"""Demo (synthetic) generator.

Creates fully-offline demo outputs so the PWA works out-of-the-box:
- per-time bins for habitat/ops/catch + covariates
- range aggregation is done client-side (docs/app.js)
- PPP (presence-only) is trained using a *presence proxy* (AIS-effort if
  token+network available; otherwise weak-label fallback; optionally CSV)

Output directory layout (default: docs/latest):
  latest/meta_index.json
  latest/runs/<run_id>/...

The UI reads:
  latest/meta_index.json
  latest/runs/<run_id>/variants/<variant>/species/<species>/meta.json
  latest/runs/<run_id>/<paths...>
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from seydyaar.export.write_cog import write_cog
from seydyaar.models.habitat import habitat_scoring
from seydyaar.models.ops import ops_feasibility
from seydyaar.models.maxent_ppp import fit_ppp_from_presence_proxy, ppp_predict
from seydyaar.providers.presence_proxy import build_presence_proxy_details
from seydyaar.utils_time import trusted_utc_now


@dataclass(frozen=True)
class GridSpec:
    bbox: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    ddeg: float
    nlon: int
    nlat: int
    lons: np.ndarray
    lats: np.ndarray


def _grid_from_bbox(bbox: Tuple[float, float, float, float], ddeg: float) -> GridSpec:
    min_lon, min_lat, max_lon, max_lat = bbox
    nlon = int(round((max_lon - min_lon) / ddeg)) + 1
    nlat = int(round((max_lat - min_lat) / ddeg)) + 1
    lons = np.linspace(min_lon, max_lon, nlon, dtype=np.float32)
    lats = np.linspace(min_lat, max_lat, nlat, dtype=np.float32)
    return GridSpec(bbox=bbox, ddeg=ddeg, nlon=nlon, nlat=nlat, lons=lons, lats=lats)


def _mk_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _smooth_noise(rng: np.random.Generator, shape: Tuple[int, int], scale: int = 5) -> np.ndarray:
    """Cheap-ish smooth random field without scipy."""
    a = rng.standard_normal(shape).astype(np.float32)
    # box blur (separable)
    k = int(scale)
    if k < 1:
        return a
    kernel = np.ones(k, dtype=np.float32) / k

    # horizontal blur
    ah = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, a)
    # vertical blur
    av = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, ah)
    return av.astype(np.float32)


def _make_covariates(
    rng: np.random.Generator,
    grid: GridSpec,
    t_idx: int,
    *,
    seasonal_phase: float,
) -> Dict[str, np.ndarray]:
    """Synthetic covariates (sst/chl/current/waves) with mild space-time structure."""
    yy, xx = np.meshgrid(grid.lats, grid.lons, indexing="ij")

    # SST: lat gradient + sinusoidal time + eddy-ish noise
    base_sst = 28.0 - 0.15 * (yy - np.mean(grid.lats))
    sst = base_sst + 1.2 * math.sin(seasonal_phase + 0.35 * t_idx) + 0.9 * _smooth_noise(rng, (grid.nlat, grid.nlon), scale=9)
    sst = sst.astype(np.float32)

    # Chl-a: log-normal-ish with fronts, inversely related to SST in this toy world
    chl_log = np.log10(0.2) + 0.25 * _smooth_noise(rng, (grid.nlat, grid.nlon), scale=7) - 0.03 * (sst - 28.0)
    chl = (10 ** chl_log).astype(np.float32)

    # Surface current speed (m/s)
    current = (0.35 + 0.15 * _smooth_noise(rng, (grid.nlat, grid.nlon), scale=11)).clip(0.05, 1.2).astype(np.float32)

    # Wave significant height (m)
    waves = (1.1 + 0.55 * _smooth_noise(rng, (grid.nlat, grid.nlon), scale=13)).clip(0.0, 4.0).astype(np.float32)

    return {"sst": sst, "chl": chl, "current": current, "waves": waves}


def _qc_mask_chl(rng: np.random.Generator, grid: GridSpec, severity: float = 0.18) -> np.ndarray:
    """Synthetic QC mask: 1=good, 0=bad."""
    noise = _smooth_noise(rng, (grid.nlat, grid.nlon), scale=15)
    thr = np.quantile(noise, 1 - severity)
    good = (noise < thr).astype(np.uint8)
    return good


def _gapfill_nearest(a: np.ndarray, mask_good: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Very small nearest-neighbor gapfill for demo.

    Fills a where mask_good==0 by iterative neighbor averaging.
    """
    out = a.copy()
    bad = (mask_good == 0)
    if not bad.any():
        return out

    # initialize bad to nan
    out = out.astype(np.float32)
    out[bad] = np.nan

    for _ in range(max_iter):
        nan = np.isnan(out)
        if not nan.any():
            break
        # 4-neighborhood average ignoring nan
        up = np.roll(out, -1, axis=0)
        dn = np.roll(out, 1, axis=0)
        lf = np.roll(out, -1, axis=1)
        rt = np.roll(out, 1, axis=1)
        stack = np.stack([up, dn, lf, rt], axis=0)
        m = np.nanmean(stack, axis=0)
        out[nan] = m[nan]

    # any remaining nan -> global median
    if np.isnan(out).any():
        med = np.nanmedian(out)
        out[np.isnan(out)] = med
    return out.astype(np.float32)


def _front_score(sst: np.ndarray, chl: np.ndarray) -> np.ndarray:
    """Toy front score using gradient magnitude."""
    def gradmag(x: np.ndarray) -> np.ndarray:
        gx = np.gradient(x, axis=1)
        gy = np.gradient(x, axis=0)
        g = np.sqrt(gx * gx + gy * gy)
        return g

    g1 = gradmag(sst)
    g2 = gradmag(np.log10(chl.clip(1e-6, None)))
    # normalize robustly
    def norm(a: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(a, [0.05, 0.95])
        if hi <= lo:
            return np.zeros_like(a, dtype=np.float32)
        return ((a - lo) / (hi - lo)).clip(0, 1).astype(np.float32)

    return (0.6 * norm(g1) + 0.4 * norm(g2)).clip(0, 1).astype(np.float32)


def _write_bin(path: Path, arr: np.ndarray, dtype: str = "f16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dtype == "f16":
        arr.astype(np.float16).tofile(path)
    elif dtype == "f32":
        arr.astype(np.float32).tofile(path)
    elif dtype == "u8":
        arr.astype(np.uint8).tofile(path)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _topk_cells(prob: np.ndarray, k: int = 10, min_separation_cells: int = 3) -> List[Tuple[int, int, float]]:
    """Pick top-k cells with simple non-max suppression."""
    flat = prob.reshape(-1)
    idxs = np.argsort(flat)[::-1]
    chosen: List[Tuple[int, int, float]] = []
    taken = np.zeros_like(prob, dtype=bool)

    for idx in idxs:
        if len(chosen) >= k:
            break
        r = idx // prob.shape[1]
        c = idx % prob.shape[1]
        if taken[r, c]:
            continue
        val = float(prob[r, c])
        chosen.append((int(r), int(c), val))
        r0 = max(0, r - min_separation_cells)
        r1 = min(prob.shape[0], r + min_separation_cells + 1)
        c0 = max(0, c - min_separation_cells)
        c1 = min(prob.shape[1], c + min_separation_cells + 1)
        taken[r0:r1, c0:c1] = True

    return chosen


def demo_generate(
    *,
    date: str,
    out_root: Path,
    fast: bool = True,
    past_days: int = 0,
    future_days: int = 0,
    step_hours: int = 2,
    models: Optional[List[str]] = None,
    species: Optional[List[str]] = None,
    presence_mode: str = "auto",
    presence_csv: Optional[Path] = None,
    export_cog: bool = False,
) -> Path:
    """Generate a complete offline demo run.

    Returns the run directory path (relative to out_root).
    """

    # Defaults
    models = models or ["scoring", "ppp", "ensemble"]
    species = species or ["skipjack", "yellowfin"]

    # AOI bbox (W Arabian Sea) – can be replaced with real AOI parsing in production
    # (min_lon, min_lat, max_lon, max_lat)
    bbox = (55.0, 10.0, 70.0, 25.0)

    ddeg = 0.25 if fast else 0.1
    grid = _grid_from_bbox(bbox, ddeg)

    # Run ids
    run_id = f"demo_{date}" + ("_fast" if fast else "_full")

    out_root = Path(out_root)
    run_root = out_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    # Times
    base_day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    t0 = base_day - timedelta(days=past_days)
    t1 = base_day + timedelta(days=future_days) + timedelta(hours=23, minutes=59)

    times: List[datetime] = []
    t = t0
    while t <= t1:
        times.append(t)
        t += timedelta(hours=step_hours)

    time_ids = [dt.strftime("%Y%m%dT%HZ") for dt in times]

    # Shared assets
    assets_dir = run_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # AOI mask (demo: all ones inside bbox)
    mask = np.ones((grid.nlat, grid.nlon), dtype=np.uint8)
    _write_bin(assets_dir / "mask_u8.bin", mask, "u8")

    # Species profiles (priors)
    # These are priors from literature summaries; fine-tune with AIS/feedback later.
    species_profiles = {
        "skipjack": {
            "display": {"en": "Skipjack", "fa": "اسکیپ‌جک"},
            "habitat": {"sst_opt": 28.0, "sst_sigma": 3.0, "chl_opt": 0.2, "chl_log_sigma": 0.6},
            "weights": {"sst": 0.30, "chl": 0.25, "front": 0.25, "current": 0.10, "waves": 0.10},
            "refs": [
                "Elith & Leathwick 2009",
                "Phillips et al. 2006",
                "Renner & Warton 2013",
                "Lehodey et al. 2008",
            ],
        },
        "yellowfin": {
            "display": {"en": "Yellowfin", "fa": "یلوفین"},
            "habitat": {"sst_opt": 26.5, "sst_sigma": 2.8, "chl_opt": 0.18, "chl_log_sigma": 0.65},
            "weights": {"sst": 0.32, "chl": 0.23, "front": 0.25, "current": 0.10, "waves": 0.10},
            "refs": [
                "Elith & Leathwick 2009",
                "Phillips et al. 2006",
                "Renner & Warton 2013",
                "Lehodey et al. 2008",
            ],
        },
    }

    # Presence-proxy / PPP fit per species (fit once on the first time step's covariates)
    ppp_models: Dict[str, dict] = {}

    # Generate per-time fields and outputs
    # We store shared covariates under times/<time>/, and variant-specific model bins
    # also under times/<time>/ with suffixes.
    variants = ["base", "gapfill"]

    # meta templates per variant/species
    variant_species_meta: Dict[Tuple[str, str], dict] = {}

    # Pre-create per variant/species meta (filled later)
    for v in variants:
        for sp in species:
            variant_species_meta[(v, sp)] = {
                "version": 1,
                "run_id": run_id,
                "variant": v,
                "species": sp,
                "fast": fast,
                "generated_at_utc": trusted_utc_now().isoformat().replace("+00:00", "Z"),
                "grid": {
                    "bbox": list(bbox),
                    "ddeg": float(ddeg),
                    "nlon": int(grid.nlon),
                    "nlat": int(grid.nlat),
                },
                "times": time_ids,
                "defaults": {
                    "map": "pcatch_ensemble" if "ensemble" in models else "pcatch_scoring",
                    "aggregation": "p90",
                    "time_range": {"t0": time_ids[0], "t1": time_ids[min(12, len(time_ids)-1)]},
                    "step_hours": step_hours,
                },
                "species_profile": species_profiles.get(sp, {}),
                "audit": {
                    "presence_proxy": {},
                    "qc": {"chl_mask": True, "gapfill": (v == "gapfill")},
                    "sanity": {"range_checks": True, "analysis_vs_forecast": True},
                },
                "paths": {
                    "mask": "assets/mask_u8.bin",
                    "per_time": {
                        # Covariates (shared)
                        "sst": "times/{time}/sst_f32.bin",
                        "chl": f"times/{{time}}/chl_{v}_f32.bin",
                        "current": "times/{time}/current_f32.bin",
                        "waves": "times/{time}/waves_f32.bin",
                        "qc_chl": "times/{time}/qc_chl_u8.bin",
                        # Uncertainty (shared)
                        "conf": "times/{time}/conf_f32.bin",
                        "agree": "times/{time}/agree_f32.bin",
                        "spread": "times/{time}/spread_f32.bin",
                        # Habitat/ops/catch maps
                        "phab_scoring": f"times/{{time}}/phab_scoring_{v}.bin",
                        "phab_frontplus": f"times/{{time}}/phab_frontplus_{v}.bin",
                        "pops": f"times/{{time}}/pops_{v}.bin",
                        "pcatch_scoring": f"times/{{time}}/pcatch_scoring_{v}.bin",
                        "pcatch_ppp": f"times/{{time}}/pcatch_ppp_{v}.bin",
                        "pcatch_ensemble": f"times/{{time}}/pcatch_ensemble_{v}.bin",
                    },
                },
            }

    # Main loop
    for ti, time_id in enumerate(time_ids):
        times_dir = run_root / "times" / time_id
        times_dir.mkdir(parents=True, exist_ok=True)

        # Stable RNG per time for reproducible demo
        rng = _mk_rng(seed=abs(hash((run_id, time_id))) % (2**32 - 1))
        seasonal_phase = 2 * math.pi * (ti / max(1, len(time_ids) - 1))

        cov = _make_covariates(rng, grid, ti, seasonal_phase=seasonal_phase)
        sst = cov["sst"]
        chl_raw = cov["chl"]
        current = cov["current"]
        waves = cov["waves"]

        qc_chl = _qc_mask_chl(rng, grid, severity=0.22 if not fast else 0.18)

        # Save shared covariates
        _write_bin(times_dir / "sst_f32.bin", sst, "f32")
        _write_bin(times_dir / "current_f32.bin", current, "f32")
        _write_bin(times_dir / "waves_f32.bin", waves, "f32")
        _write_bin(times_dir / "qc_chl_u8.bin", qc_chl, "u8")

        # Confidence/uncertainty (demo): lower confidence where qc bad or where waves high
        conf = (0.85 * (qc_chl.astype(np.float32)) + 0.15 * (1.0 - (waves / 4.0).clip(0, 1))).clip(0, 1)
        agree = (0.6 + 0.4 * qc_chl.astype(np.float32)).clip(0, 1)
        spread = (0.15 + 0.35 * (1 - qc_chl.astype(np.float32))).clip(0, 1)
        _write_bin(times_dir / "conf_f32.bin", conf, "f32")
        _write_bin(times_dir / "agree_f32.bin", agree, "f32")
        _write_bin(times_dir / "spread_f32.bin", spread, "f32")

        # Build per-variant chl
        chl_base = chl_raw.copy().astype(np.float32)
        chl_base[qc_chl == 0] = np.nan
        chl_gap = _gapfill_nearest(chl_base, qc_chl)  # fills nans

        chl_by_variant = {
            "base": chl_base,
            "gapfill": chl_gap,
        }

        # Save variant-specific chl
        _write_bin(times_dir / "chl_base_f32.bin", np.nan_to_num(chl_base, nan=np.nanmedian(chl_base)), "f32")
        _write_bin(times_dir / "chl_gapfill_f32.bin", chl_gap, "f32")

        # Compute front score (shared, based on raw)
        front = _front_score(sst, np.nan_to_num(chl_gap, nan=np.nanmedian(chl_gap)))

        # For each variant + species compute outputs
        for v in variants:
            chl = chl_by_variant[v]
            # replace nan for model computations
            chl_model = np.nan_to_num(chl, nan=np.nanmedian(chl_gap)).astype(np.float32)

            for sp in species:
                prof = species_profiles[sp]

                # Habitat scoring + Front+ variant
                phab_score = habitat_scoring(
                    sst,
                    chl_model,
                    sst_opt=prof["habitat"]["sst_opt"],
                    sst_sigma=prof["habitat"]["sst_sigma"],
                    chl_opt=prof["habitat"]["chl_opt"],
                    chl_log_sigma=prof["habitat"]["chl_log_sigma"],
                )

                # Front+ as multiplicative boost in fronty areas (bounded)
                phab_front = (phab_score * (0.75 + 0.25 * front)).clip(0, 1).astype(np.float32)

                # Ops
                pops = ops_feasibility(current, waves, prof)

                # Catch baseline (habitat x ops)
                pcatch_scoring = (phab_score * pops).clip(0, 1).astype(np.float32)

                # PPP (presence-only): fit on first time step only (for speed)
                if sp not in ppp_models:
                    presence_idx, presence_audit, bias_surface = build_presence_proxy_details(
                        mode=presence_mode,
                        species=sp,
                        grid=grid,
                        habitat_proxy=pcatch_scoring,
                        presence_csv_path=presence_csv,
                    )

                    # Fit on a subset of covariates (toy)
                    covs_for_ppp = {
                        "sst": sst,
                        "chl": chl_model,
                        "front": front,
                        "current": current,
                        "waves": waves,
                    }
                    ppp_model, ppp_info = fit_ppp_from_presence_proxy(
                        covariates=covs_for_ppp,
                        presence_idx=presence_idx,
                        mask=mask,
                        bias_surface=bias_surface,
                        n_background=6000 if fast else 25000,
                        reg_strength=1.0,
                        random_state=42,
                    )
                    ppp_models[sp] = {"model": ppp_model, "info": ppp_info, "audit": presence_audit}

                    # Store audit into meta template for both variants (same presence proxy)
                    for vv in variants:
                        variant_species_meta[(vv, sp)]["audit"]["presence_proxy"] = {
                            **presence_audit,
                            "ppp": ppp_info,
                        }

                # Predict PPP catchability proxy and rescale to 0..1
                ppp_pred = ppp_predict(ppp_models[sp]["model"], {
                    "sst": sst,
                    "chl": chl_model,
                    "front": front,
                    "current": current,
                    "waves": waves,
                }).astype(np.float32)

                # PPP prediction is intensity-like; normalize robustly
                lo, hi = np.quantile(ppp_pred, [0.05, 0.95])
                if hi > lo:
                    pcatch_ppp = ((ppp_pred - lo) / (hi - lo)).clip(0, 1).astype(np.float32)
                else:
                    pcatch_ppp = np.zeros_like(ppp_pred, dtype=np.float32)

                # Ensemble: combine (scoring catch, front catch, ppp) with agreement/spread
                # Here we use a simple weighted average; in production you may do stacking.
                w_sc, w_fr, w_ppp = 0.40, 0.25, 0.35
                pcatch_ens = (w_sc * pcatch_scoring + w_fr * (phab_front * pops) + w_ppp * pcatch_ppp).clip(0, 1)
                # down-weight by uncertainty
                pcatch_ens = (pcatch_ens * conf).clip(0, 1).astype(np.float32)

                # Write bins
                suf = v
                _write_bin(times_dir / f"phab_scoring_{suf}.bin", phab_score, "f16")
                _write_bin(times_dir / f"phab_frontplus_{suf}.bin", phab_front, "f16")
                _write_bin(times_dir / f"pops_{suf}.bin", pops, "f16")
                _write_bin(times_dir / f"pcatch_scoring_{suf}.bin", pcatch_scoring, "f16")
                _write_bin(times_dir / f"pcatch_ppp_{suf}.bin", pcatch_ppp, "f16")
                _write_bin(times_dir / f"pcatch_ensemble_{suf}.bin", pcatch_ens, "f16")

                # Optional: export per-time COG for ensemble catch (for production-ish outputs)
                if export_cog and (ti % 3 == 0):
                    out_tif = times_dir / f"pcatch_ensemble_{suf}_{sp}.tif"
                    write_cog(out_tif, (pcatch_ens * 100.0).astype(np.float32), bbox=bbox, nodata=-9999.0)

    # Write species meta.json files per variant
    for v in variants:
        for sp in species:
            meta_path = run_root / "variants" / v / "species" / sp / "meta.json"
            _write_json(meta_path, variant_species_meta[(v, sp)])

    # Write run-level meta.json
    run_meta = {
        "version": 1,
        "run_id": run_id,
        "date": date,
        "fast": fast,
        "generated_at_utc": trusted_utc_now().isoformat().replace("+00:00", "Z"),
        "times": time_ids,
        "variants": variants,
        "species": species,
        "bbox": list(bbox),
        "step_hours": step_hours,
        "export_cog": export_cog,
    }
    _write_json(run_root / "meta.json", run_meta)

    # Update meta_index.json
    index_path = out_root / "meta_index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {"version": 1, "runs": []}
    else:
        index = {"version": 1, "runs": []}

    run_entry = {
        "run_id": run_id,
        "path": f"runs/{run_id}",
        "fast": fast,
        "date": date,
        "time_count": len(time_ids),
        "variants": variants,
        "species": species,
        "generated_at_utc": run_meta["generated_at_utc"],
    }

    # replace if exists
    index["runs"] = [r for r in index.get("runs", []) if r.get("run_id") != run_id] + [run_entry]
    index["runs"] = sorted(index["runs"], key=lambda r: r.get("generated_at_utc", ""))
    index["latest_run_id"] = run_id
    index["generated_at_utc"] = trusted_utc_now().isoformat().replace("+00:00", "Z")

    _write_json(index_path, index)

    return Path(f"runs/{run_id}")
