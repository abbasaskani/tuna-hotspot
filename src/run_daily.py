import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
from shapely.prepared import prep

import copernicusmarine as cm


AOI_PATH = "config/aoi.geojson"
OUT_DIR = "docs/latest"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Your gear constraint (surface gillnet): 0–5 m
DEPTH_MIN_M = 0.0
DEPTH_MAX_M = 5.0

# ---- Datasets (Copernicus Marine "analysis+forecast" families)
# Physics temperature (daily)
TEMP_DATASET = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
# Currents (daily)
CUR_DATASET = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
# Sea level / SSH (daily) – used as a proxy for fronts/eddies via gradient
SSH_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
# Biogeochemistry optics (daily) – chlorophyll often here
CHL_DATASET_PRIMARY = "cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m"
# Waves (3-hourly) – significant wave height
WAVES_DATASET = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"

# Preferred variable candidates (we auto-pick the first that exists)
PREF_TEMP_VARS = ["thetao", "sea_water_potential_temperature"]
PREF_CUR_U_VARS = ["uo", "eastward_sea_water_velocity"]
PREF_CUR_V_VARS = ["vo", "northward_sea_water_velocity"]
PREF_SSH_VARS = ["zos", "sea_surface_height_above_geoid", "adt", "sla"]
PREF_CHL_VARS = ["CHL", "chl", "chlorophyll", "chl_a"]
PREF_WAVE_VARS = ["VHM0", "significant_height_of_combined_wind_waves_and_swell"]


# ---------------- Utilities ----------------
def read_geojson_polygon(path: str):
    gj = json.load(open(path, "r", encoding="utf-8"))
    geom = gj["features"][0]["geometry"]
    return shape(geom)

def bbox_from_geojson(path: str) -> Tuple[float, float, float, float]:
    gj = json.load(open(path, "r", encoding="utf-8"))
    coords = gj["features"][0]["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), max(lons), min(lats), max(lats)

def _parse_dataset_time_bounds(msg: str) -> Optional[Tuple[datetime, datetime]]:
    # Example:
    # "... exceed the dataset coordinates [2020-08-20 00:00:00+00:00, 2025-12-28 00:00:00+00:00]"
    m = re.search(r"dataset coordinates \[(.*?), (.*?)\]", msg)
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    try:
        # Normalize space to ISO "T"
        a = a.replace(" ", "T")
        b = b.replace(" ", "T")
        tmin = datetime.fromisoformat(a)
        tmax = datetime.fromisoformat(b)
        return tmin, tmax
    except Exception:
        return None

def _parse_dataset_depth_bounds(msg: str) -> Optional[Tuple[float, float]]:
    # Example:
    # "... depth dimension exceed the dataset coordinates [0.4940..., 5727.9169...]"
    m = re.search(r"depth dimension.*dataset coordinates \[(.*?), (.*?)\]", msg)
    if not m:
        return None
    try:
        dmin = float(m.group(1).strip())
        dmax = float(m.group(2).strip())
        return dmin, dmax
    except Exception:
        return None

def _latlon_names(ds: xr.Dataset) -> Tuple[str, str]:
    for lat in ["latitude", "lat", "nav_lat"]:
        if lat in ds.coords:
            lat_name = lat
            break
    else:
        raise RuntimeError("No latitude coord found in dataset.")
    for lon in ["longitude", "lon", "nav_lon"]:
        if lon in ds.coords:
            lon_name = lon
            break
    else:
        raise RuntimeError("No longitude coord found in dataset.")
    return lat_name, lon_name

def _time_name(ds: xr.Dataset) -> Optional[str]:
    for t in ["time", "TIME"]:
        if t in ds.coords:
            return t
    return None

def _pick_existing_var(ds: xr.Dataset, prefer: List[str]) -> str:
    vars_ = list(ds.data_vars.keys())
    for p in prefer:
        if p in vars_:
            return p
    if not vars_:
        raise RuntimeError("Dataset has no data_vars.")
    return vars_[0]

def _safe_open_dataset(
    dataset_id: str,
    bbox: Tuple[float, float, float, float],
    start_dt: datetime,
    end_dt: datetime,
    variables: Optional[List[str]] = None,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    max_tries: int = 5,
) -> Tuple[xr.Dataset, Dict[str, Any]]:
    """
    Opens a dataset subset. If requested time/depth is out of bounds, it clamps automatically.
    Returns dataset + metadata about the effective selection.
    """
    min_lon, max_lon, min_lat, max_lat = bbox
    eff = {
        "dataset_id": dataset_id,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "depth_min": depth_min,
        "depth_max": depth_max,
    }

    last_err = None
    for _ in range(max_tries):
        try:
            kwargs = dict(
                dataset_id=dataset_id,
                minimum_longitude=min_lon,
                maximum_longitude=max_lon,
                minimum_latitude=min_lat,
                maximum_latitude=max_lat,
                start_datetime=start_dt.isoformat(),
                end_datetime=end_dt.isoformat(),
                coordinates_selection_method="nearest",
                disable_progress_bar=True,
            )
            if variables:
                kwargs["variables"] = variables
            if depth_min is not None and depth_max is not None:
                kwargs["minimum_depth"] = float(depth_min)
                kwargs["maximum_depth"] = float(depth_max)

            ds = cm.open_dataset(**kwargs)
            eff.update({
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "depth_min": depth_min,
                "depth_max": depth_max,
            })
            return ds, eff

        except Exception as e:
            msg = str(e)
            last_err = e

            # Clamp time if our start is beyond dataset max
            if "time dimension exceed" in msg or "time dimension" in msg and "dataset coordinates" in msg:
                tb = _parse_dataset_time_bounds(msg)
                if tb:
                    _, tmax = tb
                    # If requested window starts after available max, move back to tmax's date
                    if start_dt > tmax:
                        new_day = tmax.date()
                        start_dt = datetime(new_day.year, new_day.month, new_day.day, tzinfo=tmax.tzinfo)
                        end_dt = start_dt + timedelta(days=1)
                        continue

            # Clamp depth if our requested range violates dataset depth coords
            if "depth dimension exceed" in msg:
                db = _parse_dataset_depth_bounds(msg)
                if db and (depth_min is not None and depth_max is not None):
                    dmin_av, dmax_av = db
                    depth_min = max(depth_min, dmin_av)
                    depth_max = min(depth_max, dmax_av)
                    # ensure valid range
                    if depth_min > depth_max:
                        depth_min = dmin_av
                        depth_max = min(dmin_av, dmax_av)
                    continue

            # Otherwise: break (real error: invalid dataset, auth, etc.)
            break

    raise RuntimeError(f"Failed opening dataset={dataset_id}. Last error: {last_err}")


def _regrid_to_target(da: xr.DataArray, target_lats: np.ndarray, target_lons: np.ndarray) -> xr.DataArray:
    """Interpolate da to 1D target lat/lon coordinates."""
    lat_name, lon_name = None, None
    for cand in ["latitude", "lat", "nav_lat"]:
        if cand in da.coords:
            lat_name = cand
            break
    for cand in ["longitude", "lon", "nav_lon"]:
        if cand in da.coords:
            lon_name = cand
            break
    if lat_name is None or lon_name is None:
        raise RuntimeError("Cannot find lat/lon coords for regridding.")

    # Make sure monotonic for interp
    da2 = da
    if np.any(np.diff(da2[lat_name].values) < 0):
        da2 = da2.sortby(lat_name)
    if np.any(np.diff(da2[lon_name].values) < 0):
        da2 = da2.sortby(lon_name)

    try:
        return da2.interp({lat_name: target_lats, lon_name: target_lons}, method="linear")
    except Exception:
        return da2.interp({lat_name: target_lats, lon_name: target_lons}, method="nearest")


def _polygon_mask(poly, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Boolean mask (True inside polygon) for a lat/lon grid defined by 1D arrays."""
    P = prep(poly)
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    # This is fine at 0.083° grids; if you later go very high-res, vectorize.
    for iy, la in enumerate(lats):
        for ix, lo in enumerate(lons):
            mask[iy, ix] = P.contains(Point(float(lo), float(la)))
    return mask


def _grad_mag(arr2d: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(arr2d)
    return np.sqrt(gx * gx + gy * gy)


def _normalize_robust(a: np.ndarray, p: float = 95.0) -> np.ndarray:
    """Robust normalize to 0..1 using percentile p."""
    v = a.copy()
    finite = np.isfinite(v)
    if not np.any(finite):
        return np.zeros_like(v)
    scale = np.nanpercentile(v[finite], p)
    if scale <= 0 or not np.isfinite(scale):
        scale = np.nanmax(v[finite]) + 1e-9
    return np.clip(v / (scale + 1e-9), 0, 1)


# ---------------- Scoring model (0–100) ----------------
def score_temp(temp_c: np.ndarray) -> np.ndarray:
    # Broad tuna-friendly surface layer: peak ~28C, sigma ~3C
    return np.exp(-((temp_c - 28.0) / 3.0) ** 2)

def score_chl(chl: np.ndarray) -> np.ndarray:
    # Use log scale; peak around ~0.2 mg/m3, tolerant (very region-dependent)
    chl2 = np.clip(chl, 1e-4, np.inf)
    x = np.log10(chl2)
    mu = np.log10(0.2)
    sigma = 0.6
    return np.exp(-((x - mu) / sigma) ** 2)

def score_current(spd: np.ndarray) -> np.ndarray:
    # Operational + ecology: prefer moderate currents (avoid extremes for gillnet ops)
    return np.exp(-((spd - 0.4) / 0.25) ** 2)

def score_waves(hs: np.ndarray) -> np.ndarray:
    # Penalize rough sea; ~1.5m pivot
    return 1.0 / (1.0 + np.exp((hs - 1.5) / 0.35))


def main():
    print("Reading AOI...")
    poly = read_geojson_polygon(AOI_PATH)
    bbox = bbox_from_geojson(AOI_PATH)

    today = datetime.utcnow().date()
    # We request today; _safe_open_dataset clamps to latest available automatically if needed.
    start = datetime(today.year, today.month, today.day)
    end = start + timedelta(days=1)

    # --- Load Temperature (0–5 m mean)
    temp_ds, temp_eff = _safe_open_dataset(
        TEMP_DATASET, bbox, start, end,
        variables=None,
        depth_min=DEPTH_MIN_M, depth_max=DEPTH_MAX_M
    )
    tvar = _pick_existing_var(temp_ds, PREF_TEMP_VARS)
    lat_name, lon_name = _latlon_names(temp_ds)
    time_name = _time_name(temp_ds)

    temp_da = temp_ds[tvar]
    if time_name:
        temp_da = temp_da.isel({time_name: 0})
    # Average over depth if present
    if "depth" in temp_da.dims:
        temp_da = temp_da.mean("depth")
    temp_da = temp_da.load()

    lats = temp_da[lat_name].values
    lons = temp_da[lon_name].values

    # AOI mask on the target grid (temp grid)
    mask = _polygon_mask(poly, lats, lons)

    # --- Load Currents (0–5 m mean), regrid to temp grid (should already match, but safe)
    cur_ok = True
    try:
        cur_ds, cur_eff = _safe_open_dataset(
            CUR_DATASET, bbox, start, end,
            variables=None,
            depth_min=DEPTH_MIN_M, depth_max=DEPTH_MAX_M
        )
        uvar = _pick_existing_var(cur_ds, PREF_CUR_U_VARS)
        vvar = _pick_existing_var(cur_ds, PREF_CUR_V_VARS)
        time_name_c = _time_name(cur_ds)
        u = cur_ds[uvar]
        v = cur_ds[vvar]
        if time_name_c:
            u = u.isel({time_name_c: 0})
            v = v.isel({time_name_c: 0})
        if "depth" in u.dims:
            u = u.mean("depth")
        if "depth" in v.dims:
            v = v.mean("depth")
        u = u.load()
        v = v.load()

        u = _regrid_to_target(u, lats, lons)
        v = _regrid_to_target(v, lats, lons)
        cur_speed = np.sqrt(u.values ** 2 + v.values ** 2)
    except Exception as e:
        print(f"WARNING: currents unavailable -> will downweight. Error: {e}")
        cur_eff = {"dataset_id": CUR_DATASET, "error": str(e)}
        cur_ok = False
        cur_speed = np.full((len(lats), len(lons)), np.nan)

    # --- Load SSH (front proxy), regrid to temp grid
    ssh_ok = True
    try:
        ssh_ds, ssh_eff = _safe_open_dataset(
            SSH_DATASET, bbox, start, end,
            variables=None
        )
        svar = _pick_existing_var(ssh_ds, PREF_SSH_VARS)
        time_name_s = _time_name(ssh_ds)
        ssh = ssh_ds[svar]
        if time_name_s:
            ssh = ssh.isel({time_name_s: 0})
        if "depth" in ssh.dims:
            # if any, take surface-ish
            ssh = ssh.isel(depth=0)
        ssh = ssh.load()
        ssh = _regrid_to_target(ssh, lats, lons)
        ssh_vals = ssh.values
    except Exception as e:
        print(f"WARNING: SSH unavailable -> will downweight. Error: {e}")
        ssh_eff = {"dataset_id": SSH_DATASET, "error": str(e)}
        ssh_ok = False
        ssh_vals = np.full((len(lats), len(lons)), np.nan)

    # --- Load Chlorophyll (surface or 0–5m), regrid to temp grid
    chl_ok = True
    try:
        chl_ds, chl_eff = _safe_open_dataset(
            CHL_DATASET_PRIMARY, bbox, start, end,
            variables=None,
            depth_min=DEPTH_MIN_M, depth_max=DEPTH_MAX_M
        )
        cvar = _pick_existing_var(chl_ds, PREF_CHL_VARS)
        time_name_ch = _time_name(chl_ds)
        chl = chl_ds[cvar]
        if time_name_ch:
            chl = chl.isel({time_name_ch: 0})
        if "depth" in chl.dims:
            chl = chl.mean("depth")
        chl = chl.load()
        chl = _regrid_to_target(chl, lats, lons)
        chl_vals = chl.values
    except Exception as e:
        print(f"WARNING: CHL unavailable -> will downweight. Error: {e}")
        chl_eff = {"dataset_id": CHL_DATASET_PRIMARY, "error": str(e)}
        chl_ok = False
        chl_vals = np.full((len(lats), len(lons)), np.nan)

    # --- Load Waves (Hs), daily max, regrid to temp grid
    wav_ok = True
    try:
        wav_ds, wav_eff = _safe_open_dataset(
            WAVES_DATASET, bbox, start, end,
            variables=None
        )
        wvar = _pick_existing_var(wav_ds, PREF_WAVE_VARS)
        time_name_w = _time_name(wav_ds)
        hs = wav_ds[wvar]
        if time_name_w:
            # daily maximum Hs for operations
            hs = hs.max(time_name_w)
        hs = hs.load()
        hs = _regrid_to_target(hs, lats, lons)
        hs_vals = hs.values
    except Exception as e:
        print(f"WARNING: waves unavailable -> will downweight. Error: {e}")
        wav_eff = {"dataset_id": WAVES_DATASET, "error": str(e)}
        wav_ok = False
        hs_vals = np.full((len(lats), len(lons)), np.nan)

    # ---------------- Build sub-scores ----------------
    temp_vals = temp_da.values

    # Front strength (gradients) – robust normalized
    temp_front = _normalize_robust(_grad_mag(temp_vals))
    chl_front = _normalize_robust(_grad_mag(chl_vals)) if chl_ok else np.zeros_like(temp_front)
    ssh_front = _normalize_robust(_grad_mag(ssh_vals)) if ssh_ok else np.zeros_like(temp_front)
    front_score = np.clip(0.5 * temp_front + 0.25 * chl_front + 0.25 * ssh_front, 0, 1)

    # Suitability
    s_temp = score_temp(temp_vals)
    s_chl = score_chl(chl_vals) if chl_ok else np.full_like(s_temp, 0.5)
    s_cur = score_current(cur_speed) if cur_ok else np.full_like(s_temp, 0.5)
    s_wav = score_waves(hs_vals) if wav_ok else np.full_like(s_temp, 0.5)

    # ---------------- Weights (auto-renormalize if a layer is missing) ----------------
    # Literature commonly uses SST + CHL + SSHA/fronts for tuna habitat/hotspots; ops adds waves/currents. 
    weights = {
        "temp": 0.30,
        "chl": 0.25,
        "front": 0.25,
        "current": 0.10,
        "waves": 0.10,
    }
    avail = {
        "temp": True,
        "chl": chl_ok,
        "front": True,   # front uses temp anyway; ssh/chl fronts degrade gracefully
        "current": cur_ok,
        "waves": wav_ok,
    }
    wsum = sum(weights[k] for k in weights if avail[k])
    if wsum <= 0:
        raise RuntimeError("No usable layers available to build score.")

    for k in list(weights.keys()):
        if not avail[k]:
            weights[k] = 0.0
    # renormalize
    if sum(weights.values()) > 0:
        scale = 1.0 / sum(weights.values())
        for k in weights:
            weights[k] *= scale

    prob01 = (
        weights["temp"] * s_temp +
        weights["chl"] * s_chl +
        weights["front"] * front_score +
        weights["current"] * s_cur +
        weights["waves"] * s_wav
    )

    prob = 100.0 * np.clip(prob01, 0, 1)

    # Apply AOI mask
    prob_masked = prob.copy()
    prob_masked[~mask] = np.nan

    # ---------------- Top-10 hotspots ----------------
    flat = prob_masked.ravel()
    valid = np.isfinite(flat)
    if not np.any(valid):
        raise RuntimeError("All scores are NaN inside AOI. Check AOI polygon and dataset coverage.")
    idx_sorted = np.argsort(flat[valid])[::-1]
    topn = 10
    valid_idx = np.where(valid)[0][idx_sorted[:topn]]
    yy, xx = np.unravel_index(valid_idx, prob_masked.shape)

    # Create GeoJSON features (include explainable fields)
    features = []
    for rank, (y, x) in enumerate(zip(yy, xx), start=1):
        props = {
            "rank": rank,
            "probability_0_100": float(prob_masked[y, x]),
            "temp_C_0_5m": float(temp_vals[y, x]) if np.isfinite(temp_vals[y, x]) else None,
            "chl": float(chl_vals[y, x]) if np.isfinite(chl_vals[y, x]) else None,
            "current_mps": float(cur_speed[y, x]) if np.isfinite(cur_speed[y, x]) else None,
            "hs_m": float(hs_vals[y, x]) if np.isfinite(hs_vals[y, x]) else None,
            "front_score_0_1": float(front_score[y, x]) if np.isfinite(front_score[y, x]) else None,
        }
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Point", "coordinates": [float(lons[x]), float(lats[y])]},
        })

    out_geo = {"type": "FeatureCollection", "features": features}
    with open(f"{OUT_DIR}/hotspots.geojson", "w", encoding="utf-8") as f:
        json.dump(out_geo, f, ensure_ascii=False, indent=2)

    # Save run metadata
    meta = {
        "run_utc": datetime.utcnow().isoformat() + "Z",
        "requested_day_utc": today.isoformat(),
        "depth_m": [DEPTH_MIN_M, DEPTH_MAX_M],
        "weights_final": weights,
        "datasets_effective": {
            "temp": temp_eff,
            "currents": cur_eff,
            "ssh": ssh_eff,
            "chl": chl_eff,
            "waves": wav_eff,
        },
    }
    with open(f"{OUT_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---------------- Plot: red (low) -> green (high) ----------------
    plt.figure(figsize=(10, 7))
    # Use pcolormesh with 1D coords
    plt.pcolormesh(lons, lats, prob_masked, shading="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(label="Fishing probability (0–100)")
    plt.title(f"Tuna Hotspot Probability (0–5m) | requested {today.isoformat()} (UTC)")

    # Overlay top-10
    for rank, (y, x) in enumerate(zip(yy, xx), start=1):
        plt.scatter(lons[x], lats[y], s=35, c="black")
        plt.text(lons[x], lats[y], f"{rank}", fontsize=9, ha="left", va="bottom", color="black")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/probability.png", dpi=150)
    plt.close()

    # Human-readable summary
    lines = []
    lines.append(f"Run UTC: {meta['run_utc']}")
    lines.append(f"Requested day UTC: {meta['requested_day_utc']}")
    lines.append(f"Depth used (m): {meta['depth_m']}")
    lines.append(f"Weights(final): {weights}")
    lines.append("")
    lines.append("Top-10 Hotspots:")
    for ft in features:
        lon, lat = ft["geometry"]["coordinates"]
        p = ft["properties"]
        lines.append(
            f"#{p['rank']:02d}  prob={p['probability_0_100']:.1f}  "
            f"lon={lon:.4f} lat={lat:.4f}  "
            f"T={p['temp_C_0_5m']:.2f}C  CHL={p['chl'] if p['chl'] is not None else 'NA'}  "
            f"U={p['current_mps'] if p['current_mps'] is not None else 'NA'} m/s  "
            f"Hs={p['hs_m'] if p['hs_m'] is not None else 'NA'} m"
        )
    with open(f"{OUT_DIR}/summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Done. Outputs written to docs/latest/")

if __name__ == "__main__":
    main()