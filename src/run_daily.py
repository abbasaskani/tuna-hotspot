import json
import os
from datetime import datetime, timedelta, date

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import copernicusmarine as cm

# -----------------------
# Paths
# -----------------------
AOI_PATH = "config/aoi.geojson"
OUT_DIR = "docs/latest"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Dataset IDs
# -----------------------
# 1) SST (Level-4 NRT)
SST_DATASET = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"

# 2) CHL (Daily, gapfree-like product; may lag behind "today" -> we auto-backtrack)
# اگر این dataset_id در اکانت شما در دسترس نبود، فقط همین خط را با یک dataset CHL دیگر جایگزین کنید.
CHL_DATASET = "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"

# 3) Surface currents (merged UV hourly)
UV_DATASET = "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i"

# 4) Sea level (merged sea level hourly)
SL_DATASET = "cmems_mod_glo_phy_anfc_merged-sl_PT1H-i"


# -----------------------
# Variable preference lists
# -----------------------
PREF_SST = ["analysed_sst", "sst"]
PREF_CHL = ["CHL", "chl", "chlorophyll", "chlor_a", "CHL1"]
PREF_UV_PAIRS = [
    ("uo", "vo"),
    ("ugos", "vgos"),
    ("u", "v"),
    ("eastward_sea_water_velocity", "northward_sea_water_velocity"),
]
PREF_SL = ["sla", "zos", "ssh", "adt", "sea_surface_height_above_geoid"]


# -----------------------
# Utilities
# -----------------------
def bbox_from_geojson(path: str):
    gj = json.load(open(path, "r", encoding="utf-8"))
    coords = gj["features"][0]["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return float(min(lons)), float(max(lons)), float(min(lats)), float(max(lats))


def _normalize_lonlat_names(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for cand in ["longitude", "lon", "LONGITUDE", "x"]:
        if cand in ds.coords or cand in ds.dims:
            rename[cand] = "lon"
            break
    for cand in ["latitude", "lat", "LATITUDE", "y"]:
        if cand in ds.coords or cand in ds.dims:
            rename[cand] = "lat"
            break
    if rename:
        ds = ds.rename(rename)
    return ds


def _ensure_lat_ascending(da: xr.DataArray) -> xr.DataArray:
    if "lat" in da.dims:
        lat = da["lat"].values
        if lat[0] > lat[-1]:
            da = da.sortby("lat")
    return da


def _safe_grad_mag(a2d: np.ndarray) -> np.ndarray:
    # fill NaNs to avoid blowing up gradients
    if np.all(~np.isfinite(a2d)):
        return np.full_like(a2d, np.nan, dtype=float)
    fill = np.nanmedian(a2d)
    arr = np.where(np.isfinite(a2d), a2d, fill)
    gy, gx = np.gradient(arr)
    g = np.sqrt(gx * gx + gy * gy)
    return g


def _norm01(x: np.ndarray) -> np.ndarray:
    if np.all(~np.isfinite(x)):
        return np.full_like(x, np.nan, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if np.isclose(mx - mn, 0.0):
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def _gaussian_score(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _subset_one_day(dataset_id: str, variables: list[str], d: date, bbox, out_name: str):
    min_lon, max_lon, min_lat, max_lat = bbox
    start = datetime(d.year, d.month, d.day)
    end = start + timedelta(days=1)

    cm.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=min_lon, maximum_longitude=max_lon,
        minimum_latitude=min_lat, maximum_latitude=max_lat,
        start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
        file_format="netcdf",
        output_directory="/tmp",
        output_filename=out_name,
        overwrite=True,
        disable_progress_bar=True,
    )
    return os.path.join("/tmp", out_name)


def _find_working_day_for_vars(dataset_id: str, variables: list[str], target: date, bbox, lookback_days: int, out_prefix: str):
    last_err = None
    for k in range(0, lookback_days + 1):
        d = target - timedelta(days=k)
        out_name = f"{out_prefix}_{dataset_id}_{d.isoformat()}.nc".replace("/", "_")
        try:
            path = _subset_one_day(dataset_id, variables, d, bbox, out_name)
            return d, path
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not find an available day within {lookback_days} days for {dataset_id}. Last error: {last_err}")


def _find_working_day_singlevar(dataset_id: str, prefer_vars: list[str], target: date, bbox, lookback_days: int, out_prefix: str):
    # try var names one by one; for each var, try backtracking days
    last_err = None
    for v in prefer_vars:
        try:
            d, path = _find_working_day_for_vars(dataset_id, [v], target, bbox, lookback_days, out_prefix)
            return v, d, path
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not read any preferred variable for {dataset_id}. Last error: {last_err}")


def _find_working_day_uv(dataset_id: str, target: date, bbox, lookback_days: int, out_prefix: str):
    last_err = None
    for (u, v) in PREF_UV_PAIRS:
        try:
            d, path = _find_working_day_for_vars(dataset_id, [u, v], target, bbox, lookback_days, out_prefix)
            return (u, v), d, path
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not read any UV pair for {dataset_id}. Last error: {last_err}")


# -----------------------
# Main
# -----------------------
def main():
    print("Reading AOI...")
    bbox = bbox_from_geojson(AOI_PATH)

    # date control
    # اگر RUN_DATE ست کنید (مثلاً 2026-01-01) همان را هدف می‌گیرد؛
    # و اگر داده نبود، خودش عقب می‌رود تا داده پیدا کند.
    run_date_str = os.environ.get("RUN_DATE", "").strip()
    if run_date_str:
        target = datetime.strptime(run_date_str, "%Y-%m-%d").date()
    else:
        target = datetime.utcnow().date()

    print(f"Finding latest available dates (target={target.isoformat()})...")

    layers = {}

    # ---- SST (lookback کوتاه‌تر چون NRT است)
    try:
        sst_var, sst_day, sst_path = _find_working_day_singlevar(
            SST_DATASET, PREF_SST, target, bbox, lookback_days=14, out_prefix="sst"
        )
        ds = _normalize_lonlat_names(xr.open_dataset(sst_path))
        da = ds[sst_var]
        if "time" in da.dims:
            da = da.isel(time=0)
        da = da.rename("sst").astype(float)
        da = da - 273.15 if np.nanmean(da.values) > 100 else da  # Kelvin->C fallback
        da = _ensure_lat_ascending(da)
        layers["sst"] = da
        print(f"SST OK: day={sst_day} var={sst_var}")
    except Exception as e:
        print(f"[WARN] SST unavailable: {e}")

    # ---- CHL (lookback بزرگ‌تر چون ممکن است lag داشته باشد)
    try:
        chl_var, chl_day, chl_path = _find_working_day_singlevar(
            CHL_DATASET, PREF_CHL, target, bbox, lookback_days=60, out_prefix="chl"
        )
        ds = _normalize_lonlat_names(xr.open_dataset(chl_path))
        da = ds[chl_var]
        if "time" in da.dims:
            da = da.isel(time=0)
        da = da.rename("chl").astype(float)
        da = _ensure_lat_ascending(da)
        layers["chl"] = da
        print(f"CHL OK: day={chl_day} var={chl_var}")
    except Exception as e:
        print(f"[WARN] CHL unavailable: {e}")

    # ---- UV (surface currents) (hourly -> daily mean)
    try:
        (u_var, v_var), uv_day, uv_path = _find_working_day_uv(
            UV_DATASET, target, bbox, lookback_days=14, out_prefix="uv"
        )
        ds = _normalize_lonlat_names(xr.open_dataset(uv_path))
        u = ds[u_var].astype(float)
        v = ds[v_var].astype(float)
        if "time" in u.dims:
            u = u.mean("time", skipna=True)
            v = v.mean("time", skipna=True)
        u = _ensure_lat_ascending(u)
        v = _ensure_lat_ascending(v)
        layers["u"] = u
        layers["v"] = v
        print(f"UV OK: day={uv_day} vars=({u_var},{v_var})")
    except Exception as e:
        print(f"[WARN] UV unavailable: {e}")

    # ---- Sea level / SSH proxy (hourly -> daily mean)
    try:
        sl_var, sl_day, sl_path = _find_working_day_singlevar(
            SL_DATASET, PREF_SL, target, bbox, lookback_days=14, out_prefix="sl"
        )
        ds = _normalize_lonlat_names(xr.open_dataset(sl_path))
        da = ds[sl_var].astype(float)
        if "time" in da.dims:
            da = da.mean("time", skipna=True)
        da = da.rename("ssh").astype(float)
        da = _ensure_lat_ascending(da)
        layers["ssh"] = da
        print(f"SSH OK: day={sl_day} var={sl_var}")
    except Exception as e:
        print(f"[WARN] SSH unavailable: {e}")

    if "sst" not in layers:
        raise RuntimeError("Cannot proceed: SST is mandatory for this MVP (no SST layer found).")

    # -----------------------
    # Align all layers to SST grid
    # -----------------------
    base = layers["sst"]
    for k in list(layers.keys()):
        if k == "sst":
            continue
        da = layers[k]
        # nearest interpolation (no SciPy)
        try:
            da2 = da.interp(lon=base["lon"], lat=base["lat"], method="nearest")
            layers[k] = da2
        except Exception:
            # if cannot interp, drop it
            print(f"[WARN] Could not align layer {k} to SST grid -> dropping.")
            layers.pop(k, None)

    # -----------------------
    # Feature engineering (gillnet/purse-seine-ish surface logic)
    # -----------------------
    sst = layers["sst"].values
    sst_front = _safe_grad_mag(sst)

    # SST suitability: broad tuna-friendly band (heuristic baseline)
    sst_suit = _gaussian_score(sst, mu=27.0, sigma=2.5)

    feats = []
    weights = []

    # fronts are strong indicators (thermal/ocean color gradients) 1
    feats.append(_norm01(sst_front)); weights.append(0.25)
    feats.append(_norm01(sst_suit));  weights.append(0.20)

    if "chl" in layers:
        chl = layers["chl"].values
        chl_clip = np.clip(chl, 1e-6, None)
        chl_log = np.log10(chl_clip)
        chl_front = _safe_grad_mag(chl_log)

        # CHL suitability: moderate productivity sweet spot (heuristic baseline)
        # (در ادبیات، CHL/SST/SSH به‌عنوان predictors متداول استفاده می‌شوند) 2
        chl_suit = _gaussian_score(chl_log, mu=np.log10(0.2), sigma=0.35)

        feats.append(_norm01(chl_front)); weights.append(0.25)
        feats.append(_norm01(chl_suit));  weights.append(0.15)

    if ("u" in layers) and ("v" in layers):
        u = layers["u"].values
        v = layers["v"].values
        spd = np.sqrt(u*u + v*v)
        # prefer moderate currents (heuristic)
        spd_suit = _gaussian_score(spd, mu=0.6, sigma=0.5)
        feats.append(_norm01(spd_suit)); weights.append(0.10)

    if "ssh" in layers:
        ssh = layers["ssh"].values
        ssh_g = _safe_grad_mag(ssh)
        feats.append(_norm01(ssh_g)); weights.append(0.05)

    # re-normalize weights if some layers missing
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    score01 = np.zeros_like(feats[0], dtype=float)
    for f, w in zip(feats, weights):
        score01 += w * f

    prob = np.clip(score01 * 100.0, 0, 100)

    # -----------------------
    # Top-10 points
    # -----------------------
    flat = prob.ravel()
    good = np.isfinite(flat)
    if not np.any(good):
        raise RuntimeError("All probability values are NaN; cannot produce hotspots.")

    idx = np.argsort(flat[good])[::-1][:10]
    good_idx = np.where(good)[0][idx]
    yy, xx = np.unravel_index(good_idx, prob.shape)

    lats = layers["sst"]["lat"].values
    lons = layers["sst"]["lon"].values

    rows = []
    features = []
    for rank, (y, x) in enumerate(zip(yy, xx), start=1):
        lon = float(lons[x])
        lat = float(lats[y])
        p = float(prob[y, x])
        rows.append({"rank": rank, "lon": lon, "lat": lat, "probability": p})

        features.append({
            "type": "Feature",
            "properties": {"rank": rank, "probability": p},
            "geometry": {"type": "Point", "coordinates": [lon, lat]}
        })

    # save table
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/top10.csv", index=False)

    # save geojson
    out_geo = {"type": "FeatureCollection", "features": features}
    with open(f"{OUT_DIR}/hotspots.geojson", "w", encoding="utf-8") as f:
        json.dump(out_geo, f, ensure_ascii=False)

    # -----------------------
    # Visualization: red->yellow->green (RdYlGn)
    # -----------------------
    Lon, Lat = np.meshgrid(lons, lats)

    plt.figure(figsize=(10, 7))
    plt.pcolormesh(Lon, Lat, prob, shading="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(label="Fishing probability (0-100)")
    plt.title("Daily Tuna Hotspot Probability (heuristic MVP)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # overlay top points
    for r in rows:
        plt.scatter(r["lon"], r["lat"], s=35, marker="o", edgecolor="black")
        plt.text(r["lon"], r["lat"], str(r["rank"]), fontsize=9, ha="left", va="bottom")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/probability.png", dpi=160)
    plt.close()

    # small metadata
    meta = {
        "run_target_date": target.isoformat(),
        "used_layers": sorted(list(layers.keys())),
        "note": "Heuristic MVP probability. Uses latest available day per dataset (auto-backtracking).",
    }
    with open(f"{OUT_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(df)


if __name__ == "__main__":
    main()