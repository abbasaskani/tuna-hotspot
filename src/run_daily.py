import json
import os
import re
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import copernicusmarine as cm

AOI_PATH = "config/aoi.geojson"
OUT_DIR = "docs/latest"
os.makedirs(OUT_DIR, exist_ok=True)

# Dataset IDs (MVP)
SST_DATASET = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
CHL_DATASET = "cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m"

PREF_SST_VARS = ["analysed_sst", "sst", "sea_surface_temperature"]
PREF_CHL_VARS = ["chl", "CHL", "chlorophyll"]


def bbox_from_geojson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    coords = gj["features"][0]["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), max(lons), min(lats), max(lats)


def _first_time_slice(da: xr.DataArray) -> xr.DataArray:
    for dim in list(da.dims):
        if "time" in dim.lower():
            return da.isel({dim: 0})
    return da


def _find_coord_name(da: xr.DataArray, candidates):
    for c in candidates:
        if c in da.coords:
            return c
        if c in da.dims:
            return c
    return None


def _get_lat_lon_names(da: xr.DataArray):
    lat = _find_coord_name(da, ["latitude", "lat", "nav_lat", "y"])
    lon = _find_coord_name(da, ["longitude", "lon", "nav_lon", "x"])
    if not lat or not lon:
        raise RuntimeError(f"Could not detect lat/lon coordinates for variable {da.name}.")
    return lat, lon


def _is_time_oob_error(e: Exception) -> bool:
    s = str(e)
    return ("CoordinatesOutOfDatasetBounds" in e.__class__.__name__) or ("exceed the dataset coordinates" in s)


def _extract_max_date_from_error(e: Exception):
    """
    From error like:
    ... dataset coordinates [2020-08-20 00:00:00+00:00, 2025-12-28 00:00:00+00:00]
    extract '2025-12-28'
    """
    s = str(e)
    m = re.search(r"dataset coordinates \[[^,]+,\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", s)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def _try_open_dataset(dataset_id: str, day, bbox):
    min_lon, max_lon, min_lat, max_lat = bbox
    start = datetime(day.year, day.month, day.day)
    end = start + timedelta(days=1)

    # open_dataset supports time range and bounds :contentReference[oaicite:2]{index=2}
    return cm.open_dataset(
        dataset_id=dataset_id,
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        minimum_depth=0,
        maximum_depth=0,
        start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def find_working_day(dataset_id: str, preferred_day, bbox, max_lookback_days=7):
    """
    Try preferred_day; if out-of-bounds, jump directly to dataset max date if we can parse it.
    Otherwise step back up to max_lookback_days.
    """
    day = preferred_day

    # 1) try preferred day
    try:
        _ = _try_open_dataset(dataset_id, day, bbox)
        return day
    except Exception as e:
        if _is_time_oob_error(e):
            max_day = _extract_max_date_from_error(e)
            if max_day:
                # jump to last available day (fast path)
                try:
                    _ = _try_open_dataset(dataset_id, max_day, bbox)
                    return max_day
                except Exception:
                    pass
        # else: fallthrough to lookback loop

    # 2) look back day-by-day
    for k in range(1, max_lookback_days + 1):
        cand = preferred_day - timedelta(days=k)
        try:
            _ = _try_open_dataset(dataset_id, cand, bbox)
            return cand
        except Exception:
            continue

    raise RuntimeError(
        f"Could not find an available day within {max_lookback_days} days for dataset {dataset_id}."
    )


def pick_var_from_data(dataset_id: str, preferred_vars, day, bbox):
    ds = _try_open_dataset(dataset_id, day, bbox)
    vars_ = list(ds.data_vars)
    if not vars_:
        raise RuntimeError(f"Dataset opened but no data variables found: {dataset_id}")
    for p in preferred_vars:
        if p in vars_:
            return p
    return vars_[0]


def subset_day(dataset_id: str, var: str, day, bbox):
    min_lon, max_lon, min_lat, max_lat = bbox
    start = datetime(day.year, day.month, day.day)
    end = start + timedelta(days=1)

    out_name = f"{dataset_id}_{day.isoformat()}_{var}.nc"
    out_path = f"/tmp/{out_name}"

    # subset expects bounds to overlap dataset; otherwise error :contentReference[oaicite:3]{index=3}
    cm.subset(
        dataset_id=dataset_id,
        variables=[var],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
        minimum_depth=0,
        maximum_depth=0,
        file_format="netcdf",
        output_directory="/tmp",
        output_filename=out_name,
        overwrite=True,
    )
    return out_path


def grad_mag(da: xr.DataArray):
    arr = da.values.astype("float64")
    arr[~np.isfinite(arr)] = np.nan
    gy, gx = np.gradient(arr)
    return np.sqrt(gx * gx + gy * gy)


def safe_norm(a: np.ndarray):
    mx = np.nanmax(a)
    if not np.isfinite(mx) or mx == 0:
        return np.zeros_like(a)
    return a / mx


def main():
    print("Reading AOI...")
    bbox = bbox_from_geojson(AOI_PATH)

    today = datetime.utcnow().date()
    print(f"Processing requested date: {today}")

    # Find available day per dataset (handles the lag)
    sst_day = find_working_day(SST_DATASET, today, bbox, max_lookback_days=7)
    chl_day = find_working_day(CHL_DATASET, today, bbox, max_lookback_days=7)

    # Use common day to avoid mixing days
    day = min(sst_day, chl_day)
    print(f"Selected working date (UTC): {day}  (SST:{sst_day} / CHL:{chl_day})")

    print("Selecting variables (from data_vars)...")
    sst_var = pick_var_from_data(SST_DATASET, PREF_SST_VARS, day, bbox)
    chl_var = pick_var_from_data(CHL_DATASET, PREF_CHL_VARS, day, bbox)
    print(f"Using SST var: {sst_var}")
    print(f"Using CHL var: {chl_var}")

    print("Downloading subsets...")
    sst_nc = subset_day(SST_DATASET, sst_var, day, bbox)
    chl_nc = subset_day(CHL_DATASET, chl_var, day, bbox)

    print("Opening downloaded NetCDFs...")
    ds_sst = xr.open_dataset(sst_nc)
    ds_chl = xr.open_dataset(chl_nc)

    sst = _first_time_slice(ds_sst[sst_var])
    chl = _first_time_slice(ds_chl[chl_var])

    lat_sst, lon_sst = _get_lat_lon_names(sst)
    lat_chl, lon_chl = _get_lat_lon_names(chl)

    print("Regridding SST onto CHL grid...")
    sst_on_chl = sst.interp(
        {lat_sst: chl[lat_chl], lon_sst: chl[lon_chl]},
        method="linear"
    )

    print("Scoring...")
    sst_g = grad_mag(sst_on_chl)
    chl_g = grad_mag(chl)

    chl_vals = chl.values.astype("float64")
    chl_target = np.nanmedian(chl_vals)
    chl_std = np.nanstd(chl_vals) + 1e-9
    chl_score = 1.0 - np.clip(np.abs(chl_vals - chl_target) / chl_std, 0, 1)

    score = (
        0.45 * safe_norm(sst_g) +
        0.45 * safe_norm(chl_g) +
        0.10 * chl_score
    )

    flat = score.ravel()
    flat_idx = np.argsort(flat)[::-1]
    flat_idx = [i for i in flat_idx if np.isfinite(flat[i])][:10]
    yy, xx = np.unravel_index(flat_idx, score.shape)

    lats = chl[lat_chl].values
    lons = chl[lon_chl].values

    features = []
    for rank, (y, x) in enumerate(zip(yy, xx), start=1):
        features.append({
            "type": "Feature",
            "properties": {
                "rank": rank,
                "score": float(score[y, x]),
                "chl": float(chl_vals[y, x]) if np.isfinite(chl_vals[y, x]) else None,
                "sst": float(sst_on_chl.values[y, x]) if np.isfinite(sst_on_chl.values[y, x]) else None,
                "date_utc": day.isoformat()
            },
            "geometry": {"type": "Point", "coordinates": [float(lons[x]), float(lats[y])]}
        })

    out_geo = {"type": "FeatureCollection", "features": features}
    with open(f"{OUT_DIR}/hotspots.geojson", "w", encoding="utf-8") as f:
        json.dump(out_geo, f, ensure_ascii=False, indent=2)

    plt.figure()
    plt.imshow(score, origin="lower")
    plt.title(f"Hotspot score {day.isoformat()} (UTC)")
    plt.colorbar()
    plt.savefig(f"{OUT_DIR}/score.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
