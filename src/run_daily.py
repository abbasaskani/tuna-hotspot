import json
import os
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import copernicusmarine as cm

# --------------------
# Paths
# --------------------
AOI_PATH = "config/aoi.geojson"     # مطمئن شوید فایل واقعاً اینجاست
OUT_DIR = "docs/latest"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------
# Dataset IDs (MVP)
# --------------------
SST_DATASET = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
CHL_DATASET = "cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m"

PREF_SST_VARS = ["analysed_sst", "sst", "sea_surface_temperature"]
PREF_CHL_VARS = ["chl", "CHL", "chlorophyll"]

# --------------------
# Helpers
# --------------------
def _model_to_dict(obj):
    # CopernicusMarineCatalogue is a Pydantic model
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj

def bbox_from_geojson(path: str):
    gj = json.load(open(path, "r", encoding="utf-8"))
    coords = gj["features"][0]["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), max(lons), min(lats), max(lats)

def list_variables(dataset_id: str):
    cat = cm.describe(dataset_id=dataset_id, disable_progress_bar=True)
    d = _model_to_dict(cat)

    vars_found = []
    for p in d.get("products", []) or []:
        for ds in p.get("datasets", []) or []:
            for v in ds.get("variables", []) or []:
                name = v.get("name")
                if name:
                    vars_found.append(name)

    # اگر چیزی پیدا نکردیم، یعنی یا dataset_id غلط است یا describe ساختار دیگری داده
    return sorted(set(vars_found))

def pick_var(dataset_id: str, preferred: list[str]):
    vars_ = list_variables(dataset_id)
    if not vars_:
        raise RuntimeError(
            f"Could not read variables for dataset_id={dataset_id}. "
            f"Either dataset_id is invalid/retired, or the account has no access."
        )
    for p in preferred:
        if p in vars_:
            return p
    return vars_[0]

def subset_day(dataset_id: str, var: str, day: datetime.date, bbox):
    min_lon, max_lon, min_lat, max_lat = bbox
    start = datetime(day.year, day.month, day.day)
    end = start + timedelta(days=1)

    out_name = f"{dataset_id}_{day.isoformat()}_{var}.nc"
    out_path = f"/tmp/{out_name}"

    cm.subset(
        dataset_id=dataset_id,
        variables=[var],
        minimum_longitude=min_lon, maximum_longitude=max_lon,
        minimum_latitude=min_lat, maximum_latitude=max_lat,
        start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
        minimum_depth=0, maximum_depth=0,
        file_format="netcdf",
        output_directory="/tmp",
        output_filename=out_name,
        overwrite=True,
        disable_progress_bar=True,
    )
    return out_path

def _first_time_slice(da: xr.DataArray) -> xr.DataArray:
    # time dim names vary across datasets
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

# --------------------
# Main
# --------------------
def main():
    print("Reading AOI...")
    bbox = bbox_from_geojson(AOI_PATH)

    day = datetime.utcnow().date()
    print(f"Processing date: {day}")
    print("Selecting variables...")

    sst_var = pick_var(SST_DATASET, PREF_SST_VARS)
    chl_var = pick_var(CHL_DATASET, PREF_CHL_VARS)

    print(f"Using SST var: {sst_var}")
    print(f"Using CHL var: {chl_var}")

    print("Downloading subsets...")
    sst_nc = subset_day(SST_DATASET, sst_var, day, bbox)
    chl_nc = subset_day(CHL_DATASET, chl_var, day, bbox)

    print("Opening datasets...")
    ds_sst = xr.open_dataset(sst_nc)
    ds_chl = xr.open_dataset(chl_nc)

    sst = _first_time_slice(ds_sst[sst_var])
    chl = _first_time_slice(ds_chl[chl_var])

    # Align grids: interpolate SST onto CHL grid (CHL is usually coarser)
    lat_sst, lon_sst = _get_lat_lon_names(sst)
    lat_chl, lon_chl = _get_lat_lon_names(chl)

    print("Regridding SST onto CHL grid...")
    sst_on_chl = sst.interp(
        {lat_sst: chl[lat_chl], lon_sst: chl[lon_chl]},
        method="linear"
    )

    # Compute gradients and scoring on CHL grid
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

    # Pick Top-10 non-nan
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
            "geometry": {
                "type": "Point",
                "coordinates": [float(lons[x]), float(lats[y])]
            }
        })

    out_geo = {"type": "FeatureCollection", "features": features}
    with open(f"{OUT_DIR}/hotspots.geojson", "w", encoding="utf-8") as f:
        json.dump(out_geo, f, ensure_ascii=False)

    # Plot
    plt.figure()
    plt.imshow(score, origin="lower")
    plt.title(f"Hotspot score {day.isoformat()}")
    plt.colorbar()
    plt.savefig(f"{OUT_DIR}/score.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Done.")

if __name__ == "__main__":
    main()
