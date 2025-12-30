import json
import os
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import copernicusmarine as cm

# -----------------------
# Paths
# -----------------------
AOI_PATH = "config/aoi.geojson"
OUT_DIR = "docs/latest"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Dataset IDs (MVP)
# -----------------------
SST_DATASET = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
CHL_DATASET = "cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m"


# -----------------------
# Helpers
# -----------------------
def bbox_from_geojson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    coords = gj["features"][0]["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    return min(lons), max(lons), min(lats), max(lats)


def pick_var(dataset_id: str, prefer: list[str]) -> str:
    """
    Safely pick a variable name from Copernicus dataset
    """
    desc = cm.describe(dataset_id=dataset_id).dict()

    datasets = desc.get("datasets", [])
    if not datasets:
        raise RuntimeError(f"No datasets found in describe() for {dataset_id}")

    variables = datasets[0].get("variables", [])
    var_names = [v["name"] for v in variables]

    for p in prefer:
        if p in var_names:
            return p

    # fallback: first available variable
    return var_names[0]


def subset_day(dataset_id: str, var: str, day, bbox):
    min_lon, max_lon, min_lat, max_lat = bbox

    start = datetime(day.year, day.month, day.day)
    end = start + timedelta(days=1)

    out_name = f"{dataset_id}_{day.isoformat()}_{var}.nc"

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

    return f"/tmp/{out_name}"


# -----------------------
# Main pipeline
# -----------------------
def main():
    print("Reading AOI...")
    bbox = bbox_from_geojson(AOI_PATH)

    day = datetime.utcnow().date()
    print("Processing date:", day)

    print("Selecting variables...")
    sst_var = pick_var(SST_DATASET, ["analysed_sst", "sst"])
    chl_var = pick_var(CHL_DATASET, ["CHL", "chl", "chlorophyll"])

    print("Downloading SST...")
    sst_nc = subset_day(SST_DATASET, sst_var, day, bbox)

    print("Downloading CHL...")
    chl_nc = subset_day(CHL_DATASET, chl_var, day, bbox)

    print("Opening datasets...")
    ds_sst = xr.open_dataset(sst_nc)
    ds_chl = xr.open_dataset(chl_nc)

    sst = ds_sst[sst_var].isel(time=0)
    chl = ds_chl[chl_var].isel(time=0)

    def grad_mag(arr):
        gy, gx = np.gradient(arr.values)
        return np.sqrt(gx**2 + gy**2)

    print("Computing gradients...")
    sst_g = grad_mag(sst)
    chl_g = grad_mag(chl)

    chl_vals = chl.values
    chl_target = np.nanmedian(chl_vals)
    chl_score = 1.0 - np.clip(
        np.abs(chl_vals - chl_target) / (np.nanstd(chl_vals) + 1e-9),
        0,
        1,
    )

    score = (
        0.45 * (sst_g / (np.nanmax(sst_g) + 1e-9))
        + 0.45 * (chl_g / (np.nanmax(chl_g) + 1e-9))
        + 0.10 * chl_score
    )

    print("Extracting top hotspots...")
    flat = score.ravel()
    idx = np.argsort(flat)[::-1][:10]
    yy, xx = np.unravel_index(idx, score.shape)

    lats = sst["latitude"].values
    lons = sst["longitude"].values

    features = []
    for i, (y, x) in enumerate(zip(yy, xx), start=1):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "rank": i,
                    "score": float(score[y, x]),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lons[x]), float(lats[y])],
                },
            }
        )

    out_geo = {"type": "FeatureCollection", "features": features}

    with open(f"{OUT_DIR}/hotspots.geojson", "w", encoding="utf-8") as f:
        json.dump(out_geo, f, indent=2)

    print("Saving score map...")
    plt.figure(figsize=(8, 6))
    plt.imshow(score, origin="lower")
    plt.title(f"Tuna Hotspot Score – {day.isoformat()}")
    plt.colorbar(label="Score")
    plt.savefig(f"{OUT_DIR}/score.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
