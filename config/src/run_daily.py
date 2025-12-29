import json
import os
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import copernicusmarine as cm

AOI_PATH = "config/aoi.geojson"
OUT_DIR = "docs/latest"
os.makedirs(OUT_DIR, exist_ok=True)

# Dataset IDs (برای MVP)
SST_DATASET = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
CHL_DATASET = "cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m"

def bbox_from_geojson(path: str):
    gj = json.load(open(path, "r", encoding="utf-8"))
    coords = gj["features"][0]["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), max(lons), min(lats), max(lats)

def pick_var(dataset_id: str, prefer: list[str]):
    desc = cm.describe(dataset_id=dataset_id).to_dict()
    vars_ = [v["name"] for v in desc["datasets"][0]["variables"]]
    for p in prefer:
        if p in vars_:
            return p
    return vars_[0]

def subset_day(dataset_id: str, var: str, day: datetime.date, bbox):
    min_lon, max_lon, min_lat, max_lat = bbox
    start = datetime(day.year, day.month, day.day)
    end = start + timedelta(days=1)
    out = f"/tmp/{dataset_id}_{day.isoformat()}_{var}.nc"
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
        output_filename=os.path.basename(out),
        overwrite=True,
    )
    return out

def main():
    bbox = bbox_from_geojson(AOI_PATH)

    day = datetime.utcnow().date()

    sst_var = pick_var(SST_DATASET, prefer=["analysed_sst", "sst"])
    chl_var = pick_var(CHL_DATASET, prefer=["CHL", "chl", "chlorophyll"])

    sst_nc = subset_day(SST_DATASET, sst_var, day, bbox)
    chl_nc = subset_day(CHL_DATASET, chl_var, day, bbox)

    ds_sst = xr.open_dataset(sst_nc)
    ds_chl = xr.open_dataset(chl_nc)

    sst = ds_sst[sst_var].isel(time=0)
    chl = ds_chl[chl_var].isel(time=0)

    def grad_mag(a):
        gy, gx = np.gradient(a.values)
        return np.sqrt(gx*gx + gy*gy)

    sst_g = grad_mag(sst)
    chl_g = grad_mag(chl)

    chl_vals = chl.values
    chl_target = np.nanmedian(chl_vals)
    chl_score = 1.0 - np.clip(np.abs(chl_vals - chl_target) / (np.nanstd(chl_vals)+1e-9), 0, 1)

    score = (0.45 * (sst_g / (np.nanmax(sst_g)+1e-9)) +
             0.45 * (chl_g / (np.nanmax(chl_g)+1e-9)) +
             0.10 * chl_score)

    flat = score.ravel()
    idx = np.argsort(flat)[::-1][:10]
    yy, xx = np.unravel_index(idx, score.shape)

    lats = sst["latitude"].values
    lons = sst["longitude"].values

    features = []
    for i, (y, x) in enumerate(zip(yy, xx), start=1):
        features.append({
            "type": "Feature",
            "properties": {"rank": i, "score": float(score[y, x])},
            "geometry": {"type": "Point", "coordinates": [float(lons[x]), float(lats[y])]}
        })

    out_geo = {"type": "FeatureCollection", "features": features}
    with open(f"{OUT_DIR}/hotspots.geojson", "w", encoding="utf-8") as f:
        json.dump(out_geo, f)

    plt.figure()
    plt.imshow(score, origin="lower")
    plt.title(f"Hotspot score {day.isoformat()}")
    plt.colorbar()
    plt.savefig(f"{OUT_DIR}/score.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Done", day.isoformat())

if __name__ == "__main__":
    main()
