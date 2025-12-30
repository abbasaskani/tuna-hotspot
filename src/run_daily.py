import json
import os
from datetime import datetime, timedelta, date

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.path import Path

import copernicusmarine as cm


# ------------------------
# Paths
# ------------------------
AOI_PATH = "config/aoi.geojson"
OUT_DIR = "docs/latest"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------
# Dataset IDs (Stable daily ops: use forecast models)
# 1) Physics (temp/currents/sea level) - daily
# 2) BGC optics (chlorophyll) - daily
# 3) Waves (SWH) - 3-hourly
# ------------------------
TEMP_DATASET = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"          # temperature
CUR_DATASET  = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"            # currents
SL_DATASET   = "cmems_mod_glo_phy_anfc_merged-sl_PT1H-i"              # sea level (hourly)

CHL_DATASET  = "cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m"           # chlorophyll optics (daily)

WAV_DATASET  = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"               # waves (3-hourly)

# Variable preferences (we will pick the first one that exists)
PREF_TEMP_VARS = ["thetao", "temperature", "to"]
PREF_U_VARS    = ["uo", "u", "eastward_sea_water_velocity"]
PREF_V_VARS    = ["vo", "v", "northward_sea_water_velocity"]
PREF_SL_VARS   = ["zos", "sea_surface_height", "ssh", "sla"]
PREF_CHL_VARS  = ["CHL", "chl", "chlorophyll", "chlor_a"]
PREF_SWH_VARS  = ["VHM0", "swh", "hs", "significant_height_of_combined_wind_waves_and_swell"]

# Scoring weights (baseline – can be tuned later)
W_FRONT_TEMP = 0.22
W_FRONT_CHL  = 0.22
W_EDDY_SL    = 0.18
W_CONVERGENCE = 0.10
W_SUIT_TEMP  = 0.10
W_SUIT_CHL   = 0.10
W_WAVE_PENALTY = 0.08  # subtracted

# Operational knobs
TOP_N = 10
LOOKBACK_DAYS = 21  # robust against data lags


def read_aoi_polygon(path: str):
    gj = json.load(open(path, "r", encoding="utf-8"))
    coords = gj["features"][0]["geometry"]["coordinates"][0]
    return coords  # list of [lon, lat]


def bbox_from_polygon(coords):
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), max(lons), min(lats), max(lats)


def polygon_mask(lons2d, lats2d, polygon_coords):
    # polygon coords: [[lon,lat], ...]
    poly = Path(np.array(polygon_coords))
    pts = np.vstack([lons2d.ravel(), lats2d.ravel()]).T
    inside = poly.contains_points(pts).reshape(lons2d.shape)
    return inside


def safe_pick_var(ds: xr.Dataset, prefer):
    for p in prefer:
        if p in ds.data_vars:
            return p
    return list(ds.data_vars.keys())[0]


def standardize_latlon(ds: xr.Dataset):
    ren = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        ren["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        ren["lon"] = "longitude"
    if ren:
        ds = ds.rename(ren)
    return ds


def try_open_small(dataset_id: str, day: date, bbox, prefer_vars, depth0=None):
    """
    Open a tiny subset (single day) to discover variable names robustly.
    Uses coordinates_selection_method='nearest' to avoid boundary errors.
    """
    min_lon, max_lon, min_lat, max_lat = bbox
    mid_lon = (min_lon + max_lon) / 2.0
    mid_lat = (min_lat + max_lat) / 2.0

    start = datetime(day.year, day.month, day.day)
    end = start  # IMPORTANT: avoid end+1day boundary issues

    kwargs = dict(
        dataset_id=dataset_id,
        minimum_longitude=mid_lon, maximum_longitude=mid_lon,
        minimum_latitude=mid_lat, maximum_latitude=mid_lat,
        start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
        coordinates_selection_method="nearest",
    )
    if depth0 is not None:
        kwargs.update(minimum_depth=depth0, maximum_depth=depth0)

    ds = cm.open_dataset(**kwargs)
    ds = standardize_latlon(ds)
    v = safe_pick_var(ds, prefer_vars)
    return v


def subset_to_netcdf(dataset_id: str, var: str, day: date, bbox, out_path: str, depth0=None):
    """
    Download subset to NetCDF.
    """
    min_lon, max_lon, min_lat, max_lat = bbox
    start = datetime(day.year, day.month, day.day)
    end = start  # IMPORTANT: same-day selection

    kwargs = dict(
        dataset_id=dataset_id,
        variables=[var],
        minimum_longitude=min_lon, maximum_longitude=max_lon,
        minimum_latitude=min_lat, maximum_latitude=max_lat,
        start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
        file_format="netcdf",
        output_directory=os.path.dirname(out_path),
        output_filename=os.path.basename(out_path),
        overwrite=True,
        coordinates_selection_method="nearest",
    )
    if depth0 is not None:
        kwargs.update(minimum_depth=depth0, maximum_depth=depth0)

    cm.subset(**kwargs)
    return out_path


def find_working_day(dataset_id: str, bbox, prefer_vars, depth0=None, lookback_days=LOOKBACK_DAYS):
    """
    Find the most recent day within lookback that can be opened.
    """
    today = datetime.utcnow().date()
    last_err = None
    for k in range(0, lookback_days + 1):
        d = today - timedelta(days=k)
        try:
            _ = try_open_small(dataset_id, d, bbox, prefer_vars, depth0=depth0)
            return d
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not find an available day within {lookback_days} days for {dataset_id}. Last error: {last_err}")


def grad_mag(da: xr.DataArray):
    a = da.values.astype("float64")
    gy, gx = np.gradient(a)
    return np.sqrt(gx * gx + gy * gy)


def norm01(x):
    x = np.array(x, dtype="float64")
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return (x - mn) / (mx - mn + 1e-12)


def gaussian_suitability(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)


def main():
    print("Reading AOI...")
    poly = read_aoi_polygon(AOI_PATH)
    bbox = bbox_from_polygon(poly)

    # 1) Pick a common date that exists across all needed datasets
    print("Finding latest available dates...")
    # Physics: use surface depth0 ~ 0 (nearest)
    d_temp = find_working_day(TEMP_DATASET, bbox, PREF_TEMP_VARS, depth0=0)
    d_cur  = find_working_day(CUR_DATASET,  bbox, PREF_U_VARS, depth0=0)   # u check
    d_sl   = find_working_day(SL_DATASET,   bbox, PREF_SL_VARS, depth0=None)
    d_chl  = find_working_day(CHL_DATASET,  bbox, PREF_CHL_VARS, depth0=None)
    d_wav  = find_working_day(WAV_DATASET,  bbox, PREF_SWH_VARS, depth0=None)

    day = min(d_temp, d_cur, d_sl, d_chl, d_wav)
    print(f"Processing date (common): {day.isoformat()}")

    # 2) Discover variable names safely
    print("Selecting variables...")
    temp_var = try_open_small(TEMP_DATASET, day, bbox, PREF_TEMP_VARS, depth0=0)
    u_var    = try_open_small(CUR_DATASET,  day, bbox, PREF_U_VARS, depth0=0)
    v_var    = try_open_small(CUR_DATASET,  day, bbox, PREF_V_VARS, depth0=0)
    sl_var   = try_open_small(SL_DATASET,   day, bbox, PREF_SL_VARS, depth0=None)
    chl_var  = try_open_small(CHL_DATASET,  day, bbox, PREF_CHL_VARS, depth0=None)
    swh_var  = try_open_small(WAV_DATASET,  day, bbox, PREF_SWH_VARS, depth0=None)

    # 3) Download subsets
    print("Downloading subsets...")
    temp_nc = subset_to_netcdf(TEMP_DATASET, temp_var, day, bbox, "/tmp/temp.nc", depth0=0)
    u_nc    = subset_to_netcdf(CUR_DATASET,  u_var,    day, bbox, "/tmp/u.nc",    depth0=0)
    v_nc    = subset_to_netcdf(CUR_DATASET,  v_var,    day, bbox, "/tmp/v.nc",    depth0=0)
    sl_nc   = subset_to_netcdf(SL_DATASET,   sl_var,   day, bbox, "/tmp/sl.nc",   depth0=None)
    chl_nc  = subset_to_netcdf(CHL_DATASET,  chl_var,  day, bbox, "/tmp/chl.nc",  depth0=None)
    wav_nc  = subset_to_netcdf(WAV_DATASET,  swh_var,  day, bbox, "/tmp/wav.nc",  depth0=None)

    # 4) Load datasets
    ds_temp = standardize_latlon(xr.open_dataset(temp_nc))
    ds_u    = standardize_latlon(xr.open_dataset(u_nc))
    ds_v    = standardize_latlon(xr.open_dataset(v_nc))
    ds_sl   = standardize_latlon(xr.open_dataset(sl_nc))
    ds_chl  = standardize_latlon(xr.open_dataset(chl_nc))
    ds_wav  = standardize_latlon(xr.open_dataset(wav_nc))

    # Time selection (some are hourly/3-hourly)
    def first_time(da):
        if "time" in da.dims:
            return da.isel(time=0)
        return da

    temp = first_time(ds_temp[temp_var])
    u    = first_time(ds_u[u_var])
    v    = first_time(ds_v[v_var])
    sl   = first_time(ds_sl[sl_var])
    chl  = first_time(ds_chl[chl_var])
    swh  = first_time(ds_wav[swh_var])

    # 5) Regrid CHL onto physics grid if needed
    # (physics is 1/12deg, chl is 1/4deg)
    if (chl.sizes.get("latitude") != temp.sizes.get("latitude")) or (chl.sizes.get("longitude") != temp.sizes.get("longitude")):
        chl = chl.interp(latitude=temp["latitude"], longitude=temp["longitude"], method="linear")

    if (swh.sizes.get("latitude") != temp.sizes.get("latitude")) or (swh.sizes.get("longitude") != temp.sizes.get("longitude")):
        swh = swh.interp(latitude=temp["latitude"], longitude=temp["longitude"], method="linear")

    if (sl.sizes.get("latitude") != temp.sizes.get("latitude")) or (sl.sizes.get("longitude") != temp.sizes.get("longitude")):
        sl = sl.interp(latitude=temp["latitude"], longitude=temp["longitude"], method="linear")

    if (u.sizes.get("latitude") != temp.sizes.get("latitude")) or (u.sizes.get("longitude") != temp.sizes.get("longitude")):
        u = u.interp(latitude=temp["latitude"], longitude=temp["longitude"], method="linear")
    if (v.sizes.get("latitude") != temp.sizes.get("latitude")) or (v.sizes.get("longitude") != temp.sizes.get("longitude")):
        v = v.interp(latitude=temp["latitude"], longitude=temp["longitude"], method="linear")

    # 6) AOI polygon mask
    lats = temp["latitude"].values
    lons = temp["longitude"].values
    Lon, Lat = np.meshgrid(lons, lats)
    inside = polygon_mask(Lon, Lat, poly)

    # Mask outside
    def apply_mask(arr):
        a = arr.values.astype("float64")
        a[~inside] = np.nan
        return a

    temp_a = apply_mask(temp)
    chl_a  = apply_mask(chl)
    sl_a   = apply_mask(sl)
    u_a    = apply_mask(u)
    v_a    = apply_mask(v)
    swh_a  = apply_mask(swh)

    # 7) Features
    front_temp = grad_mag(xr.DataArray(temp_a))
    front_chl  = grad_mag(xr.DataArray(chl_a))
    eddy_sl    = grad_mag(xr.DataArray(sl_a))

    # Convergence ~ -div(U,V) (simple finite diff proxy)
    dudx = np.gradient(u_a, axis=1)
    dvdy = np.gradient(v_a, axis=0)
    convergence = -(dudx + dvdy)

    # Suitability windows (baseline, tunable)
    # Temp peak ~ 28C, sigma ~ 2.5C (broad for "all tuna" MVP)
    suit_temp = gaussian_suitability(temp_a, mu=28.0, sigma=2.5)

    # CHL: work on log-scale; target ~ 0.15 mg/m3 (broad)
    chl_safe = np.where(chl_a <= 0, np.nan, chl_a)
    suit_chl = gaussian_suitability(np.log10(chl_safe), mu=np.log10(0.15), sigma=0.35)

    # Wave penalty (higher waves -> worse for net operations)
    wave_pen = np.clip(swh_a / 3.0, 0, 1)

    # 8) Normalize + combine
    F1 = norm01(front_temp)
    F2 = norm01(front_chl)
    F3 = norm01(eddy_sl)
    F4 = norm01(convergence)

    score = (
        W_FRONT_TEMP * F1
        + W_FRONT_CHL * F2
        + W_EDDY_SL * F3
        + W_CONVERGENCE * F4
        + W_SUIT_TEMP * suit_temp
        + W_SUIT_CHL * suit_chl
        - W_WAVE_PENALTY * wave_pen
    )

    # Convert to 0–100 (relative within AOI)
    prob = 100.0 * norm01(score)

    # 9) Top-N points
    flat = prob.ravel()
    idx = np.argsort(flat)[::-1][:TOP_N]
    yy, xx = np.unravel_index(idx, prob.shape)

    top_features = []
    rows = []
    for rank, (y, x) in enumerate(zip(yy, xx), start=1):
        lon = float(lons[x])
        lat = float(lats[y])
        p = float(prob[y, x])
        rows.append((rank, lon, lat, p, float(temp_a[y, x]), float(chl_a[y, x]), float(sl_a[y, x]), float(swh_a[y, x])))

        top_features.append({
            "type": "Feature",
            "properties": {
                "rank": rank,
                "probability_0_100": p,
                "temp_C": float(temp_a[y, x]),
                "chl_mg_m3": float(chl_a[y, x]),
                "sea_level_m": float(sl_a[y, x]),
                "swh_m": float(swh_a[y, x]),
                "date": day.isoformat()
            },
            "geometry": {"type": "Point", "coordinates": [lon, lat]}
        })

    out_geo = {"type": "FeatureCollection", "features": top_features}
    with open(f"{OUT_DIR}/hotspots.geojson", "w", encoding="utf-8") as f:
        json.dump(out_geo, f, ensure_ascii=False)

    # CSV
    with open(f"{OUT_DIR}/top10.csv", "w", encoding="utf-8") as f:
        f.write("rank,lon,lat,probability_0_100,temp_C,chl_mg_m3,sea_level_m,swh_m\n")
        for r in rows:
            f.write(",".join([str(x) for x in r]) + "\n")

    # 10) Plot (red->yellow->green) + top labels
    plt.figure(figsize=(10, 7))
    plt.imshow(prob, origin="lower", vmin=0, vmax=100, cmap="RdYlGn")
    plt.colorbar(label="Fishing probability (0–100, relative within AOI)")
    plt.title(f"Tuna Hotspot Probability | {day.isoformat()}")

    # overlay top points
    for rank, (y, x) in enumerate(zip(yy, xx), start=1):
        plt.scatter([x], [y], s=35, marker="o", edgecolors="black")
        plt.text(x + 1, y + 1, str(rank), fontsize=10, weight="bold")

    plt.savefig(f"{OUT_DIR}/probability.png", dpi=160, bbox_inches="tight")
    plt.close()

    # 11) Minimal report
    with open(f"{OUT_DIR}/report.txt", "w", encoding="utf-8") as f:
        f.write(f"Date used (common): {day.isoformat()}\n")
        f.write("Top points:\n")
        for r in rows:
            f.write(f"#{r[0]} lon={r[1]:.5f} lat={r[2]:.5f} prob={r[3]:.1f} temp={r[4]:.2f}C chl={r[5]:.3f} sl={r[6]:.3f}m swh={r[7]:.2f}m\n")

    print("Done.")


if __name__ == "__main__":
    main()
