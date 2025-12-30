import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.path import Path

import copernicusmarine as cm

AOI_PATH = "config/aoi.geojson"
OUT_DIR = "docs/latest"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# DATASETS (Robust MVP)
# ---------------------------
# 1) SST: MetOffice L4 NRT
SST_DATASET = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
PREF_SST_VARS = ["analysed_sst", "sst", "sea_surface_temperature"]

# 2) Chlorophyll: Satellite NRT L4 gapfree (stable daily)
# Product: OCEANCOLOUR_GLO_BGC_L4_NRT_009_102 (dataset below)
CHL_DATASET = "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"
PREF_CHL_VARS = ["CHL", "chl", "chlor_a", "chlorophyll"]

# 3) Surface currents (hourly, merged surface)
UV_DATASET = "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i"
PREF_U_VARS = ["uo", "u", "eastward_sea_water_velocity"]
PREF_V_VARS = ["vo", "v", "northward_sea_water_velocity"]

# 4) Sea level (hourly, merged)
SL_DATASET = "cmems_mod_glo_phy_anfc_merged-sl_PT1H-i"
PREF_SL_VARS = ["zos", "sla", "sea_surface_height", "adt"]

# 5) Waves (if available; optional but recommended)
WAV_DATASET = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
PREF_WAV_VARS = ["VHM0", "swh", "significant_height_of_combined_wind_waves_and_swell"]

# 6) Wind (NRT hourly L4; optional)
WIND_DATASET = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"
PREF_WIND_U = ["eastward_wind", "u10", "u"]
PREF_WIND_V = ["northward_wind", "v10", "v"]

# ---------------------------
# Utility
# ---------------------------

def read_aoi(aoi_path: str):
    gj = json.load(open(aoi_path, "r", encoding="utf-8"))
    poly = gj["features"][0]["geometry"]["coordinates"][0]  # list of [lon, lat]
    lons = [p[0] for p in poly]
    lats = [p[1] for p in poly]
    bbox = (min(lons), max(lons), min(lats), max(lats))
    return poly, bbox

def get_lon_lat_names(ds: xr.Dataset):
    # Common coordinate names in CMEMS
    lon_candidates = ["longitude", "lon", "LONGITUDE", "x"]
    lat_candidates = ["latitude", "lat", "LATITUDE", "y"]
    lon = next((c for c in lon_candidates if c in ds.coords), None)
    lat = next((c for c in lat_candidates if c in ds.coords), None)
    if lon is None or lat is None:
        raise RuntimeError(f"Could not detect lon/lat coords. coords={list(ds.coords)}")
    return lon, lat

def pick_var_from_dataset(ds: xr.Dataset, prefer: list[str]):
    vars_ = list(ds.data_vars.keys())
    for p in prefer:
        if p in vars_:
            return p
    # fallback: pick first variable
    if not vars_:
        raise RuntimeError("Dataset has no data_vars.")
    return vars_[0]

def to_2d_field(da: xr.DataArray):
    # reduce time/depth dimensions if present
    if "time" in da.dims:
        da = da.isel(time=0)
    if "depth" in da.dims:
        da = da.isel(depth=0)
    return da

def open_point_nearest(dataset_id: str, when: datetime, lon0: float, lat0: float):
    # minimal load (point), nearest selection avoids out-of-bounds failures
    # coordinates_selection_method applies to lon/lat/time/depth per CMEMS toolbox docs
    ds = cm.open_dataset(
        dataset_id=dataset_id,
        minimum_longitude=lon0, maximum_longitude=lon0,
        minimum_latitude=lat0, maximum_latitude=lat0,
        start_datetime=when.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=when.strftime("%Y-%m-%dT%H:%M:%S"),
        coordinates_selection_method="nearest",
    )
    return ds

def find_latest_available_time(dataset_id: str, bbox, lookback_days: int = 30):
    # Some CMEMS products lag by ~1-3 days; wind can lag more.
    min_lon, max_lon, min_lat, max_lat = bbox
    lon0 = (min_lon + max_lon) / 2.0
    lat0 = (min_lat + max_lat) / 2.0

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    last_err = None
    for d in range(lookback_days + 1):
        cand = now - timedelta(days=d)
        try:
            ds = open_point_nearest(dataset_id, cand, lon0, lat0)
            # If dataset has time coordinate, return nearest actual time
            if "time" in ds.coords:
                t = ds["time"].values
                # numpy datetime64 -> python datetime
                t0 = np.atleast_1d(t)[0]
                # convert safely
                when = np.datetime64(t0).astype("datetime64[s]").astype(datetime).replace(tzinfo=timezone.utc)
                return when
            # No time coord (static) -> return cand
            return cand
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not find available time within {lookback_days} days for {dataset_id}. Last error: {last_err}")

def subset_to_netcdf(dataset_id: str, variables: list[str], when: datetime, bbox, out_path: str):
    min_lon, max_lon, min_lat, max_lat = bbox
    cm.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=min_lon, maximum_longitude=max_lon,
        minimum_latitude=min_lat, maximum_latitude=max_lat,
        start_datetime=when.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=when.strftime("%Y-%m-%dT%H:%M:%S"),
        file_format="netcdf",
        output_directory=os.path.dirname(out_path),
        output_filename=os.path.basename(out_path),
        overwrite=True,
        coordinates_selection_method="nearest",
    )
    return out_path

def polygon_mask(poly_lonlat, grid_lons_2d, grid_lats_2d):
    poly = np.array(poly_lonlat, dtype=float)
    path = Path(poly)  # expects Nx2 (lon,lat)
    pts = np.column_stack([grid_lons_2d.ravel(), grid_lats_2d.ravel()])
    inside = path.contains_points(pts).reshape(grid_lons_2d.shape)
    return inside

def normalize01(a):
    a = np.array(a, dtype=float)
    m = np.nanmin(a)
    M = np.nanmax(a)
    return (a - m) / (M - m + 1e-12)

def grad_mag(a2d):
    gy, gx = np.gradient(a2d)
    return np.sqrt(gx * gx + gy * gy)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ---------------------------
# Main
# ---------------------------
def main():
    print("Reading AOI...")
    poly, bbox = read_aoi(AOI_PATH)

    print("Finding latest available dates...")
    t_sst = find_latest_available_time(SST_DATASET, bbox, lookback_days=10)
    t_chl = find_latest_available_time(CHL_DATASET, bbox, lookback_days=10)
    t_uv  = find_latest_available_time(UV_DATASET,  bbox, lookback_days=10)
    t_sl  = find_latest_available_time(SL_DATASET,  bbox, lookback_days=10)

    # waves/wind are optional (may lag more)
    try:
        t_wav = find_latest_available_time(WAV_DATASET, bbox, lookback_days=14)
    except Exception:
        t_wav = None

    try:
        t_wind = find_latest_available_time(WIND_DATASET, bbox, lookback_days=21)
    except Exception:
        t_wind = None

    meta = {
        "sst_time_used_utc": t_sst.isoformat(),
        "chl_time_used_utc": t_chl.isoformat(),
        "uv_time_used_utc":  t_uv.isoformat(),
        "sl_time_used_utc":  t_sl.isoformat(),
        "wav_time_used_utc": t_wav.isoformat() if t_wav else None,
        "wind_time_used_utc": t_wind.isoformat() if t_wind else None,
        "bbox": {"min_lon": bbox[0], "max_lon": bbox[1], "min_lat": bbox[2], "max_lat": bbox[3]},
    }

    print("Downloading subsets...")
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    # SST
    ds_sst_pt = open_point_nearest(SST_DATASET, t_sst, (bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2)
    sst_var = pick_var_from_dataset(ds_sst_pt, PREF_SST_VARS)
    meta["sst_var"] = sst_var
    sst_nc = subset_to_netcdf(SST_DATASET, [sst_var], t_sst, bbox, f"{tmp_dir}/sst.nc")
    ds_sst = xr.open_dataset(sst_nc)
    lon_name, lat_name = get_lon_lat_names(ds_sst)
    sst = to_2d_field(ds_sst[sst_var])

    # CHL
    ds_chl_pt = open_point_nearest(CHL_DATASET, t_chl, (bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2)
    chl_var = pick_var_from_dataset(ds_chl_pt, PREF_CHL_VARS)
    meta["chl_var"] = chl_var
    chl_nc = subset_to_netcdf(CHL_DATASET, [chl_var], t_chl, bbox, f"{tmp_dir}/chl.nc")
    ds_chl = xr.open_dataset(chl_nc)
    chl = to_2d_field(ds_chl[chl_var])

    # Regrid CHL onto SST grid (if grids differ)
    if (lon_name in ds_chl.coords) and (lat_name in ds_chl.coords):
        chl = chl.interp({lon_name: ds_sst[lon_name], lat_name: ds_sst[lat_name]})
    else:
        # try auto-detect on CHL dataset
        chl_lon, chl_lat = get_lon_lat_names(ds_chl)
        chl = chl.interp({chl_lon: ds_sst[lon_name], chl_lat: ds_sst[lat_name]})
        chl = chl.rename({chl_lon: lon_name, chl_lat: lat_name})

    # UV currents (surface)
    ds_uv_pt = open_point_nearest(UV_DATASET, t_uv, (bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2)
    u_var = pick_var_from_dataset(ds_uv_pt, PREF_U_VARS)
    v_var = pick_var_from_dataset(ds_uv_pt, PREF_V_VARS)
    meta["u_var"] = u_var
    meta["v_var"] = v_var
    uv_nc = subset_to_netcdf(UV_DATASET, [u_var, v_var], t_uv, bbox, f"{tmp_dir}/uv.nc")
    ds_uv = xr.open_dataset(uv_nc)
    u = to_2d_field(ds_uv[u_var])
    v = to_2d_field(ds_uv[v_var])
    # regrid to SST grid if needed
    uv_lon, uv_lat = get_lon_lat_names(ds_uv)
    if uv_lon != lon_name or uv_lat != lat_name:
        u = u.interp({uv_lon: ds_sst[lon_name], uv_lat: ds_sst[lat_name]})
        v = v.interp({uv_lon: ds_sst[lon_name], uv_lat: ds_sst[lat_name]})
        u = u.rename({uv_lon: lon_name, uv_lat: lat_name})
        v = v.rename({uv_lon: lon_name, uv_lat: lat_name})

    # Sea level
    ds_sl_pt = open_point_nearest(SL_DATASET, t_sl, (bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2)
    sl_var = pick_var_from_dataset(ds_sl_pt, PREF_SL_VARS)
    meta["sl_var"] = sl_var
    sl_nc = subset_to_netcdf(SL_DATASET, [sl_var], t_sl, bbox, f"{tmp_dir}/sl.nc")
    ds_sl = xr.open_dataset(sl_nc)
    sl = to_2d_field(ds_sl[sl_var])
    sl_lon, sl_lat = get_lon_lat_names(ds_sl)
    if sl_lon != lon_name or sl_lat != lat_name:
        sl = sl.interp({sl_lon: ds_sst[lon_name], sl_lat: ds_sst[lat_name]})
        sl = sl.rename({sl_lon: lon_name, sl_lat: lat_name})

    # Waves (optional)
    wav = None
    wav_var = None
    if t_wav:
        try:
            ds_wav_pt = open_point_nearest(WAV_DATASET, t_wav, (bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2)
            wav_var = pick_var_from_dataset(ds_wav_pt, PREF_WAV_VARS)
            meta["wav_var"] = wav_var
            wav_nc = subset_to_netcdf(WAV_DATASET, [wav_var], t_wav, bbox, f"{tmp_dir}/wav.nc")
            ds_wav = xr.open_dataset(wav_nc)
            wav = to_2d_field(ds_wav[wav_var])
            wlon, wlat = get_lon_lat_names(ds_wav)
            if wlon != lon_name or wlat != lat_name:
                wav = wav.interp({wlon: ds_sst[lon_name], wlat: ds_sst[lat_name]})
                wav = wav.rename({wlon: lon_name, wlat: lat_name})
        except Exception:
            wav = None

    # Wind (optional)
    wind_speed = None
    if t_wind:
        try:
            ds_wind_pt = open_point_nearest(WIND_DATASET, t_wind, (bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2)
            wu = pick_var_from_dataset(ds_wind_pt, PREF_WIND_U)
            wv = pick_var_from_dataset(ds_wind_pt, PREF_WIND_V)
            meta["wind_u_var"] = wu
            meta["wind_v_var"] = wv
            wind_nc = subset_to_netcdf(WIND_DATASET, [wu, wv], t_wind, bbox, f"{tmp_dir}/wind.nc")
            ds_wind = xr.open_dataset(wind_nc)
            w_u = to_2d_field(ds_wind[wu])
            w_v = to_2d_field(ds_wind[wv])
            wlon, wlat = get_lon_lat_names(ds_wind)
            if wlon != lon_name or wlat != lat_name:
                w_u = w_u.interp({wlon: ds_sst[lon_name], wlat: ds_sst[lat_name]})
                w_v = w_v.interp({wlon: ds_sst[lon_name], wlat: ds_sst[lat_name]})
                w_u = w_u.rename({wlon: lon_name, wlat: lat_name})
                w_v = w_v.rename({wlon: lon_name, wlat: lat_name})
            wind_speed = np.sqrt(w_u.values**2 + w_v.values**2)
        except Exception:
            wind_speed = None

    # Prepare AOI mask
    lons = ds_sst[lon_name].values
    lats = ds_sst[lat_name].values
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    mask = polygon_mask(poly, grid_lons, grid_lats)

    # ---------------------------
    # Features (literature-driven set)
    # Typical tuna habitat models use SST/CHL/SSH + fronts/eddies and often currents/wind. 4
    # ---------------------------

    sst_vals = sst.values
    chl_vals = chl.values
    cur_speed = np.sqrt(u.values**2 + v.values**2)
    sl_vals = sl.values

    # Frontness proxies
    sst_front = normalize01(grad_mag(sst_vals))
    chl_front = normalize01(grad_mag(chl_vals))
    sl_front  = normalize01(grad_mag(sl_vals))

    # CHL preference: reward moderate productive waters (not extremes).
    # Many studies report higher tuna catch around moderate CHL ranges, and strong fronts. 5
    chl_log = np.log10(np.clip(chl_vals, 1e-4, None))
    chl_med = np.nanmedian(chl_log)
    chl_std = np.nanstd(chl_log) + 1e-9
    chl_pref = 1.0 - np.clip(np.abs(chl_log - chl_med) / (2.0 * chl_std), 0, 1)

    # Current suitability for gillnet-like ops: moderate currents preferred (too strong = gear control issues)
    # Heuristic band-pass around ~0.2–0.8 m/s
    cs = cur_speed
    cur_ok = np.exp(-((cs - 0.5) ** 2) / (2 * (0.35 ** 2)))
    cur_ok = normalize01(cur_ok)

    # Operational penalties (optional)
    wav_pen = None
    if wav is not None:
        wv = wav.values
        # penalize high waves; normalize then invert
        wav_pen = 1.0 - normalize01(wv)

    wind_pen = None
    if wind_speed is not None:
        # penalize high winds; normalize then invert
        wind_pen = 1.0 - normalize01(wind_speed)

    # ---------------------------
    # Weighted index -> 0..100
    # ---------------------------
    layers = []
    weights = []

    # Core ecological (fronts + productivity)
    layers += [sst_front, chl_front, chl_pref, sl_front]
    weights += [0.25,     0.25,      0.20,     0.10]

    # Dynamics (currents)
    layers += [cur_ok]
    weights += [0.10]

    # Ops constraints (gillnet): waves/wind if available
    if wav_pen is not None:
        layers += [wav_pen]
        weights += [0.05]
    if wind_pen is not None:
        layers += [wind_pen]
        weights += [0.05]

    W = np.array(weights, dtype=float)
    W = W / (W.sum() + 1e-12)

    score_raw = np.zeros_like(sst_vals, dtype=float)
    for w, lay in zip(W, layers):
        score_raw += w * np.nan_to_num(lay, nan=0.0)

    # Mask outside AOI
    score_raw = np.where(mask, score_raw, np.nan)

    # Map to "probability-like" 0..100 (not calibrated probability; a suitability index)
    mu = np.nanmedian(score_raw)
    sd = np.nanstd(score_raw) + 1e-12
    z = (score_raw - mu) / sd
    prob = 100.0 * sigmoid(z)
    prob = np.where(mask, prob, np.nan)

    # Top-10
    flat = prob.ravel()
    valid = np.isfinite(flat)
    idx = np.argsort(flat[valid])[::-1][:10]
    valid_idx = np.where(valid)[0][idx]
    yy, xx = np.unravel_index(valid_idx, prob.shape)

    # Save GeoJSON + CSV-like text
    features = []
    rows = ["rank,lon,lat,prob,sst,chl,current_speed,ssh"]
    for i, (y, x) in enumerate(zip(yy, xx), start=1):
        lon = float(grid_lons[y, x])
        lat = float(grid_lats[y, x])
        p = float(prob[y, x])
        features.append({
            "type": "Feature",
            "properties": {
                "rank": i,
                "prob_0_100": p,
                "sst": float(sst_vals[y, x]),
                "chl": float(chl_vals[y, x]),
                "current_speed": float(cur_speed[y, x]),
                "ssh": float(sl_vals[y, x]),
            },
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        })
        rows.append(f"{i},{lon},{lat},{p:.2f},{sst_vals[y,x]:.4f},{chl_vals[y,x]:.6f},{cur_speed[y,x]:.4f},{sl_vals[y,x]:.4f}")

    out_geo = {"type": "FeatureCollection", "features": features}
    with open(f"{OUT_DIR}/hotspots.geojson", "w", encoding="utf-8") as f:
        json.dump(out_geo, f, ensure_ascii=False, indent=2)

    with open(f"{OUT_DIR}/top10.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    with open(f"{OUT_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Plot probability map (red->yellow->green)
    plt.figure(figsize=(10, 6))
    im = plt.imshow(prob, origin="lower", vmin=0, vmax=100, cmap="RdYlGn")
    plt.colorbar(im, label="Fishing suitability (0–100)")

    # overlay top10 points
    plt.scatter(xx, yy, s=30, marker="o", edgecolors="black", linewidths=0.8)

    # title with times used
    title = f"Tuna suitability (AOI) | SST:{t_sst.date()} CHL:{t_chl.date()} UV:{t_uv.date()} SL:{t_sl.date()}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/probability.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("Done.")
    print("Outputs:")
    print(f"- {OUT_DIR}/hotspots.geojson")
    print(f"- {OUT_DIR}/top10.csv")
    print(f"- {OUT_DIR}/probability.png")
    print(f"- {OUT_DIR}/metadata.json")

if __name__ == "__main__":
    main()
