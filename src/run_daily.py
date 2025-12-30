import json
import os
from datetime import datetime, timedelta, date

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import copernicusmarine as cm
from matplotlib.path import Path

# ----------------------------
# Config
# ----------------------------
AOI_PATH = "config/aoi.geojson"
OUT_DIR = "docs/latest"
os.makedirs(OUT_DIR, exist_ok=True)

# Net fishing depth envelope (user: ~20–30m, but we score 0–30m as "net-influenced layer")
DEPTH_MIN_M = 0.0
DEPTH_MAX_M = 30.0

# Use forecast/analysis products (updated daily + includes short forecasts)
TEMP_DATASET = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"         # 3D temp daily
CURR_DATASET = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"            # 3D currents daily
SSH_DATASET  = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"                # 2D includes SSH
WAV_DATASET  = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"               # waves 3-hourly

# BGC: primary = model forecast, fallback = satellite gapfree NRT
CHL_PRIMARY  = "cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m"
CHL_FALLBACK = "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"

# Preferred variable names (we still auto-detect if not found)
PREF = {
    "temp": ["thetao", "temperature", "temp"],
    "uo":   ["uo", "u", "eastward_sea_water_velocity"],
    "vo":   ["vo", "v", "northward_sea_water_velocity"],
    "ssh":  ["zos", "ssh", "sea_surface_height_above_geoid"],
    "chl":  ["CHL", "chl", "chlorophyll", "chlor_a", "CHL_A"],
    "wave": ["VHM0", "swh", "Hs", "significant_height"]
}

# ----------------------------
# Utilities
# ----------------------------
def read_aoi_polygon(path: str):
    gj = json.load(open(path, "r", encoding="utf-8"))
    coords = gj["features"][0]["geometry"]["coordinates"][0]
    # coords = [(lon,lat),...]
    return coords

def bbox_from_coords(coords):
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), max(lons), min(lats), max(lats)

def pick_var(ds: xr.Dataset, prefer_list):
    vars_ = list(ds.data_vars.keys())
    for p in prefer_list:
        if p in vars_:
            return p
    if not vars_:
        raise RuntimeError("Dataset has no data_vars.")
    return vars_[0]

def get_lat_lon_names(ds: xr.Dataset):
    for lat_name in ["latitude", "lat", "nav_lat"]:
        if lat_name in ds.coords:
            break
    else:
        raise RuntimeError("No latitude coord found.")
    for lon_name in ["longitude", "lon", "nav_lon"]:
        if lon_name in ds.coords:
            break
    else:
        raise RuntimeError("No longitude coord found.")
    return lat_name, lon_name

def safe_grad_mag(a2d: np.ndarray):
    # fill NaN for gradient calculation (keep mask later)
    a = a2d.copy()
    nanmask = np.isnan(a)
    if np.all(nanmask):
        return np.full_like(a, np.nan)
    med = np.nanmedian(a)
    a[nanmask] = med
    gy, gx = np.gradient(a)
    g = np.sqrt(gx * gx + gy * gy)
    g[nanmask] = np.nan
    return g

def poly_mask(lon2d, lat2d, poly_coords):
    # Path expects (x,y) = (lon,lat)
    p = Path(poly_coords)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    inside = p.contains_points(pts).reshape(lon2d.shape)
    return inside

def open_day_dataset(
    dataset_id: str,
    variables: list[str],
    day: date,
    bbox,
    depth_min=None,
    depth_max=None,
):
    min_lon, max_lon, min_lat, max_lat = bbox
    start = datetime(day.year, day.month, day.day)
    end = start + timedelta(days=1)

    kwargs = dict(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
        coordinates_selection_method="nearest",  # key to avoid out-of-bounds time/depth
    )
    if depth_min is not None and depth_max is not None:
        kwargs.update(
            minimum_depth=float(depth_min),
            maximum_depth=float(depth_max),
        )

    return cm.open_dataset(**kwargs)

def normalize_0_1(x):
    x = x.astype(float)
    m = np.nanmin(x)
    M = np.nanmax(x)
    if not np.isfinite(m) or not np.isfinite(M) or M - m < 1e-12:
        return np.zeros_like(x)
    return (x - m) / (M - m)

def gaussian_pref(x, mu, sigma):
    # returns 0..1
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-9)) ** 2)

# ----------------------------
# Main
# ----------------------------
def main():
    print("Reading AOI...")
    poly = read_aoi_polygon(AOI_PATH)
    bbox = bbox_from_coords(poly)

    # Allow override for testing
    run_date_str = os.getenv("RUN_DATE", "").strip()
    if run_date_str:
        req_day = datetime.strptime(run_date_str, "%Y-%m-%d").date()
    else:
        req_day = datetime.utcnow().date()

    print(f"Processing requested date: {req_day.isoformat()}")

    # 1) Temperature (0–30m mean + gradient)
    print("Loading TEMP...")
    ds_temp = open_day_dataset(
        TEMP_DATASET, variables=[], day=req_day, bbox=bbox,
        depth_min=DEPTH_MIN_M, depth_max=DEPTH_MAX_M
    )
    v_temp = pick_var(ds_temp, PREF["temp"])
    latn, lonn = get_lat_lon_names(ds_temp)

    temp = ds_temp[v_temp]
    if "time" in temp.dims:
        temp = temp.isel(time=0)
    if "depth" in temp.dims:
        temp_0_30 = temp.mean("depth", skipna=True)
    else:
        temp_0_30 = temp

    temp2d = temp_0_30.values
    temp_grad = safe_grad_mag(temp2d)

    # 2) Currents (0–30m mean speed)
    print("Loading CURRENTS...")
    ds_cur = open_day_dataset(
        CURR_DATASET, variables=[], day=req_day, bbox=bbox,
        depth_min=DEPTH_MIN_M, depth_max=DEPTH_MAX_M
    )
    v_uo = pick_var(ds_cur, PREF["uo"])
    v_vo = pick_var(ds_cur, PREF["vo"])

    uo = ds_cur[v_uo]
    vo = ds_cur[v_vo]
    if "time" in uo.dims:
        uo = uo.isel(time=0)
        vo = vo.isel(time=0)
    if "depth" in uo.dims:
        uo = uo.mean("depth", skipna=True)
        vo = vo.mean("depth", skipna=True)

    speed = np.sqrt(uo.values**2 + vo.values**2)
    speed_grad = safe_grad_mag(speed)

    # 3) SSH (front/eddy proxy via gradient)
    print("Loading SSH...")
    ds_ssh = open_day_dataset(SSH_DATASET, variables=[], day=req_day, bbox=bbox)
    v_ssh = pick_var(ds_ssh, PREF["ssh"])
    ssh = ds_ssh[v_ssh]
    if "time" in ssh.dims:
        ssh = ssh.isel(time=0)
    ssh2d = ssh.values
    ssh_grad = safe_grad_mag(ssh2d)

    # 4) Waves (penalty) - compute daily max significant wave height
    print("Loading WAVES...")
    ds_wav = open_day_dataset(WAV_DATASET, variables=[], day=req_day, bbox=bbox)
    v_wav = pick_var(ds_wav, PREF["wave"])
    wav = ds_wav[v_wav]
    if "time" in wav.dims:
        wav_day = wav.max("time", skipna=True)  # conservative for operations
    else:
        wav_day = wav
    wav2d = wav_day.values

    # 5) Chlorophyll (primary -> fallback)
    print("Loading CHL (primary)...")
    chl2d = None
    try:
        ds_chl = open_day_dataset(CHL_PRIMARY, variables=[], day=req_day, bbox=bbox, depth_min=0.0, depth_max=0.0)
        v_chl = pick_var(ds_chl, PREF["chl"])
        chl = ds_chl[v_chl]
        if "time" in chl.dims:
            chl = chl.isel(time=0)
        if "depth" in chl.dims:
            chl = chl.isel(depth=0)  # nearest depth=0
        chl2d = chl.values
        print("CHL primary OK.")
    except Exception as e:
        print(f"CHL primary failed: {e}")
        print("Loading CHL (fallback satellite gapfree)...")
        ds_chl = open_day_dataset(CHL_FALLBACK, variables=[], day=req_day, bbox=bbox)
        v_chl = pick_var(ds_chl, PREF["chl"])
        chl = ds_chl[v_chl]
        if "time" in chl.dims:
            chl = chl.isel(time=0)
        chl2d = chl.values
        print("CHL fallback OK.")

    chl_grad = safe_grad_mag(chl2d)

    # Coordinates
    lats = ds_temp[latn].values
    lons = ds_temp[lonn].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    # AOI mask
    inside = poly_mask(lon2d, lat2d, poly)
    mask_outside = ~inside

    def apply_mask(a):
        a = a.astype(float)
        a[mask_outside] = np.nan
        return a

    temp2d     = apply_mask(temp2d)
    temp_grad  = apply_mask(temp_grad)
    speed      = apply_mask(speed)
    speed_grad = apply_mask(speed_grad)
    ssh_grad   = apply_mask(ssh_grad)
    chl2d      = apply_mask(chl2d)
    chl_grad   = apply_mask(chl_grad)
    wav2d      = apply_mask(wav2d)

    # ----------------------------
    # Scoring model (0..100)
    # ----------------------------
    # Tuna habitat literature repeatedly uses SST/Temp, CHL, SSH anomaly/gradient, currents; and operational filters like wind/waves are practical.
    # We use a heuristic index now; later, if you collect catch logs, we can fit a real regression/GAM/ML model.

    # 1) Temperature preference (broad, multi-species tropical tuna proxy)
    # Use AOI median as "local optimum" to avoid hard-coded species-specific numbers
    t_mu = np.nanmedian(temp2d)
    t_sigma = np.nanstd(temp2d) if np.nanstd(temp2d) > 0 else 1.5
    temp_pref = gaussian_pref(temp2d, t_mu, t_sigma)

    # 2) Chlorophyll preference: tuna often avoids extremes; use AOI median targeting
    chl_mu = np.nanmedian(chl2d)
    chl_sigma = np.nanstd(chl2d) if np.nanstd(chl2d) > 0 else (chl_mu * 0.5 + 1e-6)
    chl_pref = gaussian_pref(chl2d, chl_mu, chl_sigma)

    # 3) Front/eddy proxies
    temp_front = normalize_0_1(temp_grad)
    chl_front  = normalize_0_1(chl_grad)
    ssh_front  = normalize_0_1(ssh_grad)
    cur_front  = normalize_0_1(speed_grad)

    # 4) Current speed preference: gillnet prefers moderate currents (too strong = gear/control issues)
    sp_mu = np.nanmedian(speed)
    sp_sigma = np.nanstd(speed) if np.nanstd(speed) > 0 else (sp_mu * 0.5 + 1e-6)
    speed_pref = gaussian_pref(speed, sp_mu, sp_sigma)

    # 5) Wave penalty: lower waves better for operations
    wave_norm = normalize_0_1(wav2d)
    wave_penalty = 1.0 - wave_norm  # higher is better

    # Weights (heuristic; can be re-tuned)
    # Fronts + productivity proxies get strong weight; operational penalty meaningful for net fishing
    w = {
        "temp_pref": 0.18,
        "chl_pref":  0.14,
        "temp_front":0.16,
        "chl_front": 0.16,
        "ssh_front": 0.14,
        "cur_front": 0.08,
        "speed_pref":0.06,
        "wave_ok":   0.08,
    }

    raw = (
        w["temp_pref"]  * temp_pref +
        w["chl_pref"]   * chl_pref +
        w["temp_front"] * temp_front +
        w["chl_front"]  * chl_front +
        w["ssh_front"]  * ssh_front +
        w["cur_front"]  * cur_front +
        w["speed_pref"] * speed_pref +
        w["wave_ok"]    * wave_penalty
    )

    raw = apply_mask(raw)

    # Scale to 0..100 within AOI
    prob = 100.0 * normalize_0_1(raw)
    prob = apply_mask(prob)

    # ----------------------------
    # Top-10 points
    # ----------------------------
    flat = prob.ravel()
    valid_idx = np.where(np.isfinite(flat))[0]
    if valid_idx.size == 0:
        raise RuntimeError("No valid grid points inside AOI to rank.")

    topn = 10
    order = valid_idx[np.argsort(flat[valid_idx])[::-1]]
    order = order[:topn]
    yy, xx = np.unravel_index(order, prob.shape)

    features = []
    lines = []
    for rank, (y, x) in enumerate(zip(yy, xx), start=1):
        props = {
            "rank": rank,
            "probability_0_100": float(prob[y, x]),
            "temp_0_30_mean": float(temp2d[y, x]),
            "chl_surface": float(chl2d[y, x]),
            "wave_max": float(wav2d[y, x]),
            "current_speed_0_30_mean": float(speed[y, x]),
            # explainability: weighted contributions (approx)
            "contrib_temp_pref": float(w["temp_pref"] * temp_pref[y, x]),
            "contrib_chl_pref": float(w["chl_pref"] * chl_pref[y, x]),
            "contrib_temp_front": float(w["temp_front"] * temp_front[y, x]),
            "contrib_chl_front": float(w["chl_front"] * chl_front[y, x]),
            "contrib_ssh_front": float(w["ssh_front"] * ssh_front[y, x]),
            "contrib_wave_ok": float(w["wave_ok"] * wave_penalty[y, x]),
        }
        lon = float(lons[x])
        lat = float(lats[y])
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Point", "coordinates": [lon, lat]}
        })

        lines.append(
            f"{rank:02d}) lon={lon:.5f}, lat={lat:.5f} | Prob={prob[y,x]:.1f} | "
            f"T(0-30)={temp2d[y,x]:.2f} | CHL={chl2d[y,x]:.4f} | WaveMax={wav2d[y,x]:.2f} | CurSpd={speed[y,x]:.2f}"
        )

    out_geo = {"type": "FeatureCollection", "features": features}
    with open(f"{OUT_DIR}/hotspots.geojson", "w", encoding="utf-8") as f:
        json.dump(out_geo, f, ensure_ascii=False)

    with open(f"{OUT_DIR}/hotspots.txt", "w", encoding="utf-8") as f:
        f.write("Top-10 Hotspots (lon,lat) + key drivers\n")
        f.write(f"Date requested: {req_day.isoformat()} (nearest-in-dataset selection)\n\n")
        f.write("\n".join(lines))
        f.write("\n")

    # ----------------------------
    # Plot probability map + points
    # ----------------------------
    plt.figure(figsize=(10, 7))
    pcm = plt.pcolormesh(lon2d, lat2d, prob, shading="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(pcm, label="Fishing Probability (0-100)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Tuna Net-Fishing Hotspot Probability (0-30m drivers) | {req_day.isoformat()}")

    # AOI outline
    poly_lons = [p[0] for p in poly] + [poly[0][0]]
    poly_lats = [p[1] for p in poly] + [poly[0][1]]
    plt.plot(poly_lons, poly_lats, linewidth=2)

    # points
    for ft in features:
        lon, lat = ft["geometry"]["coordinates"]
        r = ft["properties"]["rank"]
        plt.scatter([lon], [lat], s=40)
        plt.text(lon, lat, str(r), fontsize=9)

    plt.savefig(f"{OUT_DIR}/probability.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("Done:", req_day.isoformat(), "| outputs in", OUT_DIR)

if __name__ == "__main__":
    main()