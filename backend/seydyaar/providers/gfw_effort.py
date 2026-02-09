from __future__ import annotations
"""
Global Fishing Watch (GFW) AIS effort proxy provider.

This module is intentionally conservative:
- It provides a *working architecture* to fetch AIS effort tiles from the official GFW 4Wings API,
  stitch them, and rasterize to our AOI grid.
- To run it, you need a valid API access token (env: GFW_API_TOKEN) and network access.

Docs (API reference):
- https://globalfishingwatch.org/our-apis/documentation (4Wings raster/tiles, datasets, date-range, etc.)
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import os
import io
import math
import numpy as np
import requests
from PIL import Image

GFW_BASE = "https://gateway.api.globalfishingwatch.org"
TILE_PATH = "/v2/4wings/tile/heatmap/{z}/{x}/{y}"

@dataclass
class GFWConfig:
    token: str
    dataset: str = "public-global-fishing-effort:latest"
    interval: str = "10days"   # 'hour'|'day'|'10days'
    date_range: Optional[str] = None  # "YYYY-MM-DD,YYYY-MM-DD"
    style: Optional[str] = None       # style id/url param from GFW style endpoint
    zoom: int = 4

def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

def _tile_xyz_from_lonlat(lon: float, lat: float, z: int) -> Tuple[int,int,int]:
    # WebMercator tile conversion
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return z, xtile, ytile

def fetch_effort_tile_png(cfg: GFWConfig, z: int, x: int, y: int) -> Image.Image:
    params = {
        "format": "png",
        "interval": cfg.interval,
        "datasets[0]": cfg.dataset,
    }
    if cfg.date_range:
        params["date-range"] = cfg.date_range
    if cfg.style:
        params["style"] = cfg.style

    url = f"{GFW_BASE}{TILE_PATH.format(z=z,x=x,y=y)}"
    r = requests.get(url, headers=_headers(cfg.token), params=params, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")

def stitch_bbox_tiles(cfg: GFWConfig, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> Tuple[np.ndarray, dict]:
    """
    Fetches tiles covering bbox and stitches into one RGBA image array.
    Returns:
      img_rgba (H,W,4) uint8, and metadata dict (z, x_range, y_range)
    """
    z = cfg.zoom
    _, x0, y0 = _tile_xyz_from_lonlat(lon_min, lat_max, z)  # NW corner
    _, x1, y1 = _tile_xyz_from_lonlat(lon_max, lat_min, z)  # SE corner

    x_min, x_max = min(x0,x1), max(x0,x1)
    y_min, y_max = min(y0,y1), max(y0,y1)

    tiles = []
    for y in range(y_min, y_max+1):
        row = []
        for x in range(x_min, x_max+1):
            row.append(fetch_effort_tile_png(cfg, z, x, y))
        tiles.append(row)

    tile_size = tiles[0][0].size[0]
    W = (x_max-x_min+1)*tile_size
    H = (y_max-y_min+1)*tile_size
    canvas = Image.new("RGBA", (W, H))
    for ry, row in enumerate(tiles):
        for rx, im in enumerate(row):
            canvas.paste(im, (rx*tile_size, ry*tile_size))

    meta = {"z": z, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "tile_size": tile_size}
    return np.array(canvas), meta

def rgba_to_effort_proxy(img_rgba: np.ndarray) -> np.ndarray:
    """
    Converts a GFW-rendered heatmap RGBA image into a *proxy* numeric surface.
    This is not the raw 'hours' value, but is usable as:
      - sampling bias surface, or
      - pseudo-presence density for presence-only modeling.

    Strategy: use alpha * luminance as proxy intensity (0..1).
    """
    rgb = img_rgba[..., :3].astype(np.float32) / 255.0
    a = img_rgba[..., 3].astype(np.float32) / 255.0
    lum = (0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2])
    p = lum * a
    # normalize robustly
    lo, hi = np.percentile(p, 5), np.percentile(p, 95)
    out = (p - lo) / (hi - lo + 1e-9)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def effort_proxy_surface(cfg: GFWConfig, bbox: Tuple[float,float,float,float]) -> Tuple[np.ndarray, dict]:
    lon_min, lat_min, lon_max, lat_max = bbox
    img, meta = stitch_bbox_tiles(cfg, lon_min, lon_max, lat_min, lat_max)
    proxy = rgba_to_effort_proxy(img)
    meta["bbox"] = {"lon_min":lon_min,"lon_max":lon_max,"lat_min":lat_min,"lat_max":lat_max}
    return proxy, meta

def load_cfg_from_env(date_range: Optional[str] = None, zoom: int = 4) -> Optional[GFWConfig]:
    tok = os.environ.get("GFW_API_TOKEN") or os.environ.get("GFW_TOKEN")
    if not tok:
        return None
    return GFWConfig(token=tok, date_range=date_range, zoom=zoom)

# -----------------------------------------------------------------------------
# Raster helpers
# -----------------------------------------------------------------------------

def rasterize_effort_to_grid(
    proxy_img: np.ndarray,
    *,
    img_meta: dict,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> np.ndarray:
    """Project a stitched WebMercator tile image onto a regular lon/lat grid.

    Parameters
    - proxy_img: 2D array (H x W) of effort intensity (already normalized).
    - img_meta: metadata returned by `stitch_bbox_tiles`.
      Expected keys: z, x_min, y_min, tile_size (pixels).
    - grid_lon/grid_lat: 2D arrays of lon/lat at target grid cell centers.
    - bbox: lon_min, lat_min, lon_max, lat_max (informational; not required if meta is present)

    Notes
    - Uses standard slippy-map projection formulas to map lon/lat -> pixel.
    - If required meta fields are missing, falls back to linear bbox mapping.
    """

    H, W = proxy_img.shape

    # If we have proper slippy-tile metadata, map lon/lat -> global pixel and then
    # into the stitched image pixel coordinates.
    if all(k in img_meta for k in ("z", "x_min", "y_min", "tile_size")):
        z = int(img_meta["z"])
        x_min = int(img_meta["x_min"])
        y_min = int(img_meta["y_min"])
        tile_size = int(img_meta["tile_size"])

        n = (2 ** z) * tile_size  # global pixel size

        lon = grid_lon
        lat = np.clip(grid_lat, -85.0511, 85.0511)  # WebMercator limit

        x_global = (lon + 180.0) / 360.0 * n
        lat_rad = np.deg2rad(lat)
        y_global = (1.0 - np.log(np.tan(lat_rad) + (1.0 / np.cos(lat_rad))) / np.pi) / 2.0 * n

        x = x_global - x_min * tile_size
        y = y_global - y_min * tile_size
    else:
        # Fallback approximation: assume the image spans bbox linearly in lon/lat.
        lon_min, lat_min, lon_max, lat_max = bbox
        x = (grid_lon - lon_min) / max(lon_max - lon_min, 1e-9) * (W - 1)
        y = (lat_max - grid_lat) / max(lat_max - lat_min, 1e-9) * (H - 1)

    # Bilinear sample
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    Ia = proxy_img[y0, x0]
    Ib = proxy_img[y0, x1]
    Ic = proxy_img[y1, x0]
    Id = proxy_img[y1, x1]

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)

    # Fallback: assume proxy image is bbox-aligned in lon/lat
    lon_min, lat_min, lon_max, lat_max = bbox
    gx = (grid_lon - lon_min) / max(1e-9, (lon_max - lon_min))
    gy = (grid_lat - lat_min) / max(1e-9, (lat_max - lat_min))

    ix = np.clip(gx * (nx_img - 1), 0, nx_img - 1)
    iy = np.clip((1 - gy) * (ny_img - 1), 0, ny_img - 1)

    x0 = np.floor(ix).astype(int)
    x1 = np.clip(x0 + 1, 0, nx_img - 1)
    y0 = np.floor(iy).astype(int)
    y1 = np.clip(y0 + 1, 0, ny_img - 1)

    wx = ix - x0
    wy = iy - y0

    a = proxy_img[y0, x0]
    b = proxy_img[y0, x1]
    c = proxy_img[y1, x0]
    d = proxy_img[y1, x1]

    out = (1 - wx) * (1 - wy) * a + wx * (1 - wy) * b + (1 - wx) * wy * c + wx * wy * d
    return out.astype(np.float32)
