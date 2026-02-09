from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def write_cog(
    out_tif: Path,
    data: np.ndarray,
    *,
    bbox: Tuple[float, float, float, float],
    nodata: float = -9999.0,
    compress: str = "DEFLATE",
) -> dict:
    """Write a Cloud-Optimized GeoTIFF (COG) if GDAL supports it.

    Falls back to a "COG-friendly" tiled GeoTIFF with overviews when the COG
    driver is unavailable in the runtime environment.

    Returns a small metadata dict for auditing.
    """

    import rasterio
    from rasterio.transform import from_bounds

    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    # Ensure float32 in file to avoid driver quirks.
    arr = np.asarray(data, dtype=np.float32)

    height, width = arr.shape
    transform = from_bounds(*bbox, width=width, height=height)

    # Try GDAL COG driver first.
    try:
        profile = {
            "driver": "COG",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": nodata,
            "compress": compress,
            "blocksize": 256,
            "overview_resampling": "average",
        }
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(arr, 1)
        return {
            "path": str(out_tif),
            "driver": "COG",
            "compress": compress,
            "nodata": nodata,
        }
    except Exception:
        # Fall back to tiled GeoTIFF + overviews.
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": nodata,
            "compress": compress,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(arr, 1)
            # Build a few overviews to make it web-friendly.
            factors = [2, 4, 8]
            try:
                dst.build_overviews(factors, rasterio.enums.Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")
            except Exception:
                pass
        return {
            "path": str(out_tif),
            "driver": "GTiff",
            "compress": compress,
            "nodata": nodata,
            "note": "COG driver unavailable; wrote tiled GeoTIFF with overviews",
        }
