from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
except Exception:  # pragma: no cover
    rasterio = None
    from_bounds = None


@dataclass
class RasterSpec:
    bbox: Tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
    shape: Tuple[int, int]  # (ny, nx)
    crs: str = "EPSG:4326"


def write_geotiff(
    path: str,
    *,
    arr: np.ndarray,
    spec: RasterSpec,
    nodata: float = np.nan,
    dtype: str = "float32",
    compress: str = "deflate",
    tiled: bool = True,
    blocksize: int = 256,
) -> None:
    """Write a single-band GeoTIFF."""
    if rasterio is None:
        raise RuntimeError("rasterio is not available")

    ny, nx = spec.shape
    lon_min, lat_min, lon_max, lat_max = spec.bbox
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": dtype,
        "crs": spec.crs,
        "transform": transform,
        "compress": compress,
        "tiled": tiled,
        "blockxsize": min(blocksize, nx) if tiled else None,
        "blockysize": min(blocksize, ny) if tiled else None,
        "nodata": None if (np.isnan(nodata)) else nodata,
    }
    # drop None keys
    profile = {k: v for k, v in profile.items() if v is not None}

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


def write_cog(
    path: str,
    *,
    arr: np.ndarray,
    spec: RasterSpec,
    nodata: float = np.nan,
    dtype: str = "float32",
    compress: str = "deflate",
    overview_resampling: str = "average",
) -> None:
    """Write a Cloud-Optimized GeoTIFF-like output.

    This creates a tiled GeoTIFF with internal overviews, which is compatible with
    common COG readers. For strict COG validation, you may still want rio-cogeo,
    but this is a practical no-internet-friendly implementation.
    """
    if rasterio is None:
        raise RuntimeError("rasterio is not available")

    # Write base GeoTIFF tiled
    write_geotiff(
        path,
        arr=arr,
        spec=spec,
        nodata=nodata,
        dtype=dtype,
        compress=compress,
        tiled=True,
        blocksize=256,
    )

    # Build internal overviews
    with rasterio.open(path, "r+") as dst:
        # Determine overview levels
        ny, nx = dst.height, dst.width
        levels = []
        level = 2
        while (ny // level) >= 256 and (nx // level) >= 256:
            levels.append(level)
            level *= 2
        if levels:
            dst.build_overviews(levels, rasterio.enums.Resampling[overview_resampling])
            dst.update_tags(ns="rio_overview", resampling=overview_resampling)

