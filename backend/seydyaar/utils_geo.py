from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from shapely.geometry import shape, Point
from shapely.prepared import prep

@dataclass(frozen=True)
class GridSpec:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    width: int
    height: int
    crs: str = "EPSG:4326"

    @property
    def dx(self) -> float:
        return (self.lon_max - self.lon_min) / (self.width - 1)

    @property
    def dy(self) -> float:
        return (self.lat_max - self.lat_min) / (self.height - 1)

    def lonlat_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        lons = np.linspace(self.lon_min, self.lon_max, self.width, dtype=np.float32)
        lats = np.linspace(self.lat_max, self.lat_min, self.height, dtype=np.float32)  # north->south for images
        lon2d, lat2d = np.meshgrid(lons, lats)
        return lon2d, lat2d

def bbox_from_geojson(aoi_geojson: dict) -> Tuple[float,float,float,float]:
    geom = shape(aoi_geojson["features"][0]["geometry"])
    minx, miny, maxx, maxy = geom.bounds
    return float(minx), float(miny), float(maxx), float(maxy)

def mask_from_geojson(aoi_geojson: dict, grid: GridSpec) -> np.ndarray:
    """
    Returns uint8 mask (1 inside AOI, 0 outside) with shape (H, W), aligned with grid lon/lat mesh.
    """
    geom = shape(aoi_geojson["features"][0]["geometry"])
    pg = prep(geom)

    lon2d, lat2d = grid.lonlat_mesh()
    H, W = lon2d.shape
    mask = np.zeros((H, W), dtype=np.uint8)

    # vectorized point-in-polygon is nontrivial; do a fast loop (H*W is small in demo)
    for i in range(H):
        for j in range(W):
            if pg.contains(Point(float(lon2d[i, j]), float(lat2d[i, j]))):
                mask[i, j] = 1
    return mask
