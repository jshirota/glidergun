import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy import ndarray
from rasterio.crs import CRS
from rasterio.drivers import driver_from_extension
from shapely.geometry import Point, Polygon, mapping

from glidergun._types import FeatureCollection


def create_directory_for(file_path: str):
    directory = "/".join(re.split(r"/|\\", file_path)[0:-1])
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_crs(crs: int | str | CRS):
    if isinstance(crs, str):
        try:
            return CRS.from_string(crs)
        except Exception:
            return CRS.from_wkt(crs)
    return CRS.from_epsg(crs) if isinstance(crs, int) else crs


def get_driver(file: str):
    return "COG" if file.lower().endswith(".tif") else driver_from_extension(file)


def format_type(data: ndarray):
    if data.dtype == "float64":
        return np.asanyarray(data, dtype="float32")
    if data.dtype == "int64":
        return np.asanyarray(data, dtype="int32")
    if data.dtype == "uint64":
        return np.asanyarray(data, dtype="uint32")
    return data


def get_nodata_value(dtype: str) -> float | int | None:
    if dtype == "bool" or dtype == "uint8":
        return None
    if dtype.startswith("float"):
        return float(np.finfo(dtype).min)
    if dtype.startswith("uint"):
        return np.iinfo(dtype).max
    return np.iinfo(dtype).min


def get_geojson(features: Iterable[tuple[Point | Polygon, dict]]) -> FeatureCollection:
    return FeatureCollection(
        type="FeatureCollection",
        features=[{"type": "Feature", "geometry": mapping(g), "properties": p} for g, p in features],
    )
