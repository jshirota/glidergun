import hashlib
import os
import re
from pathlib import Path

import numpy as np
import requests
from numpy import ndarray
from rasterio.crs import CRS
from rasterio.drivers import driver_from_extension


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


def http_get(url: str, cache_dir: str | None = None) -> bytes:
    if not cache_dir:
        response = requests.get(url)
        response.raise_for_status()
        return response.content

    key = hashlib.md5(url.encode()).hexdigest()
    path = f"{cache_dir}/{key}"

    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()

    response = requests.get(url)
    response.raise_for_status()
    os.makedirs(cache_dir, exist_ok=True)

    with open(path, "wb") as f:
        f.write(response.content)

    return response.content
