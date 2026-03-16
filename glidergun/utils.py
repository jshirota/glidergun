import numpy as np
from numpy import ndarray
from rasterio.crs import CRS
from rasterio.drivers import driver_from_extension
from rasterio.io import MemoryFile
from rasterio.warp import transform


def get_crs(crs: int | str | CRS):
    if isinstance(crs, str):
        try:
            return CRS.from_string(crs)
        except Exception:
            return CRS.from_wkt(crs)
    return CRS.from_epsg(crs) if isinstance(crs, int) else crs


def project(x: float, y: float, from_crs: int | str | CRS, to_crs: int | str | CRS):
    [x], [y], *_ = transform(get_crs(from_crs), get_crs(to_crs), [x], [y])
    return x, y


def get_driver(file: str | MemoryFile):
    if not isinstance(file, str) or file.lower().endswith(".tif"):
        return "COG"
    return driver_from_extension(file)


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


def is_image_format(driver: str, file: str | MemoryFile) -> bool:
    driver = driver or get_driver(file)
    return driver.lower() in ("jpeg", "kmlsuperoverlay", "mbtiles", "png", "webp")


def is_tile_url(url: str) -> bool:
    return url.startswith(("http://", "https://")) and "{" in url and "}" in url
