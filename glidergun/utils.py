import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray
from rasterio.crs import CRS
from rasterio.drivers import driver_from_extension
from rasterio.io import MemoryFile
from rasterio.warp import transform

if TYPE_CHECKING:
    from glidergun.stac import Grid, Stack


def get_crs(crs: int | str | CRS):
    if isinstance(crs, int):
        try:
            return CRS.from_epsg(crs)
        except Exception:
            pass
        try:
            return CRS.from_string(f"ESRI:{crs}")
        except Exception as ex:
            raise ValueError(f"Unknown CRS code: {crs}") from ex

    if isinstance(crs, str):
        if crs.isdigit():
            return get_crs(int(crs))
        try:
            return CRS.from_string(crs)
        except Exception as ex:
            raise ValueError(f"Invalid CRS string: {crs}") from ex
    return crs


def get_crs_name(crs: int | str | CRS) -> str:
    try:
        return get_crs(crs).to_wkt().split('"')[1]
    except Exception:
        return "Unknown"


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


def add_to_active_map(self: "Grid | Stack"):
    file = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".tif"

    if "arcpy" in sys.modules:
        try:
            import arcpy  # type: ignore

            aprx = arcpy.mp.ArcGISProject("CURRENT")
            folder = arcpy.env.scratchFolder
            geotiff = os.path.join(folder, file)
            self.save(geotiff)
            aprx.activeMap.addDataFromPath(geotiff)
        except Exception as ex:
            print(f"Failed to add layer to active map: {ex}")
        return

    if "qgis.core" in sys.modules:
        try:
            from qgis.core import QgsProcessingUtils, QgsProject, QgsRasterLayer  # type: ignore

            folder = QgsProcessingUtils.tempFolder()
            geotiff = os.path.join(folder, file)
            self.save(geotiff)
            QgsProject.instance().addMapLayer(QgsRasterLayer(geotiff, file))
        except Exception as ex:
            print(f"Failed to add layer to active map: {ex}")
        return

    raise RuntimeError("This function is only supported within ArcGIS Pro or QGIS.")
