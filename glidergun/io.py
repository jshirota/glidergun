import hashlib
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import rasterio
import requests
from rasterio.io import MemoryFile

from glidergun.literals import DataType
from glidergun.utils import get_driver

if TYPE_CHECKING:
    from glidergun.grid import Grid


def http_get(url: str, cache_dir: str | None = None) -> bytes:
    if not cache_dir:
        response = requests.get(url)
        response.raise_for_status()
        return response.content

    cache_dir = os.path.expanduser(cache_dir)

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


def create_directory_for(file_path: str):
    directory = "/".join(re.split(r"/|\\", file_path)[0:-1])
    Path(directory).mkdir(parents=True, exist_ok=True)


def write_to_file(
    file: str | MemoryFile, driver: str, dtype: DataType, nodata: int | float | None, *grids: "Grid"
) -> None:
    count = len(grids)
    g = grids[0]
    driver = driver or get_driver(file)
    metadata = {"height": g.height, "width": g.width, "crs": g.crs, "transform": g.transform}
    if isinstance(file, str):
        create_directory_for(file)
        with rasterio.open(file, "w", driver=driver, count=count, dtype=dtype, nodata=nodata, **metadata) as dataset:
            for index, grid in enumerate(grids):
                dataset.write(grid.data, index + 1)
    elif isinstance(file, MemoryFile):
        with file.open(driver=driver, count=count, dtype=dtype, nodata=nodata, **metadata) as dataset:
            for index, grid in enumerate(grids):
                dataset.write(grid.data, index + 1)
