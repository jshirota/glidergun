import hashlib
import shutil
from functools import lru_cache

import rasterio

from glidergun import grid, search

dem = grid("./tests/input/n55_e008_1arc_v3.bil").resample(0.01)
dem_color = grid("./tests/input/n55_e008_1arc_v3.bil").resample(0.01).color("terrain")


@lru_cache
def landsat():
    return search("landsat-c2-l2", [-122.52, 37.70, -122.35, 37.82])[0].download(["red", "green", "blue"])


def save(obj, file_name):
    obj.save(file_name)
    with open(file_name, "rb") as f:
        hash = hashlib.md5(f.read()).hexdigest()
    with rasterio.open(file_name) as d:
        compress = d.profile.get("compress", None)
    shutil.rmtree("tests/output/temp")
    return hash, compress


def test_saving_dem_jpg():
    hash, compress = save(dem, "tests/output/temp/dem.jpg")
    assert hash


def test_saving_dem_tif():
    hash, compress = save(dem, "tests/output/temp/dem.tif")
    assert compress == "lzw"


def test_saving_dem_img():
    hash, compress = save(dem, "tests/output/temp/dem.img")
    assert hash == "0834c56700cf1cc3b7155a8ef6e8b922"


def test_saving_dem_bil():
    hash, compress = save(dem, "tests/output/temp/dem.bil")
    assert hash == "ce6230320c089d41ddbc8b3f17fd0c0d"


def test_saving_dem_color_jpg():
    hash, compress = save(dem_color, "tests/output/temp/dem_color.jpg")
    assert hash


def test_saving_dem_color_tif():
    hash, compress = save(dem_color, "tests/output/temp/dem_color.tif")
    assert compress == "lzw"


def test_saving_dem_color_img():
    hash, compress = save(dem_color, "tests/output/temp/dem_color.img")
    assert hash == "0834c56700cf1cc3b7155a8ef6e8b922"


def test_saving_dem_color_bil():
    hash, compress = save(dem_color, "tests/output/temp/dem_color.bil")
    assert hash == "ce6230320c089d41ddbc8b3f17fd0c0d"


def test_saving_landsat_jpg():
    hash, compress = save(landsat(), "tests/output/temp/landsat.jpg")
    assert hash


def test_saving_landsat_tif():
    hash, compress = save(landsat(), "tests/output/temp/landsat.tif")
    assert hash
    assert compress == "lzw"


def test_saving_landsat_img():
    hash, compress = save(landsat(), "tests/output/temp/landsat.img")
    assert hash
    assert compress is None


def test_saving_landsat_bil():
    hash, compress = save(landsat(), "tests/output/temp/landsat.bil")
    assert hash
    assert compress is None
