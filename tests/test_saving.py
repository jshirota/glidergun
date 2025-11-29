import hashlib
import shutil

import rasterio

from glidergun import grid, stack

dem = grid("./tests/input/n55_e008_1arc_v3.bil").resample(0.01)
dem_color = grid("./tests/input/n55_e008_1arc_v3.bil").resample(0.01).color("terrain")

landsat = stack(
    [
        "tests/input/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B1.TIF",
        "tests/input/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B2.TIF",
        "tests/input/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B3.TIF",
        "tests/input/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B4.TIF",
        "tests/input/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B5.TIF",
        "tests/input/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B6.TIF",
        "tests/input/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B7.TIF",
    ]
)


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
    hash, compress = save(landsat.color((5, 4, 3)), "tests/output/temp/landsat_543_1.jpg")
    assert hash

    hash, compress = save(landsat.extract_bands(5, 4, 3), "tests/output/temp/landsat_543_2.jpg")
    assert hash


def test_saving_landsat_tif():
    hash1, compress1 = save(landsat.color((5, 4, 3)), "tests/output/temp/landsat_543_1.tif")
    hash2, compress2 = save(landsat.color((1, 2, 3)), "tests/output/temp/landsat_543_2.tif")
    assert hash1 == hash2

    hash, compress = save(landsat.extract_bands(5, 4, 3), "tests/output/temp/landsat_543_3.tif")
    assert compress == "lzw"


def test_saving_landsat_img():
    hash1, compress1 = save(landsat.color((5, 4, 3)), "tests/output/temp/landsat_543_1.img")
    hash2, compress2 = save(landsat.color((1, 2, 3)), "tests/output/temp/landsat_543_2.img")
    assert hash1 == hash2

    hash, compress = save(landsat.extract_bands(5, 4, 3), "tests/output/temp/landsat_543_3.img")
    assert hash == "700167ccacd6f63a3f46fcf7c2e41f71"


def test_saving_landsat_bil():
    hash, compress = save(landsat.extract_bands(5, 4, 3), "tests/output/temp/landsat_543_2.bil")
    assert hash == "ff0b8c95a824c9550d12c203132ca4a9"
    assert compress is None
