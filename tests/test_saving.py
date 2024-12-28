import hashlib
import shutil

from glidergun._grid import grid
from glidergun._stack import stack

dem = grid("./.data/n55_e008_1arc_v3.bil").resample(0.01)
dem_color = grid("./.data/n55_e008_1arc_v3.bil").resample(0.01).color("terrain")

landsat = stack(
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B1.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B2.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B3.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B4.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B5.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B6.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B7.TIF",
)


def save(obj, file_name):
    obj.save(file_name)
    with open(file_name, "rb") as f:
        hash = hashlib.md5(f.read()).hexdigest()
    shutil.rmtree(".output/test")
    return hash


def test_saving_dem_png():
    hash = save(dem, ".output/test/dem.png")
    assert hash == "aa05e8bf7a3d9d450c6466392c3d96cd"


def test_saving_dem_jpg():
    hash = save(dem, ".output/test/dem.jpg")
    assert hash


def test_saving_dem_tif():
    hash = save(dem, ".output/test/dem.tif")
    assert hash == "af7d1d5efc2fe56b8ed398a50d6d07f1"


def test_saving_dem_img():
    hash = save(dem, ".output/test/dem.img")
    assert hash == "0834c56700cf1cc3b7155a8ef6e8b922"


def test_saving_dem_bil():
    hash = save(dem, ".output/test/dem.bil")
    assert hash == "ce6230320c089d41ddbc8b3f17fd0c0d"


def test_saving_dem_color_png():
    hash = save(dem_color, ".output/test/dem_color.png")
    assert hash == "3a01653a1228fd4045392d2a32814ac9"


def test_saving_dem_color_jpg():
    hash = save(dem_color, ".output/test/dem_color.jpg")
    assert hash


def test_saving_dem_color_tif():
    hash = save(dem_color, ".output/test/dem_color.tif")
    assert hash == "af7d1d5efc2fe56b8ed398a50d6d07f1"


def test_saving_dem_color_img():
    hash = save(dem_color, ".output/test/dem_color.img")
    assert hash == "0834c56700cf1cc3b7155a8ef6e8b922"


def test_saving_dem_color_bil():
    hash = save(dem_color, ".output/test/dem_color.bil")
    assert hash == "ce6230320c089d41ddbc8b3f17fd0c0d"


def test_saving_landsat_png():
    hash = save(landsat.color((5, 4, 3)), ".output/test/landsat_543_1.png")
    assert hash == "233580785423a3b0d3a21564a59d68f8"

    hash = save(landsat.extract_bands(5, 4, 3), ".output/test/landsat_543_2.png")
    assert hash == "233580785423a3b0d3a21564a59d68f8"


def test_saving_landsat_jpg():
    hash = save(landsat.color((5, 4, 3)), ".output/test/landsat_543_1.jpg")
    assert hash

    hash = save(landsat.extract_bands(5, 4, 3), ".output/test/landsat_543_2.jpg")
    assert hash


def test_saving_landsat_tif():
    hash1 = save(landsat.color((5, 4, 3)), ".output/test/landsat_543_1.tif")
    hash2 = save(landsat.color((1, 2, 3)), ".output/test/landsat_543_2.tif")
    assert hash1 == hash2

    hash = save(landsat.extract_bands(5, 4, 3), ".output/test/landsat_543_3.tif")
    assert hash == "fadf37f9d802720f7f1ebacf617699f6"


def test_saving_landsat_img():
    hash1 = save(landsat.color((5, 4, 3)), ".output/test/landsat_543_1.img")
    hash2 = save(landsat.color((1, 2, 3)), ".output/test/landsat_543_2.img")
    assert hash1 == hash2

    hash = save(landsat.extract_bands(5, 4, 3), ".output/test/landsat_543_3.img")
    assert hash == "700167ccacd6f63a3f46fcf7c2e41f71"


def test_saving_landsat_bil():
    hash = save(landsat.extract_bands(5, 4, 3), ".output/test/landsat_543_2.bil")
    assert hash == "ff0b8c95a824c9550d12c203132ca4a9"
