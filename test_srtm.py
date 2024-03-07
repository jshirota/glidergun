import os
import pytest
import rasterio
import shutil
from glidergun import *

dem = grid("./.data/n55_e008_1arc_v3.bil")


def test_properties():
    assert dem.width == 1801
    assert dem.height == 3601
    assert dem.dtype == "float32"
    assert dem.md5 == "09b41b3363bd79a87f28e3c5c4716724"


def test_con():
    g1 = con(dem > 50, 3, 7.123)
    g2 = con(30 <= dem < 40, 2.345, g1)
    assert pytest.approx(g2.min, 0.001) == 2.345
    assert pytest.approx(g2.max, 0.001) == 7.123


def test_fill_nan():
    g1 = dem.set_nan(77)
    assert g1.has_nan
    g2 = g1.fill_nan()
    assert not g2.has_nan


def test_project():
    g = dem.project(3857)
    assert g.crs.wkt.startswith('PROJCS["WGS 84 / Pseudo-Mercator",')


def test_resample():
    g = dem.resample(0.01)
    assert g.cell_size == (0.01, 0.01)


def test_save():
    memory_file = rasterio.MemoryFile()
    dem.save(memory_file)
    g = grid(memory_file)
    assert g.md5 == dem.md5


def save(file: str, strict: bool = True):
    folder = ".output/test"
    file_path = f"{folder}/{file}"
    os.makedirs(folder, exist_ok=True)
    dem.save(file_path)
    g = grid(file_path)
    if strict:
        assert g.md5 == dem.md5
    assert g.extent == dem.extent
    shutil.rmtree(folder)


def test_save_2():
    save("test_save_2.bil")


def test_save_3():
    save("test_save_2.img")


def test_save_4():
    save("test_save_2.tif")


def test_save_5():
    save("test_save_2.jpg", strict=False)


def test_save_6():
    save("test_save_2.png", strict=False)
