import os
import shutil
from functools import lru_cache

import pytest
import rasterio

from glidergun import Stack, search, stack
from glidergun._grid import _to_uint8_range


@lru_cache
def sentinel():
    return search("sentinel-2-l2a", [-122.50, 37.74, -122.48, 37.76])[0].download(["B04", "B03", "B02"])


def test_extract_bands():
    s = sentinel().extract_bands(3, 2, 1)
    assert s.grids[0].sha256 == sentinel().grids[2].sha256
    assert s.grids[1].sha256 == sentinel().grids[1].sha256
    assert s.grids[2].sha256 == sentinel().grids[0].sha256


def test_op_mul():
    s = sentinel() * 1000
    for g1, g2 in zip(sentinel().grids, s.grids, strict=False):
        assert pytest.approx(g2.min, 0.001) == g1.min * 1000
        assert pytest.approx(g2.max, 0.001) == g1.max * 1000


def test_op_div():
    s = sentinel() / 1000
    for g1, g2 in zip(sentinel().grids, s.grids, strict=False):
        assert pytest.approx(g2.min, 0.001) == g1.min / 1000
        assert pytest.approx(g2.max, 0.001) == g1.max / 1000


def test_op_add():
    s = sentinel() + 1000
    for g1, g2 in zip(sentinel().grids, s.grids, strict=False):
        assert pytest.approx(g2.min, 0.001) == g1.min + 1000
        assert pytest.approx(g2.max, 0.001) == g1.max + 1000


def test_op_sub():
    s = sentinel() - 1000
    for g1, g2 in zip(sentinel().grids, s.grids, strict=False):
        assert pytest.approx(g2.min, 0.001) == g1.min - 1000
        assert pytest.approx(g2.max, 0.001) == g1.max - 1000


def test_percent_clip():
    s = sentinel().percent_clip(1, 99)
    for g1, g2 in zip(sentinel().grids, s.grids, strict=False):
        assert pytest.approx(g2.min, 0.001) == g1.percentile(1)
        assert pytest.approx(g2.max, 0.001) == g1.percentile(99)


def test_to_uint8_range():
    s = sentinel().each(lambda _, g: _to_uint8_range(g))
    for g in s.grids:
        assert pytest.approx(g.min, 0.001) == 0
        assert pytest.approx(g.max, 0.001) == 255


def test_pca():
    s = sentinel().pca(2)
    assert len(s.grids) == 2


def test_project():
    s = sentinel().project(4326)
    assert s.crs.wkt.startswith('GEOGCS["WGS 84",DATUM["WGS_1984",')


def test_properties_2():
    for g in sentinel().grids:
        assert g.crs == sentinel().crs
        assert g.extent == sentinel().extent
        assert g.xmin == sentinel().xmin
        assert g.ymin == sentinel().ymin
        assert g.xmax == sentinel().xmax
        assert g.ymax == sentinel().ymax


def test_resample():
    s = sentinel().resample(1000)
    assert pytest.approx(s.grids[0].cell_size.x, 0.001) == 1000
    assert pytest.approx(s.grids[0].cell_size.y, 0.001) == 1000
    for g in s.grids:
        assert pytest.approx(g.cell_size.x, 0.001) == 1000
        assert pytest.approx(g.cell_size.y, 0.001) == 1000


def test_resample_2():
    s = sentinel().resample((1000, 600))
    assert pytest.approx(s.grids[0].cell_size.x, 0.001) == 1000
    assert pytest.approx(s.grids[0].cell_size.y, 0.001) == 600
    for g in s.grids:
        assert pytest.approx(g.cell_size.x, 0.001) == 1000
        assert pytest.approx(g.cell_size.y, 0.001) == 600


def test_resample_by():
    s0 = sentinel().type("int32")
    s1 = s0.resample_by(2.0)
    s2 = s0.resample_by(1.0)
    assert s0.dtype == "int32"
    assert s1.dtype == "float32"
    assert s2.dtype == "float32"
    assert pytest.approx(s1.cell_size.x, 0.001) == s0.cell_size.x * 2.0
    assert pytest.approx(s1.cell_size.y, 0.001) == s0.cell_size.y * 2.0
    assert s0.cell_size == s2.cell_size


def save(s1: Stack, file: str, strict: bool = True):
    folder = "tests/output/temp"
    file_path = f"{folder}/{file}"
    os.makedirs(folder, exist_ok=True)
    s1.save(file_path)
    s2 = stack(file_path)
    if strict:
        assert s2.sha256s == s1.sha256s
    assert s2.extent == s1.extent
    shutil.rmtree(folder)


def test_save_memory():
    memory_file = rasterio.MemoryFile()
    sentinel().save(memory_file)
    s = stack(memory_file)
    assert s.sha256s == sentinel().sha256s


def test_save_img():
    save(sentinel(), "test_stack.img")


def test_save_tif():
    save(sentinel(), "test_stack.tif")


def test_save_jpg():
    save(sentinel().extract_bands(3, 2, 1), "test_stack.jpg", strict=False)


def test_save_png():
    save(sentinel().extract_bands(2, 3, 1), "test_stack.png", strict=False)


def test_bytes():
    assert stack(sentinel().to_bytes()).sha256s == sentinel().sha256s
