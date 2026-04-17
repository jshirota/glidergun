import os
import shutil

import pytest

from glidergun import Stack, search, stack
from glidergun.grid import _to_uint8_range


@pytest.fixture(scope="session")
def sentinel():
    return search("sentinel-2-l2a", [-122.50, 37.74, -122.48, 37.76])[0].get(["B04", "B03", "B02"], resample_by=10)


def test_extract_bands(sentinel):
    s = sentinel.extract_bands(3, 2, 1)
    assert s.grids[0].sha256 == sentinel.grids[2].sha256
    assert s.grids[1].sha256 == sentinel.grids[1].sha256
    assert s.grids[2].sha256 == sentinel.grids[0].sha256


def test_op_mul(sentinel):
    s = sentinel * 1000
    for g1, g2 in zip(sentinel.grids, s.grids, strict=True):
        assert pytest.approx(g2.min, 0.001) == g1.min * 1000
        assert pytest.approx(g2.max, 0.001) == g1.max * 1000


def test_op_div(sentinel):
    s = sentinel / 1000
    for g1, g2 in zip(sentinel.grids, s.grids, strict=True):
        assert pytest.approx(g2.min, 0.001) == g1.min / 1000
        assert pytest.approx(g2.max, 0.001) == g1.max / 1000


def test_op_add(sentinel):
    s = sentinel + 1000
    for g1, g2 in zip(sentinel.grids, s.grids, strict=True):
        assert pytest.approx(g2.min, 0.001) == g1.min + 1000
        assert pytest.approx(g2.max, 0.001) == g1.max + 1000


def test_op_sub(sentinel):
    s = sentinel - 1000
    for g1, g2 in zip(sentinel.grids, s.grids, strict=True):
        assert pytest.approx(g2.min, 0.001) == g1.min - 1000
        assert pytest.approx(g2.max, 0.001) == g1.max - 1000


def test_percent_clip(sentinel):
    s = sentinel.percent_clip(1, 99)
    for g1, g2 in zip(sentinel.grids, s.grids, strict=True):
        assert pytest.approx(g2.min, 0.001) == g1.percentile(1)
        assert pytest.approx(g2.max, 0.001) == g1.percentile(99)


def test_to_uint8_range(sentinel):
    s = sentinel.each(lambda g: _to_uint8_range(g))
    for g in s.grids:
        assert pytest.approx(g.min, 0.001) == 0
        assert pytest.approx(g.max, 0.001) == 255


def test_pca(sentinel):
    s = sentinel.pca(2)
    assert len(s.grids) == 2


def test_project(sentinel):
    s = sentinel.project(4326)
    assert s.crs.wkt.startswith('GEOGCS["WGS 84",DATUM["WGS_1984",')


def test_properties_2(sentinel):
    for g in sentinel.grids:
        assert g.crs == sentinel.crs
        assert g.extent == sentinel.extent
        assert g.xmin == sentinel.xmin
        assert g.ymin == sentinel.ymin
        assert g.xmax == sentinel.xmax
        assert g.ymax == sentinel.ymax


def test_resample(sentinel):
    s = sentinel.resample(1000)
    assert pytest.approx(s.first.cell_size.x, 0.001) == 1000
    assert pytest.approx(s.first.cell_size.y, 0.001) == 1000
    for g in s.grids:
        assert pytest.approx(g.cell_size.x, 0.001) == 1000
        assert pytest.approx(g.cell_size.y, 0.001) == 1000


def test_resample_2(sentinel):
    s = sentinel.resample((1000, 600))
    assert pytest.approx(s.first.cell_size.x, 0.001) == 1000
    assert pytest.approx(s.first.cell_size.y, 0.001) == 600
    for g in s.grids:
        assert pytest.approx(g.cell_size.x, 0.001) == 1000
        assert pytest.approx(g.cell_size.y, 0.001) == 600


def test_resample_by(sentinel):
    s0 = sentinel.type("int32")
    s1 = s0.resample_by(2.0)
    s2 = s0.resample_by(1.0)
    assert s0.dtype == "int32"
    assert s1.dtype == "float32"
    assert s2.dtype == "float32"
    assert pytest.approx(s1.cell_size.x, 0.001) == s0.cell_size.x * 2.0
    assert pytest.approx(s1.cell_size.y, 0.001) == s0.cell_size.y * 2.0
    assert s0.cell_size == s2.cell_size


def save(s1: Stack, file: str):
    folder = "tests/output/temp"
    file_path = f"{folder}/{file}"
    os.makedirs(folder, exist_ok=True)
    s1.save(file_path)
    s2 = stack(file_path)
    assert s2.extent == s1.extent
    shutil.rmtree(folder)


def test_save_img(sentinel):
    save(sentinel, "test_stack.img")


def test_save_tif(sentinel):
    save(sentinel, "test_stack.tif")


def test_save_jpg(sentinel):
    save(sentinel.extract_bands(3, 2, 1), "test_stack.jpg")


def test_save_png(sentinel):
    save(sentinel.extract_bands(2, 3, 1), "test_stack.png")
