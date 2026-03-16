import numpy as np
import pytest
import rasterio

from glidergun import grid
from glidergun.stack import stack


def test_file():
    g = grid("./data/n55_e008_1arc_v3.bil")
    assert g
    assert g.crs.is_valid


def test_dataset():
    with rasterio.open("./data/n55_e008_1arc_v3.bil") as dataset:
        g = grid(dataset)
        assert g
        assert g.crs.is_valid


def test_ndarray():
    g = grid("./data/n55_e008_1arc_v3.bil").project(3857)
    g2 = grid(g.data, g.extent, g.crs)
    assert g2.extent == g.extent
    assert g.crs.is_valid
    assert g2.crs == g.crs
    assert g2.data.shape == g.data.shape


def test_ndarray_no_extent():
    g = grid(np.array([[1, 2], [3, 4]]))
    assert g.crs.is_valid
    assert g.crs == 4326


def test_box():
    g = grid((40, 30))
    assert g.extent == (0.0, 0.0, 0.04, 0.03)
    assert g.crs.is_valid
    assert g.crs == 4326
    assert g.width == 40
    assert g.height == 30


def test_constant_int():
    g = grid(123, (-120, 30, -119, 31), 4326, 0.1)
    assert g.extent == (-120, 30, -119, 31)
    assert g.crs.is_valid
    assert g.crs == 4326
    assert g.cell_size == (0.1, 0.1)
    assert pytest.approx(g.min, 0.001) == 123
    assert pytest.approx(g.max, 0.001) == 123
    assert pytest.approx(g.mean, 0.001) == 123


def test_constant_float():
    g = grid(123.456, (-120, 30, -119, 31), 4326, 0.1)
    assert g.extent == (-120, 30, -119, 31)
    assert g.crs.is_valid
    assert g.crs == 4326
    assert g.cell_size == (0.1, 0.1)
    assert pytest.approx(g.min, 0.001) == 123.456
    assert pytest.approx(g.max, 0.001) == 123.456
    assert pytest.approx(g.mean, 0.001) == 123.456


def test_bing_grid():
    g = grid("microsoft")
    assert g.crs.is_valid
    assert g.crs == 3857


def test_bing_stack():
    s = stack("microsoft")
    assert s.crs.is_valid
    assert s.crs == 3857
