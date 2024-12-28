import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from glidergun._types import CellSize, Extent, GridCore, PointValue


def test_gridcore():
    data = np.array([[1, 2], [3, 4]])
    transform = Affine.translation(0, 0)
    crs = CRS.from_epsg(4326)
    grid_core = GridCore(data, transform, crs)

    assert np.array_equal(grid_core.data, data)
    assert grid_core.transform == transform
    assert grid_core.crs == crs


def test_extent_intersect():
    extent1 = Extent(0, 0, 10, 10)
    extent2 = Extent(5, 5, 15, 15)
    expected = Extent(5, 5, 10, 10)

    assert extent1.intersect(extent2) == expected


def test_extent_union():
    extent1 = Extent(0, 0, 10, 10)
    extent2 = Extent(5, 5, 15, 15)
    expected = Extent(0, 0, 15, 15)

    assert extent1.union(extent2) == expected


def test_cellsize_mul():
    cell_size = CellSize(2, 3)
    expected = CellSize(4, 6)

    assert cell_size * 2 == expected
    assert 2 * cell_size == expected


def test_cellsize_truediv():
    cell_size = CellSize(4, 6)
    expected = CellSize(2, 3)

    assert cell_size / 2 == expected


def test_pointvalue():
    point_value = PointValue(1, 2, 3)

    assert point_value.x == 1
    assert point_value.y == 2
    assert point_value.value == 3
