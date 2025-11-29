import numpy as np
import pytest
from rasterio.transform import from_origin

from glidergun import CellSize, Extent, grid


@pytest.fixture
def sample_grid():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    transform = from_origin(0, 3, 1, 1)  # top-left corner at (0, 3), cell size 1x1
    return grid(data, transform, 4326)


def test_width(sample_grid):
    assert sample_grid.width == 3
    assert isinstance(sample_grid.width, int)


def test_height(sample_grid):
    assert sample_grid.height == 3
    assert isinstance(sample_grid.height, int)


def test_dtype(sample_grid):
    assert sample_grid.dtype == "float32"
    assert isinstance(sample_grid.dtype, str)


def test_nodata(sample_grid):
    assert isinstance(sample_grid.nodata, float)
    assert isinstance(sample_grid.type("uint16").nodata, int)


def test_has_nan(sample_grid):
    assert not sample_grid.has_nan
    assert isinstance(sample_grid.has_nan, bool)


def test_xmin(sample_grid):
    assert sample_grid.xmin == 0
    assert isinstance(sample_grid.xmin, float)


def test_ymin(sample_grid):
    assert sample_grid.ymin == 0
    assert isinstance(sample_grid.ymin, float)


def test_xmax(sample_grid):
    assert sample_grid.xmax == 3
    assert isinstance(sample_grid.xmax, float)


def test_ymax(sample_grid):
    assert sample_grid.ymax == 3
    assert isinstance(sample_grid.ymax, float)


def test_extent(sample_grid):
    assert sample_grid.extent == Extent(0, 0, 3, 3)
    assert isinstance(sample_grid.extent[0], float)
    assert isinstance(sample_grid.extent[1], float)
    assert isinstance(sample_grid.extent[2], float)
    assert isinstance(sample_grid.extent[3], float)


def test_mean(sample_grid):
    assert sample_grid.mean == 5.0
    assert isinstance(sample_grid.mean, float)


def test_std(sample_grid):
    assert sample_grid.std == 2.58198881149292
    assert isinstance(sample_grid.std, float)


def test_min(sample_grid):
    assert sample_grid.min == 1.0
    assert isinstance(sample_grid.min, float)


def test_max(sample_grid):
    assert sample_grid.max == 9.0
    assert isinstance(sample_grid.max, float)


def test_cell_size(sample_grid):
    assert sample_grid.cell_size == CellSize(1, 1)
    assert isinstance(sample_grid.cell_size[0], float)
    assert isinstance(sample_grid.cell_size[1], float)


def test_bins(sample_grid):
    expected_bins = {
        1.0: 1,
        2.0: 1,
        3.0: 1,
        4.0: 1,
        5.0: 1,
        6.0: 1,
        7.0: 1,
        8.0: 1,
        9.0: 1,
    }
    assert sample_grid.bins == expected_bins
