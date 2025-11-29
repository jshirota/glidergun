import numpy as np
import pytest
from rasterio.crs import CRS

from glidergun import Extent, grid, stack


def test_create_stack_1():
    g = grid((40, 30), (0, 0, 4, 3))
    s = stack([g, g, g])
    assert s.crs == 4326


def test_create_stack_2():
    g = grid((40, 30), (0, 0, 4, 3))
    s = stack([g, g, g])
    assert s.grids[0].extent == s.extent
    assert s.grids[1].extent == s.extent
    assert s.grids[2].extent == s.extent


def test_stack_crs():
    g1 = grid((40, 30), (0, 0, 4, 3), crs=4326)
    g2 = grid((40, 30), (0, 0, 4, 3), crs=3857)

    with pytest.raises(ValueError):
        stack([g1, g2])


def test_stack_empty():
    s = stack([])
    assert len(s.grids) == 0


def test_stack_single_grid():
    g = grid((40, 30), (0, 0, 4, 3))
    s = stack([g])
    assert len(s.grids) == 1


@pytest.fixture
def sample_grids():
    data = np.random.rand(10, 10)
    extent = Extent(0, 0, 10, 10)
    crs = CRS.from_epsg(4326)
    grid1 = grid(data, extent, crs)
    grid2 = grid(data * 2, extent, crs)
    return grid1, grid2


def test_stack_creation(sample_grids):
    grid1, grid2 = sample_grids
    s = stack([grid1, grid2])
    assert len(s.grids) == 2
    assert s.grids[0] == grid1
    assert s.grids[1] == grid2


def test_stack_repr(sample_grids):
    grid1, grid2 = sample_grids
    s = stack([grid1, grid2])
    assert (
        repr(s)
        == f"image: 10x10 float32 | crs: {grid1.crs} | count: 2 | rgb: (1, 2, 3) | cell: (1.0, 1.0) | extent: (0.0, 0.0, 10.0, 10.0)"  # noqa: E501
    )


def test_stack_addition(sample_grids):
    grid1, grid2 = sample_grids
    s = stack([grid1, grid2])
    result = s + 1
    assert np.all(result.grids[0].data == grid1.data + 1)
    assert np.all(result.grids[1].data == grid2.data + 1)


def test_stack_subtraction(sample_grids):
    grid1, grid2 = sample_grids
    s = stack([grid1, grid2])
    result = s - 1
    assert np.all(result.grids[0].data == grid1.data - 1)
    assert np.all(result.grids[1].data == grid2.data - 1)


def test_stack_multiplication(sample_grids):
    grid1, grid2 = sample_grids
    s = stack([grid1, grid2])
    result = s * 2
    assert np.all(result.grids[0].data == grid1.data * 2)
    assert np.all(result.grids[1].data == grid2.data * 2)


def test_stack_division(sample_grids):
    grid1, grid2 = sample_grids
    s = stack([grid1, grid2])
    result = s / 2
    assert np.all(result.grids[0].data == grid1.data / 2)
    assert np.all(result.grids[1].data == grid2.data / 2)


def test_stack_resample(sample_grids):
    grid1, grid2 = sample_grids
    s = stack([grid1, grid2])
    result = s.resample(5)
    assert result.grids[0].cell_size == (5, 5)


def test_stack_clip(sample_grids):
    grid1, grid2 = sample_grids
    s = stack([grid1, grid2])
    result = s.clip((2, 2, 8, 8))
    assert result.grids[0].extent == Extent(2, 2, 8, 8)
    assert result.grids[1].extent == Extent(2, 2, 8, 8)


def test_stack_save(sample_grids, tmp_path):
    grid1, grid2 = sample_grids
    s = stack([grid1, grid2])
    file_path = tmp_path / "test.tif"
    s.save(str(file_path))
    assert file_path.exists()
