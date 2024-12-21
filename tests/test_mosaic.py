from rasterio.io import MemoryFile

from glidergun._grid import Grid, grid
from glidergun._mosaic import Mosaic, mosaic
from glidergun._stack import Stack


def test_mosaic_init_with_files():
    g1 = grid((40, 30), (0, 0, 4, 3))
    g2 = grid((50, 40), (0, 0, 4, 3))
    m = Mosaic(g1, g2)
    assert len(m.files) == 2
    assert m.files[0] == g1
    assert m.files[1] == g2


def test_mosaic_init_with_memoryfile():
    with MemoryFile() as memfile:
        g = grid((40, 30), (0, 0, 4, 3))
        m = Mosaic(memfile, g)
        assert len(m.files) == 2
        assert isinstance(m.files[0], MemoryFile)
        assert m.files[1] == g


def test_mosaic_read():
    g1 = grid((40, 30), (0, 0, 4, 3))
    g2 = grid((50, 40), (0, 0, 4, 3))
    m = Mosaic(g1, g2)
    extent = (0, 0, 4, 3)
    grids = list(m._read(extent, 1))
    assert len(grids) == 2
    assert grids[0] == g1
    assert grids[1] == g2


def test_mosaic_clip_single_index():
    g1 = grid((40, 30), (0, 0, 4, 3))
    g2 = grid((50, 40), (0, 0, 4, 3))
    m = Mosaic(g1, g2)
    clipped_grid = m.clip(0, 0, 4, 3, index=1)
    assert isinstance(clipped_grid, Grid)


def test_mosaic_clip_multiple_indices():
    g1 = grid((40, 30), (0, 0, 4, 3))
    g2 = grid((50, 40), (0, 0, 4, 3))
    m = Mosaic(g1, g2)
    clipped_stack = m.clip(0, 0, 4, 3, index=(1, 2))
    assert isinstance(clipped_stack, Stack)
    assert len(clipped_stack.grids) == 2


def test_mosaic_function_with_strings():
    m = mosaic("file1.tif", "file2.tif")
    assert isinstance(m, Mosaic)
    assert len(m.files) == 2
    assert m.files[0] == "file1.tif"
    assert m.files[1] == "file2.tif"


def test_mosaic_function_with_grids():
    g1 = grid((40, 30), (0, 0, 4, 3))
    g2 = grid((50, 40), (0, 0, 4, 3))
    result_grid = mosaic(g1, g2)
    assert isinstance(result_grid, Grid)
