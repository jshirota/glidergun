import pytest

from glidergun._grid import grid
from glidergun._stack import stack


def test_create_stack_1():
    g = grid((40, 30), (0, 0, 4, 3))
    s = stack(g, g, g)
    assert s.crs == 4326


def test_create_stack_2():
    g = grid((40, 30), (0, 0, 4, 3))
    s = stack(g, g, g)
    assert s.grids[0].extent == s.extent
    assert s.grids[1].extent == s.extent
    assert s.grids[2].extent == s.extent


def test_create_stack_3():
    s1 = stack(
        grid((40, 30), (0, 0, 4, 3)),
        grid((40, 40), (0, 0, 4, 3)),
        grid((40, 50), (0, 0, 4, 3)),
    )
    s2 = stack(
        grid((50, 20), (0, 0, 4, 3)),
        grid((50, 30), (0, 0, 4, 3)),
        grid((50, 40), (0, 0, 4, 3)),
    )
    s3 = s1 + s2
    assert s3.extent == s1.extent
    assert s3.grids[0].cell_size == s1.grids[0].cell_size
    assert s3.grids[1].cell_size == s1.grids[1].cell_size
    assert s3.grids[2].cell_size == s1.grids[2].cell_size


def test_stack_crs():
    g1 = grid((40, 30), (0, 0, 4, 3), crs=4326)
    g2 = grid((40, 30), (0, 0, 4, 3), crs=3857)

    with pytest.raises(ValueError):
        stack(g1, g2)


def test_stack_different_sizes():
    g1 = grid((40, 30), (0, 0, 4, 3))
    g2 = grid((50, 40), (0, 0, 4, 3))
    s = stack(g1, g2)
    assert s.grids[0].cell_size != s.grids[1].cell_size
    assert s.grids[0].extent == s.grids[1].extent


def test_stack_empty():
    s = stack()
    assert len(s.grids) == 0


def test_stack_single_grid():
    g = grid((40, 30), (0, 0, 4, 3))
    s = stack(g)
    assert len(s.grids) == 1
