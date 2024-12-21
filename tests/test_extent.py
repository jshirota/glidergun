from glidergun._grid import grid
from glidergun._types import Extent


def test_extent_1():
    g1 = grid((40, 30), (0, 0, 4, 3))
    g2 = grid((40, 30), (0, 0, 4, 4))
    g3 = g1 + g2
    assert g1.extent == g3.extent


def test_extent_2():
    g1 = grid((40, 30), (0, 0, 4, 3))
    g2 = grid((40, 30), (0, 0, 5, 3))
    g3 = g1 + g2
    assert g1.extent == g3.extent


def test_extent_3():
    g1 = grid((40, 40), (0, 0, 4, 3))
    g2 = grid((40, 30), (0, 0, 4, 4))
    g3 = g1 + g2
    assert g1.extent == g3.extent


def test_extent_4():
    g1 = grid((40, 30), (0, 0, 4, 3))
    g2 = grid((40, 40), (0, 0, 5, 3))
    g3 = g1 + g2
    assert g1.extent == g3.extent


def test_extent_intersect():
    e1 = Extent(0, 0, 4, 4)
    e2 = Extent(2, 2, 6, 6)
    result = e1.intersect(e2)
    expected = Extent(2, 2, 4, 4)
    assert result == expected


def test_extent_union():
    e1 = Extent(0, 0, 4, 4)
    e2 = Extent(2, 2, 6, 6)
    result = e1.union(e2)
    expected = Extent(0, 0, 6, 6)
    assert result == expected


def test_extent_and_operator():
    e1 = Extent(0, 0, 4, 4)
    e2 = Extent(2, 2, 6, 6)
    result = e1 & e2
    expected = Extent(2, 2, 4, 4)
    assert result == expected


def test_extent_or_operator():
    e1 = Extent(0, 0, 4, 4)
    e2 = Extent(2, 2, 6, 6)
    result = e1 | e2
    expected = Extent(0, 0, 6, 6)
    assert result == expected
