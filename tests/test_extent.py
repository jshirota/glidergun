from glidergun import Extent, grid


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


def test_extent_contains():
    e1 = Extent(0, 0, 4, 4)
    e2 = (1, 1, 3, 3)
    assert e1.contains(e2) is True
    e3 = (0, 0, 5, 5)
    assert e1.contains(e3) is False


def test_extent_not_contains():
    e1 = Extent(0, 0, 4, 4)
    e2 = (5, 5, 6, 6)
    assert e1.contains(e2) is False


def test_extent_intersects():
    e1 = Extent(0, 0, 4, 4)
    e2 = (2, 2, 6, 6)
    assert e1.intersects(e2) is True
    e3 = (5, 5, 7, 7)
    assert e1.intersects(e3) is False


def test_extent_not_intersects():
    e1 = Extent(0, 0, 4, 4)
    e2 = (5, 5, 6, 6)
    assert e1.intersects(e2) is False


def test_intersects_but_not_contains():
    e1 = Extent(0, 0, 4, 4)
    e2 = (3, 3, 5, 5)
    assert e1.intersects(e2) is True
    assert e1.contains(e2) is False


def test_adjust():
    e1 = Extent(0, 0, 4, 4)
    e2 = e1.adjust(-1, -1, 1, 1)
    expected = Extent(-1, -1, 5, 5)
    assert e2 == expected


def test_buffer():
    e1 = Extent(0, 0, 4, 4)
    e2 = e1.buffer(1)
    expected = Extent(-1, -1, 5, 5)
    assert e2 == expected
    e3 = e1.buffer(0.5)
    expected2 = Extent(-0.5, -0.5, 4.5, 4.5)
    assert e3 == expected2
    e4 = e1.buffer(0)
    expected3 = Extent(0, 0, 4, 4)
    assert e4 == expected3
    e5 = e1.buffer(-1)
    expected4 = Extent(1, 1, 3, 3)
    assert e5 == expected4 == expected4
    try:
        e1.buffer(-3)
    except ValueError as ex:
        assert str(ex) == "Adjusted extent is not valid: (3, 3, 1, 1)"


def test_tiles():
    e1 = Extent(0, 0, 4, 4)
    tiles = e1.tiles(2, 2)
    expected = [
        Extent(0, 0, 2, 2),
        Extent(0, 2, 2, 4),
        Extent(2, 0, 4, 2),
        Extent(2, 2, 4, 4),
    ]
    assert tiles == expected
