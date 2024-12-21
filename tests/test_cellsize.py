from glidergun._types import CellSize


def test_cellsize_mul():
    cs = CellSize(2.0, 3.0)
    result = cs * 2
    expected = CellSize(4.0, 6.0)
    assert result == expected


def test_cellsize_rmul():
    cs = CellSize(2.0, 3.0)
    result = 2 * cs
    expected = CellSize(4.0, 6.0)
    assert result == expected


def test_cellsize_truediv():
    cs = CellSize(4.0, 6.0)
    result = cs / 2
    expected = CellSize(2.0, 3.0)
    assert result == expected
