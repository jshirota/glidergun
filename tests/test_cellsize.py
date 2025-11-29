from glidergun import CellSize


def test_cellsize_mul():
    assert CellSize(2.0, 3.0) * 2 == CellSize(4.0, 6.0)


def test_cellsize_rmul():
    assert 2 * CellSize(2.0, 3.0) == CellSize(4.0, 6.0)


def test_cellsize_truediv():
    assert CellSize(2.0, 3.0) == CellSize(4.0, 6.0) / 2
