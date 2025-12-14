import numpy as np

from glidergun import grid


def test_grid_reclass():
    g = grid(np.array([[1, 2], [3, 4]]))
    result = g.reclass((1, 2, 10), (2, 3, 20), (3, 4, 30))
    expected = grid(np.array([[10, 20], [30, np.nan]]))
    assert np.array_equal(result.data, expected.data, equal_nan=True)


def test_grid_percentile():
    g = grid(np.array([[1, 2], [3, 4]]))
    result = g.percentile(50)
    expected = 2.5
    assert result == expected


def test_grid_slice():
    g = grid(np.array([[1, 2], [3, 4]]))
    result = g.slice(2)
    expected = grid(np.array([[1, 1], [2, 2]]))
    assert np.array_equal(result.data, expected.data)


def test_grid_replace():
    g = grid(np.array([[1, 2], [3, 4]]))
    result = g.con(2, 20)
    expected = grid(np.array([[1, 20], [3, 4]]))
    assert np.array_equal(result.data, expected.data)


def test_grid_set_nan():
    g = grid(np.array([[1, 2], [3, 4]]))
    result = g.set_nan(2)
    expected = grid(np.array([[1, np.nan], [3, 4]]))
    assert np.array_equal(result.data, expected.data, equal_nan=True)


def test_grid_value():
    g = grid(np.array([[1, 2], [3, 4]]), extent=(0, 0, 2, 2))
    result = g.value_at(1, 1)
    expected = 4
    assert result == expected


def test_grid_interp_clough_tocher():
    points = [(1, 1, 10), (4, 7, 40), (8, 2, 7)]
    extent = (0, 0, 10, 10)
    g = grid(points, extent, 4326, 1).interp_clough_tocher()
    assert g.extent == extent
    assert g.crs == 4326
    assert g.cell_size == (1.0, 1.0)
    assert g.value_at(2, 2) == 12.54273509979248
    assert g.has_nan is True
    assert g.sha256 == "8c972e4687e1b3548c7799ca5b7f40e069e02d91c829a14cac57b6945d550b08"


def test_grid_interp_linear():
    points = [(1, 1, 10), (4, 7, 40), (8, 2, 7)]
    extent = (0, 0, 10, 10)
    g = grid(points, extent, 4326, 1).interp_linear()
    assert g.extent == extent
    assert g.crs == 4326
    assert g.cell_size == (1.0, 1.0)
    assert g.value_at(2, 2) == 12.54273509979248
    assert g.has_nan is True
    assert g.sha256 == "7f16929a87f9e8f81ca58fbab55f8a007547c790f42aa8ea8f2b10b1f9ce7ae4"


def test_grid_interp_nearest():
    points = [(40, 30, 123), (30, 34, 777)]
    extent = (28, 28, 42, 36)
    g = grid(points, extent, 4326, 1).interp_nearest()
    assert g.extent == extent
    assert g.crs == 4326
    assert g.cell_size == (1.0, 1.0)
    assert g.value_at(30, 30) == 777
    assert g.value_at(40, 30) == 123
    assert g.has_nan is False
    assert g.sha256 == "9e473990c51c3647ee8b1f80475c558ba781e5aed97e0dc9b5950d7735a4d5ba"


def test_grid_interp_rbf():
    points = [(1, 1, 10), (4, 7, 40), (8, 2, 7)]
    extent = (0, 0, 10, 10)
    g = grid(points, extent, 4326, 1).interp_rbf()
    assert g.extent == extent
    assert g.crs == 4326
    assert g.cell_size == (1.0, 1.0)
    assert g.value_at(2, 2) == 12.54273509979248
    assert g.has_nan is False
    assert g.sha256 == "6f6a07bb0b3c5e827cef96c956899f2e82bcc8d042b8f1f9544d8eaa3844250b"


def test_grid_interp_compare():
    points = [(1, 1, 10), (4, 7, 40), (8, 2, 7)]
    extent = (0, 0, 10, 10)
    g1 = grid(points, extent, 4326, 1).interp_linear()
    g2 = grid(points, extent, 4326, 1).interp_rbf()
    g3 = g1 - g2
    assert g3.min == g3.max == 0

    points = [(1, 1, 10), (4, 7, 40), (8, 2, 7), (9, 2, 7)]
    extent = (0, 0, 10, 10)
    g1 = grid(points, extent, 4326, 1).interp_linear()
    g2 = grid(points, extent, 4326, 1).interp_rbf()
    g3 = g1 - g2
    assert g3.min != g3.max


def test_slope():
    g = grid(
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
    )

    assert g.slope().min == 0
    assert g.slope(True).min == 0
