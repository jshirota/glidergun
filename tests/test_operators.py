import numpy as np

from glidergun import grid


def test_add():
    g1 = grid(np.array([[1, 2], [3, 4]]))
    g2 = grid(np.array([[5, 6], [7, 8]]))
    result = g1 + g2
    expected = grid(np.array([[6, 8], [10, 12]]))
    assert np.array_equal(result.data, expected.data)


def test_sub():
    g1 = grid(np.array([[5, 6], [7, 8]]))
    g2 = grid(np.array([[1, 2], [3, 4]]))
    result = g1 - g2
    expected = grid(np.array([[4, 4], [4, 4]]))
    assert np.array_equal(result.data, expected.data)


def test_mul():
    g1 = grid(np.array([[1, 2], [3, 4]]))
    g2 = grid(np.array([[5, 6], [7, 8]]))
    result = g1 * g2
    expected = grid(np.array([[5, 12], [21, 32]]))
    assert np.array_equal(result.data, expected.data)


def test_truediv():
    g1 = grid(np.array([[10, 20], [30, 40]]))
    g2 = grid(np.array([[2, 4], [5, 8]]))
    result = g1 / g2
    expected = grid(np.array([[5, 5], [6, 5]]))
    assert np.array_equal(result.data, expected.data)


def test_floordiv():
    g1 = grid(np.array([[10, 20], [30, 40]]))
    g2 = grid(np.array([[3, 4], [5, 8]]))
    result = g1 // g2
    expected = grid(np.array([[3, 5], [6, 5]]))
    assert np.array_equal(result.data, expected.data)


def test_mod():
    g1 = grid(np.array([[10, 20], [30, 40]]))
    g2 = grid(np.array([[3, 4], [5, 8]]))
    result = g1 % g2
    expected = grid(np.array([[1, 0], [0, 0]]))
    assert np.array_equal(result.data, expected.data)


def test_pow():
    g1 = grid(np.array([[2, 3], [4, 5]]))
    g2 = grid(np.array([[3, 2], [2, 3]]))
    result = g1**g2
    expected = grid(np.array([[8, 9], [16, 125]]))
    assert np.array_equal(result.data, expected.data)


def test_lt():
    g1 = grid(np.array([[1, 2], [3, 4]]))
    g2 = grid(np.array([[2, 2], [4, 4]]))
    result = g1 < g2
    expected = grid(np.array([[1, 0], [1, 0]]))
    assert np.array_equal(result.data, expected.data)


def test_gt():
    g1 = grid(np.array([[1, 2], [3, 4]]))
    g2 = grid(np.array([[2, 2], [4, 4]]))
    result = g1 > g2
    expected = grid(np.array([[0, 0], [0, 0]]))
    assert np.array_equal(result.data, expected.data)


def test_le():
    g1 = grid(np.array([[1, 2], [3, 4]]))
    g2 = grid(np.array([[2, 2], [4, 4]]))
    result = g1 <= g2
    expected = grid(np.array([[1, 1], [1, 1]]))
    assert np.array_equal(result.data, expected.data)


def test_ge():
    g1 = grid(np.array([[1, 2], [3, 4]]))
    g2 = grid(np.array([[2, 2], [4, 4]]))
    result = g1 >= g2
    expected = grid(np.array([[0, 1], [0, 1]]))
    assert np.array_equal(result.data, expected.data)


def test_eq():
    g1 = grid(np.array([[1, 2], [3, 4]]))
    g2 = grid(np.array([[1, 2], [3, 4]]))
    result = g1 == g2
    expected = grid(np.array([[1, 1], [1, 1]]))
    assert np.array_equal(result.data, expected.data)


def test_ne():
    g1 = grid(np.array([[1, 2], [3, 4]]))
    g2 = grid(np.array([[1, 2], [4, 4]]))
    result = g1 != g2
    expected = grid(np.array([[0, 0], [1, 0]]))
    assert np.array_equal(result.data, expected.data)


def test_and():
    g1 = grid(np.array([[1, 0], [1, 0]]))
    g2 = grid(np.array([[1, 1], [0, 0]]))
    result = g1 & g2
    expected = grid(np.array([[1, 0], [0, 0]]))
    assert np.array_equal(result.data, expected.data)


def test_or():
    g1 = grid(np.array([[1, 0], [1, 0]]))
    g2 = grid(np.array([[1, 1], [0, 0]]))
    result = g1 | g2
    expected = grid(np.array([[1, 1], [1, 0]]))
    assert np.array_equal(result.data, expected.data)


def test_xor():
    g1 = grid(np.array([[1, 0], [1, 0]]))
    g2 = grid(np.array([[1, 1], [0, 0]]))
    result = g1 ^ g2
    expected = grid(np.array([[0, 1], [1, 0]]))
    assert np.array_equal(result.data, expected.data)


def test_rshift():
    g1 = grid(np.array([[4, 8], [16, 32]]))
    result = g1 >> 1
    expected = grid(np.array([[2, 4], [8, 16]]))
    assert np.array_equal(result.data, expected.data)


def test_lshift():
    g1 = grid(np.array([[4, 8], [16, 32]]))
    result = g1 << 1
    expected = grid(np.array([[8, 16], [32, 64]]))
    assert np.array_equal(result.data, expected.data)


def test_neg():
    g = grid(np.array([[1, -2], [-3, 4]]))
    result = -g
    expected = grid(np.array([[-1, 2], [3, -4]]))
    assert np.array_equal(result.data, expected.data)


def test_pos():
    g = grid(np.array([[1, -2], [-3, 4]]))
    result = +g
    expected = grid(np.array([[1, -2], [-3, 4]]))
    assert np.array_equal(result.data, expected.data)


def test_invert():
    g = grid(np.array([[1, 0], [1, 0]]))
    result = ~g
    expected = grid(np.array([[0, 1], [0, 1]]))
    assert np.array_equal(result.data, expected.data)
