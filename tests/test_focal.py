import numpy as np
import pytest

from glidergun._grid import Grid, grid


class TestFocal:
    @pytest.fixture
    def focal(self):
        data = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        return grid(data)

    def test_focal_count(self, focal: Grid):
        result = focal.focal_count(value=5, buffer=1, circle=False)
        expected = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_ptp(self, focal: Grid):
        result = focal.focal_ptp(buffer=1, circle=False)
        expected = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 8, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_percentile(self, focal: Grid):
        result = focal.focal_percentile(percentile=50, buffer=1, circle=False)
        expected = np.array(
            [
                [3.0, 3.5, 4.0],
                [4.5, 5.0, 5.5],
                [6.0, 6.5, 7.0],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_quantile(self, focal: Grid):
        result = focal.focal_quantile(probability=0.5, buffer=1, circle=False)
        expected = np.array(
            [
                [3.0, 3.5, 4.0],
                [4.5, 5.0, 5.5],
                [6.0, 6.5, 7.0],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_median(self, focal: Grid):
        result = focal.focal_median(buffer=1, circle=False)
        expected = np.array(
            [
                [3.0, 3.5, 4.0],
                [4.5, 5.0, 5.5],
                [6.0, 6.5, 7.0],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_mean(self, focal: Grid):
        result = focal.focal_mean(buffer=1, circle=False)
        expected = np.array(
            [
                [3.0, 3.5, 4.0],
                [4.5, 5.0, 5.5],
                [6.0, 6.5, 7.0],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_std(self, focal: Grid):
        result = focal.focal_std(buffer=1, circle=False)
        expected = np.array(
            [
                [1.581139, 1.707825, 1.581139],
                [2.5, 2.581989, 2.5],
                [1.581139, 1.707825, 1.581139],
            ]
        )
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_focal_var(self, focal: Grid):
        result = focal.focal_var(buffer=1, circle=False)
        expected = np.array(
            [
                [2.5, 2.916667, 2.5],
                [6.25, 6.666667, 6.25],
                [2.5, 2.916667, 2.5],
            ]
        )
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_focal_min(self, focal: Grid):
        result = focal.focal_min(buffer=1, circle=False)
        expected = np.array(
            [
                [1, 1, 2],
                [1, 1, 2],
                [4, 4, 5],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_max(self, focal: Grid):
        result = focal.focal_max(buffer=1, circle=False)
        expected = np.array(
            [
                [5, 6, 6],
                [8, 9, 9],
                [8, 9, 9],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_sum(self, focal: Grid):
        result = focal.focal_sum(buffer=1, circle=False)
        expected = np.array(
            [
                [12, 21, 16],
                [27, 45, 33],
                [24, 39, 28],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)
