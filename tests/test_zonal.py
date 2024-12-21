import math

import numpy as np
import pytest

from glidergun._grid import Grid, grid


@pytest.fixture
def grid_data():
    data = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    return grid(data)


@pytest.fixture
def zone_grid_data():
    data = np.array(
        [
            [1, 1, 2],
            [1, 2, 2],
            [3, 3, 3],
        ]
    )
    return grid(data)


def test_zonal_count(grid_data, zone_grid_data):
    result = grid_data.zonal_count(5, zone_grid_data)
    assert result.data.tolist() == [
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
    ]


def test_zonal_ptp(grid_data, zone_grid_data):
    result = grid_data.zonal_ptp(zone_grid_data)
    assert result.data.tolist() == [
        [3, 3, 3],
        [3, 3, 3],
        [2, 2, 2],
    ]


def test_zonal_percentile(grid_data, zone_grid_data):
    result = grid_data.zonal_percentile(50, zone_grid_data)
    assert isinstance(result, Grid)


def test_zonal_quantile(grid_data, zone_grid_data):
    result = grid_data.zonal_quantile(0.5, zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 2


def test_zonal_median(grid_data, zone_grid_data):
    result = grid_data.zonal_median(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 2


def test_zonal_mean(grid_data, zone_grid_data):
    result = grid_data.zonal_mean(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 2.3333332538604736


def test_zonal_std(grid_data, zone_grid_data):
    result = grid_data.zonal_std(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 1.247219204902649


def test_zonal_var(grid_data, zone_grid_data):
    result = grid_data.zonal_var(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 1.5555557012557983


def test_zonal_min(grid_data, zone_grid_data):
    result = grid_data.zonal_min(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 1


def test_zonal_max(grid_data, zone_grid_data):
    result = grid_data.zonal_max(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 4


def test_zonal_sum(grid_data, zone_grid_data):
    result = grid_data.zonal_sum(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 7


def test_zonal_entropy(grid_data, zone_grid_data):
    result = grid_data.zonal_entropy(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 0.9556999206542969


def test_zonal_gmean(grid_data, zone_grid_data):
    result = grid_data.zonal_gmean(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 2


def test_zonal_hmean(grid_data, zone_grid_data):
    result = grid_data.zonal_hmean(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 1.7142857313156128


def test_zonal_pmean(grid_data, zone_grid_data):
    result = grid_data.zonal_pmean(2, zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 2.6457512378692627


def test_zonal_kurtosis(grid_data, zone_grid_data):
    result = grid_data.zonal_kurtosis(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == -1.5000001192092896


def test_zonal_iqr(grid_data, zone_grid_data):
    result = grid_data.zonal_iqr(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 1.5


def test_zonal_moment(grid_data, zone_grid_data):
    result = grid_data.zonal_moment(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 0


def test_zonal_skew(grid_data, zone_grid_data):
    result = grid_data.zonal_skew(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 0.3818019926548004


def test_zonal_kstat(grid_data, zone_grid_data):
    result = grid_data.zonal_kstat(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 2.3333332538604736


def test_zonal_kstatvar(grid_data, zone_grid_data):
    result = grid_data.zonal_kstatvar(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert math.isnan(g.min)
    assert math.isnan(g.max)


def test_zonal_tmean(grid_data, zone_grid_data):
    result = grid_data.zonal_tmean(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 2.3333332538604736


def test_zonal_tvar(grid_data, zone_grid_data):
    result = grid_data.zonal_tvar(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 2.3333334922790527


def test_zonal_tmin(grid_data, zone_grid_data):
    result = grid_data.zonal_tmin(zone_grid_data)
    g = result.set_nan(zone_grid_data != 1, result)
    assert g.min == g.max == 1


def test_zonal_tmax(grid_data, zone_grid_data):
    result = grid_data.zonal_tmax(zone_grid_data)
    assert isinstance(result, Grid)


def test_zonal_tstd(grid_data, zone_grid_data):
    result = grid_data.zonal_tstd(zone_grid_data)
    assert isinstance(result, Grid)


def test_zonal_variation(grid_data, zone_grid_data):
    result = grid_data.zonal_variation(zone_grid_data)
    assert isinstance(result, Grid)


def test_zonal_median_abs_deviation(grid_data, zone_grid_data):
    result = grid_data.zonal_median_abs_deviation(zone_grid_data)
    assert isinstance(result, Grid)
