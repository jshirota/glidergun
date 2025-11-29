import numpy as np
import pytest

from glidergun import grid


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
