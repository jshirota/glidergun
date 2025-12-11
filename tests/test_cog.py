import pytest
from rasterio.crs import CRS

from glidergun import grid


def test_read():
    url = "https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com/store/elevation/cdem-cdsm/cdem/cdem-canada-dem.tif"
    dem = grid(url, (-1994991, 460761, -1954991, 484761))
    assert dem.crs == CRS.from_epsg(3979)
    assert pytest.approx(dem.extent[0], 100) == -1994991
    assert pytest.approx(dem.extent[1], 100) == 460761
    assert pytest.approx(dem.extent[2], 100) == -1954991
    assert pytest.approx(dem.extent[3], 100) == 484761


def test_read_4326():
    url = "https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com/store/elevation/cdem-cdsm/cdem/cdem-canada-dem.tif"
    dem = grid(url, (-80.00, 43.00, -79.99, 43.01), 4326)
    assert dem.crs == CRS.from_epsg(4326)
    assert pytest.approx(dem.extent[0], 0.01) == -80.00
    assert pytest.approx(dem.extent[1], 0.01) == 43.00
    assert pytest.approx(dem.extent[2], 0.01) == -79.99
    assert pytest.approx(dem.extent[3], 0.01) == 43.01
