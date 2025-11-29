import pytest
from rasterio.crs import CRS

from glidergun import stack


def test_bing():
    url = "https://t.ssl.ak.tiles.virtualearth.net/tiles/a{q}.jpeg?g=15437"
    e = (-123.165, 49.271, -123.157, 49.275)
    s1 = stack(url, extent=e, max_tiles=2)
    s2 = stack(url, extent=e, max_tiles=10)
    assert pytest.approx(s1.extent) == e
    assert pytest.approx(s1.extent) == s2.extent
    assert s1.width <= s2.width
    assert s1.height <= s2.height
    assert s1.dtype == "uint8"
    assert s1.crs == CRS.from_epsg(4326)
