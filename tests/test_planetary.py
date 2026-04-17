import shutil

from glidergun import Grid, Stack, search, stack
from tests.utils import extents_equal


def test_sentinel_visual():
    items = search("sentinel-2-l2a", (-75.72, 45.40, -75.71, 45.41))

    i = items[0]
    assert i.id
    assert i.datetime
    s = i.get("visual")
    assert len(s.grids) == 3
    assert s.dtype == "uint8"

    s.save("tests/output/temp/sentinel_test.tif")
    s2 = stack("tests/output/temp/sentinel_test.tif")
    assert s2.crs == s.crs
    assert extents_equal(s2.extent, s.extent)
    assert len(s2.grids) == 3
    assert s2.dtype == "uint8"

    shutil.rmtree("tests/output/temp")


def test_sentinel_rgb():
    items = search("sentinel-2-l2a", (-75.72, 45.40, -75.71, 45.41))

    i = items[0]
    assert i.id
    assert i.datetime
    s = i.get(["B04", "B03", "B02"], resample_by=10)
    assert len(s.grids) == 3
    assert s.dtype == "uint16"

    s.save("tests/output/temp/sentinel_rgb.tif")
    s2 = stack("tests/output/temp/sentinel_rgb.tif")
    assert s2.crs == s.crs
    assert extents_equal(s2.extent, s.extent)
    assert len(s2.grids) == 3
    assert s2.dtype == "float32"

    shutil.rmtree("tests/output/temp")


def test_landsat_rgb():
    items = search("landsat-c2-l2", (-75.72, 45.40, -75.71, 45.41))

    i = items[0]
    assert i.id
    assert i.datetime
    s = i.get(["red", "green", "blue"])
    assert len(s.grids) == 3
    assert s.dtype == "float32"

    s.save("tests/output/temp/landsat_rgb.img")
    s2 = stack("tests/output/temp/landsat_rgb.img")
    assert s2.crs == s.crs
    assert extents_equal(s2.extent, s.extent)
    assert len(s2.grids) == 3
    assert s2.dtype == "float32"

    shutil.rmtree("tests/output/temp")


def test_landsat_red():
    items = search("landsat-c2-l2", (-75.72, 45.40, -75.71, 45.41))

    i = items[0]
    assert i.id
    assert i.datetime
    g = i.get("red")
    assert g.dtype == "float32"

    g.save("tests/output/temp/landsat_red.bil")
    g2 = stack("tests/output/temp/landsat_red.bil")
    assert g2.crs == g.crs
    assert extents_equal(g2.extent, g.extent)
    assert g2.dtype == "float32"

    shutil.rmtree("tests/output/temp")


def test_other():
    collection: str = "landsat-c2-l2".upper().lower()
    items = search(collection, (-75.72, 45.40, -75.71, 45.41))

    i = items[0]
    assert i.id
    assert i.datetime
    s = i.get(["red", "green", "blue"])
    assert isinstance(s, Stack)

    g = i.get("red")
    assert isinstance(g, Grid)


def test_preview():
    e = (-75.62, 45.45, -75.58, 45.47)
    item = search("landsat-c2-l2", e, datetime="1986-01-01/1987-01-01", cloud_cover_percent=5)[0]

    t = item.datetime
    assert t
    assert t.isoformat() <= "1987-01-01"
    assert t.isoformat() >= "1986-01-01"

    s1 = item.get(["red", "green", "blue"], resample_by=10)
    s2 = item.get(["red", "green", "blue"])

    assert s1.crs == s2.crs
    assert s1.width < s2.width / 2
    assert s1.height < s2.height / 2
