import shutil

from glidergun import Grid, Stack, search, stack


def test_sentinel_visual():
    items = search("sentinel-2-l2a", (-75.72, 45.40, -75.71, 45.41))

    i = items[0]
    assert i.id
    assert i.datetime
    s = i.download("visual")
    assert len(s.grids) == 3
    assert s.dtype == "uint8"

    s.save("tests/output/temp/sentinel_test.tif")
    s2 = stack("tests/output/temp/sentinel_test.tif")
    assert s2.crs == s.crs
    assert s2.extent == s.extent
    assert len(s2.grids) == 3
    assert s2.dtype == "uint8"

    shutil.rmtree("tests/output/temp")


def test_sentinel_rgb():
    items = search("sentinel-2-l2a", (-75.72, 45.40, -75.71, 45.41))

    i = items[0]
    assert i.id
    assert i.datetime
    s = i.download(["B04", "B03", "B02"])
    assert len(s.grids) == 3
    assert s.dtype == "float32"

    s.save("tests/output/temp/sentinel_rgb.tif")
    s2 = stack("tests/output/temp/sentinel_rgb.tif")
    assert s2.crs == s.crs
    assert s2.extent == s.extent
    assert len(s2.grids) == 3
    assert s2.dtype == "float32"

    shutil.rmtree("tests/output/temp")


def test_landsat_rgb():
    items = search("landsat-c2-l2", (-75.72, 45.40, -75.71, 45.41))

    i = items[0]
    assert i.id
    assert i.datetime
    s = i.download(["red", "green", "blue"])
    assert len(s.grids) == 3
    assert s.dtype == "float32"

    s.save("tests/output/temp/landsat_rgb.img")
    s2 = stack("tests/output/temp/landsat_rgb.img")
    assert s2.crs == s.crs
    assert s2.extent == s.extent
    assert len(s2.grids) == 3
    assert s2.dtype == "float32"

    shutil.rmtree("tests/output/temp")


def test_landsat_red():
    items = search("landsat-c2-l2", (-75.72, 45.40, -75.71, 45.41))

    i = items[0]
    assert i.id
    assert i.datetime
    g = i.download("red")
    assert g.dtype == "float32"

    g.save("tests/output/temp/landsat_red.bil")
    g2 = stack("tests/output/temp/landsat_red.bil")
    assert g2.crs == g.crs
    assert g2.extent == g.extent
    assert g2.dtype == "float32"

    shutil.rmtree("tests/output/temp")


def test_other():
    collection: str = "landsat-c2-l2".upper().lower()
    items = search(collection, (-75.72, 45.40, -75.71, 45.41))

    i = items[0]
    assert i.id
    assert i.datetime
    s = i.download(["red", "green", "blue"])
    assert isinstance(s, Stack)

    g = i.download("red")
    assert isinstance(g, Grid)
