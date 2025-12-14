import shutil

from glidergun import search, stack


def test_sentinel_visual():
    features = search("sentinel-2-l2a", (-75.72, 45.40, -75.67, 45.42))

    f = features[0]
    s = f.download("visual")
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
    features = search("sentinel-2-l2a", (-75.72, 45.40, -75.67, 45.42))

    f = features[0]
    s = f.download(["B04", "B03", "B02"])
    assert len(s.grids) == 3
    assert s.dtype == "float32"

    s.save("tests/output/temp/sentinel_rgb.tif")
    s2 = stack("tests/output/temp/sentinel_rgb.tif")
    assert s2.crs == s.crs
    assert s2.extent == s.extent
    assert len(s2.grids) == 3
    assert s2.dtype == "float32"

    shutil.rmtree("tests/output/temp")
