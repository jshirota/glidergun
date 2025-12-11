import shutil

from glidergun import search, stack


def test_sentinel():
    features = search("sentinel-2-l2a", (-75.72, 45.40, -75.67, 45.42))

    f = features[0]
    s = f.get_stack("visual")
    assert len(s.grids) == 3
    assert s.dtype == "uint8"

    s.save("tests/output/temp/sentinel_test.tif")
    s2 = stack("tests/output/temp/sentinel_test.tif")
    assert s2.crs == s.crs
    assert s2.extent == s.extent
    assert len(s2.grids) == 3
    assert s2.dtype == "uint8"

    shutil.rmtree("tests/output/temp")
