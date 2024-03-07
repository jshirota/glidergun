from glidergun import *


def test_properties():
    dem = grid("./.data/n55_e008_1arc_v3.bil")

    assert dem.width == 1801
    assert dem.height == 3601
    assert dem.dtype == "float32"
