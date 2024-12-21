import rasterio

from glidergun import grid


def test_file():
    g = grid("./.data/n55_e008_1arc_v3.bil")
    assert g


def test_dataset():
    with rasterio.open(
        ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B1.TIF"
    ) as dataset:
        g = grid(dataset)
        assert g


def test_ndarray():
    g = grid("./.data/n55_e008_1arc_v3.bil").project(3857)
    g2 = grid(g.data, g.extent, g.crs)
    assert g2.extent == g.extent
    assert g2.crs == g.crs
    assert g2.data.shape == g.data.shape


def test_box():
    g = grid((40, 30))
    assert g.extent == (0, 0, 1, 1)
    assert g.crs == 4326
    assert g.width == 40
    assert g.height == 30
