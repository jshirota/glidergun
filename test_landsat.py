import pytest
from glidergun import *

landsat = stack(
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B1.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B2.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B3.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B4.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B5.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B6.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B7.TIF",
)


def test_properties():
    assert landsat.width == 2010
    assert landsat.height == 2033
    assert landsat.dtype == "float32"


def test_project():
    s = landsat.project(4326)
    assert s.crs.wkt.startswith('GEOGCS["WGS 84",DATUM["WGS_1984",')


def test_resample():
    s = landsat.resample(1000)
    assert pytest.approx(s.cell_size.x, 0.001) == 1000
    assert pytest.approx(s.cell_size.y, 0.001) == 1000
    for g in s.grids:
        assert pytest.approx(g.cell_size.x, 0.001) == 1000
        assert pytest.approx(g.cell_size.y, 0.001) == 1000


def test_resample_2():
    s = landsat.resample((1000, 600))
    assert pytest.approx(s.cell_size.x, 0.001) == 1000
    assert pytest.approx(s.cell_size.y, 0.001) == 600
    for g in s.grids:
        assert pytest.approx(g.cell_size.x, 0.001) == 1000
        assert pytest.approx(g.cell_size.y, 0.001) == 600
