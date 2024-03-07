from glidergun import *


def test_properties():
    landsat = stack(
        ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B1.TIF",
        ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B2.TIF",
        ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B3.TIF",
        ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B4.TIF",
        ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B5.TIF",
        ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B6.TIF",
        ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B7.TIF",
    )

    assert landsat.width == 2010
    assert landsat.height == 2033
    assert landsat.dtype == "float32"
