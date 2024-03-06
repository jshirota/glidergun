import unittest

from glidergun import *


class TestLandsat(unittest.TestCase):
    def setUp(self):
        self.stack = stack(
            ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B1.TIF",
            ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B2.TIF",
            ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B3.TIF",
            ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B4.TIF",
            ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B5.TIF",
            ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B6.TIF",
            ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B7.TIF",
        )

    def test_properties(self):
        self.assertEqual(self.stack.width, 2010)
        self.assertEqual(self.stack.height, 2033)


if __name__ == "__main__":
    unittest.main()
