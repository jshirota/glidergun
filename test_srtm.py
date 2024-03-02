import glob
import unittest

from glidergun import *


class TestLandsat(unittest.TestCase):
    def setUp(self):
        self.stack = stack(*glob.glob(".data/LC08*.TIF"))

    def test_properties(self):
        self.assertEqual(self.stack.width, 2010)
        self.assertEqual(self.stack.height, 2033)


if __name__ == "__main__":
    unittest.main()
