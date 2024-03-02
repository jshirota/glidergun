import unittest

from glidergun import *


class TestSrtm(unittest.TestCase):

    def setUp(self):
        self.grid1 = grid("./.data/n55_e008_1arc_v3.bil")
        self.grid2 = grid("./.data/n55_e009_1arc_v3.bil")

    def test_properties(self):
        self.assertEqual(self.grid1.width, 1801)
        self.assertEqual(self.grid1.height, 3601)


if __name__ == "__main__":
    unittest.main()
