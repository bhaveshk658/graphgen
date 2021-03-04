import unittest
from graphgen.data import utils

class TestCCW(unittest.TestCase):

    def test_true(self):
        A = [0, 0]
        B = [0, 4]
        C = [-2, 2]
        self.assertTrue(utils.ccw(A, B, C))

    def test_false(self):
        A = [0, 0]
        B = [0, 4]
        C = [2, 2]
        self.assertFalse(utils.ccw(A, B, C))

    def test_extended_true(self):
        A = [0, 0]
        B = [0, 4]
        C = [-100, 50]
        self.assertTrue(utils.ccw(A, B, C))

    def test_extended_false(self):
        A = [0, 0]
        B = [0, 4]
        C = [100, 50]
        self.assertFalse(utils.ccw(A, B, C))

class TestIsIntersect(unittest.TestCase):

    def test_true(self):
        A = [0, 0]
        B = [0, 4]
        C = [2, 2]
        D = [-2, 2]
        self.assertTrue(utils.is_intersect(A, B, C, D))

    def test_false(self):
        A = [0, 0]
        B = [0, 4]
        C = [2, 2]
        D = [1, 1]

class TestMinDistance(unittest.TestCase):

    def test_touching(self):
        A = [0, 0]
        B = [0, 4]
        C = [0, 2]
        self.assertEqual(utils.minDistance(A, B, C), 0)

    def test_perpendicular(self):
        A = [0, 0]
        B = [0, 4]
        C = [2, 2]
        self.assertEqual(utils.minDistance(A, B, C), 2)

    def test_parallel(self):
        A = [0, 0]
        B = [0, 4]
        C = [0, 6]
        self.assertEqual(utils.minDistance(A, B, C), 2)


if __name__ == "__main__":
    unittest.main()
