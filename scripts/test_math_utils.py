import unittest
import quaternion
import numpy as np
import math_utils

class TestMathUtils(unittest.TestCase):
    def test_axes(self):
        deg = np.rad2deg(math_utils.angle_between(quaternion.x, quaternion.z))
        self.assertEqual(deg, 90.0)

        deg = np.rad2deg(math_utils.angle_between(quaternion.z, quaternion.y))
        self.assertEqual(deg, 90.0)

        deg = np.rad2deg(math_utils.angle_between(quaternion.x, quaternion.y))
        self.assertEqual(deg, 90.0)

        deg = np.rad2deg(math_utils.angle_between(quaternion.x, quaternion.x))
        self.assertEqual(deg, 0.0)

    def test_combination_of_axes(self):
        q1 = (quaternion.x + quaternion.y).normalized()
        deg = np.rad2deg(math_utils.angle_between(q1, quaternion.x))
        np.testing.assert_almost_equal(deg, 45.0)

    def test_shortest_arc(self):
        v1 = np.array([1., 0., 0.])
        v2 = np.array([0., 1., 0.])
        arc = math_utils.shortest_arc(v1, v2)
        rotated = arc * quaternion.x * arc.inverse()
        np.testing.assert_almost_equal((quaternion.y - rotated).norm(), 0.0)
        np.testing.assert_almost_equal(rotated.norm(), 1.0)

    def test_shortest_arc_same(self):
        v1 = np.array([1, 0, 0]) * 2.0
        v2 = np.array([1., 0., 0.])
        arc = math_utils.shortest_arc(v1, v2)
        rotated = arc * quaternion.x * arc.inverse()
        np.testing.assert_almost_equal(rotated, quaternion.x)


if __name__ == "__main__":
    unittest.main()
