import unittest
import numpy as np
import quaternion
import math_utils
from state_estimation import EKF, solve_for_orientation, GRAVITY

class TestEKF(unittest.TestCase):
    R = np.eye(4) * 0.1
    def setUp(self):
        self.default_x = np.zeros((10, 1))
        self.default_x[6] = 1.0 # Facing towards x axis

    def test_nop(self):
        ekf = EKF(initial_state=self.default_x, sampling_time=0.1,
                m=4)
        orientation = np.array([1.0, 0.0, 0.0, 0.0])[:, None]
        hover = np.zeros(6)[:, None]
        hover[5, 0] = GRAVITY
        mean, _ = ekf.step(orientation, hover, self.R)
        np.testing.assert_allclose(mean, self.default_x)

    def test_solve_for_orienation_hover(self):
        eps = 1e-15
        f = np.array([0., 0., GRAVITY]) # hover
        a = np.array([0., 0., GRAVITY + eps])
        orientation = solve_for_orientation(f, a)
        np.testing.assert_allclose(orientation.norm(), 1.0)
        np.testing.assert_allclose(orientation, np.quaternion(1, 0, 0, 0))

    def test_solve_for_orienation_at_angle(self):
        eps = 1e-15
        thrust = np.array([0., 0., GRAVITY])
        orientation = np.quaternion(np.cos(np.pi/2) + np.sin(np.pi/2),
                0, 0, 1).normalized() # at 90 degree angle to z-axis.
        imu_acc = np.array([0., GRAVITY, GRAVITY])
        solved = solve_for_orientation(thrust, imu_acc)
        angle = np.rad2deg(math_utils.angle_between(solved, quaternion.z))
        np.testing.assert_almost_equal(angle, 90.0)





if __name__ == "__main__":
    unittest.main()
