#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import rospy
import tf
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from mav_msgs.msg import RateThrust
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty

GRAVITY = 9.81000041962

class EKF(object):
    def __init__(self, initial_state, sampling_time,
            m,
            qval=1e-1):
        """States are as follows:
            x[0]: x position
            x[1]: y position
            x[2]: z position
            x[3]: x velocity
            x[4]: y velocity
            x[5]: z velocity
            x[6:10]: orientation quaternion x, y, z, w
            u[0]: angular velocity x
            u[1]: angular velocity y
            u[2]: angular velocity z
        """
        assert len(initial_state.shape) == 2
        self.x = initial_state
        self.dt = sampling_time
        self.gravity = GRAVITY
        n = self.x.shape[0]
        self.P_post = np.eye(n) * 1e-1
        self.Q = np.eye(n) * qval
        self.I = np.eye(n)

    def reset(self, x):
        self.x = x
        self.P_post = np.eye(x.shape[0]) * 1e-1

    def step(self, z, u, R):
        self.x, J = self.g(self.x, u)
        z_hat, H = self.h(self.x)

        P_pre = J * self.P_post * J.T + self.Q

        K = np.linalg.solve((H.dot(P_pre).dot(H.T)).T + R, H.dot(P_pre)).T
        assert (P_pre - P_pre.T).sum() < 1e-12

        self.x += K.dot((z - z_hat))
        self.P_post = (self.I - K.dot(H)).dot(P_pre)

        # Normalize quaternion.
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10], 2)

        return self.x, self.P_post

    def g(self, x, u):
        angular_velocity = u[0:3]
        thrust = u[2]

        x = x
        dx = np.zeros_like(x)

        # Velocity affecting position.
        dx[0:3] = self.dt * x[3:6]

        # Update velocity from thrust.
        dx[3] = self.dt * 2.0 * (x[6] * x[8] + x[7] * x[9]) * thrust
        dx[4] = self.dt * 2.0 * (x[7] * x[8] - x[6] * x[9]) * thrust
        dx[5] = self.dt * ((-x[6]**2 - x[7]**2 + x[8]**2 + x[9]**2) * thrust - self.gravity)
        dx[6] = self.dt * (angular_velocity[0] * x[9] + angular_velocity[2] * x[7] - angular_velocity[1] * x[8])
        dx[7] = self.dt * (angular_velocity[1] * x[9] - angular_velocity[2] * x[6] + angular_velocity[0] * x[8])
        dx[8] = self.dt * (angular_velocity[2] * x[9] + angular_velocity[1] * x[6] - angular_velocity[0] * x[7])
        dx[9] = self.dt * (-angular_velocity[0] * x[6] - angular_velocity[1] * x[7] - angular_velocity[2] * x[8])

        J = self.I.copy()
        J[0, 3] = self.dt
        J[1, 4] = self.dt
        J[2, 5] = self.dt
        J[3, 6] = 2.0* self.dt * x[8] * thrust
        J[3, 7] = 2.0* self.dt * x[9] * thrust
        J[3, 8] = 2.0* self.dt * x[6] * thrust
        J[3, 9] = 2.0* self.dt * x[7] * thrust
        J[4, 6] = -2.0 * self.dt * x[9] * thrust
        J[4, 7] = 2.0 * self.dt * x[8] * thrust
        J[4, 8] = 2.0 * self.dt * x[7] * thrust
        J[4, 9] = -2.0 * self.dt * x[6] * thrust
        J[5, 6] = -2.0 * self.dt * x[6] * thrust
        J[5, 7] = -2.0 * self.dt * x[7] * thrust
        J[5, 8] = 2.0 * self.dt * x[8] * thrust
        J[5, 9] = 2.0 * self.dt * x[9] * thrust
        J[6, 7] = self.dt * angular_velocity[2]
        J[6, 8] = -self.dt * angular_velocity[1]
        J[6, 9] = self.dt * angular_velocity[0]
        J[7, 6] = -self.dt * angular_velocity[2]
        J[7, 8] = self.dt * angular_velocity[0]
        J[7, 9] = self.dt * angular_velocity[1]
        J[8, 6] = self.dt * angular_velocity[1]
        J[8, 7] = -self.dt * angular_velocity[0]
        J[8, 9] = self.dt * angular_velocity[2]
        J[9, 6] = -self.dt * angular_velocity[0]
        J[9, 7] = -self.dt * angular_velocity[1]
        J[9, 8] = -self.dt * angular_velocity[2]

        return x + dx, J

    def h(self, x):
        z = x[3:6]
        return z, np.eye(3, 10)

class StateEstimatorNode(object):
    def __init__(self):
        self.thrust = self._initial_thrust()
        rospy.init_node('state_estimator')
        self.imu_sub = rospy.topics.Subscriber(
                name="/sensors/imu",
                data_class=Imu,
                callback=self._imu_callback)
        self.thrust_sub = rospy.topics.Subscriber(
                name='/uav/input/rateThrust',
                data_class=RateThrust,
                callback=self._thrust_callback)
        self.broadcaster = tf.TransformBroadcaster()

        self.reset_sub = rospy.topics.Subscriber(
                name='ekf/reset',
                data_class=Empty,
                callback=self._reset)

        x = self._initial_pose()
        self._publish_frame(x)

        self.ekf = EKF(initial_state=x,
                sampling_time=1.0/120.0, m=3)

        self.tf = tf.TransformListener(True, rospy.Duration(0.1))

    def _reset(self, *args):
        xyz, q = self.tf.lookupTransform('world', 'uav/imu', rospy.Time(0))
        x = np.array([xyz[0], xyz[1], xyz[2],
            0, 0, 0,
            q[0], q[1], q[2], q[3]])[:, None]
        self.ekf.reset(x)

    def _initial_pose(self):
        initial_pose = np.array(rospy.get_param('/uav/flightgoggles_uav_dynamics/init_pose'))[:, None]
        x = np.zeros((10, 1))
        x[0:3] = initial_pose[0:3]
        x[6:] = initial_pose[3:]
        return x

    def _initial_thrust(self):
        return np.array([0.0, 0.0, 0.0,
            0.0, 0.0, GRAVITY
            ])[:, None]

    def _thrust_callback(self, thrust_msg):
        self.thrust = np.array([
            thrust_msg.angular_rates.x,
            thrust_msg.angular_rates.y,
            thrust_msg.angular_rates.z,
            thrust_msg.thrust.x,
            thrust_msg.thrust.y,
            thrust_msg.thrust.z
        ])

    def _imu_callback(self, imu_msg):
        z = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
            ])[:, None]
        u = np.array(self.thrust[0:3])[:, None]
        R = np.array(imu_msg.angular_velocity_covariance).reshape(3, 3)
        mean, _ = self.ekf.step(z, u, R)
        self._publish_frame(mean)

    def _publish_frame(self, x):
        self.broadcaster.sendTransform((x[0], x[1], x[2]),
                (x[6], x[7], x[8], x[9]),
                rospy.Time.now(),
                'ekf/pose',
                'world')

if __name__ == "__main__":
    try:
        node = StateEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

