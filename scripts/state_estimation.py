#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import rospy
import tf
import quaternion
import math_utils
from geometry_msgs.msg import Quaternion, Vector3
from mav_msgs.msg import RateThrust
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty

GRAVITY = 9.81000041962
EPSILON = np.finfo(np.float64).eps

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
            u[3]: thrust x
            u[4]: thrust y
            u[5]: thrust z
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

        self.x += K.dot((z - z_hat))
        self.P_post = (self.I - K.dot(H)).dot(P_pre)

        return self.x, self.P_post

    def g(self, x, u):
        angular_velocity = u[0:3]
        thrust = u[5]

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

        x += dx
        # Normalize quaternion.
        x[6:10] = x[6:10] / np.linalg.norm(x[6:10], 2)

        return x, J

    def h(self, x):
        # Only the orientation is being measured as it currently stands.
        z = x[6:]
        return z, np.eye(4, 10)

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
        self.odometry = rospy.topics.Publisher("odometry/ekf",
                Odometry,
                queue_size=1)

        self.broadcaster = tf.TransformBroadcaster()
        self.reset_sub = rospy.topics.Subscriber(
                name='ekf/reset',
                data_class=Empty,
                callback=self._reset)

        x = self._initial_pose()
        self._publish_frame(x, np.eye(10) * 0.1)

        self.ekf = EKF(initial_state=x,
                sampling_time=1.0/120.0, m=3)

        self.tf = tf.TransformListener(True, rospy.Duration(0.1))

    def _reset(self, *args):
        # This is here for debugging purposes only.
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

    def _solve_for_orientation(self, imu_msg):
        # Let f be the thrust vector in the imu frame (towards the z-axis of the drone).
        # a is the measured acceleration in the imu frame.
        # q is the orientation quaternion.
        #
        # q * (f + a) * q.inverse() = g
        # I.e. we want to find the rotation q which rotates f + a to g.
        f = np.array([0.0, 0.0, self.thrust[5]])
        a = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])
        g = np.array([0.0, 0.0, -GRAVITY])

        f_plus_a = f + a
        q = math_utils.shortest_arc(f_plus_a, g)
        return quaternion.as_float_array(q)[:, None]

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
        orientation = self._solve_for_orientation(imu_msg)
        u = np.array(self.thrust[0:6])[:, None]
        # TODO figure out how to model uncertainty.
        R = np.eye(4) * imu_msg.angular_velocity_covariance[0]
        self.done = True
        mean, cov = self.ekf.step(orientation, u, R)
        self._publish_frame(mean, cov)

    def _publish_frame(self, x, cov):
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.child_frame_id = 'uav/imu'
        msg.pose.pose.position = Vector3(x[0], x[1], x[2])
        msg.pose.pose.orientation = Quaternion(x[6], x[7], x[8], x[9])
        msg.pose.covariance[0] = -1.0
        #TODO populate covariance. Has to be calculated
        # in terms of covariance in x, y, z axis rotation.

        msg.twist.twist.linear = Vector3(x[3], x[4], x[5])
        covariance = np.zeros((6, 6))
        covariance[0:3, 0:3] = cov[3:6, 3:6]
        msg.twist.covariance = covariance.ravel().tolist()
        self.odometry.publish(msg)
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

