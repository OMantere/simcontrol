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
from simcontrol.msg import State

GRAVITY = 9.81000041962
EPSILON = np.finfo(np.float64).eps
RATE = 120 # iterations a sec
gravity_vector = np.array([0., 0., -GRAVITY])

def solve_for_orientation(f, a):
    # Gravity is subtracted
    a_without_thrust = (a - f)
    return math_utils.shortest_arc(a_without_thrust, gravity_vector)

class Dynamics(object):
    def __init__(self, initial_state, sampling_time):
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

    def reset(self, x):
        self.x = x

    def step(self, u):
        self.x = self.f(self.x, u)
        return self.x

    def f(self, x, u):
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

        x += dx
        # Normalize quaternion.
        x[6:10] = x[6:10] / np.linalg.norm(x[6:10], 2)

        return x

class StateEstimatorNode(object):
    def __init__(self):
        rospy.init_node('state_estimator')

        self.parent_frame = 'init_pose'
        self.frame = 'state_estimate'

        self.thrust = self._initial_thrust()
        self.thrust_sub = rospy.topics.Subscriber(
                name='/uav/input/rateThrust',
                data_class=RateThrust,
                callback=self._thrust_callback)
        self.state_pub = rospy.topics.Publisher("simcontrol/state_estimate",
                State,
                queue_size=1)

        self.broadcaster = tf.TransformBroadcaster()
        self.reset_sub = rospy.topics.Subscriber(
                name='simcontrol/state_estimate/reset',
                data_class=State,
                callback=self._reset)

        x = self._initial_pose()

        self._publish_frame(x)

        self.dynamics = Dynamics(initial_state=x,
                sampling_time=1.0/120.0)

        self.tf = tf.TransformListener(True, rospy.Duration(0.1))

    def step(self):
        u = np.array(self.thrust[0:6])[:, None]
        x = self.dynamics.step(u)
        self._publish_frame(x)

    def _reset(self, state):
        """
        Will set the state to the sent state.
        """
        x = np.array([
            state.pose.position.x,
            state.pose.position.y,
            state.pose.position.z,
            state.linear_velocity.x,
            state.linear_velocity.y,
            state.linear_velocity.z,
            state.pose.orientation.x,
            state.pose.orientation.y,
            state.pose.orientation.z,
            state.pose.orientation.w])[:, None]
        self.dynamics.reset(x)
        self.broadcaster.sendTransform((state.pose.position.x, state.pose.position.y,
            state.pose.position.z),
            (state.pose.orientation.x, state.pose.orientation.y,
                state.pose.orientation.z, state.pose.orientation.w),
                rospy.Time.now(),
                self.frame,
                self.parent_frame)

    def _initial_pose(self):
        # Initial pose in [x y z w]
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

    def _publish_frame(self, x):
        msg = State()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.parent_frame
        msg.pose.position = Vector3(x[0], x[1], x[2])
        msg.pose.orientation = Quaternion(x[6], x[7], x[8], x[9])
        msg.linear_acceleration = Vector3(x[3], x[4], x[5])

        self.state_pub.publish(msg)
        self.broadcaster.sendTransform((x[0], x[1], x[2]),
                (x[6], x[7], x[8], x[9]),
                rospy.Time.now(),
                self.frame,
                self.parent_frame)

if __name__ == "__main__":
    try:
        node = StateEstimatorNode()
        rate = rospy.Rate(RATE)
        while not rospy.is_shutdown():
            node.step()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

