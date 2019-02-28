#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import rospy
import tf
import quaternion
from lib import math_utils
from geometry_msgs.msg import Quaternion, Vector3, Point
from mav_msgs.msg import RateThrust
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from simcontrol.msg import State
from flightgoggles.msg import IRMarkerArray
from collections import defaultdict
from lib.vision import solve_pnp

GRAVITY = 9.81000041962
EPSILON = np.finfo(np.float64).eps
RATE = 50 # iterations a sec
gravity_vector = np.array([0., 0., -GRAVITY])

def solve_for_orientation(f, a):
    # Gravity is subtracted
    a_without_thrust = (a - f)
    return math_utils.shortest_arc(a_without_thrust, gravity_vector)

class ImuIntegrator(object):
    def __init__(self, initial_state):
        self.imu_topic = '/uav/sensors/imu'
        self.imu_subscriber = rospy.Subscriber(self.imu_topic, Imu, self._imu_callback)
        initial_state = initial_state.flatten()
        self.imu_latest_x = np.float32([initial_state[0], initial_state[1], initial_state[2]])
        self.imu_latest_xdot = np.float32([0.0, 0.0, 0.0])
        self.imu_latest_xddot = np.float32([0.0, 0.0, 0.0])
        self.imu_latest_omega = np.float32([0.0, 0.0, 0.0])
        self.imu_latest_q = np.quaternion(initial_state[9], initial_state[6], initial_state[7], initial_state[8])
        self.imu_acc_xddot = np.float32([0, 0, 0])
        self.imu_acc_omega = np.float32([0, 0, 0])
        self.imu_count = 0
        self.prev_imu_time = rospy.Time.now()
        self.tf_estimator = TFStateEstimator()
        self.thrust = np.float32([0,0,0,0,0,0])
        self.thrust_sub = rospy.topics.Subscriber(name='/uav/input/rateThrust',data_class=RateThrust,callback=self._thrust_callback)
        self.state_pub = rospy.topics.Publisher("simcontrol/state_estimate",
                State,
                queue_size=1)

    def body_to_world(self, q_b, v):
        qr = np.quaternion(0.0, v[0], v[1], v[2])
        qr = q_b * qr * q_b.conjugate()
        return np.float32([qr.x, qr.y, qr.z])

    def latest_world_frame_acceleration(self, q_b=None):
        if not q_b:
            q_b = self.imu_latest_q
        return self.body_to_world(q_b, self.imu_latest_xddot)

    def imu_state_message(self):
        msg = State()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.pose.position = Point(self.imu_latest_x[0], self.imu_latest_x[1], self.imu_latest_x[2])
        msg.pose.orientation = Quaternion(self.imu_latest_q.x, self.imu_latest_q.y, self.imu_latest_q.z, self.imu_latest_q.w)
        msg.linear_velocity = Vector3(self.imu_latest_xdot[0], self.imu_latest_xdot[1], self.imu_latest_xdot[2])
        msg.angular_velocity = Vector3(self.imu_latest_omega[0], self.imu_latest_omega[1], self.imu_latest_omega[2])
        msg.linear_acceleration = Vector3(self.imu_latest_xddot[0], self.imu_latest_xddot[1], self.imu_latest_xddot[2])
        self.state_pub.publish(msg)

    def _thrust_callback(self, thrust_msg):
        self.thrust = np.array([
            thrust_msg.angular_rates.x,
            thrust_msg.angular_rates.y,
            thrust_msg.angular_rates.z,
            thrust_msg.thrust.x,
            thrust_msg.thrust.y,
            thrust_msg.thrust.z
        ])
        #self._imu_update()

    def _imu_callback(self, msg):
        self.imu_acc_omega += np.float32([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.imu_acc_xddot += np.float32([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z - GRAVITY])
        self.imu_count += 1
        if self.imu_count % 30 == 0:
            self.imu_acc_omega /= 300
            self.imu_acc_xddot /= 300
            print('imu acc', self.imu_acc_xddot)
            self._imu_update()
            self.imu_acc_omega = np.float32([0, 0, 0])
            self.imu_acc_xddot = np.float32([0, 0, 0])

    def _imu_update(self):
        now = rospy.Time.now()
        time_elapsed = now - self.prev_imu_time
        self.prev_imu_time = now
        omega = self.imu_acc_omega
        secs_elapsed = time_elapsed.to_sec()
        omegabar = (omega + self.imu_latest_omega) / 2 * secs_elapsed
        #self.imu_latest_q = math_utils.q_from_rpy(omegabar[0], omegabar[1], omegabar[2]) * self.imu_latest_q
        body_xddot = self.imu_acc_xddot
        xddot = self.body_to_world(self.imu_latest_q, body_xddot)
        xdot = self.imu_latest_xdot + ((xddot + self.imu_latest_xddot) / 2) * secs_elapsed  # Integrate from xddot
        #print('xdot', xdot)
        #print('xddot', xddot)
        x = self.imu_latest_x + (xdot + self.imu_latest_xdot) / 2 * secs_elapsed  # Integrate from xdot
        self.imu_latest_x = x
        self.imu_latest_xdot = xdot
        self.imu_latest_xddot = xddot
        self.imu_latest_omega = omega
        x, xdot, xddot, q = self.tf_estimator.state_estimate()

class IREstimator(object):
    def __init__(self):
        self.ir_sub = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.ir_subscriber)
        self.tf = TransformBroadcaster()
        self.gate_markers = {}
        self.get_nominal_locations()
        self.last_position = np.float32([0, 0, 0])
        self.last_orientation = np.quaternion(1.0, 0.0, 0.0, 0.0)
        rospy.spin()

    def update_state(self, p, q):
        self.last_position = p 
        self.last_orientation = q
     
    def get_nominal_locations(self):
        for gate in range(1, 24):
            nominal_location = np.float32(rospy.get_param('/uav/Gate%d/nominal_location' % gate))
            self.gate_markers['Gate%d' % gate] = {}
            for i in range(1, 5):
                self.gate_markers['Gate%d' % gate][i] = nominal_location[i-1, :]

    def send_pose(self, x, q, stamp=rospy.Time.now(), origin_frame='world'):
        self.tf.sendTransform((x[0], x[1], x[2]), (q.x, q.y, q.z, q.w), stamp, 'quadcopter', origin_frame)

    def get_pose(self, origin_frame):
        self.tf.sendTransform()

    def ir_subscriber(self, msg):
        image_points = defaultdict(list)
        object_points = defaultdict(list)
        for marker in msg.markers:
            markers[msg.landmarkID.data].append(np.float32([msg.x, msg.y]))
            object_points[msg.landmarkID.data].append(self.gate_markers[msg.landmarkID.data][msg.markerID.data])

        for gate in image_points:
            if len(image_points[gate]) == 4:  # If all 4 points are in view, solve a relative pose to the gate
                success, position, orientation = solvepnp(object_points[gate], image_points[gate],
                 position_guess=self.last_position, orientation_guess=self.last_position)

class TFStateEstimator(object):
    def __init__(self):
        self.tf_listener = tf.TransformListener()
        self.tf_prev_x = np.float32([0, 0, 0])
        self.tf_prev_q = math_utils.identity_quaternion()
        self.tf_prev_xdot = np.array([0.0, 0.0, 0.0])
        self.tf_prev_xddot = np.array([0.0, 0.0, 0.0])
        self.prev_tf_time = rospy.Time.now()

    def state_estimate(self, timestamp=None):
        use_timestamp = rospy.Time.now()
        if timestamp is not None:
            use_timestamp = timestamp
        try:
            translation, rotation = self.tf_listener.lookupTransform('world','uav/imu', use_timestamp)
            q = np.quaternion(rotation[3], rotation[0], rotation[1], rotation[2])
            #q = q.conjugate()
            x = np.array(translation)
            # Discrete derivatives
            xdot = self.tf_prev_xdot
            xddot = self.tf_prev_xddot
            if timestamp is None:
                time_elapsed = rospy.Time.now() - self.prev_tf_time
                nsec_elapsed = time_elapsed.to_nsec()
                if nsec_elapsed > 0:
                    true_rate = 1.0/nsec_elapsed * 1e9
                    xdot = true_rate * (x - self.tf_prev_x)
                    xddot = true_rate * (xdot - self.tf_prev_xdot)
                    self.prev_tf_time = self.prev_tf_time + time_elapsed
                    self.tf_prev_x = x
                    self.tf_prev_q = q
                    self.tf_prev_xdot = xdot
                    self.tf_prev_xddot = xddot
            return x, xdot, xddot, q
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return self.tf_prev_x, self.tf_prev_xdot, self.tf_prev_xddot, self.tf_prev_q

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
        self.prev_time = rospy.Time.now()

    def reset(self, x):
        self.x = x

    def step(self, u):
        self.x = self.f(self.x, u)
        return self.x

    def f(self, x, u):
        now = rospy.Time.now()
        elapsed_time = now - self.prev_time
        self.prev_time = now
        dt = elapsed_time.to_sec()
        dt = self.dt
        angular_velocity = u[0:3]
        thrust = u[5]
        dx = np.zeros_like(x)

        # Velocity affecting position.
        dx[0:3] = dt * x[3:6]

        # Update velocity from thrust.
        dx[3] = dt * 2.0 * (x[6] * x[8] + x[7] * x[9]) * thrust
        dx[4] = dt * 2.0 * (x[7] * x[8] - x[6] * x[9]) * thrust
        dx[5] = dt * ((-x[6]**2 - x[7]**2 + x[8]**2 + x[9]**2) * thrust - self.gravity)
        dx[6] = dt * 0.5*(angular_velocity[0] * x[9] + angular_velocity[2] * x[7] - angular_velocity[1] * x[8])
        dx[7] = dt * 0.5*(angular_velocity[1] * x[9] - angular_velocity[2] * x[6] + angular_velocity[0] * x[8])
        dx[8] = dt * 0.5*(angular_velocity[2] * x[9] + angular_velocity[1] * x[6] - angular_velocity[0] * x[7])
        dx[9] = dt * 0.5*(-angular_velocity[0] * x[6] - angular_velocity[1] * x[7] - angular_velocity[2] * x[8])

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

        self.use_imu = True
        x = self._initial_pose()
        self.dynamics = Dynamics(initial_state=x,
                sampling_time=1.0/120.0)
        self.imu_integrator = ImuIntegrator(initial_state=x)
        self.tf = tf.TransformListener(True, rospy.Duration(0.1))
        self._publish_frame(x)

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
        if self.use_imu:
            self.imu_integrator.imu_state_message()
            return

        msg = State()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.parent_frame
        msg.pose.position = Vector3(x[0], x[1], x[2])
        msg.pose.orientation = Quaternion(x[6], x[7], x[8], x[9])
        msg.linear_velocity = Vector3(x[3], x[4], x[5])

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
            rate.sleep()   # With IMU runs at about 2 khz of no sleep
    except rospy.ROSInterruptException:
        pass

