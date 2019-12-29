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
from std_msgs.msg import Bool, String
from flightgoggles.msg import IRMarkerArray
from collections import defaultdict
from lib.vision import solve_pnp
import random
import math

GRAVITY = 9.81000041962
EPSILON = np.finfo(np.float64).eps
RATE = 120 # iterations a sec
gravity_vector = np.array([0., 0., -GRAVITY])

def solve_for_orientation(f, a):
    # Gravity is subtracted
    a_without_thrust = (a - f)
    return math_utils.shortest_arc(a_without_thrust, gravity_vector)

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
    def __init__(self, initial_state):
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
        self.gravity = GRAVITY
        self.dt = 0
        self.prev_time = rospy.Time.now()

    def reset(self, x):
        self.x = x

    def step(self, u):
        self.x = self.f(self.x, u)
        return self.x

    def f(self, x, u):
        now = rospy.Time.now()
        nsec_elapsed = (now - self.prev_time).to_nsec()
        self.prev_time = now
        if nsec_elapsed > 0:
            self.dt = nsec_elapsed / 1e9
            angular_velocity = u[0:3]
            thrust = u[5]

            dx = np.zeros_like(x)

            # Velocity affecting position.
            dx[0:3] = self.dt * x[3:6]

            # Update velocity from thrust.
            dx[3] = self.dt * 2.0 * (x[6] * x[8] + x[7] * x[9]) * thrust
            dx[4] = self.dt * 2.0 * (x[7] * x[8] - x[6] * x[9]) * thrust
            dx[5] = self.dt * ((-x[6]**2 - x[7]**2 + x[8]**2 + x[9]**2) * thrust - self.gravity)
            dx[6] = self.dt * 0.5*(angular_velocity[0] * x[9] + angular_velocity[2] * x[7] - angular_velocity[1] * x[8])
            dx[7] = self.dt * 0.5*(angular_velocity[1] * x[9] - angular_velocity[2] * x[6] + angular_velocity[0] * x[8])
            dx[8] = self.dt * 0.5*(angular_velocity[2] * x[9] + angular_velocity[1] * x[6] - angular_velocity[0] * x[7])
            dx[9] = self.dt * 0.5*(-angular_velocity[0] * x[6] - angular_velocity[1] * x[7] - angular_velocity[2] * x[8])
            x += dx
            # Normalize quaternion.
            x[6:10] = x[6:10] / np.linalg.norm(x[6:10], 2)

        return x


class IREstimator(object):
    def __init__(self):
        self.init_position, self.init_orientation = self.get_init_pose()
        self.ir_sub = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.ir_subscriber, queue_size=1)
        self.gate_sub = rospy.Subscriber('/simcontrol/target_gate', String, self._gate_callback)
        self.taget_gate_name = None
        self.tf = tf.TransformBroadcaster()
        self.gate_markers = {}
        self.gate_mean = {}
        self.gate_normal = {}
        self.gate_is_perturbed = {}
        self.prev_closest_landmark = self.init_position
        self.gate_names = rospy.get_param("/gate_names") if rospy.has_param('/gate_names') else None
        self.last_orientation = self.init_orientation
        self.state_pub = rospy.Publisher('/simcontrol/state_estimate', State, queue_size=1)
        self.takeoff_pub = rospy.Publisher('/simcontrol/takeoff', Bool, queue_size=1)
        self.tf_estimator = TFStateEstimator()
        self.prev_time = rospy.Time.now()
        self.prev_time_int = rospy.Time.now()
        self.last_xdot = np.float32([0, 0, 0])
        self.wait_i = 0
        self.last_position_ir = self.init_position
        self.last_position = np.copy(self.init_position)
        self.target_gate_name = None
        x = self._initial_pose()
        self.dynamics = Dynamics(initial_state=x)
        self.imu_enabled = True
        self.latest_thrust = None
        self.thrust_sub = rospy.topics.Subscriber(
                name='/uav/input/rateThrust',
                data_class=RateThrust,
                callback=self._thrust_callback)

        self.imu_topic = '/uav/sensors/imu'
        self.imu_subscriber = rospy.Subscriber(self.imu_topic, Imu, self._imu_callback)
        self.imu_acc_xddot = np.float32([0, 0, 0])
        self.imu_acc_omega = np.float32([0, 0, 0])
        self.imu_count = 0
        self.prev_imu_time = rospy.Time.now()
        self.x_min = np.float32([-20, -45, 0])
        self.x_max = np.float32([21, 55, 19])
        self.reference_waypoints = None
        self.reference_trajectory = None
        self.gate_waypoint_distance = 2.0
        self.get_gate_info()

    def _thrust_callback(self, thrust_msg):
        self.latest_thrust = np.array([
            thrust_msg.angular_rates.x,
            thrust_msg.angular_rates.y,
            thrust_msg.angular_rates.z,
            thrust_msg.thrust.x,
            thrust_msg.thrust.y,
            thrust_msg.thrust.z
        ])

    def _initial_pose(self):
        # Initial pose in [x y z w]
        initial_pose = np.array(rospy.get_param('/uav/flightgoggles_uav_dynamics/init_pose'))[:, None]
        x = np.zeros((10, 1))
        x[0:3] = initial_pose[0:3]
        x[6:] = initial_pose[3:]
        return x

    def _gate_callback(self, msg):
        self.target_gate_name = msg.data

    def body_to_world(self, q_b, v):
        qr = np.quaternion(0.0, v[0], v[1], v[2])
        qr = q_b * qr * q_b.conjugate()
        return np.float32([qr.x, qr.y, qr.z])

    def _imu_callback(self, msg):
        self.imu_acc_omega += np.float32([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.imu_acc_xddot += np.float32([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.imu_count += 1
        if self.imu_count % 100 == 0:
            self.imu_acc_omega /= 100
            self.imu_acc_xddot /= 100
            self._imu_update()
            self.imu_acc_omega = np.float32([0, 0, 0])
            self.imu_acc_xddot = np.float32([0, 0, 0])

    def get_init_pose(self):
        if not rospy.has_param('/uav/flightgoggles_uav_dynamics/init_pose'):
            return np.float32([0, 0, 0]), np.quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            init_pose = rospy.get_param('/uav/flightgoggles_uav_dynamics/init_pose')
            x = np.float32([init_pose[0], init_pose[1], init_pose[2]])
            q = np.quaternion(init_pose[6], init_pose[3], init_pose[4], init_pose[5])
            return x, q

    def linear_interpolate(self, p, n):
        res = np.zeros((n * p.shape[0], p.shape[1]))
        ii = 0
        for i in range(p.shape[0] - 1):
            res[ii, :] = p[i, :]
            trans = p[i + 1, :] - p[i, :]
            dist = np.linalg.norm(trans)
            for j in range(1, n):
                ii += 1
                res[ii, :] = p[i, :] + trans * (1.0*j/n)
            ii += 1
        return res, ii

    def distance_to_trajectory(self, x):
        closest_dist = 1e9
        target = self.gate_names.index(self.target_gate_name)   # Consider only segment between previous and target gate
        for i in range(self.reference_trajectory.shape[0])[100*2*target:(2*target+1)*100]:
            if i == 0:
                continue
            new_d = np.linalg.norm(self.reference_trajectory[i, :] - x)
            if new_d < closest_dist:
                closest_dist = new_d
        return closest_dist

    def get_gate_info(self):
        for gate in range(1, 24):
            nominal_location = np.float32(rospy.get_param('/uav/Gate%d/nominal_location' % gate))
            self.gate_markers['Gate%d' % gate] = {}
            for i in range(1, 5):
                self.gate_markers['Gate%d' % gate][str(i)] = nominal_location[i-1, :]
            self.gate_normal['Gate%d' % gate] = -np.cross(nominal_location[1,:] - nominal_location[0,:], nominal_location[2,:] - nominal_location[0,:])
            self.gate_normal['Gate%d' % gate] /= np.linalg.norm(self.gate_normal['Gate%d' % gate])
            perturbation_bound = np.float32(rospy.get_param('/uav/Gate%d/perturbation_bound' % gate))
            self.gate_is_perturbed['Gate%d' % gate] = np.any(perturbation_bound > 0.1)
            self.gate_mean['Gate%d' % gate] = np.mean(nominal_location, axis=0)

        # Generate reference trajectory
        p = np.zeros((len(self.gate_names)*2+1, 3))
        p[0, :] = self.init_position
        for i, g in enumerate(self.gate_names):
            point_a = self.gate_mean[g] - self.gate_normal[g] * self.gate_waypoint_distance
            point_b = self.gate_mean[g] + self.gate_normal[g] * self.gate_waypoint_distance
            if np.linalg.norm(point_a - p[i*2, :]) < np.linalg.norm(point_b - p[i*2, :]):
                p[i*2+1, :] = point_a
                p[i*2+2, :] = point_b
            else:
                p[i*2+1, :] = point_b
                p[i*2+2, :] = point_a

        n = 100
        ev, total = self.linear_interpolate(p, n)
        self.reference_waypoints = p
        self.reference_trajectory = ev[:total, :]

    def _imu_update(self):
        if self.imu_enabled:
            now = rospy.Time.now()
            time_elapsed = now - self.prev_imu_time
            self.prev_imu_time = now
            nsec_elapsed = time_elapsed.to_nsec()
            sec_elapsed = nsec_elapsed / 1e9
            omega = self.imu_acc_omega
            r, p, y = math_utils.rpy(self.last_orientation)
            r_updated = omega[0] * sec_elapsed + r
            p_updated = omega[1] * sec_elapsed + p
            y_updated = omega[2] * sec_elapsed + y
            updated_q = math_utils.q_from_rpy(r_updated, p_updated, y_updated)
            body_xddot = self.imu_acc_xddot
            xddot = self.body_to_world(self.last_orientation, body_xddot) + gravity_vector
            xdot = sec_elapsed * xddot + self.last_xdot  # Integrate from xddot
            self.last_xdot = xdot
            self.last_orientation = updated_q

    def pub_state(self, x, xdot, xddot, q):
        msg = State()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.pose.position = Vector3(x[0], x[1], x[2])
        msg.pose.orientation = Quaternion(q.x, q.y, q.z, q.w)
        msg.linear_velocity = Vector3(xdot[0], xdot[1], xdot[2])
        self.state_pub.publish(msg)

    def integrate_position(self):
        time_now = rospy.Time.now()
        nsec_elapsed = (time_now - self.prev_time_int).to_nsec()
        if nsec_elapsed > 0:
            sec_elapsed = nsec_elapsed / 1e9
            self.prev_time_int = time_now
            self.last_position = sec_elapsed * self.last_xdot + self.last_position

    def _build_state(self):
        x = self.last_position
        xdot = self.last_xdot
        q = self.last_orientation
        return np.float32([x[0], x[1], x[2], xdot[0], xdot[1], xdot[2], q.x, q.y, q.z, q.w])

    def _set_from_state(self, x):
        self.last_position = np.float32(x[0:3])
        self.last_xdot = np.float32(x[3:6])
        self.last_orientation = np.quaternion(x[9], x[6], x[7], x[8])

    def step(self):
        if self.imu_enabled:
            self.integrate_position()
        elif self.latest_thrust is not None:
            self._set_from_state(self.dynamics.f(self._build_state(), self.latest_thrust))
        xi = self.last_position
        xdoti = self.last_xdot
        qi = self.last_orientation
        x, xdot, xddot, q = self.tf_estimator.state_estimate()  # This is GT
        print("Pos drift:", np.linalg.norm(xi - x))
        self.pub_state(xi, xdoti, xdoti, qi)

    def sample(self, iterator1, iterator2, k):
        """
        Samples k elements from an iterable object.

        :param iterator: an object that is iterable
        :param k: the number of items to sample
        """
        # fill the reservoir to start
        result1 = [iterator1[i] for i in range(k)]
        result2 = [iterator2[i] for i in range(k)]

        n = k - 1
        for i in range(len(iterator1)):
            n += 1
            s = random.randint(0, n)
            if s < k:
                result1[s] = iterator1[i]
                result2[s] = iterator2[i]

        return result1, result2

    def pos_sanity(self, x):
        """Check if position is inside of race hall"""
        return np.all(x > self.x_min) and np.all(x < self.x_max)

    def closest_gate(self, x):
        best_dist = 1e9
        best_gate = None
        for gate in self.gate_mean:
            dist = np.linalg.norm(x - self.gate_mean[gate])
            if dist < best_dist:
                best_dist = dist
                best_gate = self.gate_mean[gate]
        return best_gate

    def ir_subscriber(self, msg):
        self.wait_i += 1
        if self.wait_i > 40:
            takeoff_msg = Bool(data=True)
            self.takeoff_pub.publish(takeoff_msg)

        image_points = defaultdict(list)
        object_points = defaultdict(list)
        all_image_points = []
        all_object_points = []
        num_markers = 0
        for marker in msg.markers:
            image_points[marker.landmarkID.data].append(np.float32([marker.x, marker.y]))
            object_points[marker.landmarkID.data].append(self.gate_markers[marker.landmarkID.data][marker.markerID.data])
            all_image_points.append(np.float32([marker.x, marker.y]))
            all_object_points.append(self.gate_markers[marker.landmarkID.data][marker.markerID.data])
            num_markers += 1         

        p_results = []
        q_results = []
        for gate in image_points:
            if len(image_points[gate]) == 4:  # If all 4 points are in view, solve a relative pose to the gate
                success, position, orientation = solve_pnp(object_points[gate], image_points[gate], self.last_position_ir, self.last_orientation)
                if not self.pos_sanity(position):
                    continue
                p_results.append(position)
                q_results.append(orientation)

        if len(p_results) == 0:
            return
        use_pos = None
        use_q = None
        max_dist = 1e9
        for i in range(len(p_results)):
            # If we have recent IR measurement, this one cannot be too far off
            secs_elapsed = (rospy.Time.now() - self.prev_time).to_sec()
            if secs_elapsed < 0.1 and np.linalg.norm(p_results[i] - self.last_position_ir) > 1.0:
                continue
            if np.linalg.norm(self.distance_to_trajectory(p_results[i])) < max_dist:  # Choose the solution closest to the reference trajectory
                use_pos = p_results[i]
                use_q = q_results[i]
        time_now = rospy.Time.now()
        nsec_elapsed = (time_now - self.prev_time).to_nsec()
        if nsec_elapsed > 0 and use_pos is not None:    
            true_rate = 1.0/nsec_elapsed * 1e9
            xdot = true_rate * (use_pos - self.last_position_ir)
            x = use_pos
            q = use_q
            print('update')
            self.prev_time = time_now
            self.prev_time_int = time_now
            self.last_xdot = xdot
            self.last_position_ir = x
            self.last_position = x
            self.prev_closest_landmark = self.closest_gate(x)
            self.last_orientation = q
        


if __name__ == "__main__":
    try:
        rospy.init_node('ir_node')
        node = IREstimator()
        rate = rospy.Rate(RATE)
        while not rospy.is_shutdown():
            node.step()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

