#! /usr/bin/python

import os
import math
import airsim
import rospy
import tf
import time
from math import *
from collections import defaultdict
import numpy as np
import quaternion
import math_utils
from simple_pid import PID
from mav_msgs.msg import RateThrust
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion, Pose
from flightgoggles.msg import IRMarkerArray, IRMarker
from lib.vision import camera_ray, ray_p_dist
from simcontrol.msg import State

GRAVITY = 9.81

class FlightgogglesController(object):
    def __init__(self, rate, use_gt_state=False):
        self.rate = rate
        self.sample_time = 1.0 / rate
        rospy.init_node('drone_control', anonymous=True)
        self.init_position, self.init_orientation = self.get_init_pose()
        self.use_gt_state = use_gt_state   # TF topic provides GT
        self.max_acc = 30.0  # A lot more performant than the Airsim drone

        self.rate_thrust_publisher = rospy.Publisher('/uav/input/rateThrust', RateThrust, queue_size=1)
        self.ir_beacons = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self._ir_callback)
        self.target_pub = rospy.Publisher('/simcontrol/target_pose', Pose, queue_size=1)
        self.state_pub = rospy.Publisher('/simcontrol/state_estimate', State, queue_size=1)

        self.imu_topic = '/uav/sensors/imu'
        self.imu_subscriber = rospy.Subscriber(self.imu_topic, Imu, self._imu_callback)
        self.tf_listener = tf.TransformListener()
        self.tf_prev_x = self.init_position
        self.tf_prev_xdot = np.array([0.0, 0.0, 0.0])
        self.tf_prev_xddot = np.array([0.0, 0.0, 0.0])
        self.imu_latest_x = self.init_position
        self.imu_latest_xdot = np.array([0.0, 0.0, 0.0])
        self.imu_latest_xddot = np.array([0.0, 0.0, 0.0])
        self.imu_latest_omega = np.array([0.0, 0.0, 0.0])
        self.is_armed = False

        self.gate_names = rospy.get_param("/gate_names") if rospy.has_param('/gate_names') else None
        self.use_ir_markers = rospy.get_param("/use_ir_markers") if rospy.has_param('/use_ir_markers') else True
        self.target_gate = 0
        self.target_lead_distance = 5.0
        self.reference_lead_steps = 2
        self.target_update_weight = 1.0
        self.target_vector = np.float32([0, 0, 0])
        self.prev_tf_time = rospy.Time.now()

        self.latest_markers = defaultdict(list)
        self.latest_markers_time = None
        self.visible_target_markers = 0
        self.latest_gate_visible_area = 0
        self.gate_is_perturbed = {}
        self.gate_mean = {}
        self.gate_markers = {}
        self.gate_normal = {}
        self.gate_change_cutoff = 3.0
        self.gate_normal_cutoff = 2.0
        self.gate_waypoint_distance = 2.5
        self.get_gate_info()

    def linear_interpolate(self, p, n):
        res = np.zeros((n * p.shape[0] * 100, p.shape[1]))
        ii = 0
        for i in range(p.shape[0] - 1):
            res[ii, :] = p[i, :]
            trans = p[i + 1, :] - p[i, :]
            dist = np.linalg.norm(trans)
            nn = int(math.floor(dist * n))
            for j in range(nn):
                ii += 1
                res[ii, :] = p[i, :] + trans * (1.0*j/nn)
            ii += 1
        return res, ii

    def get_gate_info(self):
        for gate in range(1, 24):
            nominal_location = np.float32(rospy.get_param('/uav/Gate%d/nominal_location' % gate))
            self.gate_markers['Gate%d' % gate] = {}
            for i in range(1, 5):
                self.gate_markers['Gate%d' % gate][i] = nominal_location[i-1,:]
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

        n = 10
        ev, total = self.linear_interpolate(p, n)
        self.path = ev[:total, :]

    def get_init_pose(self):
        if not rospy.has_param('/uav/flightgoggles_uav_dynamics/init_pose'):
            return np.float32([0, 0, 0]), np.quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            init_pose = rospy.get_param('/uav/flightgoggles_uav_dynamics/init_pose')
            x = np.float32([init_pose[0], init_pose[1], init_pose[2]])
            q = np.quaternion(init_pose[6], init_pose[3], init_pose[4], init_pose[5])
            return x, q

    def tangent_unit_vector(self, x1, x2):
        return (x2 - x1) / np.linalg.norm(x2 - x1)

    def loop(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            target_gate_name = self.gate_names[self.target_gate]
            state_estimate = self.state_estimate()
            x = state_estimate[0]
            q_b = state_estimate[3]
            new_target_vector = self.reference_target_vector(x)
            # We only want to use IR markers to correct for gate perturbation
            # If we are close to the target gate or there is no perturbation, just use reference trajectory
            if self.use_ir_markers:
                ir_ray = self.ir_waypoint()
                if ir_ray is not None and self.visible_target_markers >= 3:
                    new_target_vector = ir_ray
            new_target_vector *= self.target_lead_distance
            self.target_vector += (new_target_vector - self.target_vector) * self.target_update_weight

            self._set_target(self.target_vector, state_estimate)

            target_gate_center = self.gate_mean[target_gate_name]
            target_gate_distance = np.linalg.norm(target_gate_center - x)
            gate_normal_distance = ray_p_dist(target_gate_center, self.gate_normal[target_gate_name], x)
            # TODO: Compute the gate normal plane from IR markers and check distance to it
            # because this fails for wide gates
            if target_gate_distance < self.gate_change_cutoff and gate_normal_distance < self.gate_normal_cutoff:
                self.target_gate += 1
                if self.target_gate < len(self.gate_names):
                    print("New gate: %s" % self.gate_names[self.target_gate])
                else:
                    self.target_gate = len(self.gate_names) - 1
                    print("Finished course")
            rate.sleep()

    def _gaze_direction(self, x):
        gaze_target_gate = min(len(self.gate_names) - 1, self.target_gate+1)
        pointing_direction = self.gate_mean[self.gate_names[gaze_target_gate]] - x
        if self.use_ir_markers:
            ir_ray = self.ir_waypoint()
            if ir_ray is not None:
                pointing_direction = ir_ray
        return pointing_direction

    def _orientation_quaternion(self, x):
        # The low level controller will use only the yaw angle.
        # Returns the quaternion which will rotate the x-axis to
        # point towards the next gate.
        gaze_direction = self._gaze_direction(x)
        return math_utils.shortest_arc(np.array([1., 0., 0.]), gaze_direction)

    def _set_target(self, target_vector, state_estimate):
        x, xdot, xddot, q = state_estimate

        current_state = State()
        current_state.header.stamp = rospy.Time.now()
        current_state.header.frame_id = 'state_estimate'
        current_state.pose.position = Vector3(x[0], x[1], x[2])
        current_state.pose.orientation = Quaternion(q.x, q.y, q.z, q.w)
        current_state.linear_velocity = Vector3(xdot[0], xdot[1], xdot[2])
        current_state.linear_acceleration = Vector3(xddot[0], xddot[1], xddot[2])
        self.state_pub.publish(current_state)

        target_pose = Pose()
        target_pose.position = Vector3(target_vector[0], target_vector[1], target_vector[2])

        q = self._orientation_quaternion(x)
        target_pose.orientation = Quaternion(q.x, q.y, q.z, q.w)
        self.target_pub.publish(target_pose)

    def reference_target_vector(self, x):
        closest_dist = 1e9
        closest_i = 0
        for i in range(self.path.shape[0])[1:]:
            new_d = np.linalg.norm(self.path[i, :] - x)
            if new_d < closest_dist:
                closest_dist = new_d
                closest_i = i
        return self.tangent_unit_vector(self.path[closest_i, :], self.path[closest_i+self.reference_lead_steps, :])

    def ir_waypoint(self):
        center = np.float32([0, 0])
        target_gate_name = None
        self.visible_target_markers = 0
        if self.gate_names is None:
            print("List of gate names no loaded")
        else:
            target_gate_name = self.gate_names[self.target_gate]
        if target_gate_name not in self.latest_markers:
            return None
        else:
            q_b = self.state_estimate(self.latest_markers_time)[3]
            target_markers = [marker for _, marker in self.latest_markers[target_gate_name].items()]
            mean_pixel = sum(target_markers) / len(target_markers)
            gate_visible_area = 0
            self.visible_target_markers = len(target_markers)
            if self.visible_target_markers == 4: # If all are visible, compute area
                r1 = self.latest_markers[target_gate_name]['1']
                r2 = self.latest_markers[target_gate_name]['2']
                r3 = self.latest_markers[target_gate_name]['3']
                r4 = self.latest_markers[target_gate_name]['4']
                gate_visible_area += 0.5 * np.linalg.norm(np.cross(r2 - r1, r3 - r1))
                gate_visible_area += 0.5 * np.linalg.norm(np.cross(r2 - r4, r3 - r4))
            self.latest_gate_visible_area = gate_visible_area
            # Unit vector in the direction of the average of IR markers in pixel space
            return np.float32(camera_ray(mean_pixel[0], mean_pixel[1], q_b, corr=True))

    def _ir_callback(self, msg):
        self.latest_markers = defaultdict(dict)
        self.latest_markers_time = msg.header.stamp
        for marker in msg.markers:
            self.latest_markers[marker.landmarkID.data][marker.markerID.data] = np.array([marker.x, marker.y])

    def _imu_callback(self, msg):
        omega = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        omegabar = (omega + self.imu_latest_omega) / 2
        xddot = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        xdot = self.imu_latest_xdot + (xddot + self.imu_latest_xddot) / 2 * self.sample_time  # Integrate from xddot
        x = self.imu_latest_x + (xdot + self.imu_latest_xdot) / 2 * self.sample_time  # Integrate from xdot
        self.imu_latest_x = x
        self.imu_latest_xdot = xdot
        self.imu_latest_xddot = xddot
        self.imu_latest_omega = omega

    def state_estimate(self, timestamp=None):
        if self.use_gt_state:
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
                    true_rate = 1/time_elapsed.to_sec()
                    xdot = true_rate * (x - self.tf_prev_x)
                    xddot = true_rate * (xdot - self.tf_prev_xdot)
                    self.prev_tf_time = self.prev_tf_time + time_elapsed
                    self.tf_prev_x = x
                    self.tf_prev_xdot = xdot
                    self.tf_prev_xddot = xddot
                return x, xdot, xddot, q
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                return self.init_position, np.float32([0, 0, 0]), np.float32([0, 0, 0]), self.init_orientation
        else:
            raise NotImplementedError('Non-GT state estimate not implemented')


if __name__ == '__main__':
    try:
        node = FlightgogglesController(rate=1000, use_gt_state=True)
        node.loop()
    except rospy.ROSInterruptException:
        pass
