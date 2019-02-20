#! /usr/bin/python

import airsim
import rospy
import tf
import time
from math import *
from collections import defaultdict
import numpy as np
import quaternion
from simple_pid import PID
from mav_msgs.msg import RateThrust
from sensor_msgs.msg import Imu
from flightgoggles.msg import IRMarkerArray, IRMarker
from lib.graphix import camera_ray

class ControllerBase(object):
    def __init__(self, rate):
        self.rate = rate
        self.g = 9.81
        self.sample_time = 1.0/rate
        self.max_omega = np.pi


class PIDCascadeV1(ControllerBase):
    def __init__(self, rate):
        ControllerBase.__init__(self, rate)
        pki = [1.0, 0.0, 0.0]
        pki2xy = [1.0, 0.0, 0.0]
        pki2z = [2.0, 0.0, 0.0]
        pki3 = [50.0, 0.0, 0.0]
        pki3z = [2.0, 0.0, 0.0]
        vz_lim = (-6.0, 6.0)
        vxy_lim = (-15.0, 15.0)
        axy_lim = (-2.0, 2.0)
        az_lim = (-4.0, 4.0)
        omegaxy_lim = (-self.max_omega, self.max_omega)
        omegaz_lim = (-self.max_omega, self.max_omega)
        self.px_pid = PID(*pki, sample_time=self.sample_time)
        self.py_pid = PID(*pki, sample_time=self.sample_time)
        self.pz_pid = PID(*pki, sample_time=self.sample_time)
        self.vx_pid = PID(*pki2xy, sample_time=self.sample_time)
        self.vy_pid = PID(*pki2xy, sample_time=self.sample_time)
        self.vz_pid = PID(*pki2z, sample_time=self.sample_time)
        self.thetax_pid = PID(*pki3, sample_time=self.sample_time) 
        self.thetay_pid = PID(*pki3, sample_time=self.sample_time)
        self.thetaz_pid = PID(*pki3z, sample_time=self.sample_time)
        self.vx_pid.output_limits = axy_lim
        self.vy_pid.output_limits = axy_lim
        self.vz_pid.output_limits = az_lim
        self.px_pid.output_limits = vxy_lim
        self.py_pid.output_limits = vxy_lim
        self.pz_pid.output_limits = vz_lim
        self.thetax_pid.output_limits = omegaxy_lim
        self.thetay_pid.output_limits = omegaxy_lim
        self.thetaz_pid.output_limits = omegaz_lim

    def rpy(self, q):
        w, x, y, z = q.w, q.x, q.y, q.z
        ysqr = y * y
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        roll = np.arctan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        elif t2 < -1.0:
            t2 = -1.0
        pitch = np.arcsin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        yaw = np.arctan2(t3, t4)
        return roll, pitch, yaw

    def w2b(self, q_b, a_d):
        qr = np.quaternion(0.0, a_d[0], a_d[1], a_d[2])
        qr = q_b.conjugate() * qr * q_b
        return np.array([qr.x, qr.y, qr.z])

    def shortest_arc(self, v1, v2):
        cross = np.cross(v1, v2)
        v1sqr = np.dot(v1,v1)
        v2sqr = np.dot(v2,v2)
        q = np.quaternion(np.sqrt(v1sqr * v2sqr) + np.dot(v1, v2), cross[0], cross[1], cross[2])
        norm = np.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
        return np.quaternion(q.w / norm, q.x / norm, q.y / norm, q.z / norm)

    def pid1(self, x_d):
        estimate = self.state_estimate()
        if not estimate:
            print("PID controller: Failed to get state estimate, skipping frame")
            return
        if not self.armed():
            self.command(9.81 * 1.2, np.array([0.0, 0.0, 0.0]))  # c > 1.1 * m * g to arm
            return
        x, xdot, xddot, q_b = estimate
        ray, area = self.ir_waypoint(q_b)
        if ray is not None:
            x_d = ray * 5 + x
            print('Ray is: ', ray)
            print('Area is', area)
        roll, pitch, yaw = self.rpy(q_b)
        g_vec = np.array([0.0, 0.0, -9.81])
        unit_z = np.array([0.0, 0.0, 1.0])
        print('pos error', x_d - x)

        # Position controller
        self.px_pid.setpoint = x_d[0]
        self.py_pid.setpoint = x_d[1]
        self.pz_pid.setpoint = x_d[2]
        v_d = np.array([0.0, 0.0, 0.0])
        v_d[0] = self.px_pid(x[0])
        v_d[1] = self.py_pid(x[1])
        v_d[2] = self.pz_pid(x[2])

        # Velocity controller
        self.vx_pid.setpoint = v_d[0]
        self.vy_pid.setpoint = v_d[1]
        self.vz_pid.setpoint = v_d[2]
        a_d = np.array([0.0, 0.0, 0.0])
        a_d[0] = self.vx_pid(xdot[0])
        a_d[1] = self.vy_pid(xdot[1])
        a_d[2] = self.vz_pid(xdot[2])

        # World to body transform
        a_db = self.w2b(q_b, a_d - g_vec)

        # Find desired attitude correction using shortest arc algorithm
        q_theta = self.shortest_arc(unit_z, a_db)
        roll_d, pitch_d, yaw_d = self.rpy(q_theta)
        v = x_d - x
        yaw_d = np.arctan2(v[1], v[0]) # Point towards the next trajectory point
        yaw = yaw % (2*np.pi)
        yaw_d = yaw_d % (2*np.pi)
        yaw_diff = (yaw_d - yaw) % (2*np.pi)
        if yaw_diff > np.pi:
            yaw_diff -= 2*np.pi

        # Apply attitude corrections using attitude PID
        self.thetax_pid.setpoint = roll_d
        self.thetay_pid.setpoint = pitch_d
        self.thetaz_pid.setpoint = yaw_diff
        omega_d = np.array([0.0, 0.0, 0.0])
        omega_d[0] = self.thetax_pid(0) # 0 because a correction quantity is being tracked
        omega_d[1] = self.thetay_pid(0)
        omega_d[2] = self.thetaz_pid(0)

        # Send command
        c = np.linalg.norm(a_db) # Desired thrust is norm of desired acceleration
        self.command(c, omega_d)


class AirsimController(PIDCascadeV1):
    def __init__(self, rate, client):
        PIDCascadeV1.__init__(self, rate)
        self.client = client
        self.max_acc = 6.7 + self.g
    
    def armed(self):
        return True

    def command(self, c, omega):
        omega_x = float(omega[1])
        omega_y = -float(omega[0])
        omega_z = -float(omega[2])
        c = c / self.max_acc  # Airsim takes normalized thrust
        self.client.moveByAnglerateThrottleAsync(omega_x, omega_y, omega_z, c, 1/self.rate)

    def state_estimate(self):
        state = self.client.getMultirotorState()
        p = state.kinematics_estimated.position
        v = state.kinematics_estimated.linear_velocity
        a = state.kinematics_estimated.linear_acceleration
        q = state.kinematics_estimated.orientation
        omega = state.kinematics_estimated.angular_velocity
        alpha = state.kinematics_estimated.angular_acceleration
        p.y_val = -p.y_val
        p.z_val = -p.z_val
        v.y_val = -v.y_val
        v.z_val = -v.z_val
        a.y_val = -a.y_val
        a.z_val = -a.z_val
        omega.y_val = -omega.y_val
        omega.z_val = -omega.z_val
        alpha.y_val = -alpha.y_val
        alpha.z_val = -alpha.z_val
        q.y_val = -q.y_val
        q.z_val = -q.z_val
        x = np.array([p.x_val, p.y_val, p.z_val])
        xdot = np.array([v.x_val, v.y_val, v.z_val])
        xddot = np.array([a.x_val, a.y_val, a.z_val])
        q = np.quaternion(q.w_val, q.x_val, q.y_val, q.z_val)
        return x, xdot, xddot, q


class FlightgogglesController(PIDCascadeV1):
    def __init__(self, rate, init_position, use_gt_state=False):
        PIDCascadeV1.__init__(self, rate)
        self.use_gt_state = use_gt_state   # TF topic provides GT
        self.max_acc = 30.0  # A lot more performant than the Airsim drone
        self.rate_thrust_publisher = rospy.Publisher('/uav/input/rateThrust', RateThrust, queue_size=10)
        self.ir_beacons = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.ir_subscriber)
        self.imu_topic = '/uav/sensors/imu'
        self.imu_subscriber = rospy.Subscriber(self.imu_topic, Imu, self.imu_subscriber)
        self.tf_listener = tf.TransformListener()
        self.init_position = init_position
        self.tf_prev_x = init_position
        self.tf_prev_xdot = np.array([0.0, 0.0, 0.0])
        self.imu_latest_x = init_position
        self.imu_latest_xdot = np.array([0.0, 0.0, 0.0])
        self.imu_latest_xddot = np.array([0.0, 0.0, 0.0])
        self.imu_latest_omega = np.array([0.0, 0.0, 0.0])
        self.is_armed = False
        try:
            self.gate_names = rospy.get_param("/gate_names")
        except:
            self.gate_names = None
        self.target_gate = 0

        self.max_omega = 5*np.pi
        # TODO: Extract PID configuration into a function
        pki = [1.0, 0.0, 0.0]
        pki2xy = [2.0, 0.0, 0.0]
        pki2z = [30.0, 0.0, 0.0]
        pki3 = [50.0, 0.0, 0.0]
        pki3z = [2.0, 0.0, 0.0]
        vz_lim = (-6.0, 6.0)
        vxy_lim = (-15.0, 15.0)
        axy_lim = (-2.0, 2.0)
        az_lim = (-4.0, 4.0)
        omegaxy_lim = (-self.max_omega, self.max_omega)
        omegaz_lim = (-self.max_omega, self.max_omega)
        self.px_pid = PID(*pki, sample_time=self.sample_time)
        self.py_pid = PID(*pki, sample_time=self.sample_time)
        self.pz_pid = PID(*pki, sample_time=self.sample_time)
        self.vx_pid = PID(*pki2xy, sample_time=self.sample_time)
        self.vy_pid = PID(*pki2xy, sample_time=self.sample_time)
        self.vz_pid = PID(*pki2z, sample_time=self.sample_time)
        self.thetax_pid = PID(*pki3, sample_time=self.sample_time) 
        self.thetay_pid = PID(*pki3, sample_time=self.sample_time)
        self.thetaz_pid = PID(*pki3z, sample_time=self.sample_time)
        self.vx_pid.output_limits = axy_lim
        self.vy_pid.output_limits = axy_lim
        self.vz_pid.output_limits = az_lim
        self.px_pid.output_limits = vxy_lim
        self.py_pid.output_limits = vxy_lim
        self.pz_pid.output_limits = vz_lim
        self.thetax_pid.output_limits = omegaxy_lim
        self.thetay_pid.output_limits = omegaxy_lim
        self.thetaz_pid.output_limits = omegaz_lim
        self.latest_markers = defaultdict(list)
        self.latest_markers_time = None
        self.latest_gate_visible_area = 0

    def ir_waypoint(self, q_b):
        center = np.float32([0, 0])
        target_gate_name = None
        if self.gate_names is None:
            if len(self.latest_markers) == 0:
                print("No gates in sight")
                return
            else:
                target_gate_name = list(self.latest_markers)[0]
                print("Going to gate: ", target_gate_name)
        else:
            target_gate_name = self.gate_names[self.target_gate]
        if target_gate_name not in self.latest_markers:
            print("Cannot see IR markers of target gate!")
            return None, 0
        else:
            target_markers = [marker for _, marker in self.latest_markers[target_gate_name].items()]
            mean_pixel = sum(target_markers) / len(target_markers)
            gate_visible_area = 0
            print('%d markers visible' % len(target_markers))
            if len(target_markers) == 4: # If all are visible, compute area
                r1 = self.latest_markers[target_gate_name]['1']
                r2 = self.latest_markers[target_gate_name]['2']
                r3 = self.latest_markers[target_gate_name]['3']
                r4 = self.latest_markers[target_gate_name]['4']
                gate_visible_area += 0.5 * np.linalg.norm(np.cross(r2 - r1, r3 - r1))
                gate_visible_area += 0.5 * np.linalg.norm(np.cross(r2 - r4, r3 - r4))
            # Unit vector in the direction of the average of IR markers in pixel space
            self.latest_gate_visible_area = gate_visible_area
            return np.float32(camera_ray(mean_pixel[0], mean_pixel[1], q_b, corr=True)), gate_visible_area

    def ir_subscriber(self, msg):
        self.latest_markers = defaultdict(dict)
        self.latest_markers_time = msg.header.stamp
        for marker in msg.markers:
            self.latest_markers[marker.landmarkID.data][marker.markerID.data] = np.array([marker.x, marker.y])
        if self.latest_gate_visible_area > 200000:
            self.target_gate += 1
            self.latest_gate_visible_area = 0
        
    def armed(self):
        if np.abs(self.tf_prev_x[2] - self.init_position[2]) > 0.1:
            self.is_armed = True
        return self.is_armed

    def imu_subscriber(self, msg):
        omega = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        omegabar = (omega + self.imu_latest_omega) / 2
        xddot = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        xdot = self.imu_latest_xdot + (xddot + self.imu_latest_xddot) / 2 * self.sample_time  # Integrate from xddot
        x = self.imu_latest_x + (xdot + self.imu_latest_xdot) / 2 * self.sample_time  # Integrate from xdot
        self.imu_latest_x = x
        self.imu_latest_xdot = xdot
        self.imu_latest_xddot = xddot
        self.imu_latest_omega = omega

    def command(self, c, omega):
        omega_x = float(omega[0]) / self.max_omega
        omega_y = float(omega[1]) / self.max_omega
        omega_z = float(omega[2]) / self.max_omega
        msg = RateThrust()
        msg.thrust.x = 0
        msg.thrust.y = 0
        msg.thrust.z = c
        msg.angular_rates.x = omega_x
        msg.angular_rates.y = omega_y
        msg.angular_rates.z = omega_z
        self.rate_thrust_publisher.publish(msg)

    def axis_angle(self, axis, angle):
        x = axis[0] * np.sin(angle)
        y = axis[1] * np.sin(angle)
        z = axis[2] * np.sin(angle)
        return np.quaternion(np.cos(angle), x, y, z)

    def state_estimate(self):
        if self.use_gt_state:
            timestamp = self.latest_markers_time
            if timestamp is None:
                timestamp = rospy.Time.now()
            try:
                translation, rotation = self.tf_listener.lookupTransform('world','uav/imu', timestamp)
                q = np.quaternion(rotation[3], rotation[0], rotation[1], rotation[2])
                #q = q.conjugate()
                x = np.array(translation)
                # Discrete derivatives
                xdot = self.rate * (x - self.tf_prev_x)
                xddot = self.rate * (xdot - self.tf_prev_xdot)
                self.tf_prev_x = x
                self.tf_prev_xdot = xdot
                return x, xdot, xddot, q
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException): 
                return None


class AirSimNode(object):
    def __init__(self, client):
        self.client = client
        self.rate = 60
        self.path = np.genfromtxt('/home/omantere/git/simcontrol/spline.csv', delimiter=',')
        self.path[:,2] += 2
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("Node connected to AirSim")

    def loop(self):
        rospy.init_node('drone_perception')
        rate = rospy.Rate(self.rate)
        self.controller = AirsimController(self.rate, self.client)
        i = 0
        while not rospy.is_shutdown():
            self.controller.pid1(self.path[int(floor(i)) % self.path.shape[0], :])
            i += 0.9
            rate.sleep()


class FlightgogglesNode(object):
    def __init__(self):
        self.rate = 60
        self.path = np.genfromtxt('/home/omantere/catkin_ws/src/simcontrol/splines/0.csv', delimiter=',')
        self.init_position = self.path[0, :]
        self.hover_test = np.array([self.init_position[0], self.init_position[1], self.init_position[2]+1])

    def loop(self):
        rospy.init_node('drone_perception', anonymous=True)
        rate = rospy.Rate(self.rate)
        self.controller = FlightgogglesController(self.rate, self.init_position, use_gt_state=True)
        i = 0
        while not rospy.is_shutdown():
            self.controller.pid1(self.path[int(floor(i)) % self.path.shape[0], :])
            i += 0.9 * 1.1
            rate.sleep()


if __name__ == '__main__':
    try:
        node = FlightgogglesNode()
        node.loop()
    except rospy.ROSInterruptException:
        pass
