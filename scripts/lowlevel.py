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
from simple_pid import PID
from mav_msgs.msg import RateThrust
from sensor_msgs.msg import Imu
from flightgoggles.msg import IRMarkerArray, IRMarker
from lib.graphix import camera_ray, ray_p_dist
from simcontrol.msg import State
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool

class ControllerBase(object):
    def __init__(self):
        self.g = 9.81
        self.max_omega = np.pi

class PIDCascadeV1(ControllerBase):
    def __init__(self):
        ControllerBase.__init__(self)
        self.sample_time = None   # We shouldn't limit ourselves
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

        self.x = np.float32([0,0,0])
        self.xdot = np.float32([0,0,0])
        self.xddot = np.float32([0,0,0])
        self.q_b = np.quaternion(1.0, 0.0, 0.0, 0.0)

        self.x_d = np.float32([0,0,0])
        self.yaw_d = 0.0

        self.armed = False
        self.takeoff = False

        self.state_sub = rospy.Subscriber('/simcontrol/state_estimate', State, self.state_subscriber)
        self.x_d_sub = rospy.Subscriber('/simcontrol/target_pose', Pose, self.target_subscriber)
        self.armed_sub = rospy.Subscriber('/simcontrol/armed', Bool, self.armed_subscriber)
        self.takeoff_sub = rospy.Subscriber('/simcontrol/takeoff', Bool, self.takeoff_subscriber)
        self.rate_thrust_publisher = rospy.Publisher('/uav/input/rateThrust', RateThrust, queue_size=1)

    def state_subscriber(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        v = msg.linear_velocity

    def target_subscriber(self, msg):
        q = msg.orientation
        q_d = np.quaternion([q.w, q.x, q.y, q.z])
        p = msg.position
        self.x_d = np.float32([p.x, p.y, p.z])
        _, _, self.yaw_d = self.rpy(q)

    def armed_subscriber(self, msg):
        self.armed = msg.data

    def takeoff_subscriber(self, msg):
        self.takeoff = msg.data

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

    def pid1(self, x_d, estimate):
        """x_d is relative target position"""
        if not estimate:
            print("PID controller: Failed to get state estimate, skipping frame")
            return
        if not self.armed:
            if self.takeoff:
                self.command(9.81 * 1.1, np.array([0.0, 0.0, 0.0]))  # c > 1.1 * m * g to arm
            return
        x, xdot, xddot, q_b = estimate
        roll, pitch, yaw = self.rpy(q_b)
        g_vec = np.array([0.0, 0.0, -9.81])
        unit_z = np.array([0.0, 0.0, 1.0])

        # Position controller
        self.px_pid.setpoint = x_d[0]
        self.py_pid.setpoint = x_d[1]
        self.pz_pid.setpoint = x_d[2]
        v_d = np.array([0.0, 0.0, 0.0])
        v_d[0] = self.px_pid(0.0)
        v_d[1] = self.py_pid(0.0)
        v_d[2] = self.pz_pid(0.0)

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
        roll_d, pitch_d, _ = self.rpy(q_theta)
        yaw_d = self.yaw_d
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
        omega_d[0] = self.thetax_pid(0.0) # 0 because a correction quantity is being tracked
        omega_d[1] = self.thetay_pid(0.0)
        omega_d[2] = self.thetaz_pid(0.0)

        # Send command
        c = np.linalg.norm(a_db) # Desired thrust is norm of desired acceleration
        self.command(c, omega_d)


    def loop(self):
        rospy.init_node('drone_lowlevel', anonymous=True)
        i = 0
        import time
        import math
        elapsed = time.time()
        self.armed = True
        while not rospy.is_shutdown():
            self.pid1(self.x, (self.x, self.xdot, self.xddot, self.q_b))
            i +=1
            if i % 1000 == 0:
                print(int(math.floor(1/(time.time()-elapsed)*1000)))
                elapsed = time.time()

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

if __name__ == '__main__':
    try:
        node = PIDCascadeV1()
        node.loop()
    except rospy.ROSInterruptException:
        pass
