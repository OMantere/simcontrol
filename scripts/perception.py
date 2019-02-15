#! /usr/bin/env python

import airsim
import rospy
from math import *
import numpy as np
import quaternion
from simple_pid import PID

class ControllerBase(object):
    def __init__(self, client, rate):
        self.client = client
        self.rate = rate

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
        return p, v, a, q, omega, alpha


class CascadeV1(ControllerBase):
    def __init__(self, client, rate):
        ControllerBase.__init__(self, client, rate)
        sample_time = 1.0/rate
        pki = [1.0, 0.0, 0.0]
        pki2xy = [1.0, 0.0, 0.0]
        pki2z = [2.0, 0.0, 0.0]
        pki3 = [50.0, 0.0, 0.0]
        pki3z = [2.0, 0.0, 0.0]
        c_lim = (0.4, 2.0)
        vz_lim = (-6.0, 6.0)
        vxy_lim = (-15.0, 15.0)
        axy_lim = (-2.0, 2.0)
        az_lim = (-4.0, 4.0)
        omegaxy_lim = (-1e9, 1e9)
        omegaz_lim = (-1e9, 1e9)
        self.px_pid = PID(*pki, sample_time=sample_time)
        self.py_pid = PID(*pki, sample_time=sample_time)
        self.pz_pid = PID(*pki, sample_time=sample_time)
        self.vx_pid = PID(*pki2xy, sample_time=sample_time)
        self.vy_pid = PID(*pki2xy, sample_time=sample_time)
        self.vz_pid = PID(*pki2z, sample_time=sample_time)
        self.thetax_pid = PID(*pki3, sample_time=sample_time) 
        self.thetay_pid = PID(*pki3, sample_time=sample_time)
        self.thetaz_pid = PID(*pki3z, sample_time=sample_time)
        self.vx_pid.output_limits = axy_lim
        self.vy_pid.output_limits = axy_lim
        self.vz_pid.output_limits = az_lim
        self.px_pid.output_limits = vxy_lim
        self.py_pid.output_limits = vxy_lim
        self.pz_pid.output_limits = vz_lim
        self.thetax_pid.output_limits = omegaxy_lim
        self.thetay_pid.output_limits = omegaxy_lim
        self.thetaz_pid.output_limits = omegaz_lim

    def command(self, c, omega):
        omega_x = float(omega[0])
        omega_y = float(omega[1])
        omega_z = float(omega[2])
        self.client.moveByAnglerateThrottleAsync(omega_x, omega_y, omega_z, c, 1/self.rate)

    def get_state(self):
        p, v, a, q, omega, alpha = self.state_estimate()
        x = np.array([p.x_val, p.y_val, p.z_val])
        xdot = np.array([v.x_val, v.y_val, v.z_val])
        xddot = np.array([a.x_val, a.y_val, a.z_val])
        q = np.quaternion(q.w_val, q.x_val, q.y_val, q.z_val)
        return x, xdot, xddot, q

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
        g = 9.81
        max_acc = 6.7 + g
        x, xdot, xddot, q_b = self.get_state()
        roll, pitch, yaw = self.rpy(q_b)
        g_vec = np.array([0.0, 0.0, -9.81])
        unit_z = np.array([0.0, 0.0, 1.0])

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
        omega_d[1] = self.thetax_pid(0) # 0 because a correction quantity is being tracked
        omega_d[0] = -self.thetay_pid(0)
        omega_d[2] = -self.thetaz_pid(0)

        # Send command
        c = np.linalg.norm(a_db)/max_acc # Desired thrust is norm of acceleration
        self.command(c, omega_d)
        


class PerceptionNode(object):
    def __init__(self, client):
        self.client = client
        self.rate = 60
        self.path = np.genfromtxt('/home/omantere/git/simcontrol/spline.csv', delimiter=',')
        self.path[:,2] += 2

    def airpub(self):
        rospy.init_node('drone_perception')
        rate = rospy.Rate(self.rate)
        self.controller = CascadeV1(self.client, self.rate)
        i = 0
        while not rospy.is_shutdown():
            self.controller.pid1(self.path[int(floor(i)) % self.path.shape[0], :])
            i += 0.9
            rate.sleep()


if __name__ == '__main__':
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)

        node = PerceptionNode(client)
        print("Perception node connected to AirSim")

        node.airpub()
    except rospy.ROSInterruptException:
        pass
