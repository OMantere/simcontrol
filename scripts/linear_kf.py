import numpy as np
import quaternion
import rospy
from math_utils import rpy, shortest_arc
from sensor_msgs.msg import Imu

class LinearKF(object):
    def __init__(self):
        self.x = np.float32([0, 0, 0])
        init_pose = rospy.get_param('/uav/flightgoggles_uav_dynamics/init_pose')
        self.imu_sub = rospy.Subscriber('/uav/sensors/imu', Imu, self.ir_subscriber)
        self.q = np.quaternion(init_pose[6], init_pose[3], init_pose[4], init_pose[5])

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

    def step(self):
        # x_k = F_k x_k-1 + B_k u_k


if __name__ == "__main__":
    try:
        node = LinearKF()
        node.loop()
    except rospy.ROSInterruptException:
        pass
