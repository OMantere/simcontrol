#!/usr/bin/env python

# Example ROS node for publishing AirSim images.

# AirSim Python API
import setup_path
import airsim
import numpy as np

import rospy
import cv_bridge
import cv2

# ROS Image message
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Vector3, Quaternion

class AirPub:
    def __init__(self):
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.cam0 = rospy.Publisher("cam0_image", Image, queue_size=1)
        self.cam1 = rospy.Publisher("cam1_image", Image, queue_size=1)
        self.imu = rospy.Publisher("imu", Imu, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        rospy.init_node('image_raw', anonymous=True)

    def _get_images(self):
        # get camera images from the car
        responses = self.client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("2", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array

        msgs = []
        stamp = rospy.Time.now()
        for response in responses:
            img_rgba_string = response.image_data_uint8

            # Populate image message
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = "frameId"
            msg.encoding = "rgba8"
            msg.height = response.height
            msg.width = response.width
            msg.data = img_rgba_string
            msg.is_bigendian = 0
            msg.step = msg.width * 4

            array = self.bridge.imgmsg_to_cv2(msg, "rgba8")
            grayscale = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
            converted = self.bridge.cv2_to_imgmsg(grayscale, "8UC1")
            converted.header.stamp = stamp

            msgs.append(converted)

        assert(len(msgs) == 2)

        rospy.loginfo(len(response.image_data_uint8))

        return msgs

    def _get_imu_message(self):
        state = self.client.getMultirotorState()
        kinematics = state.kinematics_estimated
        msg = Imu()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "frameId"
        msg.orientation = Quaternion(
            x=kinematics.orientation.x_val,
            y=kinematics.orientation.y_val,
            z=kinematics.orientation.z_val,
            w=kinematics.orientation.w_val)
        msg.angular_velocity = Vector3(
            x=kinematics.angular_velocity.x_val,
            y=kinematics.angular_velocity.y_val,
            z=kinematics.angular_velocity.z_val)
        msg.linear_acceleration = Vector3(
            x=kinematics.linear_acceleration.x_val,
            y=kinematics.linear_acceleration.y_val,
            z=kinematics.linear_acceleration.z_val)
        msg.orientation_covariance[0] = -1.0
        msg.angular_velocity_covariance[0] = -1.0
        msg.linear_acceleration_covariance[0] = -1.0
        return msg

    def run(self):
        rate = rospy.Rate(10) # 10hz

        while not rospy.is_shutdown():

            img0, img1 = self._get_images()
            imu_msg = self._get_imu_message()

            self.cam0.publish(img0)
            self.cam1.publish(img1)
            self.imu.publish(imu_msg)

            rate.sleep()


if __name__ == '__main__':
    try:
        AirPub().run()
    except rospy.ROSInterruptException:
        pass
