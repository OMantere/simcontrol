#! /usr/bin/python

import os
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
from lib.vision import camera_ray
from sensor_msgs.msg import Image


class Node(object):
    def __init__(self):
        rospy.init_node('train_gen', anonymous=True)
        self.ir_beacons = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.ir_subscriber)
        self.rgb_right = rospy.Subscriber('/uav/camera/right/image_rect_color', Image, self.image_right_subscriber)
        self.rgb_left = rospy.Subscriber('/uav/camera/left/image_rect_color', Image, self.image_left_subscriber)
        self.latest_markers = defaultdict()
        self.stamps = defaultdict(dict)
        rospy.spin()

    def ir_subscriber(self, msg):
        self.latest_markers = defaultdict(dict)
        self.latest_markers_time = msg.header.stamp
        print('got ir beacons with stamp', msg.header.stamp)
        for marker in msg.markers:
            self.latest_markers[marker.landmarkID.data][marker.markerID.data] = np.array([marker.x, marker.y])

    def image_left_subscriber(self, msg):
        self.latest_left_time = msg.header.stamp
        self.latest_left = msg.data
        print('got left rgb with stamp', msg.header.stamp)

    def image_right_subscriber(self, msg):
        self.latest_right_time = msg.header.stamp
        self.latest_right = msg.data
        print('got right rgb with stamp', msg.header.stamp)


if __name__ == '__main__':
    try:
        node = Node()  
    except rospy.ROSInterruptException:
        pass
