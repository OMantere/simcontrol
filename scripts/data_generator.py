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
from lib.graphix import camera_ray
from sensor_msgs.msg import Image


class Node(object):
    def __init__(self):
        rospy.init_node('train_gen', anonymous=True)
        self.tf_listener = tf.TransformListener()
        self.rgb_left = rospy.Subscriber('/uav/camera/left/image_rect_color', Image, self.image_left_subscriber)
        self.left_images {}
        self.x = {}
        self.q = {}
        rospy.spin()

    def image_left_subscriber(self, msg):
        translation, rotation = self.tf_listener.lookupTransform('world','uav/imu', msg.header.stamp)
        q = np.quaternion(rotation[3], rotation[0], rotation[1], rotation[2])
        x = np.array(translation)
        nsec = msg.header.stamp.to_sec
        self.left_images[nsec] = np.fromstring(image_msg.data, np.uint8)
        self.x[nsec] = x
        self.q[nsec] = q
        print('got left rgb with stamp', msg.header.stamp)

if __name__ == '__main__':
    try:
        node = Node()  
    except rospy.ROSInterruptException:
        pass
