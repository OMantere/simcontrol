import rospy
from flightgoggles.msg import IRMarkerArray

def VisionNode(object):
    def __init__(self):
        rospy.init_node('vision_node', anonymous=True)
        self.ir_sub = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.ir_subscriber)

    def loop(self):
        while not rospy.is_shutdown():

    def pose_estimate(self, image_points, object_points):


    def ir_subscriber(self, msg):
        for marker in msg.markers:

            


if __name__ == '__main__':
    try:
        node = VisionNode()
        node.loop()
    except rospy.ROSInterruptException:
        pass

