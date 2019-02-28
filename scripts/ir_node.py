import rospy
from flightgoggles.msg import IRMarkerArray
from collections import defaultdict
from lib.vision import solve_pnp

def VisionNode(object):
    def __init__(self):
        rospy.init_node('ir_vision_node', anonymous=True)
        self.ir_sub = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.ir_subscriber)
        self.tf = TransformBroadcaster()
        self.gate_markers = {}
        self.get_nominal_locations()
        rospy.spin()
     
    def get_nominal_locations(self):
        for gate in range(1, 24):
            nominal_location = np.float32(rospy.get_param('/uav/Gate%d/nominal_location' % gate))
            self.gate_markers['Gate%d' % gate] = {}
            for i in range(1, 5):
                self.gate_markers['Gate%d' % gate][i] = nominal_location[i-1, :]

    def send_pose(self, x, q, stamp=rospy.Time.now(), origin_frame='world'):
        self.tf.sendTransform((x[0], x[1], x[2]), (q.x, q.y, q.z, q.w), stamp, 'quadcopter', origin_frame)

    def ir_subscriber(self, msg):
        image_points = defaultdict(list)
        object_points = defaultdict(list)
        for marker in msg.markers:
            markers[msg.landmarkID.data].append(np.float32([msg.x, msg.y]))
            object_points[msg.landmarkID.data].append(self.gate_markers[msg.landmarkID.data][msg.markerID.data])

        for gate in image_points:
            if len(image_points[gate]) == 4:  # If all 4 points are in view, solve a relative pose to the gate
                success, position, orientation = solvepnp(object_points[gate], image_points[gate], )



if __name__ == '__main__':
    try:
        node = VisionNode()
        node.loop()
    except rospy.ROSInterruptException:
        pass

