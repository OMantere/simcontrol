import numpy as np
import quaternion

def shortest_arc(v1, v2):
    cross = np.cross(v1, v2)
    v1sqr = np.dot(v1,v1)
    v2sqr = np.dot(v2,v2)
    return np.quaternion(np.sqrt(v1sqr * v2sqr) + np.dot(v1, v2), cross[0], cross[1], cross[2]).normalized()

