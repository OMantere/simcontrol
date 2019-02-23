import numpy as np
import quaternion

def shortest_arc(v1, v2):
    cross = np.cross(v1, v2)
    if np.linalg.norm(cross) < 1e-15:
        # Opposite or same directions.
        if np.linalg.norm(v1 - v2):
            return np.quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            # Return rotation along x-axis.
            rotation = np.sin(np.pi/2) + np.cos(np.pi/2)
            return np.quaternion(rotation, 1.0, 0.0, 0.0).normalized()
    v1sqr = np.dot(v1,v1)
    v2sqr = np.dot(v2,v2)
    return np.quaternion(np.sqrt(v1sqr * v2sqr) + np.dot(v1, v2), cross[0], cross[1], cross[2]).normalized()

def angle_between(q1, q2):
    # Returns the angle (in radians) required to rotate one quaternion to another.
    q1 = quaternion.as_float_array(q1)
    q2 = quaternion.as_float_array(q2)
    return np.arccos(q1.dot(q2))

