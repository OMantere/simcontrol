import numpy as np
import quaternion

def shortest_arc(v1, v2):
    cross = np.cross(v1, v2)
    v1sqr = np.dot(v1,v1)
    v2sqr = np.dot(v2,v2)
    return np.quaternion(np.sqrt(v1sqr * v2sqr) + np.dot(v1, v2), cross[0], cross[1], cross[2]).normalized()


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

