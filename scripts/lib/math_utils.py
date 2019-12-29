import numpy as np
import quaternion

def identity_quaternion():
    return np.quaternion(1.0, 0.0, 0.0, 0.0)

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

def q_rot(q, r):
    """Rotate vector r by rotation described by quaternion q"""
    qr = np.quaternion(0.0, r[0], r[1], r[2])
    qr = q * qr * q.conjugate()
    return np.array([qr.x, qr.y, qr.z])

def axis_angle(q):
    """Compute axis-angle representation from quaternion"""
    n = np.float32([q.x, q.y, q.z])
    n_norm = np.linalg.norm(n)
    return n/n_norm, 2 * np.arctan2(n_norm, q.w)

def axis_angle_inv(axis, angle):
    """Compute quaternion from axis-angle representation"""
    x = axis[0] * np.sin(angle/2)
    y = axis[1] * np.sin(angle/2)
    z = axis[2] * np.sin(angle/2)
    return np.quaternion(np.cos(angle/2), x, y, z)

def q_from_rpy(r, p, y):
    """Returns the quaternion corresponding to given roll-pitch-yaw (Tait-Bryan) angles"""
    cosr = np.cos(r/2)
    sinr = np.sin(r/2)
    cosp = np.cos(p/2)
    sinp = np.sin(p/2)
    cosy = np.cos(y/2)
    siny = np.sin(y/2)
    return np.quaternion(
        cosy * cosp * cosr + siny * sinp * sinr,
        cosy * cosp * sinr - siny * sinp * cosr,
        cosy * sinp * cosr + siny * cosp * sinr,
        siny * cosp * cosr - cosy * sinp * sinr
    )

def rpy(q):
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
