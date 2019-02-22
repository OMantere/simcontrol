import numpy as np
import quaternion
import cv2

w = 1028
h = 768
fov = 70.0
baseline = 0.32

def radial_correction(x):
    k = [[-0.00012990738802631872, 1.2054033902290686, 0.0045308598380855436],
 [8.0829008163425578e-05, 0.9003035510021189, -1.2587111710371524e-05]]
    xcorr1 = k[0][0] * x[0]**2 + k[0][1] * x[0] + k[0][2]
    xcorr2 = k[1][0] * x[1]**2 + k[1][1] * x[1] + k[1][2]
    return np.float32([xcorr1, xcorr2])

def q_rot(q, r):
    qr = np.quaternion(0.0, r[0], r[1], r[2])
    qr = q * qr * q.conjugate()
    return np.array([qr.x, qr.y, qr.z])

def stereo_correction(q, o, side='left'):
    # perform origin correction because stereo cameras are spaced out
    y_trans = baseline/2 if side == 'left' else -baseline/2
    stereo_translation = q_rot(q, np.float32([0, y_trans, 0]))
    return o + stereo_translation

def screen_space(x, y):
    x = (x - w/2)/(w/2)
    y = (y - h/2)/(h/2)
    return np.float32([x, y])

def camera_proj(q, p):
    pdot = q_rot(q.conjugate(), p)
    x = pdot[1] / pdot[0]
    y = pdot[2] / pdot[0]
    x = x/(fov/2/45.0)
    y = y/(fov/2/45.0)
    return np.float32([-x, -y])

def camera_ray(x, y, q, corr=False):
    x = screen_space(x, y)
    if corr:
        x = radial_correction(x)
    x, y = x[0], x[1]
    x = x*(fov/2/45.0)
    y = y*(fov/2/45.0)
    v = np.float32([1, -x, -y])
    return q_rot(q, v / np.linalg.norm(v))

def ray_p_dist(o, d, p, pert=np.float32([0.0, 0.0, 0.0]), stereo=False):
    pdot = p - o
    return np.linalg.norm(np.cross(pdot, d))

def to_rvec(q):
    return cv2.Rodrigues(quaternion.as_rotation_matrix(q))[0]

def to_q(rvec):
    return quaternion.from_rotation_matrix(cv2.Rodrigues(rvec)[0])

def cv_point_trans(x):
    return np.float32([-x[1], -x[2], x[0]])

def cv_point_inv_trans(x):
    return np.float32([x[2], -x[0], -x[1]])

def cv_q_trans(q):
    return np.quaternion(-q.w, -q.y, -q.z, q.x)

def cv_q_inv_trans(q):
    return np.quaternion(q.w, q.z, -q.x, -q.y)

def axis_angle(q):
    n = np.float32([q.x, q.y, q.z])
    n_norm = np.linalg.norm(n)
    return n/n_norm, 2 * np.arctan2(n_norm, q.w)

def axis_angle_inv(axis, angle):
    x = axis[0] * np.sin(angle/2)
    y = axis[1] * np.sin(angle/2)
    z = axis[2] * np.sin(angle/2)
    return np.quaternion(np.cos(angle/2), x, y, z)

def rvec2q(rvec):
    angle = np.linalg.norm(rvec)
    r = rvec / angle
    return axis_angle_inv(r, angle)

def q2rvec(q):
    return to_rvec(cv_q_trans(q))

def rvec2q(rvec):
    return cv_q_inv_trans(to_q(rvec))