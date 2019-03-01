import numpy as np
import quaternion
import cv2
from math_utils import q_rot, identity_quaternion
import time

w = 1028
h = 768
fov = 70.0
baseline = 0.32
fx = 1/(fov/2/45.0)*w/2
fy = 1/(fov/2/45.0)*h/2
cx = w / 2.0
cy = h / 2.0
camera_matrix = np.float32([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def polynomial_correction(x):
    """Apply a 2nd order lens distortion correction to screen space point x"""
    k = [[-0.00012990738802631872, 1.2054033902290686, 0.0045308598380855436],
 [8.0829008163425578e-05, 0.9003035510021189, -1.2587111710371524e-05]]
    xcorr1 = k[0][0] * x[0]**2 + k[0][1] * x[0] + k[0][2]
    xcorr2 = k[1][0] * x[1]**2 + k[1][1] * x[1] + k[1][2]
    return np.float32([xcorr1, xcorr2])

def pixel_space(x):
    """From screen space to pixel space"""
    p1 = (x[0] + 1)*(w/2)
    p2 = (x[1] + 1)*(h/2)
    return np.float32([p1, p2])

def stereo_correction(q, o, side='left'):
    """Perform origin correction because stereo cameras are spaced out"""
    y_trans = baseline/2 if side == 'left' else -baseline/2
    stereo_translation = q_rot(q, np.float32([0, y_trans, 0]))
    return o + stereo_translation

def screen_space(x):
    """Transform from pixel space to screen space (x in [-1, 1], y in [-1, 1])"""
    y1 = (x[0] - w/2)/(w/2)
    y2 = (x[1] - h/2)/(h/2)
    return np.float32([y1, y2])

def camera_proj(q, p):
    """Project point p in object space to camera screen space"""
    pdot = q_rot(q.conjugate(), p)
    x = pdot[1] / pdot[0]
    y = pdot[2] / pdot[0]
    x = x/(fov/2/45.0)
    y = y/(fov/2/45.0)
    return np.float32([-x, -y])

def camera_ray(x, q, corr=False):
    """Get the ray in object space which corresponds to pixel at (x, y)"""
    x = screen_space(x)
    if corr:
        x = polynomial_correction(x)
    x, y = x[0], x[1]
    x = x*(fov/2/45.0)
    y = y*(fov/2/45.0)
    v = np.float32([1, -x, -y])
    return q_rot(q, v / np.linalg.norm(v))

def ray_p_dist(origin, direction, point, pert=np.float32([0.0, 0.0, 0.0]), stereo=False):
    """Closest distance from ray defined by origin and direction to point b"""
    pdot = point - origin
    return np.linalg.norm(np.cross(pdot, direction))

def to_rvec(q):
    return cv2.Rodrigues(quaternion.as_rotation_matrix(q))[0]

def to_q(rvec):
    return quaternion.from_rotation_matrix(cv2.Rodrigues(rvec)[0])

def cv_point_trans(x):
    """Transform position from our representation to OpenCV convention"""
    return np.float32([-x[1], -x[2], x[0]])

def cv_point_inv_trans(x):
    """Transform position from OpenCV convention to our representation"""
    return np.float32([x[2], -x[0], -x[1]])

def cv_q_trans(q):
    """Transform orientation from our representation to OpenCV convention"""
    return np.quaternion(-q.w, -q.y, -q.z, q.x)

def cv_q_inv_trans(q):
    """Transform orientation from OpenCV convention to our representation"""
    return np.quaternion(q.w, q.z, -q.x, -q.y)

def q2rvec(q):
    """Compute OpenCV Rodrigues rotation vector from our quaternion"""
    return to_rvec(cv_q_trans(q))

def rvec2q(rvec):
    """Compute our quaternion to OpenCV Rodrigues rotation vector"""
    return cv_q_inv_trans(to_q(rvec))

def solve_pnp(object_points, image_points, pos=np.float32([0,0,0]), q=identity_quaternion()):
    """
    Solves the PnP problem for the given object-point image-point correspondences

    :param object_points: array-like, shape=(n, 3), object points in flightgoggles representation
    :param image_points: array-like, shape=(n, 3), screen space points corresponding to object points
    :return tuple, (bool, numpy.float32, numpy.quaternion), (Whether a solution was found, solved position, solved orientation)
    """
    t0 = time.time()
    cv_object_points = np.apply_along_axis(cv_point_trans, 1, object_points)
    image_points = np.apply_along_axis(screen_space, 1, image_points)
    image_points = np.apply_along_axis(polynomial_correction, 1, image_points)
    image_points = np.apply_along_axis(pixel_space, 1, image_points)
    rvec = q2rvec(q)
    tvec = cv_point_trans(q_rot(q.conjugate(), -pos))
    success, rvec, tvec = cv2.solvePnP(cv_object_points, np.float32(image_points), camera_matrix, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, distCoeffs=None, flags=0)
    if not success:  # Did not converge
        return False, None, None
    else:
        orientation_result = rvec2q(rvec).conjugate()
        position_result = -q_rot(orientation_result, cv_point_inv_trans(tvec))
        #print('solve_pnp time:', (time.time() - t0) * 1000, 'ms')
        return True, position_result, orientation_result
