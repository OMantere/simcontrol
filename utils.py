import numpy as np
from airsim.types import *

# Unreal coordinates are relative to player spawn
# Player spawn is hardcoded for now
pp = np.array((2660, 1690, 2050))

scale = 100 # Unreal scale factor
x_corr = 1.5 # Something to do with the gate pivot, dont ask
z_corr = 7.0 # Gate height correction

def airsim_tf(x):
    """From unreal to airsim coordinates"""
    y = (x - pp) / scale
    y[2] = -y[2]
    return y


def unreal_tf(x):
    """From airsim to unreal coordinates"""
    x[2] = -x[2]
    y = x * scale + pp
    return y


class Quaternion(np.ndarray):
    @classmethod
    def from_airsim(cls, q):
        """To be initialized with an AirSim quaternion"""
        obj = cls([q.x_val, q.y_val, q.z_val, q.w_val])
        obj.q = q
        return obj

    def yaw(self):
        """Rotation around the z-axis in radians"""
        siny_cosp = 2.0 * (self.q.w_val * self.q.z_val + self.q.x_val * self.q.y_val)
        cosy_cosp = 1.0 - 2.0 * (self.q.y_val * self.q.y_val + self.q.z_val * self.q.z_val)
        return np.arctan2(siny_cosp, cosy_cosp)


class Position(np.ndarray):
    @classmethod
    def from_airsim(cls, pos):
        """To be initialized with an AirSim position. We should keep internal 
        representation in AirSim space"""
        obj = cls([pos.x_val, pos.y_val, pos.z_val])
        obj.pos = pos
        return obj

    def unreal_position(self):
        """Position vector in Unreal space"""
        return unreal_tf(self.pos)


def Gate(object):

    def __init__(self, pos, q):
        self.pos = pos
        self.q = q

    @classmethod
    def from_api(cls, client, idx):
        gate_name = 'gate_%d' % (idx+1)
        pose = client.simGetObjectPose(gate_name)
        if np.isnan(pose.position.x_val):
            print(gate_name, "was nan")
            return cls(Position(), Quaternion())
        else:
            pos = pose.position
            pos.x_val -= np.cos(q_yaw(pose.orientation)) * x_corr
            pos.z_val -= z_corr
            obj = cls(Position.from_airsim(pos), Quaternion.from_airsim(pose.orientation))
            return obj

    @classmethod
    def from_unreal_position(cls, x, y, z): # Because we really dont care about gate orientation
        return cls(Position([x, y, z]), Quaternion())