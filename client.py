import airsim
import numpy as np
import msgpackrpc

from utils import Gate


class Simulator(object):
    def __init__(self):
        self.client = airsim.VehicleClient()
        try:
            self.client.confirmConnection()
        except msgpackrpc.error.TransportError:
            raise Exception("Can't connect to Unreal Engine") from None

    def get_gates(self, num):
        self.gates = []
        self.num_gates = num
        for i in range(gate_n):
            self.gates.append(Gate.from_api(self.client, i))

    def track_position_matrix(self):
        p = np.zeros(self.num_gates + 1, 3) # Include the start as end for the track
        for i in range(self.num_gates + 1):
            p[i, :] = self.gates[i % num_gates].pos
