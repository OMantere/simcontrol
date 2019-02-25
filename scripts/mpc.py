import time
import cvxpy
import numpy as np
import cvxpy as cp
from cvxpy import Variable

class MPC(object):
    def __init__(self, dynamics, Q, R):
        self.dynamics = dynamics
        self.Q = Q
        self.R = R

    def solve(self, trajectory, x_0):
        A, B = self.dynamics.linearize()
        n = self.dynamics.dim_x
        m = self.dynamics.dim_u

        T = trajectory.shape[1]
        x = Variable((n, T+1))
        u = Variable((m, T))

        constraints = []
        cost = 0.0
        for t in range(T):
            constraints += [x[:, t+1] == A * x[:, t] + B * u[:, t]]

            x_diff = cp.abs(x[:, t] - trajectory[:, t])[0]
            cost += cp.sum_squares(x_diff) + cp.sum_squares(u[:, t])
            # cost += cp.quad_form(x_diff[:, None], self.Q)
            # cost += cp.quad_form(u[:, t][:, None], self.R)

        constraints += [cp.abs(u) <= 1.0]
        problem = cp.Problem(cp.Minimize(cost), constraints)

        x.value = np.zeros((n, T+1))
        for i in range(5):
            start = time.time()
            u.value = np.random.randn(u.shape[0], u.shape[1])
            result = problem.solve(warm_start=True, solver=cp.OSQP,
                    eps_abs=1e-1, eps_rel=1e-1)
            print("Took {} to solve, cost: {}".format(time.time() - start, result))


        return u, x, problem.value

if __name__ == "__main__":
    class Dynamics(object):
        def __init__(self):
            self.dim_x = 2
            self.dim_u = 1
            h = 0.1
            self.A = np.array([[1., h], [0., 1.]])
            self.B = np.array([[0.],[h * 2]])

        def linearize(self):
            return self.A, self.B

    Q = np.zeros((2, 2))
    Q[0, 0] = 1.0 # Only care about position.
    R = np.eye(1) * 0.1
    mpc = MPC(Dynamics(), Q, R)

    trajectory = np.stack([np.linspace(0, 10, 20), np.zeros((20,))])
    x_0 = np.zeros((2, 1))
    mpc.solve(trajectory, x_0)

