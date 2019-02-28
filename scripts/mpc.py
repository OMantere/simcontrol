import time
import numpy as np
import osqp
import numpy as np
import scipy as sp
import scipy.sparse as sparse


class MPC(object):
    def __init__(self, dynamics):
        self.dynamics = dynamics
	# Objective function
	self.Q = sparse.diags([10., 0.])
	self.R = 0.1 * sparse.eye(1)

    def solve(self, reference, x0):
	N = 25 # Prediction horizon

        Ad, Bd = self.dynamics.linearize()
        nx = self.dynamics.dim_x
        nu = self.dynamics.dim_u

	# - quadratic objective
	P = sparse.block_diag([sparse.kron(sparse.eye(N), self.Q), self.Q,
			       sparse.kron(sparse.eye(N), self.R)]).tocsc()
	# - linear objective
	q = np.hstack([np.kron(np.ones(N), -self.Q.dot(reference)), -self.Q.dot(reference),
		       np.zeros(N*nu)])
	# - linear dynamics
	Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
	Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
	Aeq = sparse.hstack([Ax, Bu])
	leq = np.hstack([-x0, np.zeros(N*nx)])
	ueq = leq
	# - input and state constraints
	umin = np.ones(nu) * -1.0
	umax = np.ones(nu) * 1.0
	xmin = np.ones(nx) * -100.0
	xmax = np.ones(nx) * 100.0
	Aineq = sparse.eye((N+1)*nx + N*nu)
	lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
	uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
	# - OSQP constraints
	A = sparse.vstack([Aeq, Aineq]).tocsc()
	l = np.hstack([leq, lineq])
	u = np.hstack([ueq, uineq])

	# Create an OSQP object
	prob = osqp.OSQP()

	# Setup workspace
	prob.setup(P, q, A, l, u, warm_start=True, polish=False,
		eps_abs=1e-1, eps_rel=1e-1)

	res = prob.solve()

	x = res.x[:N*nx].reshape(N, nx)
	controls = res.x[-N*nu:]
	return x, controls


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
    mpc = MPC(Dynamics())

    reference = np.array([1., 0.])
    x_0 = np.zeros(2)
    x, u = mpc.solve(reference, x_0)

    from matplotlib import pyplot
    figure = pyplot.figure()
    axis = figure.add_subplot(211)
    axis.plot(np.arange(u.shape[0]), u)
    axis = figure.add_subplot(212)
    axis.plot(np.arange(u.shape[0]), x[:, 0])
    pyplot.show()


