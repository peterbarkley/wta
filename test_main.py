import oars
import numpy as np
import cvxpy as cp # Only needed for the demo resolvent

class resolvent:
    '''L1 Norm Resolvent function'''
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    def __call__(self, x):
        u = x - self.data
        return max(abs(u)-1, 0)*np.sign(u) + self.data

    def __repr__(self):
        return "L1 norm resolvent"
        

class demo_resolvent:
    '''Demo Resolvent function'''
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        # Define the problem using CVXPY
        self.x = cp.Variable(data.shape) # variable
        self.y = cp.Parameter(data.shape) # resolvent parameter
        # self.f would be the function to minimize
        self.f = cp.huber(self.x - self.data, 1)
        self.obj = cp.Minimize(self.f + cp.sum_squares(self.x - self.y))
        self.prob = cp.Problem(self.obj)

    def __call__(self, x):
        self.y.value = x
        self.prob.solve()
        return self.x.value

    def __repr__(self):
        return "Demo resolvent"
        
if __name__ == "__main__":
    # Test L1 resolvent
    n = 4
    ldata = np.array([1, 2, 3, 4])
    lres = [resolvent]*n
    lx, lresults = oars.solveMT(n, ldata, lres, itrs=50, verbose=True)
    print("lx", lx)
    print("lresults", lresults)

    # Test demo resolvent
    n = 4
    ddata = np.array([1, 2, 3, 40])
    dres = [demo_resolvent]*n
    dx, dresults = oars.solveMT(n, ddata, dres, itrs=50, verbose=True)
    print("dx", dx)
    print("dresults", dresults)