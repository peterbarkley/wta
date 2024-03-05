import oars
import numpy as np
import cvxpy as cp # Only needed for the demo resolvent

def theoretical_min(data, L):
    '''
    Find argmin \sum_i (x - data[i])^2/(1-L[i])^2
    '''
    x = cp.Variable(1)
    obj = cp.sum(cp.power(x - data, 2)/(1-L)**2)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    return x.value

class square_resolvent:
    '''Demo Resolvent function'''
    def __init__(self, data):
        self.data = data['x']
        self.lii = data['lii']
        self.shape = 1
        # Define the problem using CVXPY
        self.x = cp.Variable(self.shape) # variable
        self.y = cp.Parameter(self.shape) # resolvent parameter
        # self.f would be the function to minimize
        self.f = cp.sum_squares(self.x - self.data)
        self.obj = cp.Minimize(self.f + 0.5*cp.sum_squares((1-self.lii)*self.x - self.y))
        self.prob = cp.Problem(self.obj)

    def __call__(self, x):
        self.y.value = x
        self.prob.solve()
        
        #print(f"Data: {self.data}, x: {self.x.value}, y: {x}", flush=True)
        return self.x.value

    def __repr__(self):
        return "Demo resolvent"
        
if __name__ == "__main__":
    # Test L1 resolvent
    n = 4
    data = [{'x': -2, 'lii': 0}, {'x': -1, 'lii': 0}, {'x': 1, 'lii': 0.25}, {'x': 2, 'lii': 0.25}]
    print(np.mean([d['x'] for d in data]))
    resolvents = [square_resolvent]*n
    #fVal = fullValueNorm(ldata)
    L = np.array([[0,0,0,0], [1,0,0,0], [0,1,0.25,0], [1,0,0.25,0.25]])
    W = np.array([[1,-1,0,0], [-1,2,-1,0], [0,-1,1.5,-0.5], [0,0,-0.5,0.5]])
    
    print(theoretical_min([d['x'] for d in data], np.diag(L)))
    itrs = 1000
    alg_x, results = oars.solve(n, data, resolvents, W, L, itrs=itrs, vartol=1e-5, parallel=False, verbose=True)   
    print("alg_x", alg_x)

    # vs MT
    resolvents = [square_resolvent]*n
    data = [{'x': -2, 'lii': 0}, {'x': -1, 'lii': 0}, {'x': 1, 'lii': 0}, {'x': 2, 'lii': 0}]    
    mt_x, mt_results = oars.solveMT(n, data, resolvents, itrs=itrs, vartol=1e-6, parallel=False, verbose=True)
    print("mt_x", mt_x)
    # Test demo resolvent
    # n = 4
    # ddata = np.array([1, 2, 3, 40])
    # dres = [demo_resolvent]*n
    # dx, dresults = oars.solveMT(n, ddata, dres, itrs=50, vartol=1e-2, verbose=True)
    # print("dx", dx)
    # print("dresults", dresults)