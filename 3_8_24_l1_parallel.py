import oars
import numpy as np
import cvxpy as cp # Only needed for the demo resolvent
import pandas as pd

class resolvent:
    '''L1 Norm Resolvent function'''
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    def __call__(self, x):
        u = x - self.data
        r = max(abs(u)-1, 0)*np.sign(u) + self.data
        #print(f"Data: {self.data}, x: {x}, u: {u}, r: {r}", flush=True)
        return r

    def __repr__(self):
        return "L1 norm resolvent"
        
def fullValue(data, x):
    '''Full value norm'''
    v = 0
    for d in data:
        v += np.abs(x - d)
    return v

class fullValueNorm:
    '''Full value norm'''
    def __init__(self, data):
        self.data = data
        

    def __call__(self, x):
        v = 0
        for d in self.data:
            v += np.abs(x - d)
        return v

    def __repr__(self):
        return "Full value norm"
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
    n = 6
    ldata = [np.array(i) for i in range(n)]
    ldata.append(ldata.copy())
    #fVal = fullValueNorm(ldata)
    Wmt, Lmt = oars.getMT(n)
    Wmax, Lmax = oars.getMaxConnect(n)
    Wpar, Lpar = oars.getParallel(n)
    Wmc, Lmc = oars.getMaxCut(n)
    pairs = [(Wmt, Lmt), (Wpar, Lpar), (Wmc, Lmc), (Wmax, Lmax)]
    pair_names = ["MT", "Parallel", "Max Cut", "MaxConnect"]
    alg_times = pd.DataFrame(columns=["MT", "Parallel", "MaxConnect", "MaxCut"])
    for j in range(10):
        print(f"Test {j+1}")
        ldata = [np.array(i) for i in np.random.rand(n)*100]
        ldata.append(ldata.copy())
        i = 0
        for W, L in pairs:                
            lres = [resolvent]*n
            print(f"Pair: {pair_names[i]}")
            #lx, lresults = oars.solve(n, ldata, lres, W, L, itrs=1000, objtol=1e-5, fval=fullValue, parallel=False, verbose=True)
            #Vartol 1e-5
            lx, lresults = oars.solve(n, ldata, lres, W, L, itrs=1000, vartol=1e-5, parallel=True, verbose=True)
            # Log alg_time from results[0]
            alg_times.loc[j, pair_names[i]] = lresults[0]['alg_time']
            i += 1
            #print("lx", lx)
            #print("lresults", lresults)

    # Save alg_times
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    alg_times.to_csv("alg_times"+timestamp+".csv")

    # Print mean for each algorithm
    print(alg_times.mean())
    # Test demo resolvent
    # n = 4
    # ddata = np.array([1, 2, 3, 40])
    # dres = [demo_resolvent]*n
    # dx, dresults = oars.solveMT(n, ddata, dres, itrs=50, vartol=1e-2, verbose=True)
    # print("dx", dx)
    # print("dresults", dresults)