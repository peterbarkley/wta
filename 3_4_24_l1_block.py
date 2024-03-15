import oars
import numpy as np
import cvxpy as cp # Only needed for the demo resolvent

class resolvent:
    '''L1 Norm Resolvent function'''
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    def __call__(self, x):
        return sum(abs(x - self.data))

    def prox(self, x):
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
      
if __name__ == "__main__":
    # Test L1 resolvent
    n = 10
    ldata = [np.array(i) for i in range(n)]
    ldata.append(ldata.copy())
    #fVal = fullValueNorm(ldata)
    Lf = np.array([[-0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 0.573, -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 0.573,  0.573, -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 0.573,  0.573,  0.573, -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 0.281,  0.281,  0.281,  0.281, -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [-0.   , -0.   , -0.   , -0.   ,  0.877, -0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   , -0.   ,  0.281, -0.   , -0.   , -0.   , -0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   , -0.   ,  0.281,  0.573, -0.   , -0.   , -0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   , -0.   ,  0.281,  0.573,  0.573, -0.   , -0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   , -0.   ,  0.281,  0.573,  0.573,  0.573, -0.   ]])
    Wf = np.array([[ 1.208, -0.309, -0.309, -0.309, -0.281, -0.   , -0.   , -0.   , -0.   , -0.   ],
       [-0.309,  1.208, -0.309, -0.309, -0.281, -0.   , -0.   , -0.   , -0.   , -0.   ],
       [-0.309, -0.309,  1.208, -0.309, -0.281, -0.   , -0.   , -0.   , -0.   , -0.   ],
       [-0.309, -0.309, -0.309,  1.208, -0.281, -0.   , -0.   , -0.   , -0.   , -0.   ],
       [-0.281, -0.281, -0.281, -0.281,  2.   , -0.877, -0.   , -0.   , -0.   , -0.   ],
       [-0.   , -0.   , -0.   , -0.   , -0.877,  2.   , -0.281, -0.281, -0.281, -0.281],
       [-0.   , -0.   , -0.   , -0.   , -0.   , -0.281,  1.208, -0.309, -0.309, -0.309],
       [-0.   , -0.   , -0.   , -0.   , -0.   , -0.281, -0.309,  1.208, -0.309, -0.309],
       [-0.   , -0.   , -0.   , -0.   , -0.   , -0.281, -0.309, -0.309,  1.208, -0.309],
       [-0.   , -0.   , -0.   , -0.   , -0.   , -0.281, -0.309, -0.309, -0.309,  1.208]])
    L_pen = np.array([[-0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 0.27 ,  1.152, -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 1.153,  0.547,  0.3  , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 0.577,  0.301,  0.278,  0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   ,  0.845, -0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.   , -0.   , -0.   , -0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.278,  0.3  , -0.   , -0.   , -0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.301,  0.547,  1.152, -0.   , -0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.577,  1.153,  0.27 ,  0.   , -0.   ]])
    W_pen = np.array([[ 0.619, -0.619, -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [-0.619,  1.683, -1.064, -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ],
       [-0.   , -1.064,  1.92 , -0.501, -0.355, -0.   , -0.   , -0.   , -0.   , -0.   ],
       [-0.   , -0.   , -0.501,  1.015, -0.513, -0.   , -0.   , -0.   , -0.   , -0.   ],
       [-0.   , -0.   , -0.355, -0.513,  1.713, -0.845, -0.   , -0.   , -0.   , -0.   ],
       [-0.   , -0.   , -0.   , -0.   , -0.845,  1.713, -0.513, -0.355, -0.   , -0.   ],
       [-0.   , -0.   , -0.   , -0.   , -0.   , -0.513,  1.015, -0.501, -0.   , -0.   ],
       [-0.   , -0.   , -0.   , -0.   , -0.   , -0.355, -0.501,  1.92 , -1.064, -0.   ],
       [-0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -1.064,  1.683, -0.619],
       [-0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.619,  0.619]])
    Wmt, Lmt = oars.getMT(n)
    Wmax, Lmax = oars.getMaxConnect(n)
    pairs = [(Wf, Lf), (W_pen, L_pen), (Wmt, Lmt), (Wmax, Lmax)]
    pair_names = ["Wf, Lf", "W_pen, L_pen", "Wmt, Lmt", "Wmax, Lmax"]
    i = 0
    for W, L in pairs:                
        lres = [resolvent]*n
        print(f"Pair: {pair_names[i]}")
        i += 1
        #lx, lresults = oars.solve(n, ldata, lres, W, L, itrs=1000, objtol=1e-5, fval=fullValue, parallel=False, verbose=True)
        #Vartol 1e-5
        lx, lresults = oars.solve(n, ldata, lres, W, L, itrs=1000, vartol=1e-5, parallel=False, verbose=True)
        print("lx", lx)
        #print("lresults", lresults)

    # Test demo resolvent
    # n = 4
    # ddata = np.array([1, 2, 3, 40])
    # dres = [demo_resolvent]*n
    # dx, dresults = oars.solveMT(n, ddata, dres, itrs=50, vartol=1e-2, verbose=True)
    # print("dx", dx)
    # print("dresults", dresults)