
import numpy as np
from numba import jit, prange
 
@jit(nopython=True, parallel=True)
def projection_simplex_sort(v):
    n_features = v.shape[0]
    ind = np.arange(n_features) + 1
    for i in prange(y.shape[1]):
        u = sorted(y[:,i], reverse=True)
        cssv = np.cumsum(u) - 1 # or - z for [0, z] simplex
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        y[:,i] = np.maximum(y[:,i] - theta, 0)
    return y

y = np.random.rand(5, 6)
print(y)
print(projection_simplex_sort(y))