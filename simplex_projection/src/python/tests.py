#boot up ipython and run
#%timeit 

import simplex_projection
import numpy as np
import time

m = 1000
n = 10000
print(f"Working with {m} x {n} arrays")
np.random.seed(0)
arr = np.random.randn(m, n)
u = np.empty((m, n), dtype=np.float64)
cumsum = np.empty((m, n), dtype=np.float64)

print("Running simplex_proj in serial")
start = time.time()
simplex_projection.simplex_proj_serial(arr, u, cumsum)
print(f'Completed in {time.time() - start} seconds')

print("Running simplex_proj in parallel")
start = time.time()
simplex_projection.simplex_proj_parallel(arr, u, cumsum)
print(f'Completed in {time.time() - start} seconds')
