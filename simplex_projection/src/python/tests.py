#boot up ipython and run
#%timeit 

import simplex_projection
import numpy as np
import time

m = 1000
n = 1000
print(f"Working with {m} x {n} arrays")
np.random.seed(0)
arr = np.random.randn(m, n)
u = np.empty((m, n), dtype=np.float64)
cumsum = np.empty((m, n), dtype=np.float64)

print("Running simplex_proj")
start = time.time()
simplex_projection.simplex_proj(arr, u, cumsum)
print(f'Completed in {time.time() - start} seconds')
