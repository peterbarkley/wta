import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/peter.barkley/wta/simplex_projection/src/python')
import simplex_projection

import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
import time

wpns = 10
tgts = 6
m = tgts
n = wpns
np.random.seed(0)
arr = np.random.randn(m, n)
u = np.empty((m, n), dtype=np.float64)
cumsum = np.empty((m, n), dtype=np.float64)

print("Running simplex_proj in serial")
print(arr)
start = time.time()
simplex_projection.simplex_proj_serial(arr, u, cumsum)
print(f'Completed in {time.time() - start} seconds')
print(u)
print(arr)

# print("Running simplex_proj in parallel")
# start = time.time()
# simplex_projection.simplex_proj_parallel(arr, u, cumsum)
# print(f'Completed in {time.time() - start} seconds')
