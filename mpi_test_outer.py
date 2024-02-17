#import oarsmpi
import mpi_test as oarsmpi
import numpy as np
import cvxpy as cp # Only needed for the demo resolvent
from mpi4py import MPI

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
        
#if __name__ == "__main__":
# Test L1 resolvent
comm = MPI.COMM_WORLD
i = comm.Get_rank()
n = comm.Get_size()
itrs = 11
gamma = 0.5
if i == 0:
    ldata = [np.array(i) for i in range(n)]
    #ldata.append(ldata.copy())
    lres = [resolvent]*n
    #fVal = fullValueNorm(ldata)
    L, W = oarsmpi.getMT(n)
    Comms_Data = oarsmpi.requiredComms(L, W)
    # Broadcast L and W
    print("Node 0 broadcasting L and W", flush=True)
    comm.Bcast(L, root=0)
    comm.Bcast(W, root=0)
    # Distribute the data
    for j in range(1, n):
        data = ldata[j]
        comm.Send(data, dest=j, tag=44)
        comm.send(lres[j], dest=j, tag=17)
        comm.send(Comms_Data[j], dest=j, tag=33)
    # Run subproblems
    print("Node 0 running subproblem", flush=True)
    print("Comms data 0", Comms_Data[0], flush=True)
    w = oarsmpi.subproblem(i, ldata[i], lres[i], W, L, Comms_Data[i], comm, gamma, itrs, verbose=True)
    #w = np.array(i)
    results = []
    results.append({'w':w})
    w_i = np.zeros(w.shape)
    for k in range(1, n):
        comm.Recv(w_i, source=k, tag=0)
        results.append({'w':w_i})
        w += w_i
    w = w/n
    print(w)
else:
    # Receive L and W
    print(f"Node {i} receiving L and W", flush=True)
    L = np.zeros((n,n))
    W = np.zeros((n,n))
    comm.Bcast(L, root=0)
    comm.Bcast(W, root=0)
    print(f"Node {i} received L and W", flush=True)
    # Receive the data
    data = np.array(0)
    comm.Recv(data, source=0, tag=44)
    res = comm.recv(source=0, tag=17)
    comms = comm.recv(source=0, tag=33)
    # Run the subproblem
    print(f"Node {i} running subproblem", flush=True)
    w = oarsmpi.subproblem(i, data, res, W, L, comms, comm, gamma, itrs, verbose=True)
    #w = np.array(i)
    comm.Send(w, dest=0, tag=0)