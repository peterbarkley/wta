#import oarsmpi
import mpi_test as oarsmpi
import numpy as np
from mpi4py import MPI

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

#if __name__ == "__main__":
# Test L1 resolvent
comm = MPI.COMM_WORLD
i = comm.Get_rank()
n = comm.Get_size()
itrs = 1000
gamma = 0.5
m = 1 # Dimension of the data
if i == 0:
    ldata = [np.array(i) for i in range(n)]
    #ldata.append(ldata.copy())
    lres = [resolvent]*n
    #fVal = fullValueNorm(ldata)
    L, W = oarsmpi.getMT(n-1)
    Comms_Data = oarsmpi.requiredComms(L, W)
    # Broadcast L and W
    print("Node 0 broadcasting L and W", flush=True)
    comm.Bcast(L, root=0)
    comm.Bcast(W, root=0)
    # Distribute the data
    for j in range(1, n-1):
        data = ldata[j]
        comm.Send(data, dest=j, tag=44) # Data
        comm.send(lres[j], dest=j, tag=17) # Resolvent
        comm.send(Comms_Data[j], dest=j, tag=33) # Comms data
    # Run subproblems
    print("Node 0 running subproblem", flush=True)
    print("Comms data 0", Comms_Data[0], flush=True)
    w = oarsmpi.subproblem(i, ldata[i], lres[i], W, L, Comms_Data[i], comm, gamma, itrs, vartol=1e-2, verbose=True)
    #w = np.array(m)
    results = []
    results.append({'w':w})
    w_i = np.zeros(w.shape)
    for k in range(1, n-1):
        comm.Recv(w_i, source=k, tag=0)
        results.append({'w':w_i})
        w += w_i
    w = w/(n-1)
    print(w)
elif i < n-1:
    # Receive L and W
    #print(f"Node {i} receiving L and W", flush=True)
    L = np.zeros((n-1,n-1))
    W = np.zeros((n-1,n-1))
    comm.Bcast(L, root=0)
    comm.Bcast(W, root=0)
    #print(f"Node {i} received L and W", flush=True)
    # Receive the data
    data = np.array(m)
    comm.Recv(data, source=0, tag=44)
    res = comm.recv(source=0, tag=17)
    comms = comm.recv(source=0, tag=33)
    # Run the subproblem
    print(f"Node {i} running subproblem", flush=True)
    w = oarsmpi.subproblem(i, data, res, W, L, comms, comm, gamma, itrs, vartol=1e-2, verbose=True)
    #w = np.array(i)
    comm.Send(w, dest=0, tag=0)
else:
    L = np.zeros((n-1,n-1))
    W = np.zeros((n-1,n-1))
    comm.Bcast(L, root=0)
    comm.Bcast(W, root=0)
    oarsmpi.evaluate(m, comm, vartol=1e-2, itrs=itrs) 