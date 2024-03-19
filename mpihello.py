from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
name=MPI.Get_processor_name()

i = comm.Get_rank()
n = comm.Get_size()
s = 1
nn = 10
buffer = np.zeros(s, dtype=np.float64)
if i == 0:
    print(f"Hello from {name}! I am the master node.")
    for v in range(nn):
        buffer = np.array([v], dtype=np.float64)
        comm.Send(buffer, dest=1, tag=11)
    for v in range(nn):
        buffer = np.zeros(s, dtype=np.float64)
        comm.Send(buffer, dest=1, tag=11)
else:
    last = np.zeros(s, dtype=np.float64)
    for v in range(nn*2):
        comm.Recv(buffer, source=0, tag=11)
        w = buffer.copy()
        print(f"Hello from {name}! I am the worker node {i}. I received {w}.")
        if np.array_equal(w, last):
            print(f"Hello from {name}! I am the worker node {i}. I received the same value twice.")
        last = w