#import oarsmpi
import mpi_test as oarsmpi
import numpy as np
from mpi4py import MPI
from scipy.special import lambertw

from time import time
np.set_printoptions(precision=3, suppress=True, linewidth=200)


def get_final_surv_prob(q, x):
    """
    Get the final probability of kill for each target.
    Inputs:
        q: (n,m) array of survival probabilities
        x: (n,m) array of weapon assignments
    """
    return np.prod(np.power(q, x), axis=1)


def generate_random_problem(n=5, m=3):
    """
    Generate a random problem.
    Inputs:
        n: number of targets
        m: number of weapon types
    """
    np.random.seed(1)
    q = np.random.rand(n,m)*.8 + .1 # Survival probability
    V = np.random.rand(n)*100 # Value of each target
    #W = np.random.randint(1,10,m) # Number of weapons of each type
    W = np.ones(m)
    return q, V, W

# From https://gist.github.com/mblondel/6f3b7aaad90606b98f71
def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def lambertw_prox(q, bv, y, v=1, verbose=False):

    d = q@y
    z = d - lambertw(bv*np.exp(d))
    lam = np.exp(z)
    x = y - lam*v*q
    return x.real

# WTA resolvent
class wtaResolvent:
    '''Resolvent function'''
    def __init__(self, data):
        self.data = data
        self.v0 = data['v0']
        # prob, w, y = buildWTAProb(data)
        # self.prob = prob
        # self.w = w
        # self.y = y
        self.q = np.log(self.data['QQ'])
        self.b = self.q@self.q
        self.bv = self.b*self.data['VV']
        self.shape = self.data['Q'].shape
        self.log = []

    def __call__(self, x):
        return self.data['VV']*np.exp(self.q@x)

    def prox(self, x):
        t = time()
        y = x[self.data['s'],:]
        log = {}
        log['start'] = time()
        w = lambertw_prox(self.q, self.bv, y, v=self.data['VV'])
        x[self.data['s'],:] = w
        log['end'] = time()
        log['time'] = log['end'] - log['start']
        log['value'] = fullValue(self.data, proj_full(x))
        self.log.append(log)
        return x

    def __repr__(self):
        return "wtaResolvent"

class simplexProj:
    '''Simplex projection resolvent for a subset of the full variable space
    Initialization:
    data: dictionary containing the following
    QQ: (tgts, wpns) array of survival probabilities
    VV: (tgts,) array of target values
    s: list of wpn indices to project
    '''
    def __init__(self, data):
        self.data = data
        self.shape = data['QQ'].shape
        self.q = data['QQ']
        self.v = data['VV']
        self.v0 = data['v0']
        self.s = data['s']
        self.log = []

    def __call__(self, y):
        return np.all(y >= 0) and np.all(y <= 1) and np.isclose(np.sum(y), 1)

    def prox(self, y):
        log = {}
        log['start'] = time()
        for i in self.s:
            y[:,i] = projection_simplex_sort(y[:,i])
        
        
        log['end'] = time()
        log['time'] = log['end'] - log['start']
        log['value'] = fullValue(self.data, y)
        self.log.append(log)
        return y

    def __repr__(self):
        return "Simplex Projection"

def fullValue(d, w):
    '''Get the full value of the problem'''
    return d['V']@get_final_surv_prob(d['Q'], w)


def smallSplit(Q, V, W, n, tgts, wpns, node_tgts, num_nodes_per_tgt, L=None):
    '''Generate the data for the splitting
    Inputs:
    n: number of nodes
    tgts: number of targets
    wpns: number of weapons
    node_tgts: list of lists of targets for each node
    num_nodes_per_tgt: list of number of nodes for each target
    L: (n,n) array of edge weights
    Outputs:
    data: list of dictionaries containing the following keys:
    QQ: (t, wpns) array of survival probabilities for the t targets in the node
    VV: (t,) array of target values for the t targets in the node
    WW: (wpns,) array of weapon counts
    v0: (len(node_tgts(i)), wpns) array of initial consensus parameters
    Lii: diagonal element of L
    Q: (tgts, wpns) array of survival probabilities for all targets
    V: (tgts,) array of target values
    i: node index'''
    if L is None:
        L = np.zeros((n,n))
    #Q, V, WW = generate_random_problem(tgts, wpns)

    m = (tgts, wpns)
    data = []
    print(node_tgts)
    for i in range(n):
        q = Q[node_tgts[i], :] # Only use the targets that are assigned to the node
        v = V[node_tgts[i]] # Only use the targets that are in the node
        #v = v/num_nodes_per_tgt # Divide the value by the number of nodes that have the target
        v0 = 1/tgts*np.ones(m) # Initial consensus parameter - equal number of weapons per tgt
        data.append({'QQ':q, 'VV':v, 'WW':W, 'v0':v0, 'Lii':L[i,i], 'Q':Q, 'V':V, 'i':i, 's':node_tgts[i]})

    #data.append({'Q':Q, 'V':V, 'WW':WW}) # Add the data for the full problem
    return data

def proj_full(x):
    proj_alg_x = x.copy()
    for i in range(x.shape[1]):
        proj_alg_x[:,i] = projection_simplex_sort(proj_alg_x[:,i])

    return proj_alg_x


# Test WTA resolvent
comm = MPI.COMM_WORLD
i = comm.Get_rank()
n = comm.Get_size()
itrs = 1000
gamma = 0.5
tgts = n-2 # Save one node for proj and one for convergence testing
wpns = tgts + 2
m = (tgts, wpns) # Dimension of the data
if i == 0:
    node_tgts = list(range(tgts))
    v0 = 1/tgts*np.ones((tgts, wpns))
    # Generate the data
    Q, V, WW = generate_random_problem(tgts, wpns)
    data = smallSplit(Q, V, WW, tgts, tgts, wpns, node_tgts, 1)
    
    proj_data = {'QQ':Q, 'VV':V, 'WW':WW, 's':range(wpns), 'Q':Q, 'V':V, 'v0':v0}
    fulldata = []
    for j in range(tgts):
        fulldata.append(data[j])
    fulldata.append(proj_data)
    fulldata.append({'Q':Q, 'V':V, 'WW':WW}) # For testing convergence
    
    resolvents = []
    for _ in range(tgts):
        resolvents.append(wtaResolvent)
    resolvents.append(simplexProj)
    #W, L = oars.getMaxConnect(n)
    #ldata.append(ldata.copy())
    #fVal = fullValueNorm(ldata)
    L, W = oarsmpi.getMT(n-1)
    Comms_Data = oarsmpi.requiredComms(L, W)
    # Broadcast L and W
    print("Node 0 broadcasting L and W", flush=True)
    comm.Bcast(L, root=0)
    comm.Bcast(W, root=0)
    # Distribute the data
    for j in range(1, n-1):
        data = fulldata[j]
        comm.send(data, dest=j, tag=44) # Data
        comm.send(resolvents[j], dest=j, tag=17) # Resolvent
        comm.send(Comms_Data[j], dest=j, tag=33) # Comms data
    # Run subproblems
    print("Node 0 running subproblem", flush=True)
    print("Comms data 0", Comms_Data[0], flush=True)
    w = oarsmpi.subproblem(i, fulldata[i], resolvents[i], W, L, Comms_Data[i], comm, gamma, itrs, vartol=1e-2, verbose=True)
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
    #data = np.array(m)
    data = comm.recv(source=0, tag=44)
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