#Test projection at beginning
import mpi_test as oarsmpi
import numpy as np
from mpi4py import MPI
from scipy.special import lambertw
import json
from time import time
np.set_printoptions(precision=3, suppress=True, linewidth=200)
#from numba import jit, prange
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/peter.barkley/wta/simplex_projection/src/python')
import simplex_projection
 
 # From https://gist.github.com/mblondel/6f3b7aaad90606b98f71

# @jit(nopython=True, parallel=True, cache=True)
# def projection_simplex_sort(v):
#     n_features = v.shape[0]
#     ind = np.arange(n_features) + 1
#     for i in prange(v.shape[1]):
#         u = np.sort(v[:,i])[::-1]
#         cssv = u.cumsum() - 1 # or - z for [0, z] simplex
#         cond = u - cssv / ind > 0
#         rho = ind[cond][-1]
#         theta = cssv[cond][-1] / float(rho)
#         v[:,i] = np.maximum(v[:,i] - theta, 0)
#     return 0


def get_final_surv_prob(q, x):
    """Get the final survival probability for each target.
    Inputs:
        q: (wpns, tgts) array of survival probabilities
        x: (wpns, tgts) array of weapon assignments
    """
    return np.prod(np.power(q, x), axis=0)


def generate_random_problem(tgts=5, wpns=3):
    """
    Generate a random problem.
    Inputs:
        tgts: number of targets
        wpns: number of weapon types
    """
    np.random.seed(1)
    q = np.random.rand(wpns, tgts)*.5 + .5 # Survival probability
    V = np.random.rand(tgts)*100 # Value of each target
    #W = np.random.randint(1,10,m) # Number of weapons of each type
    W = np.ones(wpns)
    return q, V, W



def lambertw_prox(q, bv, y, v=1, verbose=False):
    """Compute the prox_f(y) for f(x) = bv*e^{q@x}"""
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
        i = self.data['i']
        y = x[:, i]
        log = {}
        log['start'] = time()
        x[:, i] = lambertw_prox(self.q, self.bv, y, v=self.data['VV'])
        log['end'] = time()
        log['time'] = log['end'] - log['start']
        #log['value'] = fullValue(self.data, proj_full(x))
        self.log.append(log)
        return x

    def __repr__(self):
        return "wtaResolvent"

class simplexProj:
    '''Simplex projection resolvent for a subset of the full variable space
    Initialization:
    data: dictionary containing the following
    QQ: (wpns, tgts) array of survival probabilities
    VV: (tgts,) array of target values
    v0: (wpns, tgts) array of initial consensus parameters
    '''
    def __init__(self, data):
        self.data = data
        self.shape = data['QQ'].shape
        self.q = data['QQ']
        self.v = data['VV']
        self.v0 = data['v0']
        self.log = []
        self.cs = np.empty(self.shape, dtype=np.float64) # cumsum array
        self.u = np.empty(self.shape, dtype=np.float64) # output array

    def __call__(self, y):
        return np.all(y >= 0) and np.all(y <= 1) and np.isclose(np.sum(y), 1)

    def prox(self, y):
        log = {}
        log['start'] = time()
        #projection_simplex_sort(y)  
        simplex_projection.simplex_proj_parallel(y, self.u, self.cs)
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


def smallSplit(Q, V, W):
    '''Generate the data for the splitting
    Inputs:
    Q: (wpns, tgts) array of survival probabilities
    V: (tgts,) array of target values
    W: (wpns,) array of weapon counts

    Outputs:
    data: list of dictionaries containing the following keys:
    QQ: (wpns) array of survival probabilities for the target in the node
    VV: (1) target value for the target in the node
    WW: (wpns,) array of weapon counts
    v0: (tgts, wpns) array of initial consensus parameters
    Q: (wpns, tgts) array of survival probabilities for all targets
    V: (tgts,) array of target values
    i: node index'''
    
    tgts = Q.shape[1]
    data = []
    
    v0 = 1/tgts*np.ones(Q.shape) # Initial consensus parameter - equal number of weapons per tgt
    for i in range(tgts):
        data.append({'QQ':Q[:,i], 'VV':V[i], 'WW':W, 'v0':v0, 'Q':Q, 'V':V, 'i':i, 's':i})

    return data

# Test WTA resolvent
def test_wta(L, W, itrs=1000, gamma=0.5, title="WTA"):
    nodes = L.shape[0]
    tgts = nodes-1 # Save one node for proj and one for convergence testing
    wpns = tgts + 40
    m = (wpns, tgts) # Dimension of the data
    if i == 0:
        # Generate the data
        Q, V, WW = generate_random_problem(tgts, wpns)
        data = smallSplit(Q, V, WW)
        v0 = 1/tgts*np.ones(m)
        proj_data = {'QQ':Q, 'VV':V, 'WW':WW, 's':range(wpns), 'Q':Q, 'V':V, 'v0':v0}
        fulldata = []
        for j in range(tgts):
            fulldata.append(data[j])
        fulldata.append(proj_data)
        fulldata.append({'Q':Q, 'V':V, 'WW':WW}) # For testing convergence
        
        # Generate the resolvents
        resolvents = []
        for _ in range(tgts):
            resolvents.append(wtaResolvent)
        resolvents.append(simplexProj)
        Comms_Data = oarsmpi.requiredComms(L, W)

        # Distribute the data
        for j in range(1, nodes):
            #print("Node 0 sending data to node", j, flush=True)
            data = fulldata[j]
            comm.send(data, dest=j, tag=44) # Data
            comm.send(resolvents[j], dest=j, tag=17) # Resolvent
            comm.send(Comms_Data[j], dest=j, tag=33) # Comms data
        # Run subproblems
        print("Node 0 running subproblem", flush=True)
        #print("Comms data 0", Comms_Data[0], flush=True)
        t = time()
        w, log = oarsmpi.subproblem(i, fulldata[i], resolvents[i], W, L, Comms_Data[i], comm, gamma, itrs, vartol=1e-5, verbose=True)
        print("Time", time() - t)
        
        #timestamp = time()
        with open('logs_wta'+str(i)+'_'+title+'.json', 'w') as f:
            json.dump(log, f)
        #w = np.array(m)
        results = []
        results.append({'w':w})
        w_i = np.zeros(w.shape)
        for k in range(1, n-1):
            comm.Recv(w_i, source=k, tag=0)
            results.append({'w':w_i})
            w += w_i
        proj_w = w_i
        #print(w, proj_w, flush=True)
        print("alg val", fullValue(fulldata[-1], proj_w))
        t = time()
        true_p, true_x = oarsmpi.wta(Q.T, V, WW, integer=False, verbose=True, mosek_params={'MSK_DPAR_OPTIMIZER_MAX_TIME': 30.0,})
        true_time = time() - t
        print("true val", true_p)
        print("mosek time", true_time)
    elif i < n-1:
        # Receive L and W
        #print(f"Node {i} receiving L and W", flush=True)
        #L = np.zeros((n-1,n-1))
        #W = np.zeros((n-1,n-1))
        #comm.Bcast(L, root=0)
        #comm.Bcast(W, root=0)
        #print(f"Node {i} received L and W", flush=True)
        # Receive the data
        #data = np.array(m)
        data = comm.recv(source=0, tag=44)
        res = comm.recv(source=0, tag=17)
        comms = comm.recv(source=0, tag=33)
        # Run the subproblem
        #print(f"Node {i} running subproblem", flush=True)
        w, log = oarsmpi.subproblem(i, data, res, W, L, comms, comm, gamma, itrs, vartol=1e-2, verbose=True)
        #timestamp = time()
        with open('logs_wta'+str(i)+'_'+title+'.json', 'w') as f:
            json.dump(log, f)
        #w = np.array(i)
        comm.Send(w, dest=0, tag=0)
    elif i == n-1:
        #L = np.zeros((n-1,n-1))
        #W = np.zeros((n-1,n-1))
        #comm.Bcast(L, root=0)
        #comm.Bcast(W, root=0)
        oarsmpi.evaluate(m, comm, vartol=1e-5, itrs=itrs) 


comm = MPI.COMM_WORLD
i = comm.Get_rank()
n = comm.Get_size()
nodes = n-1
# if n - 1 < nodes:
#     print("Not enough nodes for the given problem")
#     exit()

LW_titles = ['full', 'MT', 'block2', 'SLEM', 'Ryu']
Lfull, Wfull = oarsmpi.getMaxConnect(nodes)
Lmt, Wmt = oarsmpi.getMT(nodes)
m = nodes//2
L = np.zeros((nodes, nodes))
L[m:, :m] = 2/m
W = np.eye(nodes)*2 - L - L.T
Ws = np.eye(nodes)*2 - np.ones((nodes, nodes))*(2/nodes)

Lr, Wr = oarsmpi.getRyu(nodes)

LW_list = [(Lfull, Wfull), (Lmt, Wmt), (L, W)] #, (L, Ws), (Lr, Wr)]
for j, (L, W) in enumerate(LW_list):
    if i == 0:
        print("Running WTA with L and W on "+LW_titles[j]+'nodes'+str(n), flush=True)
    test_wta(L, W, itrs=1000, gamma=0.5, title=LW_titles[j])