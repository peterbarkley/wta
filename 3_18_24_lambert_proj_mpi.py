import oars
import wta
import numpy as np
from time import time
from collections import defaultdict, deque

import json
from scipy.special import lambertw

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
    return d['V']@wta.get_final_surv_prob(d['Q'], w)

def generateSplitData(n, tgts, wpns, node_tgts, num_nodes_per_tgt, L):
    '''Generate the data for the splitting'''
    Q, V, WW = wta.generate_random_problem(tgts, wpns)
    m = (tgts, wpns)
    data = []
    for i in range(n):
        q = np.ones(m)
        q[node_tgts[i]] = Q[node_tgts[i]] # Only use the targets that are in the node
        v = np.zeros(tgts)
        v[node_tgts[i]] = V[node_tgts[i]] # Only use the targets that are in the node
        v = v/num_nodes_per_tgt # Divide the value by the number of nodes that have the target
        v0 = 1/tgts*np.array(WW)*np.ones(m) # Initial consensus parameter - equal number of weapons per tgt
        data.append({'QQ':q, 'VV':v, 'WW':WW, 'v0':v0, 'Lii':L[i,i], 'Q':Q, 'V':V})

    data.append({'Q':Q, 'V':V, 'WW':WW}) # Add the data for the full problem
    return data

def smallSplit(n, tgts, wpns, node_tgts, num_nodes_per_tgt, L=None):
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
    Q, V, WW = generate_random_problem(tgts, wpns)
    WW = np.ones(wpns)
    m = (tgts, wpns)
    data = []
    print(node_tgts)
    for i in range(n):
        q = Q[node_tgts[i], :] # Only use the targets that are assigned to the node
        v = V[node_tgts[i]] # Only use the targets that are in the node
        #v = v/num_nodes_per_tgt # Divide the value by the number of nodes that have the target
        v0 = 1/tgts*np.ones(m) # Initial consensus parameter - equal number of weapons per tgt
        data.append({'QQ':q, 'VV':v, 'WW':WW, 'v0':v0, 'Lii':L[i,i], 'Q':Q, 'V':V, 'i':i, 's':node_tgts[i]})

    #data.append({'Q':Q, 'V':V, 'WW':WW}) # Add the data for the full problem
    return data

def generate_random_problem(n=5, m=3):
    """
    Generate a random problem.
    Inputs:
        n: number of targets
        m: number of weapon types
    """
    np.random.seed(1)
    q = np.random.rand(n,m)*.5 + .5 # Survival probability
    V = np.random.rand(n)*100 # Value of each target
    #W = np.random.randint(1,10,m) # Number of weapons of each type
    W = np.ones(m)
    return q, V, W


def proj_full(x):
    proj_alg_x = x.copy()
    for i in range(x.shape[1]):
        proj_alg_x[:,i] = projection_simplex_sort(proj_alg_x[:,i])

    return proj_alg_x

def get_rand_solution_value(Q, V, tgts, wpns, integer=False):
        # Generate random solution for comparison
    # Generate random solution
    y = np.random.rand(tgts, wpns)
    # Set the max value in each column to 1, the rest to 0
    if integer:
        x = np.zeros(y.shape)
        max_index = np.argmax(y, axis=0)
        for j in range(wpns):
            x[max_index[j],j] = 1
        y = x

    # Get the value of the solution
    z = wta.fullValue(Q, V, y)
    print('Random value', z)
    return z


if __name__ == "__main__":
    # Problem data
    n = 7
    tgts = 6
    wpns = 46
    itrs = 3000

    # Survival probabilities
    Q, V, WW = generate_random_problem(tgts, wpns)
    WW = np.ones(wpns)
    # Reference values
    t = time()
    true_p, true_x = wta.wta(Q, V, WW, integer=False, verbose=True, mosek_params={'MSK_DPAR_OPTIMIZER_MAX_TIME': 30.0,})
    true_time = time() - t
    print("true time", true_time)
    print("true x", true_x)

    # Generate random solution for comparison
    get_rand_solution_value(Q, V, tgts, wpns, integer=False)

    # Generate splitting
    num_nodes_per_function = 1
    node_tgts = list(range(tgts))
    num_nodes_per_tgt = 1 # num_nodes_per_function*np.ones(tgts) # Assumes even splitting
    v0 = 1/tgts*np.ones((tgts, wpns))
    # Generate the data
    data = smallSplit(tgts, tgts, wpns, node_tgts, num_nodes_per_tgt)
    
    # Append data for the simplex projection
    proj_data = {'QQ':Q, 'VV':V, 'WW':WW, 's':range(wpns), 'Q':Q, 'V':V, 'v0':v0}
    fulldata = []
    for i in range(tgts):
        fulldata.append(data[i])
    fulldata.append(proj_data)
    fulldata.append({'Q':Q, 'V':V, 'WW':WW}) # For testing convergence

    # Generate the resolvents
    resolvents = []
    for i in range(tgts):
        resolvents.append(wtaResolvent)
    resolvents.append(simplexProj)
    W, L = oars.getMaxConnect(n)
    t = time()
    alg_x, results = oars.solve(n, fulldata, resolvents, W, L, itrs=itrs, vartol= 1e-6, objtol = 1e-4, fval=fullValue, parallel=False, verbose=True)  
    print("alg time", time()-t)

    print("true x")
    print(true_x) 
    print("alg x")
    print(alg_x)

    alg_p = results[n-1]['log'][-1]['value']
    print("alg val", alg_p)
    print("True cont val", true_p)



    logs = [results[i]['log'] for i in range(n)]

    timestamp = time()
    with open('logs_lambert'+str(timestamp)+'.json', 'w') as f:
        json.dump(logs, f)
