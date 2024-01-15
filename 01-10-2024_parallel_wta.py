# Parallel WTA using MT algorithm over 3 nodes

from collections import defaultdict
import multiprocessing as mp
import numpy as np
import pandas as pd
import cvxpy as cp
import wta
from microbench import MicroBench
basic_bench = MicroBench()

np.set_printoptions(precision=3, suppress=True, linewidth=200)

def requiredQueues(man, W, L):
    '''
    Returns the queues for the given W and L matrices
    Inputs:
    man is the multiprocessing manager
    W is the W matrix
    L is the L matrix

    Returns:
    WQ is the dictionary of queues for the W matrix
    up_LQ is the dictionary of queues for nodes which feed L into node i
    down_LQ is the dictionary of queues for nodes which node i feeds into
    up_BQ is the dictionary of queues for both W and L data flow into node i and node i feeds W back to node j
    down_BQ is the dictionary of queues from which j receives W data
    '''
    # Get the number of nodes
    n = W.shape[0]
    Queue_Array = {} # Queues required by non-zero off diagonal elements of W
    WQ = defaultdict(list) # Queues required by non-zero off diagonal elements of W
    up_LQ = defaultdict(list) # Queues required by non-zero off diagonal elements of tril(L)
    down_LQ = defaultdict(list) # Queues required by non-zero off diagonal elements of triu(L)
    up_BQ = defaultdict(list) # Queues required by both W and L
    down_BQ = defaultdict(list) # Queues required by both W and L
    for i in range(n):
        for j in range(i):
            if W[i,j] != 0:
                
                if (i,j) not in Queue_Array:
                    queue_ij = man.Queue()
                    Queue_Array[i,j] = queue_ij
                if (j,i) not in Queue_Array:
                    queue_ji = man.Queue()
                    Queue_Array[j,i] = queue_ji
                if L[i,j] != 0:
                    up_BQ[i].append(j) # i receives L&W data from j
                    down_BQ[j].append(i) # j sends L data to i, i sends W data to j
                else:
                    WQ[j].append(i) # j sends/receives W data to/from i
                    WQ[i].append(j) # i sends/receives W data to/from j
            elif L[i,j] != 0:
                if (j,i) not in Queue_Array:
                    queue_ji = man.Queue()
                    Queue_Array[j,i] = queue_ji
                up_LQ[i].append(j) # i receives L data from j
                down_LQ[j].append(i) # j sends L data to i

    return Queue_Array, WQ, up_LQ, down_LQ, up_BQ, down_BQ

def buildWTAProb(data):
    '''
    Builds the WTA problem

    Inputs:
    data is a dictionary containing the following keys:
    QQ is the survival probabilities for the targets in the node
    VV is the value of the targets in the node
    WW is the number of weapons for all weapons
    v0 is the initial consensus parameter
    Lii is the diagonal element of L (typically 0)

    Returns:
    prob is the problem
    w is the variable
    v is the consensus parameter
    r is the resolvent parameter
    '''
    QQ = data['QQ']
    VV = data['VV']
    WW = data['WW']
    v0 = data['v0']
    Lii = data['Lii']
    # Get the number of targets and weapons
    m = QQ.shape

    # Create the variables
    w = cp.Variable(m)

    # Create the parameters
    r = cp.Parameter(m) # holds sum(L[i,j]*all_x[j] for j in range(i))
    r.value = np.zeros(m)
    v = cp.Parameter(m) # consensus parameter
    v.value = v0
    
    # Create the obj
    y = v + r + Lii*w # resolvent argument
    weighted_weapons = cp.multiply(w, np.log(QQ)) # (tgts, wpns)
    survival_probs = cp.exp(cp.sum(weighted_weapons, axis=1)) # (tgts,)
    obj = cp.Minimize(VV@survival_probs + .5*cp.sum_squares(w - y))

    # Create the constraints
    cons = [w >= 0, cp.sum(w, axis=0) <= WW]

    # Create the problem 
    prob = cp.Problem(obj, cons)

    # Return the problem, variable, and parameters
    return prob, w, v, r

def subproblem(data, W, L, i, WQ, up_LQ, down_LQ, up_BQ, down_BQ, queue, gamma=0.5, itrs=501, verbose=False):
    '''
    Solves the subproblem for node i
    data contains arguments for the problem
    W is the W matrix
    L is the L matrix
    WQ is the list of queues for the W matrix
    up_LQ is the list of queues for nodes which feed into node i
    down_LQ is the list of queues for nodes which node i feeds into
    up_BQ is the list of queues for both W and L data flow into node i
    down_BQ is the list of queues from which j receives W data
    i is the node number
    '''

    # Create the problem
    prob, w, v, r = buildWTAProb(data)

    # Log files
    logw = []
    logv = []

    m = v.value.shape
    v_temp = np.zeros(m)
    z_n = np.zeros(m)
    for itr in range(itrs):
        #print(f'Node {i} iteration {itr}')
        # Get data from upstream L queue
        if len(up_LQ) > 0:
            #print(f'Node {i} getting data from upstream L queue')
            r.value = sum([L[i,k]*queue[k,i].get() for k in up_LQ])
        else:
            r.value = z_n

        # Pull from the B queues, update r and v_temp
        for k in up_BQ:
            #print(f'Node {i} getting data from upstream B queue {k}')
            temp = queue[k,i].get()
            r.value += L[i,k]*temp
            v_temp += W[i,k]*temp

        # Solve the problem
        #print(f'Node {i} solving problem')
        prob.solve(verbose=False)

        # Log results
        logw.append(w.value)
        logv.append(v.value)

        # Put data in downstream L queue
        for k in down_LQ:
            queue[i,k].put(w.value)
        for k in down_BQ:
            queue[i,k].put(w.value)

        # Put data in upstream W queue
        for k in WQ:
            queue[i,k].put(w.value)
        for k in up_BQ:
            queue[i,k].put(w.value)

        # Update v from all W queues
        v_temp += sum([W[i,k]*queue[k,i].get() for k in WQ])
        v_temp += sum([W[i,k]*queue[k,i].get() for k in down_BQ])
        v.value = v.value - gamma*(W[i,i]*w.value + v_temp)
        
        # Zero out v_temp without reallocating memory
        v_temp.fill(0)

    # Return the solution
    return {'logw':logw, 'w':w.value, 'logv':logv, 'v':v.value}

@basic_bench
def parallelAlgorithm(n, m, Q, V, WW, W, L, node_tgts, num_nodes_per_tgt, itrs=1001, gamma=0.5, verbose=False):
    # Create the queues
    man = mp.Manager()
    Queue_Array, WQ, up_LQ, down_LQ, up_BQ, down_BQ = requiredQueues(man, W, L)

    tgts = m[0]
    data = []
    for i in range(n):
        q = np.ones(m)
        q[node_tgts[i]] = Q[node_tgts[i]] # Only use the targets that are in the node
        v = np.zeros(tgts)
        v[node_tgts[i]] = V[node_tgts[i]] # Only use the targets that are in the node
        v = v/num_nodes_per_tgt # Divide the value by the number of nodes that have the target
        data.append({'QQ':q, 'VV':v, 'WW':WW, 'v0':np.zeros(m), 'Lii':L[i,i]})
    # Run subproblems in parallel
    with mp.Pool(processes=n) as p:
        params = [(data[i], W, L, i, WQ[i], up_LQ[i], down_LQ[i], up_BQ[i], down_BQ[i], Queue_Array, 0.5, itrs) for i in range(n)]
        results = p.starmap(subproblem, params)
    w = results[0]['w']
    print(V@wta.get_final_surv_prob(Q, w))
    return w

@basic_bench
def serialAlgorithm(n, m, Q, V, WW, W, L, node_tgts, num_nodes_per_tgt, itrs=1001, gamma=0.5, verbose=False):
    
    tgts = m[0]
    v0 = []
    vk = []
    log = []
    log_e = []
    for i in range(n):
        vi = 1/tgts*np.array(WW)*np.ones(m)
        v0.append(vi)
        vk.append(vi.copy())

    # Create variables/params/objs for the algorithm
    probs = [] # List of problems for each node
    all_x = [] # List of last x solutions for each node as params
    all_v = [] # List of last v solutions for each node as params
    all_w = []# List of variables for each node
    for i in range(n):
        w = cp.Variable(m)
        all_w.append(w)
        x = cp.Parameter(m)
        x.value = np.zeros(m)
        all_x.append(x)
        v = cp.Parameter(m)
        v.value = v0[i]
        all_v.append(v)
        y = v + sum(L[i,j]*all_x[j] for j in range(i)) + L[i,i]*w
        qq = np.ones(m)
        qq[node_tgts[i]] = Q[node_tgts[i]] # Only use the targets that are in the node
        weighted_weapons = cp.multiply(w, np.log(qq)) # (tgts, wpns)
        survival_probs = cp.exp(cp.sum(weighted_weapons, axis=1)) # (tgts,)
        VV = np.zeros(tgts)
        VV[node_tgts[i]] = V[node_tgts[i]]
        VV = VV/num_nodes_per_tgt
        obj = cp.Minimize(VV@survival_probs + .5*cp.sum_squares(w - y))
        cons = [w >= 0, cp.sum(w, axis=0) <= WW]
        probs.append(cp.Problem(obj, cons))

    # Run the algorithm
    for itr in range(itrs):
        #print("Iteration", itr)
        e = 0
        for i in range(n):
            probs[i].solve()
            # if itr % 500 == 0:
            #     print("Iteration", itr, "Node", i)
            #     print(all_w[i].value)
            #e += np.linalg.norm(all_w[i].value - all_x[i].value)
            all_x[i].value = all_w[i].value
        #log_e.append(e)
        for i in range(n):
            vk[i] -= gamma*sum(W[i,j]*all_x[j].value for j in range(n))
            all_v[i].value = vk[i]
        #log.append(V@get_final_surv_prob(Q, all_x[0].value))
        # if itr % 500 == 0:
        #     print("v", vk)
    
    print(V@wta.get_final_surv_prob(Q, all_w[0].value))
    return all_w[0].value

if __name__ == '__main__':
    #mp.freeze_support()
    # Problem data
    # Data
    n = 3
    tgts = 3
    wpns = 4
    m = (tgts, wpns)
    # Survival probabilities
    Q = np.array([[0.8, 0.4, 0.6, 0.5],
                [0.6, 0.5, 0.7, 0.5],
                [0.4, 0.6, 0.6, 0.5]])

    # Target values
    V = np.array([10, 6, 10])

    # Weapon values
    WW = np.array([1, 1, 1, 1])

    # Reference values
    true_p, true_x = wta.wta(Q, V, WW, integer=False, verbose=False)

    # W and L for Malitsky-Tam
    M = np.array([
        [-1, 1, 0],
        [0, -1, 1]])

    W = M.T@M
    L = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]])

    node_tgts = {0:[0, 1], 1:[1,2], 2:[0,2]}
    num_nodes_per_tgt = [2, 2, 2]

    parallelAlgorithm(n, m, Q, V, WW, W, L, node_tgts, num_nodes_per_tgt)    
    serialAlgorithm(n, m, Q, V, WW, W, L, node_tgts, num_nodes_per_tgt)

    results = pd.read_json(basic_bench.outfile.getvalue(), lines=True)
    results['delta'] =   results['finish_time'] - results['start_time']
    print(results.groupby('function_name').mean())
    
    