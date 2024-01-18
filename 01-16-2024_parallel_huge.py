# Parallel WTA using MT algorithm over 3 nodes

from collections import defaultdict
import multiprocessing as mp
import numpy as np
import pandas as pd
import cvxpy as cp
import wta
from time import time

np.set_printoptions(precision=3, suppress=True, linewidth=200)

def getMT(n):
    '''Get Malitsky-Tam values for W and L
    n: number of agents'''
    W = np.zeros((n,n))
    W[0,0] = 1
    W[0,1] = -1
    for r in range(1,n-1):
        W[r,r-1] = -1
        W[r,r] = 2
        W[r,r+1] = -1
    W[n-1,n-1] = 1
    W[n-1,n-2] = -1

    L = np.zeros((n,n))
    # Add ones just below the diagonal
    for i in range(n-1):
        L[i+1,i] = 1
    L[n-1,0] = 1
    return W, L

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
    # print("Building queues for #", n)
    # print(L)
    # print(W)
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

    #print(WQ, up_LQ, down_LQ, up_BQ, down_BQ)
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
    #print(data)
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
    return prob, w, v, r, y

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
    prob, w, v, r, y = buildWTAProb(data)

    # Log files
    logw = []
    logv = []

    m = v.value.shape
    v_temp = np.zeros(m)
    z_n = np.zeros(m)
    for itr in range(itrs):
        print(f'Node {i} iteration {itr}')
        # Get data from upstream L queue
        r.value = z_n
        for k in up_LQ:
            #print(f'Node {i} getting data from upstream L queue')
            # if i==2:print("node",i,"itr", itr, "from", k, "before value for r", r.value)
            temp = queue[k,i].get()
            # if i==2:print("node",i,"itr", itr, "from", k, "value for temp", temp, "value for L", L[i,k])
            r.value = r.value + L[i,k]*temp
            # if i==2:print("node",i,"itr", itr, "from", k, "after value for r", r.value)
            

        # Pull from the B queues, update r and v_temp
        for k in up_BQ:
            #print(f'Node {i} getting data from upstream B queue {k}')
            temp = queue[k,i].get()
            # if i==2:print("node",i,"itr", itr, "from", k, "before value for r", r.value)
            # if i==2:print("node",i,"itr", itr, "from", k, "value for temp", temp, "value for L", L[i,k])
            r.value = r.value + L[i,k]*temp
            # if i==2:print("node",i,"itr", itr, "from", k, "after value for r", r.value)
            v_temp += W[i,k]*temp

        # Solve the problem
        prob.solve(ignore_dpp = True, verbose=False)
        # print("p node",i,"itr", itr, " value for w", w.value)
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
        for k in WQ:
            # if i ==2:print("node",i,"itr", itr, "from", k, "before value for v_temp", v_temp)
            temp = queue[k,i].get()            
            # if i ==2:print("node",i,"itr", itr, "from", k, "value for temp", temp, "value for W", W[i,k])
            v_temp += W[i,k]*temp
            # if i ==2:print("node",i,"itr", itr, "from", k, "after value for v_temp", v_temp)
        #v_temp += sum([W[i,k]*queue[k,i].get() for k in WQ])
        v_temp += sum([W[i,k]*queue[k,i].get() for k in down_BQ])
        v.value = v.value - gamma*(W[i,i]*w.value + v_temp)
        
        # Zero out v_temp without reallocating memory
        v_temp.fill(0)

    # Return the solution
    return {'logw':logw, 'w':w.value, 'logv':logv, 'v':v.value}

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
        v0 = 1/tgts*np.array(WW)*np.ones(m)
        data.append({'QQ':q, 'VV':v, 'WW':WW, 'v0':v0, 'Lii':L[i,i]})
    # Run subproblems in parallel
    with mp.Pool(processes=n) as p:
        params = [(data[i], W, L, i, WQ[i], up_LQ[i], down_LQ[i], up_BQ[i], down_BQ[i], Queue_Array, 0.5, itrs) for i in range(n)]
        results = p.starmap(subproblem, params)
    w = sum(results[i]['w'] for i in range(n))/n
    prob_val = V@wta.get_final_surv_prob(Q, w)
    # for i in [0, 2, 4]:
    #     print("Node", i)
    #     print(results[i]['logw'])
    #     print(results[i]['logv'])
    return prob_val, w, results

def serialAlgorithm(n, m, Q, V, WW, W, L, node_tgts, num_nodes_per_tgt, itrs=1001, gamma=0.5, verbose=False):
    tgts = m[0]
    v0 = []
    vk = []
    log = []
    log_e = []
    log_w = {}
    for i in range(n):
        vi = 1/tgts*np.array(WW)*np.ones(m)
        v0.append(vi)
        vk.append(vi.copy())
        log_w[i] = []

    # Create variables/params/objs for the algorithm
    probs = [] # List of problems for each node
    all_x = [] # List of last x solutions for each node as params
    all_v = [] # List of last v solutions for each node as params
    all_w = []# List of variables for each node
    all_y = [] # List of y values for each node
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
        all_y.append(y)
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
            # if i == 2:print("y", i, itr, all_y[i].value)
            probs[i].solve()
            log_w[i].append(all_w[i].value)
            # if itr % 500 == 0:
            #     print("Iteration", itr, "Node", i)
            #print("s", i, itr, all_w[i].value)
            e += np.linalg.norm(all_w[i].value - all_x[i].value)
            all_x[i].value = all_w[i].value
        log_e.append(e)
        for i in range(n):
            vk[i] -= gamma*sum(W[i,j]*all_x[j].value for j in range(n))
            all_v[i].value = vk[i]
        #log.append(V@get_final_surv_prob(Q, all_x[0].value))
        # if itr % 500 == 0:
        # print("v", itr, vk)
    w_val = sum(all_w[i].value for i in range(n))/n
    prob_val = V@wta.get_final_surv_prob(Q, w_val)
    return prob_val, w_val, log, log_e, log_w

if __name__ == '__main__':
    #mp.freeze_support()
    # Problem data
    # Data
    n = 8
    tgts = 160
    wpns = 200
    itrs = 30
    m = (tgts, wpns)

    # Survival probabilities
    Q, V, WW = wta.generate_random_problem(tgts, wpns)

    # Reference values
    t = time()
    true_p, true_x = wta.wta(Q, V, WW, integer=False, verbose=False)
    true_time = time() - t
    # W and L for Malitsky-Tam
    W, L = getMT(n)
    # L = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1.5, 0.5, 0, 0], [0.5,1.5,0,0]])
    # W = np.array([[1, 0, -1, 0], [0, 2, -0.5, -1.5], [-1, -0.5, 1.67218, -0.17218], [0,-1.5,-0.17218,1.67218]])
    # Generate splitting
    node_tgts = {}
    #num_nodes_per_tgt = {}
    for i in range(n):
        end = (i+1)*tgts//n
        if i == n-1:
            end = tgts
        node_tgts[i] = list(range(i*tgts//n, end))
    num_nodes_per_tgt = np.ones(tgts) # Assumes even splitting
    #print(node_tgts)
    t = time()
    alg_p, alg_x, results = parallelAlgorithm(n, m, Q, V, WW, W, L, node_tgts, num_nodes_per_tgt, itrs=itrs)   
    print("alg time", time()-t)
    t = time() 
    #s_alg_p, s_alg_x, log, log_e, log_w = serialAlgorithm(n, m, Q, V, WW, W, L, node_tgts, num_nodes_per_tgt, itrs=itrs)
    # print("s alg time", time()-t)
    print("alg val", alg_p)
    print("s alg val", s_alg_p)
    print("True val", true_p)
    # print("alg x", alg_x)
    # print("s alg x", s_alg_x)
    # print("true x", true_x)
    
    # print("direct time", true_time)
    # print("V", V)
    # v = 1/tgts*np.array(WW)*np.ones(m)
    # print("v", v)
    # print(results[2]['logv'][0])
    # for i in [2]:#range(n):
    #     print("Node", i)
    #     for j in range(1,itrs):
    #         print("Iteration", j)
    #         print("s", log_w[i][j])
    #         print("p", results[i]['logw'][j])
    #         print("v", results[i]['logv'][j])
    #         v = v - 0.5*sum(W[i,k]*results[k]['logw'][j-1] for k in range(n))
    #         print("correct v", v)
    #         r = sum(L[i,k]*results[k]['logw'][j] for k in range(i))
    #         print("correct r", r)
    #         y = v + r
    #         print("correct y", y)