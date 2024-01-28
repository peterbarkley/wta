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
    Queue_Array is the dictionary of the queues with keys (i,j) for the queues from i to j
    Comms_Data is a list of the required comms data for each node
    The comms data entry for node i is a dictionary with the following keys:
    WQ: nodes which feed only W data into node i
    up_LQ: nodes which feed only L data into node i
    down_LQ: nodes which receive only L data from node i
    up_BQ: nodes which feed both W and L data into node i, and node i feeds W back to
    down_BQ: nodes which receive W and L data from node i
    '''
    # Get the number of nodes
    n = W.shape[0]

    Queue_Array = {} # Queues required by non-zero off diagonal elements of W
    Comms_Data = []
    for i in range(n):
        WQ = []
        up_LQ = []
        down_LQ = []
        up_BQ = []
        down_BQ = []
        Comms_Data.append({'WQ':WQ, 'up_LQ':up_LQ, 'down_LQ':down_LQ, 'up_BQ':up_BQ, 'down_BQ':down_BQ})

    for i in range(n):
        comms_i = Comms_Data[i]
        for j in range(i):
            comms_j = Comms_Data[j]
            if W[i,j] != 0:
                if (i,j) not in Queue_Array:
                    queue_ij = man.Queue()
                    Queue_Array[i,j] = queue_ij
                if (j,i) not in Queue_Array:
                    queue_ji = man.Queue()
                    Queue_Array[j,i] = queue_ji
                if L[i,j] != 0:
                    comms_i['up_BQ'].append(j)
                    comms_j['down_BQ'].append(i)
                else:
                    comms_j['WQ'].append(i)
                    comms_i['WQ'].append(j)
            elif L[i,j] != 0:
                if (j,i) not in Queue_Array:
                    queue_ji = man.Queue()
                    Queue_Array[j,i] = queue_ji
                comms_i['up_LQ'].append(j)
                comms_j['down_LQ'].append(i)

    return Queue_Array, Comms_Data

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
    Lii = data['Lii']

    # Get the number of targets and weapons
    m = QQ.shape

    # Create the variable
    w = cp.Variable(m)

    # Create the parameter
    y = cp.Parameter(m) # resolvent parameter, sum of weighted previous resolvent outputs and v_i
    
    # Create the objective
    weighted_weapons = cp.multiply(w, np.log(QQ)) # (tgts, wpns)
    survival_probs = cp.exp(cp.sum(weighted_weapons, axis=1)) # (tgts,)
    obj = cp.Minimize(VV@survival_probs + .5*cp.sum_squares((1-Lii)*w - y))

    # Create the constraints
    cons = [w >= 0, cp.sum(w, axis=0) <= WW]

    # Create the problem 
    prob = cp.Problem(obj, cons)

    # Return the problem, variable, and parameters
    return prob, w, y

class resolvent:
    '''Resolvent function'''
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    def __call__(self, x):
        u = x - self.data
        return max(abs(u)-1, 0)*np.sign(u) + self.data

    def __repr__(self):
        return "L1 norm resolvent"

# WTA resolvent
class wtaResolvent:
    '''Resolvent function'''
    def __init__(self, data):
        self.data = data
        prob, w, y = buildWTAProb(data)
        self.prob = prob
        self.w = w
        self.y = y
        self.shape = w.shape

    def __call__(self, x):
        self.y.value = x
        self.prob.solve(verbose=False)
        return self.w.value

    def __repr__(self):
        return "wtaResolvent"

def subproblem(i, data, problem_builder, W, L, comms_data, queue, gamma=0.5, itrs=501, verbose=False):
    '''
    Solves the subproblem for node i
    Inputs:
    i is the node number
    data is a dictionary contains arguments for the problem
    W is the W matrix
    L is the L matrix
    comms_data is a dictionary with the following keys:
    WQ: nodes which feed only W data into node i
    up_LQ: nodes which feed only L data into node i
    down_LQ: nodes which receive only L data from node i
    up_BQ: nodes which feed both W and L data into node i, and node i feeds W back to
    down_BQ: nodes which receive W and L data from node i
    queue is the array of queues
    gamma is the consensus parameter
    itrs is the number of iterations
    '''

    # Create the problem
    resolvent = problem_builder(data)
    m = resolvent.shape
    v_temp = np.zeros(m)
    if isinstance(data, dict) and 'v0' in data:
        local_v = data['v0']
    else:
        local_v = np.zeros(m)
    local_r = np.zeros(m)

    # if i == 0:
    #     log_val = []
    # Iterate over the problem
    for itr in range(itrs):
        if itr % 10 == 0 and verbose:
            print(f'Node {i} iteration {itr}')

        # Get data from upstream L queue
        for k in comms_data['up_LQ']:
            temp = queue[k,i].get()
            local_r += L[i,k]*temp

        # Pull from the B queues, update r and v_temp
        for k in comms_data['up_BQ']:
            temp = queue[k,i].get()
            local_r += L[i,k]*temp
            v_temp += W[i,k]*temp

        # Solve the problem
        y_value = local_v + local_r
        w_value = resolvent(y_value)

        # Put data in downstream queues
        for k in comms_data['down_LQ']:
            queue[i,k].put(w_value)
        for k in comms_data['down_BQ']:
            queue[i,k].put(w_value)

        # Put data in upstream W queues
        for k in comms_data['WQ']:
            queue[i,k].put(w_value)
        for k in comms_data['up_BQ']:
            queue[i,k].put(w_value)

        # Update v from all W queues
        for k in comms_data['WQ']:
            temp = queue[k,i].get()            
            v_temp += W[i,k]*temp
            
        # Update v from all B queues
        v_temp += sum([W[i,k]*queue[k,i].get() for k in comms_data['down_BQ']])
        local_v = local_v - gamma*(W[i,i]*w_value + v_temp)
        
        # Zero out v_temp without reallocating memory
        v_temp.fill(0)
        local_r.fill(0)

        # Log the value -- needs to be done in another process
        # if i == 0:
        #     prob_val = fullValue(data, w_value)
        #     log_val.append(prob_val)

    # Return the solution
    # if i == 0:
    #     return {'w':w_value, 'v':local_v, 'log':log_val}
    return {'w':w_value, 'v':local_v}


def fullValue(d, w):
    '''Get the full value of the problem'''
    return d['V']@wta.get_final_surv_prob(d['Q'], w)

def parallelAlgorithm(n, data, resolvents, W, L, itrs=1001, gamma=0.5, verbose=False):
    # Create the queues
    man = mp.Manager()
    Queue_Array, Comms_Data = requiredQueues(man, W, L)

    # Run subproblems in parallel
    with mp.Pool(processes=n) as p:
        params = [(i, data[i], resolvents[i], W, L, Comms_Data[i], Queue_Array, 0.5, itrs, verbose) for i in range(n)]
        results = p.starmap(subproblem, params)

    # Get the final value
    w = sum(results[i]['w'] for i in range(n))/n

    return w, results

def serialAlgorithm(n, m, Q, V, WW, W, L, node_tgts, num_nodes_per_tgt, itrs=1001, gamma=0.5, verbose=False):
    tgts = m[0]
    v0 = []
    vk = []
    log = []
    log_e = []
    #log_w = {}
    for i in range(n):
        vi = 1/tgts*np.array(WW)*np.ones(m)
        v0.append(vi)
        vk.append(vi.copy())
        #log_w[i] = []

    # Create variables/params/objs for the algorithm
    probs = [] # List of problems for each node
    all_x = [] # List of last x solutions for each node as params
    all_v = [] # List of last v solutions for each node as params
    all_w = []# List of variables for each node
    #all_y = [] # List of y values for each node
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
        #all_y.append(y)
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
            #log_w[i].append(all_w[i].value)
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
    return prob_val, w_val, log, log_e

def graphResults(logs, labels, ref):
    '''Graph the results'''
    import matplotlib.pyplot as plt
    i=0
    for log in logs:
        plt.plot(log, label=labels[i])
        i+=1
    
    # Horizontal line for reference
    plt.axhline(y=ref, color='red', linestyle='dashed')
    
    # Add labels
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Parallel WTA convergence')

    # Add legend
    plt.legend()
    #plt.show()

    # Save the figure
    timestamp = time()
    plt.savefig('parallel_wta'+str(timestamp)+'.png')

def wrapper(name):
    if name == '__main__':
        main_routine()

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

def distributeFunctions(num_nodes, num_functions, num_nodes_per_function=1):
    '''Distribute the functions to the nodes'''
    '''Can send the same function to the same node multiple times'''
    from random import shuffle

    functs = list(range(num_functions))
    functs = functs*num_nodes_per_function
    shuffle(functs)
    
    return np.array_split(functs, num_nodes)

def distributeFunctionsLinear(num_nodes, num_functions, num_nodes_per_function=1):
    '''Distribute the functions to the nodes'''
    '''Assumes even splitting'''
    from itertools import cycle, islice
    functs = list(range(num_functions))
    stride = num_functions*num_nodes_per_function//num_nodes
    step = stride//num_nodes_per_function
    funcs_per_node = []

    for i in range(num_nodes):
        start = i*step
        end = start + stride
        funcs_per_node.append(list(islice(cycle(functs), start, end)))
    
    return funcs_per_node


def main_routine():
    # Problem data
    n = 4
    tgts = 120
    wpns = 10
    itrs = 20

    # Survival probabilities
    Q, V, WW = wta.generate_random_problem(tgts, wpns)

    # Reference values
    t = time()
    true_p, true_x = wta.wta(Q, V, WW, integer=False, verbose=False)
    true_time = time() - t
    # W and L for Malitsky-Tam
    #W, L = getMT(n)

    # W and L for 4 node example
    L = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1.5, 0.5, 0, 0], [0.5,1.5,0,0]])
    W = np.array([[1, 0, -1, 0], [0, 2, -0.5, -1.5], [-1, -0.5, 1.67218, -0.17218], [0,-1.5,-0.17218,1.67218]])

    # Generate splitting
    num_nodes_per_function = 2
    node_tgts = distributeFunctionsLinear(n, tgts, num_nodes_per_function)
    num_nodes_per_tgt = num_nodes_per_function*np.ones(tgts) # Assumes even splitting

    # Generate the data
    data = generateSplitData(n, tgts, wpns, node_tgts, num_nodes_per_tgt, L)
    resolvents = [wtaResolvent]*n
    t = time()
    alg_x, results = parallelAlgorithm(n, data, resolvents, W, L, itrs=itrs, verbose=True)   
    print("alg time", time()-t)
    print("direct time", true_time)
    
    alg_p = fullValue(data[-1], alg_x)
    print("alg val", alg_p)
    print("True val", true_p)

    # Test L1 resolvent
    ldata = np.array([1, 2, 3, 4])
    lres = [resolvent]*n
    lx, lresults = parallelAlgorithm(n, ldata, lres, W, L, itrs=itrs, verbose=True)
    print("lx", lx)
    print("lresults", lresults)
    # log_alg = results[0]['log']
    # log_mt = mt_results[0]['log']
    # graphResults([log_alg, log_mt], ["New Algorithm", "Malitsky Tam"], true_p)
    

    # # Save the alg_x as a json file
    # import json
    # with open('alg_x.json', 'w') as f:
    #     json.dump(alg_x.tolist(), f)


    # 
