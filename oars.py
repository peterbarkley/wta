import multiprocessing as mp
import numpy as np
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

    # Iterate over the problem
    for itr in range(itrs):
        if verbose and itr % 10 == 0:
            print(f'Node {i} iteration {itr}')

        # Get data from upstream L queue
        for k in comms_data['up_LQ']:
            local_r += L[i,k]*queue[k,i].get()
            
        # Pull from the B queues, update r and v_temp
        for k in comms_data['up_BQ']:
            temp = queue[k,i].get()
            local_r += L[i,k]*temp
            v_temp += W[i,k]*temp

        # Solve the problem
        w_value = resolvent(local_v + local_r)

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
            #temp = queue[k,i].get()            
            v_temp += W[i,k]*queue[k,i].get() 
            
        # Update v from all B queues
        for k in comms_data['down_BQ']:
            v_temp += W[i,k]*queue[k,i].get()
        #v_temp += sum([W[i,k]*queue[k,i].get() for k in comms_data['down_BQ']])
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
    if hasattr(resolvent, 'log'):
        return {'w':w_value, 'v':local_v, 'log':resolvent.log}
    return {'w':w_value, 'v':local_v}

def parallelAlgorithm(n, data, resolvents, W, L, itrs=1001, gamma=0.5, verbose=False):
    # Create the queues
    man = mp.Manager()
    Queue_Array, Comms_Data = requiredQueues(man, W, L)

    # Run subproblems in parallel
    if verbose:
        print('Starting Parallel Algorithm')
        t = time()
    with mp.Pool(processes=n) as p:
        params = [(i, data[i], resolvents[i], W, L, Comms_Data[i], Queue_Array, 0.5, itrs, verbose) for i in range(n)]
        results = p.starmap(subproblem, params)
    if verbose:
        print('Parallel Algorithm Loop Time:', time()-t)
    # Get the final value
    w = sum(results[i]['w'] for i in range(n))/n

    return w, results

def serialAlgorithm(n, data, resolvents, W, L, itrs=1001, gamma=0.5, verbose=False):
    
    # Initialize the resolvents and variables
    all_x = []
    all_v = []
    for i in range(n):
        resolvents[i] = resolvents[i](data[i])
        x = np.zeros(resolvents[i].shape)
        all_x.append(x)
        if isinstance(data[i], dict) and 'v0' in data[i]:
            v = data[i]['v0']
        else:
            v = np.zeros(resolvents[i].shape)
        all_v.append(v)

    # Run the algorithm
    if verbose:
        print('Starting Serial Algorithm')
        t = time()
    for itr in range(itrs):
        if verbose and itr % 10 == 0:
            print(f'Iteration {itr}')

        for i in range(n):
            resolvent = resolvents[i]
            y = all_v[i] + sum(L[i,j]*all_x[j] for j in range(i))
            all_x[i] = resolvent(y)

        for i in range(n):            
            all_v[i] = all_v[i] - gamma*sum(W[i,j]*all_x[j] for j in range(n))
    if verbose:
        print('Serial Algorithm Loop Time:', time()-t)
    x = sum(all_x)/n
    # Build results list
    results = []
    for i in range(n):
        if hasattr(resolvents[i], 'log'):
            results.append({'w':all_x[i], 'v':all_v[i], 'log':resolvents[i].log})
        else:
            results.append({'w':all_x[i], 'v':all_v[i]})
    return x, results

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

def solve(n, data, resolvents, W, L, itrs=1001, gamma=0.5, parallel=True, verbose=False):
    '''Solve the problem with a given W and L matrix
    Inputs:
    n is the number of nodes
    data is a list of dictionaries containing the problem data
    resolvents is a list of resolvent functions
    W is the W matrix
    L is the L matrix
    itrs is the number of iterations
    gamma is the consensus parameter
    parallel is a boolean for parallel or serial
    verbose is a boolean for verbose output
    Outputs:
    alg_x is the solution
    results is a dictionary of the results of the algorithm for each node
    '''

    if parallel:
        alg_x, results = parallelAlgorithm(n, data, resolvents, W, L, itrs=itrs, verbose=verbose)
    else:
        alg_x, results = serialAlgorithm(n, data, resolvents, W, L, itrs=itrs, verbose=verbose)

    return alg_x, results

def solveMT(n, data, resolvents, itrs=1001, gamma=0.5, parallel=True, verbose=False):
    # Solve the problem with the Malitsky-Tam W and L matrices
    W, L = getMT(n)
    return solve(n, data, resolvents, W, L, itrs=itrs, gamma=gamma, parallel=parallel, verbose=verbose)