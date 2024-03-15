import multiprocessing as mp
import numpy as np
import cvxpy as cvx
from time import time

np.set_printoptions(precision=3, suppress=True, linewidth=200)

def getParallel(n):
    """Return the W and L matrices for parallel computation
    n: number of agents

    Returns:
    W: W matrix n x n numpy array
    L: L matrix n x n numpy array
    """
    fixed = {(i, j): 0 for i in range(n) for j in range(n) if j<i and (i < n//2 or j > n//2)}
    W, L = getMaxFiedlerSum(n, fixed_Z=fixed)
    return W, L

def getMaxCut(n):
    """Return the W and L matrices for parallel computation
    n: number of agents

    Returns:
    W: W matrix n x n numpy array
    L: L matrix n x n numpy array
    """
    fixed = {(n-1, 0): 0 }
    W, L = getMaxFiedlerSum(n, fixed_Z=fixed)
    return W, L
 
def getMaxFiedlerSum(n, fixed_Z=None, fixed_W=None, vz=1.0, vw=1.0, verbose=False):
    
    # Variables
    cw = cvx.Variable(1, pos=True)  # Fiedler value for W
    cz = cvx.Variable(1, pos=True)  # Fiedler value for Z
    Z = cvx.Variable((n,n), symmetric=True)
    W = cvx.Variable((n,n), PSD=True)

    # Constraints
    c = 1-np.cos(np.pi/n)
    D = Z - W
    cons = [D >> 0, # Z - W is PSD
            cvx.lambda_sum_smallest(W, 2) >= cw, # Fiedler value constraint
            cw >= c, # Fiedler value constraint
            cvx.lambda_sum_smallest(Z, 2) >= cz, # Fiedler value constraint
            cz >= c, # Fiedler value constraint
            cvx.sum(W, axis=1) == 0, # W is row stochastic
            cvx.sum(Z) == 0] # Z is sums to zero
    cons += [Z[i,i] == 2 for i in range(n)] # Z diagonal entries equal 2

    # Set fixed L and W values
    if fixed_Z is not None:
        cons += [Z[idx] == val for idx,val in fixed_Z.items()]
    if fixed_W is not None:
        cons += [W[idx] == val for idx,val in fixed_W.items()]
    obj_fun = vw*cw + vz*cz

    # Solve
    obj = cvx.Maximize(obj_fun)
    prob = cvx.Problem(obj, cons)
    prob.solve()

    if verbose:
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal cw", cw.value)
        print("optimal cz", cz.value)
        print("optimal W")
        print(W.value)
        print("optimal Z")
        print(Z.value)
    if prob.status == 'infeasible':
        return None, None
    return W.value, -np.tril(Z.value,-1)

def getMT(n):
    '''Get Malitsky-Tam values for W and L
    n: number of agents
    
    Returns:
    W: W matrix n x n numpy array
    L: L matrix n x n numpy array
    '''
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

def getMaxConnect(n):
    '''
    Return W, L for maximum connectivity
    '''
    v = 2/(n-1)
    W = -v*np.ones((n,n))
    # Set diagonal of W to 2
    for i in range(n):
        W[i,i] = 2
    
    L = np.zeros((n,n))
    # Set lower triangle of L to v
    for i in range(n):
        for j in range(i):
            L[i,j] = v

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

def subproblem(i, data, problem_builder, W, L, comms_data, queue, gamma=0.5, itrs=501, terminate=None, verbose=False):
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
    if hasattr(resolvent, 'v0'):
        local_v = resolvent.v0
    elif isinstance(data, dict) and 'v0' in data:
        local_v = data['v0']
    else:
        local_v = np.zeros(m)
    local_r = np.zeros(m)

    # Iterate over the problem
    itr = 0
    while itr < itrs:
        if terminate is not None and terminate.value != 0:
            if terminate.value < itr:
                terminate.value = itr + 1
            itrs = terminate.value
        if verbose and itr % 1000 == 999:
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
        w_value = resolvent.prox(local_v + local_r)

        # Terminate if needed
        if i==0 and terminate is not None:
            queue['terminate'].put(w_value)
            

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
        

        local_v = local_v - gamma*(W[i,i]*w_value+v_temp)
        
        # Zero out v_temp without reallocating memory
        v_temp.fill(0)
        local_r.fill(0)
        itr += 1
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

def evaluate(terminateQueue, terminate, vartol, objtol, data, fval, itrs, verbose):
    """Evaluate the termination conditions and set the terminate value if needed
    The terminate value is set a number of iterations ahead of the convergence iteration
    
    Inputs:
    terminateQueue is the queue for termination
    terminate is the multiprocessing value for termination
    vartol is the variable tolerance
    objtol is the objective tolerance
    data is the data for the objective function
    fval is the objective function
    itrs is the number of iterations
    verbose is a boolean for verbose output


    """
    x = terminateQueue.get()
    varcounter = 0
    objcounter = 0
    itr = 0
    itrs -= 10
    while itr < itrs:
        prev_x = x
        x = terminateQueue.get()        
        if vartol is not None:
            if np.linalg.norm(x-prev_x) < vartol:
                varcounter += 1
                if varcounter >= 10:
                    terminate.value = itr + 10
                    if verbose:
                        print('Converged on vartol on iteration', itr)
                    break
            else:
                varcounter = 0
        if objtol is not None:
            if abs(fval(data, x)-fval(data, prev_x)) < objtol:
                objcounter += 1
                if objcounter >= 10:
                    terminate.value = itr + 10
                    if verbose:
                        print('Converged on objtol')
                    break
            else:
                objcounter = 0
        itr += 1
        
def parallelAlgorithm(n, data, resolvents, W, L, itrs=1001, gamma=0.5, vartol=None, objtol=None, fval=None, verbose=False):
    # Create the queues
    man = mp.Manager()
    Queue_Array, Comms_Data = requiredQueues(man, W, L)
    if vartol is not None or objtol is not None:
        terminate = man.Value('i',0) #man.Event()
        Queue_Array['terminate'] = man.Queue()
        # Create evaluation process
        if objtol is not None:
            d = data[n]
        else:
            d = None
        evalProcess = mp.Process(target=evaluate, args=(Queue_Array['terminate'], terminate, vartol, objtol, d, fval, itrs, verbose))
        evalProcess.start()
    else:
        terminate = None

    # Run subproblems in parallel
    if verbose:
        #print('Starting Parallel Algorithm')
        t = time()
    with mp.Pool(processes=n) as p:
        params = [(i, data[i], resolvents[i], W, L, Comms_Data[i], Queue_Array, 0.5, itrs, terminate, verbose) for i in range(n)]
        results = p.starmap(subproblem, params)
    if verbose:
        alg_time = time()-t
        print('Parallel Algorithm Loop Time:', alg_time)

    # Join the evaluation process
    if terminate is not None:        
        evalProcess.join()
    # Get the final value
    w = sum(results[i]['w'] for i in range(n))/n
    if verbose:
        results[0]['alg_time'] = alg_time
    return w, results

def serialAlgorithm(n, data, resolvents, W, L, itrs=1001, gamma=0.5, vartol=None, objtol=None, fval=None, verbose=False):
    
    # Initialize the resolvents and variables
    all_x = []
    all_v = []
    for i in range(n):
        resolvents[i] = resolvents[i](data[i])
        if i == 0:
            m = resolvents[0].shape
        x = np.zeros(m)
        all_x.append(x)
        if isinstance(data[i], dict) and 'v0' in data[i]:
            v = data[i]['v0']
        else:
            v = np.zeros(m)
        all_v.append(v)

    # Run the algorithm
    if verbose:
        print('Starting Serial Algorithm')
        start_time = time()
    checkperiod = 10
    if vartol is not None:
        tracker = 0
    if objtol is not None:
        objtracker = 0
        lastVal = fval(data[n], all_x[n-1])
    for itr in range(itrs):
        if verbose and itr % 10 == 0:
            print(f'Iteration {itr}')

        for i in range(n):
            resolvent = resolvents[i]
            y = all_v[i] + sum(L[i,j]*all_x[j] for j in range(i))
            all_x[i] = resolvent.prox(y)

        for i in range(n):     
            wx = sum(W[i,j]*all_x[j] for j in range(n))       
            all_v[i] = all_v[i] - gamma*wx
        
        if vartol is not None and itr%checkperiod == 0:
            if np.linalg.norm(wx) < vartol:
                tracker += 1
                checkperiod = 1
                if tracker >= n:
                    print('Converged on variable value, iteration', itr)
                    break
            else:
                tracker = 0
                checkperiod = 10

        if objtol is not None and itr%checkperiod == 0:
            newVal = fval(data[n], all_x[0])
            if abs(newVal-lastVal) < objtol:
                objtracker += 1
                checkperiod = 1
                if objtracker >= n:
                    print('Converged in objective value, iteration', itr)
                    break
            else:
                objtracker = 0
                checkperiod = 10
            lastVal = newVal
            
    if verbose:
        print('Serial Algorithm Loop Time:', time()-start_time)
    x = sum(all_x)/n
    # Build results list
    results = []
    for i in range(n):
        if hasattr(resolvents[i], 'log'):
            results.append({'w':all_x[i], 'v':all_v[i], 'log':resolvents[i].log})
        else:
            results.append({'w':all_x[i], 'v':all_v[i]})
        if verbose:
            print('results', i, 'w', all_x[i], 'v', all_v[i])
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
    '''Distribute the functions to the nodes
    Assumes even splitting
    Inputs:
    num_nodes is the number of nodes
    num_functions is the number of functions
    num_nodes_per_function is the number of nodes per function

    Returns:
    funcs_per_node is a list of length num_nodes
    Each entry is a list of the functions assigned to that node
    '''

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

def solve(n, data, resolvents, W, L, itrs=1001, gamma=0.5, vartol=None, objtol=None, fval=None, parallel=False, verbose=False):
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
        alg_x, results = parallelAlgorithm(n, data, resolvents, W, L, itrs=itrs, vartol=vartol, objtol=objtol, fval=fval, verbose=verbose)
    else:
        alg_x, results = serialAlgorithm(n, data, resolvents, W, L, itrs=itrs, vartol=vartol, objtol=objtol, fval=fval, verbose=verbose)

    return alg_x, results

def solveMT(n, data, resolvents, itrs=1001, gamma=0.5, vartol=None, objtol=None, fval=None, parallel=True, verbose=False):
    # Solve the problem with the Malitsky-Tam W and L matrices
    W, L = getMT(n)
    return solve(n, data, resolvents, W, L, itrs=itrs, gamma=gamma, vartol=vartol, objtol=objtol, fval=fval, parallel=parallel, verbose=verbose)