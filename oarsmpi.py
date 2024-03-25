from mpi4py import MPI
import numpy as np
from time import time


# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

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
    return L, W

def requiredComms(L, W):
    '''
    Returns a dictionary of the communications required by the given W and L matrices
    Inputs:
    L is the L matrix
    W is the W matrix

    Returns:
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
            if not np.isclose(W[i,j], 0, atol=1e-3):
                if not np.isclose(L[i,j], 0, atol=1e-3):
                    comms_i['up_BQ'].append(j)
                    comms_j['down_BQ'].append(i)
                else:
                    comms_j['WQ'].append(i)
                    comms_i['WQ'].append(j)
            elif not np.isclose(L[i,j], 0, atol=1e-3):
                comms_i['up_LQ'].append(j)
                comms_j['down_LQ'].append(i)

    return Comms_Data

def subproblem(i, data, problem_builder, W, L, comms_data, comm, gamma=0.5, itrs=501, terminate=None, verbose=False):
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
    v_temp = np.zeros(m, dtype=np.float64)
    buffer = np.zeros(m, dtype=np.float64)
    if isinstance(data, dict) and 'v0' in data:
        local_v = data['v0']
    else:
        local_v = np.zeros(m, dtype=np.float64)
    local_r = np.zeros(m, dtype=np.float64)

    # Iterate over the problem
    itr = 0
    while itr < itrs:
        if terminate is not None and terminate.value != 0:
            if terminate.value < itr:
                terminate.value = itr + 1
            itrs = terminate.value
        if verbose and itr % 10 == 0:
            print(f'Node {i} iteration {itr}')

        # Get data from upstream L queue
        for k in comms_data['up_LQ']:
            comm.Recv(buffer, source=k, tag=itr)
            local_r += L[i,k]*buffer
            
        # Pull from the B queues, update r and v_temp
        for k in comms_data['up_BQ']:
            comm.Recv(buffer, source=k, tag=itr)
            local_r += L[i,k]*buffer
            v_temp += W[i,k]*buffer

        # Solve the problem
        w_value = resolvent.prox(local_v + local_r)

        # Terminate if needed
        if i==0 and terminate is not None:
            comm.send(w_value, dest=n, tag=itr)            

        # Put data in downstream queues
        for k in comms_data['down_LQ']:
            comm.Send(w_value, dest=k, tag=itr)
        for k in comms_data['down_BQ']:
            comm.Send(w_value, dest=k, tag=itr)

        # Put data in upstream W queues
        for k in comms_data['WQ']:
            comm.Send(w_value, dest=k, tag=itr)
        for k in comms_data['up_BQ']:
            comm.Send(w_value, dest=k, tag=itr)

        # Update v from all W queues
        for k in comms_data['WQ']:
            comm.Recv(buffer, source=k, tag=itr)
            v_temp += W[i,k]*buffer
            
        # Update v from all B queues
        for k in comms_data['down_BQ']:
            comm.Recv(buffer, source=k, tag=itr)
            v_temp += W[i,k]*buffer
        
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
    # if hasattr(resolvent, 'log'):
    #     if i != 0:
    #         comm.Send(resolvent.log, dest=0, tag='FinalLog')
    
    return w_value

def evaluate(terminateQueue, terminate, vartol, objtol, data, fval, itrs, verbose):
    '''Evaluate the termination conditions'''
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
                        print('Converged on vartol')
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
        

def parallelAlgorithm(comm, n, data, resolvents, W, L, itrs=1001, gamma=0.5, vartol=None, objtol=None, fval=None, verbose=False):
    # Create the communicator
    #comm = MPI.COMM_WORLD
    i = comm.Get_rank()
    size = comm.Get_size()
    Comms_Data = requiredComms(L, W)
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
        print('Starting Parallel Algorithm node ', i)
        t = time()
    w = subproblem(i, data[i], resolvents[i], W, L, Comms_Data[i], comm, 0.5, itrs, terminate, verbose)
    if verbose:
        print(f'Node {i} Parallel Algorithm Loop Time: {time()-t}')

    # Join the evaluation process
    if terminate is not None:        
        evalProcess.join()
    # Get the final valueelif i != 0:
    if i != 0:
        comm.send(w, dest=0, tag=0)
    elif i == 0:
        results = []
        results.append({'w':w})
        for k in range(1, size):
            w_i = comm.recv(source=k, tag=0)
            results.append({'w':w_i})
            w += w_i
        w /= size
        return w, results

def serialAlgorithm(n, data, resolvents, W, L, itrs=1001, gamma=0.5, vartol=None, objtol=None, fval=None, verbose=False):
    
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
        start_time = time()
    checkperiod = 10
    if vartol is not None:
        tracker = 0
    if objtol is not None:
        objtracker = 0
        lastVal = fval(data[n], all_x[0])
    for itr in range(itrs):
        if verbose and itr % 10 == 0:
            print(f'Iteration {itr}')

        for i in range(n):
            resolvent = resolvents[i]
            y = all_v[i] + sum(L[i,j]*all_x[j] for j in range(i))
            all_x[i] = resolvent(y)

        for i in range(n):     
            t = sum(W[i,j]*all_x[j] for j in range(n))       
            all_v[i] = all_v[i] - gamma*t
        
        if vartol is not None and itr%checkperiod == 0:
            if np.linalg.norm(t) < vartol:
                tracker += 1
                checkperiod = 1
                if tracker >= n:
                    print('Converged')
                    break
            else:
                tracker = 0
                checkperiod = 10

        if objtol is not None and itr%checkperiod == 0 and fval is not None:
            newVal = fval(data[n], all_x[0])
            if abs(newVal-lastVal) < objtol:
                objtracker += 1
                checkperiod = 1
                if objtracker >= n:
                    print('Converged')
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

def solve(comm, n, data, resolvents, W, L, itrs=1001, gamma=0.5, vartol=None, objtol=None, fval=None, parallel=True, verbose=False):
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
        alg_x, results = parallelAlgorithm(comm, n, data, resolvents, W, L, itrs=itrs, vartol=vartol, objtol=objtol, fval=fval, verbose=verbose)
    elif comm.Get_rank() == 0:
        alg_x, results = serialAlgorithm(n, data, resolvents, W, L, itrs=itrs, vartol=vartol, objtol=objtol, fval=fval, verbose=verbose)

    if comm.Get_rank() == 0:
        return alg_x, results

def solveMT(comm, n, data, resolvents, itrs=1001, gamma=0.5, vartol=None, objtol=None, fval=None, parallel=True, verbose=False):
    # Solve the problem with the Malitsky-Tam W and L matrices
    W, L = getMT(n)
    return solve(comm, n, data, resolvents, W, L, itrs=itrs, gamma=gamma, vartol=vartol, objtol=objtol, fval=fval, parallel=parallel, verbose=verbose)