# Parallel WTA using MT algorithm over 3 nodes

from collections import defaultdict
import multiprocessing as mp
import numpy as np
# import cvxpy as cp
# import wta

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

def subproblem(W, L, i, WQ, up_LQ, down_LQ, up_BQ, down_BQ, queue, gamma=0.5, itrs=5, verbose=False):
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
    r = 1
    v = 0
    logv = []
    for itr in range(itrs):
        print(f'Node {i} iteration {itr}')
        # Get data from upstream L queue
        if len(up_LQ) > 0:
            print(f'Node {i} getting data from upstream L queue')
            temp = [queue[k,i].get() for k in up_LQ]
            r = temp
            logv.append(temp)
        else:
            r = 0

        # Pull from the B queues, update r and v_temp
        for k in up_BQ:
            print(f'Node {i} getting data from upstream B queue {k}')
            temp = queue[k,i].get()
            print(f'Node {i} received data from upstream B queue {k}', temp)
            logv.append(temp)
            # r += L[i,k]*temp
            # v_temp += W[i,k]*temp

        # Solve the problem
        print(f'Node {i} solving problem')
        #prob.solve(verbose=False)
        w = (i, itr)
        # Log results
        # logw.append(w.value)
        # logv.append(v.value)

        # Put data in downstream L queue
        for k in down_LQ:
            queue[i,k].put(w)
        for k in down_BQ:
            queue[i,k].put(w)

        # Put data in upstream W queue
        for k in WQ:
            queue[i,k].put(w)
        for k in up_BQ:
            queue[i,k].put(w)

        # Update v from all W queues
        logv.append([queue[k,i].get() for k in WQ])
        
        # Zero out v_temp without reallocating memory
        # v_temp.fill(0)

    # Return the solution
    return {'logv':logv,}


if __name__ == '__main__':
    mp.freeze_support()

    n = 3 # number of nodes
    # W and L for Malitsky-Tam
    M = np.array([
        [-1, 1, 0],
        [0, -1, 1]])

    W = M.T@M
    L = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]])

    # print(W)
    # print(L)

    # Create the queues
    man = mp.Manager()
    Queue_Array, WQ, up_LQ, down_LQ, up_BQ, down_BQ = requiredQueues(man, W, L)

    # Run subproblems in parallel
    with mp.Pool(processes=n) as p:
        params = [(W, L, i, WQ[i], up_LQ[i], down_LQ[i], up_BQ[i], down_BQ[i], Queue_Array) for i in range(n)]
        results = p.starmap(subproblem, params)
        print(results)

   


