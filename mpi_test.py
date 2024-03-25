#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
#import cvxpy as cp
#import oars

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

    return L, W


# def wta(q, V, W, integer=True, lasso=False, verbose=False, **kwargs):
#     """
#     Solve the weapon-target assignment problem.
#     Inputs:
#         q: (n,m) array of survival probabilities
#         V: (n,) array of target values
#         W: (m,) array of weapon counts
#         integer: boolean, whether to solve the integer or continuous problem
#         lasso: boolean, whether to solve the lasso problem
#     Outputs:
#         probval: optimal value of the problem
#         x: (n,m) array of weapon assignments
#     """
#     if len(q.shape) == 1:
#         n = q.shape[0]
#         m = 1
#         q = q.reshape((n,m))
#     else:
#         n, m = q.shape

#     # Define the CVXPY problem.
#     if integer:
#         x = cp.Variable((n,m), integer=True)
#     else:
#         x = cp.Variable((n,m))
#     weighted_weapons = cp.multiply(x, np.log(q)) # (n,m)
#     survival_probs = cp.exp(cp.sum(weighted_weapons, axis=1)) # (n,)
    
#     if lasso:
#         v = 0.1*min(V)
#         obj_fun = V@survival_probs + v*cp.sum(x)
#     else:
#         obj_fun = V@survival_probs
#     objective = cp.Minimize(obj_fun)
#     cons = [cp.sum(x, axis=0) <= W, x >= 0]

#     # Solve
#     prob = cp.Problem(objective, cons)
#     prob.solve(**kwargs)
#     print(prob.status) # Optimal
#     if verbose and prob.status == 'Optimal':
#         print("The optimal value is", prob.value)
#         print("A solution x is")
#         print(x.value)

#     return prob.value, x.value


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

#def solve(s, itrs=100, gamma=0.5, verbose=False, terminate=None):
def subproblem(i, data, pb, W, L, comms_data, comm, gamma=0.5, itrs=100, verbose=False, vartol=None, terminate=None):
    
    # comm = MPI.COMM_WORLD
    # i = comm.Get_rank()
    # size = comm.Get_size()

    # L, W = oars.getMT(size)
    # comms_data_all = requiredComms(L, W)
    # comms_data = comms_data_all[i]
    #s = 10
    resolvent = pb(data)
    s = resolvent.shape
    #s = data.shape
    buffer = np.ones(s, dtype=np.float64)
    local_v = np.zeros(s, dtype=np.float64)
    local_r = np.zeros(s, dtype=np.float64)
    v_temp = np.zeros(s, dtype=np.float64)
    n = W.shape[0]
    itr = 0
    terminated = False
    while itr < itrs:
        if vartol is not None and comm.Iprobe(source=n, tag=0):
            itrs = comm.recv(source=n, tag=0)
            terminated = True
        #     if terminate.value < itr:
        #         terminate.value = itr + 1
        #     itrs = terminate.value
        if verbose and itr % 10 == 0:
            print(f'Node {i} iteration {itr}', flush=True)

        # Get data from upstream L queue
        for k in comms_data['up_LQ']:
            req = comm.Irecv(buffer, source=k, tag=itr)
            req.Wait()
            local_r += L[i,k]*buffer

        # Pull from the B queues, update r and v_temp
        for k in comms_data['up_BQ']:
            req = comm.Irecv(buffer, source=k, tag=itr)
            req.Wait()
            local_r += L[i,k]*buffer
            v_temp += W[i,k]*buffer

        # Solve the problem
        w_value = resolvent.prox(local_v + local_r)

        # Terminate if needed
        if not terminated and i==0 and vartol is not None:
             #print(f'Node {i} w_value sending for eval: {w_value}', flush=True)
             comm.Send(w_value, dest=n, tag=itr)

            

        # Put data in downstream queues
        for k in comms_data['down_LQ']:
            comm.Isend(w_value, dest=k, tag=itr)
        for k in comms_data['down_BQ']:
            comm.Isend(w_value, dest=k, tag=itr)

        # Put data in upstream W queues
        for k in comms_data['WQ']:
            comm.Isend(w_value, dest=k, tag=itr)
        for k in comms_data['up_BQ']:
            comm.Isend(w_value, dest=k, tag=itr)

        # Update v from all W queues
        for k in comms_data['WQ']:
            req = comm.Irecv(buffer, source=k, tag=itr)
            req.Wait()
            v_temp += W[i,k]*buffer
            
        # Update v from all B queues
        for k in comms_data['down_BQ']:
            req = comm.Irecv(buffer, source=k, tag=itr)
            req.Wait()
            v_temp += W[i,k]*buffer
        #v_temp += sum([W[i,k]*queue[k,i].get() for k in comms_data['down_BQ']])
        

        local_v = local_v - gamma*(W[i,i]*w_value+v_temp)
        
        # Zero out v_temp without reallocating memory
        v_temp.fill(0)
        local_r.fill(0)
        itr += 1

    #print(f'Node {i} w_value: {w_value}', flush=True)
    # return w_value and log if it is in the resolvent
    return w_value, resolvent.log

def evaluate(s, comm, vartol=1e-7, itrs=100):
    """Evaluate the convergence of the algorithm
    s: the shape of the data
    comm: the MPI communicator
    itrs: the number of iterations to run
    
    """
    last = np.zeros(s, dtype=np.float64)
    buffer = np.zeros(s, dtype=np.float64)
    counter = 0
    itr = 0
    while counter < comm.Get_size() - 1 and counter < itrs and itr < itrs:
        comm.Recv(buffer, source=0, tag=itr)
        w = buffer.copy()
        # Print last and buffer
        #print(f'Counter: {counter}, Last: {last}, Buffer: {w}', flush=True)
        if np.linalg.norm(w - last) < vartol:
            counter += 1
        else:
            counter = 0
        last = w
        itr += 1
    # print counter, last and buff
    #print(f'Counter: {counter}, Last: {last}, Buffer: {w}', flush=True)
    print(f'Reached termination criteria on Iteration {itr}', flush=True)

    # Terminate the other processes
    terminate_itr = itr + 50 # comm.Get_size()
    if itr < itrs - 50:
        for i in range(comm.Get_size()-1):
            #print(f'Sending termination criteria {terminate_itr} to {i}', flush=True)
            comm.send(terminate_itr, dest=i, tag=0)