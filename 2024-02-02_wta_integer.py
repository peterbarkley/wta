import oars
import wta
import numpy as np
import cvxpy as cp
from time import time
import pandas as pd
from collections import defaultdict, deque
import plotly.express as px
import json

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
    w = cp.Variable(m, integer=True)

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
   
def dynamicProx(i, q_row, y, Vj, verbose=False):
    pk_i = 1 - q_row
    tgts, wpns = y.shape
    x = np.zeros(y.shape)
    # Put the index of the max value in each column of y into an array of size wpns
    max_index = np.argmax(y, axis=0)
    if verbose:print(max_index)
    # If index i is the max value in column j, set x[i,j] = 1
    # These are weapons favored by the prox matrix y
    remaining = set()
    for j in range(wpns):
        if max_index[j] == i:
            x[i,j] = 1
        else:
            remaining.add(j)
    if verbose:print("remaining",remaining)
    # Look at remaining weapons
    y_sq = np.power(y, 2)
    y_sq_total = np.sum(y_sq, axis=0)
    y_sq_total_i = y_sq_total - y_sq[i,:] + (1-y[i,:])**2
    cost = 0.5*y_sq_total_i
    if verbose:print("cost", cost)
    alt_cost = 0.5*(y_sq_total + [(1-y[max_index[j],j])**2-y_sq[max_index[j],j] for j in range(wpns)])
    if verbose:print("alt_cost", alt_cost)
    cost_diff = cost-alt_cost
    full_val = np.prod(q_row)
    if verbose:print("full_val", full_val)
    full_val_i = (full_val/q_row)*pk_i
    if verbose:print("full_val", full_val_i)
    if verbose:print("cost_diff", cost_diff)
    high_q = set(filter(lambda j: full_val_i[j] > cost_diff[j], remaining))
    remaining = remaining.difference(high_q)
    if verbose:print("high_q", high_q)
    if verbose:print("remaining", remaining)

    # Set x[i,j] = 1 if j in remove
    # These weapons have such a high q value that they are worth the cost even if all weapons are used
    for j in high_q:
        x[i,j] = 1

    # We can get the lowest possible base value now with these weapon values set
    if verbose:print(q_row, i, x[i,:], np.power(q_row, x[i,:]), Vj)
    base_value = Vj*np.prod(np.power(q_row, x[i,:]))
    if verbose:print(base_value)

    # Find promising weapons - those with a base value higher than the cost if they are the only one added
    base_value_i = base_value*pk_i
    if verbose:print("base_value_i", base_value_i)
    bv = {}
    val = {}
    promising = []
    for j in remaining:
        v = base_value*pk_i[j] - cost_diff[j]
        if v > 0:
            bv[tuple([j])] = base_value*q_row[j]
            val[tuple([j])] = v
            promising.append(j)            
        else:
            x[max_index[j],j] = 1
    if verbose:print("promising", promising)
    if verbose:print("vals", val)
    if verbose:print("bv", bv)
    if promising:
        prom_tuples = [tuple([j]) for j in promising]
        #p = dynProgExp(bv, val, q_row, pk_i, cost_diff, max_index, prom_tuples, verbose=verbose)
        p = singleWTA(q_row, base_value, cost_diff, promising, verbose=verbose)
        for j in promising:
            if j in p:
                x[i,j] = 1
            else:
                x[max_index[j],j] = 1
    return x
    
def dynProgExp(bv, val, q_row, pk_i, cost_diff, max_index, promising, verbose=False):
    
    queue = deque()
    
    remains = {}
    prohibited = defaultdict(list)

    promising_set = set(promising)
    for j in promising:
        remains[j] = promising_set.difference({j})
    queue.extend(promising)
    new_queue = []
    tail = []

    # Dynamic programming to find the best combination of promising weapons
    while queue:
        j = queue.popleft()

        settail = True
        for k in remains[j]:
            if verbose:print("j", j, "k", k)
            
            if type(j) is int:
                jk = {j,k}
            else:
                jk = set(j)
                jk.add(k)
            tjk = tuple(jk)
            if tjk not in bv:
                new_bv = bv[j]*q_row[k]
                v = bv[j]*pk_i[k] - cost_diff[k]
                if v > 0:
                    settail = False
                    if verbose:print("v", v, "val[j]", val[j], "k", k, "j", j, "tjk", tjk)
                    val[tjk] = v + val[j]
                    bv[tjk] = new_bv
                    new_queue.append(tjk)
                else:
                    prohibited[j].append(k)
        if settail:
            tail.append(j)
        for p in new_queue:
            if verbose:print(p)
            r = promising_set.difference(p).difference({j for i in p for j in prohibited[i]})
            if r:
                queue.append(p)
                remains[tuple(p)] = r
            else:
                tail.append(p)
        new_queue = []

    # Find p in tail with the highest value
    if verbose:print("tail", tail)
    if verbose:print("val", val)
    if tail:
        p = max(tail, key=lambda x: val[x])
    else:
        p = []
    if verbose:print("p", p)
    # Set x[i,j] = 1 if j in p
    if verbose:print("last remaining", remaining)
    

    return tuple(p)

def singleWTA(q, V, cost_diff, promising, verbose=False):
    '''
    Solve the weapon-target assignment problem for a single target
    Inputs:
    q: (m,) array of survival probabilities
    V: target value (float)
    cost_diff: (m,) array of the difference between the cost of each weapon and the base value
    promising: list of promising weapons
    Outputs:
    p: list of weapons to use
    '''
    m = len(promising)
    small_q = q[promising]
    #print("small_q", small_q)
    small_c = cost_diff[promising]
    #print("small_c", small_c)
    #print("V", V)
    x = cp.Variable(m, boolean=True)
    ww = x@cp.log(small_q)
    obj = cp.Minimize(V*cp.exp(ww) + x@small_c)
    prob = cp.Problem(obj)
    prob.solve()
    if verbose:print("prob status", prob.status)
    if verbose:print("prob value", prob.value)
    if verbose:print("prob x", x.value)
    used = [i for i in range(m) if x.value[i] > 0.5]
    #print(used)
    return np.array(promising)[used]


class wtaBProx:

    def __init__(self, data):
        self.data = data
        self.shape = data['Q'].shape
        self.q = data['QQ']
        self.v = data['VV']
        self.i = data['i']

    def __call__(self, y):
        t = time()
        x = dynamicProx(self.i, self.q, y, self.v)
        return x

    def __repr__(self):
        return "WTA Better Proximal Operator"
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
        self.log = []

    def __call__(self, x):
        t = time()
        self.y.value = x
        self.prob.solve(verbose=True)
        st = time()
        self.log.append((t,st))
        # You can implement logging here
        #self.log.append(fullValue(self.data, self.w.value))
        return self.w.value

    def __repr__(self):
        return "wtaResolvent"

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

def smallSplit(n, tgts, wpns, node_tgts, num_nodes_per_tgt, L):
    '''Generate the data for the splitting'''
    Q, V, WW = wta.generate_random_problem(tgts, wpns)
    WW = np.ones(wpns)
    m = (tgts, wpns)
    data = []
    for i in range(n):
        q = Q[i, :] # Only use the targets that are in the node
        v = V[i] # Only use the targets that are in the node
        #v = v/num_nodes_per_tgt # Divide the value by the number of nodes that have the target
        v0 = 1/tgts*np.ones(m) # Initial consensus parameter - equal number of weapons per tgt
        data.append({'QQ':q, 'VV':v, 'WW':WW, 'v0':v0, 'Lii':L[i,i], 'Q':Q, 'V':V, 'i':i})

    data.append({'Q':Q, 'V':V, 'WW':WW}) # Add the data for the full problem
    return data

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

def getDataGantt(logs, title):
    n = len(logs)
    #tt, s, x = getCycleTime(t, l, L, W)
    dflist = []

    j = 0
    for log in logs: # Node logs
        i = 0
        for start,stop in log: # Operations
            if j==0 and i==0:
                offset = start
            start = start - offset
            stop = stop - offset
            #start = s[i,j]
            #stop = start + t[j]
            dflist.append(dict(Task="Iter %s" % i, Start=start, Finish=stop, Resource="Node %s" % j))
            i += 1
        j += 1
    df = pd.DataFrame(dflist)
    df['delta'] = df['Finish'] - df['Start']
    # Export df to csv
    df.to_csv(title+'.csv')

    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", color="Task")
    fig.update_yaxes(autorange="reversed") 

    fig.layout.xaxis.type = 'linear'
    for d in fig.data:
        filt = df['Task'] == d.name
        d.x = df[filt]['delta'].tolist()

    fig.update_layout(title_text=title)
    return fig

if __name__ == "__main__":
    # Problem data
    n = 4
    tgts = 4
    wpns = 150
    itrs = 200

    # Survival probabilities
    Q, V, WW = wta.generate_random_problem(tgts, wpns)
    WW = np.ones(wpns)
    # Reference values
    t = time()
    true_p, true_x = wta.wta(Q, V, WW, integer=True, verbose=True)
    true_time = time() - t
    print("true time", true_time)

    # W and L for 4 node example
    L = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1.5, 0.5, 0, 0], [0.5,1.5,0,0]])
    W = np.array([[1, 0, -1, 0], [0, 2, -0.5, -1.5], [-1, -0.5, 1.67218, -0.17218], [0,-1.5,-0.17218,1.67218]])

    # Generate splitting
    num_nodes_per_function = 1
    node_tgts = oars.distributeFunctionsLinear(n, tgts, num_nodes_per_function)
    num_nodes_per_tgt = num_nodes_per_function*np.ones(tgts) # Assumes even splitting

    # Generate the data
    data = smallSplit(n, tgts, wpns, node_tgts, num_nodes_per_tgt, L)
    #print(data)
    # WTA
    resolvents = [wtaBProx]*n
    t = time()
    alg_x, results = oars.solve(n, data, resolvents, W, L, vartol=1e-3, itrs=itrs, parallel=False, verbose=True)  
    
    print("true val")
    print(true_x) 
    print("alg time", time()-t)
    # print("alg x")
    # print(alg_x)
    # for i in range(n):
    #     print("node", i, "x")
    #     print(results[i]['w'])
    alg_p = fullValue(data[-1], alg_x)
    print("alg val", alg_p)
    print("True val", true_p)

    # Generate random solution for comparison
    # Generate random solution
    y = np.random.rand(4, wpns)
    # Set the max value in each column to 1, the rest to 0
    x = np.zeros(y.shape)
    max_index = np.argmax(y, axis=0)
    for j in range(wpns):
        x[max_index[j],j] = 1

    # Get the value of the solution
    z = wta.fullValue(Q, V, x)
    print(z)
    # logs = [results[i]['log'] for i in range(n)]
    # fig = getDataGantt(logs, "Parallel WTA")
    # #fig.show()
    # fig.write_html('parallel_wta_gantt'+str(time())+'.html')

    # Malitsky-Tam
    # log_alg = results[0]['log']
    # t = time()
    # mt_resolvents = [wtaResolvent]*n
    # mt_x, mt_results = oars.solveMT(n, data, mt_resolvents, itrs=itrs, verbose=True)
    # print("mt time", time()-t)
    # print("mt val", fullValue(data[-1], mt_x))
    # log_mt = mt_results[0]['log']
    # graphResults([log_alg, log_mt], ["New Algorithm", "Malitsky Tam"], true_p)
    

    # # Save the alg_x as a json file
    
    # with open('alg_x.json', 'w') as f:
    #     json.dump(alg_x.tolist(), f)
