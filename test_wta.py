import oars
import wta
import numpy as np
import cvxpy as cp
from time import time
import pandas as pd
from collections import defaultdict
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

    # Not implemented
    def __call__(self, x):
        return None

    def prox(self, x):
        t = time()
        self.y.value = x
        self.prob.solve(verbose=False)
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
    tgts = 120
    wpns = 10
    itrs = 50

    # Survival probabilities
    Q, V, WW = wta.generate_random_problem(tgts, wpns)

    # Reference values
    t = time()
    true_p, true_x = wta.wta(Q, V, WW, integer=False, verbose=False)
    true_time = time() - t

    # W and L for 4 node example
    L = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1.5, 0.5, 0, 0], [0.5,1.5,0,0]])
    W = np.array([[1, 0, -1, 0], [0, 2, -0.5, -1.5], [-1, -0.5, 1.67218, -0.17218], [0,-1.5,-0.17218,1.67218]])

    # Generate splitting
    num_nodes_per_function = 2
    node_tgts = oars.distributeFunctionsLinear(n, tgts, num_nodes_per_function)
    num_nodes_per_tgt = num_nodes_per_function*np.ones(tgts) # Assumes even splitting

    # Generate the data
    data = generateSplitData(n, tgts, wpns, node_tgts, num_nodes_per_tgt, L)
    data.append({'Q':Q, 'V':V, 'WW':WW})
    
    # Serial WTA
    resolvents = [wtaResolvent]*n
    t = time()
    alg_x, results = oars.solve(n, data, resolvents, W, L, itrs=itrs, vartol=1e-5, parallel=False, verbose=True)   
    logs = [results[i]['log'] for i in range(n)]
    fig = getDataGantt(logs, "Serial WTA")
    #fig.show()
    fig.write_html('serial_wta_gantt'+str(time())+'.html')

    print("alg time", time()-t)
    print("direct time", true_time)
    
    alg_p = fullValue(data[-1], alg_x)
    print("alg val", alg_p)
    print("True val", true_p)

    # Parallel WTA
    resolvents = [wtaResolvent]*n
    t = time()
    alg_x, results = oars.solve(n, data, resolvents, W, L, itrs=itrs, objtol=1e-5, fval=fullValue, parallel=True, verbose=True)   
    print("alg time", time()-t)
    
    alg_p = fullValue(data[-1], alg_x)
    print("alg val", alg_p)
    print("True val", true_p)
    logs = [results[i]['log'] for i in range(n)]
    fig = getDataGantt(logs, "Parallel WTA")
    #fig.show()
    fig.write_html('parallel_wta_gantt'+str(time())+'.html')

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
