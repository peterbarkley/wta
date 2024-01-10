import cvxpy as cp
import numpy as np

def wta(q, V, W, integer=True, lasso=False, verbose=False):
    """
    Solve the weapon-target assignment problem.
    Inputs:
        q: (n,m) array of survival probabilities
        V: (n,) array of target values
        W: (m,) array of weapon counts
    """
    if len(q.shape) == 1:
        n = q.shape[0]
        m = 1
        q = q.reshape((n,m))
    else:
        n, m = q.shape

    # Define the CVXPY problem.
    if integer:
        x = cp.Variable((n,m), integer=True)
    else:
        x = cp.Variable((n,m))
    weighted_weapons = cp.multiply(x, np.log(q)) # (n,m)
    survival_probs = cp.exp(cp.sum(weighted_weapons, axis=1)) # (n,)
    
    if lasso:
        v = 0.1*min(V)
        obj_fun = V@survival_probs + v*cp.sum(x)
    else:
        obj_fun = V@survival_probs
    objective = cp.Minimize(obj_fun)
    cons = [cp.sum(x, axis=0) <= W, x >= 0]

    # Solve
    prob = cp.Problem(objective, cons)
    prob.solve(verbose=verbose)
    print(prob.status) # Optimal
    print("The optimal value is", prob.value)
    print("A solution x is")
    print(x.value)

    return prob.value, x.value

def test_wta():
        
    # Define the problem data
    n = 5 # number of targets
    m = 3 # number of weapon types
    np.random.seed(1)
    q = np.random.rand(n,m)*.9 + .1 # Survival probability
    V = np.random.rand(n)*100 # Value of each target
    W = np.random.randint(1,10,m) # Number of weapons of each type

    probval, x = wta(q, V, W)
    assert(np.allclose(probval, 2.309884155995462))
    assert(np.allclose(x, [[1., 0., 2.],
       [0., 3., 0.],
       [4., 0., 0.],
       [0., 4., 0.],
       [3., 0., 0.]]))

class platform:
    """
    A platform is a vehicle with a location.
    """
    def __init__(self, platformtype, lat, lon):
        self.platformtype = platformtype
        self.lat = lat
        self.lon = lon

    def __repr__(self):
        return f"platform({self.platformtype}, {self.lat}, {self.lon})"
    # Map
    BASE_URL = 'https://raw.githubusercontent.com/peterbarkley/wta/main/images/'
    FERRY_URL = BASE_URL + 'ferry_right.png'
    SHIP_URL = BASE_URL + 'warship_054A_down_right.png'
    SUB_URL = BASE_URL + 'sub_up_left.png'
    UAV_URL = BASE_URL + 'drone_icon_up_left.png'
    USV_URL = BASE_URL + 'ship_icon_up_left.png'

    # Dictionary of URLs by platform
    ICON_URL = {
        "ferry": FERRY_URL,
        "ship": SHIP_URL,
        "sub": SUB_URL,
        "uav": UAV_URL,
        "uuv": SUB_URL,
        "usv": USV_URL,
    }

    def get_icon_url(self):
        if self.platformtype in self.ICON_URL:
            return self.ICON_URL[self.platformtype]
        else:
            return self.ICON_URL["ship"]

class weapon(platform):

    def __init__(self, weapontype, lat, lon, count):
        super().__init__(weapontype, lat, lon)
        self.count = count


class target(platform):
    """
    A target is a platform with a value.
    """
    def __init__(self, targettype, lat, lon, value):
        super().__init__(targettype, lat, lon)
        self.value = value

    def __repr__(self):
        return f"target({self.platformtype}, {self.lat}, {self.lon}, {self.value})"

def get_final_surv_prob(q, x):
    """
    Get the final probability of kill for each target.
    Inputs:
        q: (n,m) array of survival probabilities
        x: (n,m) array of weapon assignments
    """
    return np.prod(np.power(q, x), axis=1)

def get_ind_value(q, V, W):
    """
    Get the total value if each platform solves independently.
    Inputs:
        q: (n,m) array of survival probabilities
        V: (n,) array of target values
        W: (m,) array of weapon counts
    """
    # Loop through platforms
    n, m = q.shape
    x = np.zeros((n,m))
    for i in range(m):
        # Solve the WTA problem for platform i
        q_i = q[:,i]
        pv, x_i = wta(q_i, V, W[i])
        x[:,i] = x_i[:,0]
    return V@get_final_surv_prob(q, x), x