import test_module_wta
import numpy as np

name = __name__

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

class absValResolvent:
    def __init__(self, data):
        self.data = data

    def __call__(self, x):
        return np.abs(x - self.data).argmin()

    def __repr__(self):
        return "absValResolvent"
        
test_module_wta.wrapper(name)

