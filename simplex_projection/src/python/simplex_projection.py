from ctypes import c_void_p, c_long, c_double, cdll
from numpy.ctypeslib import ndpointer
import numpy as np
import os
this_path = os.path.dirname(__file__)

#import the library.

plib = cdll.LoadLibrary(os.path.join(this_path, '..', '..', 'build', 'parallel_simplex_proj.so'))
slib = cdll.LoadLibrary(os.path.join(this_path, '..', '..', 'build', 'serial_simplex_proj.so'))

#argument types of swap_min_max
plib.simplex_proj.argtypes = [ndpointer(c_double, ndim=2, flags="C_CONTIGUOUS"),\
                            ndpointer(c_double, ndim=2, flags="C_CONTIGUOUS"),
                            ndpointer(c_double, ndim=2, flags="C_CONTIGUOUS"),
                            c_long,
                            c_long]
slib.simplex_proj.argtypes = [ndpointer(c_double, ndim=2, flags="C_CONTIGUOUS"),\
                            ndpointer(c_double, ndim=2, flags="C_CONTIGUOUS"),
                            ndpointer(c_double, ndim=2, flags="C_CONTIGUOUS"),
                            c_long,
                            c_long]

#return type
plib.simplex_proj.restype = c_void_p
slib.simplex_proj.restype = c_void_p

def simplex_proj_parallel(arr, u, cumsum):
    '''arr, u, and cumsum are all double precision arrays of dimension (m,n)'''
    plib.simplex_proj(arr, u, cumsum, arr.shape[0], arr.shape[1])

def simplex_proj_serial(arr, u, cumsum):
    '''arr, u, and cumsum are all double precision arrays of dimension (m,n)'''
    slib.simplex_proj(arr, u, cumsum, arr.shape[0], arr.shape[1])

