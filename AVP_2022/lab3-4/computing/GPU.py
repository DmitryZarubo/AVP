import pycuda.driver as cuda
import pycuda.autoinit
from  pycuda import gpuarray 
import numpy as np
from pycuda.compiler import SourceModule

class gpu(object):
    dim_x: np.int32 = None
    dim_y: np.int32 = None
    matrix: gpuarray = None

    
