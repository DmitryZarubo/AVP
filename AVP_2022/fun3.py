import pycuda.driver as cuda
import pycuda.autoinit
from  pycuda import gpuarray 
import numpy as np
from time import time
from pycuda.compiler import SourceModule

gpu_kernel = SourceModule("""
__global__ void mul_kernel(float* in, float scalar, float* out){
    int i = threadIdx.x;
    out[i] = scalar*in[i];
}    
    """
)
#size of the array
N = 512

host_data = np.float32( np.random.random(N))

def speed_test():
    t1 = time()
    host_res = host_data * np.float32(2)
    t2 = time()
    print("CPU computing consumed : %f" % (t2-t1))

    device_data = gpuarray.to_gpu(host_data)
    device_res = gpuarray.empty_like(device_data)
    t1 = time()
    gpu_kernel_fun = gpu_kernel.get_function("mul_kernel")
    gpu_kernel_fun(device_data, np.float32(2), device_res, block=(N,1,1), grid=(1,1,1))
    t2 = time()
    to_host_res = device_res.get()
    print("GPU computing consumer : %f" %(t2 - t1))
    print("Is resulting matrixes are the same ? {}\n{host_res}\n{to_host_res}".format(np.allclose(to_host_res, host_res), host_res=host_res, to_host_res=to_host_res))


if __name__ == "__main__":
    speed_test()
