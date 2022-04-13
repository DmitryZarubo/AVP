import pycuda.driver as cuda
import pycuda.autoinit
from  pycuda import gpuarray 
import numpy as np
from time import time
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ker = SourceModule("""
#define _X  ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )
#define _WIDTH  ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y  )
#define _XM(x)  ( (x + _WIDTH) % _WIDTH )
#define _YM(y)  ( (y + _HEIGHT) % _HEIGHT )
#define _INDEX(x,y)  ( _XM(x)  + _YM(y) * _WIDTH )

// return the number of living neighbors for a given cell                
__device__ int nbrs(int x, int y, int * in)
{
     return ( in[ _INDEX(x -1, y+1) ] + in[ _INDEX(x-1, y) ] + in[ _INDEX(x-1, y-1) ] \
                   + in[ _INDEX(x, y+1)] + in[_INDEX(x, y - 1)] \
                   + in[ _INDEX(x+1, y+1) ] + in[ _INDEX(x+1, y) ] + in[ _INDEX(x+1, y-1) ] );
}

__global__ void conway_ker(int * p_lattice, int* out)
{
   // x, y are the appropriate values for the cell covered by this thread
   int x = _X, y = _Y;
   __shared__ int lattice[32*32];
   lattice[_INDEX(x,y)] = p_lattice[_INDEX(x,y)];
   __syncthreads();
   // count the number of neighbors around the current cell
   int n = nbrs(x, y, lattice);
   int cell_value;                
    
    // if the current cell is alive, then determine if it lives or dies for the next generation.
    if ( lattice[_INDEX(x,y)] == 1)
       switch(n)
       {
          // if the cell is alive: it remains alive only if it has 2 or 3 neighbors.
          case 2:
          case 3: cell_value = 1;
                  break;
          default: cell_value = 0;                   
       }
    else if( lattice[_INDEX(x,y)] == 0 )
         switch(n)
         {
            // a dead cell comes to life only if it has 3 neighbors that are alive.
            case 3: cell_value = 1;
                    break;
            default: cell_value = 0;         
         }
         __syncthreads();
         lattice[_INDEX(x,y)] = cell_value;
         __syncthreads();
         out[_INDEX(x,y)] = lattice[_INDEX(x,y)];
         __syncthreads();
}
""")

conway_ker = ker.get_function("conway_ker")
     

def update_gpu(frameNum, img, new_lattice_gpu, lattice_gpu, N):
    block_size = 32
    conway_ker( lattice_gpu, new_lattice_gpu, grid=(N//block_size,N//block_size,1), block=(block_size,block_size,1))
    
    img.set_data(new_lattice_gpu.get() )

    lattice_gpu[:] = new_lattice_gpu[:]
    return img
    

if __name__ == '__main__':
    # set lattice size
    N = 32
    
    lattice = np.int32( np.random.choice([1,0], N*N, p=[0.25, 0.75]).reshape(N, N) )
    lattice_gpu = gpuarray.to_gpu(lattice)
    new_lattice_gpu = gpuarray.empty_like(lattice_gpu)    
    
    fig, ax = plt.subplots()
    img = ax.imshow(lattice_gpu.get(), interpolation='nearest')
    ani = animation.FuncAnimation(fig, update_gpu, fargs=(img,  new_lattice_gpu, lattice_gpu, N) , interval=0, frames=1000, save_count=1000)    
    plt.show()