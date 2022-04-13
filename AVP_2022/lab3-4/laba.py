from computing.CPU import cpu
#from computing.GPU import gpu 
import numpy as np
from time import time

def main() -> None:
    x = 1300
    y = 1200
    host_matrix = np.int32(np.random.randint(0, 256, x*y).reshape(x, y))
    cpu_matrix = cpu(x, y, host_matrix)

    t1 = time()
    new_cpu_matrix = cpu_matrix.compute()
    t2 = time()

    print(cpu_matrix, new_cpu_matrix, "is the same?", new_cpu_matrix == cpu_matrix, "Computing time on CPU : {:.4f}".format(t2-t1), sep="\n") 


    
if __name__ == "__main__":
    main()