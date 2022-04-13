import numpy as np

class cpu(object):
    dim_x: np.int32 = None
    dim_y: np.int32 = None
    matrix: np.matrix = None

    def __init__(self, dim_x: np.int32 = 0, dim_y: np.int32 = 0, matrix: np.matrix = np.matrix(None)) -> None:
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.matrix = matrix[:]


    @property
    def whole_size(self) -> np.int32:
        return self.dim_x*self.dim_y


    @property
    def shape(self) -> tuple:
        return self.matrix.shape


    def compute(self) -> object:
        x = self.matrix.shape[0]
        y = self.matrix.shape[1]
        new_matrix = np.empty_like(self.matrix).reshape(self.matrix.shape[::-1])
        for i in range(x):
            for j in range(y):
                new_matrix[j][x-i-1] = (self.matrix[i][y-j-1])
        return cpu(y, x, new_matrix)        


    def __eq__(self, __o: object) -> bool:
        res: bool = False
        if isinstance(__o, (cpu)):
            if (self.dim_x, self.dim_y) == (__o.dim_x, __o.dim_y):
                if (self.matrix == __o.matrix):
                    res = True
        return res

        
    def __repr__(self) -> str:
        return """
CPU(dim_x : {x}, dim_y : {y}, whole_size : {size}, shape : {shape}
matrix :    
{matrix} 
) 
""".format(x=self.dim_x, y=self.dim_y, matrix=self.matrix, size=self.whole_size, shape=self.shape)




if __name__ == "__main__":
    x = 3
    y = 6
    CPU = cpu( np.int32(x), np.int32(y), np.float32(np.random.random_integers(0, 255, x*y).reshape(x,y)))
    new_CPU = CPU.compute()
    print(CPU, new_CPU, "is the same?", new_CPU == CPU, sep="\n")   
    