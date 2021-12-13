#1. Create a square matrix with random integer values(use randint()) and use appropriate functions to find:
#i) inverse
#ii) rank of matrix
#iii) Determinant
#iv) transform matrix into 1D array
#v) eigen values and vectors

import numpy as np

matrix=np.random.randint(0,10,4).reshape(2,2)
print(matrix)
inverse=np.linalg.inv(matrix)
print("inverse of matrix")
print(inverse)
rank=np.linalg.matrix_rank(matrix)
print("rank of matrix",rank)
det=np.linalg.det(matrix)
print("Determinant of matrix",det)
array_1d=matrix.flatten()
print("transform matrix into 1D array")
print(array_1d)
eigen=np.linalg.eig(matrix)
print("eigen values and vectors")
print(eigen)

