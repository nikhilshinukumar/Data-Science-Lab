#2. Create a 2 dimensional array (2X3) with elements belonging to complex data type and print it. Also display
#a. the no: of rows and columns
#b. dimension of an array
#c. reshape the same array to 3X2

import numpy as np
x = np.array([[2, 4, 6], [6.5, 8, 10]])
print(type(x))	
print(x)
numOfRows = np. size(x, 0)
print(numOfRows)
numOfColumns = np. size(x, 1)
print(numOfColumns)
print("No. of dimensions: ", x.ndim)
rs=np.reshape(x, (3, 2))
print(rs)
