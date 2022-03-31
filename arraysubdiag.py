import matplotlib.pyplot as np
import numpy as np

x=np.array([[1,4],[3,4]])
y=np.array([[5,3],[7,6]])
print("Subtract two matrix:")
print(np.subtract(x,y))
print("Sum of the diagonal elements of a matrix : ")
print(np.trace(x))
print(np.trace(y))