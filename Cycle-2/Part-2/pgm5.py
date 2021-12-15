#5. Write a program to check whether given matrix is symmetric or Skew Symmetric.

#Solving systems of equations with numpy
#One of the more common problems in linear algebra is solving a matrix-vector equation. 

#Here is an example. We seek the vector x that solves the equation

#A X = b          Where                                       
#And    X=A-1 b.
#Numpy provides a function called solve for solving such equations.

import numpy as np
A = np.array([[6, 1, 1],
              [4, -2, 5],
              [2, 8, 7]])
inv=np.transpose(A)
print (inv)
neg=np.negative(A)
comparison = A == inv
comparison1 = inv== neg
equal_arrays = comparison.all()
skew=comparison1.all()
if equal_arrays :
    print("Symmetric")
else:
    print("not Symmetric")
    
if skew:
    print("Skew Symmetric")
else:
    print("Not Skew Symmetric")
