#7.Write a program to perform the SVD of a given matrix. Also reconstruct the given matrix   from the 3 matrices obtained after performing SVD.

from numpy import array
from scipy.linalg import svd
from numpy import diag
from numpy import dot
from numpy import zeros
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# SVD
U, s, VT = svd(A)
print("first" ,U)
print("second",s)
print("3rd" ,VT)
Sigma = zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
# reconstruct matrix
B = U.dot(Sigma.dot(VT))
print(B)
