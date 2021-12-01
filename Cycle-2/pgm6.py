#6. Create a 2 Dimensional array with 4 rows and 4 columns.

#a. Display all elements excluding the first row
#b. Display all elements excluding the last column
#c. Display the elements of 1 st and 2 nd column in 2 nd and 3 rd row
#d. Display the elements of 2 nd and 3 rd column
#e. Display 2 nd and 3 rd element of 1 st row
#f. Display the elements from indices 4 to 10 in descending order(use–values)

import numpy as np

array_2d=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(array_2d)

print("Display all elements excluding the first row")
print(array_2d[1:4,:])
print("Display all elements excluding the last column")
print(array_2d[:,0:3])
print("Display the elements of 1 st and 2 nd column in 2 nd and 3 rd row")
print(array_2d[1:3,1:3])
print("Display the elements of 2 nd and 3 rd column")
print(array_2d[:,1:3])
print("Display 2 nd and 3 rd element of 1 st row")
print(array_2d[0,1:3])

array=np.array([0,1,2,3,4,5,6,7,8,9,10])
print("Display the elements from indices 4 to 10 in descending order(use–values)")
print(array[10:4:-1])
