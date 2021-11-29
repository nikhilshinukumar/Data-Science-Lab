import numpy as np

arr_2d=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(arr_2d)

print("Display all elements excluding the first row")
print(arr_2d[1:4,:])
print("Display all elements excluding the last column")
print(arr_2d[:,0:3])
print("Display the elements of 1 st and 2 nd column in 2 nd and 3 rd row")
print(arr_2d[1:3,1:3])
print("Display the elements of 2 nd and 3 rd column")
print(arr_2d[:,1:3])
print("Display 2 nd and 3 rd element of 1 st row")
print(arr_2d[0,1:3])

arr=np.array([0,1,2,3,4,5,6,7,8,9,10])
print("Display the elements from indices 4 to 10 in descending order(use–values)")
print(arr[10:4:-1])