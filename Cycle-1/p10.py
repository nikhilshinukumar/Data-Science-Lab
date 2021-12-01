rows = int(input("Enter the Number of rows : "))
column = int(input("Enter the Number of Columns: "))

print("Enter the elements of First Matrix:")
X = [[int(input()) for i in range(column)] for i in range(rows)]
print("First Matrix is: ")
for n in X:
    print(n)

print("Enter the elements of Second Matrix:")
Y = [[int(input()) for i in range(column)] for i in range(rows)]
for n in Y:
    print(n)

result = [[0 for i in range(column)] for i in range(rows)]

for i in range(len(X)):
   for j in range(len(X[0])):
       result[i][j] = X[i][j] + Y[i][j]

for r in result:
   print(r)