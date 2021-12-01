lower = int(input("Enter Lower Bound:"))
upper = int(input("Enter Upper Bound:"))
list1 = []
for num in range(lower,upper + 1):
   if num > 1:
       for i in range(2,num):
           if (num % i) == 0:
              print(num)
              break