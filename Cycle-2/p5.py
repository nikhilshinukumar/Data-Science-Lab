import numpy as np

a = np.arange(0, 15, 2)
print(a)

s2 = slice(2, 8, 2)
print(a[s2])

s = slice(-1, -15, -1)
print(a[s])


ab = np.arange(1, 15, 2)
print(ab)

first_element = ab[-3::]

print(first_element)