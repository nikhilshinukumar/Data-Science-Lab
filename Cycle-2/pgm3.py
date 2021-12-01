#3. Familiarize with the functions to create
#a) an uninitialized array
#b) array with all elements as 1,
#c) all elements as 0

import numpy as np
x=np.empty([2, 2])
print(x)
y=np.full((2, 2), 1)
print(y)
z=np.full((2, 2), 0)
print(z)