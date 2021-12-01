#4. Create an one dimensional array using arange function containing 10 elements.
#Display
#a. First 4 elements
#b. Last 6 elements
#c. Elements from index 2 to 7

import numpy as np
a = np.arange(1, 11, 1)
print(a)
first_element = a[:4]
print(first_element)
first_element1 = a[5:]
print(first_element1)
first_element2 = a[1:7]
print(first_element2)
