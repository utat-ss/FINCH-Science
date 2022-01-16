import numpy as np

"""
# horizontal vector
print(np.full((1, 20), 2))

# vertical vector
print(np.full((20, 1), 2))

x = np.linspace(0,1,num=11)
print(x)

a = np.array([1, 2, 3])
print(a)
print(a.shape)
print(np.transpose(a))
print(a[np.newaxis])
print(a[np.newaxis].shape)
print(np.transpose(a[np.newaxis]))
"""

print(np.cov([[1, 2], [3, 4], [5, 6]]))
