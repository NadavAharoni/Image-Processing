import numpy as np

a = np.array([[1], [2], [3], [4]])  # column vector (4,1)
b = np.array([[10, 20, 30, 40]])    # row vector (1,4)

print(a.shape)
print(b.shape)
print(a)
print(b)
print(a * b)
print(b * a)
print(a @ b)
print(b @ a)
print(a + b)
print(b + a)
