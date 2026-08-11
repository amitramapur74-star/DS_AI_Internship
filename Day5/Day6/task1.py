import numpy as np

# Create two matrices
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

# Matrix multiplication using np.dot()
matrix_product = np.dot(A, B)
print("\nMatrix Multiplication (np.dot):")
print(matrix_product)

# Element-wise multiplication using *
element_product = A * B
print("\nElement-wise Multiplication (*):")
print(element_product)

# Check shapes
print("\nShape of A:", A.shape)
print("Shape of B:", B.shape)
print("Shape of Matrix Product:", matrix_product.shape)

# Swap the matrices
swapped_product = np.dot(B, A)
print("\nAfter Swapping A and B:")
print(swapped_product)