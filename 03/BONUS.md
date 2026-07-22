NumPy Bonus Content: Advanced Topics

This file contains advanced NumPy topics beyond daily data science operations.

# Advanced Universal Functions (ufuncs)

**Reference:**

```python
import numpy as np

arr = np.array([1, 4, 9, 16, 25])

# Advanced mathematical functions
sqrt_arr = np.sqrt(arr)              # Square root
log_arr = np.log(arr)                # Natural log
log10_arr = np.log10(arr)            # Base-10 log
exp_arr = np.exp([1, 2, 3])          # Exponential
sin_arr = np.sin(np.pi * arr)        # Trigonometric
```

# Advanced Broadcasting

**Reference:**

```python
# Broadcasting 1D to 2D
row = np.array([1, 2, 3])
col = np.array([[1], [2]])
result = row + col          # Shape (2, 3)

# Broadcasting rules:
# 1. If arrays have different dimensions, prepend 1s to smaller shape
# 2. Arrays are compatible if dimensions are equal or one is 1
# 3. After broadcasting, each array behaves as if it had shape equal to elementwise max
```

# Array Stacking and Concatenation

**Reference:**

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Stacking
vstacked = np.vstack([arr1, arr2])   # Vertical: shape (2, 3)
hstacked = np.hstack([arr1, arr2])   # Horizontal: shape (6,)
dstacked = np.dstack([arr1, arr2])   # Depth: shape (1, 3, 2)

# Concatenation with axis
arr_2d1 = np.array([[1, 2], [3, 4]])
arr_2d2 = np.array([[5, 6], [7, 8]])
concatenated = np.concatenate([arr_2d1, arr_2d2], axis=0)  # Stack rows
```

# Linear Algebra Operations

**Reference:**

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B                   # or np.dot(A, B)
C_alt = np.matmul(A, B)     # Alternative

# Matrix operations
det = np.linalg.det(A)      # Determinant
inv = np.linalg.inv(A)      # Matrix inverse
rank = np.linalg.matrix_rank(A)  # Matrix rank

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solve linear system Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
```

# Advanced Indexing

**Reference:**

```python
# Using np.ix_ for outer indexing
arr = np.arange(20).reshape(4, 5)
rows = [0, 2]
cols = [1, 3, 4]
result = arr[np.ix_(rows, cols)]

# Using ellipsis for arbitrary dimensions
arr_3d = np.random.randn(2, 3, 4)
result = arr_3d[..., 0]  # Same as arr_3d[:, :, 0]
```

# Random Number Generation

**Reference:**

```python
# Modern random number generation (NumPy 1.17+)
from numpy.random import default_rng
rng = default_rng(seed=42)

# Generate random arrays
uniform = rng.uniform(0, 1, size=(3, 3))       # Uniform [0, 1)
normal = rng.normal(0, 1, size=(3, 3))         # Normal distribution
integers = rng.integers(1, 10, size=(3, 3))    # Random integers

# Random sampling
choices = rng.choice([1, 2, 3, 4, 5], size=10, replace=True)
shuffled = rng.permutation([1, 2, 3, 4, 5])

# Legacy interface (still works)
np.random.seed(42)
old_style = np.random.randn(3, 3)
```

# Set Operations

**Reference:**

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([3, 4, 5, 6, 7])

# Set operations
unique_vals = np.unique(arr1)                    # Unique values
intersection = np.intersect1d(arr1, arr2)        # array([3, 4, 5])
union = np.union1d(arr1, arr2)                   # array([1, 2, 3, 4, 5, 6, 7])
difference = np.setdiff1d(arr1, arr2)            # array([1, 2]) - in arr1 not arr2
symmetric_diff = np.setxor1d(arr1, arr2)         # Elements in one but not both

# Test membership
is_member = np.in1d(arr1, arr2)                  # Boolean array
```

# Advanced Sorting

**Reference:**

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Sorting
sorted_arr = np.sort(arr)                        # Returns sorted copy
arr.sort()                                       # Sorts in place

# Indirect sort (get indices)
sorted_indices = np.argsort(arr)                 # Indices that would sort
original_arr = arr[sorted_indices]               # Reconstruct sorted array

# Partial sort (find k smallest/largest)
k = 3
partition_indices = np.argpartition(arr, k)      # k smallest at start
k_smallest = np.sort(arr[partition_indices[:k]]) # Get k smallest, sorted

# 2D sorting
arr_2d = np.array([[3, 2, 1], [6, 5, 4]])
sorted_2d = np.sort(arr_2d, axis=1)              # Sort each row
```

# File I/O Operations

**Reference:**

```python
# Save and load arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Binary format (fast, preserves dtype)
np.save('data.npy', arr)
loaded = np.load('data.npy')

# Text format (human-readable)
np.savetxt('data.txt', arr, fmt='%d')
loaded_txt = np.loadtxt('data.txt', dtype=int)

# CSV with header
np.savetxt('data.csv', arr, delimiter=',', header='col1,col2,col3', comments='')
loaded_csv = np.loadtxt('data.csv', delimiter=',', skiprows=1)

# Multiple arrays in one file
np.savez('arrays.npz', arr1=arr, arr2=arr*2)
loaded_dict = np.load('arrays.npz')
arr1 = loaded_dict['arr1']
arr2 = loaded_dict['arr2']

# Compressed format
np.savez_compressed('arrays_compressed.npz', arr1=arr, arr2=arr*2)
```

# Conditional Logic with np.where

**Reference:**

```python
arr = np.array([1, 5, 3, 8, 2, 9, 4])

# np.where for conditional replacement
result = np.where(arr > 5, arr, 0)               # Keep if >5, else 0
result = np.where(arr > 5, 'high', 'low')        # String labels

# Multiple conditions
result = np.where(arr > 7, 'high',
                 np.where(arr > 4, 'medium', 'low'))

# Get indices where condition is true
indices = np.where(arr > 5)[0]                   # Returns tuple of arrays

# np.select for multiple conditions
conditions = [arr < 3, arr < 6, arr >= 6]
choices = ['low', 'medium', 'high']
result = np.select(conditions, choices, default='unknown')
```

# Structured Arrays

**Reference:**

```python
# Define structured array dtype
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('score', 'f8')])

# Create structured array
data = np.array([('Alice', 25, 92.5),
                 ('Bob', 30, 87.3),
                 ('Charlie', 28, 95.1)], dtype=dt)

# Access fields
names = data['name']
ages = data['age']

# Access individual records
alice = data[0]
alice_score = data[0]['score']

# Sort by field
sorted_data = np.sort(data, order='score')
```

# Memory-Mapped Files

For working with arrays larger than RAM:

**Reference:**

```python
# Create memory-mapped file
shape = (1000000, 100)
mmap_array = np.memmap('large_array.dat', dtype='float64', mode='w+', shape=shape)

# Use like normal array (but stored on disk)
mmap_array[0] = np.random.randn(100)
mmap_array.flush()  # Write to disk

# Load existing memory-mapped file
loaded_mmap = np.memmap('large_array.dat', dtype='float64', mode='r', shape=shape)
```

These advanced topics are useful for specialized applications but not required for daily data science work.
