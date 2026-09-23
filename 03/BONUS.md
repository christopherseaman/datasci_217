---
notion:
  title_line: "# DLC: Advanced NumPy and Shell Reference"
  role: bonus
  status: mapped
  page_id: "3d6d9fdd-1a1a-8129-8a17-fb53e6f1afe8"
  url: "https://app.notion.com/p/3d6d9fdd1a1a81298a17fb53e6f1afe8"
---

# DLC: Advanced NumPy and Shell Reference

Advanced NumPy topics and optional shell-processing reference.

# Advanced Universal Functions (ufuncs)

Beyond the square roots and exponentials in the lecture, these transform a whole array at once: logs compress a skewed lab value, and the trigonometric functions handle periodic signals.

## Code Snippet: Apply Less Common Math Functions

```python
import numpy as np

arr = np.array([1, 4, 9, 16, 25])

# Advanced mathematical functions
log_arr = np.log(arr)                # Natural log
log10_arr = np.log10(arr)            # Base-10 log
sin_arr = np.sin(np.pi * arr)        # Trigonometric
```

# Advanced Broadcasting

The lecture broadcast a single number across an array. The same rules combine arrays of different shapes, such as subtracting one baseline per column from every row.

## Code Snippet: Broadcast a Row Against a Column

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

Use these to combine arrays that arrived separately, such as one array per clinic visit, into a single array.

## Code Snippet: Stack Arrays

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

Matrix arithmetic is the machinery under regression and other models; Lecture 10 uses libraries that call these routines for you.

## Code Snippet: Multiply, Invert, and Solve

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

Use these when a selection needs a grid of chosen rows and columns at once, or when an array has more than two dimensions.

## Code Snippet: Select Grids and Higher Dimensions

```python
# Using np.ix_ for outer indexing
arr = np.arange(20).reshape(4, 5)
rows = [0, 2]
cols = [1, 3, 4]
result = arr[np.ix_(rows, cols)]

# Using ellipsis for arbitrary dimensions
arr_3d = np.random.default_rng(0).standard_normal((2, 3, 4))
result = arr_3d[..., 0]  # Same as arr_3d[:, :, 0]
```

# Random Number Generation

The lecture drew random integers. The same generator draws from named distributions and samples from a list, which is how you simulate a study population or pick a random subset of patient IDs.

## Code Snippet: Draw from Distributions and Samples

```python
# Modern random number generation (NumPy 1.17+)
from numpy.random import default_rng
rng = default_rng(seed=42)

# Generate random arrays
uniform = rng.uniform(0, 1, size=(3, 3))       # Uniform [0, 1)
normal = rng.normal(0, 1, size=(3, 3))         # Normal distribution

# Random sampling
choices = rng.choice([1, 2, 3, 4, 5], size=10, replace=True)
shuffled = rng.permutation([1, 2, 3, 4, 5])

# Legacy interface (still works)
np.random.seed(42)
old_style = np.random.randn(3, 3)
```

# Set Operations

Use these to compare two ID lists, such as patients enrolled at both sites.

## Code Snippet: Compare Two ID Lists

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
is_member = np.isin(arr1, arr2)                  # Boolean array
```

# Advanced Sorting

The lecture sorted values and positions for a whole array. Use these when you need only the k smallest values, or when each row of a table must be sorted on its own.

## Code Snippet: Partial and Row-Wise Sorting

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Partial sort (find k smallest/largest)
k = 3
partition_indices = np.argpartition(arr, k)      # k smallest at start
k_smallest = np.sort(arr[partition_indices[:k]]) # Get k smallest, sorted

# 2D sorting
arr_2d = np.array([[3, 2, 1], [6, 5, 4]])
sorted_2d = np.sort(arr_2d, axis=1)              # Sort each row
```

# File I/O Operations

Use these to save an array between runs without writing and re-parsing a CSV each time; `.npy` keeps the dtype exactly.

## Code Snippet: Save and Load Arrays

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

The lecture labeled values with a single `np.where`. Use these when one test is not enough: several bands at once, or the positions rather than the labels.

## Code Snippet: Multiple Conditions and Positions

```python
arr = np.array([1, 5, 3, 8, 2, 9, 4])

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

One array normally holds one dtype. A structured array holds columns of different types, which is the job the pandas DataFrame does more conveniently from Lecture 04 on.

## Code Snippet: Build a Record Array

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

For working with arrays larger than RAM: the array stays on disk and NumPy reads only the parts you touch.

## Code Snippet: Work with an On-Disk Array

```python
# Create memory-mapped file
shape = (1000000, 100)  # 800,000,000 bytes (~800 MB) of float64 storage
mmap_array = np.memmap('large_array.dat', dtype='float64', mode='w+', shape=shape)

# Use like normal array (but stored on disk)
mmap_array[0] = np.random.default_rng(0).standard_normal(100)
mmap_array.flush()  # Write to disk

# Load existing memory-mapped file
loaded_mmap = np.memmap('large_array.dat', dtype='float64', mode='r', shape=shape)
```

These advanced topics are useful for specialized applications but not required for daily data science work.

# Optional shell reference

This reference extends the core activity with `tr`, `sed`, `awk`, and longer pipelines. The canonical visualization lecture is Lecture 07.

The advanced examples below are optional reference only.

## Optional: Advanced Processing

### Code Snippet: Transform Text with tr, sed, and awk

```bash
# tr: Translate characters
tr 'a-z' 'A-Z' < file.txt       # Uppercase
tr -d ' ' < file.txt            # Delete spaces

# sed: Stream editor
sed 's/old/new/g' file.txt      # Replace all
sed '/pattern/d' file.txt       # Delete lines

# awk: Pattern processing
awk '{print $1, $3}' file.txt   # Print columns 1, 3
awk -F',' '$3 > 50' data.csv    # Filter rows
```

## Optional: Longer Data Pipelines

### Code Snippet: Chain Several Stages

```bash
# Complex pipeline
cat data.csv | \
  cut -d',' -f2,4 | \
  tr '[:lower:]' '[:upper:]' | \
  sort -t',' -k2,2n | \
  head -n 10 > results.csv
```

## Optional reference: Quick Data Visualization

Terminal visualization is also optional/reference-only. Lecture 07 is the canonical place for visualization; these commands are included only as a quick shell-based supplement.

Command line tools for quick data visualization without leaving the terminal.

### Code Snippet: Plot in the Terminal

```bash
# sparklines: Inline Unicode graphs
# Install: pip install sparklines

# Visualize systolic readings inline
cut -d',' -f3 data/raw/encounters.csv | tail -n +2 | sparklines
#     Extract column 3 -> Skip header (line 1) -> Graph
#     tail -n +2 means "start at line 2" (skip the header)
# Output: ▅█▃▆▇▄▇▂▆▅

# With statistics
cut -d',' -f3 data/raw/encounters.csv | tail -n +2 | sparklines --stat-min --stat-max --stat-mean

# gnuplot: Create terminal plots (optional - many dependencies)
# Install: brew install gnuplot (Mac) or apt install gnuplot (Linux)

# Simple plot of systolic readings
cut -d',' -f3 data/raw/encounters.csv | tail -n +2 | \
  gnuplot -e "set terminal dumb; plot '-' with linespoints"

# Bar chart: count encounters by clinic
cut -d',' -f4 data/raw/encounters.csv | tail -n +2 | sort | uniq -c | \
  gnuplot -e "set terminal dumb; plot '-' using 1 with boxes"
```

Use cases:

- Quick trend checks in terminal sessions
- Data quality sanity checks
- Pipeline debugging visualization
- Terminal dashboards
