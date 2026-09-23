---
notion:
  title_line: "# DLC: Advanced NumPy and Shell Reference"
  role: bonus
  status: mapped
  page_id: "3d6d9fdd-1a1a-8129-8a17-fb53e6f1afe8"
  url: "https://app.notion.com/p/3d6d9fdd1a1a81298a17fb53e6f1afe8"
---

# DLC: Advanced NumPy and Shell Reference

# Direct and Transitive Dependencies

The lecture's `requirements.txt` lists only the **direct** dependencies, the packages the project's own code imports. Those packages need packages of their own, the **transitive** dependencies, and the installer adds them automatically. Installing pandas, which Lecture 04 uses, into the Lecture 03 environment shows the difference.

## Code Snippet: List Every Installed Package

```bash
uv pip install pandas==3.0.5
uv pip freeze
```

```text
numpy==2.3.3
pandas==3.0.5
python-dateutil==2.9.0.post0
six==1.17.0
```

`uv pip freeze` records everything currently installed, including packages that arrived as dependencies of what you asked for: `python-dateutil` came with pandas, `six` came with `python-dateutil`, and on Windows `tzdata` arrives too. That is a record of one environment, which is different from the hand-written list of what the project chose, so keep writing `requirements.txt` by hand.

A **lock file** records the resolved transitive dependencies with their exact versions; generate one when a project needs that complete record. In a project started with `uv init`, `uv lock` writes one named `uv.lock`, covering every platform at once.

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
difference = np.setdiff1d(arr1, arr2)            # array([1, 2]): in arr1, not arr2
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
systolic = np.array([118, 142, 127, 135, 151, 109, 131])

# Nested np.where: test the highest band first
bands = np.where(systolic >= 140, "stage 2",
                 np.where(systolic >= 130, "stage 1", "below 130"))
print(bands)
# ['below 130' 'stage 2' 'below 130' 'stage 1' 'stage 2' 'below 130'
#  'stage 1']

# np.select: the same bands as a list; the first true condition wins
conditions = [systolic >= 140, systolic >= 130]
choices = ["stage 2", "stage 1"]
print(np.select(conditions, choices, default="below 130"))  # same labels as above

# Positions where a condition is true
positions = np.where(systolic >= 140)[0]
print(positions)  # [1 4]
```

With only a condition, `np.where` returns a tuple holding one array of positions per dimension; `[0]` takes the array for this 1D input.

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

# Optional shell reference

The lecture's pipelines select and count. `tr`, `sed`, and `awk` also rewrite text as it passes through, and longer pipelines chain them.

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

These tools plot a column without leaving the terminal: a quick look at a trend, a sanity check on a pipeline's output, or a small dashboard. Lecture 07 covers plotting in Python.

### Code Snippet: Plot in the Terminal

```bash
# sparklines: Inline Unicode graphs
# Install into the active environment: uv pip install sparklines

# Visualize systolic readings inline
cut -d',' -f3 data/raw/encounters.csv | tail -n +2 | sparklines
#     Extract column 3 -> Skip header (line 1) -> Graph
#     tail -n +2 means "start at line 2" (skip the header)
# Output: one bar per record, six for this file

# Taller bars: two rows per sparkline
cut -d',' -f3 data/raw/encounters.csv | tail -n +2 | sparklines -n 2

# gnuplot: Create terminal plots (optional; many dependencies)
# Install: brew install gnuplot (Mac) or apt install gnuplot (Linux)

# Simple plot of systolic readings
cut -d',' -f3 data/raw/encounters.csv | tail -n +2 | \
  gnuplot -e "set terminal dumb; plot '-' with linespoints"

# Bar chart: count encounters by clinic
cut -d',' -f4 data/raw/encounters.csv | tail -n +2 | sort | uniq -c | \
  gnuplot -e "set terminal dumb; plot '-' using 1 with boxes"
```
