NumPy Arrays and Professional Python Development

Welcome to week 3! Now that you've mastered Python basics and Git, it's time to dive into NumPy - the foundation of all scientific computing in Python. We'll also learn professional development practices including virtual environments, type checking, and debugging techniques.

**Note:** This lecture covers NumPy arrays (the building blocks of pandas), virtual environments for project isolation, and debugging skills that will save you hours of frustration.

By the end of today, you'll understand how NumPy makes Python 10-100x faster for numerical computing, manage Python environments like a pro, and debug code effectively. These are the essential skills that separate hobbyist Python from professional data science.

![xkcd 353: Python](media/xkcd_353.png)

Python's data structures really are that elegant and powerful - especially when you add NumPy to the mix!

# Why NumPy Matters for Data Science

## The Python Performance Problem

Python is famously slow for numerical computing. Consider multiplying a million numbers by 2:

```python
# Pure Python approach (SLOW)
my_list = list(range(1_000_000))
result = [x * 2 for x in my_list]  # 46.4 ms per loop
```

Now compare with NumPy:

```python
# NumPy approach (FAST)
import numpy as np
my_array = np.arange(1_000_000)
result = my_array * 2  # 0.3 ms per loop - 150x faster!
```

**NumPy is 10-100x faster** than pure Python for numerical operations. This isn't just optimization - it's the difference between "works for homework" and "works for real data science."

<!-- FIXME: Add visual showing performance comparison chart (Python vs NumPy speed) -->

## The NumPy Solution

NumPy provides:

- **ndarray**: Fast, memory-efficient multidimensional arrays
- **Vectorized operations**: Apply functions to entire arrays at once
- **Broadcasting**: Smart handling of different-sized arrays
- **Universal functions (ufuncs)**: Fast element-wise operations

Think of NumPy as "Python with superpowers for numbers." It's like giving Python a calculator that can handle millions of numbers at once - no more slow loops! 🚀

<!-- FIXME: Add visual showing NumPy array structure vs Python list structure -->

# NumPy Array Fundamentals

## Creating NumPy Arrays

NumPy arrays are the foundation of all scientific computing in Python. They're fast, memory-efficient, and enable vectorized operations.

### Basic Array Creation

NumPy arrays can be created from Python lists or using built-in functions. The key difference from Python lists is that NumPy arrays are homogeneous (all elements same type) and optimized for numerical operations. It's like the difference between a messy desk and a perfectly organized filing cabinet - both hold stuff, but one is much more efficient! 📁

**Reference:**

```python
import numpy as np

# From Python lists
data = [1, 2, 3, 4, 5]
arr = np.array(data)         # array([1, 2, 3, 4, 5])

# From nested lists (2D arrays)
data_2d = [[1, 2, 3], [4, 5, 6]]
arr_2d = np.array(data_2d)  # array([[1, 2, 3], [4, 5, 6]])

# Array creation functions
zeros = np.zeros(5)          # array([0., 0., 0., 0., 0.])
ones = np.ones((2, 3))       # array([[1., 1., 1.], [1., 1., 1.]])
range_arr = np.arange(10)    # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
empty = np.empty((2, 3))     # Uninitialized array (contains garbage! Use with caution)
full = np.full((2, 3), 7)    # array([[7, 7, 7], [7, 7, 7]])
```

### Array Properties and Attributes

Every NumPy array has important properties that tell you about its structure and memory usage. These are crucial for debugging and understanding your data. Think of them as the "nutrition facts" for your arrays - they tell you everything you need to know! 🏷️

![NumPy Memory Layout](media/numpy_memory_layout.png)

*NumPy arrays are stored contiguously in memory for speed - like a perfectly organized filing cabinet vs a messy desk!*

**Reference:**

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Array shape and dimensions
print(arr.shape)             # (2, 3) - 2 rows, 3 columns
print(arr.ndim)              # 2 - number of dimensions
print(arr.size)              # 6 - total number of elements
print(arr.dtype)             # int64 - data type
print(arr.itemsize)          # 8 - bytes per element
print(arr.nbytes)            # 48 - total bytes
```

### Data Types and Type Conversion

NumPy arrays are homogeneous - all elements must be the same type. This is different from Python lists and enables fast operations, but requires careful type management. It's like a strict bouncer at a club - everyone must be the same "type" to get in, but once they're in, the party is much more organized! 🎭

**Reference:**

```python
# Explicit data types
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = np.array([1, 2, 3], dtype=np.float64)

# Type conversion
arr = np.array([1, 2, 3, 4, 5])
float_arr = arr.astype(np.float64)  # Convert to float
int_arr = arr.astype(np.int32)      # Convert to int

# String to numeric conversion
str_arr = np.array(["1.25", "-9.6", "42"])
num_arr = str_arr.astype(float)     # array([1.25, -9.6, 42.])
```

**Brief Example:**

```python
# Create a 2D array and check its properties
grades = np.array([[85, 92, 78], [95, 88, 91]])
print(f"Shape: {grades.shape}")      # (2, 3)
print(f"Mean: {grades.mean():.1f}")  # 88.2
```

# Array Indexing and Slicing

NumPy arrays support powerful indexing that's much more flexible than Python lists. It's like having a Swiss Army knife for data selection - you can slice, dice, and filter your data in ways that would make a chef jealous! 🔪

<!-- FIXME: Add visual showing different indexing methods (basic, boolean, fancy) -->

### Basic Indexing

**Reference:**

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

Single element access
first = arr[0]              # 0
last = arr[-1]              # 9

Slicing (like Python lists)
subset = arr[2:7]           # array([2, 3, 4, 5, 6])
every_other = arr[::2]      # array([0, 2, 4, 6, 8])
```

### Multidimensional Indexing

**Reference:**

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

Access elements
first_row = arr_2d[0]       # array([1, 2, 3])
element = arr_2d[1, 2]      # 6 (row 1, column 2)

Slicing 2D arrays
first_two_rows = arr_2d[:2]     # array([[1, 2, 3], [4, 5, 6]])
middle_column = arr_2d[:, 1]    # array([2, 5, 8])
```

### Boolean Indexing

**Reference:**

```python
arr = np.array([1, 5, 3, 8, 2, 9, 4])

# Boolean indexing
mask = arr > 5               # array([False, False, False, True, False, True, False])
high_values = arr[mask]      # array([8, 9])

# Conditional operations
arr[arr > 5] = 0             # Set values > 5 to 0: array([1, 5, 3, 0, 2, 0, 4])

# Multiple conditions
arr = np.array([1, 5, 3, 8, 2, 9, 4])
mask = (arr > 2) & (arr < 8)  # Use & for AND, | for OR
filtered = arr[mask]         # array([5, 3])
```

### Fancy Indexing

**Reference:**

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])

# Integer array indexing
indices = [1, 3, 5]
selected = arr[indices]      # array([20, 40, 60])

# Negative indexing
selected = arr[[-1, -3, -5]] # array([80, 60, 40])

# 2D fancy indexing
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_indices = [0, 2]
col_indices = [1, 2]
selected = arr_2d[row_indices, col_indices]  # array([2, 9])
```

### Array Views vs Copies

**Reference:**

```python
arr = np.array([1, 2, 3, 4, 5])

# Slicing creates views (shares memory)
view = arr[1:4]              # array([2, 3, 4])
view[0] = 99                 # Modifies original array!
print(arr)                   # array([1, 99, 3, 4, 5])

# Explicit copy
arr = np.array([1, 2, 3, 4, 5])
copy = arr[1:4].copy()       # Creates independent copy
copy[0] = 99                 # Doesn't affect original
print(arr)                   # array([1, 2, 3, 4, 5])
```

**Brief Example:**

```python
# Find high achievers with boolean indexing
grades = np.array([78, 92, 85, 88, 91, 82, 95])
high_achievers = grades[grades > 85]
print(f"High achievers: {high_achievers}")  # [92 88 91 95]
```

LIVE DEMO!
*Working with real data: loading student grades and finding patterns using NumPy operations*

# NumPy Operations and Universal Functions

### Vectorized Operations

NumPy's real power comes from vectorized operations - applying functions to entire arrays at once, without loops. It's like having a magic wand that can transform thousands of numbers simultaneously, instead of doing it one by one like a muggle! ✨

![Vectorized Operations](media/vectorized_vs_loops.png)

*Vectorized operations vs loops - why NumPy is so much faster!*

### Arithmetic Operations

**Reference:**

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([5, 4, 3, 2, 1])

Element-wise operations
sum_arr = arr1 + arr2          # array([6, 6, 6, 6, 6])
mult_arr = arr1 * arr2         # array([5, 8, 9, 8, 5])
power_arr = arr1 ** 2          # array([1, 4, 9, 16, 25])

Scalar operations
doubled = arr1 * 2             # array([2, 4, 6, 8, 10])
squared = arr1 ** 2            # array([1, 4, 9, 16, 25])
```

### Universal Functions (ufuncs)

**Reference:**

```python
arr = np.array([1, 4, 9, 16, 25])

Mathematical functions
sqrt_arr = np.sqrt(arr)        # array([1., 2., 3., 4., 5.])
log_arr = np.log(arr)          # Natural logarithm
exp_arr = np.exp([1, 2, 3])    # array([2.72, 7.39, 20.09])

Statistical functions
mean_val = np.mean(arr)        # 11.0
std_val = np.std(arr)          # 8.94
max_val = np.max(arr)          # 25
```

### Array Aggregations

**Reference:**

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Aggregation functions
total = arr_2d.sum()           # 45 (sum of all elements)
mean = arr_2d.mean()           # 5.0 (mean of all elements)
std = arr_2d.std()             # 2.58 (standard deviation)
var = arr_2d.var()             # 6.67 (variance)

# Axis-specific aggregations
row_sums = arr_2d.sum(axis=1)  # array([6, 15, 24]) - sum each row
col_means = arr_2d.mean(axis=0) # array([4., 5., 6.]) - mean each column

# Finding min/max positions
max_pos = arr_2d.argmax()      # 8 (flattened index of maximum)
max_pos_2d = np.unravel_index(arr_2d.argmax(), arr_2d.shape)  # (2, 2)
```

### Broadcasting

Broadcasting is NumPy's superpower for handling arrays of different shapes. It's like having a universal remote that works with any TV - NumPy figures out how to make operations work between arrays of different sizes automatically! 📺

![Broadcasting Visual](media/broadcasting_diagram.png)

*How NumPy handles different array shapes automatically - it's like magic, but with math!*

**Reference:**

```python
# Broadcasting allows operations between arrays of different shapes
arr = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
scalar = 10
result = arr + scalar          # Broadcasts scalar to all elements

# Row and column broadcasting
row = np.array([1, 2, 3])     # Shape (3,)
col = np.array([[1], [2]])    # Shape (2, 1)
result = row + col            # Shape (2, 3) - broadcasts to (2, 3)

# More complex broadcasting
arr_3d = np.ones((2, 3, 4))
arr_1d = np.array([1, 2, 3, 4])
result = arr_3d + arr_1d      # Broadcasts (4,) to (2, 3, 4)
```

### Linear Algebra Operations

**Reference:**

```python
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B                     # Matrix multiplication
C_alt = np.dot(A, B)          # Alternative syntax

# Transpose
A_T = A.T                     # Transpose matrix
A_T_alt = A.transpose()       # Alternative syntax

# Determinant and inverse
det = np.linalg.det(A)        # Determinant
inv = np.linalg.inv(A)        # Matrix inverse
```

### Random Number Generation

**Reference:**

```python
# Set seed for reproducibility
np.random.seed(42)

# Generate random arrays
uniform = np.random.uniform(0, 1, size=(3, 3))  # Uniform distribution
normal = np.random.normal(0, 1, size=(3, 3))     # Normal distribution
integers = np.random.randint(1, 10, size=(3, 3)) # Random integers

# Random sampling
choices = np.random.choice([1, 2, 3, 4, 5], size=10)  # Random choices
shuffled = np.random.permutation([1, 2, 3, 4, 5])     # Random permutation
```

**Brief Example:**

```python
# Calculate class statistics with vectorized operations
grades = np.array([[85, 92, 78], [95, 88, 91], [82, 90, 87]])
print(f"Class average: {grades.mean():.1f}")  # 88.2
print(f"Highest grade: {grades.max()}")       # 95
```

## Command Line Data Processing

### Essential Shell Tools for Data Science

The command line is incredibly powerful for data processing. These tools can handle massive datasets that would crash Python, and they're often faster for simple operations. Think of them as your data science Swiss Army knife - simple tools that can do amazing things when combined! 🔧

![CLI Pipeline](media/cli_pipeline.png)

*Command line tools working together - like a well-oiled machine!*

### Text Processing with `cut`, `sort`, and `grep`

**Reference:**

```bash
# cut: Extract columns from structured data
cut -d',' -f1,3 data.csv          # Extract columns 1 and 3 from CSV
cut -d'\t' -f2-4 data.tsv        # Extract columns 2-4 from TSV
cut -c1-10 file.txt              # Extract characters 1-10

# sort: Sort data
sort data.txt                     # Alphabetical sort
sort -n data.txt                 # Numerical sort
sort -k2 -n data.csv             # Sort by 2nd column numerically
sort -r data.txt                 # Reverse sort

# grep: Search and filter
grep "pattern" file.txt           # Find lines containing "pattern"
grep -v "pattern" file.txt       # Find lines NOT containing "pattern"
grep -i "pattern" file.txt       # Case-insensitive search
grep -E "regex" file.txt         # Extended regex
```

### Advanced Text Processing

**Reference:**

```bash
# tr: Translate characters
tr 'a-z' 'A-Z' < file.txt        # Convert to uppercase
tr -d ' ' < file.txt             # Delete spaces
tr '\n' ' ' < file.txt           # Replace newlines with spaces

# sed: Stream editor
sed 's/old/new/g' file.txt       # Replace all occurrences
sed 's/^/PREFIX: /' file.txt     # Add prefix to each line
sed '/pattern/d' file.txt        # Delete lines matching pattern
sed -n '10,20p' file.txt         # Print lines 10-20

# awk: Pattern scanning and processing
awk '{print $1, $3}' file.txt     # Print columns 1 and 3
awk -F',' '{print $2}' data.csv  # Use comma as delimiter
awk '$3 > 50 {print $1}' data.txt # Print $1 where $3 > 50
```

### Building Data Pipelines

**Reference:**

```bash
# Complex data processing pipeline
cat data.csv | \
  cut -d',' -f2,4 | \
  tr '[:lower:]' '[:upper:]' | \
  tr ',' '\t' | \
  sort -k2 -n | \
  head -n 10 > top_results.tsv

# Process log files
grep "ERROR" logfile.txt | \
  cut -d' ' -f1,2,5 | \
  sort | \
  uniq -c | \
  sort -nr > error_summary.txt

# Extract and analyze data
grep "student" grades.csv | \
  cut -d',' -f2,3 | \
  awk -F',' '$2 > 85 {print $1}' | \
  sort > high_achievers.txt
```

**Brief Example:**

```bash
# Process student grades
echo "Name,Grade,Subject
Alice,85,Math
Bob,92,Science
Charlie,78,Math
David,95,Science" > grades.csv

# Find top performers in Science
grep "Science" grades.csv | \
  cut -d',' -f1,2 | \
  awk -F',' '$2 > 90 {print $1}'
# Output: Bob
```

## Git Review: Essential Version Control

### Why Git Matters for Data Science

Git isn't just for software development - it's essential for data science projects. Think of Git as your "undo button for everything" - you'll use it to:
- Track changes to your analysis code (because you WILL break things) 💥
- Collaborate on data projects (because data science is a team sport) 👥
- Maintain different versions of your analysis (because your first attempt is never the best) 🎯
- Backup your work to GitHub (because hard drives die, but GitHub lives forever) 💾

<!-- FIXME: Add visual showing Git workflow (working directory -> staging -> commit -> push) -->

![xkcd 1296: Git](media/xkcd_1296.png)

*The reality of Git - it's powerful but can be confusing at first! Don't worry, we'll make it simple.*

### Essential Git Commands for Data Science

**Reference:**
```bash
# Check status and see what's changed
git status                    # Show modified files
git diff                      # Show detailed changes
git log --oneline            # Show commit history

# Stage and commit changes
git add filename.py           # Stage specific file
git add .                     # Stage all changes
git commit -m "Add NumPy analysis"  # Commit with message

# Working with branches
git branch                    # List branches
git checkout -b new-feature   # Create and switch to new branch
git checkout main            # Switch back to main branch
git merge new-feature        # Merge branch into main

# Remote operations
git push origin main          # Push changes to GitHub
git pull origin main         # Pull latest changes
git clone [url]              # Copy repository from GitHub
```

### Git Workflow for Data Projects

**Reference:**
```bash
# Daily workflow
git status                   # Check what's changed
git add .                    # Stage all changes
git commit -m "Update analysis with new data"
git push origin main         # Backup to GitHub

# Starting new analysis
git checkout -b analysis-v2  # Create new branch
# ... do your analysis ...
git add .
git commit -m "Complete v2 analysis"
git push origin analysis-v2  # Push branch
```

**Brief Example:**
```bash
# Track your NumPy analysis
git add student_analysis.py
git commit -m "Add NumPy student performance analysis"
git push origin main
```

# Professional Python Development

### Type Checking and Debugging

Understanding data types is crucial for debugging. Python's dynamic typing means variables can change type, so type checking is essential. It's like being a detective - you need to know what you're dealing with before you can solve the case! 🕵️‍♀️

![xkcd 1205: Code Quality](media/xkcd_1205.png)

*The reality of debugging - sometimes you need to be a detective to figure out what's going on!*

### Type Checking Basics

**Reference:**

```python
# Type checking for debugging
user_input = "42"  # This is a string, not a number!
print(f"Input: {user_input}")
print(f"Type: {type(user_input)}")  # <class 'str'>

# Convert and verify
number = int(user_input)
print(f"Converted: {number}")
print(f"New type: {type(number)}")  # <class 'int'>

# Debugging data processing
data = [1, 2, "3", 4, 5]  # Mixed types!
for item in data:
    print(f"Item: {item}, Type: {type(item)}")
    if isinstance(item, str):
        print(f"  Converting string '{item}' to int")
        item = int(item)
```

### Error Handling

**Reference:**

```python
# Basic error handling
try:
    number = int("not_a_number")
except ValueError:
    print("Could not convert to number")

try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
```

## F-String Formatting

F-strings are the modern, readable way to format strings in Python. They're like having a conversation with your data - you can ask it questions and get nicely formatted answers! 💬

### F-String Basics

**Reference:**

```python
name = "Alice"
grade = 87.5
year = 2024

# F-strings (preferred, Python 3.6+)
message = f"Student {name} earned {grade:.1f}% in {year}"

# Formatting options
print(f"Grade: {grade:.2f}")        # 87.50
print(f"Grade: {grade:>8.1f}")      # Right-aligned: "    87.5"
print(f"Grade: {grade:<8.1f}")      # Left-aligned: "87.5    "
print(f"Grade: {grade:^8.1f}")      # Center-aligned: " 87.5   "
```

### Advanced F-String Features

**Reference:**

```python
# Expressions in f-strings
arr = np.array([1, 2, 3, 4, 5])
print(f"Array sum: {arr.sum()}")
print(f"Array mean: {arr.mean():.2f}")

# Multi-line f-strings
result = f"""
Analysis Results:
- Total students: {len(arr)}
- Average score: {arr.mean():.1f}
- Highest score: {arr.max()}
- Lowest score: {arr.min()}
"""
```

**Brief Example:**

```python
# Clean student names with f-strings
messy_names = ["  alice SMITH  ", "BOB jones", "  Charlie Brown "]

for name in messy_names:
    clean_name = name.strip().title()
    print(f"Original: '{name}' -> Clean: '{clean_name}'")
```

## Virtual Environments

Virtual environments create isolated Python installations for each project. This prevents package conflicts and ensures reproducible environments. Think of them as separate apartments for each project - everyone gets their own space and can't mess with each other's stuff! 🏠

<!-- FIXME: Add visual showing virtual environment isolation concept -->

![xkcd 1987: Python Environment](media/xkcd_1987.png)

*Why virtual environments matter - package chaos without them! Each project gets its own clean space.*

### Why Virtual Environments Matter

**The Problem:** Different projects need different package versions. Without isolation:

- Project A needs pandas 1.3.0
- Project B needs pandas 2.0.0  
- Installing one breaks the other!

**The Solution:** Each project gets its own Python environment with its own packages.

### Creating Virtual Environments

**Reference (using conda - recommended):**

```bash
# Create environment
conda create -n datasci-practice python=3.11

# Activate environment
conda activate datasci-practice

# Install packages
conda install pandas numpy matplotlib

# Deactivate when done
conda deactivate
```

**Reference (using venv):**

```bash
# Create environment
python -m venv datasci-practice

# Activate environment
# Mac/Linux:
source datasci-practice/bin/activate
# Windows:
datasci-practice\Scripts\activate

# Install packages
pip install pandas numpy matplotlib

# Deactivate
deactivate
```

### Managing Dependencies

**Reference:**

```bash
# Save current environment packages
conda list --export > requirements.txt

# Or create environment file (better for conda)
conda env export > environment.yml

# For pip-only environments
pip freeze > requirements.txt
```

**Environment File Example (environment.yml):**

```yaml
name: datasci-practice
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pandas>=1.5.0
  - numpy>=1.20.0
  - matplotlib>=3.5.0
  - jupyter
  - pip
  - pip:
    - requests>=2.28.0
```

**Brief Example:**

```bash
# Professional workflow
# 1. Start new project
conda create -n my-analysis python=3.11
conda activate my-analysis

# 2. Install packages
conda install pandas numpy matplotlib

# 3. Document environment
conda env export > environment.yml

# 4. Commit to Git
git add environment.yml
git commit -m "Add environment configuration"
```

### Array Manipulation and Reshaping

**Reference:**

```python
# Reshaping arrays
arr = np.arange(12)
reshaped = arr.reshape(3, 4)      # Reshape to 3x4
flattened = reshaped.flatten()    # Flatten back to 1D

# Transposing
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
transposed = arr_2d.T             # Transpose

# Concatenation
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
combined = np.concatenate([arr1, arr2])  # [1, 2, 3, 4, 5, 6]

# Stacking
stacked = np.vstack([arr1, arr2])  # Vertical stack
hstacked = np.hstack([arr1, arr2]) # Horizontal stack
```

### Array Set Operations

**Reference:**

```python
# Unique elements
arr = np.array([1, 2, 2, 3, 3, 3, 4])
unique_vals = np.unique(arr)      # array([1, 2, 3, 4])

# Set operations
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([3, 4, 5, 6, 7])

# Intersection
intersection = np.intersect1d(arr1, arr2)  # array([3, 4, 5])

# Union
union = np.union1d(arr1, arr2)    # array([1, 2, 3, 4, 5, 6, 7])

# Set difference
diff = np.setdiff1d(arr1, arr2)   # array([1, 2]) - elements in arr1 not in arr2
```

### Array File I/O

**Reference:**

```python
# Save arrays to disk
arr = np.array([[1, 2, 3], [4, 5, 6]])
np.save('data.npy', arr)         # Save as binary
np.savetxt('data.txt', arr)     # Save as text

# Load arrays from disk
loaded = np.load('data.npy')    # Load binary
loaded_txt = np.loadtxt('data.txt')  # Load text

# CSV operations
np.savetxt('data.csv', arr, delimiter=',', header='col1,col2,col3')
loaded_csv = np.loadtxt('data.csv', delimiter=',', skiprows=1)
```

### Array Sorting and Searching

**Reference:**

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Sorting
sorted_arr = np.sort(arr)         # Returns sorted copy
arr.sort()                        # Sorts in place

# Finding elements
max_idx = np.argmax(arr)          # Index of maximum
min_idx = np.argmin(arr)          # Index of minimum
sorted_indices = np.argsort(arr)  # Indices that would sort array

# Searching
where_result = np.where(arr > 5)  # Indices where condition is true
```

LIVE DEMO!
*Working with NumPy arrays: loading data, performing vectorized operations, and generating statistical summaries*

# Real-World NumPy Example: Student Performance Analysis

This is where NumPy's power becomes clear - analyzing real data with vectorized operations.

**Reference Pattern:**

```python
import numpy as np

# Simulate student data (in practice, you'd load from CSV)
np.random.seed(42)  # For reproducible results
n_students = 100
n_assignments = 5

# Generate random grades (70-100 range)
grades = np.random.randint(70, 101, size=(n_students, n_assignments))
student_names = [f"Student_{i:03d}" for i in range(n_students)]

# Calculate statistics using vectorized operations
student_averages = grades.mean(axis=1)  # Average per student
assignment_averages = grades.mean(axis=0)  # Average per assignment
overall_average = grades.mean()  # Overall class average

# Find high performers (above 90 average)
high_performers = student_averages > 90
high_performer_names = np.array(student_names)[high_performers]
high_performer_grades = student_averages[high_performers]

# Statistical analysis
print(f"Class Statistics:")
print(f"Overall average: {overall_average:.1f}")
print(f"Highest student average: {student_averages.max():.1f}")
print(f"Lowest student average: {student_averages.min():.1f}")
print(f"Standard deviation: {student_averages.std():.1f}")

print(f"\nHigh Performers ({len(high_performer_names)} students):")
for name, grade in zip(high_performer_names, high_performer_grades):
    print(f"  {name}: {grade:.1f}")

# Assignment difficulty analysis
print(f"\nAssignment Averages:")
for i, avg in enumerate(assignment_averages):
    print(f"  Assignment {i+1}: {avg:.1f}")
```

This example demonstrates NumPy's power for real data analysis - fast, vectorized operations on large datasets. It's like having a supercomputer in your pocket! 🚀

![xkcd 2083: Data Analysis](media/xkcd_2083.png)

*The reality of data analysis - sometimes the data has other plans!*

# Key Takeaways

1. **NumPy arrays** are 10-100x faster than Python lists for numerical operations
2. **Vectorized operations** apply functions to entire arrays without loops
3. **Boolean indexing** enables powerful data filtering and selection
4. **Virtual environments** keep projects isolated and reproducible
5. **Type checking** is essential for debugging and understanding data
6. **F-strings** provide clean, readable string formatting

You now have the foundation for professional Python data science. NumPy's vectorized operations, combined with proper environment management and debugging skills, form the backbone of all scientific computing in Python. You've gone from Python novice to data science superhero! 🦸‍♀️

Next week: We'll dive into pandas - the data manipulation library built on NumPy that makes working with tabular data intuitive and powerful!

Practice Challenge

Before next class:

1. Create a virtual environment for this week's practice
2. Generate a NumPy array with 1000 random numbers (0-100)
3. Calculate and print:
   - Mean, median, and standard deviation
   - Number of values above 80
   - Values in the 90th percentile
4. Use f-strings for all output formatting
5. Add type checking to verify your calculations

Remember: NumPy is the foundation of pandas - master these concepts and you'll be ready for the next level!

These professional development skills - NumPy arrays, virtual environments, and debugging - are what separate hobbyist Python from production-ready data science.
