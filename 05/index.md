Python Libraries and Environment Management

Welcome to week 5! Today we're diving into Python's powerful ecosystem of libraries and learning how to manage project environments like a professional data scientist. You'll discover NumPy, the foundation of scientific computing in Python, and master virtual environments.

By the end of today, you'll know how to set up isolated Python environments for different projects and perform efficient numerical computations with NumPy arrays.

*[xkcd 1987: "Python Environment" - Shows a person saying "My Python environment has become so complex that I keep a docker container just to update my package manager to install a new version of a program to manage my Python versions to install libraries."]*

Don't worry - we'll keep your environment manageable with best practices!

Why Package Management Matters

The Dependency Problem

Imagine working on multiple data science projects:
- **Project A** needs pandas 1.3.0 and numpy 1.21.0
- **Project B** needs pandas 2.0.0 and numpy 1.24.0  
- **Your system** has pandas 1.5.0 and numpy 1.20.0 installed globally

Without proper environment management, you get:
- Version conflicts that break code
- "It works on my machine" problems
- Difficulty sharing reproducible analyses
- System-wide package chaos

The Virtual Environment Solution

Virtual environments create isolated Python installations for each project:
- Each project gets its own package versions
- No conflicts between different projects
- Easy to share exact requirements with teammates
- Clean, reproducible development environments

Think of it as giving each project its own private Python toolkit.

Virtual Environment Management

Understanding the Options

Python has several environment management tools. We'll focus on the two most important:

**conda** - Preferred for data science (handles non-Python dependencies, optimized packages)  
**venv/pip** - Built into Python (lighter weight, universal compatibility)

When to Use Which

**Use conda when:**
- Doing data science or scientific computing
- Need packages like NumPy, pandas, matplotlib, scikit-learn
- Working with complex dependencies (C libraries, etc.)
- Want optimized, pre-compiled packages

**Use venv + pip when:**
- Building web applications or general Python projects
- Working in resource-constrained environments
- Need packages not available in conda
- Want to stick with Python's built-in tools

Creating Virtual Environments with conda

Basic conda Workflow

**Reference:**
```bash
Create new environment with specific Python version
conda create -n project-name python=3.11

Activate the environment
conda activate project-name

Install packages
conda install numpy pandas matplotlib

Deactivate when done
conda deactivate

List all environments
conda env list

Remove environment (if needed)
conda env remove -n project-name
```

Project-Specific Environment Setup

**Reference:**
```bash
Create environment for specific project
conda create -n datasci-analysis python=3.11
conda activate datasci-analysis

Install data science essentials
conda install numpy pandas matplotlib jupyter seaborn scikit-learn

Install additional packages with pip if needed
pip install some-package-not-in-conda

Save current environment
conda env export > environment.yml

Recreate environment on another machine
conda env create -f environment.yml
```

**Brief Example:**
```bash
Set up environment for DataSci 217 assignments
conda create -n datasci217 python=3.11
conda activate datasci217
conda install numpy pandas matplotlib jupyter
pip install pytest  # For assignment testing

echo "Environment ready! Remember to activate before working."
```

Alternative: venv + pip

Basic venv Workflow

**Reference:**
```bash
Create virtual environment
python -m venv project-env

Activate environment
On macOS/Linux:
source project-env/bin/activate
On Windows:
project-env\Scripts\activate

Install packages
pip install numpy pandas matplotlib

Save requirements
pip freeze > requirements.txt

Deactivate
deactivate

Install from requirements (new machine)
pip install -r requirements.txt
```

Project Requirements Management

**Reference (requirements.txt):**
```txt
Core data science packages
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
jupyter>=1.0.0

For testing
pytest>=7.0.0

Optional: specify exact versions for reproducibility
numpy==1.24.3
pandas==2.0.2
```

**Brief Example:**
```bash
Create and setup venv environment
python -m venv datasci217-env
source datasci217-env/bin/activate  # or activate.bat on Windows
pip install -r requirements.txt
pip freeze > installed-packages.txt  # Document what was actually installed
```

LIVE DEMO!
*Setting up a conda environment, installing packages, and demonstrating the difference between global and environment-specific installations*

Introduction to NumPy

Why NumPy is Essential

NumPy (Numerical Python) is the foundation of the entire Python data science ecosystem. It provides:
- **Fast arrays** - 10-100x faster than Python lists for numerical operations
- **Memory efficiency** - Much less memory usage than Python lists
- **Mathematical functions** - Optimized implementations of common operations
- **Broadcasting** - Elegant handling of operations between different-sized arrays

Every major data science library (pandas, scikit-learn, matplotlib) is built on NumPy.

The Performance Difference

**Python lists (slow):**
```python
Calculate squares of 1 million numbers
numbers = list(range(1000000))
squares = [x**2 for x in numbers]  # Slow, memory-intensive
```

**NumPy arrays (fast):**
```python
import numpy as np
numbers = np.arange(1000000)
squares = numbers**2  # Fast, memory-efficient
```

The NumPy version is typically 10-50x faster!

NumPy Array Basics

Array Creation

**Reference:**
```python
import numpy as np

From Python lists
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array

Built-in creation functions
zeros = np.zeros(5)                    # [0. 0. 0. 0. 0.]
ones = np.ones((2, 3))                 # 2x3 array of ones
full = np.full((3, 3), 7)              # 3x3 array filled with 7

Ranges and sequences
range_arr = np.arange(0, 10, 2)        # [0 2 4 6 8] (start, stop, step)
linspace = np.linspace(0, 1, 5)        # [0. 0.25 0.5 0.75 1.] (start, stop, count)

Random arrays
random_arr = np.random.rand(3, 4)      # 3x4 array of random numbers [0,1)
random_int = np.random.randint(1, 11, 10)  # 10 random integers [1,10]
```

Array Attributes

**Reference:**
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

Shape and dimensions
print(arr.shape)      # (2, 3) - 2 rows, 3 columns
print(arr.ndim)       # 2 - number of dimensions
print(arr.size)       # 6 - total number of elements

Data type information
print(arr.dtype)      # int64 (or similar, depends on system)
print(arr.itemsize)   # 8 - bytes per element

Memory usage
print(arr.nbytes)     # 48 - total bytes used
```

**Brief Example:**
```python
Create sample grade data
grades = np.array([85, 92, 78, 96, 88, 91, 84, 87])
print(f"Class grades: {grades}")
print(f"Number of students: {grades.size}")
print(f"Data type: {grades.dtype}")
print(f"Memory usage: {grades.nbytes} bytes")
```

Array Indexing and Slicing

Basic Indexing

**Reference:**
```python
arr = np.array([10, 20, 30, 40, 50])

Single elements (0-indexed)
first = arr[0]        # 10
last = arr[-1]        # 50
middle = arr[2]       # 30

Slicing (start:stop:step)
first_three = arr[:3]      # [10 20 30]
last_two = arr[-2:]        # [40 50]
every_other = arr[::2]     # [10 30 50]
reversed_arr = arr[::-1]   # [50 40 30 20 10]
```

2D Array Indexing

**Reference:**
```python
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

Individual elements
element = matrix[1, 2]     # 6 (row 1, column 2)
element = matrix[1][2]     # Alternative syntax (less efficient)

Rows and columns
first_row = matrix[0, :]   # [1 2 3] (row 0, all columns)
second_col = matrix[:, 1]  # [2 5 8] (all rows, column 1)

Sub-arrays
sub_matrix = matrix[0:2, 1:3]  # [[2 3] [5 6]] (first 2 rows, columns 1-2)
```

Boolean Indexing

**Reference:**
```python
grades = np.array([85, 92, 78, 96, 88, 91, 84, 87])

Boolean conditions
high_grades = grades > 90        # [False True False True False True False False]
passing = grades >= 80          # [True True False True True True True True]

Filter arrays with boolean indexing
excellent_grades = grades[grades > 90]    # [92 96 91]
failing_grades = grades[grades < 80]      # [78]

Multiple conditions (use & for and, | for or)
good_grades = grades[(grades >= 85) & (grades <= 95)]  # [85 92 88 91 87]
```

**Brief Example:**
```python
Analyze student grades
grades = np.array([85, 92, 78, 96, 88, 91, 84, 87])

Find various grade categories
excellent = grades[grades >= 90]
good = grades[(grades >= 80) & (grades < 90)]
needs_help = grades[grades < 80]

print(f"Excellent grades (≥90): {excellent}")
print(f"Good grades (80-89): {good}")
print(f"Needs help (<80): {needs_help}")
```

Basic Mathematical Operations

Element-wise Operations

**Reference:**
```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

Arithmetic operations (element-wise)
addition = arr1 + arr2        # [11 22 33 44]
subtraction = arr2 - arr1     # [9 18 27 36]
multiplication = arr1 * arr2   # [10 40 90 160]
division = arr2 / arr1        # [10. 10. 10. 10.]
power = arr1 ** 2             # [1 4 9 16]

Operations with scalars (broadcasting)
scaled = arr1 * 10            # [10 20 30 40]
shifted = arr1 + 5            # [6 7 8 9]
```

Statistical Functions

**Reference:**
```python
grades = np.array([85, 92, 78, 96, 88, 91, 84, 87])

Basic statistics
mean_grade = np.mean(grades)      # 87.625
median_grade = np.median(grades)  # 87.5
std_dev = np.std(grades)         # 5.74 (approximately)

Range statistics
min_grade = np.min(grades)        # 78
max_grade = np.max(grades)        # 96
range_grades = np.ptp(grades)     # 18 (peak-to-peak, max-min)

Aggregations
total_points = np.sum(grades)     # 701
student_count = np.size(grades)   # 8

Position-based statistics
min_index = np.argmin(grades)     # 2 (index of minimum value)
max_index = np.argmax(grades)     # 3 (index of maximum value)
```

Mathematical Functions

**Reference:**
```python
data = np.array([1, 4, 9, 16, 25])

Common mathematical functions
sqrt_data = np.sqrt(data)         # [1. 2. 3. 4. 5.]
log_data = np.log(data)          # Natural logarithm
exp_data = np.exp(np.array([1, 2, 3]))  # [2.72 7.39 20.09]

Trigonometric functions
angles = np.array([0, np.pi/2, np.pi])
sin_values = np.sin(angles)       # [0. 1. 0.]
cos_values = np.cos(angles)       # [1. 0. -1.]

Rounding and absolute values
decimals = np.array([3.14159, 2.71828, 1.41421])
rounded = np.round(decimals, 2)   # [3.14 2.72 1.41]
abs_values = np.abs(np.array([-5, -3, 2, 7]))  # [5 3 2 7]
```

**Brief Example:**
```python
Analyze temperature data
temps_celsius = np.array([22.5, 25.1, 19.8, 28.3, 21.7, 26.4])

Calculate statistics
avg_temp = np.mean(temps_celsius)
temp_range = np.ptp(temps_celsius)
std_temp = np.std(temps_celsius)

Convert to Fahrenheit
temps_fahrenheit = temps_celsius * 9/5 + 32

print(f"Average temperature: {avg_temp:.1f}°C ({np.mean(temps_fahrenheit):.1f}°F)")
print(f"Temperature range: {temp_range:.1f}°C")
print(f"Standard deviation: {std_temp:.1f}°C")
```

LIVE DEMO!
*Working with real dataset using NumPy: loading data, performing calculations, comparing performance with Python lists*

Array Reshaping and Combining

Reshaping Arrays

**Reference:**
```python
Create 1D array
arr = np.arange(12)  # [0 1 2 3 4 5 6 7 8 9 10 11]

Reshape to 2D
matrix_3x4 = arr.reshape(3, 4)   # 3 rows, 4 columns
matrix_4x3 = arr.reshape(4, 3)   # 4 rows, 3 columns
matrix_2x6 = arr.reshape(2, -1)  # 2 rows, auto-calculate columns

Flatten back to 1D
flattened = matrix_3x4.flatten()  # Back to 1D array
raveled = matrix_3x4.ravel()      # Also flattens (may share memory)

Transpose (flip dimensions)
transposed = matrix_3x4.T         # 4x3 (columns become rows)
```

Combining Arrays

**Reference:**
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

Concatenation
horizontal = np.concatenate([arr1, arr2])      # [1 2 3 4 5 6]
vertical = np.stack([arr1, arr2])             # [[1 2 3] [4 5 6]]

Specific joining functions
hstacked = np.hstack([arr1, arr2])            # [1 2 3 4 5 6]
vstacked = np.vstack([arr1, arr2])            # [[1 2 3] [4 5 6]]

For 2D arrays
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

combined_horizontal = np.hstack([matrix1, matrix2])  # [[1 2 5 6] [3 4 7 8]]
combined_vertical = np.vstack([matrix1, matrix2])    # [[1 2] [3 4] [5 6] [7 8]]
```

**Brief Example:**
```python
Combine grade data from different assignments
assignment1 = np.array([85, 92, 78])
assignment2 = np.array([88, 89, 82])
assignment3 = np.array([91, 95, 85])

Stack assignments as rows (students as columns)
all_grades = np.vstack([assignment1, assignment2, assignment3])
print("Grade matrix (assignments × students):")
print(all_grades)

Calculate average per student
student_averages = np.mean(all_grades, axis=0)
print(f"Student averages: {student_averages}")
```

Working with Real Data

Loading Data from Files

**Reference:**
```python
Save arrays to files
grades = np.array([85, 92, 78, 96, 88])
np.savetxt('grades.txt', grades, fmt='%d')        # Save as text
np.save('grades.npy', grades)                     # Save as NumPy binary

Load arrays from files
loaded_grades = np.loadtxt('grades.txt')          # From text file
loaded_binary = np.load('grades.npy')             # From NumPy binary

Work with CSV-like data
data = np.loadtxt('student_data.csv', delimiter=',', skiprows=1)  # Skip header
```

Integration with CSV Processing

**Reference:**
```python
import csv
import numpy as np

def load_grades_as_numpy(filename):
    """Load student grades from CSV into NumPy array"""
    grades_list = []
    
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Extract numeric grades
            grade = float(row['grade'])
            grades_list.append(grade)
    
    # Convert to NumPy array for fast processing
    return np.array(grades_list)

Usage
grades = load_grades_as_numpy('student_grades.csv')

Now we can use NumPy's fast operations
class_average = np.mean(grades)
grade_std = np.std(grades)
top_10_percent = np.percentile(grades, 90)

print(f"Class statistics:")
print(f"Average: {class_average:.1f}")
print(f"Standard deviation: {grade_std:.1f}")
print(f"90th percentile: {top_10_percent:.1f}")
```

**Brief Example:**
```python
Process large dataset efficiently
import time

Generate sample data (simulating large dataset)
large_dataset = np.random.normal(85, 10, 10000)  # 10,000 grades, mean=85, std=10

Time NumPy operations
start_time = time.time()
mean_grade = np.mean(large_dataset)
std_grade = np.std(large_dataset)
above_90 = np.sum(large_dataset > 90)
numpy_time = time.time() - start_time

print(f"NumPy processing of 10,000 records:")
print(f"Mean: {mean_grade:.1f}")
print(f"Standard deviation: {std_grade:.1f}")
print(f"Grades above 90: {above_90}")
print(f"Processing time: {numpy_time:.4f} seconds")
```

Key Takeaways

1. **Virtual environments** isolate project dependencies and prevent version conflicts
2. **conda** is preferred for data science; **venv + pip** works for general Python projects
3. **NumPy arrays** are much faster and more memory-efficient than Python lists
4. **Array operations** are vectorized - they apply to entire arrays at once
5. **Boolean indexing** provides powerful data filtering capabilities
6. **Mathematical functions** in NumPy are optimized and work element-wise
7. **Integration** with CSV processing creates efficient data analysis workflows

You now have the foundation for numerical computing in Python and professional environment management. NumPy's speed and efficiency will become essential as you work with larger datasets.

Next week: We'll introduce Jupyter notebooks and pandas for even more powerful data analysis!

Practice Challenge

Before next class:
1. **Environment Setup:**
   - Create a conda environment named `numpy-practice`
   - Install numpy, matplotlib, and jupyter
   - Document the process and package versions
   
2. **NumPy Practice:**
   - Create arrays using different methods (lists, arange, random)
   - Practice indexing, slicing, and boolean filtering
   - Implement basic statistical analysis using NumPy functions
   
3. **Real Data:**
   - Generate or find a small CSV dataset
   - Load it and convert numeric columns to NumPy arrays
   - Calculate summary statistics and compare with manual calculations

Remember: NumPy is the foundation - master these basics before moving to pandas!