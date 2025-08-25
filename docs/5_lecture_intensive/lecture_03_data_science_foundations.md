# Lecture 03: Data Science Foundations

*NumPy Arrays, Pandas Fundamentals, and Data Loading/Cleaning*

## Learning Objectives

By the end of this lecture, you will be able to:
- Understand and manipulate NumPy arrays for numerical computing
- Create and work with Pandas Series and DataFrames
- Load data from various file formats (CSV, JSON, Excel)
- Perform basic data cleaning and preprocessing
- Handle missing data and data type conversions
- Understand the relationship between NumPy and Pandas

## Introduction: The Data Science Stack

Today we enter the core of data science computing in Python. NumPy and Pandas form the foundation of virtually every data science project in Python. NumPy provides the computational engine with its powerful n-dimensional arrays, while Pandas adds the data manipulation and analysis capabilities that make working with real-world data practical and intuitive.

Understanding these libraries deeply will accelerate everything you do in data science - from simple data exploration to complex machine learning pipelines. We'll focus on building a solid foundation that will serve you throughout your career.

## Part 1: NumPy - Numerical Computing Foundation

### What Makes NumPy Special?

NumPy (Numerical Python) revolutionized data science by providing:
- **Speed**: Operations are implemented in C, making them 10-100x faster than pure Python
- **Memory Efficiency**: Arrays use contiguous memory, reducing overhead
- **Vectorization**: Apply operations to entire arrays without writing loops
- **Broadcasting**: Perform operations on arrays of different shapes
- **Integration**: Foundation for virtually every other data science library

### NumPy Array Fundamentals

**Creating Arrays:**
```python
import numpy as np

# From Python lists
simple_array = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Built-in creation functions
zeros = np.zeros((3, 4))           # 3x4 array of zeros
ones = np.ones((2, 3))             # 2x3 array of ones
identity = np.eye(3)               # 3x3 identity matrix
arange_array = np.arange(0, 10, 2) # [0, 2, 4, 6, 8]
linspace_array = np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# Random arrays
random_array = np.random.random((3, 3))      # Random values [0, 1)
random_int = np.random.randint(0, 10, (2, 4)) # Random integers
normal_dist = np.random.normal(0, 1, 1000)   # Normal distribution

# Structured arrays with data types
structured = np.array([1.5, 2.3, 3.7], dtype=np.float32)
```

**Understanding Array Properties:**
```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(f"Shape: {arr.shape}")          # (3, 4) - 3 rows, 4 columns
print(f"Dimensions: {arr.ndim}")      # 2 - number of dimensions
print(f"Size: {arr.size}")           # 12 - total number of elements
print(f"Data type: {arr.dtype}")     # int64 (or int32 on some systems)
print(f"Item size: {arr.itemsize}")  # 8 bytes per element
print(f"Memory usage: {arr.nbytes}") # 96 bytes total
```

### Array Indexing and Slicing

**Basic Indexing:**
```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# Single element access
element = arr[1, 2]                   # 7 (row 1, column 2)
element = arr[1][2]                   # Alternative syntax (less efficient)

# Row and column access
first_row = arr[0, :]                 # [1, 2, 3, 4]
second_column = arr[:, 1]             # [2, 6, 10]
last_row = arr[-1, :]                 # [9, 10, 11, 12]

# Slicing
subarray = arr[0:2, 1:3]             # [[2, 3], [6, 7]]
every_other = arr[::2, ::2]          # [[1, 3], [9, 11]]
```

**Advanced Indexing:**
```python
# Boolean indexing
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = data > 5                       # [False, False, False, False, False, True, True, True, True, True]
filtered = data[mask]                 # [6, 7, 8, 9, 10]

# Multiple conditions
mask = (data > 3) & (data < 8)        # [False, False, False, True, True, True, True, False, False, False]
filtered = data[mask]                 # [4, 5, 6, 7]

# Fancy indexing
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])
selected = arr[indices]               # [10, 30, 50]

# 2D fancy indexing
matrix = np.array([[1, 2], [3, 4], [5, 6]])
rows = np.array([0, 2])
cols = np.array([1, 0])
selected = matrix[rows, cols]         # [2, 5] - elements at (0,1) and (2,0)
```

### Array Operations and Broadcasting

**Element-wise Operations:**
```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Basic arithmetic (element-wise)
addition = a + b                      # [11, 22, 33, 44]
multiplication = a * b                # [10, 40, 90, 160]
power = a ** 2                        # [1, 4, 9, 16]

# Mathematical functions
sqrt_vals = np.sqrt(a)                # [1.0, 1.414, 1.732, 2.0]
log_vals = np.log(a)                  # Natural logarithm
sin_vals = np.sin(a)                  # Sine values
```

**Broadcasting - NumPy's Superpower:**
```python
# Broadcasting allows operations between arrays of different shapes
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Add scalar to matrix (broadcasts scalar to all elements)
result = matrix + 10                  # [[11, 12, 13], [14, 15, 16], [17, 18, 19]]

# Add 1D array to 2D matrix (broadcasts across rows)
row_vector = np.array([1, 2, 3])
result = matrix + row_vector          # [[2, 4, 6], [5, 7, 9], [8, 10, 12]]

# Add column vector (broadcasts across columns)
col_vector = np.array([[1], [2], [3]])
result = matrix + col_vector          # [[2, 3, 4], [6, 7, 8], [10, 11, 12]]

# More complex broadcasting
a = np.array([[[1, 2]], [[3, 4]]])    # Shape: (2, 1, 2)
b = np.array([10, 20, 30])            # Shape: (3,)
# Broadcasting will create shape (2, 3, 2)
result = a + b                        # Complex but predictable result
```

**Broadcasting Rules:**
1. Arrays are aligned from the rightmost dimension
2. Dimensions of size 1 can be stretched to match
3. Missing dimensions are assumed to be size 1

### Array Reshaping and Manipulation

**Changing Array Shape:**
```python
arr = np.arange(12)                   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Reshape to 2D
reshaped = arr.reshape(3, 4)          # 3 rows, 4 columns
reshaped = arr.reshape(4, 3)          # 4 rows, 3 columns
reshaped = arr.reshape(2, 6)          # 2 rows, 6 columns

# Automatic dimension calculation
reshaped = arr.reshape(3, -1)         # 3 rows, automatically calculate columns (4)
reshaped = arr.reshape(-1, 2)         # Automatically calculate rows (6), 2 columns

# Flatten to 1D
flattened = reshaped.flatten()        # Creates copy
flattened = reshaped.ravel()          # Returns view if possible (more efficient)

# Transpose
matrix = np.array([[1, 2, 3], [4, 5, 6]])
transposed = matrix.T                 # [[1, 4], [2, 5], [3, 6]]
transposed = np.transpose(matrix)     # Same result
```

**Combining Arrays:**
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Vertical stacking (along rows)
vertical = np.vstack((a, b))          # [[1, 2], [3, 4], [5, 6], [7, 8]]
vertical = np.concatenate((a, b), axis=0)  # Same result

# Horizontal stacking (along columns)
horizontal = np.hstack((a, b))        # [[1, 2, 5, 6], [3, 4, 7, 8]]
horizontal = np.concatenate((a, b), axis=1)  # Same result

# 3D stacking
depth = np.dstack((a, b))             # Stack along third dimension

# Splitting arrays
arr = np.arange(8)
split_arrays = np.split(arr, 4)       # Split into 4 equal parts
split_arrays = np.array_split(arr, 3) # Split into 3 parts (handles uneven splits)
```

### Statistical Operations

**Aggregation Functions:**
```python
data = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])

# Global statistics
total_sum = np.sum(data)              # 78
mean_val = np.mean(data)              # 6.5
std_val = np.std(data)                # Standard deviation
var_val = np.var(data)                # Variance
min_val = np.min(data)                # 1
max_val = np.max(data)                # 12

# Along specific axes
row_sums = np.sum(data, axis=1)       # [10, 26, 42] - sum each row
col_sums = np.sum(data, axis=0)       # [15, 18, 21, 24] - sum each column
col_means = np.mean(data, axis=0)     # [5.0, 6.0, 7.0, 8.0]

# Cumulative operations
cumulative_sum = np.cumsum(data)      # Running sum
cumulative_product = np.cumprod(data) # Running product

# Finding positions
max_index = np.argmax(data)           # 11 (flat index of maximum)
max_indices = np.unravel_index(np.argmax(data), data.shape)  # (2, 3)
```

**Statistical Functions:**
```python
# Generate sample data
np.random.seed(42)  # For reproducible results
sample_data = np.random.normal(100, 15, 1000)  # Mean=100, std=15, n=1000

# Descriptive statistics
print(f"Mean: {np.mean(sample_data):.2f}")
print(f"Median: {np.median(sample_data):.2f}")
print(f"Standard deviation: {np.std(sample_data):.2f}")
print(f"Variance: {np.var(sample_data):.2f}")

# Percentiles
percentiles = np.percentile(sample_data, [25, 50, 75])
print(f"25th, 50th, 75th percentiles: {percentiles}")

# Correlation between arrays
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5  # y correlated with x
correlation = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {correlation:.3f}")
```

### Linear Algebra with NumPy

**Matrix Operations:**
```python
# Matrix creation
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
matrix_mult = np.dot(A, B)            # [[19, 22], [43, 50]]
matrix_mult = A @ B                   # Python 3.5+ syntax
element_mult = A * B                  # Element-wise multiplication

# Matrix properties
determinant = np.linalg.det(A)        # -2.0
inverse = np.linalg.inv(A)            # Matrix inverse
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solving linear systems: Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([8, 13])
x = np.linalg.solve(A, b)             # [1, 6] - solution to system
```

### Performance Considerations

**Vectorization vs Loops:**
```python
import time

# Inefficient: Pure Python loop
def slow_sum_of_squares(arr):
    total = 0
    for element in arr:
        total += element ** 2
    return total

# Efficient: NumPy vectorized operation
def fast_sum_of_squares(arr):
    return np.sum(arr ** 2)

# Performance comparison
large_array = np.random.randn(1000000)

start = time.time()
slow_result = slow_sum_of_squares(large_array)
slow_time = time.time() - start

start = time.time()
fast_result = fast_sum_of_squares(large_array)
fast_time = time.time() - start

print(f"Slow method: {slow_time:.4f} seconds")
print(f"Fast method: {fast_time:.4f} seconds")
print(f"Speedup: {slow_time / fast_time:.1f}x")
```

## Part 2: Pandas - Data Analysis Powerhouse

### Introduction to Pandas

Pandas provides two primary data structures:
- **Series**: 1-dimensional labeled array
- **DataFrame**: 2-dimensional labeled data structure (like a spreadsheet)

Both are built on top of NumPy arrays but add powerful indexing, alignment, and data manipulation capabilities.

### Series: 1D Labeled Data

**Creating Series:**
```python
import pandas as pd
import numpy as np

# From list
temperatures = pd.Series([20, 25, 30, 35, 40])
print(temperatures)

# With custom index
temperatures = pd.Series([20, 25, 30, 35, 40], 
                        index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])

# From dictionary
city_temps = pd.Series({
    'New York': 15,
    'London': 8,
    'Tokyo': 22,
    'Sydney': 25
})

# From NumPy array
random_data = pd.Series(np.random.randn(10))
```

**Series Operations:**
```python
# Accessing elements
monday_temp = temperatures['Mon']         # 20
first_temp = temperatures.iloc[0]         # 20 (position-based)

# Slicing
week_start = temperatures['Mon':'Wed']    # Mon, Tue, Wed
first_three = temperatures.iloc[:3]       # First 3 elements

# Boolean indexing
hot_days = temperatures[temperatures > 25]

# Mathematical operations
celsius_to_fahrenheit = temperatures * 9/5 + 32
temp_diff = temperatures - temperatures.mean()

# String operations (for text data)
cities = pd.Series(['New York', 'Los Angeles', 'Chicago'])
lower_cities = cities.str.lower()
city_lengths = cities.str.len()
```

### DataFrames: 2D Labeled Data

**Creating DataFrames:**
```python
# From dictionary
student_data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [20, 21, 19, 22],
    'grade': [85, 92, 78, 95],
    'major': ['Physics', 'Math', 'Chemistry', 'Biology']
}
df = pd.DataFrame(student_data)

# From list of dictionaries
students = [
    {'name': 'Alice', 'age': 20, 'grade': 85},
    {'name': 'Bob', 'age': 21, 'grade': 92},
    {'name': 'Charlie', 'age': 19, 'grade': 78}
]
df = pd.DataFrame(students)

# From NumPy array with column names
data = np.random.randn(4, 3)
df = pd.DataFrame(data, 
                  columns=['A', 'B', 'C'],
                  index=['Row1', 'Row2', 'Row3', 'Row4'])
```

**Basic DataFrame Properties:**
```python
# Display basic information
print(df.shape)                       # (4, 4) - rows, columns
print(df.columns)                     # Column names
print(df.index)                       # Row indices
print(df.dtypes)                      # Data types of each column
print(df.info())                      # Comprehensive overview
print(df.describe())                  # Statistical summary
```

### Data Selection and Indexing

**Column Selection:**
```python
# Single column (returns Series)
names = df['name']
grades = df.grade                     # Dot notation (if no spaces/special chars)

# Multiple columns (returns DataFrame)
subset = df[['name', 'grade']]

# Column selection with conditions
high_performers = df[df['grade'] > 85]
physics_students = df[df['major'] == 'Physics']
```

**Row Selection:**
```python
# By position (iloc)
first_student = df.iloc[0]            # First row
first_two = df.iloc[:2]               # First two rows
last_student = df.iloc[-1]            # Last row

# By label (loc)
specific_rows = df.loc[0:2]           # Rows 0 through 2 (inclusive)

# Boolean indexing
young_students = df[df['age'] < 21]
top_grades = df[df['grade'] >= 90]

# Multiple conditions
physics_high_performers = df[(df['major'] == 'Physics') & (df['grade'] > 80)]
young_or_top = df[(df['age'] < 21) | (df['grade'] >= 90)]
```

**Advanced Indexing:**
```python
# Setting custom index
df_indexed = df.set_index('name')
alice_data = df_indexed.loc['Alice']

# Multi-level indexing
df['semester'] = ['Fall', 'Fall', 'Spring', 'Spring']
multi_index = df.set_index(['semester', 'name'])
fall_students = multi_index.loc['Fall']
```

## Part 3: Data Loading and File I/O

### Reading Data from Files

**CSV Files:**
```python
# Basic CSV reading
df = pd.read_csv('data.csv')

# Common parameters
df = pd.read_csv('data.csv',
                 sep=',',                    # Delimiter
                 header=0,                   # Which row contains column names
                 index_col=0,                # Which column to use as index
                 names=['col1', 'col2'],     # Custom column names
                 skiprows=1,                 # Skip first row
                 nrows=1000,                 # Read only first 1000 rows
                 encoding='utf-8',           # File encoding
                 na_values=['NA', 'null'])   # Values to treat as NaN

# Handling different delimiters
df = pd.read_csv('data.tsv', sep='\t')      # Tab-separated
df = pd.read_csv('data.txt', sep='|')       # Pipe-separated

# Reading from URL
url = 'https://raw.githubusercontent.com/datasets/iris/master/data/iris.csv'
iris_df = pd.read_csv(url)
```

**Excel Files:**
```python
# Read Excel file
df = pd.read_excel('data.xlsx')

# Specify sheet
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_excel('data.xlsx', sheet_name=0)        # By position

# Read multiple sheets
all_sheets = pd.read_excel('data.xlsx', sheet_name=None)  # Dictionary of DataFrames

# Common parameters
df = pd.read_excel('data.xlsx',
                   sheet_name='Data',
                   header=2,                # Data starts at row 3
                   usecols='A:D',          # Only columns A through D
                   skiprows=[0, 1])        # Skip first two rows
```

**JSON Files:**
```python
# Read JSON file
df = pd.read_json('data.json')

# Different orientations
df = pd.read_json('data.json', orient='records')    # List of objects
df = pd.read_json('data.json', orient='index')      # Object of objects
df = pd.read_json('data.json', orient='values')     # Array of arrays

# From JSON string
json_string = '{"name": ["Alice", "Bob"], "age": [25, 30]}'
df = pd.read_json(json_string)
```

**Other Formats:**
```python
# Parquet (efficient binary format)
df = pd.read_parquet('data.parquet')

# Pickle (Python serialization)
df = pd.read_pickle('data.pkl')

# SQL databases
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table_name', conn)

# HDF5 (hierarchical data format)
df = pd.read_hdf('data.h5', key='data')
```

### Writing Data to Files

```python
# CSV
df.to_csv('output.csv', index=False)           # Don't include row indices
df.to_csv('output.csv', sep='\t')              # Tab-separated

# Excel
df.to_excel('output.xlsx', sheet_name='Data', index=False)

# Multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')

# JSON
df.to_json('output.json', orient='records', indent=2)

# Parquet
df.to_parquet('output.parquet')

# Pickle
df.to_pickle('output.pkl')
```

## Part 4: Data Cleaning and Preprocessing

### Handling Missing Data

**Detecting Missing Data:**
```python
# Create sample data with missing values
data = {
    'name': ['Alice', 'Bob', None, 'Diana'],
    'age': [25, None, 30, 22],
    'salary': [50000, 60000, None, 55000]
}
df = pd.DataFrame(data)

# Check for missing values
print(df.isnull())                    # Boolean DataFrame
print(df.isnull().sum())              # Count missing per column
print(df.isnull().sum().sum())        # Total missing values

# Check for any missing values
print(df.isnull().any())              # Any missing in each column
print(df.isnull().any().any())        # Any missing in entire DataFrame

# Visualize missing data pattern
print(df.info())                      # Shows non-null counts
```

**Handling Missing Data:**
```python
# Drop rows with any missing values
df_complete = df.dropna()

# Drop rows where specific columns are missing
df_no_missing_salary = df.dropna(subset=['salary'])

# Drop columns with any missing values
df_no_missing_cols = df.dropna(axis=1)

# Fill missing values
df_filled = df.fillna(0)              # Fill with zero
df_filled = df.fillna({'age': df['age'].mean(),
                       'name': 'Unknown'})  # Different values per column

# Forward fill and backward fill
df_ffill = df.fillna(method='ffill')  # Forward fill
df_bfill = df.fillna(method='bfill')  # Backward fill

# Interpolate missing values
numeric_df = pd.DataFrame({'values': [1, 2, None, 4, 5, None, 7]})
interpolated = numeric_df.interpolate()
```

### Data Type Conversions

**Understanding and Converting Data Types:**
```python
# Check data types
print(df.dtypes)

# Convert data types
df['age'] = df['age'].astype('float64')
df['name'] = df['name'].astype('string')

# Convert to category (memory efficient for repeated values)
df['department'] = df['department'].astype('category')

# Numeric conversion with error handling
df['salary_numeric'] = pd.to_numeric(df['salary'], errors='coerce')  # Invalid -> NaN

# DateTime conversion
dates = pd.Series(['2023-01-01', '2023-02-01', '2023-03-01'])
dates_converted = pd.to_datetime(dates)

# Multiple formats
dates_mixed = pd.Series(['01/15/2023', '2023-02-20', 'March 10, 2023'])
dates_converted = pd.to_datetime(dates_mixed, infer_datetime_format=True)
```

### String Data Cleaning

**String Operations:**
```python
# Sample messy text data
messy_names = pd.Series([
    '  ALICE SMITH  ',
    'bob johnson',
    'Charlie Brown Jr.',
    'DIANA PRINCE-WAYNE'
])

# Basic cleaning
cleaned = messy_names.str.strip()                    # Remove whitespace
cleaned = cleaned.str.title()                        # Title case
cleaned = cleaned.str.lower()                        # Lowercase
cleaned = cleaned.str.upper()                        # Uppercase

# String replacement
cleaned = messy_names.str.replace('-', ' ')          # Replace hyphens
cleaned = messy_names.str.replace(r'\s+', ' ')       # Replace multiple spaces

# Extract information
first_names = messy_names.str.split().str[0]         # First word
initials = messy_names.str.extract(r'([A-Z])')       # First capital letter

# String validation
emails = pd.Series(['alice@example.com', 'invalid-email', 'bob@test.org'])
valid_emails = emails.str.contains('@')               # Boolean mask
```

### Data Validation and Quality Checks

**Creating Validation Functions:**
```python
def validate_dataframe(df):
    """
    Comprehensive data validation function.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_data': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'issues': []
    }
    
    # Check for issues
    if validation_results['duplicate_rows'] > 0:
        validation_results['issues'].append(f"Found {validation_results['duplicate_rows']} duplicate rows")
    
    missing_percentage = (df.isnull().sum() / len(df) * 100)
    high_missing = missing_percentage[missing_percentage > 50]
    if not high_missing.empty:
        validation_results['issues'].append(f"Columns with >50% missing: {list(high_missing.index)}")
    
    return validation_results

# Example usage
sample_df = pd.DataFrame({
    'id': [1, 2, 3, 2, 5],  # Duplicate
    'name': ['Alice', 'Bob', None, 'Bob', 'Eve'],
    'score': [85, 90, None, 90, 95]
})

validation = validate_dataframe(sample_df)
print("Validation Results:")
for key, value in validation.items():
    print(f"{key}: {value}")
```

### Data Cleaning Pipeline

**Comprehensive Cleaning Function:**
```python
def clean_dataset(df, config=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Raw dataset
        config (dict): Cleaning configuration
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    if config is None:
        config = {
            'drop_duplicates': True,
            'handle_missing': 'drop',  # 'drop', 'fill', 'interpolate'
            'standardize_strings': True,
            'convert_types': True
        }
    
    df_clean = df.copy()
    
    # Remove duplicates
    if config['drop_duplicates']:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_dups = initial_rows - len(df_clean)
        if removed_dups > 0:
            print(f"Removed {removed_dups} duplicate rows")
    
    # Handle missing data
    if config['handle_missing'] == 'drop':
        df_clean = df_clean.dropna()
    elif config['handle_missing'] == 'fill':
        # Fill numeric columns with median, text with mode
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown')
    
    # Standardize strings
    if config['standardize_strings']:
        string_columns = df_clean.select_dtypes(include=['object']).columns
        for col in string_columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
    
    # Convert data types
    if config['convert_types']:
        # Try to convert object columns to numeric
        for col in df_clean.select_dtypes(include=['object']).columns:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col])
            except (ValueError, TypeError):
                pass  # Keep as string if conversion fails
    
    return df_clean

# Example usage
raw_data = pd.DataFrame({
    'name': ['  alice  ', 'BOB', 'charlie', 'alice', None],
    'age': ['25', '30', None, '25', '35'],
    'score': [85.5, 90.0, 78.5, 85.5, 92.0]
})

cleaned_data = clean_dataset(raw_data)
print("Original data:")
print(raw_data)
print("\nCleaned data:")
print(cleaned_data)
print(f"\nData types:\n{cleaned_data.dtypes}")
```

## Part 5: Pandas and NumPy Integration

### When to Use NumPy vs Pandas

**NumPy is ideal for:**
- Pure numerical computations
- Mathematical operations on homogeneous data
- Memory-efficient operations
- Building blocks for other libraries

**Pandas is ideal for:**
- Heterogeneous data (mixed types)
- Data with labels/indices
- Data cleaning and transformation
- Real-world messy data

### Converting Between NumPy and Pandas

```python
# NumPy array to Pandas
np_array = np.random.randn(5, 3)
df_from_numpy = pd.DataFrame(np_array, columns=['A', 'B', 'C'])

# Pandas to NumPy
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
np_array_from_df = df.values           # Or df.to_numpy()

# Series to NumPy
series = pd.Series([1, 2, 3, 4, 5])
np_array_from_series = series.values

# Accessing underlying NumPy array
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
np_array = df['A'].values              # NumPy array of column A
```

### Using NumPy Functions on Pandas Objects

```python
# NumPy functions work directly on Pandas objects
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

# Statistical functions
print(np.mean(df))                     # Column means
print(np.std(df))                      # Column standard deviations
print(np.corrcoef(df['A'], df['B']))   # Correlation coefficient

# Mathematical functions
df['A_sqrt'] = np.sqrt(df['A'])
df['B_log'] = np.log(df['B'])

# Broadcasting still works
df_normalized = (df - np.mean(df)) / np.std(df)
```

### Performance Optimization

**Memory-Efficient Operations:**
```python
# Use appropriate data types
df = pd.DataFrame({
    'small_int': pd.array([1, 2, 3], dtype='int8'),      # 8-bit integer
    'category': pd.Categorical(['A', 'B', 'A']),          # Category for repeated values
    'float32': pd.array([1.1, 2.2, 3.3], dtype='float32') # 32-bit float
})

# Check memory usage
print(df.memory_usage(deep=True))

# Chunking for large files
chunk_size = 10000
processed_chunks = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    processed_chunk = chunk.groupby('category').sum()
    processed_chunks.append(processed_chunk)

# Combine results
final_result = pd.concat(processed_chunks).groupby(level=0).sum()
```

## Summary and Practice Exercises

### What We've Accomplished

Today we've built a solid foundation in the core tools of data science:

1. **NumPy Mastery**: Understanding arrays, broadcasting, reshaping, and vectorized operations
2. **Pandas Fundamentals**: Working with Series and DataFrames for real-world data
3. **Data Loading**: Reading from various file formats with proper configuration
4. **Data Cleaning**: Handling missing data, data types, and validation
5. **Integration**: Understanding how NumPy and Pandas work together

### Key Takeaways

- **Vectorization**: Always prefer NumPy/Pandas operations over Python loops
- **Data Types**: Choose appropriate data types for memory efficiency
- **Missing Data**: Have a strategy for handling missing values
- **Validation**: Always validate your data before analysis
- **Integration**: NumPy and Pandas complement each other perfectly

### Practice Exercises

1. **NumPy Challenge**: Create a function that normalizes a 2D array (subtract mean, divide by standard deviation) for each column using NumPy operations.

2. **Data Loading Project**: Download a real dataset (e.g., from Kaggle) and create a complete data loading and cleaning pipeline.

3. **Performance Comparison**: Compare the performance of a calculation using pure Python loops vs NumPy vectorization.

4. **Data Validation**: Build a comprehensive data validation system that checks for outliers, missing patterns, and data quality issues.

### Preparation for Next Lecture

In our next lecture, we'll dive into advanced data analysis and visualization. To prepare:

1. Practice the concepts covered today, especially data cleaning
2. Install matplotlib and seaborn: `pip install matplotlib seaborn`
3. Find a dataset you're interested in and practice loading and cleaning it
4. Review basic statistics concepts (correlation, distributions)

The foundation you've built today with NumPy and Pandas will serve you throughout your data science journey. Every analysis, every model, every insight starts with properly loaded and cleaned data - and now you have the tools to do it professionally.