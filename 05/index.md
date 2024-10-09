---
marp: true
theme: default
paginate: true
---

# Lecture 05: Welcome to Data Management

1. Remote Jupyter Notebooks
2. Saving your place in SSH: `screen`, `tmux`, `mosh`
3. Introduction to NumPy
4. NumPy Arrays and Operations
5. Introduction to Pandas
6. Working with Data Files in Pandas
7. DataFrame and Series Objects
8. Preview of Data Manipulation with Pandas
9. Data munging in the shell: `cut`, `tr`, & `sed`

---

## Remote Jupyter Notebooks

- Run Jupyter on a remote server, access via web browser
- Benefits: More computing power, centralized data storage
- Use cases: Large datasets, ML model training, collaboration
- Setup: Install on server, configure for remote access, SSH tunneling

---

## Introduction to SSH Session Management

- Importance: Maintains work across disconnections
- Common issues with standard SSH:
  - Lost work on network interruptions
  - Difficulty managing multiple tasks
- Solutions: screen, tmux, mosh

---

## screen

- Terminal multiplexer: multiple virtual terminals in one session
- Persists sessions across disconnects
- Commands:
  - `screen`: Start new session
  - `Ctrl-a d`: Detach
  - `screen -r`: Reattach
  - `Ctrl-a c`: New window
  - `Ctrl-a n`: Next window

#FIXME[add image of screen session]

---

## tmux

- Modern terminal multiplexer, highly customizable
- Commands:
  - `tmux`: Start new session
  - `Ctrl-b d`: Detach
  - `tmux attach`: Reattach
  - `Ctrl-b c`: New window
  - `Ctrl-b n`: Next window

#FIXME[add image of tmux session]

---

## mosh (Mobile Shell)

- SSH replacement for unreliable networks
- Supports roaming and intermittent connectivity
- Pros: Works well on unreliable networks
- Cons: Requires server-side installation

---

## What is NumPy?

- Numerical Python: fundamental for scientific computing
- Provides support for large, multi-dimensional arrays and matrices
- Efficient array operations
- Comprehensive mathematical functions
- Tools for integrating C/C++ and Fortran code
- Linear algebra, Fourier transform, and random number capabilities

---

## The NumPy ndarray Object

- n-dimensional array
- Homogeneous data type
- Fixed size at creation

#FIXME[add image of ndarray structure]

---

## Creating and Manipulating NumPy Arrays

```python
import numpy as np

# Creation
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3, 3))
c = np.ones((2, 2))
d = np.arange(0, 10, 2)

# Manipulation
e = a.reshape((5, 1))
f = b[1:, 1:]
g = c + 10
h = d * 2
```

---

## Basic Array Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
c = a + b
d = a * b

# Matrix operations
e = np.dot(a, b)
f = np.outer(a, b)
```

---

## Universal Functions (ufuncs)

- Fast element-wise array operations
- Examples: 

```python
a = np.array([1, 4, 9])
b = np.sqrt(a)  # [1, 2, 3]
c = np.exp(a)   # [2.72, 54.60, 8103.08]
d = np.sin(a)   # [0.84, -0.76, 0.41]
```

---

## Broadcasting

- Allows operations on arrays of different sizes
- NumPy's way of treating arrays with different shapes during arithmetic

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
c = a * b  # Element-wise: [10, 40, 90, 160]

d = np.array([[1, 2, 3], [4, 5, 6]])
e = np.array([10, 20, 30])
f = d + e  # Broadcasting: [[11, 22, 33], [14, 25, 36]]
```

#FIXME[add image illustrating broadcasting]

---

## What is Pandas?

- Python Data Analysis Library
- Built on top of NumPy
- Provides high-performance, easy-to-use data structures and tools
- Designed for working with labeled and relational data
- Key data structures: Series (1D) and DataFrame (2D)

---

## Pandas Series

- 1-dimensional labeled array
- Can hold data of any type

```python
import pandas as pd

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)
```

Output:
```
0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64
```

---

## Pandas DataFrame

- 2-dimensional labeled data structure
- Like a spreadsheet or SQL table

```python
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)
print(df)
```

Output:
```
   A  B  C
0  1  4  7
1  2  5  8
2  3  6  9
```

---

## Creating Series and DataFrames

```python
# Series from dictionary
d = {'a': 1, 'b': 2, 'c': 3}
series = pd.Series(d)

# DataFrame from list of dictionaries
data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)

print(series)
print("\n")
print(df)
```

---

## Reading Data Files in Pandas

```python
# CSV
df_csv = pd.read_csv('filename.csv')

# JSON
df_json = pd.read_json('filename.json')

# Excel
df_excel = pd.read_excel('filename.xlsx', sheet_name='Sheet1')

# Handling data types
df_types = pd.read_csv('filename.csv', 
                       dtype={'column1': str, 'column2': float})
```

---

## Writing Data Files in Pandas

```python
# CSV
df.to_csv('output.csv', index=False)

# JSON
df.to_json('output.json')

# Excel
df.to_excel('output.xlsx', sheet_name='Sheet1')
```

---

## Dealing with Missing Data

```python
# Checking for missing values
print(df.isnull().sum())

# Dropping missing values
df_clean = df.dropna()

# Filling missing values
df_filled = df.fillna(value={'column1': 0, 'column2': 'Unknown'})
```

---

## Basic DataFrame Operations

```python
# Viewing data
print(df.head())
print(df.tail())

# Information about DataFrame
print(df.info())
print(df.describe())

# Selecting columns
print(df['column_name'])
print(df[['column1', 'column2']])
```

---

## Accessing Data in DataFrames

```python
# Using loc for label-based indexing
print(df.loc['row_label', 'column_label'])

# Using iloc for integer-based indexing
print(df.iloc[0, 2])

# Boolean indexing
print(df[df['column'] > 5])
```

---

## Preview: Advanced Pandas Operations

- Indexing and Selection: `.loc[]`, `.iloc[]`, Boolean indexing
- Data Cleaning: Handle missing data, remove duplicates, type conversion
- Merging and joining data
- Grouping and aggregation
- Pivoting and reshaping data
- Time series functionality

---

## Data Munging in the Shell

- Unix command-line tools for text processing
- Useful for quick data transformations
- Can be combined with pipes for complex operations

---

## cut: Extracting Columns

- Selects specific columns from tabular data
- Usage: `cut OPTION... [FILE]...`

Examples:
```bash
# Extract 1st and 3rd columns from CSV
cut -d',' -f1,3 data.csv

# Extract characters 5-10 from each line
cut -c5-10 data.txt
```

---

## tr: Translating Characters

- Transforms or deletes characters
- Usage: `tr [OPTION]... SET1 [SET2]`

Examples:
```bash
# Convert lowercase to uppercase
cat file.txt | tr 'a-z' 'A-Z'

# Delete all digits
echo "hello123world" | tr -d '0-9'

# Squeeze repeated characters
echo "hello    world" | tr -s ' '
```

---

## sed: Stream Editor

- Powerful text transformation tool
- Can search, find and replace, insert, and delete

Examples:
```bash
# Replace first occurrence of 'old' with 'new'
sed 's/old/new/' file.txt

# Replace all occurrences
sed 's/old/new/g' file.txt

# Delete lines containing 'pattern'
sed '/pattern/d' file.txt

# Insert 'text' at beginning of each line
sed 's/^/text /' file.txt
```

---

## Combining Shell Commands

- Use pipes (`|`) to chain commands
- Create powerful data processing pipelines

Example:
```bash
# Extract 2nd column, convert to uppercase, 
# replace spaces with underscores
cat data.csv | cut -d',' -f2 | tr 'a-z' 'A-Z' | tr ' ' '_'
```
