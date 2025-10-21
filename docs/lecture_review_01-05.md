# Comprehensive Lecture Review: Lectures 01-05
## What Students Should Know by Assignment 05

---

## Lecture 01: Command Line & Python Basics

### Command Line Essentials

#### Navigation
- `pwd` - Print working directory
- `ls` / `ls -la` - List directory contents (with details)
- `cd [path]` / `cd ..` / `cd ~` - Change directory
- File viewing: `cat`, `head`, `tail`, `head -n N`, `tail -n N`

#### File Operations
- `mkdir [name]` / `mkdir -p [path]` - Create directories
- `touch [filename]` - Create empty file
- `cp [source] [dest]` - Copy files
- `mv [source] [dest]` - Move/rename files
- `rm [file]` / `rm -r [dir]` - Remove files/directories

#### Help & Control
- `man [command]` - Manual pages
- `which [command]` - Find command location
- `Ctrl+C` - Cancel current command
- `exit` - Close terminal

### Python Fundamentals

#### Data Types & Variables
- **Numbers**: `int`, `float`, scientific notation (`1.4e9`)
- **Strings**: Text data, string methods (`.upper()`, `.lower()`, `.strip()`)
- **Booleans**: `True`, `False`, logical operators (`and`, `or`, `not`)
- **Type checking**: `type()`, `isinstance()`

#### Basic Operations
- Arithmetic: `+`, `-`, `*`, `/`, `//` (integer division), `%` (modulo), `**` (power)
- String concatenation: `+` operator
- F-strings: `f"Hello {name}"` (preferred formatting method)

#### Control Structures
- **If statements**: `if`, `elif`, `else`
- **For loops**: `for item in iterable:`, `range(n)`
- **Comparison operators**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Membership**: `in`, `not in`

#### Input/Output
- `print()` - Basic output
- F-strings for formatting: `f"{value:.2f}"` (2 decimal places)
- `input()` - User input (returns string)

#### Debugging Basics
- Reading error messages: `NameError`, `TypeError`, `ValueError`
- Using `print()` for debugging
- Checking types: `type(variable)`

---

## Lecture 02: Git, VS Code, and Python Deeper

### VS Code Essentials

#### Interface
- Command Palette: `Cmd+Shift+P` (Mac) / `Ctrl+Shift+P` (Windows)
- Quick Open: `Cmd+P` / `Ctrl+P`
- Terminal: `Ctrl+\``
- Explorer: `Cmd+Shift+E` / `Ctrl+Shift+E`
- Source Control: `Cmd+Shift+G` / `Ctrl+Shift+G`

#### Settings
- Format on Save
- Python Interpreter selection
- Extensions: Python, Pylance, Jupyter, Markdown

### Git Version Control

#### Core Concepts
- **Repository**: Project folder with version history
- **Commit**: Snapshot of project at specific time
- **Remote**: Shared version (e.g., GitHub)
- **Branch**: Parallel timeline (main branch is default)

#### Essential Commands
- `git init` - Initialize repository
- `git add [file]` - Stage changes
- `git commit -m "message"` - Create commit
- `git status` - Show working directory status
- `git log` - Show commit history
- `git diff` - Show changes
- `git clone [url]` - Copy remote repository
- `git push` / `git pull` - Sync with remote

#### .gitignore
- Pattern matching for files to ignore
- Common patterns: `*.pyc`, `__pycache__/`, `.env`, `data/*.csv`
- Use `!` to negate patterns

### Markdown Documentation
- Headers: `#`, `##`, `###`
- Bold: `**text**`, Italic: `*text*`
- Code: `` `inline` ``, code blocks: ``` ```language ```
- Lists: `- item` or `1. item`
- Links: `[text](url)`, Images: `![alt](url)`

### Python Advanced Basics

#### Object Model
- Everything is an object with type information
- Duck typing: "If it walks like a duck..."
- Introspection: `type()`, `dir()`, `help()`, `isinstance()`

#### Data Structures
- **Lists**: `[1, 2, 3]`, mutable, `.append()`, `.insert()`, `.remove()`, `.pop()`
- **Tuples**: `(1, 2, 3)`, immutable, unpacking
- **Dictionaries**: `{key: value}`, `.get()`, `.keys()`, `.values()`, `.items()`
- **Sets**: `{1, 2, 3}`, unique values, `.union()`, `.intersection()`

#### String Operations
- `.strip()`, `.lower()`, `.upper()`, `.title()`
- `.split(sep)`, `.replace(old, new)`
- `.startswith()`, `.endswith()`, `.find()`, `.index()`
- `.isdigit()`, `.isalpha()`, `.isalnum()`

#### List Comprehensions
- `[expr for item in iterable if condition]`
- `enumerate(iterable)` - Get index and value
- `zip(iter1, iter2)` - Combine sequences
- `sorted()`, `reversed()`, `sum()`, `min()`, `max()`, `len()`

#### Functions
- `def function_name(parameters):` - Definition
- `return value` - Return value
- Default parameters: `def func(param=default):`
- `if __name__ == "__main__":` - Script execution guard

#### File I/O
- `open(file, mode)` - Modes: `'r'` (read), `'w'` (write), `'a'` (append)
- `with open(file) as f:` - Context manager (auto-closes)
- `file.read()`, `file.readline()`, `file.readlines()`
- `file.write(string)`

---

## Lecture 03: NumPy and Virtual Environments

### Virtual Environments

#### venv (Standard)
```bash
python -m venv env_name
source env_name/bin/activate  # Mac/Linux
env_name\Scripts\activate     # Windows
pip install package_name
pip freeze > requirements.txt
deactivate
```

#### uv (Modern)
```bash
uv venv env_name
source env_name/bin/activate
uv pip install package_name
uv pip freeze > requirements.txt
```

#### conda
```bash
conda create -n env_name python=3.11
conda activate env_name
conda install package_name
conda env export > environment.yml
```

### NumPy Arrays

#### Array Creation
- `np.array([1, 2, 3])` - From list
- `np.zeros(5)`, `np.ones((2, 3))` - Filled arrays
- `np.arange(10)` - Range of values
- `np.full((2, 3), 7)` - Filled with specific value

#### Array Properties
- `.shape` - Dimensions (e.g., `(2, 3)`)
- `.ndim` - Number of dimensions
- `.size` - Total number of elements
- `.dtype` - Data type

#### Data Types
- `dtype=np.int32`, `dtype=np.float64`
- `.astype(np.float64)` - Type conversion
- String to numeric: `str_arr.astype(float)`

#### Indexing & Slicing
- Basic: `arr[0]`, `arr[-1]`, `arr[2:7]`, `arr[::2]`
- 2D: `arr_2d[0]`, `arr_2d[1, 2]`, `arr_2d[:2]`, `arr_2d[:, 1]`
- Boolean indexing: `arr[arr > 5]`, `arr[(arr > 2) & (arr < 8)]`
- Fancy indexing: `arr[[1, 3, 5]]`

#### Views vs Copies
- Slicing creates views (shares memory)
- `.copy()` creates independent copy
- **Critical**: Modifying a view modifies the original array

#### Operations
- Vectorized operations: `arr1 + arr2`, `arr * 2`, `arr ** 2`
- Statistical: `.mean()`, `.std()`, `.max()`, `.min()`, `.sum()`
- Axis-specific: `.mean(axis=0)` (columns), `.mean(axis=1)` (rows)

#### Array Manipulation
- Reshaping: `.reshape(3, 4)`, `.flatten()`
- Transposing: `.T`
- Sorting: `.sort()` (in-place), `np.sort()` (returns copy)

#### Universal Functions (ufuncs)
- `np.sqrt()`, `np.exp()`, `np.log()`
- `np.maximum()`, `np.minimum()`

#### Conditional Logic
- `np.where(condition, true_val, false_val)`
- `.any()`, `.all()` - Boolean array checks

#### Random Number Generation
```python
rng = np.random.default_rng(seed=42)  # Reproducible
rng.random(5)              # Random floats [0, 1)
rng.integers(1, 10, 5)     # Random integers
rng.standard_normal(5)     # Normal distribution
```

### Command Line Data Processing
- `cut -d',' -f1,3` - Extract columns
- `sort -n`, `sort -k2` - Sort data
- `uniq`, `uniq -c` - Unique values, count occurrences
- `grep`, `tr`, `sed`, `awk` - Text processing
- Pipes: `|` - Chain commands
- Redirection: `>` (write), `>>` (append), `<` (input)

---

## Lecture 04: Pandas Fundamentals

### Jupyter Notebooks

#### Interface
- **Code cells**: Execute Python code (`Shift+Enter`)
- **Markdown cells**: Documentation and explanations
- **Magic commands**: `%matplotlib inline`, `%pwd`, `%pip install`

#### display() vs print()
- `print()` - Plain text, works everywhere (scripts and notebooks)
- `display()` - Rich HTML tables, **Jupyter only**
- Use `display()` for DataFrames in notebooks for better formatting

#### Kernel Management
- Restart Kernel: Fresh start, clears all variables
- Run All: Execute all cells top to bottom
- Restart & Run All: Test notebook runs from scratch
- **Critical**: Clear outputs before committing to git (removes sensitive data)

### Pandas Data Structures

#### Series
- 1D labeled array: `pd.Series([1, 2, 3], index=['a', 'b', 'c'])`
- Properties: `.index`, `.values`, `.name`, `.dtype`, `.size`
- Methods: `.head()`, `.tail()`, `.describe()`, `.value_counts()`

#### DataFrame
- 2D labeled table: `pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})`
- Properties: `.index`, `.columns`, `.shape`, `.dtypes`, `.values`
- Methods: `.head()`, `.tail()`, `.info()`, `.describe()`, `.sample()`

### Column Selection
- Single column (Series): `df['column']`
- Single column (DataFrame): `df[['column']]`
- Multiple columns: `df[['col1', 'col2']]`
- Dot notation: `df.column` (avoid if column has spaces/special chars)
- By type: `df.select_dtypes(include=['number'])`

### Data Selection and Indexing

#### .loc vs .iloc
- **`.loc[row_label, col_label]`** - Uses labels (names)
- **`.iloc[row_pos, col_pos]`** - Uses positions (0, 1, 2...)
- **Slicing difference**:
  - `.loc[1:3]` - Includes BOTH endpoints (1, 2, AND 3)
  - `.iloc[1:3]` - Excludes end (1, 2, but NOT 3)

#### Boolean Indexing
- `df[df['Age'] > 30]` - Filter rows by condition
- Multiple conditions: `df[(df['Age'] > 30) & (df['Salary'] < 60000)]`
- Use `&` (AND), `|` (OR), not `and`/`or`

### Adding Columns
- Direct assignment: `df['new_col'] = expression`
- `.assign()`: `df.assign(new_col=lambda d: d['col'] * 2)` (returns new DataFrame)
- `.insert(loc, 'col', values)` - Insert at specific position

### Missing Data
- Detection: `.isnull()`, `.notnull()`, `.isna()`, `.notna()`
- Count: `.isnull().sum()`
- Removal: `.dropna()`, `.dropna(subset=['col'])`, `.dropna(how='all')`
- Filling:
  - `.fillna(value)` - Constant value
  - `.fillna(df.mean())` - Column mean
  - `.fillna(df.median())` - Column median
  - `.ffill()` - Forward fill (use previous value)
  - `.bfill()` - Backward fill (use next value)
  - `.interpolate()` - Interpolate values

### Data Type Conversion
- `.astype('int64')`, `.astype('float64')`, `.astype('string')`
- `pd.to_datetime(df['date_col'])`
- `pd.to_numeric(df['col'], errors='coerce')` - Invalid → NaN

### Essential Operations

#### Sorting
- By values: `.sort_values('column')`, `.sort_values(['col1', 'col2'])`
- By index: `.sort_index()`
- Descending: `ascending=False`

#### Unique Values
- `.unique()` - Array of unique values
- `.nunique()` - Count unique values
- `.value_counts()` - Frequency of each value
- `.isin(['A', 'B'])` - Membership check

#### GroupBy
- Basic: `df.groupby('col')['target'].mean()`
- Aggregation: `df.groupby('col').agg(['mean', 'count', 'sum'])`
- Multiple keys: `df.groupby(['col1', 'col2']).sum()`
- Filter: `grouped.filter(lambda g: len(g) > 5)`
- Transform: `grouped.transform(lambda x: x - x.mean())`

### Data I/O

#### CSV
- Read: `pd.read_csv('file.csv', sep=',', header=0, index_col=None)`
- Write: `df.to_csv('file.csv', index=False, na_rep='')`

#### Excel
- Read: `pd.read_excel('file.xlsx', sheet_name='Sheet1')`
- Write: `df.to_excel('file.xlsx', sheet_name='Data', index=False)`

#### JSON
- Read: `pd.read_json('file.json', orient='records')`
- Write: `df.to_json('file.json', orient='records', indent=2)`

### Summary Statistics
- `.describe()` - Numeric summary (count, mean, std, min, quartiles, max)
- `.describe(include='all')` - All columns (numeric + categorical)
- `.info()` - Data types, non-null counts, memory usage
- `.count()` - Non-null values per column
- `.nunique()` - Unique values per column
- `.memory_usage()` - Memory per column

---

## Lecture 05: Data Cleaning

### Data Cleaning Workflow
1. Load and inspect data
2. Handle missing values
3. Remove duplicates
4. Convert data types
5. Handle outliers
6. Validate data quality
7. Export clean data

### Missing Data Patterns
- **MCAR** (Missing Completely At Random)
- **MAR** (Missing At Random)
- **MNAR** (Missing Not At Random)

### Missing Data Techniques

#### Detection
- `.isnull()`, `.notnull()`, `.isna()`, `.notna()`
- `.isnull().sum()` - Count per column
- `.isnull().sum(axis=1)` - Count per row
- `.isnull().mean()` - Proportion missing
- `.isnull().any()`, `.isnull().all()` - Boolean checks

#### Removal
- `.dropna()` - Remove rows with any missing
- `.dropna(axis=1)` - Remove columns with any missing
- `.dropna(thresh=n)` - Keep rows with at least n non-null

#### Imputation
- `.fillna(0)` - Constant value
- `.fillna(df.mean())` - Mean imputation
- `.fillna(df.median())` - Median imputation
- `.fillna(df.mode().iloc[0])` - Mode imputation
- `.ffill()` - Forward fill (previous value)
- `.bfill()` - Backward fill (next value)
- `.interpolate()` - Linear interpolation

### Data Transformation

#### Removing Duplicates
- `.duplicated()` - Check for duplicates
- `.duplicated().sum()` - Count duplicates
- `.drop_duplicates()` - Remove duplicates
- `.drop_duplicates(subset=['col1', 'col2'])` - Based on specific columns
- `.drop_duplicates(keep='first')` - Keep first occurrence

#### Replacing Values
- `.replace(old, new)` - Single value
- `.replace([val1, val2], new)` - Multiple → same replacement
- `.replace([val1, val2], [new1, new2])` - Multiple → different replacements
- `.replace({val1: new1, val2: new2})` - Dictionary mapping
- `.replace(regex=True)` - Regular expressions

#### Applying Functions
- `.map(func)` - Series element-wise
- `.apply(func)` - Series or DataFrame
- `.apply(func, axis=0)` - Apply to columns (default)
- `.apply(func, axis=1)` - Apply to rows
- Lambda functions: `lambda x: x * 2`

#### Renaming
- `.rename(columns={'old': 'new'})` - Specific columns
- `.rename(columns=str.lower)` - Function to all columns
- `.rename(columns=str.strip)` - Remove whitespace
- `.rename(index={'old': 'new'})` - Rename rows
- `inplace=True` - Modify in place

#### Creating Categories
- `pd.cut(series, bins=[0, 18, 35, 50, 100])` - Equal-width bins
- `pd.qcut(series, q=4)` - Equal-frequency bins (quartiles)
- `labels=['Young', 'Middle', 'Senior']` - Custom labels

#### Outlier Detection & Handling
- `.describe()` - Check for extreme values
- Z-score method: `abs(df - df.mean()) < 3 * df.std()`
- IQR method:
  ```python
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
  IQR = Q3 - Q1
  lower = Q1 - 1.5 * IQR
  upper = Q3 + 1.5 * IQR
  df_clean = df[(df >= lower) & (df <= upper)]
  ```
- `.clip(lower, upper)` - Cap values at bounds

### Categorical Data Encoding

#### Categorical Type
- `.astype('category')` - Convert to categorical
- `.cat.categories` - View categories
- `.cat.codes` - View numeric codes
- Memory efficient for repeated values

#### Dummy Variables
- `pd.get_dummies(series)` - Create binary columns
- `prefix='category'` - Add prefix to column names
- `drop_first=True` - Avoid multicollinearity
- `dtype='int64'` - Use int64 (not bool - can't represent NaN)

### String Manipulation

#### Basic Operations
- `.str.upper()`, `.str.lower()`, `.str.title()`
- `.str.strip()`, `.str.lstrip()`, `.str.rstrip()`
- `.str.replace(old, new)`
- `.str.contains(pattern)` - Check if contains
- `.str.startswith(prefix)`, `.str.endswith(suffix)`

#### Splitting & Joining
- `.str.split(sep)` - Split into list
- `.str.split(sep, expand=True)` - Split into columns
- `.str.cat(sep=' ')` - Join strings
- `.str[0]` - First character
- `.str[:3]` - First 3 characters

### Random Sampling

#### Sampling
- `.sample(n=10)` - Sample n rows
- `.sample(frac=0.5)` - Sample fraction of rows
- `.sample(replace=True)` - Sample with replacement (bootstrap)
- `.sample(random_state=42)` - Reproducible sampling
- Stratified: `df.groupby('col').apply(lambda x: x.sample(2))`

#### Permutation
- `.sample(frac=1)` - Shuffle all rows
- `.sample(n=len(df), replace=True)` - Bootstrap
- `np.random.permutation(df.index)` - Permute index

### Data Validation

#### Quality Checks
- `.isnull().sum()` - Missing values
- `.duplicated().sum()` - Duplicate rows
- `.nunique()` - Unique values per column
- `.dtypes` - Data types
- `.describe()` - Summary statistics
- `.info()` - Detailed information
- `.memory_usage()` - Memory usage

#### Validation Rules
- `.between(left, right)` - Check if in range
- `.isin(values)` - Check membership
- `.str.contains(pattern)` - Pattern matching
- `.str.match(pattern)` - Regex matching
- `.str.len()` - String length
- `.str.isdigit()` - Check if digits

### Running Notebooks from Command Line
```bash
# Execute notebook
jupyter nbconvert --execute --to notebook notebook.ipynb

# Execute with output to new file
jupyter nbconvert --execute --to notebook --output new_name notebook.ipynb

# Execute in place
jupyter nbconvert --execute --to notebook --inplace notebook.ipynb

# Pipeline with error checking
jupyter nbconvert --execute --to notebook notebook.ipynb || {
    echo "ERROR: Notebook failed"
    exit 1
}
```

---

## Key Pedagogical Gaps Identified

### Topics Taught in Lectures
✅ **Covered thoroughly:**
- Command line basics
- Python fundamentals
- Git version control
- NumPy arrays
- Pandas DataFrames
- Data selection (.loc, .iloc)
- Missing data handling
- Basic data cleaning

### Topics NOT Covered in Lectures 01-05

**Advanced Pandas:**
- Merging/joining DataFrames
- Pivot tables and reshaping
- Time series analysis
- Advanced groupby operations
- Multi-indexing
- Window functions

**Data Visualization:**
- matplotlib basics
- seaborn
- Plotting directly from pandas

**Statistical Concepts:**
- Correlation analysis
- Distribution analysis
- Hypothesis testing
- Statistical significance

**Advanced File Operations:**
- Working with multiple files
- Batch processing
- Complex data pipelines

**Programming Patterns:**
- Error handling (try/except)
- Classes and objects
- Modules and packages
- Testing code

### Assignment 05 Expectations

**Students should be comfortable with:**
1. Command line file operations
2. Git version control basics
3. Python fundamentals (loops, functions, data types)
4. NumPy array operations
5. Pandas DataFrame manipulation
6. Reading/writing CSV files
7. Missing data detection and handling
8. Basic data cleaning workflow
9. Jupyter notebook execution
10. Data type conversions

**Students may need help with:**
- Complex data transformations not explicitly covered
- Combining multiple techniques in sequence
- Interpreting statistical measures not yet discussed
- Advanced pandas operations beyond basics
- Efficient coding patterns and best practices
