# Lectures 01-06: Allowed Concepts Inventory

**Purpose:** This document establishes the baseline of concepts covered in lectures 01-06. Use this to validate that assignment solutions only use techniques taught in class.

---

## Lecture 01: Getting Started - Command Line & Python Basics

### Command Line Concepts
- **Terminal access**: WSL, PowerShell, Terminal, GitHub Codespaces
- **Navigation commands**: `pwd`, `ls`, `ls -la`, `cd`, `cd ..`, `cd ~`, `cd -`
- **File/Directory operations**: `mkdir`, `mkdir -p`, `touch`, `cp`, `mv`, `rm`, `rm -r`
- **Viewing files**: `cat`, `head`, `tail`, `head -n`, `tail -n`
- **Help commands**: `man`, `--help`, `which`
- **Control**: Ctrl+C (cancel), `exit`

### Python Fundamentals
- **Running Python**: Interactive mode (REPL), script mode, `python script.py`
- **Variables and data types**: `int`, `float`, `str`, `bool`, `None`
- **Type checking**: `type()`, `isinstance()`
- **Arithmetic operations**: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- **String operations**: `.strip()`, `.lower()`, `.upper()`, `.title()`, `.split()`, `.replace()`, `.startswith()`, `.endswith()`, `.find()`, `.index()`, `.isdigit()`, `.isalpha()`, `.isalnum()`
- **F-string formatting**: `f"text {variable}"`, `f"{value:.2f}"`, `f"{value:>8.1f}"`
- **Comparison operators**: `==`, `!=`, `<`, `>`, `<=`, `>=`, `in`, `not in`
- **Boolean logic**: `and`, `or`, `not`
- **Control structures**:
  - `if`/`elif`/`else`
  - `for item in iterable`
  - `while condition`
  - `break`, `continue`
  - `range(start, stop, step)`
- **Data structures**:
  - Lists: `[item1, item2]`, `.append()`, `.insert()`, `.remove()`, `.pop()`, slicing
  - Tuples: `(item1, item2)`, unpacking
  - Dictionaries: `{key: value}`, `.get()`, `.keys()`, `.values()`, `.items()`
  - Sets: `{item1, item2}`, `.union()`, `.intersection()`, `.difference()`
- **List comprehensions**: `[expr for item in iterable if condition]`
- **Functions**:
  - `def function_name(parameters):`
  - `return value`
  - Default parameters
- **Built-in functions**: `sum()`, `min()`, `max()`, `len()`, `sorted()`, `reversed()`, `enumerate()`, `zip()`
- **Print and output**: `print()`, multiple values, f-strings
- **Error reading**: `NameError`, `TypeError`, `ValueError`, reading error messages
- **File I/O**:
  - `open(file, mode)` with modes: `'r'`, `'w'`, `'a'`, `'x'`
  - `.read()`, `.readline()`, `.readlines()`, `.write()`
  - Context manager: `with open() as file:`
  - Printing to files: `print(value, file=file_handle)`

---

## Lecture 02: Git, Version Control & Python Deepening

### VS Code Basics
- **Command Palette**: `Cmd+Shift+P` (Mac), `Ctrl+Shift+P` (Windows/Linux)
- **Quick Open**: `Cmd+P` (Mac), `Ctrl+P` (Windows/Linux)
- **Source Control panel**: `Cmd+Shift+G` (Mac), `Ctrl+Shift+G` (Windows/Linux)
- **Terminal**: `Ctrl+`` (backtick)
- **Extensions**: Installing and managing extensions
- **Settings**: Format on Save, Python interpreter selection

### Git Version Control
- **Basic Git commands**:
  - `git init` - Initialize repository
  - `git clone [url]` - Copy remote repository
  - `git status` - Show working directory status
  - `git add [file]` - Stage changes
  - `git commit -m "message"` - Create commit
  - `git push [remote] [branch]` - Send commits to remote
  - `git pull [remote] [branch]` - Fetch and merge from remote
  - `git log` - Show commit history
  - `git diff` - Show changes
- **Branching concepts**: `main` branch, creating branches, switching branches
- **Remotes**: GitHub as remote repository
- **Commit messages**: Good vs bad practices
- **.gitignore**: Patterns to exclude files/directories

### Markdown Documentation
- **Headers**: `#`, `##`, `###`
- **Formatting**: `**bold**`, `*italic*`
- **Code**: `` `inline code` ``, ``` code blocks ```
- **Lists**: `- item` (unordered), `1. item` (ordered)
- **Links**: `[text](url)`
- **Images**: `![alt](url)`
- **Tables**: `| col1 | col2 |`

### Python Language Semantics
- **Indentation**: 4 spaces for code blocks (required!)
- **Comments**: `#` for single-line comments
- **Object introspection**: `type()`, `dir()`, `help()`, `id()`
- **Duck typing**: If it walks like a duck and quacks like a duck...
- **Scalar types**: `int`, `float`, `str`, `bool`, `None` (revisited)
- **String methods** (deeper): All from Lecture 01 plus context
- **Print vs display**: `print()` works everywhere, `display()` is Jupyter-only
- **File I/O** (expanded): All modes, context managers, common patterns

---

## Lecture 03: NumPy Arrays & Virtual Environments

### Virtual Environments
- **venv**: `python -m venv name`, activation (Mac/Linux/Windows), `deactivate`
- **uv**: Fast alternative to venv with similar commands
- **conda**: `conda create`, `conda activate`, `conda deactivate`
- **Package management**: `pip install`, `pip freeze > requirements.txt`, `uv pip install`

### NumPy Fundamentals
- **Importing**: `import numpy as np`
- **Array creation**:
  - `np.array([list])` - From Python list
  - `np.zeros(shape)`, `np.ones(shape)` - Filled arrays
  - `np.arange(start, stop, step)` - Range of values
  - `np.full(shape, value)` - Fill with value
- **Array properties**: `.shape`, `.ndim`, `.size`, `.dtype`
- **Data types**: `np.int32`, `np.int64`, `np.float64`, `.astype()`
- **Type conversion**: String to numeric with `.astype(float)`

### Array Indexing and Slicing
- **Basic indexing**: `arr[0]`, `arr[-1]`, `arr[start:stop:step]`
- **Multidimensional indexing**: `arr_2d[row, col]`, `arr_2d[:, col]`
- **Boolean indexing**: `arr[arr > 5]`, `mask = arr > 5; arr[mask]`
- **Multiple conditions**: `(arr > 2) & (arr < 8)` using `&`, `|`
- **Fancy indexing**: `arr[[1, 3, 5]]` - integer array indexing
- **Views vs copies**: Slicing creates views (shared memory), `.copy()` creates independent copy

### NumPy Operations
- **Arithmetic**: Element-wise `+`, `-`, `*`, `/`, `**`, scalar operations
- **Vectorized operations**: Operations on entire arrays at once
- **Statistical functions**: `.mean()`, `.std()`, `.max()`, `.min()`, `.sum()`
- **Axis operations**: `axis=0` (columns), `axis=1` (rows)
- **Reshaping**: `.reshape()`, `.flatten()`, `.T` (transpose)
- **Universal functions (ufuncs)**: `np.sqrt()`, `np.exp()`, `np.maximum()`
- **Conditional logic**: `np.where(condition, true_value, false_value)`
- **Boolean array methods**: `.any()`, `.all()`
- **Sorting**: `.sort()` (in-place), `np.sort()` (returns sorted copy)
- **Random number generation**: `np.random.default_rng()`, `.random()`, `.integers()`, `.standard_normal()`

### Command Line Data Processing
- **Text processing**: `cut`, `sort`, `uniq`, `grep`, `tr`, `sed`, `awk`
- **Pipelines**: `|` (pipe), `>` (redirect), `>>` (append), `<` (input)
- **Command chaining**: `&&` (and), `||` (or)

---

## Lecture 04: Pandas on Jupyter - Data Structures & I/O

### Jupyter Notebooks
- **Interface**: Code cells, markdown cells
- **Execution**: `Shift+Enter` (run and advance), `Ctrl+Enter` (run and stay)
- **Cell management**: Add above/below, delete cells
- **Magic commands**:
  - `%matplotlib inline` - Display plots inline
  - `%pwd`, `%ls` - Directory operations
  - `%pip install package` - Install packages
- **Output**: `print()` vs `display()` - display() is Jupyter-specific for rich formatting
- **Kernel management**: Restart kernel, run all, restart & run all
- **Clear outputs**: Important before Git commits!

### Pandas Data Structures
- **Importing**: `import pandas as pd`
- **Series**:
  - `pd.Series(data, index=None, name=None)`
  - `.index`, `.values`, `.name`, `.dtype`, `.size`
  - `.head(n)`, `.tail(n)`, `.describe()`, `.value_counts()`
- **DataFrame**:
  - `pd.DataFrame(data, index=None, columns=None)`
  - `.index`, `.columns`, `.values`, `.shape`, `.dtypes`
  - `.info()`, `.describe()`, `.head(n)`, `.tail(n)`, `.sample(n)`

### Column Selection
- `df['column']` - Single column (returns Series)
- `df[['col1', 'col2']]` - Multiple columns (returns DataFrame)
- `df.column_name` - Dot notation (if valid Python identifier)
- `df.select_dtypes(include=['number'])` - Select by data type

### Data Selection and Indexing
- **Label-based**: `df.loc[row_labels, column_labels]`
- **Position-based**: `df.iloc[row_positions, column_positions]`
- **Slicing differences**: `.loc[1:3]` includes 3, `.iloc[1:3]` excludes 3
- **Boolean indexing**: `df[df['column'] > value]`
- **Query method**: `df.query("expression")`
- **Membership**: `df['col'].isin(['A', 'B'])`, `df['col'].between(left, right)`

### Adding Columns
- `df['new_col'] = expression` - Direct assignment
- `df.assign(new=lambda d: d['col'] * 2)` - Functional approach
- `df.insert(loc, column, value)` - Control position

### Missing Data
- **Detection**: `df.isnull()`, `df.notnull()`, `.isnull().sum()`
- **Filling**: `df.fillna(value)`, `df.fillna(method='ffill')`, `df.fillna(method='bfill')`
- **Dropping**: `df.dropna()`, `df.dropna(subset=['col'])`, `df.dropna(how='any'/'all')`

### Data Type Conversion
- `df.astype('int64')`, `df.astype('float64')`, `df.astype('string')`
- `pd.to_datetime(df['date_column'])`
- `pd.to_numeric(df['column'], errors='coerce')` - Errors become NaN

### Essential Operations
- **Sorting**: `df.sort_values('column')`, `df.sort_values(['col1', 'col2'])`, `ascending=False`, `df.sort_index()`
- **Unique values**: `series.unique()`, `series.nunique()`, `series.value_counts()`, `series.isin(['A', 'B'])`
- **GroupBy**:
  - `df.groupby('col')['target'].agg(['mean', 'count'])`
  - `df.groupby(['col1', 'col2']).sum(numeric_only=True)`
  - `.transform(func)` - Broadcast aggregated values back
  - `.filter(lambda g: ...)` - Keep groups matching criteria

### Data Loading and Storage
- **CSV**:
  - `pd.read_csv('file.csv')`, `sep=';'`, `header=0`, `index_col=0`
  - `df.to_csv('file.csv')`, `index=False`, `na_rep=''`, `columns=[...]`
- **Excel**:
  - `pd.read_excel('file.xlsx', sheet_name='Sheet1')`
  - `df.to_excel('file.xlsx', sheet_name='Summary', index=False)`
- **JSON**:
  - `pd.read_json('file.json', orient='records')`
  - `df.to_json('file.json', orient='records', indent=2)`

### Data Exploration
- **Summary statistics**: `.describe()`, `.info()`, `.shape`, `.count()`, `.nunique()`, `.memory_usage()`
- **Data quality**: `.isnull().sum()`, `.duplicated()`, `.drop_duplicates()`

---

## Lecture 05: Assignment 5 Specifics

### Assignment-Specific Concepts
- **Shell scripting**:
  - Shebang: `#!/bin/bash`
  - Making scripts executable: `chmod +x script.sh`
  - Directory creation: `mkdir -p path/to/nested`
  - Running Python scripts: `python3 script.py`
  - Output redirection: `> file.txt`
  - `tree` command for directory listings
- **Python config file processing**:
  - Parsing key=value config files
  - String manipulation for parsing
  - Validation logic with if/elif/else
  - File I/O for reading configs and writing outputs
- **Pandas utility functions**:
  - Reusable function design
  - Data loading, cleaning, filtering
  - Missing data handling strategies
  - Type transformations
  - Binning with `pd.cut()`
  - GroupBy aggregation with custom agg_dict
- **Jupyter workflow**:
  - Importing custom modules
  - Executing cells in order
  - Saving outputs to CSV/TXT
  - Documentation in markdown cells
- **Pipeline automation**:
  - Running notebooks with `jupyter nbconvert --execute --to notebook`
  - Error handling with `||` operator
  - Logging execution status

---

## Lecture 06: Data Wrangling - Join, Combine, and Reshape

### Database-Style Joins
- **pd.merge() fundamentals**:
  - `pd.merge(left, right, on='key')`
  - `left_on='key1', right_on='key2'` - Different column names
  - `on=['col1', 'col2']` - Multiple key columns
  - `suffixes=('_left', '_right')` - Handle overlapping names
- **Join types**:
  - `how='inner'` - Intersection (default)
  - `how='left'` - All left rows, matching right
  - `how='right'` - All right rows, matching left
  - `how='outer'` - Union of all rows
- **Relationship types**:
  - Many-to-one: Multiple left rows match one right row
  - Many-to-many: Multiple rows in both (creates Cartesian product)
  - Checking row counts before/after merge

### Concatenating DataFrames
- **Vertical concatenation** (adding rows):
  - `pd.concat([df1, df2, df3])` or `pd.concat([df1, df2], axis=0)`
  - `ignore_index=True` - Reset to sequential index
  - `join='outer'` (default) - Union of columns
  - `join='inner'` - Intersection of columns only
- **Horizontal concatenation** (adding columns):
  - `pd.concat([df1, df2], axis=1)`
  - Index alignment - matching indexes are joined
  - NaN where indexes don't match

### Reshaping Data
- **Wide vs Long format**:
  - Wide: One row per entity, many columns
  - Long: Multiple rows per entity, fewer columns
- **Pivot (Long → Wide)**:
  - `df.pivot(index='row_labels', columns='col_labels', values='data')`
  - Requires unique index/columns combinations
- **Melt (Wide → Long)**:
  - `pd.melt(df, id_vars=['id'], value_vars=['col1', 'col2'])`
  - `var_name='variable'`, `value_name='value'` - Custom column names

### Working with Indexes
- **set_index()**: Move column(s) to index
  - `df.set_index('column')`
  - `df.set_index(['col1', 'col2'])` - MultiIndex
  - `drop=False` - Keep column in DataFrame
  - `inplace=True` - Modify in place
- **reset_index()**: Move index to column(s)
  - `df.reset_index()`
  - `drop=True` - Discard index
  - `inplace=True` - Modify in place

### MultiIndex Basics
- **Creation**: From `groupby()`, `pivot_table()`, `set_index()`, `concat()` with keys
- **Selection**:
  - `df.loc['outer_label']` - All rows with outer level
  - `df.loc[('outer', 'inner')]` - Specific combination
  - `.xs('label', level='level_name')` - Cross-section
- **Flattening**: `df.reset_index()` - Convert back to regular columns

---

## Forbidden Concepts (NOT in Lectures 01-06)

### Visualization and Plotting
- ❌ matplotlib plotting (`plt.plot()`, `plt.scatter()`, etc.)
- ❌ seaborn visualizations
- ❌ pandas plotting methods (`.plot()`, `.plot.bar()`, etc.)
- ❌ plotly or other visualization libraries
- ❌ Any chart/graph creation

### Statistical Analysis
- ❌ Hypothesis testing (t-tests, chi-square, etc.)
- ❌ Statistical modeling (regression, ANOVA, etc.)
- ❌ Correlation analysis beyond basic `.corr()`
- ❌ Statistical inference
- ❌ P-values, confidence intervals

### Machine Learning
- ❌ scikit-learn or any ML libraries
- ❌ Model training, prediction, evaluation
- ❌ Feature selection algorithms
- ❌ Cross-validation
- ❌ Any supervised/unsupervised learning

### Advanced Pandas
- ❌ `pivot_table()` with aggregation functions (basic `pivot()` is OK)
- ❌ `crosstab()`
- ❌ Window functions (`.rolling()`, `.expanding()`, `.ewm()`)
- ❌ Time series resampling (`.resample()`)
- ❌ Advanced string methods (`.str.extract()`, regex patterns)
- ❌ Categorical data type operations beyond basic `.astype('category')`
- ❌ Complex MultiIndex operations beyond reset_index()

### Advanced NumPy
- ❌ Broadcasting rules beyond basic scalar operations
- ❌ Linear algebra functions (`np.dot()`, `np.linalg.*`)
- ❌ FFT and signal processing
- ❌ Advanced indexing with `np.ix_`

### Databases and SQL
- ❌ SQL queries (while mentioned, not taught yet)
- ❌ Database connections
- ❌ `pd.read_sql()`, `pd.to_sql()`

### Web and APIs
- ❌ Web scraping
- ❌ API requests
- ❌ `pd.read_html()`

### Other Advanced Topics
- ❌ Regular expressions (regex) beyond basic string methods
- ❌ Object-oriented programming (classes, inheritance)
- ❌ Decorators, generators, context managers (advanced Python)
- ❌ Parallel processing, multiprocessing
- ❌ Performance profiling

---

## Summary: Core Allowed Toolbox for Assignments

### Data Loading
✅ `pd.read_csv()`, `pd.read_excel()`, `pd.read_json()`

### Data Inspection
✅ `.head()`, `.tail()`, `.info()`, `.describe()`, `.shape`, `.dtypes`, `.isnull().sum()`

### Data Selection
✅ `.loc[]`, `.iloc[]`, boolean indexing, column selection

### Data Cleaning
✅ `.fillna()`, `.dropna()`, `.drop_duplicates()`, `.astype()`, `pd.to_numeric()`, `pd.to_datetime()`

### Data Transformation
✅ Adding columns, `pd.cut()` for binning, string methods (`.strip()`, `.lower()`, etc.)

### Data Aggregation
✅ `.groupby()`, `.agg()`, `.value_counts()`, `.unique()`, `.nunique()`

### Data Combination
✅ `pd.merge()`, `pd.concat()`, `.pivot()`, `pd.melt()`

### Index Management
✅ `.set_index()`, `.reset_index()`

### Data Output
✅ `.to_csv()`, `.to_excel()`, `.to_json()`

### NumPy
✅ Array creation, indexing, slicing, boolean masking, basic operations, statistical functions

### Python Fundamentals
✅ Lists, dictionaries, sets, functions, control structures, file I/O, string manipulation

### Command Line
✅ Navigation, file operations, pipes, basic text processing

---

**Use this inventory to validate assignment solutions. If a concept is not listed in the "Allowed" sections above, it should NOT appear in assignments for lectures 01-06.**
