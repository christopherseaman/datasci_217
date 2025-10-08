Pandas on Jupyter: Data Structures & I/O

See [BONUS.md](BONUS.md) for advanced topics:

- Alignment and broadcasting tricks
- Function application patterns and method chaining notes
- Ranking strategies and working with duplicate index labels
- Extended I/O and performance tips (Excel, JSON, chunked reads)

# Jupyter Notebooks: Interactive Data Analysis

In Lectures 1-3, you wrote Python scripts (`.py` files) that run top-to-bottom. Jupyter notebooks (`.ipynb` files) let you run code in any order, see results immediately, and mix code with documentation - perfect for data exploration and analysis. Think of `.py` files for production code and automation, and `.ipynb` files for interactive analysis and storytelling with data.

Jupyter notebooks provide an interactive environment for data analysis, combining code execution with rich output display. They're essential for exploratory data analysis, prototyping, and sharing results with stakeholders.

<!-- FIXME: Add screenshot of Jupyter interface in VS Code showing:
     - Cell types (code/markdown dropdown)
     - Run button and keyboard shortcuts
     - Kernel selector (Python 3.x)
     - Variable explorer panel
     File: media/jupyter_interface.png -->

## Jupyter Notebook Interface

Jupyter notebooks organize work into cells that can contain code or markdown. This structure enables iterative analysis and clear documentation of the analytical process.

<!-- FIXME: xkcd 1906 "Making Progress"
     About learning curves and feeling like you're not making progress
     Great for students learning Jupyter - normalizes the learning process
     https://xkcd.com/1906/
     File: media/xkcd_1906.png -->

**Reference:**

- **Code cells**: Execute Python code and display output
- **Markdown cells**: Write documentation and explanations
- **Cell execution**: `Shift+Enter` (run and advance), `Ctrl+Enter` (run and stay)
- **Cell management**: `A` (add above), `B` (add below), `DD` (delete cell)
- **Magic commands**: `%matplotlib inline` (display plots), `%timeit` (time execution)
- **Kernel**: Python interpreter that executes code cells

**Example:**

```python
# Cell 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Cell 2: Load data
df = pd.read_csv('data.csv')
display(f"Data shape: {df.shape}")

# Cell 3: Quick visualization
df.head().plot(kind='bar')
plt.title('Sample Data')
plt.show()
```

## `display()` vs `print()`

*Think of `print()` as the reliable Honda Civic - works everywhere, gets the job done, but nothing fancy. `display()` is the sports car - looks amazing, handles beautifully, but only in the right environment (Jupyter).*

When working with DataFrames and Series in Jupyter, you have two options for viewing output. `print()` works in any Python environment (scripts, notebooks, REPL) and shows plain text. `display()` is Jupyter-specific and renders rich HTML tables with formatting, making data much easier to read.

**Reference:**

- `print(df)` - Plain text output, works everywhere (scripts and notebooks)
- `display(df)` - Rich HTML table output, **Jupyter notebooks only**
- Use `display()` for DataFrames/Series in notebooks for better readability
- Use `print()` for simple values, strings, or when writing `.py` scripts
- `display()` will fail in regular Python scripts (`.py` files run from terminal)

**Example:**

```python
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
})

# Plain text - works everywhere, but harder to read
print(df)
#      Name  Age
# 0   Alice   25
# 1     Bob   30
# 2 Charlie   35

# Rich formatting - Jupyter only, beautiful tables
display(df)
# [Renders as a formatted HTML table with borders, alternating row colors, etc.]

# For simple values, print() is fine
print(f"Total rows: {len(df)}")  # Total rows: 3
```

**Pro tip:** In Jupyter, if a DataFrame is the last line in a cell, it automatically displays without calling `display()` - but being explicit is clearer!

## Jupyter Magic Commands

Magic commands are like cheat codes for Jupyter - they give you special powers that normal Python doesn't have. Think of them as the "konami code" of data science, except instead of getting 30 extra lives, you get inline plots and package installation!

Magic commands provide special functionality for notebook environments. They start with `%` and extend Jupyter's capabilities for data analysis.

**Reference:**

- `%matplotlib inline` - Display plots within notebook cells
- `%pwd` - Print current working directory
- `%ls` - List directory contents
- `%pip install package_name` - Install Python packages
- `%pip list` - List installed packages
- `%pip show package_name` - Show package information

**Example:**

```python
# Display plots inline
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.show()

# Install packages
%pip install pandas numpy matplotlib

# Check your location
%pwd
%ls
```

## Jupyter Notebooks in VS Code

VS Code provides excellent Jupyter notebook support with integrated terminal, git management, and debugging capabilities. Understanding how to work with notebooks in VS Code is essential for modern data science workflows.

**Reference:**

- **Create notebook**: `Ctrl+Shift+P` → "Jupyter: Create New Jupyter Notebook"
- **Open notebook**: `Ctrl+O` → Select `.ipynb` file
- **Run cell**: `Shift+Enter` (run and advance), `Ctrl+Enter` (run and stay)
- **Add cell**: Click `+` button above cells, or right-click → "Insert Cell Above/Below"
- **Delete cell**: Right-click cell → "Delete Cell", or select cell and press `Delete` key
- **Save**: `Ctrl+S` (auto-saves frequently)

**Note:** Keyboard shortcuts like `A` and `DD` only work in command mode (when cell is selected but not editing). For beginners, using the GUI buttons and right-click menu is more reliable.

**Example:**

```python
# VS Code automatically detects .ipynb files
# Just open any .ipynb file and start coding
import pandas as pd
df = pd.read_csv('data.csv')
display(df.head())
```

## Kernel Management Basics

The kernel is the Python interpreter running your code. Sometimes it gets stuck or needs a fresh start.

**Reference:**

- **Restart Kernel**: Clears all variables from memory, fresh start
- **Run All**: Executes all cells from top to bottom
- **Restart & Run All**: Combines both - useful for testing if code works from scratch
- Common issues: Variable conflicts, memory issues, stuck computations

**Example:**

When to restart your kernel:

- Code behaves unexpectedly
- Variables seem to have wrong values
- "It worked before but now it doesn't"
- Before submitting assignments (test it runs from top to bottom!)

## Notebook Outputs and Git: The Memory Problem

**Warning:** Jupyter notebooks are like that one friend who screenshots everything you text them. They save both your code AND all the outputs (results, data, plots) in the same file.

This means if you accidentally print your password, patient data, or that embarrassing test result, it's now permanently saved in your notebook file. It's like having a photographic memory of your most awkward moments.

**Before committing to git (the "digital hygiene" moment):**

1. **Clear all outputs** - Click the "Clear All Outputs" button in VS Code
2. **Check for sensitive data** - Make sure no personal information, passwords, or confidential data is visible
3. **Save the notebook** - The outputs will be removed from the file

**Example:**

```python
# This output contains sensitive data and will be saved in the notebook
df = pd.read_csv('patient_data.csv')
display(df.head())  # Shows patient names, IDs, medical info
# Oops! Now everyone can see John Doe's blood pressure on GitHub
```

**After running this code, the patient data will be visible in your notebook file. Always clear outputs before sharing or committing to git.**

# LIVE DEMO!

(Demo 1: Jupyter Basics - interface, cells, magic commands)

# Introduction to Pandas

Pandas provides powerful data structures and tools for working with structured data. It's built on NumPy but adds labeled axes and missing data handling, making it essential for data analysis workflows.

<!-- FIXME: Add diagram showing both Series and DataFrame:
     - Series: 1D labeled array (index + values)
     - DataFrame: 2D table (row index + column names + data)
     - Visual comparison showing the relationship
     File: media/pandas_structures.png -->

<!-- FIXME: Alternative/Additional - xkcd 2180 "Spreadsheets"
     Shows why we need better tools than Excel for data
     Perfect for introducing DataFrames as "spreadsheets done right"
     https://xkcd.com/2180/
     File: media/xkcd_2180.png -->

*Fun fact: Pandas got its name from "Panel Data" - the economics term for time-series data. The cute bear logo? That's just a happy accident that makes data science more approachable! 🐼*

## Pandas Data Structures

*Think of pandas data structures like Russian nesting dolls - Series fit inside DataFrames, which can contain other DataFrames, which can contain... well, you get the idea. It's data structures all the way down!*

<!-- FIXME: Add diagram showing Series vs DataFrame relationships (media/pandas_structures.png) -->

**Reference:**

- `pd.Series(data, index=None, name=None)` — create a labeled vector
- `pd.DataFrame(data, index=None, columns=None)` — create a table with labeled axes
- `.index`, `.columns`, `.shape`, `.dtypes` — inspect structure metadata
- `.info()`, `.describe()` — quick structure and summary diagnostics

### Series

A Series is a one-dimensional labeled array that can hold any data type. It's like a column in a spreadsheet with an index that labels each value, enabling powerful data manipulation and analysis.

**Reference:**

- `pd.Series(data, index=None, name=None)` - Create Series
- `series.index` - Access index labels
- `series.values` - Get values as NumPy array
- `series.name` - Get/set Series name
- `series.dtype` - Get data type
- `series.size` - Number of elements
- `series.head(n=5)` - First n elements
- `series.tail(n=5)` - Last n elements
- `series.describe()` - Summary statistics
- `series.value_counts()` - Value frequencies

**Example:**

```python
# Create Series
ages = pd.Series([25, 30, 35, 40], index=['Alice', 'Bob', 'Charlie', 'Diana'])
display(ages)  # Alice: 25, Bob: 30, Charlie: 35, Diana: 40
display(ages.index)  # ['Alice', 'Bob', 'Charlie', 'Diana']
display(ages.values)  # [25 30 35 40]

# Series operations
display(ages.mean())  # 32.5
display(ages.describe())  # count, mean, std, min, 25%, 50%, 75%, max
```

### DataFrame

A DataFrame is a two-dimensional labeled data structure with columns of potentially different types. DataFrames combine multiple Series so you can operate column-wise or row-wise with shared labels. Because columns can hold different dtypes, keep an eye on schema when merging disparate sources.

*Pro tip: DataFrames are like Excel spreadsheets, but with superpowers. They can handle millions of rows without breaking a sweat, and they never ask you to "save as" or complain about circular references.*

**Reference:**

- `pd.DataFrame(data, index=None, columns=None)` - Create DataFrame
- `df.index` - Access row index
- `df.columns` - Access column names
- `df.values` - Get values as NumPy array
- `df.shape` - (rows, columns) tuple
- `df.dtypes` - Data types per column
- `df.info()` - Detailed information
- `df.describe()` - Summary statistics
- `df.head(n=5)` - First n rows
- `df.tail(n=5)` - Last n rows
- `df.sample(n=5)` - Random n rows

**Example:**

```python
# Create DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
})

display(df.shape)  # (3, 3)
display(df.dtypes)  # Name: object, Age: int64, Salary: int64
display(df.describe())  # Summary statistics for numeric columns
```

### Selecting Columns from a DataFrame

Thankfully, we don't have to use the whole DataFrame at all times. We can select subsets of columns to work with instead.

*Think of column selection like picking your team for dodgeball - sometimes you want just your star player (single column), sometimes you want your entire A-team (multiple columns), and sometimes you want everyone except that one person who always gets you out (column exclusion).*

**Reference:**

- `df['column_name']` - Select single column (returns Series)
- `df[['col1', 'col2']]` - Select multiple columns (returns DataFrame)
- `df.column_name` - Dot notation for single column (if name has no spaces/special chars)
- `df.select_dtypes(include=['number'])` - Select by data type

**Example:**

```python
# Create sample DataFrame
employees = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000],
    'Department': ['Engineering', 'Sales', 'Engineering', 'Marketing'],
    'Years_Experience': [2, 5, 8, 3]
})

# Single column selection (returns Series)
names = employees['Name']
display(type(names))  # <class 'pandas.core.series.Series'>
display(names)

# Single column selection (returns DataFrame)
names = employees[['Name']]
display(type(names))  # <class 'pandas.core.frame.DataFrame'>
display(names)

# Multiple column selection (returns DataFrame)
basic_info = employees[['Name', 'Age']]
display(type(basic_info))  # <class 'pandas.core.frame.DataFrame'>
display(basic_info)

# Dot notation (careful with column names!)
ages = employees.Age  # Works if column name is valid Python identifier
display(ages)

# Select numeric columns only
numeric_data = employees.select_dtypes(include=['number'])
display(numeric_data.columns)  # ['Age', 'Salary', 'Years_Experience']
```

/callout("This is confusing!")
**Tips:**

- Use `df['column']` for single columns when you want a Series
- Use `df[['column']]` for single columns when you want a DataFrame
- Bracket notation `df['column']` is safer than dot notation `df.column`
- Multiple column selection always returns a DataFrame, even if you select just one column

## Data Selection and Indexing

Data selection in pandas uses label-based and position-based indexing. Understanding these methods is crucial for data manipulation and analysis.

<!-- FIXME: Add visual comparing .loc vs .iloc:
     - Show DataFrame with labeled index (e.g., ['A', 'B', 'C']) AND numeric positions [0, 1, 2]
     - Side-by-side examples:
       * df.loc['B', 'Name'] → selects by label
       * df.iloc[1, 0] → selects by position (same cell)
     - Highlight: loc uses labels, iloc uses positions
     - Include slicing behavior difference (loc is inclusive, iloc is exclusive)
     File: media/loc_vs_iloc.png -->

*Warning: Indexing in pandas is like a choose-your-own-adventure book - there are multiple ways to get to the same destination, and sometimes you end up in a completely different story than you intended.*

### .loc vs .iloc

The key difference: **`.loc` uses LABELS, `.iloc` uses POSITIONS** (like list indexing).

*Think of it this way: `.loc` is like asking "Give me the data for employee 'Alice'" (using names/labels), while `.iloc` is like saying "Give me the data from the 2nd row" (using positions like 0, 1, 2...).*

**Critical Differences:**

1. **What they use:**
   - `.loc[row_label, col_label]` — uses index labels and column names
   - `.iloc[row_position, col_position]` — uses integer positions (0, 1, 2...)

2. **Slicing behavior:**
   - `.loc[1:3]` — includes BOTH endpoints (1, 2, AND 3)
   - `.iloc[1:3]` — excludes the end (1, 2, but NOT 3)

3. **When to use:**
   - Use `.loc` when you know the names/labels
   - Use `.iloc` when you know the positions

**Reference:**

- `df.loc[row_labels, column_labels]` — label-based selection
- `df.iloc[row_positions, column_positions]` — position-based selection
- `df.query("expression")` — filter with readable expressions
- `df[df['column'] > value]` — boolean masking
- `df.isin(sequence)` / `df.between(left, right)` — membership utilities

**Example:**

```python
# Create DataFrame with custom index to show the difference clearly
employees = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000]
}, index=['emp001', 'emp002', 'emp003', 'emp004'])

display(employees)
#          Name  Age  Salary
# emp001  Alice   25   50000
# emp002    Bob   30   60000
# emp003 Charlie  35   70000
# emp004  Diana   28   55000

# .loc uses LABELS (index names and column names)
display(employees.loc['emp002', 'Name'])        # 'Bob' - using row label 'emp002'
display(employees.loc['emp001':'emp003', 'Age']) # Ages for emp001, emp002, AND emp003 (inclusive!)

# .iloc uses POSITIONS (like list indexing: 0, 1, 2, 3...)
display(employees.iloc[1, 0])      # 'Bob' - position 1 (2nd row), position 0 (1st column)
display(employees.iloc[0:3, 1])    # Ages for positions 0, 1, 2 (NOT including position 3!)

# Common mistake: mixing them up!
# employees.loc[1, 'Name']    # ERROR! No row with label '1' 
# employees.iloc['emp002', 0] # ERROR! Can't use string labels with iloc

# Boolean indexing (works with either)
adults = employees[employees['Age'] >= 30]  # Bob, Charlie, Diana
high_earners = employees.loc[employees['Salary'] > 60000]  # Charlie
```

**Memory Trick:**

- **`.loc`** = **"L"abels** (names, strings, custom indices)
- **`.iloc`** = **"i"nteger** **"L"ocations** (0, 1, 2, 3... like list positions)

### Adding Columns to DataFrames

Derived columns capture new features and align automatically with existing indexes. Choose between direct assignment for quick mutations and `.assign()` when you need a non-mutating pipeline step.

**Reference:**

- `df['column_name'] = expression` — insert or overwrite a column
- `df.assign(name=lambda d: ...)` — add columns while returning a new DataFrame
- `df.insert(loc, column, value)` — control column ordering
- `df.eval("new = ...")` — expression syntax for simple arithmetic

**Example:**

```python
salaries = pd.DataFrame({
    'Name': ['Avery', 'Bianca', 'Cheng'],
    'Salary': [120000, 95000, 88000],
    'Department': ['Engineering', 'Sales', 'People Ops']
})

salaries['HourlyRate'] = salaries['Salary'] / 2080
augmented = salaries.assign(
    Bonus=lambda d: d['Salary'] * 0.05,
    TotalComp=lambda d: d['Salary'] + d['Salary'] * 0.05
)

display(augmented[['Name', 'HourlyRate', 'TotalComp']])
```

### Handling Missing Data

Missing data decisions begin when files are read and continue throughout transformations. Detect gaps, decide whether to fill or drop, and capture read-time options so messy inputs become reproducible.

**Reference:**

- `series.isnull()`, `series.notnull()` — null diagnostics
- `df.fillna(value | method='ffill' | method='bfill')` — replacement strategies
- `df.dropna(subset=..., how='any' | 'all')` — remove incomplete rows
- `df.isnull().sum()` — column-level null counts

**Example:**

```python
survey = pd.read_csv(
    'employee_survey.csv',
    na_values=['NA', 'NULL', '?'],
    keep_default_na=True,
    skiprows=[1],
    usecols=['name', 'role', 'bonus', 'start_date']
)

survey['bonus'] = survey['bonus'].fillna(0)
clean = survey.dropna(subset=['start_date'])
display(survey.isnull().sum())
```

## Data Type Conversion

Converting data types is like trying to convince your data to identify as something else. Sometimes it works smoothly (string "42" → int 42), sometimes you need therapy (error handling), and sometimes it just refuses and throws a ValueError. Just remember: you can't force a square peg into a round hole, but pandas will try its best!

Converting data to the correct types is essential for proper analysis. This includes converting strings to numbers, dates, and other appropriate types.

**Reference:**

- `df.astype('int64')` - Convert to integer
- `df.astype('float64')` - Convert to float
- `df.astype('string')` - Convert to string
- `pd.to_datetime(df['date_column'])` - Convert to datetime
- `pd.to_numeric(df['column'], errors='coerce')` - Convert to numeric, errors become NaN

**Example:**

```python
# Convert data types
df = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4.5, 5.5, 6.5]})
df['A'] = df['A'].astype('int64')  # Convert string to integer
df['B'] = df['B'].astype('int64')  # Convert float to integer
display(df.dtypes)  # A: int64, B: int64

# Handle conversion errors
df['C'] = ['1', '2', 'invalid', '4']
df['C'] = pd.to_numeric(df['C'], errors='coerce')  # Invalid becomes NaN
display(df['C'])  # [1.0, 2.0, NaN, 4.0]
```

# LIVE DEMO!

(Demo 2: Pandas DataFrames - selection, filtering, groupby, operations)

# Essential Pandas Operations

## Sorting Data

Sorting organizes your data by values or index, making it easier to find patterns and outliers. This is one of the most common operations in data analysis.

**Reference:**

- `df.sort_values('column')` - Sort by column values
- `df.sort_values(['col1', 'col2'])` - Sort by multiple columns
- `ascending=False` - Sort in descending order
- `df.sort_index()` - Sort by index

**Example:**

```python
# Sort by age
df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 20]})
df.sort_values('Age')  # Sort by age (youngest first)
df.sort_values('Age', ascending=False)  # Sort by age (oldest first)
```

## Finding Unique Values

Exploring unique values helps you understand your data and identify categories. This is essential for data exploration and cleaning.

**Reference:**

- `series.unique()` - Get unique values
- `series.nunique()` - Count unique values
- `series.value_counts()` - Count how often each value appears
- `series.isin(['A', 'B'])` - Check if values are in a list

**Example:**

```python
# Find unique values
categories = pd.Series(['A', 'B', 'A', 'C', 'B'])
display(categories.unique())  # ['A' 'B' 'C']
display(categories.value_counts())  # A: 2, B: 2, C: 1

# Filter by membership
display(categories.isin(['A', 'B']))  # [True, True, True, False, True]
```

## GroupBy

GroupBy enables split-apply-combine analytics: split data into groups, apply aggregations or filters, then combine results into aligned outputs. It replaces manual loops with expressive transformations.

**Reference:**

- `df.groupby('col')[target].agg(['mean', 'count'])` — summarize groups
- `df.groupby(['col1', 'col2']).sum(numeric_only=True)` — multi-key aggregation
- `grouped.filter(lambda g: ...)` — keep groups matching criteria
- `grouped.transform(func)` — broadcast aggregated values back to rows

**Example:**

```python
comp = pd.DataFrame({
    'Department': ['Eng', 'Eng', 'Sales', 'Sales', 'People Ops'],
    'Salary': [120000, 115000, 95000, 98000, 88000]
})

dept_summary = comp.groupby('Department')['Salary'].agg(['count', 'mean'])
display(dept_summary)
```

# Data Loading and Storage

*Pro tip: If you're ever stuck with a weird file format, remember: "There's a pandas function for that!" (Usually `pd.read_[format]()` - pandas is surprisingly comprehensive at reading data from just about anywhere)*

## Reading and Writing CSV Files

CSV files are the most common format for data analysis. Pandas makes it easy to read CSV files with sensible defaults.

*Fun fact: CSV stands for "Comma-Separated Values," but in reality, it's more like "Comma-Separated Values (unless someone used semicolons, or tabs, or pipes, or any other delimiter they felt like using that day)."*

**Reference:**

Reading

- `pd.read_csv('filename.csv')` - Read CSV file
- `pd.read_csv('filename.csv', sep=';')` - Custom separator
- `pd.read_csv('filename.csv', header=0)` - Specify header row
- `pd.read_csv('filename.csv', index_col=0)` - Use first column as index
- `pd.read_csv(path, sep=',', header=0, index_col=None)` — all options

Writing

- `df.to_csv('asdf.csv') - no frills
- `df.to_csv('tab_separated.tsv', sep='\t')
- `df.to_csv(path, index=False, na_rep='')` — write cleaned results
- `df.to_csv(path, columns=[...])` — export selected columns

**Example:**

```python
# Basic CSV reading
df = pd.read_csv('data.csv')
display(df.head())

# Custom options
df = pd.read_csv('data.csv', sep=';', index_col=0)
display(df.head())
```

## Reading and Writing Other Formats

**Reference:**

Excel

- `pd.read_excel(path, sheet_name=0, usecols=None)` — ingest worksheets
- `df.to_excel(path, sheet_name='Summary', index=False)` — share spreadsheets

JSON

- `pd.read_json(path_or_buf, orient='records')` — parse structured payloads
- `df.to_json(path_or_buf, orient='records', indent=2)` — export API-friendly data

**Example:**

```python
#Excel
sales = pd.read_excel('quarterly_sales.xlsx', sheet_name='Q2')
sales.to_excel('quarterly_sales_clean.xlsx', sheet_name='Q2', index=False)

# JSON
payload = pd.read_json('inventory_payload.json')
payload.to_json('inventory_payload_export.json', orient='records', indent=2)
```

**Note:** Database access and sql will be covered later course content.

# Data Exploration and Summary Statistics

## Summary Statistics

Summary statistics provide a quick overview of your data's distribution and characteristics. They're essential for understanding data quality and identifying patterns.

*Remember: Correlation does not imply causation! (But it's still useful for understanding patterns in your data)*

**Reference:**

- `df.describe()` - Summary statistics for numeric columns
- `df.info()` - Data types and memory usage
- `df.shape` - (rows, columns) tuple
- `df.count()` - Non-null values per column
- `df.nunique()` - Unique values per column
- `df.memory_usage()` - Memory usage per column
- `df.isnull().sum()` - Missing values per column

**Example:**

```python
# Summary statistics
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})
display(df.describe())  # count, mean, std, min, 25%, 50%, 75%, max
display(df.info())  # Data types and memory usage
display(df.isnull().sum())  # Missing values per column
```

## Data Quality Assessment

Data quality assessment identifies issues like missing values, duplicates, and outliers. This is crucial for ensuring reliable analysis results.

**Reference:**

- `df.isnull()` - Boolean DataFrame: True for missing values
- `df.notnull()` - Boolean DataFrame: True for non-missing values
- `df.duplicated()` - Boolean Series: True for duplicate rows
- `df.drop_duplicates()` - Remove duplicate rows
- `df.nunique()` - Count unique values per column
- `df.value_counts()` - Value frequencies
- `df.describe()` - Summary statistics

**Example:**

```python
# Check for missing values
display(df.isnull().sum())  # Missing values per column

# Check for duplicates
display(df.duplicated().sum())  # Number of duplicate rows
df_clean = df.drop_duplicates()  # Remove duplicates

# Check data types
display(df.dtypes)  # Data types per column
display(df.info())  # Detailed information
```

# LIVE DEMO!

(Demo 3: Data I/O - CSV, Excel, JSON, real-world cleaning workflow)
