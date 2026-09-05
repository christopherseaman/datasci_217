---
notion:
  role: lecture
  status: mapped
  page_id: "281d9fdd-1a1a-800a-897d-cafb5971c23f"
  url: "https://app.notion.com/p/281d9fdd1a1a800a897dcafb5971c23f"
---

Pandas on Jupyter: Data Structures & I/O

See [BONUS.md](BONUS.md) for advanced topics:

- Alignment and broadcasting tricks
- Function application patterns and method chaining notes
- Ranking strategies and working with duplicate index labels
- Extended I/O and performance tips (Excel, JSON, chunked reads)
- Safe non-interactive notebook execution and failure handling

# Jupyter Notebooks: Interactive Data Analysis

In Lectures 1-3, Python scripts (`.py`) ran top-to-bottom. Jupyter notebooks
(`.ipynb`) arrange code and Markdown in interactive cells, making them useful
for exploration and explanation; scripts remain the better fit for automation.

The lecture examples use pandas 3.x APIs. Executable demos and assignments record their exact tested package pins in each activity's requirements.

## Jupyter Notebook Interface

This conceptual map names the parts you will use in a Jupyter notebook. The
exact buttons vary slightly between JupyterLab, VS Code, and other clients.

| Notebook part | Purpose |
| --- | --- |
| Code / Markdown cell | Run Python, or explain the analysis |
| Run controls | Execute the selected cell and choose whether to advance |
| Kernel selector | Choose the Python interpreter that runs the cells |
| Output area | Inspect the value, table, or error produced by a cell |
| Variable explorer | Review names and values currently held by the kernel |

![xkcd 1906, “Making Progress”: after hours of work, the same problems are now in a spreadsheet.](media/xkcd_1906.png)

*[Making Progress](https://xkcd.com/1906/) by xkcd — progress, now with columns.*

**Reference:**

- **Code cells**: Execute Python code and display output
- **Markdown cells**: Write documentation and explanations
- **Cell execution**: `Shift+Enter` (run and advance), `Ctrl+Enter` (run and stay)
- **Cell management**: `A` (add above), `B` (add below), `DD` (delete cell)
- **Magic commands**: `%pwd`, `%ls`, `%timeit`, and `%pip` (notebook utilities)
- **Kernel**: Python interpreter that executes code cells

**Example:**

```python
# Cell 1: Core Python values
name = "Ada"
scores = [8, 9, 10]

# Cell 2: Run a calculation
average = sum(scores) / len(scores)
print(f"{name}'s average: {average:.1f}")

# Cell 3: Markdown can explain the result
```

## Jupyter Magic Commands

Magic commands are like cheat codes for Jupyter - they give you special powers that normal Python doesn't have. Think of them as the "konami code" of data science, except instead of getting 30 extra lives, you get inline plots and package installation!

Magic commands provide special functionality for notebook environments. They start with `%` and extend Jupyter's capabilities for data analysis.

**Reference:**

- `%pwd` - Print current working directory
- `%ls` - List directory contents
- `%timeit expression` - Time a Python expression
- `%pip install -r requirements.txt` - Install an activity's recorded packages into the notebook kernel environment
- `%pip list` - List installed packages
- `%pip show package_name` - Show package information

**Example:**

```python
# These examples use only notebook mechanics and core Python.
%pwd
%ls
%timeit sum(range(100))

# Install the requirements recorded for the current activity
%pip install -r requirements.txt
```

Plotting is deferred until Lecture 07, after the plotting libraries and workflow have been introduced.

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
message = "Notebook cells can be rerun independently."
print(message)
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

`Restart & Run All` is the interactive reproducibility check. For optional batch
execution that preserves the source notebook and stops on failed cells, see
[Running notebooks non-interactively](BONUS.md#running-notebooks-non-interactively).

## Notebook Outputs and Git: The Memory Problem

**Warning:** Jupyter notebooks are like that one friend who screenshots everything you text them. They save both your code AND all the outputs (results, data, plots) in the same file.

Accidentally printed passwords, patient data, or embarrassing test results are saved in the notebook too—like having a photographic memory of your most awkward moments.

**Before committing to git (the "digital hygiene" moment):**

1. **Clear all outputs** - Click the "Clear All Outputs" button in VS Code
2. **Check for sensitive data** - Make sure no personal information, passwords, or confidential data is visible
3. **Save the notebook** - The outputs will be removed from the file

**Example:**

```python
# This output contains sensitive data and will be saved in the notebook
patient_name = "Example Patient"
blood_pressure = "120/80"
print(patient_name, blood_pressure)
# Clear the output before sharing or committing the notebook.
```

# LIVE DEMO!

(Demo 1: Jupyter Basics - interface, cells, magic commands)

# Introduction to Pandas

Pandas builds labeled Series and DataFrames on NumPy and adds tabular I/O and
missing-data tools.

![xkcd 2180, “Spreadsheets”: a joke about spreadsheet formulas becoming
 unexpectedly elaborate.](media/xkcd_2180.png)

*[Spreadsheets](https://xkcd.com/2180/) by xkcd — a reminder that a
DataFrame is useful when the spreadsheet is becoming a program.*

*Fun fact: Pandas got its name from "Panel Data" - the economics term for time-series data. The cute bear logo? That's just a happy accident that makes data science more approachable! 🐼*

Pandas is conventionally imported with the short alias `pd`, which the examples below use:

```python
import pandas as pd
```

## Pandas Data Structures

A Series is one labeled dimension; a DataFrame combines labeled columns under a shared row index. That shared index is what makes selection and alignment more than simple list positioning.

*Think of Series inside DataFrames like Russian nesting dolls: one labeled
column fits inside the larger labeled table.*

| Structure | Shape | Labels | Example |
| --- | --- | --- | --- |
| `Series` | 1D | One index + values | `age['Ada'] → 36` |
| `DataFrame` | 2D | Row index + column names | `people.loc['Ada', 'age'] → 36` |

One DataFrame column is a Series; several aligned Series form a DataFrame.

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
display(df.dtypes)  # Name: str, Age: int64, Salary: int64
display(df.describe())  # Summary statistics for numeric columns
```

## `display()` vs `print()`

Now that Series and DataFrames are defined, we can compare notebook output choices. `print()` works in scripts and notebooks and shows plain text. In a Jupyter notebook, `display()` renders a Series or DataFrame as rich HTML, which is usually easier to scan. Use `print()` for simple values or code that should also run as a `.py` script; use `display()` when the notebook presentation matters. A DataFrame or Series written as the last expression in a cell is also displayed automatically.

*Think of `print()` as the reliable Honda Civic—works almost anywhere—while
`display()` is the sports car: prettier, but happiest in Jupyter.*

**Example:**

```python
df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
print(df)       # Plain text, works everywhere
display(df)     # Rich table output in Jupyter
print(len(df))  # A simple value: 2
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

> **This is confusing!**

**Tips:**

- Use `df['column']` for single columns when you want a Series
- Use `df[['column']]` for single columns when you want a DataFrame
- Bracket notation `df['column']` is safer than dot notation `df.column`
- Multiple column selection always returns a DataFrame, even if you select just one column

## Data Selection and Indexing

Pandas supports both label-based and position-based selection.

*Warning: Indexing in pandas is like a choose-your-own-adventure book—there are multiple ways to get to the same destination, and sometimes you end up in a completely different story than you intended.*

### .loc vs .iloc

| Selector | Uses | Same cell | Slice ending |
| --- | --- | --- | --- |
| `.loc` | Row and column labels | `employees.loc['emp002', 'Name']` → `'Bob'` | `employees.loc['emp001':'emp003']` includes `emp003` |
| `.iloc` | Integer positions | `employees.iloc[1, 0]` → `'Bob'` | `employees.iloc[0:3]` stops before position `3` |

*Think of it this way: `.loc` is like asking "Give me the data for employee 'Alice'" (using names/labels), while `.iloc` is like saying "Give me the data from the 2nd row" (using positions like 0, 1, 2...).*

**Reference:**

- `df.loc[row_labels, column_labels]` — label-based selection
- `df.iloc[row_positions, column_positions]` — position-based selection
- `df.query("expression")` — filter with readable expressions
- `df[df['column'] > value]` — boolean masking
- `df.isin(sequence)` / `df['column'].between(left, right)` — membership and range tests

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

# A Boolean Series carries index labels, so .loc aligns it by label
age_mask = employees['Age'] >= 30
adults = employees.loc[age_mask]  # Bob and Charlie

# .iloc is positional and does not accept an indexed Boolean Series.
# Convert deliberately to a positional Boolean array when position is intended.
adults_by_position = employees.iloc[age_mask.to_numpy()]
high_earners = employees.loc[employees['Salary'] > 60000]  # Charlie
```

**Memory Trick:**

- **`.loc`** = **"L"abels** (names, strings, custom indices)
- **`.iloc`** = **"i"nteger** **"L"ocations** (0, 1, 2, 3... like list positions)

### Adding Columns to DataFrames

Derived columns capture new features and align automatically with existing indexes. Mutate the owning DataFrame directly when that is the intent; use `.assign()` and bind its returned DataFrame when you want a new result.

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

### Label Alignment and Safe Assignment

Pandas aligns Series and DataFrame operations by index label, not merely by row position. Unmatched labels can produce missing values. Use `reindex()` when you need to make the target labels, order, and missing-label policy explicit.

```python
scores = pd.DataFrame(
    {'score': [80, 90, 70]},
    index=['student_a', 'student_b', 'student_c']
)
bonus = pd.Series({'student_c': 5, 'student_a': 2})

# Series assignment aligns labels; reindex also supplies the missing student_b value.
scores['bonus'] = bonus.reindex(scores.index, fill_value=0)
scores['adjusted_score'] = scores['score'] + scores['bonus']

# Assign through .loc in one operation so the original DataFrame is updated.
scores['status'] = 'ok'
scores.loc[scores['score'] < 75, 'status'] = 'review'
```

With Copy-on-Write, a subset behaves independently: mutating it does not mutate `scores`. Therefore, chained assignment such as `scores[scores['score'] < 75]['status'] = 'review'` never updates the original DataFrame. Update the owner in one statement with `.loc[row_mask, column] = value` (or `.iloc[...] = value` for positional assignment), as above; for a separate result, transform the subset and assign the returned object to a name.

For the version-specific details behind these examples, see the official
[pandas 3.0 release notes](https://pandas.pydata.org/pandas-docs/version/3.0/whatsnew/v3.0.0.html),
[string-dtype migration guide](https://pandas.pydata.org/docs/user_guide/migration-3-strings.html),
and [Copy-on-Write guide](https://pandas.pydata.org/docs/user_guide/copy_on_write.html).

### Detecting Missing Data at Read Time

Missing-data work begins by telling pandas which source tokens represent missing values and measuring the gaps. This lecture focuses on detection and reproducible read-time handling; [Lecture 05](../05/README.md) covers decisions such as filling or dropping values.

Inspect dtypes and missing-value counts together. In pandas 3, ordinary inferred text columns report `str`; use an explicit nullable `string` dtype only when that distinction is part of the data contract.

**Reference:**

- `pd.read_csv(..., na_values=[...], keep_default_na=True)` — define missing tokens while reading
- `series.isna()`, `series.notna()` — null diagnostics
- `df.isna().sum()` — column-level null counts

**Example:**

```python
survey = pd.read_csv(
    'employee_survey.csv',
    na_values=['NA', 'NULL', '?'],
    keep_default_na=True,
    skiprows=[1],
    usecols=['name', 'role', 'bonus', 'start_date']
)

display(survey.isna().sum())
display(survey.dtypes)
```

## Data Type Conversion

Converting data to the correct types is essential for proper analysis. This short section introduces the mechanics; Lecture 05 ties each conversion to a data contract and decides what to do with invalid or lossy values.

**Reference:**

- `df.astype('int64')` - Convert to integer
- `df.astype('float64')` - Convert to float
- `series.astype('string')` - Explicitly request nullable string data
- `pd.to_datetime(df['date_column'])` - Convert to datetime
- `pd.to_numeric(df['column'], errors='coerce')` - Convert to numeric, errors become NaN

**Example:**

```python
# Convert text digits and whole-valued floats
df = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4.0, 5.0, 6.0]})
df['A'] = df['A'].astype('int64')  # Convert string to integer
df['B'] = df['B'].astype('int64')  # Values are already mathematically whole
display(df.dtypes)  # A: int64, B: int64

# Handle conversion errors
df['C'] = ['1', 'invalid', '4']
df['C'] = pd.to_numeric(df['C'], errors='coerce')  # Invalid becomes NaN
display(df['C'])  # [1.0, NaN, 4.0]
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

## GroupBy Preview

GroupBy follows a split-apply-combine idea: split rows by a key, compute within each group, and combine the results. For a first glimpse, this calculates one mean per department:

**Example:**

```python
pay = pd.DataFrame({
    'Department': ['Engineering', 'Engineering', 'Sales'],
    'Salary': [120000, 115000, 95000],
})
display(pay.groupby('Department')['Salary'].mean())
```

[Lecture 08](../08/README.md) is the canonical treatment of grouping, aggregation, `transform()`, and `filter()`.

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

- `df.to_csv('asdf.csv')` - no frills
- `df.to_csv('tab_separated.tsv', sep='\t')`
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
- `df.isna().sum()` - Missing values per column

**Example:**

```python
# Summary statistics
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})
display(df.describe())  # count, mean, std, min, 25%, 50%, 75%, max
df.info()  # Prints data types and memory usage
display(df.isna().sum())  # Missing values per column
```

## Data Quality Assessment

### Inspection preview (decisions come next lecture)

This short preview shows how to inspect missing values and duplicate rows after the pandas introduction. Cleaning decisions and transformations belong to Lecture 05, where they are tied to a documented data contract.

**Reference:**

- `df.isna().sum()` - Count missing values per column
- `df.duplicated().sum()` - Count duplicate rows

**Example:**

```python
# Inspect without changing the table
display(df.isna().sum())       # Missing values per column
display(df.duplicated().sum()) # Number of duplicate rows
```

Lecture 05 picks up from this inspection and documents the cleaning decisions before transforming and validating a working table.

# LIVE DEMO!

(Demo 3: Data I/O - CSV, Excel, JSON, and quality inspection)

> Never be afraid to make a mistake. Unless it's in Git. Then be afraid. Be very afraid.
