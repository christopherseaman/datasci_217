# Pandas on Jupyter: Data Structures & I/O

See [BONUS.md](BONUS.md) for advanced topics:
- Data alignment and arithmetic operations
- Function application with `apply()` and `map()`
- Ranking data
- Working with duplicate index labels

# Jupyter Notebooks: Interactive Data Analysis

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

**Brief Example:**

```python
# Cell 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Cell 2: Load data
df = pd.read_csv('data.csv')
print(f"Data shape: {df.shape}")

# Cell 3: Quick visualization
df.head().plot(kind='bar')
plt.title('Sample Data')
plt.show()
```

## Jupyter Magic Commands

Magic commands provide special functionality for notebook environments. They start with `%` and extend Jupyter's capabilities for data analysis.

**Reference:**

- `%matplotlib inline` - Display plots within notebook cells
- `%pwd` - Print current working directory
- `%ls` - List directory contents
- `%pip install package_name` - Install Python packages
- `%pip list` - List installed packages
- `%pip show package_name` - Show package information

**Brief Example:**

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

**Brief Example:**

```python
# VS Code automatically detects .ipynb files
# Just open any .ipynb file and start coding
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
```

## Kernel Management Basics

The kernel is the Python interpreter running your code. Sometimes it gets stuck or needs a fresh start.

**Reference:**

- **Restart Kernel**: Clears all variables from memory, fresh start
- **Run All**: Executes all cells from top to bottom
- **Restart & Run All**: Combines both - useful for testing if code works from scratch
- Common issues: Variable conflicts, memory issues, stuck computations

**Brief Example:**

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

**Brief Example:**

```python
# This output contains sensitive data and will be saved in the notebook
df = pd.read_csv('patient_data.csv')
print(df.head())  # Shows patient names, IDs, medical info
# Oops! Now everyone can see John Doe's blood pressure on GitHub
```

**After running this code, the patient data will be visible in your notebook file. Always clear outputs before sharing or committing to git.**

**LIVE DEMO!**

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

**Brief Example:**

```python
# Create Series
ages = pd.Series([25, 30, 35, 40], index=['Alice', 'Bob', 'Charlie', 'Diana'])
print(ages)  # Alice: 25, Bob: 30, Charlie: 35, Diana: 40
print(ages.index)  # ['Alice', 'Bob', 'Charlie', 'Diana']
print(ages.values)  # [25 30 35 40]

# Series operations
print(ages.mean())  # 32.5
print(ages.describe())  # count, mean, std, min, 25%, 50%, 75%, max
```

### DataFrame

A DataFrame is a two-dimensional labeled data structure with columns of potentially different types. It's like a spreadsheet or SQL table, providing powerful data manipulation capabilities.

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

**Brief Example:**

```python
# Create DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
})
print(df.shape)  # (3, 3)
print(df.dtypes)  # Name: object, Age: int64, Salary: int64
print(df.describe())  # Summary statistics for numeric columns
```

## Data Type Conversion

Converting data to the correct types is essential for proper analysis. This includes converting strings to numbers, dates, and other appropriate types.

**Reference:**

- `df.astype('int64')` - Convert to integer
- `df.astype('float64')` - Convert to float
- `df.astype('string')` - Convert to string
- `pd.to_datetime(df['date_column'])` - Convert to datetime
- `pd.to_numeric(df['column'], errors='coerce')` - Convert to numeric, errors become NaN

**Brief Example:**

```python
# Convert data types
df = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4.5, 5.5, 6.5]})
df['A'] = df['A'].astype('int64')  # Convert string to integer
df['B'] = df['B'].astype('int64')  # Convert float to integer
print(df.dtypes)  # A: int64, B: int64

# Handle conversion errors
df['C'] = ['1', '2', 'invalid', '4']
df['C'] = pd.to_numeric(df['C'], errors='coerce')  # Invalid becomes NaN
print(df['C'])  # [1.0, 2.0, NaN, 4.0]
```

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

**Reference:**

- `df.loc[row_indexer, col_indexer]` - Label-based selection (uses labels)
- `df.iloc[row_indexer, col_indexer]` - Position-based selection (uses integer positions)
- `df['column_name']` - Select single column
- `df[['col1', 'col2']]` - Select multiple columns
- `df[df['column'] > value]` - Boolean indexing
- `df.query('condition')` - Query with string expression
- `df.isin(values)` - Check membership
- `df.between(left, right)` - Range selection

**Brief Example:**

```python
# Label-based selection
df.loc[0, 'Name']  # First row, Name column
df.loc[0:2, ['Name', 'Age']]  # Rows 0-2, Name and Age columns

# Position-based selection
df.iloc[0, 0]  # First row, first column
df.iloc[0:2, 0:2]  # First 2 rows, first 2 columns

# Boolean indexing
adults = df[df['Age'] >= 18]  # Rows where Age >= 18
high_earners = df[df['Salary'] > 60000]  # High salary rows
```

**LIVE DEMO!**

# Essential Pandas Operations

## Sorting Data

Sorting organizes your data by values or index, making it easier to find patterns and outliers. This is one of the most common operations in data analysis.

**Reference:**

- `df.sort_values('column')` - Sort by column values
- `df.sort_values(['col1', 'col2'])` - Sort by multiple columns
- `ascending=False` - Sort in descending order
- `df.sort_index()` - Sort by index

**Brief Example:**

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

**Brief Example:**

```python
# Find unique values
categories = pd.Series(['A', 'B', 'A', 'C', 'B'])
print(categories.unique())  # ['A' 'B' 'C']
print(categories.value_counts())  # A: 2, B: 2, C: 1

# Filter by membership
print(categories.isin(['A', 'B']))  # [True, True, True, False, True]
```

**LIVE DEMO!**

# Data Loading and Storage

*Pro tip: If you're ever stuck with a weird file format, remember: "There's a pandas function for that!" (Usually `pd.read_[format]()` - pandas is surprisingly comprehensive at reading data from just about anywhere)*

## Reading CSV Files

CSV files are the most common format for data analysis. Pandas makes it easy to read CSV files with sensible defaults.

*Fun fact: CSV stands for "Comma-Separated Values," but in reality, it's more like "Comma-Separated Values (unless someone used semicolons, or tabs, or pipes, or any other delimiter they felt like using that day)."*

**Reference:**

- `pd.read_csv('filename.csv')` - Read CSV file
- `pd.read_csv('filename.csv', sep=';')` - Custom separator
- `pd.read_csv('filename.csv', header=0)` - Specify header row
- `pd.read_csv('filename.csv', index_col=0)` - Use first column as index

**Brief Example:**

```python
# Basic CSV reading
df = pd.read_csv('data.csv')
print(df.head())

# Custom options
df = pd.read_csv('data.csv', sep=';', index_col=0)
print(df.head())
```

## Reading Excel Files

Excel files are common in business environments. Pandas can read Excel files and handle multiple sheets.

**Reference:**

- `pd.read_excel('filename.xlsx')` - Read Excel file
- `pd.read_excel('filename.xlsx', sheet_name='Sheet1')` - Read specific sheet
- `pd.read_excel('filename.xlsx', sheet_name=None)` - Read all sheets

**Brief Example:**

```python
# Read Excel file
df = pd.read_excel('data.xlsx')
print(df.head())

# Read specific sheet
df = pd.read_excel('data.xlsx', sheet_name='Sales')
print(df.head())
```

## Writing Data Files

Saving your cleaned and analyzed data is essential for sharing results and creating reports.

**Reference:**

- `df.to_csv('filename.csv')` - Write CSV file
- `df.to_excel('filename.xlsx')` - Write Excel file
- `df.to_csv('filename.csv', index=False)` - Don't include index
- `df.to_excel('filename.xlsx', sheet_name='Results')` - Custom sheet name

**Brief Example:**

```python
# Write CSV file
df.to_csv('output.csv', index=False)

# Write Excel file
df.to_excel('output.xlsx', sheet_name='Results')
```

## Handling Missing Values in CSV Files

Real-world CSV files often have missing or invalid data. Pandas provides options to handle this during reading.

**Reference:**

- `na_values=['NA', 'NULL', '']` - Specify additional NA markers
- `keep_default_na=False` - Disable default NA recognition
- `nrows=1000` - Read only first N rows (useful for testing)
- `skiprows=[0, 2, 3]` - Skip specific rows

**Brief Example:**

```python
# Read CSV with custom missing value markers
df = pd.read_csv('data.csv', na_values=['NA', 'NULL', 'missing', '?'])

# Read only first 100 rows to test
df_sample = pd.read_csv('large_file.csv', nrows=100)
print(df_sample.head())
```

## Reading Large Files in Chunks

For files too large to fit in memory, read them in chunks and process iteratively.

<!-- FIXME: Add visual showing chunked reading workflow:
     - Large file → chunks → process → combine
     - Memory usage comparison: full load vs chunked
     File: media/chunked_reading.png -->

**Reference:**

- `chunksize=10000` - Read file in chunks of N rows
- `TextFileReader` - Iterator object returned when using chunksize
- Process each chunk with a loop

**Brief Example:**

```python
# Read and process large file in chunks
chunk_iter = pd.read_csv('huge_file.csv', chunksize=10000)
results = []

for chunk in chunk_iter:
    # Process each chunk
    processed = chunk[chunk['value'] > 0].groupby('category').sum()
    results.append(processed)

# Combine results
final = pd.concat(results, axis=0).groupby(level=0).sum()
```

## Reading JSON Data

JSON is a common format for web APIs and modern data exchange.

**Reference:**

- `pd.read_json('file.json')` - Read JSON file
- `df.to_json('file.json')` - Write JSON file
- `orient='records'` - JSON format option

**Brief Example:**

```python
# Read JSON file
df = pd.read_json('data.json')

# Write to JSON
df.to_json('output.json', orient='records')
```

## Working with Databases

For data in databases, pandas can connect and read directly using SQL queries.

*Note: Database connections covered in detail in advanced lectures. For now, just know it's possible!*

**Reference:**

- `pd.read_sql(query, connection)` - Execute SQL query, return DataFrame
- Requires database connection (SQLAlchemy or similar)

**Brief Example:**

```python
# Basic SQL reading (requires connection setup)
# import sqlalchemy as sqla
# db = sqla.create_engine('sqlite:///mydb.sqlite')
# df = pd.read_sql('SELECT * FROM table', db)
```

**LIVE DEMO!**

# Data Exploration and Summary Statistics

## Summary Statistics

Summary statistics provide a quick overview of your data's distribution and characteristics. They're essential for understanding data quality and identifying patterns.

<!-- FIXME: Add xkcd 552 about correlation
     https://xkcd.com/552/
     File: media/xkcd_552.png -->

*Remember: Correlation does not imply causation! (But it's still useful for understanding patterns in your data)*

**Reference:**

- `df.describe()` - Summary statistics for numeric columns
- `df.info()` - Data types and memory usage
- `df.shape` - (rows, columns) tuple
- `df.count()` - Non-null values per column
- `df.nunique()` - Unique values per column
- `df.memory_usage()` - Memory usage per column
- `df.isnull().sum()` - Missing values per column

**Brief Example:**

```python
# Summary statistics
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})
print(df.describe())  # count, mean, std, min, 25%, 50%, 75%, max
print(df.info())  # Data types and memory usage
print(df.isnull().sum())  # Missing values per column
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

**Brief Example:**

```python
# Check for missing values
print(df.isnull().sum())  # Missing values per column

# Check for duplicates
print(df.duplicated().sum())  # Number of duplicate rows
df_clean = df.drop_duplicates()  # Remove duplicates

# Check data types
print(df.dtypes)  # Data types per column
print(df.info())  # Detailed information
```

**LIVE DEMO!**


---

## Want to Learn More?

See [BONUS.md](BONUS.md) for advanced topics:
- Data alignment and arithmetic operations
- Function application with `apply()` and `map()`
- Ranking data
- Working with duplicate index labels
