Demo Guide - Lecture 4: Pandas on Jupyter

# Demo 1: Jupyter Notebooks Basics

**Objective**: Get comfortable with Jupyter notebooks in VS Code - creating, running, and managing notebooks.

**Key Concepts**: Notebook interface, cell execution, kernel management, git safety

## Step 1: Create Your First Notebook

1. Open VS Code
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Type "Jupyter: Create New Jupyter Notebook"
4. Save as `demo1_jupyter_basics.ipynb` in the demo folder

**What this demonstrates**: VS Code's Jupyter integration makes notebooks feel like native code files

## Step 2: Code and Markdown Cells

Create a markdown cell (click `+ Markdown` or change cell type):

```markdown
# My First Data Analysis

This notebook demonstrates:
- Loading data
- Basic exploration
- Simple visualization
```

Create a code cell below it:

```python
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

print("Setup complete!")
```

Run the cell with `Shift+Enter`

**What this demonstrates**: Notebooks mix documentation (markdown) with executable code, perfect for analysis storytelling

## Step 3: Magic Commands in Action

Create a new code cell:

```python
# Display plots inline
%matplotlib inline

# Check working directory
%pwd
```

Run it, then create another cell:

```python
# Install a package (if needed)
%pip install pandas

# List installed packages
%pip list | grep pandas
```

**What this demonstrates**: Magic commands give special notebook powers - inline plots, package management, system commands

## Step 4: Interactive Data Exploration

Create a code cell:

```python
# Create sample data
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000]
}

df = pd.DataFrame(data)
df
```

Run it - notice pandas displays the DataFrame nicely

Add another cell to explore:

```python
# Summary statistics
print(df.describe())

# Quick visualization
df.plot(x='name', y='salary', kind='bar')
plt.title('Salaries by Person')
plt.show()
```

**What this demonstrates**: Notebooks show results immediately - no need to print everything, DataFrames render as tables, plots appear inline

## Step 5: Kernel Management

1. Add a variable in a new cell: `test_var = "I'm in memory"`
2. Run it with `Shift+Enter`
3. In a new cell, reference it: `print(test_var)` - it works!
4. Click the "Restart" button (or `Ctrl+Shift+P` → "Jupyter: Restart Kernel")
5. Try running `print(test_var)` again - it fails! Variable is gone

**What this demonstrates**:
- Kernel holds all variables in memory
- Restarting clears everything - useful for debugging
- Good practice: "Restart & Run All" before sharing to ensure code works top-to-bottom

## Step 6: Git Safety - The Critical Step

Your notebook now has OUTPUT saved in it (that bar chart, the DataFrames, etc.)

**Danger**: If that data was sensitive (patient records, financial data), it's now in your git history forever!

**Safe workflow**:
1. Click "Clear All Outputs" button (looks like an eraser)
2. Save the notebook (`Ctrl+S`)
3. Check the file - outputs are gone
4. NOW it's safe to commit

**What this demonstrates**:
- Notebooks save code AND outputs
- Always clear sensitive outputs before committing
- This is the #1 security mistake with notebooks

---

# Demo 2: Pandas DataFrames Exploration

**Objective**: Master pandas DataFrame basics - creation, selection, filtering, and exploration.

**Key Concepts**: DataFrame creation, indexing (.loc/.iloc), boolean filtering, basic operations

## Step 1: Create DataFrame from Dictionary

Create `demo2_pandas_basics.md` (we'll convert to .ipynb later):

```python
import pandas as pd
import numpy as np

# Create from dictionary
employee_data = {
    'employee_id': ['E001', 'E002', 'E003', 'E004', 'E005'],
    'name': ['Alice Johnson', 'Bob Smith', 'Charlie Davis', 'Diana Martinez', 'Eve Wilson'],
    'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales'],
    'salary': [85000, 65000, 92000, 58000, 71000],
    'years_experience': [5, 3, 8, 2, 4]
}

df = pd.DataFrame(employee_data)
print(df)
print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
```

**What this demonstrates**:
- Dictionaries → DataFrames (keys become column names)
- `.shape` shows (rows, columns)
- `.dtypes` shows column data types

## Step 2: Column Selection (THE FIRST THING YOU DO!)

```python
# Single column (returns Series)
names = df['name']
print("Single column (Series):")
print(type(names))
print(names)

# Multiple columns (returns DataFrame)
basic_info = df[['name', 'department']]
print("\nMultiple columns (DataFrame):")
print(type(basic_info))
print(basic_info)

# Select only numeric columns
numeric_cols = df.select_dtypes(include=['number'])
print("\nNumeric columns only:")
print(numeric_cols)
```

**What this demonstrates**:

- **CRITICAL**: Column selection is the first operation after loading data
- `df['col']` → Series, `df[['col1', 'col2']]` → DataFrame
- `select_dtypes()` filters by data type (super useful!)

## Step 3: DataFrame Inspection & Summary Statistics

```python
# First few rows
print("First 3 rows:")
print(df.head(3))

# Summary statistics - THE most important exploration tool
print("\nSummary statistics:")
print(df.describe())

# DataFrame structure and memory
print("\nDataFrame info:")
df.info()

# Count values
print(f"\nTotal rows: {len(df)}")
print(f"\nNon-null counts per column:")
print(df.count())

# Unique values in categorical columns
print(f"\nDepartments: {df['department'].unique()}")
print(f"Number of unique departments: {df['department'].nunique()}")
print(f"\nValue counts for department:")
print(df['department'].value_counts())
```

**What this demonstrates**:

- `.describe()` is your go-to for numeric column summaries
- `.info()` shows data types, memory, and null counts
- `.value_counts()` essential for categorical data exploration

## Step 4: Label vs Position Selection (.loc vs .iloc)

```python
# .loc - label-based selection
print("Using .loc (labels):")
print(df.loc[0, 'name'])  # Row 0, column 'name'
print(df.loc[0:2, ['name', 'salary']])  # Rows 0-2 (inclusive!), specific columns

print("\nUsing .iloc (positions):")
print(df.iloc[0, 1])  # Row 0, column 1 (position)
print(df.iloc[0:2, [1, 3]])  # Rows 0-1 (exclusive!), columns at positions 1 and 3
```

**What this demonstrates**:
- `.loc[]` uses labels (column names, index labels) - INCLUSIVE slicing
- `.iloc[]` uses integer positions - EXCLUSIVE slicing (like Python lists)
- This is the #1 confusion point for pandas beginners!

## Step 5: Boolean Filtering & .query()

```python
# Single condition
high_earners = df[df['salary'] > 70000]
print("High earners (>$70k):")
print(high_earners)

# Multiple conditions (& = AND, | = OR, must use parentheses!)
experienced_engineers = df[(df['department'] == 'Engineering') & (df['years_experience'] > 4)]
print("\nExperienced engineers:")
print(experienced_engineers)

# Using .isin() for multiple values
sales_or_marketing = df[df['department'].isin(['Sales', 'Marketing'])]
print("\nSales or Marketing:")
print(sales_or_marketing)

# Using .query() for more readable filtering
high_earner_query = df.query('salary > 70000')
print("\nUsing .query() - more readable:")
print(high_earner_query)

# Complex query with multiple conditions
complex_query = df.query('department == "Engineering" and years_experience > 4')
print("\nComplex query:")
print(complex_query)
```

**What this demonstrates**:

- Boolean masks filter rows based on conditions
- Use `&` (and), `|` (or), `~` (not) - MUST use parentheses
- `.isin()` checks membership in a list
- `.query()` provides SQL-like readable syntax (great for complex filters!)

## Step 6: Column Operations

```python
# Create new column
df['salary_per_year_exp'] = df['salary'] / df['years_experience']
print("Added calculated column:")
print(df[['name', 'salary_per_year_exp']])

# Rename columns
df_renamed = df.rename(columns={'years_experience': 'experience_yrs'})
print("\nRenamed column:")
print(df_renamed.columns)

# Drop columns
df_subset = df.drop(columns=['employee_id'])
print("\nAfter dropping employee_id:")
print(df_subset.columns)
```

**What this demonstrates**: DataFrames are mutable - add, rename, drop columns easily

## Step 7: Sorting and Ranking

```python
# Sort by salary
df_sorted = df.sort_values('salary', ascending=False)
print("Sorted by salary (high to low):")
print(df_sorted[['name', 'salary']])

# Sort by multiple columns
df_multi_sort = df.sort_values(['department', 'salary'], ascending=[True, False])
print("\nSorted by department, then salary within each:")
print(df_multi_sort[['name', 'department', 'salary']])

# Sort by index
df_sorted_index = df.sort_index()
print("\nSorted by index:")
print(df_sorted_index)
```

**What this demonstrates**: Sorting is crucial for finding patterns and top/bottom values

---

# Demo 3: Data I/O and Real-World Workflow

**Objective**: Load real data from files, handle missing values, and perform basic analysis.

**Key Concepts**: CSV reading, missing data, value counts, groupby basics

## Step 1: Create Sample CSV Data

First, create a sample CSV file to work with:

```python
# Create sample sales data
sales_data = """date,product,quantity,price,region
2024-01-15,Widget A,5,29.99,North
2024-01-16,Widget B,3,49.99,South
2024-01-16,Widget A,2,29.99,East
2024-01-17,Widget C,7,19.99,North
2024-01-17,Widget B,,49.99,South
2024-01-18,Widget A,4,29.99,West
2024-01-18,Widget C,6,,North
2024-01-19,Widget B,8,49.99,East"""

# Write to file
with open('sales_data.csv', 'w') as f:
    f.write(sales_data)

print("Created sales_data.csv")
```

**What this demonstrates**: Creating test data programmatically (useful for demos and testing)

## Step 2: Load and Inspect Data

```python
# Read CSV
df_sales = pd.read_csv('sales_data.csv')

print("Data loaded:")
print(df_sales)

print("\nData types:")
print(df_sales.dtypes)

print("\nMissing values per column:")
print(df_sales.isnull().sum())
```

**What this demonstrates**:
- `read_csv()` infers data types automatically
- Missing values appear as NaN
- Always check for missing data after loading

## Step 3: Data Quality Assessment

```python
# Check for missing values
print("Missing values per column:")
print(df_sales.isnull().sum())

# See rows with missing values
print("\nRows with missing values:")
print(df_sales[df_sales.isnull().any(axis=1)])

# Check for duplicates
print(f"\nNumber of duplicate rows: {df_sales.duplicated().sum()}")

# Check data types
print("\nData types:")
print(df_sales.dtypes)
```

**What this demonstrates**:

- Always check for missing values, duplicates, and data types after loading
- `.isnull().sum()` shows missing count per column
- `.duplicated()` finds duplicate rows

## Step 4: Handle Missing Data & Type Conversion

```python
# Fill missing quantity with 0
df_sales['quantity'] = df_sales['quantity'].fillna(0)

# Fill missing price with median price for that product
df_sales['price'] = df_sales.groupby('product')['price'].transform(lambda x: x.fillna(x.median()))

print("After handling missing values:")
print(df_sales)

# Convert quantity to integer (was float due to NaN)
df_sales['quantity'] = df_sales['quantity'].astype('int64')

# Convert date column to datetime
df_sales['date'] = pd.to_datetime(df_sales['date'])

print("\nAfter type conversion:")
print(df_sales.dtypes)
print(f"\nRemaining missing values: {df_sales.isnull().sum().sum()}")
```

**What this demonstrates**:

- Different strategies for different columns (0 for quantity, median for price)
- `groupby().transform()` fills within groups
- `.astype()` converts data types (useful after filling NaN)
- `pd.to_datetime()` converts strings to datetime objects
- Real-world data is messy - always have a cleaning strategy!

## Step 5: Basic Analysis

```python
# Add calculated column
df_sales['total_sale'] = df_sales['quantity'] * df_sales['price']

print("Sales with totals:")
print(df_sales[['date', 'product', 'quantity', 'price', 'total_sale']])

# Summary by product
print("\nSales by product:")
product_summary = df_sales.groupby('product').agg({
    'quantity': 'sum',
    'total_sale': 'sum'
}).round(2)
print(product_summary)

# Summary by region
print("\nSales by region:")
region_summary = df_sales.groupby('region')['total_sale'].sum().sort_values(ascending=False)
print(region_summary)
```

**What this demonstrates**:
- Calculated columns enable analysis
- `groupby().agg()` summarizes data
- Multiple aggregations at once

## Step 6: Value Counts and Quick Insights

```python
# Most common products
print("Product frequencies:")
print(df_sales['product'].value_counts())

# Regions ranked by number of sales
print("\nSales transactions by region:")
print(df_sales['region'].value_counts())

# Quick statistics
print(f"\nTotal revenue: ${df_sales['total_sale'].sum():.2f}")
print(f"Average transaction: ${df_sales['total_sale'].mean():.2f}")
print(f"Largest sale: ${df_sales['total_sale'].max():.2f}")
```

**What this demonstrates**:
- `.value_counts()` is your best friend for categorical data
- Quick statistical summaries guide deeper analysis

## Step 7: Save Results

```python
# Save cleaned data
df_sales.to_csv('sales_data_clean.csv', index=False)
print("Saved cleaned data to sales_data_clean.csv")

# Save summary
product_summary.to_csv('product_summary.csv')
print("Saved product summary to product_summary.csv")

print("\nFiles created:")
import os
for file in ['sales_data.csv', 'sales_data_clean.csv', 'product_summary.csv']:
    if os.path.exists(file):
        print(f"  ✓ {file}")
```

**What this demonstrates**:
- `to_csv()` saves DataFrames
- `index=False` prevents writing row numbers as a column
- Always save cleaned data and analysis results

---

# Key Takeaways

**Demo 1 - Jupyter**:
- Notebooks are perfect for interactive data analysis
- Mix code and documentation
- Magic commands provide special powers
- Always clear outputs before committing to git

**Demo 2 - DataFrames**:
- DataFrames are like supercharged spreadsheets
- `.loc[]` = labels (inclusive), `.iloc[]` = positions (exclusive)
- Boolean filtering is powerful - use parentheses with `&`/`|`
- Build new columns from existing data

**Demo 3 - Real Workflow**:
- Real data is messy - missing values are normal
- Always inspect data after loading
- Different missing value strategies for different situations
- Save cleaned data and results

**Next Steps for Students**:
- Practice with their own CSV files
- Try different missing value strategies
- Explore groupby aggregations
- Build a mini analysis pipeline
