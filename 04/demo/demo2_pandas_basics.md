---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Pandas DataFrame Exploration

Mastering pandas basics: creation, selection, filtering, and operations

## Step 1: Create DataFrame from Dictionary

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
