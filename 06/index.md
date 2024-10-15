---
marp: true
theme: sqrl
paginate: true
class: invert
---

# Lecture 06: Are you ready to wrangle?!?

1. Introduction to Data Wrangling with pandas
2. Combining and Reshaping Data
3. Practical Data Cleaning Techniques

---

# 1. Introduction to Data Wrangling with pandas

---

## What is Data Wrangling?

- Process of transforming and mapping data from one "raw" form into another format
- Aims to make data more appropriate and valuable for downstream purposes
- Key step in data preprocessing for analysis

---

## Why pandas?

- Powerful Python library for data manipulation and analysis
- Built on top of NumPy
- Provides high-performance, easy-to-use data structures and tools
- Essential for working with structured data in Python

---

## Key pandas Data Structures

1. Series: 1-dimensional labeled array
2. DataFrame: 2-dimensional labeled data structure with columns of potentially different types

---

## Creating a Series

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

## Creating a DataFrame

```python
data = {'name': ['John', 'Jane', 'Bob'],
        'age': [25, 30, 35],
        'city': ['New York', 'Paris', 'London']}
df = pd.DataFrame(data)
print(df)
```

Output:
```
   name  age      city
0  John   25  New York
1  Jane   30     Paris
2   Bob   35    London
```

---

## Reading Data into pandas

Common file formats:
- CSV: `pd.read_csv()`
- Excel: `pd.read_excel()`
- JSON: `pd.read_json()`
- SQL databases: `pd.read_sql()`

Example:
```python
df = pd.read_csv('patient_data.csv')
```

---

## Basic DataFrame Operations

- Viewing data: `df.head()`, `df.tail()`, `df.info()`
- Selecting columns: `df['column_name']` or `df.column_name`
- Filtering rows: `df[df['column_name'] > value]`
- Adding new columns: `df['new_column'] = values`

---

## Handling Missing Data

- Detecting missing values: `df.isna()`, `df.isnull()`
- Dropping missing values: `df.dropna()`
- Filling missing values: `df.fillna(value)`

```python
# Fill missing values with the mean of the column
df['age'].fillna(df['age'].mean(), inplace=True)
```

---

## Data Type Conversion

- Checking data types: `df.dtypes`
- Converting types: `df['column'].astype(type)`

```python
# Convert 'age' column to integer type
df['age'] = df['age'].astype(int)
```

---

## Renaming Columns

```python
df = df.rename(columns={'old_name': 'new_name'})
```

or

```python
df.columns = ['new_name1', 'new_name2', 'new_name3']
```

---

## Sorting Data

```python
# Sort by a single column
df_sorted = df.sort_values('age')

# Sort by multiple columns
df_sorted = df.sort_values(['age', 'name'], ascending=[True, False])
```

---

## Grouping and Aggregation

```python
# Group by 'city' and calculate mean age
df.groupby('city')['age'].mean()

# Multiple aggregations
df.groupby('city').agg({'age': 'mean', 'name': 'count'})
```

---

# BREAK TIME!

Let's take a 5-minute break to stretch and process what we've learned so far.

When we return, we'll dive into combining and reshaping data with pandas!

---

# 2. Combining and Reshaping Data

---

## Concatenating DataFrames

```python
df1 = pd.DataFrame({'A': ['A0', 'A1'], 'B': ['B0', 'B1']})
df2 = pd.DataFrame({'A': ['A2', 'A3'], 'B': ['B2', 'B3']})

result = pd.concat([df1, df2])
```

Result:
```
    A   B
0  A0  B0
1  A1  B1
0  A2  B2
1  A3  B3
```

---

## Merging DataFrames

```python
left = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                     'A': ['A0', 'A1', 'A2']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K3'],
                      'B': ['B0', 'B1', 'B2']})

result = pd.merge(left, right, on='key')
```

Result:
```
  key   A   B
0  K0  A0  B0
1  K1  A1  B1
```

---

## Types of Joins

- Inner join (default): `pd.merge(left, right, how='inner')`
- Outer join: `pd.merge(left, right, how='outer')`
- Left join: `pd.merge(left, right, how='left')`
- Right join: `pd.merge(left, right, how='right')`

---

## Reshaping Data: Melt

Converting from wide to long format:

```python
df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                   'B': {0: 1, 1: 3, 2: 5},
                   'C': {0: 2, 1: 4, 2: 6}})

melted = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
```

Result:
```
     A variable  value
0    a        B      1
1    b        B      3
2    c        B      5
3    a        C      2
4    b        C      4
5    c        C      6
```

---

## Reshaping Data: Pivot

Converting from long to wide format:

```python
pivoted = melted.pivot(index='A', columns='variable', values='value')
```

Result:
```
variable  B  C
A            
a         1  2
b         3  4
c         5  6
```

---

## Stacking and Unstacking

```python
# Stacking
stacked = df.stack()

# Unstacking
unstacked = stacked.unstack()
```

---

# 3. Practical Data Cleaning Techniques

---

## Handling Duplicates

```python
# Check for duplicates
df.duplicated()

# Remove duplicates
df_clean = df.drop_duplicates()
```

---

## Handling Outliers

1. Identify outliers (e.g., using Z-score or IQR)
2. Decide on a strategy: remove, cap, or transform

```python
# Example: Capping outliers using quantiles
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['column'] = df['column'].clip(lower_bound, upper_bound)
```

---

## String Manipulation

pandas provides vectorized string methods:

```python
# Convert to lowercase
df['name'] = df['name'].str.lower()

# Remove whitespace
df['name'] = df['name'].str.strip()

# Replace values
df['name'] = df['name'].str.replace('old', 'new')
```

---

## Working with Dates and Times

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Calculate time differences
df['time_diff'] = df['end_date'] - df['start_date']
```

---

## Binning Data

```python
# Create age groups
bins = [0, 18, 35, 50, 65, 100]
labels = ['0-18', '19-35', '36-50', '51-65', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
```

---

## Handling Categorical Data

```python
# Convert to category type
df['category'] = df['category'].astype('category')

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'])
```

---

## Putting It All Together: A Data Cleaning Pipeline

1. Load the data
2. Handle missing values
3. Remove duplicates
4. Handle outliers
5. Convert data types
6. Feature engineering
7. Save cleaned data

---

# Conclusion

- pandas is a powerful tool for data wrangling in Python
- Key operations: reading, cleaning, reshaping, and combining data
- Practice and experimentation are key to mastering pandas

Remember: "Data scientists spend 80% of their time cleaning data, and 20% of their time complaining about cleaning data." - Unknown

---

# Questions?

Thank you for your attention! Any questions about pandas or data wrangling?
