---
title: "Bonus: Advanced Data Wrangling"
---

Data Wrangling: Advanced Topics

*These are more advanced or specialized operations from McKinney Chapter 8. They're incredibly powerful but you won't need them daily as a beginner. Come back to these when you encounter specific use cases that require hierarchical data management or specialized joining techniques.*

See [README.md](README.md) for core data wrangling operations - master those first!

## Advanced Topics Covered

1. **Advanced MultiIndex Operations** - Deep dive into hierarchical indexing with swaplevel(), level-specific sorting, and summary statistics by level
2. **Merging on Index** - Join DataFrames using index values instead of columns
3. **DataFrame.join() Method** - Simplified index-based merging
4. **combine_first() for Patching Missing Data** - Fill gaps by preferentially choosing non-null values
5. **Advanced concat Options** - Using keys, levels, names, and verify_integrity for complex concatenations
6. **MultiIndex Creation Methods** - Programmatically build hierarchical indexes with from_tuples(), from_product(), from_arrays()
7. **Stack/Unstack with dropna Parameter** - Control how missing data is handled during reshaping
8. **Hierarchical Columns from Pivot** - Create and work with MultiIndex in column headers

---

## 1. Advanced MultiIndex Operations

*You've seen basic MultiIndex - now let's go deeper. MultiIndex becomes essential when working with hierarchical data like time series with multiple metrics, or nested business hierarchies.*

### Swapping and Reordering Index Levels

When you have multiple index levels, you may need to change their order for different analyses.

**Reference:**

- `df.swaplevel(0, 1)` - Exchange two index levels by position
- `df.swaplevel('level1', 'level2')` - Exchange by name
- `df.sort_index(level=0)` - Sort by specific level
- `df.sort_index(level='level_name')` - Sort by named level
- Combine swaplevel() + sort_index() for reordering

**Example:**

```python
import pandas as pd
import numpy as np

# Hierarchical data: Store performance by region and quarter
data = pd.DataFrame(
    np.arange(12).reshape((4, 3)),
    index=[['West', 'West', 'East', 'East'], ['Q1', 'Q2', 'Q1', 'Q2']],
    columns=['Revenue', 'Costs', 'Profit']
)
data.index.names = ['Region', 'Quarter']
print(data)
#                 Revenue  Costs  Profit
# Region Quarter
# West   Q1            0      1       2
#        Q2            3      4       5
# East   Q1            6      7       8
#        Q2            9     10      11

# Swap levels - Quarter becomes outer, Region becomes inner
swapped = data.swaplevel('Region', 'Quarter')
print(swapped)
#                 Revenue  Costs  Profit
# Quarter Region
# Q1      West          0      1       2
# Q2      West          3      4       5
# Q1      East          6      7       8
# Q2      East          9     10      11

# Now sort by the new outer level (Quarter)
sorted_data = swapped.sort_index(level=0)
print(sorted_data)
#                 Revenue  Costs  Profit
# Quarter Region
# Q1      East          6      7       8
#         West          0      1       2
# Q2      East          9     10      11
#         West          3      4       5

# Shorthand: swap and sort in one go
result = data.swaplevel(0, 1).sort_index(level=0)
```

**When you need this:**
- Changing perspective on hierarchical data (year→month→day vs day→month→year)
- Preparing data for specific groupby operations
- Making partial selection easier (e.g., all Q1 data across regions)

**Gotcha:** Sorting is critical for performance with MultiIndex. Always sort after creating or modifying MultiIndex for faster .loc[] operations.

---

### Summary Statistics by Level

Aggregate data at specific levels of a MultiIndex without flattening the entire structure.

**Reference:**

- `df.groupby(level='level_name').sum()` - Aggregate by named level
- `df.groupby(level=0).mean()` - Aggregate by level position
- `df.groupby(level=['level1', 'level2']).agg(['sum', 'mean'])` - Multiple levels and functions
- Works with any aggregation function (sum, mean, count, std, etc.)

**Example:**

```python
# Sum across all quarters for each region
regional_totals = data.groupby(level='Region').sum()
print(regional_totals)
#         Revenue  Costs  Profit
# Region
# East         15     17      19
# West          3      5       7

# Average by quarter across all regions
quarterly_avg = data.groupby(level='Quarter').mean()
print(quarterly_avg)
#          Revenue  Costs  Profit
# Quarter
# Q1           3.0    4.0     5.0
# Q2           6.0    7.0     8.0

# Both axes - sum columns by level too (if you had MultiIndex columns)
frame = pd.DataFrame(
    np.arange(12).reshape((3, 4)),
    index=['a', 'b', 'c'],
    columns=[['Ohio', 'Ohio', 'Colorado', 'Colorado'],
             ['Green', 'Red', 'Green', 'Red']]
)
frame.columns.names = ['state', 'color']
print(frame)
# state      Ohio     Colorado
# color     Green Red    Green Red
# a             0   1        2   3
# b             4   5        6   7
# c             8   9       10  11

# Sum across states (collapse state level)
by_color = frame.groupby(level='color', axis='columns').sum()
print(by_color)
# color  Green  Red
# a          2    4
# b         10   12
# c         18   20
```

**Real-world use case:**
- Sales data: Total by product category ignoring individual products
- Time series: Monthly totals from daily data with Year/Month/Day index
- Organizational data: Department totals ignoring individual teams

---

## 2. Merging on Index

*Sometimes your "key" isn't a column - it's the index itself. This is common with time series or when you've already structured data with meaningful indexes.*

Instead of merging on columns, you can merge using the index of one or both DataFrames.

**Reference:**

- `pd.merge(left, right, left_index=True, right_index=True)` - Merge both indexes
- `pd.merge(left, right, left_on='col', right_index=True)` - Column to index
- `pd.merge(left, right, left_index=True, right_on='col')` - Index to column
- `how='inner'/'left'/'right'/'outer'` - Still applies

**Example:**

```python
# Customer lookup table (index = customer_id)
customers = pd.DataFrame(
    {'name': ['Alice', 'Bob', 'Charlie'],
     'city': ['Seattle', 'Portland', 'Eugene']},
    index=['C001', 'C002', 'C003']
)
customers.index.name = 'customer_id'

# Purchase data (customer_id as regular column)
purchases = pd.DataFrame({
    'customer_id': ['C001', 'C001', 'C002', 'C004'],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'amount': [999.99, 25.99, 79.99, 299.99]
})

print(customers)
#              name      city
# customer_id
# C001        Alice   Seattle
# C002          Bob  Portland
# C003      Charlie    Eugene

# Merge: purchases column 'customer_id' to customers index
merged = pd.merge(purchases, customers,
                  left_on='customer_id', right_index=True)
print(merged)
#   customer_id  product  amount     name      city
# 0        C001   Laptop  999.99    Alice   Seattle
# 1        C001    Mouse   25.99    Alice   Seattle
# 2        C002 Keyboard   79.99      Bob  Portland

# Notice C003 (Charlie) and C004 (Monitor) are missing - inner join!
# Use how='left' to keep all purchases
merged_left = pd.merge(purchases, customers,
                       left_on='customer_id', right_index=True, how='left')
print(merged_left)
#   customer_id  product  amount     name      city
# 0        C001   Laptop  999.99    Alice   Seattle
# 1        C001    Mouse   25.99    Alice   Seattle
# 2        C002 Keyboard   79.99      Bob  Portland
# 3        C004  Monitor  299.99      NaN       NaN  # No customer info

# Both DataFrames using index
left_indexed = purchases.set_index('customer_id')
both_index = pd.merge(left_indexed, customers,
                      left_index=True, right_index=True, how='outer')
print(both_index)
```

**When you need this:**
- Time series with datetime indexes
- Lookup tables where index is the key
- After set_index() operations
- Joining dimension tables to fact tables (data warehouse style)

**Gotcha:** When merging on index, the index values are preserved in many-to-one joins (unlike column merges which discard indexes).

---

## 3. DataFrame.join() Method

*DataFrame.join() is a convenience method for index-based merging. It's less flexible than merge() but cleaner for common cases.*

The join() method is an instance method (called on a DataFrame) that defaults to joining on indexes.

**Reference:**

- `left_df.join(right_df)` - Join on indexes (default how='left')
- `left_df.join(right_df, how='inner')` - Change join type
- `left_df.join(right_df, on='column')` - Join left's column to right's index
- `left_df.join([df2, df3, df4])` - Join multiple DataFrames at once
- Simpler than merge() for index-based joins

**Example:**

```python
# Product prices (indexed by product_id)
prices = pd.DataFrame(
    {'price': [999.99, 25.99, 79.99, 299.99]},
    index=['P001', 'P002', 'P003', 'P004']
)
prices.index.name = 'product_id'

# Product categories (indexed by product_id)
categories = pd.DataFrame(
    {'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics']},
    index=['P001', 'P002', 'P003', 'P004']
)
categories.index.name = 'product_id'

# Join prices with categories (both on index)
products = prices.join(categories)
print(products)
#               price     category
# product_id
# P001        999.99  Electronics
# P002         25.99  Accessories
# P003         79.99  Accessories
# P004        299.99  Electronics

# Join multiple DataFrames at once
inventory = pd.DataFrame(
    {'stock': [5, 50, 20, 3]},
    index=['P001', 'P002', 'P003', 'P004']
)
inventory.index.name = 'product_id'

full_catalog = prices.join([categories, inventory])
print(full_catalog)
#               price     category  stock
# product_id
# P001        999.99  Electronics      5
# P002         25.99  Accessories     50
# P003         79.99  Accessories     20
# P004        299.99  Electronics      3

# Join column to index (like left_on/right_index)
orders = pd.DataFrame({
    'product_id': ['P001', 'P002', 'P001'],
    'quantity': [2, 5, 1]
})
orders_with_prices = orders.join(prices, on='product_id')
print(orders_with_prices)
#   product_id  quantity   price
# 0       P001         2  999.99
# 1       P002         5   25.99
# 2       P001         1  999.99
```

**Comparison to merge():**

```python
# These are equivalent:
result1 = left_df.join(right_df, how='left')
result2 = pd.merge(left_df, right_df, left_index=True, right_index=True, how='left')

# join() is cleaner when:
# - Joining on indexes (the common case)
# - Joining multiple DataFrames at once
# - You prefer method chaining style

# merge() is better when:
# - Joining on columns
# - Different column names (left_on/right_on)
# - You need explicit control over everything
```

---

## 4. combine_first() for Patching Missing Data

*Think of combine_first() as saying "use my data, but if I'm missing something, fill it in from the other DataFrame." It's perfect for data patching and fallback values.*

The combine_first() method fills missing values with corresponding values from another DataFrame or Series.

**Reference:**

- `df1.combine_first(df2)` - Use df1 values; fill NaN from df2
- Works with both Series and DataFrame
- Aligns on index automatically
- Result includes union of both indexes
- Preserves dtypes when possible

**Example:**

```python
# Survey responses - Week 1 (incomplete)
week1 = pd.Series([95.0, np.nan, 88.0, np.nan],
                  index=['Alice', 'Bob', 'Charlie', 'Diana'])

# Survey responses - Week 2 (filled some gaps)
week2 = pd.Series([np.nan, 92.0, np.nan, 85.0, 90.0],
                  index=['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'])

print("Week 1:", week1)
# Alice      95.0
# Bob         NaN
# Charlie    88.0
# Diana       NaN

print("Week 2:", week2)
# Alice       NaN
# Bob        92.0
# Charlie     NaN
# Diana      85.0
# Eve        90.0

# Combine: Prefer week1, fill gaps from week2
combined = week1.combine_first(week2)
print(combined)
# Alice      95.0  # From week1
# Bob        92.0  # From week2 (week1 was NaN)
# Charlie    88.0  # From week1
# Diana      85.0  # From week2 (week1 was NaN)
# Eve        90.0  # From week2 (not in week1)

# Works with DataFrames too
df1 = pd.DataFrame({
    'A': [1.0, np.nan, 5.0],
    'B': [np.nan, 2.0, np.nan],
    'C': range(2, 18, 4)
}, index=[0, 1, 2])

df2 = pd.DataFrame({
    'A': [5.0, 4.0, np.nan, 3.0],
    'B': [np.nan, 3.0, 4.0, 6.0]
}, index=[0, 1, 2, 3])

result = df1.combine_first(df2)
print(result)
#      A    B     C
# 0  1.0  NaN   2.0  # A from df1, C from df1
# 1  4.0  2.0   6.0  # A filled from df2, B from df1
# 2  5.0  4.0  10.0  # A from df1, B filled from df2
# 3  3.0  6.0   NaN  # Entire row from df2
```

**Real-world use cases:**
- Merging corrected data with original (corrections take priority)
- Filling data gaps from multiple sources
- Combining model predictions with actual values
- Patching incomplete time series

**Gotcha:** combine_first() always prefers the calling DataFrame (left side). If you want different priority, swap the order: `df2.combine_first(df1)` instead.

---

## 5. Advanced concat Options

*Basic concat is straightforward, but these options give you fine control over how pieces are labeled and validated.*

Beyond basic concatenation, you can add hierarchical labels, name levels, and validate data integrity.

**Reference:**

- `keys=['name1', 'name2']` - Add outer level with these labels
- `names=['level1', 'level2']` - Name the hierarchical levels
- `verify_integrity=True` - Raise error if indexes overlap
- `join='inner'/'outer'` - Handle column mismatches
- `ignore_index=True` - Discard existing indexes

**Example with keys:**

```python
# Sales from two different systems
system_a = pd.DataFrame({
    'product': ['Laptop', 'Mouse'],
    'amount': [999.99, 25.99]
})

system_b = pd.DataFrame({
    'product': ['Keyboard', 'Monitor'],
    'amount': [79.99, 299.99]
})

# Concatenate with source labels
combined = pd.concat([system_a, system_b],
                     keys=['SystemA', 'SystemB'])
print(combined)
#            product  amount
# SystemA 0   Laptop  999.99
#         1    Mouse   25.99
# SystemB 0 Keyboard   79.99
#         1  Monitor  299.99

# Now you can select by source
print(combined.loc['SystemA'])
#   product  amount
# 0  Laptop  999.99
# 1   Mouse   25.99

# Name the levels for clarity
combined_named = pd.concat([system_a, system_b],
                           keys=['SystemA', 'SystemB'],
                           names=['source', 'original_index'])
print(combined_named)
#                           product  amount
# source  original_index
# SystemA 0                  Laptop  999.99
#         1                   Mouse   25.99
# SystemB 0                Keyboard   79.99
#         1                 Monitor  299.99
```

**Example with verify_integrity:**

```python
# Data with overlapping indexes
df1 = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
df2 = pd.DataFrame({'A': [4, 5, 6]}, index=[2, 3, 4])  # Index 2 overlaps!

# Default: allows duplicate indexes
result = pd.concat([df1, df2])
print(result)
#    A
# 0  1
# 1  2
# 2  3  # Duplicate!
# 2  4  # Duplicate!
# 3  5
# 4  6

# With verify_integrity: raises error
try:
    result = pd.concat([df1, df2], verify_integrity=True)
except ValueError as e:
    print(f"Error: {e}")
# Error: Indexes have overlapping values: Int64Index([2], dtype='int64')

# Solution: Use ignore_index or handle duplicates
result = pd.concat([df1, df2], ignore_index=True)
print(result)
#    A
# 0  1
# 1  2
# 2  3
# 3  4
# 4  5
# 5  6
```

**When to use these options:**
- **keys**: Tracking data source after concatenation
- **names**: Making MultiIndex levels meaningful
- **verify_integrity**: Ensuring no accidental duplicates in production
- **join='inner'**: Only keeping columns common to all DataFrames

---

## 6. MultiIndex Creation Methods

*Sometimes you need to build a MultiIndex programmatically rather than getting it from groupby or pivot. These methods give you precise control.*

Pandas provides several factory methods for creating MultiIndex objects from scratch.

**Reference:**

- `pd.MultiIndex.from_tuples(tuples, names=['level1', 'level2'])` - From list of tuples
- `pd.MultiIndex.from_product([list1, list2], names=[...])` - Cartesian product
- `pd.MultiIndex.from_arrays([array1, array2], names=[...])` - From parallel arrays
- `pd.MultiIndex.from_frame(df)` - From DataFrame columns

**Example with from_tuples:**

```python
# Create MultiIndex from list of tuples
index_tuples = [
    ('California', 'San Francisco'),
    ('California', 'Los Angeles'),
    ('Texas', 'Houston'),
    ('Texas', 'Dallas')
]

multi_idx = pd.MultiIndex.from_tuples(index_tuples,
                                      names=['state', 'city'])
population = pd.Series([875000, 3980000, 2320000, 1340000],
                       index=multi_idx)
print(population)
# state      city
# California San Francisco     875000
#            Los Angeles      3980000
# Texas      Houston          2320000
#            Dallas           1340000
```

**Example with from_product:**

```python
# Create all combinations of two lists (Cartesian product)
years = [2021, 2022, 2023]
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

multi_idx = pd.MultiIndex.from_product([years, quarters],
                                       names=['year', 'quarter'])
# Creates: (2021, Q1), (2021, Q2), ... (2023, Q4) - all 12 combinations

data = pd.Series(np.random.randint(100, 500, size=12), index=multi_idx)
print(data)
# year quarter
# 2021 Q1        145
#      Q2        389
#      Q3        212
#      Q4        456
# 2022 Q1        278
# ...
```

**Example with from_arrays:**

```python
# Create from parallel arrays (aligned by position)
states = ['CA', 'CA', 'CA', 'TX', 'TX', 'TX']
cities = ['SF', 'LA', 'SD', 'Houston', 'Dallas', 'Austin']
stores = [1, 2, 3, 1, 2, 3]

multi_idx = pd.MultiIndex.from_arrays([states, cities, stores],
                                      names=['state', 'city', 'store_num'])
sales = pd.Series([100, 200, 150, 180, 220, 190], index=multi_idx)
print(sales)
# state  city     store_num
# CA     SF       1            100
#        LA       2            200
#        SD       3            150
# TX     Houston  1            180
#        Dallas   2            220
#        Austin   3            190
```

**When you need manual MultiIndex creation:**
- Building test data with hierarchical structure
- Creating time period indexes (year/month/day combinations)
- Setting up templates for data entry
- Programmatically generating report structures

---

## 7. Stack/Unstack with dropna Parameter

*By default, stack() drops NaN values. The dropna parameter lets you control this behavior - critical when you need to preserve missing data patterns.*

The stack() and unstack() methods have a dropna parameter that controls how missing values are handled.

**Reference:**

- `df.stack(dropna=True)` - Default: drop rows where result is NaN
- `df.stack(dropna=False)` - Keep NaN values in the result
- `df.unstack(fill_value=0)` - Fill missing values with 0 (or other value)
- Useful for maintaining data completeness

**Example:**

```python
# Survey data with missing responses
survey = pd.DataFrame({
    'Q1': [5, 4, np.nan, 3],
    'Q2': [4, np.nan, 5, 4],
    'Q3': [np.nan, 5, 4, np.nan]
}, index=['Alice', 'Bob', 'Charlie', 'Diana'])

print(survey)
#           Q1   Q2   Q3
# Alice    5.0  4.0  NaN
# Bob      4.0  NaN  5.0
# Charlie  NaN  5.0  4.0
# Diana    3.0  4.0  NaN

# Default stack: drops NaN values
stacked_drop = survey.stack()
print(stacked_drop)
# Alice    Q1    5.0
#          Q2    4.0
# Bob      Q1    4.0
#          Q3    5.0
# Charlie  Q2    5.0
#          Q3    4.0
# Diana    Q1    3.0
#          Q2    4.0
# dtype: float64
# Notice: Only 8 values (missing values dropped)

# Keep NaN values
stacked_keep = survey.stack(dropna=False)
print(stacked_keep)
# Alice    Q1    5.0
#          Q2    4.0
#          Q3    NaN  # Kept!
# Bob      Q1    4.0
#          Q2    NaN  # Kept!
#          Q3    5.0
# Charlie  Q1    NaN  # Kept!
#          Q2    5.0
#          Q3    4.0
# Diana    Q1    3.0
#          Q2    4.0
#          Q3    NaN  # Kept!
# dtype: float64
# All 12 values preserved

# Unstack with fill_value
unstacked = stacked_drop.unstack(fill_value=0)
print(unstacked)
#           Q1   Q2   Q3
# Alice    5.0  4.0  0.0  # NaN became 0
# Bob      4.0  0.0  5.0
# Charlie  0.0  5.0  4.0
# Diana    3.0  4.0  0.0
```

**When dropna=False matters:**

```python
# Complete time series with gaps
dates = pd.date_range('2024-01-01', periods=4)
data = pd.DataFrame({
    'Store1': [100, np.nan, 150, 200],
    'Store2': [120, 140, np.nan, 180]
}, index=dates)

# With dropna=True (default): loses temporal structure
stacked_drop = data.stack()
print(len(stacked_drop))  # Only 6 values

# With dropna=False: maintains complete time grid
stacked_keep = data.stack(dropna=False)
print(len(stacked_keep))  # All 8 values (4 dates × 2 stores)

# Critical for time series: you need to know when data is missing
# vs when the observation didn't occur
```

**Real-world use case:**
- Time series: Preserve missing data patterns (gaps vs zeros)
- Survey analysis: Distinguish "no response" from non-applicable questions
- Data quality checks: Count how many NaN values exist
- Machine learning: Some algorithms need explicit missing value markers

---

## 8. Hierarchical Columns from Pivot

*pivot() can create MultiIndex not just in rows, but in columns too. This happens when you don't specify the values parameter or when pivoting multiple value columns.*

When pivoting with multiple value columns or without specifying values, pandas creates hierarchical column headers.

**Reference:**

- `df.pivot(index='row', columns='col')` - Creates MultiIndex columns (all values)
- `df.pivot(index='row', columns='col', values='val')` - Single level columns
- Access: `df['value_name', 'column_name']` or `df['value_name']['column_name']`
- Flatten: `df.columns = ['_'.join(col) for col in df.columns]`

**Example:**

```python
# Long format sales data
sales = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    'product': ['Laptop', 'Mouse', 'Laptop', 'Mouse'],
    'revenue': [1000, 50, 1200, 60],
    'units': [1, 5, 1, 6]
})

print(sales)
#         date product  revenue  units
# 0 2024-01-01  Laptop     1000      1
# 1 2024-01-01   Mouse       50      5
# 2 2024-01-02  Laptop     1200      1
# 3 2024-01-02   Mouse       60      6

# Pivot without specifying values - creates hierarchical columns
wide = sales.pivot(index='date', columns='product')
print(wide)
#            revenue        units
# product     Laptop Mouse Laptop Mouse
# date
# 2024-01-01    1000    50      1     5
# 2024-01-02    1200    60      1     6

# The columns are MultiIndex!
print(wide.columns)
# MultiIndex([('revenue',  'Laptop'),
#             ('revenue',   'Mouse'),
#             (  'units',  'Laptop'),
#             (  'units',   'Mouse')],
#            names=[None, 'product'])

# Access specific column
print(wide['revenue', 'Laptop'])
# date
# 2024-01-01    1000
# 2024-01-02    1200

# Or access top level first
print(wide['revenue'])
# product  Laptop  Mouse
# date
# 2024-01-01    1000     50
# 2024-01-02    1200     60

# Flatten MultiIndex columns to single level
wide.columns = ['_'.join(col) for col in wide.columns]
print(wide)
#            revenue_Laptop  revenue_Mouse  units_Laptop  units_Mouse
# date
# 2024-01-01            1000             50             1            5
# 2024-01-02            1200             60             1            6

# Now normal column access
print(wide['revenue_Laptop'])
```

**More complex example with naming:**

```python
# Pivot table with hierarchical columns
summary = sales.pivot_table(
    values=['revenue', 'units'],
    index='date',
    columns='product',
    aggfunc='sum'
)

# Name the column levels
summary.columns.names = ['metric', 'product']
print(summary)
# metric      revenue        units
# product      Laptop Mouse Laptop Mouse
# date
# 2024-01-01     1000    50      1     5
# 2024-01-02     1200    60      1     6

# Select by level
print(summary.xs('revenue', axis=1, level='metric'))
# product  Laptop  Mouse
# date
# 2024-01-01    1000     50
# 2024-01-02    1200     60

# Swap column levels (like swaplevel for rows)
swapped = summary.swaplevel(axis=1)
print(swapped)
# product  Laptop          Mouse
# metric  revenue units revenue units
# date
# 2024-01-01  1000     1      50     5
# 2024-01-02  1200     1      60     6
```

**When you'll encounter hierarchical columns:**
- Pivot tables with multiple metrics
- Time series with multiple measurements per timestamp
- Cross-tabulations showing multiple statistics
- Financial reports (multiple quarters, multiple metrics)

**Gotcha:** Hierarchical columns can be confusing. Often it's cleaner to either:
1. Flatten them to single-level columns with descriptive names
2. Use .xs() to extract just the metric/dimension you need
3. Restructure the data to long format and avoid hierarchical columns

---

## When to Revisit These Topics

You'll know it's time to come back to these advanced topics when you encounter:

**Advanced MultiIndex Operations:**
- Working with hierarchical business data (Region → Store → Department)
- Multi-level time series (Year → Quarter → Month)
- Need to aggregate at different hierarchical levels
- Performance issues with complex MultiIndex selection

**Merging on Index:**
- Time series joins where datetime is the index
- Dimension tables using index as primary key
- After extensive use of set_index()
- Working with data from databases (often indexed)

**DataFrame.join():**
- Frequently joining multiple DataFrames by index
- Building data pipelines with consistent index-based joins
- Need cleaner syntax than pd.merge() for simple cases

**combine_first():**
- Patching data gaps from multiple sources
- Implementing fallback/default values
- Merging corrected data with original datasets
- Time series with overlapping coverage

**Advanced concat Options:**
- Need to track data provenance (which source?)
- Building complex hierarchical datasets
- Data validation in production (verify_integrity)
- Combining data from multiple systems/files

**MultiIndex Creation Methods:**
- Programmatically generating reports with fixed structure
- Creating test data with hierarchical indexes
- Building time period hierarchies (year/quarter/month)
- Need precise control over MultiIndex structure

**Stack/Unstack with dropna:**
- Time series where gaps matter (NaN ≠ 0)
- Survey data preserving "no response" vs "N/A"
- Data quality analysis (counting missing patterns)
- Maintaining rectangular data structure despite gaps

**Hierarchical Columns from Pivot:**
- Complex pivot tables with multiple metrics
- Financial reports (products × metrics × time periods)
- Need to represent multi-dimensional data in 2D table
- Building sophisticated summary tables

**Bottom Line:** If the basic operations in the main lecture feel limiting, come back here. These advanced topics solve real problems that emerge in complex data wrangling scenarios.

## Patching Missing Data with combine_first()

When you have overlapping data sources, combine_first() fills gaps intelligently - like having a backup copy that fills in the blanks.

**Visual: Data Jigsaw Puzzle**

```
Puzzle Piece 1:        Puzzle Piece 2:        Completed Puzzle:
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ A │ B │ NaN │        │NaN│ B │  C  │        │ A │ B │  C  │
├───┼───┼─────┤        ├───┼───┼─────┤        ├───┼───┼─────┤
│NaN│NaN│  C  │   +    │ D │NaN│ NaN │   =    │ D │ B │  C  │
└───┴───┴─────┘        └───┴───┴─────┘        └───┴───┴─────┘
  Has: A, B, C            Has: B, C, D            Has: A, B, C, D
```

`*combine_first()` is like completing a jigsaw puzzle - each piece fills in the missing parts!*

**Reference:**

- `df1.combine_first(df2)` - Fill missing values in df1 with values from df2
- Works by index alignment - matching index values are combined
- Preserves non-null values from calling DataFrame
- Fills NaN values with values from other DataFrame

**Example:**

```python
# Two data sources with overlapping but incomplete data
sales_q1 = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'sales': [100, np.nan, 150]
})

sales_q2 = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'sales': [120, 200, np.nan]
})

# Combine to get complete picture
complete = sales_q1.combine_first(sales_q2)
display(complete)
#   product  sales
# 0       A  100.0  # Kept original (non-null)
# 1       B  200.0  # Filled from Q2
# 2       C  150.0  # Kept original (non-null)

# Why this matters: You get the best of both datasets!
# Q1 had A and C, Q2 had B - now you have all three

```

**Real-world example:** Combining survey responses from different time periods, or merging partial datasets from different sources.
