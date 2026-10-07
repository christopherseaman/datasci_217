---
notion:
  title_line: "# DLC: Advanced Data Wrangling"
  role: bonus
  status: mapped
  page_id: "3d2d9fdd-1a1a-810a-9c14-db8f8818cb36"
  url: "https://app.notion.com/p/3d2d9fdd1a1a810a9c14db8f8818cb36"
---

# DLC: Advanced Data Wrangling

_These are more advanced or specialized operations from McKinney Chapter 8. They're incredibly powerful but you won't need them daily as a beginner. Come back to these when you encounter specific use cases that require hierarchical data management or specialized joining techniques._

See README.md for the core data wrangling operations; master those first!

# 1. Advanced MultiIndex Operations

A **MultiIndex** identifies rows or columns with several label levels. Changing which level comes first lets the same clinic-by-quarter table support selection by clinic or by quarter without changing its values.

## Swapping and Reordering Index Levels

When you have multiple index levels, you may need to change their order for different analyses.

### Reference Card: Swapping and reordering index levels

- `df.index.names = ['level1', 'level2']`: Name the levels; the labels themselves stay the same
- `df.swaplevel(0, 1)`: Exchange two index levels by position
- `df.swaplevel('level1', 'level2')`: Exchange by name
- `df.sort_index(level=0)`: Sort by specific level
- `df.sort_index(level='level_name')`: Sort by named level
- Combine swaplevel() + sort_index() for reordering

### Code Snippet: Swap and sort index levels

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
# West   Q1             0      1       2
#        Q2             3      4       5
# East   Q1             6      7       8
#        Q2             9     10      11

# Swap levels: Quarter becomes outer, Region becomes inner
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

### When You'll Need This

- Changing perspective on hierarchical data (year→month→day vs day→month→year)
- Preparing data for specific groupby operations
- Making partial selection easier (e.g., all Q1 data across regions)

Sorting the levels prepares a MultiIndex for label-range slices with `.loc`; an unsorted index can reject those slices. Selecting one complete label does not require sorting first.


# 2. Merging on Index

_Sometimes your "key" isn't a column; it's the index itself. This is common with time series or when you've already structured data with meaningful indexes._

Instead of merging on columns, you can merge using the index of one or both DataFrames.

## Index-Based Merge Keys

### Reference Card: `pd.merge()` index options

- `pd.merge(left, right, left_index=True, right_index=True)`: Merge both indexes
- `pd.merge(left, right, left_on='col', right_index=True)`: Column to index
- `pd.merge(left, right, left_index=True, right_on='col')`: Index to column
- `how='inner'/'left'/'right'/'outer'`: Still applies

### Code Snippet: Merge a column key against an index

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
#                 name      city
# customer_id
# C001           Alice   Seattle
# C002             Bob  Portland
# C003         Charlie    Eugene

# Merge: purchases column 'customer_id' to customers index
merged = pd.merge(purchases, customers,
                  left_on='customer_id', right_index=True)
print(merged)
#   customer_id   product  amount   name      city
# 0        C001    Laptop  999.99  Alice   Seattle
# 1        C001     Mouse   25.99  Alice   Seattle
# 2        C002  Keyboard   79.99    Bob  Portland

# C003 (Charlie) and C004 (Monitor) are missing: the default is an inner join
# Use how='left' to keep all purchases
merged_left = pd.merge(purchases, customers,
                       left_on='customer_id', right_index=True, how='left')
print(merged_left)
#   customer_id   product  amount   name      city
# 0        C001    Laptop  999.99  Alice   Seattle
# 1        C001     Mouse   25.99  Alice   Seattle
# 2        C002  Keyboard   79.99    Bob  Portland
# 3        C004   Monitor  299.99    NaN       NaN  # No customer info

# Both DataFrames using index
left_indexed = purchases.set_index('customer_id')
both_index = pd.merge(left_indexed, customers,
                      left_index=True, right_index=True, how='outer')
print(both_index)
```

### When You'll Need This

- Time series with datetime indexes
- Lookup tables where index is the key
- After set_index() operations
- Joining dimension tables to fact tables (data warehouse style)

**Gotcha:** An index used as a merge key is not automatically preserved as the result's index in every merge. Column-key merges generally create a new result index; index-key merges use the participating index labels as keys, but the resulting index structure depends on the join and key choices. Inspect `result.index` or call `reset_index()` when you need a predictable column form.

## DataFrame.join(): Shorthand for Index Merges

`join()` is a shorter way to write an index merge. `df1.join(df2)` is a left join on index labels, so it suits tables that already share a meaningful index, such as dates.

### Reference Card: `join()`

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `df1.join(df2)` | Left join on index (default) | Index-aligned `DataFrame` |
| `df1.join(df2, how='outer')` | Outer join on index | Index-aligned `DataFrame`; missing fields become `NaN` |
| `df1.join(df2, on='key')` | Match df1's key column against df2's index | Joined `DataFrame`, retaining df1's index |

### Code Snippet: Join aligned indexes

```python
# Time series data with dates as index
prices = pd.DataFrame({'price': [100, 101, 102]},
                      index=pd.to_datetime(['2023-01', '2023-02', '2023-03']))
volumes = pd.DataFrame({'volume': [1000, 1100, 1200]},
                       index=pd.to_datetime(['2023-01', '2023-02', '2023-03']))

# Join on index
combined = prices.join(volumes)
print(combined)
#             price  volume
# 2023-01-01    100    1000
# 2023-02-01    101    1100
# 2023-03-01    102    1200
```


# 3. Validating concat with verify_integrity

_Vertical concat keeps each piece's index, so labels can repeat. `verify_integrity=True` turns that into an error._

## Rejecting Repeated Labels

### Reference Card: `verify_integrity`

- `pd.concat([df1, df2], verify_integrity=True)`: Raise `ValueError` if the pieces share an index label on the concatenation axis
- It does not detect duplicate records that carry different labels
- Fix an overlap with `ignore_index=True` or by making the labels unique

### Code Snippet: Validate indexes with verify_integrity

```python
df1 = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
df2 = pd.DataFrame({'A': [4, 5, 6]}, index=[2, 3, 4])  # Index 2 overlaps

try:
    pd.concat([df1, df2], verify_integrity=True)
except ValueError as e:
    print(f"Error: {e}")
# Error: Indexes have overlapping values: Index([2], dtype='int64')

print(pd.concat([df1, df2], ignore_index=True)['A'].tolist())
# [1, 2, 3, 4, 5, 6]
```

# 4. MultiIndex Creation Methods

_Sometimes you need to build a MultiIndex programmatically rather than getting it from groupby or pivot. These methods give you precise control._

Pandas provides several factory methods for creating MultiIndex objects from scratch.

## Building a MultiIndex with Factory Methods

### Reference Card: MultiIndex factory methods

- `pd.MultiIndex.from_tuples(tuples, names=['level1', 'level2'])`: From list of tuples
- `pd.MultiIndex.from_product([list1, list2], names=[...])`: Cartesian product
- `pd.MultiIndex.from_arrays([array1, array2], names=[...])`: From parallel arrays
- `pd.MultiIndex.from_frame(df)`: From DataFrame columns

### Code Snippet: Build from tuples

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
# state       city
# California  San Francisco     875000
#             Los Angeles      3980000
# Texas       Houston          2320000
#             Dallas           1340000
```

### Code Snippet: Build from a Cartesian product

```python
# Create all combinations of two lists (Cartesian product)
years = [2021, 2022, 2023]
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

multi_idx = pd.MultiIndex.from_product([years, quarters],
                                       names=['year', 'quarter'])
# Creates all 12 combinations: (2021, Q1), (2021, Q2), ... (2023, Q4)

rng = np.random.default_rng(42)
data = pd.Series(rng.integers(100, 500, size=12), index=multi_idx)
print(data)
# year  quarter
# 2021  Q1         135
#       Q2         409
#       Q3         361
#       Q4         275
# 2022  Q1         273
# ...
```

### Code Snippet: Build from parallel arrays

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

### When You'll Need Manual MultiIndex Creation

- Building test data with hierarchical structure
- Creating time period indexes (year/month/day combinations)
- Setting up templates for data entry
- Programmatically generating report structures


# 5. Hierarchical Columns from Pivot

_pivot() can create MultiIndex not just in rows, but in columns too. This happens when you don't specify the values parameter or when pivoting multiple value columns._

When pivoting with multiple value columns or without specifying values, pandas creates hierarchical column headers.

## Creating Hierarchical Columns

### Reference Card: Hierarchical pivot columns

- `df.pivot(index='row', columns='col')`: Creates MultiIndex columns (all values)
- `df.pivot(index='row', columns='col', values='val')`: Single level columns
- Access: `df['value_name', 'column_name']` or `df['value_name']['column_name']`
- Flatten: `df.columns = ['_'.join(col) for col in df.columns]`

### Code Snippet: Pivot into hierarchical columns

```python
# Long format sales data
sales = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    'product': ['Laptop', 'Mouse', 'Laptop', 'Mouse'],
    'revenue': [1000, 50, 1200, 60],
    'units': [1, 5, 1, 6]
})

print(sales)
#          date product  revenue  units
# 0  2024-01-01  Laptop     1000      1
# 1  2024-01-01   Mouse       50      5
# 2  2024-01-02  Laptop     1200      1
# 3  2024-01-02   Mouse       60      6

# Pivot without specifying values: creates hierarchical columns
wide = sales.pivot(index='date', columns='product')
print(wide)
#            revenue        units
# product     Laptop Mouse Laptop Mouse
# date
# 2024-01-01    1000    50      1     5
# 2024-01-02    1200    60      1     6

# The columns are MultiIndex!
print(wide.columns)
# MultiIndex([('revenue', 'Laptop'),
#             ('revenue',  'Mouse'),
#             (  'units', 'Laptop'),
#             (  'units',  'Mouse')],
#            names=[None, 'product'])

# Access specific column
print(wide['revenue', 'Laptop'])
# date
# 2024-01-01    1000
# 2024-01-02    1200

# Or access top level first
print(wide['revenue'])
# product     Laptop  Mouse
# date
# 2024-01-01    1000     50
# 2024-01-02    1200     60

# Flatten MultiIndex columns to single level
wide.columns = ['_'.join(col) for col in wide.columns]
print(wide)
#             revenue_Laptop  revenue_Mouse  units_Laptop  units_Mouse
# date
# 2024-01-01            1000             50             1            5
# 2024-01-02            1200             60             1            6

# Now normal column access
print(wide['revenue_Laptop'])
```

### Code Snippet: Name and select column levels

```python
# Long format sales data, same shape as the previous example
sales = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    'product': ['Laptop', 'Mouse', 'Laptop', 'Mouse'],
    'revenue': [1000, 50, 1200, 60],
    'units': [1, 5, 1, 6]
})

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
# metric     revenue        units
# product     Laptop Mouse Laptop Mouse
# date
# 2024-01-01    1000    50      1     5
# 2024-01-02    1200    60      1     6

# Select by level
print(summary.xs('revenue', axis=1, level='metric'))
# product     Laptop  Mouse
# date
# 2024-01-01    1000     50
# 2024-01-02    1200     60

# Swap column levels (like swaplevel for rows)
swapped = summary.swaplevel(axis=1)
print(swapped)
# product     Laptop   Mouse Laptop Mouse
# metric     revenue revenue  units units
# date
# 2024-01-01    1000      50      1     5
# 2024-01-02    1200      60      1     6
```

### When You'll Encounter Hierarchical Columns

- Pivot tables with multiple metrics
- Time series with multiple measurements per timestamp
- Cross-tabulations showing multiple statistics
- Financial reports (multiple quarters, multiple metrics)

**Gotcha:** Hierarchical columns can be confusing. Often it's cleaner to either:
1. Flatten them to single-level columns with descriptive names
2. Use .xs() to extract just the metric/dimension you need
3. Restructure the data to long format and avoid hierarchical columns


# 6. Repeated Pairs and pivot_table()

`pivot()` stops with `ValueError: Index contains duplicate entries, cannot reshape` when an index/column pair holds more than one value. `pivot_table(values=..., index=..., columns=..., aggfunc='mean')` aggregates the repeats into one cell; choosing `mean` or `sum` is an analysis decision, and Lecture 08 teaches it.

# Further Reading

- Hadley Wickham, [Tidy Data](https://www.jstatsoft.org/article/view/v059i10), _Journal of Statistical Software_ 59(10), 2014: the paper behind the name "tidy" for long data with one observation per row and one variable per column.
- Wes McKinney, _Python for Data Analysis_, 3rd edition, Chapter 8 (Data Wrangling: Join, Combine, and Reshape): the source of most topics on this page.
