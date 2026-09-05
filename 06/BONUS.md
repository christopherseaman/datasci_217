---
notion:
  role: bonus
  status: unmapped
  page_id: null
  url: null
  target:
    role: dlc_subpage
    parent_page_id: "293d9fdd-1a1a-801c-bef2-e6140976408c"
    anchor: top
  note: "Notion links a generic BONUS.md reference, but no separate matching bonus page was found."
---

Data Wrangling: Advanced Topics

*These are more advanced or specialized operations from McKinney Chapter 8. They're incredibly powerful but you won't need them daily as a beginner. Come back to these when you encounter specific use cases that require hierarchical data management or specialized joining techniques.*

See [README.md](README.md) for core data wrangling operations - master those first!

## Advanced Topics Covered

1. **Advanced MultiIndex Operations** - Deep dive into hierarchical indexing with swaplevel(), level-specific sorting, and summary statistics by level
2. **Merging on Index** - Join DataFrames using index values instead of columns
3. **Advanced concat Options** - Using keys, levels, names, and verify_integrity for complex concatenations
4. **MultiIndex Creation Methods** - Programmatically build hierarchical indexes with from_tuples(), from_product(), from_arrays()
5. **Stack/Unstack with dropna Parameter** - Control how missing data is handled during reshaping
6. **Hierarchical Columns from Pivot** - Create and work with MultiIndex in column headers

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

# Sum across states (transpose so color is a row-index level while grouping)
by_color = frame.T.groupby(level='color').sum().T
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

**Gotcha:** An index used as a merge key is not automatically preserved as the
result's index in every merge. Column-key merges generally create a new result
index; index-key merges use the participating index labels as keys, but the
resulting index structure depends on the join and key choices. Inspect
`result.index` or call `reset_index()` when you need a predictable column form.

---

## 3. Advanced concat Options

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
# Error: Indexes have overlapping values: Index([2], dtype='int64')

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

## 4. MultiIndex Creation Methods

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

## 5. Stack/Unstack and Missing Values in pandas 3

In pandas 3, `stack()` uses the new implementation and preserves missing combinations. The former `dropna=` argument is no longer accepted. Remove missing values explicitly after stacking when that is the intended analysis.

**Reference:**

- `df.stack()` - Move columns into an index level while preserving missing combinations
- `df.stack().dropna()` - Keep only observed values after reshaping
- `series.unstack(fill_value=0)` - Rebuild a table and fill combinations absent from the Series index
- Preserving a missing marker is different from replacing it with zero

**Example:**

```python
# Survey data with missing responses
survey = pd.DataFrame({
    'Q1': [5, 4, np.nan, 3],
    'Q2': [4, np.nan, 5, 4],
    'Q3': [np.nan, 5, 4, np.nan]
}, index=['Alice', 'Bob', 'Charlie', 'Diana'])

# pandas 3 preserves all 12 respondent-question combinations.
stacked_all = survey.stack()
print(len(stacked_all))       # 12
print(stacked_all.isna().sum())  # 4 missing responses

# Drop missing responses only when the question calls for observed values.
stacked_observed = stacked_all.dropna()
print(len(stacked_observed))  # 8 observed responses

# Filling with zero is a separate substantive decision.
zero_filled = stacked_observed.unstack(fill_value=0)
print(zero_filled)
#           Q1   Q2   Q3
# Alice    5.0  4.0  0.0
# Bob      4.0  0.0  5.0
# Charlie  0.0  5.0  4.0
# Diana    3.0  4.0  0.0
```

For a time-indexed table, the same distinction applies:

```python
dates = pd.date_range('2024-01-01', periods=4)
data = pd.DataFrame({
    'Store1': [100, np.nan, 150, 200],
    'Store2': [120, 140, np.nan, 180]
}, index=dates)

stacked_all = data.stack()
print(len(stacked_all))            # 8 time-store combinations
print(len(stacked_all.dropna()))   # 6 observed values
```

Keeping the full result lets a later analysis distinguish a recorded missing value from a combination removed from the table.

---

## 6. Hierarchical Columns from Pivot

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
