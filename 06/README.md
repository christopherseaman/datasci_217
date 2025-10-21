Data Wrangling: Join, Combine, and Reshape

See [BONUS.md](BONUS.md) for advanced topics:

- Merging on index with left_index/right_index parameters
- DataFrame.join() method as alternative to merge
- combine_first() for patching missing data
- Advanced concat options (keys, levels, names, verify_integrity)
- Manual MultiIndex creation methods
- Stack/unstack with dropna parameter
- Hierarchical columns from pivot operations

*Fun fact: Data scientists spend 80% of their time wrangling data - merging datasets, reshaping tables, and combining sources. The remaining 20% is spent wondering why their join returned 10x more rows than expected (yes, that's 100% - data wrangling is just that intense!)*

Data wrangling is the art of transforming messy, disconnected datasets into clean, analysis-ready structures. This lecture focuses on the three fundamental operations you'll use every single day: **merging datasets**, **concatenating DataFrames**, and **reshaping data formats**.

**Learning Objectives:**

- Master pd.merge() for database-style joins (inner, outer, left, right)
- Combine multiple DataFrames with pd.concat()
- Transform between wide and long formats with pivot() and melt()
- Manage DataFrame indexes with set_index() and reset_index()
- Recognize and work with basic MultiIndex structures

# Database-Style DataFrame Joins

*Reality check: Merging datasets is the single most common data wrangling task you'll perform. Master pd.merge() and you'll save yourself countless hours of frustration.*

Joining (or merging) DataFrames combines data from multiple sources by linking rows using shared keys. If you've worked with SQL databases, this will feel familiar - pandas implements database-style join operations.

#FIXME: Add diagram showing "Inner vs Outer Join Venn Diagram" - visual representation of how different join types include/exclude rows

## The Basics of pd.merge()

The `pd.merge()` function is your workhorse for combining datasets. At its simplest, it links two DataFrames based on shared column values.

**Reference:**

- `pd.merge(left, right)` - Merge two DataFrames (auto-detects common columns)
- `pd.merge(left, right, on='key')` - Merge on specific column (explicit is better!)
- `pd.merge(left, right, left_on='key1', right_on='key2')` - Different column names
- `pd.merge(left, right, how='inner')` - Join type: inner (default), left, right, outer
- `pd.merge(left, right, on=['col1', 'col2'])` - Merge on multiple columns
- `pd.merge(left, right, suffixes=('_left', '_right'))` - Handle overlapping column names

**Example:**

```python
import pandas as pd

# Customer data
customers = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003', 'C004'],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'city': ['Seattle', 'Portland', 'Seattle', 'Eugene']
})

# Purchase data
purchases = pd.DataFrame({
    'customer_id': ['C001', 'C001', 'C002', 'C005'],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'amount': [999.99, 25.99, 79.99, 299.99]
})

# Basic merge - combines on common column 'customer_id'
merged = pd.merge(customers, purchases)
display(merged)
#   customer_id     name      city  product  amount
# 0        C001    Alice   Seattle   Laptop  999.99
# 1        C001    Alice   Seattle    Mouse   25.99
# 2        C002      Bob  Portland Keyboard   79.99

# Explicit is better - specify the key
merged = pd.merge(customers, purchases, on='customer_id')
display(merged)  # Same result
```

**Why this matters:** Notice that C003 (Charlie) and C004 (Diana) disappeared from the result - they had no matching purchases. C005's monitor purchase also disappeared - no matching customer. This is an **inner join**, the default behavior.

## Join Types: The Four Horsemen of Data Merging

Understanding join types is crucial. Each type answers a different question about your data.

**Reference:**

- `how='inner'` - Only rows with matching keys in BOTH DataFrames (intersection)
- `how='left'` - ALL rows from left DataFrame, matching rows from right (left dominates)
- `how='right'` - ALL rows from right DataFrame, matching rows from left (right dominates)
- `how='outer'` - ALL rows from BOTH DataFrames (union)

**Example:**

```python
# Inner join (default) - only customers with purchases
inner = pd.merge(customers, purchases, on='customer_id', how='inner')
display(inner)
# Result: 3 rows (Alice twice, Bob once) - only matching customers

# Left join - ALL customers, even without purchases
left = pd.merge(customers, purchases, on='customer_id', how='left')
display(left)
#   customer_id     name      city    product   amount
# 0        C001    Alice   Seattle     Laptop   999.99
# 1        C001    Alice   Seattle      Mouse    25.99
# 2        C002      Bob  Portland   Keyboard    79.99
# 3        C003  Charlie   Seattle        NaN      NaN  # No purchase
# 4        C004    Diana    Eugene        NaN      NaN  # No purchase

# Right join - ALL purchases, even without customer info
right = pd.merge(customers, purchases, on='customer_id', how='right')
display(right)
# Result: 4 rows - includes C005's monitor (customer info is NaN)

# Outer join - EVERYTHING (all customers and all purchases)
outer = pd.merge(customers, purchases, on='customer_id', how='outer')
display(outer)
# Result: 5 rows - all customers AND all purchases, NaN where no match
```

**Pro tip:** Most beginners default to inner joins and lose data without realizing it. Use left joins when the left DataFrame is your "master" list (e.g., all customers), right joins for the opposite, and outer joins when you need to see ALL the data from both sides.

**LIVE DEMO!** (Demo 1: Customer Purchase Analysis)
- Merge customer data with purchase records
- Practice different join types (inner, left, right, outer)
- Handle duplicate keys and validation

## Many-to-One and Many-to-Many Merges

Real-world data rarely has perfect one-to-one relationships. Understanding relationship types (how many rows match) is key.

**Reference:**

- **Many-to-one**: Multiple rows in left DataFrame match one row in right (e.g., many purchases per customer)
- **Many-to-many**: Multiple rows in both DataFrames match (creates Cartesian product - beware!)
- Check row counts before and after merge - unexpected growth means many-to-many

**Example:**

```python
# Many-to-one: Multiple purchases per customer
# customers has 1 row per customer_id
# purchases has multiple rows per customer_id
# Result: Each purchase gets customer info attached

many_to_one = pd.merge(customers, purchases, on='customer_id')
display(many_to_one)
# Alice appears twice (2 purchases), Bob once (1 purchase)

# Many-to-many: DANGER ZONE
categories = pd.DataFrame({
    'product': ['Laptop', 'Mouse', 'Laptop', 'Mouse'],
    'category': ['Electronics', 'Accessories', 'Computing', 'Peripherals']
})

# Multiple rows for 'Laptop' in purchases, multiple rows for 'Laptop' in categories
many_to_many = pd.merge(purchases, categories, on='product')
display(many_to_many)
# Every combination of matching rows! Alice's Laptop matches BOTH Electronics AND Computing
# This creates: 1 purchase × 2 categories = 2 rows per purchase
# With 100 purchases × 100 categories? Could become 10,000 rows!
```

**Common pitfall:** If you expect 1,000 rows and get 10,000 after a merge, you probably have an accidental many-to-many join (where keys repeat in both DataFrames). Check for duplicate keys!

## Merging on Multiple Columns

Sometimes a single column isn't enough to uniquely identify matches - you need to match on multiple columns together (like matching on BOTH store_id AND date).

**Reference:**

- `on=['col1', 'col2', 'col3']` - Match on multiple columns simultaneously
- All specified columns must match for rows to merge
- Useful for hierarchical data (year + month, store + date, etc.)

**Example:**

```python
# Sales data by store and date
sales_q1 = pd.DataFrame({
    'store_id': ['S01', 'S01', 'S02', 'S02'],
    'quarter': ['Q1', 'Q1', 'Q1', 'Q1'],
    'sales': [50000, 55000, 42000, 48000]
})

# Target data by store and quarter
targets = pd.DataFrame({
    'store_id': ['S01', 'S02', 'S01', 'S02'],
    'quarter': ['Q1', 'Q1', 'Q2', 'Q2'],
    'target': [52000, 45000, 58000, 50000]
})

# Merge on BOTH store_id AND quarter
merged = pd.merge(sales_q1, targets, on=['store_id', 'quarter'])
display(merged)
#   store_id quarter  sales  target
# 0      S01      Q1  50000   52000
# 1      S01      Q1  55000   52000  # Same store/quarter appears twice
# 2      S02      Q1  42000   45000
# 3      S02      Q1  48000   45000
```

**Why this matters:** If you merged only on 'store_id', you'd match Q1 sales with Q2 targets - wrong! The composite key ensures correct matching.

## Handling Overlapping Column Names

When both DataFrames have columns with the same name (besides the merge key), pandas adds suffixes to distinguish them.

**Reference:**

- Default suffixes: `_x` (left DataFrame) and `_y` (right DataFrame)
- `suffixes=('_left', '_right')` - Custom suffixes for clarity
- `suffixes=('_old', '_new')` - Useful for comparing versions

**Example:**

```python
# Both DataFrames have 'total' column
sales = pd.DataFrame({
    'product_id': ['P001', 'P002', 'P003'],
    'total': [100, 200, 150]  # Sales total
})

inventory = pd.DataFrame({
    'product_id': ['P001', 'P002', 'P003'],
    'total': [50, 75, 30]  # Inventory total
})

# Default suffixes (_x and _y)
merged = pd.merge(sales, inventory, on='product_id')
display(merged)
#   product_id  total_x  total_y
# 0       P001      100       50
# 1       P002      200       75
# 2       P003      150       30

# Custom suffixes for clarity
merged = pd.merge(sales, inventory, on='product_id',
                  suffixes=('_sales', '_inventory'))
display(merged)
#   product_id  total_sales  total_inventory
# 0       P001          100               50
# 1       P002          200               75
# 2       P003          150               30
```

**Pro tip:** Always use descriptive suffixes! `_sales` and `_inventory` are much clearer than `_x` and `_y`.

# Concatenating DataFrames Along an Axis

*Think of concatenation as stacking LEGO bricks - you can stack them vertically (add more rows) or horizontally (add more columns). Just make sure they fit together!*

Concatenation combines DataFrames by stacking them together, either adding rows (vertical) or columns (horizontal). Unlike merging, concatenation doesn't use keys - it simply glues DataFrames together.

#FIXME: Add diagram showing "Vertical vs Horizontal Concatenation" - visual showing axis=0 (rows) vs axis=1 (columns)

## Vertical Concatenation: Adding More Rows

The most common use case - combining datasets with the same columns.

**Reference:**

- `pd.concat([df1, df2, df3])` - Stack DataFrames vertically (default axis=0)
- `pd.concat([df1, df2], axis=0)` - Explicit vertical stacking
- `pd.concat([df1, df2], ignore_index=True)` - Reset index to 0, 1, 2, ...
- `pd.concat([df1, df2], join='outer')` - Union of columns (default)
- `pd.concat([df1, df2], join='inner')` - Intersection of columns only

**Example:**

```python
# Sales data from different months
jan_sales = pd.DataFrame({
    'product': ['Laptop', 'Mouse', 'Keyboard'],
    'quantity': [5, 20, 15],
    'month': ['Jan', 'Jan', 'Jan']
})

feb_sales = pd.DataFrame({
    'product': ['Laptop', 'Monitor', 'Tablet'],
    'quantity': [8, 3, 12],
    'month': ['Feb', 'Feb', 'Feb']
})

# Stack them vertically - combines rows
combined = pd.concat([jan_sales, feb_sales])
display(combined)
#    product  quantity month
# 0   Laptop         5   Jan
# 1    Mouse        20   Jan
# 2 Keyboard        15   Jan
# 0   Laptop         8   Feb  # Index repeats! (0, 1, 2 again)
# 1  Monitor         3   Feb
# 2   Tablet        12   Feb

# Clean indexes with ignore_index=True
combined = pd.concat([jan_sales, feb_sales], ignore_index=True)
display(combined)
#    product  quantity month
# 0   Laptop         5   Jan
# 1    Mouse        20   Jan
# 2 Keyboard        15   Jan
# 3   Laptop         8   Feb  # Clean sequential index
# 4  Monitor         3   Feb
# 5   Tablet        12   Feb
```

**When to use concat vs merge:**
- Use **concat** when stacking similar datasets (same columns, different rows)
- Use **merge** when joining related datasets (shared keys, different information)

## Horizontal Concatenation: Adding More Columns

Less common but useful for adding related information side-by-side.

**Reference:**

- `pd.concat([df1, df2], axis=1)` - Stack DataFrames horizontally
- Indexes are aligned - matching index values are joined
- Missing indexes result in NaN values
- Use when adding new features from separate sources

**Example:**

```python
# Student grades
grades = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'grade': [95, 88, 92]
}, index=[0, 1, 2])

# Student attendance (different students!)
attendance = pd.DataFrame({
    'days_present': [18, 20, 19],
    'days_total': [20, 20, 20]
}, index=[1, 2, 3])  # Different index!

# Horizontal concatenation
combined = pd.concat([grades, attendance], axis=1)
display(combined)
#       name  grade  days_present  days_total
# 0    Alice   95.0           NaN         NaN  # No attendance data
# 1      Bob   88.0          18.0        20.0  # Match!
# 2  Charlie   92.0          20.0        20.0  # Match!
# 3      NaN    NaN          19.0        20.0  # No grade data
```

**Common pitfall:** Horizontal concat uses indexes for alignment. If your indexes don't match, you'll get lots of NaN values. Usually you want to merge instead!

**Visual guide - Index alignment in horizontal concat:**
```
DataFrame 1:              DataFrame 2:              Result (axis=1):
Index | Data              Index | Data              Index | Data1 | Data2
  0   | A                   1   | X                   0   |  A    | NaN
  1   | B                   2   | Y                   1   |  B    |  X
  2   | C                   3   | Z                   2   |  C    |  Y
                                                       3   | NaN   |  Z

Only matching index values are aligned → NaN where indexes don't match!
```

## Handling Different Columns with join Parameter

When DataFrames don't have identical columns, concat needs to know what to do.

**Reference:**

- `join='outer'` (default) - Keep ALL columns from both DataFrames (union)
- `join='inner'` - Keep only COMMON columns (intersection)

**Example:**

```python
# Different columns in each DataFrame
df1 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

df2 = pd.DataFrame({
    'B': [7, 8, 9],
    'C': [10, 11, 12]
})

# Outer join (default) - keeps all columns
outer = pd.concat([df1, df2], join='outer')
display(outer)
#      A  B     C
# 0  1.0  4   NaN  # From df1
# 1  2.0  5   NaN
# 2  3.0  6   NaN
# 0  NaN  7  10.0  # From df2
# 1  NaN  8  11.0
# 2  NaN  9  12.0

# Inner join - keeps only column B (common to both)
inner = pd.concat([df1, df2], join='inner')
display(inner)
#    B
# 0  4
# 1  5
# 2  6
# 0  7
# 1  8
# 2  9
```

# Reshaping: Wide vs Long Format

*Fun fact: 90% of data reshaping confusion comes from not understanding which format you have and which format you need. Once you know that, the solution is usually obvious!*

Data can be organized in two fundamental formats: **wide** (one row per entity, many columns) and **long** (multiple rows per entity, fewer columns). Different analyses and visualizations require different formats.

#FIXME: Add table comparing "Wide vs Long Format Examples" - side-by-side comparison showing same data in both formats

## Understanding Wide Format

Wide format has one row per entity and separate columns for each variable/time period.

**Example:**

```python
# Wide format: Student test scores
wide_data = pd.DataFrame({
    'student': ['Alice', 'Bob', 'Charlie'],
    'math': [95, 88, 92],
    'english': [90, 85, 94],
    'science': [92, 90, 89]
})
display(wide_data)
#    student  math  english  science
# 0    Alice    95       90       92
# 1      Bob    88       85       90
# 2  Charlie    92       94       89

# Wide format is good for:
# - Reading (humans prefer wide tables)
# - Pivot tables and cross-tabulations
# - Spreadsheet-style analysis
```

## Understanding Long Format

Long format has multiple rows per entity, with a variable column indicating what each value represents.

**Example:**

```python
# Long format: Same data, different structure
long_data = pd.DataFrame({
    'student': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Bob',
                'Charlie', 'Charlie', 'Charlie'],
    'subject': ['math', 'english', 'science', 'math', 'english', 'science',
                'math', 'english', 'science'],
    'score': [95, 90, 92, 88, 85, 90, 92, 94, 89]
})
display(long_data)
#    student  subject  score
# 0    Alice     math     95
# 1    Alice  english     90
# 2    Alice  science     92
# 3      Bob     math     88
# 4      Bob  english     85
# 5      Bob  science     90
# 6  Charlie     math     92
# 7  Charlie  english     94
# 8  Charlie  science     89

# Long format is good for:
# - Groupby operations (df.groupby('subject').mean())
# - Plotting with seaborn/plotly (they prefer long format)
# - Statistical modeling (most models expect long format)
```

## Pivoting Long to Wide with pivot()

The `pivot()` method converts long format to wide format - perfect for creating summary tables.

**Reference:**

- `df.pivot(index='row_labels', columns='col_labels', values='data')` - Basic pivot
- `index` - Column to use for row labels
- `columns` - Column to use for column labels
- `values` - Column containing the data values
- **Critical:** Works only when index/columns combinations are unique!

**Example:**

```python
# Convert long format to wide format
wide = long_data.pivot(index='student', columns='subject', values='score')
display(wide)
# subject  english  math  science
# student
# Alice         90    95       92
# Bob           85    88       90
# Charlie       94    92       89

# Why this matters: Now you can easily compare subjects across students!
# Wide format is easier to read for humans

# Pivot makes the column names the new column headers
# And index becomes the row labels
# Values fill the cells
```

**Common error:** If your index/columns combinations aren't unique, pivot() will fail. Use `pivot_table()` instead (covered in BONUS.md).

## Melting Wide to Long with melt()

The `melt()` function converts wide format to long format - essential for analysis and plotting.

**Reference:**

- `pd.melt(df, id_vars=['id_col'], value_vars=['col1', 'col2'])` - Basic melt
- `id_vars` - Columns to keep as identifier variables
- `value_vars` - Columns to unpivot (if None, uses all columns except id_vars)
- `var_name` - Name for the new 'variable' column (default: 'variable')
- `value_name` - Name for the new 'value' column (default: 'value')

**Example:**

```python
# Convert wide format to long format
long = pd.melt(wide_data,
               id_vars=['student'],
               value_vars=['math', 'english', 'science'],
               var_name='subject',
               value_name='score')
display(long)
#    student  subject  score
# 0    Alice     math     95
# 1      Bob     math     88
# 2  Charlie     math     92
# 3    Alice  english     90
# 4      Bob  english     85
# 5  Charlie  english     94
# 6    Alice  science     92
# 7      Bob  science     90
# 8  Charlie  science     89

# Now you can easily do:
long.groupby('subject')['score'].mean()
# subject
# english    89.67
# math       91.67
# science    90.33
```

**Real-world example:** Survey data often comes wide (Q1, Q2, Q3 columns) but needs to be long for analysis.

**Visual guide - Wide to Long to Wide workflow:**
```
WIDE FORMAT                           LONG FORMAT
student | math | english | science    student | subject | score
Alice   |  95  |   90    |   92   →   Alice   | math    |  95
Bob     |  88  |   85    |   90       Alice   | english |  90
                                      Alice   | science |  92
      melt() ────────────────→        Bob     | math    |  88
      ←────────────── pivot()         Bob     | english |  85
                                      Bob     | science |  90

Wide: Easy to read            Long: Easy to analyze with groupby()
      Spreadsheet style              Ready for plotting (seaborn, plotly)
```

**LIVE DEMO!** (Demo 2: Survey Data Reshaping)
- Convert wide survey data (Q1, Q2, Q3 columns) to long format for analysis
- Use groupby on long data to calculate summary statistics
- Pivot back to wide for reporting

# Working with DataFrame Indexes

*Pro tip: Understanding when to move columns to the index (and back) is like understanding when to put your keys in your pocket vs. your hand - it's all about what you need to access quickly!*

The index is special in pandas - it's the "name" of each row. Moving columns to/from the index is a common operation that makes certain operations easier.

## set_index(): Moving Columns to Index

Converting columns to index labels makes certain operations faster and more intuitive.

**Reference:**

- `df.set_index('column')` - Make column the new index
- `df.set_index(['col1', 'col2'])` - Create MultiIndex from multiple columns
- `drop=False` - Keep the column in the DataFrame (default is True, removes it)
- `inplace=True` - Modify DataFrame in place (default is False, returns new DataFrame)

**Example:**

```python
# Employee data
employees = pd.DataFrame({
    'emp_id': ['E001', 'E002', 'E003'],
    'name': ['Alice', 'Bob', 'Charlie'],
    'department': ['Engineering', 'Sales', 'Engineering'],
    'salary': [95000, 75000, 88000]
})
display(employees)
#   emp_id     name   department  salary
# 0   E001    Alice  Engineering   95000
# 1   E002      Bob        Sales   75000
# 2   E003  Charlie  Engineering   88000

# Make emp_id the index
indexed = employees.set_index('emp_id')
display(indexed)
#           name   department  salary
# emp_id
# E001     Alice  Engineering   95000
# E002       Bob        Sales   75000
# E003   Charlie  Engineering   88000

# Now you can access by emp_id directly
display(indexed.loc['E002'])  # Bob's record
# name              Bob
# department      Sales
# salary          75000
```

**Why this matters:**
- Makes .loc[] selection more meaningful (`indexed.loc['E002']` vs `employees[employees['emp_id'] == 'E002']`)
- Required for some operations like merging on index
- Common for time series (dates as index)

## reset_index(): Moving Index to Columns

The opposite operation - converts index back to a regular column.

**Reference:**

- `df.reset_index()` - Move index to column(s)
- `drop=True` - Discard the index instead of converting to column
- `inplace=True` - Modify DataFrame in place

**Example:**

```python
# Move index back to a column
reset = indexed.reset_index()
display(reset)
#   emp_id     name   department  salary
# 0   E001    Alice  Engineering   95000
# 1   E002      Bob        Sales   75000
# 2   E003  Charlie  Engineering   88000

# Back to original structure with default numeric index

# Discard index instead of converting
indexed.reset_index(drop=True)
#       name   department  salary
# 0    Alice  Engineering   95000
# 1      Bob        Sales   75000
# 2  Charlie  Engineering   88000
```

**Common use case:** After a groupby operation, you often want to reset_index() to make the grouping columns regular columns again.

**LIVE DEMO!** (Demo 3: Time Series Concatenation)
- Combine quarterly data files into single dataset
- Use set_index() to create datetime index
- Handle index management during concatenation
- Practice ignore_index vs preserving indexes

# Understanding MultiIndex (Hierarchical Indexing)

*Warning: MultiIndex looks scary at first, but it's actually just nested row labels. Think of it like a file system - folders within folders. Once you get that mental model, it's much less intimidating!*

MultiIndex (hierarchical indexing) appears when you have multiple levels of row labels. It's common in real-world data analysis, especially after groupby operations or pivot tables.

## When You'll Encounter MultiIndex

You don't usually create MultiIndex manually - it shows up as a result of other operations.

**Reference:**

- `df.groupby(['col1', 'col2']).sum()` - Creates MultiIndex automatically
- `df.pivot_table()` - Often creates MultiIndex
- `df.set_index(['col1', 'col2'])` - Explicit MultiIndex creation
- `pd.concat([df1, df2], keys=['source1', 'source2'])` - Adds outer level

**Example:**

```python
# Sales data
sales = pd.DataFrame({
    'region': ['West', 'West', 'East', 'East', 'West', 'East'],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q1', 'Q2'],
    'sales': [100, 150, 120, 180, 110, 190]
})

# Groupby creates MultiIndex automatically
summary = sales.groupby(['region', 'quarter'])['sales'].sum()
display(summary)
# region  quarter
# East    Q1         120  # MultiIndex! Two levels: region and quarter
#         Q2         370  # (180 + 190)
# West    Q1         210  # (100 + 110)
#         Q2         150

# Check the index
display(summary.index)
# MultiIndex([('East', 'Q1'),
#             ('East', 'Q2'),
#             ('West', 'Q1'),
#             ('West', 'Q2')],
#            names=['region', 'quarter'])
```

**What's happening:** The index now has TWO levels - region (outer) and quarter (inner). This is a MultiIndex.

## Basic MultiIndex Selection

Selecting from MultiIndex uses partial indexing - you can select by outer level only.

**Reference:**

- `df.loc['outer_label']` - Select all rows with that outer level value
- `df.loc[('outer', 'inner')]` - Select specific combination (tuple)
- `df.loc['outer_label', 'inner_label']` - Alternative syntax
- `.xs('label', level='level_name')` - Cross-section selection

**Example:**

```python
# Select all East region data
display(summary.loc['East'])
# quarter
# Q1    120
# Q2    370
# Name: sales, dtype: int64

# Select specific combination (East, Q1)
display(summary.loc[('East', 'Q1')])  # 120

# Alternative syntax
display(summary.loc['East', 'Q1'])  # 120
```

**Common pattern:** After groupby with MultiIndex, use `.reset_index()` to convert back to regular columns.

```python
# Convert MultiIndex back to regular columns
flattened = summary.reset_index()
display(flattened)
#   region quarter  sales
# 0   East      Q1    120
# 1   East      Q2    370
# 2   West      Q1    210
# 3   West      Q2    150

# Now easier to work with for most people
```

## The "Escape Hatch" - reset_index()

*Pro tip: If MultiIndex confuses you, just use .reset_index() to bail out! It converts the hierarchical index back to regular columns, and suddenly everything makes sense again.*

**Reference:**

- `df.reset_index()` - Converts ALL index levels to columns
- `df.reset_index(level=0)` - Convert only outermost level
- `df.reset_index(level='level_name')` - Convert specific named level

**Example:**

```python
# MultiIndex from pivot
pivoted = sales.pivot_table(values='sales',
                             index='region',
                             columns='quarter',
                             aggfunc='sum')
display(pivoted)
# quarter    Q1   Q2
# region
# East      120  370
# West      210  150

# Has MultiIndex columns - confusing for beginners!
# Reset to make it simpler
simple = pivoted.reset_index()
display(simple)
# quarter region   Q1   Q2
# 0         East  120  370
# 1         West  210  150

# Much easier to understand!
```

# Practical Workflow Example

Let's combine everything we've learned into a real-world scenario.

**Example:**

```python
# Three separate data sources
customers = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003'],
    'name': ['Alice', 'Bob', 'Charlie'],
    'city': ['Seattle', 'Portland', 'Eugene']
})

purchases = pd.DataFrame({
    'purchase_id': ['P001', 'P002', 'P003', 'P004'],
    'customer_id': ['C001', 'C001', 'C002', 'C003'],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'amount': [999.99, 25.99, 79.99, 299.99],
    'quarter': ['Q1', 'Q1', 'Q2', 'Q2']
})

products = pd.DataFrame({
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics']
})

# Step 1: Merge customers with purchases (left join to keep all customers)
customer_purchases = pd.merge(customers, purchases, on='customer_id', how='left')
display(customer_purchases.head())

# Step 2: Add product categories
full_data = pd.merge(customer_purchases, products, on='product', how='left')
display(full_data.head())

# Step 3: Reshape for analysis - sales by category and quarter
summary = full_data.groupby(['category', 'quarter'])['amount'].sum()
display(summary)
# MultiIndex result

# Step 4: Convert to wide format for reporting
report = summary.reset_index().pivot(index='category',
                                     columns='quarter',
                                     values='amount')
display(report)
# quarter    Accessories  Electronics
# category
# Accessories       79.99          NaN
# Electronics         NaN      1299.98

# Clean presentation!
```

**Why this matters:** This workflow demonstrates the full cycle:
1. **Merge** - Combine related datasets
2. **Group** - Aggregate by meaningful categories
3. **Reshape** - Present data in readable format

# Key Takeaways

1. **pd.merge()** is your daily workhorse - master the four join types (inner, left, right, outer)
2. Use **pd.concat()** for stacking similar datasets, not joining by keys
3. **Wide format** (pivot) is readable; **long format** (melt) is analyzable
4. **set_index()** and **reset_index()** move columns to/from the index
5. **MultiIndex** appears naturally from groupby/pivot - use reset_index() if confused
6. Always check row counts before and after joins to catch unexpected behavior
7. Explicit is better than implicit - specify join keys and join types

You now have the tools to wrangle messy, disconnected data into clean, analysis-ready DataFrames. Practice with real datasets - these operations become second nature with repetition!

Next class: Visualization basics - turning your wrangled data into insights!
