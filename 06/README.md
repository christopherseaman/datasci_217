06) Data Wrangling: Join, Combine, and Reshape

**Assignment 6:** [assignment instructions](assignment/README.md)

[LIVE DEMO!](demo/DEMO_GUIDE.md)

See [BONUS.md](BONUS.md) for advanced topics:

- Merging on index with left_index/right_index parameters
- Advanced concat options (keys, levels, names, verify_integrity)
- Manual MultiIndex creation methods
- Advanced stack/unstack options and missing combinations
- Hierarchical columns from pivot operations

*Fun fact: The word “wrangling” comes from the Old English “wranglian” meaning “to dispute or argue.” This is surprisingly accurate - data wrangling is basically arguing with your data until it finally agrees to cooperate.*


Join keys are the table's name tags: if two rows share a tag, pandas brings
their columns together. A good key makes matching boring; a bad key turns the
merge into an enthusiastic photocopier.

Data wrangling is the art of transforming messy, disconnected datasets into clean, analysis-ready structures. This lecture focuses on the three fundamental operations you’ll use every single day: **merging datasets**, **concatenating DataFrames**, and **reshaping data formats**.

**Learning Objectives:**

- Master pd.merge() for database-style joins (inner, outer, left, right)
- Combine multiple DataFrames with pd.concat()
- Transform between wide and long formats with pivot() and melt()
- Manage DataFrame indexes with set_index() and reset_index()
- Recognize and work with basic MultiIndex structures

# Database-Style DataFrame Joins

*Reality check: Merging datasets is the single most common data wrangling task you’ll perform. Master pd.merge() and you’ll save yourself countless hours of frustration.*

Joining (or merging) DataFrames combines data from multiple sources by linking rows using shared keys. If you’ve worked with SQL databases, this will feel familiar - pandas implements database-style join operations.

**Important difference from SQL:** pandas matches null key values with other
null key values. SQL usually treats `NULL = NULL` as unknown, so SQL joins do
not match two null keys. If missing keys should never match, filter them or
replace missing keys on the left and right with **different** sentinels that
cannot occur in the data. Never use one shared sentinel—it would recreate the
same match. Document whichever policy you choose.

**Visual Guide - Join Types:**

```
Table A: customers          Table B: purchases
┌─────────────┬─────────┐   ┌─────────────┬─────────┐
│ customer_id │  name   │   │ customer_id │ amount  │
├─────────────┼─────────┤   ├─────────────┼─────────┤
│     1       │  Alice  │   │     1       │   $50   │
│     2       │   Bob   │   │     2       │   $30   │
│     3       │ Charlie │   │     4       │   $25   │
└─────────────┴─────────┘   └─────────────┴─────────┘

INNER JOIN (how='inner')     LEFT JOIN (how='left')
┌─────────────┬─────────┬─────────┐  ┌─────────────┬─────────┬─────────┐
│ customer_id │  name   │ amount  │  │ customer_id │  name   │ amount  │
├─────────────┼─────────┼─────────┤  ├─────────────┼─────────┼─────────┤
│     1       │  Alice  │   $50   │  │     1       │  Alice  │   $50   │
│     2       │   Bob   │   $30   │  │     2       │   Bob   │   $30   │
└─────────────┴─────────┴─────────┘  │     3       │Charlie  │   NaN   │
 (Only matching rows)                └─────────────┴─────────┴─────────┘
                                      (All from A, missing from B = NaN)

RIGHT JOIN (how='right')     OUTER JOIN (how='outer')
┌─────────────┬─────────┬─────────┐  ┌─────────────┬─────────┬─────────┐
│ customer_id │  name   │ amount  │  │ customer_id │  name   │ amount  │
├─────────────┼─────────┼─────────┤  ├─────────────┼─────────┼─────────┤
│     1       │  Alice  │   $50   │  │     1       │  Alice  │   $50   │
│     2       │   Bob   │   $30   │  │     2       │   Bob   │   $30   │
│     4       │   NaN   │   $25   │  │     3       │Charlie  │   NaN   │
└─────────────┴─────────┴─────────┘  │     4       │   NaN   │   $25   │
 (All from B, missing from A = NaN)  └─────────────┴─────────┴─────────┘
                                      (Everything from both tables)

```

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

# Basic merge: pandas detects the common 'customer_id' column
merged = pd.merge(customers, purchases)
display(merged)
#   customer_id   name      city   product  amount
# 0        C001  Alice   Seattle    Laptop  999.99
# 1        C001  Alice   Seattle     Mouse   25.99
# 2        C002    Bob  Portland  Keyboard   79.99

# Explicit is better: specify the key
merged = pd.merge(customers, purchases, on='customer_id')
display(merged)  # Same result
```

**Why this matters:** Inner join only keeps matching records - customers without purchases are dropped.

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
# Result: 6 rows - matching key C001 produces two rows; unmatched values get NaN

```

**Pro tip:** Most beginners default to inner joins and lose data without realizing it. Use left joins when the left DataFrame is your “master” list (e.g., all customers), right joins for the opposite, and outer joins when you need to see ALL the data from both sides.

**Why this matters:** Wrong join type = lost data. Use left join to keep all customers.


## Merge Cardinality and Integrity Checks

Merge cardinality describes whether each key is unique or repeated on the left and right. State the relationship you expect before merging, then ask pandas to validate it.

**Reference:**

- **One-to-one**: Keys are unique in both DataFrames
- **One-to-many**: Left keys are unique; right keys may repeat
- **Many-to-one**: Left keys may repeat; right keys are unique
- **Many-to-many**: Keys may repeat on both sides; each matching key produces every pair of rows
- `validate='one_to_one'`, `'one_to_many'`, or `'many_to_one'` — raise an error if the expected uniqueness is violated
- `indicator=True` — add a `_merge` column showing `left_only`, `right_only`, or `both`

**Example:**

```python
# One-to-one: each customer_id occurs at most once on both sides
customer_status = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003', 'C004'],
    'status': ['gold', 'silver', 'silver', 'bronze']
})
one_to_one = pd.merge(
    customers, customer_status,
    on='customer_id', validate='one_to_one'
)

# One-to-many: customers is unique; purchases repeats C001
one_to_many = pd.merge(
    customers, purchases,
    on='customer_id', validate='one_to_many'
)

# Reversing the inputs describes the same data as many-to-one
many_to_one = pd.merge(
    purchases, customers,
    on='customer_id', validate='many_to_one'
)

# Many-to-many: 'Laptop' repeats twice on each side
order_lines = pd.DataFrame({
    'order_id': ['O1', 'O2', 'O3'],
    'product': ['Laptop', 'Laptop', 'Mouse']
})
product_tags = pd.DataFrame({
    'product': ['Laptop', 'Laptop', 'Mouse'],
    'tag': ['portable', 'computing', 'accessory']
})
many_to_many = pd.merge(
    order_lines, product_tags,
    on='product', validate='many_to_many'
)
display(many_to_many)
# The Laptop key contributes 2 × 2 = 4 rows; Mouse contributes 1 row.

# Audit which keys matched while preserving both sides
audit = pd.merge(
    customers, purchases,
    on='customer_id', how='outer',
    validate='one_to_many', indicator=True
)
display(audit['_merge'].value_counts())
# both: 3 rows, left_only: 2 rows, right_only: 1 row

```

Row growth alone does not prove a many-to-many merge: an intended one-to-many merge also adds rows. Inspect key uniqueness and use `validate=` to make the expected relationship executable. (`validate='many_to_many'` permits repeats on both sides, so it documents rather than constrains uniqueness.)

# LIVE DEMO!

(Demo 1: Customer Purchase Analysis)

## Merging on Multiple Columns

Sometimes a single column isn’t enough to uniquely identify matches - you need to match on multiple columns together (like matching on BOTH store_id AND date).

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

**Why this matters:** Composite keys prevent mismatched data (Q1 sales with Q2 targets).

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

## Alternative Data Combination Methods

### Prerequisite: Index Labels and Alignment

An index supplies row labels. When pandas aligns objects, equal labels identify corresponding rows; label order and coverage do not have to match. Unmatched labels are retained or discarded according to the operation and its alignment mode, and retained gaps become `NaN`. Before using index-based combination, confirm that the labels mean the same thing in every object; a shared default `RangeIndex` does not establish shared identity.

### DataFrame.join(): Index-Based Merging

Lecture 04 introduced labels and alignment, and the later index section builds on this prerequisite with index-management mechanics. Keep the distinction in mind here: `join()` and `combine_first()` match by labels rather than by row position.

`join()` is a simpler alternative to `merge()` when working with indexes: it defaults to a left join on index labels.

**Reference:**
- `df1.join(df2)` - Left join on index (default)
- `df1.join(df2, how='outer')` - Outer join on index
- `df1.join(df2, on='key')` - Join df2's index to df1's 'key' column

**Example:**

```python
# Time series data with dates as index
prices = pd.DataFrame({'price': [100, 101, 102]}, 
                      index=pd.to_datetime(['2023-01', '2023-02', '2023-03']))
volumes = pd.DataFrame({'volume': [1000, 1100, 1200]}, 
                       index=pd.to_datetime(['2023-01', '2023-02', '2023-03']))

# Join on index
combined = prices.join(volumes)
display(combined)
#           price  volume
# 2023-01     100    1000
# 2023-02     101    1100  
# 2023-03     102    1200
```

### Patching Missing Data with combine_first()

Use `combine_first()` when two aligned sources represent the same variables and the caller is the authoritative source while the other is a fallback. This is a label-based patch operation, not a way to combine observations from different periods.

**Reference:**

- `df1.combine_first(df2)` - Fill missing values in df1 with values from df2
- Works by index alignment - matching index values are combined
- Preserves non-null values from calling DataFrame
- Fills NaN values with values from other DataFrame

**Example:**

```python
# A primary extract plus a lower-priority repair source, keyed by product
primary_sales = pd.DataFrame(
    {'sales': [100.0, None, 150.0]},
    index=pd.Index(['A', 'B', 'C'], name='product')
)
backup_sales = pd.DataFrame(
    {'sales': [200.0, 175.0, 90.0]},
    index=pd.Index(['B', 'C', 'D'], name='product')
)

complete = primary_sales.combine_first(backup_sales)
display(complete)
#          sales
# product
# A        100.0  # Kept from the primary source
# B        200.0  # Filled from the backup source
# C        150.0  # Primary value wins over backup value 175.0
# D         90.0  # Label found only in the backup source

```

**Real-world example:** Applying a reviewed repair table to gaps in a primary extract. Confirm that row and column labels have the same meaning in both sources before combining them.


It's important to make sure your analysis destroys as much information as it produces.

# Concatenating DataFrames Along an Axis

*Think of concatenation as stacking LEGO bricks - you can stack them vertically (add more rows) or horizontally (add more columns). Just make sure they fit together!*

Concatenation combines DataFrames by stacking them together, either adding rows (vertical) or columns (horizontal). Unlike merging, concatenation doesn’t use keys - it simply glues DataFrames together.

**Visual Guide - Concatenation Types:**

```
VERTICAL CONCATENATION (axis=0)     HORIZONTAL CONCATENATION (axis=1)
DataFrame A:                       DataFrame A:    DataFrame B:
┌─────────┐                        ┌─────────┐    ┌─────────┐
│ A │ B   │                        │ A │ B   │    │ C │ D   │
├─────────┤                        ├─────────┤    ├─────────┤
│ 1 │ 2   │                        │ 1 │ 2   │    │ 5 │ 6   │
│ 3 │ 4   │                        │ 3 │ 4   │    │ 7 │ 8   │
└─────────┘                        └─────────┘    └─────────┘
         +
DataFrame B:                               =
┌─────────┐                        ┌─────────────────┐
│ A │ B   │                        │ A │ B │ C │ D   │
├─────────┤                        ├─────────────────┤
│ 5 │ 6   │                        │ 1 │ 2 │ 5 │ 6   │
│ 7 │ 8   │                        │ 3 │ 4 │ 7 │ 8   │
└─────────┘                        └─────────────────┘
         =
┌─────────┐
│ A │ B   │
├─────────┤
│ 1 │ 2   │  ← Stacked vertically
│ 3 │ 4   │
│ 5 │ 6   │
│ 7 │ 8   │
└─────────┘

```

## Vertical Concatenation: Adding More Rows

The most common use case - combining datasets with the same columns.

**Reference:**

- `pd.concat([df1, df2, df3])` - Stack DataFrames vertically (default axis=0)
- `pd.concat([df1, df2], axis=0)` - Explicit vertical stacking
- `pd.concat([df1, df2], ignore_index=True)` - Reset index to 0, 1, 2, …
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

**Why this matters:** Use concat for similar data, merge for related data.

## Horizontal Concatenation: Adding More Columns

Horizontal concatenation is useful for adding related columns side-by-side when both objects already use the same row labels.

**Reference:**

- `pd.concat([df1, df2], axis=1)` - Stack DataFrames horizontally
- Indexes are aligned - matching index values are joined
- Missing indexes result in NaN values
- Use only when index labels represent the same row identity in every object

**Example:**

```python
# Put the real identity in the index before aligning independent sources.
grades = pd.DataFrame({
    'student_id': ['S001', 'S002', 'S003'],
    'name': ['Alice', 'Bob', 'Charlie'],
    'grade': [95, 88, 92]
}).set_index('student_id')

attendance = pd.DataFrame({
    'student_id': ['S002', 'S003', 'S004'],
    'days_present': [18, 20, 19],
    'days_total': [20, 20, 20]
}).set_index('student_id')

# Horizontal concatenation aligns the student_id labels.
combined = pd.concat([grades, attendance], axis=1)
display(combined)
#                name  grade  days_present  days_total
# student_id
# S001          Alice   95.0           NaN         NaN
# S002            Bob   88.0          18.0        20.0
# S003        Charlie   92.0          20.0        20.0
# S004            NaN    NaN          19.0        20.0

```

Do not rely on default `RangeIndex` values from independently loaded tables: two unrelated first rows would both have label `0` and would appear to match. If identity is stored in ordinary columns rather than the index, use `merge()` on those keys.

## Column-Set Alignment with the join Parameter

For the vertical concatenation shown here, `join=` is the **column-set alignment mode**, not a relational join. It tells `concat()` whether to keep the union or intersection of columns on the non-concatenation axis.

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

The same observations can often be represented in **wide** or **long** form. In the student-score example below, wide form gives each subject its own column, while long form gives each student-subject observation its own row. Neither shape is universally better; choose the shape required by the next operation.

**Visual Guide - Wide vs Long Format:**

```
WIDE FORMAT (subject columns)       LONG FORMAT (subject-value rows)
┌─────────┬──────┬─────────┬────────┐  ┌─────────┬─────────┬───────┐
│ student │ math │ english │ science│  │ student │ subject │ score │
├─────────┼──────┼─────────┼────────┤  ├─────────┼─────────┼───────┤
│ Alice   │  95  │   90    │   92   │  │ Alice   │ math    │  95   │
│ Bob     │  88  │   85    │   90   │  │ Alice   │ english │  90   │
│ Charlie │  92  │   94    │   89   │  │ Alice   │ science │  92   │
└─────────┴──────┴─────────┴────────┘  │ Bob     │ math    │  88   │
                                       │ Bob     │ english │  85   │
                                       │ Bob     │ science │  90   │
                                       │ Charlie │ math    │  92   │
                                       │ Charlie │ english │  94   │
                                       │ Charlie │ science │  89   │
                                       └─────────┴─────────┴───────┘

Wide: One row per student in this example      Long: One row per student-subject observation
Useful for: matrix-like comparisons            Useful for: grouping/filtering by subject

```


## Understanding Wide Format

In this example, wide format has one row per student and a separate column for each subject. In other datasets, the row meaning may be different; “wide” describes values spread across columns, not a universal one-row-per-entity rule.

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

# This shape is convenient for side-by-side subject comparisons
# and tools that require one fixed feature column per subject.

```

## Understanding Long Format

Here, long format records one measured value per row: the `student` and `subject` columns identify the observation, and `score` contains its value.

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

# This shape makes subject a value that can be grouped or filtered.
# Many plotting APIs also accept this tidy representation directly.

```


## Pivoting Long to Wide with pivot()

The `pivot()` method reshapes long data to wide form without aggregating values.

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

# Pivot makes the column names the new column headers
# And index becomes the row labels
# Values fill the cells

```

**Common error:** If an `index`/`columns` pair identifies more than one value, `pivot()` raises an error because it cannot choose a cell value. First determine whether the duplicates are data errors or repeated observations. Use `pivot_table()` only when an explicit aggregation is part of the question.

### pivot_table(): Aggregating Before Reshaping (Preview)

`pivot()` needs each index/column pair to identify one value. If repeated observations are valid, an aggregation must decide how those values become one cell; `pivot_table()` performs that aggregation before reshaping. The choice of `sum`, `mean`, or another function changes the question being answered. Aggregation and pivot tables are taught canonically in [Lecture 08](../08/README.md#pivot-tables-and-cross-tabulations).

```python
sales = pd.DataFrame({
    'month': ['Jan', 'Jan'],
    'category': ['Electronics', 'Electronics'],
    'amount': [100, 150],
})

# pivot() would fail because Jan/Electronics appears twice.
# Use this only when summing those rows is part of the question.
sales_pivot = pd.pivot_table(sales, values='amount',
                             index='month', columns='category',
                             aggfunc='sum')
```

## Melting Wide to Long with melt()

The `melt()` function reshapes selected wide columns into variable-value rows.

**Reference:**

- `pd.melt(df, id_vars=['id_col'], value_vars=['col1', 'col2'])` - Basic melt
- `id_vars` - Columns to keep as identifier variables
- `value_vars` - Columns to unpivot (if None, uses all columns except id_vars)
- `var_name` - Name for the new ‘variable’ column (default: ‘variable’)
- `value_name` - Name for the new ‘value’ column (default: ‘value’)

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

# The subject label is now a value in a tidy column, ready for a later
# aggregation or visualization step. Aggregation is introduced in Lecture 08.

```

**Real-world example:** Survey responses stored in Q1, Q2, and Q3 columns can be melted when a downstream operation needs the question name as a row value.

**Why this matters:** Reshaping changes structure, not the underlying observations; the appropriate form depends on the plotting, grouping, or modeling interface you are using.

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

Wide: subject columns         Long: subject stored as a value
      matrix-like layout            convenient for grouping by subject

```


If a reshape feels mysterious, write down what one row represents before
choosing `pivot()` or `melt()`—your future self will thank you for the labels.

# LIVE DEMO!

(Demo 2: Survey Data Reshaping)

# Working with DataFrame Indexes

*Pro tip: Understanding when to move columns to the index (and back) is like understanding when to put your keys in your pocket vs. your hand - it’s all about what you need to access quickly!*

The index is pandas' row-label axis. Pandas uses it for `.loc` selection and automatic alignment, but it is not automatically a database primary key or a guarantee of row uniqueness.

In the pandas 3 Copy-on-Write model, treat index changes as transformations: capture the returned DataFrame or deliberately reassign the variable.

**Key Properties:**

- Index labels may be unique or duplicated; `.loc[label]` returns every matching row.
- A default `RangeIndex` is appropriate when rows do not need meaningful labels.
- Use meaningful labels when label-based selection or alignment serves the task.
- If an operation requires uniqueness, check `df.index.is_unique` after creating the index.

## set_index(): Moving Columns to Index

`set_index()` moves one or more columns into the row labels, enabling label-based selection and alignment on those values.

**Reference:**

- `df.set_index('column')` - Make column the new index
- `df.set_index(['col1', 'col2'])` - Create MultiIndex from multiple columns
- `drop=False` - Keep the column in the DataFrame (default is True, removes it)
- `df.index.is_unique` - Check whether the resulting index labels are unique
- Capture the returned object, as in `indexed = df.set_index('column')`

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

# Make emp_id the index and assert the expected uniqueness
indexed = employees.set_index('emp_id')
assert indexed.index.is_unique
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

**Why this matters:** The employee ID can now be used explicitly with `.loc[]` and for label alignment. Setting an index does not otherwise establish relational integrity unless you request or check uniqueness.

## reset_index(): Moving Index to Columns

The opposite operation - converts index back to a regular column.

**Reference:**

- `df.reset_index()` - Move index to column(s)
- `drop=True` - Discard the index instead of converting to column
- Capture or reassign the returned object, as in `flat = df.reset_index()`

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
dropped = indexed.reset_index(drop=True)
display(dropped)
#       name   department  salary
# 0    Alice  Engineering   95000
# 1      Bob        Sales   75000
# 2  Charlie  Engineering   88000

```

**Common use case:** After an index-based reshape or combination, you may want
to reset_index() to make index labels regular columns again.

## Basic MultiIndex Operations

*MultiIndex (hierarchical indexing) represents each row or column label with more than one level, such as a `(region, quarter)` pair.*

**Reference:**

- `df.set_index(['col1', 'col2'])` - Create MultiIndex from multiple columns
- `df.index.names = ['level1', 'level2']` - Name the index levels
- `df.loc[('key1', 'key2'), :]` - Access a specific two-level row label
- `df.swaplevel(0, 1)` - Swap index levels
- `df.sort_index(level=0)` - Sort by specific level

**Example:**

```python
# Sales data
sales = pd.DataFrame({
    'region': ['West', 'West', 'East', 'East'],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
    'sales': [100, 150, 120, 180]
})

# Build the two-level index directly from unique row labels.
summary = sales.set_index(['region', 'quarter']).sort_index()
assert summary.index.is_unique
display(summary)

# Check the index
display(summary.index)
# MultiIndex([('East', 'Q1'),
#             ('East', 'Q2'),
#             ('West', 'Q1'),
#             ('West', 'Q2')],
#            names=['region', 'quarter'])

```

**Common pattern:** Use `.reset_index()` to convert MultiIndex labels back to
regular columns.

```python
# Convert MultiIndex back to regular columns
flattened = summary.reset_index()
display(flattened)
#   region quarter  sales
# 0   East      Q1    120
# 1   East      Q2    180
# 2   West      Q1    100
# 3   West      Q2    150

# Now easier to work with for most people

```


*“The data clearly shows that our hypothesis is correct, assuming we ignore all the data that doesn’t support our hypothesis.”*

# LIVE DEMO! (Demo 3: Index Management and Concatenation)
