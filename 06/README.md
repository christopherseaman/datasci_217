---
notion:
  title_line: "# 06) Data Wrangling: Join, Combine, and Reshape"
  role: lecture
  status: mapped
  page_id: "293d9fdd-1a1a-801c-bef2-e6140976408c"
  url: "https://app.notion.com/p/293d9fdd1a1a801cbef2e6140976408c"
---

# 06) Data Wrangling: Join, Combine, and Reshape

**Assignment 6:** [assignment instructions](assignment/README.md)

See [BONUS.md](BONUS.md) for the optional extensions.

**Live notebooks in Colab:** [Demo 1](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo1_merge_operations.ipynb) · [Demo 2](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo2_pivot_melt.ipynb) · [Demo 3](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo3_concat_timeseries.ipynb)

*Fun fact: “wrangle” comes from the Low German “wrangeln,” meaning “to dispute; to wrestle.” Surprisingly accurate - data wrangling is arguing and wrestling with your data until it finally agrees to cooperate.*

# Database-Style DataFrame Joins

*Reality check: Merging datasets is the single most common data wrangling task you’ll perform. Master pd.merge() and you’ll save yourself countless hours of frustration.*

Health data rarely arrives in one table. A clinic keeps a patients table (one row per patient: ID, birth year, clinic) and a lab results table (one row per test: patient ID, test name, value). To ask whether older patients have higher A1c, you need each lab value next to that patient's birth year. **Joining** lines those rows up using a **key**: a column, such as `patient_id`, whose values name the same thing in both tables. In pandas the main tool is `pd.merge()`, so joins are often called **merges**.

Join keys are the table's name tags: if two rows share a tag, pandas brings their columns together. A good key makes matching boring; a bad key turns the merge into an enthusiastic photocopier.

This builds on Lecture 05's row meaning, which is also called the table's **grain**: what one row represents (one patient, or one lab test). Four terms describe how keys behave in a join:

- A **primary key** uniquely identifies each row of its own table: `patient_id` in the patients table.
- A **candidate key** is any column, or combination of columns, that could serve as the primary key; Lecture 05 called it a candidate identifier.
- A **foreign key** refers to another table's primary key: `patient_id` in the lab table, where one patient can appear many times.
- **Cardinality** says how many rows can match on each side, for example one patient to many lab results (**one-to-many**).

The examples below use the same shape with customers (one row per customer) and purchases (one row per purchase). First get a merge working with `pd.merge()`; at the end of this topic, `validate=` makes pandas check that the keys behave the way you expect.

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

### Reference Card: `merge()` keys and options

| Argument | Purpose | Typical output |
| --- | --- | --- |
| `left`, `right` | Input tables; `left.merge(right, on=...)` is the same call written as a DataFrame method | Combined `DataFrame` |
| `on='key'` or `on=['k1', 'k2']` | Shared key column, or several that must all match | Key-based matches |
| `left_on`, `right_on` | Different key names | Matches equivalent fields |
| `how` | `inner`, `left`, `right`, `outer`, or `cross` | Controls retained rows |
| `validate` / `indicator` | Check cardinality / match status | `MergeError` or `_merge` audit column |
| `suffixes` | Rename overlapping non-key columns | Unambiguous columns |

### Code Snippet: Merge on a key

```python
import pandas as pd

customers = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003', 'C004'],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'city': ['Seattle', 'Portland', 'Seattle', 'Eugene']
})
purchases = pd.DataFrame({
    'customer_id': ['C001', 'C001', 'C002', 'C005'],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'amount': [999.99, 25.99, 79.99, 299.99]
})

# pandas would find the shared 'customer_id' column on its own, but name it anyway
merged = pd.merge(customers, purchases, on='customer_id')
display(merged)
#   customer_id   name      city   product  amount
# 0        C001  Alice   Seattle    Laptop  999.99
# 1        C001  Alice   Seattle     Mouse   25.99
# 2        C002    Bob  Portland  Keyboard   79.99
```

## Join Types: The Four Horsemen of Data Merging (Plus One)

When a key appears in only one table, the join type decides whether that row survives. The four keyed joins each answer a different question about your data. The fifth rider, `cross`, ignores keys entirely and pairs every left row with every right row; it gets its own section below.

### Reference Card: Join types

| `how` | Retains | Unmatched side |
| --- | --- | --- |
| `inner` | Matching keys only | Dropped |
| `left` | Every left row | Right fields become `NaN` |
| `right` | Every right row | Left fields become `NaN` |
| `outer` | Every row from both | Missing fields become `NaN` |
| `cross` | Every left row paired with every right row; takes no `on=` | Nothing to match; `len(left) × len(right)` rows |

Add `indicator=True` to any merge to get a `_merge` column that says where each row came from: `both`, `left_only`, or `right_only`.

### Code Snippet: Compare join types

```python
# Inner join (default) - only customers with purchases
inner = pd.merge(customers, purchases, on='customer_id', how='inner')
display(inner)
# Result: 3 rows (Alice twice, Bob once) - only matching customers

# Left join - ALL customers, even without purchases
left = pd.merge(customers, purchases, on='customer_id', how='left')
display(left)
#   customer_id     name      city   product  amount
# 0        C001    Alice   Seattle    Laptop  999.99
# 1        C001    Alice   Seattle     Mouse   25.99
# 2        C002      Bob  Portland  Keyboard   79.99
# 3        C003  Charlie   Seattle       NaN     NaN  # No purchase
# 4        C004    Diana    Eugene       NaN     NaN  # No purchase

# Right join - ALL purchases, even without customer info
right = pd.merge(customers, purchases, on='customer_id', how='right')
display(right)
# Result: 4 rows - includes C005's monitor (customer info is NaN)

# Outer join - EVERYTHING; indicator=True labels where each row came from
outer = pd.merge(customers, purchases, on='customer_id', how='outer', indicator=True)
display(outer)
#   customer_id     name      city   product  amount      _merge
# 0        C001    Alice   Seattle    Laptop  999.99        both
# 1        C001    Alice   Seattle     Mouse   25.99        both
# 2        C002      Bob  Portland  Keyboard   79.99        both
# 3        C003  Charlie   Seattle       NaN     NaN   left_only
# 4        C004    Diana    Eugene       NaN     NaN   left_only
# 5        C005      NaN       NaN   Monitor  299.99  right_only
```

**Pro tip:** Most beginners default to inner joins and lose data without realizing it. Use left joins when the left DataFrame is your “master” list (e.g., all customers), right joins for the opposite, and outer joins when you need to see ALL the data from both sides. An inner join drops customers without purchases; a left join keeps them.

## Merging on Multiple Columns

Sometimes one key column isn’t enough to identify a match: several columns must match together (like matching on BOTH `store_id` AND `quarter`). Lecture 05’s visit table worked the same way: `patient_id` repeats across visits, so its candidate identifier was `patient_id` + `visit_date`. Pass a list, `on=['store_id', 'quarter']`, and every listed column must match.

### Code Snippet: Merge on multiple columns

```python
# Monthly sales for each store in Q1
sales_q1 = pd.DataFrame({
    'store_id': ['S01', 'S01', 'S02', 'S02'],
    'quarter': ['Q1', 'Q1', 'Q1', 'Q1'],
    'month': ['Jan', 'Feb', 'Jan', 'Feb'],
    'sales': [50000, 55000, 42000, 48000]
})

# One target per store and quarter
targets = pd.DataFrame({
    'store_id': ['S01', 'S02', 'S01', 'S02'],
    'quarter': ['Q1', 'Q1', 'Q2', 'Q2'],
    'target': [52000, 45000, 58000, 50000]
})

# Merge on BOTH store_id AND quarter
merged = pd.merge(sales_q1, targets, on=['store_id', 'quarter'])
display(merged)
#   store_id quarter month  sales  target
# 0      S01      Q1   Jan  50000   52000  # Each monthly row gets its store's Q1 target (many-to-one)
# 1      S01      Q1   Feb  55000   52000
# 2      S02      Q1   Jan  42000   45000
# 3      S02      Q1   Feb  48000   45000

# Merging on store_id alone pairs every Q1 sale with the Q2 target too
wrong = pd.merge(sales_q1, targets, on='store_id')
print(len(wrong))
# 8
```

## Listing Every Combination with a Cross Join

A clinic network expects every clinic to send an arrival report every hour. A missing report leaves no row at all, and no join type can flag a row that doesn’t exist. First build the **expected grid**, every clinic at every hour, with `how='cross'`: it pairs each left row with each right row and needs no key. Then left-merge the observed reports onto the grid, and combinations without a report show up as `left_only`.

### Code Snippet: Find missing reports with a cross join

```python
clinic_list = pd.DataFrame({'clinic': ['North', 'South']})
hours = pd.DataFrame({'hour': [8, 9, 10]})
expected = pd.merge(clinic_list, hours, how='cross')
print(len(expected))
# 6   <- 2 clinics × 3 hours

reports = pd.DataFrame({
    'clinic': ['North', 'North', 'South', 'South'],
    'hour': [8, 10, 8, 9],
    'arrivals': [3, 0, 5, 2],
})
coverage = pd.merge(expected, reports, on=['clinic', 'hour'],
                    how='left', indicator=True)
display(coverage)
#   clinic  hour  arrivals     _merge
# 0  North     8       3.0       both
# 1  North     9       NaN  left_only   <- North sent no 9:00 report
# 2  North    10       0.0       both    <- a real report of zero arrivals
# 3  South     8       5.0       both
# 4  South     9       2.0       both
# 5  South    10       NaN  left_only
```

North’s 10:00 report of zero arrivals is real data; its 9:00 report never arrived, so that value stays `NaN` rather than 0.

## Handling Overlapping Column Names

When both DataFrames have columns with the same name (besides the merge key), pandas adds suffixes to distinguish them.

### Reference Card: Overlapping columns

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| Default suffixes (`_x`, `_y`) | Distinguish overlapping columns | `total_x` from the left table, `total_y` from the right |
| `suffixes=('_sales', '_inventory')` | Say what each column means | Named output columns |

### Code Snippet: Name overlapping columns

```python
# Both tables have a 'total' column: revenue in one, units in stock in the other
sales = pd.DataFrame({'product_id': ['P001', 'P002'], 'total': [100, 200]})
inventory = pd.DataFrame({'product_id': ['P001', 'P002'], 'total': [50, 75]})

display(pd.merge(sales, inventory, on='product_id'))
#   product_id  total_x  total_y
# 0       P001      100       50
# 1       P002      200       75

display(pd.merge(sales, inventory, on='product_id',
                 suffixes=('_sales', '_inventory')))
#   product_id  total_sales  total_inventory
# 0       P001          100               50
# 1       P002          200               75
```

**Pro tip:** Always use descriptive suffixes! `_sales` and `_inventory` are much clearer than `_x` and `_y`.

## Checking Merge Cardinality

A merge can run without errors and still be wrong. If a lookup table that should have one row per key repeats a key, every matching row is copied once per repeat and the result quietly grows. The cardinality from the introduction is a contract: `validate=` makes pandas check it and stop with an error when the data breaks it, and `indicator=True` shows which rows found a match.

### Reference Card: Merge cardinality and audit

| Contract | Meaning | Check |
| --- | --- | --- |
| One-to-one | Unique keys on both sides | `validate='one_to_one'` |
| One-to-many | Unique left keys; repeated right keys allowed | `validate='one_to_many'` |
| Many-to-one | Repeated left keys; unique right keys | `validate='many_to_one'` |
| Many-to-many | Repeated keys on both sides | `validate='many_to_many'` documents, not constrains |
| Broken contract | A side you declared unique has repeats | Raises `pd.errors.MergeError`; catch it with `try`/`except` (Lecture 02) |
| Find the repeats | Every row whose key appears more than once | `df[df.duplicated(subset=['key'], keep=False)]` (Lecture 05); list every key column, such as `subset=['k1', 'k2']`, for a composite key |
| Match audit | Show `left_only`, `right_only`, `both` | `indicator=True` |

### Code Snippet: Catch a broken merge contract

```python
# Each visit should get exactly one clinic, but K2 has two lookup rows
clinics = pd.DataFrame({
    'clinic_id': ['K1', 'K2', 'K2'],
    'clinic_name': ['Mission Bay', 'Parnassus', 'Parnassus Annex'],
    'record_status': ['current', 'retired', 'current'],
})
visits = pd.DataFrame({
    'visit_id': ['V1', 'V2', 'V3', 'V4'],
    'clinic_id': ['K1', 'K2', 'K2', 'K9'],
})

# Without a check, the repeated key silently photocopies rows
unchecked = pd.merge(visits, clinics, on='clinic_id', how='left')
print(len(unchecked))
# 6   <- 4 visits went in; V2 and V3 each matched two clinic rows

display(clinics[clinics.duplicated(subset=['clinic_id'], keep=False)])
#   clinic_id      clinic_name record_status
# 1        K2        Parnassus       retired
# 2        K2  Parnassus Annex       current

try:
    pd.merge(visits, clinics, on='clinic_id', how='left', validate='many_to_one')
except pd.errors.MergeError as error:
    print('MergeError:', error)
# MergeError: Merge keys are not unique in right dataset; not a many-to-one merge
#
# Duplicates in right:
#  clinic_id
#        K2 ...

# Fix the lookup with a documented rule, then merge and audit
current = clinics[clinics['record_status'] == 'current']
audit = pd.merge(visits, current, on='clinic_id', how='left',
                 validate='many_to_one', indicator=True)
display(audit)
#   visit_id clinic_id      clinic_name record_status     _merge
# 0       V1        K1      Mission Bay       current       both
# 1       V2        K2  Parnassus Annex       current       both
# 2       V3        K2  Parnassus Annex       current       both
# 3       V4        K9              NaN           NaN  left_only
```

Row growth alone does not prove a many-to-many merge: an intended one-to-many merge also adds rows. Inspect key uniqueness and use `validate=` to make the expected relationship executable.

Missing keys match each other. If two visits and two clinic rows all have a blank `clinic_id`, pandas pairs every blank with every blank and adds four false matches. Check `df['clinic_id'].notna().all()` on both tables before merging, or set rows with missing keys aside first.

# LIVE DEMO!

# Working with DataFrame Indexes

*Pro tip: Understanding when to move columns to the index (and back) is like understanding when to put your keys in your pocket vs. your hand - it’s all about what you need to access quickly!*

Lecture 04 gave `visits` an index of patient IDs, the row labels printed down the left side, so `visits.loc["P002"]` found a patient by name. A table built from a dict or read from a CSV gets the default `0, 1, 2, …` instead, a **RangeIndex** that only counts rows. When a column already identifies rows, such as `patient_id` or `emp_id`, you can move it into the index. Then `.loc['E002']` finds that row directly, and pandas can line up rows from two tables by label, just as Lecture 04’s derived columns lined up with their rows by index label.

Later in this lecture, `pivot()` builds its result’s index from an identifier column, and horizontal concatenation and `combine_first()` match rows by index label.

- An index is not automatically unique. Check `df.index.is_unique` when each label should appear once.
- `set_index()` and `reset_index()` return a new DataFrame, so assign the result: `indexed = df.set_index('emp_id')`.

## Row Labels Before and After

```
employees as built from a dict: the RangeIndex down the left edge only counts rows

  emp_id     name   department  salary
0   E001    Alice  Engineering   95000
1   E002      Bob        Sales   75000
2   E003  Charlie  Engineering   88000

employees.set_index('emp_id'): the IDs are the row labels, so .loc['E002'] finds Bob

           name   department  salary
emp_id
E001      Alice  Engineering   95000
E002        Bob        Sales   75000
E003    Charlie  Engineering   88000
```

## set_index(): Moving Columns to Index

`set_index()` moves one or more columns into the row labels, enabling label-based selection and alignment on those values.

### Reference Card: `set_index()`

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `df.set_index('column')` | Make column the new index | DataFrame with new index |
| `drop=False` | Keep the column in the DataFrame (default is True, removes it) | DataFrame retains the source column |
| `df.index.is_unique` | Test that all row labels are unique | `True` when no labels repeat; otherwise `False` |

### Code Snippet: Set an identifier index

```python
employees = pd.DataFrame({
    'emp_id': ['E001', 'E002', 'E003'],
    'name': ['Alice', 'Bob', 'Charlie'],
    'department': ['Engineering', 'Sales', 'Engineering'],
    'salary': [95000, 75000, 88000]
})

indexed = employees.set_index('emp_id')  # row labels become E001, E002, E003
assert indexed.index.is_unique           # each employee should appear once

display(indexed.loc['E002'])             # Bob's record, found by label
# name            Bob
# department    Sales
# salary        75000
# Name: E002, dtype: object
```

## reset_index(): Moving Index to Columns

The opposite operation - converts index back to a regular column.

### Reference Card: `reset_index()`

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `df.reset_index()` | Move index labels into columns | New `DataFrame` with a default `RangeIndex` |
| `df.reset_index(drop=True)` | Discard old labels instead of saving them as columns | New `DataFrame` with a default `RangeIndex` |

### Code Snippet: Restore an identifier column

```python
# Move the labels back into a column: the original structure returns
display(indexed.reset_index())
#   emp_id     name   department  salary
# 0   E001    Alice  Engineering   95000
# 1   E002      Bob        Sales   75000
# 2   E003  Charlie  Engineering   88000

# Discard the labels instead of keeping them
display(indexed.reset_index(drop=True))
#       name   department  salary
# 0    Alice  Engineering   95000
# 1      Bob        Sales   75000
# 2  Charlie  Engineering   88000
```

## Two-Level Row Labels

One column does not always name a row on its own: a quarterly sales figure is identified by region *and* quarter together. Passing `set_index()` a list of columns gives each row a two-part label, and pandas calls the result a **MultiIndex** (hierarchical index): each row label has more than one level, such as a `(region, quarter)` pair. Later in this lecture, `pivot()` with a list of identifier columns builds the same kind of label.

### Reference Card: Two-level row labels

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `df.set_index(['col1', 'col2'])` | Build two-level row labels from two columns | DataFrame with a `MultiIndex` |
| `df.loc[('key1', 'key2'), :]` | Select a row by both parts of its label; `df.loc['key1']` selects every row under one outer label | Selected row or rows |
| `df.reset_index()` | Turn both levels back into columns | New `DataFrame` with a default `RangeIndex` |

### Code Snippet: Build a MultiIndex

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
#                 sales
# region quarter
# East   Q1         120
#        Q2         180
# West   Q1         100
#        Q2         150

# Every row under one outer label
display(summary.loc['East'])
#          sales
# quarter
# Q1         120
# Q2         180

# Both levels back into ordinary columns
display(summary.reset_index())
#   region quarter  sales
# 0   East      Q1    120
# 1   East      Q2    180
# 2   West      Q1    100
# 3   West      Q2    150
```

After `pivot()` or a horizontal concatenation later in this lecture, `reset_index()` turns the identifiers left in the index back into ordinary columns.

# Reshaping: Wide vs Long Format

*Fun fact: 90% of data reshaping confusion comes from not understanding which format you have and which format you need. Once you know that, the solution is usually obvious!*

A blood-pressure study measures each patient at baseline and again at follow-up. The clinic’s spreadsheet has one row per patient and a column per visit: `baseline_sbp`, `followup_sbp`. That layout is **wide**: several measurements of the same kind spread across columns. Plotting and grouping tools usually want **long** layout: one row per patient-visit, a `visit` column saying which measurement it is, and a single `sbp` column holding the value.

Both tables hold exactly the same numbers; only the row meaning (Lecture 05) changes. In long format the **identifier columns** (`patient_id`, `visit`) say which observation a row is, and the **value column** holds the measurement. Long data with one observation per row and one variable per column is often called **tidy** data, after Hadley Wickham's [Tidy Data](https://www.jstatsoft.org/article/view/v059i10) paper.

Reshaping never adds or removes observations. `melt()` turns wide into long. `pivot()` turns long back into wide, which works only when each identifier combination appears once. The examples below use a small student-score table with the same structure: one row per student with a column per subject (wide), or one row per student-subject score (long).

| Shape | One row represents | Best for | Conversion |
| --- | --- | --- | --- |
| Wide | One entity with several measured columns | Side-by-side comparison | `melt()` to long |
| Long | One entity/variable observation | Grouping and tidy plotting | `pivot()` to wide |

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

## Building Both Shapes

The two frames below hold exactly the numbers in the diagram above. The wide one keeps a column per subject, which suits side-by-side comparison; the long one makes `subject` a value that can be filtered, grouped, or handed to a plotting library. “Wide” describes values spread across columns, not a universal one-row-per-entity rule: another dataset may put something else in a row.

### Code Snippet: Wide and long versions of the same scores

```python
# Wide: one row per student, one column per subject
wide_data = pd.DataFrame({
    'student': ['Alice', 'Bob', 'Charlie'],
    'math': [95, 88, 92],
    'english': [90, 85, 94],
    'science': [92, 90, 89]
})

# Long: one row per student-subject score
long_data = pd.DataFrame({
    'student': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Bob',
                'Charlie', 'Charlie', 'Charlie'],
    'subject': ['math', 'english', 'science'] * 3,
    'score': [95, 90, 92, 88, 85, 90, 92, 94, 89]
})
display(long_data.head(3))
#   student  subject  score
# 0   Alice     math     95
# 1   Alice  english     90
# 2   Alice  science     92
```

## Melting Wide to Long with melt()

The `melt()` function, written `pd.melt(df, ...)` or `df.melt(...)`, is the wide-to-long step: it holds the identifier columns fixed and turns each selected wide column into variable-value rows. The blood-pressure table that opened this topic needs exactly this before anything can group or plot by visit.

### Reference Card: `melt()`

| Argument | Purpose | Output |
| --- | --- | --- |
| `id_vars` | Columns to keep fixed | Identifier columns repeated |
| `value_vars` | Columns to unpivot | One row per source row and selected value column |
| `var_name` | Name for former column labels | Variable column |
| `value_name` | Name for former cell values | Value column |

### Code Snippet: Melt wide data

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
```

The subject label is now a value in a tidy column, ready for a later aggregation (Lecture 08) or plot. Survey responses stored in `Q1`, `Q2`, and `Q3` columns melt the same way when a downstream step needs the question name as a row value.

## Pivoting Long Back to Wide with pivot()

`pivot()` runs `melt()` backwards: the variable column supplies the new headers, the value column fills the cells, and the identifier column becomes the row labels. Reach for it when data arrives long, one row per observation the way `long_data` is, and a reader or a matrix-shaped tool wants a column per subject. `pivot()` never aggregates, so each `index`/`columns` pair must identify exactly one value.

### Reference Card: `pivot()`

| Argument | Purpose | Constraint / output |
| --- | --- | --- |
| `index` | Column becoming row labels | One row per index value; a list such as `['id1', 'id2']` gives two-level row labels |
| `columns` | Column becoming headers | One column per value |
| `values` | Column filling cells | Duplicate index/column pairs raise an error |
| `.reset_index()` afterward | Turn the row labels back into an ordinary column | Leaves the `columns` name as a header label |
| `result.columns.name = None` | Drop that leftover header label | Plain column headers, as in the original wide table |

### Code Snippet: Pivot long data back to wide

```python
# long_data was typed by hand and long came out of melt(): the same nine scores
# in a different row order, which pivot() files by label into one frame.
wide = long_data.pivot(index='student', columns='subject', values='score')
display(wide)
# subject  english  math  science
# student
# Alice         90    95       92
# Bob           85    88       90
# Charlie       94    92       89

print(wide.equals(long.pivot(index='student', columns='subject', values='score')))
# True

back = long.pivot(index='student', columns='subject', values='score').reset_index()
display(back)
# subject  student  english  math  science   <- 'subject' is a leftover header label
# 0          Alice       90    95       92
# 1            Bob       85    88       90
# 2        Charlie       94    92       89

back.columns.name = None                                # drop the leftover label
back = back[['student', 'math', 'english', 'science']]  # pivot sorted the columns
print(back.equals(wide_data))
# True
```

Melting `wide_data` and pivoting the result puts every score back in the cell it came from, and carrying the values back is what makes the two operations inverses. The frame itself does not come back equal: `long.pivot(index='student', columns='subject', values='score').equals(wide_data)` is `False`, because the pivot moved `student` into the row labels, left `subject` behind as a header label, and sorted the columns into `english, math, science`. The fixups in the snippet undo those differences: `reset_index()` restores the `student` column, `back.columns.name = None` clears the leftover label, and reselecting the columns restores the original order, after which `equals()` returns `True`. With two identifier columns, `index=['id1', 'id2']` builds the two-level row index from earlier in this lecture, and `reset_index()` turns both levels back into columns.

If an `index`/`columns` pair identifies more than one value, `pivot()` cannot choose a cell value and stops: pandas reports `ValueError: Index contains duplicate entries, cannot reshape`. First determine whether the duplicates are data errors or repeated observations. If repeated observations are valid, `pivot_table()` aggregates them into one cell before reshaping, and the choice of `sum`, `mean`, or another function changes the question being answered. [BONUS.md](BONUS.md) shows that one call; aggregation and pivot tables are taught canonically in [Lecture 08](../08/README.md#pivot-tables-and-cross-tabulations).

If a reshape feels mysterious, write down what one row represents before choosing `pivot()` or `melt()`. Your future self will thank you for the labels.

# LIVE DEMO!

# Concatenating DataFrames Along an Axis

*Think of concatenation as stacking LEGO bricks - you can stack them vertically (add more rows) or horizontally (add more columns). Just make sure they fit together!*

A hospital system exports admissions one month at a time: `jan_admissions.csv`, `feb_admissions.csv`, and so on. Every file has the same columns and the same row meaning (one admission), so nothing needs matching by key: the files just need to go one after another. That is **concatenation**: gluing tables together along an **axis**, numbered as in NumPy (Lecture 03). Stacking rows is `axis=0`; placing columns side by side is `axis=1`.

The two directions line things up differently:

- **Vertical** (`axis=0`): rows are appended and columns are matched by **column name**. A column missing from one file becomes `NaN` for that file’s rows.
- **Horizontal** (`axis=1`): columns are added and rows are matched by **index label**, the alignment from the indexes topic. Use it only after `set_index()` has put a real identifier, such as `patient_id`, in the index.

Use `merge()` when rows must be matched by the values in a key column; use `concat()` when the tables are already pieces of one table. Add a label column (for example `source_file`) before stacking if you need to know where each row came from, its provenance (Lecture 05).

| Goal | Call | Alignment rule | Main risk |
| --- | --- | --- | --- |
| Add observations | `pd.concat(frames, ignore_index=True)` | Rows are appended | Duplicate or incompatible columns |
| Add measured fields | `pd.concat(frames, axis=1)` | Index labels align | Unrelated indexes appear to match |
| Keep only shared fields | `pd.concat(frames, join='inner')` | Columns intersect | Silent loss of columns |

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

### Reference Card: Vertical `concat()`

| Call | Purpose | Result |
| --- | --- | --- |
| `pd.concat([df1, df2])` | Stack rows (`axis=0`) | Appended rows; indexes may repeat |
| `ignore_index=True` | Replace source indexes | Fresh `RangeIndex` |
| `piece['source_file'] = 'jan.csv'` on each table before stacking | Record where each row came from | An ordinary label column that survives the stack |
| `keys=['jan', 'feb']`, `names=['source', 'row']` | Label each piece with an outer index level instead of a column | Two-level row index; `reset_index()` turns the labels into columns |

### Code Snippet: Stack rows

```python
# Sales from different months; the month column records which file each row came from
jan_sales = pd.DataFrame({
    'product': ['Laptop', 'Mouse', 'Keyboard'],
    'quantity': [5, 20, 15],
})
feb_sales = pd.DataFrame({
    'product': ['Laptop', 'Monitor', 'Tablet'],
    'quantity': [8, 3, 12],
})
jan_sales['month'] = 'Jan'
feb_sales['month'] = 'Feb'

# Stack them vertically - combines rows
combined = pd.concat([jan_sales, feb_sales])
display(combined)
#     product  quantity month
# 0    Laptop         5   Jan
# 1     Mouse        20   Jan
# 2  Keyboard        15   Jan
# 0    Laptop         8   Feb  # Index repeats! (0, 1, 2 again)
# 1   Monitor         3   Feb
# 2    Tablet        12   Feb

# Clean indexes with ignore_index=True
combined = pd.concat([jan_sales, feb_sales], ignore_index=True)
display(combined)
#     product  quantity month
# 0    Laptop         5   Jan
# 1     Mouse        20   Jan
# 2  Keyboard        15   Jan
# 3    Laptop         8   Feb  # Clean sequential index
# 4   Monitor         3   Feb
# 5    Tablet        12   Feb
```

## Horizontal Concatenation: Adding More Columns

Horizontal concatenation is useful for adding related columns side-by-side when both objects already use the same row labels, such as identifiers moved into the index with `set_index()` earlier in this lecture.

### Reference Card: Horizontal `concat()`

| Call / rule | Purpose | Result |
| --- | --- | --- |
| `pd.concat([df1, df2], axis=1)` | Add columns | Index labels align rows |
| Shared index labels | Define row identity | Matching labels share a row |
| Missing labels | Preserve unmatched rows | `NaN` in absent fields |

### Code Snippet: Align columns by index

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

The `join=` argument of `concat()` is not a merge. When stacking rows, it decides whether to keep the union or the intersection of the columns. With `axis=1`, it makes the same choice for row labels: `join='inner'` keeps only the labels found in every input.

### Reference Card: `concat(join=...)`

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `join='outer'` (default) | Keep all columns from both frames | Union of columns |
| `join='inner'` | Keep only columns shared by all inputs | Stacked `DataFrame` with the column intersection |
| `axis=1, join='inner'` | Keep only row labels found in every input | Side-by-side `DataFrame` without unmatched rows |

### Code Snippet: Choose a column set

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

## Patching Gaps with combine_first()

Sometimes two sources hold the same variables for the same labeled rows, and one is more trustworthy: a primary extract with a few gaps, and a lower-priority repair table. `combine_first()` keeps every non-missing value of the table you call it on and fills only its gaps from the fallback, matching cells by row and column label. It patches values; it is not a way to stack observations from different periods (that is `concat()`).

### Reference Card: `combine_first()`

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `df1.combine_first(df2)` | Fill gaps using the same row and column labels in df2 | New `DataFrame`; non-null df1 values win |
| Row and column coverage | Keep the union of both sets of labels | Rows or columns found only in df2 are included |

### Code Snippet: Patch from a fallback source

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

Confirm that row and column labels mean the same thing in both sources before combining them.

*“The data clearly shows that our hypothesis is correct, assuming we ignore all the data that doesn’t support our hypothesis.”*

# LIVE DEMO!
