---
notion:
  title_line: "# 06) Data Wrangling: Join, Combine, and Reshape"
  role: lecture
  status: mapped
  page_id: "293d9fdd-1a1a-801c-bef2-e6140976408c"
  url: "https://app.notion.com/p/293d9fdd1a1a801cbef2e6140976408c"
---

# 06) Data Wrangling: Join, Combine, and Reshape

See [BONUS.md](BONUS.md) for the optional extensions.

**Live notebooks in Colab:** [Demo 1](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo1_merge_operations.ipynb) · [Demo 2](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo2_pivot_melt.ipynb) · [Demo 3](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo3_concat_timeseries.ipynb)

_Fun fact: “wrangle” comes from the Low German “wrangeln,” meaning “to dispute; to wrestle.” Surprisingly accurate: data wrangling is arguing and wrestling with your data until it finally agrees to cooperate._

# Database-Style DataFrame Joins

_Reality check: Merging datasets is the single most common data wrangling task you’ll perform. Master pd.merge() and you’ll save yourself countless hours of frustration._

Health data rarely arrives in one table. A clinic keeps a patients table (one row per patient: ID and birth year) and a lab results table (one row per test: patient ID, test name, value). To ask whether older patients have higher A1c, you need each lab value next to that patient's birth year. **Joining** lines those rows up using a **key**: a column, such as `patient_id`, whose values name the same thing in both tables. In pandas the main tool is `pd.merge()`, so joins are often called **merges**.

Join keys are the table's name tags: if two rows share a tag, pandas brings their columns together. A good key makes matching boring; a bad key turns the merge into an enthusiastic photocopier.

This builds on Lecture 05's row meaning, which is also called the table's **grain**: what one row represents (one patient, or one lab test). Four terms describe how keys behave in a join:

- A **primary key** uniquely identifies each row of its own table: `patient_id` in the patients table.
- A **candidate key** is any column, or combination of columns, that could serve as the primary key; Lecture 05 called it a candidate identifier.
- A **foreign key** refers to another table's primary key: `patient_id` in the lab table, where one patient can appear many times.
- **Cardinality** says how many rows can match on each side, for example one patient to many lab results (**one-to-many**).

The examples below use three patients and four lab results. First get a merge working with `pd.merge()`; at the end of this topic, `validate=` makes pandas check that the keys behave the way you expect.

```
patients                    labs
┌────────────┬────────────┐ ┌────────────┬──────┬───────┐
│ patient_id │ birth_year │ │ patient_id │ test │ value │
├────────────┼────────────┤ ├────────────┼──────┼───────┤
│ P001       │ 1958       │ │ P001       │ A1c  │   6.8 │
│ P002       │ 1971       │ │ P001       │ LDL  │ 131.0 │
│ P003       │ 1985       │ │ P002       │ A1c  │   5.4 │
└────────────┴────────────┘ │ P004       │ A1c  │   7.9 │
                            └────────────┴──────┴───────┘
```

Every row either table can contribute, and the join types (`how=`) that keep it:

| patient_id | birth_year | test | value | Why | `inner` | `left` | `right` | `outer` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P001 | 1958 | A1c | 6.8 | Key in both tables | ✓ | ✓ | ✓ | ✓ |
| P001 | 1958 | LDL | 131.0 | Key in both tables | ✓ | ✓ | ✓ | ✓ |
| P002 | 1971 | A1c | 5.4 | Key in both tables | ✓ | ✓ | ✓ | ✓ |
| P003 | 1985 | NaN | NaN | Patient with no labs yet |  | ✓ |  | ✓ |
| P004 | NaN | A1c | 7.9 | Lab with no patient record |  |  | ✓ | ✓ |

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

patients = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003'],
    'birth_year': [1958, 1971, 1985],
})
labs = pd.DataFrame({
    'patient_id': ['P001', 'P001', 'P002', 'P004'],
    'test': ['A1c', 'LDL', 'A1c', 'A1c'],    # A1c in %, LDL in mg/dL
    'value': [6.8, 131.0, 5.4, 7.9],
})

# pandas would find the shared 'patient_id' column on its own, but name it anyway
merged = pd.merge(patients, labs, on='patient_id')
display(merged)
#   patient_id  birth_year test  value
# 0       P001        1958  A1c    6.8
# 1       P001        1958  LDL  131.0
# 2       P002        1971  A1c    5.4
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
# Inner join (default): only patients with labs
inner = pd.merge(patients, labs, on='patient_id', how='inner')
print(len(inner))
# 3   <- P001 twice, P002 once

# Left join: ALL patients, even without labs
left = pd.merge(patients, labs, on='patient_id', how='left')
display(left)
#   patient_id  birth_year test  value
# 0       P001        1958  A1c    6.8
# 1       P001        1958  LDL  131.0
# 2       P002        1971  A1c    5.4
# 3       P003        1985  NaN    NaN   <- no labs yet

# Right join: ALL labs, even without a patient record
right = pd.merge(patients, labs, on='patient_id', how='right')
print(len(right))
# 4   <- includes P004's A1c with birth_year NaN

# Outer join: EVERYTHING; indicator=True labels where each row came from
outer = pd.merge(patients, labs, on='patient_id', how='outer', indicator=True)
display(outer)
#   patient_id  birth_year test  value      _merge
# 0       P001      1958.0  A1c    6.8        both
# 1       P001      1958.0  LDL  131.0        both
# 2       P002      1971.0  A1c    5.4        both
# 3       P003      1985.0  NaN    NaN   left_only
# 4       P004         NaN  A1c    7.9  right_only
```

`birth_year` prints as `1958.0` in the outer join because P004's missing birth year makes the whole column `float64` (Lecture 05).

**Pro tip:** Most beginners default to inner joins and lose data without realizing it. Use a left join when the left table is your master list (every patient in the registry), a right join for the opposite, and an outer join when you need to see everything from both sides. An inner join silently drops P003; a left join keeps P003 with `NaN` labs, which is exactly the patient who is due for a test.

## Merging on Multiple Columns

Sometimes one key column isn’t enough to identify a match: several columns must match together. Lecture 05’s visit table worked the same way: `patient_id` repeats across visits, so its candidate identifier was `patient_id` + `visit_date`. Below, each patient has a baseline and a follow-up visit in both a vitals table and a lab table, so a row is named by `patient_id` and `visit` together. Pass a list, `on=['patient_id', 'visit']`, and every listed column must match.

### Code Snippet: Merge on multiple columns

```python
vitals = pd.DataFrame({
    'patient_id': ['P001', 'P001', 'P002', 'P002'],
    'visit': ['baseline', 'followup', 'baseline', 'followup'],
    'sbp': [152, 138, 128, 124],          # systolic blood pressure, mmHg
})
a1c = pd.DataFrame({
    'patient_id': ['P001', 'P001', 'P002', 'P002'],
    'visit': ['baseline', 'followup', 'baseline', 'followup'],
    'a1c': [7.4, 6.9, 5.6, 5.5],          # %
})

# Merge on BOTH patient_id AND visit
merged = pd.merge(vitals, a1c, on=['patient_id', 'visit'])
display(merged)
#   patient_id     visit  sbp  a1c
# 0       P001  baseline  152  7.4
# 1       P001  followup  138  6.9
# 2       P002  baseline  128  5.6
# 3       P002  followup  124  5.5

# Merging on patient_id alone pairs every visit with every other visit
wrong = pd.merge(vitals, a1c, on='patient_id')
print(len(wrong))
# 8   <- the baseline SBP now sits next to the follow-up A1c too
print(list(wrong.columns))
# ['patient_id', 'visit_x', 'sbp', 'visit_y', 'a1c']
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

When both DataFrames have columns with the same name (besides the merge key), pandas adds suffixes to distinguish them. The multi-column snippet above already produced `visit_x` and `visit_y` when `visit` was left out of the key.

### Reference Card: Overlapping columns

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| Default suffixes (`_x`, `_y`) | Distinguish overlapping columns | `a1c_x` from the left table, `a1c_y` from the right |
| `suffixes=('_lab', '_poc')` | Say what each column means | Named output columns |

### Code Snippet: Name overlapping columns

```python
# Both tables have an 'a1c' column: the central lab's result and a point-of-care device's
lab = pd.DataFrame({'patient_id': ['P001', 'P002'], 'a1c': [7.4, 5.6]})
poc = pd.DataFrame({'patient_id': ['P001', 'P002'], 'a1c': [7.1, 5.9]})

display(pd.merge(lab, poc, on='patient_id'))
#   patient_id  a1c_x  a1c_y
# 0       P001    7.4    7.1
# 1       P002    5.6    5.9

display(pd.merge(lab, poc, on='patient_id', suffixes=('_lab', '_poc')))
#   patient_id  a1c_lab  a1c_poc
# 0       P001      7.4      7.1
# 1       P002      5.6      5.9
```

**Pro tip:** Always use descriptive suffixes! `_lab` and `_poc` say which device measured each value; `_x` and `_y` do not.

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

_Pro tip: Knowing when to move columns into the index (and back) is like knowing when to keep your keys in your hand or in your pocket: it’s all about what you need to reach quickly!_

A DataFrame’s **index** is its row labels, printed down the left side. A table built from a dict or read from a CSV gets the default `0, 1, 2, …`, a **RangeIndex** that only counts rows. `set_index('patient_id')` makes an identifier column the row labels, so `.loc['P002']` finds that patient’s row (label selection, Lecture 04) and pandas can line up rows from two tables by label. `reset_index()` moves the labels back into an ordinary column.

Later in this lecture, `pivot()` builds its result’s index from an identifier column, and horizontal concatenation and `combine_first()` match rows by index label.

- An index is not automatically unique. Check `df.index.is_unique` when each label should appear once.
- `set_index()` and `reset_index()` return a new DataFrame, so assign the result: `indexed = df.set_index('patient_id')`.

## Row Labels Before and After

```
patients as built from a dict: the RangeIndex down the left edge only counts rows

  patient_id  age clinic
0       P001   67  North
1       P002   54  South
2       P003   41  North

patients.set_index('patient_id'): the IDs are the row labels, so .loc['P002'] finds that patient

            age clinic
patient_id
P001         67  North
P002         54  South
P003         41  North
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
patients = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003'],
    'age': [67, 54, 41],
    'clinic': ['North', 'South', 'North'],
})

indexed = patients.set_index('patient_id')  # row labels become P001, P002, P003
assert indexed.index.is_unique              # each patient should appear once

display(indexed.loc['P002'])                # one patient's record, found by label
# age          54
# clinic    South
# Name: P002, dtype: object
```

## reset_index(): Moving Index to Columns

`reset_index()` is the opposite: it moves the index labels back into an ordinary column. With `drop=True` it discards them instead, which renumbers a table whose labels carry no information, such as the gaps a filter leaves in a RangeIndex.

### Reference Card: `reset_index()`

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `df.reset_index()` | Move index labels into columns | New `DataFrame` with a default `RangeIndex` |
| `df.reset_index(drop=True)` | Discard old labels instead of saving them as columns | New `DataFrame` with a default `RangeIndex` |

### Code Snippet: Restore an identifier column

```python
# Move the labels back into a column: the original structure returns
display(indexed.reset_index())
#   patient_id  age clinic
# 0       P001   67  North
# 1       P002   54  South
# 2       P003   41  North

# Discard the labels instead of keeping them
display(indexed.reset_index(drop=True))
#    age clinic
# 0   67  North
# 1   54  South
# 2   41  North
```

## Two-Level Row Labels

One column does not always name a row on its own: a clinic’s quarterly visit count is identified by clinic _and_ quarter together. Passing `set_index()` a list of columns gives each row a two-part label, and pandas calls the result a **MultiIndex** (hierarchical index): each row label has more than one level, such as a `(clinic, quarter)` pair. Later in this lecture, `pivot()` with a list of identifier columns builds the same kind of label.

### Reference Card: Two-level row labels

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `df.set_index(['col1', 'col2'])` | Build two-level row labels from two columns | DataFrame with a `MultiIndex` |
| `df.loc[('key1', 'key2'), :]` | Select a row by both parts of its label; `df.loc['key1']` selects every row under one outer label | Selected row or rows |
| `df.reset_index()` | Turn both levels back into columns | New `DataFrame` with a default `RangeIndex` |

### Code Snippet: Build a MultiIndex

```python
# Visits per clinic and quarter
visits = pd.DataFrame({
    'clinic': ['North', 'North', 'South', 'South'],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
    'visits': [410, 455, 380, 362],
})

# Build the two-level index directly from unique row labels.
summary = visits.set_index(['clinic', 'quarter']).sort_index()
assert summary.index.is_unique
display(summary)
#                 visits
# clinic quarter
# North  Q1          410
#        Q2          455
# South  Q1          380
#        Q2          362

# Every row under one outer label
display(summary.loc['South'])
#          visits
# quarter
# Q1          380
# Q2          362

# Both levels back into ordinary columns
display(summary.reset_index())
#   clinic quarter  visits
# 0  North      Q1     410
# 1  North      Q2     455
# 2  South      Q1     380
# 3  South      Q2     362
```

# Reshaping: Wide vs Long Format

_Fun fact: 90% of data reshaping confusion comes from not understanding which format you have and which format you need. Once you know that, the solution is usually obvious!_

A blood-pressure study measures each patient’s systolic blood pressure (SBP, mmHg) at baseline and again at follow-up. The clinic’s spreadsheet has one row per patient and a column per visit: `baseline` and `followup`. That layout is **wide**: several measurements of the same kind spread across columns. Plotting and grouping tools usually want **long** layout: one row per patient-visit, a `visit` column saying which measurement it is, and a single `sbp` column holding the value.

Both tables hold exactly the same numbers; only the row meaning (Lecture 05) changes. In long format the **identifier columns** (`patient_id`, `visit`) say which observation a row is, and the **value column** holds the measurement. Long data with one observation per row and one variable per column is often called **tidy** data.

Reshaping never adds or removes observations. `melt()` turns wide into long. `pivot()` turns long back into wide, which works only when each identifier combination appears once.

| Shape | One row represents | Best for | Conversion |
| --- | --- | --- | --- |
| Wide | One entity with several measured columns | Side-by-side comparison | `melt()` to long |
| Long | One entity/variable observation | Grouping and tidy plotting | `pivot()` to wide |

```
WIDE FORMAT (a column per visit)          LONG FORMAT (a row per patient-visit)
┌────────────┬──────────┬──────────┐      ┌────────────┬──────────┬─────┐
│ patient_id │ baseline │ followup │      │ patient_id │ visit    │ sbp │
├────────────┼──────────┼──────────┤      ├────────────┼──────────┼─────┤
│ P001       │   152    │   138    │      │ P001       │ baseline │ 152 │
│ P002       │   138    │   132    │      │ P001       │ followup │ 138 │
│ P003       │   145    │   129    │      │ P002       │ baseline │ 138 │
└────────────┴──────────┴──────────┘      │ P002       │ followup │ 132 │
                                          │ P003       │ baseline │ 145 │
Useful for: each patient's change         │ P003       │ followup │ 129 │
at a glance                               └────────────┴──────────┴─────┘
                                          Useful for: grouping or plotting by visit
```

## Building Both Shapes

The two frames below hold exactly the numbers in the diagram above. The wide one keeps a column per visit, which suits side-by-side comparison; the long one makes `visit` a value that can be filtered, grouped, or handed to a plotting library. “Wide” describes values spread across columns, not a universal one-row-per-entity rule: another dataset may put something else in a row.

### Code Snippet: Wide and long versions of the same readings

```python
# Wide: one row per patient, one SBP column per visit
wide_data = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003'],
    'baseline': [152, 138, 145],
    'followup': [138, 132, 129],
})

# Long: one row per patient-visit reading
long_data = pd.DataFrame({
    'patient_id': ['P001', 'P001', 'P002', 'P002', 'P003', 'P003'],
    'visit': ['baseline', 'followup'] * 3,
    'sbp': [152, 138, 138, 132, 145, 129],
})
display(long_data.head(3))
#   patient_id     visit  sbp
# 0       P001  baseline  152
# 1       P001  followup  138
# 2       P002  baseline  138
```

## Melting Wide to Long with melt()

The `melt()` function, written `pd.melt(df, ...)` or `df.melt(...)`, is the wide-to-long step: it holds the identifier columns fixed and turns each selected wide column into variable-value rows. The old column names become the values of the new variable column.

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
               id_vars=['patient_id'],
               value_vars=['baseline', 'followup'],
               var_name='visit',
               value_name='sbp')
display(long)
#   patient_id     visit  sbp
# 0       P001  baseline  152
# 1       P002  baseline  138
# 2       P003  baseline  145
# 3       P001  followup  138
# 4       P002  followup  132
# 5       P003  followup  129
```

The visit is now a value in its own column, ready for a later aggregation (Lecture 08) or a plot of SBP by visit. Survey responses stored in `Q1`, `Q2`, and `Q3` columns melt the same way when a downstream step needs the question name as a row value.

## Pivoting Long Back to Wide with pivot()

`pivot()` runs `melt()` backwards: the variable column supplies the new headers, the value column fills the cells, and the identifier column becomes the row labels. Reach for it when data arrives long, one row per observation the way `long_data` is, and a reader or a matrix-shaped tool wants a column per visit. `pivot()` never aggregates, so each `index`/`columns` pair must identify exactly one value.

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
# long_data was typed by hand and long came out of melt(): the same six readings
# in a different row order, which pivot() files by label into one frame.
wide = long_data.pivot(index='patient_id', columns='visit', values='sbp')
display(wide)
# visit       baseline  followup
# patient_id
# P001             152       138
# P002             138       132
# P003             145       129

print(wide.equals(long.pivot(index='patient_id', columns='visit', values='sbp')))
# True

back = long.pivot(index='patient_id', columns='visit', values='sbp').reset_index()
back.columns.name = None                 # drop the leftover 'visit' header label
display(back)
#   patient_id  baseline  followup
# 0       P001       152       138
# 1       P002       138       132
# 2       P003       145       129
print(back.equals(wide_data))
# True
```

Melting and then pivoting puts every reading back in the cell it came from; `reset_index()` restores the `patient_id` column, so the round trip reproduces `wide_data` exactly.

## When pivot() Finds a Repeated Pair

If an `index`/`columns` pair identifies more than one value, `pivot()` cannot choose a cell value and stops with `ValueError: Index contains duplicate entries, cannot reshape`. List the repeated pairs with `duplicated(subset=[...], keep=False)` (Lecture 05), then decide whether they are data errors or real repeated observations.

### Code Snippet: Find the pair that stops pivot()

```python
# P002's follow-up blood pressure was rechecked, so that pair appears twice
rechecked = pd.DataFrame({
    'patient_id': ['P001', 'P001', 'P002', 'P002', 'P002'],
    'visit': ['baseline', 'followup', 'baseline', 'followup', 'followup'],
    'sbp': [152, 138, 138, 148, 136],
})

display(rechecked[rechecked.duplicated(subset=['patient_id', 'visit'], keep=False)])
#   patient_id     visit  sbp
# 3       P002  followup  148
# 4       P002  followup  136

try:
    rechecked.pivot(index='patient_id', columns='visit', values='sbp')
except ValueError as error:
    print('ValueError:', error)
# ValueError: Index contains duplicate entries, cannot reshape

# Clinic rule: after a recheck, record the second reading
fixed = rechecked.drop_duplicates(subset=['patient_id', 'visit'], keep='last')
display(fixed.pivot(index='patient_id', columns='visit', values='sbp'))
# visit       baseline  followup
# patient_id
# P001             152       138
# P002             138       136
```

A recheck is a real repeated observation, so the fix is a documented rule, not a guess. If both readings should count, `pivot_table()` aggregates them into one cell instead, and the choice of `sum`, `mean`, or another function changes the question being answered. [BONUS.md](BONUS.md) shows that one call; aggregation and pivot tables are taught canonically in [Lecture 08](../08/README.md#pivot-tables-and-cross-tabulations).

If a reshape feels mysterious, write down what one row represents before choosing `pivot()` or `melt()`. Your future self will thank you for the labels.

# LIVE DEMO!

# Concatenating DataFrames Along an Axis

_Think of concatenation as stacking LEGO bricks: you can stack them vertically (add more rows) or horizontally (add more columns). Just make sure they fit together!_

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

The most common case stacks monthly extracts that share the same columns.

### Reference Card: Vertical `concat()`

| Call | Purpose | Result |
| --- | --- | --- |
| `pd.concat([df1, df2])` | Stack rows (`axis=0`) | Appended rows; indexes may repeat |
| `ignore_index=True` | Replace source indexes | Fresh `RangeIndex` |
| `piece['source_file'] = 'jan.csv'` on each table before stacking | Record where each row came from | An ordinary label column that survives the stack |
| `keys=['jan', 'feb']`, `names=['source', 'row']` | Label each piece with an outer index level instead of a column | Two-level row index; `reset_index()` turns the labels into columns |

### Code Snippet: Stack rows

```python
# Two monthly admission extracts; source_file records which file each row came from
jan = pd.DataFrame({
    'admission_id': ['A101', 'A102', 'A103'],
    'unit': ['ICU', 'Med', 'Surg'],
    'los_days': [4, 2, 3],                # length of stay
})
feb = pd.DataFrame({
    'admission_id': ['A201', 'A202'],
    'unit': ['Med', 'ICU'],
    'los_days': [5, 1],
})
jan['source_file'] = 'jan_admissions.csv'
feb['source_file'] = 'feb_admissions.csv'

# Stack them vertically: the rows of feb go under the rows of jan
combined = pd.concat([jan, feb])
display(combined)
#   admission_id  unit  los_days         source_file
# 0         A101   ICU         4  jan_admissions.csv
# 1         A102   Med         2  jan_admissions.csv
# 2         A103  Surg         3  jan_admissions.csv
# 0         A201   Med         5  feb_admissions.csv   <- index repeats (0, 1 again)
# 1         A202   ICU         1  feb_admissions.csv

# Clean indexes with ignore_index=True
combined = pd.concat([jan, feb], ignore_index=True)
display(combined)
#   admission_id  unit  los_days         source_file
# 0         A101   ICU         4  jan_admissions.csv
# 1         A102   Med         2  jan_admissions.csv
# 2         A103  Surg         3  jan_admissions.csv
# 3         A201   Med         5  feb_admissions.csv   <- clean sequential index
# 4         A202   ICU         1  feb_admissions.csv
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
a1c = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003'],
    'a1c': [7.4, 5.6, 6.1],
}).set_index('patient_id')

vitals = pd.DataFrame({
    'patient_id': ['P002', 'P003', 'P004'],
    'sbp': [128, 141, 119],
    'weight_kg': [81.5, 67.0, 90.2],
}).set_index('patient_id')

# Horizontal concatenation aligns the patient_id labels.
combined = pd.concat([a1c, vitals], axis=1)
display(combined)
#             a1c    sbp  weight_kg
# patient_id
# P001        7.4    NaN        NaN
# P002        5.6  128.0       81.5
# P003        6.1  141.0       67.0
# P004        NaN  119.0       90.2
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
# The February extract dropped 'age' and added 'payer'
jan = pd.DataFrame({
    'admission_id': ['A101', 'A102'],
    'age': [67, 45],
    'los_days': [4, 2],
})
feb = pd.DataFrame({
    'admission_id': ['A201', 'A202'],
    'los_days': [5, 1],
    'payer': ['Medicare', 'Private'],
})

# Outer (default): keeps every column; a column a file lacks is NaN for its rows
outer = pd.concat([jan, feb], join='outer', ignore_index=True)
display(outer)
#   admission_id   age  los_days     payer
# 0         A101  67.0         4       NaN
# 1         A102  45.0         2       NaN
# 2         A201   NaN         5  Medicare
# 3         A202   NaN         1   Private

# Inner: keeps only the columns both files share
inner = pd.concat([jan, feb], join='inner', ignore_index=True)
display(inner)
#   admission_id  los_days
# 0         A101         4
# 1         A102         2
# 2         A201         5
# 3         A202         1
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
# Weights from the primary record, plus a lower-priority intake form
primary = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003'],
    'weight_kg': [81.5, None, 67.0],
}).set_index('patient_id')
intake = pd.DataFrame({
    'patient_id': ['P002', 'P003', 'P004'],
    'weight_kg': [74.8, 66.1, 90.2],
}).set_index('patient_id')

complete = primary.combine_first(intake)
display(complete)
#             weight_kg
# patient_id
# P001             81.5   <- kept from the primary source
# P002             74.8   <- filled from the intake form
# P003             67.0   <- primary value wins over the intake value 66.1
# P004             90.2   <- label found only in the intake form
```

Confirm that row and column labels mean the same thing in both sources before combining them.

_“The data clearly shows that our hypothesis is correct, assuming we ignore all the data that doesn’t support our hypothesis.”_

# LIVE DEMO!
