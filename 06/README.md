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

**Live notebooks in Colab:** [Demo 1](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo1_merge_operations.ipynb) · [Demo 2](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo2_pivot_melt.ipynb) · [Demo 3](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo3_concat_combine.ipynb)

**Run locally:**

```shell
curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/06/demo/setup_demo.sh | sh
```

# Overview

## What This Lecture Covers

- **Database-style joins**: keys, cardinality, `pd.merge()`, the join types, overlapping columns, multi-column keys, and cross joins.
- **Reshaping**: wide and long formats with `melt()`, `pivot()`, and `pivot_table()`, two-level row labels, and `stack()` and `unstack()`.
- **Concatenation**: stacking rows and aligning columns with `concat()`, and patching gaps with `combine_first()`.

### Readings

McKinney, _Python for Data Analysis_ (3rd ed.):

- Chapter 8 (data wrangling: join, combine, and reshape)

## What You Will Be Able to Do

- Join a patient table to lab results with `pd.merge()`, and see which patients drop out under each join type with `indicator=True`.
- Catch a merge that duplicates rows by checking cardinality with `validate=`.
- Reshape wide blood-pressure visits to long with `melt()` and back with `pivot()`, and find the repeated pair that stops a pivot.
- Stack monthly admission extracts with `concat()`, record which file each row came from, and fill gaps from a second source with `combine_first()`.

# Database-Style DataFrame Joins

- **Database-style**: a **relational database** stores data as separate tables linked by shared key columns, and pandas borrows its vocabulary for combining them.
- **Join** (a **merge** in pandas): puts related rows from two tables side by side by matching a key.
- **Key**: a column, such as `patient_id`, whose values name the same thing in both tables.
- Health data rarely arrives in one table: asking whether older patients have higher A1c means joining patients to lab results first.

## Keys and Cardinality

- A table's **grain** is what one row represents (Lecture 05's row meaning): one patient, or one lab test.
- A **primary key** uniquely identifies each row of its own table: `patient_id` in `patients`.
- A **candidate key** is any column, or combination of columns, that could serve as the primary key; Lecture 05 called it a candidate identifier.
- A **foreign key** refers to another table's primary key: `patient_id` in `labs`, where one patient can appear many times.
- **Cardinality** says how many rows can match on each side, such as one patient to many lab results (**one-to-many**).

```
patients                    labs (A1c in %, LDL in mg/dL)
┌────────────┬────────────┐ ┌────────────┬──────┬───────┐
│ patient_id │ birth_year │ │ patient_id │ test │ value │
├────────────┼────────────┤ ├────────────┼──────┼───────┤
│ P001       │ 1958       │ │ P001       │ A1c  │   6.8 │
│ P002       │ 1971       │ │ P001       │ LDL  │ 131.0 │
│ P003       │ 1985       │ │ P002       │ A1c  │   5.4 │
└────────────┴────────────┘ │ P004       │ A1c  │   7.9 │
                            └────────────┴──────┴───────┘
```

_Join keys are the table's name tags: if two rows share a tag, pandas brings their columns together. A good key makes matching boring; a bad key turns the merge into an enthusiastic photocopier._

![xkcd 2801: Contact Merge. A shared key connects records, so verify what it identifies](media/xkcd_2801.png)

## The Basics of pd.merge()

`pd.merge(left, right, on='key')` pairs each row of `left` with every row of `right` that has the same key value, and returns a new DataFrame.

### Reference Card: `merge()` keys and options

| Argument | Purpose | Typical output |
| --- | --- | --- |
| `left`, `right` | Input tables; `left.merge(right, on=...)` is the same call written as a DataFrame method | Combined `DataFrame` |
| `on='key'` or `on=['k1', 'k2']` | Shared key column, or several that must all match; without `on=`, every shared column name becomes part of the key | Key-based matches |
| `left_on`, `right_on` | Different key names | Matches equivalent fields |
| `how` | `inner` (the default), `left`, `right`, `outer`, or `cross` | Controls retained rows |
| `validate` / `indicator` | Check cardinality / match status | `MergeError` or `_merge` audit column |
| `suffixes` | Rename overlapping non-key columns | Unambiguous columns |

### Code Snippet: Merge on a key

```python
merged = pd.merge(patients, labs, on='patient_id')
display(merged)
```

|  | patient_id | birth_year | test | value |
| --- | --- | --- | --- | --- |
| 0 | P001 | 1958 | A1c | 6.8 |
| 1 | P001 | 1958 | LDL | 131.0 |
| 2 | P002 | 1971 | A1c | 5.4 |

### Code Snippet: Match differently named keys

`referrals` calls its patient key `registry_id`; `patients` uses `patient_id`:

```text
referrals
registry_id  service
P002         cardiology
```

```python
display(pd.merge(referrals, patients, left_on='registry_id', right_on='patient_id'))
```

|  | registry_id | service | patient_id | birth_year |
| --- | --- | --- | --- | --- |
| 0 | P002 | cardiology | P002 | 1971 |

Both key columns remain because their names differ.

## Join Types: The Four Horsemen of Data Merging (Plus One)

When a key appears in only one table, the join type decides whether that row survives. The fifth rider, `cross`, ignores keys and pairs every left row with every right row.

Every row either table can contribute, and the join types (`how=`) that keep it:

| patient_id | birth_year | test | value | Why | `inner` | `left` | `right` | `outer` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P001 | 1958 | A1c | 6.8 | Key in both tables | ✓ | ✓ | ✓ | ✓ |
| P001 | 1958 | LDL | 131.0 | Key in both tables | ✓ | ✓ | ✓ | ✓ |
| P002 | 1971 | A1c | 5.4 | Key in both tables | ✓ | ✓ | ✓ | ✓ |
| P003 | 1985 | NaN | NaN | Patient with no labs yet |  | ✓ |  | ✓ |
| P004 | NaN | A1c | 7.9 | Lab with no patient record |  |  | ✓ | ✓ |

Add `indicator=True` to any merge to get a `_merge` column that says where each row came from: `both`, `left_only`, or `right_only`.

### Reference Card: Join types

| `how` | Retains | Unmatched side |
| --- | --- | --- |
| `inner` | Matching keys only | Dropped |
| `left` | Every left row | Right fields become `NaN` |
| `right` | Every right row | Left fields become `NaN` |
| `outer` | Every row from both | Missing fields become `NaN` |
| `cross` | Every left row paired with every right row; takes no `on=` | Nothing to match; `len(left) × len(right)` rows |

### Code Snippet: Compare join types

```python
inner = pd.merge(patients, labs, on='patient_id', how='inner')
print(len(inner))    # 3: P001 twice, P002 once

right = pd.merge(patients, labs, on='patient_id', how='right')
print(len(right))    # 4: includes P004's A1c with birth_year NaN

left = pd.merge(patients, labs, on='patient_id', how='left')
display(left)

outer = pd.merge(patients, labs, on='patient_id', how='outer', indicator=True)
display(outer)
```

|  | patient_id | birth_year | test | value |
| --- | --- | --- | --- | --- |
| 0 | P001 | 1958 | A1c | 6.8 |
| 1 | P001 | 1958 | LDL | 131.0 |
| 2 | P002 | 1971 | A1c | 5.4 |
| 3 | P003 | 1985 | NaN | NaN |

|  | patient_id | birth_year | test | value | _merge |
| --- | --- | --- | --- | --- | --- |
| 0 | P001 | 1958.0 | A1c | 6.8 | both |
| 1 | P001 | 1958.0 | LDL | 131.0 | both |
| 2 | P002 | 1971.0 | A1c | 5.4 | both |
| 3 | P003 | 1985.0 | NaN | NaN | left_only |
| 4 | P004 | NaN | A1c | 7.9 | right_only |

`birth_year` prints as `1958.0` in the outer join because P004's missing birth year makes the whole column `float64` (Lecture 05).

<callout icon="⚠️" color="yellow_bg">
	## `merge()` is an inner join unless you say otherwise!
	Unmatched rows vanish without a warning, as P003 does above. Use `how='left'` when the left table is your master list, and compare row counts before and after.
</callout>

_“The data clearly shows that our hypothesis is correct, assuming we ignore all the data that doesn’t support our hypothesis.”_

## Handling Overlapping Column Names

When both tables have a non-key column with the same name, `merge()` keeps both and adds the suffixes `_x` (left table) and `_y` (right table).

### Reference Card: Overlapping columns

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| Default suffixes (`_x`, `_y`) | Distinguish overlapping columns | `a1c_x` from the left table, `a1c_y` from the right |
| `suffixes=('_lab', '_poc')` | Say what each column means | Named output columns |

### Code Snippet: Name overlapping columns

`lab` holds central-lab A1c results and `poc` holds point-of-care device results for the same patients:

```text
lab                 poc
patient_id  a1c     patient_id  a1c
P001        7.4     P001        7.1
P002        5.6     P002        5.9
```

```python
display(pd.merge(lab, poc, on='patient_id'))
display(pd.merge(lab, poc, on='patient_id', suffixes=('_lab', '_poc')))
```

|  | patient_id | a1c_x | a1c_y |
| --- | --- | --- | --- |
| 0 | P001 | 7.4 | 7.1 |
| 1 | P002 | 5.6 | 5.9 |

|  | patient_id | a1c_lab | a1c_poc |
| --- | --- | --- | --- |
| 0 | P001 | 7.4 | 7.1 |
| 1 | P002 | 5.6 | 5.9 |

![xkcd 1459: Documents. Name overlapping columns with suffixes=, not a1c_x and a1c_y](media/xkcd_1459.png)

## Merging on Multiple Columns

- **Composite key**: several columns that together name a row, such as `patient_id` and `visit` (Lecture 05's `patient_id` + `visit_date`).
- Pass a list, `on=['patient_id', 'visit']`; every listed column must match.

```text
vitals (sbp in mmHg)            a1c (a1c in %)
patient_id  visit     sbp       patient_id  visit     a1c
P001        baseline  152       P001        baseline  7.4
P001        followup  138       P001        followup  6.9
P002        baseline  128       P002        baseline  5.6
P002        followup  124       P002        followup  5.5
```

### Code Snippet: Merge on multiple columns

```python
wrong = pd.merge(vitals, a1c, on='patient_id')
print(len(wrong))            # 8: each visit's SBP now sits next to both of that patient's A1c values
print(list(wrong.columns))   # ['patient_id', 'visit_x', 'sbp', 'visit_y', 'a1c']: _x and _y mark the clashing visit columns

merged = pd.merge(vitals, a1c, on=['patient_id', 'visit'])
display(merged)
```

|  | patient_id | visit | sbp | a1c |
| --- | --- | --- | --- | --- |
| 0 | P001 | baseline | 152 | 7.4 |
| 1 | P001 | followup | 138 | 6.9 |
| 2 | P002 | baseline | 128 | 5.6 |
| 3 | P002 | followup | 124 | 5.5 |

## Listing Every Combination with a Cross Join

- A report that never arrived leaves no row, so no keyed join can flag it.
- Build the **expected grid** of every combination, such as every clinic at every hour, with `how='cross'`, then left-merge the observed reports onto it: each missing report shows up as `left_only`.

### Code Snippet: Find missing reports with a cross join

`clinic_list` has North and South; `hours` has 8, 9, and 10. `reports` holds the four hourly reports that arrived:

```text
clinic  hour  arrivals
North   8     3
North   10    0
South   8     5
South   9     2
```

```python
expected = pd.merge(clinic_list, hours, how='cross')
print(len(expected))   # 6: 2 clinics × 3 hours

coverage = pd.merge(expected, reports, on=['clinic', 'hour'], how='left', indicator=True)
display(coverage)
```

|  | clinic | hour | arrivals | _merge |
| --- | --- | --- | --- | --- |
| 0 | North | 8 | 3.0 | both |
| 1 | North | 9 | NaN | left_only |
| 2 | North | 10 | 0.0 | both |
| 3 | South | 8 | 5.0 | both |
| 4 | South | 9 | 2.0 | both |
| 5 | South | 10 | NaN | left_only |

North's 9:00 `NaN` means no report arrived; its 10:00 `0.0` is a real report of zero arrivals.

## Checking Merge Cardinality

<callout icon="⚠️" color="yellow_bg">
	## A repeated key copies rows without an error!
	Every row that matches a repeated key in a **lookup table** (one row of details per key, such as clinics) is copied once per repeat. Row growth alone proves nothing, because a one-to-many merge adds rows too, so state the expected relationship with `validate=`.
</callout>

### Reference Card: Merge cardinality and audit

| Contract | Meaning | Check |
| --- | --- | --- |
| One-to-one | Unique keys on both sides | `validate='one_to_one'` |
| One-to-many | Unique left keys; repeated right keys allowed | `validate='one_to_many'` |
| Many-to-one | Repeated left keys; unique right keys | `validate='many_to_one'` |
| Many-to-many | Repeated keys on both sides | `validate='many_to_many'` documents, not constrains |
| Broken contract | A side you declared unique has repeats | Raises `pd.errors.MergeError`; catch it with `try`/`except` (Lecture 02) |
| Find the repeats | Every row whose key appears more than once | `df[df.duplicated(subset=['key'], keep=False)]` (Lecture 05); list every key column, such as `subset=['k1', 'k2']`, for a composite key |
| Missing keys | Missing (`NA`) keys match each other: two missing-key rows on each side make four false matches | `df['key'].notna().all()` on both tables before merging; this does not reject empty strings or whitespace, which need separate cleaning |
| Match audit | Show `left_only`, `right_only`, `both` | `indicator=True` |

### Code Snippet: Catch a Broken Merge Contract

Each visit should get exactly one clinic, but the lookup lists K2 twice:

```text
visits                   clinics
visit_id  clinic_id      clinic_id  clinic_name      record_status
V1        K1             K1         Mission Bay      current
V2        K2             K2         Parnassus        retired
V3        K2             K2         Parnassus Annex  current
V4        K9
```

```python
pd.merge(visits, clinics, on='clinic_id', how='left', validate='many_to_one')
```

- Expected error: `MergeError: Merge keys are not unique in right dataset; not a many-to-one merge`.
- Without `validate=`, four visits become six rows: V2 and V3 each match twice.
- Fix: list the repeated keys, apply a documented lookup rule (here, keep `record_status == 'current'`), and merge again with `validate=` and `indicator=True`: four rows, with V4's K9 `left_only`.

```python
current = clinics[clinics['record_status'] == 'current']
checked = pd.merge(visits, current, on='clinic_id', how='left',
                   validate='many_to_one', indicator=True)
display(checked[['visit_id', 'clinic_id', 'clinic_name', '_merge']])
```

|  | visit_id | clinic_id | clinic_name | _merge |
| --- | --- | --- | --- | --- |
| 0 | V1 | K1 | Mission Bay | both |
| 1 | V2 | K2 | Parnassus Annex | both |
| 2 | V3 | K2 | Parnassus Annex | both |
| 3 | V4 | K9 | NaN | left_only |

# LIVE DEMO!

[Open Demo 1 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo1_merge_operations.ipynb)

# Reshaping: Wide vs Long Format

- **Reshaping**: changing the layout of a table. A lossless reshape preserves every measurement; **aggregation** reduces several values to one summary and cannot recover the originals.
- **Wide**: measurements of one kind spread across columns, such as a systolic blood pressure (SBP) column per visit.
- **Long**: one row per measurement, such as one row per patient-visit; grouping and plotting tools usually want this layout.

```
wide: a column per visit                  long: a row per patient-visit
┌────────────┬──────────┬──────────┐      ┌────────────┬──────────┬─────┐
│ patient_id │ baseline │ followup │      │ patient_id │ visit    │ sbp │
├────────────┼──────────┼──────────┤      ├────────────┼──────────┼─────┤
│ P001       │   152    │   138    │      │ P001       │ baseline │ 152 │
│ P002       │   138    │   132    │      │ P002       │ baseline │ 138 │
│ P003       │   145    │   129    │      │ P003       │ baseline │ 145 │
└────────────┴──────────┴──────────┘      │ P001       │ followup │ 138 │
                                          │ P002       │ followup │ 132 │
                                          │ P003       │ followup │ 129 │
                                          └────────────┴──────────┴─────┘
```

- In long data, the **identifier columns** (`patient_id`, `visit`) say which observation a row is, and the **value column** (`sbp`) holds the measurement.
- Long data with one observation per row and one variable per column is often called **tidy** data.

| Shape | One row represents | Best for | Conversion |
| --- | --- | --- | --- |
| Wide | One entity with several measured columns | Side-by-side comparison | `melt()` to long |
| Long | One entity/variable observation | Grouping and tidy plotting | `pivot()` to wide |

## Melting Wide to Long with melt()

`melt()`, written `pd.melt(df, ...)` or `df.melt(...)`, keeps the identifier columns fixed and turns each selected column into rows: the old column name goes into a variable column and its cell into a value column. Questionnaire answers stored in `Q1`, `Q2`, and `Q3` columns melt the same way, with the question name as the variable.

### Reference Card: `melt()`

| Argument | Purpose | Output |
| --- | --- | --- |
| `id_vars` | Columns to keep fixed | Identifier columns repeated |
| `value_vars` | Columns to unpivot | One row per source row and selected value column |
| `var_name` | Name for former column labels | Variable column |
| `value_name` | Name for former cell values | Value column |

### Code Snippet: Melt wide data

```python
long = pd.melt(wide, id_vars=['patient_id'], value_vars=['baseline', 'followup'],
               var_name='visit', value_name='sbp')
display(long)
```

|  | patient_id | visit | sbp |
| --- | --- | --- | --- |
| 0 | P001 | baseline | 152 |
| 1 | P002 | baseline | 138 |
| 2 | P003 | baseline | 145 |
| 3 | P001 | followup | 138 |
| 4 | P002 | followup | 132 |
| 5 | P003 | followup | 129 |

## Pivoting Long Back to Wide with pivot()

- `pivot()` runs `melt()` backwards, for a reader or a comparison of baseline with follow-up that needs one column per visit.
- It files each value by its labels, so the long rows may come in any order.
- It never aggregates, so each `index`/`columns` pair must identify exactly one value.

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
back = long.pivot(index='patient_id', columns='visit', values='sbp')
display(back)

back = back.reset_index()
back.columns.name = None   # drop the leftover 'visit' header label
display(back)
print(back.equals(wide))   # True: the round trip rebuilt the wide table exactly
```

| visit | baseline | followup |
| --- | --- | --- |
| **patient_id** |  |  |
| P001 | 152 | 138 |
| P002 | 138 | 132 |
| P003 | 145 | 129 |

|  | patient_id | baseline | followup |
| --- | --- | --- | --- |
| 0 | P001 | 152 | 138 |
| 1 | P002 | 138 | 132 |
| 2 | P003 | 145 | 129 |

![xkcd 2313: Wrong Times Table. A grid can look right and still be wrong, so check the round trip](media/xkcd_2313.png)

## When pivot() Finds a Repeated Pair

If an `index`/`columns` pair identifies more than one value, `pivot()` cannot choose a cell value and stops with `ValueError: Index contains duplicate entries, cannot reshape`. List the repeated pairs with `duplicated(subset=[...], keep=False)` (Lecture 05), then decide whether they are data errors or real repeated observations.

### Code Snippet: Find the Pair that Stops a Pivot

`rechecked` holds one baseline reading per patient and two follow-up readings for P002, so `rechecked.pivot(index='patient_id', columns='visit', values='sbp')` raises the repeated-pair error:

```python
display(rechecked[rechecked.duplicated(subset=['patient_id', 'visit'], keep=False)])
```

|  | patient_id | visit | sbp | reading_time |
| --- | --- | --- | --- | --- |
| 3 | P002 | followup | 148 | 10:15 |
| 4 | P002 | followup | 136 | 10:22 |

- A recheck is a real repeated observation, so the fix is a documented rule, not a guess: to keep the later recheck, `sort_values('reading_time')`, then `drop_duplicates(subset=['patient_id', 'visit'], keep='last')`.
- If both readings should count, `pivot_table()` aggregates them into one cell instead, and the choice of `sum`, `mean`, or another function changes the question being answered.

### Code Snippet: Summarize repeated readings with `pivot_table()`

For just the two P002 rechecks shown above, stored in `repeat_readings`, the mean is `(148 + 136) / 2 = 142` mmHg.

```python
display(repeat_readings.pivot_table(index='patient_id', columns='visit',
                                   values='sbp', aggfunc='mean'))
```

| visit | followup |
| --- | --- |
| **patient_id** |  |
| P002 | 142.0 |

`aggfunc='mean'` is the default; naming it records the decision. This answers “What was the average reading?”, while keeping the later recheck answers “What reading did the protocol record?” Aggregation is not lossless: the mean alone cannot recover the two readings.

![xkcd 1762: Moving Boxes. A good row label says what is inside, like a patient ID](media/xkcd_1762.png)

## Two-Level Row Labels

- The index, as in Lecture 04, holds the row labels that `loc` uses to find rows.
- In long data one column often cannot name a row: after `melt()` a patient has a row per visit, and a clinic with one visit count per quarter needs a clinic _and_ a quarter.
- **MultiIndex** (hierarchical index): row labels with more than one level, such as a `(clinic, quarter)` pair. The outer label prints once, with the inner labels under it.
- Build one by passing `set_index()` a list of columns.

### Reference Card: Two-level row labels

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `df.set_index(['col1', 'col2'])` | Build two-level row labels from two columns | DataFrame with a `MultiIndex` |
| `df.loc[('key1', 'key2'), :]` | Select a row by both parts of its label; `df.loc['key1']` selects every row under one outer label | Selected row or rows |
| `df.reset_index()` | Turn both levels back into columns | New `DataFrame` with a default `RangeIndex` |
| `df.index.is_unique` | Test that all row labels are unique; an index can repeat a label | `True` when no labels repeat; otherwise `False` |

### Code Snippet: Select a two-part row label

`quarterly` has one `visits` count per clinic (North, South) and quarter (Q1, Q2):

|  | clinic | quarter | visits |
| --- | --- | --- | --- |
| 0 | North | Q1 | 410 |
| 1 | North | Q2 | 395 |
| 2 | South | Q1 | 380 |
| 3 | South | Q2 | 362 |

```python
summary = quarterly.set_index(['clinic', 'quarter'])
display(summary)
print(summary.loc[('South', 'Q2'), 'visits'])  # 362
display(summary.loc['South'])
```

| clinic | quarter | visits |
| --- | --- | --- |
| North | Q1 | 410 |
|  | Q2 | 395 |
| South | Q1 | 380 |
|  | Q2 | 362 |

The `loc` line prints `362`; `loc['South']` shows the two South rows:

| quarter | visits |
| --- | --- |
| Q1 | 380 |
| Q2 | 362 |

## Moving Index Levels with stack() and unstack()

- **unstack()** moves an index level from the rows to the columns: each quarter becomes its own column.
- **stack()** does the reverse, moving the column labels down into an inner row level.
- Both change the layout only; every value keeps its clinic and quarter labels.
- By default each moves the innermost level; `level=` names a different one.

```text
stacked: summary['visits']            unstacked: summary['visits'].unstack()
clinic  quarter                       quarter  Q1   Q2
North   Q1        410                 clinic
        Q2        395                 North    410  395
South   Q1        380                 South    380  362
        Q2        362
```

### Reference Card: `stack()` and `unstack()`

| Call | Purpose | Output / note |
| --- | --- | --- |
| `s.unstack()` | Move the inner row level to columns | `DataFrame`: one row per outer label, one column per inner label |
| `s.unstack(level='clinic')` | Move a named level (or a number, `0` for outer) instead | Clinics become the columns |
| `df.stack()` | Move the column labels into an inner row level | `Series` with a two-level index |
| Missing combination | A pair absent from the data | `unstack()` fills `NaN`; `stack()` keeps that `NaN` row, so add `.dropna()` to remove it |

### Code Snippet: Unstack quarters into columns

```python
by_quarter = summary['visits'].unstack()
display(by_quarter)
display(summary['visits'].unstack(level='clinic'))
print(by_quarter.stack().equals(summary['visits']))   # True: stack() undoes unstack()
```

| quarter | Q1 | Q2 |
| --- | --- | --- |
| **clinic** |  |  |
| North | 410 | 395 |
| South | 380 | 362 |

| clinic | North | South |
| --- | --- | --- |
| **quarter** |  |  |
| Q1 | 410 | 380 |
| Q2 | 395 | 362 |

`pivot(index='clinic', columns='quarter', values='visits')` gives the same layout as `set_index(['clinic', 'quarter'])['visits'].unstack()`. Both preserve each clinic-quarter value; `melt()` similarly moves values from ordinary columns into rows.

# LIVE DEMO!

[Open Demo 2 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo2_pivot_melt.ipynb)

# Concatenating DataFrames Along an Axis

- **Concatenation**: gluing together tables that are pieces of one table, such as monthly admission extracts with the same columns.
- **Axis**: `axis=0` stacks rows; `axis=1` puts columns side by side (Lecture 03).
- `concat()` does not match key values the way `merge()` does. Stacking (`axis=0`) aligns columns by name; side by side (`axis=1`) aligns rows by index label.

```
VERTICAL CONCATENATION (axis=0)
DataFrame A:      DataFrame B:      Result:
┌─────────┐       ┌─────────┐       ┌─────────┐
│ A │ B   │   +   │ A │ B   │   =   │ A │ B   │
├─────────┤       ├─────────┤       ├─────────┤
│ 1 │ 2   │       │ 5 │ 6   │       │ 1 │ 2   │
│ 3 │ 4   │       │ 7 │ 8   │       │ 3 │ 4   │
└─────────┘       └─────────┘       │ 5 │ 6   │
                                    │ 7 │ 8   │
                                    └─────────┘

HORIZONTAL CONCATENATION (axis=1): rows line up by index label
DataFrame A:      DataFrame B:
┌───────────┐     ┌───────────┐
│   │ A │ B │     │   │ C │ D │
├───────────┤     ├───────────┤
│ 0 │ 1 │ 2 │  +  │ 1 │ 5 │ 6 │
│ 1 │ 3 │ 4 │     │ 2 │ 7 │ 8 │
└───────────┘     └───────────┘
Result (outer join keeps every label):
┌───────────────────┐
│   │ A │ B │ C │ D │
├───────────────────┤
│ 0 │ 1 │ 2 │NaN│NaN│
│ 1 │ 3 │ 4 │ 5 │ 6 │
│ 2 │NaN│NaN│ 7 │ 8 │
└───────────────────┘
```

## Vertical Concatenation: Adding More Rows

### Code Snippet: Count each extract's lines with `%%bash`

Lecture 04's `!command` runs one shell line; a cell whose first line is `%%bash` (a **cell magic**) runs the whole cell in the shell. `wc -l` counts the header too, so each extract has one row fewer than its count:

```python
%%bash
wc -l data/*_admissions.csv
```

```text
 3 data/feb_admissions.csv
 4 data/jan_admissions.csv
 7 total
```

### Reference Card: Vertical `concat()`

| Call | Purpose | Result |
| --- | --- | --- |
| `pd.concat([df1, df2])` | Stack rows (`axis=0`) | Appended rows; indexes may repeat |
| `ignore_index=True` | Replace source indexes | Fresh `RangeIndex` |
| `piece['source_file'] = 'jan.csv'` on each table before stacking | Record where each row came from, its provenance (Lecture 05) | An ordinary label column that survives the stack |
| `keys=['jan', 'feb']`, `names=['source', 'row']` | Label each piece with an outer index level instead of a column | Two-level row index; `reset_index()` turns the labels into columns |

### Code Snippet: Stack rows

`jan` holds three January admissions and `feb` two February ones, with the same columns (`los_days` is the length of stay):

```text
jan                         feb
admission_id  unit  los_days admission_id  unit  los_days
A101          ICU   4        A201          Med   5
A102          Med   2        A202          ICU   1
A103          Surg  3
```

```python
jan['source_file'] = 'jan_admissions.csv'
feb['source_file'] = 'feb_admissions.csv'
combined = pd.concat([jan, feb])
display(combined)

combined = pd.concat([jan, feb], ignore_index=True)
print(list(combined.index))   # [0, 1, 2, 3, 4]: a clean sequential index
```

|  | admission_id | unit | los_days | source_file |
| --- | --- | --- | --- | --- |
| 0 | A101 | ICU | 4 | jan_admissions.csv |
| 1 | A102 | Med | 2 | jan_admissions.csv |
| 2 | A103 | Surg | 3 | jan_admissions.csv |
| 0 | A201 | Med | 5 | feb_admissions.csv |
| 1 | A202 | ICU | 1 | feb_admissions.csv |

![xkcd 2502: Every Data Table. A source column records where each stacked row came from](media/xkcd_2502.png)

## Column-Set Alignment with the join Parameter

When stacked extracts do not share every column, the `join=` argument of `concat()` chooses which columns to keep.

### Reference Card: `concat(join=...)`

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `join='outer'` (default) | Keep all columns from both frames | Union of columns; `NaN` where a piece lacks a column |
| `join='inner'` | Keep only columns shared by all inputs | Stacked `DataFrame` with the column intersection |

### Code Snippet: Choose a column set

This `jan` and `feb` are two admission extracts whose columns changed: February dropped `age` and added `payer`.

```text
jan                            feb
admission_id  age  los_days    admission_id  los_days  payer
A101          67   4           A201          5         Medicare
A102          45   2           A202          1         Private
```

```python
display(pd.concat([jan, feb], join='outer', ignore_index=True))
display(pd.concat([jan, feb], join='inner', ignore_index=True))
```

|  | admission_id | age | los_days | payer |
| --- | --- | --- | --- | --- |
| 0 | A101 | 67.0 | 4 | NaN |
| 1 | A102 | 45.0 | 2 | NaN |
| 2 | A201 | NaN | 5 | Medicare |
| 3 | A202 | NaN | 1 | Private |

|  | admission_id | los_days |
| --- | --- | --- |
| 0 | A101 | 4 |
| 1 | A102 | 2 |
| 2 | A201 | 5 |
| 3 | A202 | 1 |

## Horizontal Concatenation: Adding More Columns

Separate systems can report different variables for the same patients. Put the patient identifier in each index first: `axis=1` uses those labels to line up measurements, even when the rows arrive in different orders.

### Reference Card: Horizontal `concat()`

| Call / rule | Purpose | Result |
| --- | --- | --- |
| `pd.concat([df1, df2], axis=1)` | Add columns | Index labels align rows |
| Shared index labels | Define row identity | Matching labels share a row |
| Missing labels | Preserve unmatched rows | `NaN` in absent fields |
| `axis=1, join='inner'` | Keep only row labels found in every input | Side-by-side `DataFrame` without unmatched rows |

### Code Snippet: Align columns by index

`a1c` and `vitals` both use `patient_id` as their index, and only P002 and P003 appear in both:

```text
a1c                 vitals
patient_id  a1c     patient_id  sbp  weight_kg
P001        7.4     P002        128  81.5
P002        5.6     P003        141  67.0
P003        6.1     P004        119  90.2
```

```python
combined = pd.concat([a1c, vitals], axis=1)
display(combined)
```

| patient_id | a1c | sbp | weight_kg |
| --- | --- | --- | --- |
| P001 | 7.4 | NaN | NaN |
| P002 | 5.6 | 128.0 | 81.5 |
| P003 | 6.1 | 141.0 | 67.0 |
| P004 | NaN | 119.0 | 90.2 |

<callout icon="⚠️" color="yellow_bg">
	## `axis=1` lines rows up by label, not by position!
	Two tables loaded separately both number their rows `0, 1, 2, …`, so `axis=1` pairs their unrelated first rows. `set_index()` the real identifier first, or `merge()` on the key column instead.
</callout>

## Patching Gaps with combine_first()

`primary.combine_first(fallback)` patches the gaps in a trusted source from a lower-priority one that holds the same variables, such as a primary weight record and an intake form.

### Reference Card: `combine_first()`

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `df1.combine_first(df2)` | Fill gaps using the same row and column labels in df2 | New `DataFrame`; non-null df1 values win |
| Row and column coverage | Keep the union of both sets of labels | Rows or columns found only in df2 are included |
| Not a stack | Patches values of the same observations from a second source | To add another period's observations, use `concat()` |
| Matching labels | `P002` and `weight_kg` must mean the same patient and unit in both sources | Check before combining; pandas matches names, not meaning |

### Code Snippet: Patch from a fallback source

`primary` and `intake` both use `patient_id` as their index; the primary record lacks P002's weight:

```text
primary                   intake
patient_id  weight_kg     patient_id  weight_kg
P001        81.5          P002        74.8
P002        NaN           P003        66.1
P003        67.0          P004        90.2
```

```python
complete = primary.combine_first(intake)
display(complete)
```

| patient_id | weight_kg |
| --- | --- |
| P001 | 81.5 |
| P002 | 74.8 |
| P003 | 67.0 |
| P004 | 90.2 |

P002 is filled from the intake form, P003 keeps the primary 67.0 over the intake's 66.1, and P004 comes only from the intake form.

# LIVE DEMO!

[Open Demo 3 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/06/demo/demo3_concat_combine.ipynb)
