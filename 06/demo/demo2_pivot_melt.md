---
jupyter:
  jupytext:
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    name: python
    version: 3.13
---

# Demo 2: Row Labels and Reshaping Blood-Pressure Visits

- A small hypertension study records systolic blood pressure (SBP, mmHg) at baseline, week 4, and week 12.
- Give the table meaningful row labels, including two-level ones.
- Reshape it from wide to long and back with `melt()`, `unstack()`, `stack()`, and `pivot()`.
- Fix the repeated reading that stops `pivot()`.

Run the cells from top to bottom; each **Expect** line says what you should see.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
```

## Core walkthrough

### 1. Load the study table (wide)

This supplied cell keeps any Parquet file already in `data/` and downloads the rest (`urlretrieve(url, path)` saves the file at a web address to `path`; `data_dir.glob("demo2_*.parquet")` lists the matching files).

```python
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/06/demo/data"
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
for filename in ["demo2_bp_wide.parquet", "demo2_device_week4.parquet"]:
    path = data_dir / filename
    if not path.exists():
        urlretrieve(f"{REPO_RAW}/{filename}", path)
print(sorted(p.name for p in data_dir.glob("demo2_*.parquet")))
```

**Expect:** `['demo2_bp_wide.parquet', 'demo2_device_week4.parquet']`.

One row per patient, one SBP column per visit. `ids` lists the columns that describe a patient and do not change between visits.

```python
bp_wide = pd.read_parquet(data_dir / "demo2_bp_wide.parquet")
ids = ['patient_id', 'age', 'sex', 'clinic']
print(bp_wide.shape)
display(bp_wide)
```

**Expect:** `(4, 7)`: patients P001 to P004, and the columns `patient_id`, `age`, `sex`, `clinic`, `baseline`, `week_04`, `week_12`.

### 2. Patient IDs as row labels

As in Lecture 04, `set_index()` turns a column into row labels.

```python
bp = bp_wide.set_index('patient_id')
print(bp.index.is_unique)
print(bp.loc['P003', 'week_12'])
```

**Expect:** `True` (each patient appears once), then `129`.

### 3. Wide to long with `melt()`

Plotting SBP over time and grouping by visit both want one row per patient-visit.

```python
bp_long = bp_wide.melt(
    id_vars=ids,
    value_vars=['baseline', 'week_04', 'week_12'],
    var_name='visit',
    value_name='sbp',
)
print(bp_long.shape)
display(bp_long.head(6))
```

**Expect:** `(12, 6)`: 4 patients × 3 visits = 12 rows, with columns `patient_id`, `age`, `sex`, `clinic`, `visit`, `sbp`. The first four rows are every patient's `baseline`, then the `week_04` rows begin.

### 4. Two-level row labels

In the long table one patient has three rows, so `patient_id` alone no longer names a row; `patient_id` and `visit` together do.

```python
by_visit = bp_long.set_index(['patient_id', 'visit']).sort_index()
print(by_visit.index.is_unique)
display(by_visit.loc['P003'])
print(by_visit.loc[('P003', 'week_12'), 'sbp'])
```

**Expect:** `True`, then P003's three visits (`baseline`, `week_04`, `week_12`) with SBP 147, 140, and 129, then `129`. The labels sort alphabetically, which here is also time order.

### 5. Move a level with `unstack()` and `stack()`

`unstack()` moves the inner level, `visit`, up into the columns: one row per patient again.

```python
sbp_by_visit = by_visit['sbp'].unstack()
display(sbp_by_visit)
display(by_visit['sbp'].unstack(level='patient_id'))
```

**Expect:** first a 4 × 3 table with patients as rows and `baseline`, `week_04`, `week_12` as columns (P001: 152, 144, 138). Then the same numbers turned the other way: visits as rows, P001 to P004 as columns.

`stack()` moves the visit columns back down into the inner row level.

```python
print(sbp_by_visit.stack().equals(by_visit['sbp']))
```

**Expect:** `True`: no value was lost or moved to another patient.

### 6. Long back to wide with `pivot()`

`pivot()` does `set_index()` and `unstack()` in one call: `visit` supplies the headers and `sbp` fills the cells. With four identifier columns in `ids`, the result has four-level row labels, so `reset_index()` turns them back into columns, and `columns.name = None` drops the leftover `visit` header label.

```python
bp_wide_again = bp_long.pivot(
    index=ids,
    columns='visit',
    values='sbp',
).reset_index()
bp_wide_again.columns.name = None
print(bp_wide_again.equals(bp_wide))
display(bp_wide_again)
```

**Expect:** `True`: the round trip reproduces the original table exactly, the same 4 rows, 7 columns, values, and dtypes.

### 7. When `pivot()` finds a repeated pair

The BP cuff's own export for the week 4 visit arrives.

```python
device_week4 = pd.read_parquet(data_dir / "demo2_device_week4.parquet")
display(device_week4)
```

**Expect:** 5 rows for 4 patients; P003 appears twice.

**Intentional error:** one `patient_id`/`visit` pair has two readings, so `pivot()` cannot choose a cell value.

```python
try:
    device_week4.pivot(index='patient_id', columns='visit', values='sbp')
except ValueError as error:
    print('ValueError:', error)
```

**Expect:** `ValueError: Index contains duplicate entries, cannot reshape`.

List every row in the repeated set before deciding what to do.

```python
display(device_week4[device_week4.duplicated(subset=['patient_id', 'visit'], keep=False)])
```

**Expect:** 2 rows for P003: 152 at 10:15 and 140 at 10:22. The readings differ, so this is not a double entry: the nurse rechecked P003's pressure after a rest.

**The fix:** the clinic's protocol records the recheck, the later reading. Sort by time, keep the last row of each pair, and pivot again.

```python
recorded = (device_week4.sort_values('reading_time')
            .drop_duplicates(subset=['patient_id', 'visit'], keep='last'))
print(len(recorded))
display(recorded.pivot(index='patient_id', columns='visit', values='sbp'))
```

**Expect:** `4` rows, then a one-column table (`week_04`) with P001 144, P002 135, P003 140, and P004 136. These match the `week_04` column of `bp_wide`, so the study table recorded the recheck too.

If both readings should count, `pivot_table()` would average them instead.

## Independent practice

Continue on your own after class. These cells reuse the core results; if the runtime closed, run the cells above again first.

### 8. Visit numbers instead of labels

The visit labels are text. A number of weeks is easier to plot and sort, so look each label up in a dictionary with `.map()` (Lecture 05).

```python
weeks = {'baseline': 0, 'week_04': 4, 'week_12': 12}
bp_long['week'] = bp_long['visit'].map(weeks)
print(bp_long['week'].isna().sum())
display(bp_long.sort_values(['patient_id', 'week']).head(3))
```

**Expect:** `0` unmapped labels, then P001's three rows in time order: 152 at week 0, 144 at week 4, and 138 at week 12.

### 9. A missed visit after `unstack()`

P002 missed the week 12 visit. Drop that row and unstack again.

```python
missed = by_visit['sbp'].drop(('P002', 'week_12'))
gappy = missed.unstack()
display(gappy)
print(len(gappy.stack()), len(gappy.stack().dropna()))
```

**Expect:** P002's `week_12` cell is `NaN`, and the SBP columns become floats (`138.0`) because `NaN` is a float. Then `12 11`: `stack()` keeps the empty cell as a row, and `.dropna()` removes it.

### 10. Renumber rows after a filter

Filtering keeps the original row labels, so the numbers now have gaps.

```python
high_baseline = bp_wide[bp_wide['baseline'] >= 140]
print(list(high_baseline.index))

high_baseline = high_baseline.reset_index(drop=True)
print(list(high_baseline.index))
display(high_baseline[['patient_id', 'age', 'baseline']])
```

**Expect:** `[0, 2, 3]`, then `[0, 1, 2]`: P001, P003, and P004 started at 140 mmHg or higher. `drop=True` throws the old labels away instead of saving them as a column, because `0, 2, 3` carried no information.
