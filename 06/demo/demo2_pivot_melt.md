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

A small hypertension study records systolic blood pressure (SBP, mmHg) at baseline, week 4, and week 12. This demo gives the table meaningful row labels, reshapes it from wide to long and back, and fixes the repeated reading that stops `pivot()`. Everything here comes from Lecture 06 up to the second demo break, plus Lectures 01 to 05.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-25 with Python 3.13 and pandas 3.0.5; the whole notebook runs in a few seconds. The patient IDs and values are synthetic.

## Setup

The first cell installs pandas 3.0.5, the course version. Colab ships an older pandas (2.2). A `.venv` made with `uv venv --seed` includes pip, so the same `%pip` cell works locally too.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

**Expect:** `Note: you may need to restart the kernel to use updated packages.`, perhaps after a notice that a newer pip is available; neither needs any action. Locally, with the requirements already installed, the cell changes nothing and you can go on. In Colab, pip may also print a dependency conflict because some preinstalled packages expect pandas 2.2; that is expected, and this demo does not use them. If Colab asks you to restart the session (or says pandas was previously imported), choose **Runtime → Restart session**, then continue with the next cell. You do not need to rerun the install.

```python
import pandas as pd

print('pandas', pd.__version__)
```

**Expect:** `pandas 3.0.5`.

## 1. The study table (wide)

One row per patient, one SBP column per visit.

```python
bp_wide = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003', 'P004'],
    'clinic': ['North', 'South', 'North', 'South'],
    'baseline': [152, 138, 147, 141],
    'week_04': [144, 135, 140, 136],
    'week_12': [138, 131, 129, 130],
})
print(bp_wide.shape)
bp_wide
```

**Expect:** `(4, 5)`: 4 patients, and the columns `patient_id`, `clinic`, `baseline`, `week_04`, `week_12`. The row labels down the left are `0` to `3`, a RangeIndex that only counts rows.

## 2. Patient IDs as row labels

Move `patient_id` into the index and check that each patient appears once.

```python
bp = bp_wide.set_index('patient_id')
print(bp.index.is_unique)
bp
```

**Expect:** `True`, and the row labels are now `P001` to `P004`; `patient_id` is no longer an ordinary column.

Now a patient's row is one label away.

```python
print(bp.loc['P003'])
print(bp.loc['P003', 'week_12'])
```

**Expect:** P003's row as a Series (`clinic` North, `baseline` 147, `week_04` 140, `week_12` 129, `dtype: object` because it mixes text and numbers), then `129`.

Labels also line rows up. Each patient's SBP goal from the care plan arrives in a different order; subtracting matches patients by label, not by position.

```python
goal = pd.Series([130, 140, 130, 130], index=['P004', 'P003', 'P002', 'P001'])
above_goal = bp['week_12'] - goal
above_goal
```

**Expect:** P001 `8`, P002 `1`, P003 `-11`, P004 `0`. P003 is the only patient below goal at week 12 (P003's goal is 140), and P004 is exactly at goal.

`reset_index()` moves the labels back into a column, and the table is exactly what we started with.

```python
print(bp.reset_index().equals(bp_wide))
```

**Expect:** `True`.

## 3. Renumber rows after a filter

Filtering keeps the original row labels, so the numbers now have gaps.

```python
high_baseline = bp_wide[bp_wide['baseline'] >= 140]
print(list(high_baseline.index))

high_baseline = high_baseline.reset_index(drop=True)
print(list(high_baseline.index))
high_baseline[['patient_id', 'baseline']]
```

**Expect:** `[0, 2, 3]`, then `[0, 1, 2]`: P001, P003, and P004 started at 140 mmHg or higher. `drop=True` throws the old labels away instead of saving them as a column, because `0, 2, 3` carried no information.

## 4. Wide to long with `melt()`

Plotting SBP over time and grouping by visit both want one row per patient-visit.

```python
bp_long = bp_wide.melt(
    id_vars=['patient_id', 'clinic'],
    value_vars=['baseline', 'week_04', 'week_12'],
    var_name='visit',
    value_name='sbp',
)
print(bp_long.shape)
bp_long.head(6)
```

**Expect:** `(12, 4)`: 4 patients × 3 visits = 12 rows, with columns `patient_id`, `clinic`, `visit`, `sbp`. The first four rows are every patient's `baseline`, then the `week_04` rows begin.

The visit labels are text. A number of weeks is easier to plot and sort, so look each label up in a dictionary with `.map()` (Lecture 05).

```python
weeks = {'baseline': 0, 'week_04': 4, 'week_12': 12}
bp_long['week'] = bp_long['visit'].map(weeks)
print(bp_long['week'].isna().sum())
bp_long.sort_values(['patient_id', 'week']).head(3)
```

**Expect:** `0` unmapped labels, then P001's three rows in time order: 152 at week 0, 144 at week 4, and 138 at week 12.

## 5. Two-level row labels on the long table

In the long table one patient has three rows, so `patient_id` alone no longer names a row; `patient_id` and `visit` together do.

```python
by_visit = bp_long.set_index(['patient_id', 'visit']).sort_index()
print(by_visit.index.is_unique)
by_visit.loc['P003']
```

**Expect:** `True`, then P003's three visits (`baseline`, `week_04`, `week_12`) with SBP 147, 140, and 129. The labels sort alphabetically, which here is also time order because `baseline` comes before `week_04` and `week_12`.

```python
by_visit.loc[('P003', 'week_12'), :]
```

**Expect:** one row as a Series: `clinic` North, `sbp` 129, `week` 12.

```python
print(by_visit.reset_index().shape)
```

**Expect:** `(12, 5)`: both label levels are ordinary columns again.

## 6. Long back to wide with `pivot()`

`pivot()` rebuilds the wide table: `visit` supplies the headers and `sbp` fills the cells. With two identifier columns, the result has two-level row labels, so `reset_index()` turns them back into columns.

```python
bp_wide_again = bp_long.pivot(
    index=['patient_id', 'clinic'],
    columns='visit',
    values='sbp',
).reset_index()
bp_wide_again.columns.name = None
print(bp_wide_again.equals(bp_wide))
bp_wide_again
```

**Expect:** `True`: the round trip reproduces the original table exactly, the same 4 rows, 5 columns, values, and dtypes. The extra `week` column is simply not used, because `values='sbp'` names the one column that fills the cells.

## 7. When `pivot()` finds a repeated pair

The BP cuff's own export for the week 4 visit arrives. Pivot it to one column per visit.

```python
device_week4 = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003', 'P003', 'P004'],
    'visit': ['week_04'] * 5,
    'sbp': [144, 135, 152, 140, 136],
    'reading_time': ['09:05', '09:40', '10:15', '10:22', '11:02'],
})
print(len(device_week4))
```

**Expect:** `5` rows for 4 patients.

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
device_week4[device_week4.duplicated(subset=['patient_id', 'visit'], keep=False)]
```

**Expect:** 2 rows for P003: 152 at 10:15 and 140 at 10:22. The readings differ, so this is not a double entry: the nurse rechecked P003's pressure after a rest.

**The fix:** the clinic's protocol records the recheck, the later reading. Sort by time, keep the last row of each pair, and pivot again.

```python
recorded = (device_week4.sort_values('reading_time')
            .drop_duplicates(subset=['patient_id', 'visit'], keep='last'))
print(len(recorded))
recorded.pivot(index='patient_id', columns='visit', values='sbp')
```

**Expect:** `4` rows, then a one-column table (`week_04`) with P001 144, P002 135, P003 140, and P004 136. These match the `week_04` column of `bp_wide`, so the study table recorded the recheck too.

If both readings should count, `pivot_table()` would average them instead; that is a different question, and aggregation waits for Lecture 08.
