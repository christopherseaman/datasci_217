---
jupyter:
  jupytext:
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

# Demo 2: From NumPy arrays to labeled pandas

This demo turns NumPy arrays of patient measurements into a labeled Series and DataFrame, then selects columns, cells, blocks, and rows. The patient IDs and values are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5
```

## 2.1 Series and DataFrame

### 2.1a Build a Series

Lecture 03 stored measurements in a NumPy **ndarray**. A pandas **Series** adds an **index**: a label for each value, here the patient ID. `pd.Series()` accepts a plain list or an array, and the results match. Its `dtype` describes the stored values and its `name` identifies the Series.

```python
import numpy as np
import pandas as pd

patient_ids = ["P001", "P002", "P003"]

temp_from_list = pd.Series([36.8, 38.1, 37.2], index=patient_ids, name="temp_c")
temp_from_array = pd.Series(np.array([36.8, 38.1, 37.2]), index=patient_ids, name="temp_c")

display(temp_from_list)
display(temp_from_array)
print("same Series:", temp_from_list.equals(temp_from_array))
print("P002:", temp_from_list["P002"])
```

Expect two identical displays with rows `P001` to `P003` and the footer `Name: temp_c, dtype: float64`, then `same Series: True` and `P002: 38.1`.

### 2.1b Build a DataFrame

A pandas **DataFrame** is a labeled table. Each row is a patient and the two columns are systolic blood pressure (mmHg) at a baseline visit and at follow-up. `index=` labels the rows and `columns=` labels the columns. Three kinds of input build the same table:

- A list of lists: each inner list is one row.
- A 2D NumPy array: each inner row is one row.
- A dict: each key is a column name and each value is that column's list.

```python
patient_ids = ["P001", "P002", "P003", "P004"]
column_names = ["baseline_sbp", "follow_up_sbp"]

from_lists = pd.DataFrame(
    [[128, 124], [142, 136], [150, 138], [118, 131]],
    index=patient_ids,
    columns=column_names,
)

sbp_readings = np.array([[128, 124], [142, 136], [150, 138], [118, 131]])
sbp = pd.DataFrame(sbp_readings, index=patient_ids, columns=column_names)

from_dict = pd.DataFrame(
    {"baseline_sbp": [128, 142, 150, 118], "follow_up_sbp": [124, 136, 138, 131]},
    index=patient_ids,
)

display(from_lists)
display(sbp)
display(from_dict)
print("all three match:", from_lists.equals(sbp) and sbp.equals(from_dict))
```

Expect three identical tables with four patients and two columns, then `all three match: True`.

The rest of the demo uses `sbp`. Naming its index says what the row labels are.

```python
sbp.index.name = "patient_id"

print("shape:", sbp.shape)
display(sbp.dtypes)
sbp
```

Expect `shape: (4, 2)`, both columns `int64`, and a table with `patient_id` shown above the four row labels.

## 2.2 Select columns and cells

### 2.2a Select columns with brackets

One label in brackets returns a Series; a list of labels (double brackets) returns a DataFrame, even when the list holds one name.

```python
baseline = sbp["baseline_sbp"]
baseline_table = sbp[["baseline_sbp"]]

print(type(baseline))
print(type(baseline_table))
print(baseline_table.shape)
```

Expect `<class 'pandas.Series'>`, then `<class 'pandas.DataFrame'>`, then `(4, 1)`.

`display()` shows the two objects differently.

```python
display(baseline)
display(baseline_table)
```

Expect the Series as plain text with `Name: baseline_sbp, dtype: int64` and the DataFrame as a table with a `baseline_sbp` column header.

### 2.2b Labels with `.loc`, positions with `.iloc`

`.loc` selects by row and column **labels**; `.iloc` selects by zero-based integer **positions**. A label slice includes its end label; a position slice stops before its end position, as in ordinary Python. `.equals()` checks that two selections hold the same labels and values.

```python
by_label = sbp.loc["P002", "baseline_sbp"]
by_position = sbp.iloc[1, 0]
print("one cell by label:", by_label)
print("one cell by position:", by_position)

label_block = sbp.loc["P002":"P003", ["baseline_sbp", "follow_up_sbp"]]
position_block = sbp.iloc[1:3, 0:2]
display(label_block)
print("same block:", label_block.equals(position_block))
```

Expect `142` twice, a block with rows `P002` and `P003`, and `same block: True`.

### 2.2c Square brackets look up labels

On a Series, `[]` means index labels, not positions. Here the labels are patient IDs, so asking for position `1` fails. This error is intentional; `try`/`except` prints it instead of stopping the notebook.

```python
try:
    baseline[1]
except KeyError as error:
    print("KeyError:", error)

print("by label:", baseline["P002"])
print("by position with .iloc:", baseline.iloc[1])
```

Expect `KeyError: 1`, then `by label: 142` and `by position with .iloc: 142`.

## 2.3 Filter rows

### 2.3a Filter rows with a mask

A **mask** is a Boolean Series with the same index as the table. Build it on its own line with a descriptive name, then pass it to `.loc` with the columns you want. This one asks which patients still had a systolic pressure of 130 mmHg or higher at follow-up.

```python
high_at_follow_up = sbp["follow_up_sbp"] >= 130

display(high_at_follow_up)
print("rows:", high_at_follow_up.sum())
sbp.loc[high_at_follow_up, ["baseline_sbp", "follow_up_sbp"]]
```

Expect `False` for `P001` and `True` for the other three, `rows: 3`, and a table of `P002`, `P003`, and `P004`.

### 2.3b A filter leaves gaps in the labels

A default `0, 1, 2, ...` index hides the label/position difference until a filter drops a row. `plain` holds the same readings with the default index; the mask keeps rows labeled `1`, `2`, and `3`.

```python
plain = pd.DataFrame(sbp_readings, columns=["baseline_sbp", "follow_up_sbp"])
high_plain = plain["follow_up_sbp"] >= 130
kept = plain[high_plain]["baseline_sbp"]

display(kept)
try:
    kept[0]
except KeyError as error:
    print("KeyError:", error)

print("first kept row with .iloc:", kept.iloc[0])
```

Expect `kept` to show labels `1`, `2`, `3` with `142`, `150`, `118`, then `KeyError: 0` (the filter removed label `0`) and `first kept row with .iloc: 142`.

### 2.3c Narrow the selection with a second condition

`&` keeps rows where both masks are `True`; `|` keeps rows where either is. Written inline, each comparison needs its own parentheses. Which of those patients were below 130 mmHg at baseline, so their high reading is new?

```python
normal_at_baseline = sbp["baseline_sbp"] < 130
newly_high = high_at_follow_up & normal_at_baseline

# The same mask written inline; the parentheses are required
inline_mask = (sbp["follow_up_sbp"] >= 130) & (sbp["baseline_sbp"] < 130)

print("newly high:", newly_high.sum(), "row")
print("inline version:", inline_mask.sum(), "row")
print("same mask:", newly_high.equals(inline_mask))
sbp.loc[newly_high]
```

Expect `1 row` twice, `same mask: True`, and one row: `P004`, which went from 118 to 131 mmHg. `P001` was below 130 at baseline too, but its follow-up reading (124) stayed below 130.

## 2.4 Summarize and count

### 2.4a Summarize down the columns and across the rows

A reduction turns many values into one. By default it runs down each column; `axis="columns"` runs across each row, here giving each patient's average over the two visits. `idxmax()` names the row with the largest value, and `corr()` measures how closely two columns move together.

```python
display(pd.DataFrame({"mean": sbp.mean()}))
display(pd.DataFrame({"patient_mean": sbp.mean(axis="columns")}))

highest_follow_up = sbp["follow_up_sbp"].idxmax()
r = sbp["baseline_sbp"].corr(sbp["follow_up_sbp"])
display(pd.DataFrame({"highest_at_follow_up": [highest_follow_up], "baseline_follow_up_r": [round(r, 2)]}))
```

Expect column means of `134.50` and `132.25` mmHg, patient means from `124.5` (`P004`) to `144.0` (`P003`), and one row with `P003` (138 mmHg) and `0.72`: patients high at baseline tended to stay high.

### 2.4b Count clinics and test membership

Each patient was seen at one clinic. `value_counts()` counts each clinic, `nunique()` counts how many distinct clinics there are, and `isin()` builds a mask that is `True` for any clinic in a list, one mask instead of two joined with `|`.

```python
clinic = pd.Series(["North", "South", "North", "East"], index=sbp.index, name="clinic")

display(clinic.value_counts())
print("distinct clinics:", clinic.nunique())

south_or_east = clinic.isin(["South", "East"])
sbp.loc[south_or_east]
```

Expect `North 2`, `South 1`, `East 1`, `distinct clinics: 3`, and the rows for `P002` (South) and `P004` (East). The mask's index matches `sbp`'s, so `.loc` pairs each `True` with the right patient.

## 2.5 Independent practice

These cells reuse the results above; if the runtime closed, run the cells above again first.

### 2.5a First look at the table

- `head(3)` shows the first three rows.
- `info()` prints the index, column names, **non-null counts** (values present rather than missing), and dtypes; it returns `None`, so call it on its own line.
- `describe()` summarizes each numeric column.

```python
display(sbp.head(3))

sbp.info()

display(sbp.describe())
```

Expect `4 non-null` for both columns (nothing is missing), and in the summary a `mean` of `134.5` mmHg for `baseline_sbp` and `132.25` for `follow_up_sbp`, with minimums of `118` and `124`.

### 2.5b Intentional error: a position given to `.loc`

`.loc` only understands labels, and no row is _labeled_ `1`. The next cell asks for one anyway, catches the error with `try`/`except` from Lecture 02, and prints it instead of stopping the notebook.

```python
try:
    sbp.loc[1, "baseline_sbp"]
except KeyError as error:
    print("KeyError:", error)

print("fixed with .iloc:", sbp.iloc[1, 0])
print("fixed with the label:", sbp.loc["P002", "baseline_sbp"])
```

Expect `KeyError: 1`, then `142` from each fix: use `.iloc` for a position, or `.loc` with the label.
