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

## Core walkthrough

### A 1D array becomes a Series

Lecture 03 stored measurements in a NumPy **ndarray** and selected them by integer position. A pandas **Series** adds an **index**: a label for each value, here the patient ID. Its `dtype` describes the stored values, and its `name` identifies the Series.

```python
import numpy as np
import pandas as pd

temps_c = np.array([36.8, 38.1, 37.2])

temp_by_patient = pd.Series(
    temps_c,
    index=["P001", "P002", "P003"],
    name="temp_c",
)

display(temp_by_patient)
print("index:", temp_by_patient.index)
print("P002:", temp_by_patient["P002"])
```

Expect three rows labeled `P001` to `P003`, the footer `Name: temp_c, dtype: float64`, and `P002: 38.1`: the label finds the value without knowing its position.

### A 2D array becomes a DataFrame

A pandas **DataFrame** is a labeled table. Each row is a patient and the two columns are systolic blood pressure (mmHg) at a baseline visit and at follow-up. `index=` labels the rows, `columns=` labels the columns, and naming the index says what the labels are.

```python
sbp_readings = np.array(
    [
        [128, 124],
        [142, 136],
        [150, 138],
        [118, 131],
    ]
)

sbp = pd.DataFrame(
    sbp_readings,
    index=["P001", "P002", "P003", "P004"],
    columns=["baseline_sbp", "follow_up_sbp"],
)
sbp.index.name = "patient_id"

print("shape:", sbp.shape)
display(sbp.dtypes)
sbp
```

Expect `shape: (4, 2)`, both columns `int64`, and a table with `patient_id` shown above the four row labels.

### Select columns with brackets

One label in brackets returns a Series; a list of labels (double brackets) returns a DataFrame, even when the list holds one name.

```python
baseline = sbp["baseline_sbp"]
baseline_table = sbp[["baseline_sbp"]]

print(type(baseline))
print(type(baseline_table))
print(baseline_table.shape)
```

Expect `<class 'pandas.Series'>`, then `<class 'pandas.DataFrame'>`, then `(4, 1)`: the same column, two different shapes.

### Labels with `.loc`, positions with `.iloc`

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

### Filter rows with a mask

A **mask** is a Boolean Series with the same index as the table. Build it on its own line with a descriptive name, then pass it to `.loc` with the columns you want. This one asks which patients still had a systolic pressure of 130 mmHg or higher at follow-up.

```python
high_at_follow_up = sbp["follow_up_sbp"] >= 130

display(high_at_follow_up)
print("rows:", high_at_follow_up.sum())
sbp.loc[high_at_follow_up, ["baseline_sbp", "follow_up_sbp"]]
```

Expect `False` for `P001` and `True` for the other three, `rows: 3`, and a table of `P002`, `P003`, and `P004`.

### Narrow the selection with a second condition

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

### Summarize down the columns and across the rows

A reduction turns many values into one. By default it runs down each column; `axis="columns"` runs across each row, here giving each patient's average over the two visits. `idxmax()` names the row with the largest value, and `corr()` measures how closely two columns move together.

```python
display(sbp.mean())
patient_mean = sbp.mean(axis="columns")
display(patient_mean)

highest_follow_up = sbp["follow_up_sbp"].idxmax()
print("highest at follow-up:", highest_follow_up)
r = sbp["baseline_sbp"].corr(sbp["follow_up_sbp"])
print(f"baseline vs follow-up r: {r:.2f}")
```

Expect column means of `134.50` and `132.25` mmHg, patient means from `124.5` (`P004`) to `144.0` (`P003`), `highest at follow-up: P003` (138 mmHg), and `r: 0.72`: patients high at baseline tended to stay high, though `P004` rose while the others fell.

### Count clinics and test membership

Each patient was seen at one clinic. `value_counts()` counts each clinic, `nunique()` counts how many distinct clinics there are, and `isin()` builds a mask that is `True` for any clinic in a list, one mask instead of two joined with `|`.

```python
clinic = pd.Series(["North", "South", "North", "East"], index=sbp.index, name="clinic")

display(clinic.value_counts())
print("distinct clinics:", clinic.nunique())

south_or_east = clinic.isin(["South", "East"])
sbp.loc[south_or_east]
```

Expect `North 2`, `South 1`, `East 1`, `distinct clinics: 3`, and the rows for `P002` (South) and `P004` (East). The mask's index matches `sbp`'s, so `.loc` pairs each `True` with the right patient.

### Fresh-run check

Restart, then **Run All** up to here. This cell checks the walkthrough's checkpoints.

```python
assert temp_by_patient["P002"] == 38.1
assert temp_by_patient.name == "temp_c"
assert sbp.index.name == "patient_id"
assert sbp.shape == (4, 2)
assert isinstance(baseline, pd.Series)
assert isinstance(baseline_table, pd.DataFrame)
assert by_label == by_position == 142
assert label_block.equals(position_block)
assert high_at_follow_up.sum() == 3
assert newly_high.sum() == inline_mask.sum() == 1
assert list(sbp.loc[newly_high].index) == ["P004"]
assert patient_mean["P003"] == 144.0
assert highest_follow_up == "P003"
assert round(r, 2) == 0.72
assert clinic.nunique() == 3
assert list(sbp.loc[south_or_east].index) == ["P002", "P004"]

print("Demo 2 fresh-run check passed")
```

Expect `Demo 2 fresh-run check passed`.

## Independent practice

These cells reuse the core results; if the runtime closed, run the cells above again first.

### First look at the table

- `head(3)` shows the first three rows.
- `info()` prints the index, column names, **non-null counts** (values present rather than missing), and dtypes; it returns `None`, so call it on its own line.
- `describe()` summarizes each numeric column.

```python
display(sbp.head(3))

sbp.info()

display(sbp.describe())
```

Expect `4 non-null` for both columns (nothing is missing), and in the summary a `mean` of `134.5` mmHg for `baseline_sbp` and `132.25` for `follow_up_sbp`, with minimums of `118` and `124`.

### Intentional error: a position given to `.loc`

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
