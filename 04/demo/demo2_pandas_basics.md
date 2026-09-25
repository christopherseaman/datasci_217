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

This demo turns NumPy arrays of patient measurements into a labeled Series and DataFrame, then selects columns, cells, blocks, and rows the way the lecture did. Each step says what to expect, so you can tell whether it worked.

Run this notebook in Colab from the course page, or locally in VS Code with the course `.venv` selected as the kernel. Colab does not save your edits back to the course repository; to keep them, use **File → Save a copy in Drive**. The patient IDs and values here are synthetic.

## Setup

Run this cell first. It installs pandas 3.0.5, the course version, into the notebook's environment: in Colab, which ships an older pandas, and in your local `.venv` alike.

- pip may print a warning that other Colab packages expect a different pandas. That is expected; this demo does not use those packages.
- If Colab asks you to restart after the install, choose **Runtime → Restart session**, then run the notebook from the top.
- Locally, the `.venv` you made in Lecture 03 with `uv venv --seed` includes pip, so `%pip` installs into it too. When pandas 3.0.5 is already installed there, the cell prints `Note: you may need to restart the kernel to use updated packages.`, perhaps with a notice that a newer pip exists; neither needs any action.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

```python
import sys

import numpy as np
import pandas as pd

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
assert pd.__version__ == "3.0.5", "Restart the session (Runtime → Restart session), then run all cells from the top"
```

Expect `pandas: 3.0.5`. If the check fails in Colab, pandas was imported before the install finished: restart the session and run all cells again.

## A 1D array becomes a Series

Lecture 03 stored measurements in a NumPy **ndarray** and selected them by integer position. A pandas **Series** adds an **index**: a label for each value, here the patient ID. Its `dtype` describes the stored values, and its `name` identifies the Series.

```python
temps_c = np.array([36.8, 38.1, 37.2])

temp_by_patient = pd.Series(
    temps_c,
    index=["P001", "P002", "P003"],
    name="temp_c",
)

print(temp_by_patient)
print("index:", temp_by_patient.index)
print("P002:", temp_by_patient["P002"])
```

Expect three rows labeled `P001` to `P003`, the footer `Name: temp_c, dtype: float64`, and `P002: 38.1`: the label finds the value without knowing its position.

## A 2D array becomes a DataFrame

A pandas **DataFrame** is a labeled table. Here each row is a patient and the two columns are systolic blood pressure (mmHg) at a baseline visit and at follow-up. `index=` labels the rows, `columns=` labels the columns, and naming the index says what the labels are.

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
print("dtypes:")
print(sbp.dtypes)
sbp
```

Expect `shape: (4, 2)`, both columns `int64`, and a table with `patient_id` shown above the four row labels.

## First look at the table

`head(3)` shows the first three rows. `info()` prints the index, column names, **non-null counts** (values present rather than missing), dtypes, and memory; it prints its report and returns `None`, so call it on its own line. `describe()` summarizes each numeric column.

```python
print(sbp.head(3))

sbp.info()

sbp_summary = sbp.describe()
print(sbp_summary)
```

Expect `4 non-null` for both columns (nothing is missing), and in the summary a `mean` of `134.5` mmHg for `baseline_sbp` and `132.25` for `follow_up_sbp`, with minimums of `118` and `124`.

## Select columns with brackets

One label in brackets returns a Series; a list of labels (double brackets) returns a DataFrame, even when the list holds one name.

```python
baseline = sbp["baseline_sbp"]
baseline_table = sbp[["baseline_sbp"]]

print(type(baseline))
print(type(baseline_table))
print(baseline_table.shape)
```

Expect `<class 'pandas.Series'>`, then `<class 'pandas.DataFrame'>`, then `(4, 1)`: the same column, two different shapes.

## Labels with `.loc`, positions with `.iloc`

`.loc` selects by row and column **labels**; `.iloc` selects by zero-based integer **positions**. A label slice includes its end label; a position slice stops before its end position, as in ordinary Python. `.equals()` checks that two selections hold the same labels and values.

```python
by_label = sbp.loc["P002", "baseline_sbp"]
by_position = sbp.iloc[1, 0]
print("one cell by label:", by_label)
print("one cell by position:", by_position)

label_block = sbp.loc["P002":"P003", ["baseline_sbp", "follow_up_sbp"]]
position_block = sbp.iloc[1:3, 0:2]
print(label_block)
print("same block:", label_block.equals(position_block))
```

Expect `142` twice, a block with rows `P002` and `P003`, and `same block: True`. The label slice ends at `"P003"` and includes it; the position slice `1:3` stops before position 3, which is the same row.

## Intentional error: a position given to `.loc`

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

## Filter rows with a mask

A **mask** is a Boolean Series with the same index as the table. Build it on its own line with a descriptive name, then pass it to `.loc` with the columns you want. Here the question is which patients still had a systolic pressure of 130 mmHg or higher at follow-up.

```python
high_at_follow_up = sbp["follow_up_sbp"] >= 130

print(high_at_follow_up)
print("rows:", high_at_follow_up.sum())
sbp.loc[high_at_follow_up, ["baseline_sbp", "follow_up_sbp"]]
```

Expect `False` for `P001` and `True` for the other three, `rows: 3`, and a table of `P002`, `P003`, and `P004`.

## Narrow the selection with a second condition

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

## Fresh-run check

Run **Restart session and run all** (VS Code: **Restart**, then **Run All**). This cell checks the checkpoints above.

```python
assert temp_by_patient["P002"] == 38.1
assert temp_by_patient.name == "temp_c"
assert sbp.index.name == "patient_id"
assert sbp.shape == (4, 2)
assert sbp_summary.loc["mean", "baseline_sbp"] == 134.5
assert isinstance(baseline, pd.Series)
assert isinstance(baseline_table, pd.DataFrame)
assert by_label == by_position == 142
assert label_block.equals(position_block)
assert high_at_follow_up.sum() == 3
assert newly_high.sum() == inline_mask.sum() == 1
assert list(sbp.loc[newly_high].index) == ["P004"]

print("Demo 2 fresh-run check passed")
```

Expect `Demo 2 fresh-run check passed`.
