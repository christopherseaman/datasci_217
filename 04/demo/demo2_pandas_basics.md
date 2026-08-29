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
    version: 3.12.13
---

# Demo 2 — From NumPy arrays to labeled pandas

**Learning objectives**

- Convert one 1D ndarray to a Series and one 2D ndarray to a DataFrame.
- Inspect labels, shape, dtypes, and bounded numeric summaries.
- Select with brackets, `.loc`, `.iloc`, and one boolean mask.
- Add one derived column and sort with a unique tie-breaker.

Colab is the default launch experience; local Jupyter runs the same cells. See `DEMO_GUIDE.md` for launch links and checkpoints. GitHub source opened in Colab is not automatically updated by edits in the Colab tab.

Compatibility candidate: Python 3.12.13, NumPy 2.0.2, pandas 3.0.5. This is not the final course lock until fresh local and Colab certification is complete. Never place credentials, tokens, protected records, or identifying data in notebook source or output.

```python
from importlib.metadata import version
import sys

PANDAS_CANDIDATE = "3.0.5"

import numpy as np
import pandas as pd

assert version("pandas") == PANDAS_CANDIDATE, (
    "Install the demo requirements before running this notebook; "
    f"expected pandas {PANDAS_CANDIDATE}, found {version('pandas')}"
)
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
```

## A 1D ndarray becomes a Series

Lecture 03 used a NumPy **ndarray**, a homogeneous array selected by integer position. A pandas **Series** is a one-dimensional labeled object. Its **index** contains row labels, its `dtype` describes the stored values, and its optional `name` identifies the Series.

```python
temperatures = np.array([18.5, 21.0, 19.5])

temperature_by_site = pd.Series(
    temperatures,
    index=["north", "south", "west"],
    name="temperature_c",
)

print(temperature_by_site)
print("index:", temperature_by_site.index)
print("dtype:", temperature_by_site.dtype)
```

## A 2D ndarray becomes a DataFrame

A pandas **DataFrame** is a two-dimensional labeled table. Its `index` labels rows, its `columns` label columns, its `shape` reports `(rows, columns)`, and its `dtypes` report one dtype for each column.

```python
measurements = np.array(
    [
        [10, 20],
        [20, 30],
        [30, 40],
        [10, 20],
    ]
)

measurement_table = pd.DataFrame(
    measurements,
    index=["obs-001", "obs-002", "obs-003", "obs-004"],
    columns=["baseline", "follow_up"],
)
measurement_table.index.name = "record_id"

print("index:", measurement_table.index)
print("columns:", measurement_table.columns)
print("shape:", measurement_table.shape)
print("dtypes:")
print(measurement_table.dtypes)
measurement_table
```

## Bounded inspection

`head(3)` returns a small structural preview. `info()` prints index details, column names, **non-null counts** (entries present rather than missing), dtypes, and a memory summary. Numeric `describe()` reports count, mean, standard deviation (`std`), minimum, percentile cut points, and maximum. This lecture inspects those results; decisions about missing values or cleaning belong to Lecture 05. Call `info()` directly because it prints its report and returns `None`.

```python
first_three = measurement_table.head(3)
print(first_three)

measurement_table.info()

numeric_summary = measurement_table.describe()
print(numeric_summary)
```

## Select columns with brackets

Brackets select columns by label. `df["column"]` returns a Series, while a list of column labels inside double brackets returns a DataFrame. Bracket notation also works with names containing spaces or names shared by DataFrame methods.

```python
baseline_series = measurement_table["baseline"]
baseline_table = measurement_table[["baseline"]]
two_columns = measurement_table[["baseline", "follow_up"]]

print(type(baseline_series))
print(type(baseline_table))
two_columns
```

## Labels with `.loc`; positions with `.iloc`

`.loc` selects by row and column **labels**. `.iloc` selects by zero-based integer **positions**. Label slices include both named endpoints when present; positional slices follow ordinary Python slicing and exclude the stop position.

```python
same_value_by_label = measurement_table.loc["obs-002", "baseline"]
same_value_by_position = measurement_table.iloc[1, 0]

label_block = measurement_table.loc[
    "obs-002":"obs-003",
    ["baseline", "follow_up"],
]
position_block = measurement_table.iloc[1:3, 0:2]

print(same_value_by_label)
print(same_value_by_position)
print(label_block)
print(position_block)
```

## Filter with one mask

Lecture 03 defined a **mask** as `True` and `False` values used to select elements. A pandas mask carries the same row index as the Series or DataFrame. Build it separately and give it a descriptive name so the selection condition remains visible.

```python
follow_up_at_least_30 = measurement_table["follow_up"] >= 30

selected_measurements = measurement_table.loc[
    follow_up_at_least_30,
    ["baseline", "follow_up"],
]

print(follow_up_at_least_30)
selected_measurements
```

## Add one derived column

A **derived column** is calculated from existing columns. pandas aligns the arithmetic by row labels, so this calculation needs no explicit Python loop.

```python
measurement_table["change"] = (
    measurement_table["follow_up"] - measurement_table["baseline"]
)

measurement_table
```

## Sort deterministically

A **deterministic sort** produces the same observable row order for the same input. State every key and direction, and finish with a unique tie-breaker. Here `record_id` names the unique index labels and resolves equal `change` values.

```python
ordered_measurements = measurement_table.sort_values(
    by=["change", "record_id"],
    ascending=[False, True],
)

ordered_measurements
```

```python
assert np.array_equal(temperatures, np.array([18.5, 21.0, 19.5]))
assert temperature_by_site.index.tolist() == ["north", "south", "west"]
assert temperature_by_site.name == "temperature_c"
assert measurement_table.index.name == "record_id"
assert measurement_table.index.tolist() == [
    "obs-001",
    "obs-002",
    "obs-003",
    "obs-004",
]
assert measurement_table[["baseline", "follow_up"]].shape == (4, 2)
assert first_three.shape == (3, 2)
assert numeric_summary.shape == (8, 2)
assert isinstance(baseline_series, pd.Series)
assert isinstance(baseline_table, pd.DataFrame)
assert same_value_by_label == same_value_by_position == 20
pd.testing.assert_frame_equal(label_block, position_block)
assert selected_measurements.index.tolist() == ["obs-002", "obs-003"]
assert measurement_table["change"].tolist() == [10, 10, 10, 10]
assert ordered_measurements.index.tolist() == [
    "obs-001",
    "obs-002",
    "obs-003",
    "obs-004",
]

print("Demo 2 fresh-run verification passed")
```
