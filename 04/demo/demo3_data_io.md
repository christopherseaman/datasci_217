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

# Demo 3 — From a CSV file to a saved result

**Learning objectives**

- Resolve one immutable CSV input without assuming an absolute path or launch directory.
- Read the file with pandas and take a first look: gaps, categories, repeats.
- See how an unrecognized missing marker changes a column's dtype, and fix it with `na_values`.
- Copy a selection, add a derived column and a flag, and sort it deterministically.
- Write the result to CSV, read it back, and verify the round trip in fresh state.

Colab is the default launch experience; local Jupyter runs the same cells. Run this notebook from a fresh kernel and follow its path, inspection, and round-trip checks. GitHub source opened in Colab is not automatically updated by edits in the Colab tab.

Compatibility candidate: Python 3.13, NumPy 2.3.3, pandas 3.0.5. This is not the final course lock until fresh local and Colab certification is complete. Never place credentials, tokens, protected records, or identifying data in notebook source or output.

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

## Resolve and verify a portable input

A **portable path** identifies the same course input without assuming one absolute location or one notebook launch directory. A **checksum** is a short digest of file bytes; matching the expected digest confirms that local and downloaded inputs are identical.

The supplied bootstrap searches upward for the committed fixture. If it is unavailable, as it is when Colab opens only this notebook, the bootstrap downloads the file from one immutable upstream commit. It verifies either source before pandas reads it, creates every required directory in code, and names the three files this notebook writes so that the cells below only need those names. Run `%pwd` from Demo 1 if you want to see the directory this search starts from.

```python
from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve

SOURCE_RELATIVE_PATH = Path("04") / "demo" / "data" / "anscombe.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/"
    "71e2436a092d714350de0fc409ca8a8714e7e78f/anscombe.csv"
)
EXPECTED_SHA256 = (
    "a0c1f636aa0347101de76271e7efe4c8"
    "6a22ef28cda62886eaff23a1bf1924b1"
)


def find_course_file(start, relative_path):
    current = start.resolve()
    while True:
        candidate = current / relative_path
        if candidate.is_file():
            return candidate
        if current.parent == current:
            return None
        current = current.parent


DATA_PATH = find_course_file(Path.cwd(), SOURCE_RELATIVE_PATH)
lecture_readme = find_course_file(Path.cwd(), Path("04") / "README.md")

if DATA_PATH is None:
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    DATA_PATH = data_dir / "anscombe.csv"
    if not DATA_PATH.is_file():
        urlretrieve(SOURCE_URL, DATA_PATH)

actual_sha256 = sha256(DATA_PATH.read_bytes()).hexdigest()
assert actual_sha256 == EXPECTED_SHA256, "Unexpected anscombe.csv content"

if lecture_readme is None:
    demo_base = Path.cwd()
else:
    demo_base = lecture_readme.parent / "demo"

OUTPUT_DIR = demo_base / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "selected_anscombe.csv"
ROW_NUMBER_PATH = OUTPUT_DIR / "selected_with_row_numbers.csv"
MESSY_PATH = OUTPUT_DIR / "messy_readings.csv"

print("Input:", DATA_PATH)
print("Output:", OUTPUT_PATH)
```

## Read the pinned CSV

A **CSV file** is a text table whose first row normally gives field names and whose later rows give records. `pd.read_csv()` creates a labeled DataFrame and chooses a dtype for each column. Inspect its shape, columns, and dtypes before selecting values.

`anscombe.csv` holds Anscombe's quartet: four small `dataset` groups of `x` and `y` points that share almost the same straight-line fit. `print()` shows the plain-text table; `display()` renders it as a formatted table in Jupyter, so a cell can show more than its last line.

```python
anscombe = pd.read_csv(DATA_PATH)

print("shape:", anscombe.shape)
print("columns:", anscombe.columns)
print("dtypes:")
print(anscombe.dtypes)
display(anscombe.head(3))
anscombe.tail(3)
```

## Take a first look

Before using a new table, ask what is missing, which categories it holds, and whether any record was entered twice. Each question is one line, and each answer here is a number you can check: no gaps, four datasets of eleven rows, no repeated records.

```python
print("missing values per column:")
print(anscombe.isna().sum())

print("rows per dataset:")
print(anscombe["dataset"].value_counts(dropna=False))
print("distinct datasets:", anscombe["dataset"].nunique())

print("repeated records:", anscombe.duplicated().sum())
```

## Missing markers change a column's dtype

This fixture is clean. Clinic exports usually are not, and a single unrecognized marker is enough to turn a numeric column into text. pandas already treats a blank, `NA`, `N/A`, and `NULL` as missing; a `?` is read as ordinary text unless you name it in `na_values`. Write a tiny messy file here with `open()` from Lecture 02, then read it both ways.

```python
with open(MESSY_PATH, "w", encoding="utf-8") as file:
    file.write("patient_id,temp_c\n")
    file.write("P001,36.8\n")
    file.write("P002,?\n")
    file.write("P003,37.2\n")

as_text = pd.read_csv(MESSY_PATH)
with_markers = pd.read_csv(MESSY_PATH, na_values=["?"])

print("temp_c dtype without na_values:", as_text["temp_c"].dtype)
print("temp_c dtype with na_values=['?']:", with_markers["temp_c"].dtype)
print(with_markers)
print("missing temp_c values:", with_markers["temp_c"].isna().sum())
```

## Copy the rows you will change

Build the mask separately and give it a descriptive name, as in Demo 2, then select with `.loc`. `.copy()` says plainly that `selected_points` is a separate table you intend to modify, leaving `anscombe` as it was read.

```python
x_at_least_13 = anscombe["x"] >= 13

selected_points = anscombe.loc[
    x_at_least_13,
    ["dataset", "x", "y"],
].copy()

print("selected rows:", x_at_least_13.sum())
selected_points
```

## Add a derived column and a flag

A **derived column** is calculated from columns you already have. All four datasets fit close to the same line, `y = 3 + 0.5 * x`, and `gap_from_line` subtracts that line's height from each point's `y`. It is a **signed gap**: positive above the line, negative below it, zero on it. pandas computes the whole column at once and keeps every result on its own row.

A gap near zero means the point sits on the line, not that the point is ordinary. Dataset IV's `x = 19` point gets `0.00` here, and it is the lone far-right point that fixes where that dataset's line goes at all. Judging how unusual a point is takes more than one column; this section is about deriving a column and flagging rows with it.

A flag column is two steps: assign the common value to every row, then overwrite only the rows a mask selects. Binary floating point cannot store `-0.04` exactly, so the saved file keeps values such as `-0.03999999999999915`; the printed table rounds them for reading.

```python
selected_points["gap_from_line"] = (
    selected_points["y"] - (3 + 0.5 * selected_points["x"])
)

on_or_above = selected_points["gap_from_line"] >= 0

selected_points["position"] = "below the fitted line"
selected_points.loc[on_or_above, "position"] = "on or above the fitted line"

print("points on or above the line:", on_or_above.sum())
selected_points
```

## Sort deterministically

A **deterministic sort** produces the same observable row order for the same rows. State every key and direction, and finish with a key that makes each combination of sort keys unique. Three of these rows share `x = 14`, so sorting by `x` alone says nothing about how those three are ordered: pandas is free to leave them in whatever order they arrived in.

To see that, sort the same seven rows twice from two different starting orders. Sorting by `x` alone puts the `x = 14` rows as `I, II, III` from the rows as read and as `II, III, I` from the same rows arranged by `y`. Adding `dataset` as a second key gives one order from both.

```python
same_rows_by_y = selected_points.sort_values("y")

print("x alone, rows as read:")
print(selected_points.sort_values("x", ascending=False)[["dataset", "x"]])

print("x alone, same rows arranged by y first:")
print(same_rows_by_y.sort_values("x", ascending=False)[["dataset", "x"]])

ordered_points = selected_points.sort_values(
    by=["x", "dataset"],
    ascending=[False, True],
)
ordered_from_other_order = same_rows_by_y.sort_values(
    by=["x", "dataset"],
    ascending=[False, True],
)

print("x then dataset, rows as read:")
print(ordered_points[["dataset", "x"]])

print("x then dataset, same rows arranged by y first:")
print(ordered_from_other_order[["dataset", "x"]])

ordered_points
```

## Write the result and read it back

A **round trip** writes data and then reads the new file back. It catches path, column, selection, and serialization mistakes immediately. `index=False` keeps the row index out of the file, which is what you want here: the index is just the row numbers this selection happened to land on in `anscombe`.

```python
ordered_points.to_csv(OUTPUT_PATH, index=False)
print("wrote:", OUTPUT_PATH)

round_trip = pd.read_csv(OUTPUT_PATH)
print("round-trip shape:", round_trip.shape)
round_trip
```

## Row labels: keep them or drop them

The first line of a CSV shows which labels survived, so write the selection both ways and read the two header lines back with `open()`. `index=False` left the row index out, so the saved file starts with `dataset`; writing with the default index instead adds an unnamed first column holding those row numbers.

Reading works the other way: `index_col` names a column to use as the index. It needs a column that labels one row each, and `dataset` names a group of eleven points rather than a single row. The readings file written earlier has one `patient_id` per row, so that is the column to read back as the index, and `.loc` on one of its labels returns that patient's row.

```python
ordered_points.to_csv(ROW_NUMBER_PATH)

with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
    no_index_lines = file.readlines()

with open(ROW_NUMBER_PATH, "r", encoding="utf-8") as file:
    default_index_lines = file.readlines()

print("index=False:", no_index_lines[0].strip())
print("default index:", default_index_lines[0].strip())

by_patient = pd.read_csv(MESSY_PATH, na_values=["?"], index_col="patient_id")
print("index name:", by_patient.index.name)
print("index:", by_patient.index)
print(by_patient.loc["P003"])
by_patient
```

## Fresh-runtime verification

A **fresh-runtime execution** starts without names or files created by earlier interactive work, so every required input and directory must be resolved by the visible cells above.

```python
assert list(anscombe.columns) == ["dataset", "x", "y"]
assert anscombe.shape == (44, 3)
assert anscombe.isna().sum().sum() == 0
assert anscombe["dataset"].nunique() == 4
assert anscombe.duplicated().sum() == 0

assert as_text["temp_c"].dtype == "str"
assert with_markers["temp_c"].dtype == "float64"
assert with_markers["temp_c"].isna().sum() == 1

assert x_at_least_13.sum() == 7
assert selected_points.shape == (7, 5)
assert list(anscombe.columns) == ["dataset", "x", "y"], "anscombe itself must be unchanged"
assert selected_points["position"].tolist().count("on or above the fitted line") == 2

assert ordered_points["dataset"].tolist() == ["IV", "I", "II", "III", "I", "II", "III"]
assert ordered_points["x"].tolist() == [19.0, 14.0, 14.0, 14.0, 13.0, 13.0, 13.0]
assert ordered_from_other_order["dataset"].tolist() == ordered_points["dataset"].tolist()

assert list(round_trip.columns) == ["dataset", "x", "y", "gap_from_line", "position"]
assert round_trip.shape == (7, 5)
assert (round_trip["x"] >= 13).all()
assert np.allclose(
    round_trip["gap_from_line"].to_numpy(),
    np.array([0.0, -0.04, -1.9, -1.16, -1.92, -0.76, 3.24]),
)
assert by_patient.index.name == "patient_id"
assert with_markers["patient_id"].nunique() == with_markers.shape[0]
assert by_patient.index.tolist() == ["P001", "P002", "P003"]
assert list(by_patient.columns) == ["temp_c"]
assert by_patient.loc["P003", "temp_c"] == 37.2

print("Demo 3 fresh-run verification passed")
round_trip
```
