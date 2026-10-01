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

# Demo 3: From a clinic CSV to a saved result

A clinic exported a small file of visits, and the nurse lead wants every visit with a temperature of 37.5 °C or higher, hottest first, with the temperature in Fahrenheit and a fever flag. This demo reads the file, takes a first look, derives the new columns, sorts the rows so the order never changes between runs, saves the result, and reads it back. Each step says what to expect.

**Coming back to it:** in Colab, open this notebook from the lecture page's **Live notebooks in Colab** link; a new runtime needs only the setup cells below, which also download the data file. On your computer, open the `04-demo` folder that Demo 1's setup made (**File → Open Folder…** in VS Code), open this notebook, and choose the `.venv` kernel; if that folder does not exist yet, run Demo 1's five setup lines in a terminal first. Colab does not save your edits back to the course repository; to keep them, use **File → Save a copy in Drive**. The patient IDs and values are synthetic.

Choose a route below. The **core walkthrough** is the demonstration path; **independent practice** is for you to work through after class. In a fresh runtime, run Setup and the core first. **Run all** completes both routes.

| Route | Work and visible checkpoint |
| --- | --- |
| [Core walkthrough](#core-walkthrough) | Read the missing marker, select and flag six warm visits, sort ties, and read `warm_visits.csv` back as `(6, 6)`. |
| [Independent practice](#independent-practice) | Trigger and repair chained assignment; compare saved patient labels; save and verify a typed Parquet table. |

## Setup

Run this cell first. It installs pandas 3.0.5 and the Parquet backend pyarrow 25.0.0 into the kernel's environment.

- In Colab, pip may print `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...` and a line such as `google-colab ... requires pandas==..., but you have pandas 3.0.5 which is incompatible.` That is expected: the install still succeeded, this demo does not use those Colab packages, and the version check in the next cell confirms pandas 3.0.5.
- If Colab asks you to restart after the install, choose **Runtime → Restart session**, then run the notebook from the top.
- On your computer, `uv sync` already installed pandas 3.0.5 and pyarrow 25.0.0, so the cell changes nothing. It prints `Note: you may need to restart the kernel to use updated packages.`, perhaps with a notice that a newer pip exists; neither needs any action.

```python
# Setup: install pandas and the Parquet backend (Colab and local)
%pip install -q pandas==3.0.5 pyarrow==25.0.0
```

```python
import sys

import pandas as pd

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
assert pd.__version__ == "3.0.5", "Restart the session (Runtime → Restart session), then run all cells from the top"
```

Expect `pandas: 3.0.5`. If the check fails in Colab, pandas was imported before the install finished: restart the session and run all cells again.

## Core walkthrough

### Get the data file

This demo reads one file, `data/clinic_visits.csv`. On your computer, Demo 1's setup already saved it in the `data/` folder next to this notebook. Colab opens only the notebook, so this cell downloads the file from the course repository on GitHub when it is missing. It is supplied plumbing: it keeps a file that is already there, and `urlretrieve(url, path)` saves the file at a web address to `path`. Results go to an `output/` folder, created here so a fresh runtime has it.

```python
from pathlib import Path
from urllib.request import urlretrieve

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/04/demo"
DATA_PATH = Path("data") / "clinic_visits.csv"

if DATA_PATH.exists():
    status = "already here"
else:
    DATA_PATH.parent.mkdir(exist_ok=True)
    urlretrieve(f"{REPO_RAW}/{DATA_PATH.as_posix()}", DATA_PATH)
    status = "downloaded"
print(DATA_PATH.as_posix(), status, DATA_PATH.stat().st_size, "bytes")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
```

Expect `data/clinic_visits.csv already here 302 bytes` in `~/04-demo`, or `data/clinic_visits.csv downloaded 302 bytes` in Colab. If a run in `~/04-demo` says `downloaded`, the kernel's working directory is not the notebook's folder, so the copy landed somewhere else; check it with `%pwd` from Demo 1.

### Read the file as it is

Read the file with no options first and look at the dtypes. A numeric column that comes back as text means some entry in it is not a number.

```python
raw = pd.read_csv(DATA_PATH)

print(raw.dtypes)
raw.loc[raw["temp_c"] == "?"]
```

Expect `temp_c` to be `str`, not `float64`, while `age` and `systolic` are `float64`. The mask finds the reason: `P002`'s temperature was recorded as `?`. Because `temp_c` is text, `raw["temp_c"] >= 37.5` would raise a `TypeError`.

### Read it again with the missing marker

`na_values=["?"]` adds `?` to the markers pandas already treats as missing (blank, `NA`, `NULL`, and others).

```python
visits = pd.read_csv(DATA_PATH, na_values=["?"])

print("shape:", visits.shape)
print(visits.dtypes)
visits
```

Expect `shape: (12, 5)`, `temp_c` now `float64`, and `NaN` in four places: `P002`'s temperature, `P004`'s age (a blank), `P006`'s systolic pressure (a blank), and `P007`'s clinic (`NULL`).

### Take a first look

Before any analysis, ask what is missing, which categories the table holds, and whether any visit was entered twice.

```python
visits.info()

print(visits.isna().sum())
print(visits["clinic"].value_counts(dropna=False))
print("distinct clinics:", visits["clinic"].nunique())
print("repeated rows:", visits.duplicated().sum())
visits.loc[visits.duplicated()]
```

Expect `11 non-null` for `clinic`, `age`, `temp_c`, and `systolic` in `info()`, and one missing value in each of those columns. Clinic counts are North 5, South 3, East 3, and `NaN` 1, with `distinct clinics: 3` because `nunique()` leaves out missing. `repeated rows: 1`: the row labeled 10 repeats `P003`'s visit exactly. Lecture 05 removes repeats; here you only find them.

### Summarize the temperatures

`describe()` summarizes every numeric column; one-column summaries answer a single question. Missing values are skipped.

```python
print(visits.describe())

print("mean temp_c:", visits["temp_c"].mean())
print(f"mean temp_c, rounded: {visits['temp_c'].mean():.2f}")
print("highest temp_c:", visits["temp_c"].max())

hottest = visits["temp_c"].idxmax()
print("row label of the highest:", hottest)
visits.loc[hottest]
```

Expect a `temp_c` count of `11` in `describe()`, a mean of `37.69` when rounded, a highest temperature of `39.0`, and row label `8`, which is `P009` from the South clinic. The repeated `P003` visit is counted twice in these numbers, one reason to find repeats before summarizing.

### Copy the rows you will change

Build the mask on its own line, then select with `.loc`. `.copy()` makes `warm_visits` a separate table you intend to change, leaving `visits` as it was read.

```python
warm = visits["temp_c"] >= 37.5

warm_visits = visits.loc[warm, ["patient_id", "clinic", "temp_c", "systolic"]].copy()

print("warm visits:", warm.sum())
warm_visits
```

Expect `warm visits: 6`: `P004`, `P005`, `P007`, `P008`, `P009`, and `P011`. `P002` is left out because a missing temperature is not `>= 37.5`, so its mask value is `False`.

### Add a derived column

A **derived column** is computed from columns you already have. pandas converts the whole column at once and keeps each result on its own row.

```python
warm_visits["temp_f"] = warm_visits["temp_c"] * 9 / 5 + 32
warm_visits
```

Expect `temp_f` beside `temp_c`, such as `102.20` for `P009`'s `39.0` and `101.12` for the two `38.4` readings. The table rounds for display; the stored values can end in binary-rounding digits, such as `101.11999999999999`, which the saved file will show.

### Set the fever flag in one step

Start every selected row as elevated, then use one `.loc` assignment to flag fever.

```python
warm_visits["flag"] = "elevated"
```

```python
has_fever = warm_visits["temp_c"] >= 38.0
warm_visits.loc[has_fever, "flag"] = "fever"

print(warm_visits["flag"].value_counts())
warm_visits
```

Expect `fever 4` and `elevated 2`: `P007` (37.8) and `P011` (37.6) are warm but below 38.0 °C.

### Sort so the order never changes

`P004` and `P005` both have `38.4`, a **tie**. Sorting by `temp_c` alone does not guarantee which of the two comes first. To see that, sort the same six rows twice: once as they are, and once after arranging them by systolic pressure.

```python
by_systolic = warm_visits.sort_values("systolic")

print("temp_c only, rows as selected:")
print(warm_visits.sort_values("temp_c", ascending=False)[["patient_id", "temp_c"]])

print("temp_c only, rows arranged by systolic first:")
print(by_systolic.sort_values("temp_c", ascending=False)[["patient_id", "temp_c"]])
```

Expect `P004` before `P005` in the first result and `P005` before `P004` in the second: same rows, same key, different order. A unique second key, `patient_id`, settles every tie, so both starting orders give one answer:

```python
ordered = warm_visits.sort_values(by=["temp_c", "patient_id"], ascending=[False, True])
ordered_other = by_systolic.sort_values(by=["temp_c", "patient_id"], ascending=[False, True])

print(ordered[["patient_id", "temp_c"]])
print("same order both ways:", list(ordered["patient_id"]) == list(ordered_other["patient_id"]))
```

Expect `P009`, `P004`, `P005`, `P008`, `P007`, `P011`, and `same order both ways: True`.

### Write the result and read it back

`index=False` leaves the row index out of the file; here the index is only the row numbers these visits had in `visits`. A **round trip**, reading the saved file back, confirms the file holds what you meant to write. `display()` shows the table formatted in a notebook even when it is not the cell's last line.

```python
result_path = OUTPUT_DIR / "warm_visits.csv"
ordered.to_csv(result_path, index=False)

round_trip = pd.read_csv(result_path)
print("round-trip shape:", round_trip.shape)
print("columns:", list(round_trip.columns))
display(round_trip)
print("same patients, same order:", list(round_trip["patient_id"]) == list(ordered["patient_id"]))
```

Expect `round-trip shape: (6, 6)`, the columns `patient_id`, `clinic`, `temp_c`, `systolic`, `temp_f`, and `flag`, and `same patients, same order: True`. `P007`'s missing clinic was written as an empty field and reads back as `NaN`.

### Fresh-run check

Restart, then run Setup and the core cells in order. This cell checks the core checkpoints; use **Run All** when completing both routes.

```python
assert raw["temp_c"].dtype == "str"
assert visits["temp_c"].dtype == "float64"
assert visits.shape == (12, 5)
assert visits.duplicated().sum() == 1
assert visits["clinic"].nunique() == 3
assert f"{visits['temp_c'].mean():.2f}" == "37.69"
assert hottest == 8
assert warm.sum() == 6
assert list(visits.columns) == ["patient_id", "clinic", "age", "temp_c", "systolic"], "visits itself must be unchanged"
assert (warm_visits["flag"] == "fever").sum() == 4
assert list(ordered["patient_id"]) == ["P009", "P004", "P005", "P008", "P007", "P011"]
assert list(ordered_other["patient_id"]) == list(ordered["patient_id"])
assert round_trip.shape == (6, 6)
assert list(round_trip["patient_id"]) == list(ordered["patient_id"])

print("Demo 3 fresh-run check passed")
```

Expect `Demo 3 fresh-run check passed`.
## Independent practice

Continue on your own after class. These cells reuse the core results; if the runtime closed, run Setup and the core again first.

### Intentional mistake: chained assignment

A flag column takes two steps: give every row the common value, then overwrite only the rows a mask selects. Doing the second step with two bracket selections in a row changes a temporary copy, so pandas warns with `ChainedAssignmentError` and `warm_visits` stays unchanged.

```python
mistake = warm_visits.copy()
mistake["flag"] = "elevated"

# Intentional mistake: two bracket steps change a temporary copy
mistake[mistake["temp_c"] >= 38.0]["flag"] = "fever"

print(mistake["flag"].value_counts())
```

Expect a `ChainedAssignmentError` warning and `elevated 6`: no row became `fever`. The fix is one `.loc` step, with the mask and the column together:

```python
has_fever = warm_visits["temp_c"] >= 38.0
warm_visits.loc[has_fever, "flag"] = "fever"

print(warm_visits["flag"].value_counts())
warm_visits
```

Expect `fever 4` and `elevated 2`: `P007` (37.8) and `P011` (37.6) are warm but below 38.0 °C.

### Row labels: keep them or drop them

The first line of a CSV shows which labels were saved. Write the same table with the default index and compare the two header lines, read with `open()` from Lecture 02.

```python
numbered_path = OUTPUT_DIR / "warm_visits_row_numbers.csv"
ordered.to_csv(numbered_path)

with open(result_path, "r", encoding="utf-8") as file:
    no_index_lines = file.readlines()
with open(numbered_path, "r", encoding="utf-8") as file:
    numbered_lines = file.readlines()

print("index=False:  ", no_index_lines[0].strip())
print("default index:", numbered_lines[0].strip())
print("first record: ", numbered_lines[1].strip())
```

Expect the default-index file to start with an extra unnamed column (a leading comma in the header) and its first record to start with `8,`: the leftover row number, which means nothing to the nurse lead.

A patient ID is a meaningful label. `index_col="patient_id"` reads that column as the row index, so `.loc` finds a patient by ID, and writing with the default index then keeps it as the first column, headed `patient_id`.

```python
by_patient = pd.read_csv(result_path, index_col="patient_id")
print(by_patient.loc["P009"])

by_patient_path = OUTPUT_DIR / "warm_visits_by_patient.csv"
by_patient.to_csv(by_patient_path)
with open(by_patient_path, "r", encoding="utf-8") as file:
    print("header:", file.readlines()[0].strip())
```

Expect `P009`'s row (South, 39.0, 158.0, 102.2, fever) and `header: patient_id,clinic,temp_c,systolic,temp_f,flag`.


### Save a typed Parquet table

This uses `ordered`, the six warm visits from the core, and saves a separate file. Parquet preserves the dtypes and missing cells; the saved index is omitted because these old row numbers have no meaning. The Setup installer and the supplied local project both include `pyarrow`, the Parquet backend.

```python
parquet_path = OUTPUT_DIR / "warm_visits.parquet"
ordered.to_parquet(parquet_path, index=False)
parquet_back = pd.read_parquet(parquet_path)

print("Parquet shape:", parquet_back.shape)
print(parquet_back.dtypes)
print("missing clinic:", parquet_back["clinic"].isna().sum())
print("same patients, same order:", list(parquet_back["patient_id"]) == list(ordered["patient_id"]))
display(parquet_back)
```

Expect `Parquet shape: (6, 6)`, three text (`str`) columns (`patient_id`, `clinic`, `flag`) and three `float64` columns (`temp_c`, `systolic`, `temp_f`). P007 still has a missing clinic, so `missing clinic: 1` prints. `same patients, same order: True` confirms the sorted IDs survived. Reading back uses row labels 0 to 5 because the old row labels were omitted.

```python
for column in ["patient_id", "temp_c", "systolic", "temp_f", "flag"]:
    assert list(parquet_back[column]) == list(ordered[column])
assert parquet_back.dtypes.equals(ordered.dtypes)
assert parquet_back["clinic"].isna().sum() == 1
assert list(parquet_back.loc[~parquet_back["clinic"].isna(), "clinic"]) == ["South", "East", "South", "North", "North"]
print("Parquet round-trip check passed")
```

Expect `Parquet round-trip check passed`. Download `output/warm_visits.parquet` from Colab's Files pane if you want to keep it after the runtime closes.
