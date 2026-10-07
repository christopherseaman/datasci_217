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

A clinic exported a small file of visits. The nurse lead wants every visit at 37.5 °C or higher, hottest first, with the temperature in Fahrenheit and a fever flag. This demo reads, checks, derives, sorts, saves, and reads back. The patient IDs and values are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
```

## Core walkthrough

### Get the data file

This cell downloads `data/clinic_visits.csv` if it is missing and creates the `output/` folder.

```python
from pathlib import Path
from urllib.request import urlretrieve

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/04/demo"
DATA_PATH = Path("data") / "clinic_visits.csv"

# You can probably guess what this is doing, note the import at the top of the cell
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

Expect `data/clinic_visits.csv downloaded 302 bytes`, or `already here` if the file was there.

### Read the file as it is

Read the file with no options first and look at the dtypes. A numeric column that comes back as text means some entry in it is not a number.

```python
import pandas as pd

raw = pd.read_csv(DATA_PATH)

display(raw.dtypes)
```

- :x: Expect `temp_c` to be `str`, not `float64`, while `age` and `systolic` are `float64`. Because `temp_c` is text, `raw["temp_c"] >= 37.5` would raise a `TypeError`.
    - → Find the non-numeric entries: list the distinct values, then mask on the one that is not a number.

```python
display(raw["temp_c"].unique())
raw.loc[raw["temp_c"] == "?"]
```

Expect the distinct values to include `'?'` beside the temperatures, and the mask to find `P002`, whose temperature was recorded as `?`.

### Read it again with the missing marker

`na_values=["?"]` adds `?` to the markers pandas already treats as missing (blank, `NA`, `NULL`, and others).

```python
visits = pd.read_csv(DATA_PATH, na_values=["?"])

print("shape:", visits.shape)
display(visits.dtypes)
visits
```

Expect `shape: (12, 5)`, `temp_c` now `float64`, and `NaN` in four places: `P002`'s temperature, `P004`'s age (a blank), `P006`'s systolic pressure (a blank), and `P007`'s clinic (`NULL`).

### Several missing markers in one file

A lab export may mark a missing value more than one way. List every marker in `na_values`; a number written as text, such as `"-999"`, matches the number too.

```python
labs_path = OUTPUT_DIR / "labs_markers.csv"
with open(labs_path, "w", encoding="utf-8") as file:
    file.write("patient_id,glucose_mg_dl,ldl_mg_dl,lab_site\n")
    file.write("P001,98,131,North\n")
    file.write("P002,?,-999,South\n")
    file.write("P003,110,unknown,unknown\n")
    file.write("P004,-999,142,North\n")

labs = pd.read_csv(labs_path, na_values=["?", "-999", "unknown"])
display(labs)
display(labs.count())
```

Expect `glucose_mg_dl` and `ldl_mg_dl` as `float64` with `NaN` for every marker, `P003`'s `lab_site` as `NaN`, and `count()` showing 4, 2, 2, and 3 values present.

### Preview a few columns

A large export is quicker to check with a preview: `usecols=` reads only the named columns, and `nrows=` stops after that many records. `dtype=` sets a column's type instead of letting pandas guess.

```python
preview = pd.read_csv(
    DATA_PATH,
    usecols=["patient_id", "temp_c"],
    nrows=4,
    na_values=["?"],
    dtype={"temp_c": "float64"},
)
display(preview.dtypes)
preview
```

Expect two columns, `patient_id` (`str`) and `temp_c` (`float64`), and four rows, `P001` to `P004`. Without `na_values=["?"]`, `dtype=` would raise a `ValueError`, because `?` cannot be read as a number.

### Take a first look

Before any analysis, ask what is missing, which categories the table holds, and whether any patient appears more than once.

```python
visits.info()

display(visits["clinic"].value_counts(dropna=False))
print("distinct clinics:", visits["clinic"].nunique())

id_counts = visits["patient_id"].value_counts()
display(id_counts.head(3))
visits.loc[visits["patient_id"] == "P003"]
```

- `info()`: `11 non-null` for `clinic`, `age`, `temp_c`, and `systolic`, one missing value in each.
- Clinic counts: North 5, South 3, East 3, `NaN` 1; `distinct clinics: 3`, since `nunique()` leaves out missing.
- `P003` is counted twice; its rows, labeled 2 and 10, match in every column: one visit entered twice, which the summaries below still count.

### Summarize the temperatures

`describe()` summarizes every numeric column; one-column summaries answer a single question. Missing values are skipped.

```python
display(visits.describe())

print("mean temp_c:", visits["temp_c"].mean())
print(f"mean temp_c, rounded: {visits['temp_c'].mean():.2f}")
print("highest temp_c:", visits["temp_c"].max())

hottest = visits["temp_c"].idxmax()
print("row label of the highest:", hottest)
visits.loc[hottest]
```

Expect a `temp_c` count of `11` in `describe()`, a mean of `37.69` when rounded, a highest temperature of `39.0`, and row label `8`, which is `P009` from the South clinic. The repeated `P003` visit is counted twice in these numbers, one reason to find repeats before summarizing.

### Compare every visit with the average

Subtracting a Series of column means from a table **broadcasts**: pandas matches the Series' labels to the columns and subtracts each mean from every row of its column.

```python
vitals = visits[["temp_c", "systolic"]]
display(vitals.mean())

from_mean = vitals - vitals.mean()
from_mean.loc[[0, 8]]
```

Expect means of about `37.69` °C and `136.27` mmHg. `P001` (row 0) is about 0.89 °C and 18.27 mmHg below them (`-0.890909`, `-18.272727`), and `P009` (row 8) about 1.31 °C and 21.73 mmHg above.

### Select the rows and columns you need

Build the mask on its own line, then select the rows with `.loc`. `drop(columns=["age"])` leaves out the column the nurse lead does not need. It returns a new table, so `warm_visits` is separate from `visits`, as `.copy()` would make it, and changing it leaves `visits` as it was read.

```python
warm = visits["temp_c"] >= 37.5

warm_visits = visits.loc[warm].drop(columns=["age"])

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

display(warm_visits["flag"].value_counts())
warm_visits
```

Expect `fever 4` and `elevated 2`: `P007` (37.8) and `P011` (37.6) are warm but below 38.0 °C.

### Sort so the order never changes

`P004` and `P005` both have `38.4`, a **tie**. Sorting by `temp_c` alone does not guarantee which of the two comes first. To see that, sort the same six rows twice: once as they are, and once after arranging them by systolic pressure.

```python
by_systolic = warm_visits.sort_values("systolic")

print("temp_c only, rows as selected:")
display(warm_visits.sort_values("temp_c", ascending=False)[["patient_id", "temp_c"]])

print("temp_c only, rows arranged by systolic first:")
display(by_systolic.sort_values("temp_c", ascending=False)[["patient_id", "temp_c"]])
```

Expect `P004` before `P005` in the first result and `P005` before `P004` in the second: same rows, same key, different order. A unique second key, `patient_id`, settles every tie, so both starting orders give one answer:

```python
ordered = warm_visits.sort_values(by=["temp_c", "patient_id"], ascending=[False, True])
ordered_other = by_systolic.sort_values(by=["temp_c", "patient_id"], ascending=[False, True])

display(ordered[["patient_id", "temp_c"]])
print("same order both ways:", list(ordered["patient_id"]) == list(ordered_other["patient_id"]))
```

Expect `P009`, `P004`, `P005`, `P008`, `P007`, `P011`, and `same order both ways: True`.

A rank numbers each visit by its place without moving rows. `method="min"` gives tied values the same, best place:

```python
ranked = ordered[["patient_id", "temp_c"]].copy()
ranked["temp_rank"] = ranked["temp_c"].rank(ascending=False, method="min")
ranked
```

Expect ranks `1`, `2`, `2`, `4`, `5`, `6`: `P004` and `P005` share second place at 38.4 °C, so no visit is third. `ordered` itself is unchanged.

### Look up a visit by patient ID

`set_index("patient_id")` makes the IDs the row labels, so `.loc` finds a patient by ID instead of by row number. It returns a new table; `visits` is unchanged.

```python
by_id = visits.set_index("patient_id")

print("P009 temperature:", by_id.loc["P009", "temp_c"])
display(by_id.loc["P003"])
```

Expect `P009 temperature: 39.0`. `P003` appears twice, so `.loc["P003"]` returns a two-row table instead of one row.

### Line up arithmetic by patient ID

Subtracting two Series pairs values by label, whatever their order. A label on only one side gives `NaN`.

```python
first = by_id.loc[["P009", "P005", "P004"], "temp_c"]
recheck = pd.Series([38.1, 38.6, 36.9], index=["P005", "P009", "P020"])

display(recheck - first)
```

Expect `P004` and `P020` as `NaN` (each is on one side only), `P005` as `-0.3` and `P009` as `-0.4`, matched by ID although the two Series list them in different orders.

### Renumber rows after a filter

A filter keeps each row's original label, so the default index shows gaps. `reset_index(drop=True)` discards the old labels and renumbers from 0.

```python
warm_ids = visits.loc[warm, ["patient_id", "temp_c"]]
display(warm_ids)
display(warm_ids.reset_index(drop=True))
```

Expect row labels `3`, `4`, `6`, `7`, `8`, `11` in the first table, with the six visits `P004`, `P005`, `P007`, `P008`, `P009`, `P011`, and `0` to `5` in the second.

### Turn the ID index back into a column

`reset_index()` without `drop=True` moves the index into a column, which is what you want before saving the IDs as data.

```python
back_to_column = by_id.reset_index()

print(list(back_to_column.columns))
display(back_to_column.head(3))
```

Expect `['patient_id', 'clinic', 'age', 'temp_c', 'systolic']` and a default index `0`, `1`, `2` again.

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

### The saved index and reading it back

`to_csv()` writes the index as the first column unless `index=False`. This cell saves `by_id` (IDs as the index) three ways and reads each back.

```python
by_id_path = OUTPUT_DIR / "by_id.csv"
by_id.to_csv(by_id_path)
no_ids_path = OUTPUT_DIR / "by_id_no_ids.csv"
by_id.to_csv(no_ids_path, index=False)

plain = pd.read_csv(by_id_path)
indexed = pd.read_csv(by_id_path, index_col="patient_id")
dropped = pd.read_csv(no_ids_path)

print("plain:", list(plain.columns), "index:", plain.index[:3].tolist())
print("index_col:", list(indexed.columns), "index:", indexed.index[:3].tolist())
print("index=False:", list(dropped.columns))
display(plain.head(3))
```

Expect these lines:

- `plain`: `patient_id` is an ordinary column and the index is the new row numbers `[0, 1, 2]`.
- `index_col`: `patient_id` is the index (`['P001', 'P002', 'P003']`) and is no longer a column.
- `index=False`: the columns are `clinic`, `age`, `temp_c`, `systolic`; the patient IDs were not saved.

### Fresh-run check

Restart, then **Run All** up to here. This cell checks the walkthrough's checkpoints.

```python
assert raw["temp_c"].dtype == "str"
assert visits["temp_c"].dtype == "float64"
assert visits.shape == (12, 5)
assert list(preview.columns) == ["patient_id", "temp_c"] and len(preview) == 4
assert id_counts["P003"] == 2 and (id_counts > 1).sum() == 1
assert round(from_mean.loc[8, "temp_c"], 2) == 1.31
assert list(ranked["temp_rank"]) == [1, 2, 2, 4, 5, 6]
assert visits["clinic"].nunique() == 3
assert f"{visits['temp_c'].mean():.2f}" == "37.69"
assert hottest == 8
assert warm.sum() == 6
assert list(visits.columns) == ["patient_id", "clinic", "age", "temp_c", "systolic"], "visits itself must be unchanged"
assert (warm_visits["flag"] == "fever").sum() == 4
assert list(ordered["patient_id"]) == ["P009", "P004", "P005", "P008", "P007", "P011"]
assert list(ordered_other["patient_id"]) == list(ordered["patient_id"])
assert round_trip.shape == (6, 6)
assert by_id.loc["P009", "temp_c"] == 39.0 and "patient_id" in back_to_column.columns
assert list(warm_ids.reset_index(drop=True).index) == list(range(6))
assert "patient_id" in plain.columns and indexed.index.name == "patient_id" and "patient_id" not in dropped.columns
assert list(round_trip["patient_id"]) == list(ordered["patient_id"])

print("Demo 3 fresh-run check passed")
```

Expect `Demo 3 fresh-run check passed`.

## Independent practice

These cells reuse the core results; if the runtime closed, run the cells above again first.

### Intentional mistake: chained assignment

A flag column takes two steps: give every row the common value, then overwrite only the rows a mask selects. Doing the second step with two bracket selections in a row changes a temporary copy, so pandas warns with `ChainedAssignmentError` and `warm_visits` stays unchanged.

```python
mistake = warm_visits.copy()
mistake["flag"] = "elevated"

# Intentional mistake: two bracket steps change a temporary copy
mistake[mistake["temp_c"] >= 38.0]["flag"] = "fever"

display(mistake["flag"].value_counts())
```

Expect a `ChainedAssignmentError` warning and `elevated 6`: no row became `fever`. The fix is one `.loc` step, with the mask and the column together:

```python
has_fever = warm_visits["temp_c"] >= 38.0
warm_visits.loc[has_fever, "flag"] = "fever"

display(warm_visits["flag"].value_counts())
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

### Save a typed Parquet table

This uses `ordered`, the six warm visits from the core, and saves a separate file. Parquet preserves the dtypes and missing cells; the saved index is omitted because these old row numbers have no meaning. The install cell at the top includes `pyarrow`, the Parquet backend.

```python
parquet_path = OUTPUT_DIR / "warm_visits.parquet"
ordered.to_parquet(parquet_path, index=False)
parquet_back = pd.read_parquet(parquet_path)

print("Parquet shape:", parquet_back.shape)
display(parquet_back.dtypes)
print("clinics recorded:", parquet_back["clinic"].count(), "of", len(parquet_back))
print("same patients, same order:", list(parquet_back["patient_id"]) == list(ordered["patient_id"]))
display(parquet_back)
```

Expect `Parquet shape: (6, 6)`, three text (`str`) columns (`patient_id`, `clinic`, `flag`) and three `float64` columns (`temp_c`, `systolic`, `temp_f`). P007 still has a missing clinic, so `clinics recorded: 5 of 6` prints, then `same patients, same order: True`. The rows read back as 0 to 5 because the old row labels were omitted.

```python
for column in ["patient_id", "temp_c", "systolic", "temp_f", "flag"]:
    assert list(parquet_back[column]) == list(ordered[column])
assert parquet_back.dtypes.equals(ordered.dtypes)
assert parquet_back["clinic"].count() == 5
assert parquet_back["clinic"].equals(ordered["clinic"].reset_index(drop=True))
print("Parquet round-trip check passed")
```

Expect `Parquet round-trip check passed`. Download `output/warm_visits.parquet` from Colab's Files pane if you want to keep it after the runtime closes.
