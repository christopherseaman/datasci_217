# Notebooks and Labeled Data with pandas

Lecture 04 is the transition from terminal-run Python scripts to reproducible notebook work. It also connects the positional NumPy arrays from Lecture 03 to pandas objects with row and column labels.

Optional extensions are collected in [BONUS.md](BONUS.md). They are not prerequisites for the demos, assignment, or Lecture 05.

## Prerequisites

Before starting this lecture, you should be able to:

- run a Python script from a terminal and read a traceback;
- activate a project environment and import NumPy;
- explain an ndarray's dimension, shape, dtype, axis, positional index, slice, and boolean mask;
- use vectorized arithmetic and basic reductions; and
- use a working directory and relative paths.

No prior notebook or pandas experience is expected.

## Learning objectives

By the end of this lecture, you should be able to:

1. Define **notebook**, **cell**, **kernel/runtime**, **state**, **execution order**, **output**, **restart**, and **run all**, and distinguish a notebook runtime from a fresh script process.
2. Reproduce a stale-state failure, explain why visible source and output can disagree with current state, and repair the notebook so restart-and-run-all succeeds.
3. Construct a labeled `Series` from a 1D ndarray and a labeled `DataFrame` from a 2D ndarray, then inspect `index`, `columns`, `shape`, and `dtypes`.
4. Use `[]`, `.loc`, `.iloc`, and one boolean mask; add one derived column; and sort deterministically with explicit keys, directions, and a unique tie-breaker.
5. Read a pinned CSV and write a result with `index=False` using a portable local/Colab path, then verify the notebook from top to bottom in a fresh runtime.

## From a script process to a notebook

A terminal-run script normally starts a fresh Python process, executes from top to bottom, and then exits. A **notebook** is an `.ipynb` document containing an ordered sequence of cells and, when saved, possibly their stored outputs.

A **cell** is one unit in that document:

- a **code cell** contains executable Python; and
- a **Markdown cell** contains headings, explanations, links, or other documentation.

A **kernel** is the separate Python process that executes code cells. A **runtime** is the wider execution environment that hosts the kernel, installed packages, and runtime-local files. Local Jupyter usually emphasizes the word *kernel*; Colab usually emphasizes *runtime*. For this course, both words point to the execution environment behind the visible notebook.

**State** is the collection of names, values, imports, and other in-memory results currently held by the kernel. Unlike a finished script process, notebook state remains available after a cell finishes.

**Execution order** is the order in which code cells were actually run. It can differ from the visual top-to-bottom order of the document.

An **output** is a result captured below a code cell, such as text, a value, a table, or an error. Saving a notebook can store that output in the `.ipynb` file. A stored output records what happened at some earlier execution; it does not prove that the visible source still produces that result.

**Stale state** occurs when the current kernel values or stored outputs no longer agree with the visible source and its top-to-bottom order.

A **restart** discards the kernel's in-memory state. **Run all** executes the notebook's code cells in visual order. **Restart and run all** combines those actions and is the basic reproducibility check for notebook work.

### A concrete stale-state failure

Suppose these are separate code cells in visual order:

```python
# Cell 1
units = 12
rate = 2
```

```python
# Cell 2
total = units * rate
total
```

The first top-to-bottom run produces `24`. Now edit Cell 1 so it visibly says `rate = 3`, but do not run it. The source says `3` while the kernel still holds `2` and the stored output still says `24`.

Next, run only the edited Cell 1. The kernel now holds `rate == 3`, but `total` is still the earlier value `24`. The notebook contains two values that cannot come from one clean top-to-bottom run.

Finally, restart and try Cell 2 by itself. It raises a `NameError` because the new kernel has never executed Cell 1. The repair is to keep producer cells before the cells that depend on them, restart, and run all. The consistent result is then `36`.

Execution counters can help reveal what ran, but they are diagnostic hints rather than proof. A clean run is the proof.

## Colab first, local Jupyter equivalent

[Google Colab](https://colab.research.google.com/) is the default launch experience for compatible notebook demos from this lecture onward. Colab provides a hosted Jupyter interface, so students can open and run a notebook without first starting a local notebook server.

Local Jupyter remains fully supported. The concepts and Python code are the same; only interface labels differ.

| Action | Colab | Local Jupyter or VS Code |
|---|---|---|
| Run the current cell | Run button or `Shift+Enter` | Run button or `Shift+Enter` |
| Choose Python | Runtime settings | Kernel selector |
| Restart in-memory state | Restart session/runtime | Restart kernel |
| Execute top to bottom | Run all | Run all cells |
| Inspect a saved notebook | File menu | File browser or editor |

When a notebook is opened from GitHub in Colab, edits are not automatically saved back to the repository. Demo notebooks use Colab by default, but notebook assignments must run in clean local Jupyter until the save-to-repository and Classroom 50 submission pilot is approved.

## Outputs, privacy, and evidence

A notebook file can contain its source, Markdown, stored outputs, and execution metadata. Treat all four as shareable content.

- Clear any output containing credentials, private records, or other sensitive information before saving or sharing.
- Ordinary nonsensitive output may remain when a human reviewer needs it, but it is never accepted as execution evidence.
- Files created under an `output/` directory are separate generated artifacts; they are not the same as a cell's stored output.
- Before submission, restart and run all.
- A grader must execute a fresh copy and evaluate the newly created results rather than trust stored output.

Do not put passwords, access tokens, or private data in a teaching notebook, even temporarily.

## The Colab filesystem is ephemeral

An **ephemeral filesystem** belongs to a temporary runtime. Files installed or created inside a Colab runtime can disappear when that virtual machine is disconnected and deleted. The saved notebook document may remain in Drive or GitHub while its runtime-local packages and files do not.

For that reason, a reproducible notebook must include the setup and data-acquisition steps it needs. Course notebooks do not mount Google Drive by default and do not require manual uploads. Small teaching data is committed to the course repository and fetched from a stable HTTPS address when the local committed copy is unavailable.

## Candidate runtime and package setup

The current compatibility candidate is:

| Component | Candidate |
|---|---|
| Python | 3.12.13 |
| NumPy | 2.0.2 |
| pandas | 3.0.3 |

This is not the final course lock. It becomes the supported release only after the required notebooks and graders pass both clean local-Jupyter and fresh-Colab validation.

Colab's selectable 2026.04 runtime already supplies the candidate Python and NumPy versions, but its recorded image has an earlier pandas version. A single setup cell therefore checks pandas before importing it and installs only the candidate when necessary:

```python
from importlib.metadata import PackageNotFoundError, version
import subprocess
import sys

PANDAS_CANDIDATE = "3.0.3"

try:
    installed_pandas = version("pandas")
except PackageNotFoundError:
    installed_pandas = None

if installed_pandas != PANDAS_CANDIDATE:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            f"pandas=={PANDAS_CANDIDATE}",
        ],
        check=True,
    )

import numpy as np
import pandas as pd

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
```

Run package setup before any pandas import. Avoid reinstalling the complete Colab package collection.

Do not install pandas 3.0.4. That release was yanked for a reported defect and is not part of the course candidate.

## Portable project paths

A **portable path** identifies the same course input without assuming one absolute location or one notebook launch directory. The local path and Colab path may differ, but the teaching cells receive the same `DATA_PATH`, `OUTPUT_DIR`, and `OUTPUT_PATH` names.

The pinned input is `anscombe.csv` from Michael Waskom's [`seaborn-data` repository](https://github.com/mwaskom/seaborn-data), fixed at commit `71e2436a092d714350de0fc409ca8a8714e7e78f`. It has 44 rows and the exact columns `dataset`, `x`, and `y`.

A supplied bootstrap cell follows this contract:

- use a future course-local `04/demo/data/anscombe.csv` copy if it is present;
- otherwise fetch the file from the immutable upstream commit;
- verify its checksum;
- create a runtime-local data directory only when needed; and
- create the output directory before writing.

The bootstrap does not assume that the course-local copy already exists.

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

print("Input:", DATA_PATH)
print("Output:", OUTPUT_PATH)
```

The fallback address already names one immutable upstream commit. A later demo rebuild may add the verified file under `04/demo/data/`, but the HTTPS fallback remains executable when that local file is absent.

## LIVE DEMO 1: Notebook runtime and stale-state repair

The first required demonstration uses Colab first and names the local-Jupyter equivalents. It creates code and Markdown cells, shows stored output, deliberately produces the stale-state sequence above, restarts into a real failure, and repairs the notebook with restart-and-run-all. It closes by identifying ephemeral runtime files and applying the output/privacy policy.

## From NumPy positions to pandas labels

Lecture 03 introduced a NumPy ndarray as a homogeneous array whose elements are selected by integer position. pandas keeps that positional array foundation and adds labels.

The two central pandas structures are:

| NumPy foundation | pandas structure | Added labels |
|---|---|---|
| 1D ndarray | `Series` | one row `index` |
| 2D ndarray | `DataFrame` | a row `index` and column `columns` |

A **Series** is a one-dimensional labeled pandas object. Its **index** contains the row labels, its `dtype` describes the stored values, and its optional `name` identifies the series.

### A 1D ndarray becomes a Series

```python
import numpy as np
import pandas as pd

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

The three values still have positions `0` through `2`, but they now also have the labels `north`, `south`, and `west`.

A **DataFrame** is a two-dimensional labeled pandas table. Its `index` labels rows, its `columns` label columns, its `shape` reports `(rows, columns)`, and its `dtypes` report one dtype for each column.

### A 2D ndarray becomes a DataFrame

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

The ndarray has one dtype for the full array. A DataFrame reports dtype by column, so later tables can represent columns with different kinds of values while retaining one shared row index.

### Bounded DataFrame inspection

Three bounded inspection tools answer different questions:

A **non-null count** is the number of entries present rather than missing in a column. Lecture 04 only reports that count; interpreting why entries are absent or deciding what action to take belongs to Lecture 05.

- `head(3)` returns the first three rows for a small structural preview;
- `info()` prints the index, column names, non-null counts, dtypes, and memory summary; and
- numeric `describe()` returns count, mean, spread, and percentile summaries for numeric columns.

In `describe()` output, `std` is a standard-deviation measure of spread, and the `25%`, `50%`, and `75%` rows are percentile cut points. Only the count and already-familiar mean are interpreted in this lecture; later material develops the other statistical summaries.

```python
first_three = measurement_table.head(3)
print(first_three)

measurement_table.info()

numeric_summary = measurement_table.describe()
print(numeric_summary)
```

`info()` prints its report and returns `None`, so call it directly rather than wrapping it in `display()`. Inspection describes the table that arrived; deciding what to change is a Lecture 05 cleaning task.

### Column selection with brackets

Brackets select columns by label:

```python
baseline_series = measurement_table["baseline"]
baseline_table = measurement_table[["baseline"]]
two_columns = measurement_table[["baseline", "follow_up"]]

print(type(baseline_series))
print(type(baseline_table))
two_columns
```

- `df["column"]` returns a Series.
- `df[["column"]]` returns a one-column DataFrame.
- `df[["first", "second"]]` returns a DataFrame.

Use bracket notation rather than attribute-style column access. Brackets work consistently with names containing spaces and names that overlap with DataFrame methods.

### Labels with `.loc` and positions with `.iloc`

`.loc` selects by row and column **labels**. `.iloc` selects by zero-based integer **positions**.

| Request | Label form | Position form |
|---|---|---|
| One cell | `df.loc["obs-002", "baseline"]` | `df.iloc[1, 0]` |
| Rows 2 through 3, both measurement columns | `df.loc["obs-002":"obs-003", ["baseline", "follow_up"]]` | `df.iloc[1:3, 0:2]` |

Label slices with `.loc` include both named endpoints when both are present. Positional slices with `.iloc` follow ordinary Python slicing and exclude the stop position.

```python
same_value_by_label = measurement_table.loc["obs-002", "baseline"]
same_value_by_position = measurement_table.iloc[1, 0]

print(same_value_by_label)
print(same_value_by_position)

label_block = measurement_table.loc[
    "obs-002":"obs-003",
    ["baseline", "follow_up"],
]
position_block = measurement_table.iloc[1:3, 0:2]

print(label_block)
print(position_block)
```

### Filter with one boolean mask

Lecture 03 defined a mask as an array of `True` and `False` values used to select elements. A pandas mask is labeled by the same index as its Series or DataFrame.

```python
follow_up_at_least_30 = measurement_table["follow_up"] >= 30

selected_measurements = measurement_table.loc[
    follow_up_at_least_30,
    ["baseline", "follow_up"],
]

print(follow_up_at_least_30)
selected_measurements
```

Build the mask separately and give it a descriptive name. This makes the selection condition visible and debuggable.

### Add one derived column

A **derived column** is calculated from existing columns. pandas applies the arithmetic to corresponding labeled rows without an explicit Python loop.

```python
measurement_table["change"] = (
    measurement_table["follow_up"] - measurement_table["baseline"]
)

measurement_table
```

The example adds one new column and leaves the two source columns available for inspection.

### Sort deterministically

A **deterministic sort** gives the same observable order every time the same input is used. State every sort key and direction, and finish with a unique tie-breaker.

The index name `record_id` identifies the unique row labels, so it can serve as the final tie-break key:

```python
ordered_measurements = measurement_table.sort_values(
    by=["change", "record_id"],
    ascending=[False, True],
)

ordered_measurements
```

This requests the largest change first. Rows with the same change are then ordered by their unique record label.

## LIVE DEMO 2: NumPy to labeled pandas

The second required demonstration converts one 1D ndarray to a Series and one 2D ndarray to a DataFrame. It inspects labels and structure with `head(3)`, `info()`, and numeric `describe()`, then practices bracket selection, `.loc`, `.iloc`, one mask, one derived column, and a deterministic sort with a unique tie-breaker.

## Portable CSV input and output

A **CSV file** is a text table in which the first row normally provides field names and later rows provide records. Lecture 03 used the standard library to load a fixed CSV into an ndarray. pandas provides `read_csv()` to create a DataFrame directly.

### Read the pinned input

The portable bootstrap defines `DATA_PATH` before this teaching cell runs:

```python
anscombe = pd.read_csv(DATA_PATH)

print("shape:", anscombe.shape)
print("columns:", anscombe.columns)
print("dtypes:")
print(anscombe.dtypes)
anscombe
```

The expected shape is `(44, 3)` and the expected columns are `dataset`, `x`, and `y`. Inspect those labels and the dtypes before selecting values.

### Select and write one result

```python
x_at_least_13 = anscombe["x"] >= 13

selected_anscombe = anscombe.loc[
    x_at_least_13,
    ["dataset", "x", "y"],
]

selected_anscombe.to_csv(OUTPUT_PATH, index=False)
print("wrote:", OUTPUT_PATH)
```

`index=False` prevents pandas from writing the DataFrame's row index as an extra CSV field. The output contains seven rows and only the three explicitly selected columns.

### Read back and verify

A **round-trip** writes data and then reads the new file back. This catches path, column, and serialization mistakes immediately.

```python
round_trip = pd.read_csv(OUTPUT_PATH)

assert list(round_trip.columns) == [
    "dataset",
    "x",
    "y",
]
assert round_trip.shape == (7, 3)
assert (round_trip["x"] >= 13).all()

print("Fresh-run verification passed")
round_trip
```

The verification cell is deliberately small and observable. It confirms the exact columns, seven-row result, and selection condition.

### Fresh-runtime check

A **fresh-runtime execution** begins without relying on earlier interactive work. For routine notebook work:

- save the visible source;
- restart the kernel or Colab session;
- run all cells from top to bottom;
- confirm that setup reacquires any runtime-local input;
- confirm that every required directory is created by code; and
- reach the final verification result without manually defining a name or uploading a file.

For formal Colab certification, repeat the run after disconnecting and deleting the prior runtime. For assignment grading, the grader executes a fresh notebook copy and ignores stored outputs.

## LIVE DEMO 3: Portable CSV round-trip

The third required demonstration uses the shared package and path bootstrap, reads the pinned CSV, inspects its labeled structure, selects rows and columns, writes one `index=False` result, reads it back, and reaches the final verification in both fresh Colab and clean local Jupyter.

## Handoff to Lecture 05

Lecture 04 establishes the mechanics needed for reliable notebook work:

- a notebook's visible order and kernel state are different things;
- restart-and-run-all checks whether the document reproduces its results;
- 1D and 2D NumPy arrays become labeled Series and DataFrames;
- labels, positions, masks, derived columns, and deterministic order support precise selection; and
- portable paths make the same CSV workflow usable locally and in Colab.

Lecture 05 can now begin with raw tables, row meaning, schema expectations, provenance, and explicit data-quality decisions. It can assume that the notebook itself starts cleanly and that input and output paths are reproducible.
