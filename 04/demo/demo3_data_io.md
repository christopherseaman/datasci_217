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

# Demo 3 — Portable CSV round trip

**Learning objectives**

- Resolve one immutable CSV input without assuming an absolute path or launch directory.
- Verify the input checksum before reading it with pandas.
- Select explicit rows and columns, create an output directory, and write one CSV.
- Read the output back and verify the round trip in fresh state.

Colab is the default launch experience; local Jupyter runs the same cells. See `DEMO_GUIDE.md` for launch links and path-case checks. GitHub source opened in Colab is not automatically updated by edits in the Colab tab.

Compatibility candidate: Python 3.12.13, NumPy 2.0.2, pandas 3.0.3. This is not the final course lock until fresh local and Colab certification is complete. Never place credentials, tokens, protected records, or identifying data in notebook source or output.

```python
from importlib.metadata import version
import sys

PANDAS_CANDIDATE = "3.0.3"

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

The supplied bootstrap searches upward for the committed fixture. If it is unavailable, as it is when Colab opens only this notebook, the bootstrap downloads the file from one immutable upstream commit. It verifies either source before pandas reads it and creates every required directory in code.

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

## Read the pinned CSV

A **CSV file** is a text table whose first row normally gives field names and whose later rows give records. `pd.read_csv()` creates a labeled DataFrame. Inspect its shape, columns, and dtypes before selecting values.

```python
anscombe = pd.read_csv(DATA_PATH)

print("shape:", anscombe.shape)
print("columns:", anscombe.columns)
print("dtypes:")
print(anscombe.dtypes)
anscombe.head(3)
```

## Select and write one result

Reuse the labeled mask and `.loc` selection from Demo 2. `index=False` prevents pandas from writing the DataFrame's row index as an extra CSV field.

```python
x_at_least_13 = anscombe["x"] >= 13

selected_anscombe = anscombe.loc[
    x_at_least_13,
    ["dataset", "x", "y"],
]

selected_anscombe.to_csv(OUTPUT_PATH, index=False)
print("wrote:", OUTPUT_PATH)
```

## Read back and verify

A **round trip** writes data and then reads the new file back. It catches path, column, selection, and serialization mistakes immediately. A **fresh-runtime execution** starts without names or files created by earlier interactive work, so every required input and directory must be resolved by the visible cells above.

```python
round_trip = pd.read_csv(OUTPUT_PATH)

assert list(anscombe.columns) == ["dataset", "x", "y"]
assert anscombe.shape == (44, 3)
assert list(round_trip.columns) == ["dataset", "x", "y"]
assert round_trip.shape == (7, 3)
assert (round_trip["x"] >= 13).all()
assert round_trip["dataset"].tolist() == [
    "I",
    "I",
    "II",
    "II",
    "III",
    "III",
    "IV",
]
assert round_trip["x"].tolist() == [13.0, 14.0, 13.0, 14.0, 13.0, 14.0, 19.0]
assert np.allclose(
    round_trip["y"].to_numpy(),
    np.array([7.58, 9.96, 8.74, 8.1, 12.74, 8.84, 12.5]),
)

print("Demo 3 fresh-run verification passed")
round_trip
```
