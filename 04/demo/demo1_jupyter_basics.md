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

# Demo 1: Notebook runtime, state, and fresh execution

A **notebook** is a document made of Markdown cells and code cells. A **kernel** is the Python process that executes code. The kernel's **state** is the collection of names and values currently held in memory. A Colab **runtime** includes that kernel and the files it can see. **Stored output** is text or another result saved beneath a cell; it can remain visible even when it no longer describes current state.

Run the cells from top to bottom; after each step, the text says what to expect. Never put credentials, tokens, or real patient data in a notebook's source or output.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5
```

## Core walkthrough

### Cell types and execution

Markdown cells explain, predict, and interpret. Code cells send Python to the kernel. Running a code cell can change state and create stored output; merely editing its visible source does neither.

Before you change anything, add a Markdown cell (**+ Text** in Colab, **+ Markdown** in VS Code) and write what you expect the next code cells to show. Predicting first is what turns a surprise into information.

### Producer and dependent cells

The producer cell defines names: here a dictionary of lists (Lecture 02) holding ten synthetic clinic visits, one list per column. `pd.DataFrame()` turns it into a table so `display()` can draw it; here the table is just something to look at. The dependent cell needs those names and computes another value. Run top to bottom, they always give the same answer; run out of order, or in a fresh kernel, they may not.

```python
clinic_data = {
    "patient_id": ["P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008", "P009", "P010"],
    "clinic": ["North", "North", "South", "East", "South", "North", "East", "South", "North", "East"],
    "visit_date": ["2026-09-01", "2026-09-01", "2026-09-02", "2026-09-02", "2026-09-03",
                   "2026-09-03", "2026-09-04", "2026-09-04", "2026-09-05", "2026-09-05"],
    "age": [34, 58, 71, 45, 29, 66, 52, 80, 39, 62],
    "temp_c": [36.8, 37.2, 38.4, 36.9, 37.6, 36.6, 38.1, 37.0, 36.7, 37.9],
    "systolic_bp": [118, 132, 145, 124, 112, 138, 128, 150, 116, 141],
    "vaccine_doses": [2, 3, 4, 2, 1, 3, 2, 4, 1, 3],
}

import pandas as pd

visits = pd.DataFrame(clinic_data)  # pandas & dataframes are coming up next in lecture
display(visits)
```

Expect a formatted table of ten rows, `P001` to `P010`, with seven columns from `patient_id` to `vaccine_doses`.

```python
doses = clinic_data["vaccine_doses"]
total_doses = sum(doses)

print("visits:", len(doses))
print("total doses:", total_doses)
```

```python
print("P005 doses in kernel:", clinic_data["vaccine_doses"][4])
print("total doses in kernel:", total_doses)
```

Expect `visits: 10` and `total doses: 25` from the dependent cell. The third cell, the **check cell**, reports what the kernel holds right now: expect `P005 doses in kernel: 1` and `total doses in kernel: 25`.

### Show a result three ways

`display()` drew the table above; `print()` shows the same table as plain text, and a cell's bare last line is shown without either.

```python
print(visits)
len(visits)
```

Expect the table in plain monospaced text, then `10` below it, shown because `len(visits)` is the cell's last line.

### Repair the hidden dependency

The check cell printed `25` because the producer cell ran first. Do this by hand now, in this notebook, to see the two failures for yourself:

1. In the producer cell, change P005's dose count, the fifth `vaccine_doses` value, from `1` to `2`, and run **only** that cell. The table now shows `2`. Run the check cell again: it prints `P005 doses in kernel: 2` but `total doses in kernel: 25`, a stale value: `total_doses` still holds the old sum, because nothing recomputed it.
2. Run the dependent cell and the check cell again. Now the total is `26`. The notebook's visible source never said `25` or `26` was correct; execution order decided.
3. Change the value back to `1`, then restart the kernel (Colab: **Runtime → Restart session**; VS Code: **Restart**) and run the check cell on its own. It raises `NameError: name 'clinic_data' is not defined`, because a fresh kernel holds nothing at all.

**Restart-and-run-all** means starting with empty kernel state and executing every cell from top to bottom. Restart, then **Run All**: the check cell must print `25` again. Stored output alone is never evidence that this happened.

### Fresh-run check

Restart, then run every cell above in order once more. This cell checks the values a fresh run should produce.

```python
assert len(clinic_data["patient_id"]) == 10
assert clinic_data["vaccine_doses"][4] == 1
assert total_doses == 25

print("Demo 1 fresh-run check passed: total_doses = 25")
```

Expect `Demo 1 fresh-run check passed: total_doses = 25`. An `AssertionError` means a value was changed without rerunning the cells after it; restart and run all again.
## Independent practice

Continue on your own after class. These cells reuse the core results; if the runtime closed, run the cells above again first.

### Move a dependent cell

Now break the order on purpose: move the dependent cell (`total_doses = sum(doses)`) above the producer cell. In VS Code, drag it by the bar at its left, or click into it, press `Esc`, then `Alt+Up` (`Option+Up` on Mac); in Colab, click into it and press `Ctrl+M K`. Restart and run all: the run stops at the moved cell with `NameError: name 'clinic_data' is not defined`. Move it back below the producer (`Alt+Down`, or `Ctrl+M J` in Colab), then restart and run all once more; the check cell prints `25`.

### Magic commands

A **magic command** is a notebook-only shortcut that starts with `%`. The setup cell used `%pip install`. `%pwd` reports the kernel's current working directory and `%ls` lists the files it can see from there. Check both first whenever a notebook cannot find a file. A cell shows the value of its last line only, so `%pwd` gets a cell to itself.

```python
%pwd
```

Expect `'/content'` in Colab; locally, the folder that holds this notebook.

```python
%ls
```

In Colab, expect `sample_data/`, Colab's own example folder. On your computer, in `~/04-demo`, expect `data/`, the three demo notebooks, `pyproject.toml`, `setup_demo.sh`, and `uv.lock`, plus `output/` once a later cell or demo has created it. `%ls` leaves out names that start with a dot, such as `.venv` and `.python-version`.

`%timeit` runs one line many times and reports how long it takes:

```python
%timeit sum(range(1000))
```

Expect a line such as `18.5 μs ± 8.55 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)`; the numbers vary by machine.

### Runtime-local files

A **runtime-local file** exists only where the kernel runs. In Colab it disappears when the runtime shuts down, so reliable code recreates it rather than assuming it is still there. The path below is relative, so it lands inside the working directory `%pwd` just showed.

```python
from pathlib import Path

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
runtime_note = output_dir / "runtime_note.txt"

# Write and read it back with open(), from Lecture 02.
with open(runtime_note, "w", encoding="utf-8") as file:
    file.write("runtime-local; safe demo content\n")

with open(runtime_note, "r", encoding="utf-8") as file:
    saved_text = file.read()

print("runtime-local file:", runtime_note)
print(saved_text, end="")
```

Expect `runtime-local file: output/runtime_note.txt` and the line of text you wrote. In Colab, the file also appears under **Files** (the folder icon at the left).

### Clear outputs before you commit

A notebook saves each cell's output inside the `.ipynb` file. Run this cell, which prints a made-up identifier standing in for something private:

```python
fake_patient = "Test Patient, MRN 000000"
print("Now viewing:", fake_patient)
```

The printed line is now part of the notebook file. Remove it:

1. Clear the outputs: VS Code **Clear All Outputs**; Colab **Edit → Clear all outputs**. The printed line disappears from under the cell.
2. Save the notebook (`Ctrl+S`, or `Cmd+S` on macOS).
3. In VS Code, look inside the saved file, as the lecture's "What Git actually commits" snippet does: in the **Explorer**, right-click `demo1_jupyter_basics.ipynb`, choose **Open With…**, then **Text Editor**, and search it (`Ctrl+F`, or `Cmd+F` on macOS) for the made-up name the cell printed. The only match is the code line that defines `fake_patient`; before step 1, a second match sat in that cell's `"outputs"`. In a Git repository, **Source Control** shows the same change in the notebook's diff (Lecture 02). Colab has no text view of the file, so this check is VS Code only.

Clearing outputs does not change kernel state: `fake_patient` still exists until the kernel restarts. A made-up value is safe here; a real one should never be printed in a notebook you commit.
