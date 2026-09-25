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

Run this notebook in Colab from the course page, or locally in VS Code with the course `.venv` selected as the kernel (Lecture 03 setup, plus `uv pip install ipykernel`). Colab does not save your edits back to the course repository; to keep them, use **File → Save a copy in Drive**. Never put credentials, tokens, or real patient data in a notebook's source or output.

## Setup

Every demo notebook in this course starts with this cell. It installs pandas 3.0.5, the course version, into the notebook's environment: in Colab, which ships an older pandas, and in your local `.venv` alike. This demo does not use pandas yet; Demos 2 and 3 do.

- pip may print a warning that other Colab packages expect a different pandas. That is expected; the demos do not use those packages.
- If Colab asks you to restart after the install, choose **Runtime → Restart session**, then run the notebook from the top.
- Locally, the `.venv` you made in Lecture 03 with `uv venv --seed` includes pip, so `%pip` installs into it too. When pandas 3.0.5 is already installed there, the cell prints `Note: you may need to restart the kernel to use updated packages.`, perhaps with a notice that a newer pip exists; neither needs any action.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

```python
import sys

print("Python:", sys.version.split()[0])
```

Expect a Python version: 3.12 in Colab, 3.13 in the course `.venv`.

## Cell types and execution

Markdown cells explain, predict, and interpret. Code cells send Python to the kernel. Running a code cell can change state and create stored output; merely editing its visible source does neither.

Before you change anything, add a Markdown cell (**+ Text** in Colab, **+ Markdown** in VS Code) and write what you expect the next code cells to print. Predicting first is what turns a surprise into information.

## Producer and dependent cells

The producer cell defines names. The dependent cell requires those names and computes another value. Run top to bottom, they always give the same answer; run out of order, or in a fresh kernel, they may not.

```python
days = 12
doses_per_day = 3

print("days:", days)
print("doses per day:", doses_per_day)
```

```python
total_doses = days * doses_per_day
print("total doses:", total_doses)
```

```python
print("doses per day in kernel:", doses_per_day)
print("total doses in kernel:", total_doses)
```

Expect `total doses: 36` from the dependent cell and `total doses in kernel: 36` from the last cell.

## Repair the hidden dependency

The cell above printed `36` because the producer cell ran first. Do this by hand now, in this notebook, to see the two failures for yourself:

1. Change `doses_per_day = 3` to `doses_per_day = 2` in the producer cell and run **only** that cell. Then run the last cell again. It prints `total doses in kernel: 36`, a stale value: `total_doses` still holds the old product, because nothing recomputed it.
2. Run the dependent cell (`total_doses = days * doses_per_day`) and the last cell again. Now it prints `24`. The notebook's visible source never said `36` or `24` was correct; execution order decided.
3. Change `doses_per_day` back to `3`, then restart the kernel (Colab: **Runtime → Restart session**; VS Code: **Restart**) and run the last cell on its own. It raises `NameError: name 'doses_per_day' is not defined`, because a fresh kernel holds nothing at all.

**Restart-and-run-all** means starting with empty kernel state and executing every cell from top to bottom. Do that now (Colab: **Runtime → Restart session and run all**): the last cell must print `36` again. Stored output alone is never evidence that this happened.

## Magic commands

A **magic command** is a notebook-only shortcut that starts with `%`. The setup cell used `%pip install`. `%pwd` reports the kernel's current working directory and `%ls` lists the files it can see from there. Check both first whenever a notebook cannot find a file. A cell shows the value of its last line only, so `%pwd` gets a cell to itself.

```python
%pwd
```

Expect `'/content'` in Colab; locally, the folder that holds this notebook.

```python
%ls
```

In Colab, expect `sample_data/`, Colab's own example folder. Locally, expect the demo notebooks, `requirements.txt`, and the `data/` folder, plus `output/` once a later cell or demo has created it.

`%timeit` runs one line many times and reports how long it takes:

```python
%timeit sum(range(1000))
```

Expect a line such as `18.5 μs ± 8.55 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)`; the numbers vary by machine.

## Runtime-local files

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

## Clear outputs before you commit

A notebook saves each cell's output inside the `.ipynb` file. Run this cell, which prints a made-up identifier standing in for something private:

```python
fake_patient = "Test Patient, MRN 000000"
print("Now viewing:", fake_patient)
```

The printed line is now part of the notebook file. Remove it:

1. Clear the outputs: VS Code **Clear All Outputs**; Colab **Edit → Clear all outputs**. The printed line disappears from under the cell.
2. Save the notebook (`Ctrl+S`, or `Cmd+S` on macOS).
3. In VS Code, open **Source Control** and click the notebook to see its diff. The `Now viewing:` output is gone from the saved file; only the code remains. Colab has no Source Control view, so this check is VS Code only.

Clearing outputs does not change kernel state: `fake_patient` still exists until the kernel restarts. A made-up value is safe here; a real one should never be printed in a notebook you commit.

## Fresh-run check

Run **Restart session and run all** (VS Code: **Restart**, then **Run All**) once more. This last cell checks the values a fresh run should produce.

```python
assert days == 12
assert doses_per_day == 3
assert total_doses == 36
assert saved_text == "runtime-local; safe demo content\n"

print("Demo 1 fresh-run check passed: total_doses = 36")
```

Expect `Demo 1 fresh-run check passed: total_doses = 36`. An `AssertionError` means a value was changed without rerunning the cells after it; restart and run all again.
