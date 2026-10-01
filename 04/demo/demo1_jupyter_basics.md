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

Run the cells from top to bottom, in Colab or on your computer; after each step, the text says what to expect. Never put credentials, tokens, or real patient data in a notebook's source or output.

Choose a route below. The **core walkthrough** is the demonstration path; **independent practice** is for you to work through after class. In a fresh runtime, run Setup and the core first. **Run all** completes both routes.

| Route | Work and visible checkpoint |
| --- | --- |
| [Core walkthrough](#core-walkthrough) | Predict, run, repair stale state, and restart: `total_doses = 36`. |
| [Independent practice](#independent-practice) | Move a dependent cell; inspect magic commands, runtime files, and saved outputs. |

## Setup

**In Colab**, open this notebook from the lecture page's **Live notebooks in Colab** link. A new runtime needs only the install cell below. Colab does not save your edits back to the course repository; to keep them, use **File → Save a copy in Drive**.

**On your computer**, run these lines once in VS Code's terminal (**Terminal → New Terminal**; on Windows, the **WSL: Ubuntu** window from Lecture 01). The first line downloads the three demo notebooks, their data, and the project's environment files (`pyproject.toml`, `uv.lock`, and `.python-version`) into a new folder, `~/04-demo`; the rest create, activate, and fill its environment, as in Lecture 03:

```shell
curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/04/demo/setup_demo.sh | sh
cd ~/04-demo
uv venv --seed
source .venv/bin/activate
uv sync
```

Among the lines they print, these show each step worked; `3.13.x` is whichever 3.13 release you have:

```text
Made ~/04-demo with the Lecture 04 demo notebooks, their environment files, and data/clinic_visits.csv.
Using CPython 3.13.x
 + ipykernel==6.29.5
 + pandas==3.0.5
```

`pyproject.toml` lists **ipykernel**, the package a notebook kernel needs, so `uv sync` installed it with pandas. In VS Code, choose **File → Open Folder…**, open `04-demo` in your home folder, open `demo1_jupyter_basics.ipynb`, click **Select Kernel** at the top right, and choose the Python in `.venv` (VS Code lists it under **Python Environments**). In Git Bash, activate with `source .venv/Scripts/activate` instead.

The setup script never overwrites earlier work: run it a second time and `mkdir` reports that `~/04-demo` already exists (`File exists`), and nothing else happens. To start over, rename the old folder first with `mv ~/04-demo ~/04-demo-old`, then run the `curl` line again. To come back to the demos later, open the `04-demo` folder in VS Code and choose the `.venv` kernel; the environment stays in the folder.

## Install the course's pandas

Every demo notebook in this course starts with this cell. It installs pandas 3.0.5, the course version, into the kernel's environment. This demo does not use pandas yet; Demos 2 and 3 do.

- In Colab, which ships an older pandas, pip may print `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...` and a line such as `google-colab ... requires pandas==..., but you have pandas 3.0.5 which is incompatible.` That is expected: the install still succeeded, and the demos do not use those Colab packages.
- If Colab asks you to restart after the install, choose **Runtime → Restart session**, then run the notebook from the top.
- On your computer, `uv sync` already installed pandas 3.0.5, so the cell changes nothing. It prints `Note: you may need to restart the kernel to use updated packages.`, perhaps with a notice that a newer pip exists; neither needs any action.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

```python
import sys

print("Python:", sys.version.split()[0])
```

Expect a Python version: Colab manages its runtime's version, while `~/04-demo`'s `.venv` uses the course's Python 3.13. A version other than 3.13 on your computer means the kernel is not that `.venv`: click the kernel name at the top right and choose it.

## Core walkthrough

### Cell types and execution

Markdown cells explain, predict, and interpret. Code cells send Python to the kernel. Running a code cell can change state and create stored output; merely editing its visible source does neither.

Before you change anything, add a Markdown cell (**+ Text** in Colab, **+ Markdown** in VS Code) and write what you expect the next code cells to print. Predicting first is what turns a surprise into information.

### Producer and dependent cells

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

Expect `total doses: 36` from the dependent cell. The third cell, the **check cell**, reports what the kernel holds right now: expect `total doses in kernel: 36`.

### Repair the hidden dependency

The cell above printed `36` because the producer cell ran first. Do this by hand now, in this notebook, to see the two failures for yourself:

1. Change `doses_per_day = 3` to `doses_per_day = 2` in the producer cell and run **only** that cell. Then run the check cell again. It prints `total doses in kernel: 36`, a stale value: `total_doses` still holds the old product, because nothing recomputed it.
2. Run the dependent cell (`total_doses = days * doses_per_day`) and the check cell again. Now it prints `24`. The notebook's visible source never said `36` or `24` was correct; execution order decided.
3. Change `doses_per_day` back to `3`, then restart the kernel (Colab: **Runtime → Restart session**; VS Code: **Restart**) and run the check cell on its own. It raises `NameError: name 'doses_per_day' is not defined`, because a fresh kernel holds nothing at all.

**Restart-and-run-all** means starting with empty kernel state and executing every cell from top to bottom. For the core route, restart (Colab: **Runtime → Restart session**; VS Code: **Restart**) and run Setup and the core cells in order: the check cell must print `36` again. Use **Run All** when completing both routes. Stored output alone is never evidence that this happened.

### Fresh-run check

Restart, then run Setup and the core cells in order once more. This cell checks the values a fresh run should produce.

```python
assert days == 12
assert doses_per_day == 3
assert total_doses == 36

print("Demo 1 fresh-run check passed: total_doses = 36")
```

Expect `Demo 1 fresh-run check passed: total_doses = 36`. An `AssertionError` means a value was changed without rerunning the cells after it; restart and run all again.
## Independent practice

Continue on your own after class. These cells reuse the core results; if the runtime closed, run Setup and the core again first.

### Move a dependent cell

Now break the order on purpose: move the dependent cell (`total_doses = days * doses_per_day`) above the producer cell. In VS Code, drag it by the bar at its left, or click into it, press `Esc`, then `Alt+Up` (`Option+Up` on Mac); in Colab, click into it and press `Ctrl+M K`. Restart and run all: the run stops at the moved cell with `NameError: name 'days' is not defined`. Move it back below the producer (`Alt+Down`, or `Ctrl+M J` in Colab), then restart and run all once more; the check cell prints `36`.

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
