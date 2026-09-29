---
notion:
  title_line: "# NumPy Arrays & Virtual Environments"
  role: lecture
  status: mapped
  page_id: "27ed9fdd-1a1a-80a7-8532-e70d7000dae8"
  url: "https://app.notion.com/p/27ed9fdd1a1a80a78532e70d7000dae8"
---

# NumPy Arrays & Virtual Environments

See [BONUS.md](BONUS.md) for the optional extensions.

[Live Demo Guide](demo/DEMO_GUIDE.md)

# Virtual Environments

![xkcd 1987: Python Environment. Virtual environments prevent package chaos](media/xkcd_1987.png)

Lecture 02 imported `math`, which ships with Python. NumPy does not: it is a third-party **package**, installable code that provides modules, and `import numpy` works only after it is installed. A **virtual environment** is a project folder, `.venv`, holding its own Python **interpreter** (the program that runs Python code) and its own installed packages, so each project keeps the versions it was written with.

```text
Project A → A/.venv → Python 3.13, numpy 2.3.3
Project B → B/.venv → Python 3.12, numpy 1.26.4
```

## Course Environment

- Python 3.13, installed and set as the default with uv
- One environment per project, in a folder named `.venv`, with the project's packages listed in `pyproject.toml`
- NumPy 2.3.3 in this lecture; pandas 3.0.5 from Lecture 04
- uv manages all of it; `requirements.txt`, standard-library `venv`, and Conda are alternatives later in this topic

Make 3.13 the default when you install it, or afterwards:

| When | Command | What it sets |
| --- | --- | --- |
| During install | `uv python install 3.13 --default` | Installs 3.13 and makes the `python` and `python3` commands run it (Lecture 01). uv may warn that `--default` is experimental; the install still works. |
| After install | `uv python pin --global 3.13` | Makes 3.13 the version uv uses for new environments. |

Run the second one even if you ran the first: `--default` sets only the `python` commands, and without a pin `uv venv` takes the newest Python uv has installed.

## Using uv

[uv documentation](https://docs.astral.sh/uv/)

`uv venv --seed` creates `.venv` in the current folder with uv's default Python. **Activation** switches the current terminal to that environment, so `python` runs the copy inside `.venv` and uv installs packages there; the prompt then starts with the project folder's name.

```text
~/assignment-03 $ uv venv --seed
Using CPython 3.13.14
Creating virtual environment with seed packages at: .venv
 + pip==26.2.1
Activate with: source .venv/bin/activate
~/assignment-03 $ source .venv/bin/activate
(assignment-03) ~/assignment-03 $ python --version
Python 3.13.14
```

<callout icon="⚠️" color="yellow_bg">
	Keep `.venv/` out of Git: add the line `.venv/` to the project's `.gitignore` (Lecture 02). It holds thousands of installed files that work only on the computer that built them; the project's records, below, rebuild it anywhere.
</callout>

### Reference Card: uv Environment Workflow

| Task | Command | Result |
| :--- | :--- | :--- |
| Set the default Python | `uv python pin --global 3.13` | New environments use 3.13 wherever no project pin applies. |
| Pin one project | `uv python pin 3.13` | Writes `.python-version` in this folder; `uv venv` here uses that version even when the global pin differs, so a project can pin 3.12. |
| Create environment | `uv venv --seed` | Creates `.venv` with the pinned Python; `--seed` adds pip, which Lecture 04's notebooks use. |
| Activate | `source .venv/bin/activate` (PowerShell: `.\.venv\Scripts\Activate.ps1`) | The prompt starts with the folder name, such as `(assignment-03)`; `python` now runs the environment's interpreter. |
| Verify | `python --version` and `python -c "import numpy as np; print(np.__version__)"` | Confirms the Python and NumPy versions. |
| Leave environment | `deactivate` | Returns to the previous shell environment. |

If `.venv` already exists, `uv venv` asks whether to replace it: `n` keeps it (uv then prints `error: Failed to create virtual environment` and changes nothing), and `y` makes a fresh, empty one that needs its packages installed again.

## Recording Packages in `pyproject.toml`

A **dependency** is a package a project needs. `pyproject.toml` is Python's standard project file and the course's default record of dependencies: its `dependencies` list names the packages the project's own code imports. `uv init --bare` writes the file, and `uv add` installs a package and adds it to the list in one step:

```toml
[project]
name = "clinic-project"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "numpy==2.3.3",
]
```

`uv add` also writes `uv.lock`, a **lock file** listing the exact version of every package the environment needs, including any that your packages need in turn, so a rebuild installs the same versions. uv keeps `uv.lock` up to date; commit it beside `pyproject.toml` and leave its contents to uv.

### Reference Card: `pyproject.toml` Projects

| Task | Command | Result |
| :--- | :--- | :--- |
| Start a project | `uv init --bare` | Writes `pyproject.toml` with an empty `dependencies` list; the project is named after the folder. |
| Add a package | `uv add numpy==2.3.3` | Installs that exact version into `.venv`, adds it to `pyproject.toml`, and updates `uv.lock`. |
| Rebuild from the records | `uv sync` | Makes `.venv` match `pyproject.toml` and `uv.lock`: installs what is missing and removes packages they do not list. |
| Run in the environment | `uv run python script.py` | Runs the command with the project's `.venv`, active or not. |

### Code Snippet: Create and Verify an Environment

In a new project folder:

```bash
mkdir ~/clinic-project
cd ~/clinic-project
uv python pin 3.13                                      # Pinned `.python-version` to `3.13`
uv init --bare                                          # Initialized project `clinic-project`
uv venv --seed
source .venv/bin/activate
uv add numpy==2.3.3                                     # + numpy==2.3.3
python --version                                        # Python 3.13.14
python -c "import numpy as np; print(np.__version__)"   # 2.3.3
deactivate
uv run python -c "import numpy as np; print(np.__version__)"   # 2.3.3, with no environment active
```

`.python-version`, `pyproject.toml`, and `uv.lock` are the project's records: together they say which Python and which packages to rebuild. `uv init --bare` runs once per project; in a folder that already has `pyproject.toml`, such as an assignment handout, it stops with `error: Project is already initialized`.

`uv sync` also creates `.venv` when it is missing, but without pip, so run `uv venv --seed` first.

## Using `requirements.txt` (alternative)

Many projects, and tools such as pip and Google Colab, list their packages in a **requirements file**, `requirements.txt`, instead: one package per line, with `==` pinning an exact version. No lock file sits beside it, and uv installs from it with `uv pip` commands.

```text
numpy==2.3.3
pandas==3.0.5
```

### Reference Card: Requirements Files

- `uv pip install -r requirements.txt`: Install every package the file lists into the active environment; `pyproject.toml` stays unchanged.
- `uv export --no-hashes > requirements.txt`: Write the packages `uv.lock` records as a requirements file; `--no-hashes` leaves out long download checksums.
- `uv pip freeze > requirements.txt`: Write every package installed in the active environment, pip included.
- `python -m pip install -r requirements.txt`: The same install with pip, as Colab and the alternatives below do.

In a `pyproject.toml` project, add packages with `uv add`, not `uv pip install`: `uv sync` removes any package that `pyproject.toml` and `uv.lock` do not list, printing a line such as `- six==1.17.0` for each.

### Code Snippet: Share the Environment as `requirements.txt`

From `~/clinic-project`:

```bash
uv export --no-hashes > requirements.txt
cat requirements.txt
```

```text
# This file was autogenerated by uv via the following command:
#    uv export --no-hashes
numpy==2.3.3
    # via clinic-project
```

In a new folder holding a copy of that file, `uv venv --seed`, `source .venv/bin/activate`, and `uv pip install -r requirements.txt` install the same NumPy.

## Which Python Is Running?

`python --version` reports the version, and `sys.executable` reports the exact interpreter file, which shows whether `python` runs the project's `.venv`; `-c` runs the Python string that follows it:

```text
~/assignment-03 $ source .venv/bin/activate
(assignment-03) ~/assignment-03 $ python -c "import sys; print(sys.executable)"
/home/alice/assignment-03/.venv/bin/python
```

### When `import numpy` Fails

```text
$ python -c "import numpy as np"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

`ModuleNotFoundError` means the Python that ran the code cannot find the package. The usual causes:

- **The environment is not active**, or VS Code picked another interpreter: `sys.executable` does not end in `.venv/bin/python`. Run `source .venv/bin/activate` in this terminal, and in VS Code run **Python: Select Interpreter** (Lecture 02) and choose `./.venv/bin/python`.
- **The project's packages are not installed**: run `uv sync`, or `uv pip install -r requirements.txt` for a `requirements.txt` project.
- **The package is new to this project**: `uv add pandas==3.0.5` installs it and records it in `pyproject.toml`.

## Recreate an Environment

A result is **reproducible** when another person can rebuild the software environment and rerun the program on the same inputs. Commit the records, `.python-version`, `pyproject.toml`, and `uv.lock`, and check them by rebuilding in a new folder.

### Code Snippet: Recreate from the Records

With no environment active, from `~/clinic-project`:

```bash
mkdir recreation-check
cp .python-version pyproject.toml uv.lock recreation-check/
cd recreation-check
uv venv --seed
uv sync                                                        # + numpy==2.3.3
uv run python -c "import numpy as np; print(np.__version__)"   # 2.3.3
cd ..
```

`uv venv` reads the copied `.python-version`, so the rebuild uses Python 3.13, and `uv sync` installs the versions `uv.lock` records. The same two commands set up a project someone else made, such as an assignment handout. Run with another environment active, `uv sync` warns that `VIRTUAL_ENV` does not match the project environment and ignores it; `deactivate` first.

## Using standard-library venv (alternative)

With Python 3.13 installed, `venv` creates an environment with pip included. pip reads `requirements.txt` rather than `uv.lock`:

### Reference Card: standard-library `venv`

| Task | Command | Note |
| :--- | :--- | :--- |
| Create | `python -m venv .venv` | Use the already-installed Python 3.13 interpreter. |
| Activate | `source .venv/bin/activate` | In PowerShell, use `.\.venv\Scripts\Activate.ps1`. |
| Install | `python -m pip install -r requirements.txt` | Uses the active environment's pip. |
| Leave | `deactivate` | Returns to the previous shell environment. |

Choose one environment tool for a project; these are alternative routes, not consecutive steps.

## Using Conda (alternative comparison)

[Conda documentation](https://docs.conda.io/)

Conda manages Python environments and packages, including non-Python dependencies.

### Reference Card: Conda alternative

| Task | Command | Result |
| :--- | :--- | :--- |
| Create | `conda create --prefix ./.venv python=3.13 pip` | Creates the same `.venv` location with Conda. |
| Activate | `conda activate ./.venv` (PowerShell: `conda activate .\.venv`) | Selects the Conda environment. |
| Install | `python -m pip install -r requirements.txt` | Installs the packages a requirements file lists. |
| Leave | `conda deactivate` | Returns to the previous environment. |

![xkcd 2347: Dependency. Every project stands on packages other people maintain, which is why yours records exactly which versions it needs](media/xkcd_2347.png)

# Shell Pipelines and Scripts

## Pipelines

A **pipe** (`|`) sends a command's output into another command, where `>` (Lecture 01) sends it into a file. A **pipeline** chains small commands, each doing one job on the previous command's output, and answers quick questions about a file before any Python, such as how many encounters each clinic had.

Demo 1 creates `data/raw/encounters.csv`, which has a header and six rows. Each stage receives the previous stage's output:

| Stage | Command | Output |
| --- | --- | --- |
| Input | `cat data/raw/encounters.csv` | `patient_id,age,systolic_bp,clinic`, `P001,54,128,Cardiology`, … 7 lines |
| Skip the header | `tail -n +2` | `P001,54,128,Cardiology`, `P002,39,118,Primary Care`, … 6 lines |
| Keep field 4 | `cut -d',' -f4` | `Cardiology`, `Primary Care`, `Nephrology`, `Cardiology`, `Nephrology`, `Cardiology` |
| Put equal lines together | `sort` | `Cardiology`, `Cardiology`, `Cardiology`, `Nephrology`, `Nephrology`, `Primary Care` |
| Count each group | `uniq -c` | `3 Cardiology`, `2 Nephrology`, `1 Primary Care` |

### Reference Card: Pipeline Building Blocks

- `tail -n +2 FILE`: Skip the first line (the header).
- `cut -d',' -f4`: Select the fourth comma-separated field.
- `sort`: Put equal lines next to one another.
- `uniq -c`: Count adjacent equal lines.
- `head -n 5`: Keep the first five lines.
- `wc -l`: Count lines.
- `command > FILE` / `command >> FILE`: Replace / append file contents.

### Code Snippet: Count Encounters per Clinic

```bash
tail -n +2 data/raw/encounters.csv | cut -d',' -f4 | sort | uniq -c
tail -n +2 data/raw/encounters.csv | wc -l
```

```text
      3 Cardiology
      2 Nephrology
      1 Primary Care
6
```

`uniq` only merges lines that are next to each other, so `sort` first: without `sort`, the same pipeline prints six separate lines such as `1 Cardiology` and `1 Nephrology`. `cut` splits at every comma, so use it only on simple files with no commas inside a field.

## Variables and Timestamps

Running a pipeline again with `>` replaces the previous summary. Naming each run's output with the time it ran keeps every result and lets the log show which run wrote which file. The shell can store text in a **variable** and capture a command's output with **command substitution**, `$(...)`.

### Reference Card: Shell Variables and Timestamps

- `name=value`: Store text; no spaces around `=`.
- `"$name"` / `"${name}"`: Use the value; quotes keep spaces, and braces mark where the name ends, as in `summary_${timestamp}.txt`.
- `$(command)`: Replace the expression with the command's output.
- `date +"%Y%m%d_%H%M%S"`: Print the current time as year, month, day, underscore, hour, minute, second, such as `20260918_161539`, which sorts in time order.

### Code Snippet: Name and Log a Run

```bash
mkdir -p results logs
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "Run: $timestamp" > "results/summary_${timestamp}.txt"
echo "${timestamp} complete" >> logs/processing.log
ls results
cat logs/processing.log
```

```text
summary_20260918_162001.txt
20260918_162001 complete
```

One saved value labels both the result file and the log line; your timestamp will differ.

## Shell Scripts

A **shell script** saves commands in a file, so a whole pipeline reruns with one command whenever the data change. Build it the way Lecture 01 did: `cat > FILE.sh`, paste, then run it with `bash FILE.sh`.

### Reference Card: Shell Scripts

- `cat > count_clinics.sh`: Paste the script, press **Enter**, then **Ctrl+C** (Lecture 01).
- `#!/bin/bash`: The first line; names the shell the script expects.
- `# note`: A comment; Bash skips it.
- `\` at the end of a line: Continue the same command on the next line.
- `bash count_clinics.sh`: Run the script from top to bottom.

### Code Snippet: Save a Pipeline as a Script

```bash
#!/bin/bash
# Count encounters per clinic; save the counts under this run's timestamp.
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p results
tail -n +2 data/raw/encounters.csv \
  | cut -d',' -f4 | sort | uniq -c > "results/clinic_counts_${timestamp}.txt"
echo "Saved results/clinic_counts_${timestamp}.txt"
```

```bash
bash count_clinics.sh
cat results/clinic_counts_*.txt
```

```text
Saved results/clinic_counts_20260918_162310.txt
      3 Cardiology
      2 Nephrology
      1 Primary Care
```

Each run adds another timestamped file. Scripts can also take arguments, define functions, and stop at the first failing command; [the bonus page](BONUS.md) covers those.

# LIVE DEMO!

# Checking Types and Looping over Lists

Three built-in tools shorten everyday list code: `isinstance()` checks what a value is, `zip()` and `reversed()` walk lists, and a list comprehension builds a list in one line.

## Checking Types

Values read from a file start as text, so check a value's type before calculating with it. **Introspection** means asking an object what it is while the program runs: `type()` (Lecture 01) names the type, and `isinstance()` answers yes or no, which suits an `if`.

### Reference Card: Object Introspection

- `type(value)`: Show the value's exact type.
- `isinstance(value, type)`: Test whether a value has the requested type.
- `dir(value)`: List available attributes and methods.
- `help(value)` / `help(str.split)`: Read documentation; press **Q** to leave a paged view.
- `id(value)`: Get an identity number for an object during its lifetime.

### Code Snippet: Inspect Values Before Using Them

```python
readings = ["72", 80, "88"]   # some typed in, some read from a file as text
print(type(readings[0]), type(readings[1]))  # <class 'str'> <class 'int'>
clean = []
for value in readings:
    if isinstance(value, str):
        value = int(value)
    clean.append(value)
print(clean, sum(clean))      # [72, 80, 88] 240
```

`sum(readings)` on the original list raises `TypeError: unsupported operand type(s) for +: 'int' and 'str'`.

![xkcd 1537: Types. A language that guesses what mixed types mean gives surprises; Python raises TypeError instead, so check types first](media/xkcd_1537.png)

## Sequence Functions

Beside `enumerate()` and `sorted()`, two more built-ins handle common loop needs: `zip()` walks two lists side by side, keeping each patient's ID with that patient's reading, and `reversed()` walks a sequence from the end.

### Reference Card: Sequence Functions

- `enumerate(items, start=0)`: Yield position-value pairs.
- `zip(left, right)`: Yield pairs until the shorter input ends.
- `reversed(items)`: Iterate from the last item to the first.
- `sorted(items)`: Return a new sorted list.

### Code Snippet: Keep Related Values Together

```python
patients = ["P001", "P002", "P003"]
systolic = [128, 142, 118]
for number, patient in enumerate(patients, start=1):
    print(f"{number}: {patient}")     # 1: P001, then 2: P002, 3: P003
for patient, reading in zip(patients, systolic):
    print(f"{patient}: {reading}")    # P001: 128, then P002: 142, P003: 118
print(list(reversed(patients)))       # ['P003', 'P002', 'P001']
```

## List Comprehensions

A **list comprehension** is an optional shorthand for a loop that builds a list: `[expression for item in items if condition]`. Read it as "make this, for each item, keeping only items that pass." A plain loop always works, so treat comprehensions as a shortcut to recognize now; [the bonus page](BONUS.md) shows more forms.

### Reference Card: List Comprehensions

- `[expr for x in items]`: Transform every item; returns a new list of the same length.
- `[x for x in items if condition]`: Keep only items that pass; returns a new, possibly shorter list.
- `[expr for x in items if condition]`: Filter, then transform.

### Code Snippet: The Same List Two Ways

```python
temps_f = [98.6, 101.2, 99.5, 103.1]

fevers = []                                   # a loop that builds a list
for t in temps_f:
    if t >= 100.4:
        fevers.append(t)
print(fevers)                                 # [101.2, 103.1]

fevers = [t for t in temps_f if t >= 100.4]   # the same list in one line
print(fevers)                                 # [101.2, 103.1]

doses_g = [mg / 1000 for mg in [250, 500, 125]]
print(doses_g)                                # [0.25, 0.5, 0.125]
```

# NumPy Basics

![It's pronounced "num pie": NumPy is short for Numerical Python, whatever the cat says](media/numpy.webp)

## Why NumPy

**NumPy** (Numerical Python) is the package for calculating on many numbers at once, and pandas (Lecture 04) is built on it. Its core object is the **array**, a grid of values that all share one data type, so one expression such as `readings * 2` applies to every element with no loop. Writing calculations this way is called **vectorization**, and on a wearable's 86,400 once-a-second readings per day it is far faster than looping over a list.

```text
Python list: [1, 2, 3] * 2  → [1, 2, 3, 1, 2, 3]   repeats the list
NumPy array: [1, 2, 3] * 2  → [2, 4, 6]            multiplies each value
```

### Code Snippet: The Same Calculation Two Ways

```python
my_list = [1, 2, 3, 4, 5]
doubled_list = [x * 2 for x in my_list]
print(doubled_list)         # [2, 4, 6, 8, 10]

import numpy as np
my_array = np.array(my_list)
doubled_array = my_array * 2
print(doubled_array)        # [ 2  4  6  8 10]
```

`import numpy as np` loads NumPy under its standard alias `np` (Lecture 02's `import module as alias`), and `np.array()` turns a list into an array. With five numbers the time difference is too small to notice. Demo 2 adds a 2 bpm calibration offset to one million heart-rate readings both ways; on one test machine the list took about 24 ms and the array under 1 ms.

## NumPy Data Types

NumPy stores values in its own **data types** (**dtypes**), such as `int64` and `float64`, rather than Python's `int` and `float`. Each dtype has a fixed size, so an array's values sit side by side in memory and NumPy's compiled code processes them in one pass; that is where the speed comes from. NumPy's types mix with Python's in arithmetic and convert back and forth, and pandas' numeric columns (Lecture 04) use these same dtypes.

| Python type | NumPy dtype | `np.array(...)` of | Holds |
| --- | --- | --- | --- |
| `int` | `int64` | `[1, 2, 3]` | Whole numbers up to about 9.2 × 10¹⁸ |
| `float` | `float64` | `[1.5, 2]` | Decimals, to about 15 significant digits |
| `bool` | `bool` | `[True, False]` | `True` or `False` |
| `str` | `<U5` | `["98.6", "101.2"]` | Text up to 5 characters |

### Reference Card: Data Types

- `arr.dtype`: The element type, such as `int64`.
- `np.array(values, dtype=np.float64)`: Choose the dtype when creating the array.
- `arr.astype(float)`: Return a new array converted to another dtype; also parses numeric text. Text that is not a number, such as `"NA"`, raises `ValueError`, and converting decimals to `int` drops the decimal part without rounding.
- `arr.tolist()` / `float(arr[0])`: Convert back to plain Python values.

### Code Snippet: Convert Numeric Text

```python
temps = np.array(["98.6", "101.2", "99.5"])
print(temps.dtype)          # <U5: text of up to 5 characters
temps_f = temps.astype(float)
print(temps_f.dtype)        # float64
print(temps_f + 1)          # [ 99.6 102.2 100.5]
print(temps_f.astype(int))  # [ 98 101  99]: decimals dropped, not rounded
print(temps_f.tolist())     # [98.6, 101.2, 99.5]: plain Python floats
```

![xkcd 571: Can't Sleep. A 16-bit integer counts to 32,767 and then wraps to -32,768; NumPy's fixed-size integers wrap the same way when a value outgrows its dtype](media/xkcd_571.png)

## NumPy Arrays

`np.array()` turns a list into a one-dimensional (1-D) array and a list of lists into a 2-D array, one inner list per row. An array's **shape** is the length of each dimension, rows first:

```text
          column 0  column 1  column 2
row 0  →       1         2         3
row 1  →       4         5         6        shape (2, 3)
```

### Creating Arrays

#### Reference Card: Array Creation

| Function | Purpose | Example output |
| :--- | :--- | :--- |
| `np.array(values)` | Converts a list or nested lists to an array. | 1D or 2D `ndarray` |
| `np.zeros(shape)` | Fills an array with zeros. | `float` array |
| `np.ones(shape)` | Fills an array with ones. | `float` array |
| `np.arange(stop)` / `np.arange(start, stop)` | Creates evenly spaced integers, stopping before `stop`. | `np.arange(5)` → `[0 1 2 3 4]`; `np.arange(1, 13)` → 1 through 12 |
| `np.full(shape, value)` | Fills an array with one value. | Array matching `shape` |

#### Code Snippet: Create Arrays

```python
arr = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)                  # [1 2 3 4 5]
print(arr_2d)               # [[1 2 3]
                            #  [4 5 6]]
print(np.zeros(5))          # [0. 0. 0. 0. 0.]: floats print with a trailing dot
print(np.ones((2, 3)))      # [[1. 1. 1.]
                            #  [1. 1. 1.]]
print(np.arange(10))        # [0 1 2 3 4 5 6 7 8 9]
print(np.full((2, 3), 7))   # [[7 7 7]
                            #  [7 7 7]]
```

### Array Properties

#### Reference Card: Array Properties

| Attribute | Meaning | Example for `arr_2d` |
| :--- | :--- | :--- |
| `arr_2d.shape` | Length along each dimension. | `(2, 3)` |
| `arr_2d.ndim` | Number of dimensions. | `2` |
| `arr_2d.size` | Total number of elements. | `6` |
| `arr_2d.dtype` | Element data type. | `int64` |

### Random Arrays

Simulated data lets you practice an analysis before touching patient records. A **random number generator** produces values that look random, and a **seed** makes it produce the same sequence every run, so your results match your classmates'. `integers(low, high)` never returns `high`, so use `101` to include 100, and `size` can be a shape such as `(100, 5)`.

#### Reference Card: Random Number Generation

| Method | Purpose | Example |
| :--- | :--- | :--- |
| `np.random.default_rng(seed)` | Creates a random generator; the seed makes the sequence reproducible. | `rng = np.random.default_rng(seed=42)` |
| `rng.random(size)` | Uniform floats in `[0, 1)`. | `rng.random(5)` |
| `rng.integers(low, high, size)` | Integers in `[low, high)`; `size` may be a shape. | `rng.integers(60, 101, size=(100, 5))` |
| `rng.standard_normal(size)` | Standard normal draws. | `rng.standard_normal(5)` |

#### Code Snippet: Generate Reproducible Practice Data

```python
rng = np.random.default_rng(seed=42)
ages = rng.integers(18, 91, size=5)               # 18 through 90
heart_rates = rng.integers(60, 101, size=(2, 3))  # 2 patients × 3 readings, 60 through 100 bpm
print(ages)          # [24 74 65 50 49]
print(heart_rates)   # [[95 63 88]
                     #  [68 63 81]]
```

Run it again with `seed=42` and the same numbers print. Older tutorials call `np.random.seed()` and `np.random.randn()`; use `default_rng()` in new code.

### Vectorized Arithmetic

The arithmetic operators `+`, `-`, `*`, `/`, and `**` work on whole arrays: each applies to every element and returns a new array of the same shape. Two arrays of the same shape combine position by position. An array and a single number combine by applying that number to every element, which NumPy calls **broadcasting**.

| Expression | Printed result | Meaning |
| --- | --- | --- |
| `systolic` | `[128 142 118]` | Three readings in mmHg |
| `systolic - 120` | `[ 8 22 -2]` | How far each reading is above 120 mmHg |
| `systolic * 0.133` | `[17.024 18.886 15.694]` | The same readings in kPa |

#### Code Snippet: Calculate Without an Explicit Loop

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([5, 4, 3, 2, 1])
print(arr1 + arr2)   # [6 6 6 6 6]
print(arr1 * arr2)   # [5 8 9 8 5]
print(arr1 ** 2)     # [ 1  4  9 16 25]
print(arr1 * 2)      # [ 2  4  6  8 10]: one number broadcast to every element
```

Two arrays combine position by position, so their shapes must match, or one must stretch to fit the other the way a single number does in `arr1 * 2` ([the bonus page](BONUS.md) gives the stretching rules). Shapes that do neither raise an error: `np.array([1, 2, 3]) + np.array([1, 2])` raises `ValueError: operands could not be broadcast together with shapes (3,) (2,)`.

## Indexing and Slicing

Array indexing extends Lecture 02's list indexing and slicing: the same square brackets, positions counted from 0, negative positions counted from the end, and half-open `start:stop` slices. A multidimensional array takes one index per dimension, separated by commas.

### One-Dimensional Arrays

#### Reference Card: One-Dimensional Indexing

| Pattern | Purpose | Example |
| :--- | :--- | :--- |
| `arr[i]` | Select one element. | `arr[0]` |
| `arr[-1]` | Select the last element. | `arr[-1]` |
| `arr[start:stop]` | Select a half-open slice. | `arr[2:7]` |
| `arr[::step]` | Select every `step`th element. | `arr[::2]` |

#### Code Snippet: Slice a One-Dimensional Array

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr[0], arr[-1])   # 0 9
print(arr[2:7])          # [2 3 4 5 6]
print(arr[::2])          # [0 2 4 6 8]
```

### Multidimensional Arrays

A 2-D array is a table: the first index picks rows (**axis 0**) and the second picks columns (**axis 1**). Leaving out the column index selects a whole row, so `bp[1]` is the same as `bp[1, :]`. Each further dimension adds one more index: a 3-D array of patients × days × readings takes `arr[patient, day, reading]`.

```text
bp              visit 1  visit 2  visit 3
patient 0  →      128      131      126
patient 1  →      142      145     [139]   ← bp[1, 2]
patient 2  →      118      121      119
bp[:, 0] is the visit 1 column: [128 142 118]
```

#### Reference Card: Rows, Columns, and Blocks

| Selection | Meaning | Result shape |
| --- | --- | --- |
| `arr[row, col]` | One element of a 2-D array | Scalar |
| `arr[row, :]` or `arr[row]` | All columns in one row | 1-D row |
| `arr[:, col]` | All rows in one column | 1-D column |
| `arr[:2, 1:3]` | First two rows, columns 1 and 2 | 2-D block |
| `arr3d[i]`, `arr3d[i, j]`, `arr3d[:, :, k]` | In a 3-D array, each single-number index removes one dimension; `:` keeps it | 2-D, 1-D, 2-D |

#### Code Snippet: Select Cells, Rows, Columns, and Blocks

```python
bp = np.array([[128, 131, 126],   # patient 0: systolic at visits 1-3
               [142, 145, 139],   # patient 1
               [118, 121, 119]])  # patient 2
print(bp[1, 2])    # 139: patient 1, visit 3
print(bp[1])       # [142 145 139]: every visit for patient 1
print(bp[:, 0])    # [128 142 118]: visit 1 for every patient
print(bp[:2, 1:])  # [[131 126]
                   #  [145 139]]: patients 0-1, visits 2-3

week = np.zeros((2, 7, 3))   # 2 patients × 7 days × 3 readings a day
print(week[0].shape)         # (7, 3): patient 0's whole week
print(week[0, 6].shape)      # (3,): patient 0, day 7
print(week[:, :, 0].shape)   # (2, 7): every patient's first reading of each day
```

# LIVE DEMO!

# Selecting and Reshaping Arrays

NumPy selects values by a condition or by a list of positions, and rearranges values into new shapes. Some results share data with the original array and some are copies, so this topic starts with when a change to one array shows up in another.

## Names, Aliases, and Mutability

An object is **mutable** when its contents can change in place: lists (Lecture 02), dictionaries, sets, and arrays are mutable, while numbers, strings, and tuples are not. Assignment binds a name to an object without copying it, so `same_values = values` makes an **alias**, a second name for the same array:

```text
values ───────┐
              ├──> [10, 20, 30]   one array
same_values ──┘
copied_values ───> [10, 20, 30]   a separate array
```

### Code Snippet: Alias Versus Copy

```python
values = np.array([10, 20, 30])
same_values = values
same_values[0] = 99
print(values)                 # [99 20 30]
print(same_values is values)  # True: `is` compares identity, `==` compares values

copied_values = values.copy()
copied_values[0] = 10
print(values)                 # [99 20 30]
print(copied_values)          # [10 20 30]
```

<callout icon="⚠️" color="yellow_bg">
	`b = a` never copies. When you meant a separate array, a change through `b`, such as `b[0] = 99` or `b += 1`, silently changes `a` too, and the same happens to an array passed into a function. Write `b = a.copy()` when you need a copy.
</callout>

### Code Snippet: Functions Share the Caller's Array

```python
def add_one(values):
    values += 1          # changes the caller's array

data = np.array([1, 2, 3])
add_one(data)
print(data)              # [2 3 4]
```

To leave an input unchanged, return a new array (`return values + 1`) or work on `values.copy()`.

## Views and Copies

A slice is a **view**: a second window onto the same numbers, not a copy of them. Changing a value through the view changes the original array, just as the alias did. When you need to experiment without touching the source data, make an independent **copy** with `.copy()`.

### Reference Card: Views and Copies

- `arr[1:3]`, `arr[:, 0]`, `arr[:2, 1:]`: A slice is a view; it shares data with `arr`.
- `arr[1:3].copy()`: An independent copy; changes stay local.
- `arr[mask]`: Boolean indexing (below) always returns a new copy, even when every mask value is `True`.
- `arr[[0, 3]]`: Fancy indexing (below) always returns a new copy.
- `arr[mask] = value`: Assignment through a mask, or through `arr[[0, 3]]`, changes `arr` itself.
- `np.shares_memory(a, b)`: `True` when two arrays share data.

### Code Snippet: Change a View, Keep a Copy

```python
arr = np.array([10, 20, 30])
view = arr[:2]
independent = arr[:2].copy()
view[0] = 99
print(arr)          # [99 20 30]
print(independent)  # [10 20]
print(np.shares_memory(view, arr))         # True
print(np.shares_memory(independent, arr))  # False
```

Assign through a mask in one step: `arr[arr > 25][0] = 0` changes only the temporary copy that `arr[arr > 25]` returned, so `arr` stays the same.

## Boolean Indexing

**Boolean indexing** selects values by a condition instead of by position, and it is one of NumPy's most powerful tools: one line finds every reading above a threshold in an array of any shape. A comparison such as `bp >= 140` builds a **Boolean mask**, an array of `True` and `False` with the same shape as `bp`. `bp[mask]` keeps the values at the `True` positions, and `mask.sum()` counts them, because `True` counts as 1.

```text
bp                 bp >= 140 (the mask)
[[128 131 126]     [[False False False]
 [142 145 139]      [ True  True False]
 [118 121 119]]     [False False False]]

bp[bp >= 140]  →  [142 145]     the matching values, as a 1-D array
```

### Reference Card: Boolean Indexing

| Pattern | Purpose | Example |
| :--- | :--- | :--- |
| `arr > value` | Builds a Boolean mask; in this card `arr` is a 1-D array. | `arr > 5` |
| `arr[mask]` | Keeps matching elements; a 2-D array gives a 1-D result. | `arr[arr > 5]` |
| `arr2d[arr2d[:, 0] >= x]` | Keeps whole rows whose column 0 passes; the result stays 2-D. | `bp[bp[:, 0] >= 140]` |
| `values[labels == "x"]` | Keeps one group's values, where `labels` names the group at each position. | `systolic[clinics == "Cardiology"]` |
| `(a) & (b)` | Combines conditions with AND. | `(arr > 2) & (arr < 8)` |
| `(a) \| (b)` | Combines conditions with OR. | `(arr < 2) \| (arr > 8)` |
| `~mask` | Flips every `True` and `False`. | `arr[~(arr > 5)]` |
| `mask.sum()` | Counts `True` values. | `(arr > 5).sum()` |
| `mask.any()` / `mask.all()` | Is any / every value `True`? | `(arr > 5).any()` |
| `arr[mask] = value` | Replaces matching elements in place; changes `arr`. | `arr[arr > 5] = 0` |

### Code Snippet: Filter and Count with a Mask

```python
bp = np.array([[128, 131, 126],
               [142, 145, 139],
               [118, 121, 119]])
high = bp >= 140
print(bp[high])                       # [142 145]
print(high.sum())                     # 2
print(bp[(bp >= 120) & (bp < 130)])   # [128 126 121]
print(bp[bp[:, 0] >= 140])            # [[142 145 139]]: patients whose visit 1 was high
```

Use `&` and `|`, not `and` and `or`, and wrap each comparison in parentheses. `bp >= 120 and bp < 130` raises `ValueError: The truth value of an array with more than one element is ambiguous`; write `(bp >= 120) & (bp < 130)` instead.

## Fancy Indexing

**Fancy indexing** selects by a list of positions instead of a mask, in any order you choose. Like a mask, it returns a copy.

### Code Snippet: Pick Positions and Rows

```python
ids = np.array(["P001", "P002", "P003"])
print(ids[[2, 0]])   # ['P003' 'P001']
print(bp[[2, 0]])    # [[118 121 119]
                     #  [128 131 126]]: rows 2 and 0, in that order
```

## Array Reshaping

Reshaping rearranges the same values into a different grid without changing any of them: a flat run of 12 readings becomes 3 patients by 4 visits. The grid fills row by row, and `-1` in one position tells NumPy to work out that length from the others. `reshape` and `ravel` return a view when possible but may need to copy data; `flatten` always returns a copy.

![NumPy reshape: 12 values fill a 3×4 grid row by row, and ravel() turns the grid back into one row](media/numpy_reshape_panel.png)

### Reference Card: Reshape and Transpose

| Operation | Purpose | Result |
| :--- | :--- | :--- |
| `arr.reshape(rows, columns)` | Changes dimensions without changing values. | New shape `(rows, columns)` |
| `arr.reshape(-1, columns)` | `-1` lets NumPy compute that length from the array's size. | `np.arange(1, 13).reshape(-1, 4)` → shape `(3, 4)` |
| `arr.flatten()` / `arr.ravel()` | Back to 1D; `flatten` always copies, `ravel` returns a view when it can. | `[1 2 3 4 5 6]` for `arr` below |
| `arr.T` | Swaps rows and columns. | Transposed view when possible |

### Code Snippet: Reshape Versus Transpose

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.reshape(3, 2))  # [[1 2]
                          #  [3 4]
                          #  [5 6]]: same flat order
print(arr.T)              # [[1 4]
                          #  [2 5]
                          #  [3 6]]: rows become columns
```

![xkcd 2295: Garbage Math. Averaging independent garbage only gives better garbage; a summary is only as good as the readings behind it](media/xkcd_2295.png)

# Analyzing Arrays

NumPy's analysis tools work on a whole array or along one axis: they summarize values, transform each one, label them by rule, and rank them.

## Summary Statistics

A **reduction** collapses many values into one, such as a mean. Passing `axis` reduces along one dimension instead: `axis=0` collapses the rows and gives one result per column, and `axis=1` collapses the columns and gives one result per row. For a patients × visits table, `axis=1` gives each patient's mean and `axis=0` each visit's.

```text
[[1, 2, 3],  → axis=1 mean: 2.0 for this row
 [4, 5, 6]]  → axis=1 mean: 5.0 for this row
  ↓  ↓  ↓
 axis=0 means: [2.5, 3.5, 4.5], one per column
```

Most summaries work as a method (`arr.mean()`) or a function (`np.mean(arr)`), and both accept `axis`; `np.median()` and `np.percentile()` are functions only.

### Reference Card: Summaries by Axis

For `arr = np.array([[1, 2, 3], [4, 5, 6]])`:

| Task | Method or function | Purpose and key arguments | Typical output |
| --- | --- | --- | --- |
| Total | `arr.sum()` | Add values; `axis=0` gives one total per column | `21`; `axis=0` → `[5 7 9]` |
| Average | `arr.mean()` / `np.mean(arr, axis=1)` | Mean; `axis=1` gives one mean per row | `3.5`; `axis=1` → `[2. 5.]` |
| Middle value | `np.median(arr)` | Median; less affected by one extreme reading than the mean | `3.5`; `axis=0` → `[2.5 3.5 4.5]` |
| Spread | `arr.std()` | Population SD (divides by n); `ddof=1` divides by n − 1 for the sample SD, which pandas uses by default in Lecture 04 | `1.708`; `ddof=1` → `1.871` |
| Variance | `arr.var()` | The SD squared; also takes `ddof` | `2.917` |
| Extremes | `arr.min()` / `arr.max()` | Smallest / largest value | `1` / `6`; `axis=1` → `[1 4]` / `[3 6]` |
| Percentile | `np.percentile(arr, q)` | Value below which `q` percent of values fall; `q` may be a list | `np.percentile(arr, [25, 75])` → `[2.25 4.75]` |
| Running total | `arr.cumsum(axis=1)` | Cumulative sum along the axis | `[[ 1  3  6] [ 4  9 15]]` |
| Count matches | `(arr > 2).sum(axis=1)` | Count of `True` values per row | `[1 3]` |

### Code Snippet: Reduce a 2×3 Array

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.mean(axis=0))            # [2.5 3.5 4.5]: one mean per column
print(arr.mean(axis=1))            # [2. 5.]: one mean per row
print(arr.mean())                  # 3.5: one mean for the whole array
print(np.median(arr, axis=1))      # [2. 5.]
print(arr.max(axis=0))             # [4 5 6]: the largest value in each column
```

### Code Snippet: Select One Group by a Label

A mask can come from a text array too. When `clinics` names the clinic for each reading in `systolic`, comparing `clinics` with one name keeps that clinic's readings, and `.mean()` summarizes them.

```python
clinics = np.array(["Cardiology", "Primary Care", "Cardiology", "Nephrology", "Primary Care"])
systolic = np.array([142, 118, 151, 135, 128])   # one reading per visit, same order as clinics
print(clinics == "Cardiology")                   # [ True False  True False False]
print(systolic[clinics == "Cardiology"])         # [142 151]
print(systolic[clinics == "Cardiology"].mean())  # 146.5
for clinic in sorted(set(clinics)):              # each clinic once, in order
    print(clinic, systolic[clinics == clinic].mean())
# Cardiology 146.5
# Nephrology 135.0
# Primary Care 123.0
```

The mask and the values must line up position by position: position 2 of `clinics` has to describe the same visit as position 2 of `systolic`. Printing the whole list with `print(sorted(set(clinics)))` shows each name as `np.str_('Cardiology')`, NumPy's text type; printing one name at a time, as the loop does, shows it plainly.

## Universal Functions (ufuncs)

A **universal function** (ufunc) applies one math operation to every element and returns a new array. Lecture 02's `math.sqrt(16)` accepts one number and raises `TypeError` when given an array; `np.sqrt(arr)` takes the square root of every element.

### Reference Card: ufuncs

| Function | Purpose | Example |
| :--- | :--- | :--- |
| `np.sqrt(arr)` | Square root element-wise. | `np.sqrt(arr)` |
| `np.exp(arr)` | Exponential element-wise. | `np.exp([1, 2, 3])` |
| `np.maximum(a, b)` | Larger value at each position. | `np.maximum(arr1, arr2)` |

### Code Snippet: Apply Mathematical Functions

```python
arr = np.array([1, 4, 9, 16, 25])
print(np.sqrt(arr))                      # [1. 2. 3. 4. 5.]
print(np.exp([1, 2, 3]))                 # [ 2.71828183  7.3890561  20.08553692]
print(np.maximum([1, 5, 3], [4, 2, 6]))  # [4 5 6]
```

## Conditional Logic

`np.where(condition, value_if_true, value_if_false)` is the array version of `if`/`else`: it checks every position and picks one of two values. For more than two labels, nest `np.where` calls or give `np.select` a list of conditions; at each position it uses the first condition that is true.

### Reference Card: Labels by Rule

- `np.where(cond, a, b)`: `a` where `cond` is `True`, `b` elsewhere; `a` and `b` may be arrays, so `np.where(cond, arr, 0)` keeps values that pass.
- `np.where(c1, a, np.where(c2, b, c))`: Three labels; test the highest band first.
- `np.select([c1, c2, c3], [a, b, c], default=d)`: Any number of labels; the first true condition wins, and `default` fills positions where none is true.

### Code Snippet: Label Readings by Rules

```python
systolic = np.array([118, 142, 127, 135, 151, 109])
print(np.where(systolic >= 140, "high", "ok"))   # ['ok' 'high' 'ok' 'ok' 'high' 'ok']
print(np.where(systolic >= 140, systolic, 0))    # [  0 142   0   0 151   0]

category = np.select(
    [systolic >= 140, systolic >= 130, systolic >= 120],   # highest band first
    ["stage 2", "stage 1", "elevated"],
    default="normal",
)
print(category)   # ['normal' 'stage 2' 'elevated' 'stage 1' 'stage 2' 'normal']

nested = np.where(systolic >= 140, "stage 2", np.where(systolic >= 130, "stage 1", "below 130"))
print(nested)     # ['below 130' 'stage 2' 'below 130' 'stage 1' 'stage 2' 'below 130']
```

A reading of 142 passes both `>= 140` and `>= 130`; listing the highest band first gives it `stage 2`.

## Sorting and Ranking

`np.sort()` returns the _values_ in order. `np.argsort()` returns the _positions_ that would put them in order, which tells you _which_ patient has the highest value; indexing an array of patient IDs kept in the same order turns a position into an ID. A slice step of `-1` walks backward, so `[::-1]` reverses an order.

On a 2-D array both sort along one axis, the last by default, so each row is sorted on its own. That breaks the link between a value and its column. To reorder whole rows, `argsort` one column and index the rows with the result, as fancy indexing did.

### Reference Card: Values Versus Positions

| Operation | Result | Changes the original? |
| --- | --- | --- |
| `np.sort(arr)` | Sorted copy | No |
| `arr.sort()` | `None`; sorts `arr` in place | Yes |
| `np.argsort(arr)` | Positions that would sort the array | No |
| `arr.argmin()` / `arr.argmax()` | Position of the smallest / largest value in a 1-D array; `ids[arr.argmax()]` looks up the matching ID | No |
| `np.sort(arr2d, axis=1)` | Each row sorted on its own (the default for 2-D) | No |
| `np.sort(arr2d, axis=0)` | Each column sorted on its own | No |
| `np.argsort(arr2d, axis=1)` | Sorting positions within each row | No |
| `arr2d[np.argsort(arr2d[:, 0])]` | Whole rows, ordered by column 0 | No |

### Code Snippet: Find the Highest Values and Who Has Them

```python
ids = np.array(["P001", "P002", "P003", "P004", "P005"])
avg_glucose = np.array([112, 98, 145, 101, 130])  # one average per patient, same order as ids
print(np.sort(avg_glucose))    # [ 98 101 112 130 145]
order = np.argsort(avg_glucose)
print(order)                   # [1 3 0 4 2]: position 1 lowest, position 2 highest
top_two = order[-2:][::-1]     # last two positions, largest first
print(top_two)                 # [2 4]
print(avg_glucose[top_two])    # [145 130]
print(avg_glucose.argmax())    # 2
print(ids[avg_glucose.argmax()])  # P003
```

### Code Snippet: Sort Along an Axis and Order Rows

```python
ids = np.array(["P001", "P002", "P003"])   # one ID per row of bp
print(np.sort(bp, axis=1))       # [[126 128 131]
                                 #  [139 142 145]
                                 #  [118 119 121]]: each patient's readings in order
print(np.sort(bp, axis=0)[-1])   # [142 145 139]: the highest reading at each visit
order = np.argsort(bp[:, 0])[::-1]   # patients by visit 1, highest first
print(ids[order])                # ['P002' 'P001' 'P003']
print(bp[order])                 # [[142 145 139]
                                 #  [128 131 126]
                                 #  [118 121 119]]: each row stays intact
```

![Learning to code, day 1: a husky at the keyboard, which is how everyone starts](media/learning_to_code.png)

# LIVE DEMO!
