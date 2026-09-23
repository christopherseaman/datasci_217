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

## Why Virtual Environments?

Unlike `math`, NumPy does not ship with Python: it is a third-party **package**, which must be installed in the active environment before `import` works. A **virtual environment** gives each project its own set of installed packages.

Without one, every project shares one Python installation. Last year's readmission report ran with an older NumPy; this term's wearable-sensor project needs NumPy 2.3.3. Upgrading for the new project changes the old one, and the published numbers may no longer rerun.

```text
Project A → A/.venv → its Python and package versions
Project B → B/.venv → its Python and package versions
```

## Course Environment

- Python 3.13
- Project environment: `.venv`
- NumPy 2.3.3, recorded as `numpy==2.3.3` in `requirements.txt`

Use uv for the course workflow; standard-library `venv` and Conda are alternatives below.

## Reproducibility vocabulary

A result is **reproducible** when another person can reconstruct the needed software environment and rerun the documented program with the same supplied inputs.

### Interpreter

The Python **interpreter** is the executable program that reads and runs Python code. Two terminals can resolve the command `python` to different interpreter files, so both version and location matter.

Check the version:

```bash
python --version
```

After activation below, Python can report the exact interpreter path without a platform-specific shell command:

```bash
python -c "import sys; print(sys.executable)"
```

`-c` runs the Python string that follows it.

### Package, module, and dependency

A package is installable software that can provide one or more modules, the importable Python files from Lecture 02. NumPy is a package; code normally loads its top-level module with `import numpy`.

A **dependency** is software a project needs: **direct** dependencies are chosen by the project; **transitive** dependencies are required by those packages.

A **requirements file** lists packages to install. For this project, record the direct dependency in `requirements.txt`:

```text
numpy==2.3.3
```

`==` pins an exact version. Write this file by hand so it lists only the direct dependencies you chose; `uv pip freeze` instead records everything currently installed, including packages that arrived as dependencies of what you asked for, which is a different record. A **lock file** also records the resolved transitive dependencies; it can be generated when the project needs that complete record.

Lecture 04 installs this same file from inside a notebook with `%pip install -r requirements.txt`.

### Environment and activation

An **environment** is the interpreter plus the packages available to it. A virtual environment is an isolated directory containing a project-specific Python command and package installation location.

This course uses `.venv` as the environment directory. Add it to `.gitignore`:

```gitignore
.venv/
```

The environment is recreated from instructions and requirements; it is not synchronized through Git.

**Activation** changes the current shell so `python` and installed commands resolve to the selected environment. Activation does not install a package and does not change Python source files.

```text
$ source .venv/bin/activate
(.venv) $ python -c "import sys; print(sys.executable)"
/home/alice/assignment-03/.venv/bin/python
(.venv) $ deactivate
$
```

The `(.venv)` prefix shows the environment is active, and `python` now runs the copy inside the project's `.venv` folder.

## Using uv

[uv documentation](https://docs.astral.sh/uv/)

### Reference Card: uv environment workflow

| Task | Command | Result |
| :--- | :--- | :--- |
| Pin Python | `uv python pin 3.13` | Writes `.python-version` containing `3.13`; later `uv venv` commands in this folder use it. |
| Create environment | `uv venv --python 3.13 .venv` | Creates the project environment. |
| Activate | `source .venv/bin/activate` (PowerShell: `.\.venv\Scripts\Activate.ps1`) | The prompt shows `(.venv)`; `python` now runs the environment's interpreter. |
| Install requirements | `uv pip install -r requirements.txt` | Installs the deliberate direct dependencies. |
| Verify | `python --version` and `python -c "import numpy as np; print(np.__version__)"` | Confirms Python and NumPy versions. |
| Leave environment | `deactivate` | Returns to the previous shell environment. |

### Code Snippet: Create and Verify an Environment

```bash
uv --version
uv python pin 3.13                                      # Pinned `.python-version` to `3.13`
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python --version                                        # Python 3.13.14
python -c "import numpy as np; print(np.__version__)"   # 2.3.3
deactivate
```

`uv python pin 3.13` writes that printed version into a `.python-version` file, so later `uv venv` commands in the folder use it without `--python`. For larger projects, `uv init` starts a project with a `pyproject.toml` that records `requires-python` instead of a pinned version file plus `requirements.txt`.

In native Windows PowerShell, replace the Bash activation line with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### When `import numpy` Fails

```text
$ python -c "import numpy as np"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

`ModuleNotFoundError` means the Python that ran the code is not the project's environment. Activation applies only to the terminal where you ran it, so look for the `(.venv)` prefix and run `source .venv/bin/activate` again if it is missing. VS Code's **Run** and **Debug** buttons use the interpreter chosen with **Python: Select Interpreter** (Lecture 02); for this project, choose the one inside `.venv` (`./.venv/bin/python`).

## Recreate an Environment

Keep `requirements.txt` and `.python-version` in Git, not `.venv/`. Recreate the environment in a new directory from those records to check that someone else can reproduce it.

### Code Snippet: Recreate from the Records

```bash
mkdir recreation-check
cp .python-version requirements.txt recreation-check/
cd recreation-check
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import numpy as np; print(np.__version__)"   # 2.3.3
deactivate
cd ..
```

`uv venv` reads the copied `.python-version`, so it creates a Python 3.13 environment without `--python 3.13`.

## Using standard-library venv (alternative)

With Python 3.13 installed, `venv` creates an environment with pip included:

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
| Activate | `conda activate ./.venv` | Selects the Conda environment. |
| Activate (PowerShell) | `conda activate .\.venv` | Windows alternative. |
| Install | `python -m pip install -r requirements.txt` | Installs the deliberate requirements. |
| Leave | `conda deactivate` | Returns to the previous environment. |

# Shell pipelines and small automation

## Pipelines

A **pipe** (`|`) sends a command's output into another command, where `>` (Lecture 01) sends it into a file. A **pipeline** chains small commands, each doing one job on the previous command's output. Before writing any Python, a pipeline can answer quick questions about a file, such as how many participants each study site enrolled.

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

# LIVE DEMO!

# Python Tools for Collections

Before switching to NumPy, three groups of built-in Python tools make everyday code shorter and safer: introspection functions that check what a value is, sequence functions that walk lists together, and list comprehensions that build a new list in one line. Several extend Lecture 01's `type()` and `enumerate()` and Lecture 02's lists and `sorted()`.

## Object introspection

Values read from a file start as text, so checking a value before calculating avoids a `TypeError`. **Introspection** means asking an object what it is or what it can do while the program runs. `type()`, from Lecture 01, shows the exact type; `isinstance()` answers yes or no and is the usual test inside an `if`; `dir()` lists what the object can do; and `help()` shows its documentation.

### Reference Card: object introspection

- `type(value)`: Show the value's exact type.
- `isinstance(value, type)`: Test whether a value has the requested type.
- `dir(value)`: List available attributes and methods.
- `help(value)` / `help(str.split)`: Read documentation; press **Q** to leave a paged view.
- `id(value)`: Get an identity number for an object during its lifetime.

### Code Snippet: Inspect values before using them

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

## Sequence functions

Beside `enumerate()` and `sorted()`, two more built-ins handle common loop needs: `zip()` walks two lists side by side, keeping each patient's ID with that patient's reading, and `reversed()` walks a sequence from the end.

### Reference Card: sequence functions

- `enumerate(items, start=0)`: Yield position-value pairs.
- `zip(left, right)`: Yield pairs until the shorter input ends.
- `reversed(items)`: Iterate from the last item to the first.
- `sorted(items)`: Return a new sorted list.

### Code Snippet: Keep related values together

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

A **list comprehension** writes a `for` loop that builds a list with `append()` in one line: `[expression for item in items if condition]`. Read it as "make this, for each item, keeping only items that pass."

```text
loop                                 comprehension
fevers = []                          fevers = [t for t in temps_f if t >= 100.4]
for t in temps_f:
    if t >= 100.4:
        fevers.append(t)
```

### Reference Card: List Comprehensions

- `[expr for x in items]`: Transform every item; returns a new list of the same length.
- `[x for x in items if condition]`: Keep only items that pass; returns a new, possibly shorter list.
- `[expr for x in items if condition]`: Filter, then transform.

### Code Snippet: Filter and Convert a List

```python
temps_f = [98.6, 101.2, 99.5, 103.1]
fevers = [t for t in temps_f if t >= 100.4]
print(fevers)   # [101.2, 103.1]

doses_mg = [250, 500, 125]
doses_g = [mg / 1000 for mg in doses_mg]
print(doses_g)  # [0.25, 0.5, 0.125]
```

A comprehension is concise, but Python still handles one item at a time. NumPy, next, removes that loop.

# Why NumPy Matters

![It's pronounced...](media/numpy.webp)

A wearable heart monitor records one reading per second: 86,400 readings per patient per day. A list comprehension handles those readings one at a time in Python. **NumPy** (Numerical Python) is the package that stores and calculates on numbers in bulk; pandas, which starts in Lecture 04, is built on it.

NumPy's core object is the **array** (`ndarray`, for n-dimensional array): a grid of values that all share one data type. A Python list can hold anything in each slot, so Python must check every item as it goes. An array's values share one type and sit side by side in memory, so NumPy's built-in routines, written in the faster C language, can process them all in one pass. Writing one expression that applies to every element, instead of a loop, is called **vectorization**.

## Lists Versus Arrays

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

Both give the same values. On one test machine the list version took about 44 ms and the array version about 1.4 ms; Demo 2 times both on your computer.

# NumPy Arrays

`import numpy as np` is the standard way to load NumPy; `np` is an alias, as in Lecture 02's `import module as alias`. `np.array()` turns a list, or a list of lists, into an array. Each inner list becomes one row, so two patients' readings at three visits become a 2×3 array. Every array has a **shape**, the length of each dimension with rows first, and a **dtype**, the one data type shared by every element.

One 2×3 array has two rows and three columns:

```text
          column 0  column 1  column 2
row 0  →       1         2         3
row 1  →       4         5         6
```

## Creating Arrays

### Reference Card: array creation

| Function | Purpose | Example output |
| :--- | :--- | :--- |
| `np.array(values)` | Converts a list or nested lists to an array. | 1D or 2D `ndarray` |
| `np.zeros(shape)` | Fills an array with zeros. | `float` array |
| `np.ones(shape)` | Fills an array with ones. | `float` array |
| `np.arange(stop)` | Creates evenly spaced integer values. | `[0, 1, ..., stop - 1]` |
| `np.full(shape, value)` | Fills an array with one value. | Array matching `shape` |

### Code Snippet: Create arrays

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
zeros = np.zeros(5)         # array([0., 0., 0., 0., 0.])
ones = np.ones((2, 3))      # 2x3 array of ones
range_arr = np.arange(10)   # array([0, 1, 2, ..., 9])
full = np.full((2, 3), 7)   # 2x3 array filled with 7
```

## Array Properties

### Reference Card: array properties

| Attribute | Meaning | Example |
| :--- | :--- | :--- |
| `arr_2d.shape` | Length along each dimension. | `(2, 3)` |
| `arr_2d.ndim` | Number of dimensions. | `2` |
| `arr_2d.size` | Total number of elements. | `6` |
| `arr_2d.dtype` | Element data type. | `int64` on a 64-bit system |

## Data Types

Values read from a file arrive as text: `"98.6"` is a string until you convert it, just as Lecture 01 converted text with `float("25.5")`. `astype()` converts a whole array at once and returns a new array. If any value is not a number, such as `"NA"`, conversion stops with `ValueError`. Converting decimals to integers drops the decimal part without rounding.

### Reference Card: data types

| Operation | Purpose | Example |
| :--- | :--- | :--- |
| `np.array(values, dtype=...)` | Chooses the initial element type. | `dtype=np.int32` |
| `arr.astype(dtype)` | Returns a new converted array; also parses numeric text. | `arr.astype(np.float64)`, `str_arr.astype(float)` |

### Code Snippet: Convert Numeric Text

```python
temps = np.array(["98.6", "101.2", "99.5"])
print(temps.dtype)          # <U5: text of up to 5 characters
temps_f = temps.astype(float)
print(temps_f)              # [ 98.6 101.2  99.5]
print(temps_f.astype(int))  # [ 98 101  99]: decimals dropped, not rounded
```

## Arithmetic and Vectorized Operations

### Reference Card: vectorized arithmetic

| Operation | Meaning | Example |
| :--- | :--- | :--- |
| `a + b` | Element-wise addition. | `arr1 + arr2` |
| `a * b` | Element-wise multiplication. | `arr1 * arr2` |
| `a ** n` | Element-wise power. | `arr1 ** 2` |
| `a * scalar` | **Broadcasting**: NumPy stretches one value across every element. | `arr1 * 2` → `[ 2  4  6  8 10]` |

### Code Snippet: Calculate without an explicit loop

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([5, 4, 3, 2, 1])
print(arr1 + arr2)   # [6 6 6 6 6]
print(arr1 * arr2)   # [5 8 9 8 5]
print(arr1 ** 2)     # [ 1  4  9 16 25]
print(arr1 * 2)      # [ 2  4  6  8 10]: one number broadcast to every element
```

Two arrays combine position by position, so they must have the same shape: `np.array([1, 2, 3]) + np.array([1, 2])` raises `ValueError: operands could not be broadcast together with shapes (3,) (2,)`.

# Array Indexing and Slicing

Arrays use the same square brackets and half-open slices as lists, such as `items[0]` and `items[1:4]`, then add one index per dimension, separated by commas. That is how a table of patients by visits gets read one cell, one row, one column, or one block at a time.

## Basic Indexing

### Reference Card: one-dimensional indexing

| Pattern | Purpose | Example |
| :--- | :--- | :--- |
| `arr[i]` | Select one element. | `arr[0]` |
| `arr[-1]` | Select the last element. | `arr[-1]` |
| `arr[start:stop]` | Select a half-open slice. | `arr[2:7]` |
| `arr[::step]` | Select every `step`th element. | `arr[::2]` |

### Code Snippet: Slice a one-dimensional array

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr[0], arr[-1])   # 0 9
print(arr[2:7])          # [2 3 4 5 6]
print(arr[::2])          # [0 2 4 6 8]
```

## Multidimensional Indexing

A 2D array is a table: the first index picks rows and the second picks columns, separated by a comma. Axis 0 is the rows and axis 1 the columns. Leaving out the column index selects a whole row, so `bp[1]` is the same as `bp[1, :]`.

```text
bp              visit 1  visit 2  visit 3
patient 0  →      128      131      126
patient 1  →      142      145     [139]   ← bp[1, 2]
patient 2  →      118      121      119
bp[:, 0] is the visit 1 column: [128 142 118]
```

### Reference Card: Rows, Columns, and Blocks

| Selection | Meaning for a 2D array | Result shape |
| --- | --- | --- |
| `arr[row, col]` | One element | Scalar |
| `arr[row, :]` | All columns in one row | 1D row |
| `arr[row]` | Same as `arr[row, :]` | 1D row |
| `arr[:, col]` | All rows in one column | 1D column |
| `arr[:2, 1:3]` | First two rows, columns 1 and 2 | 2D block |

### Code Snippet: Select Cells, Rows, Columns, and Blocks

```python
bp = np.array([[128, 131, 126],   # patient 0: systolic at visits 1-3
               [142, 145, 139],   # patient 1
               [118, 121, 119]])  # patient 2
print(bp[1, 2])    # 139: patient 1, visit 3
print(bp[1])       # [142 145 139]: every visit for patient 1
print(bp[:, 0])    # [128 142 118]: visit 1 for every patient
print(bp[:2, 1:])  # [[131 126]
                   #  [145 139]]: patients 0-1, visits 2-3
```

# LIVE DEMO!

# Views, Copies, and Boolean Selection

Slicing a Python list makes a new list. NumPy is built for arrays of millions of values, so it avoids that copying wherever it can: a slice hands back a window onto the same numbers. The saved time is why arrays are fast, and it is also behind the most common NumPy surprise, where changing one array changes another. This topic covers when NumPy shares data, when it copies, and how to select values by a condition instead of by position.

## Names, aliases, and mutability

Assignment binds a name to an object; it does not copy the object. Lists, dictionaries, sets, and arrays are **mutable**: their contents can change. Two names can be **aliases** for the same object:

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
print(same_values is values)  # True

copied_values = values.copy()
copied_values[0] = 10
print(values)                 # [99 20 30]
print(copied_values)          # [10 20 30]
```

### Code Snippet: Functions Share the Caller's Array

Passing an array to a function does not copy it either: the parameter is another name for the caller's array.

```python
def add_one(values):
    values += 1          # changes the caller's array

data = np.array([1, 2, 3])
add_one(data)
print(data)              # [2 3 4]
```

To leave an input unchanged, return a new array (`return values + 1`) or work on `values.copy()`.

Use `==` to compare values and `is` to compare object identity. A list's `copy()` copies the outer list only; nested mutable objects remain shared.

## Views vs Copies

A slice is a **view**: a second window onto the same numbers, not a copy of them. Changing a value through the view changes the original array, just as `same_values` did above. When you need to experiment without touching the source data, make an independent **copy** with `.copy()`.

### Reference Card: Views and Copies

- `arr[1:3]`, `arr[:, 0]`, `arr[:2, 1:]`: View; shares data with `arr`.
- `arr[1:3].copy()`: Independent copy; changes stay local.
- `arr[mask]`, `arr[[0, 3]]`: Always a new copy (Boolean and fancy indexing, below).
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

## Boolean Indexing

Clinical questions are often filters: which readings are at or above 140 mmHg? Comparing an array with a value, as Lecture 01 compared single numbers, produces a **Boolean mask**: an array of `True` and `False` with the same shape. Putting the mask inside square brackets keeps only the `True` positions. Because `True` counts as 1, `mask.sum()` counts the matches.

```text
systolic            [128    142    118    151    135  ]
systolic >= 140     [False  True   False  True   False]   ← high, the mask
systolic[high]              [142           151]
```

### Reference Card: boolean indexing

| Pattern | Purpose | Example |
| :--- | :--- | :--- |
| `arr > value` | Builds a Boolean mask. | `arr > 5` |
| `arr[mask]` | Keeps matching elements. | `arr[arr > 5]` |
| `(a) & (b)` | Combines conditions with AND. | `(arr > 2) & (arr < 8)` |
| `(a) \| (b)` | Combines conditions with OR. | `(arr < 2) \| (arr > 8)` |
| `mask.sum()` | Count `True` values. | `high.sum()` → `2` |
| `mask.any()` / `mask.all()` | Is any / every value `True`? | `high.any()` → `True`; `high.all()` → `False` |
| `arr[mask] = value` | Replace matching elements in place; changes `arr`. | `arr[arr > 5] = 0` |

### Code Snippet: Filter and Count with a Mask

```python
systolic = np.array([128, 142, 118, 151, 135])
high = systolic >= 140
print(high)            # [False  True False  True False]
print(systolic[high])  # [142 151]
print(high.sum())      # 2
print(systolic[(systolic >= 120) & (systolic < 140)])  # [128 135]
```

Use `&` and `|`, not `and` and `or`, and wrap each comparison in parentheses. `systolic >= 120 and systolic < 140` raises `ValueError: The truth value of an array with more than one element is ambiguous`.

**Fancy indexing** selects by a list of positions rather than by a mask, and also returns a copy: `systolic[[0, 3]]` gives `[128 151]`.

# NumPy Operations

Indexing answers "which values?"; a **reduction** answers "what are they, taken together?" by collapsing many numbers into one. The same summaries work on a whole array or along one axis, which is how a table of readings gives a per-patient average and a per-visit average from the same data.

## Statistical Operations

```text
[[1, 2, 3],  → axis=1 mean: 2.0 for this row
 [4, 5, 6]]  → axis=1 mean: 5.0 for this row
  ↓  ↓  ↓
 axis=0 means: [2.5, 3.5, 4.5], one per column
```

Each summary works as a method (`arr.mean()`) or a function (`np.mean(arr)`); both accept `axis`.

### Reference Card: Summaries by Axis

For `arr = np.array([[1, 2, 3], [4, 5, 6]])`:

| Task | Method or function | Purpose and key arguments | Typical output |
| --- | --- | --- | --- |
| Total | `arr.sum()` / `np.sum(arr)` | Add values; `axis=0` gives one total per column | `21`; `axis=0` → `[5 7 9]` |
| Average | `arr.mean()` / `np.mean(arr, axis=1)` | Mean; `axis=1` gives one mean per row | `3.5`; `axis=1` → `[2. 5.]` |
| Spread | `arr.std()` | Population SD (divides by n); `ddof=1` divides by n − 1 for the sample SD, which pandas uses by default in Lecture 04 | `1.708`; `ddof=1` → `1.871` |
| Extremes | `arr.min()` / `arr.max()` | Smallest / largest value; accepts `axis` | `1` / `6` |

### Code Snippet: Reduce a 2×3 array

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.mean(axis=0))  # [2.5 3.5 4.5]: one mean per column
print(arr.mean(axis=1))  # [2. 5.]: one mean per row
print(arr.mean())        # 3.5: one mean for the whole array
```

## Array Reshaping

Reshaping rearranges the same values into a different grid without changing any of them: a flat run of 12 readings becomes 3 patients by 4 visits. `reshape` returns a view when possible but may need to copy data; `flatten` always returns a copy.

![NumPy reshaping cheatsheet: this lecture uses only the top-left reshape panel; the bonus page covers stacking](media/nparray_cheatsheet.png)

### Reference Card: reshape and transpose

| Operation | Purpose | Result |
| :--- | :--- | :--- |
| `arr.reshape(rows, columns)` | Changes dimensions without changing values. | New shape `(rows, columns)` |
| `arr.flatten()` | Makes a 1D copy. | Independent 1D array |
| `arr.T` | Swaps rows and columns. | Transposed view when possible |
| `np.reshape(arr, (rows, columns))` | Function form of `reshape`; `np.reshape(arr, arr.size)` makes a 1D array. | `[1 2 3 4 5 6]` for `arr.size` |

### Code Snippet: Reshape versus transpose

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.reshape(3, 2))  # [[1 2]
                          #  [3 4]
                          #  [5 6]]: same flat order
print(arr.T)              # [[1 4]
                          #  [2 5]
                          #  [3 6]]: rows become columns
```

# Array Tools for Analysis

Indexing picked out values and reductions summarized them. These tools answer the next analysis questions about a whole array at once: transform every value, label values by a rule, rank patients, and generate practice data. More specialized tools are in [the bonus page](BONUS.md).

## Universal Functions (ufuncs)

A **universal function** (ufunc) applies one math operation to every element and returns a new array. Lecture 02's `math.sqrt(16)` accepts one number and raises `TypeError` when given an array; `np.sqrt(arr)` takes the square root of every element.

### Reference Card: ufuncs

| Function | Purpose | Example |
| :--- | :--- | :--- |
| `np.sqrt(arr)` | Square root element-wise. | `np.sqrt(arr)` |
| `np.exp(arr)` | Exponential element-wise. | `np.exp([1, 2, 3])` |
| `np.maximum(a, b)` | Larger value at each position. | `np.maximum(arr1, arr2)` |

### Code Snippet: Apply mathematical functions

```python
arr = np.array([1, 4, 9, 16, 25])
print(np.sqrt(arr))                      # [1. 2. 3. 4. 5.]
print(np.exp([1, 2, 3]))                 # [ 2.71828183  7.3890561  20.08553692]
print(np.maximum([1, 5, 3], [4, 2, 6]))  # [4 5 6]
```

## Conditional Logic

`np.where(condition, value_if_true, value_if_false)` is the array version of `if`/`else`: it checks every position and picks one of two values.

### Code Snippet: Label Readings by a Rule

```python
systolic = np.array([128, 142, 118, 151, 135])
labels = np.where(systolic >= 140, "high", "ok")
print(labels)  # ['ok' 'high' 'ok' 'high' 'ok']

kept = np.where(systolic >= 140, systolic, 0)
print(kept)    # [  0 142   0 151   0]
```

The second call keeps each original value where the test passes and substitutes `0` elsewhere, instead of a text label.

## Sorting and Ranking

Sorting answers two questions. `np.sort()` returns the _values_ in order. `np.argsort()` returns the _positions_ that would put them in order, which tells you _which_ patient has the highest value. A slice step of `-1` walks backward, so `[::-1]` reverses an order.

### Reference Card: Values Versus Positions

| Operation | Result | Changes the original? |
| --- | --- | --- |
| `np.sort(arr)` | Sorted copy | No |
| `arr.sort()` | `None` | Yes |
| `np.sort(arr, axis=0)` | Each column sorted independently in 2D | No |
| `arr.argmin()` / `arr.argmax()` | Index of the smallest / largest value in a 1D array | No |
| `np.argsort(arr)` | Indices that would sort the array | No |

### Code Snippet: Find the Highest Values and Who Has Them

```python
avg_glucose = np.array([112, 98, 145, 101, 130])  # one average per patient
print(np.sort(avg_glucose))    # [ 98 101 112 130 145]
order = np.argsort(avg_glucose)
print(order)                   # [1 3 0 4 2]: patient 1 lowest, patient 2 highest
top_two = order[-2:][::-1]     # last two positions, largest first
print(top_two)                 # [2 4]
print(avg_glucose[top_two])    # [145 130]
print(avg_glucose.argmax())    # 2
```

## Random Number Generation

Simulated data lets you practice an analysis before touching patient records. A **random number generator** produces values that look random, and a **seed** makes it produce the same sequence every run, so your results match your classmates'. `integers(low, high)` never returns `high`, so use `101` to include 100. `size` can be a shape: `size=(100, 5)` makes 100 rows and 5 columns.

### Reference Card: random number generation

| Method | Purpose | Example |
| :--- | :--- | :--- |
| `np.random.default_rng()` | Creates a modern random generator. | `rng = ...` |
| `default_rng(seed)` | Makes the sequence reproducible. | `seed=42` |
| `rng.random(size)` | Uniform floats in `[0, 1)`. | `rng.random(5)` |
| `rng.integers(low, high, size)` | Integers in `[low, high)`. | `rng.integers(1, 10, size=5)` |
| `rng.integers(low, high, size=(rows, cols))` | 2D array of integers. | `rng.integers(70, 101, size=(100, 5))` |
| `rng.standard_normal(size)` | Standard normal draws. | `rng.standard_normal(5)` |

### Code Snippet: Generate Reproducible Practice Data

```python
rng = np.random.default_rng(seed=42)
ages = rng.integers(18, 91, size=5)          # 18 through 90
scores = rng.integers(70, 101, size=(2, 3))  # 2 rows, 3 columns, 70 through 100
print(ages)    # [24 74 65 50 49]
print(scores)  # [[96 72 91]
               #  [76 72 86]]
```

Run it again with `seed=42` and the same numbers print. Older tutorials call `np.random.seed()` and `np.random.randn()`; use `default_rng()` in new code.

![Learning to Code...](media/learning_to_code.png)

# LIVE DEMO!
