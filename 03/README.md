---
notion:
  title_line: "# NumPy Arrays & Virtual Environments"
  role: lecture
  status: mapped
  page_id: "27ed9fdd-1a1a-80a7-8532-e70d7000dae8"
  url: "https://app.notion.com/p/27ed9fdd1a1a80a78532e70d7000dae8"
---

# NumPy Arrays & Virtual Environments

[Live Demo Guide](demo/DEMO_GUIDE.md)

# Virtual Environments

![xkcd 1987: Python Environment](media/xkcd_1987.png)

*Virtual environments prevent package chaos*

## Why Virtual Environments?

Different projects may need incompatible versions of the same package. A virtual environment keeps each project's packages separate.

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

The `-c` option runs the short Python string that follows it.

### Package, module, and dependency

A **module** is a Python file that can be imported. A **package** is installable software that can provide one or more modules. NumPy is a package; code normally loads its top-level module with `import numpy`.

A **dependency** is software a project needs: **direct** dependencies are chosen by the project; **transitive** dependencies are required by those packages.

A **requirements file** lists packages to install. For this project, record the direct dependency in `requirements.txt`:

```text
numpy==2.3.3
```

`==` pins an exact version. A **lock file** also records the resolved transitive dependencies; it can be generated when the project needs that complete record.

### Environment and activation

An **environment** is the interpreter plus the packages available to it. A **virtual environment** is an isolated directory containing a project-specific Python command and package installation location.

This course uses `.venv` as the environment directory. Add it to `.gitignore`:

```gitignore
.venv/
```

The environment is recreated from instructions and requirements; it is not synchronized through Git.

**Activation** changes the current shell so `python` and installed commands resolve to the selected environment. Activation does not install a package and does not change Python source files.

## Using uv

[uv documentation](https://docs.astral.sh/uv/)

### Reference Card: uv environment workflow

| Task | Command | Result |
| :--- | :--- | :--- |
| Pin Python | `uv python pin 3.13` | Records the course interpreter version. |
| Create environment | `uv venv --python 3.13 .venv` | Creates the project environment. |
| Install requirements | `uv pip install -r requirements.txt` | Installs the deliberate direct dependencies. |
| Verify | `python --version` and `python -c "import numpy as np; print(np.__version__)"` | Confirms Python and NumPy versions. |
| Leave environment | `deactivate` | Returns to the previous shell environment. |

### Code Snippet: Create and Verify an Environment

```bash
uv --version
uv python pin 3.13
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
python -c "import numpy as np; print(np.__version__)"
deactivate
```

In native Windows PowerShell, replace the Bash activation line with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Recreate an Environment

Keep `requirements.txt` and `.python-version` in Git, not `.venv/`. Recreate the environment in a new directory from those records to check that someone else can reproduce it.

## Using standard-library venv (alternative)

With Python 3.13 installed, `venv` creates an environment with pip included:

### Reference Card: standard-library `venv`

| Task | Command | Note |
| :--- | :--- | :--- |
| Create | `python -m venv .venv` | Use the already-installed Python 3.13 interpreter. |
| Activate | `source .venv/bin/activate` | In PowerShell, use `.\.venv\Scripts\Activate.ps1`. |
| Install | `python -m pip install -r requirements.txt` | Uses the active environment's pip. |
| Leave | `deactivate` | Returns to the previous shell environment. |

In native Windows PowerShell, replace the Bash activation line with:

```powershell
.\.venv\Scripts\Activate.ps1
```

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

A **pipeline** connects small commands so each performs one step:

Pipes (`|`) send one command's output to the next. This example uses a simple CSV with no commas inside its fields.

### Reference Card: Pipeline Building Blocks

- `tail -n +2 FILE`: Skip the first line (the header).
- `cut -d',' -f4`: Select the fourth comma-separated field.
- `sort`: Put equal lines next to one another.
- `uniq -c`: Count adjacent equal lines.
- `head -n 5`: Keep the first five lines.
- `wc -l`: Count lines.
- `command > FILE` / `command >> FILE`: Replace / append file contents.

```text
CSV → skip header → select subject → sort → count
                                         1 English
                                         3 Math
                                         2 Science
```

### Code Snippet: Inspect a Small CSV

```bash
tail -n +2 data/raw/students.csv | cut -d',' -f4 | sort | uniq -c | head -n 5
tail -n +2 data/raw/students.csv | wc -l
```

## Variables and Timestamps

- `name=value`: Store text; no spaces around `=`.
- `"$name"` / `"${name}"`: Use its value; quotes preserve spaces.
- `$(command)`: Capture a command's output.

### Code Snippet: Name and Log a Run

```bash
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "Run: $timestamp" > "results/summary_${timestamp}.txt"
echo "${timestamp} complete" >> logs/processing.log
```

The timestamp looks like `20260917_093000`: year, month, day, hour, minute, second. One saved value labels both the result and the log.

# LIVE DEMO!

# Python Tools for Collections

## Object introspection

Introspection means asking an object what it is or what it contains while a program runs. `type()` shows the exact type; `isinstance()` checks whether a value matches a type and is usually the safer test in a program.

### Reference Card: object introspection

- `type(value)`: Show the value's exact type.
- `isinstance(value, type)`: Test whether a value has the requested type.
- `dir(value)`: List available attributes and methods.
- `help(value)` / `help(str.split)`: Read documentation; press **Q** to leave a paged view.
- `id(value)`: Get an identity number for an object during its lifetime.

### Code Snippet: Inspect values before using them

```python
value = "42"
print(type(value))             # <class 'str'>
if isinstance(value, str):
    value = int(value)
print(type(value), value)      # <class 'int'> 42
```

## Sequence functions

These built-in functions make common loops easier to read. `enumerate()` adds positions, `zip()` pairs items, and `reversed()` walks a sequence from the end. `sorted()` returns a new ordered list; the original is unchanged.

### Reference Card: sequence functions

- `enumerate(items, start=0)`: Yield position-value pairs.
- `zip(left, right)`: Yield pairs until the shorter input ends.
- `reversed(items)`: Iterate from the last item to the first.
- `sorted(items)`: Return a new sorted list.

### Code Snippet: Keep related values together

```python
names = ["Alice", "Bob", "Charlie"]
grades = [85, 92, 78]
for number, name in enumerate(names, start=1):
    print(f"{number}: {name}")  # 1: Alice, then 2: Bob, 3: Charlie
for name, grade in zip(names, grades):
    print(f"{name}: {grade}")   # Alice: 85, then Bob: 92, Charlie: 78
print(list(reversed(names)))   # ['Charlie', 'Bob', 'Alice']
```

## List Comprehensions

List comprehensions build a new list by applying an expression to each item, optionally keeping only items that pass a condition. They are concise Python, but still run one Python-level operation at a time.

```python
squares = [x * x for x in range(5)]
even_squares = [x * x for x in range(5) if x % 2 == 0]
```

# Why NumPy Matters

![It's pronounced...](media/numpy.webp)

An **array** stores values of one data type in a regular grid. **Vectorized operations** apply an operation to the whole array instead of writing a Python loop.

```text
Python list: [1, 2, 3] * 2  → [1, 2, 3, 1, 2, 3]
NumPy array: [1, 2, 3] * 2  → [2, 4, 6]
```

Python is famously slow for numerical computing:

```python
# Pure Python approach (SLOW)
my_list = list(range(1_000_000))
result = [x * 2 for x in my_list]

# NumPy approach (FAST)
import numpy as np
my_array = np.arange(1_000_000)
result = my_array * 2
```

NumPy can be much faster for large numerical arrays; the speedup depends on the operation, array size, and machine.

## The NumPy Solution

- **ndarray**: Fast, memory-efficient multidimensional arrays
- **Vectorized operations**: Apply functions to entire arrays at once
- **Broadcasting**: Smart handling of different-sized arrays
- **Universal functions (ufuncs)**: Fast element-wise operations

# NumPy Arrays

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

# From Python lists
arr = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Array creation functions
zeros = np.zeros(5)              # array([0., 0., 0., 0., 0.])
ones = np.ones((2, 3))           # 2x3 array of ones
range_arr = np.arange(10)        # array([0, 1, 2, ..., 9])
full = np.full((2, 3), 7)        # 2x3 array filled with 7
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

### Reference Card: data types

| Operation | Purpose | Example |
| :--- | :--- | :--- |
| `np.array(values, dtype=...)` | Chooses the initial element type. | `dtype=np.int32` |
| `arr.astype(dtype)` | Returns a converted array. | `arr.astype(np.float64)` |
| `astype(float)` | Converts numeric strings to numbers. | `str_arr.astype(float)` |

### Code Snippet: Convert Numeric Text

```python
str_arr = np.array(["1.25", "-9.6", "42"])
print(str_arr.astype(float))  # [ 1.25 -9.6  42.  ]
```

## Arithmetic and Vectorized Operations

NumPy's vectorized operations calculate element-wise across arrays without explicit loops.

### Reference Card: vectorized arithmetic

| Operation | Meaning | Example |
| :--- | :--- | :--- |
| `a + b` | Element-wise addition. | `arr1 + arr2` |
| `a * b` | Element-wise multiplication. | `arr1 * arr2` |
| `a ** n` | Element-wise power. | `arr1 ** 2` |
| `a * scalar` | Broadcasts one value across the array. | `arr1 * 2` |

### Code Snippet: Calculate without an explicit loop

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([5, 4, 3, 2, 1])

# Element-wise operations
sum_arr = arr1 + arr2       # array([6, 6, 6, 6, 6])
mult_arr = arr1 * arr2      # array([5, 8, 9, 8, 5])
power_arr = arr1 ** 2       # array([1, 4, 9, 16, 25])

# Scalar operations
doubled = arr1 * 2          # array([2, 4, 6, 8, 10])
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr + 10           # Adds 10 to all elements
```

# LIVE DEMO!

# Array Indexing and Slicing

## Basic Indexing

NumPy extends familiar Python indexing and slicing across multiple dimensions.

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

# Single element
first = arr[0]          # 0
last = arr[-1]          # 9

# Slicing
subset = arr[2:7]       # array([2, 3, 4, 5, 6])
every_other = arr[::2]  # array([0, 2, 4, 6, 8])
```

## Multidimensional Indexing

Comma-separated indices access elements, rows, or columns in multidimensional arrays.

### Reference Card: Rows, Columns, and Blocks

| Selection | Meaning for a 2D array | Result shape |
| --- | --- | --- |
| `arr[row, col]` | One element | Scalar |
| `arr[row, :]` | All columns in one row | 1D row |
| `arr[:, col]` | All rows in one column | 1D column |
| `arr[:2, 1:3]` | First two rows, columns 1 and 2 | 2D block |

## Boolean Indexing

Boolean indexing filters arrays with conditional logic, selecting elements that meet specific criteria.

### Reference Card: boolean indexing

| Pattern | Purpose | Example |
| :--- | :--- | :--- |
| `arr > value` | Builds a Boolean mask. | `arr > 5` |
| `arr[mask]` | Keeps matching elements. | `arr[arr > 5]` |
| `(a) & (b)` | Combines conditions with AND. | `(arr > 2) & (arr < 8)` |
| `(a) \| (b)` | Combines conditions with OR. | `(arr < 2) \| (arr > 8)` |

### Code Snippet: Filter with a Boolean mask

```python
arr = np.array([1, 5, 3, 8, 2, 9, 4])

# Boolean mask
mask = arr > 5              # array([False, False, False, True, False, True, False])
high_values = arr[mask]     # array([8, 9])

# Conditional operations
arr[arr > 5] = 0            # Set values > 5 to 0

# Multiple conditions (use & for AND, | for OR)
mask = (arr > 2) & (arr < 8)
filtered = arr[mask]
```

## Fancy Indexing

Fancy indexing uses integer arrays to select multiple elements at arbitrary positions without explicit loops.

```python
arr = np.array([10, 20, 30, 40])
arr[[0, 3]]  # array([10, 40]); a copy
```

## Names, aliases, and mutability

Assignment binds a name to an object; it does not copy the object. Lists, dictionaries, sets, and arrays are **mutable**: their contents can change. Two names can be **aliases** for the same object:

```text
values ───────┐
              ├──> [10, 20, 30]   one array
same_values ──┘
copied_values ───> [10, 20, 30]   a separate array
```

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

Use `==` to compare values and `is` to compare object identity. A list's `copy()` copies the outer list only; nested mutable objects remain shared.

## Views vs Copies

Slicing creates views that share memory; explicit copies are independent.

```python
arr = np.array([10, 20, 30])
view = arr[:2]
independent = arr[:2].copy()
view[0] = 99
print(arr)          # [99 20 30]
print(independent)  # [10 20]
```

# NumPy Operations

## Statistical Operations

NumPy provides built-in statistics across entire arrays or specific axes.

```text
[[1, 2, 3],  → axis=1 mean: 2.0 for this row
 [4, 5, 6]]  → axis=1 mean: 5.0 for this row
  ↓  ↓  ↓
 axis=0 means: [2.5, 3.5, 4.5], one per column
```

`arr.mean()` and `np.mean(arr)` are equivalent forms; NumPy also provides `np.sum(arr)` and `np.reshape(arr, shape)`.

### Reference Card: statistics

| Method | Meaning | Axis example |
| :--- | :--- | :--- |
| `.mean()` | Average of all values. | `.mean(axis=1)` per row |
| `.std()` | Standard deviation. | `.std()` overall |
| `.max()` / `.min()` | Largest or smallest value. | `.max()` overall |
| `axis=0` / `axis=1` | Reduce down rows / across columns in 2D. | One result per column / per row |

### Code Snippet: Reduce a 2×3 array

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.mean(axis=0)  # [2.5, 3.5, 4.5]: one mean per column
arr.mean(axis=1)  # [2., 5.]: one mean per row
arr.mean()        # 3.5: one mean for the whole array
```

## Array Reshaping

Reshaping operations change an array's dimensions. `reshape` returns a view when possible but may need to copy data; `flatten` always returns a copy.

![NumPy array reference](media/nparray_cheatsheet.png)

### Reference Card: reshape and transpose

| Operation | Purpose | Result |
| :--- | :--- | :--- |
| `arr.reshape(rows, columns)` | Changes dimensions without changing values. | New shape `(rows, columns)` |
| `arr.flatten()` | Makes a 1D copy. | Independent 1D array |
| `arr.T` | Swaps rows and columns. | Transposed view when possible |

### Code Snippet: Reshape versus transpose

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.reshape(3, 2)  # [[1, 2], [3, 4], [5, 6]]: same flat order
arr.T             # [[1, 4], [2, 5], [3, 6]]: rows become columns
```

# Semi-Advanced NumPy

The following short reference keeps a few useful NumPy operations visible; the more specialized material is collected in [the bonus page](BONUS.md).

## Universal Functions (ufuncs)

### Reference Card: ufuncs

| Function | Purpose | Example |
| :--- | :--- | :--- |
| `np.sqrt(arr)` | Square root element-wise. | `np.sqrt(arr)` |
| `np.exp(arr)` | Exponential element-wise. | `np.exp([1, 2, 3])` |
| `np.maximum(a, b)` | Larger value at each position. | `np.maximum(arr1, arr2)` |

### Code Snippet: Apply mathematical functions

```python
arr = np.array([1, 4, 9, 16, 25])

# Common mathematical functions
sqrt_arr = np.sqrt(arr)         # array([1., 2., 3., 4., 5.])
exp_arr = np.exp([1, 2, 3])     # array([2.718, 7.389, 20.086])

# Binary functions
arr1 = np.array([1, 5, 3])
arr2 = np.array([4, 2, 6])
max_arr = np.maximum(arr1, arr2) # array([4, 5, 6])
```

## Conditional Logic

### Reference Card: conditional logic

| Function | Purpose | Example |
| :--- | :--- | :--- |
| `np.where(condition, x, y)` | Select `x` where true, `y` elsewhere. | `np.where(arr < 0, 0, arr)` replaces negatives with zero. |

## Boolean Array Methods

Use `any()` to ask whether at least one value meets a condition; `all()` asks whether every value does.

### Reference Card: Boolean Reductions

- `(arr > 0).any()`: Is at least one value positive?
- `(arr > 0).all()`: Is every value positive?
- `(arr > 0).sum()`: How many values are positive? `True` counts as 1.

## Sorting

### Reference Card: Values Versus Positions

| Operation | Result | Changes the original? |
| --- | --- | --- |
| `np.sort(arr)` | Sorted copy | No |
| `arr.sort()` | `None` | Yes |
| `np.sort(arr, axis=0)` | Each column sorted independently in 2D | No |
| `arr.argmin()` / `arr.argmax()` | Index of the smallest / largest value in a 1D array | No |
| `np.argsort(arr)` | Indices that would sort the array | No |

## Random Number Generation

### Reference Card: random number generation

| Method | Purpose | Example |
| :--- | :--- | :--- |
| `np.random.default_rng()` | Creates a modern random generator. | `rng = ...` |
| `default_rng(seed)` | Makes the sequence reproducible. | `seed=42` |
| `rng.random(size)` | Uniform floats in `[0, 1)`. | `rng.random(5)` |
| `rng.integers(low, high, size)` | Integers in `[low, high)`. | `rng.integers(1, 10, size=5)` |
| `rng.standard_normal(size)` | Standard normal draws. | `rng.standard_normal(5)` |

### Code Snippet: Generate reproducible random data

```python
# Create a modern random generator
rng = np.random.default_rng()  # No seed (different each time)
rng_seeded = np.random.default_rng(seed=42)  # Reproducible

# Generate random numbers
random_nums = rng.random(5)              # 5 random floats [0, 1)
random_ints = rng.integers(1, 10, size=5) # 5 random ints [1, 10)
normal_nums = rng.standard_normal(5)     # 5 from normal distribution

# With seed for reproducibility
rng = np.random.default_rng(seed=123)
data = rng.random((3, 3))  # Same result every time
```

Use `default_rng()` for new code. It keeps the generator's state in one object instead of changing NumPy's legacy global random state.

![Learning to Code...](media/learning_to_code.png)

# LIVE DEMO!
