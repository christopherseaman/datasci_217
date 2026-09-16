---
notion:
  title_line: "# NumPy Arrays & Virtual Environments"
  role: lecture
  status: mapped
  page_id: "27ed9fdd-1a1a-80a7-8532-e70d7000dae8"
  url: "https://app.notion.com/p/27ed9fdd1a1a80a78532e70d7000dae8"
---

# NumPy Arrays & Virtual Environments

Lecture 03 makes execution reproducible: create a known environment, then use it for NumPy-based numerical work. Lectures 01–03 use scripts and the terminal; notebooks begin in Lecture 04.

**LIVE DEMO!**

[Live Demo Guide](demo/DEMO_GUIDE.md)

# Virtual Environments

![xkcd 1987: Python Environment](media/xkcd_1987.png)

*Virtual environments prevent package chaos*

## Why Virtual Environments?

**The Problem:** Different projects need different package versions.

- Project A needs pandas 1.3.0
- Project B needs pandas 2.0.0
- Installing one breaks the other!

**The Solution:** Each project gets its own Python environment.

## Lecture 03 candidate environment

This lecture uses one tested course candidate:

- CPython 3.13
- a project environment stored in `.venv`
- NumPy 2.3.3 as the core lecture examples' only direct Python dependency
- a deliberate `requirements.txt` containing only `numpy==2.3.3`

These versions define the tested candidate for this lecture's NumPy work, not a permanent dependency set for every later lecture. Later activities may add packages through their own recorded requirements. The primary setup below uses uv. The standard-library `venv` and Conda sections are alternatives for comparison; here they reproduce the same version, directory, and dependency contract rather than defining separate learning outcomes.

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

A **dependency** is software a project needs in order to run.

- A **direct dependency** is deliberately chosen by this project and imported by its Python code. NumPy is the only direct Python dependency used by the core lecture examples. The optional command-line supplement later in the lecture names separate tools that are not part of this candidate environment.
- A **transitive dependency** is needed by a direct dependency rather than chosen directly by this project.

A **requirements file** is a plain-text list of the direct packages a project deliberately needs. Record those dependencies in `requirements.txt`; do not generate that file from every package currently installed in an environment.

A **lock artifact** records exact resolved direct and transitive versions for a tested release. When the course needs one, it is generated and reviewed separately from the deliberate direct-dependency list.

For the candidate environment, create `requirements.txt` in VS Code with exactly:

```text
numpy==2.3.3
```

`==` pins the direct dependency to one exact candidate version. It can be changed later only as an intentional, tested course update.

### Environment and activation

An **environment** is the interpreter plus the packages available to it. A **virtual environment** is an isolated directory containing a project-specific Python command and package installation location.

This course uses `.venv` as the environment directory. Add it to `.gitignore`:

```gitignore
.venv/
```

The environment is recreated from instructions and requirements; it is not synchronized through Git.

**Activation** changes the current shell so `python` and installed commands resolve to the selected environment. Activation does not install a package and does not change Python source files.

## Using uv (course candidate)

[uv documentation](https://docs.astral.sh/uv/)

### Reference Card: uv environment workflow

| Task | Command | Result |
| :--- | :--- | :--- |
| Pin Python | `uv python pin 3.13` | Records the course interpreter version. |
| Create environment | `uv venv --python 3.13 .venv` | Creates the project environment. |
| Install requirements | `uv pip install -r requirements.txt` | Installs the deliberate direct dependencies. |
| Verify | `python --version` and `python -c "import numpy as np; print(np.__version__)"` | Confirms Python and NumPy versions. |
| Leave environment | `deactivate` | Returns to the previous shell environment. |

### Code Snippet: Create and verify the candidate environment

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

## Recreate instead of assuming

An import from the first environment proves only that the first environment works. Recreate the dependency set in a separate disposable directory to test the recorded instructions:

The recreated environment should independently report Python 3.13 and NumPy 2.3.3.

## Using standard-library venv (alternative)

Use this concise fallback only when uv is unavailable and the candidate Python interpreter is already installed. Confirm that `python` reports 3.13 before creating the environment:

### Reference Card: standard-library `venv`

| Task | Command | Note |
| :--- | :--- | :--- |
| Create | `python -m venv .venv` | Use the already-installed Python 3.13 interpreter. |
| Activate | `source .venv/bin/activate` | In PowerShell, use `\.venv\Scripts\Activate.ps1`. |
| Install | `python -m pip install -r requirements.txt` | Uses the active environment's pip. |
| Leave | `deactivate` | Returns to the previous shell environment. |

In native Windows PowerShell, replace the Bash activation line with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The outcome is the same: an activated `.venv` created from the deliberate direct-dependency file. Choose one setup route; do not nest one environment inside another.

## Using Conda (alternative comparison)

[Conda documentation](https://docs.conda.io/)

Conda is not the course's primary setup route. Use it only if the configured channels provide the exact candidate Python version; otherwise use uv. This example keeps the same `.venv` location and installs the same deliberate requirements file.

### Reference Card: Conda alternative

| Task | Command | Result |
| :--- | :--- | :--- |
| Create | `conda create --prefix ./.venv python=3.13 pip` | Creates the same `.venv` location with Conda. |
| Activate | `conda activate ./.venv` | Selects the Conda environment. |
| Activate (PowerShell) | `conda activate .\.venv` | Windows alternative. |
| Install | `python -m pip install -r requirements.txt` | Installs the deliberate requirements. |
| Leave | `conda deactivate` | Returns to the previous environment. |

# Shell pipelines and small automation

Before the Python and NumPy examples, use the terminal to inspect a small CSV. Run `demo/01_cli_pipeline_demo.sh` from a disposable directory; it creates only paths below the current working directory.

Pipes (`|`) send one command's output to the next command. Redirection (`>` and `>>`) writes output to a file; `>>` appends. The core bounded pipeline skips a header with `tail`, selects a field with `cut`, sorts it for `uniq`, counts records with `wc`, and limits displayed output with `head`:

### Code Snippet: Inspect a small CSV

```bash
tail -n +2 data/raw/students.csv | cut -d',' -f4 | sort | uniq -c | head -n 5
tail -n +2 data/raw/students.csv | wc -l
```

Shell variables use `name=value` with no spaces. Capture one timestamp with `timestamp=$(date +"%Y%m%d_%H%M%S")`, then reuse `$timestamp` in output names and log lines so one run has one identifier. Append a concise status message with `echo "${timestamp} complete" >> logs/processing.log`.

# Brief Python refresher

Lectures 01–02 introduced type checking and f-string formatting. Keep those core-Python tools available while reading the NumPy examples below; no NumPy objects are needed for this refresher.

```python
name = "Alice"
grade = 87.5
print(f"Student {name} earned {grade:.1f}%")
```

**LIVE DEMO!**

![It's pronounced...](media/numpy.webp)

# Why NumPy Matters

List comprehensions build a new list by applying an expression to each item, optionally keeping only items that pass a condition. They are concise Python, but still run one Python-level operation at a time.

```python
squares = [x * x for x in range(5)]
even_squares = [x * x for x in range(5) if x % 2 == 0]
```

Python is famously slow for numerical computing:

```python
# Pure Python approach (SLOW)
my_list = list(range(1_000_000))
result = [x * 2 for x in my_list]  # 46.4 ms

# NumPy approach (FAST)
import numpy as np
my_array = np.arange(1_000_000)
result = my_array * 2  # 0.3 ms - 150x faster!
```

**NumPy is 10-100x faster** than pure Python for numerical operations.

## The NumPy Solution

- **ndarray**: Fast, memory-efficient multidimensional arrays
- **Vectorized operations**: Apply functions to entire arrays at once
- **Broadcasting**: Smart handling of different-sized arrays
- **Universal functions (ufuncs)**: Fast element-wise operations

# NumPy Arrays

## NumPy Quick Reference

![NumPy Cheatsheet](media/nparray_cheatsheet.png)

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
| `arr.shape` | Length along each dimension. | `(2, 3)` |
| `arr.ndim` | Number of dimensions. | `2` |
| `arr.size` | Total number of elements. | `6` |
| `arr.dtype` | Element data type. | `int64` |

## Data Types

### Reference Card: data types

| Operation | Purpose | Example |
| :--- | :--- | :--- |
| `np.array(values, dtype=...)` | Chooses the initial element type. | `dtype=np.int32` |
| `arr.astype(dtype)` | Returns a converted array. | `arr.astype(np.float64)` |
| `astype(float)` | Converts numeric strings to numbers. | `str_arr.astype(float)` |

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

## Statistical Operations

NumPy provides built-in statistics across entire arrays or specific axes.

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

### Reference Card: reshape and transpose

| Operation | Purpose | Result |
| :--- | :--- | :--- |
| `arr.reshape(rows, columns)` | Changes dimensions without changing values. | New shape `(rows, columns)` |
| `arr.flatten()` | Makes a 1D copy. | Independent 1D array |
| `arr.T` | Swaps rows and columns. | Transposed view when possible |

### Code Snippet: Reshape versus transpose

```python
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

| Expression | Result |
| --- | --- |
| `(arr > 0).any()` | One Boolean: is any value positive? |
| `(arr > 0).all()` | One Boolean: are all values positive? |
| `(arr > 0).sum()` | Number of positive values |

## Sorting

| Operation | Result | Changes the original? |
| --- | --- | --- |
| `np.sort(arr)` | Sorted copy | No |
| `arr.sort()` | `None` | Yes |
| `np.sort(arr, axis=0)` | Each column sorted independently in 2D | No |

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
# Create random generator
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

![Learning to Code...](media/learning_to_code.png)

# LIVE DEMO!
