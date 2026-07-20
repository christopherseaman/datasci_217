# Reproducible Environments, Terminal Pipelines, and NumPy Arrays

## Learning objectives

By the end of this lecture, you should be able to:

1. Distinguish an interpreter, package, direct dependency, transitive dependency, environment, activation, and requirements file, then create, verify, and recreate the candidate Python 3.12.13 environment from deliberate direct requirements.
2. Use `head`, `tail`, `cut`, `sort`, `uniq`, `wc`, a pipe, and overwrite redirection to preview, select, count, and save a result from a small supplied text file whose fields contain no commas.
3. Create homogeneous 1D and 2D NumPy arrays, inspect `shape`, `ndim`, `size`, and `dtype`, select by position or basic slice, and demonstrate a slice view versus an explicit copy.
4. Create one boolean mask, perform vectorized same-shape and scalar arithmetic, and calculate whole-array and axis reductions while predicting each output shape.
5. Reshape and transpose a compatible array, predict one scalar-to-1D broadcast, and run an import-safe NumPy analysis as a terminal `.py` program.

## Starting point and execution boundary

Lecture 02 established the required GUI Git workflow, functions, minimal dictionaries, small text input/output, local modules, imports, and the main guard. Before continuing, make sure you can:

- locate a project with `pwd` and `ls`;
- run `python main.py` from the intended working directory;
- define a function that returns a value;
- import a local module without triggering its driver workflow; and
- inspect, stage, commit, and synchronize through VS Code Source Control or GitHub Desktop.

This is the final terminal-and-script lecture. All required Python work runs in `.py` files from the POSIX-style shell established in Lecture 01: Bash on Linux, WSL, or the supported cloud environment, and default zsh on macOS. Command-line Git is not required.

Lecture 04 introduces notebooks and Google Colab. Do not create or run a notebook in this lecture.

## Candidate runtime, not the final release lock

The currently tested candidate for this lecture is:

- Python 3.12.13;
- NumPy 2.0.2; and
- a project environment named `.venv`.

These exact versions make the current examples reproducible. They are not yet the final course lock: the complete course stack, graders, and Lecture 04–11 notebook environments still need release testing.

# Reproducibility vocabulary

A result is **reproducible** when another person can reconstruct the needed software environment and rerun the documented program with the same supplied inputs.

## Interpreter

The Python **interpreter** is the executable program that reads and runs Python code. Two terminals can resolve the command `python` to different interpreter files, so both version and location matter.

Check the version:

```bash
python --version
```

After activation below, Python can report the exact interpreter path without a platform-specific shell command:

```bash
python -c "import sys; print(sys.executable)"
```

The `-c` option runs the short Python string that follows it. Lecture 02 already used this pattern to check import safety.

## Package, module, and dependency

A **module** is a Python file that can be imported. A **package** is installable software that can provide one or more modules. NumPy is a package; code normally loads its top-level module with `import numpy`.

A **dependency** is software a project needs in order to run.

- A **direct dependency** is deliberately chosen by this project and imported or invoked by its code. NumPy is the only direct dependency in this lecture.
- A **transitive dependency** is needed by a direct dependency rather than chosen directly by this project.

A **requirements file** is a plain-text list of the direct packages a project deliberately needs. Record those dependencies in `requirements.txt`; do not generate that file from every package currently installed in an environment.

A **lock artifact** records exact resolved direct and transitive versions for a tested release. When the course needs one, it is generated and reviewed separately from the deliberate direct-dependency list.

For the candidate environment, create `requirements.txt` in VS Code with exactly:

```text
numpy==2.0.2
```

`==` pins the direct dependency to one exact candidate version. It can be changed later only as an intentional, tested course update.

## Environment and activation

An **environment** is the interpreter plus the packages available to it. A **virtual environment** is an isolated directory containing a project-specific Python command and package installation location.

This course uses `.venv` as the environment directory. Add it to `.gitignore`:

```gitignore
.venv/
```

The environment is recreated from instructions and requirements; it is not synchronized through Git.

**Activation** changes the current shell so `python` and installed commands resolve to the selected environment. Activation does not install a package and does not change Python source files.

# One primary uv workflow

**uv** is the course's primary Python and package-management tool. Install it before this lecture using one reviewed method from the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/). The lecture does not use a downloaded-script pipeline.

Confirm that uv is available:

```bash
uv --version
```

From the project directory containing `requirements.txt`, install and pin the exact candidate interpreter. A **pin** records the selected Python request in `.python-version`:

```bash
uv python install 3.12.13
uv python pin 3.12.13
```

Astral documents exact version requests and project pins in its [Python version guide](https://docs.astral.sh/uv/concepts/python-versions/).

Create `.venv` with that interpreter and activate it:

```bash
uv venv --python 3.12.13 .venv
source .venv/bin/activate
```

The activation command is the supported form for Bash and zsh. Once the environment is active, use `python` consistently:

```bash
python --version
python -c "import sys; print(sys.executable)"
```

The first command should report `Python 3.12.13`; the path from the second should point inside the project's `.venv`.

Install the deliberate direct requirements into the active environment:

```bash
uv pip install -r requirements.txt
python -c "import numpy as np; print(np.__version__)"
```

The version check should print `2.0.2`. The install command follows Astral's official [package installation](https://docs.astral.sh/uv/pip/packages/) and [environment](https://docs.astral.sh/uv/pip/environments/) guidance.

Do not create the direct-dependency file from a complete listing of everything installed. Such a listing does not distinguish choices made by this project from transitive packages.

Leave the active environment when the project work is complete:

```bash
deactivate
```

## Recreate instead of assuming

An import from the first environment proves only that the first environment works. Recreate the dependency set in a separate disposable directory to test the recorded instructions:

```bash
mkdir recreation-check
cp requirements.txt recreation-check/requirements.txt
cd recreation-check
uv venv --python 3.12.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
python -c "import numpy as np; print(np.__version__)"
deactivate
cd ..
```

The recreated environment should independently report Python 3.12.13 and NumPy 2.0.2.

## Standard-library fallback

Use this concise fallback only when uv is unavailable and the candidate Python interpreter is already installed. Confirm that `python` reports 3.12.13 before creating the environment:

```bash
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import numpy as np; print(np.__version__)"
deactivate
```

The outcome is the same: an activated `.venv` created from the deliberate direct-dependency file. The course does not require a second environment manager.

# LIVE DEMO!

**Reproduce one environment:** identify the initial interpreter, create and activate `.venv` with uv and Python 3.12.13, install `numpy==2.0.2` from `requirements.txt`, verify both versions, and reproduce those checks in a separate clean directory.

# A bounded terminal data pipeline

The shell can connect small commands when the supplied data is deliberately simple.

**Standard output** is the normal text a command writes to the terminal. **Standard input** is text a command receives. A **pipe**, written `|`, sends one command's standard output to the next command's standard input. A **pipeline** is the connected sequence of commands.

**Overwrite redirection**, written `>`, sends standard output to a file and replaces that file's previous contents. Always confirm the destination path before running it.

A **fixture** is a small supplied file with fixed, known contents. This fixture uses the comma-separated values (**CSV**) format and one comma as a **delimiter**, a character that separates fields. A **field** is one value between delimiters. The first line is the **header**, which names the fields:

```text
site,score,status
north,18,complete
south,21,complete
north,24,review
west,19,complete
south,22,review
north,20,complete
```

These commands are valid for this fixture because no field contains a comma, quoted newline, or other CSV complication. `cut` is not a general CSV parser.

## Preview with `head` and `tail`

`head` shows the beginning of a file; `tail` shows the end:

```bash
head -n 3 observations.csv
tail -n 2 observations.csv
```

`tail -n +2` starts at line 2, so it excludes this fixture's one-line header:

```bash
tail -n +2 observations.csv
```

## Select one field with `cut`

`cut -d','` declares the comma delimiter. `-f1` selects the first field:

```bash
cut -d',' -f1 observations.csv
```

Remove the header before selecting the site values:

```bash
tail -n +2 observations.csv | cut -d',' -f1
```

## Order and count with `sort`, `uniq`, and `wc`

`sort` places equal lines next to one another. `uniq -c` then counts adjacent equal lines:

```bash
tail -n +2 observations.csv | cut -d',' -f1 | sort | uniq -c
```

Save that result by overwriting a named output file:

```bash
tail -n +2 observations.csv | cut -d',' -f1 | sort | uniq -c > site_counts.txt
```

Inspect the saved result before using it:

```bash
cat site_counts.txt
wc -l site_counts.txt
```

`wc -l` reports the number of output lines. For this fixture there are three distinct site lines: `north`, `south`, and `west`.

This is the entire required shell-processing surface: preview, select, order, count, and save. More complex parsing belongs in a language or library that understands the data format.

# NumPy's array model

NumPy provides the `ndarray`, or **N-dimensional array**, for regular collections of values. An ndarray is **homogeneous**: its elements share one NumPy data type.

Load NumPy with its conventional short module name:

```python
import numpy as np
```

An **element** is one value in an array. A **dimension** is one direction needed to locate elements. A one-dimensional array needs one positional index; a two-dimensional array uses row and column positions.

An **axis** is a numbered dimension. Axis `0` is the first dimension and axis `1` is the second dimension.

## Create 1D and 2D arrays

`np.array()` converts supplied Python lists to ndarrays:

```python
import numpy as np

scores = np.array([18, 21, 24, 19], dtype=np.float64)
score_table = np.array(
    [
        [18, 21, 24],
        [20, 22, 26],
    ],
    dtype=np.float64,
)

print(scores)
print(score_table)
```

The nested list becomes a two-dimensional array. `dtype=np.float64` makes the shared element type explicit.

## Shape, dimensions, size, and dtype

An array's **shape** is its length along each dimension. Python displays that shape as a **tuple**, an ordered fixed sequence in parentheses.

- `shape` is the shape tuple;
- `ndim` is the number of dimensions;
- `size` is the total number of elements; and
- `dtype` is the shared NumPy element type.

```python
print(scores.shape)
print(scores.ndim)
print(scores.size)
print(scores.dtype)

print(score_table.shape)
print(score_table.ndim)
print(score_table.size)
print(score_table.dtype)
```

Expected output:

```text
(4,)
1
4
float64
(2, 3)
2
6
float64
```

The comma in `(4,)` distinguishes a one-element shape tuple from ordinary parentheses around a number.

# Positional indexing and basic slicing

NumPy positional indices are zero-based, as list indices were in Lecture 01.

```python
first_score = scores[0]
second_row = score_table[1]
third_column_value = score_table[0, 2]

print(first_score)
print(second_row)
print(third_column_value)
```

Expected output:

```text
18.0
[20. 22. 26.]
24.0
```

A **slice** selects a regular range. In `start:stop`, the start is included and the stop is excluded. A colon by itself keeps all positions on that axis:

```python
middle_scores = scores[1:3]
first_row = score_table[0, :]
second_column = score_table[:, 1]

print(middle_scores)
print(first_row)
print(second_column)
```

Expected output:

```text
[21. 24.]
[18. 21. 24.]
[21. 22.]
```

## Basic slice views and explicit copies

A **view** is an array that looks at the same underlying data as another array. A **copy** has independent data.

Basic NumPy slices are views. Changing the view can therefore change the source:

```python
source = np.array([10, 20, 30, 40])
middle_view = source[1:3]
middle_view[0] = 99

print(source)
```

Expected output:

```text
[10 99 30 40]
```

Use `.copy()` when the selected data must change independently:

```python
source = np.array([10, 20, 30, 40])
middle_copy = source[1:3].copy()
middle_copy[0] = 99

print(source)
print(middle_copy)
```

Expected output:

```text
[10 20 30 40]
[99 30]
```

This rule is deliberately specific: basic slicing creates a view, but not every kind of NumPy indexing does. Optional advanced indexing is covered in the bonus material. See NumPy's official [indexing](https://numpy.org/doc/2.0/user/basics.indexing.html) and [copy/view](https://numpy.org/doc/2.0/user/basics.copies.html) explanations.

# Masks and vectorized arithmetic

A **boolean mask** is an array of `True` and `False` values used to select elements at corresponding positions:

```python
scores = np.array([18, 21, 24, 19])
review_mask = scores >= 20
review_scores = scores[review_mask]

print(review_mask)
print(review_scores)
```

Expected output:

```text
[False  True  True False]
[21 24]
```

A **vectorized operation** applies one array expression element by element. Same-shape arrays align by position:

```python
baseline = np.array([18, 21, 24])
follow_up = np.array([20, 20, 27])
change = follow_up - baseline

print(change)
```

Expected output:

```text
[ 2 -1  3]
```

# Reductions and axis meaning

A **reduction** combines many array elements into fewer summary values. With no axis argument, a reduction uses the whole array. With an axis argument, it combines values along that numbered dimension.

For a table with shape `(2, 3)`, reducing `axis=0` removes the first dimension and leaves one result for each of the three columns. Reducing `axis=1` removes the second dimension and leaves one result for each of the two rows.

```python
measurements = np.array(
    [
        [10, 20, 30],
        [20, 30, 40],
    ],
    dtype=np.float64,
)

overall_mean = measurements.mean()
column_means = measurements.mean(axis=0)
row_means = measurements.mean(axis=1)

print(f"Overall mean: {overall_mean:.1f}")
print(column_means)
print(column_means.shape)
print(row_means)
print(row_means.shape)
```

Expected output:

```text
Overall mean: 25.0
[15. 25. 35.]
(3,)
[20. 30.]
(2,)
```

Predict which input dimension disappears before running an axis reduction. NumPy defines `mean(axis=...)` as computing along the selected axis; retaining reduced axes is a separate optional behavior.

# Compatible reshape and transpose

**Reshaping** changes the shape used to organize the same number of elements. The requested shape must have a compatible total size:

```python
values = np.array([1, 2, 3, 4, 5, 6])
grid = values.reshape(2, 3)

print(grid)
print(grid.shape)
```

Expected output:

```text
[[1 2 3]
 [4 5 6]]
(2, 3)
```

A **transpose** reverses the axes of this two-dimensional array. `.T` is the concise transpose attribute:

```python
transposed = grid.T

print(transposed)
print(transposed.shape)
```

Expected output:

```text
[[1 4]
 [2 5]
 [3 6]]
(3, 2)
```

Do not assume that every reshape shares data. NumPy returns a view when possible and a copy otherwise, as documented for [`reshape`](https://numpy.org/doc/2.0/reference/generated/numpy.reshape.html). When independent mutation matters, make that intention explicit with `.copy()`.

# One core broadcasting case: scalar to 1D

**Broadcasting** lets NumPy apply an operation to compatible shapes. The sole required case is a scalar combined with a one-dimensional array. Conceptually, the scalar is used at every array position:

```python
scores = np.array([18, 21, 24])
adjusted_scores = scores + 1

print(adjusted_scores)
print(adjusted_scores.shape)
```

Expected output:

```text
[19 22 25]
(3,)
```

The output keeps the array's `(3,)` shape. General multidimensional broadcasting rules are optional bonus material. NumPy documents scalar broadcasting as its [simplest broadcasting example](https://numpy.org/doc/2.0/user/basics.broadcasting.html).

# LIVE DEMO!

**Build the ndarray mental model:** progress from small 1D and 2D literal arrays through shape and dtype, positional selection, a visible slice-view mutation and explicit copy, one mask, same-shape arithmetic, whole/axis reductions, reshape, transpose, and one scalar-to-1D broadcast.

# One import-safe terminal analysis

Lecture 02's module boundary still applies when a third-party package is imported. Save this file as `array_summary.py`:

```python
import numpy as np


def summarize(measurements):
    """Return the overall mean, column means, and review count."""
    overall_mean = measurements.mean()
    column_means = measurements.mean(axis=0)
    review_values = measurements.reshape(measurements.size)
    review_mask = review_values >= 30
    review_count = int(review_mask.sum())
    return {
        "overall_mean": overall_mean,
        "column_means": column_means,
        "review_count": review_count,
    }


def main():
    """Run one deterministic array summary."""
    measurements = np.array(
        [
            [10, 20, 30],
            [20, 30, 40],
        ],
        dtype=np.float64,
    )
    summary = summarize(measurements)

    print(f'Overall mean: {summary["overall_mean"]:.1f}')
    print(f'Column means: {summary["column_means"]}')
    print(f'Values at or above 30: {summary["review_count"]}')


if __name__ == "__main__":
    main()
```

`review_mask.sum()` is a whole-array reduction. In that sum, each `True` contributes one and each `False` contributes zero, so the result is the number of selected elements.

With the candidate environment active, verify that import is quiet:

```bash
python -c "import array_summary"
```

Then run the program:

```bash
python array_summary.py
```

Expected output:

```text
Overall mean: 25.0
Column means: [15. 25. 35.]
Values at or above 30: 3
```

The main guard prevents the driver workflow from running during import. NumPy is a direct dependency because this module imports it.

# LIVE DEMO!

**Bounded pipeline plus terminal analysis:** preview and count the supplied delimiter-safe fixture with the required shell pipeline, save and inspect the count file, then use a supplied loader to return a homogeneous 2D ndarray to an import-safe analysis module. Reshape the array to 1D before applying the scalar mask, then calculate the documented axis reductions. Students do not implement CSV parsing.

# Handoff to Lecture 04

Each terminal script run in Lectures 01–03 starts a fresh Python process, executes the file from top to bottom, and ends. Lecture 04 will define a notebook, cell, kernel/runtime, persistent state, and execution order before asking you to use them. The key contrast is that a notebook's kernel can retain values between separate cell executions, while a new script process does not retain values from the previous run.

Lecture 04 will also define pandas objects and their labels. The conceptual bridge is:

- a 1D ndarray supplies the positional array model underneath a pandas **Series**, a labeled one-dimensional object; and
- a 2D ndarray supplies the positional array model underneath a pandas **DataFrame**, a labeled two-dimensional table.

Do not use pandas APIs yet. Carry forward the array's dtype, dimensions, shape, positional indexing, masks, and axis reasoning; Lecture 04 adds labels, notebook state, and portable tabular input/output.

# Key takeaways

- An environment is reproducible only when its interpreter and deliberate direct dependencies can be recreated from recorded instructions.
- Activation changes which `python` the shell resolves; verify the version and executable path before installing or running.
- A bounded shell pipeline can preview, select, order, count, and save simple delimiter-safe data, but it is not a general CSV parser.
- An ndarray has one homogeneous dtype and explicit dimensions, shape, size, and numbered axes.
- Basic slices are views; `.copy()` requests independent data.
- Masks, vectorized arithmetic, scalar broadcasting, reductions, reshape, and transpose all have predictable shape consequences.
- Lecture 04 changes the execution model and adds labels; it does not erase the environment, import-safety, or array reasoning established here.

# Optional bonus material

The single optional extension for this lecture is [Lecture 03 bonus: Additional NumPy patterns](BONUS.md). It is not assessed or assumed by Lecture 04.
