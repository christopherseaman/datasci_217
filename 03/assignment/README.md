# Assignment 03: Reproducible Terminal NumPy Analysis

This assignment is the final terminal-and-script assignment before Lecture 04 introduces Jupyter and Google Colab. It combines the exact candidate environment records, one bounded POSIX terminal pipeline, and ordinary homogeneous NumPy arrays.

Work in the `03/assignment` directory, or in the standalone assignment repository created from this subtree. Use the supported POSIX-style shell from Lecture 01 and terminal-executed `.py` files. Do not create a notebook.

The required GUI delivery sequence is in [`PLATFORM_CHECK.md`](PLATFORM_CHECK.md). It is an unassessed platform checklist, separate from the public code checks, because a grading checkout cannot prove that you activated or recreated an environment, operated a GUI, or followed a particular local workflow.

## Starter files

- `.python-version` and `requirements.txt`: Task 1 records; edit both.
- `.gitignore`: supplied recreation/environment exclusions; do not edit it.
- `environment_check.py`: supplied version probe; do not edit it.
- `PIPELINE.md`: Task 2 command block; edit only its four `TODO` lines.
- `observations.csv`: supplied deterministic seven-line fixture; do not edit it.
- `data_loader.py`: supplied CSV-to-ndarray boundary; do not edit it.
- `array_analysis.py`: Task 3 NumPy functions; edit this file.
- `analysis.py`: Task 3 terminal driver; edit this file.
- `output/.gitkeep`: keeps the output directory in the starter repository.
- `check_assignment.py` and `_public_checks.py`: supplied checker machinery; do not edit them.
- `test_assignment.py`: public managed-pytest facade; do not edit it. You do not need pytest locally.

The supplied loader and checker use later Python features internally. Run them, but do not copy their implementation patterns into the two student Python files.

## Task 1: Record, verify, and recreate the candidate environment

Replace the single `TODO` line in `.python-version` with exactly:

```text
3.12.13
```

Replace the single `TODO` line in `requirements.txt` with the only deliberate direct dependency:

```text
numpy==2.0.2
```

Do not add uv, pytest, or transitive packages to `requirements.txt`. The file records what this project deliberately imports, not everything installed in one environment.

From the assignment directory, use uv to install/pin the interpreter, create the named local environment, activate it, install the direct requirement, and verify the selected interpreter:

```bash
uv python install 3.12.13
uv python pin 3.12.13
uv venv --python 3.12.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
python -c "import sys; print(sys.executable)"
python environment_check.py > output/environment_check.txt
cat output/environment_check.txt
```

The exact saved probe is:

```text
Python: 3.12.13
NumPy: 2.0.2
```

There is one newline after each line. Leave the environment and recreate it from only the committed records and supplied probe:

```bash
deactivate
mkdir recreation-check
cp .python-version requirements.txt environment_check.py recreation-check/
cd recreation-check
uv venv --python 3.12.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python environment_check.py
deactivate
cd ..
```

Both `.venv/` directories and `recreation-check/` are generated local state. They are ignored and must never be submitted. The checker can validate the exact records, compare the saved probe with a fresh probe, and reject a tracked `.venv` when Git metadata is available. It cannot prove that you activated the local environment or performed the recreation steps; those are workflow evidence, not facts encoded in Python output.

## Task 2: Complete and run the bounded terminal pipeline

Replace the four `TODO` lines inside the fenced block in `PIPELINE.md` with these exact commands, in this order:

```bash
head -n 3 observations.csv > output/head_preview.txt
tail -n 2 observations.csv > output/tail_preview.txt
tail -n +2 observations.csv | cut -d',' -f1 | sort | uniq -c > output/site_counts.txt
wc -l output/site_counts.txt > output/site_count_lines.txt
```

Run the four commands from the assignment directory. They preview the fixed input, remove its one-line header, select the site field, put equal site names next to one another, count adjacent equal names, and save the number of count lines.

The fixture is deliberately bounded: it has one header, no quoted newlines, and no comma inside a field. `cut` is not a general CSV parser. `uniq -c` may pad counts differently on supported systems; after whitespace normalization, the saved count/name pairs must be:

```text
3 north
2 south
1 west
```

Use one overwrite redirection (`>`) per command, never append redirection (`>>`). The checker parses the block before executing it in temporary copies. It accepts only the stated `head`, `tail`, `cut`, `sort`, `uniq`, and `wc` commands, pipes, and one final overwrite redirect; it does not pass student text to a shell.

## Task 3: Implement the NumPy functions

Complete the seven functions in `array_analysis.py`. Keep every exact signature, replace each one-line TODO docstring, and return the named dictionaries or scalar. Do not print or mutate an input.

### `create_and_describe(values)`

Create `array = np.array(values, dtype=np.float64)` and return:

```python
{
    "array": array,
    "shape": array.shape,
    "ndim": array.ndim,
    "size": array.size,
    "dtype": array.dtype,
}
```

### `select_parts(values)`

For a 2D ndarray with at least two rows and two columns, return selections made directly from `values`:

```python
{
    "first_value": values[0, 0],
    "second_row": values[1],
    "second_column": values[:, 1],
    "top_left_block": values[:2, :2],
}
```

### `view_and_copy(values)`

Create `middle_view = values[1:3]` and `middle_copy = values[1:3].copy()`. Return both without mutating either result or `values` during the call:

```python
{"view": middle_view, "copy": middle_copy}
```

The returned view must share memory with the input. The returned copy must not share memory with the input or view.

### `vector_operations(values, baseline, threshold, offset)`

The two array inputs are 1D and have the same shape; `threshold` and `offset` are scalars. Create one boolean `mask = values >= threshold`, select `values[mask]`, calculate `difference = values - baseline`, and apply the scalar broadcast `adjusted = values + offset`. Return:

```python
{
    "mask": mask,
    "selected": selected,
    "difference": difference,
    "adjusted": adjusted,
}
```

### `reduction_summary(values)`

For a 2D ndarray, calculate `np.mean(values)`, `np.mean(values, axis=0)`, and `np.mean(values, axis=1)`. Return:

```python
{
    "overall_mean": overall_mean,
    "column_means": column_means,
    "column_means_shape": column_means.shape,
    "row_means": row_means,
    "row_means_shape": row_means.shape,
}
```

### `reshape_and_transpose(values, rows, columns)`

Create `grid = np.reshape(values, (rows, columns))`, then `transposed = grid.T`. Return:

```python
{
    "grid": grid,
    "grid_shape": grid.shape,
    "transposed": transposed,
    "transposed_shape": transposed.shape,
}
```

### `count_at_or_above(values, threshold)`

First reshape the input to one dimension with `flattened = np.reshape(values, values.size)`. Then create the scalar-comparison mask `flattened >= threshold` and return its whole count with `np.sum(...)`. Do not reduce one original axis separately.

## Task 3 driver

Complete `analysis.py` while keeping its supplied imports and exact main guard. Inside `main()`, in this order:

1. call `load_measurements("observations.csv")` once;
2. pass that returned array directly to `create_and_describe()` once;
3. pass the same array directly to `reduction_summary()` once;
4. pass the same array and scalar `30` directly to `count_at_or_above()` once; and
5. use those returned results to print the exact eight-line summary below.

Do not ignore the returned data, recalculate it in the driver, or hard-code the displayed values. The supplied loader resolves the fixture relative to its own file, so this program works when invoked by absolute path from another working directory.

Running `python analysis.py` must print exactly:

```text
Measurements shape: (6, 2)
Measurements dtype: float64
Overall mean: 25.0
Column means: [20. 30.]
Column means shape: (2,)
Row means: [15. 25. 35. 15. 25. 35.]
Row means shape: (6,)
Values at or above 30: 6
```

Importing `array_analysis`, `data_loader`, or `analysis` must be quiet and must not create, remove, or change any file. The driver writes no report; its output goes only to stdout.

## Check your work

With the candidate environment active, run:

```bash
python analysis.py
python check_assignment.py
```

A complete submission ends with `All public checks passed.` The optional GitHub Actions workflow may run the ten public facade tests with managed pytest. Instructor or TA grading may run the same written contract from a trusted checkout; the workflow is feedback, not a submission requirement.

The checker fresh-executes the probe, pipeline, imports, functions, and driver in temporary copies. Stored output alone is never treated as proof that current code works.

## Explicit scope boundaries

These are assignment boundaries, not hidden style rules. They apply to `array_analysis.py` and `analysis.py`; the supplied loader and grader machinery may use standard-library facilities needed for their jobs.

- Keep the supplied module docstring and exact top-level layout in both student files. `array_analysis.py` contains only its NumPy import and the seven required functions in the supplied order; `analysis.py` contains only its two supplied imports, `main()`, and the exact main guard. Do not add nested or extra functions or top-level driver state.
- No `for`/`while` loops, comprehensions, generator expressions or functions, exceptions, classes, lambdas, decorators, async code, annotations, default parameters, `*args`, `**kwargs`, `global`, or `nonlocal`.
- No `pathlib`, file I/O, CSV parsing, or other parsing in student code. The supplied `data_loader.py` owns the path and CSV boundary.
- `array_analysis.py` has only `import numpy as np`. `analysis.py` has only the supplied exact local imports.
- No indirect or dynamic calls, aliases for calls, attribute lookup used to choose a call, `eval`, `exec`, `compile`, or `__import__`.
- Do not use augmented assignment, assign through an index or attribute, delete data, or otherwise mutate any input or returned selection during a function call.
- Use only the required `np.array`, `.copy()`, `np.mean`, `np.reshape`, `np.sum`, `.T`, basic indexing/slicing, one boolean selection, comparisons, same-shape subtraction, and scalar addition. Do not replace required NumPy calls with methods or hand calculations.
- No structured arrays, random generation, fancy integer indexing, sorting/ranking, stacking, concatenation, multidimensional broadcasting, pandas, notebook, Colab, or shell script.
- The pipeline may not use command substitution, variables, append redirection, semicolons, compound commands, extra tools, CLI Git, `awk`, `sed`, or `tr`.
