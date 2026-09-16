# Assignment 03: Reproducible Terminal NumPy Analysis

## Files

```text
assignment/
├── .python-version, requirements.txt  # environment records to complete
├── PIPELINE.md                        # command scaffold to complete
├── array_analysis.py, analysis.py     # Python scaffolds to complete
├── observations.csv                   # supplied input; keep unchanged
├── environment_check.py, data_loader.py # supplied helpers; keep unchanged
├── .gitignore, check_assignment.py     # supplied configuration/checker
└── output/                            # generate and commit six text artifacts
```

## Task 1: Record, verify, and recreate the environment

Work in `03/assignment` or its standalone repository with the Lecture 01 POSIX-style shell. In VS Code, sync `main`, finish outstanding changes, and use **Git: Create Branch** to create `feature/numpy-analysis`.

Open **Terminal → New Terminal** in VS Code at the assignment directory. If you use a native terminal or WSL Ubuntu instead, first `cd` into the assignment directory.

### 1.1 Record the environment

Replace the single `TODO` line in `.python-version` with exactly:

```text
3.13
```

Replace the single `TODO` line in `requirements.txt` with the only deliberate direct dependency:

```text
numpy==2.3.3
```

The file records the project's direct dependency.

> **Checkpoint — `.python-version`**
> Save `3.13` with a final newline.

> **Checkpoint — `requirements.txt`**
> Save `numpy==2.3.3` with a final newline.

### 1.2 Verify and recreate the environment

From the assignment directory, use uv to install/pin the interpreter, create the named local environment, activate it, install the direct requirement, and verify the selected interpreter:

```bash
uv python install 3.13
uv python pin 3.13
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
python -c "import sys; print(sys.executable)"
python environment_check.py > output/environment_check.txt
cat output/environment_check.txt
```

The exact saved probe is:

```text
Python: 3.13
NumPy: 2.3.3
```

> **Checkpoint — `output/environment_check.txt`**
> Save the two version lines above.

There is one newline after each line. Leave the environment and recreate it from only the committed records and supplied probe:

```bash
deactivate
mkdir recreation-check
cp .python-version requirements.txt environment_check.py recreation-check/
cd recreation-check
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python environment_check.py
deactivate
cd ..
```

Both `.venv/` directories and `recreation-check/` are generated local state. Keep them ignored and out of your submission.

## Task 2: Complete and run the bounded terminal pipeline

### 2.1 Complete the pipeline

Replace the four `TODO` lines inside the fenced block in `PIPELINE.md` with these exact commands, in this order:

```bash
head -n 3 observations.csv > output/head_preview.txt
tail -n 2 observations.csv > output/tail_preview.txt
tail -n +2 observations.csv | cut -d',' -f1 | sort | uniq -c > output/site_counts.txt
wc -l output/site_counts.txt > output/site_count_lines.txt
```

> **Checkpoint — `output/head_preview.txt`**
> Save the first three fixture lines.

> **Checkpoint — `output/tail_preview.txt`**
> Save the last two fixture lines.

> **Checkpoint — `output/site_counts.txt`**
> Save the three site counts shown below.

> **Checkpoint — `output/site_count_lines.txt`**
> Save the `wc` result: `3 output/site_counts.txt` (spacing may vary).

### 2.2 Run and inspect the results

Run the four commands from the assignment directory. They preview the fixed input, remove its one-line header, select the site field, put equal site names next to one another, count adjacent equal names, and save the number of count lines.

The fixture is deliberately bounded: it has one header, no quoted newlines, and no comma inside a field. `cut` is not a general CSV parser. `uniq -c` may pad counts differently on supported systems; after whitespace normalization, the saved count/name pairs must be:

```text
3 north
2 south
1 west
```

Commit the four saved results. Count-column padding may vary by platform; the checker normalizes whitespace in the site counts and `wc` result.

## Task 3: Implement the NumPy functions

Complete the seven functions in `array_analysis.py`. Document their behavior and return the dictionaries or scalar described below, leaving inputs unchanged.

### 3.1 create_and_describe(values)

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

### 3.2 select_parts(values)

For a 2D ndarray with at least two rows and two columns, return selections made directly from `values`:

```python
{
    "first_value": values[0, 0],
    "second_row": values[1],
    "second_column": values[:, 1],
    "top_left_block": values[:2, :2],
}
```

### 3.3 view_and_copy(values)

Create `middle_view = values[1:3]` and `middle_copy = values[1:3].copy()`. Return both without mutating either result or `values` during the call:

```python
{"view": middle_view, "copy": middle_copy}
```

The returned view must share memory with the input. The returned copy must not share memory with the input or view.

### 3.4 vector_operations(values, baseline, threshold, offset)

The two array inputs are 1D and have the same shape; `threshold` and `offset` are scalars. Create one boolean `mask = values >= threshold`, select `values[mask]`, calculate `difference = values - baseline`, and apply the scalar broadcast `adjusted = values + offset`. Return:

```python
{
    "mask": mask,
    "selected": selected,
    "difference": difference,
    "adjusted": adjusted,
}
```

### 3.5 reduction_summary(values)

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

### 3.6 reshape_and_transpose(values, rows, columns)

Create `grid = np.reshape(values, (rows, columns))`, then `transposed = grid.T`. Return:

```python
{
    "grid": grid,
    "grid_shape": grid.shape,
    "transposed": transposed,
    "transposed_shape": transposed.shape,
}
```

### 3.7 count_at_or_above(values, threshold)

Reshape the input to one dimension with `flattened = np.reshape(values, values.size)`. Create the scalar-comparison mask `flattened >= threshold` and return its count with `np.sum(...)`.

### 3.8 Complete the driver

Complete `analysis.py` using its supplied imports and main guard. Inside `main()`, in this order:

1. call `load_measurements("observations.csv")` once;
2. use the loaded array with all seven helpers;
3. print the existing shape, dtype, reduction, and count summary;
4. print the four selections, the vector-operation results, and the reshape/transposed arrays; and
5. use a separate copy of the loaded data to print the view and copy before and after changing that copy's first middle value to `-99`.

Use the values returned by the helpers for the displayed results.

Running `python analysis.py` must print a readable transcript and capture it as `output/analysis.txt`. Start with the existing six summary lines, then print these labeled learning results in this order:

```text
Measurements shape: (6, 2)
Measurements dtype: float64
Overall mean: 25.0
Column means: [20. 30.]
Row means: [15. 25. 35. 15. 25. 35.]
Values at or above 30: 6
First value: 10.0
Second row: [20. 30.]
Second column: [20. 30. 20. 20. 30. 40.]
Top-left block: [[10. 20.]
 [20. 30.]]
View before change: [[20. 30.]
 [30. 40.]]
Copy before change: [[20. 30.]
 [30. 40.]]
View after source change: [[-99.  30.]
 [ 30.  40.]]
Copy after source change: [[20. 30.]
 [30. 40.]]
Mask at or above 30: [False  True False False  True  True]
Selected values: [30. 30. 40.]
Difference from baseline: [10. 10. 10. 10. 10. 10.]
Adjusted values: [25. 35. 45. 25. 35. 45.]
Grid: [[10. 20. 20. 30.]
 [30. 40. 10. 20.]
 [20. 30. 30. 40.]]
Transpose: [[10. 30. 20.]
 [20. 40. 30.]
 [20. 10. 30.]
 [30. 20. 40.]]
```

> **Checkpoint — `output/analysis.txt`**
> Run `python analysis.py > output/analysis.txt` to save the complete transcript above.

## Check your work

With the candidate environment active, run:

```bash
python analysis.py
python check_assignment.py
```

A complete submission passes every check. Correct the named artifacts, regenerate them, and check again.

### Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifacts | Format and completion criteria | Points |
|---|---|---:|
| `.python-version`, `requirements.txt`, `output/environment_check.txt` | Exact environment records with final newlines and the two probe lines in Task 1. | 20 |
| `output/head_preview.txt`, `output/tail_preview.txt`, `output/site_counts.txt`, `output/site_count_lines.txt` | UTF-8 text with the specified fixture previews, ordered site/count pairs, and `3 output/site_counts.txt`. Whitespace is normalized for counts and `wc` fields. | 40 |
| `output/analysis.txt` | UTF-8 text with every transcript line shown in Task 3, in that order, including array spacing. | 40 |

## Submit

In VS Code Source Control, inspect and stage the environment records, `PIPELINE.md`, and all six output text files. Keep `.venv/` and `recreation-check/` ignored. Commit with `Record reproducible terminal workflow`. Inspect and stage `array_analysis.py` and `analysis.py`, then commit with `Implement NumPy array analysis`.

Publish or sync the branch. With no unfinished changes, switch to `main`, run **Git: Merge...**, and select `feature/numpy-analysis`. Resolve any unexpected conflict, inspect the resolution, and sync. Confirm the completed records, pipeline, six output files, and Python files on `main` in the repository browser. GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork. If a required VS Code control is unavailable, record its message and contact the instructor.
