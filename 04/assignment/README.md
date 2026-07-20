# Assignment 04: Fresh Notebooks and Labeled pandas Data

This assignment is the first required notebook assignment in the course. It checks that you can repair notebook state, construct and select labeled pandas objects, and complete one portable CSV round trip.

Complete it in clean local Jupyter or the VS Code notebook interface. Google Colab is used for Lecture 04 demos, but it is not yet a supported assignment-submission path. Do not upload this assignment to Colab or add a Colab badge.

The repository-delivery steps are in [`PLATFORM_CHECK.md`](PLATFORM_CHECK.md). Classroom50 delivers and grades Assignments 01–11; Assignment 04 is only the notebook-grading pilot.

## Starter files

- `assignment.ipynb`: the notebook you repair and complete;
- `.python-version` and `requirements.txt`: the candidate environment records; do not edit them;
- `data/purchases.csv` and `data/fixture.json`: the immutable synthetic input and its manifest; do not edit them;
- `output/.gitkeep`: keeps the generated-output directory in the starter repository;
- `check_assignment.py`: the discoverable public checker; do not edit it; and
- `PLATFORM_CHECK.md`: the unassessed local-Jupyter and GUI delivery checklist; do not edit it.

The supplied notebook setup cell uses later standard-library code to locate and verify the fixture. Run that cell, but do not edit it. It searches for both a flattened Classroom50 layout and this course-repository layout, so it does not need an absolute path, upload, network request, or Drive mount.

## Candidate environment

The implementation candidate is Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. These exact records support the course certification pass; they are not an invitation to install pandas 3.0.4.

From the assignment directory, create the environment and install the two deliberate notebook dependencies:

```bash
uv python install 3.12.13
uv venv --python 3.12.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Use the course-supported local Jupyter or VS Code host and select this Python 3 environment as the notebook kernel. Jupyter hosting and kernel support are platform tooling, not imports used by your assignment code.

## Task 1: repair notebook state

The starter deliberately places this dependent cell first:

```python
adjusted_rate = base_rate + 2
```

The producer cell defining `base_rate = 3` appears later. Move the complete producer cell above the dependent cell. Do not copy the definition into another cell. A restart-and-run-all must produce:

```text
base_rate: 3
adjusted_rate: 5
```

Replace the TODO in the supplied Markdown explanation with a short explanation that distinguishes:

- visible cell order from actual execution order;
- the notebook source from retained kernel state;
- stored output from evidence of a fresh execution; and
- the repair and restart-and-run-all check you performed.

This explanation is human-reviewed. The grader independently checks the fresh executable result and ignores stored output.

## Task 2: construct and select labeled pandas objects

Keep the supplied arrays. Complete the Task 2 code cell using the exact variable names below.

Create `reading_by_site` as a pandas Series from `reading_values`:

- index: `north`, `south`, `east`, `west`;
- name: `reading_c`.

Create `measurement_table` as a DataFrame from `measurement_values`:

- index: `site-101`, `site-102`, `site-103`, `site-104`;
- index name: `record_id`;
- columns: `baseline_c`, `follow_up_c`.

Then create:

```python
baseline_series = measurement_table["baseline_c"]
baseline_table = measurement_table[["baseline_c"]]
```

`baseline_series` must be a Series. `baseline_table` must be a one-column DataFrame.

Use label selection for:

```python
label_block = measurement_table.loc[
    "site-102":"site-103",
    ["baseline_c", "follow_up_c"],
]
```

Use the equivalent positional selection for `position_block` with `.iloc[1:3, 0:2]`. Recall that `.loc` includes the named stop label while `.iloc` excludes its positional stop.

Verify that the two blocks contain the same values. Write `label_block` to `LABELED_OUTPUT_PATH` while preserving its named row index. Reading `output/labeled_block.csv` as an ordinary CSV must produce the columns `record_id`, `baseline_c`, and `follow_up_c`.

## Task 3: portable CSV round trip

Read the immutable input exactly through the supplied path:

```python
purchases = pd.read_csv(DATA_PATH)
```

Inspect its shape, columns, dtypes, and first rows. Create exactly this named, index-aligned Boolean Series mask:

```python
quantity_at_least_two = purchases["quantity"] >= 2
```

Use `.loc` with that mask and these explicit source columns:

```text
purchase_id, item, quantity, unit_price
```

Copy that selection, then add one arithmetic derived column:

```python
line_total = quantity * unit_price
```

Sort deterministically with the exact keys and directions:

```python
by=["line_total", "purchase_id"]
ascending=[False, True]
```

`purchase_id` is the unique tie-breaker. With the supplied fixture, the nine selected IDs must be:

```text
P008, P003, P004, P006, P001, P011, P007, P009, P012
```

Write `selected_purchases` to `SELECTED_OUTPUT_PATH` with `index=False`. Read that file back through the same supplied path into `round_trip`. The final supplied verification cell checks the exact schema, nine-row count, mask condition, arithmetic, and deterministic order.

## Generated artifacts and execution evidence

Submit exactly these student-authored or generated artifacts:

1. `assignment.ipynb`;
2. `output/labeled_block.csv`; and
3. `output/selected_purchases.csv`.

A generated CSV is a separate file artifact; it is not the same thing as output stored under a notebook cell. Before submission:

1. save the visible notebook source;
2. restart the kernel;
3. run all cells from top to bottom;
4. confirm both CSV files were recreated; and
5. run the public checker from the assignment directory:

```bash
python check_assignment.py
```

A complete artifact set ends with `All public checks passed.` The public checker derives expected results from the fixture and does not trust editable assertions or displayed notebook output.

The centrally managed grader copies the submission to disposable directories, deletes generated CSVs, clears stored output and execution counts, appends instructor-owned verification to the disposable notebook, and executes a fresh kernel. It also repeats with a relocated checkout and a second valid fixture. Published grader logic is discoverable and enforces only this written contract.

## Scope and human-review boundary

Do not add cleaning, missing-value decisions, type conversion, dates, joins, concatenation, reshape, GroupBy, aggregation, plotting, modeling, performance work, network access, absolute paths, `/content` paths, or Drive mounts.

Automated checks cover notebook structure and fresh execution, pandas object types and metadata, label/position selection, the mask, arithmetic, deterministic order, fixture integrity, and both CSV artifacts. Human review checks only that the state explanation is understandable, distinguishes source/kernel/execution/output, identifies restart-and-run-all, uses clear task headings, and contains no sensitive information.
