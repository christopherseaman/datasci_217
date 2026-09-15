# Assignment 04: Fresh Notebooks and Labeled pandas Data

This assignment is the first required notebook assignment in the course. It checks that you can repair notebook state, construct and select labeled pandas objects, and complete one portable CSV round trip.

Complete and commit the supplied notebook along with its CSV artifacts. Automated grading reads the committed CSVs without executing the notebook.

The repository-delivery steps are in [`PLATFORM_CHECK.md`](PLATFORM_CHECK.md). Commit the notebook and generated CSVs in the assignment repository; the optional Actions workflow is feedback only.

## Starter files

- `assignment.ipynb`: the notebook you repair and complete;
- `.python-version` and `requirements.txt`: the candidate environment records; do not edit them;
- `data/purchases.csv` and `data/fixture.json`: the immutable synthetic input and its manifest; do not edit them;
- `output/.gitkeep`: keeps the generated-output directory in the starter repository;
- `check_assignment.py` and `grading.py`: the discoverable public checker and scoring rules; do not edit them; and
- `PLATFORM_CHECK.md`: the unassessed local-Jupyter and GUI delivery checklist; do not edit it.

The supplied setup cell locates and verifies the fixture; do not edit it. It supports both a standalone exported layout and this course repository.

## Candidate environment

The implementation candidate is Python 3.14, NumPy 2.3.3, and pandas 3.0.5. These exact records are the tested assignment contract.

From the assignment directory, create the environment and install the two deliberate notebook dependencies:

```bash
uv python install 3.14
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

In the supplied notebook, use the course-supported local Jupyter or VS Code host and select this Python 3 environment as its kernel. Jupyter hosting and kernel support are platform tooling, not imports used by your assignment code.

## Task 1: repair notebook state

The starter deliberately places this dependent cell first:

```python
adjusted_rate = base_rate + 2
```

The producer cell defining `base_rate = 3` appears later. Move the complete producer cell above the dependent cell. Do not copy the definition into another cell. When run, the repaired order should produce:

```text
base_rate: 3
adjusted_rate: 5
```

Replace the TODO in the supplied Markdown explanation with a short explanation that distinguishes:

- visible cell order from actual execution order;
- the notebook source from retained kernel state;
- stored output from evidence of a fresh execution; and
- the repair and resulting cell order.

This explanation is human-reviewed. Automated checks read the committed CSV artifacts directly and do not infer how they were produced.

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

Submit these required student-authored or generated artifacts:

1. `output/labeled_block.csv`; and
2. `output/selected_purchases.csv`.

Additional input or diagnostic files are allowed; grading checks the required artifacts and ignores extras.

A generated CSV is a separate committed milestone artifact; it is not the same thing as output stored under a notebook cell. Before submission:

1. save the visible notebook source;
2. create or regenerate both CSV files;
3. confirm both CSV files are committed; and
5. run the public checker from the assignment directory:

```bash
python check_assignment.py
```

A complete artifact set ends with `All public checks passed.` The public checker derives expected results from the fixture and does not trust editable assertions or displayed notebook output. It is also the grader contract: instructors run a trusted copy against submitted artifacts and never execute the notebook.

The automated grader reads the two committed CSV artifacts directly. Optional notebook execution is useful local QA but is not required for grading.

## Scope and assessment

Do not add cleaning, missing-value decisions, type conversion, dates, joins, concatenation, reshape, GroupBy, aggregation, plotting, modeling, performance work, network access, absolute paths, `/content` paths, or Drive mounts.

Automated grading totals 100 points: 40 for the labeled-block artifact and 60 for the selected-purchases artifact. The trusted grader supplies its own fixture; it does not require starter files, the notebook, or protected package files in a submission. There are no separate human-review points; the notebook explanation and task headings remain required coursework context.

### Artifact comparison

CSV checks compare parsed columns and values, not file hashes or quoting. Preserve the row order explicitly requested for selection, sorting, concatenation, and reshaping. Each milestone is assessed independently.
