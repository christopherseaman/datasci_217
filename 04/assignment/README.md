# Assignment 04: Fresh Notebooks and Labeled pandas Data

## Files

```text
assignment/
├── assignment.ipynb                 # notebook scaffold to complete
├── .python-version, requirements.txt # supplied environment records
├── data/purchases.csv, data/fixture.json # supplied synthetic input/manifest
├── check_assignment.py, grading.py  # supplied checker; keep unchanged
└── output/                         # generate and commit two CSV artifacts
```

## Setup

Open **Terminal → New Terminal** in VS Code at the assignment directory. If you use a native terminal or WSL Ubuntu instead, first `cd` into the assignment directory.

Use Python 3.13, NumPy 2.3.3, and pandas 3.0.5. Open `04/assignment` or its standalone repository in VS Code. Sync `main` and finish outstanding changes before editing.

From the assignment directory, create the environment and install the two deliberate notebook dependencies:

```bash
uv python install 3.13
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

In the supplied notebook, use the course-supported local Jupyter or VS Code host and select this Python 3 environment as its kernel. Keep the work local and use the supplied synthetic data. Never include credentials or private information in notebook source or output.

## Task 1: repair notebook state

### 1.1 Reorder the cells

The scaffold deliberately places this dependent cell first:

```python
adjusted_rate = base_rate + 2
```

The producer cell defining `base_rate = 3` appears later. Move the complete producer cell above the dependent cell. Do not copy the definition into another cell. When run, the repaired order should produce:

```text
base_rate: 3
adjusted_rate: 5
```

### 1.2 Explain the repair

Replace the TODO in the supplied Markdown explanation with a short explanation that distinguishes:

- visible cell order from actual execution order;
- the notebook source from retained kernel state;
- stored output from evidence of a fresh execution; and
- the repair and resulting cell order.


## Task 2: construct and select labeled pandas objects

### 2.1 Construct the labeled objects

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

### 2.2 Select and save the block

Use label selection for:

```python
label_block = measurement_table.loc[
    "site-102":"site-103",
    ["baseline_c", "follow_up_c"],
]
```

Use the equivalent positional selection for `position_block` with `.iloc[1:3, 0:2]`. Recall that `.loc` includes the named stop label while `.iloc` excludes its positional stop.

Verify that the two blocks contain the same values. Write `label_block` to `LABELED_OUTPUT_PATH` while preserving its named row index. Reading `output/labeled_block.csv` as an ordinary CSV must produce the columns `record_id`, `baseline_c`, and `follow_up_c`.

> **Checkpoint: `output/labeled_block.csv`**
> Save the two selected rows with the named `record_id` index and both measurement columns.

## Task 3: portable CSV round trip

### 3.1 Read and select purchases

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

### 3.2 Sort, save, and read back

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

> **Checkpoint: `output/selected_purchases.csv`**
> Save the nine selected purchases in the order above with `index=False`, then read the CSV back.

## Check your work

Save the notebook, regenerate both CSVs, and run from the assignment directory:

```bash
python check_assignment.py
```

Correct any reported artifact and repeat until every check passes.

### Completion contract

Grading totals 100 points and reads the two UTF-8 CSV files below. Column and row order matter; numeric values are compared with tolerance, while CSV quoting may vary.

| Artifact | Columns and completion criteria | Points |
|---|---|---:|
| `output/labeled_block.csv` | `record_id,baseline_c,follow_up_c`; rows `site-102,15,23` and `site-103,10,17`, in that order. | 40 |
| `output/selected_purchases.csv` | `purchase_id,item,quantity,unit_price,line_total`; the nine fixture purchases with quantity at least 2, their unchanged item/quantity/price values, and correct line totals. Sort by total descending and ID ascending as specified in Task 3. | 60 |

Additional files are allowed and ignored by the artifact checks.

## Submit

In VS Code Source Control, inspect the notebook and CSV diffs and confirm they contain only synthetic course data. Keep the supplied fixture, environment records, and checker unchanged. Stage `assignment.ipynb` and both CSV files, commit with `Complete Assignment 04 notebook`, and select **Sync Changes**. Confirm all three files in the assignment repository browser. GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork. If a required local notebook or VS Code control is unavailable, record its message and contact the instructor.
