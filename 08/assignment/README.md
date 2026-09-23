# Assignment 08: Grouped Results with an Explicit Grain

## Files

```text
assignment/
├── assignment.ipynb     # Provided notebook scaffold to complete
├── data/                # Provided fixtures
├── requirements.txt     # Provided pinned environment
├── check_assignment.py  # Provided completion checker
└── output/              # Generated artifacts to submit
```

## Setup

In VS Code, open the assignment folder and choose **Terminal → New Terminal**. You can also use your native terminal or WSL Ubuntu; change to the assignment directory before running these commands.

From this assignment directory, create a Python 3.13 environment and install the pinned requirements:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. If Python 3.13 is missing, run `uv python install 3.13`. Select this environment as the kernel when opening the notebook in VS Code or Jupyter. The course uses pandas 3.0.5.

Keep the supplied `data/` files unchanged. Open the complete assignment directory; the setup cell locates and verifies its fixtures in either a standalone assignment repository or the course repository. Restore missing or checksum-mismatched fixtures before continuing.

## Terms and data contract

- **Input row grain** says what one source row represents.
- A **grouping key** is one column or bounded column combination whose values
  determine group membership. A **group** is the input rows sharing one observed
  key value or combination. The **grouping unit** is the real-world category or
  entity represented by one group.
- **Output row grain** says what one result row represents. An **aggregation**
  reduces each group's rows to one or more summary values. A **GroupBy object**
  records how rows are split; it is not itself a summary table.
- The **observed-category policy** controls whether unused categorical levels or
  combinations appear; use `observed=True`. The **missing-key policy** controls
  whether missing grouping keys form a group; the supplied keys are complete and
  use `dropna=True`. **Output order** is deliberate; use `sort=True` with the
  declared ordered categories.
- `size` counts input rows regardless of missing values elsewhere. Selected-
  column `count` counts nonmissing values in that column. Selected-column
  `nunique` counts distinct nonmissing values in that column.
- A **named aggregation** pairs an output name with a source column and
  calculation. `as_index=False` keeps grouping keys as ordinary flat columns.
- `transform` performs a within-group calculation and returns one same-index
  value per input row.
- A **two-key group** contains rows sharing one observed combination of both
  key values.
- Lecture 06 structural `pivot` reshapes uniquely keyed values and performs no
  aggregation. An aggregating `pivot_table` groups repeated combinations and
  places their summaries across row and column axes.
- An **absent combination** has no input row for its key combination. It is not
  a measured zero.

The fixture grain is one synthetic support request. `request_id` identifies a row; `center` is the first grouping key; `channel` is the second; `agent_id` may repeat; `resolution_minutes` is a complete measurement; and `satisfaction_score` is optional. Center order is Central, Harbor, Ridge, Valley; channel order is Email, Phone, Chat. Valley is an unused category. Harbor--Phone is absent, and three satisfaction scores are missing. These are prepared facts to analyze, not cleaning decisions.

## Question 1: Grain and count semantics

### 1.1 Count observed requests, scores, and agents

Before grouping, state the input grain, grouping key and unit, predicted observed groups, observed-category policy, and output grain. Choose the operation that answers each question:

- How many support-request rows were recorded? Use `size`.
- How many requests have a recorded satisfaction score? Use selected-column
  `count`.
- How many distinct agents appear? Use selected-column `nunique`.

Implement `build_count_summary(request_table)` with explicit `observed=True`, `sort=True`, and `dropna=True`. Return one flat row per observed center and save/read back `center_count_summary.csv`.

> **Checkpoint: `output/center_count_summary.csv`**

## Question 2: Aggregation, transform, and two keys

### 2.1 Create the center summary

Implement `build_center_summary(request_table)` with flat named aggregation and deliberate `as_index=False`. Save and read back the center summary without rounding results.

> **Checkpoint: `output/center_summary.csv`**

### 2.2 Add request context

Implement `add_center_context(request_table)` with selected-Series `transform("mean")`; its result must preserve the input row count and exact index. Save and read back the request-level result without mutating the source table.

> **Checkpoint: `output/requests_with_context.csv`**

### 2.3 Summarize center/channel pairs

Implement `build_center_channel_summary(request_table)` as one flat two-key summary. Save and read back the result without rounding values.

> **Checkpoint: `output/center_channel_summary.csv`**

## Question 3: Aggregating pivot and equivalence

### 3.1 Compare the pivot with grouped means

Implement `build_resolution_pivot(request_table)` with the assignment's only `pd.pivot_table` call. Its five roles are `index="center"`, `columns="channel"`, `values="resolution_minutes"`, `aggfunc="mean"`, and `observed=True`; also use explicit `sort=True` and `dropna=True`. Compare every populated pivot cell with the equivalent GroupBy mean. Keep Harbor--Phone missing; do not replace it with zero. Save and read back `mean_resolution_pivot.csv`.

> **Checkpoint: `output/mean_resolution_pivot.csv`**

## Check Your Work

Run this from the assignment directory after saving your artifacts:

```bash
python check_assignment.py
```

Fix each failed check, regenerate the affected files, and run the checker again. It reads saved artifacts without running your code.

### Completion contract

Save these five CSVs in `output/` with the columns below in order and no extra index column. Row identities must be unique and match the supplied requests; row order may differ. Values and missingness must match the requested calculations; retain full calculation precision when exporting.

| Artifact | Columns in order and completion criteria |
|---|---|
| `output/center_count_summary.csv` | `center`, `request_count`, `satisfaction_count`, `unique_agent_count`: one row per observed center, counting requests, recorded satisfaction values, and distinct agents. |
| `output/center_summary.csv` | The four count-summary columns, then `total_resolution_minutes`, `mean_resolution_minutes`: source-derived sum and mean per observed center. |
| `output/requests_with_context.csv` | Original columns `request_id`, `center`, `agent_id`, `channel`, `resolution_minutes`, `satisfaction_score`, then `center_mean_resolution_minutes`, `difference_from_center_mean`: every original request and its center mean and resolution-minus-mean difference. |
| `output/center_channel_summary.csv` | `center`, `channel`, `request_count`, `mean_resolution_minutes`: one row per observed center/channel pair. |
| `output/mean_resolution_pivot.csv` | `center`, `Email`, `Phone`, `Chat`: mean resolution minutes per observed center/channel pair; Harbor–Phone stays missing. |

Task 1 is worth 29 points, Task 2 is worth 47, and Task 3 is worth 24: 100 total.

## Submit

In VS Code Source Control, inspect your completed notebook and required `output/` files, then commit and push them. Alternatively, use **Add file → Upload files** on the GitHub website and commit the files at their required paths. Keep private data, credentials, virtual environments, and notebook checkpoints out of your submission. GitHub Actions runs the assignment checks automatically on every push. If your fork has Actions disabled, enable it once in the Actions tab. Review the feedback, then regenerate, check, commit, and push corrected artifacts if needed.
