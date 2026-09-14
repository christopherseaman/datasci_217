# Assignment 08: Grouped Results with an Explicit Grain

Build one reproducible pandas notebook that makes the grain of every grouped
result explicit. You will choose among three counting operations, create flat
named summaries, preserve source-row alignment with `transform`, and prove that
one aggregating pivot agrees with its equivalent two-key GroupBy result.

This is a local-first assignment. The notebook (or an equivalent documented
source file) remains a required coursework deliverable, though automated grading
does not execute it. The single
prepared table is course-authored synthetic support-request data; it has no
real, identifying, or customer records. Assignment Colab is not supported.
Do not use manual uploads, Drive mounts, network access, absolute paths, or
`/content` paths. The supplied setup supports standalone exported assignment
repositories and full course checkouts, including nested launch directories
inside the assignment.

## Setup

Use CPython 3.14. From this directory, create and activate a virtual
environment, and install the two exact dependency records. If you use the
notebook, open it through Jupyter or the VS Code notebook interface:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Complete
[PLATFORM_CHECK.md](PLATFORM_CHECK.md) before preparing artifacts. If you run
the notebook, its kernel must use the environment you checked.

## Deliverables

Complete every `TODO` in `assignment.ipynb` or your working copy. Create and commit these five CSV milestone artifacts in the assignment repository:

- `output/center_count_summary.csv`
- `output/center_summary.csv`
- `output/requests_with_context.csv`
- `output/center_channel_summary.csv`
- `output/mean_resolution_pivot.csv`

The five CSVs are intentionally visible in VS Code Source Control and GitHub
Desktop. Commit them with your completed notebook. Do not edit `data/`,
the supplied notebook cells, environment records, checker, or instructions.
Automated grading reads the committed CSVs; retain and submit your completed notebook.

After creating the committed artifacts, use the discoverable student check:

```bash
python check_assignment.py
```

The shared checker reads committed CSVs and reports milestone points without executing notebook code or judging written explanations. Fix each failed check, regenerate the artifacts, and check again.

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

The fixture grain is one synthetic support request. `request_id` identifies a
row; `center` is the first grouping key; `channel` is the second; `agent_id` may
repeat; `resolution_minutes` is a complete measurement; and
`satisfaction_score` is optional. Center order is Central, Harbor, Ridge,
Valley; channel order is Email, Phone, Chat. Valley is an unused category.
Harbor--Phone is absent, and three satisfaction scores are missing. These are
prepared facts to analyze, not cleaning decisions.

## Task 1: grain and count semantics

Before grouping, state the input grain, grouping key and unit, predicted observed
groups, observed-category policy, and output grain. Choose the operation that
answers each question:

- How many support-request rows were recorded? Use `size`.
- How many requests have a recorded satisfaction score? Use selected-column
  `count`.
- How many distinct agents appear? Use selected-column `nunique`.

Implement `build_count_summary(request_table)` with explicit
`observed=True`, `sort=True`, and `dropna=True`. Return one flat row per observed
center and save/read back `center_count_summary.csv`.

## Task 2: aggregation, transform, and two keys

Implement `build_center_summary(request_table)` with flat named aggregation and
deliberate `as_index=False`. Implement `add_center_context(request_table)` with
selected-Series `transform("mean")`; its result must preserve the input row count
and exact index. Implement `build_center_channel_summary(request_table)` as one
bounded flat two-key summary. Do not mutate inputs or round results. Save and
read back all three Task 2 outputs.

## Task 3: one aggregating pivot and equivalence

Implement `build_resolution_pivot(request_table)` with the assignment's only
`pd.pivot_table` call. Its five roles are `index="center"`,
`columns="channel"`, `values="resolution_minutes"`, `aggfunc="mean"`, and
`observed=True`; also use explicit `sort=True` and `dropna=True`. Compare every
populated pivot cell with the equivalent GroupBy mean. Keep Harbor--Phone
missing; do not replace it with zero. Save and read back
`mean_resolution_pivot.csv`.

For your own confidence, try the completed functions with a prepared table that
has different category labels, values, group sizes, row order, and a shuffled
nondefault index. All five functions should derive their behavior from their
argument rather than canonical literals, global data, or files.

## Scope and assessment boundary

Required work does not clean, impute, join, structurally reshape, filter groups,
use `GroupBy.apply`, manipulate MultiIndex, create crosstabs, visualize, analyze
dates/time series, calculate statistics or models, access remote/performance
tools, fetch network data, generate random data, or depend on a mutable date.

Automated grading totals 100 points: 10 for fixture integrity, 25 for Task 1,
40 for Task 2, 20 for Task 3, and 5 for artifact inventory. There are no
separate human-review points; the grain/count, aggregate/transform, pivot,
privacy, and readability explanations remain required coursework context.

## Public automated grading

grading.py is the shared ruleset for students, pytest, and graders. Run
python check_assignment.py [submission_dir], or add --json for a
datasci217/grading-result/v1 result. It reads committed artifacts only, never
notebooks or submission code. The visible tests award fixture integrity (10),
Task 1 (25), Task 2 (40), Task 3 (20), and artifact inventory (5).
