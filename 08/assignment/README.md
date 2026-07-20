# Assignment 08: Grouped Results with an Explicit Grain

Build one reproducible pandas notebook that makes the grain of every grouped
result explicit. You will choose among three counting operations, create flat
named summaries, preserve source-row alignment with `transform`, and prove that
one aggregating pivot agrees with its equivalent two-key GroupBy result.

This assignment uses clean local Jupyter or the VS Code notebook interface. The
single prepared table is course-authored synthetic support-request data; it has
no real, identifying, or customer records. Assignment Colab is not supported.
Do not use manual uploads, Drive mounts, network access, absolute paths, or
`/content` paths. The supplied setup supports standalone Classroom50 and full
course checkouts, including nested launch directories inside the assignment.

## Setup

Use CPython 3.12.13. From this directory, create and activate a virtual
environment, install the two exact dependency records, and open Jupyter or the
VS Code notebook interface:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Complete
[PLATFORM_CHECK.md](PLATFORM_CHECK.md) before editing the notebook. Jupyter is
the host application; the selected notebook kernel must use the environment you
checked.

## Deliverables

Complete every `TODO` in `assignment.ipynb`. Restart the kernel and run all 25
cells from top to bottom. Submit these six files through Classroom50:

- `assignment.ipynb`
- `output/center_count_summary.csv`
- `output/center_summary.csv`
- `output/requests_with_context.csv`
- `output/center_channel_summary.csv`
- `output/mean_resolution_pivot.csv`

The five CSVs are intentionally visible in VS Code Source Control and GitHub
Desktop. Commit them with the notebook. Do not edit `data/`, the supplied
notebook cells, environment records, checker, or instructions. Stored notebook
output is not execution evidence: the grader clears it, removes generated CSVs,
and executes a disposable copy from fresh state.

After restart-and-run, use the discoverable student check:

```bash
python check_assignment.py
```

The checker reads files and notebook source but does not execute notebook code,
award points, or judge the quality of your written explanations. Fix each
`[FIX]` message, rerun the notebook from a fresh kernel, and check again.

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

The grader publishes an alternate valid prepared table with different category
labels, values, group sizes, row order, and a shuffled nondefault index. All five
functions must derive their behavior from their argument rather than canonical
literals, global data, or files.

## Scope and assessment boundary

Required work does not clean, impute, join, structurally reshape, filter groups,
use `GroupBy.apply`, manipulate MultiIndex, create crosstabs, visualize, analyze
dates/time series, calculate statistics or models, access remote/performance
tools, fetch network data, generate random data, or depend on a mutable date.

The provisional automated result has a maximum of 90: 20 for Task 1, 35 for
Task 2, 20 for Task 3, and 15 for shared integrity and portability. A separate
10-point human review covers only the grain/count, aggregate/transform, pivot,
privacy, and readability reasoning in your Markdown. Classroom50 provides that
review through its submission `review` link. The revised syllabus will decide
how this diagnostic evidence maps to course pass/fail policy; this notebook and
checker do not declare a threshold or grade conversion.
