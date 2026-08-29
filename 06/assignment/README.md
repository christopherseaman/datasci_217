# Assignment 06: Validated Combination and Structural Reshaping

Build one reproducible pandas notebook that combines tables only after stating
their grain, keys, cardinality, and preservation goal. Then examine vertical and
horizontal alignment and complete a reversible wide/long reshape without
aggregation.

This is a local Jupyter assignment. The supplied synthetic fixtures contain no
human-subject data and are different from the Lecture 06 demo data. Do not use
Colab, manual uploads, Drive mounts, network access, or `/content` paths. The
portable setup supports both a standalone exported assignment repository and
this full course repository.

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
the host application; the notebook kernel must use the environment you checked.

## Deliverables

Complete every `TODO` in `assignment.ipynb`. Restart the kernel and run all 25
cells from top to bottom. Commit these six files in the assignment repository:

- `assignment.ipynb`
- `output/specimen_merge_audit.csv`
- `output/combined_specimens.csv`
- `output/aligned_features.csv`
- `output/sensor_scores_long.csv`
- `output/sensor_scores_round_trip.csv`

The required CSVs are intentionally visible in VS Code Source Control and
GitHub Desktop. Commit them with the notebook. Do not edit the six files in
`data/`, `data/fixture.json`, the supplied notebook cells, environment records,
checker, or instructions. Stored notebook output is not evidence that the code
runs; the grader clears it and executes a disposable copy from fresh state.

After restart-and-run, use the discoverable student check:

```bash
python check_assignment.py
```

The checker reads files and notebook source but does not execute notebook code.
Fix each `[FIX]` message, rerun the notebook from a fresh kernel, and check again.

## Task 1: contract-first validated merge

State the row grain, primary/candidate/foreign keys, predicted cardinality,
preservation goal, join type, and predicted row count before combining tables.
Verify that the specimen keys are nonmissing and unique. Make the duplicate `R`
station-history key visible.

Attempt the unfiltered left merge with explicit `on="station_code"` and
`validate="many_to_one"`. Catch the pandas `MergeError` that the duplicated
right key naturally causes. Do not manufacture the failure flag.

Implement:

- `select_current_stations(history_table)`, which applies only the supplied
  `record_status == "current"` rule and returns the three ordered lookup columns;
- `validated_station_merge(specimen_table, station_table)`, which explicitly
  performs a left, many-to-one validated merge with `indicator=True` and does
  not mutate either input.

The canonical result preserves all seven specimens. Its indicator counts are
six `both`, one `left_only`, and zero `right_only`; `SP106`/`X` is the only
orphan. Save and explicitly read back `specimen_merge_audit.csv`.

## Task 2: concatenation and label alignment

Implement `stack_specimen_partitions(partition_map)`. For each insertion-ordered
mapping entry, copy the table, add the source label as an ordinary string
`source_partition` column, and concatenate rows with a fresh RangeIndex. Reject
an input that already uses the reserved column. Preserve first-seen column order
and do not mutate inputs or put provenance in a MultiIndex.

Use the function to reproduce all seven specimen rows from batches A and B. On
disposable copies, remove `mass_g` from batch B and add `review_note` only to
batch B. The resulting three missing masses and four missing notes demonstrate
column-label alignment; observe them without cleaning them.

Implement `align_specimen_features(mass_table, review_table)`. Validate unique,
nonmissing `specimen_id` keys, build named indexes, and concatenate the feature
columns horizontally with outer label alignment. Preserve first-seen union order
without resetting indexes before alignment. The canonical index is `SP101`,
`SP102`, `SP103`, `SP108`. Save and read back `combined_specimens.csv` and
`aligned_features.csv`; only the latter intentionally serializes its named
index.

## Task 3: reversible structural reshape

Implement `wide_to_long_scores(wide_table)` with `melt` and
`long_to_wide_scores(long_table, ordered_columns)` with `pivot`. Validate the
wide (`sensor_id`, `station_code`) key and the long (`sensor_id`,
`station_code`, `measurement_label`) key, preserve first-seen row order, and do
not mutate inputs.

The canonical long table has eight rows: four `baseline_value` rows followed by
four `followup_value` rows. Its structural key is unique, and pivoting it back
must exactly reproduce the original values, dtypes, rows, and columns. On a
disposable copy, append the first long row, show the two-row duplicate set, and
catch the natural `ValueError` when the wide function rejects that ambiguity.
Do not delete or aggregate the duplicate. Save and read back the long and
round-trip artifacts.

## Scope boundary

Use explicit keys, merge validation and indicators, `concat`, `melt`, and
structural `pivot`. General cleaning decisions, arbitrary deduplication,
GroupBy, aggregation, `transform`, `pivot_table`, crosstabs, plotting, dates,
time series, modeling, remote data, notebook magics, and shell commands are out
of scope. GroupBy and aggregation begin in Lecture 08.

## Assessment boundary

The implementation has a provisional 90-point automated overlay: 40 points for
Task 1, 27 for Task 2, and 23 for Task 3. Ten points of human review cover the
four explanations, organization, and privacy. The revised syllabus will decide
how that evidence maps to course policy; the notebook and public checker do not
declare a pass threshold or grade.
