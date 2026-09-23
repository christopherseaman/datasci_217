# Assignment 06: Validated Combination and Structural Reshaping

## Files

```text
assignment/
├── assignment.ipynb                 # notebook scaffold to complete
├── .python-version, requirements.txt # supplied environment records
├── data/                            # supplied synthetic CSVs and fixture manifest
├── check_assignment.py, grading.py  # supplied checker; keep unchanged
└── output/                         # generate and commit five CSV artifacts
```

## Setup

Open **Terminal → New Terminal** in VS Code at the assignment directory. If you use a native terminal or WSL Ubuntu instead, first `cd` into the assignment directory.

Use CPython 3.13. From this directory, create and activate a virtual
environment, and install the two exact dependency records. If you use the
notebook, open it through Jupyter or the VS Code notebook interface:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Select the same environment as the local notebook kernel. Run:

```bash
python --version
python -c "import sys, numpy, pandas; print(sys.executable); print(numpy.__version__); print(pandas.__version__)"
```

Expect Python 3.13, NumPy 2.3.3, and pandas 3.0.5, with an interpreter inside your activated environment. Run the supplied setup cell unchanged; it verifies fixture set `a06-structural-wrangling-v1` in either the standalone repository or `06/assignment`. Restore missing or checksum-mismatched fixtures before continuing. Use the synthetic course data locally and keep private information out of source and output.

## Deliverables

Complete every `TODO` in `assignment.ipynb`. Create these five CSV milestone artifacts:

- `output/specimen_merge_audit.csv`
- `output/combined_specimens.csv`
- `output/aligned_features.csv`
- `output/sensor_scores_long.csv`
- `output/sensor_scores_round_trip.csv`

## Task 1: contract-first validated merge

### 1.1 State and test the merge contract

State the row grain, primary/candidate/foreign keys, predicted cardinality,
preservation goal, join type, and predicted row count before combining tables.
Verify that the specimen keys are nonmissing and unique. Make the duplicate `R`
station-history key visible.

Attempt the unfiltered left merge with explicit `on="station_code"` and
`validate="many_to_one"`. Catch the pandas `MergeError` that the duplicated
right key naturally causes. Do not manufacture the failure flag.

### 1.2 Select current stations and merge

Implement:

- `select_current_stations(history_table)`, which applies only the supplied
  `record_status == "current"` rule and returns the three ordered lookup columns;
- `validated_station_merge(specimen_table, station_table)`, which explicitly
  performs a left, many-to-one validated merge with `indicator=True` and does
  not mutate either input.

The canonical result preserves all seven specimens. Its indicator counts are
six `both`, one `left_only`, and zero `right_only`; `SP106`/`X` is the only
orphan. Save and explicitly read back the merge audit.

> **Checkpoint: `output/specimen_merge_audit.csv`**
> Save all seven specimens in source order, with station fields and the merge indicator.

## Task 2: concatenation and label alignment

### 2.1 Stack the partitions

Implement `stack_specimen_partitions(partition_map)`. For each insertion-ordered
mapping entry, copy the table, add the source label as an ordinary string
`source_partition` column, and concatenate rows with a fresh RangeIndex. Reject
an input that already uses the reserved column. Preserve first-seen column order
and do not mutate inputs or put provenance in a MultiIndex.

Use the function to reproduce all seven specimen rows from batches A and B. On
disposable copies, remove `mass_g` from batch B and add `review_note` only to
batch B. The resulting three missing masses and four missing notes demonstrate
column-label alignment; observe them without cleaning them.

> **Checkpoint: `output/combined_specimens.csv`**
> Save the seven canonical rows, with `source_partition` labels `batch_a` and `batch_b`.

### 2.2 Align feature columns

Implement `align_specimen_features(mass_table, review_table)`. Validate unique,
nonmissing `specimen_id` keys, build named indexes, and concatenate the feature
columns horizontally with outer label alignment. Preserve first-seen union order
without resetting indexes before alignment. The canonical index is `SP101`,
`SP102`, `SP103`, `SP108`. Save and read back both artifacts; only the aligned features intentionally serialize their named index.

> **Checkpoint: `output/aligned_features.csv`**
> Save columns `specimen_id,mass_g,review_score` in the canonical index order above. Leave structural missingness as empty fields.

## Task 3: reversible structural reshape

### 3.1 Reshape and validate the keys

Implement `wide_to_long_scores(wide_table)` with `melt` and
`long_to_wide_scores(long_table, ordered_columns)` with `pivot`. Validate the
wide (`sensor_id`, `station_code`) key and the long (`sensor_id`,
`station_code`, `measurement_label`) key, preserve first-seen row order, and do
not mutate inputs.

### 3.2 Verify the round trip and duplicate failure

The canonical long table has eight rows: four `baseline_value` rows followed by
four `followup_value` rows. Its structural key is unique, and pivoting it back
must exactly reproduce the original values, dtypes, rows, and columns. On a
disposable copy, append the first long row, show the two-row duplicate set, and
catch the natural `ValueError` when the wide function rejects that ambiguity.
Do not delete or aggregate the duplicate. Save and read back the long and
round-trip artifacts.

> **Checkpoint: `output/sensor_scores_long.csv`**
> Save eight rows with columns `sensor_id,station_code,measurement_label,value`, preserving source order within each measurement label.

> **Checkpoint: `output/sensor_scores_round_trip.csv`**
> Save the four reconstructed rows with the original wide columns and source order.

## Check your work

Regenerate the five CSV artifacts and run from the assignment directory:

```bash
python check_assignment.py
```

Correct any reported artifact and repeat until every check passes.

### Completion contract

Grading totals 100 points and reads the five UTF-8 CSVs below. Column order, row order, text values, and empty fields must match the specified results. Numeric values are compared with tolerance; quoting may vary.

| Artifacts | Columns and completion criteria | Points |
|---|---|---:|
| `output/specimen_merge_audit.csv` | Original specimen columns followed by `station_name,region,_merge`; seven rows in source order, current station values, six `both` rows and orphan `SP106`/`X` as `left_only`. | 45 |
| `output/combined_specimens.csv`, `output/aligned_features.csv` | Combined: original specimen columns then `source_partition`, batch A followed by B, seven rows. Aligned: `specimen_id,mass_g,review_score`, rows `SP101,SP102,SP103,SP108`, source values and empty unmatched fields. | 30 |
| `output/sensor_scores_long.csv`, `output/sensor_scores_round_trip.csv` | Long: `sensor_id,station_code,measurement_label,value`, four baseline rows then four follow-up rows. Round trip: `sensor_id,station_code,baseline_value,followup_value`, reproducing all four source rows and values. | 25 |

Additional files are allowed and ignored by the artifact checks.

## Submit

Save the notebook, inspect the notebook and five CSV diffs in VS Code Source Control, commit them, and sync. Confirm all six files appear in your assignment repository.

GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork.
