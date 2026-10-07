# Assignment 06: Merge, Stack, and Reshape Clinic Data

## Overview

Merge specimens with their clinic records, stack two delivery batches, line up volumes with transit times, and reshape blood pressure readings between wide and long, saving each result for the checks.

All six files are synthetic. A hospital's central lab receives specimens collected at its neighborhood clinics.

- `specimens.csv`: one row per specimen. `specimen_id` is its unique ID, `patient_id` the patient it came from, `collection_number` that patient's first, second, ... specimen, `clinic_id` the clinic that collected it, `specimen_type` blood, urine, or swab, and `volume_ml` the volume in mL.
- `clinics_history.csv`: one row per clinic record: `clinic_id`, `clinic_name`, `region`, and `record_status`. When a clinic is renamed, its old record stays in the file marked `retired`, so `clinic_id` repeats here.
- `specimens_batch_a.csv` and `specimens_batch_b.csv`: the same seven specimens, delivered as two batches with the same columns as `specimens.csv`.
- `transit_times.csv`: the courier's log, one row per delivered specimen: `specimen_id` and `transit_min`, the minutes from collection to lab receipt.
- `sbp_wide.csv`: one row per patient in a blood pressure follow-up: `patient_id`, then systolic blood pressure in mmHg at the `baseline` and `followup` visits.

```text
specimen_id,patient_id,collection_number,clinic_id,specimen_type,volume_ml
SP101,P201,1,K01,blood,5.0
SP102,P201,2,K01,urine,30.0

clinic_id,clinic_name,region,record_status
K01,Bayview Annex,southeast,retired
K01,Bayview Clinic,southeast,current

patient_id,baseline,followup
P201,148,136
```

## Setup

1. Fork the assignment repository on GitHub, clone your fork as in Lecture 01, and open the cloned folder itself.
2. In the integrated terminal, create the environment, activate it, and install the packages `pyproject.toml` and `uv.lock` list. Do not run `uv init`: the project files already exist.

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```

    - Expect: `uv sync` lists `+ pandas==3.0.5` and `+ ipykernel==6.29.5` among the packages it installs.
3. Open `assignment.ipynb`, click **Select Kernel**, and choose the Python inside this project's `.venv`.
4. Run the notebook's first two code cells.
    - Expect: `pandas: 3.0.5` and `data folder found: True`, then `specimens: (7, 6)` first among the shapes. `False` means the notebook is not running from the assignment folder.

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/                   # supplied files; keep them exactly as handed out
│   ├── specimens.csv
│   ├── clinics_history.csv
│   ├── specimens_batch_a.csv
│   ├── specimens_batch_b.csv
│   ├── transit_times.csv
│   └── sbp_wide.csv
├── pyproject.toml          # supplied: the project's packages, numpy, pandas, and ipykernel
├── uv.lock                 # supplied: the exact versions `uv sync` installs
├── .python-version         # supplied: tells uv to use Python 3.13
├── CHECKS.md               # supplied: what each check looks for
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── specimen_merge_audit.csv  # you generate in Task 1.2
    ├── combined_specimens.csv    # you generate in Task 2.1
    ├── aligned_features.csv      # you generate in Task 2.3
    ├── sbp_long.csv              # you generate in Task 3.1
    └── sbp_round_trip.csv        # you generate in Task 3.2
```

## Task 1: Merge specimens with their clinics

Each specimen should get the name and region of the clinic that collected it: a left merge of `specimens` with the clinic records on `clinic_id`, many specimens to one clinic.

### 1.1 Find the repeated clinic key

In the Task 1.1 cells:

1. Set `specimen_ids_unique` to `specimens["specimen_id"].is_unique`. It prints `specimen_id unique: True`.
2. Select every `clinics_history` row whose `clinic_id` appears more than once, with `duplicated(subset=["clinic_id"], keep=False)` as the mask, and name the result `repeated_clinic_rows`. It shows K01's two rows: `Bayview Annex` (retired) and `Bayview Clinic` (current).
3. In the next cell, try the merge with the contract written down, and catch the error it raises with `try`/`except` (Lecture 02). The lecture's "Catch a Broken Merge Contract" snippet shows the merge that raises it:

```python
try:
    pd.merge(specimens, clinics_history, on="clinic_id", how="left", validate="many_to_one")
except pd.errors.MergeError as error:
    print("MergeError:", error)
```

It prints `MergeError: Merge keys are not unique in right dataset; not a many-to-one merge`, followed by the repeated key.

### 1.2 Keep the current records, merge, and save

In the Task 1.2 cell:

1. Build `current_clinics`: the `clinics_history` rows whose `record_status` is `"current"`, with the columns `clinic_id`, `clinic_name`, and `region`, in one `.loc`. It prints `current clinic_id unique: True`.
2. Left-merge `specimens` with `current_clinics` on `clinic_id`, with `validate="many_to_one"` and `indicator=True`, and name the result `merge_audit`. It prints `rows: 7`, then the `_merge` counts: `both 6`, `left_only 1`, `right_only 0`. The one `left_only` row is SP106, whose clinic K09 has no record.
3. Save `merge_audit` to `MERGE_AUDIT_PATH` with `index=False`.

> **Checkpoint: `output/specimen_merge_audit.csv`**
> The header line `specimen_id,patient_id,collection_number,clinic_id,specimen_type,volume_ml,clinic_name,region,_merge`, then seven rows, SP101 to SP107, one per specimen. Each has its clinic's current name and region (SP101 reads `Bayview Clinic,southeast`), and SP106 has empty `clinic_name` and `region` and `_merge` `left_only`. A `record_status` column as well, which merging the current rows without selecting columns keeps, is accepted.

## Task 2: Stack the batches and line up features

### 2.1 Stack the two batches

In the Task 2.1 cell:

1. Record where each row came from: add a `source_partition` column holding `"batch_a"` to `batch_a`, and one holding `"batch_b"` to `batch_b`, as the lecture's "Stack rows" snippet adds `source_file`.
2. Stack `batch_a` above `batch_b` with `pd.concat()` and `ignore_index=True`, and name the result `combined_specimens`. It prints `rows: 7` and `same specimens as specimens.csv: True`.
3. Save `combined_specimens` to `COMBINED_PATH` with `index=False`.

> **Checkpoint: `output/combined_specimens.csv`**
> The header line `specimen_id,patient_id,collection_number,clinic_id,specimen_type,volume_ml,source_partition`, then seven rows: SP101 to SP104 with `batch_a`, then SP105 to SP107 with `batch_b`.

### 2.2 Stack batches whose columns differ

This step saves nothing. It shows what the lecture's "Choose a column set" snippet describes: stacking tables whose columns differ keeps every column and leaves empty cells where a table lacks one.

1. Build `batch_b_changed`: `batch_b` without its `volume_ml` column (`drop(columns="volume_ml")`, Lecture 04), then add a `courier_note` column holding `"late pickup"`.
2. Stack `batch_a` above `batch_b_changed` with `pd.concat()` and `ignore_index=True`, and name the result `drift_preview`.

The printed missing counts show `volume_ml 3` (batch B's rows have none) and `courier_note 4` (batch A's rows have none); every other column shows 0.

### 2.3 Line up volumes and transit times

The courier's log and batch A overlap only in part: SP101 and SP104 have no transit time, and SP108 is in the log but in neither batch. Line them up by label, as the lecture's "Align columns by index" snippet does:

1. Build `volumes`: `batch_a`'s `specimen_id` and `volume_ml` columns, with `set_index("specimen_id")`.
2. Build `transit`: `transit_times.set_index("specimen_id")`. Both indexes print as unique.
3. Put them side by side with `pd.concat([volumes, transit], axis=1)` and name the result `aligned_features`.
4. Save `aligned_features` to `ALIGNED_PATH` with its index: the specimen IDs are meaningful labels, so leave out `index=False`.

> **Checkpoint: `output/aligned_features.csv`**
> The header line `specimen_id,volume_ml,transit_min`, then five rows, SP101 to SP104 and SP108. SP101 and SP104 have an empty `transit_min`, and SP108 has an empty `volume_ml`.

## Task 3: Reshape the blood pressure readings

### 3.1 Melt wide to long

In the Task 3.1 cell:

1. Melt `sbp_wide` to one row per patient and visit, as the lecture's "Melt wide data" snippet does: `id_vars=["patient_id"]`, `value_vars=["baseline", "followup"]`, `var_name="visit"`, and `value_name="sbp"`. Name the result `sbp_long`. It prints `rows: 8` and `repeated patient-visit pairs: 0`.
2. Save `sbp_long` to `SBP_LONG_PATH` with `index=False`.

> **Checkpoint: `output/sbp_long.csv`**
> The header line `patient_id,visit,sbp`, then eight rows: the four `baseline` readings (P201 to P204), then the four `followup` readings. The first row is `P201,baseline,148`.

### 3.2 Pivot back to wide

In the Task 3.2 cell:

1. Pivot `sbp_long` back with `index="patient_id"`, `columns="visit"`, and `values="sbp"`, then `reset_index()`, and name the result `sbp_round_trip`.
2. Drop the leftover header label with `sbp_round_trip.columns.name = None`, as the lecture's "Pivot long data back to wide" snippet does. It prints `round trip matches sbp_wide: True`.
3. Save `sbp_round_trip` to `SBP_ROUND_TRIP_PATH` with `index=False`.

> **Checkpoint: `output/sbp_round_trip.csv`**
> The header line `patient_id,baseline,followup`, then four rows, P201 to P204, with the same readings as `data/sbp_wide.csv`.

### 3.3 Find the pair that stops `pivot()`

This step saves nothing. P202's follow-up blood pressure was rechecked, so the supplied lines at the top of the cell add a second P202 `followup` reading, 147 mmHg, in `sbp_rechecked`.

1. Select every `sbp_rechecked` row whose `patient_id` and `visit` pair appears more than once, with `duplicated(subset=["patient_id", "visit"], keep=False)` as the mask, and name the result `repeated_pairs`. It shows two rows: P202 `followup` 151 and P202 `followup` 147.
2. Try the Task 3.2 pivot on `sbp_rechecked` inside `try`, catch `ValueError`, and print it. The lecture's "Find the Pair that Stops a Pivot" snippet explains the error. It prints `ValueError: Index contains duplicate entries, cannot reshape`.

## Check your work

- Click **Restart**, then **Run All**.
- The last cell prints your score and what to fix, using the latest checks from the course repository: the same checks GitHub runs on each push.
- Commit `assignment.ipynb` and the `output/` files, then push.

What each check looks for: [CHECKS.md](CHECKS.md)
