# Assignment 06: Merge, Stack, and Reshape Clinic Data

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
├── requirements.txt        # supplied: numpy, pandas, and ipykernel
├── .python-version         # supplied: tells uv to use Python 3.13
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

## The data

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

Fork the assignment repository on GitHub and clone your fork the way Lecture 01 did: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open it. Then open **Terminal → New Terminal** in VS Code at the assignment directory (Ctrl+Shift+backtick, also Control on Mac). If you use a native terminal or WSL Ubuntu instead, `cd` into the assignment directory first. Run `ls data` and expect the six CSV files above. This clone is a new repository, so before your first commit run Lecture 02's two `git config user.name "..."` and `git config user.email "..."` lines in this terminal, with your name and GitHub noreply email.

Create the project environment, activate it, and install the supplied requirements, as in Lecture 03:

```bash
uv venv --seed --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` includes **ipykernel**, the package that lets a notebook run on this environment's Python (Lecture 04), so this one install is all the notebook needs. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`. If VS Code offers to install the **Jupyter** extension, accept.

> **Windows:** work in the **WSL: Ubuntu** window from Lecture 01's setup. In PowerShell instead, activate with `.\.venv\Scripts\Activate.ps1`.

Run the notebook's first two code cells. The first prints the pandas version and `data folder found: True`; `False` means the notebook is not running from the assignment directory, so open the folder itself in VS Code, not a folder above it. The second reads the six files and prints their shapes, starting with `specimens: (7, 6)`.

## Task 1: Merge specimens with their clinics

Each specimen should get the name and region of the clinic that collected it: a left merge of `specimens` with the clinic records on `clinic_id`, many specimens to one clinic. Lecture 06's "Checking Merge Cardinality" section walks through the same steps.

### 1.1 Find the repeated clinic key

In the Task 1.1 cells:

1. Set `specimen_ids_unique` to `specimens["specimen_id"].is_unique`. It prints `specimen_id unique: True`.
2. Select every `clinics_history` row whose `clinic_id` appears more than once, with `duplicated(subset=["clinic_id"], keep=False)` as the mask, and name the result `repeated_clinic_rows`. It shows K01's two rows: `Bayview Annex` (retired) and `Bayview Clinic` (current).
3. In the next cell, try the merge with the contract written down, and catch the error it raises, as the lecture's "Catch a broken merge contract" snippet does:

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
2. Try the Task 3.2 pivot on `sbp_rechecked` inside `try`, catch `ValueError`, and print it, as the lecture's "Find the pair that stops pivot()" snippet does. It prints `ValueError: Index contains duplicate entries, cannot reshape`.

## Check your work

Click **Restart**, then **Run All**. The last cell prints `Fresh-run check passed`, or names the task to fix. Then, with the environment active, run the checks from the assignment directory:

```bash
python check_assignment.py
```

`check_assignment.py` runs the same checks GitHub runs. They read only the five CSV files in `output/` and compare them with the supplied data. They never run or read your notebook, so any way of producing correct files counts.

Each check prints `PASS` or `FIX` and the points it earned, and a `FIX` says what to fix on the line beneath it. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/8  merge audit: columns
         output/specimen_merge_audit.csv is missing; run the Task 1.2 cell to write it, then commit it.
```

Fix what it names, rerun the notebook and then the checks, and repeat until every check passes. A clean run ends with:

```text
[PASS]  4/4  SBP round trip: followup values

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only that check's points.
- Spacing, line endings, quoting, column order, and row order never cost points.
- Numbers are compared as numbers, so `5`, `5.0`, and `5.00` are the same value.
- IDs, labels, and column names are compared in any letter case.
- An empty cell may be written empty or as `NaN`.
- A leading column of row numbers, which `to_csv()` writes when `index=False` is left out, is ignored.
- A first column with no header that holds the IDs, which `to_csv()` writes for an index without a name, counts as the ID column.

Every push also runs GitHub Actions, which downloads the course's current copy of the checks and reruns them on the files you committed and pushed. That run is what counts, and a check corrected after handout reaches you there on your next push.

### Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/specimen_merge_audit.csv` | Its columns are the nine in the Task 1.2 header line; `record_status` may be there too. | merge audit: columns | 8 |
| `output/specimen_merge_audit.csv` | It holds SP101 to SP107, once each. | merge audit: one row per specimen | 8 |
| `output/specimen_merge_audit.csv` | Each specimen's own columns match `data/specimens.csv`. | merge audit: specimen values | 8 |
| `output/specimen_merge_audit.csv` | Each specimen has its clinic's current `clinic_name` and `region`, and SP106's are empty. | merge audit: current clinic names and regions | 8 |
| `output/specimen_merge_audit.csv` | `_merge` is `left_only` for SP106 and `both` for the other six. | merge audit: _merge indicator | 8 |
| `output/combined_specimens.csv` | Its columns are the seven in the Task 2.1 header line. | combined specimens: columns | 3 |
| `output/combined_specimens.csv` | It holds SP101 to SP107, once each. | combined specimens: one row per specimen | 4 |
| `output/combined_specimens.csv` | Each row's values match its batch file. | combined specimens: specimen values | 4 |
| `output/combined_specimens.csv` | `source_partition` is `batch_a` for SP101 to SP104 and `batch_b` for SP105 to SP107. | combined specimens: source_partition labels | 4 |
| `output/aligned_features.csv` | Its columns are `specimen_id`, `volume_ml`, and `transit_min`. | aligned features: columns | 3 |
| `output/aligned_features.csv` | It holds SP101 to SP104 and SP108, once each. | aligned features: one row per specimen | 4 |
| `output/aligned_features.csv` | `volume_ml` matches batch A, and SP108's is empty. | aligned features: volume_ml values | 4 |
| `output/aligned_features.csv` | `transit_min` matches `data/transit_times.csv`, and SP101's and SP104's are empty. | aligned features: transit_min values | 4 |
| `output/sbp_long.csv` | Its columns are `patient_id`, `visit`, and `sbp`. | SBP long: columns | 5 |
| `output/sbp_long.csv` | It holds each of the eight patient and visit pairs once. | SBP long: one row per patient and visit | 5 |
| `output/sbp_long.csv` | Each `sbp` matches that patient's reading at that visit in `data/sbp_wide.csv`. | SBP long: sbp values | 5 |
| `output/sbp_round_trip.csv` | Its columns are `patient_id`, `baseline`, and `followup`. | SBP round trip: columns | 3 |
| `output/sbp_round_trip.csv` | It holds P201 to P204, once each. | SBP round trip: one row per patient | 4 |
| `output/sbp_round_trip.csv` | Each `baseline` matches `data/sbp_wide.csv`. | SBP round trip: baseline values | 4 |
| `output/sbp_round_trip.csv` | Each `followup` matches `data/sbp_wide.csv`. | SBP round trip: followup values | 4 |

Extra files are ignored.

## Submit

Before you commit a notebook, follow Lecture 04's "Before You Commit a Notebook": click **Clear All Outputs**, then save. In VS Code Source Control, stage `assignment.ipynb` and the five files in `output/`. Commit with `Complete Assignment 06 notebook` and select **Sync Changes**. Keep `.venv/` out of the commit; `.gitignore` already lists it.

Confirm the notebook and the five CSV files on `main` in the repository browser. GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork. If a run cannot download the course's current checks, it grades with the copy in your repository and says so in its log. If your local run and the GitHub run ever disagree, the GitHub run counts, because it uses the course's current checks. If a required VS Code control is unavailable, record its message and contact the instructor.
