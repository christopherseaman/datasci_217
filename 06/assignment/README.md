# Assignment 06: Clinic Data Wrangling Pipeline (Midterm)

## Overview

Profile five raw clinic tables, audit and clean them, stack the two site visit exports, reshape the lab results, join everything into one analysis table, and pivot blood pressure to one row per patient. Each result is saved as a file in `output/`; the notebook cells and decision reasons are read by human review.

The five files in `data/` are a synthetic follow-up study of adults with high blood pressure and type 2 diabetes at two clinic sites; no real patients are in them.

| File | One row is | Columns |
| --- | --- | --- |
| `patients.csv` | One registered patient | `patient_id`; `sex` (`F` or `M`, may be empty); `birth_year` (`9999` means not recorded); `site_code` |
| `sites.csv` | One site record | `site_code`; `site_name`; `record_status` (`current` or `retired`) |
| `visits_north.csv` | One clinic visit at the north site | `patient_id`; `visit` (`baseline` or `week12`); `visit_date` (`YYYY-MM-DD`); `visit_time` (`HH:MM`); `sbp_mmhg` (systolic blood pressure in mmHg; `999` means not measured); `weight_kg` |
| `visits_south.csv` | One clinic visit at the south site | The same columns, except the weight column is named `wt_kg` |
| `a1c_wide.csv` | One patient's central lab results | `patient_id`; `a1c_baseline` and `a1c_week12` (hemoglobin A1c in %; empty means no result) |

The values arrive as staff typed them, so expect stray spaces, mixed letter case, repeated rows, and entries that break these rules. Leave the files exactly as they ship: the notebook's first code cell compares each checksum with `data/manifest.json` and stops if a file has changed.

## Setup

1. Fork the assignment repository on GitHub and clone your fork as in Lecture 01: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open the cloned folder itself, not a folder above it.
2. In the integrated terminal, create the environment, activate it, and install the packages `pyproject.toml` and `uv.lock` list. Do not run `uv init`: the project files already exist.

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```

    - Expect: `uv sync` lists `+ numpy==2.3.3` and `+ pandas==3.0.5`.
3. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`.
4. Run the first code cell.
    - Expect: the Python, NumPy, and pandas versions, then `Verified 5 data files`.

The first cell supplies `DATA_DIR`, `OUTPUT_DIR`, a path for each output file, and `raw`, a dictionary holding each raw table with every column read as text (`raw["patients"]`, `raw["visits_south"]`, ...). Function names in the scaffold are suggestions: grading reads the files in `output/` and the cells named in CHECKS.md. The "Cumulative midterm checkpoint" cell, just below the first code cell, asks for an evidence map; fill it in whenever you like.

## Files

```text
assignment/
├── assignment.ipynb            # notebook scaffold: complete every TODO, then run it top to bottom
├── data/                       # supplied raw tables (the five files above); never edit them
├── data/manifest.json          # supplied: each data file's row count, columns, and sha256 checksum
├── pyproject.toml, uv.lock     # supplied: the packages and exact versions `uv sync` installs
├── .python-version             # supplied: Python 3.13
├── CHECKS.md                   # supplied: what earns the points
├── .gitattributes, .gitignore  # supplied: keep the data byte for byte, keep .venv/ out of Git
└── output/
    ├── file_inventory.txt      # you make in Task 1.1, in the terminal
    ├── table_profile.csv       # you make in Task 1.2
    ├── issue_audit.csv         # you make in Task 2.3
    ├── patients_clean.csv      # you make in Task 3.1
    ├── visits_clean.csv        # you make in Task 3.2
    ├── a1c_long.csv            # you make in Task 4.1
    ├── merge_audit.csv         # you make in Task 4.2
    ├── analysis_visits.csv     # you make in Task 5.2
    ├── sbp_wide.csv            # you make in Task 5.2
    └── decision_log.csv        # you make in Task 5.2
```

## Task 1: Inventory the files

### 1.1 Count lines from the terminal

In the terminal, from the assignment directory, run one command that counts the lines of every CSV file in `data/` and saves what it prints:

```bash
wc -l data/*.csv > output/file_inventory.txt
```

Then paste the command into the notebook's Task 1.1 cell.

> **Checkpoint: `output/file_inventory.txt`**
> 6 lines: one count and path per data file, in alphabetical order starting with `data/a1c_wide.csv`, then a `total` line. Each count includes the header line.

### 1.2 Profile each table with a function

Write a function that takes a table's name and the table and returns a dictionary with four keys: `table` (the file name, such as `patients.csv`), `rows` (data rows, not the header), `columns`, and `exact_duplicate_rows` (rows identical to an earlier row: `table.duplicated().sum()`). Call it for each table in `raw`, build a DataFrame from the results, and save it with `to_csv(path, index=False)`.

> **Checkpoint: `output/table_profile.csv`**
> 6 lines: the header `table,rows,columns,exact_duplicate_rows` and one row per data file, in the order of the Overview table.

## Task 2: Contract, stack, and audit

### 2.1 Write the data contract

In the notebook's Task 2.1 Markdown cell, state for each of the five tables its grain (what one row represents) and its key: a single column or a composite key (Lectures 05 and 06). Then name the cardinality you expect for each join in Task 4.3: visits to patients, patients to sites, and visits to A1c results. Human review reads this cell.

### 2.2 Stack the two visit files

Build `visits_raw`, one table of every raw visit row from both sites:

1. Rename `wt_kg` to `weight_kg` in a copy of `raw["visits_south"]` with `rename(columns={...})` (Lecture 05).
2. Add a `source_file` column to each table holding its file name, `visits_north.csv` or `visits_south.csv`, before stacking (Lecture 06's provenance column).
3. Stack them with `pd.concat([...], ignore_index=True)`, north first.

- Expect: 48 rows and 7 columns, with no column of only missing values (one would mean a column name still differs).

### 2.3 Audit the raw tables

Count each issue below in the raw tables, every row before any duplicate is removed; the `visits` issues count `visits_raw`. Save the counts to `output/issue_audit.csv` with these labels, in this order:

```text
issue,count
patients: sex values needing normalization,<count>
patients: empty sex,<count>
patients: birth_year sentinel 9999,<count>
patients: site codes needing normalization,<count>
sites: rows with a repeated site_code,<count>
visits: visit labels needing normalization,<count>
visits: sbp sentinel 999,<count>
visits: sbp outside 70 through 250,<count>
visits: date parse failures,<count>
visits: rows in repeated patient_id and visit pairs,<count>
```

| Issue | What it counts |
| --- | --- |
| `sex values needing normalization` | Rows whose `sex` is not empty after `.str.strip()` and changes under `.str.strip().str.upper()` |
| `empty sex` | Rows whose `sex` is empty after `.str.strip()` |
| `birth_year sentinel 9999` | Rows whose `birth_year` is `9999` |
| `site codes needing normalization` | Rows whose `site_code` changes under `.str.strip().str.upper()` |
| `rows with a repeated site_code` | Every `sites.csv` row whose `site_code` also appears on another row: `duplicated(subset=["site_code"], keep=False)` |
| `visit labels needing normalization` | Rows whose `visit` changes under `.str.strip().str.lower()` |
| `sbp sentinel 999` | Rows whose `sbp_mmhg` is `999` |
| `sbp outside 70 through 250` | Rows whose `sbp_mmhg` is a number other than `999` below 70 or above 250 |
| `date parse failures` | Rows whose `visit_date` is not exact `YYYY-MM-DD` text for a date on the calendar (Lecture 05's "Accept only exact dates") |
| `rows in repeated patient_id and visit pairs` | Every row whose `patient_id` and stripped, lowercased `visit` together also appear on another row: `duplicated(subset=[...], keep=False)` |

A count of 0 is a finding too; keep its row. Auditing must not change `raw`: the notebook's snapshot comparison stays `True`.

> **Checkpoint: `output/issue_audit.csv`**
> 11 lines: the header `issue,count` and the 10 issues in the order above, each with a whole-number count.

## Task 3: Clean

Build each cleaned table from a `copy(deep=True)` and never change `raw` or `visits_raw`. Leave missing values missing.

### 3.1 Clean the patients

Build `patients_clean` from `raw["patients"]`:

- Remove exact duplicate rows, keeping the first.
- `sex`: strip and uppercase; an empty value becomes missing.
- `birth_year`: the `9999` sentinel becomes missing; dtype `Int64`.
- `site_code`: strip and uppercase.
- Renumber the rows with `reset_index(drop=True)`.

Save it with `index=False`, columns in this order: `patient_id`, `sex`, `birth_year`, `site_code`.

> **Checkpoint: `output/patients_clean.csv`**
> 25 lines: the header `patient_id,sex,birth_year,site_code` and one row per patient.

### 3.2 Clean the visits

Build `visits_clean` from `visits_raw`:

1. Remove exact duplicate rows, keeping the first.
2. `visit`: strip and lowercase.
3. `sbp_mmhg`: convert with `pd.to_numeric(..., errors="coerce")`; keep values from 70 through 250, and make everything else, `999` included, missing; dtype `Int64`.
4. `weight_kg`: convert to numbers (`float64`); an empty value stays missing.
5. `visit_date`: keep a value only when its text is exact `YYYY-MM-DD` and the date exists on the calendar; anything else becomes missing.
6. Show the rows whose `patient_id` and `visit` pair still repeats, with `duplicated(subset=[...], keep=False)`. A repeat here is a recheck: keep the later reading with `sort_values("visit_time")`, then `drop_duplicates(subset=["patient_id", "visit"], keep="last")` (Lecture 06's "When pivot() Finds a Repeated Pair").
7. Sort by `patient_id` and `visit`, renumber with `reset_index(drop=True)`, and keep these columns in this order: `patient_id`, `visit`, `visit_date`, `sbp_mmhg`, `weight_kg`, `source_file`.

- Expect: step 6 shows exactly two rows, the same patient's baseline at two times.

> **Checkpoint: `output/visits_clean.csv`**
> 47 lines: the header `patient_id,visit,visit_date,sbp_mmhg,weight_kg,source_file` and one row per visit.

### 3.3 Record the decisions

Build `decision_table`, a DataFrame with the columns `table`, `decision`, and `reason`: one row for each decision below, in this order, with `table` and `decision` copied exactly. Write each `reason` yourself: one sentence, on one line, about these files. Human review reads each reason.

| `table` | `decision` |
| --- | --- |
| `patients` | `remove the exact duplicate row, keeping the first` |
| `patients` | `strip and uppercase sex and site_code; an empty sex becomes missing` |
| `patients` | `convert the birth_year sentinel 9999 to missing` |
| `visits` | `rename wt_kg to weight_kg before stacking` |
| `visits` | `convert sbp_mmhg outside 70 through 250, including 999, to missing` |
| `visits` | `keep only exact YYYY-MM-DD calendar dates` |
| `visits` | `keep the later reading of a repeated patient_id and visit pair` |
| `sites` | `keep only current site records before joining` |

Commit your work so far with a message that says what it holds, such as `Clean patients and visits` (Lecture 02).

## Task 4: Reshape and join

### 4.1 Melt the A1c results to long form

Turn `raw["a1c_wide"]` into `a1c_long`, one row per patient and visit:

1. `melt()` with `id_vars=["patient_id"]`, `value_vars=["a1c_baseline", "a1c_week12"]`, `var_name="visit"`, and `value_name="a1c_pct"`.
2. Map the visit labels to the ones `visits_clean` uses: `{"a1c_baseline": "baseline", "a1c_week12": "week12"}`.
3. Convert `a1c_pct` to numbers (`float64`), drop the rows with no result with `dropna(subset=["a1c_pct"])`, and renumber with `reset_index(drop=True)`.

Save it with `index=False`, columns `patient_id`, `visit`, `a1c_pct`.

> **Checkpoint: `output/a1c_long.csv`**
> 47 lines: the header `patient_id,visit,a1c_pct` and one row per result.

### 4.2 Check the keys before joining

In the Task 4.2 cells:

1. Merge `patients_clean` with `sites` on `site_code` with `how="left"` and `validate="many_to_one"` inside `try`/`except pd.errors.MergeError`, and print the error.
2. Show the repeated site rows with `duplicated(subset=["site_code"], keep=False)`, then build `sites_current`: the rows whose `record_status` is `current`, with the columns `site_code` and `site_name`.
3. Merge `visits_clean` and `a1c_long` on `["patient_id", "visit"]` with `how="outer"` and `indicator=True`, and show `value_counts()` of `_merge`.

Save the counts below to `output/merge_audit.csv` with these labels, in this order:

```text
check,count
visits with an A1c result,<count>
visits without an A1c result,<count>
A1c results without a visit,<count>
visits whose patient is not in the registry,<count>
registry patients with no visit,<count>
registry patients whose site is not in the site list,<count>
site list rows removed by the current-record rule,<count>
```

| Check | What it counts |
| --- | --- |
| `visits with an A1c result` | Rows of the step 3 merge whose `_merge` is `both` |
| `visits without an A1c result` | Rows whose `_merge` is `left_only` |
| `A1c results without a visit` | Rows whose `_merge` is `right_only` |
| `visits whose patient is not in the registry` | `visits_clean` rows whose `patient_id` is not in `patients_clean` |
| `registry patients with no visit` | `patients_clean` rows whose `patient_id` is not in `visits_clean` |
| `registry patients whose site is not in the site list` | `patients_clean` rows whose `site_code` is not in `sites.csv` |
| `site list rows removed by the current-record rule` | `sites.csv` rows minus `sites_current` rows |

> **Checkpoint: `output/merge_audit.csv`**
> 8 lines: the header `check,count` and the 7 checks in the order above, each with a whole-number count.

### 4.3 Join the analysis table

Build `analysis`, one row per clean visit, with three left merges onto `visits_clean`, each with the `validate=` that matches your Task 2.1 contract:

1. `patients_clean` on `patient_id`.
2. `sites_current` on `site_code`.
3. `a1c_long` on `["patient_id", "visit"]`.

Keep these columns in this order, with these dtypes, so the read-back in Task 5.3 matches. `astype()` takes a dictionary of column to dtype, as `read_csv(dtype={...})` does in Lecture 05: `analysis = analysis.astype({"patient_id": "string", "sbp_mmhg": "Int64", ...})`.

| Column | dtype |
| --- | --- |
| `patient_id`, `visit` | `string` |
| `visit_date` | `datetime64[us]` |
| `sbp_mmhg` | `Int64` |
| `weight_kg`, `a1c_pct` | `float64` |
| `sex` | `string` |
| `birth_year` | `Int64` |
| `site_code`, `site_name`, `source_file` | `string` |

- Expect: as many rows as `visits_clean`; a visit whose patient or site is unknown keeps its row with missing patient or site values.

### 4.4 Pivot blood pressure to one row per patient

Build `sbp_wide` from `visits_clean` with `pivot(index="patient_id", columns="visit", values="sbp_mmhg")`, then `reset_index()` and set `columns.name = None` (Lecture 06). Add `sbp_change`, `week12` minus `baseline`; it is missing when either reading is missing.

- Expect: one row per patient with a visit and the columns `patient_id`, `baseline`, `week12`, `sbp_change`.

## Task 5: Validate, save, and read back

### 5.1 Validate before saving

In the Task 5.1 cell, collect named `True`/`False` checks in a Series, as Lecture 05's "Stop before saving a bad table" does, and display it. Check at least that:

- `analysis` has as many rows as `visits_clean`;
- no `patient_id` and `visit` pair repeats in `analysis`;
- every `sbp_mmhg` that is present in `analysis` is from 70 through 250;
- every `site_name` that is present is a current site name;
- `sbp_wide["patient_id"]` is unique;
- every raw table still equals its snapshot.

Then `assert checks.all(), checks[~checks]` directly before the cell that saves, so a failed check means no file is written.

### 5.2 Save the joined, pivoted, and decision tables

Save `analysis` to `output/analysis_visits.csv`, `sbp_wide` to `output/sbp_wide.csv`, and `decision_table` to `output/decision_log.csv`, each with `to_csv(path, index=False)`.

> **Checkpoint: `output/analysis_visits.csv`**
> 47 lines: the header `patient_id,visit,visit_date,sbp_mmhg,weight_kg,a1c_pct,sex,birth_year,site_code,site_name,source_file` and one row per visit.

> **Checkpoint: `output/sbp_wide.csv`**
> 25 lines: the header `patient_id,baseline,week12,sbp_change` and one row per patient with a visit.

> **Checkpoint: `output/decision_log.csv`**
> 9 lines: the header `table,decision,reason` and the eight decisions in the order of Task 3.3.

### 5.3 Read the files back

Read `output/analysis_visits.csv` back with `dtype={...}` for every column but `visit_date` and `parse_dates=["visit_date"]`, and print `round_trip.equals(analysis)`, which should be `True` (Lecture 05's "Read the saved file back").

## Check your work

1. Click **Restart**, then **Run All**.
    - Expect: no errors, the validation cell shows only `True`, and the read-back prints `True`.
2. Open each file in `output/` and compare it with this table. The line count is the number beside the last line with text; VS Code may number one empty line after it; do not count that.

| File | Made in | First line | Lines |
| --- | --- | --- | --- |
| `output/file_inventory.txt` | Task 1.1 | `<count> data/a1c_wide.csv` | 6 |
| `output/table_profile.csv` | Task 1.2 | `table,rows,columns,exact_duplicate_rows` | 6 |
| `output/issue_audit.csv` | Task 2.3 | `issue,count` | 11 |
| `output/patients_clean.csv` | Task 3.1 | `patient_id,sex,birth_year,site_code` | 25 |
| `output/visits_clean.csv` | Task 3.2 | `patient_id,visit,visit_date,sbp_mmhg,weight_kg,source_file` | 47 |
| `output/a1c_long.csv` | Task 4.1 | `patient_id,visit,a1c_pct` | 47 |
| `output/merge_audit.csv` | Task 4.2 | `check,count` | 8 |
| `output/analysis_visits.csv` | Task 5.2 | `patient_id,visit,visit_date,sbp_mmhg,weight_kg,a1c_pct,sex,birth_year,site_code,site_name,source_file` | 47 |
| `output/sbp_wide.csv` | Task 5.2 | `patient_id,baseline,week12,sbp_change` | 25 |
| `output/decision_log.csv` | Task 5.2 | `table,decision,reason` | 9 |

3. Commit `assignment.ipynb` (with its outputs) and the ten `output/` files, then push. Keep `.venv/` out of the commit; `.gitignore` already does. The course grades your fork's `main` branch after the deadline.

What the points reward and how files are read: [CHECKS.md](CHECKS.md)
