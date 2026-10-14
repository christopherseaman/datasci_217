# Assignment 06: Clinic Data Wrangling Pipeline (Midterm)

## Overview

Profile five raw clinic tables, audit and clean them, recode gender as a category, stack the two site visit exports, reshape the lab results, join everything into one analysis table, and pivot blood pressure to one row per patient. You save audits, summaries, and small extracts in `output/`; the full cleaned tables stay in the notebook.

`generate_data.py` writes the five files into `data/`, two as Parquet and three as CSV: a synthetic follow-up study of adults with high blood pressure and type 2 diabetes, with no real patients. The registry holds about 30,000 patients with about 57,000 visits, so every count comes from code. Two clinics, North Clinic (`N01`) and South Valley Clinic (`S02`), each export a visit file. `sites.csv` lists three site codes, `N01`, `S02`, and `W03` (a site with no patients yet), and some registry rows carry `E04`, a code `sites.csv` does not list.

| File | One row is | Columns |
| --- | --- | --- |
| `patients.parquet` | One registered patient | `patient_id`; `gender`; `birth_year` (`9999` means not recorded); `site_code` |
| `sites.csv` | One site record | `site_code`; `site_name`; `record_status` (`current` or `retired`) |
| `visits_north.csv` | One clinic visit at the north site | `patient_id`; `visit`; `visit_date` (`YYYY-MM-DD`); `visit_time` (`HH:MM`); `sbp_mmhg` (systolic blood pressure in mmHg; `999` means not measured); `weight_kg` |
| `visits_south.csv` | One clinic visit at the south site | The same columns, except the weight column is named `wt_kg` |
| `a1c_wide.parquet` | One patient's central lab results | `patient_id`; `a1c_baseline` and `a1c_week12` (hemoglobin A1c in %, `float64`; missing means no result) |

Every `patients.parquet` column is text, as staff typed it; the A1c columns are numbers. The CSV tables are read with every column as text.

The coded columns must hold these values once cleaned:

| Column | Code | Meaning |
| --- | --- | --- |
| `gender` | `F` | Female |
| `gender` | `M` | Male |
| `gender` | `X` | Nonbinary |
| `gender` | missing | Empty or `unknown`: not recorded |
| `site_code` | A code from `sites.csv`, such as `N01` | Uppercase, with no spaces or hyphens |
| `visit` | `baseline` | The enrollment visit |
| `visit` | `week12` | The follow-up visit 12 weeks later |

Expect other spellings of these codes, such as `woman` or `wk12`, stray spaces, mixed letter case, repeated rows, and entries that break these rules. Leave the generated files as they are: the first code cell checks each one against `data_manifest.json` and stops if a table has changed.

The two extracts (Tasks 4.3 and 5.2) hold the rows of 16 sample patients, `SAMPLE_IDS` in the first code cell: `P00001`, `P00004`, `P00088`, `P00260`, `P00446`, `P00487`, `P00631`, `P00851`, `P02354`, `P03836`, `P05763`, `P07510`, `P14287`, `P16486`, `P17390`, `P30001`.

## Setup

1. Fork the assignment repository on GitHub and clone your fork as in Lecture 01: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open the cloned folder itself, not a folder above it.
2. In the integrated terminal, create the environment, activate it, and install the packages `pyproject.toml` and `uv.lock` list. Do not run `uv init`.

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```

    - Expect: `uv sync` lists `+ ipykernel==6.29.5`, `+ numpy==2.3.3`, `+ pandas==3.0.5`, and `+ pyarrow==25.0.0`.
3. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`.
4. Run the first code cell. When `data/` is missing, it runs `generate_data.py` (`python3 generate_data.py` in the terminal does the same) to write the five files.
    - Expect: the Python, NumPy, and pandas versions, `Wrote 5 data files to .../data` on the first run, then `Verified 5 data files`.

The first cell supplies `DATA_DIR`, `OUTPUT_DIR`, a path for each output file, `SAMPLE_IDS`, and `raw`, a dictionary of the raw tables (`raw["patients"]`, `raw["visits_south"]`, ...). The Parquet files are read with `pd.read_parquet(path)`; the CSV files with `pd.read_csv(path, dtype="string", keep_default_na=False)`, so blanks stay empty text. Function names in the scaffold are suggestions. Fill in the "Cumulative midterm checkpoint" cell, below the first code cell, at any time.

## Files

```text
assignment/
├── assignment.ipynb            # notebook scaffold: complete every TODO, then run it top to bottom
├── generate_data.py            # supplied: writes the five data files into data/
├── data_manifest.json          # supplied: each data file's row count, columns, and sha256 checksum
├── data/                       # made by generate_data.py; never edit it (Git ignores it)
├── pyproject.toml, uv.lock     # supplied: the packages and exact versions `uv sync` installs
├── .python-version             # supplied: Python 3.13
├── CHECKS.md                   # supplied: what earns the points
├── HINTS.md                    # supplied: how to find the cause when a task goes wrong
├── .gitignore                  # supplied: keeps data/ and .venv/ out of Git
└── output/
    ├── file_inventory.txt      # you make in Task 1.1, in the terminal
    ├── table_profile.csv       # you make in Task 1.2
    ├── issue_audit.csv         # you make in Task 2.3
    ├── patients_summary.csv    # you make in Task 3.1
    ├── gender_categories.csv   # you make in Task 3.1
    ├── visits_summary.csv      # you make in Task 3.2
    ├── decision_log.csv        # you make in Task 3.3
    ├── a1c_summary.csv         # you make in Task 4.1
    ├── merge_audit.csv         # you make in Task 4.2
    ├── sbp_wide_sample.csv     # you make in Task 4.3
    └── analysis_sample.csv     # you make in Task 5.2
```

Each task saves its file as soon as it is made. An error stops Run All, and a cell that never runs writes no file: fix the cause and run again.

## Task 1: Inventory the files

### 1.1 Count lines from the terminal

In the terminal, from the assignment directory, run one command that counts the lines of every CSV file in `data/` and saves what it prints; Task 1.2 counts the Parquet rows.

```bash
wc -l data/*.csv > output/file_inventory.txt
```

Then paste the command into the `%%bash` cell under Task 1.1 in the notebook.

> **Checkpoint: `output/file_inventory.txt`**
> 4 lines: one count and path per CSV file, in alphabetical order starting with `data/sites.csv`, then a `total` line. Each count includes the header line.

### 1.2 Profile each table with a function

Write a function that takes a table's name and the table and returns a dictionary with four keys: `table` (the file name, such as `patients.parquet`), `rows` (data rows, not the header), `columns`, and `exact_duplicate_rows` (rows identical to an earlier row: `table.duplicated().sum()`). Call it for each table in `raw`, build a DataFrame from the results, and save it with `to_csv(path, index=False)`.

- Self-check: each CSV file's `rows` is one less than its `wc -l` count in Task 1.1.

> **Checkpoint: `output/table_profile.csv`**
> 6 lines: the header `table,rows,columns,exact_duplicate_rows` and one row per data file, in the order of the Overview table.

## Task 2: Contract, stack, and audit

### 2.1 Write the data contract

In the notebook's Task 2.1 Markdown cell, state for each of the five tables its grain (what one row represents) and its key, single or composite. Then name the cardinality you expect for each Task 4.4 join: visits to patients, patients to sites, and visits to A1c results.

### 2.2 Stack the two visit files

Build `visits_raw`, one table of every raw visit row from both sites:

1. Rename `wt_kg` to `weight_kg` in a copy of `raw["visits_south"]` with `rename(columns={...})`.
2. Add a `source_file` column to each table holding its file name, `visits_north.csv` or `visits_south.csv`, before stacking.
3. Stack them with `pd.concat([...], ignore_index=True)`, north first.

- Expect: 57321 rows and 7 columns, with no column of only missing values.

### 2.3 Audit the raw tables

Count each issue below in the raw tables, every row before any duplicate is removed; the `visits` issues count `visits_raw`. Save the counts to `output/issue_audit.csv` with these labels, in this order:

```text
issue,count
patients: gender values that are not a code,<count>
patients: empty gender,<count>
patients: birth_year sentinel 9999,<count>
patients: site codes needing normalization,<count>
sites: rows with a repeated site_code,<count>
visits: visit labels that are not baseline or week12,<count>
visits: sbp sentinel 999,<count>
visits: sbp outside 70 through 250,<count>
visits: date parse failures,<count>
visits: rows in repeated patient_id and visit pairs,<count>
```

| Issue | What it counts |
| --- | --- |
| `gender values that are not a code` | Rows whose `gender` is not empty after `.str.strip()` and, as written, is not exactly `F`, `M`, or `X`: `~isin([...])` |
| `empty gender` | Rows whose `gender` is empty after `.str.strip()` |
| `birth_year sentinel 9999` | Rows whose `birth_year` is `9999` |
| `site codes needing normalization` | Rows whose `site_code` changes when uppercased with every space and hyphen removed (Task 3.1's rule) |
| `rows with a repeated site_code` | Every `sites.csv` row whose `site_code` also appears on another row: `duplicated(subset=["site_code"], keep=False)` |
| `visit labels that are not baseline or week12` | Rows whose `visit`, as written, is not exactly `baseline` or `week12` |
| `sbp sentinel 999` | Rows whose `sbp_mmhg` is `999` |
| `sbp outside 70 through 250` | Rows whose `sbp_mmhg` is a number other than `999` below 70 or above 250 |
| `date parse failures` | Rows whose `visit_date` is not exact `YYYY-MM-DD` text for a date on the calendar (Lecture 05's "Accept only exact dates") |
| `rows in repeated patient_id and visit pairs` | Every row whose `patient_id` and `visit`, mapped to `baseline` or `week12` as in Task 3.2 step 2, together also appear on another row: `duplicated(subset=[...], keep=False)` |

A worked example on five made-up registry rows, not the real data:

```text
patient_id,gender,birth_year,site_code
P1,F,1950,N01
P2,woman,9999,n-01
P3, ,1962,S02
P3, ,1962,S02
P4,M ,1971,S02
```

Here `woman` and `M ` (a trailing space) are not codes as written, so the first count is 2; `P3`'s blank gender is counted on both of its rows, before the repeat is removed, so the second is also 2; one `9999`; and one site code, `n-01`, needs normalization. The first four lines of this example's audit file would read:

```text
issue,count
patients: gender values that are not a code,2
patients: empty gender,2
patients: birth_year sentinel 9999,1
patients: site codes needing normalization,1
```

Self-checks before you save:

- Every count is a whole number from 0 up to its table's `rows` in `table_profile.csv`.
- A row with `999` is counted as a sentinel, never also as outside 70 through 250.
- A count from `keep=False` includes the first copy, so `sites: rows with a repeated site_code` and the repeated-pair count are 0 or at least 2.
- A count of 0 is a finding too; keep its row.
- Auditing must not change `raw`: the notebook's snapshot comparison stays `True`.

> **Checkpoint: `output/issue_audit.csv`**
> 11 lines: the header `issue,count` and the 10 issues in the order above, each with a whole-number count.

## Task 3: Clean

Build each cleaned table from a `copy(deep=True)` and never change `raw` or `visits_raw`. Leave missing values missing. A summary file has two columns, `metric` and `value`, one row per metric; build it as a DataFrame from two lists:

```python
summary = pd.DataFrame({
    "metric": ["patients", "birth_year missing"],
    "value": [len(patients_clean), patients_clean["birth_year"].isna().sum()],
})
```

### 3.1 Clean the patients

Build `patients_clean` from `raw["patients"]`:

1. Remove exact duplicate rows, keeping the first.
2. `gender`: list the spellings with `value_counts()`, then strip and lowercase them and map each to its code from the data dictionary with a dictionary and `Series.map()` (Lecture 05), such as `{"female": "F", "woman": "F", ...}`. A spelling the dictionary leaves out, such as `unknown` or an empty value, becomes missing. Convert the result with `astype("category")`.
3. `birth_year`: the `9999` sentinel becomes missing; dtype `Int64`.
4. `site_code`: uppercase, then remove every hyphen and space with `.str.replace("-", "")` and `.str.replace(" ", "")`.
5. Renumber the rows with `reset_index(drop=True)`.

- Expect: `patients_clean["gender"].cat.categories` shows `['F', 'M', 'X']`.

Save the patients summary to `output/patients_summary.csv` with these metrics, in this order:

```text
metric,value
patients,<count>
birth_year missing,<count>
birth_year min,<year>
birth_year max,<year>
site_code values,<count>
patients at N01,<count>
patients at S02,<count>
patients at E04,<count>
```

| Metric | What it holds |
| --- | --- |
| `patients` | `len(patients_clean)` |
| `birth_year missing` | `patients_clean["birth_year"].isna().sum()` |
| `birth_year min`, `birth_year max` | `.min()` and `.max()` of `birth_year` |
| `site_code values` | `patients_clean["site_code"].nunique()` |
| `patients at N01`, `S02`, `E04` | The rows whose `site_code` is that code, such as `(patients_clean["site_code"] == "N01").sum()` |

Self-checks: `patients` equals the `patients.parquet` rows minus its `exact_duplicate_rows` in `table_profile.csv`; `patients at N01`, `S02`, and `E04` add up to `patients`; `birth_year min` and `max` are birth years of adults in 2026, never `9999`.

Then save the category counts, missing included, with each category's code:

```python
gender_counts = patients_clean["gender"].value_counts(dropna=False).reset_index()
gender_counts["code"] = gender_counts["gender"].cat.codes
gender_counts.to_csv(GENDER_CATEGORIES_PATH, index=False)
```

- Self-check: the four counts add up to `patients`.

> **Checkpoint: `output/patients_summary.csv`**
> 9 lines: the header `metric,value` and the 8 metrics in the order above.

> **Checkpoint: `output/gender_categories.csv`**
> 5 lines: the header `gender,count,code`, one row per code with its category code (`F` 0, `M` 1, `X` 2), and a row with an empty `gender` and code `-1` for the missing values.

### 3.2 Clean the visits

Build `visits_clean` from `visits_raw`:

1. Remove exact duplicate rows, keeping the first.
2. `visit`: list the spellings with `value_counts()`, then strip and lowercase them and map each to `baseline` or `week12` with a dictionary and `Series.map()`, as for `gender`.
3. `sbp_mmhg`: convert with `pd.to_numeric(..., errors="coerce")`; keep values from 70 through 250, and make everything else, `999` included, missing; dtype `Int64`.
4. `weight_kg`: convert to numbers (`float64`); an empty value stays missing.
5. `visit_date`: keep a value only when its text is exact `YYYY-MM-DD` and the date exists on the calendar; anything else becomes missing.
6. Show the rows whose `patient_id` and `visit` pair still repeats, with `duplicated(subset=[...], keep=False)`. A repeat here is a recheck: keep the later reading with `sort_values("visit_time")`, then `drop_duplicates(subset=["patient_id", "visit"], keep="last")`.
7. Sort by `patient_id` and `visit`, renumber with `reset_index(drop=True)`, and keep these columns in this order: `patient_id`, `visit`, `visit_date`, `sbp_mmhg`, `weight_kg`, `source_file`.

- Expect: step 6 shows 240 rows: 120 `patient_id` and `visit` pairs, each read twice at different times.
- Expect: `visits_clean["visit"].value_counts(dropna=False)` shows only `baseline` and `week12`, with no missing.

Save the visits summary to `output/visits_summary.csv` with these metrics, in this order:

```text
metric,value
visits,<count>
baseline visits,<count>
week12 visits,<count>
visit_date missing,<count>
visit_date min,<date>
visit_date max,<date>
sbp_mmhg missing,<count>
sbp_mmhg min,<mmHg>
sbp_mmhg max,<mmHg>
weight_kg missing,<count>
weight_kg mean,<kg>
visits from visits_north.csv,<count>
visits from visits_south.csv,<count>
```

| Metric | What it holds |
| --- | --- |
| `visits` | `len(visits_clean)` |
| `baseline visits`, `week12 visits` | The rows with that `visit` label |
| `visit_date missing`, `sbp_mmhg missing`, `weight_kg missing` | `.isna().sum()` of that column |
| `visit_date min`, `visit_date max` | `.min()` and `.max()` of `visit_date` |
| `sbp_mmhg min`, `sbp_mmhg max` | `.min()` and `.max()` of `sbp_mmhg` |
| `weight_kg mean` | `round(visits_clean["weight_kg"].mean(), 2)` |
| `visits from visits_north.csv`, `visits from visits_south.csv` | The rows with that `source_file` |

Self-checks: `baseline visits` plus `week12 visits` equals `visits`, and so do the two `visits from` counts; `visits` is below the 57321 rows of `visits_raw`; `sbp_mmhg min` is at least 70 and `max` at most 250; both dates fall in 2026 on clinic weekdays (Monday to Friday); each missing count is a small share of `visits`; `weight_kg mean` is an adult weight in kg.

> **Checkpoint: `output/visits_summary.csv`**
> 14 lines: the header `metric,value` and the 13 metrics in the order above.

### 3.3 Record the decisions

Build `decision_table`, a DataFrame with the columns `table`, `decision`, and `reason`: one row for each decision below, in this order, with `table` and `decision` copied exactly. Write each `reason` yourself: one sentence, on one line, about these files. Save it to `output/decision_log.csv` with `to_csv(path, index=False)`.

| `table` | `decision` |
| --- | --- |
| `patients` | `remove the exact duplicate row, keeping the first` |
| `patients` | `map gender to F, M, or X with a dictionary and store it as a category` |
| `patients` | `uppercase site_code and remove its spaces and hyphens` |
| `patients` | `convert the birth_year sentinel 9999 to missing` |
| `visits` | `rename wt_kg to weight_kg before stacking` |
| `visits` | `convert sbp_mmhg outside 70 through 250, including 999, to missing` |
| `visits` | `keep only exact YYYY-MM-DD calendar dates` |
| `visits` | `keep the later reading of a repeated patient_id and visit pair` |
| `sites` | `keep only current site records before joining` |

> **Checkpoint: `output/decision_log.csv`**
> 10 lines: the header `table,decision,reason` and the nine decisions in the order above.

Commit your work so far with a message that says what it holds, such as `Clean patients and visits`.

## Task 4: Reshape and join

### 4.1 Melt the A1c results to long form

Turn `raw["a1c_wide"]` into `a1c_long`, one row per patient and visit:

1. `melt()` with `id_vars=["patient_id"]`, `value_vars=["a1c_baseline", "a1c_week12"]`, `var_name="visit"`, and `value_name="a1c_pct"`.
2. Map the visit labels to the ones `visits_clean` uses: `{"a1c_baseline": "baseline", "a1c_week12": "week12"}`.
3. Check that `a1c_pct` is `float64`, as Parquet stored it, then drop the rows with no result with `dropna(subset=["a1c_pct"])` and renumber with `reset_index(drop=True)`.

Save a per-visit summary of `a1c_long` to `output/a1c_summary.csv`: the columns `visit`, `results` (the rows with that label), and `a1c_pct_mean` (the mean of their `a1c_pct`, rounded to two decimals), one row for `baseline`, then one for `week12`. For one label, `(a1c_long["visit"] == "baseline").sum()` counts the rows and `a1c_long.loc[a1c_long["visit"] == "baseline", "a1c_pct"].mean()` takes the mean.

Self-checks: the two `results` add up to `len(a1c_long)`, which is below twice the `a1c_wide.parquet` rows; each mean is between 5 and 14 %; the `week12` mean is the lower one.

> **Checkpoint: `output/a1c_summary.csv`**
> 3 lines: the header `visit,results,a1c_pct_mean`, then the `baseline` and `week12` rows.

### 4.2 Check the keys before joining

In the Task 4.2 cells:

1. Merge `patients_clean` with `sites` (`raw["sites"]`, supplied in the cell) on `site_code` with `how="left"` and `validate="many_to_one"` inside `try`/`except pd.errors.MergeError`, and print the error.
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

A worked example with made-up keys, not the real data: three clean visits and three A1c results.

```text
visits_clean            a1c_long
patient_id,visit        patient_id,visit,a1c_pct
P1,baseline             P1,baseline,7.9
P1,week12               P2,baseline,8.4
P2,baseline             P9,baseline,6.8
```

The outer merge pairs `P1 baseline` and `P2 baseline` (`both`, 2), keeps `P1 week12` with no result (`left_only`, 1), and keeps `P9`'s result with no visit (`right_only`, 1). `both` plus `left_only` is 3, the rows of `visits_clean`; `both` plus `right_only` is 3, the rows of `a1c_long`.

Self-checks before you save:

- `visits with an A1c result` plus `visits without an A1c result` equals `visits` in your `visits_summary.csv`.
- `visits with an A1c result` plus `A1c results without a visit` equals the two `results` in your `a1c_summary.csv` added together.
- `patients at N01` plus `patients at S02` plus `registry patients whose site is not in the site list` equals `patients` in your `patients_summary.csv`.
- `visits whose patient is not in the registry` is at most `visits`, and `registry patients with no visit` at most `patients`; both are small shares.
- `site list rows removed by the current-record rule` is `sites.csv`'s 5 rows minus the rows of `sites_current`.

> **Checkpoint: `output/merge_audit.csv`**
> 8 lines: the header `check,count` and the 7 checks in the order above, each with a whole-number count.

### 4.3 Pivot blood pressure to one row per patient

Build `sbp_wide` from `visits_clean` with `pivot(index="patient_id", columns="visit", values="sbp_mmhg")`, then `reset_index()` and set `columns.name = None`. Add `sbp_change`, `week12` minus `baseline`; it is missing when either reading is missing.

Save the rows for the sample patients with `sbp_wide[sbp_wide["patient_id"].isin(SAMPLE_IDS)]` to `output/sbp_wide_sample.csv`, with `index=False`.

- Expect: `sbp_wide` has one row per patient with a visit and the columns `patient_id`, `baseline`, `week12`, `sbp_change`; a sample patient with no visit has no row.

> **Checkpoint: `output/sbp_wide_sample.csv`**
> 16 lines: the header `patient_id,baseline,week12,sbp_change` and one row per sample patient with a visit.

### 4.4 Join the analysis table

Build `analysis`, one row per clean visit, with three left merges onto `visits_clean`, each with the `validate=` that matches your Task 2.1 contract:

1. `patients_clean` on `patient_id`.
2. `sites_current` on `site_code`.
3. `a1c_long` on `["patient_id", "visit"]`.

Keep these columns in this order, with these dtypes: `analysis = analysis.astype({"patient_id": "string", "sbp_mmhg": "Int64", ...})`.

| Column | dtype |
| --- | --- |
| `patient_id`, `visit` | `string` |
| `visit_date` | `datetime64[us]` |
| `sbp_mmhg` | `Int64` |
| `weight_kg`, `a1c_pct` | `float64` |
| `gender` | `string` |
| `birth_year` | `Int64` |
| `site_code`, `site_name`, `source_file` | `string` |

- Expect: as many rows as `visits_clean` after each merge; a visit whose patient or site is unknown keeps its row with missing patient or site values.

> **A `MergeError` here means an earlier step left a problem.**
> `analysis_sample.csv` is not written until it is fixed. Fix the cause upstream; do not remove `validate=`. [HINTS.md](HINTS.md) traces each error.

## Task 5: Validate, save, and read back

### 5.1 Validate before saving

In the Task 5.1 cell, collect named `True`/`False` checks in a Series, and display it. Check at least that:

- `analysis` has as many rows as `visits_clean`;
- no `patient_id` and `visit` pair repeats in `analysis`;
- every `sbp_mmhg` that is present in `analysis` is from 70 through 250;
- every `site_name` that is present is a current site name;
- `sbp_wide["patient_id"]` is unique;
- every raw table still equals its snapshot.

Then `assert checks.all(), checks[~checks]` directly before the cell that saves the analysis extract.

> **An `AssertionError` here means an earlier step left a problem.**
> `analysis_sample.csv` is not written until every check is `True`. Print `checks[~checks]`, then trace each failure in [HINTS.md](HINTS.md). Never loosen or skip the assert.

### 5.2 Save the analysis extract

Keep the `analysis` rows for the sample patients and renumber them: `analysis_sample = analysis[analysis["patient_id"].isin(SAMPLE_IDS)].reset_index(drop=True)`. Save it to `output/analysis_sample.csv` with `to_csv(path, index=False)`.

- Self-check: it has as many rows as `visits_clean["patient_id"].isin(SAMPLE_IDS).sum()`.

> **Checkpoint: `output/analysis_sample.csv`**
> 29 lines: the header `patient_id,visit,visit_date,sbp_mmhg,weight_kg,a1c_pct,gender,birth_year,site_code,site_name,source_file` and one row per visit of a sample patient.

### 5.3 Read the file back

Read `output/analysis_sample.csv` back with `dtype={...}` for every column but `visit_date` and `parse_dates=["visit_date"]`, and print `round_trip.equals(analysis_sample)`, which should be `True`.

## Check your work

1. Click **Restart**, then **Run All**.
    - Expect: no errors, the validation cell shows only `True`, and the read-back prints `True`.
2. In the terminal, compare each file in `output/` with this table: `wc -l output/*` prints each file's line count, and `head -n 1 output/FILE` its first line.

| File | Made in | First line | Lines |
| --- | --- | --- | --- |
| `output/file_inventory.txt` | Task 1.1 | `<count> data/sites.csv` | 4 |
| `output/table_profile.csv` | Task 1.2 | `table,rows,columns,exact_duplicate_rows` | 6 |
| `output/issue_audit.csv` | Task 2.3 | `issue,count` | 11 |
| `output/patients_summary.csv` | Task 3.1 | `metric,value` | 9 |
| `output/gender_categories.csv` | Task 3.1 | `gender,count,code` | 5 |
| `output/visits_summary.csv` | Task 3.2 | `metric,value` | 14 |
| `output/decision_log.csv` | Task 3.3 | `table,decision,reason` | 10 |
| `output/a1c_summary.csv` | Task 4.1 | `visit,results,a1c_pct_mean` | 3 |
| `output/merge_audit.csv` | Task 4.2 | `check,count` | 8 |
| `output/sbp_wide_sample.csv` | Task 4.3 | `patient_id,baseline,week12,sbp_change` | 16 |
| `output/analysis_sample.csv` | Task 5.2 | `patient_id,visit,visit_date,sbp_mmhg,weight_kg,a1c_pct,gender,birth_year,site_code,site_name,source_file` | 29 |

3. Commit `assignment.ipynb` (with its outputs) and the eleven `output/` files, then push. The course grades your fork's `main` branch after the deadline.

What the points reward and how files are read: [CHECKS.md](CHECKS.md)
