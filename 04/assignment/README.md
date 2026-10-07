# Assignment 04: Blood Pressure Follow-Up

## Overview

Load a hypertension program's blood pressure export with `read_csv` options, summarize and count it, build a ranked follow-up list, and compare clinic with home readings, saving each result for the checks.

Both files are synthetic and come from one clinic network's hypertension program, which measures each patient's **systolic blood pressure** (SBP, the top number, in mmHg) at a baseline visit, at week 4, and at week 8.

- `data/bp_followup.csv` is the clinic system's export: one row per patient with their `clinic`, `age` in years, the three readings, and a free-text `coordinator_note`. Like many real exports, it separates fields with semicolons, puts a row of units under the header, and writes `-999` for a reading that was not taken.
- `data/home_bp.csv` holds week-8 readings that some patients took at home with a cuff the program lent them, in another order. P115 is in the home file only.

```text
patient_id;clinic;age;sbp_baseline;sbp_week4;sbp_week8;coordinator_note
id;text;years;mmHg;mmHg;mmHg;text
P101;North;58;152;146;138;Started lisinopril at baseline
```

## Setup

1. Fork the assignment repository on GitHub and clone your fork as in Lecture 01: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open the cloned folder itself, not a folder above it.
2. Open **Terminal → New Terminal** (Ctrl+Shift+backtick, also Control on Mac). In a native terminal or WSL Ubuntu, `cd` into the assignment folder first. Run `ls data`.
    - Expect: `bp_followup.csv  home_bp.csv`
3. This clone is a new repository, so set your Git identity before your first commit, with Lecture 02's two lines: `git config user.name "..."` and `git config user.email "..."`, using your name and GitHub noreply email.
4. Create the environment, activate it, and install the packages `pyproject.toml` and `uv.lock` list. Do not run `uv init`: the project files already exist.

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```

    - Expect: `uv venv` prints `Using CPython 3.13.x`, and `uv sync` lists `+ pandas==3.0.5` and `+ pyarrow==25.0.0` among the packages it installs.
5. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`. If VS Code offers to install the **Jupyter** extension, accept.
6. Run the notebook's first code cell.
    - Expect: `NumPy: 2.3.3`, `pandas: 3.0.5`, `pyarrow: 25.0.0`, and `data files found: True`.

> **Windows:** work in the **WSL: Ubuntu** window from Lecture 01's setup. Git Bash also works: activate with `source .venv/Scripts/activate` and type `python` wherever these instructions say `python3`. In PowerShell, activate with `.\.venv\Scripts\Activate.ps1` and type `python` as well.

If something goes wrong:

- **`uv venv` asks `Do you want to replace it? [y/n]`:** `.venv` already exists. Answer `n`; the `error: Failed to create virtual environment` that follows is harmless, and the next two lines work. If you answered `y`, run `uv sync` again.
- **`ModuleNotFoundError` in the first cell:** the kernel is not this project's `.venv`. Click the kernel name at the top right and choose it. If it already shows `.venv`, run `uv sync` in the terminal with the environment active, then rerun the cell.
- **`data files found: False`:** the notebook cannot see `data/`. Run `%pwd` and `%ls` in new cells: `%pwd` should end in your assignment folder, and `%ls` should list `data/` beside `assignment.ipynb`. If not, open the cloned folder itself in VS Code.

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/
│   ├── bp_followup.csv     # supplied clinic export; keep it exactly as handed out
│   └── home_bp.csv         # supplied home readings; keep it exactly as handed out
├── pyproject.toml          # supplied: the project's packages, numpy, pandas, pyarrow, and ipykernel
├── uv.lock                 # supplied: the exact versions `uv sync` installs
├── .python-version         # supplied: tells uv to use Python 3.13
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── bp_loaded.csv               # you generate in Task 2
    ├── visit_summary.csv           # you generate in Task 3.1
    ├── clinic_counts.csv           # you generate in Task 3.2
    ├── followup_priority.csv       # you generate in Task 4
    ├── followup_priority.parquet   # you generate in Task 4
    └── white_coat_gap.csv          # you generate in Task 5
```

## Task 1: Put the cells in running order

### 1.1 Reorder the cells

The first Task 1 code cell uses `vials_on_hand`, which the second cell defines, so **Run All** stops with `NameError: name 'vials_on_hand' is not defined`.

1. Move the whole `vials_on_hand` cell above the `doses_available` cell: drag it by the bar to the left of its code, or click into it, press `Esc`, then `Alt+Up` (`Option+Up` on Mac). Do not copy the definition into another cell.
2. Click **Restart**, then **Run All**.
    - Expect:

    ```text
    vials_on_hand: 3
    doses_available: 30
    ```

### 1.2 Explain the repair

Replace the TODO in the Markdown cell under 1.2 with a few sentences that explain:

- the difference between the order the cells appear in and the order the kernel ran them;
- why the dependent cell can print a result while it sits above the cell it needs;
- why output saved under a cell does not prove the notebook works now; and
- what you changed, and how Restart and Run All shows that it worked.

Task 1 writes no file, but **Run All** stops at the `NameError` until 1.1 is done.

## Task 2: Load the blood pressure export

### 2.1 Look at the raw file

1. Run the Task 2.1 cell as it is. It prints the file, then reads it with no options.
    - Expect: `shape: (15, 1)`.
2. Find the four problems the output shows:
    - the whole header is one column, because the fields are separated by `;`, not `,`;
    - the first record is the units row (`id;text;years;mmHg;...`);
    - `coordinator_note` is free text the analysis does not use; and
    - `-999` stands for a missing reading, so it must not count as a blood pressure.

### 2.2 Read with options and save

1. In the Task 2.2 cell, replace `None` with one `pd.read_csv()` call that fixes all four problems and makes `patient_id` the index, using Lecture 04's `pd.read_csv()` reference card:
    - `sep=";"`;
    - `skiprows=[1]`, which skips line 1, the units row (the header line is 0);
    - `usecols=["patient_id", "clinic", "age", "sbp_baseline", "sbp_week4", "sbp_week8"]`;
    - `na_values=["-999"]`; and
    - `index_col="patient_id"`.
    - Expect: `shape: (14, 5)`; `clinic` is `str`, `age` is `int64`, and the three readings are `float64`, since a missing value is a float. `P104`'s `sbp_week4` shows `NaN`.
2. Display `P104`'s row with `bp.loc["P104"]` and the first three rows with `bp.iloc[0:3]`.
    - Expect: P104's `sbp_week4` is `NaN`; the three rows are P101, P102, and P103.
3. Save with `bp.to_csv(LOADED_PATH)`. Keep the index: it holds the patient IDs.

> **Checkpoint: `output/bp_loaded.csv`**
> The header line `patient_id,clinic,age,sbp_baseline,sbp_week4,sbp_week8`, then 14 patients, P101 to P114. Three cells are blank: P104's `sbp_week4`, P109's `sbp_week8`, and P111's `sbp_baseline`.

## Task 3: Summarize the readings

### 3.1 Summarize each visit

The cell selects `readings = bp[["sbp_baseline", "sbp_week4", "sbp_week8"]]`. Each reduction on it returns one value per visit, labeled by visit.

1. Line three reductions up as columns:

    ```python
    visit_summary = pd.DataFrame({
        "mean": readings.mean(),
        "median": readings.median(),
        "count": readings.count(),
    })
    ```

2. Name the index with `visit_summary.index.name = "visit"`.
3. Set `highest_baseline` to the patient with the highest baseline, with `idxmax()`, and `baseline_week8_r` to the correlation of `sbp_baseline` with `sbp_week8`, with `corr()`.
    - Expect: `count` is 13, not 14, because each visit has one missing reading and every reduction skips it.

    ```text
                        mean  median  count
    visit                                  
    sbp_baseline  152.153846   152.0     13
    sbp_week4     145.000000   145.0     13
    sbp_week8     140.461538   138.0     13
    highest baseline: P108
    baseline vs week 8 r: 0.9764184782678083
    ```

4. Save with `visit_summary.to_csv(SUMMARY_PATH)`, keeping the index.

> **Checkpoint: `output/visit_summary.csv`**
> The header line `visit,mean,median,count`, then the three rows above.

### 3.2 Count patients per clinic

1. Set `clinic_counts` to `bp["clinic"].value_counts()`.
2. Replace `None` in the `clinics` line with the number of distinct clinics, from `nunique()`.
    - Expect: `clinics: 4`.
3. Save with `clinic_counts.to_csv(COUNTS_PATH)`.

> **Checkpoint: `output/clinic_counts.csv`**
> The header line `clinic,count`, then `North,5`, `East,4`, `West,3`, and `South,2`, most common first.

## Task 4: Build the follow-up list

The nurses at the North and East clinics call every patient who started at stage 2 hypertension, a baseline SBP of 140 mmHg or more, largest week-8 drop first.

### 4.1 Select the program patients

1. Build the mask `in_program`: `True` where `bp["clinic"].isin(["North", "East"])` and `bp["sbp_baseline"] >= 140` are both true. Join them with `&`, with each comparison in parentheses. P111 has no baseline, so its comparison is `False`.
2. Set `followup` to the `in_program` rows of `bp` with `.loc`, then drop the `age` column with `.drop(columns=["age"])`.
    - Expect: `program patients: 7`: P101, P104, P106, P107, P110, P113, and P114.

### 4.2 Add the derived columns

1. `sbp_mean`: the mean of each patient's three readings, across the row: `followup[["sbp_baseline", "sbp_week4", "sbp_week8"]].mean(axis="columns")`.
    - Expect: P104 missed week 4, so its mean is of two readings, 157.0.
2. `change_week8`: `sbp_week8` minus `sbp_baseline`. A negative change means the pressure fell.
    - Expect: P110's change is -15.0.

### 4.3 Sort, rank, and save

1. Sort `followup` by `change_week8` from the most negative (the largest drop) up, breaking ties by patient: `by=["change_week8", "patient_id"]` (`sort_values()` accepts the index name). Assign the result back to `followup`.
2. Add `improvement_rank` with `followup["change_week8"].rank(method="min")`: the largest drop is 1, and tied patients share the better place.
3. Save with `followup.to_csv(FOLLOWUP_PATH)` and `followup.to_parquet(PARQUET_PATH)`. Keep the index in both (no `index=False`).
4. Read the Parquet file back into `back` with `pd.read_parquet()`.
    - Expect: `Parquet round trip matches: True`.

> **Checkpoint: `output/followup_priority.csv`**
> The header line `patient_id,clinic,sbp_baseline,sbp_week4,sbp_week8,sbp_mean,change_week8,improvement_rank`, then seven patients in this order of `patient_id`: P110, P101, P104, P106, P113, P114, P107. The first line is `P110,East,160.0,152.0,145.0,152.33333333333334,-15.0,1.0`. P101 and P104 tie at -14 and share rank 2; P106 and P113 tie at -13 and share rank 4.

> **Checkpoint: `output/followup_priority.parquet`**
> The same table, with the same eight columns, `patient_id` included.

## Task 5: Compare clinic and home readings

A clinic reading that runs higher than the patient's home reading is the **white-coat effect**.

1. Read `HOME_PATH` into `home` with `index_col="patient_id"`.
2. Set `white_coat_gap` to `bp["sbp_week8"] - home["home_sbp_week8"]`. pandas pairs the readings by patient ID, not by position, and the result has one row for every patient in either table, with `NaN` where one side has no reading.
3. Name the Series with `white_coat_gap.name = "white_coat_gap_mmhg"`.
    - Expect: 15 values and `patients with both readings: 5`: P101 (7.0), P104 (4.0), P106 (3.0), P110 (5.0), and P113 (7.0). P109 (no clinic week-8 reading) and P115 (home file only) are `NaN`.
4. Save with `white_coat_gap.to_csv(GAP_PATH)`, keeping the index.

> **Checkpoint: `output/white_coat_gap.csv`**
> The header line `patient_id,white_coat_gap_mmhg`, then 15 patients, P101 to P115, with a number for the five above and a blank for the other ten.

## Check your work

1. Click **Restart**, then **Run All**.
    - Expect: the last cell prints `Fresh-run check passed`, or names the task to fix.
2. With the environment active, run the checks from the assignment folder:

    ```bash
    python3 check_assignment.py
    ```

    - Expect: one `PASS` or `FIX` line per check with its points; a `FIX` says what to fix on the line beneath it, and `(same fix as above)` marks checks that need the same fix. Before Task 2, for example, the loaded-table checks report:

    ```text
    [FIX ]  0/4  bp loaded: patient_id column
             output/bp_loaded.csv is missing; run the Task 2.2 cell to write it with bp.to_csv(LOADED_PATH), then commit it.
    [FIX ]  0/4  bp loaded: units row skipped  (same fix as above)
    ```

3. Fix what `Left to fix` lists under the score, rerun the notebook and the checks, and repeat until a clean run ends with:

    ```text
    [PASS] 10/10 white-coat gap: gaps matched by patient

    Score: 100/100
    All checks passed.
    ```

The checks read only the six files in `output/`, never your notebook, so any way of producing correct files counts. GitHub runs the same checks, with the course's current copy, on every push; that run is the one that counts.

How the files are read:

- Each check is scored on its own, so one mistake costs only the checks it gets wrong.
- A later file is judged against your own earlier one: if `-999` stayed in `bp_loaded.csv`, only the `-999 read as missing` check loses points, and the values computed from it still count.
- Spacing, line endings, quoting, letter case, column order, and number format (`13`, `13.0`, `13.00`) never cost points; a mean may be rounded to one decimal.
- A leading column of row numbers, which `to_csv()` writes when the index holds only row numbers, is ignored.

### Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/bp_loaded.csv` | It has a `patient_id` column holding the saved index. | bp loaded: patient_id column | 4 |
| `output/bp_loaded.csv` | The units row is not in it. | bp loaded: units row skipped | 4 |
| `output/bp_loaded.csv` | It has no `coordinator_note` column. | bp loaded: coordinator_note left out | 4 |
| `output/bp_loaded.csv` | It holds each of P101 to P114 once. | bp loaded: all 14 patients once | 4 |
| `output/bp_loaded.csv` | The three `-999` readings are blank. | bp loaded: -999 read as missing | 4 |
| `output/bp_loaded.csv` | Every clinic, age, and reading matches the export. | bp loaded: clinic, age, and readings | 4 |
| `output/visit_summary.csv` | It has one row for each of the three visits. | visit summary: one row per visit | 3 |
| `output/visit_summary.csv` | Its `mean` column matches the readings. | visit summary: mean | 5 |
| `output/visit_summary.csv` | Its `median` column matches the readings. | visit summary: median | 5 |
| `output/visit_summary.csv` | Its `count` column counts the readings present. | visit summary: count | 5 |
| `output/clinic_counts.csv` | It has each clinic's number of patients. | clinic counts: patients per clinic | 4 |
| `output/clinic_counts.csv` | The most common clinic comes first. | clinic counts: most common first | 2 |
| `output/followup_priority.csv` | It has a `patient_id` column holding the saved index. | follow-up list: patient_id column | 3 |
| `output/followup_priority.csv` | It holds the seven program patients once each. | follow-up list: program patients | 6 |
| `output/followup_priority.csv` | Its derived columns are named `sbp_mean`, `change_week8`, and `improvement_rank`. | follow-up list: derived column names | 2 |
| `output/followup_priority.csv` | It has no `age` column. | follow-up list: age dropped | 2 |
| `output/followup_priority.csv` | Each `sbp_mean` is the mean of the patient's readings. | follow-up list: sbp_mean values | 4 |
| `output/followup_priority.csv` | Each `change_week8` is week 8 minus baseline. | follow-up list: change_week8 values | 4 |
| `output/followup_priority.csv` | Each `improvement_rank` is the change's rank with `method="min"`. | follow-up list: improvement_rank values | 2 |
| `output/followup_priority.csv` | Its patients run from the largest drop to the smallest. | follow-up list: largest drop first | 3 |
| `output/followup_priority.csv` | Patients with the same change are in `patient_id` order. | follow-up list: ties in patient_id order | 2 |
| `output/followup_priority.parquet` | It is a Parquet file. | follow-up Parquet: Parquet file | 4 |
| `output/followup_priority.parquet` | It has the follow-up list's columns, `patient_id` included. | follow-up Parquet: same columns | 4 |
| `output/white_coat_gap.csv` | It has one row for each of P101 to P115. | white-coat gap: one row per patient in either table | 6 |
| `output/white_coat_gap.csv` | Each gap is that patient's clinic minus home reading, blank where one is missing. | white-coat gap: gaps matched by patient | 10 |

Extra files are ignored.

## Submit

1. After **Restart** and **Run All**, save the notebook with its outputs: they hold only synthetic data and show your results.
2. In VS Code Source Control, check the notebook's diff, then stage `assignment.ipynb` and the six files in `output/`. `.venv/` stays out; `.gitignore` already lists it.
3. Commit with `Complete Assignment 04 notebook` and select **Sync Changes**.
4. On GitHub, confirm the notebook and the six output files on `main`, and check the Actions run's score. In a new fork, enable Actions once if GitHub prompts you.

If a VS Code control these steps need is unavailable, record its message and contact the instructor.
