# Assignment 08: Grouped Summaries of Clinic Visits

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/
│   └── clinic_visits.csv   # supplied visit log; keep it exactly as handed out
├── requirements.txt        # supplied: numpy, pandas, and ipykernel
├── .python-version         # supplied: tells uv to use Python 3.13
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── clinic_counts.csv              # you generate in Task 1.2
    ├── clinic_summary.csv             # you generate in Task 2.1
    ├── visits_with_context.csv        # you generate in Task 2.2
    ├── clinic_visit_type_summary.csv  # you generate in Task 2.3
    └── mean_wait_pivot.csv            # you generate in Task 3.1
```

## The data

`data/clinic_visits.csv` is a synthetic visit log from three neighborhood clinics, one row per visit: `visit_id`, the `clinic`, the `patient_id` (some patients come more than once), the `visit_type` (`New`, `Follow-up`, or `Telehealth`), `wait_min`, the minutes from check-in to being seen, and `satisfaction`, the patient's 1 to 5 survey score, blank for the three visits whose survey never came back. A fourth clinic, Excelsior, opened this month and has no visits yet, and Sunset had no telehealth visits.

```text
visit_id,clinic,patient_id,visit_type,wait_min,satisfaction
V001,Mission,P101,New,18,4
V002,Mission,P102,Follow-up,12,5
V003,Mission,P101,Follow-up,9,
```

## Setup

Fork the assignment repository on GitHub and clone your fork the way Lecture 01 did: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open it. Then open **Terminal → New Terminal** in VS Code at the assignment directory (Ctrl+Shift+backtick, also Control on Mac). If you use a native terminal or WSL Ubuntu instead, `cd` into the assignment directory first. Run `ls data` and expect `clinic_visits.csv`. This clone is a new repository, so before your first commit run Lecture 02's two `git config user.name "..."` and `git config user.email "..."` lines in this terminal, with your name and GitHub noreply email.

Create the project environment, activate it, and install the supplied requirements, as in Lecture 03:

```bash
uv venv --seed --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` includes **ipykernel**, the package that lets a notebook run on this environment's Python (Lecture 04), so this one install is all the notebook needs. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`. If VS Code offers to install the **Jupyter** extension, accept.

> **Windows:** work in the **WSL: Ubuntu** window from Lecture 01's setup. In PowerShell instead, activate with `.\.venv\Scripts\Activate.ps1`.

Run the notebook's first two code cells. The first prints the pandas version and `data folder found: True`; `False` means the notebook is not running from the assignment directory, so open the folder itself in VS Code, not a folder above it. The second reads the visit log and prints `visits: (15, 6)`.

## Task 1: Count each clinic's visits

The visit log has one row per visit, and every answer in Tasks 1 and 2.1 has one row per clinic: grouping changes the grain, as Lecture 08's "The Split-Apply-Combine Paradigm" section shows.

### 1.1 Put the clinics in reporting order

This step saves nothing. The clinic manager lists clinics as Mission, Sunset, Bayview, Excelsior, which is not alphabetical, and wants to see that Excelsior has had no visits. Lecture 08's "A Reporting Order and an Unused Clinic" snippet does the same. In the Task 1.1 cell:

1. Replace `visits["clinic"]` with an ordered categorical: `pd.Categorical(visits["clinic"], categories=CLINIC_ORDER, ordered=True)`. It prints `clinic dtype: category`.
2. Count the visits at each clinic with `groupby("clinic", observed=False).size()` and name the result `visits_per_clinic`. It lists Mission 6, Sunset 5, Bayview 4, and Excelsior 0, in that order.

The rest of the assignment groups with `observed=True`, which lists only the three clinics that have visits, still in this order.

### 1.2 Count visits, satisfaction scores, and patients

In the Task 1.2 cell:

1. Group `visits` by `clinic` with `as_index=False` and `observed=True`, and build `clinic_counts` with named aggregation, as Lecture 08's "Count and Summarize Each Clinic" snippet does:
    - `visit_count=("visit_id", "size")`: every visit at the clinic.
    - `satisfaction_count=("satisfaction", "count")`: the visits with a survey score; `count` skips the blanks.
    - `patient_count=("patient_id", "nunique")`: each patient once, however many times they came.

    It prints `rows: 3`.
2. Save `clinic_counts` to `CLINIC_COUNTS_PATH` with `index=False`.

> **Checkpoint: `output/clinic_counts.csv`**
> The header line `clinic,visit_count,satisfaction_count,patient_count`, then three rows: Mission, Sunset, and Bayview. Mission reads `Mission,6,5,4`: six visits, five with a survey score, from four patients.

## Task 2: Summarize, compare, and use two keys

### 2.1 Add wait totals and means

In the Task 2.1 cell:

1. Build `clinic_summary` the way you built `clinic_counts`, with the same three counts and two more named aggregations: `total_wait_min`, the `"sum"` of `wait_min`, and `mean_wait_min`, its `"mean"`. It prints `rows: 3`.
2. Save `clinic_summary` to `CLINIC_SUMMARY_PATH` with `index=False`. Leave the means unrounded.

> **Checkpoint: `output/clinic_summary.csv`**
> The header line `clinic,visit_count,satisfaction_count,patient_count,total_wait_min,mean_wait_min`, then three rows. Sunset reads `Sunset,5,3,3,126,25.2`.

### 2.2 Compare each visit with its clinic

`transform` gives every visit its own clinic's mean, so the result keeps one row per visit, as Lecture 08's "Compare Each Visit with Its Clinic" snippet shows. In the Task 2.2 cell:

1. Make `visits_with_context`, a `.copy()` of `visits`, so `visits` itself stays unchanged.
2. Add the column `clinic_mean_wait`: `visits.groupby("clinic", observed=True)["wait_min"].transform("mean")`.
3. Add the column `wait_vs_clinic`: `wait_min` minus `clinic_mean_wait`. A visit that waited longer than its clinic's mean gets a positive number.

    It prints `rows: 15` and `visits unchanged: True`.
4. Save `visits_with_context` to `CONTEXT_PATH` with `index=False`.

> **Checkpoint: `output/visits_with_context.csv`**
> The header line `visit_id,clinic,patient_id,visit_type,wait_min,satisfaction,clinic_mean_wait,wait_vs_clinic`, then 15 rows, V001 to V015, one per visit. V007 waited 32 minutes at Sunset, whose mean is 25.2, so its `wait_vs_clinic` is 6.8.

### 2.3 Summarize each clinic and visit type

Grouping by two keys gives one row per observed pair, as Lecture 08's "Grouping by Two Keys" section shows. In the Task 2.3 cell:

1. Group `visits` by `["clinic", "visit_type"]` with `as_index=False` and `observed=True`, and build `clinic_type_summary` with two named aggregations: `visit_count=("visit_id", "size")` and `mean_wait_min=("wait_min", "mean")`. It prints `rows: 8`: Sunset had no telehealth visits, so it has two pairs instead of three.
2. Save `clinic_type_summary` to `CLINIC_TYPE_PATH` with `index=False`.

> **Checkpoint: `output/clinic_visit_type_summary.csv`**
> The header line `clinic,visit_type,visit_count,mean_wait_min`, then eight rows, one per clinic and visit type pair that has visits. Sunset's new-patient row reads `Sunset,New,2,36.5`.

## Task 3: Pivot the mean waits

### 3.1 Build the pivot table and check it

A pivot table lays Task 2.3's means out as a grid, one row per clinic and one column per visit type, as Lecture 08's "A Pivot Table and Its GroupBy Twin" snippet shows. In the Task 3.1 cell:

1. Build `mean_wait_pivot` with `pd.pivot_table(visits, values="wait_min", index="clinic", columns="visit_type", aggfunc="mean", observed=True)`.
2. Build its twin, `groupby_twin`: `visits.groupby(["clinic", "visit_type"], observed=True)["wait_min"].mean().unstack()`. It prints `pivot matches its groupby twin: True` and `empty cells: 1`. That empty cell is Sunset's `Telehealth`: no telehealth visit happened there, so there is no wait to average.
3. Save `mean_wait_pivot` to `PIVOT_PATH` with its index, so leave out `index=False`: the clinic names are the index, and saving it writes them as the first column. Leave the empty cell empty. `fill_value=0` would report a zero-minute wait that never happened.

> **Checkpoint: `output/mean_wait_pivot.csv`**
> The header line `clinic,Follow-up,New,Telehealth`, then three rows: Mission, Sunset, and Bayview. Sunset's `Telehealth` cell is empty.

### 3.2 Count the visits in each pair

This step saves nothing. In the Task 3.2 cell, count the visits in each clinic and visit type pair with `pd.crosstab(visits["clinic"], visits["visit_type"])` and name the result `visit_counts`. Sunset's `Telehealth` cell shows `0`: zero visits is a real count, while the pivot of means leaves the same cell empty.

## Check your work

Click **Restart**, then **Run All**. The last cell prints `Fresh-run check passed`, or names the task to fix. Then, with the environment active, run the checks from the assignment directory:

```bash
python check_assignment.py
```

`check_assignment.py` runs the same checks GitHub runs. They read only the five CSV files in `output/` and compare them with values computed from the supplied visit log. They never run or read your notebook, so any way of producing correct files counts.

Each check prints `PASS` or `FIX` and the points it earned, and a `FIX` says what to fix on the line beneath it. When the next checks need the same fix, such as a missing file, they say `(same fix as above)`. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/6  clinic counts: columns
         output/clinic_counts.csv is missing; run the Task 1.2 cell to write it, then commit it.
```

Below the score, `Left to fix` lists the checks still failing and the points they are worth. Fix what they name, rerun the notebook and then the checks, and repeat until every check passes. A clean run ends with:

```text
[PASS]  4/4  mean wait pivot: empty cell stays empty

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only that check's points.
- Spacing, line endings, quoting, column order, and row order never cost points.
- Numbers are compared as numbers, so `6`, `6.0`, and `6.00` are the same value. A mean may keep every digit or be rounded to one or two decimals.
- Clinic names, visit types, IDs, and column names are compared in any letter case.
- An empty cell may be written empty or as `NaN`.
- A leading column of row numbers, which `to_csv()` writes when `index=False` is left out, is ignored.
- A row for Excelsior showing zero visits, which `observed=False` writes, is accepted but not needed; so are Task 2.3's other pairs with zero visits, and an all-empty Excelsior row in the pivot.

Every push also runs GitHub Actions, which downloads the course's current copy of the checks and reruns them on the files you committed and pushed. That run is what counts, and a check corrected after handout reaches you there on your next push.

### Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/clinic_counts.csv` | Its columns are the four in the Task 1.2 header line. | clinic counts: columns | 6 |
| `output/clinic_counts.csv` | It holds Mission, Sunset, and Bayview, once each. | clinic counts: one row per clinic | 6 |
| `output/clinic_counts.csv` | Each `visit_count` is the clinic's number of visits. | clinic counts: visit_count values | 4 |
| `output/clinic_counts.csv` | Each `satisfaction_count` is the clinic's number of visits with a survey score. | clinic counts: satisfaction_count values | 4 |
| `output/clinic_counts.csv` | Each `patient_count` is the clinic's number of distinct patients. | clinic counts: patient_count values | 4 |
| `output/clinic_summary.csv` | Its columns are the six in the Task 2.1 header line. | clinic summary: columns | 4 |
| `output/clinic_summary.csv` | It holds Mission, Sunset, and Bayview, once each. | clinic summary: one row per clinic | 4 |
| `output/clinic_summary.csv` | Its three counts match Task 1.2's. | clinic summary: count values | 4 |
| `output/clinic_summary.csv` | Each `total_wait_min` is the sum of the clinic's waits. | clinic summary: total_wait_min values | 4 |
| `output/clinic_summary.csv` | Each `mean_wait_min` is the mean of the clinic's waits. | clinic summary: mean_wait_min values | 4 |
| `output/visits_with_context.csv` | Its columns are the eight in the Task 2.2 header line. | visit context: columns | 4 |
| `output/visits_with_context.csv` | It holds V001 to V015, once each. | visit context: one row per visit | 4 |
| `output/visits_with_context.csv` | Each visit's `clinic`, `patient_id`, `visit_type`, `wait_min`, and `satisfaction` match `data/clinic_visits.csv`. | visit context: original visit values | 4 |
| `output/visits_with_context.csv` | Each `clinic_mean_wait` is the mean wait at that visit's clinic. | visit context: clinic_mean_wait values | 4 |
| `output/visits_with_context.csv` | Each `wait_vs_clinic` is `wait_min` minus `clinic_mean_wait`. | visit context: wait_vs_clinic values | 4 |
| `output/clinic_visit_type_summary.csv` | Its columns are the four in the Task 2.3 header line. | clinic and visit type: columns | 4 |
| `output/clinic_visit_type_summary.csv` | It holds each of the eight clinic and visit type pairs with visits once. | clinic and visit type: one row per pair | 5 |
| `output/clinic_visit_type_summary.csv` | Each `visit_count` is the pair's number of visits. | clinic and visit type: visit_count values | 4 |
| `output/clinic_visit_type_summary.csv` | Each `mean_wait_min` is the mean of the pair's waits. | clinic and visit type: mean_wait_min values | 5 |
| `output/mean_wait_pivot.csv` | Its columns are `clinic`, `Follow-up`, `New`, and `Telehealth`. | mean wait pivot: columns | 4 |
| `output/mean_wait_pivot.csv` | It holds Mission, Sunset, and Bayview, once each. | mean wait pivot: one row per clinic | 4 |
| `output/mean_wait_pivot.csv` | Each filled cell is the mean wait for that clinic and visit type. | mean wait pivot: mean waits | 6 |
| `output/mean_wait_pivot.csv` | Sunset's row is there and its `Telehealth` cell is empty, not 0. | mean wait pivot: empty cell stays empty | 4 |

Extra files are ignored.

## Submit

Before you commit a notebook, follow Lecture 04's "Before You Commit a Notebook": click **Clear All Outputs**, then save. In VS Code Source Control, stage `assignment.ipynb` and the five files in `output/`. Commit with `Complete Assignment 08 notebook` and select **Sync Changes**. Keep `.venv/` out of the commit; `.gitignore` already lists it.

Confirm the notebook and the five CSV files on `main` in the repository browser. GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork. If a run cannot download the course's current checks, it grades with the copy in your repository and says so in its log. If your local run and the GitHub run ever disagree, the GitHub run counts, because it uses the course's current checks. If a required VS Code control is unavailable, record its message and contact the instructor.
