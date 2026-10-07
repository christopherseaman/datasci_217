# Assignment 08: Grouped Summaries of Clinic Visits

## Overview

Group a clinic visit log by clinic and by clinic and visit type, compare each visit with its clinic's mean, and pivot the mean waits, saving each result for the checks.

`data/clinic_visits.csv` is a synthetic visit log from three neighborhood clinics, one row per visit.

- `visit_id`, `clinic`, and `patient_id` (some patients come more than once).
- `visit_type`: `New`, `Follow-up`, or `Telehealth`.
- `wait_min`: minutes from check-in to being seen.
- `satisfaction`: the 1 to 5 survey score, blank for the three visits whose survey never came back.
- A fourth clinic, Excelsior, opened this month and has no visits yet; Sunset had no telehealth visits.

```text
visit_id,clinic,patient_id,visit_type,wait_min,satisfaction
V001,Mission,P101,New,18,4
V002,Mission,P102,Follow-up,12,5
V003,Mission,P101,Follow-up,9,
```

## Setup

1. Fork the assignment repository on GitHub and clone your fork as in Lecture 01: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open the cloned folder itself, not a folder above it.
2. In the integrated terminal, create the environment, activate it, and install the packages `pyproject.toml` and `uv.lock` list. Do not run `uv init`: the project files already exist.

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```

    - Expect: `uv venv` prints `Using CPython 3.13.x`, and `uv sync` lists `+ pandas==3.0.5` among the packages it installs.
3. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`.
4. Run the notebook's first two code cells.
    - Expect: `pandas: 3.0.5` and `data folder found: True`, then `visits: (15, 6)`.

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/
│   └── clinic_visits.csv   # supplied visit log; keep it exactly as handed out
├── pyproject.toml          # supplied: the project's packages, numpy, pandas, and ipykernel
├── uv.lock                 # supplied: the exact versions `uv sync` installs
├── .python-version         # supplied: tells uv to use Python 3.13
├── CHECKS.md               # supplied: what each check looks for
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

- Click **Restart**, then **Run All**.
- The last cell prints your score and what to fix, using the latest checks from the course repository: the same checks GitHub runs on each push.
- Commit `assignment.ipynb` and the `output/` files, then push.

What each check looks for: [CHECKS.md](CHECKS.md)
