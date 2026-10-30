# Assignment 08: Grouped Summaries of an Imaging Network

## Overview

Summarize a year of imaging exams by site and modality, flag delayed STAT exams against their peers, build a CT radiation-dose review list, pull each site's slowest STAT reports, and pivot report times, saving each result for the checks.

`data/imaging_exams.parquet` is a synthetic export from a regional imaging network: 540,228 exams in 2025, one row per exam, at three hospitals (Downtown, Harbor, Northgate) and five outpatient imaging centers. A ninth site, Eastport, opens in 2026 and has no exams yet. Valley has no MRI scanner.

- `exam_id`, `patient_id` (most patients have several exams), `site`, and `exam_date`.
- `age`, `sex`, and `setting`: `ED`, `Inpatient`, or `Outpatient`.
- `priority`: `STAT` (scan now, mostly from the ED), `Urgent`, or `Routine`.
- `modality` (`CT`, `MRI`, `Ultrasound`, or `X-ray`) and `body_region`.
- `scan_min`: minutes the patient spent on the scanner table.
- `dose_msv`: the exam's **effective radiation dose** in millisieverts (mSv), blank for MRI and ultrasound, which use no ionizing radiation. A chest X-ray is about 0.02 mSv; an abdomen and pelvis CT is about 10.
- `wait_hr`: hours from the order to the scan.
- `report_hr`: hours from the scan to the radiologist's signed report, blank for the 2,262 exams whose report was not signed when the data were pulled.
- `reader_id`: the radiologist who signed it.

```text
exam_id,patient_id,site,exam_date,age,sex,setting,priority,modality,body_region,scan_min,dose_msv,wait_hr,report_hr,reader_id
E0000001,P077750,Northgate,2025-01-01,67,M,Inpatient,Routine,Ultrasound,Abdomen/Pelvis,28,,43.3,11.12,T04
E0000003,P198046,Northgate,2025-01-01,70,M,Inpatient,STAT,CT,Head,12,2.188,2.6,2.11,T02
```

## Setup

1. Fork and clone the repository, then open its folder in VS Code.
2. Create the environment:

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```
3. Open `assignment.ipynb` and select the `.venv` Python as the kernel.
4. Run the notebook's first two code cells. The second downloads `data/imaging_exams.parquet` (11 MB) if the folder lacks it, then reads it.
    - Expect: `pandas: 3.0.5` and `pyarrow: 25.0.0`, then `exams: (540228, 15)` and the first five exams.

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/
│   └── imaging_exams.parquet  # supplied exam log; keep it exactly as handed out
├── pyproject.toml          # supplied: numpy, pandas, pyarrow (for Parquet), and ipykernel
├── uv.lock                 # supplied: the exact versions `uv sync` installs
├── .python-version         # supplied: tells uv to use Python 3.13
├── CHECKS.md               # supplied: what each check looks for
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── site_summary.csv           # you generate in Task 1.2
    ├── site_modality_summary.csv  # you generate in Task 2.1
    ├── delayed_stat_exams.csv     # you generate in Task 2.2
    ├── ct_dose_review.csv         # you generate in Task 2.3
    ├── slowest_stat_reports.csv   # you generate in Task 2.4
    └── median_report_pivot.csv    # you generate in Task 3.1
```

## Task 1: Summarize each site

The exam log has one row per exam. Every answer in Task 1 has one row per site: grouping changes the grain, as Lecture 08's "The Split-Apply-Combine Paradigm" section shows.

### 1.1 Put the sites in reporting order

This step saves nothing. The network's board lists sites in the order of `SITE_ORDER`, hospitals first, and wants to see that Eastport has no exams yet.

1. Overwrite the `site` column with an ordered categorical in `SITE_ORDER`, as Lecture 08's "A Reporting Order and an Unused Clinic" snippet does.
    - Expect: `site dtype: category`.
2. Count the exams at each site as `exams_per_site`, keeping the unused category, as the same snippet does with `observed`.
    - Expect: nine sites in `SITE_ORDER`, starting with Downtown 129,876 and ending with Valley 25,384 and Eastport 0.

The rest of the assignment groups with `observed=True`, which lists the eight sites with exams, still in this order.

### 1.2 Count exams, reports, and patients, and total the scanner time

The operations director wants one row per site. In the Task 1.2 cell:

1. Build `site_summary` from one `groupby()` on `site` with named aggregation, keeping `site` as a column, as Lecture 08's "Count and Summarize Each Clinic" snippet does. Its reference card lists the functions. The five new columns:
    - `exam_count`: every exam at the site.
    - `report_count`: the exams with a signed report. Pick the function that skips blank `report_hr` cells.
    - `patient_count`: each patient once, however many exams they had.
    - `total_scan_min` and `mean_scan_min`: the total and the mean of `scan_min`.
    - Expect: `rows: 8`.
2. Save it to `SITE_SUMMARY_PATH` without the index. Leave the means unrounded.

> **Checkpoint: `output/site_summary.csv`**
> The header line `site,exam_count,report_count,patient_count,total_scan_min,mean_scan_min`, then eight rows, Downtown to Valley. Valley reads `Valley,25384,25280,12905,359150,14.148...`: 25,384 exams from 12,905 patients.

## Task 2: Compare sites, modalities, and single exams

### 2.1 Summarize each site and modality

Grouping by two keys gives one row per observed pair, as Lecture 08's "Grouping by Two Keys" section shows. In the Task 2.1 cell:

1. Build `site_modality` from a `groupby()` on `site` and `modality`, with both kept as columns and three named aggregations: `exam_count` (every exam), `median_wait_hr`, and `median_report_hr`. Waits are skewed: a few routine MRIs wait weeks, so the median describes a typical exam better than the mean.
    - Expect: `rows: 31`. Valley has no MRI pair.
2. Save it to `SITE_MODALITY_PATH` without the index.

> **Checkpoint: `output/site_modality_summary.csv`**
> The header line `site,modality,exam_count,median_wait_hr,median_report_hr`, then 31 rows. Northgate's MRI row reads `Northgate,MRI,12900,227.5,10.86...`: a typical MRI there waited about nine and a half days.

### 2.2 Flag STAT exams that waited far longer than their peers

A STAT exam should be scanned within about an hour. The ED quality lead wants every STAT exam that waited more than 8 hours, with the typical STAT wait for the same site and modality beside it. `transform` gives every row its group's summary, as Lecture 08's "Compare Each Visit with Its Clinic" snippet does with a mean and Demo 2's step 3 does for each clinic and department. In the Task 2.2 cell:

1. Build `stat`, the STAT exams only, with a row mask on `priority` (Lecture 04). Take a copy so new columns go on it.
2. Add `peer_median_wait_hr`: the median `wait_hr` of each exam's site and modality, from a grouped `transform`. Compute it on every STAT exam before keeping the delayed ones; a median of only the delayed exams would compare them with one another.
3. Add `wait_vs_peer_hr`: how many hours longer than its peer median the exam waited.
    - Expect: `STAT exams: 152925`.
4. Build `delayed_stat`: the STAT exams that waited more than 8 hours (strictly more), with the six columns of the checkpoint header.
    - Expect: `delayed: 190`.
5. Save it to `DELAYED_STAT_PATH` without the index.

> **Checkpoint: `output/delayed_stat_exams.csv`**
> The header line `exam_id,site,modality,wait_hr,peer_median_wait_hr,wait_vs_peer_hr`, then 190 rows. Exam E0000544, a STAT MRI at Downtown, waited 17.4 hours against a peer median of 1.7, so its `wait_vs_peer_hr` is 15.7.

### 2.3 Build a CT radiation-dose review list

The radiation safety officer reviews every patient whose CT doses this year add up to 100 mSv or more, a level linked to a measurable rise in lifetime cancer risk. The review needs all of those patients' CTs, the low-dose head scans included. `filter` tests each group as a whole and keeps all of its rows or none, as Lecture 08's "Keep Clinics by Size, Mean, or Total" snippet does with a total and Demo 2's step 4 does with a patient count. In the Task 2.3 cell:

1. Build `ct`, the CT exams only, with a row mask on `modality`.
2. Build `dose_review`: group `ct` by patient and `.filter()` with a `lambda` that keeps a patient whose `dose_msv` adds up to at least 100. The `lambda` runs once for each of the 87,663 CT patients, so the cell takes 10 to 30 seconds.
    - Expect: `rows: 1253` and `patients: 159`.
3. Build `row_mask` for contrast: a plain row mask that keeps each CT whose own `dose_msv` is at least 100.
    - Expect: `row mask rows: 0`. No single CT comes close to 100 mSv (the highest is 46.5), so a per-scan test misses every patient the review is for.
4. Build `fast_review`, the same rows from a row mask on each patient's total, broadcast to every CT by a grouped `transform` as in Task 2.2. Demo 3's step 5 shows why a built-in function beats a `lambda`.
    - Expect: `same rows as filter: True` in well under a second.
5. Save `dose_review`'s `exam_id`, `patient_id`, and `dose_msv` to `CT_DOSE_PATH` without the index.

> **Checkpoint: `output/ct_dose_review.csv`**
> The header line `exam_id,patient_id,dose_msv`, then 1,253 rows, every CT of the 159 patients. Patient P177810 has 32 of them, adding up to 192.4 mSv.

### 2.4 Pull each site's three slowest STAT reports

For the monthly quality meeting, each site reviews its three STAT exams whose signed report took longest. `apply` hands each site's rows to your function, as Lecture 08's "Sort Within Clinics and Keep the Two Longest Waits" snippet and Demo 3's step 2 show with `nlargest()`. In the Task 2.4 cell:

1. Build `slowest`: group `stat` from Task 2.2 by site and `.apply()` a `lambda` that keeps each site's three largest `report_hr`. Leave the grouping column out of what the `lambda` receives, as the snippet does.
    - Expect: `rows: 24` and `index levels: 2`: the site, then each exam's original row number.
2. Move both index levels back into columns with Lecture 06's `reset_index()`, then keep the four columns of the checkpoint header. The row numbers land in a column named `level_1`, which those four leave out. Demo 3's step 2 also shows the flat route without `apply`, which keeps the same 24 exams.
3. Save it to `SLOWEST_PATH` without the index.

> **Checkpoint: `output/slowest_stat_reports.csv`**
> The header line `exam_id,site,modality,report_hr`, then 24 rows, three per site. Downtown's slowest is E0313443, an X-ray whose report took 14.82 hours.

## Task 3: Pivot the report times

### 3.1 Build the pivot table and check it

A pivot table lays Task 2.1's median report times out as a grid, one row per site and one column per modality, as Lecture 08's "A Pivot Table and Its GroupBy Twin" snippet shows. In the Task 3.1 cell:

1. Build `report_pivot` with `pd.pivot_table()` on `exams`: the median `report_hr`, sites as rows, modalities as columns, observed pairs only.
2. Build its twin, `groupby_twin`: the same medians from a two-key `groupby()`, with `modality` moved into columns by `.unstack()`, as the snippet does.
    - Expect: `pivot matches its groupby twin: True` and `empty cells: 1`. That cell is Valley's MRI: no MRI happened there, so there is no report time to take the median of.
3. Save `report_pivot` to `PIVOT_PATH` with its index: the site names are the index, and saving writes them as the first column. Leave the empty cell empty. Filling it with 0 would report reads signed instantly at a scanner that does not exist.

The hospitals sign a CT in about 1.1 to 1.6 hours and the centers in about 10: more than half of the hospitals' CTs are STAT exams from the ED.

> **Checkpoint: `output/median_report_pivot.csv`**
> The header line `site,CT,MRI,Ultrasound,X-ray`, then eight rows, Downtown to Valley. Valley's `MRI` cell is empty.

### 3.2 Count the exams in each pair

This step saves nothing. In the Task 3.2 cell, count the exams for each site and modality with `pd.crosstab()`, sites down the rows and modalities across the columns, and name the result `exam_counts`. The same Lecture 08 snippet ends with one.

- Expect: eight rows and four columns. Valley's `MRI` cell shows `0`: zero exams is a real count, while the pivot of medians leaves the same cell empty.

## Check your work

- Run the checks any time, even partway through: `python3 check_assignment.py` in the terminal (with the environment active) prints your score and what to fix.
- When **Restart** and **Run All** finish without errors, the notebook's last cell runs the same checks.
- Both use the latest checks from the course repository, the same checks GitHub runs on each push. Commit `assignment.ipynb` and the `output/` files, then push.
- Expect: a passing run prints `Score: 100/100` and `All checks passed.`

What each check looks for: [CHECKS.md](CHECKS.md)
