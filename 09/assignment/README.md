# Assignment 09: Time Series Features from Patient Vitals

## Overview

Prepare a hospital step-down unit's heart rates as a time series, change their frequency, and build features that use only what was known by the prediction time, saving each result for the checks.

`data/vitals.csv` is a synthetic export from a hospital step-down unit on January 20, 2026: one row per charted heart rate, in beats per minute, for two patients. P01's heart rate climbs from 84 to 110 over the day, and P02's stays between 70 and 76. Readings were charted on the hour but not every hour, and the rows are out of order. `recorded_at` is the unit's New York wall-clock time, written as text with no time zone. At 10:00 a nurse started a row for P01 but charted no heart rate.

```text
patient_id,recorded_at,heart_rate
P02,2026-01-20 12:00,76
P01,2026-01-20 07:00,84
P02,2026-01-20 08:00,72
```

`data/labs.csv` holds the same two patients' six lab orders: the `test`, when the sample was drawn (`collected_at`), and when the result was reported (`resulted_at`), on the same New York clock.

```text
patient_id,test,collected_at,resulted_at
P01,lactate,2026-01-20 11:05,2026-01-20 11:50
P01,procalcitonin,2026-01-20 10:40,2026-01-21 09:30
```

## Setup

1. Fork the assignment repository on GitHub and clone your fork as in Lecture 01: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open the cloned folder itself, not a folder above it.
2. In the integrated terminal, create the environment, activate it, and install the packages `pyproject.toml` and `uv.lock` list. Do not run `uv init`: the project files already exist.

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```

    - Expect: `uv venv` prints `Using CPython 3.13.x`, and `uv sync` lists `+ pandas==3.0.5` and `+ ipykernel==6.29.5` among the packages it installs.
3. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`.
4. Run the notebook's first two code cells.
    - Expect: `data folder found: True`, then `vitals: (12, 3)` and `labs: (6, 4)`.

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/
│   ├── vitals.csv          # supplied heart rates; keep it exactly as handed out
│   └── labs.csv            # supplied lab orders; keep it exactly as handed out
├── pyproject.toml          # supplied: numpy, pandas, and ipykernel
├── uv.lock                 # supplied: the exact versions `uv sync` installs
├── .python-version         # supplied: tells uv to use Python 3.13
├── CHECKS.md               # supplied: what each check looks for
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── prepared_vitals.csv       # you generate in Task 1.1
    ├── hourly_grid.csv           # you generate in Task 2.1
    ├── two_hour_summary.csv      # you generate in Task 2.2
    ├── past_features.csv         # you generate in Task 3.1
    ├── lab_availability.csv      # you generate in Task 3.2
    └── chronological_blocks.csv  # you generate in Task 3.3
```

## Task 1: Prepare the vitals panel

`data/vitals.csv` stacks two patients' histories in one table, with `patient_id` naming whose reading each row is: a panel, in Lecture 09's terms. Every later task starts from the table this task prepares.

### 1.1 Parse, localize, and sort the readings

In the Task 1.1 cell, `vitals` starts as a copy of the supplied table:

1. Parse `recorded_at` with `pd.to_datetime(vitals["recorded_at"], format="%Y-%m-%d %H:%M")`, as Lecture 09's "Text Column to DatetimeIndex" snippet does.
2. Attach the zone the unit charted in with `.dt.tz_localize("America/New_York")`, then convert with `.dt.tz_convert("UTC")`, as Lecture 09's "Local Clinic Times to UTC" snippet does. New York is five hours behind UTC in January, so 07:00 on the unit's clock becomes `2026-01-20 12:00:00+00:00`. The cell prints `recorded_at dtype: datetime64[us, UTC]`.
3. Sort by `["patient_id", "recorded_at"]`, then `.reset_index(drop=True)`, so each patient's readings sit together, oldest first. Shifts and rolling windows read the rows in this order.
4. Add `source_row`, a column of `1`s. Task 2 uses it to tell the rows a grid creates from the rows that were charted.
5. Save `vitals` to `PREPARED_PATH` with `index=False`.

> **Checkpoint: `output/prepared_vitals.csv`**
> The header line `patient_id,recorded_at,heart_rate,source_row`, then 12 rows, P01's six readings first. The first row reads `P01,2026-01-20 12:00:00+00:00,84.0,1`, and P01's row at 15:00 UTC keeps its heart rate empty: `P01,2026-01-20 15:00:00+00:00,,1`.

## Task 2: Change the frequency

### 2.1 Build each patient's hourly grid

An hourly grid gives each patient one row per hour, from their first reading to their last, so an hour with no reading becomes a visible row. Group by patient first, so each patient gets their own grid: P01's runs from 12:00 to 19:00 UTC and P02's from 13:00 to 20:00. Lecture 09's "Hourly Grid per Patient" snippet builds the same grid. In the Task 2.1 cell:

1. The first line prints `all on the hour: True`. `asfreq()` keeps only readings exactly on the grid, and every reading here is on the hour.
2. Build `hourly_grid`: `vitals.set_index("recorded_at").groupby("patient_id")[["heart_rate", "source_row"]].resample("h").asfreq().reset_index()`.
3. Add the column `grid_created`: `hourly_grid["source_row"].isna()`, `True` on an hour that had no charted row.
4. Add the column `value_missing`: `hourly_grid["source_row"].notna() & hourly_grid["heart_rate"].isna()`, `True` on a charted row with no heart rate.

    It prints `rows: 16`, then shows `grid_created 4` and `value_missing 1`.
5. Save `hourly_grid` to `HOURLY_GRID_PATH` with `index=False`. Leave the empty heart rates empty: filling them would invent readings nobody took.

> **Checkpoint: `output/hourly_grid.csv`**
> The header line `patient_id,recorded_at,heart_rate,source_row,grid_created,value_missing`, then 16 rows. P01's hour at 14:00 UTC, which the grid created, reads `P01,2026-01-20 14:00:00+00:00,,,True,False`; its 15:00 row, charted with no heart rate, reads `P01,2026-01-20 15:00:00+00:00,,1.0,False,True`.

### 2.2 Summarize two-hour bins

Downsampling to two-hour bins gives each patient one row per bin. For `"2h"`, pandas' default bins are left-closed and left-labeled: a bin holds the readings from its start time up to, but not including, the next bin's start, and is named by its start. Lecture 09's "Two-Hour Summaries per Patient" snippet builds the same table. In the Task 2.2 cell:

1. Build `two_hour_summary`: `vitals.set_index("recorded_at").groupby("patient_id").resample("2h").agg(mean_hr=("heart_rate", "mean"), n_rows=("source_row", "count")).reset_index()`. `mean_hr` averages the heart rates in each bin. `n_rows` counts the charted rows, including the one with no heart rate, which counting `heart_rate` would skip. It prints `rows: 9` and `readings counted: 12`.
2. Save `two_hour_summary` to `TWO_HOUR_PATH` with `index=False`.

> **Checkpoint: `output/two_hour_summary.csv`**
> The header line `patient_id,recorded_at,mean_hr,n_rows`, then 9 rows: four bins for P01 and five for P02. P02's first bin is named 12:00 even though its first reading is at 13:00: `P02,2026-01-20 12:00:00+00:00,72.0,1`. P01's 14:00 bin holds only the row with no heart rate, so its mean is empty and its count is 1: `P01,2026-01-20 14:00:00+00:00,,1`.

## Task 3: Build past-only evidence

An early-warning score runs at `2026-01-20 18:00:00+00:00`, which is 13:00 on the unit's clock. It may use only what was known by then: features that read earlier rows of the same patient, and lab results that had already been reported.

### 3.1 Calculate past-only features

Lecture 09's "Grouped Lag" and "Grouped Past-Only Windows" snippets build the same four features. In the Task 3.1 cell, `past_features` starts as a copy of three columns of `vitals`, and `by_patient` groups its heart rates by patient:

1. Add the column `previous_hr`: `by_patient.shift(1)`, the same patient's previous reading, empty on each patient's first row. A plain `shift(1)` on the stacked column would hand P02's first row P01's last heart rate.
2. Add the column `hr_change`: `by_patient.diff()`, the current heart rate minus the previous one.
3. Add the column `mean_prev_2`: `by_patient.transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())`, the mean of the previous two readings, however far apart they are.
4. Build `prev_2h`: `past_features.set_index("recorded_at").groupby("patient_id")["heart_rate"].rolling("2h", closed="left").mean().rename("mean_prev_2h").reset_index()`, the mean of the readings in the two hours before each reading. `closed="left"` leaves the current reading out.
5. Merge it back: `past_features = past_features.merge(prev_2h, on=["patient_id", "recorded_at"], validate="one_to_one")`. It prints `rows: 12`.
6. Save `past_features` to `FEATURES_PATH` with `index=False`.

> **Checkpoint: `output/past_features.csv`**
> The header line `patient_id,recorded_at,heart_rate,previous_hr,hr_change,mean_prev_2,mean_prev_2h`, then 12 rows, one per reading. P01's row at 16:00 UTC has an empty `previous_hr` and `hr_change`, because the reading before it has no heart rate. At P02's 20:00 reading, `mean_prev_2` is `74.5`, the mean of the 76 and 73 readings, but `mean_prev_2h` is `73.0`, because only the 18:00 reading falls in the two hours before 20:00.

### 3.2 Check which labs were known at the prediction time

A lab result can feed the score only if it was reported by the prediction time, whenever its sample was drawn. Lecture 09's "Recorded Is Not Available" snippet makes the same comparison. In the Task 3.2 cell:

1. Set `prediction_time = pd.Timestamp("2026-01-20 18:00", tz="UTC")`, the instant `2026-01-20 18:00:00+00:00`.
2. `labs` starts as a copy of the supplied lab orders. Convert its `collected_at` and `resulted_at` columns the way Task 1.1 converted `recorded_at`: parse with the same format, localize to `America/New_York`, and convert to UTC.
3. Add the column `available`: `labs["resulted_at"] <= prediction_time`. `<=` counts a result reported exactly at the prediction time. It prints `available: 3 of 6`.
4. Save `labs` to `LABS_PATH` with `index=False`.

> **Checkpoint: `output/lab_availability.csv`**
> The header line `patient_id,test,collected_at,resulted_at,available`, then 6 rows, one per lab order. P02's troponin was reported exactly at the prediction time: `P02,troponin,2026-01-20 17:30:00+00:00,2026-01-20 18:00:00+00:00,True`. P01's creatinine was drawn at `2026-01-20 17:20:00+00:00` but reported at `2026-01-20 18:25:00+00:00`, so it is `False`.

### 3.3 Label a chronological holdout

A chronological holdout builds a method on the earlier rows and tests it on the rows from a cutoff onward, the way it would meet later patients. The cutoff here is the prediction time. Lecture 09's "Chronological Holdout" snippet labels the same blocks. In the Task 3.3 cell:

1. `chronological_blocks` starts as a `.copy()` of `vitals`. Add the column `block`: `np.where(chronological_blocks["recorded_at"] < prediction_time, "earlier", "later_holdout")`. `<` puts a reading at exactly `2026-01-20 18:00:00+00:00` in `later_holdout`. The crosstab prints 4 `earlier` and 2 `later_holdout` readings for each patient.
2. Save `chronological_blocks` to `BLOCKS_PATH` with `index=False`.

> **Checkpoint: `output/chronological_blocks.csv`**
> The header line `patient_id,recorded_at,heart_rate,source_row,block`, then 12 rows: 8 `earlier` and 4 `later_holdout`. P01's reading at the cutoff reads `P01,2026-01-20 18:00:00+00:00,104.0,1,later_holdout`.

## Check your work

- Run the checks any time, even partway through: `python3 check_assignment.py` in the terminal (with the environment active) prints your score and what to fix.
- When **Restart** and **Run All** finish without errors, the notebook's last cell runs the same checks.
- Both use the latest checks from the course repository, the same checks GitHub runs on each push. Commit `assignment.ipynb` and the `output/` files, then push.
- Expect: a passing run ends with `Score: 100/100`.

What each check looks for: [CHECKS.md](CHECKS.md)
