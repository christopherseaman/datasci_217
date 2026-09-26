# Assignment 09: Time Series Features from Patient Vitals

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/
│   ├── vitals.csv          # supplied heart rates; keep it exactly as handed out
│   └── labs.csv            # supplied lab orders; keep it exactly as handed out
├── requirements.txt        # supplied: numpy, pandas, and ipykernel
├── .python-version         # supplied: tells uv to use Python 3.13
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

## The data

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

Fork the assignment repository on GitHub and clone your fork the way Lecture 01 did: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open it. Then open **Terminal → New Terminal** in VS Code at the assignment directory (Ctrl+Shift+backtick, also Control on Mac). If you use a native terminal or WSL Ubuntu instead, `cd` into the assignment directory first. Run `ls data` and expect `labs.csv` and `vitals.csv`. This clone is a new repository, so before your first commit run Lecture 02's two `git config user.name "..."` and `git config user.email "..."` lines in this terminal, with your name and GitHub noreply email.

Create the project environment, activate it, and install the supplied requirements, as in Lecture 03:

```bash
uv venv --seed --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` includes **ipykernel**, the package that lets a notebook run on this environment's Python (Lecture 04), so this one install is all the notebook needs. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`. If VS Code offers to install the **Jupyter** extension, accept.

> **Windows:** work in the **WSL: Ubuntu** window from Lecture 01's setup. In PowerShell instead, activate with `.\.venv\Scripts\Activate.ps1`.

Run the notebook's first two code cells. The first prints the pandas version and `data folder found: True`; `False` means the notebook is not running from the assignment directory, so open the folder itself in VS Code, not a folder above it. The second reads both files and prints `vitals: (12, 3)` and `labs: (6, 4)`.

## Task 1: Prepare the vitals panel

`data/vitals.csv` stacks two patients' histories in one table, with `patient_id` naming whose reading each row is: a panel, in Lecture 09's terms. Every later task starts from the table this task prepares.

### 1.1 Parse, localize, and sort the readings

In the Task 1.1 cell, `vitals` starts as a copy of the supplied table:

1. Parse `recorded_at` with `pd.to_datetime(vitals["recorded_at"], format="%Y-%m-%d %H:%M")`, as Lecture 09's "Text Column to DatetimeIndex" snippet does.
2. Attach the zone the unit charted in with `.dt.tz_localize("America/New_York")`, then convert with `.dt.tz_convert("UTC")`, as the "Local Clinic Times to UTC" snippet and section 2 of Demo 3 do. New York is five hours behind UTC in January, so 07:00 on the unit's clock becomes `2026-01-20 12:00:00+00:00`. The cell prints `recorded_at dtype: datetime64[us, UTC]`.
3. Sort by `["patient_id", "recorded_at"]`, then `.reset_index(drop=True)`, so each patient's readings sit together, oldest first. Shifts and rolling windows read the rows in this order.
4. Add `source_row`, a column of `1`s. Task 2 uses it to tell the rows a grid creates from the rows that were charted.
5. Save `vitals` to `PREPARED_PATH` with `index=False`.

> **Checkpoint: `output/prepared_vitals.csv`**
> The header line `patient_id,recorded_at,heart_rate,source_row`, then 12 rows, P01's six readings first. The first row reads `P01,2026-01-20 12:00:00+00:00,84.0,1`, and P01's row at 15:00 UTC keeps its heart rate empty: `P01,2026-01-20 15:00:00+00:00,,1`.

## Task 2: Change the frequency

### 2.1 Build each patient's hourly grid

An hourly grid gives each patient one row per hour, from their first reading to their last, so an hour with no reading becomes a visible row. Group by patient first, so each patient gets their own grid: P01's runs from 12:00 to 19:00 UTC and P02's from 13:00 to 20:00. Lecture 09's "Hourly Grid per Patient" snippet and section 5 of Demo 2 build the same grid. In the Task 2.1 cell:

1. The first line prints `all on the hour: True`. `asfreq()` keeps only readings exactly on the grid, and every reading here is on the hour.
2. Build `hourly_grid`: `vitals.set_index("recorded_at").groupby("patient_id")[["heart_rate", "source_row"]].resample("h").asfreq().reset_index()`.
3. Add the column `grid_created`: `hourly_grid["source_row"].isna()`, `True` on an hour that had no charted row.
4. Add the column `value_missing`: `hourly_grid["source_row"].notna() & hourly_grid["heart_rate"].isna()`, `True` on a charted row with no heart rate.

    It prints `rows: 16`, then `grid_created 4` and `value_missing 1`.
5. Save `hourly_grid` to `HOURLY_GRID_PATH` with `index=False`. Leave the empty heart rates empty: filling them would invent readings nobody took.

> **Checkpoint: `output/hourly_grid.csv`**
> The header line `patient_id,recorded_at,heart_rate,source_row,grid_created,value_missing`, then 16 rows. P01's hour at 14:00 UTC, which the grid created, reads `P01,2026-01-20 14:00:00+00:00,,,True,False`; its 15:00 row, charted with no heart rate, reads `P01,2026-01-20 15:00:00+00:00,,1.0,False,True`.

### 2.2 Summarize two-hour bins

Downsampling to two-hour bins gives each patient one row per bin. For `"2h"`, pandas' default bins are left-closed and left-labeled: a bin holds the readings from its start time up to, but not including, the next bin's start, and is named by its start. Lecture 09's "Two-Hour Summaries per Patient" snippet and section 4 of Demo 2 build the same table. In the Task 2.2 cell:

1. Build `two_hour_summary`: `vitals.set_index("recorded_at").groupby("patient_id").resample("2h").agg(mean_hr=("heart_rate", "mean"), n_rows=("source_row", "count")).reset_index()`. `mean_hr` averages the heart rates in each bin. `n_rows` counts the charted rows, including the one with no heart rate, which counting `heart_rate` would skip. It prints `rows: 9` and `readings counted: 12`.
2. Save `two_hour_summary` to `TWO_HOUR_PATH` with `index=False`.

> **Checkpoint: `output/two_hour_summary.csv`**
> The header line `patient_id,recorded_at,mean_hr,n_rows`, then 9 rows: four bins for P01 and five for P02. P02's first bin is named 12:00 even though its first reading is at 13:00: `P02,2026-01-20 12:00:00+00:00,72.0,1`. P01's 14:00 bin holds only the row with no heart rate, so its mean is empty and its count is 1: `P01,2026-01-20 14:00:00+00:00,,1`.

## Task 3: Build past-only evidence

An early-warning score runs at `2026-01-20 18:00:00+00:00`, which is 13:00 on the unit's clock. It may use only what was known by then: features that read earlier rows of the same patient, and lab results that had already been reported.

### 3.1 Calculate past-only features

Lecture 09's "Grouped Lag" and "Grouped Past-Only Windows" snippets and sections 5 and 6 of Demo 3 build the same four features. In the Task 3.1 cell, `past_features` starts as a copy of three columns of `vitals`, and `by_patient` groups its heart rates by patient:

1. Add the column `previous_hr`: `by_patient.shift(1)`, the same patient's previous reading, empty on each patient's first row. A plain `shift(1)` on the stacked column would hand P02's first row P01's last heart rate.
2. Add the column `hr_change`: `by_patient.diff()`, the current heart rate minus the previous one.
3. Add the column `mean_prev_2`: `by_patient.transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())`, the mean of the previous two readings, however far apart they are.
4. Build `prev_2h`: `past_features.set_index("recorded_at").groupby("patient_id")["heart_rate"].rolling("2h", closed="left").mean().rename("mean_prev_2h").reset_index()`, the mean of the readings in the two hours before each reading. `closed="left"` leaves the current reading out.
5. Merge it back: `past_features = past_features.merge(prev_2h, on=["patient_id", "recorded_at"], validate="one_to_one")`. It prints `rows: 12`.
6. Save `past_features` to `FEATURES_PATH` with `index=False`.

> **Checkpoint: `output/past_features.csv`**
> The header line `patient_id,recorded_at,heart_rate,previous_hr,hr_change,mean_prev_2,mean_prev_2h`, then 12 rows, one per reading. P01's row at 16:00 UTC has an empty `previous_hr` and `hr_change`, because the reading before it has no heart rate. At P02's 20:00 reading, `mean_prev_2` is `74.5`, the mean of the 76 and 73 readings, but `mean_prev_2h` is `73.0`, because only the 18:00 reading falls in the two hours before 20:00.

### 3.2 Check which labs were known at the prediction time

A lab result can feed the score only if it was reported by the prediction time, whenever its sample was drawn. Lecture 09's "Recorded Is Not Available" snippet and section 7 of Demo 3 make the same comparison. In the Task 3.2 cell:

1. Set `prediction_time = pd.Timestamp("2026-01-20 18:00", tz="UTC")`, the instant `2026-01-20 18:00:00+00:00`.
2. `labs` starts as a copy of the supplied lab orders. Convert its `collected_at` and `resulted_at` columns the way Task 1.1 converted `recorded_at`: parse with the same format, localize to `America/New_York`, and convert to UTC.
3. Add the column `available`: `labs["resulted_at"] <= prediction_time`. `<=` counts a result reported exactly at the prediction time. It prints `available: 3 of 6`.
4. Save `labs` to `LABS_PATH` with `index=False`.

> **Checkpoint: `output/lab_availability.csv`**
> The header line `patient_id,test,collected_at,resulted_at,available`, then 6 rows, one per lab order. P02's troponin was reported exactly at the prediction time: `P02,troponin,2026-01-20 17:30:00+00:00,2026-01-20 18:00:00+00:00,True`. P01's creatinine was drawn at `2026-01-20 17:20:00+00:00` but reported at `2026-01-20 18:25:00+00:00`, so it is `False`.

### 3.3 Label a chronological holdout

A chronological holdout builds a method on the earlier rows and tests it on the rows from a cutoff onward, the way it would meet later patients. The cutoff here is the prediction time. Lecture 09's "Chronological Holdout" snippet and section 8 of Demo 3 label the same blocks. In the Task 3.3 cell:

1. `chronological_blocks` starts as a `.copy()` of `vitals`. Add the column `block`: `np.where(chronological_blocks["recorded_at"] < prediction_time, "earlier", "later_holdout")`. `<` puts a reading at exactly `2026-01-20 18:00:00+00:00` in `later_holdout`. The crosstab prints 4 `earlier` and 2 `later_holdout` readings for each patient.
2. Save `chronological_blocks` to `BLOCKS_PATH` with `index=False`.

> **Checkpoint: `output/chronological_blocks.csv`**
> The header line `patient_id,recorded_at,heart_rate,source_row,block`, then 12 rows: 8 `earlier` and 4 `later_holdout`. P01's reading at the cutoff reads `P01,2026-01-20 18:00:00+00:00,104.0,1,later_holdout`.

## Check your work

Click **Restart**, then **Run All**. The last cell prints `Fresh-run check passed`, or names the task to fix. Then, with the environment active, run the checks from the assignment directory:

```bash
python check_assignment.py
```

`check_assignment.py` runs the same checks GitHub runs. They read only the six CSV files in `output/` and compare them with values computed from the supplied data. They never run or read your notebook, so any way of producing correct files counts.

Each check prints `PASS` or `FIX` and the points it earned, and a `FIX` says what to fix on the line beneath it. When the next checks need the same fix, such as a missing file, they say `(same fix as above)`. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/3  prepared vitals: columns
         output/prepared_vitals.csv is missing; run the Task 1.1 cell to write it, then commit it.
```

Below the score, `Left to fix` lists the checks still failing and the points they are worth. Fix what they name, rerun the notebook and then the checks, and repeat until every check passes. A clean run ends with:

```text
[PASS]  6/6  chronological blocks: block labels

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only that check's points.
- Spacing, line endings, quoting, column order, and row order never cost points.
- Numbers are compared as numbers, so `84`, `84.0`, and `84.00` are the same value. A mean may keep every digit or be rounded to one decimal.
- Timestamps are compared as instants: `2026-01-20 18:00:00+00:00`, `2026-01-20T18:00:00Z`, `2026-01-20 18:00 UTC`, and `2026-01-20 13:00:00-05:00` all name the same moment. A timestamp with no offset is read as UTC.
- Only Task 1.1 checks that `recorded_at` was converted to UTC; Task 3.2 checks its own lab times. If every `recorded_at` in a later file is off by the same number of hours because Task 1.1 skipped the conversion, that file's rows are still matched and its other values still count.
- `True` and `False` may be written in any letter case, or as `1` and `0`.
- Patient IDs, test names, block labels, and column names are compared in any letter case, and `later holdout` matches `later_holdout`.
- An empty cell may be written empty or as `NaN`.
- A leading column of row numbers, which `to_csv()` writes when `index=False` is left out, is ignored.
- `source_row` may be kept in or left out of `hourly_grid.csv`, `past_features.csv`, and `chronological_blocks.csv`.

Every push also runs GitHub Actions, which downloads the course's current copy of the checks and reruns them on the files you committed and pushed. That run is what counts, and a check corrected after handout reaches you there on your next push.

### Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/prepared_vitals.csv` | Its columns are the four in the Task 1.1 header line. | prepared vitals: columns | 3 |
| `output/prepared_vitals.csv` | It holds each of the 12 readings once. | prepared vitals: one row per reading | 4 |
| `output/prepared_vitals.csv` | Each `recorded_at` is the reading's UTC instant. | prepared vitals: recorded_at in UTC | 6 |
| `output/prepared_vitals.csv` | Each `heart_rate` matches `data/vitals.csv`, with P01's missing value still empty. | prepared vitals: heart_rate values | 4 |
| `output/prepared_vitals.csv` | Each `source_row` is 1. | prepared vitals: source_row values | 3 |
| `output/hourly_grid.csv` | Its columns are the ones in the Task 2.1 header line. | hourly grid: columns | 3 |
| `output/hourly_grid.csv` | It holds each patient's 8 hours once. | hourly grid: one row per patient-hour | 5 |
| `output/hourly_grid.csv` | Each charted hour keeps its `heart_rate`, and each created hour stays empty. | hourly grid: heart_rate values | 4 |
| `output/hourly_grid.csv` | `grid_created` is `True` on exactly the 4 hours with no charted row. | hourly grid: grid_created flags | 4 |
| `output/hourly_grid.csv` | `value_missing` is `True` on exactly the charted row with no heart rate. | hourly grid: value_missing flags | 4 |
| `output/two_hour_summary.csv` | Its columns are the four in the Task 2.2 header line. | two-hour summary: columns | 3 |
| `output/two_hour_summary.csv` | It holds each of the 9 patient bins once. | two-hour summary: one row per patient and bin | 5 |
| `output/two_hour_summary.csv` | Each `mean_hr` is the mean heart rate in the bin, empty where there is none. | two-hour summary: mean_hr values | 4 |
| `output/two_hour_summary.csv` | Each `n_rows` counts the bin's charted rows. | two-hour summary: n_rows values | 4 |
| `output/past_features.csv` | Its columns are the seven in the Task 3.1 header line. | past features: columns | 3 |
| `output/past_features.csv` | It holds each of the 12 readings once. | past features: one row per reading | 3 |
| `output/past_features.csv` | Each `previous_hr` is the same patient's previous heart rate. | past features: previous_hr values | 4 |
| `output/past_features.csv` | Each `hr_change` is the heart rate minus `previous_hr`. | past features: hr_change values | 3 |
| `output/past_features.csv` | Each `mean_prev_2` is the mean of the patient's previous two readings. | past features: mean_prev_2 values | 4 |
| `output/past_features.csv` | Each `mean_prev_2h` is the mean of the patient's readings in the two hours before, not counting the current one. | past features: mean_prev_2h values | 4 |
| `output/lab_availability.csv` | Its columns are the five in the Task 3.2 header line. | lab availability: columns | 2 |
| `output/lab_availability.csv` | It holds each of the 6 lab orders once. | lab availability: one row per lab | 2 |
| `output/lab_availability.csv` | Each `collected_at` and `resulted_at` is the UTC instant of the supplied clock time. | lab availability: collected_at and resulted_at in UTC | 4 |
| `output/lab_availability.csv` | `available` is `True` exactly where `resulted_at` is at or before `2026-01-20 18:00:00+00:00`. | lab availability: available flags | 4 |
| `output/chronological_blocks.csv` | Its columns are the ones in the Task 3.3 header line. | chronological blocks: columns | 2 |
| `output/chronological_blocks.csv` | It holds each of the 12 readings once. | chronological blocks: one row per reading | 3 |
| `output/chronological_blocks.csv` | `block` is `earlier` before `2026-01-20 18:00:00+00:00` and `later_holdout` from then on. | chronological blocks: block labels | 6 |

Extra files are ignored.

## Submit

Before you commit a notebook, follow Lecture 04's "Before You Commit a Notebook": click **Clear All Outputs**, then save. In VS Code Source Control, stage `assignment.ipynb` and the six files in `output/`. Commit with `Complete Assignment 09 notebook` and select **Sync Changes**. Keep `.venv/` out of the commit; `.gitignore` already lists it.

Confirm the notebook and the six CSV files on `main` in the repository browser. GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork. If a run cannot download the course's current checks, it grades with the copy in your repository and says so in its log. If your local run and the GitHub run ever disagree, the GitHub run counts, because it uses the course's current checks. If a required VS Code control is unavailable, record its message and contact the instructor.
