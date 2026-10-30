# Assignment 09: Monitoring a Hospital-at-Home Program

## Overview

Audit a wearable heart-rate feed by New York calendar day, measure each episode's monitoring gaps, compare three heart-rate alert rules, put lab times on UTC, build past-only features at a fixed prediction time, and summarize a chronological holdout, saving each result for the checks.

All three files are synthetic and come from one New York health system's **hospital-at-home** program, January 5 to March 27, 2026: patients with an acute illness (pneumonia, a COPD or heart failure flare, cellulitis, a urinary infection) are treated in their own homes instead of a hospital ward. A wearable patch sends a heart rate every 10 minutes, and a visiting nurse draws blood. An **episode** runs from admission to the program until the patient is discharged home or **escalated**, sent back to the hospital because they got worse.

- `data/episodes.parquet`: 1,400 episodes, one row each: `episode_id`, `patient_id` (some patients come back), `age`, `sex`, `deprivation_decile` (1 for the least deprived neighborhoods, 10 for the most), `diagnosis`, `admitted_at` and `ended_at` (already aware UTC datetimes), and `outcome`: `discharged home` or `escalated`.
- `data/vitals.parquet`: 577,935 patch readings: `episode_id`, `recorded_at` as text from the vendor's feed in UTC (the trailing `Z` means UTC), and `heart_rate` in beats per minute (bpm), blank when the patch could not compute one.
- `data/labs.parquet`: 7,356 blood tests: `episode_id`, `test` (`BMP`, `BNP`, `CBC`, `INR`, or `Lactate`), `collected_at` and `resulted_at` as text on the New York wall clock, and the result `value` (lactate in mmol/L).

```text
episode_id,recorded_at,heart_rate
H00001,2026-01-05T16:11:00Z,80.0
H00001,2026-01-05T16:21:00Z,80.0
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
4. Run the notebook's first two code cells. The second downloads the three Parquet files (3 MB in all) into `data/` if the folder lacks them, then reads them.
    - Expect: `pandas: 3.0.5` and `pyarrow: 25.0.0`, then `episodes: (1400, 9)`, `vitals: (577935, 3)`, and `labs: (7356, 5)`.

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/                   # supplied; keep these files exactly as handed out
│   ├── episodes.parquet    # one row per episode
│   ├── vitals.parquet      # one row per patch reading
│   └── labs.parquet        # one row per blood test
├── pyproject.toml          # supplied: numpy, pandas, pyarrow (for Parquet), matplotlib, and ipykernel
├── uv.lock                 # supplied: the exact versions `uv sync` installs
├── .python-version         # supplied: tells uv to use Python 3.13
├── CHECKS.md               # supplied: what each check looks for
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── daily_feed.csv          # you generate in Task 1.2
    ├── episode_coverage.csv    # you generate in Task 2.1
    ├── alert_rules.csv         # you generate in Task 2.2
    ├── lab_turnaround.csv      # you generate in Task 3.1
    ├── score_features.csv      # you generate in Task 3.2
    └── holdout_summary.csv     # you generate in Task 3.3
```

## Task 1: Audit the feed by New York day

### 1.1 Parse and sort the readings

This step saves nothing. In the Task 1.1 cell, `vitals` starts as a copy of the feed:

1. Overwrite `recorded_at` with aware UTC datetimes: `pd.to_datetime()` with `utc=True`, which reads the trailing `Z` (Lecture 09's "Time Zones" card).
2. Sort by episode, then time, and reset the index, as Lecture 09's "Grouped Lag" snippet does. Every later shift and rolling window reads the rows in this order.
    - Expect: `recorded_at dtype: datetime64[us, UTC]`, the first reading at `2026-01-05 16:11:00+00:00`, and `no heart rate: 5780`.

### 1.2 Count each day's readings

The program manager wants one row per New York calendar day: how many readings arrived and from how many episodes. A day binned in UTC would start at 19:00 or 20:00 the evening before in New York, so convert first (Lecture 09's "Localize in the recording zone" callout). Demo 1's step 6 runs the same daily feed check, and Demo 3's independent practice 1 converts to New York time before `resample("D")`. In the Task 1.2 cell:

1. Build `local`, a copy of `vitals` whose `recorded_at` is converted to `America/New_York` with `.dt.tz_convert()`.
2. Build `daily`: set `recorded_at` as the index and `resample("D")` with named aggregation (Lecture 08's `(column, function)` pairs):
    - `readings`: every row the feed sent that day, with or without a heart rate (`"size"` of `heart_rate`).
    - `episodes`: each episode once (`"nunique"` of `episode_id`).

    Expect `days: 87`, January 5 to April 1.
3. Select February 14 to 20 from `daily` with a `.loc` date range (Lecture 09's "Calendar Selection" card); the cell adds `per_episode`, readings divided by episodes.
    - Expect: about 96 readings per episode on most days, but `73.9` on February 17, when the vendor's cloud was down for six hours.
4. Move the dates into a column with `reset_index()`, rename it from `recorded_at` to `date`, and save `daily_feed` to `DAILY_FEED_PATH` with `index=False`.

> **Checkpoint: `output/daily_feed.csv`**
> The header line `date,readings,episodes`, then 87 rows. The first reads `2026-01-05 00:00:00-05:00,828,17`, and February 17 reads `2026-02-17 00:00:00-05:00,5836,79`.

## Task 2: Measure gaps and compare alert rules

### 2.1 Find each episode's monitoring gaps

A patch that stops sending, because the home hub lost its connection, the patch came off, or its battery ran out, leaves hours with no reading. The quality team wants one row per episode: how many hours its grid spans, how many of them are empty, and how long its gaps run. Lecture 09's "Gap Runs" section and Demo 2's step 4 build the same audit. In the Task 2.1 cell:

1. Build `hourly_counts`: each episode's readings per hour, with or without a heart rate, from a grouped `resample("h").count()` of a `source_row` column of 1s (Lecture 09's "Grouped Resampling" card), with `reset_index()` and the count named `readings`. Each episode's grid runs from its first reading's hour to its last, and an hour with no reading counts 0.
2. Add `source_observed`, `True` where `readings` is above 0, and `gap_hour`, its opposite (`~`).
3. Number the gap runs: `run_id` is a grouped `cumsum()` of `source_observed`, as the "Count Gap Runs per Patient" snippet does. Keep the gap hours, count each `(episode_id, run_id)` run's hours with `.size()`, then summarize each episode's runs with `.agg()`: `gap_runs` (`"count"`) and `longest_gap_h` (`"max"`).
    - Expect: `hourly rows: 108293  gap runs: 3102`.
4. Build `coverage`, one row per episode: `grid_hours` (the `"size"` of the episode's hourly rows) and `gap_hours` (the `"sum"` of `gap_hour`). Left-merge the run summary onto it (Lecture 06, `validate="one_to_one"`), and fill the missing `gap_runs` and `longest_gap_h` of the episodes with no gap with 0 (`fillna()`), as Demo 2 does.
    - Expect: `episodes: 1400  no gap: 201`.
5. Save `coverage` to `COVERAGE_PATH` with `index=False`.
6. The cell then merges `deprivation_decile` from `episodes` and prints the mean share of grid hours lost per decile.
    - Expect: about `0.045` in decile 1 and `0.072` in decile 10. Home broadband is patchier in deprived neighborhoods, so those patients are watched less.

> **Checkpoint: `output/episode_coverage.csv`**
> The header line `episode_id,grid_hours,gap_hours,gap_runs,longest_gap_h`, then 1,400 rows. `H00002` reads `H00002,117,7,3.0,3.0`: three gaps, the longest 3 hours. `H00004` has no gap: `H00004,46,0,0.0,0.0`.

### 2.2 Compare three heart-rate alert rules

The clinical lead wants to know which rule should page a nurse when the heart rate reaches `ALERT_BPM` (120 bpm). A single reading can spike when the patient moves (a **motion artifact**); a whole hour, or three in a row, is harder to fake. Lecture 09's "Resampling Aggregations" and "Rolling Windows" cards, and Demo 2's independent practice 2 (the morning huddle list), compare the same choices. In the Task 2.2 cell:

1. Build `hourly`: each episode's hourly median heart rate from a grouped `resample("h").median()`, with `reset_index()`. An hour with no reading is `NaN`.
2. Add `rolling_3h`: within each episode, the mean of the current hour's median and the two before it, from `rolling(3)` inside a grouped `transform()` (Lecture 09's "Past-Only Panel Features" card shows the pattern). With the default `min_periods`, a window with an empty hour is `NaN`, so the rule needs three straight hours.
    - Expect: `hourly: (108293, 4)`.
3. Build `peaks`, one row per episode and one column per rule, each the episode's highest value: `any_reading` (the raw `heart_rate`), `hourly_median`, and `rolling_3h`. A grouped `max()` gives each one, and `pd.DataFrame()` of a dict of Series lines them up by episode, as Lecture 09's "Upsampling Choices" snippet does. The cell compares it with `ALERT_BPM` (`>=`) to make `flags`, and marks the escalated episodes with `isin()`.
4. Build `alert_rules`, one row per rule, with the columns `rule`, `episodes_flagged` (the flagged episodes), and `escalations_flagged` (the flagged episodes that escalated), as Lecture 04's `sum()` of a Boolean column counts `True` values. On `flags`, `sum()` counts each column at once, and `flags[escalated]` keeps the escalated episodes' rows.
    - Expect: `escalated: 179`, and `any_reading` flags 1,211 episodes, 138 of them escalations: almost every patient trips it, so a nurse paged by it learns to ignore it.
5. Save `alert_rules` to `ALERT_RULES_PATH` with `index=False`.

> **Checkpoint: `output/alert_rules.csv`**
> The header line `rule,episodes_flagged,escalations_flagged`, then 3 rows, one per rule. `any_reading,1211,138` is the first. The hourly median flags about one episode in 20 and the rolling rule about half as many. Compare how many of each rule's flags are escalations, and how many of the 179 escalations each rule misses: a fever or a fast atrial fibrillation raises the heart rate and settles at home, and some patients get worse without a fast heart rate.

### 2.3 Plot one escalated episode

This step saves nothing. Episode `H00839`, a 77-year-old treated for pneumonia, was escalated on February 20 after nearly five days at home. In the Task 2.3 cell, draw on `ax`, as Lecture 09's "Time-Series Plotting" snippet and Demo 3's step 7 do:

- the raw readings as faint gray dots (`'.'`, `alpha=0.3`),
- the episode's `hourly` medians and its `rolling_3h` as lines,
- a dashed line at `ALERT_BPM` with `ax.axhline()`,
- a title, `xlabel="Time (UTC)"`, a `ylabel` with the unit, and a legend.

Expect dots between about 85 and 110 bpm for all five days, with two isolated spikes above 150 that the hourly median ignores, and breaks in both lines where the feed stopped, including the February 17 outage. The lines never come near 120: this patient got worse without a fast heart rate, so no heart-rate rule would have paged a nurse before the escalation at 19:14 UTC on February 20.

## Task 3: Score each episode at 24 hours

The program's early-warning score runs once per episode, at the **prediction time**: the hour the episode was admitted (`admitted_at` rounded down with `.dt.floor("h")`) plus `SCORE_AFTER`, 24 hours. It may use only what was known by then.

### 3.1 Put the lab times on UTC and measure turnaround

The lab system writes New York wall-clock text. On the spring-forward night, March 8, 2026, clocks jumped from 02:00 to 03:00, yet a few results carry times from 02:00 to 02:59, which never happened in New York. In the Task 3.1 cell, `labs` starts as a copy of the lab table:

1. Convert both `collected_at` and `resulted_at`: parse with `pd.to_datetime()` and `format="%Y-%m-%d %H:%M"`, localize to `America/New_York` with `ambiguous="NaT"` and `nonexistent="NaT"`, then convert to `UTC`, as Lecture 09's "Flag Clock-Change Timestamps" snippet and Demo 3's step 2 do. A loop over the two column names saves writing it twice.
    - Expect: `set aside: {'collected_at': 0, 'resulted_at': 10}`, and the ten results printed below it, all stamped 02:00 to 02:29 on March 8.
2. Add `turnaround_h`, the hours from collection to result: subtract the two columns and divide `.dt.total_seconds()` by 3600 (Lecture 09's "Recorded Is Not Available" snippet).
3. Build `lab_turnaround`, one row per test, with `groupby("test", as_index=False)` and named aggregation: `orders` (every row, `"size"`), `timed_results` (the results with a time, `"count"` of `resulted_at`), and `median_turnaround_h` (the `"median"` of `turnaround_h`).
4. Save it to `TURNAROUND_PATH` with `index=False`.

> **Checkpoint: `output/lab_turnaround.csv`**
> The header line `test,orders,timed_results,median_turnaround_h`, then 5 rows, one per test. `Lactate` has 1,253 orders, 1,250 of them timed, and a median turnaround of about 3.5 hours: a sample drawn at home rides a courier to the lab.

### 3.2 Build past-only features at the prediction time

Each feature must use only hours that had ended by the prediction time. `hourly`'s row labeled 14:00 holds the readings from 14:00 to 14:59, which a score running at 14:00 has not seen yet, so every feature starts with a grouped `shift(1)`. Lecture 09's "Grouped Past-Only Windows" snippet and Demo 3's steps 3 to 5 build features the same way. In the Task 3.2 cell:

1. Build `scored` from `episodes`' `episode_id`, `admitted_at`, and `outcome`, and add `prediction_time` as defined above Task 3.1.
2. `features` starts as `hourly`'s `episode_id`, `recorded_at`, and `heart_rate`. Within each episode, add:
    - `prev_hour_hr`: the previous hour's median, from a grouped `shift(1)`.
    - `mean_prev_6h`: the mean of the six hours before, `shift(1)` then `rolling(6, min_periods=1)` inside a grouped `transform()`. `min_periods=1` keeps the mean when some of those hours are empty.
3. Merge `scored` onto `features` by `episode_id` (`validate="many_to_one"`), and keep the rows whose `recorded_at` equals their `prediction_time` as `score_features`. An episode that ended, or whose patch went silent for good, before its prediction time has no row then and gets no score.
    - Expect: `scored: 1359`.
4. Find each episode's latest lactate known by then: merge the `Lactate` rows of `labs` with `scored`'s `prediction_time`, keep the results with `resulted_at <=` the prediction time, sort by `resulted_at`, and keep each episode's last row with `groupby().tail(1)` (Lecture 08, Demo 3's step 5), its `value` renamed `last_lactate`. A result set aside as `NaT` fails the comparison, so it counts as not known.
    - Expect: `lactates: 1253  known: 708 from 618 episodes`.
5. Left-merge `last_lactate` onto `score_features` (`validate="one_to_one"`): a blank means no lactate had been reported, not a normal result.
6. Save the columns `episode_id`, `prev_hour_hr`, `mean_prev_6h`, and `last_lactate` to `FEATURES_PATH` with `index=False`.

> **Checkpoint: `output/score_features.csv`**
> The header line `episode_id,prev_hour_hr,mean_prev_6h,last_lactate`, then 1,359 rows. `H00001`, scored at `2026-01-06 14:00:00+00:00`, reads `H00001,86.5,78.75,` with no lactate yet; `H00007` has a `last_lactate` of `1.71`. `prev_hour_hr` is blank for 82 episodes whose previous hour was a gap.

### 3.3 Summarize a chronological holdout

The team picked the flag `mean_prev_6h >= FLAG_BPM` (100 bpm) while looking at the winter's episodes. A chronological holdout tests it on later ones, the way it would meet next month's patients (Lecture 09's "Chronological Holdout" snippet and Demo 3's step 6). In the Task 3.3 cell, `blocks` starts as a copy of `score_features`:

1. Add `block`: `earlier` where `prediction_time` is before `HOLDOUT_FROM` (`2026-03-01 00:00` UTC, `<`) and `later_holdout` otherwise, with `np.where()`.
2. Add three Boolean columns: `escalated` (`outcome` is `escalated`), `flagged` (`mean_prev_6h` is at least `FLAG_BPM`), and `flagged_escalation` (both, with `&`).
3. Build `holdout`, one row per block, with `groupby("block", as_index=False)` and named aggregation: `episodes` (`"size"`), then the `"sum"` of each Boolean column as `escalations`, `flagged`, and `flagged_escalations`.
4. Save it to `HOLDOUT_PATH` with `index=False`. The cell then prints each block's escalation rate and the share of flagged episodes that escalated.
    - Expect: an escalation rate of `0.108` in `earlier` and `0.085` in `later_holdout`, as the flu season fades. About 4 in 10 flagged episodes escalate in `earlier`, but March flags only 6 episodes, too few to show the rule holds.

> **Checkpoint: `output/holdout_summary.csv`**
> The header line `block,episodes,escalations,flagged,flagged_escalations`, then 2 rows. `earlier` holds 994 episodes and 107 escalations; `later_holdout` holds 365.

## Check your work

- Run the checks any time, even partway through: `python3 check_assignment.py` in the terminal (with the environment active) prints your score and what to fix.
- When **Restart** and **Run All** finish without errors, the notebook's last cell runs the same checks.
- Both use the latest checks from the course repository, the same checks GitHub runs on each push. Commit `assignment.ipynb` and the `output/` files, then push.
- Expect: a passing run prints `Score: 100/100` and `All checks passed.`

What each check looks for: [CHECKS.md](CHECKS.md)
