# Assignment 09 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/3  prepared vitals: columns
         output/prepared_vitals.csv is missing; run the Task 1.1 cell to write it, then commit it.
```

A clean run ends with:

```text
[PASS]  6/6  chronological blocks: block labels

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own. Later tables also accept the result of transforming your own prepared readings, so a mistake in Task 1.1 costs points once, not again downstream.
- The checks read only the six CSV files in `output/`, never your notebook, so any way of producing correct files counts.
- Spacing, line endings, quoting, column order, and row order never cost points.
- Numbers are compared as numbers (`84`, `84.0`, `84.00`); a mean may be rounded to one decimal.
- Timestamps are compared as instants: `2026-01-20 18:00:00+00:00`, `2026-01-20T18:00:00Z`, and `2026-01-20 13:00:00-05:00` are the same moment. A timestamp with no offset is read as UTC.
- Only Task 1.1 checks that `recorded_at` was converted to UTC; Task 3.2 checks its own lab times. If Task 1.1 skipped the conversion, later files are judged on the same clock, so their other values still count.
- `True` and `False` may be written in any letter case, or as `1` and `0`.
- Patient IDs, test names, block labels, and column names are compared in any letter case, and `later holdout` matches `later_holdout`.
- An empty cell may be written empty or as `NaN`.
- A leading column of row numbers, which `to_csv()` writes when `index=False` is left out, is ignored.
- `source_row` may be kept in or left out of `hourly_grid.csv`, `past_features.csv`, and `chronological_blocks.csv`.

## Completion contract

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
