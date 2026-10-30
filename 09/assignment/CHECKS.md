# Assignment 09 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/3  daily feed: columns
         output/daily_feed.csv is missing; run the Task 1.2 cell to write it, then commit it.
```

A clean run ends with:

```text
[PASS]  4/4  holdout summary: flagged values

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, and a mistake is charged once. A missing column, or one saved under another name with the right values, costs only the columns check; values in the other columns are still checked. An empty table or one with no recognizable rows earns no value points.
- The checks read only the six CSV files in `output/`, never your notebook, so any way of producing correct files counts.
- Spacing, line endings, quoting, column order, and row order never cost points, and cells may be separated by commas, semicolons, or tabs.
- Numbers are compared as numbers (`4`, `4.0`, `4.00`); a mean may be rounded to one decimal, not to a whole number.
- A date in `daily_feed.csv` may be written with or without its time and offset: `2026-02-17`, `2026-02-17 00:00:00-05:00`, and `02/17/2026` match.
- Episode IDs, test names, rule names, block labels, and column names are compared in any letter case, and `later holdout` matches `later_holdout`.
- An empty cell may be written empty or as `NaN`.
- A leading column of row numbers, which `to_csv()` writes when `index=False` is left out, is ignored.
- `episode_coverage.csv` also counts when its hours were counted from `heart_rate`, which skips the readings with no heart rate.
- The heart-rate features in `score_features.csv` also count when built from hourly means, so a mean in place of the median costs points only in Task 2.2. `holdout_summary.csv` also counts when its numbers follow from your own `score_features.csv`, so a mistake there costs points only in Task 3.2.
- An episode with a lactate result whose time Task 3.1 set aside as `NaT` has a `last_lactate` that depends on how that time was handled, so its `last_lactate` is not checked.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/daily_feed.csv` | Its columns are the three in the Task 1.2 header line. | daily feed: columns | 3 |
| `output/daily_feed.csv` | It holds each New York calendar day from January 5 to April 1 once. | daily feed: one row per New York day | 4 |
| `output/daily_feed.csv` | Each `readings` counts the rows the feed sent that day. | daily feed: readings values | 5 |
| `output/daily_feed.csv` | Each `episodes` counts the episodes that sent a reading that day. | daily feed: episodes values | 4 |
| `output/episode_coverage.csv` | Its columns are the five in the Task 2.1 header line. | episode coverage: columns | 3 |
| `output/episode_coverage.csv` | It holds each of the 1,400 episodes once. | episode coverage: one row per episode | 4 |
| `output/episode_coverage.csv` | Each `grid_hours` counts the episode's hourly grid and each `gap_hours` its hours with no reading. | episode coverage: grid_hours and gap_hours values | 5 |
| `output/episode_coverage.csv` | Each `gap_runs` counts the episode's runs of empty hours and `longest_gap_h` is the longest, 0 when there is none. | episode coverage: gap_runs and longest_gap_h values | 5 |
| `output/alert_rules.csv` | Its columns are the three in the Task 2.2 header line. | alert rules: columns | 3 |
| `output/alert_rules.csv` | It holds `any_reading`, `hourly_median`, and `rolling_3h` once each. | alert rules: one row per rule | 3 |
| `output/alert_rules.csv` | Each `episodes_flagged` counts the episodes whose highest value under the rule is at least 120. | alert rules: episodes_flagged values | 5 |
| `output/alert_rules.csv` | Each `escalations_flagged` counts the flagged episodes that escalated. | alert rules: escalations_flagged values | 5 |
| `output/lab_turnaround.csv` | Its columns are the four in the Task 3.1 header line. | lab turnaround: columns | 3 |
| `output/lab_turnaround.csv` | It holds each of the 5 tests once. | lab turnaround: one row per test | 2 |
| `output/lab_turnaround.csv` | Each `orders` counts the test's rows and `timed_results` those whose result time was not set aside. | lab turnaround: orders and timed_results values | 5 |
| `output/lab_turnaround.csv` | Each `median_turnaround_h` is the median hours from collection to result, both in UTC. | lab turnaround: median_turnaround_h values | 5 |
| `output/score_features.csv` | Its columns are the four in the Task 3.2 header line. | score features: columns | 3 |
| `output/score_features.csv` | It holds each of the 1,359 episodes with an hourly row at its prediction time once. | score features: one row per scored episode | 5 |
| `output/score_features.csv` | Each `prev_hour_hr` is the episode's hourly heart rate in the hour before the prediction time. | score features: prev_hour_hr values | 5 |
| `output/score_features.csv` | Each `mean_prev_6h` is the mean of the six hours before the prediction time. | score features: mean_prev_6h values | 6 |
| `output/score_features.csv` | Each `last_lactate` is the latest lactate reported by the prediction time, empty when none was. | score features: last_lactate values | 6 |
| `output/holdout_summary.csv` | Its columns are the five in the Task 3.3 header line. | holdout summary: columns | 2 |
| `output/holdout_summary.csv` | It holds `earlier` and `later_holdout` once each. | holdout summary: one row per block | 2 |
| `output/holdout_summary.csv` | Each block's `episodes` and `escalations` count its scored episodes split at `2026-03-01 00:00` UTC. | holdout summary: episodes and escalations values | 3 |
| `output/holdout_summary.csv` | Each block's `flagged` and `flagged_escalations` count the episodes with `mean_prev_6h` of at least 100, and those that escalated. | holdout summary: flagged values | 4 |

Extra files are ignored.
