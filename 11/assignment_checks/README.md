# Assignment 11 checks (course-owned)

These checks grade the final exam. The handout in `11/assignment/` ships none of them, with no checker, workflow, or test, so its README names every output file with the question that makes it, its first line, and its line count instead. After the deadline, `uv run scripts/grade_submissions.py 11` clones each fork and runs `check_assignment.py` from this folder on the fork's committed files. To grade one submission by hand:

```bash
uv run --python 3.13 --with-requirements 11/assignment/requirements.txt \
    python 11/assignment_checks/check_assignment.py path/to/submission          # readable report
uv run --python 3.13 --with-requirements 11/assignment/requirements.txt \
    python 11/assignment_checks/check_assignment.py path/to/submission --json   # datasci217/grading-result/v1
```

The report covers the 85 points graded from files; the other 15 come from human review of `report.md` and the notebooks, as the handout README's Completion contract describes. The command exits 0 only when every check passes, 1 when any does not, and 2 when the supplied release is missing or changed, so judge a run by its JSON, not its exit status. A run takes about 15 seconds.

The checks read only the CSV and PNG files in the submission's `output/` and its `report.md`. They never import, run, or read submitted code, and they ignore the submission's own `data/`: every expected value is recomputed from `11/assignment/data/` in this repository, after its SHA-256 is checked against the frozen release. They need NumPy and pandas at the versions `11/assignment/requirements.txt` pins.

## What each check scores

| Check | Points | Unit scored |
| --- | ---: | --- |
| `q1_release_audit.csv` | 2 | In proportion to the 7 checks with right `expected`, `observed`, and `passed` |
| `q1_station_coverage.csv` | 3 | In proportion to the 12 station values right |
| `q1_visualizations.png`, `q5_patterns.png`, `q8_final_visualizations.png` | 1 each | A PNG image at least 50 pixels on each side |
| `q2_cleaned_observations.csv: rows` | 2 | 1 for every valid release row present, 1 for no other, repeated, or unreadable row |
| `q2_cleaned_observations.csv: measurement_timestamp_utc` | 2 | In proportion to the rows right, rounded down |
| `q2_cleaned_observations.csv: solar_radiation_w_m2` | 1 | The column right in every row |
| `q2_cleaned_observations.csv: interval_rain_mm, wind_speed_mps, maximum_wind_speed_mps` | 2 | In proportion to the 3 columns right (the only range rules this release triggers) |
| `q2_cleaned_observations.csv: the other nine sensor columns` | 1 | All nine unchanged apart from their rules |
| `q2_cleaning_audit.csv` | 2 | In proportion to the 3 result totals (`rows_rejected`, `set_missing`, `set_to_zero`) right |
| `q2_missingness.csv` | 1 | All 26 station and column counts and percents right |
| `q3_hourly_panel.csv: rows` | 2 | As for the Q2 rows, over the station-hour grid |
| `q3_hourly_panel.csv: sensor columns` | 2 | In proportion to the 13 columns right |
| `q3_hourly_panel.csv: source_observed` | 2 | In proportion to the rows right |
| `q3_hourly_panel.csv: hour`, `day_of_week`, `month` | 1 each | The column right in every row |
| `q3_panel_summary.csv` | 2 | In proportion to the 10 station values right |
| `q4_features.csv`: rows, `row_id`, `target_timestamp_utc`, target, `model_eligible`, `_t` copies, wind, lags, 24-hour mean, 1-hour change, target hour, target day | 1 each | The column or column group right in every row present |
| `q4_feature_manifest.csv` | 2 | In proportion to the 19 predictors with right offsets and role and a nonblank source |
| `q5_monthly_station_summary.csv` | 3 | In proportion to the 240 station-month values right |
| `q5_correlations.csv` | 2 | In proportion to the 49 correlations right |
| `q6_X_*.csv: rows` | 3 | 1 per split with exactly its eligible rows |
| `q6_X_*.csv: values` | 3 | In proportion to the 63 split columns right |
| `q6_y_*.csv` | 3 | In proportion to the 6 parts (each split's rows and targets) right |
| `q6_split_summary.csv` | 2 | In proportion to the 12 split values right |
| `q7_model_spec.csv` | 4 | 1 each for the module and class, `parameters_json`, `feature_columns`, and `random_state` |
| `q7_validation_predictions.csv` | 5 | 1 each for the rows, the IDs, `actual`, `persistence_prediction`, and a finite `model_prediction` |
| `q7_validation_metrics.csv` | 2 | In proportion to the 8 metric values right |
| `q7_permutation_importance.csv` | 2 | 1 for the 19 features, 1 for finite values with a nonnegative standard deviation |
| `q8_test_predictions.csv` | 7 | As for Q7, plus 1 each for `model_error` and `model_absolute_error` |
| `q8_test_metrics.csv` | 2 | In proportion to the 8 metric values right |
| `q8_station_metrics.csv` | 3 | In proportion to the 16 metric values right |
| `report.md` structure | 0 | Notes for the human reviewer: headings, placeholders, the metrics table against the saved metric files, and the three image embeds |

"In proportion" means the check's points times the share right, rounded down. A check scores only its own file, so no check depends on another passing, and one wrong value costs only its own share. Every failure says what was expected, what was found (with the first differing row), and which question and section to fix.

What never costs points: line endings, a byte-order mark, spaces around cells or header names, blank lines, a missing final newline, column order, extra columns, a leading unnamed index column, row order (rows are matched by key), number format (values within 0.006, which admits two-decimal rounding; percentages within 0.051; counts exact), the letter case of labels, boolean spellings (`True`, `true`, `1`, `yes`), missing-value spellings (empty, `NaN`, `<NA>`), and the form of a time (any offset or `Z`; a naive `_utc` time is read as UTC, and a naive coverage or split-summary time may be UTC or Chicago local). Also accepted: the `column_names` audit row written as a printed list, `missing_pct` as a fraction, the population standard deviation, an unnamed first column of correlation labels (or none, in the fixed order), `parameters_json` as a Python dict literal, comma-separated `feature_columns`, and the regressor's name in place of `student_model` when it is the only other model label.

A missing or extra row costs only the rows check: the value checks compare the rows present, provided at least half of the expected rows are there. A downstream file also passes when it follows from the student's own upstream file: the Q3 panel from their Q2 table, Q4 features from their Q3 panel, the Q5 summaries and Q6 splits from their Q4 file, the Q7 and Q8 `actual` and `persistence_prediction` from their Q6 files, the Q2 missingness from their Q2 table, the Q3 summary from their panel, the split summary from their X files, and the metrics from their prediction files. So one mistake is charged once, where it was made. Model accuracy is never graded.

## Changing a check

Edit `grading.py`, keep the handout README's checklist and Completion contract and `assignment.md` in agreement with it, and rerun both tests:

```bash
uv run --python 3.13 --with-requirements 11/assignment/requirements.txt python 11/assignment_checks/_grader_selftest/run.py
uv run scripts/test_assignment_grading.py 11
```
