# Assignment 11 checks

## Artifact checks

The exam totals 100 points: 75 points graded from your committed files after the deadline, 25 by human review. Your `output/` files are the proof of work: each check reads one file and compares its values with values recomputed from the frozen release in `data/`. Most files are split into several parts, so a wrong value costs only the part it belongs to.

| File | Check | Points |
| --- | --- | ---: |
| `output/q1_release_audit.csv` | Each of the seven checks with the manifest value in `expected`, your measured value in `observed`, and `passed` True | 2 |
| `output/q1_station_coverage.csv` | Each station's six values, in proportion to the values right | 2 |
| `output/q1_visualizations.png` | A PNG image | 1 |
| `output/q2_cleaned_observations.csv` | 2 for exactly the valid release rows; 2 for `measurement_timestamp_utc`; 2 for `solar_radiation_w_m2` and the other sensor columns whose values a range rule changes, in proportion to those right; 1 for the sensor columns no rule changes | 7 |
| `output/q2_cleaning_audit.csv` | The `affected_values` totals of the `rows_rejected`, `set_missing`, and `set_to_zero` rows | 2 |
| `output/q2_missingness.csv` | Every station and sensor column's missing count and percent | 1 |
| `output/q3_hourly_panel.csv` | 2 for exactly one row per station and hour; 2 for the 13 sensor columns; 2 for `source_observed`; 2 for `hour`, `day_of_week`, and `month`, in proportion to those right | 8 |
| `output/q3_panel_summary.csv` | Each station's five values, in proportion to the values right | 2 |
| `output/q4_features.csv` | 1 each for: the rows; `row_id`; `target_timestamp_utc`; `target_air_temperature_c`; `model_eligible`; the seven `_t` copies. 2 for the three lags, the 24-hour mean, and the 1-hour change, and 2 for the sines and cosines of wind direction, target hour, and target day of year, each in proportion to the columns right | 10 |
| `output/q4_feature_manifest.csv` | Each predictor's offsets, role, and a nonblank source, in proportion to the rows right | 2 |
| `output/q5_monthly_station_summary.csv` | Each station-month's five values, in proportion to the values right | 2 |
| `output/q5_correlations.csv` | The 49 correlations, in proportion to the values right | 2 |
| `output/q5_patterns.png` | A PNG image | 1 |
| `output/q6_X_*.csv` | 1 for each file's rows; 2 for the values, in proportion to the columns right across the three files | 5 |
| `output/q6_y_*.csv` | Each file's rows and target values, in proportion to the six parts right | 3 |
| `output/q6_split_summary.csv` | Each split's four values, in proportion to the values right | 2 |
| `output/q7_model_spec.csv` | 1 each for: one row naming a scikit-learn module and class; `parameters_json` as a dictionary with `random_state` 217 and `n_jobs` 1 where the estimator has them; the 19 predictors in `feature_columns`; `random_state` 217 | 4 |
| `output/q7_validation_predictions.csv` | 1 each for the rows and a finite `model_prediction` in every row; 2 for `station_name`, `target_timestamp_utc`, `actual`, and `persistence_prediction`, in proportion to those right | 4 |
| `output/q7_validation_metrics.csv` | The eight values, computed from your `q7_validation_predictions.csv`, in proportion to the values right | 2 |
| `output/q7_permutation_importance.csv` | 1 for one row per fixed predictor; 1 for finite values with a nonnegative `std_mae_increase` | 2 |
| `output/q8_test_predictions.csv` | 1 each for: the rows; a finite `model_prediction`; `model_error`; `model_absolute_error`. 2 for `station_name`, `target_timestamp_utc`, `actual`, and `persistence_prediction`, in proportion to those right | 6 |
| `output/q8_test_metrics.csv` | The eight values, computed from your `q8_test_predictions.csv`, in proportion to the values right | 2 |
| `output/q8_station_metrics.csv` | The 16 values, computed from your `q8_test_predictions.csv`, in proportion to the values right | 2 |
| `output/q8_final_visualizations.png` | A PNG image | 1 |

"In proportion" means the part's points times the share right, rounded down.

## How the files are read

- Line endings, a byte-order mark, spaces around a value or a header name, blank lines, and a missing final newline do not matter. Neither do column order, extra columns, a leading column of row numbers, or row order: rows are matched by their key (station and time, `row_id`, `split`, `model`, and so on).
- Numbers are compared as numbers, so `2`, `2.0`, and `2.00` are the same, and a value rounded to two decimals or more passes (percentages to one decimal). Counts must be exact.
- Times are compared as instants: `2024-07-01 05:00:00+00:00`, `2024-07-01T05:00:00Z`, and `2024-07-01 00:00:00-05:00` are the same hour. In `q1_station_coverage.csv` and `q6_split_summary.csv`, a time written without an offset may be either UTC or Chicago local time.
- `True`/`False`, `true`/`false`, `1`/`0`, and `yes`/`no` all read as booleans; an empty field, `NaN`, and `<NA>` all read as missing. Labels such as station names, `model`, `split`, `role`, and `result` may use any letter case.
- Also accepted: `missing_pct` as a fraction instead of a percent, the population standard deviation (`ddof=0`) instead of the sample one, `q5_correlations.csv` saved with an unnamed first column, and your regressor's name, such as `Ridge`, in place of `student_model`.
- Repeated column names are accepted when their values agree. Conflicting copies fail the affected value check rather than silently choosing one.
- A missing or extra row costs only the rows part; the value parts judge the rows you have, as long as at least half of the expected rows are there.
- A file built correctly from one of your own earlier files counts as right even when that earlier file has a mistake: for example, a Q3 panel joined from your Q2 table, Q4 features computed from your Q3 panel, Q6 splits taken from your Q4 file, and metrics computed from your prediction files. A mistake costs points once, where you made it.
- The model's accuracy is not graded: it does not need to beat persistence.

## Human review

Human review reads `report.md` and the notebooks, 5 points for each category:

| Category | Full credit (5) |
| --- | --- |
| Data and cleaning decisions | The release audit result and each station's coverage from your Q1 files; the UTC conversion and why the ambiguous fall-back rows are rejected; the sensor rules and why invalid values become missing rather than filled; the complete panel that keeps gaps visible; each with a reason that comes from this data; a Q1 figure whose labeled panels show a sensor distribution and each station's time series |
| Training patterns | One or two monthly or local-hour patterns with numbers from `q5_monthly_station_summary.csv` or `q5_correlations.csv`, stated as training-period results; a correlation described as a pattern, not as proof that a predictor helps; a figure that shows both patterns with titles and labeled axes |
| Forecast design and model choice | The next-hour target, the past-only predictors, the persistence baseline, the preprocessing fit inside the pipeline on training rows, and the split by target time, each with its reason; section 4.4 prints rows that show one lag and one target matching the panel; the 7.2 cell names the regressor you kept and why, from its validation results |
| Model results tied to evidence | The four table rows match your Q7 and Q8 metric files; the model compared with persistence on validation and test using the table's numbers; one station difference from `q8_station_metrics.csv`; the test results reported, not used to tune; a figure with its three views, titles, and labeled axes |
| Summary, limitations, and reproducibility | The summary states the question, the data, the model, and one test result; the limitations are specific to this release and design (two stations, sensor gaps, one test period); every notebook runs top to bottom with its outputs saved |

Missing or unclear parts earn 2 to 4. A category whose `report.md` section still holds its bracketed placeholder earns 0.
