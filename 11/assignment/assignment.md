# Assignment Contract

## Forecasting Question

For each weather station and cutoff UTC hour, predict the air temperature in degrees Celsius exactly one elapsed hour later:

```text
(station_name, cutoff_timestamp_utc) -> air temperature at cutoff + 1 hour
```

The source timestamp is local civil time in `America/Chicago`. Parse it as a naive timestamp, localize with `ambiguous="NaT"` and `nonexistent="NaT"`, then convert accepted values to UTC. Use UTC for panel construction, sorting, features, targets, and splits so that one hour always means one elapsed hour.

## Frozen Source

- Release: `data/chicago_beach_sensors_2022_2024.csv`
- SHA-256: `7209cddd9b80e9475f9af17169b935e1ac2ef4a7a32fb72963ad0566b3474139`
- Rows: 50,895
- Stations: Foster Weather Station and Oak Street Weather Station
- Period: local 2022-01-01 through local 2025-01-01
- Provenance and release facts: `data/release_manifest.json`

Do not use live downloads or external data.

## Shared Rules

- Write CSV artifacts to `output/` with `index=False` and exactly the listed columns in the listed order.
- Write UTC timestamps as parseable ISO 8601 values with an explicit UTC offset.
- Panel, feature, and Q6 X/y artifacts must follow their stated chronological/key sequences; summary, metric, correlation, importance, and feature-manifest rows are matched by their documented identifiers.
- The persistence baseline uses current `air_temperature_c_t` as the next-hour prediction.
- MAE is the primary metric. Also report RMSE and R2, calculated from unrounded predictions.
- Choose one regressor from pinned scikit-learn. Do not use XGBoost or add model libraries.
- Use an sklearn `Pipeline` with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for `station_name` and `SimpleImputer(strategy="median")` for numeric predictors. Fit both preprocessing steps from the fitting rows only.
- Set `random_state=217` and `n_jobs=1` when the selected estimator supports those parameters.
- Do not use external data, tune on test results, or claim that a performance threshold is required.

## Fixed Predictors

Q4 and Q6 use these predictors in this order:

```text
station_name
air_temperature_c_t
relative_humidity_pct_t
interval_rain_mm_t
wind_speed_mps_t
maximum_wind_speed_mps_t
barometric_pressure_hpa_t
solar_radiation_w_m2_t
wind_direction_sin_t
wind_direction_cos_t
air_temperature_lag_1h_c
air_temperature_lag_24h_c
air_temperature_lag_168h_c
air_temperature_mean_past_24h_c
air_temperature_change_1h_c
target_hour_sin
target_hour_cos
target_day_of_year_sin
target_day_of_year_cos
```

## Split Boundaries

Apply boundaries to the **target instant**, represented in UTC. These instants correspond to local midnight boundaries in `America/Chicago`.

| Split | Target-time rule |
|---|---|
| train | target local time before `2024-01-01 00:00:00` |
| validation | target local time from `2024-01-01 00:00:00` through before `2024-07-01 00:00:00` |
| test | target local time from `2024-07-01 00:00:00` through before `2025-01-01 00:00:00` |

Finish model and feature selection in Q7 without accessing test rows, labels, predictions, metrics, or error slices.

## Artifact Contract

### Q1: Setup and Exploration (7 points)

> **Checkpoint — `output/q1_release_audit.csv`**

Columns: `check_name`, `expected`, `observed`, `passed`.

Exactly seven rows, in this order: `release_filename`, `release_sha256`, `release_byte_size`, `row_count`, `column_count`, `column_names`, `source_timezone`. Read expected values from the manifest and independently observe release facts. Join column names with `|` in file order.

> **Checkpoint — `output/q1_station_coverage.csv`**

Columns: `station_name`, `expected_hours`, `observed_hours`, `missing_hours`, `coverage_pct`, `first_timestamp`, `last_timestamp`.

Use the full local release window for expected hours and observed, valid localized station-hour keys for observed hours. Include each station once.

> **Checkpoint — `output/q1_visualizations.png`**

One readable figure containing an ordinary sensor distribution and a station time-series preview, with labels.

### Q2: Data Cleaning (9 points)

> **Checkpoint — `output/q2_cleaned_observations.csv`**

Columns are the exact 15 release columns, in release order, followed by `measurement_timestamp_utc`.

Reject rows with an invalid/unparseable station-timestamp key and the six ambiguous fall-back rows produced by the required localization. Retain otherwise valid rows, but set sensor values outside these rules to missing:

| Column | Valid values |
|---|---|
| `air_temperature_c` | -50 through 50 |
| `wet_bulb_temperature_c` | -50 through 50 |
| `relative_humidity_pct` | 0 through 100 |
| `rain_intensity_mm_per_hour` | 0 through 300 |
| `interval_rain_mm` | 0 through 100 |
| `total_rain_mm` | 0 through 2000 |
| `precipitation_type_code` | 0, 40, 60, or 70 |
| `wind_direction_deg` | 0 through 359 |
| `wind_speed_mps` | 0 through 75 |
| `maximum_wind_speed_mps` | 0 through 100 |
| `barometric_pressure_hpa` | 850 through 1100 |
| `solar_radiation_w_m2` | -20 through 1500; change values from -20 through below 0 to 0 |
| `battery_voltage_v` | 0 through 20 |

Bounds are inclusive. Coerce unparseable sensor values to missing before applying the rules. Do not interpolate, fill, or clip outliers. The solar near-zero correction is the only value replacement.

> **Checkpoint — `output/q2_cleaning_audit.csv`**

Columns: `rule`, `affected_values`, `result`. Include a clear row for timestamp/key rejection and each sensor rule so every cleaning decision is auditable.

The `rule` value may be any concise, unique, nonblank description. Grading checks the required result categories and their affected counts, not exact prose in `rule`.

> **Checkpoint — `output/q2_missingness.csv`**

Columns: `station_name`, `column_name`, `missing_count`, `missing_pct`. Report every sensor measurement column for both stations; each station/column key must be unique.

### Q3: Data Wrangling (11 points)

> **Checkpoint — `output/q3_hourly_panel.csv`**

Columns: `station_name`, `measurement_timestamp_utc`, the 13 cleaned sensor measurement columns in release order, then `source_observed`, `hour`, `day_of_week`, `month`.

Build every station crossed with every elapsed UTC hour from local `2022-01-01 00:00:00` inclusive through local `2025-01-01 00:00:00` exclusive. Join cleaned observations to that grid. Structural gaps remain missing in all sensor columns; do not fill them. `source_observed` is true only when a cleaned source row exists. Calendar columns come from the corresponding `America/Chicago` local instant, with Monday equal to 0.

> **Checkpoint — `output/q3_panel_summary.csv`**

Columns: `station_name`, `expected_hours`, `observed_hours`, `missing_hours`, `gap_runs`, `longest_gap_hours`. A gap run is a consecutive sequence of unobserved elapsed UTC hours. Each station key must be unique.

### Q4: Feature Engineering (14 points)

> **Checkpoint — `output/q4_features.csv`**

Exact columns:

```text
row_id,station_name,cutoff_timestamp_utc,target_timestamp_utc,
target_air_temperature_c,model_eligible,air_temperature_c_t,
relative_humidity_pct_t,interval_rain_mm_t,wind_speed_mps_t,
maximum_wind_speed_mps_t,barometric_pressure_hpa_t,
solar_radiation_w_m2_t,wind_direction_sin_t,wind_direction_cos_t,
air_temperature_lag_1h_c,air_temperature_lag_24h_c,
air_temperature_lag_168h_c,air_temperature_mean_past_24h_c,
air_temperature_change_1h_c,target_hour_sin,target_hour_cos,
target_day_of_year_sin,target_day_of_year_cos
```

Calculate every feature separately within station on the complete panel. `target_timestamp_utc` is cutoff plus one elapsed hour. The target is that exact next-hour panel value. The 24-hour rolling mean includes the cutoff and prior 23 rows, ignores missing values, and uses `min_periods=1`.

For cyclic target-calendar features, convert the target instant to `America/Chicago`. Let `hour` be its integer hour from 0 through 23 and `dayofyear` be its calendar day number from 1 through 366. Use exactly:

```text
hour_angle = 2 * pi * hour / 24
target_hour_sin = sin(hour_angle)
target_hour_cos = cos(hour_angle)
day_of_year_angle = 2 * pi * (dayofyear - 1) / 366
target_day_of_year_sin = sin(day_of_year_angle)
target_day_of_year_cos = cos(day_of_year_angle)
```

A row is `model_eligible` if and only if the current and exact next-hour air temperatures are both observed. Do not drop ineligible rows from Q4. Form `row_id` as the lowercase station slug joined by underscores, followed by `_` and cutoff `YYYYMMDDHH` in UTC.

> **Checkpoint — `output/q4_feature_manifest.csv`**

Columns: `feature_name`, `source`, `earliest_offset_hours`, `latest_offset_hours`, `role`. Include one uniquely identified row for each fixed predictor. Use a clear source description and role (`categorical` or `numeric`). Every `latest_offset_hours` must be at most 0.

The `source` value may be concise, nonblank student text. Grading fixes feature names, offsets, and roles, but does not require exact prose in `source`.

### Q5: Pattern Analysis (7 points)

Use only Q4 rows with `model_eligible=True` whose target local time is before `2024-01-01 00:00:00` in `America/Chicago`. Use this same subset for the monthly summary, correlations, and pattern figure.

> **Checkpoint — `output/q5_monthly_station_summary.csv`**

Columns: `station_name`, `year`, `month`, `n_observed`, `mean_air_temperature_c`, `std_air_temperature_c`, `min_air_temperature_c`, `max_air_temperature_c`. Summarize observed target temperatures with unique station/year/month keys.

> **Checkpoint — `output/q5_correlations.csv`**

A square Pearson correlation matrix whose row-label identities and columns are exactly: `air_temperature_c_t`, `relative_humidity_pct_t`, `interval_rain_mm_t`, `wind_speed_mps_t`, `maximum_wind_speed_mps_t`, `barometric_pressure_hpa_t`, `solar_radiation_w_m2_t`. Save the row labels as the first CSV column, named `feature`, using `to_csv(..., index=True, index_label="feature")`.

> **Checkpoint — `output/q5_patterns.png`**

One readable figure showing training-only monthly and local-hour temperature patterns.

### Q6: Modeling Preparation (11 points)

Use eligible Q4 rows and the target-time split boundaries.

> **Checkpoint — `output/q6_X_train.csv`**

> **Checkpoint — `output/q6_X_validation.csv`**

> **Checkpoint — `output/q6_X_test.csv`**

Columns: `row_id`, `station_name`, `cutoff_timestamp_utc`, `target_timestamp_utc`, followed by all fixed predictors except the already-listed `station_name`, in fixed order.

> **Checkpoint — `output/q6_y_train.csv`**

> **Checkpoint — `output/q6_y_validation.csv`**

> **Checkpoint — `output/q6_y_test.csv`**

Columns: `row_id`, `target_air_temperature_c`.

X and y files must have matching unique IDs and row order within each split. Sort by target UTC, then station.

> **Checkpoint — `output/q6_split_summary.csv`**

Columns: `split`, `n_rows`, `target_start`, `target_end`, `n_features`. Include one row for each of train, validation, and test. Ranges are observed inclusive ranges and `n_features` counts all fixed predictors, including station.

### Q7: Modeling (13 points)

Use training rows to fit candidates and validation rows to select one final model. Do not access any test artifact in this phase.
Review the pipeline and validation pattern from [Lecture 10](../../10/README.md) and [Lecture 10 Demo 2](../../10/demo/demo2_sklearn_prediction.ipynb) before starting this phase.

> **Checkpoint — `output/q7_model_spec.csv`**

Columns: `estimator_module`, `estimator_class`, `parameters_json`, `feature_columns`, `random_state`. Exactly one row. Record the regressor module and class, a JSON object of its shallow parameters, fixed feature names joined by `|`, and `217`. The fitted object used for prediction must be a pipeline with the required train-fitted preprocessing.

> **Checkpoint — `output/q7_validation_predictions.csv`**

Columns: `row_id`, `station_name`, `target_timestamp_utc`, `actual`, `persistence_prediction`, `model_prediction`.

> **Checkpoint — `output/q7_validation_metrics.csv`**

Columns: `model`, `mae`, `rmse`, `r2`, `n`. Include one uniquely identified row for each of `persistence_baseline` and `student_model`. Use identical validation rows.

> **Checkpoint — `output/q7_permutation_importance.csv`**

Columns: `feature`, `mean_mae_increase`, `std_mae_increase`. Calculate validation permutation importance through the fitted pipeline with `scoring="neg_mean_absolute_error"`, `n_repeats=10`, and `random_state=217`. Save `importances_mean` as `mean_mae_increase` and `importances_std` as `std_mae_increase` for each uniquely identified fixed feature. With sklearn's negative-MAE scorer, a positive importance means that permutation increased MAE.

### Q8: Results (13 points)

Freeze the Q7 choice, recreate its pipeline, refit on training plus validation, and evaluate test exactly once.

> **Checkpoint — `output/q8_test_predictions.csv`**

Columns: `row_id`, `station_name`, `target_timestamp_utc`, `actual`, `persistence_prediction`, `model_prediction`, `model_error`, `model_absolute_error`. Error is model prediction minus actual.

> **Checkpoint — `output/q8_test_metrics.csv`**

Columns: `model`, `mae`, `rmse`, `r2`, `n`. Include one uniquely identified row for each of `persistence_baseline` and `student_model`.

> **Checkpoint — `output/q8_station_metrics.csv`**

Columns: `model`, `station_name`, `n`, `mae`, `rmse`, `r2`. Include each unique model/station combination.

> **Checkpoint — `output/q8_final_visualizations.png`**

One readable multi-panel figure containing a validation baseline/model comparison, test actual-versus-predicted view, and residual diagnostics.

### Q9: Writeup (15 human-review points)

> **Checkpoint — `report.md`**

Complete root `report.md` with exactly these level-two headings, in order:

1. Executive Summary
2. Data and Cleaning
3. Patterns
4. Forecast Design
5. Model Results
6. Limitations

Include the accepted six-column metrics table with columns `Evaluation set`, `Model`, `MAE`, `RMSE`, `R2`, and `n`. It has exactly four data rows: the two Q7 validation metric rows followed by the two Q8 test metric rows. Also include these three image embeds with valid relative paths:

```markdown
![Release exploration](output/q1_visualizations.png)
![Training patterns](output/q5_patterns.png)
![Final model results](output/q8_final_visualizations.png)
```

Q9's automated check verifies the required structure, numeric table, and image
links as readiness feedback. The 15 points are human review; model performance
is not scored.

| Criterion | Points |
|---|---:|
| Justified analysis, cleaning, and forecast decisions | 5 |
| Interpretation tied to reported evidence | 5 |
| Limitations and clear communication | 5 |

## Check Your Work

Run `python check_assignment.py` from this directory. See the [completion contract and submission steps](README.md#check-your-work). Q1–Q8 total 85 automated points; Q9 is worth 15 human-review points.
