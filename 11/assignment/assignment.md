# Assignment 11 Contract

This file defines the forecasting question, the rules every question follows, and each output file: its path, its first line (the header), its line count, and how its values are computed. The [README](README.md) says how the files are graded.

## Forecasting Question

For each weather station and cutoff hour, predict the air temperature in degrees Celsius exactly one elapsed hour later:

```text
(station_name, cutoff_timestamp_utc) -> air temperature at cutoff + 1 hour
```

The source timestamp is Chicago wall-clock time with no time zone written on it. Parse it with `pd.to_datetime()`, localize it with `.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT")`, then convert the accepted values to UTC with `.dt.tz_convert("UTC")` (Lecture 09). Use UTC for the panel, sorting, features, targets, and splits, so one hour always means one elapsed hour.

## Frozen Source

- Release: `data/chicago_beach_sensors_2022_2024.csv`
- SHA-256: `7209cddd9b80e9475f9af17169b935e1ac2ef4a7a32fb72963ad0566b3474139`
- Rows: 50,895
- Stations: Foster Weather Station and Oak Street Weather Station
- Period: local 2022-01-01 00:00 up to local 2025-01-01 00:00
- Provenance and release facts: `data/release_manifest.json`

Do not use live downloads or external data.

## Shared Rules

- Save every CSV in `output/` with `index=False` (only `q5_correlations.csv` keeps its row labels), with the columns listed in its first line. Select the column list explicitly before saving so no extra index or column slips in.
- A column whose name ends in `_utc` holds UTC times with their offset, such as `2024-07-01 05:00:00+00:00`; pandas writes that form for a UTC column.
- Sort as each checkpoint says. Rows are matched by their key when graded, so the sort order is for you, not the grade.
- The persistence baseline predicts that the next hour equals the current `air_temperature_c_t` (Lecture 10).
- MAE is the primary metric. Also report RMSE and R2, calculated from unrounded predictions.
- Choose one regressor from the pinned scikit-learn. Do not add model libraries such as XGBoost.
- Use an sklearn `Pipeline` whose `ColumnTransformer` applies `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` to `station_name` and `SimpleImputer(strategy="median")` to the numeric predictors (Lecture 10). Fit the pipeline on the fitting rows only.
- Set `random_state=217` and `n_jobs=1` when the chosen estimator has those parameters.
- Do not tune on test results. The model does not need to beat persistence.

## Fixed Predictors

Q4, Q6, and Q7 use these 19 predictors in this order:

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

Apply the boundaries to the **target time** (cutoff plus one hour). Each boundary is local midnight in `America/Chicago`; compare it with the UTC target times as `pd.Timestamp("2024-01-01", tz="America/Chicago")` (Lecture 10's "Splitting on Target Time" card).

| Split | Target local time |
| --- | --- |
| train | before `2024-01-01 00:00` |
| validation | from `2024-01-01 00:00` up to but not including `2024-07-01 00:00` |
| test | from `2024-07-01 00:00` up to but not including `2025-01-01 00:00` |

Finish model and feature choices in Q7 without reading any test row, label, prediction, or metric.

## Q1: Setup and Exploration

> **Checkpoint: `output/q1_release_audit.csv`**
> First line `check_name,expected,observed,passed`; 8 lines.

One row for each check, in this order: `release_filename`, `release_sha256`, `release_byte_size`, `row_count`, `column_count`, `column_names`, `source_timezone`. `expected` holds the manifest's value. `observed` holds the value you measure from the CSV file itself: its name, SHA-256, and size in bytes (`path.name`, `hashlib.sha256(path.read_bytes()).hexdigest()`, and `path.stat().st_size`, as in Lecture 05's "Fingerprint the source file" snippet), and the loaded table's rows, columns, and column names. In the `column_names` row, write both lists of names joined with `|` in file order, such as `"|".join(manifest["columns"])`. For `source_timezone`, observe `America/Chicago`, the zone you localize with. `passed` is True when the two agree.

> **Checkpoint: `output/q1_station_coverage.csv`**
> First line `station_name,expected_hours,observed_hours,missing_hours,coverage_pct,first_timestamp,last_timestamp`; 3 lines.

One row per station. `expected_hours` counts the elapsed UTC hours from local 2022-01-01 00:00 up to local 2025-01-01 00:00. `observed_hours` counts the station's rows whose timestamp localizes without becoming `NaT`. `missing_hours` is the difference, `coverage_pct` is observed over expected times 100, and `first_timestamp` and `last_timestamp` are the station's earliest and latest valid times.

> **Checkpoint: `output/q1_visualizations.png`**

One figure with at least two labeled panels: the distribution of one ordinary sensor, and a time-series preview for each station (a short slice or a daily summary reads better than every point).

## Q2: Data Cleaning

> **Checkpoint: `output/q2_cleaned_observations.csv`**
> First line `station_name,measurement_timestamp,air_temperature_c,wet_bulb_temperature_c,relative_humidity_pct,rain_intensity_mm_per_hour,interval_rain_mm,total_rain_mm,precipitation_type_code,wind_direction_deg,wind_speed_mps,maximum_wind_speed_mps,barometric_pressure_hpa,solar_radiation_w_m2,battery_voltage_v,measurement_timestamp_utc`; the 50,895 release rows less the `rows_rejected` count in your `q2_cleaning_audit.csv`, plus the header. Sort by `measurement_timestamp_utc`, then `station_name`.

Reject a row only when its station is not one of the two release stations or its timestamp cannot become one UTC instant. In this release, the rows this rule removes are ambiguous fall-back hours (01:00 on a November clock change), which the required localization turns into `NaT`. Keep every other row, but set a sensor value to missing when it breaks its rule:

| Column | Valid values |
| --- | --- |
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
| `solar_radiation_w_m2` | -20 through 1500; then change values from -20 up to but not including 0 to 0 |
| `battery_voltage_v` | 0 through 20 |

Bounds are inclusive (`.between(low, high)`). Turn unreadable sensor values into missing with `pd.to_numeric(..., errors="coerce")` before applying the rules. Do not fill, interpolate, or clip. The solar near-zero correction is the only value replaced by another number. `measurement_timestamp` may stay as the source text or hold the localized Chicago time; `measurement_timestamp_utc` holds the UTC time.

> **Checkpoint: `output/q2_cleaning_audit.csv`**
> First line `rule,affected_values,result`; 16 lines, one row per rule below.

| Row | `affected_values` counts | `result` |
| --- | --- | --- |
| The station and timestamp key rule | rows rejected | `rows_rejected` |
| One row for each of the 13 sensor columns in the table above | values that were present and broke that column's rule; a value already missing is not counted | `set_missing` |
| The solar near-zero correction | values changed to 0 | `set_to_zero` |

`rule` is your own short, unique description, such as `air_temperature_c outside -50 to 50`. A rule that changes nothing still gets its row, with 0.

> **Checkpoint: `output/q2_missingness.csv`**
> First line `station_name,column_name,missing_count,missing_pct`; 27 lines.

One row per station and sensor column (2 stations times the 13 sensor columns from `air_temperature_c` through `battery_voltage_v`), counted in your cleaned table after the rules. `missing_pct` is the missing count over that station's cleaned rows, times 100.

## Q3: Data Wrangling

> **Checkpoint: `output/q3_hourly_panel.csv`**
> First line `station_name,measurement_timestamp_utc,air_temperature_c,wet_bulb_temperature_c,relative_humidity_pct,rain_intensity_mm_per_hour,interval_rain_mm,total_rain_mm,precipitation_type_code,wind_direction_deg,wind_speed_mps,maximum_wind_speed_mps,barometric_pressure_hpa,solar_radiation_w_m2,battery_voltage_v,source_observed,hour,day_of_week,month`; 2 times the `expected_hours` in your `q1_station_coverage.csv`, plus the header. Sort by `measurement_timestamp_utc`, then `station_name`.

Build every station crossed with every elapsed UTC hour from local 2022-01-01 00:00 up to local 2025-01-01 00:00 (`pd.date_range(..., freq="h", inclusive="left")` and `pd.merge(..., how="cross")`, Lectures 09 and 06), then left-join the cleaned observations onto that grid by station and UTC hour. An hour with no source row keeps every sensor column missing; do not fill it. `source_observed` is True exactly where a cleaned source row exists (`indicator=True`, Lecture 06). `hour`, `day_of_week` (Monday is 0), and `month` come from the row's time converted to `America/Chicago`.

> **Checkpoint: `output/q3_panel_summary.csv`**
> First line `station_name,expected_hours,observed_hours,missing_hours,gap_runs,longest_gap_hours`; 3 lines.

One row per station, counted from the panel. A gap run is one or more consecutive hours with `source_observed` False; `gap_runs` counts the runs and `longest_gap_hours` is the longest run's length (Lecture 09's "Count Gap Runs per Patient" snippet).

## Q4: Feature Engineering

> **Checkpoint: `output/q4_features.csv`**
> First line `row_id,station_name,cutoff_timestamp_utc,target_timestamp_utc,target_air_temperature_c,model_eligible,air_temperature_c_t,relative_humidity_pct_t,interval_rain_mm_t,wind_speed_mps_t,maximum_wind_speed_mps_t,barometric_pressure_hpa_t,solar_radiation_w_m2_t,wind_direction_sin_t,wind_direction_cos_t,air_temperature_lag_1h_c,air_temperature_lag_24h_c,air_temperature_lag_168h_c,air_temperature_mean_past_24h_c,air_temperature_change_1h_c,target_hour_sin,target_hour_cos,target_day_of_year_sin,target_day_of_year_cos`; the same line count as `q3_hourly_panel.csv`: one row per panel row, eligible or not. Sort by `cutoff_timestamp_utc`, then `station_name`.

Each panel row is a cutoff: `cutoff_timestamp_utc` is its `measurement_timestamp_utc`. Sort the panel by station and time, then compute every feature within station with `groupby("station_name")` (Lecture 09):

| Column | Rule |
| --- | --- |
| `target_timestamp_utc` | The cutoff plus one hour |
| `target_air_temperature_c` | The station's panel air temperature one row later: `shift(-1)` |
| `model_eligible` | True when both the cutoff's and the target's air temperatures are observed |
| The seven `_t` columns | The panel value at the cutoff, from the column named before `_t` |
| `wind_direction_sin_t`, `wind_direction_cos_t` | `np.sin` and `np.cos` of `2 * np.pi * wind_direction_deg / 360`, the Lecture 10 cycle formula with a 360-degree cycle |
| `air_temperature_lag_1h_c`, `_lag_24h_c`, `_lag_168h_c` | Air temperature `shift(1)`, `shift(24)`, and `shift(168)` rows earlier |
| `air_temperature_mean_past_24h_c` | The mean of the cutoff row and the 23 rows before it, ignoring missing values: `rolling(24, min_periods=1).mean()` with no shift |
| `air_temperature_change_1h_c` | `air_temperature_c_t` minus `air_temperature_lag_1h_c` |
| `target_hour_sin`, `target_hour_cos` | With `hour` the target's `America/Chicago` hour (0 to 23): `np.sin` and `np.cos` of `2 * np.pi * hour / 24` |
| `target_day_of_year_sin`, `target_day_of_year_cos` | With `dayofyear` the target's `America/Chicago` day of the year (1 to 366): `np.sin` and `np.cos` of `2 * np.pi * (dayofyear - 1) / 366`, with 366 every year |
| `row_id` | The lowercase station name with each run of spaces replaced by `_`, then `_`, then the cutoff's UTC hour as `YYYYMMDDHH`, such as `oak_street_weather_station_2024070105` |

Keep ineligible rows in this file. Modeling in Q6 uses only the eligible ones, and the pipeline's imputer fills their missing predictors.

> **Checkpoint: `output/q4_feature_manifest.csv`**
> First line `feature_name,source,earliest_offset_hours,latest_offset_hours,role`; 20 lines, one per fixed predictor.

`source` is your short, nonblank description of what the feature reads. The offsets give the earliest and latest hour each feature reads, relative to the cutoff: 0 is the cutoff hour, -1 the hour before, and so on, so `latest_offset_hours` is never above 0. `station_name` and the target calendar features read no earlier hour, so their offsets are 0 and 0. `role` is `categorical` for `station_name` and `numeric` for the rest.

## Q5: Pattern Analysis

Use only the Q4 rows with `model_eligible` True whose target falls in the train period (target local time before 2024-01-01). Use this same subset for all three files.

> **Checkpoint: `output/q5_monthly_station_summary.csv`**
> First line `station_name,year,month,n_observed,mean_air_temperature_c,std_air_temperature_c,min_air_temperature_c,max_air_temperature_c`; 49 lines (2 stations times the 24 training months, plus the header).

One row per station and target local year and month, summarizing `target_air_temperature_c`: the count, mean, standard deviation (pandas' default `.std()`), minimum, and maximum.

> **Checkpoint: `output/q5_correlations.csv`**
> First line `feature,air_temperature_c_t,relative_humidity_pct_t,interval_rain_mm_t,wind_speed_mps_t,maximum_wind_speed_mps_t,barometric_pressure_hpa_t,solar_radiation_w_m2_t`; 8 lines.

The Pearson correlation matrix of those seven predictors, `training_rows[columns].corr()` (Lecture 07), saved with its row labels as the first column: `to_csv(path, index=True, index_label="feature")`.

> **Checkpoint: `output/q5_patterns.png`**

One labeled figure showing the training period's monthly and local-hour temperature patterns.

## Q6: Modeling Preparation

Use the eligible Q4 rows and the split boundaries above. Keep rows with missing predictors; Q7's imputer handles them.

> **Checkpoint: `output/q6_X_train.csv`, `output/q6_X_validation.csv`, `output/q6_X_test.csv`**
> First line `row_id,station_name,cutoff_timestamp_utc,target_timestamp_utc,air_temperature_c_t,relative_humidity_pct_t,interval_rain_mm_t,wind_speed_mps_t,maximum_wind_speed_mps_t,barometric_pressure_hpa_t,solar_radiation_w_m2_t,wind_direction_sin_t,wind_direction_cos_t,air_temperature_lag_1h_c,air_temperature_lag_24h_c,air_temperature_lag_168h_c,air_temperature_mean_past_24h_c,air_temperature_change_1h_c,target_hour_sin,target_hour_cos,target_day_of_year_sin,target_day_of_year_cos`; that split's `n_rows` in your `q6_split_summary.csv`, plus the header.

> **Checkpoint: `output/q6_y_train.csv`, `output/q6_y_validation.csv`, `output/q6_y_test.csv`**
> First line `row_id,target_air_temperature_c`; the same line count as the matching X file.

Sort each split by `target_timestamp_utc`, then `station_name`, and save X and y from the same sorted rows so their `row_id` values line up.

> **Checkpoint: `output/q6_split_summary.csv`**
> First line `split,n_rows,target_start,target_end,n_features`; 4 lines (`train`, `validation`, `test`).

`n_rows` counts the split's rows, `target_start` and `target_end` are its earliest and latest `target_timestamp_utc`, and `n_features` counts all 19 fixed predictors, including `station_name`.

## Q7: Modeling

Fit candidates on the training rows and use the validation rows to freeze one final pipeline. Do not open any test file in this phase. [Lecture 10](https://github.com/christopherseaman/datasci_217/blob/main/10/README.md) and its [Demo 2](https://github.com/christopherseaman/datasci_217/blob/main/10/demo/demo2_sklearn_prediction.md) show the pipeline and validation pattern.

> **Checkpoint: `output/q7_validation_predictions.csv`**
> First line `row_id,station_name,target_timestamp_utc,actual,persistence_prediction,model_prediction`; the same line count as `q6_X_validation.csv`.

`actual` is the validation target, `persistence_prediction` is `air_temperature_c_t`, and `model_prediction` is your fitted pipeline's prediction for the same row.

> **Checkpoint: `output/q7_validation_metrics.csv`**
> First line `model,mae,rmse,r2,n`; 3 lines, one for `persistence_baseline` and one for `student_model`, both over the same validation rows.

> **Checkpoint: `output/q7_model_spec.csv`**
> First line `estimator_module,estimator_class,parameters_json,feature_columns,random_state`; 2 lines.

One row: the module you imported the regressor from and its class name (`sklearn.linear_model` and `Ridge` for `from sklearn.linear_model import Ridge`), its settings from `model.get_params(deep=False)` (Lecture 10) written as text with `json.dumps()` (Lecture 07), the 19 fixed predictors joined with `|`, and `217`. Record the regressor itself, not the whole pipeline.

> **Checkpoint: `output/q7_permutation_importance.csv`**
> First line `feature,mean_mae_increase,std_mae_increase`; 20 lines, one per fixed predictor.

Run `permutation_importance(pipeline, X_validation, y_validation, scoring="neg_mean_absolute_error", n_repeats=10, random_state=217)` on the fitted pipeline (Lecture 10). Save `importances_mean` as `mean_mae_increase` and `importances_std` as `std_mae_increase`. With this scorer, a positive value means shuffling that feature increased MAE.

## Q8: Results

Keep the Q7 choice: build the same regressor with the parameters in `q7_model_spec.csv`, put it in the same pipeline, fit it on train plus validation, and predict the test rows once.

> **Checkpoint: `output/q8_test_predictions.csv`**
> First line `row_id,station_name,target_timestamp_utc,actual,persistence_prediction,model_prediction,model_error,model_absolute_error`; the same line count as `q6_X_test.csv`.

`model_error` is `model_prediction` minus `actual`, and `model_absolute_error` is its absolute value.

> **Checkpoint: `output/q8_test_metrics.csv`**
> First line `model,mae,rmse,r2,n`; 3 lines, as in Q7, over the test rows.

> **Checkpoint: `output/q8_station_metrics.csv`**
> First line `model,station_name,n,mae,rmse,r2`; 5 lines, one per model and station.

> **Checkpoint: `output/q8_final_visualizations.png`**

One multi-panel figure: a validation comparison of the baseline and the model, a test actual-versus-predicted view, and residual diagnostics.

## Q9: Writeup

> **Checkpoint: `report.md`**

Complete the root `report.md` with exactly these level-two headings, in order:

1. Executive Summary
2. Data and Cleaning
3. Patterns
4. Forecast Design
5. Model Results
6. Limitations

Under **Model Results**, keep the six-column metrics table with columns `Evaluation set`, `Model`, `MAE`, `RMSE`, `R2`, and `n`, and fill its four rows: the two `q7_validation_metrics.csv` rows labeled Validation, then the two `q8_test_metrics.csv` rows labeled Test. Keep the three image embeds:

```markdown
![Release exploration](output/q1_visualizations.png)
![Training patterns](output/q5_patterns.png)
![Final model results](output/q8_final_visualizations.png)
```

The report and the notebooks earn the 25 human-review points, 5 for each of five categories; the [README](README.md#completion-contract) says what each category reads and what earns full credit. The model does not need to beat persistence.
