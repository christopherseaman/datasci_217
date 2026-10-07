# Assignment 10 checks

Each check prints `PASS` or `FIX` and the points it earned, and a `FIX` says what to fix on the line beneath it. When the next checks need the same fix, such as a missing file, they say `(same fix as above)`. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/2  coefficients: columns
         output/ols_coefficients.csv is missing; run the Task 1.1 cell to write it, then commit it.
```

Below the score, `Left to fix` lists the checks still failing and the points they are worth. Fix what they name, rerun the notebook and then the checks, and repeat until every check passes. A clean run ends with:

```text
[PASS]  2/2  readmission metrics: recall values

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only that check's points.
- A missing column costs the columns check once; values in the remaining columns are still checked. An empty table or one with no recognizable rows earns no value points.
- Spacing, line endings, quoting, column order, and row order never cost points.
- Numbers are compared as numbers, so `2`, `2.0`, and `2.00` are the same value. Any number may be rounded to one or two decimals, as `3.3` or `3.34` for an MAE of 3.3379. Accuracy, precision, recall, and R² may also be written as percents, as `85%` for 0.85.
- Labels, IDs, and column names are compared in any letter case, with spaces and underscores alike.
- Timestamps are compared as instants in UTC.
- A leading column of row numbers, which `to_csv()` writes when `index=False` is left out, is ignored.
- The test metrics and predictions may come from the pipeline fitted on the training rows or refitted on training plus validation rows.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/ols_coefficients.csv` | Its columns are the five in the Task 1.1 header line. | coefficients: columns | 2 |
| `output/ols_coefficients.csv` | It holds `Intercept`, `age`, and `bmi`, once each. | coefficients: one row per term | 2 |
| `output/ols_coefficients.csv` | Each `coef` is the fitted coefficient. | coefficients: coef values | 3 |
| `output/ols_coefficients.csv` | Each `std_err` is the coefficient's standard error. | coefficients: std_err values | 2 |
| `output/ols_coefficients.csv` | Each `ci_lower` and `ci_upper` bounds the 95% confidence interval. | coefficients: confidence interval values | 3 |
| `output/new_patient_intervals.csv` | Its columns are those in the Task 1.2 header line; `mean_se` is optional. | new-patient intervals: columns | 2 |
| `output/new_patient_intervals.csv` | It holds one row, for age 60 and BMI 31.0. | new-patient intervals: one row for the new patient | 1 |
| `output/new_patient_intervals.csv` | `mean` is the fitted SBP for the new patient. | new-patient intervals: mean | 2 |
| `output/new_patient_intervals.csv` | `mean_ci_lower` and `mean_ci_upper` bound the mean-response interval. | new-patient intervals: mean-response interval | 2 |
| `output/new_patient_intervals.csv` | `obs_ci_lower` and `obs_ci_upper` bound the prediction interval. | new-patient intervals: prediction interval | 2 |
| `output/ols_residuals.csv` | Its columns are the four in the Task 1.3 header line. | residuals: columns | 2 |
| `output/ols_residuals.csv` | It holds `P01` to `P20`, once each. | residuals: one row per patient | 2 |
| `output/ols_residuals.csv` | Each `observed` is the patient's `sbp`. | residuals: observed values | 1 |
| `output/ols_residuals.csv` | Each `fitted` is the patient's fitted SBP. | residuals: fitted values | 2 |
| `output/ols_residuals.csv` | Each `residual` is observed minus fitted. | residuals: residual values | 2 |
| `output/residuals_vs_fitted.png` | It is a PNG image. | residual plot: PNG image | 5 |
| `output/availability_decisions.csv` | Its columns are the four in the Task 2.1 header line. | availability: columns | 2 |
| `output/availability_decisions.csv` | It holds each of the five candidates once. | availability: one row per candidate | 2 |
| `output/availability_decisions.csv` | Each `hours_after_visit` matches `data/feature_availability.csv`. | availability: hours_after_visit values | 1 |
| `output/availability_decisions.csv` | `available` is true exactly for the features known when the visit ends. | availability: available values | 3 |
| `output/availability_decisions.csv` | Each `decision` keeps the available features and excludes the rest. | availability: decision values | 3 |
| `output/split_summary.csv` | Its columns are the four in the Task 2.2 header line. | split summary: columns | 2 |
| `output/split_summary.csv` | It holds `train`, `validation`, and `test`, once each. | split summary: one row per partition | 2 |
| `output/split_summary.csv` | Each `row_count` counts the partition's visits, split on `followup_time`. | split summary: row_count values | 4 |
| `output/split_summary.csv` | Each `first_target_time` is the partition's earliest `followup_time`. | split summary: first_target_time values | 3 |
| `output/split_summary.csv` | Each `last_target_time` is the partition's latest `followup_time`. | split summary: last_target_time values | 3 |
| `output/validation_metrics.csv` | Its columns are the four in the Task 3.1 header line. | validation metrics: columns | 2 |
| `output/validation_metrics.csv` | It holds `mean_baseline` and `linear_pipeline`, once each. | validation metrics: one row per approach | 2 |
| `output/validation_metrics.csv` | Each `mae` is the approach's validation MAE. | validation metrics: mae values | 3 |
| `output/validation_metrics.csv` | Each `rmse` is the approach's validation RMSE. | validation metrics: rmse values | 3 |
| `output/validation_metrics.csv` | Each `r2` is the approach's validation R². | validation metrics: r2 values | 3 |
| `output/test_metrics.csv` | Its columns are the four in the Task 3.2 header line. | test metrics: columns | 2 |
| `output/test_metrics.csv` | It holds one row, `linear_pipeline`. | test metrics: one row for the frozen approach | 1 |
| `output/test_metrics.csv` | `mae` is the frozen pipeline's test MAE. | test metrics: mae value | 2 |
| `output/test_metrics.csv` | `rmse` is the frozen pipeline's test RMSE. | test metrics: rmse value | 2 |
| `output/test_metrics.csv` | `r2` is the frozen pipeline's test R². | test metrics: r2 value | 2 |
| `output/test_predictions.csv` | Its columns are the four in the Task 3.2 header line. | test predictions: columns | 2 |
| `output/test_predictions.csv` | It holds `V39` to `V48`, once each. | test predictions: one row per test visit | 2 |
| `output/test_predictions.csv` | Each `followup_time` and `sbp_followup` matches `data/followup_visits.csv`. | test predictions: followup_time and sbp_followup values | 2 |
| `output/test_predictions.csv` | Each `predicted_sbp` is the frozen pipeline's prediction. | test predictions: predicted_sbp values | 3 |
| `output/readmission_metrics.csv` | Its columns are the four in the Task 3.3 header line. | readmission metrics: columns | 2 |
| `output/readmission_metrics.csv` | It holds `model_flag` and `never_flag`, once each. | readmission metrics: one row per flag | 1 |
| `output/readmission_metrics.csv` | Each `accuracy` is the flag's accuracy. | readmission metrics: accuracy values | 2 |
| `output/readmission_metrics.csv` | Each `precision` is the flag's precision, 0 when it flags no one. | readmission metrics: precision values | 2 |
| `output/readmission_metrics.csv` | Each `recall` is the flag's recall. | readmission metrics: recall values | 2 |

Extra files are ignored.
