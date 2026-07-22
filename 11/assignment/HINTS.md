# Hints and Troubleshooting

Use these nudges after trying the corresponding phase. They intentionally stop short of complete implementations.

## Paths and Handoffs

- Run notebooks from `11/assignment`, not the repository root.
- Create `output/` with `Path("output").mkdir(exist_ok=True)`.
- Start each notebook by loading the prior saved artifact. Do not depend on variables from another notebook's kernel.
- Before saving a CSV, select the contract column list explicitly. This catches accidental index columns, reordered fields, and extras.
- Use `index=False` except for the Q5 square correlation matrix, whose row labels must be saved.

## Q1: Release Audit and Coverage

- Read expected values from `data/release_manifest.json`, but compute observed hash, byte size, shape, and columns from the CSV itself.
- `hashlib.sha256()` can process bytes in chunks. Do not place the manifest hash directly in `observed`.
- Parse the source timestamps before localization. They are naive Chicago wall times.
- `.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT")` marks daylight-saving times that cannot identify one instant.
- Count unique valid station-hour keys for observed coverage. Construct expected hours from localized release endpoints converted to UTC.
- Plotting every point can hide patterns. A station-colored histogram and a short time-series slice or daily summary are ordinary, readable choices.

## Q2: Cleaning Without Inventing Data

- Keep a copy of each pre-rule series when counting affected values.
- Use `pd.to_numeric(..., errors="coerce")` at the data boundary.
- Inclusive bounds can be checked with `.between(lower, upper, inclusive="both")`.
- Precipitation type is a membership rule, not a numeric interval.
- Solar values need two separate masks: valid negative near-zero values become 0, while values outside the full valid range become missing.
- An observation with one invalid sensor value still contains useful measurements. Reject invalid keys, not rows with sensor missingness.
- There should be six ambiguous fall-back timestamp rows. If your count differs, verify that you localized naive timestamps rather than parsing them as UTC.

## Q3: Complete Elapsed-Hour Panel

- Convert local start/end boundaries to UTC, then use a left-inclusive UTC `date_range`. This naturally handles 23- and 25-hour local days.
- A cross join between station names and UTC hours gives the required key grid.
- Set `source_observed` from join membership, not from whether air temperature is missing. A source row can have an invalid temperature value.
- Never fill sensor columns after the left join. Missing panel values represent real structural gaps.
- To identify runs, compare each missing flag with its prior value within station and cumulatively count transitions.

## Q4: Past-Only Forecast Features

- Sort by station and UTC before grouped `shift` or `rolling` operations.
- The target and lags are shifts of panel rows because Q3 made the elapsed-hour grid complete.
- `air_temperature_change_1h_c` is current temperature minus the exact one-hour lag.
- Convert degrees to radians before applying sine and cosine.
- The target hour and day-of-year cycles describe cutoff plus one hour in Chicago local time.
- A station slug can come from lowercasing the station name, replacing non-alphanumeric runs with `_`, and trimming separators.
- Eligibility depends on current and next-hour observed temperatures. Do not require every predictor to be present; the training-fitted imputer handles predictor missingness.
- The rolling mean includes the cutoff, so do not shift before the 24-row roll in this assignment.

## Q5: Training-Only Exploration

- Derive local target year, month, and hour from the UTC target timestamp with `.dt.tz_convert("America/Chicago")`.
- Filter to targets before local 2024 before grouping, correlation, or plotting.
- Named aggregation makes output names and observed-value counts explicit.
- `training_rows[CORRELATION_FEATURES].corr(method="pearson")` returns a square matrix in input-column order.

## Q6: Fixed Splits

- Convert each local boundary to a timezone-aware timestamp, then to UTC for direct comparison with target timestamps.
- Use only `model_eligible == True` rows. Preserve rows with missing non-target predictors.
- Sort once, then select X and y by the same split mask so IDs remain aligned.
- `n_features` is 19: station plus 18 numeric predictors. IDs and timestamps are not model features.

## Q7: Train-Fitted Pipeline

- A `ColumnTransformer` can apply `OneHotEncoder(handle_unknown="ignore")` to station and `SimpleImputer(strategy="median")` to numeric predictors.
- Put preprocessing and the regressor in one `Pipeline`; fitting that object on training data prevents validation leakage.
- Choose only a regressor included in the pinned scikit-learn version. Do not install XGBoost.
- Inspect `estimator.get_params(deep=False)` to verify support before setting `random_state=217` or `n_jobs=1`.
- RMSE is `np.sqrt(mean_squared_error(actual, prediction))`.
- R2 can be negative for a weak model. That is valid and does not reduce credit.
- For MAE permutation importance, sklearn's `scoring="neg_mean_absolute_error"` produces a positive importance decrease when permutation worsens MAE.
- Record the regressor's module/class and shallow parameters. Keep fixed feature names joined with `|`.

## Q8: Test Once

- Freeze Q7 before loading test. Recreate the regressor from the recorded module, class, and JSON parameters.
- Recreate the same preprocessing pipeline, concatenate train and validation, fit once, then predict test.
- Persistence uses test X's current air temperature; eligibility guarantees this baseline value exists.
- Build metrics from the saved prediction columns so overall and station results share the same rows.
- Residuals are model prediction minus actual. Include a zero reference line where useful.
- A weak test score is not a processing error. Do not return to Q7 after inspecting test results.

## Q9: Report

- Keep exactly the six required level-two headings from the starter.
- Replace every bracketed placeholder.
- Include numeric validation and/or test MAE, RMSE, and R2 values in a Markdown table.
- Keep all three required image paths unchanged so the structural check can find them.
- Q9 does not score length, style, or whether the model beats persistence.

## Pairing and Submission Checks

After editing either member of a pair, synchronize and test it:

```bash
jupytext --sync q4_feature_engineering.md
jupytext --to ipynb --test-strict q4_feature_engineering.md
```

Before submission, restart kernels and run notebooks Q1 through Q9 in order. Inspect artifact values and shapes, then clear notebook outputs and run the checks in [`README.md`](README.md).
