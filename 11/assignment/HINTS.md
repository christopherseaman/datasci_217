# Hints and Troubleshooting

Use these nudges after trying the question yourself. Each names the lecture that taught the tool; they stop short of complete answers.

## Paths and Handoffs

- Run the notebooks from this assignment folder, the one that holds `data/` and `output/`, so the relative paths work.
- Start each notebook by reading the files the previous one saved. Do not rely on variables from another notebook's kernel.
- Before saving a CSV, select the column list from the checkpoint explicitly, as in `frame[COLUMNS].to_csv(path, index=False)` (Lecture 04). That drops accidental index columns and extras.
- Read a saved UTC column back with `pd.to_datetime(frame[column], utc=True)` (Lecture 09) so it keeps its time zone.

## Q1: Release Audit and Coverage

- Read the expected values from `data/release_manifest.json` with `json.load()` (Lecture 07), and measure the observed ones from the file with `path.name`, `hashlib.sha256(path.read_bytes()).hexdigest()`, and `path.stat().st_size` (Lecture 05's "Fingerprint the source file" snippet).
- `"|".join(weather.columns)` joins the column names in file order (Lecture 02's string methods).
- Parse the source timestamps with `pd.to_datetime()` first. They are naive Chicago wall times, so `.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT")` (Lecture 09's clock-change snippet) marks the times that cannot name one instant.
- Build the expected hours from the two local endpoints converted to UTC, with `pd.date_range(start, end, freq="h", inclusive="left")` (Lecture 09). Its length is the expected hour count.
- A histogram of one sensor and a line plot of a daily mean (`resample("D")`, Lecture 09) for each station make a readable pair of panels (Lecture 07).

## Q2: Cleaning Without Inventing Data

- Count affected values before changing them: build the rule's Boolean mask, save `int(mask.sum())`, then set `frame.loc[mask, column] = np.nan` (Lecture 05). `.between()` is False for a missing value, so `~values.between(low, high)` also selects values that were already missing; add `& values.notna()` so the count holds only values the rule changed.
- `pd.to_numeric(frame[column], errors="coerce")` turns unreadable values into `NaN` (Lecture 05).
- `.between(low, high)` includes both bounds (Lecture 05). Precipitation type is a membership rule: `.isin([0, 40, 60, 70])`.
- Solar needs two masks: values outside -20 to 1500 become missing, and values from -20 up to but not including 0 become 0 (`(values >= -20) & (values < 0)`, Lecture 05).
- A row with one invalid sensor value still holds useful measurements. Reject rows only for their station and time, never for sensor values.
- Only ambiguous fall-back hours become `NaT` here. If many more rows disappear, check that you localized the naive times rather than parsing them as UTC.

## Q3: Complete Elapsed-Hour Panel

- Convert the local start and end to UTC, then build the hours with `pd.date_range(..., freq="h", inclusive="left")`; this handles the 23-hour and 25-hour local days for you (Lecture 09).
- `pd.merge(stations, hours, how="cross")` builds every station at every hour (Lecture 06's cross-join snippet).
- Left-join the cleaned rows with `validate="one_to_one"` and `indicator=True`, and set `source_observed` from `_merge == "both"` (Lecture 06). A source row can have a missing temperature, so do not use temperature to decide.
- Never fill the sensor columns after the join. A missing panel value is a real gap.
- For gap runs, sort by station and time, then number the hours with `groupby("station_name")["source_observed"].cumsum()`, a running count of observed hours that stays flat through each gap. Group the rows with `source_observed` False by station and that number: each group's `size()` is one run's length (Lecture 09's "Count Gap Runs per Patient" snippet).

## Q4: Past-Only Forecast Features

- Sort by station and UTC time before any grouped `shift` or `rolling` (Lecture 09).
- The target and the lags are shifts of panel rows, because Q3 made every hour a row: `frame.groupby("station_name")["air_temperature_c"].shift(1)` is the value one hour earlier, and `shift(-1)` one hour later (Lecture 09).
- For the rolling mean, `groupby("station_name")["air_temperature_c"].transform(lambda s: s.rolling(24, min_periods=1).mean())` includes the cutoff row; do not shift before rolling (Lecture 09).
- The cyclic features use Lecture 10's "Cyclic Time Features" card, `np.sin(2 * np.pi * value / cycle_length)`: 360 for wind direction, 24 for the hour, and 366 for the day of the year minus 1.
- The target's calendar features describe cutoff plus one hour in Chicago local time: `.dt.tz_convert("America/Chicago").dt.hour` and `.dt.dayofyear` (Lecture 09).
- A station slug comes from `.str.lower().str.replace(" ", "_")` (Lecture 05); `.dt.strftime("%Y%m%d%H")` writes the cutoff hour (Lecture 09).
- Eligibility depends on the cutoff and next-hour temperatures only. The training-fitted imputer handles other missing predictors.

## Q5: Training-Only Exploration

- Filter to eligible rows with targets before local 2024-01-01 before grouping, correlating, or plotting.
- Get the target's local year, month, and hour with `.dt.tz_convert("America/Chicago")` and then `.dt.year`, `.dt.month`, `.dt.hour` (Lecture 09).
- Named aggregation, `.agg(n_observed=("target_air_temperature_c", "count"), ...)`, names the output columns for you (Lecture 08).
- `training_rows[CORRELATION_FEATURES].corr()` returns the square matrix in column order (Lecture 07).

## Q6: Fixed Splits

- Compare the UTC target times with local boundaries written as `pd.Timestamp("2024-01-01", tz="America/Chicago")`; pandas compares the instants (Lecture 10's "Splitting on Target Time" card).
- Use only `model_eligible` rows and keep the rows with missing predictors.
- Sort once, then take X and y from the same sorted rows so their `row_id` values line up.
- `n_features` is 19: the station plus 18 numeric predictors. The IDs and timestamps are not model features.

## Q7: Train-Fitted Pipeline

- [Lecture 10](https://github.com/christopherseaman/datasci_217/blob/main/10/README.md) and its [Demo 2](https://github.com/christopherseaman/datasci_217/blob/main/10/demo/demo2_sklearn_prediction.md) build a `ColumnTransformer` with a `OneHotEncoder` and a `SimpleImputer` inside a `Pipeline`.
- Fitting the whole pipeline on training rows fits the imputer and the encoder on training rows too, which keeps validation out of preprocessing.
- `LinearRegression`, `Ridge`, and `RandomForestRegressor` are all in the pinned scikit-learn. Check `model.get_params(deep=False)` for `random_state` and `n_jobs` before setting them.
- RMSE is `np.sqrt(mean_squared_error(actual, prediction))` (Lecture 10).
- R2 can be negative for a weak model. That is a valid result and does not cost points.
- With `scoring="neg_mean_absolute_error"`, `result.importances_mean` is positive when shuffling a feature makes MAE worse (Lecture 10).
- `json.dumps(model.get_params(deep=False))` writes the settings as text, and Q8's `json.loads()` reads them back (Lecture 07's "Saving Charts and Records" card).
- `estimator_module` and `estimator_class` come from your import line: `from sklearn.linear_model import Ridge` gives `sklearn.linear_model` and `Ridge` (Lecture 02's imports).

## Q8: Test Once

- Rebuild the regressor with the same arguments you used in Q7, which `parameters_json` records, such as `Ridge(alpha=parameters["alpha"], random_state=217)`, before reading any test file.
- Put train and validation together with `pd.concat([...], ignore_index=True)` (Lecture 06), fit once, then predict the test rows.
- Persistence uses the test rows' `air_temperature_c_t`; eligibility guarantees it exists.
- Compute overall and station metrics from the saved prediction columns so they share the same rows (`groupby("station_name")`, Lecture 08).
- Residuals are prediction minus actual. A horizontal line at zero helps a residual plot (`ax.axhline(0)`, Lecture 09).
- A weak test score is not a processing error. Do not return to Q7 after seeing test results.

## Q9: Report

- Keep the six headings from the scaffold exactly, and replace every bracketed placeholder.
- Copy the metrics from the Q9 notebook's results cell into the table; rounding to three decimals is fine.
- Keep the three image paths unchanged, and preview `report.md` in VS Code to see the images.
- Length and style are not scored, and the model does not need to beat persistence.
