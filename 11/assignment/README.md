# Assignment 11: Chicago Beach Weather Forecasting (Final Exam)

## Files

```text
assignment/
├── assignment.md                 # the forecasting question, rules, and every file's columns: read it first
├── q1_setup_exploration.ipynb    # Q1 notebook: complete every TODO, then run it top to bottom
├── q2_data_cleaning.ipynb        # Q2 notebook
├── q3_data_wrangling.ipynb       # Q3 notebook
├── q4_feature_engineering.ipynb  # Q4 notebook
├── q5_pattern_analysis.ipynb     # Q5 notebook
├── q6_modeling_preparation.ipynb # Q6 notebook
├── q7_modeling.ipynb             # Q7 notebook
├── q8_results.ipynb              # Q8 notebook
├── q9_writeup.ipynb              # Q9 notebook: shows your saved results while you write report.md
├── q1_setup_exploration.md ... q9_writeup.md  # supplied plain-text copies of the notebooks; leave them as they are
├── report.md                     # you complete in Q9
├── HINTS.md                      # supplied nudges for each question
├── example_report/               # supplied formatting example with made-up numbers
├── data/
│   ├── chicago_beach_sensors_2022_2024.csv  # supplied frozen release; never edit it
│   └── release_manifest.json                # supplied facts about the release, including its SHA-256
├── download_data.sh              # supplied: verifies the two data files (it downloads nothing)
├── requirements.txt              # supplied: the pinned package versions
├── .python-version               # supplied: Python 3.13
├── .gitattributes, .gitignore    # supplied: keep the data byte for byte, keep .venv/ out of Git
└── output/
    ├── q1_release_audit.csv            # you make in Q1 (section 1.2)
    ├── q1_station_coverage.csv         # you make in Q1 (section 1.3)
    ├── q1_visualizations.png           # you make in Q1 (section 1.4)
    ├── q2_cleaned_observations.csv     # you make in Q2 (sections 2.2 and 2.3)
    ├── q2_cleaning_audit.csv           # you make in Q2 (section 2.4)
    ├── q2_missingness.csv              # you make in Q2 (section 2.4)
    ├── q3_hourly_panel.csv             # you make in Q3 (section 3.2)
    ├── q3_panel_summary.csv            # you make in Q3 (section 3.3)
    ├── q4_features.csv                 # you make in Q4 (section 4.2)
    ├── q4_feature_manifest.csv         # you make in Q4 (section 4.3)
    ├── q5_monthly_station_summary.csv  # you make in Q5 (section 5.2)
    ├── q5_correlations.csv             # you make in Q5 (section 5.3)
    ├── q5_patterns.png                 # you make in Q5 (section 5.4)
    ├── q6_X_train.csv, q6_X_validation.csv, q6_X_test.csv  # you make in Q6 (section 6.3)
    ├── q6_y_train.csv, q6_y_validation.csv, q6_y_test.csv  # you make in Q6 (section 6.3)
    ├── q6_split_summary.csv            # you make in Q6 (section 6.4)
    ├── q7_validation_predictions.csv   # you make in Q7 (section 7.3)
    ├── q7_validation_metrics.csv       # you make in Q7 (section 7.3)
    ├── q7_model_spec.csv               # you make in Q7 (section 7.4)
    ├── q7_permutation_importance.csv   # you make in Q7 (section 7.4)
    ├── q8_test_predictions.csv         # you make in Q8 (section 8.2)
    ├── q8_test_metrics.csv             # you make in Q8 (section 8.2)
    ├── q8_station_metrics.csv          # you make in Q8 (section 8.3)
    └── q8_final_visualizations.png     # you make in Q8 (section 8.4)
```

Every output file, its first line, and its line count are listed under [Check Your Work](#check-your-work).

## The data

`data/chicago_beach_sensors_2022_2024.csv` is a frozen extract of the City of Chicago's beach weather sensors: 50,895 hourly readings from two stations, Foster Weather Station and Oak Street Weather Station, for local 2022-01-01 through 2024-12-31. Each row is one station's reading for one hour:

```text
station_name,measurement_timestamp,air_temperature_c,wet_bulb_temperature_c,relative_humidity_pct,...
Foster Weather Station,2022-01-01 00:00:00,3.39,,87,...
```

`measurement_timestamp` is the Chicago wall-clock time with no time zone written on it. `data/release_manifest.json` records the release's file name, SHA-256, size, row and column counts, columns, stations, and time zone. Leave both files exactly as they ship: Q1 checks them against each other.

## Setup

Fork the assignment repository on GitHub and clone your fork the way Lecture 01 did: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open it. Then open **Terminal → New Terminal** in VS Code at the assignment directory (Ctrl+Shift+backtick, also Control on Mac). If you use a native terminal or WSL Ubuntu instead, `cd` into the assignment directory first. Run `ls data` and expect `chicago_beach_sensors_2022_2024.csv  release_manifest.json`. This clone is a new repository, so before your first commit run Lecture 02's two `git config user.name "..."` and `git config user.email "..."` lines in this terminal, with your name and GitHub noreply email.

> **Windows:** work in the **WSL: Ubuntu** window from Lecture 01's setup; `download_data.sh` is a Bash script. Git Bash also works; there the environment activates with `source .venv/Scripts/activate` instead.

Create the project environment, activate it, and install the supplied requirements, as in Lecture 03. Then verify the data:

```bash
uv venv --seed --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
bash download_data.sh
```

The last command prints `Verified frozen release and manifest: data/chicago_beach_sensors_2022_2024.csv (4731351 bytes)`. If it reports a mismatch instead, discard your changes to the `data/` files in Source Control and run it again.

`requirements.txt` includes **ipykernel**, the package that lets a notebook run on this environment's Python (Lecture 04), so this one install is all the notebooks need. Open `q1_setup_exploration.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`. If VS Code offers to install the **Jupyter** extension, accept. Run the first code cell: it prints the Python, NumPy, pandas, scikit-learn, and Matplotlib versions and shows the first rows of the release. Select the same kernel in each notebook.

## How the exam works

Read [`assignment.md`](assignment.md) first. It defines the forecasting question (each station's air temperature one hour after a cutoff hour), the cleaning rules, the fixed predictors, the train, validation, and test periods, and every output file's columns and rules.

Then complete the nine notebooks in order, Q1 through Q9. Each notebook starts by reading the files the previous one saved in `output/`, so you can close one notebook and open the next without rerunning anything. Every notebook section that saves a file ends with a **Checkpoint** naming that file, its first line, and its line count. [`HINTS.md`](HINTS.md) has a nudge for each question when you are stuck, and each hint names the lecture that taught the tool.

| Question | Notebook | What you build | Points |
| --- | --- | --- | ---: |
| Q1 | [`q1_setup_exploration.ipynb`](q1_setup_exploration.ipynb) | Audit the release against its manifest, measure each station's coverage, and plot a first look | 5 |
| Q2 | [`q2_data_cleaning.ipynb`](q2_data_cleaning.ipynb) | Convert times to UTC, apply the sensor rules, and record what changed | 10 |
| Q3 | [`q3_data_wrangling.ipynb`](q3_data_wrangling.ipynb) | Build the complete station-by-hour panel and summarize its gaps | 10 |
| Q4 | [`q4_feature_engineering.ipynb`](q4_feature_engineering.ipynb) | Build the next-hour target and the past-only predictors | 12 |
| Q5 | [`q5_pattern_analysis.ipynb`](q5_pattern_analysis.ipynb) | Describe monthly and hourly patterns in the training period only | 5 |
| Q6 | [`q6_modeling_preparation.ipynb`](q6_modeling_preparation.ipynb) | Split the eligible rows into train, validation, and test by time | 10 |
| Q7 | [`q7_modeling.ipynb`](q7_modeling.ipynb) | Fit one scikit-learn pipeline and compare it with persistence on validation | 12 |
| Q8 | [`q8_results.ipynb`](q8_results.ipynb) | Refit the frozen choice and evaluate the test period once | 11 |
| Q9 | [`q9_writeup.ipynb`](q9_writeup.ipynb) | Complete `report.md` | 25, human review |

## Check Your Work

Restart each notebook's kernel and **Run All**, Q1 through Q8, and confirm every cell finishes without an error. Then open each file in `output/` in VS Code, where the number beside the last line is the file's line count, and check it against this list. A count written as a formula depends on your own earlier file; work it out from that file.

| File | Made in | First line | Lines |
| --- | --- | --- | --- |
| `output/q1_release_audit.csv` | Q1, 1.2 | `check_name,expected,observed,passed` | 8 |
| `output/q1_station_coverage.csv` | Q1, 1.3 | `station_name,expected_hours,observed_hours,missing_hours,coverage_pct,first_timestamp,last_timestamp` | 3 |
| `output/q1_visualizations.png` | Q1, 1.4 | a PNG image | |
| `output/q2_cleaned_observations.csv` | Q2, 2.2 and 2.3 | `station_name,measurement_timestamp,air_temperature_c,wet_bulb_temperature_c,relative_humidity_pct,rain_intensity_mm_per_hour,interval_rain_mm,total_rain_mm,precipitation_type_code,wind_direction_deg,wind_speed_mps,maximum_wind_speed_mps,barometric_pressure_hpa,solar_radiation_w_m2,battery_voltage_v,measurement_timestamp_utc` | the 50,895 release rows less the `rows_rejected` count in your `q2_cleaning_audit.csv`, plus the header |
| `output/q2_cleaning_audit.csv` | Q2, 2.4 | `rule,affected_values,result` | 16: one row for each of the 15 rules in [assignment.md](assignment.md#q2-data-cleaning), plus the header |
| `output/q2_missingness.csv` | Q2, 2.4 | `station_name,column_name,missing_count,missing_pct` | 27: 2 stations times 13 sensor columns, plus the header |
| `output/q3_hourly_panel.csv` | Q3, 3.2 | `station_name,measurement_timestamp_utc,air_temperature_c,wet_bulb_temperature_c,relative_humidity_pct,rain_intensity_mm_per_hour,interval_rain_mm,total_rain_mm,precipitation_type_code,wind_direction_deg,wind_speed_mps,maximum_wind_speed_mps,barometric_pressure_hpa,solar_radiation_w_m2,battery_voltage_v,source_observed,hour,day_of_week,month` | 2 times the `expected_hours` in your `q1_station_coverage.csv`, plus the header |
| `output/q3_panel_summary.csv` | Q3, 3.3 | `station_name,expected_hours,observed_hours,missing_hours,gap_runs,longest_gap_hours` | 3 |
| `output/q4_features.csv` | Q4, 4.2 | `row_id,station_name,cutoff_timestamp_utc,target_timestamp_utc,target_air_temperature_c,model_eligible,air_temperature_c_t,relative_humidity_pct_t,interval_rain_mm_t,wind_speed_mps_t,maximum_wind_speed_mps_t,barometric_pressure_hpa_t,solar_radiation_w_m2_t,wind_direction_sin_t,wind_direction_cos_t,air_temperature_lag_1h_c,air_temperature_lag_24h_c,air_temperature_lag_168h_c,air_temperature_mean_past_24h_c,air_temperature_change_1h_c,target_hour_sin,target_hour_cos,target_day_of_year_sin,target_day_of_year_cos` | the same as `q3_hourly_panel.csv` |
| `output/q4_feature_manifest.csv` | Q4, 4.3 | `feature_name,source,earliest_offset_hours,latest_offset_hours,role` | 20: the 19 fixed predictors, plus the header |
| `output/q5_monthly_station_summary.csv` | Q5, 5.2 | `station_name,year,month,n_observed,mean_air_temperature_c,std_air_temperature_c,min_air_temperature_c,max_air_temperature_c` | 49: 2 stations times the 24 training months, plus the header |
| `output/q5_correlations.csv` | Q5, 5.3 | `feature,air_temperature_c_t,relative_humidity_pct_t,interval_rain_mm_t,wind_speed_mps_t,maximum_wind_speed_mps_t,barometric_pressure_hpa_t,solar_radiation_w_m2_t` | 8 |
| `output/q5_patterns.png` | Q5, 5.4 | a PNG image | |
| `output/q6_X_train.csv`, `output/q6_X_validation.csv`, `output/q6_X_test.csv` | Q6, 6.3 | `row_id,station_name,cutoff_timestamp_utc,target_timestamp_utc,air_temperature_c_t,relative_humidity_pct_t,interval_rain_mm_t,wind_speed_mps_t,maximum_wind_speed_mps_t,barometric_pressure_hpa_t,solar_radiation_w_m2_t,wind_direction_sin_t,wind_direction_cos_t,air_temperature_lag_1h_c,air_temperature_lag_24h_c,air_temperature_lag_168h_c,air_temperature_mean_past_24h_c,air_temperature_change_1h_c,target_hour_sin,target_hour_cos,target_day_of_year_sin,target_day_of_year_cos` | that split's `n_rows` in your `q6_split_summary.csv`, plus the header |
| `output/q6_y_train.csv`, `output/q6_y_validation.csv`, `output/q6_y_test.csv` | Q6, 6.3 | `row_id,target_air_temperature_c` | the same as the matching X file |
| `output/q6_split_summary.csv` | Q6, 6.4 | `split,n_rows,target_start,target_end,n_features` | 4 |
| `output/q7_validation_predictions.csv` | Q7, 7.3 | `row_id,station_name,target_timestamp_utc,actual,persistence_prediction,model_prediction` | the same as `q6_X_validation.csv` |
| `output/q7_validation_metrics.csv` | Q7, 7.3 | `model,mae,rmse,r2,n` | 3 |
| `output/q7_model_spec.csv` | Q7, 7.4 | `estimator_module,estimator_class,parameters_json,feature_columns,random_state` | 2 |
| `output/q7_permutation_importance.csv` | Q7, 7.4 | `feature,mean_mae_increase,std_mae_increase` | 20 |
| `output/q8_test_predictions.csv` | Q8, 8.2 | `row_id,station_name,target_timestamp_utc,actual,persistence_prediction,model_prediction,model_error,model_absolute_error` | the same as `q6_X_test.csv` |
| `output/q8_test_metrics.csv` | Q8, 8.2 | `model,mae,rmse,r2,n` | 3 |
| `output/q8_station_metrics.csv` | Q8, 8.3 | `model,station_name,n,mae,rmse,r2` | 5: 2 models times 2 stations, plus the header |
| `output/q8_final_visualizations.png` | Q8, 8.4 | a PNG image | |
| `report.md` | Q9 | `# Next-Hour Chicago Beach Air Temperature Forecast` | |

- [ ] All 28 files in `output/` exist, each with the first line and line count above, and `report.md` is complete.
- [ ] No CSV starts with an extra column of row numbers: each was saved with `index=False`, except `q5_correlations.csv`, which saves its row labels as a first column named `feature`.
- [ ] Every `*_utc` time carries its offset, such as `2024-07-01 05:00:00+00:00`; pandas writes it that way when the column is in UTC.
- [ ] Each PNG opens in VS Code and shows labeled axes and titles.
- [ ] `report.md` has the six headings, the four-row metrics table with your saved numbers, and the three images, and no bracketed placeholder remains.
- [ ] `data/` is unchanged: Source Control lists no change to either file.

### Completion contract

The exam totals 100 points: 75 points graded from your committed files after the deadline, 25 by human review. Each file is graded on its own, and most files are split into several parts, so a wrong value costs only the part it belongs to.

| File | What earns the points | Points |
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

How the files are read:

- Line endings, a byte-order mark, spaces around a value or a header name, blank lines, and a missing final newline do not matter. Neither do column order, extra columns, a leading column of row numbers, or row order: rows are matched by their key (station and time, `row_id`, `split`, `model`, and so on).
- Numbers are compared as numbers, so `2`, `2.0`, and `2.00` are the same, and a value rounded to two decimals or more passes (percentages to one decimal). Counts must be exact.
- Times are compared as instants: `2024-07-01 05:00:00+00:00`, `2024-07-01T05:00:00Z`, and `2024-07-01 00:00:00-05:00` are the same hour. In `q1_station_coverage.csv` and `q6_split_summary.csv`, a time written without an offset may be either UTC or Chicago local time.
- `True`/`False`, `true`/`false`, `1`/`0`, and `yes`/`no` all read as booleans; an empty field, `NaN`, and `<NA>` all read as missing. Labels such as station names, `model`, `split`, `role`, and `result` may use any letter case.
- Also accepted: `missing_pct` as a fraction instead of a percent, the population standard deviation (`ddof=0`) instead of the sample one, `q5_correlations.csv` saved with an unnamed first column, and your regressor's name, such as `Ridge`, in place of `student_model`.
- A missing or extra row costs only the rows part; the value parts judge the rows you have, as long as at least half of the expected rows are there.
- A file built correctly from one of your own earlier files counts as right even when that earlier file has a mistake: for example, a Q3 panel joined from your Q2 table, Q4 features computed from your Q3 panel, Q6 splits taken from your Q4 file, and metrics computed from your prediction files. A mistake costs points once, where you made it.
- The model's accuracy is not graded: it does not need to beat persistence.

Human review reads `report.md` and the notebooks, 5 points for each category:

| Category | What it reads | Full credit (5) | Partial credit |
| --- | --- | --- | --- |
| Data and cleaning decisions | `report.md` section **Data and Cleaning** and `q1_visualizations.png`; notebook sections 2.2, 2.3, and 3.2 | The release audit result and each station's coverage from your Q1 files; the UTC conversion and why the ambiguous fall-back rows are rejected; the sensor rules and why invalid values become missing rather than filled; the complete panel that keeps gaps visible; each with a reason that comes from this data; a Q1 figure whose labeled panels show a sensor distribution and each station's time series | 2 to 4 when decisions are listed without reasons or numbers, one area is missing, or the figure is hard to read |
| Training patterns | `report.md` section **Patterns** and `q5_patterns.png` | One or two monthly or local-hour patterns with numbers from `q5_monthly_station_summary.csv` or `q5_correlations.csv`, stated as training-period results; a correlation described as a pattern, not as proof that a predictor helps; a figure that shows both patterns with titles and labeled axes | 2 to 4 when numbers are missing, a claim goes beyond the files, or the figure is hard to read |
| Forecast design and model choice | `report.md` section **Forecast Design**; notebook section 4.4 and the Markdown cell in section 7.2 | The next-hour target, the past-only predictors, the persistence baseline, the preprocessing fit inside the pipeline on training rows, and the split by target time, each with its reason; section 4.4 prints rows that show one lag and one target matching the panel; the 7.2 cell names the regressor you kept and why, from its validation results | 2 to 4 when decisions are listed without reasons, section 4.4 shows no lag or target, or the 7.2 cell is missing |
| Model results tied to evidence | `report.md` section **Model Results**, its metrics table, and `q8_final_visualizations.png` | The four table rows match your Q7 and Q8 metric files; the model compared with persistence on validation and test using the table's numbers; one station difference from `q8_station_metrics.csv`; the test results reported, not used to tune; a figure with its three views, titles, and labeled axes | 2 to 4 when numbers are missing or differ from the files, or a claim goes beyond what the files show |
| Summary, limitations, and reproducibility | `report.md` sections **Executive Summary** and **Limitations**, and the notebooks as a whole | The summary states the question, the data, the model, and one test result; the limitations are specific to this release and design (two stations, sensor gaps, one test period); every notebook runs top to bottom with its outputs saved | 2 to 4 when the summary misses a part, the limitations are generic, or a notebook stops with an error or shows no outputs |

A category whose `report.md` section still holds its bracketed placeholder earns 0.

## Submit

Save each notebook after **Run All**, so its outputs are part of the commit. In VS Code Source Control, stage the nine `.ipynb` notebooks, `report.md`, and all 28 files in `output/`, commit with a message such as `Complete the final exam`, and sync. Keep `.venv/` out of the commit; `.gitignore` already lists it. You can also upload files on the GitHub website with **Add file → Upload files**, keeping each file at its path.

The course grades the files on your fork's `main` branch after the deadline. Open your fork on GitHub and confirm that `output/` shows all 28 files, that `report.md` shows your table and images, and that each notebook shows its outputs.
