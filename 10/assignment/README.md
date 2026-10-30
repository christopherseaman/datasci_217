# Assignment 10: Modeling Blood Pressure and Testing Predictions Honestly

## Overview

Fit and read an OLS model of blood pressure across eight clinics, audit a follow-up program's candidate features for leakage, split its visits on target time, beat two baselines with a pipeline, test the frozen winner once, and score a readmission screening rule, saving each result for the checks.

All four files are synthetic extracts from one health system in the East Bay; the IDs belong to no one. Keep them exactly as handed out.

- `data/bp_patients.parquet` (Task 1): 150,000 adult primary-care patients at eight clinics, one row each, from their 2025 visit: `patient_id`, `clinic`, `visit_date`, `age`, `sex`, `insurance`, `adi_decile` (neighborhood deprivation, 1 least to 10 most deprived), `smoker`, `activity_min_wk` (self-reported exercise minutes a week), `diabetes`, `bmi` in kg/m², blank for 10,344 patients who were not weighed, `on_bp_med`, and `sbp`, systolic blood pressure in mmHg. Fruitvale and Eastmont still use manual cuffs.
- `data/followup_visits.parquet` (Tasks 2 and 3): a hypertension program's 515,750 visits by 62,000 patients from January 2024 to December 2025, one row per visit that has a completed follow-up visit. The program predicts each patient's SBP at the next visit, two to twelve weeks later, as soon as today's visit ends.
- `data/feature_candidates.parquet` (Task 2.1): the nine columns proposed as features, with `becomes_known` and `hours_after_visit`, how many hours after today's visit ends each one is known (negative: before the visit).
- `data/discharges.parquet` (Task 3.3): 210,000 inpatient discharges from four hospitals in 2024 and 2025, with each stay's `lace_score` and `readmitted_30d` (1 if readmitted within 30 days, else 0).

| `followup_visits` column | Meaning |
| --- | --- |
| `visit_id`, `patient_id`, `clinic` | Synthetic IDs and the clinic |
| `visit_time` | When today's visit ended: the prediction time (UTC) |
| `followup_time` | When the next visit measured the target: the target time (UTC) |
| `age`, `bmi` | Years and kg/m²; `bmi` is blank when the patient was not weighed |
| `n_bp_meds`, `med_change_today` | Blood-pressure drugs before today, and 1 if the clinician added one today |
| `sbp_today`, `prior_sbp` | SBP at today's visit and at the previous one (blank at a first visit), mmHg |
| `a1c_result` | HbA1c from blood drawn today, %; the lab reports it the next day |
| `callback_sbp` | The home reading a nurse collects by phone three days later, mmHg |
| `refill_gap_days` | Days without pills over the next 30 days, from pharmacy claims |
| `sbp_followup` | SBP at the next visit, mmHg: the target |

```text
visit_id,patient_id,clinic,visit_time,followup_time,age,bmi,n_bp_meds,med_change_today,sbp_today,prior_sbp,a1c_result,callback_sbp,refill_gap_days,sbp_followup
FV000001,H16330,Fruitvale,2024-01-01 16:02:00+00:00,2024-01-22 19:44:00+00:00,51,24.3,0,0,140,,6.6,152,0,166
```

## Setup

1. Fork and clone the repository, then open its folder in VS Code.
2. Create the environment:

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```
3. Open `assignment.ipynb` and select the `.venv` Python as the kernel.
4. Run the notebook's first two code cells. The second downloads the four Parquet files (16 MB) if `data/` lacks them, then reads them.
    - Expect: `pandas: 3.0.5`, `statsmodels: 0.14.6`, and `scikit-learn: 1.9.0`, then `patients: (150000, 13)`, `visits: (515750, 15)`, `candidates: (9, 3)`, and `discharges: (210000, 11)`.

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/                   # supplied; keep these files exactly as handed out
│   ├── bp_patients.parquet         # Task 1
│   ├── followup_visits.parquet     # Tasks 2.2 to 3.2
│   ├── feature_candidates.parquet  # Task 2.1
│   └── discharges.parquet          # Task 3.3
├── pyproject.toml          # supplied: the packages the notebook uses, pyarrow (for Parquet), and ipykernel
├── uv.lock                 # supplied: the exact versions `uv sync` installs
├── .python-version         # supplied: tells uv to use Python 3.13
├── CHECKS.md               # supplied: what each check looks for
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── ols_coefficients.csv         # you generate in Task 1.1
    ├── new_patient_intervals.csv    # you generate in Task 1.2
    ├── residual_bands.csv           # you generate in Task 1.3
    ├── residuals_vs_fitted.png      # you generate in Task 1.3
    ├── availability_decisions.csv   # you generate in Task 2.1
    ├── split_summary.csv            # you generate in Task 2.2
    ├── validation_metrics.csv       # you generate in Task 3.1
    ├── test_metrics.csv             # you generate in Task 3.2
    ├── clinic_test_errors.csv       # you generate in Task 3.2
    └── readmission_metrics.csv      # you generate in Task 3.3
```

## Task 1: Model blood pressure with statsmodels

The population-health team asks how much higher SBP runs with BMI among patients of the same age. Lecture 10's `statsmodels` workflow answers it on `patients`, as Demo 1 steps 3 to 6 do on another clinic network.

### 1.1 Fit the model and save its coefficients

In the Task 1.1 cell:

1. Fit `results`: an OLS model with the formula API, outcome `sbp`, and predictors `age` and `bmi`, on `patients` (Lecture 10's "Code Snippet: OLS with the Formula API").
    - Expect: `rows used: 139656 of 150000` and `R-squared: 0.179`. `statsmodels` dropped the 10,344 patients with no BMI, as the lecture's "Check `results.nobs` after every fit" callout warns.
2. Build `coefficients`, one row per term, with the columns `coef`, `std_err`, `ci_lower`, and `ci_upper`. The values come from the results' coefficients, standard errors, and the two columns of its confidence intervals (Lecture 10's "Code Snippet: Uncertainty and a New-Patient Interval").
3. Name its index `term` (Lecture 04) and save it to `COEFFICIENTS_PATH`. Keep the index this time, so the terms become the first column.
    - Expect: about 0.88 mmHg more SBP per 1 kg/m² of BMI among patients of the same age, 95% CI 0.87 to 0.90. With 139,656 rows the interval is narrow; it does not say losing weight would lower SBP by that much.

> **Checkpoint: `output/ols_coefficients.csv`**
> The header line `term,coef,std_err,ci_lower,ci_upper`, then three rows: `Intercept`, `age`, and `bmi`. The `bmi` row reads about `bmi,0.884,0.0075,0.870,0.899`. Numbers may keep every digit or be rounded to two decimals. The intercept may be named `const`, as the array interface names it.

### 1.2 Give a new patient both intervals

A new patient is 60 years old with a BMI of 31.0; `new_patient` is built for you. In the Task 1.2 cell:

1. Get `intervals`, the 95% mean-response and prediction intervals for `new_patient`, from the fitted results (the same Lecture 10 snippet).
2. Put the patient's `age` and `bmi` beside the intervals, column by column (Lecture 06), and name the result `new_patient_intervals`.
    - Expect: `rows: 1`.
3. Save it to `INTERVALS_PATH` without the index.
    - Expect: a `mean` of about 125.9 mmHg. The mean-response interval, 125.8 to 126.0, pins down the _average_ SBP of patients like this one. The prediction interval, 97.3 to 154.6, is where _this_ patient's SBP probably lies; more rows shrink the first, not the second.

> **Checkpoint: `output/new_patient_intervals.csv`**
> The header line `age,bmi,mean,mean_se,mean_ci_lower,mean_ci_upper,obs_ci_lower,obs_ci_upper`, then one row for age 60 and BMI 31.0. Values may be rounded to one decimal, and `mean_se` may be left out.

### 1.3 Check the residuals by age band

A straight line in age may miss a curve, and older patients' readings vary more. Demo 1 step 6 plots a sample of residuals, then puts numbers on the pattern with bins; here the bins are ages. In the Task 1.3 cells:

1. Build `diagnostics`, a DataFrame with the columns `age` (from `patients`) and `residual` (from the results). The residuals line up with `patients` by index, so the 10,344 dropped patients get a blank residual.
2. Add `age_band`: `age` binned with `pd.cut()` at the scaffold's `AGE_BINS`, named with `AGE_LABELS` (Lecture 05).
3. Build `residual_bands`, one row per age band, with named aggregation (Lecture 08): `patients` (how many residuals, skipping the blanks), `mean_residual`, and `sd_residual`. Keep only the bands that occur, and keep `age_band` as a column.
    - Expect: `rows: 6`.
4. Save it to `RESIDUAL_BANDS_PATH` without the index.
    - Expect: a mean residual of +1.8 mmHg at 18 to 39, -1.4 at 50 to 59, and +2.6 at 80 and over. That U shape means SBP rises faster with age after 50 than a straight line allows (Demo 1 practice 4 fits the curve). The standard deviation grows from 12.9 to 16.7 mmHg, a funnel, so the default standard errors and the Task 1.2 prediction interval are approximate.
5. In the plot cell, scatter a random 5,000 fitted values (x) against their residuals (y), as Demo 1 step 6 does. Add a gray dashed line at 0 and the axis labels `Fitted SBP (mmHg)` and `Residual (observed - fitted)`. Save to `RESIDUAL_PLOT_PATH` with `dpi=100` (which keeps the file near 100 KB) and `bbox_inches="tight"`, before `plt.show()`.

> **Checkpoint: `output/residual_bands.csv`**
> The header line `age_band,patients,mean_residual,sd_residual`, then six rows, `18-39` to `80+`. The first reads about `18-39,24239,1.79,12.94`. Values may be rounded to one decimal.

> **Checkpoint: `output/residuals_vs_fitted.png`**
> A PNG of the step 5 scatter. The check confirms only that the file is a PNG.

## Task 2: Frame the prediction problem

The program wants each visit's `sbp_followup` (the target) predicted at `visit_time` (the prediction time), so its nurses can call patients likely to stay uncontrolled. The target is measured at `followup_time` (the target time). Lecture 10's "Prediction: Features, Targets, and Honest Splits" section and Demo 1 steps 7 to 9 frame a problem the same way.

### 2.1 Audit the candidate features

A feature the model will not have at prediction time is **leakage**, however useful it looks. In the Task 2.1 cell:

1. Make `decisions`, a copy of `candidates`.
2. Add `available`: `True` when the feature is known by the end of the visit, by the rule in Lecture 10's "Feature Availability" section.
3. Add `decision` with `np.where()`: `"Keep"` where available, `"Exclude (leakage)"` where not.
4. Save `decisions` to `AVAILABILITY_PATH` without the index.
5. Set `FEATURES` to the kept `candidate_feature` names, as a list: select them with a row mask, then convert (Lecture 04).
    - Expect: `features: ['age', 'bmi', 'sbp_today', 'prior_sbp', 'n_bp_meds', 'med_change_today']`. `callback_sbp` and `refill_gap_days` would make a model look better on old visits, but the first measures the patient days after the decision and the second records whether the new pills were taken.

> **Checkpoint: `output/availability_decisions.csv`**
> The header line `candidate_feature,becomes_known,hours_after_visit,available,decision`, then nine rows, one per candidate. `prior_sbp`, known 672 hours _before_ the visit, is kept; `a1c_result` reads `a1c_result,lab report the next day,24,False,Exclude (leakage)`. `available` may be written `True`/`False` or `1`/`0`, and any `decision` starting with `keep` or `exclude` counts. Dropping `becomes_known` is fine.

### 2.2 Split on the target time

Split on when each target is measured, `followup_time`, as Lecture 10's "Code Snippet: A Chronological Split" does with dates. Then keep only the training labels the program had in hand at its first validation prediction, as Demo 1 step 9 does. In the Task 2.2 cell:

1. Select `train` (`followup_time` before `VALIDATION_START`, July 1, 2025, UTC), `valid` (from `VALIDATION_START` up to, not including, `TEST_START`, October 1), and `test` (from `TEST_START` on).
    - Expect: `rows: 327532 97175 91043`.
2. Select `fit`, the training rows whose target was measured by the first validation prediction: the earliest `visit_time` among the validation rows (Lecture 10's "Reference Card: Splitting on Target Time").
    - Expect: `fit rows: 245361` and `first validation prediction: 2025-04-08 18:48:00+00:00`. Follow-ups come up to twelve weeks after the prediction, so that prediction comes almost three months before the first validation target. The other 82,171 training targets had not been measured yet, so the model may not learn from them.
3. Build `split_summary`, one row per part in the order `train`, `fit`, `validation`, `test`, with the columns `partition`, `row_count`, `first_target_time`, and `last_target_time` (the part's earliest and latest `followup_time`).
4. Save it to `SPLIT_PATH` without the index.

> **Checkpoint: `output/split_summary.csv`**
> The header line `partition,row_count,first_target_time,last_target_time`, then four rows. The test row reads `test,91043,2025-10-01 15:00:00+00:00,2026-01-01 00:54:00+00:00`. Timestamps are compared as instants, so `2025-10-01T15:00:00Z` is the same value, and one written without a zone is read as UTC. `valid` for `validation` is fine.

## Task 3: Compare, freeze, and test once

### 3.1 Compare two baselines and a pipeline on the validation rows

A model has to beat the rules a program could use with no model. Lecture 10's "Code Snippet: A Baseline and a Linear Pipeline" and "Code Snippet: Comparing on Validation Rows", and Demo 2 steps 2 and 3, show each piece. In the Task 3.1 cell, fit every approach on the fit rows' `FEATURES` and `TARGET`:

1. `baseline`: a `DummyRegressor` that predicts the fit rows' mean.
2. Persistence needs no fitting: it predicts that the next reading equals `sbp_today`, the way Demo 2 step 2 scores `sbp_week` as its own forecast.
3. `pipeline`: a `Pipeline` with three named steps: `"impute"`, a `SimpleImputer` that fills with the median (`bmi` and `prior_sbp` have blanks that `LinearRegression` cannot take); `"scale"`, a `StandardScaler`; and `"model"`, a `LinearRegression`.
4. In the loop, record each approach's validation `mae`, `rmse` (the square root of the mean squared error), and `r2`, with the actual values as the first argument.
5. Save `validation_metrics` to `VALIDATION_PATH` without the index.
6. Set `winner` to the `approach` in the row with the lowest `mae` (Lecture 04's `.idxmin()` and `.loc`).
    - Expect: `frozen choice: linear_pipeline`. Its MAE, about 9.01 mmHg, is 1.9 below persistence's 10.95. One office reading swings about 9 mmHg from visit to visit, so `sbp_today` alone overshoots; the pipeline averages it with `prior_sbp` and lowers the forecast when a drug was just added.

> **Checkpoint: `output/validation_metrics.csv`**
> The header line `approach,mae,rmse,r2`, then three rows: `mean_baseline`, `persistence`, and `linear_pipeline`. Numbers may keep every digit or be rounded to two decimals.

### 3.2 Test the frozen pipeline once, and by clinic

Lecture 10's "Freeze, Then Test Once" section and Demo 2 step 8 show this step. Keep the pipeline Task 3.1 fitted, unchanged: refitting on the validation rows would need its own label-availability cut. In the Task 3.2 cell:

1. Predict the test rows once with the fitted `pipeline`, and name the result `test_predicted`.
2. Build `test_metrics`, one row with `approach` `"linear_pipeline"` and its test `mae`, `rmse`, and `r2`. A dictionary of single values makes `pd.DataFrame()` stop with `If using all scalar values, you must pass an index`; put each value in a one-item list.
    - Expect: a test MAE of about 8.96 mmHg, close to validation's 9.01, so the frozen pipeline held up through the winter months.
3. Build `errors`: `test`'s `clinic` column with two new columns, `error` (the actual `sbp_followup` minus the prediction) and `abs_error` (its absolute value).
4. Build `clinic_errors`, one row per clinic with `clinic` as a column, with named aggregation: `test_visits` (the number of rows), `mae` (the mean `abs_error`), and `mean_error` (the mean `error`).
    - Expect: `rows: 8`. Eastmont's and Fruitvale's follow-ups run about 3.1 and 2.3 mmHg above the predictions; every other clinic is within 1.3. Those two clinics use manual cuffs, which read high, and the features carry no clinic; Lecture 10's "Code Snippet: Variation with Mixed Numeric and Categorical Columns" adds one.
5. Save `test_metrics` to `TEST_METRICS_PATH` and `clinic_errors` to `CLINIC_ERRORS_PATH`, both without the index.

> **Checkpoint: `output/test_metrics.csv`**
> The header line `approach,mae,rmse,r2`, then one row, `linear_pipeline`. Numbers may be rounded to two decimals.

> **Checkpoint: `output/clinic_test_errors.csv`**
> The header line `clinic,test_visits,mae,mean_error`, then eight rows, one per clinic. Fruitvale reads about `Fruitvale,11190,9.22,2.32`.

### 3.3 Score the LACE readmission rules

The care-transitions team calls discharged patients flagged by the **LACE index** (length of stay, acuity, comorbidity, and ED visits, 0 to 19). It is weighing a cutoff of 13 against 10. Lecture 10's "Code Snippet: Accuracy Hides Missed Readmissions" computes these numbers. In the Task 3.3 cell:

1. Build each rule's flag as 0/1: `flag_no_one` flags nobody, an array of zeros (Lecture 03's `np.zeros()`, as Demo 2 step 6's "always no" rule does); `lace_13_plus` flags a `lace_score` of 13 or more; `lace_10_plus`, 10 or more. A comparison gives `True`/`False`; converting to `int` gives 1/0.
2. In the loop, record each rule's `flagged` (how many it flags), `accuracy`, `precision` (0 when it flags no one, with `zero_division=0`), and `recall`, with `discharges["readmitted_30d"]` as the first argument.
3. Save `readmission_metrics` to `READMISSION_PATH` without the index.
    - Expect: flagging nobody is right for 86.8% of discharges, more than either LACE rule, and catches none of the 27,808 readmissions. A cutoff of 13 flags 20,414 patients and catches about a quarter of the readmissions; 10 flags three times as many and catches about half, with more calls to patients who stay home.

> **Checkpoint: `output/readmission_metrics.csv`**
> The header line `approach,flagged,accuracy,precision,recall`, then three rows: `flag_no_one`, `lace_13_plus`, and `lace_10_plus`. `flag_no_one` reads about `flag_no_one,0,0.868,0.0,0.0`. Numbers may be rounded to two decimals.

## Check your work

- Run the checks any time, even partway through: `python3 check_assignment.py` in the terminal (with the environment active) prints your score and what to fix.
- When **Restart** and **Run All** finish without errors, the notebook's last cell runs the same checks.
- Both use the latest checks from the course repository, the same checks GitHub runs on each push. Commit `assignment.ipynb` and the `output/` files, then push.
- Expect: a passing run prints `Score: 100/100` and `All checks passed.`

What each check looks for: [CHECKS.md](CHECKS.md)
