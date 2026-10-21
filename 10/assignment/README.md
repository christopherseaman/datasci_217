# Assignment 10: Modeling Blood Pressure and Testing Predictions Honestly

## Overview

Fit and read an OLS model of blood pressure with `statsmodels`, audit features for leakage and split visits by target time, then compare a baseline with a pipeline on validation rows and test the frozen winner once, saving each result for the checks.

All four files are synthetic; the IDs belong to no one. Keep them exactly as handed out.

`data/clinic_bp.csv` holds 20 primary-care patients, one row each: `patient_id`, `age` in years, `bmi` in kg/m², and `sbp`, systolic blood pressure in mmHg.

```text
patient_id,age,bmi,sbp
P01,67,24.2,149
```

`data/followup_visits.csv` comes from a hypertension follow-up program: 48 visits, one a day from March 18 to May 4, 2026. The program predicts each patient's blood pressure at the follow-up visit two weeks later, and makes that prediction when today's visit ends.

| Column | Meaning |
| --- | --- |
| `visit_id` | `V01` to `V48`, in time order |
| `visit_time` | When today's visit ended: the prediction time (UTC) |
| `followup_time` | When the follow-up visit, 14 days later, measures the target: the target time (UTC) |
| `age`, `bmi` | Years and kg/m² |
| `sbp_today` | Systolic blood pressure measured at today's visit, mmHg |
| `a1c_result` | HbA1c from blood drawn today, %; the lab reports it the next day |
| `callback_sbp` | The home reading a nurse collects by phone three days later, mmHg |
| `sbp_followup` | Systolic blood pressure at the follow-up visit, mmHg: the target |

From late April the program also enrolled patients referred from the emergency department with uncontrolled blood pressure, so the latest visits run higher.

`data/feature_availability.csv` lists the five candidate features and `hours_after_visit`, how many hours after the visit ends each one becomes known.

`data/readmission_flags.csv` holds 20 discharged patients: `patient_id`, `readmitted_30d` (1 if readmitted within 30 days, else 0), `model_flag` (a supplied model's 1 or 0 flag), and `never_flag` (a policy that flags no one, so every value is 0).

## Setup

1. Fork and clone the repository, then open its folder in VS Code.
2. Create the environment:

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```
3. Open `assignment.ipynb` and select the `.venv` Python as the kernel.
4. Run the notebook's first two code cells.
    - Expect: `pandas: 3.0.5`, `statsmodels: 0.14.6`, `scikit-learn: 1.9.0`, and `data folder found: True` from the first; `patients: (20, 4)`, `candidates: (5, 2)`, `visits: (48, 9)`, and `flags: (20, 4)` from the second.

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/
│   ├── clinic_bp.csv             # supplied: Task 1's patients
│   ├── feature_availability.csv  # supplied: Task 2.1's candidate features
│   ├── followup_visits.csv       # supplied: Tasks 2.2 to 3.2's visits
│   └── readmission_flags.csv     # supplied: Task 3.3's flags
├── pyproject.toml          # supplied: the packages the notebook uses, and ipykernel
├── uv.lock                 # supplied: the exact versions `uv sync` installs
├── .python-version         # supplied: tells uv to use Python 3.13
├── CHECKS.md               # supplied: what each check looks for
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── ols_coefficients.csv         # you generate in Task 1.1
    ├── new_patient_intervals.csv    # you generate in Task 1.2
    ├── ols_residuals.csv            # you generate in Task 1.3
    ├── residuals_vs_fitted.png      # you generate in Task 1.3
    ├── availability_decisions.csv   # you generate in Task 2.1
    ├── split_summary.csv            # you generate in Task 2.2
    ├── validation_metrics.csv       # you generate in Task 3.1
    ├── test_metrics.csv             # you generate in Task 3.2
    ├── test_predictions.csv         # you generate in Task 3.2
    └── readmission_metrics.csv      # you generate in Task 3.3
```

## Task 1: Model blood pressure with statsmodels

Is higher BMI associated with higher systolic blood pressure among patients of the same age? Task 1 answers with Lecture 10's `statsmodels` workflow on `patients`.

### 1.1 Fit the model and save its coefficients

Lecture 10's "Code Snippet: OLS with the Formula API" and "Code Snippet: Uncertainty and a New-Patient Interval" show each step. In the Task 1.1 cell:

1. Fit `results`: build the model with `smf.ols()`, a formula with outcome `sbp` and predictors `age` and `bmi`, and `data=patients`, then call `.fit()`. It prints `rows used: 20.0` and `R-squared: 0.744`.
2. Build `coefficients`, a DataFrame with four columns: `coef` from `results.params`, `std_err` from `results.bse`, and `ci_lower` and `ci_upper` from columns `0` and `1` of `results.conf_int()`. Its index holds the three terms.
3. Name that index with `coefficients.index.name = "term"` (Lecture 04), then save `coefficients` to `COEFFICIENTS_PATH` keeping the index: leave out `index=False`, so the terms become the first column, headed `term`.

> **Checkpoint: `output/ols_coefficients.csv`**
> The header line `term,coef,std_err,ci_lower,ci_upper`, then three rows: `Intercept`, `age`, and `bmi`. The `bmi` row reads about `bmi,1.61,0.32,0.94,2.28`. Numbers may keep every digit or be rounded to one or two decimals. The intercept may be named `const`, as the array interface names it, and the term column may be saved without a header.

Read the `bmi` row as: at the same age, each 1 kg/m² higher BMI goes with about 1.6 mmHg higher fitted SBP (95% CI 0.94 to 2.28). These are observational records, so that is an association; it does not say that losing weight would lower a patient's SBP by that much.

### 1.2 Give a new patient both intervals

A new patient is 60 years old with a BMI of 31.0. In the Task 1.2 cell:

1. Get `intervals`: call `results.get_prediction()` on `new_patient`, then `.summary_frame()` with `alpha=0.05`. `new_patient` is already built for you.
2. Put the patient's `age` and `bmi` beside the intervals: combine `new_patient` and `intervals` side by side with `pd.concat()` and `axis=1` (Lecture 06), and name the result `new_patient_intervals`. It prints `rows: 1`.
3. Save `new_patient_intervals` to `INTERVALS_PATH` with `index=False`.

> **Checkpoint: `output/new_patient_intervals.csv`**
> The header line `age,bmi,mean,mean_se,mean_ci_lower,mean_ci_upper,obs_ci_lower,obs_ci_upper`, then one row for age 60 and BMI 31.0. Its `mean` is about 150.1 mmHg. Values may be rounded to one decimal, and `mean_se` may be left out.

The mean-response interval, about 146.1 to 154.2 mmHg, is where the _average_ SBP of patients like this one probably lies. The prediction interval, about 136.3 to 164.0 mmHg, is where _this_ patient's SBP probably lies; it adds person-to-person variation, so it is much wider.

### 1.3 Check the residuals

A residual is observed minus fitted SBP. Lecture 10's "Code Snippet: Residuals Versus Fitted Values" builds the table and the plot. In the Task 1.3 cells:

1. Build `residual_table` with four columns: `patient_id` from `patients["patient_id"]`, `observed` from `patients["sbp"]`, `fitted` from `results.fittedvalues`, and `residual` from `results.resid`. It prints `rows: 20`.
2. Save `residual_table` to `RESIDUALS_PATH` with `index=False`.
3. In the plot cell, draw `results.fittedvalues` (x) against `results.resid` (y) with `ax.scatter()`. Add a gray dashed horizontal line at 0 with `ax.axhline()`. Label the x axis `Fitted SBP (mmHg)` and the y axis `Residual (observed - fitted)`. Save to `RESIDUAL_PLOT_PATH` with `fig.savefig()`, `dpi=150`, and `bbox_inches="tight"`, before `plt.show()`.

> **Checkpoint: `output/ols_residuals.csv`**
> The header line `patient_id,observed,fitted,residual`, then 20 rows, `P01` to `P20`. P01 reads about `P01,149,142.71,6.29`. Values may be rounded to one decimal.

> **Checkpoint: `output/residuals_vs_fitted.png`**
> A PNG of the step 3 scatter. The check confirms only that the file is a PNG.

The residuals scatter around the dashed line with no clear curve or funnel, which is what a straight-line model should leave. The plot can show a curve or a changing spread; it cannot show that BMI _causes_ higher SBP.

## Task 2: Frame the prediction problem

The follow-up program wants to predict `sbp_followup` for each visit (the **prediction unit**) as soon as the visit ends (the **prediction time**, `visit_time`). The target is measured 14 days later, at `followup_time` (the **target time**). Lecture 10's "Prediction: Features, Targets, and Honest Splits" section and Demo 1 Part 8 frame a problem the same way.

### 2.1 Audit the candidate features

A feature the model will not have at prediction time is **leakage**, however useful it looks. Lecture 10's "Feature Availability" table and Demo 1's "Audit the candidate features" step run this check. In the Task 2.1 cell:

1. Make `decisions`, a `.copy()` of `candidates`.
2. Add the column `available`: `True` when `hours_after_visit` is 0 or less, so the feature is known by the end of the visit.
3. Add the column `decision` with `np.where()`: `"Keep"` where `available` is `True`, `"Exclude (leakage)"` where it is `False`.
4. Save `decisions` to `AVAILABILITY_PATH` with `index=False`.
5. Set `FEATURES` to the kept feature names: select the `candidate_feature` values of the available rows with `.loc` and a mask (Lecture 04), then turn them into a list with `.tolist()`. It prints `features: ['age', 'bmi', 'sbp_today']`.

> **Checkpoint: `output/availability_decisions.csv`**
> The header line `candidate_feature,hours_after_visit,available,decision`, then five rows, one per candidate. `a1c_result` reads `a1c_result,24,False,Exclude (leakage)`. `available` may be written `True`/`False`, `true`/`false`, or `1`/`0`, and any `decision` starting with `keep` or `exclude`, in any letter case, counts.

`callback_sbp` would look like the best feature of all, because it measures the same patient a few days before the target. It does not exist when the prediction is made, so a model trained with it would score well on old visits and could not be used on a new one.

### 2.2 Split on the target time

Split on when each **target** is measured, `followup_time`, so no training outcome is measured during the validation or test weeks. This is a **retrospective comparison**: every outcome is already known, so the split does not simulate forecasting at the visit times. Lecture 10's "Code Snippet: A Chronological Split" does the same with dates. In the Task 2.2 cell:

1. Select `train`, the visits whose `followup_time` is before `VALIDATION_START` (May 1, 2026, UTC); `valid`, those on or after `VALIDATION_START` and before `TEST_START` (May 9); and `test`, those on or after `TEST_START`. It prints `rows: 30 8 10`.
2. Build `split_summary` with one row per partition and four columns: `partition` (`"train"`, `"validation"`, `"test"`), `row_count` (each part's `len()`), `first_target_time` (each part's `["followup_time"].min()`), and `last_target_time` (its `.max()`).
3. Save `split_summary` to `SPLIT_PATH` with `index=False`.

> **Checkpoint: `output/split_summary.csv`**
> The header line `partition,row_count,first_target_time,last_target_time`, then three rows: `train`, `validation`, and `test`. The test row reads `test,10,2026-05-09 18:30:00+00:00,2026-05-18 21:51:00+00:00`. Timestamps are compared as instants, so `2026-05-09T18:30:00Z` and `2026-05-09 18:30:00+00:00` are the same value, and one written without a zone is read as UTC. `valid` for `validation` is fine.

## Task 3: Compare, freeze, and test once

### 3.1 Compare a baseline and a pipeline on the validation rows

A model has to beat a guess. Lecture 10's "Code Snippet: A Baseline and a Linear Pipeline" and "Code Snippet: Comparing on Validation Rows" show every step. In the Task 3.1 cell:

1. Create `baseline`, a `DummyRegressor` with `strategy="mean"`, and fit it on `train[FEATURES]` and `train[TARGET]`; it predicts the training mean for everyone.
2. Create `pipeline`, a `Pipeline` of two named steps, `"scale"` (a `StandardScaler`) then `"model"` (a `LinearRegression`), and fit it on the same training rows.
3. In the loop, predict the validation rows with each fitted approach and record its `mae` (`mean_absolute_error`), `rmse` (the square root, `np.sqrt()`, of `mean_squared_error`), and `r2` (`r2_score`), each comparing `valid[TARGET]` with the predictions.
4. Save `validation_metrics` to `VALIDATION_PATH` with `index=False`.
5. Freeze the winner, the approach with the lower validation MAE: `.idxmin()` on the `mae` column gives that row's label (Lecture 04), and `.loc` reads its `approach` into `winner`. It prints `frozen choice: linear_pipeline`.

> **Checkpoint: `output/validation_metrics.csv`**
> The header line `approach,mae,rmse,r2`, then two rows: `mean_baseline` and `linear_pipeline`. The pipeline's MAE is about 3.34 mmHg. Numbers may keep every digit or be rounded to one or two decimals.

The baseline's R² is below 0 on the validation rows: its training mean misses these patients by more than their own mean would. The pipeline cuts the typical miss from about 7.0 to 3.3 mmHg.

### 3.2 Freeze the winner and test it once

Lecture 10's "Freeze, Then Test Once" section and Demo 2 Part 11 show this step: same steps and settings, one test prediction, and that number goes in the report. In the Task 3.2 cell:

1. Create `final`, the frozen pipeline: a new `Pipeline` with the same `"scale"` and `"model"` steps as Task 3.1.
2. Fit it on the training rows, or refit it on the training plus validation rows as the lecture and Demo 2 do: stack `train` and `valid` into `train_valid` with `pd.concat()`, then fit on its `FEATURES` and `TARGET`. Either counts.
3. Predict the test rows once with `final.predict()` on `test[FEATURES]`, and name the result `test_predicted`.
4. Build `test_metrics`, one row with `approach` `"linear_pipeline"` and its `mae`, `rmse`, and `r2` for `test[TARGET]` and `test_predicted`. Put each value in a one-item list, as in `"approach": ["linear_pipeline"]`; a dictionary of single values makes `pd.DataFrame()` stop with `If using all scalar values, you must pass an index`.
5. Build `test_predictions`, a `.copy()` of the `visit_id`, `followup_time`, and `TARGET` columns of `test`, and add the column `predicted_sbp`, holding `test_predicted`.
6. Save both with `index=False`: `test_metrics` to `TEST_METRICS_PATH` and `test_predictions` to `TEST_PREDICTIONS_PATH`.

> **Checkpoint: `output/test_metrics.csv`**
> The header line `approach,mae,rmse,r2`, then one row, `linear_pipeline`. Its MAE is about 2.64 mmHg fitted on the training rows, or 2.54 refitted on training plus validation rows; both are accepted. Numbers may keep every digit or be rounded to one or two decimals.

> **Checkpoint: `output/test_predictions.csv`**
> The header line `visit_id,followup_time,sbp_followup,predicted_sbp`, then ten rows, `V39` to `V48`. V39 reads about `V39,2026-05-09 18:30:00+00:00,155,154.81` fitted on the training rows, or `154.94` refitted. Predictions may be rounded to one decimal, and `followup_time` may be written in any form Task 2.2 accepts.

The test visits include the referred patients whose blood pressure runs higher. The mean baseline would miss them badly; the pipeline follows them because `sbp_today` carries their higher readings.

### 3.3 Score the supplied readmission flags

For a yes/no outcome, accuracy alone can hide a flag that finds no one. Lecture 10's "Code Snippet: Accuracy Hides Missed Readmissions" computes the same three numbers. In the Task 3.3 cell:

1. In the loop, record each flag column's `accuracy` (`accuracy_score`), `precision` (`precision_score` with `zero_division=0`), and `recall` (`recall_score`), each comparing `flags["readmitted_30d"]` with `flags[column]`.
2. Save `readmission_metrics` to `READMISSION_PATH` with `index=False`.

> **Checkpoint: `output/readmission_metrics.csv`**
> The header line `approach,accuracy,precision,recall`, then two rows: `model_flag` and `never_flag`. `never_flag` reads `never_flag,0.8,0.0,0.0`. Numbers may keep every digit or be rounded to one or two decimals.

Flagging no one is right for 80% of patients and finds none of the four readmissions; the model's flag is right 85% of the time and finds three of them.

## Check your work

- Run the checks any time, even partway through: `python3 check_assignment.py` in the terminal (with the environment active) prints your score and what to fix.
- When **Restart** and **Run All** finish without errors, the notebook's last cell runs the same checks.
- Both use the latest checks from the course repository, the same checks GitHub runs on each push. Commit `assignment.ipynb` and the `output/` files, then push.
- Expect: a passing run ends with `Score: 100/100`.

What each check looks for: [CHECKS.md](CHECKS.md)
