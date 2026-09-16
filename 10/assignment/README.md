# Assignment 10: bounded modeling and honest evaluation

## Files

```text
assignment/
├── assignment.ipynb     # Provided notebook scaffold to complete
├── data/                # Provided fixtures
├── requirements.txt     # Provided pinned environment
├── check_assignment.py  # Provided completion checker
└── output/              # Generated artifacts to submit
```

## Setup

In VS Code, open the assignment folder and choose **Terminal → New Terminal**. You can also use your native terminal or WSL Ubuntu; change to the assignment directory before running these commands.

From this assignment directory, create a Python 3.13 environment and install the pinned requirements:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. If Python 3.13 is missing, run `uv python install 3.13`. Select this environment as the kernel when opening the notebook in VS Code or Jupyter. The course uses pandas 3.0.5.

Keep the supplied `data/` files unchanged. Open the complete assignment directory; the setup cell locates and verifies its fixtures in either a standalone assignment repository or the course repository. Restore missing or checksum-mismatched fixtures before continuing.

The supplied records are course-authored synthetic data and do not describe real people, customers, or operations. Complete the scaffold in `assignment.ipynb`.

## Question 1: Bounded OLS inference

### 1.1 Fit and interpret the model

Fit `finish_quality_score ~ mix_minutes + initial_temp_c` with an intercept. Export the coefficient summary, intervals for the supplied new case, and residuals for each run. Interpret the conditional associations and interval meanings.

> **Checkpoint — `output/inference_summary.csv`**

> **Checkpoint — `output/inference_case_intervals.csv`**

> **Checkpoint — `output/inference_residuals.csv`**

### 1.2 Inspect residuals

Save a residuals-versus-fitted figure and explain one assumption it can probe.

> **Checkpoint — `output/inference_residuals.png`**

## Question 2: Prediction contract and chronological split

### 2.1 Audit feature availability

State the prediction unit, prediction time, target, and target time. Record when each supplied candidate becomes available and whether to keep it.

> **Checkpoint — `output/availability_decisions.csv`**

### 2.2 Split chronologically

Create the supplied train, validation, and test periods and save their row counts and target-time ranges.

> **Checkpoint — `output/split_manifest.csv`**

## Question 3: Compare, freeze, and evaluate

### 3.1 Compare on validation

Fit a mean baseline and the linear pipeline using training rows. Compare their validation MAE, RMSE, and R2, then freeze the validation winner.

> **Checkpoint — `output/validation_metrics.csv`**

### 3.2 Evaluate the frozen choice

Evaluate the frozen approach on test once and export aligned predictions and metrics.

> **Checkpoint — `output/final_test_metrics.csv`**

> **Checkpoint — `output/final_predictions.csv`**

### 3.3 Interpret supplied binary predictions

Calculate accuracy, precision, and recall for the supplied model and dummy baseline, using zero precision when there are no predicted positives.

> **Checkpoint — `output/binary_metrics.csv`**

## Check Your Work

Run this from the assignment directory after saving your artifacts:

```bash
python check_assignment.py
```

Fix each failed check, regenerate the affected files, and run the checker again. It reads saved artifacts without running your code.

### Completion contract

Save the nine CSVs and one PNG below in `output/`. CSV columns must match the listed order, with no extra index or missing values. IDs and category rows must match the supplied data; row order may differ. Numeric results must match the requested calculations within 0.0001.

| Artifact | Columns in order | Completion criteria |
|---|---|---|
| `output/inference_summary.csv` | `term`, `estimate`, `standard_error`, `confidence_low_95`, `confidence_high_95` | Three OLS terms: Intercept, mix_minutes, initial_temp_c; estimates, standard errors, and 95% confidence bounds from the supplied inference data. |
| `output/inference_case_intervals.csv` | `mix_minutes`, `initial_temp_c`, `predicted_mean`, `mean_ci_low_95`, `mean_ci_high_95`, `prediction_ci_low_95`, `prediction_ci_high_95` | One supplied case (26.0 minutes, 22.0 °C), with fitted mean, mean-response interval, and individual prediction interval. |
| `output/inference_residuals.csv` | `run_id`, `actual`, `fitted`, `residual` | All 18 run IDs M01–M18 with observed, fitted, and observed-minus-fitted values. |
| `output/availability_decisions.csv` | `candidate_feature`, `latest_required_offset_hours`, `available_by_prediction_time`, `decision` | All five candidate features, with offsets 0 or 24, Boolean availability, and keep/exclude decisions. |
| `output/split_manifest.csv` | `partition`, `row_count`, `first_target_timestamp`, `last_target_timestamp` | Train: 29 target rows, April 2–30; validation: 8, May 1–8; test: 11, May 9–19, 2026. Timestamp strings use YYYY-MM-DDT00:00:00Z. |
| `output/validation_metrics.csv` | `approach`, `mae`, `rmse`, `r2` | MAE, RMSE, and R2 for mean_baseline and linear_pipeline on the same validation rows. |
| `output/final_test_metrics.csv` | `approach`, `mae`, `rmse`, `r2` | MAE, RMSE, and R2 for the frozen linear_pipeline on test. |
| `output/final_predictions.csv` | `batch_id`, `target_timestamp`, `actual_strength_mpa`, `predicted_strength_mpa` | One aligned row per test batch B038–B048, with its source target timestamp, actual strength, and final prediction. |
| `output/binary_metrics.csv` | `approach`, `accuracy`, `precision`, `recall` | Accuracy, precision, and recall for supplied_model and dummy_baseline from the supplied binary predictions. |
| `output/inference_residuals.png` | PNG file | Saved residuals-versus-fitted figure; the check verifies PNG format. |

Question 1 is worth 30 points, Question 2 is worth 35, Question 3 is worth 30, and the PNG is worth 5: 100 total.

## Submit

In VS Code Source Control, inspect your completed notebook and required `output/` files, then commit and push them. Alternatively, use **Add file → Upload files** on the GitHub website and commit the files at their required paths. Keep private data, credentials, virtual environments, and notebook checkpoints out of your submission. GitHub Actions runs the assignment checks automatically on every push. If your fork has Actions disabled, enable it once in the Actions tab. Review the feedback, then regenerate, check, commit, and push corrected artifacts if needed.
