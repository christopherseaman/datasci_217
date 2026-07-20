# Assignment 10 redesign blueprint

Status: **design-correction author evidence; implementation correction should
wait for independent reverification**.

This is an implementation contract, not an implemented assignment. The
candidate author evidence below was recomputed outside the repository with
CPython 3.12.13 and the direct Lecture 10 package candidates. It is not
certified-release evidence until the course-wide transitive constraints and
container lock pass the release gate in section 5. No Assignment 10 source file
is changed by this blueprint.

## 1. Decision

Replace the legacy Assignment 10 rather than adapting it in place.

The replacement is one practical, competence-oriented notebook with three
cumulative tasks:

1. fit and interpret one bounded multivariable OLS model, including a 95%
   coefficient interval, a mean-response interval, an individual prediction
   interval, and one residual diagnostic;
2. state a prediction contract, audit feature availability, identify leakage,
   and make a chronological train/validation/test split;
3. compare a mean baseline with one train-only linear Pipeline on validation,
   freeze that choice, evaluate it once on test, and calculate classification
   metrics from supplied binary predictions without fitting a classifier.

The assignment deliberately stops there. It does not ask learners to fit or
compare advanced models.

## 2. Legacy disposition

Remove the current `10/assignment/**` learner surface when the replacement is
implemented. Preserve it only in Git history.

The old assignment is unsuitable as a starting template because it:

- downloads California Housing during execution rather than using a pinned
  course-authored fixture;
- asks for p-values, AIC, interaction terms, full-dataset prediction, random
  forest, feature importance, and XGBoost;
- compares models on the test set and therefore spends the test set on model
  selection;
- has multiple Markdown/notebook surfaces that can drift;
- does not provide a dependency-free readiness checker, independent central
  grader, exact output inventory, alternate-function tests, or adversarial
  release harness;
- carries legacy Jupytext/lower-bound environment assumptions instead of the
  course's exact runtime contract.

Delete the legacy `assignment.md`, notebook, README, and requirements when the
replacement is implemented. The notebook, README, and requirements are replaced
at the same paths; they are not retained as compatibility surfaces. Do not
retain a California Housing fixture, runtime fetch, forest/XGBoost cell, or
compatibility alias for an old question.

## 3. Alignment and scope

### 3.1 Required prior knowledge

The notebook may freely use only material already established by Lectures
1–10:

- Python expressions, functions, dictionaries, lists, and assertions;
- pandas loading, selection, filtering, grouping, missingness checks, and
  timestamp conversion;
- Matplotlib Figure/Axes construction and saving;
- past-only temporal reasoning and chronological partitions from Lecture 9;
- the Lecture 10 OLS, prediction-contract, leakage, baseline, Pipeline, and
  metric vocabulary.

Every modeling term specific to this assignment is defined in a protected
Markdown cell before the first student cell that uses it.

### 3.2 In scope

- one statsmodels formula-interface OLS model with its implicit intercept and
  two supplied predictors;
- coefficient estimate, standard error, and 95% confidence interval;
- association language and the distinction between association and causation;
- fitted values, residuals, and a residuals-versus-fitted diagnostic;
- one confidence interval for a mean response and one prediction interval for
  an individual outcome at the same supplied predictor values;
- prediction unit, prediction time, target, target time, feature availability,
  and leakage;
- one chronological train/validation/test split on `target_timestamp`;
- `DummyRegressor(strategy="mean")`;
- exactly one `Pipeline` containing `StandardScaler` followed by
  `LinearRegression`;
- MAE, RMSE, and R-squared for regression;
- accuracy, precision, and recall on supplied binary labels/predictions;
- validation-only choice followed by one frozen final test evaluation.

### 3.3 Out of scope

Reject these from student source and omit them from prompts, hints, and rubric:

- p-values, significance-star tables, hypothesis-test decisions, AIC, BIC,
  interaction terms, polynomial terms, stepwise selection, and causal claims;
- regularization, cross-validation, hyperparameter search, feature selection,
  and feature importance;
- decision trees, random forests, boosting, XGBoost, deep learning, forecasting
  packages, survival analysis, deployment, monitoring, and model serving;
- fitting any classifier; the binary section uses predictions supplied in a
  fixture;
- refitting after validation, evaluating both candidates on test, or using test
  results to revise the model;
- remote data, runtime data generation, random data, notebook shell commands,
  Colab drive mounting, and environment-specific absolute paths;
- a bonus task. Advanced optional material remains in `10/BONUS.md`, outside
  this core assignment and outside its points.

## 4. Learner-facing assessment contract

### Task 1 — bounded inference and model interpretation

Use the 18 course-authored synthetic mixing runs. The row grain is one mixing
run. Fit:

```text
finish_quality_score ~ mix_minutes + initial_temp_c
```

Required work:

1. Define `fit_bounded_ols(inference_table, predictor_columns,
   outcome_column)` with `statsmodels.formula.api` / `smf.ols`. Build the
   formula from `outcome_column` and the ordered `predictor_columns`; do not
   hard-code canonical column names or use the matrix API. The formula
   interface supplies the implicit `Intercept` term. The function has no file
   I/O or canonical IDs.
2. Call it with predictors in the exact order `mix_minutes`,
   `initial_temp_c`.
3. Build and save the three-row coefficient table with estimates, standard
   errors, and 95% confidence bounds.
4. For the supplied case `mix_minutes=26.0`, `initial_temp_c=22.0`, save the
   predicted mean, its 95% mean-response interval, and the 95% individual
   prediction interval.
5. Create and save a residuals-versus-fitted Figure with a visible zero line,
   title, and axis labels.
6. Explain the `mix_minutes` coefficient in conditional association language,
   explain what its 95% confidence interval describes, compare the two new-case
   intervals, and identify one assumption the residual plot can probe and one
   assumption it cannot establish.

The prompt must say that synthetic observational association is not evidence
that changing mixing time causes the outcome to change.

### Task 2 — prediction contract, availability, leakage, and split

Use the 48 synthetic batch cases and five-row feature-availability inventory.
The protected term cell defines:

- **prediction unit**: the entity for which one prediction is produced;
- **prediction time**: when inputs must be available;
- **target**: the value to predict;
- **target time**: when that value is observed;
- **feature availability**: whether all information needed for a feature exists
  no later than prediction time;
- **leakage**: use of information that would not be available for the real
  prediction;
- **training set**: rows used to fit parameters;
- **validation set**: later rows used to compare the two supplied approaches;
- **test set**: later untouched rows used once after the choice is frozen.

Required work:

1. Fill the exact machine-readable contract values from `fixture.json`:
   unit `one synthetic batch`, prediction time `prediction_timestamp`, target
   `next_day_strength_mpa`, target time `target_timestamp`, and feature list
   `batch_sequence`, `ambient_temp_c`, `pre_mix_moisture_pct`.
2. Define `audit_feature_availability(candidate_table)`. A candidate is
   available when `latest_required_offset_hours <= 0`; output `keep` for
   available rows and `exclude` otherwise. Preserve input order and do not
   mutate the input.
3. Explain why the two `+24` candidates leak future information.
4. Define `build_chronological_splits(prediction_table, validation_start,
   test_start)`. Sort stably by `target_timestamp`, assign rows before
   `validation_start` to train, rows from validation start through immediately
   before test start to validation, and later rows to test. Return the three
   copied DataFrames and a three-row manifest.
5. Use UTC-aware cutoffs `2026-05-01T00:00:00Z` and
   `2026-05-09T00:00:00Z`. Save the availability decisions and split manifest.
6. Explain why a chronological split matches this contract better than a
   shuffled split and why validation and test have different roles.

The split function rejects `validation_start >= test_start`, missing
timestamps, duplicate `batch_id`, and any row for which
`prediction_timestamp >= target_timestamp` with a clear `ValueError`.

### Task 3 — baseline, validation choice, frozen test, supplied binary metrics

Required work:

1. Define `regression_metrics(actual, predicted)` returning exactly the keys
   `mae`, `rmse`, and `r2` as Python floats.
2. Define `fit_prediction_candidates(train_table, feature_columns,
   target_column)` returning exactly:
   - `mean_baseline`: a fitted `DummyRegressor(strategy="mean")`;
   - `linear_pipeline`: a fitted Pipeline whose named steps are
     `scale=StandardScaler()` and `linear=LinearRegression()`.
3. Fit both candidates only on the 29 training rows.
4. Use the protected `record_predictions` helper once per candidate on the
   validation rows and save both metric rows.
5. Define `choose_validation_winner(metrics_table, metric_column)`. Ignore
   nonfinite metric rows, choose the finite minimum, and break an exact numeric
   tie by lexicographically smaller approach name. Do not mutate the input.
   Call it with `metric_column="mae"`.
6. Execute the protected freeze cell. It freezes the selected approach and
   opens one test-evaluation call only after the two expected validation calls.
7. Use `record_predictions` exactly once on test and only for the frozen
   approach. Do not refit. Save one metric row and all 11 predictions.
8. Define `compute_binary_metrics(prediction_table, actual_column,
   prediction_columns)`. Calculate accuracy, precision, and recall for the
   supplied model and supplied dummy prediction; use `zero_division=0`. Do not
   fit a classifier.
9. Explain why the validation result may guide the choice while the test result
   may not; interpret the negative validation R-squared for the mean baseline;
   state one limitation of the single final test estimate; and compare what
   accuracy versus recall reveals for the supplied binary predictions.

## 5. Runtime and platform contract

### 5.1 Candidate direct environment

Use CPython `3.12.13` and these direct requirement candidates for author
recomputation and implementation work:

```text
matplotlib==3.11.1
numpy==2.0.2
pandas==3.0.3
scikit-learn==1.9.0
statsmodels==0.14.6
```

The notebook kernelspec is exactly
`{"display_name":"Python 3","language":"python","name":"python3"}` and
notebook format is 4.5. The grader additionally pins `ipykernel==6.29.5`,
`nbclient==0.10.2`, `nbformat==5.10.4`, and `Pillow==12.3.0` as direct
candidates. These direct pins do not constitute the release lock.

`check_assignment.py` begins with exact PEP 723 metadata, after an optional
shebang and before its module docstring:

```python
# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "matplotlib==3.11.1",
#   "numpy==2.0.2",
#   "pandas==3.0.3",
#   "scikit-learn==1.9.0",
#   "statsmodels==0.14.6",
# ]
# ///
```

The five dependency entries exactly mirror `requirements.txt`, including order
and spelling; no grader-only or transitive package appears in this block. The
checker implementation remains standard-library-only: it imports none of these
five packages and uses `importlib.metadata` only to verify the provisioned
versions. PEP 723 is wrapper provisioning so a direct
`uv run check_assignment.py` creates/uses the declared candidate environment.
Plain `python check_assignment.py` remains valid only in an already-provisioned
exact environment. This distinction does not make the five libraries checker
implementation dependencies.

### 5.2 Required transitive-lock and container gate

Release requires both an exact transitive `constraints.txt` in the
instructor-only grader bundle and an immutable digest for the certified central
execution container. The constraints must cover the notebook, all direct and
transitive dependencies, the public checker probes, notebook execution, image
decoding, and both grader entry points. The plain-Python bootstrap installs
`requirements.txt` under that constraint file. The author harness and central
grader run in the same locked container and verify every installed distribution
against the lock before executing learner code.

The instructor-bundle copy is byte-identical to the course-wide constraints
artifact used to provision and certify clean local Jupyter. It is distributed
through the course environment setup, not accepted as a learner-editable
assignment file and not included in the copied learner surface.

The lock contents and container digest are deliberately unresolved: the
course-wide local/Colab/central certification pass in
`work/course_dependency_alignment.md` has not occurred. Do not infer or freeze
transitive versions from the environment used for this blueprint. The current
section 9 bytes and numbers are candidate author evidence. They become
certified-release evidence only if two clean executions in the locked container
reproduce all eight CSV byte strings and the stated numeric/live-object
semantics. Any difference requires updating this blueprint, rerunning the
adversarial harness, and obtaining another independent design verification
before official release hashes are frozen. No release may waive this gate.

After this blueprint passes independent design verification, candidate
implementation may proceed without inventing the unresolved lock. In that
explicitly non-release candidate state, `_grader_selftest/constraints.txt` is
the sole target-release path that is absent; no empty, partial, or environment
freeze placeholder is permitted. The author harness uses the direct candidate
environment, and production `autograder.py` must report a nonzero
infrastructure failure if invoked without the real lock. Learner surfaces,
grader logic, and adversarial cases may be developed. Non-release candidate
integrity constants in section 8.1 are required during that work and are not
blocked by the missing lock. Plain-Python production-bootstrap acceptance,
official release-frozen hashes, exact release evidence, and any release claim
remain blocked until the real file and container digest exist.

### 5.3 Local Jupyter and Colab

Local Jupyter execution is mandatory and is the release reference. Validate by
restart kernel, clear all output, run all cells, run the public checker, and run
all cells a second time.

Assignment Colab support is conditional. It is supported only when the course
launch route places the entire assignment tree, including `data/`, in the Colab
runtime. A standalone notebook upload is not a supported data transport. Do not
add Colab-only cells, `drive.mount`, `files.upload`, remote URLs, package
installs, or `/content` paths.

The setup cell searches only the current working directory and its parents,
checking each for either the assignment marker directly or `10/assignment`.
It does not recursively crawl the filesystem. It verifies fixture bytes before
removing any owned output.

### 5.4 Git workflow

Learner instructions use GUI Git: accept/open the Classroom50 assignment, clone
or open it with the approved GUI, work in local Jupyter, restart/run all and run
the checker, inspect required outputs in the GUI, then commit and push the
notebook plus outputs through the GUI. Do not teach shell `git` commands.

## 6. Exact target-release author package and distributed inventories

```text
10/assignment/
├── .gitignore
├── .python-version
├── PLATFORM_CHECK.md
├── README.md
├── assignment.ipynb
├── check_assignment.py
├── requirements.txt
├── data/
│   ├── fixture.json
│   ├── mixing_runs.csv
│   ├── batch_strength.csv
│   ├── feature_availability.csv
│   └── supplied_binary_predictions.csv
├── output/
│   └── .gitkeep
└── _grader_selftest/
    ├── README.md
    ├── autograder.py
    ├── classroom50_grader.py
    ├── constraints.txt
    ├── requirements.txt
    └── run.py
```

This is the exact target-release author package before a solution is executed.
The exact distributed learner starter is the same tree with
`_grader_selftest/` excluded; its `output/` contains only `.gitkeep`. The sole
permitted non-release author-candidate exception is the absent instructor-owned
`_grader_selftest/constraints.txt` described in section 5.2. That exception
does not alter the distributed learner starter and is not a target-release
author topology.

`_grader_selftest/` is instructor-owned, is never distributed as learner
content, and is excluded from the disposable learner surface used by the
grader. The exact distributed `.gitignore` is:

```text
.ipynb_checkpoints/
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
```

The distributed starter and solved/accepted learner package outside `output/`
both contain exactly:

```text
.gitignore
.python-version
PLATFORM_CHECK.md
README.md
assignment.ipynb
check_assignment.py
requirements.txt
data/fixture.json
data/mixing_runs.csv
data/batch_strength.csv
data/feature_availability.csv
data/supplied_binary_predictions.csv
```

`check_assignment.py` is one of those immutable learner support files. Its
exact static contract includes the section 5.1 PEP 723 block, standard-library
imports only, and no dependency entry beyond the five exact lines mirrored from
`requirements.txt`. Candidate and release integrity profiles hash its complete
raw bytes; topology acceptance without this static contract is insufficient.

The solved reference and an accepted learner submission add exactly these nine
generated regular files beside the starter's retained `output/.gitkeep`:

```text
output/inference_summary.csv
output/inference_case_intervals.csv
output/inference_residuals.png
output/availability_decisions.csv
output/split_manifest.csv
output/validation_metrics.csv
output/final_test_metrics.csv
output/final_predictions.csv
output/binary_metrics.csv
```

No generated artifact is present in the distributed starter. The accepted
output inventory is exactly `.gitkeep` plus those nine files.

At the learner-package root, the delivery system may additionally provide
exactly these regular files:

```text
.classroom50.yaml
.github/workflows/autograde.yaml
```

The repository-metadata boundary is top-level only. Inventory checks ignore
only descendants of the genuine top-level `.git/` directory and accept either
or both exact delivery-owned regular files only at the paths above. Their
delivery-controlled contents are not hashed. A same-named `.git`, `.github`,
`.classroom50.yaml`, workflow, or delivery-marker entry below an ordinary
directory is learner content and is rejected. An alternate workflow path,
symlinked delivery file, injected `_grader_selftest/`, other extra file, and any
symlink anywhere in the learner package are rejected. The assignment root may
itself have any name and may be located at any absolute depth; boundary rules
are relative to the resolved assignment root. Output inventory is checked
separately and exactly. Do not ignore `output/`, CSV, JSON, PNG, notebook
output, or nested delivery metadata.

## 7. Pinned fixture contract

All five fixture files are course-authored, synthetic, static UTF-8 files with
LF line endings and a final newline. No fixture is regenerated during learner
or grader execution.

| File | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `data/fixture.json` | manifest | 2170 | `aa50eeffc2b07c5d98cb56a0e3d18115909958f777899d5d403cf6323dd1de41` |
| `data/mixing_runs.csv` | 18 | 370 | `00b8a1ce84110f4a7fa85620742283c82a4b9d600dbe0ebea0d4721956938957` |
| `data/batch_strength.csv` | 48 | 3449 | `f14faf7da64347dfc255aa84b14e79eef7f2d0de94b394c747323319d937baa3` |
| `data/feature_availability.csv` | 5 | 155 | `a47b8df048607045640b9a6785b038fe1c70036f58d5b61ed20ec98860b556da` |
| `data/supplied_binary_predictions.csv` | 12 | 184 | `7a8809010fa94345cd04787c826ef86ee5fd13cbf0bd95953e2220c3294a239a` |

### 7.1 Exact schemas and data recipe

`mixing_runs.csv` has columns `run_id`, `mix_minutes`, `initial_temp_c`,
`finish_quality_score`; rows are `M01` through `M18`. Materialize once from:

```python
mix_minutes = np.array([12,18,25,31,14,22,28,35,16,24,30,38,20,27,33,40,19,29], dtype="int64")
initial_temp_c = np.array([19.5,21,20,22.5,24,23,18.5,25,20.5,24.5,21.5,19,22,26,23.5,20,25.5,18])
noise = np.array([0.8,-0.6,1.2,-1.0,0.4,-0.8,1.1,-0.5,0.2,-1.2,0.7,0.9,-0.4,1.0,-0.7,0.3,-0.9,0.6])
finish_quality_score = np.round(48 + 0.65*mix_minutes + 0.9*initial_temp_c + noise, 2)
```

`batch_strength.csv` has `batch_id`, `prediction_timestamp`,
`target_timestamp`, `batch_sequence`, `ambient_temp_c`,
`pre_mix_moisture_pct`, `next_day_strength_mpa`. Rows are `B001` through
`B048`; prediction timestamps are consecutive UTC days starting 2026-04-01;
target timestamps are one day later. Use NumPy 2.0.2:

```python
i = np.arange(1, 49)
prediction_timestamp = pd.date_range(
    start="2026-04-01T00:00:00Z",
    periods=48,
    freq="D",
)
target_timestamp = prediction_timestamp + pd.Timedelta(days=1)
ambient = np.round(19 + 4.2*np.sin(i/4.8) + 0.7*np.cos(i/2.3), 3)
moisture = np.round(5.4 + 0.45*np.cos(i/5.2) + 0.12*np.sin(i/2.1), 3)
noise = np.array([
 0.42,-0.31,0.18,-0.52,0.27,0.05,-0.24,0.36,
-0.11,0.49,-0.38,0.16,0.31,-0.47,0.08,0.22,
-0.19,0.41,-0.33,0.12,0.28,-0.44,0.07,0.35,
-0.15,0.46,-0.29,0.14,0.25,-0.40,0.09,0.32,
-0.17,0.43,-0.35,0.11,0.29,-0.45,0.06,0.37,
-0.13,0.48,-0.27,0.15,0.24,-0.39,0.10,0.34])
strength = np.round(22 + 0.18*i + 0.75*ambient - 1.7*moisture + noise, 3)
batch_strength = pd.DataFrame({
    "batch_id": [f"B{row:03d}" for row in i],
    "prediction_timestamp": prediction_timestamp,
    "target_timestamp": target_timestamp,
    "batch_sequence": i,
    "ambient_temp_c": ambient,
    "pre_mix_moisture_pct": moisture,
    "next_day_strength_mpa": strength,
})
```

Build all four CSV DataFrames in the stated column order. Serialize
`batch_strength.csv` exactly with:

```python
batch_strength.to_csv(
    index=False,
    encoding="utf-8",
    lineterminator="\n",
    date_format="%Y-%m-%dT%H:%M:%SZ",
)
```

This produces literal UTC fields such as `2026-04-01T00:00:00Z`, not pandas'
default space/offset representation. Serialize the other three CSV fixtures
with `to_csv(index=False, encoding="utf-8", lineterminator="\n")`. The explicit
write format does not change the protected loaded dtype contract: both batch
timestamp columns are parsed with `pd.to_datetime(..., utc=True)` and must load
as `datetime64[us, UTC]` under the candidate pandas pin.

`feature_availability.csv` is exactly:

```csv
candidate_feature,latest_required_offset_hours
batch_sequence,0
ambient_temp_c,0
pre_mix_moisture_pct,0
early_24h_strength_mpa,24
next_day_strength_mpa,24
```

`supplied_binary_predictions.csv` uses `C01` through `C12` and:

```python
actual_label = [0,0,1,0,1,0,0,0,1,0,0,0]
supplied_model_prediction = [0,0,1,0,1,1,0,0,0,0,0,0]
dummy_prediction = [0,0,0,0,0,0,0,0,0,0,0,0]
```

The exact indent-2/final-newline manifest records fixture ID
`a10-bounded-modeling-v1`, provenance `Course-authored synthetic laboratory and
batch records; no real, identifying, operational, or customer data.`, all four
file paths/grains/counts/ordered columns/hashes, the supplied new case, the
prediction contract, and the two UTC split cutoffs with
`split_on: target_timestamp`. Freeze it at the stated byte count/hash.

The manifest's exact key order and parsed value are:

```json
{
  "fixture_id": "a10-bounded-modeling-v1",
  "provenance": "Course-authored synthetic laboratory and batch records; no real, identifying, operational, or customer data.",
  "files": [
    {
      "path": "mixing_runs.csv",
      "row_grain": "one synthetic mixing run",
      "row_count": 18,
      "columns": ["run_id", "mix_minutes", "initial_temp_c", "finish_quality_score"],
      "sha256": "00b8a1ce84110f4a7fa85620742283c82a4b9d600dbe0ebea0d4721956938957"
    },
    {
      "path": "batch_strength.csv",
      "row_grain": "one synthetic batch prediction case",
      "row_count": 48,
      "columns": ["batch_id", "prediction_timestamp", "target_timestamp", "batch_sequence", "ambient_temp_c", "pre_mix_moisture_pct", "next_day_strength_mpa"],
      "sha256": "f14faf7da64347dfc255aa84b14e79eef7f2d0de94b394c747323319d937baa3"
    },
    {
      "path": "feature_availability.csv",
      "row_grain": "one candidate feature availability claim",
      "row_count": 5,
      "columns": ["candidate_feature", "latest_required_offset_hours"],
      "sha256": "a47b8df048607045640b9a6785b038fe1c70036f58d5b61ed20ec98860b556da"
    },
    {
      "path": "supplied_binary_predictions.csv",
      "row_grain": "one supplied binary prediction case",
      "row_count": 12,
      "columns": ["case_id", "actual_label", "supplied_model_prediction", "dummy_prediction"],
      "sha256": "7a8809010fa94345cd04787c826ef86ee5fd13cbf0bd95953e2220c3294a239a"
    }
  ],
  "inference_new_case": {"mix_minutes": 26.0, "initial_temp_c": 22.0},
  "prediction_contract": {
    "unit": "one synthetic batch",
    "prediction_time": "prediction_timestamp",
    "target": "next_day_strength_mpa",
    "target_time": "target_timestamp",
    "feature_columns": ["batch_sequence", "ambient_temp_c", "pre_mix_moisture_pct"]
  },
  "split_boundaries": {
    "validation_start": "2026-05-01T00:00:00Z",
    "test_start": "2026-05-09T00:00:00Z",
    "split_on": "target_timestamp"
  }
}
```

For the 2170-byte file, serialize this insertion-ordered object with
`json.dumps(value, indent=2) + "\n"`; the arrays expand to one item per line.

### 7.2 Exact loaded dtypes and keys

The protected load contract is explicit; pandas inference is not part of the
assessment. Every table has a fresh `RangeIndex` after load, in file order.
String columns use `pd.StringDtype(storage="python")`, integers use NumPy
`int64`, floats use NumPy `float64`, and UTC values use pandas
`datetime64[us, UTC]` under the candidate pandas pin.

| Fixture | Ordered loaded columns and exact dtypes | Key and required value checks |
|---|---|---|
| `fixture.json` | insertion-ordered Python `dict`; top-level keys exactly `fixture_id`, `provenance`, `files`, `inference_new_case`, `prediction_contract`, `split_boundaries`; `files` is a four-element `list[dict]`; remaining nested key order is exactly the JSON in section 7.1 | `fixture_id` is the nonempty manifest key; all path values are safe single-level fixture names; file paths are unique and match the four tables |
| `mixing_runs.csv` | `run_id: StringDtype(storage="python")`, `mix_minutes: int64`, `initial_temp_c: float64`, `finish_quality_score: float64` | `run_id` is unique/nonmissing and exactly `M01`–`M18`; all numeric values are finite |
| `batch_strength.csv` | `batch_id: StringDtype(storage="python")`, `prediction_timestamp: datetime64[us, UTC]`, `target_timestamp: datetime64[us, UTC]`, `batch_sequence: int64`, `ambient_temp_c: float64`, `pre_mix_moisture_pct: float64`, `next_day_strength_mpa: float64` | `batch_id` is unique/nonmissing and exactly `B001`–`B048`; `batch_sequence` is exactly `1`–`48`; timestamps are nonmissing and prediction is strictly before target row by row; numeric values are finite |
| `feature_availability.csv` | `candidate_feature: StringDtype(storage="python")`, `latest_required_offset_hours: int64` | `candidate_feature` is unique/nonmissing in the exact file order; offsets are nonmissing; availability is inferred only by `offset <= 0` |
| `supplied_binary_predictions.csv` | `case_id: StringDtype(storage="python")`, `actual_label: int64`, `supplied_model_prediction: int64`, `dummy_prediction: int64` | `case_id` is unique/nonmissing and exactly `C01`–`C12`; every label/prediction is exactly integer `0` or `1` |

The loader rejects a missing/extra/reordered column, duplicate or missing key,
dtype mismatch after explicit loading/conversion, nonfinite numeric value,
invalid UTC timestamp, invalid binary value, or mismatch with the manifest
before any output cleanup.

## 8. Notebook topology and starter/solution behavior

The notebook has exactly 30 cells, the IDs and types below, and no other cells.
`P` means protected; `S` means student-editable.

| # | Cell ID | Type | Owner | Exact purpose and accepted solution behavior |
|---:|---|---|---|---|
| 0 | `a10-header` | Markdown | P | Title, three-task map, local-first/conditional-Colab statement, restart/run-all/checker/GUI-Git workflow, and scope boundary. |
| 1 | `a10-setup` | Code | P | Imports only candidate pinned libraries, including `statsmodels.formula.api as smf`; resolves root; verifies manifest and fixture hashes before cleanup; creates `output/`; deletes only the nine owned artifact paths; defines paths, expected schemas, `record_predictions`, and empty fit/prediction ledgers. No model is fitted. |
| 2 | `a10-terms-inference` | Markdown | P | Defines response, predictor, conditional coefficient, association/causation, standard error, confidence interval, fitted value, residual, mean-response interval, individual prediction interval, and residual assumptions before Task 1. |
| 3 | `a10-load` | Code | S | Load four fixtures with explicit dtypes, parse timestamps with `utc=True`, assert schemas/grains/keys/required values, and make deep source snapshots. Starter has explicit TODO load scaffolds; solution has no imports or path literals beyond supplied constants. |
| 4 | `a10-task1-prompt` | Markdown | P | Exact Task 1 formula, supplied new case, artifacts, and noncausal language constraint. |
| 5 | `a10-ols-function` | Code | S | Define `fit_bounded_ols`; validate/copy the ordered argument-named columns, derive `outcome ~ predictor_1 + predictor_2` from its arguments, fit with `smf.ols(..., data=copied_table).fit()`, and return the result with ordered `Intercept`/predictor terms; no matrix API, I/O, canonical names, or p-value/AIC access. |
| 6 | `a10-task1-run` | Code | S | Fit canonical model; construct coefficient and case-interval DataFrames; assert row/column order, finite values, nested interval widths, input nonmutation, and visibly display both tables. |
| 7 | `a10-residual-figure` | Code | S | Sole PNG-save owner. Create one 7.2x4.8 inch Figure at 100 DPI, scatter fitted versus residual, add the horizontal zero line and exact title/labels, apply tight layout, save once with fixed metadata, visibly display the same live Figure, and close it without discarding the `fig` reference. No other cell calls `savefig`. |
| 8 | `a10-task1-save` | Code | S | Write only the two inference CSVs with index false, UTF-8, LF, final newline, `float_format='%.6f'`; round-trip explicit dtypes; display the two CSV paths and the already-saved PNG path. No Figure creation or save. |
| 9 | `a10-task1-explain` | Markdown | S | Conditional coefficient/CI interpretation, interval comparison, one residual-checkable assumption, one non-checkable assumption, explicit noncausal statement. Starter has prompts, not solution prose. |
| 10 | `a10-terms-prediction` | Markdown | P | Defines prediction contract, availability, leakage, train/validation/test, chronological split, baseline, and freeze before Task 2. |
| 11 | `a10-task2-contract` | Markdown | S | State prediction question and explain unit, prediction time, target, and target time. |
| 12 | `a10-task2-values` | Code | S | Fill exact machine-readable contract dictionary, feature list, and UTC cutoffs from manifest; visibly display. |
| 13 | `a10-availability-function` | Code | S | Define `audit_feature_availability`; deep copy, vectorized `<=0`, `np.where` keep/exclude, preserve order; no canonical feature strings in function. |
| 14 | `a10-split-function` | Code | S | Define `build_chronological_splits`; boundary checks, stable time sort, nonoverlapping half-open rules, copied frames, exact manifest; derive values from arguments and do no I/O. |
| 15 | `a10-task2-run` | Code | S | Run both; assert canonical decisions, 29/8/11 counts, complete/exclusive ID conservation, chronological boundaries, source nonmutation; display decisions/manifest. |
| 16 | `a10-task2-save` | Code | S | Write/round-trip the two exact CSVs with stable ordering and visible paths. |
| 17 | `a10-task2-explain` | Markdown | S | Explain both leaked candidates, why chronology matches the contract, why validation/test roles differ, and why shuffle can misrepresent future use. |
| 18 | `a10-terms-evaluation` | Markdown | P | Defines train-only transformation, Pipeline, MAE, RMSE, R-squared including negative values, labels, accuracy, precision, recall, and zero-denominator policy. |
| 19 | `a10-regression-metrics-function` | Code | S | Define `regression_metrics`; use sklearn metrics and square-root MSE; return three Python floats. |
| 20 | `a10-candidates-function` | Code | S | Define `fit_prediction_candidates`; instantiate only required baseline/Pipeline, fit each only on supplied train argument, call protected fit audit, return exact dictionary; no prediction. |
| 21 | `a10-validation-run` | Code | S | Define `choose_validation_winner`; fit candidates on train; protected recorder once per candidate on validation; build/order/round metrics; call the function on unrounded metric values with `"mae"`; display metrics/name. No test/final/refit/direct `.predict`. |
| 22 | `a10-validation-save` | Code | S | Write/round-trip validation CSV and display path; no final/test references. |
| 23 | `a10-freeze` | Code | P | Require two exact train fits and two exact validation ledger entries, no test entry; require the name returned by `choose_validation_winner`; assign immutable `FROZEN_SELECTED_APPROACH`; assert `linear_pipeline`; open one-use test gate. |
| 24 | `a10-final-test-run` | Code | S | Protected recorder exactly once for frozen candidate/test; compute one metric row and 11 ordered predictions; no fit, comparison, or selection assignment; display both. |
| 25 | `a10-final-test-save` | Code | S | Write/round-trip final metrics/predictions with exact schemas/order/precision; display paths. |
| 26 | `a10-binary-function` | Code | S | Define `compute_binary_metrics`; accept actual column and ordered approach-to-column mapping, validate binary labels, calculate metrics with `zero_division=0`, preserve mapping order; no fit. |
| 27 | `a10-binary-run-save` | Code | S | Run supplied predictions; assert approaches and broad relationships; write/round-trip/display exact CSV. |
| 28 | `a10-task3-explain` | Markdown | S | Explain selection/test roles, negative baseline R-squared, one-test limitation, and accuracy/recall tradeoff without claiming classifier training. |
| 29 | `a10-final-verify` | Code | P | Assert exact output inventory, unchanged sources, exact ledgers, frozen selection, one test call, schemas/rows/finite values, committed PNG signature/IHDR/dimensions/nontrivial bytes, retained live Figure semantics, and no open Figures; print concise readiness summary. Pillow decoding of committed and fresh PNGs belongs only to the independent central grader. |

### 8.1 Non-release candidate integrity and official release re-freeze

Candidate implementation must not wait for the unresolved transitive lock to
protect immutable learner surfaces. It defines two explicit non-release
constant maps in the independent central grader:

```text
CANDIDATE_PROTECTED_FILE_SHA256
CANDIDATE_PROTECTED_CELL_SHA256
```

`CANDIDATE_PROTECTED_FILE_SHA256` contains exactly these 12 learner paths and
the lowercase 64-hex SHA-256 of each file's raw bytes:

```text
.gitignore
.python-version
PLATFORM_CHECK.md
README.md
check_assignment.py
requirements.txt
data/fixture.json
data/mixing_runs.csv
data/batch_strength.csv
data/feature_availability.csv
data/supplied_binary_predictions.csv
output/.gitkeep
```

Delivery-owned metadata and `_grader_selftest/**` are outside this learner-file
map. `assignment.ipynb` is intentionally not whole-file hashed because student
cells and stored outputs are editable. Instead,
`CANDIDATE_PROTECTED_CELL_SHA256` contains exactly these eight protected cell
IDs and the SHA-256 of normalized source:

```text
a10-header
a10-setup
a10-terms-inference
a10-task1-prompt
a10-terms-prediction
a10-terms-evaluation
a10-freeze
a10-final-verify
```

For a cell whose JSON `source` is a list, concatenate its elements; otherwise
use the source string. Normalize only line endings by replacing CRLF and bare CR
with LF, then hash the exact UTF-8 bytes. Do not trim whitespace, add/remove a
final newline, normalize Unicode, or include outputs/execution counts. Cell ID,
type, and order remain separate exact topology checks.

The implementation author computes both maps only after all intentional
candidate support-file and protected-cell wording changes for that iteration,
embeds the values independently in the central grader, and records the exact
maps in author-harness evidence. These are testable implementation constants
labelled `candidate-nonrelease`; they confer no lock, container, platform, or
release certification. The current pre-correction implementation values are
not blueprint constants because adding the required PEP 723 block and integrity
checks changes at least `check_assignment.py`.

The public checker embeds the same candidate values for the other 11 immutable
files and all eight protected cells. It reports `[FIX] integrity: ...` for any
mismatch before readiness checks. It cannot securely self-hash because an edit
could change both its code and expected self-digest; therefore only the central
grader enforces the raw-byte hash of `check_assignment.py`. The central grader
reads and hashes checker bytes directly and never imports, executes, or trusts
the checker. It validates all 12 files and eight cells before executing any
notebook code or trusting committed outputs.

Any missing, extra-keyed, or mismatched candidate integrity constant is a
template/integrity failure. Any one-byte immutable-file change or any protected
source change, including a harmless comment/whitespace edit to `a10-setup`,
scores `0/10` on the Template test; the other four automated tests are recorded
as blocked at `0`, for an exact automated total of `0/90`. Grading still writes
a valid Classroom50 result and uses the documented completed-grading exit.

After the transitive lock, immutable container digest, final learner wording,
and independent review gates pass, recompute all 20 digests from the certified
release candidate. Replace the candidate-labelled maps with official
release-frozen maps, update both central and public copies where applicable,
rerun every integrity mutant and full harness, and freeze only those recomputed
values as release evidence. Equality with a former candidate value must be
established by recomputation, not assumed.

### 8.2 Exact public function signatures

```python
fit_bounded_ols(inference_table, predictor_columns, outcome_column)
audit_feature_availability(candidate_table)
build_chronological_splits(prediction_table, validation_start, test_start)
regression_metrics(actual, predicted)
fit_prediction_candidates(train_table, feature_columns, target_column)
choose_validation_winner(metrics_table, metric_column)
compute_binary_metrics(prediction_table, actual_column, prediction_columns)
```

Canonical `prediction_columns` is insertion-ordered:

```python
{
    "supplied_model": "supplied_model_prediction",
    "dummy_baseline": "dummy_prediction",
}
```

The complete public I/O contract is:

| Function | Arguments | Exact return and ordering | Index/dtypes/copy contract | Required errors |
|---|---|---|---|---|
| `fit_bounded_ols` | `inference_table`: pandas DataFrame; `predictor_columns`: ordered list/tuple of exactly two distinct valid Python-identifier column names; `outcome_column`: a distinct valid Python-identifier column name | one fitted `statsmodels.regression.linear_model.RegressionResultsWrapper` from `smf.ols`; formula is derived exactly as `outcome_column + " ~ " + " + ".join(predictor_columns)`; `params`, `bse`, and `conf_int()` term index is exactly `Intercept`, then the two argument-ordered predictors | uses a deep copied three-column numeric `float64` model table while retaining the input row labels in model row labels; the caller's DataFrame, index, column order, and dtypes remain exactly unchanged | `TypeError` for wrong container/name-sequence/name types; `KeyError` for absent columns; `ValueError` for invalid identifiers, duplicate/overlapping names, fewer than four rows, missing/nonfinite/non-numeric values, or non-full-rank intercept-plus-predictor design |
| `audit_feature_availability` | `candidate_table`: DataFrame containing unique nonmissing `candidate_feature` and nonmissing integer `latest_required_offset_hours` | copied DataFrame with exactly `candidate_feature`, `latest_required_offset_hours`, `available_by_prediction_time`, `decision`, in that order; availability is `offset <= 0`; decision is `keep`/`exclude` | preserves input row order and exact index; first two dtypes remain `StringDtype(storage="python")`/`int64`; added dtypes are NumPy `bool` and `StringDtype(storage="python")`; no mutation | `TypeError` for non-DataFrame; `KeyError` for a missing required column; `ValueError` for missing/duplicate/blank feature names, missing offsets, or offsets not exactly representable as `int64` |
| `build_chronological_splits` | `prediction_table`: DataFrame containing `batch_id`, UTC `prediction_timestamp`, and UTC `target_timestamp` plus arbitrary carried columns; `validation_start`, `test_start`: UTC-aware timestamp-compatible scalars | two-tuple `(parts, manifest)`; `parts` is insertion-ordered `dict` with exact keys `train`, `validation`, `test` and copied DataFrames; `manifest` has exact columns `partition`, `row_count`, `first_target_timestamp`, `last_target_timestamp` and rows in that key order | stable sort by target timestamp, retaining input order for equal timestamps; each partition contains every input column/dtype in input column order and gets a fresh `RangeIndex`; manifest gets `RangeIndex`, `partition: StringDtype(storage="python")`, `row_count: int64`, and UTC ISO strings ending `Z` in `StringDtype(storage="python")`; caller unchanged | `TypeError` for non-DataFrame or unusable cutoffs; `KeyError` for missing required columns; `ValueError` for naive/non-UTC cutoffs, `validation_start >= test_start`, missing/non-UTC timestamps, missing/duplicate/blank IDs, `prediction_timestamp >= target_timestamp`, an empty partition, or failed conservation/exclusivity |
| `regression_metrics` | equal-length one-dimensional numeric array-like `actual`, `predicted`, each with at least two values | insertion-ordered plain `dict` with exact keys `mae`, `rmse`, `r2`; values are unrounded Python `float` | input indices are ignored for positional metric calculation; inputs are never mutated | `TypeError` for unusable/non-numeric array-like input; `ValueError` for non-1D, unequal length, fewer than two values, or missing/nonfinite values |
| `fit_prediction_candidates` | `train_table`: DataFrame; `feature_columns`: nonempty ordered list/tuple of distinct column names; `target_column`: distinct column name | insertion-ordered plain `dict`: `mean_baseline` is fitted `DummyRegressor(strategy="mean")`; `linear_pipeline` is fitted `Pipeline` with exact steps `scale=StandardScaler()` then `linear=LinearRegression()` | fits both exactly once on deep copied train-only features/target in argument column order; protected audit records `batch_id` values when that column exists and otherwise the exact input index labels; caller's values/index/columns/dtypes unchanged | `TypeError` for wrong containers/name types; `KeyError` for absent columns; `ValueError` for empty/duplicate/overlapping names, zero rows, or missing/nonfinite/non-numeric model values |
| `choose_validation_winner` | `metrics_table`: DataFrame containing `approach` and the argument-named `metric_column`; `metric_column`: nonempty column-name string | one Python `str`: lexicographically smallest approach among rows whose unrounded metric equals the finite minimum; nonfinite metric rows are ignored | row/index order is irrelevant to the result and remains unchanged; numeric values are converted only in a local copy; no rounding and no mutation | `TypeError` for wrong argument types; `KeyError` for missing `approach` or metric column; `ValueError` for empty/blank/non-string approach values, non-numeric metric values, or no finite metric row |
| `compute_binary_metrics` | `prediction_table`: nonempty DataFrame; `actual_column`: column-name string; `prediction_columns`: nonempty insertion-ordered mapping from unique nonblank approach names to distinct prediction-column names | DataFrame with exact columns `approach`, `accuracy`, `precision`, `recall`, one row per mapping entry in insertion order | fresh `RangeIndex`; `approach: StringDtype(storage="python")`; metric columns `float64`; source table and mapping unchanged; calculation uses `zero_division=0` | `TypeError` for wrong containers/name types; `KeyError` for any absent named column; `ValueError` for empty input/mapping, duplicate/blank names or mapped columns, missing values, or any actual/prediction value outside integer `{0,1}` |

All seven functions derive behavior from arguments, leave inputs unchanged,
contain no I/O/import/plot/random/path logic, and are tested on the fully
disclosed alternates in section 13.1. The student cells may rely on protected
imports; no public function imports a library itself.

### 8.3 Protected audit behavior

`record_predictions(partition_name, approach_name, fitted_model,
feature_frame)` is the only permitted prediction path. It accepts validation
before the freeze and test only after the gate opens; requires exact partition
IDs; calls `predict` once; records partition, approach, ordered IDs, estimator
object identity, and prediction count; permits each candidate once on
validation and only the frozen candidate once on test; and returns a copied
float array.

The protected fit recorder logs approach, estimator identity, fit count, and
either exact `batch_id` values when present or exact input index labels on
renamed alternates. The central grader independently instruments fit/predict;
the notebook ledger is evidence, not a trusted self-report. Student source has
no direct `.predict`, no `.score`, and no fit outside the candidate function.
Final cells have no fit, candidate construction, or frozen-selection
assignment.

## 9. Candidate author artifacts and values

The eight CSVs use UTF-8, LF, final newline, index false, stated row/column
order, and six decimal places for floating outputs. Their candidate bytes below
were produced under section 5.1. They are implementation targets, not
certified-release claims until section 5.2 passes.

| Artifact | Rows | Candidate bytes | Candidate SHA-256 |
|---|---:|---:|---|
| `inference_summary.csv` | 3 | 214 | `36965b53df5133e3e05f86502d230ec9241b58e9ffd93163eba588385c9f3f48` |
| `inference_case_intervals.csv` | 1 | 186 | `345e0d3aefc422606fa9a9ee1b35a06bd7a9f9007873fc7b05162cb9ef3e0951` |
| `availability_decisions.csv` | 5 | 251 | `36042dc19dd45f75603f2fb2d5783b0a7750dad274a54bd39e8d21d5f5c2ac81` |
| `split_manifest.csv` | 3 | 221 | `2b0f3f57e323fa7bfe7a0703c671755ed7b009854236e62dd0c3459b1aa67b21` |
| `validation_metrics.csv` | 2 | 106 | `65b105be797b109c2031ccde552972320c1d08cb59174cde628a23c1879832dc` |
| `final_test_metrics.csv` | 1 | 64 | `ca1bd6d4320ed84cd2ca5befe97c3c0f238746452b648e64103522517b9a77ce` |
| `final_predictions.csv` | 11 | 575 | `60b7457821655c387b07694e18cad262a873c50bc69093a9638bd8ea99239a1d` |
| `binary_metrics.csv` | 2 | 119 | `25d7b50cdb8160f8e275812010a9a90b295d700b03591b3ce7bfd712483616fa` |

`inference_residuals.png` has no public or committed byte count/hash contract.
The public checker requires a regular nonsymlink file, PNG signature, one valid
IHDR declaring 720x480 pixels, and length greater than 8192 bytes. The central
grader additionally fully decodes and verifies the committed file with Pillow,
requires decoded 720x480 RGB/RGBA content with at least 16 distinct colors and
at least one nonwhite pixel, and checks the fresh live Figure semantics below.
Raw equality between committed and fresh output applies only to the eight
deterministic CSVs. The PNG is compared byte-for-byte only between two clean
executions in the same certified central container; it is never compared
byte-for-byte with a learner-committed PNG across platforms.

### 9.1 Inference outputs

```csv
term,estimate,standard_error,confidence_low_95,confidence_high_95
Intercept,51.959310,1.715679,48.302426,55.616194
mix_minutes,0.651471,0.021679,0.605262,0.697679
initial_temp_c,0.720189,0.070929,0.569008,0.871370
```

The derived formula is exactly
`finish_quality_score ~ mix_minutes + initial_temp_c`; the fitted term order is
`["Intercept", "mix_minutes", "initial_temp_c"]`. Additional candidate
evidence: `nobs=18`, residual df `15`, R-squared `0.984441340663`, residual mean
zero to 12 decimals.

```csv
mix_minutes,initial_temp_c,predicted_mean,mean_ci_low_95,mean_ci_high_95,prediction_ci_low_95,prediction_ci_high_95
26.000000,22.000000,84.741704,84.376661,85.106747,83.154332,86.329076
```

The prediction interval strictly contains the mean-response interval. The live
Figure has one Axes, 18 scatter points, one horizontal zero line, title
`Residuals versus fitted values`, x label `Fitted finish quality score`, y label
`Residual`, size 7.2x4.8 inches, 100 DPI, no extra Axes/legend.

### 9.2 Availability and split outputs

```csv
candidate_feature,latest_required_offset_hours,available_by_prediction_time,decision
batch_sequence,0,True,keep
ambient_temp_c,0,True,keep
pre_mix_moisture_pct,0,True,keep
early_24h_strength_mpa,24,False,exclude
next_day_strength_mpa,24,False,exclude
```

```csv
partition,row_count,first_target_timestamp,last_target_timestamp
train,29,2026-04-02T00:00:00Z,2026-04-30T00:00:00Z
validation,8,2026-05-01T00:00:00Z,2026-05-08T00:00:00Z
test,11,2026-05-09T00:00:00Z,2026-05-19T00:00:00Z
```

### 9.3 Validation, final test, and binary outputs

```csv
approach,mae,rmse,r2
mean_baseline,4.259573,4.504803,-8.441848
linear_pipeline,0.255929,0.312760,0.954488
```

```csv
approach,mae,rmse,r2
linear_pipeline,0.265686,0.332477,0.830552
```

```csv
batch_id,target_timestamp,actual_strength_mpa,predicted_strength_mpa
B038,2026-05-09T00:00:00Z,35.985000,36.619379
B039,2026-05-10T00:00:00Z,36.810000,36.938898
B040,2026-05-11T00:00:00Z,37.331000,37.153664
B041,2026-05-12T00:00:00Z,36.924000,37.249329
B042,2026-05-13T00:00:00Z,37.497000,37.213505
B043,2026-05-14T00:00:00Z,36.581000,37.045450
B044,2026-05-15T00:00:00Z,36.709000,36.749396
B045,2026-05-16T00:00:00Z,36.403000,36.348188
B046,2026-05-17T00:00:00Z,35.299000,35.868423
B047,2026-05-18T00:00:00Z,35.278000,35.350631
B048,2026-05-19T00:00:00Z,35.014000,34.842601
```

```csv
approach,accuracy,precision,recall
supplied_model,0.833333,0.666667,0.666667
dummy_baseline,0.750000,0.000000,0.000000
```

The solved reference and an accepted learner output directory contain exactly
`.gitkeep` plus these nine artifacts. The distributed starter contains only
`.gitkeep`. Setup deletes only owned generated artifact names and preserves a
foreign sentinel; solved/accepted final inventory rejects the extra sentinel
until removed.

## 10. Starter, solution, and output-state rules

The committed starter has complete protected cells, TODO markers only in
student cells, function skeletons raising `NotImplementedError`, only
`output/.gitkeep`, and no stored output suggesting correctness.

The solved reference is materialized only in the author self-test's disposable
directory. The harness replaces exact student cells, clears stored outputs and
counts, executes from scratch, and compares the eight generated CSV byte
strings plus live/decoded image semantics. It does not require committed/fresh
PNG byte equality.

An accepted submission retains visible notebook outputs for both inference
tables and Figure; availability/split tables; validation metrics and selected
approach before freeze; final metrics/predictions after freeze; binary metrics;
and the readiness summary. Central grading clears stored output before
execution. Stored visibility is communication/hygiene evidence, never result
correctness.

## 11. Scoring contract

Automated central grading reports 90 points. Human review reports 10 points
through the official Classroom50 `review` URL. Do not add timing, completion
estimates, pass/fail thresholds, or a second unofficial total.

### 11.1 Automated: 90 points

| Official test | Points | Machine evidence |
|---|---:|---|
| Template, environment, fixture, notebook, protected integrity | 10 | exact PEP 723/direct versions; inventory; candidate-nonrelease or official-release integrity profile as applicable; raw-byte hashes for all 12 immutable learner files; normalized-source hashes for all eight protected cells; fixture semantics; topology; safe root/output setup; completed scaffolds |
| Task 1 bounded inference and intervals | 20 | argument-derived formula-interface OLS; implicit `Intercept`/order/no mutation; canonical/alternate behavior; exact tables; live/decoded residual Figure; no matrix API/p-value/AIC/causal mechanism |
| Task 2 contract, availability, leakage, chronological split | 25 | exact machine contract; alternate availability; stable half-open split; 29/8/11 canonical rows; conservation/exclusivity/chronology; induced boundary failures; artifacts |
| Task 3 train-only comparison, freeze, final test, binary metrics | 30 | estimator types/steps; fit audit; alternate metrics/candidates; public finite-minimum/tie chooser; protected freeze; one selected test call/no refit; exact final rows; binary zero-division; scope checks |
| Portability, visible output, repeatability, resubmission | 5 | fresh cleared execution; eight-CSV committed/fresh equality; same-container two-run PNG determinism; path layouts; top-level metadata boundary; root resolution; committed/fresh PNG decode and semantics; output hygiene; corrected resubmission; result/context behavior |

Each is independently recorded in the Classroom50 result. Concise diagnostic
subchecks are retained without exposing hidden solution source.

### 11.2 Human: 10 points

| Review dimension | Points | Evidence |
|---|---:|---|
| Task 1 interpretation | 3 | accurate conditional association; CI and mean-versus-individual interval meaning; diagnostic limits; no causal overclaim |
| Task 2 reasoning | 3 | coherent contract; leakage uses availability at prediction time; chronology and validation/test rationale match use |
| Task 3 evaluation judgment | 4 | validation versus test roles; negative R-squared relative to baseline; single-test limitation; useful accuracy/recall comparison |

Grammar, verbosity, cosmetic styling, and exact phrase matching are not point
categories. Automated checks require nonempty evidence/free of starter markers
but do not judge prose quality.

## 12. Dependency-free public readiness checker

`check_assignment.py` has the exact PEP 723 provisioning metadata from section
5.1 but its implementation uses only the standard library. It does not import
or execute notebook code, trust stored output, award points, reproduce the
central grader, or assess qualitative Markdown.

It checks:

1. Exact PEP 723 `requires-python` and five dependency entries; exact agreement
   with the five-line `requirements.txt`; CPython patch and installed
   distribution versions via `importlib.metadata`.
2. Exact learner inventory; only genuine top-level `.git/**` ignored; only the
   two exact top-level regular Classroom50 delivery files optional; no nested
   metadata bypass, alternate workflow, symlink, instructor bundle, or legacy
   file.
3. Candidate integrity hashes for the other 11 immutable learner files and all
   eight normalized protected-cell sources. The checker may report its own PEP
   block shape but cannot establish its own integrity.
4. Exact fixture/manifest bytes, hashes, LF/final newline, safe paths, schemas,
   row counts, and parsed semantics.
5. Valid UTF-8 nbformat 4.5, exact 30 unique IDs/order/types and portable
   kernelspec.
6. AST-parsed student source: exact seven signatures; no TODO,
   `NotImplementedError`, `pass`, imports, public-function I/O/canonical
   hardcoding, direct `.predict`, unexpected `.fit`, forbidden models/APIs,
   network/Colab/shell/random/absolute paths, or test use in validation cells.
7. Required argument-derived `smf.ols` formula and implicit intercept,
   baseline/Pipeline, UTC/stable chronology, metrics, validation chooser,
   `zero_division=0`, recorder calls, and freeze/final source separation.
8. For solved/accepted readiness, exact nine-plus-`.gitkeep` artifacts; eight CSV
   headers/counts/bytes/hashes/line endings; PNG signature, IHDR dimensions,
   and nontrivial length without a committed PNG hash.
9. Required stored output is nonempty and has no traceback/error; this validates
   visibility, not values.

Diagnostics use `[FIX] <surface>: <actionable message>`. A ready result says
that points and prose quality come from central/human review. The independent
central grader hashes the checker and every protected support surface directly;
it never imports the checker.

## 13. Independent central grader

`_grader_selftest/classroom50_grader.py` is instructor-controlled with PEP 723
metadata for the candidate direct grader pins. Release additionally requires
the exact transitive lock/container gate. Before learner execution, the grader
validates the applicable integrity profile, PEP metadata,
runtime/inventory/protected surfaces/source, and the raw checker bytes without
importing or executing the checker. It copies only the learner surface to a disposable directory,
clears output/counts, verifies fixtures before owned-output cleanup,
fresh-executes with nbclient, and appends one in-memory grader cell to call all
seven functions on disclosed alternates. It independently instruments fit and
predict, checks canonical live objects/text values, fully decodes committed and
fresh PNGs, compares only the eight committed/fresh CSVs, compares PNG bytes
only between two fresh runs in the same certified central container, repeats
execution, and writes only the official Classroom50 result.

### 13.1 Disclosed alternate checks

Every alternate is literal and public. Floating expected values below are
candidate author evidence under section 5.1; the grader uses `rtol=1e-12` and
`atol=1e-12` for ordinary numeric values rather than serialized alternate
bytes.

1. **Formula OLS.** Construct this table with the displayed non-range index:

   ```csv
   _index,dose,temperature,quality
   41,0.0,2.0,9.0
   7,1.0,0.0,12.0
   90,2.0,1.0,13.5
   3,3.0,3.0,14.5
   55,4.0,2.0,17.0
   12,5.0,4.0,18.0
   ```

   Call `fit_bounded_ols(table, ["dose", "temperature"], "quality")`.
   The derived formula is `quality ~ dose + temperature`; term order is
   `Intercept`, `dose`, `temperature`; parameters are
   `[10.000000000000002, 1.999999999999999, -0.4999999999999988]`;
   standard errors are
   `[1.97614301576421e-15, 8.19372310471143e-16,
   1.083927682339524e-15]`; 95% confidence rows are
   `[[9.999999999999995,10.000000000000009],
   [1.9999999999999962,2.0000000000000013],
   [-0.5000000000000022,-0.49999999999999534]]`; fitted values are
   `[9.0, 12.0, 13.5, 14.5, 17.0, 18.0]` within tolerance; every residual is
   within `4e-15` of zero; `nobs=6`, residual df is `3`, and R-squared is `1.0`.
   The input, including its index, remains exact. A source/formula inspection
   and a second rename to `x_one`, `x_two`, `response_y` reject a hard-coded
   canonical or alternate formula.

2. **Availability.** Construct the first three columns below, then require the
   exact four-column returned table, original index/order, and dtypes from
   section 8.2:

   ```csv
   _index,candidate_feature,latest_required_offset_hours,available_by_prediction_time,decision
   8,late_8,8,False,exclude
   3,at_issue,0,True,keep
   21,past_2,-2,True,keep
   1,late_1,1,False,exclude
   13,past_24,-24,True,keep
   ```

3. **Chronological split.** Construct the following literal table in this
   shuffled input order and index. Parse both timestamp columns with
   `utc=True`; use validation cutoff `2027-02-13T00:00:00Z` and test cutoff
   `2027-02-16T00:00:00Z`.

   ```csv
   _index,batch_id,prediction_timestamp,target_timestamp
   44,X07,2027-02-06T18:00:00Z,2027-02-07T00:00:00Z
   7,X02,2027-02-01T18:00:00Z,2027-02-02T00:00:00Z
   105,X19,2027-02-18T18:00:00Z,2027-02-19T00:00:00Z
   3,X12,2027-02-11T18:00:00Z,2027-02-12T00:00:00Z
   91,X01,2027-01-31T18:00:00Z,2027-02-01T00:00:00Z
   18,X15,2027-02-14T18:00:00Z,2027-02-15T00:00:00Z
   62,X09,2027-02-08T18:00:00Z,2027-02-09T00:00:00Z
   5,X04,2027-02-03T18:00:00Z,2027-02-04T00:00:00Z
   77,X14,2027-02-13T18:00:00Z,2027-02-14T00:00:00Z
   22,X06,2027-02-05T18:00:00Z,2027-02-06T00:00:00Z
   130,X18,2027-02-17T18:00:00Z,2027-02-18T00:00:00Z
   11,X03,2027-02-02T18:00:00Z,2027-02-03T00:00:00Z
   58,X10,2027-02-09T18:00:00Z,2027-02-10T00:00:00Z
   9,X16,2027-02-15T18:00:00Z,2027-02-16T00:00:00Z
   73,X05,2027-02-04T18:00:00Z,2027-02-05T00:00:00Z
   31,X13,2027-02-12T18:00:00Z,2027-02-13T00:00:00Z
   99,X08,2027-02-07T18:00:00Z,2027-02-08T00:00:00Z
   14,X17,2027-02-16T18:00:00Z,2027-02-17T00:00:00Z
   66,X11,2027-02-10T18:00:00Z,2027-02-11T00:00:00Z
   ```

   Exact sorted memberships are train `X01`–`X12`, validation
   `X13`–`X15`, and test `X16`–`X19`; every returned partition has a fresh
   zero-based `RangeIndex`. The exact manifest value is:

   ```csv
   partition,row_count,first_target_timestamp,last_target_timestamp
   train,12,2027-02-01T00:00:00Z,2027-02-12T00:00:00Z
   validation,3,2027-02-13T00:00:00Z,2027-02-15T00:00:00Z
   test,4,2027-02-16T00:00:00Z,2027-02-19T00:00:00Z
   ```

   Separate one-change cases require `ValueError` for equal/reversed or naive
   cutoffs, one missing timestamp, duplicate `X07`, and prediction time equal
   to target time. Removing `batch_id` requires `KeyError`. Input snapshots,
   exact membership, conservation, exclusivity, and copies are all checked.

4. **Regression metrics.** For actual `[1.0, 2.0, 4.0]` and prediction
   `[1.0, 3.0, 2.0]`, expect the exact ordered keys and Python-float candidate
   values `mae=1.0`, `rmse=1.2909944487358056`, and
   `r2=-0.07142857142857162` (`-1/14` mathematically). Empty, unequal-length,
   two-dimensional, and nonfinite one-change inputs raise `ValueError`.

5. **Prediction candidates.** Construct this literal train table with the
   displayed non-range index:

   ```csv
   _index,u,v,z
   81,2.0,10.0,5.2
   5,-1.0,14.0,-1.1
   44,4.0,8.0,8.7
   12,0.0,12.0,0.4
   99,3.0,9.0,7.1
   3,1.0,11.0,2.6
   70,5.0,7.0,10.5
   ```

   Call with `feature_columns=["u", "v"]`, `target_column="z"`. The exact
   keys/types/step names follow section 8.2; exactly two fits see only index
   `[81,5,44,12,99,3,70]`. Candidate fitted state is baseline
   `constant_=[[4.771428571428571]]`; scaler
   `mean_=[2.0,10.142857142857142]`,
   `var_=[4.0,4.979591836734693]`, and
   `scale_=[2.0,2.231499907401901]`; scaled-space linear
   `coef_=[4.4799999999999995,0.4909299796284197]` and
   `intercept_=4.771428571428571`. Training predictions are respectively seven
   copies of `4.771428571428571` and
   `[4.74,-1.099999999999996,8.779999999999998,
   0.700000000000002,6.759999999999999,2.7200000000000006,
   10.799999999999997]`.

   The held-out table is literal and is never passed to either `fit`:

   ```csv
   _index,u,v,z
   404,-999.0,999.0,-1000.0
   808,999.0,-999.0,1000.0
   ```

   Post-fit predictions on its `u,v` columns are
   `[4.771428571428571,4.771428571428571]` and
   `[-2019.9199999999992,2016.039999999999]`. These extremes plus independent
   fit instrumentation prove train-only state. Both source tables remain exact.

6. **Validation winner.** The ordinary case is the literal table
   `approach=["zeta","alpha","beta","ignored_nan"]`,
   `validation_loss=[0.40,0.25,0.30,np.nan]`, index `[12,4,99,7]`; calling with
   `"validation_loss"` returns `"alpha"`. The tie case is
   `approach=["zeta","alpha","beta"]`,
   `validation_loss=[0.25,0.25,0.40]`, index `[8,2,30]`; it also returns
   `"alpha"`. Exact error cases are: `{"validation_loss":[0.1]}` lacks
   `approach` (`KeyError`); `{"approach":["a"]}` lacks the named metric
   (`KeyError`); `approach=["a","b","c"]` with
   `validation_loss=[np.nan,np.inf,-np.inf]` has no finite row (`ValueError`);
   and `approach=["a","b"]` with `validation_loss=["bad",0.2]` is nonnumeric
   (`ValueError`). Every input snapshot remains exact. Canonically, the
   function receives unrounded MAEs `4.259573275862069` and
   `0.2559293869691168` and returns `linear_pipeline`; serialized rounding is
   not selection input.

7. **Binary metrics.** Use actual `[1,1,0,0]`, model `[1,0,1,0]`, dummy
   `[0,0,0,0]`, and mapping
   `{"model_alt":"model_prediction","dummy_alt":"dummy_prediction"}`.
   The exact returned rows are `model_alt,0.5,0.5,0.5` then
   `dummy_alt,0.5,0.0,0.0`, with the container/order/dtypes from section 8.2.
   One nonbinary actual value and one nonbinary prediction value each require
   `ValueError`; the input and mapping remain exact.

The appended cell writes one grader-owned JSON evidence file, removed before
inventory checking and never accepted as a student artifact.

### 13.2 Freeze and test evidence

Fresh execution must establish:

- two estimator objects fitted exactly once each on exactly 29 train IDs;
- scaler means and baseline constant equal train-only calculations;
- exactly two validation predictions, one per candidate, before freeze;
- `choose_validation_winner` receives unrounded validation MAE and selects
  `linear_pipeline`;
- no test prediction or final artifact exists at the freeze boundary;
- the same selected estimator predicts exactly 11 test rows once after freeze;
- neither estimator is refitted and the other never sees test;
- final outputs derive from that one prediction array;
- a second execution starts clean, reproduces the eight CSV bytes, and
  reproduces first-fresh-run PNG bytes inside the same certified container.

Public fixture visibility does not make test data secret. This proves an honest
executable workflow, not cryptographic secrecy from someone opening the CSV.

### 13.3 Official result and bootstrap

Result topology:

```json
{
  "schema": "classroom50/result/v1",
  "classroom": "<CLASSROOM>",
  "assignment": "<ASSIGNMENT>",
  "submission": "<SUBMISSION_TAG>",
  "commit": "<COMMIT_URL>",
  "release": "<RELEASE_URL>",
  "review": "<REVIEW_URL or COMMIT_URL>",
  "datetime": "<UTC ISO-8601>",
  "score": 0,
  "max-score": 90,
  "tests": []
}
```

The five automated tests populate `tests`; `score` is their sum. Missing
required context or inability to write `result.json` is infrastructure failure
and returns nonzero without inventing a grade. A student grading failure still
writes a valid result and uses the platform's documented grading-success exit.
Optional runner fields are accepted if supplied but not synthesized.

Production invokes a plain-Python, stdlib-only `autograder.py`. It verifies pip,
requires the frozen sibling `constraints.txt`, installs sibling grader
requirements under those constraints, verifies the complete installed lock,
imports the central grader, and writes the result; it does not assume runner
PEP 723 support. The candidate author harness may use `uv run` and PEP 723
direct pins. The release harness must use the certified container and exact
transitive lock from section 5.2.

## 14. Adversarial release matrix

The harness materializes a correct solution temporarily, proves 90/90 and
public readiness, then creates one isolated mutant per row. Every mutant is
rejected for the named evidence; a corrected resubmission returns to full
automated credit.

| Mutant or condition | Required evidence |
|---|---|
| untouched starter | checker reports scaffolds; central below full |
| malformed notebook; missing/duplicate/reordered/retagged cell | topology rejects |
| independently alter one byte in each of the 12 immutable learner files | central candidate integrity rejects each before execution at 0/90; public rejects the other 11, while central alone proves checker integrity |
| independently make a harmless source edit in each of the eight protected cells | normalized-source candidate integrity rejects each before execution at 0/90; includes `a10-setup` comment/whitespace edit |
| omit/add a candidate integrity-map key or substitute one digest | exact key-set/digest contract rejects before execution; no partial profile accepted |
| change PEP 723 Python requirement, remove/add/reorder a dependency, change a pin, or diverge from `requirements.txt` | independent static/PEP checks reject; direct clean `uv run` provisioning gate fails as applicable |
| missing/unresolved/edited transitive lock or wrong central container digest | release gate fails before grading; no certified claim |
| fixture missing/corrupt/CRLF/extra/unsafe path | reject before cleanup; sentinel remains |
| fake stored success with broken source | clear output; source/alternate/fresh execution rejects |
| canonical IDs/counts/labels/features/cutoffs hard-coded in functions | alternate/source rejects |
| matrix API/`add_constant`; hard-coded formula/name; omit implicit intercept; reverse predictors; wrong response; p-value/AIC/interaction | Task 1 source/mechanism/output rejects |
| mutate Task 1 input or function I/O | snapshot/source rejects |
| swap intervals/wrong alpha/blank or malformed residual plot | numeric/live Figure/Pillow rejects |
| causal wording inserted | required noncausal evidence flag and human surface identify it; no fake semantic score |
| availability `<0`, reversed decisions, reordered, canonical hardcode | canonical/alternate rejects |
| random split, wrong timestamp/boundary, overlap/omission/wrong sort/cutoff | Task 2 canonical/alternate rejects |
| accept reversed cutoffs/missing time/duplicate ID/prediction at or after target | induced failure test rejects |
| include future feature or target in predictors | feature contract/fit audit rejects |
| scaler outside Pipeline or fit before split/full data | AST/scaler/train-ID audit rejects |
| wrong baseline/Pipeline step/order/extra model | type/key/fit-count rejects |
| validation uses train/test; chooser uses RMSE/test/rounded values; chooser wrong on finite minimum/tie/nonfinite/error case | ledger/source/public alternate rejects |
| direct predict; test before freeze; both on test; test twice | source/gate/instrumentation rejects |
| refit after validation | identity/fit-count rejects |
| overwrite frozen selection or fit/construct in final cell | protected/source/audit rejects |
| MAE/MSE/RMSE confusion, `.score`, rounded selection | alternate/canonical rejects |
| classifier fit, wrong labels, omit zero division, precision/recall swap | source/binary checks reject |
| advanced model/regularization/CV/search/forecast/deployment | scope scan rejects |
| URL/request/upload/mount/`/content`/absolute/shell/magic/random | portability scan rejects |
| missing/stale/extra/ignored/CRLF/wrong precision CSV artifact | hash/inventory/fresh CSV byte rejects |
| committed PNG malformed/truncated/wrong IHDR/blank/trivial/wrong dimensions | public structure or central Pillow/content rejects; no raw committed hash used |
| cross-platform committed PNG bytes differ but image/live semantics pass | accepted; committed/fresh PNG equality is not a contract |
| setup deletes foreign sentinel | destructive-cleanup test rejects |
| foreign sentinel remains at final | exact output inventory rejects; removal/rerun succeeds |
| hidden/errored required notebook output | visibility/hygiene rejects independently |
| assignment root has arbitrary name/depth, including course/nested/relocated/space path | root resolves; all reproduce artifacts/PNG |
| genuine top-level `.git/` plus both exact regular delivery files | accepted without hashing delivery-controlled content |
| same-named `.git/` or `.github/` directory nested below an ordinary folder | rejected as unexpected learner content; no recursive metadata ignore |
| nested `.classroom50.yaml` delivery-marker name | rejected as unexpected learner content |
| alternate workflow such as top-level `.github/workflows/grade.yaml` | rejected; only exact `autograde.yaml` path is optional |
| symlinked `.classroom50.yaml` or `.github/workflows/autograde.yaml` | rejected before target traversal |
| injected `_grader_selftest/` inside learner submission | rejected; instructor bundle never enters copied learner surface |
| lone notebook without data | clear missing-package error; no remote fallback |
| second run accumulates ledger/Figure/state, changes a CSV byte, or changes fresh PNG bytes in the same central container | repeatability rejects |
| missing Classroom50 context | infrastructure failure; no fabricated result |
| unwritable result destination | nonzero infrastructure failure |
| grading CLI captures student failure | valid official 0–90 result still written |
| corrected resubmission | exact artifacts, readiness, 90/90 restored |

Harness summary reports exact rejected-mutant count, alternate checks, layouts,
artifacts, fit/predict calls, result-schema behavior, and corrected resubmission.
It reports no elapsed time.

The integrity family is not one aggregate smoke test. The harness materializes
20 isolated learner mutations: one for every named immutable file and one for
every named protected cell. It explicitly includes the four observed bypasses:
`README.md`, `check_assignment.py`, `requirements.txt`, and a harmless
`a10-setup` edit. Each must produce Template `0/10`, four blocked `0` tests, and
total `0/90`; restoring the exact candidate returns `90/90`. Separate
instructor-logic cases prove missing/extra/wrong integrity-map entries reject.
PEP/static cases cover wrong `requires-python`, missing dependency, extra
dependency, reordered dependency, wrong version pin, and disagreement with
`requirements.txt`. Finally, an empty-cache direct
`uv run check_assignment.py` under the exact PEP block must provision the five
pins and reach ordinary checker diagnostics/readiness without an environment
diagnostic; a preprovisioned exact environment must also support plain Python.

## 15. Validation gates

1. **Content:** independent reviewer confirms core coverage, exclusions,
   definitions-before-use, and no dependency on later lectures.
2. **Fixture:** regenerate once out of tree under candidate direct pins; compare
   every byte/hash; commit the five static fixtures only.
3. **Static integrity and starter:** exact PEP 723/requirements agreement,
   12-file candidate map, eight-cell candidate map, normalization rule, and
   central/public key-set split match section 8.1; untouched starter fails
   actionably and exposes no solution.
4. **Solution:** temporary solution restart/clear/run-all succeeds; nine
   artifacts exist; the eight committed/fresh CSVs are byte-identical;
   committed and fresh PNGs pass structure/full-decode/content checks; the
   fresh live Figure passes semantics; no Figure remains open. A second clean
   execution in the same certified container reproduces the eight CSVs and the
   first fresh PNG bytes. Committed/fresh PNG byte equality is not required.
5. **Checker:** empty-cache direct `uv run` provisions the exact five pins;
   preprovisioned plain Python also runs; the stdlib-only checker accepts correct
   work and rejects starter plus public-detectable integrity/behavior mutants
   without executing notebook source.
6. **Central:** independently hash all 12 immutable files and eight normalized
   protected cells before execution, including checker raw bytes without import;
   fresh/alternate checks pass and automated total is 90.
7. **Freeze:** instrumentation proves two train fits, two validation predictions,
   freeze, one selected test prediction, no refit.
8. **Portability:** every arbitrary-root local layout reproduces output;
   top-level metadata boundaries and symlink rejection pass; repository-backed
   Colab smoke test runs only if assignment Colab delivery is enabled.
9. **Release lock:** the exact course-wide transitive `constraints.txt` and
   immutable central-container digest exist; bootstrap uses the constraints;
   installed distributions verify against the lock; two clean locked runs
   reproduce the eight candidate CSV byte strings and numeric/live semantics.
   A mismatch reopens the design and blocks official release-hash re-freeze;
   candidate integrity constants remain non-release evidence only.
10. **Classroom50:** metadata inventory, plain-Python bootstrap, context/result,
    student failure, and infrastructure failure paths match pilot contract.
11. **Adversarial:** all 20 isolated learner-integrity mutants, integrity-map
    cases, PEP/static cases, and every remaining matrix mutant reject as
    intended; corrected resubmission returns to exact accepted result.
12. **Human review:** official review URL exposes qualitative Markdown and
    readable outputs; human points do not duplicate numeric checks.
13. **Scope:** `rg` finds no runtime fetch, classifier fit, advanced model,
    regularization, CV, bonus, pass threshold, or timing estimate.

## 16. Implementation order

1. Independently verify this design against final Lecture 10 material and the
   Lecture 8–11 sequence.
2. Materialize/freeze section 7 fixtures in a disposable directory under the
   candidate direct environment.
3. Build the explicitly non-release author-candidate topology, omitting only
   the unresolved instructor `constraints.txt`, and the exact distributed
   learner starter with only `output/.gitkeep`; finalize the eight protected
   candidate cell sources without claiming official release hashes.
4. Implement the exact PEP 723 plus stdlib-only checker, compute/embed its
   candidate values for the other 11 immutable files and eight protected cells,
   and run public static/readiness checks.
5. After checker bytes are final for this candidate iteration, compute/embed the
   independent central 12-file/eight-cell candidate maps and implement the
   result contract without importing the checker.
6. Implement the author-only solution materializer/adversarial harness; run all
   20 isolated learner-integrity mutants, map/PEP cases, full behavior matrix,
   and corrected resubmission.
7. Complete course-wide transitive locking/container certification, add the
   real instructor `constraints.txt`, remove the non-release designation, and
   recompute all 20 official release digests after final wording. Recompute
   section 9 in that environment; if any value changes, update this blueprint
   and repeat independent design verification.
8. Replace candidate labels/maps with the recomputed official release-frozen
   values in public/central surfaces as applicable; rerun all gates and avoid
   wording drift after the release freeze.
9. Pilot through Classroom50 with accepted metadata and GUI Git path.

## 17. Unresolved items

There is no assignment-design blocker. Certified release remains blocked until
the mandatory course-wide exact transitive lock and immutable container digest
are produced and gate 9 passes. Their contents are intentionally not invented
here. Independently accepted candidate implementation may proceed under the
single-file non-release exception in section 5.2, but it must enforce the
candidate-nonrelease integrity constants from section 8.1. Those constants
close implementation bypasses without certifying release.

Two delivery choices remain conditional for the release owner:

- whether Assignment 10 receives an official repository-backed Colab launch;
  local Jupyter remains mandatory either way;
- final delivery-owned contents of `.classroom50.yaml` and
  `.github/workflows/autograde.yaml`; paths/inventory and grader interface are
  fixed here, while Classroom50 owns release-specific values.

Neither conditional delivery choice changes the task, fixture, notebook,
artifact, rubric, checker, or central-grader contract. The mandatory release
lock may change candidate numeric bytes; if so, the update/reverification rule
above applies.
