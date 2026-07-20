# Lecture 10 demo implementation blueprint

Status: author-complete design awaiting independent verification. The Lecture
10 core and bonus narrative passed independent review on 2026-07-19, and the
Lecture 09 demo handoff passed independent implementation review. This document
authorizes only an independent design review; it does not authorize edits under
`10/demo/**` or `10/assignment/**` until that review passes.

## Accepted demo role

The required demonstrations form one cumulative, bounded progression:

1. frame one inferential question; state its unit, population estimand,
   assumptions, and noncausal claim; fit one OLS association model; interpret
   one coefficient and confidence interval conditionally; inspect one residual
   plot; and distinguish a mean-response confidence interval from an individual
   prediction interval;
2. define a temporal prediction contract before fitting; contrast a seeded
   random split for exchangeable rows with fixed chronological cutoffs for
   future prediction; write a stable split manifest; and explicitly reject a
   post-outcome feature and full-data preprocessing as leakage; and
3. compare a training-mean baseline with one train-only
   `Pipeline(StandardScaler, LinearRegression)` on validation MAE, while also
   reporting validation RMSE and R2; evaluate the selected approach on test
   once; inspect the familiar residual plot; and calculate accuracy, precision,
   and recall from supplied binary predictions without fitting a classifier.

The notebooks use two tightly bounded course-authored fixtures: one 12-row
cross-sectional workshop sample for inference and one 40-row single-station
daily table for prediction. Demo 2 and Demo 3 use the same prediction fixture,
but every notebook derives its state independently and never reads another
notebook's generated output.

Required code does not fetch data, use credentials, mount Drive, require an
upload, generate random observations, use a mutable date, fit a classifier,
fit more than the baseline and one linear regression Pipeline, or introduce
regularization, cross-validation, a tree model, feature importance, boosting,
deep learning, deployment, production monitoring, or time-series forecasting.
The optional topics retained in `10/BONUS.md` remain outside the required demo
package.

## Legacy evidence and disposition

The current three legacy notebooks have 24, 24, and 23 cells. Their code cells
match their paired same-stem Markdown fences exactly. They contain 12, 12, and
169 stored outputs respectively and use executed state as though it were
evidence. The package also requires a runtime California Housing fetch,
Altair, Random Forest, Ridge, Lasso, XGBoost, TensorFlow/Keras, several neural
networks, and a Jupytext conversion path. The guide includes instructor talking
points and lesson-duration estimates.

Apply these dispositions during implementation:

| Legacy material | Disposition | Reason |
|---|---|---|
| statsmodels formula OLS and coefficient extraction | keep and bound | Supports the accepted inferential question and coefficient/interval contract. |
| array API comparison, full summary survey, F tests, p-value thresholding, AIC/BIC model selection, categorical encoding | drop from required demos | Duplicates mechanics or enters optional/untaught inference and model-selection scope. |
| California Housing runtime fetch | replace | Network/cache behavior cannot be an exact Colab/local or adversarial fixture contract. |
| Altair modeling charts | replace with familiar Matplotlib residual plots | Lecture 07 already supplies the plotting vocabulary; no incidental plotting library is needed. |
| two-way train/test-only comparison | replace | The accepted core requires distinct train, validation, and one-use test roles. |
| Ridge, Lasso, Random Forest, XGBoost, early stopping, and feature-importance comparison | remove from core | These exceed the accepted baseline-plus-one-linear-Pipeline contract; the legacy path also uses test data for development. |
| TensorFlow/Keras, architecture experiments, dropout/L2, framework comparison | remove | Deep learning is excluded from required Lecture 10 and Lecture 11 prerequisites. |
| paired Markdown/Jupytext source and notebook conversion instructions | remove | One clean notebook is the executable source of each demo. |
| instructor tips, meta-instructions, duration estimates | replace | The guide must give learners direct actions, observable outcomes, and concise explanations without a timing claim. |

## Exact package

Replace the legacy required demo package atomically with:

```text
10/demo/
|-- .gitignore
|-- .python-version
|-- DEMO_GUIDE.md
|-- requirements.txt
|-- data/
|   |-- station_next_day.csv
|   `-- workshop_participants.csv
|-- demo1_bounded_ols.ipynb
|-- demo2_split_availability_leakage.ipynb
`-- demo3_honest_prediction_evaluation.ipynb
```

Delete the three legacy notebooks, their three paired same-stem Markdown files,
and the legacy guide and requirements file as part of the atomic replacement.
Do not retain California Housing download/cache code, Wine data, Jupytext
metadata, generated HTML, or a second guide.

`.gitignore` must ignore exactly the generated-output directory and common
notebook/Python/environment state:

```gitignore
.ipynb_checkpoints/
output/
__pycache__/
*.py[cod]
.venv/
venv/
env/
```

It must not hide either fixture, the notebooks, guide, requirements, or Python
record. Generated files under `10/demo/output/` are demonstrations of fresh
execution, not version-controlled inputs.

Use this candidate environment record:

```text
.python-version: 3.12.13
requirements.txt:
numpy==2.0.2
pandas==3.0.3
statsmodels==0.14.6
scikit-learn==1.9.0
matplotlib==3.11.1
```

Notebook hosting, kernel, format, and execution tools used only for validation
are test/platform dependencies rather than teaching imports. These five library
pins have executed the lecture narrative and the design calculations locally;
they remain candidates until the complete local and fresh-Colab matrix and the
course-wide version freeze pass.

## Exact deterministic fixtures

Both fixtures are synthetic, non-identifying, UTF-8 CSV with comma delimiters,
LF line endings, the displayed column order, and a final newline. The notebook
must verify the exact checksum before parsing a present fixture.

### Workshop inference fixture

`data/workshop_participants.csv` is exactly 200 bytes with SHA-256
`eefb5f1023e9b84106f407800fa0db72853b6876d58e61255c346ed2d2d32f05`:

```csv
participant_id,study_hours,prior_score,assessment_score
p01,1,58,65
p02,2,61,69
p03,2,67,72
p04,3,63,71
p05,4,70,78
p06,4,74,80
p07,5,69,79
p08,6,76,86
p09,6,82,90
p10,7,79,88
p11,8,85,94
p12,9,88,98
```

Its parsed semantic dtypes, in column order, are pandas `string`, NumPy
`int64`, `int64`, and `int64`. Its grain is one synthetic workshop
participant, and `participant_id` is the unique row key. The inferential
question is: in the hypothetical participant population represented by this
teaching process, what is the coefficient on study hours in the conditional
mean model for assessment score after accounting for prior score? The target
estimand is that population coefficient. The sample demonstrates mechanics and
does not support a real-world population or causal conclusion.

### Station prediction fixture

`data/station_next_day.csv` is exactly 3,879 bytes with SHA-256
`f95330b252c6e0f12026577602c69e21d01dbec232b5e523c6c41b0b62cf85a8`:

```csv
row_id,prediction_timestamp,target_timestamp,day_number,current_temperature_c,previous_temperature_c,target_next_day_temperature_c
station-a-20260102,2026-01-02T00:00:00Z,2026-01-03T00:00:00Z,1,10.752852,10.400000,11.150020
station-a-20260103,2026-01-03T00:00:00Z,2026-01-04T00:00:00Z,2,11.150020,10.752852,12.284133
station-a-20260104,2026-01-04T00:00:00Z,2026-01-05T00:00:00Z,3,12.284133,11.150020,12.891635
station-a-20260105,2026-01-05T00:00:00Z,2026-01-06T00:00:00Z,4,12.891635,12.284133,12.500011
station-a-20260106,2026-01-06T00:00:00Z,2026-01-07T00:00:00Z,5,12.500011,12.891635,12.432889
station-a-20260107,2026-01-07T00:00:00Z,2026-01-08T00:00:00Z,6,12.432889,12.500011,12.810600
station-a-20260108,2026-01-08T00:00:00Z,2026-01-09T00:00:00Z,7,12.810600,12.432889,12.319227
station-a-20260109,2026-01-09T00:00:00Z,2026-01-10T00:00:00Z,8,12.319227,12.810600,11.265068
station-a-20260110,2026-01-10T00:00:00Z,2026-01-11T00:00:00Z,9,11.265068,12.319227,11.008799
station-a-20260111,2026-01-11T00:00:00Z,2026-01-12T00:00:00Z,10,11.008799,11.265068,11.042981
station-a-20260112,2026-01-12T00:00:00Z,2026-01-13T00:00:00Z,11,11.042981,11.008799,10.294535
station-a-20260113,2026-01-13T00:00:00Z,2026-01-14T00:00:00Z,12,10.294535,11.042981,9.694338
station-a-20260114,2026-01-14T00:00:00Z,2026-01-15T00:00:00Z,13,9.694338,10.294535,10.196415
station-a-20260115,2026-01-15T00:00:00Z,2026-01-16T00:00:00Z,14,10.196415,9.694338,10.705477
station-a-20260116,2026-01-16T00:00:00Z,2026-01-17T00:00:00Z,15,10.705477,10.196415,10.582814
station-a-20260117,2026-01-17T00:00:00Z,2026-01-18T00:00:00Z,16,10.582814,10.705477,11.069374
station-a-20260118,2026-01-18T00:00:00Z,2026-01-19T00:00:00Z,17,11.069374,10.582814,12.415247
station-a-20260119,2026-01-19T00:00:00Z,2026-01-20T00:00:00Z,18,12.415247,11.069374,13.203857
station-a-20260120,2026-01-20T00:00:00Z,2026-01-21T00:00:00Z,19,13.203857,12.415247,13.408874
station-a-20260121,2026-01-21T00:00:00Z,2026-01-22T00:00:00Z,20,13.408874,13.203857,14.297838
station-a-20260122,2026-01-22T00:00:00Z,2026-01-23T00:00:00Z,21,14.297838,13.408874,15.417233
station-a-20260123,2026-01-23T00:00:00Z,2026-01-24T00:00:00Z,22,15.417233,14.297838,15.482652
station-a-20260124,2026-01-24T00:00:00Z,2026-01-25T00:00:00Z,23,15.482652,15.417233,15.179048
station-a-20260125,2026-01-25T00:00:00Z,2026-01-26T00:00:00Z,24,15.179048,15.482652,15.559942
station-a-20260126,2026-01-26T00:00:00Z,2026-01-27T00:00:00Z,25,15.559942,15.179048,15.665661
station-a-20260127,2026-01-27T00:00:00Z,2026-01-28T00:00:00Z,26,15.665661,15.559942,14.738241
station-a-20260128,2026-01-28T00:00:00Z,2026-01-29T00:00:00Z,27,14.738241,15.665661,14.027121
station-a-20260129,2026-01-29T00:00:00Z,2026-01-30T00:00:00Z,28,14.027121,14.738241,14.098535
station-a-20260130,2026-01-30T00:00:00Z,2026-01-31T00:00:00Z,29,14.098535,14.027121,13.708819
station-a-20260131,2026-01-31T00:00:00Z,2026-02-01T00:00:00Z,30,13.708819,14.098535,12.768661
station-a-20260201,2026-02-01T00:00:00Z,2026-02-02T00:00:00Z,31,12.768661,13.708819,12.688712
station-a-20260202,2026-02-02T00:00:00Z,2026-02-03T00:00:00Z,32,12.688712,12.768661,13.310430
station-a-20260203,2026-02-03T00:00:00Z,2026-02-04T00:00:00Z,33,13.310430,12.688712,13.338624
station-a-20260204,2026-02-04T00:00:00Z,2026-02-05T00:00:00Z,34,13.338624,13.310430,13.290932
station-a-20260205,2026-02-05T00:00:00Z,2026-02-06T00:00:00Z,35,13.290932,13.338624,14.302447
station-a-20260206,2026-02-06T00:00:00Z,2026-02-07T00:00:00Z,36,14.302447,13.290932,15.487204
station-a-20260207,2026-02-07T00:00:00Z,2026-02-08T00:00:00Z,37,15.487204,14.302447,15.821827
station-a-20260208,2026-02-08T00:00:00Z,2026-02-09T00:00:00Z,38,15.821827,15.487204,16.311473
station-a-20260209,2026-02-09T00:00:00Z,2026-02-10T00:00:00Z,39,16.311473,15.821827,17.563960
station-a-20260210,2026-02-10T00:00:00Z,2026-02-11T00:00:00Z,40,17.563960,16.311473,18.266177
```

Its parsed semantic dtypes are pandas `string`; two
`datetime64[us, UTC]` columns; NumPy `int64`; and three NumPy `float64`
columns. Parse both timestamps with the exact `%Y-%m-%dT%H:%M:%SZ` format and
`utc=True`. The grain is one Station A daily prediction issue; `row_id` is
unique; target timestamp is exactly one day after prediction timestamp; and
all rows are chronological.

The prediction contract is fixed before features:

| field | exact definition |
|---|---|
| unit | one Station A daily prediction issue |
| prediction timestamp | current day at 00:00 UTC |
| target | next-day temperature in degrees C |
| target timestamp | prediction timestamp plus one day |
| horizon | one day |
| feature cutoff | prediction timestamp |
| primary validation metric | MAE |

## Portable path and fixture behavior

Every notebook resolves one demo root before reading or writing. Starting with
the current directory and each ancestor, test both that directory and its
`10/demo/` child for `DEMO_GUIDE.md` plus `.python-version`. This supports the
course root, `10/demo/`, a directory nested inside `10/demo/`, and a flattened
standalone copy. If no marked directory exists, use the current directory as an
ephemeral standalone/Colab root.

Read the needed fixture below `<resolved-demo-root>/data/`. If and only if that
path is absent, reconstruct the exact literal bytes above and verify the
checksum before parsing. A present but corrupt fixture must stop execution; it
must not be overwritten, bypassed, or replaced from a second path. Fixture
integrity is checked before creating the output directory or removing any
owned stale output.

Write only under `<resolved-demo-root>/output/`. Each notebook removes and
replaces only its own named outputs. It must preserve outputs owned by the
other notebooks and any unrelated sentinel. No absolute path, repository-only
working-directory assumption, network fallback, upload, Drive mount, or
credential path is allowed.

## Notebook-wide contract

All three notebooks must have:

- a portable Python 3 kernelspec, null execution counts, zero stored outputs,
  and the exact stable globally unique IDs listed below;
- a first Markdown cell stating the learning question, input and output grain,
  Colab-first/local-Jupyter equivalence, ephemeral Colab filesystem, privacy
  rule, fresh-execution rule, that Colab edits are not automatically saved to
  GitHub, and that assignment use of Colab remains conditional on the
  repository-save/Classroom 50 pilot;
- one supplied setup cell that conditionally installs only mismatched candidate
  packages before their first teaching import and then prints and asserts
  Python 3.12.13, NumPy 2.0.2, pandas 3.0.3, statsmodels 0.14.6,
  scikit-learn 1.9.0, and Matplotlib 3.11.1;
- exact checksum verification and explicit semantic parsing of its fixture;
- concise Markdown defining each newly demanding term before first code use;
- executable assertions for shape, keys, dtypes, timestamp/horizon semantics,
  formulas or split roles, exact values, paths, output schema-aware readback,
  and repeat determinism;
- UTF-8 CSV output with `index=False`, `lineterminator="\n"`,
  `float_format="%.6f"`, and `%Y-%m-%dT%H:%M:%SZ` UTC timestamp formatting;
  and
- a final verification cell whose PASS is reachable only after every prior
  source cell runs freshly and all owned artifacts pass external-style
  readback.

The setup uses `importlib.metadata.version()` to compare distribution versions.
The candidate distribution names are `numpy`, `pandas`, `statsmodels`,
`scikit-learn`, and `matplotlib`; the Python import name is `sklearn`. Set the
Matplotlib backend to `Agg` before importing `matplotlib.pyplot`, then explicitly
display learner-facing figures. Stored display output is still not execution
evidence.

### Exact topology and global IDs

`demo1_bounded_ols.ipynb` has exactly 11 cells in this order:

```text
l10d1-01-question            markdown
l10d1-02-setup               code
l10d1-03-inference-contract  markdown
l10d1-04-fixture             code
l10d1-05-ols-contract        markdown
l10d1-06-fit                 code
l10d1-07-residual-check      markdown
l10d1-08-residual-plot       code
l10d1-09-interval-contract   markdown
l10d1-10-interval-output     code
l10d1-11-verify              code
```

`demo2_split_availability_leakage.ipynb` has exactly 11 cells:

```text
l10d2-01-question            markdown
l10d2-02-setup               code
l10d2-03-prediction-contract markdown
l10d2-04-fixture             code
l10d2-05-availability        markdown
l10d2-06-availability-audit  code
l10d2-07-split-contract      markdown
l10d2-08-exchangeable-split  code
l10d2-09-temporal-split      code
l10d2-10-leakage-rejection   markdown
l10d2-11-verify              code
```

`demo3_honest_prediction_evaluation.ipynb` has exactly 13 cells:

```text
l10d3-01-question            markdown
l10d3-02-setup               code
l10d3-03-evaluation-contract markdown
l10d3-04-fixture-split       code
l10d3-05-model-contract      markdown
l10d3-06-fit                 code
l10d3-07-metric-contract     markdown
l10d3-08-validation          code
l10d3-09-final-evaluation    markdown
l10d3-10-test-and-plot       code
l10d3-11-binary-contract     markdown
l10d3-12-binary-metrics      code
l10d3-13-verify              code
```

The 35 IDs must be unique across the full three-notebook package, not merely
inside each file. The only permitted code-to-code adjacencies are exactly
`l10d1-10-interval-output` to `l10d1-11-verify`,
`l10d2-08-exchangeable-split` to `l10d2-09-temporal-split`, and
`l10d3-12-binary-metrics` to `l10d3-13-verify`. The middle pair deliberately
places the exchangeable and temporal split demonstrations consecutively; the
other two pairs end with the final verification cell. No hidden helper, raw
cell, or bonus appendix is added.

## Exact generated-output contract

Each successful run contributes only to these ten ignored outputs:

```text
output/
|-- availability_decisions.csv
|-- binary_metrics.csv
|-- final_predictions.csv
|-- final_test_metrics.csv
|-- inference_summary.csv
|-- ols_residuals.png
|-- prediction_intervals.csv
|-- prediction_residuals.png
|-- split_manifest.csv
`-- validation_metrics.csv
```

The tree contains eight CSVs and two PNGs. Demo 1 owns `inference_summary.csv`,
`prediction_intervals.csv`, and `ols_residuals.png`. Demo 2 owns
`availability_decisions.csv` and `split_manifest.csv`. Demo 3 owns the
remaining five files.

The reference CSV serializations have these exact byte and SHA-256 contracts:

| file | bytes | SHA-256 |
|---|---:|---|
| `inference_summary.csv` | 95 | `feccc3b50dcd46cd3bfb0c73d246940244823d2d9711da91ca1515d7bd9a1066` |
| `prediction_intervals.csv` | 200 | `6bbe7d288b39efc870ead781363dc747223d336b1fe521c59eec724927469663` |
| `availability_decisions.csv` | 416 | `e765b4426412525b06fcfad9717af158e786e7e0ff32c570341927192d6020f2` |
| `split_manifest.csv` | 2,755 | `92a1fb047929e318d0f0259634cfa454bad58076779dbc5058992a5b70e3d9d0` |
| `validation_metrics.csv` | 116 | `899d246c84c5116e857e81a4327055819d3bef1d214e4199e6247f82fd69d25f` |
| `final_test_metrics.csv` | 79 | `8c4163c7d9f5de0a49f41646d30a68b4569e96c1e7b69d0d89f7706fbc3ffc6a` |
| `final_predictions.csv` | 843 | `eb2b5a5ffa2d62cdfb6f92b1a6e7fea0c2c5c4b1d33e433f4d2cd5c568c08090` |
| `binary_metrics.csv` | 119 | `52180318d2393b626ceeafc07a93ba751bcd7bb2f56907a6a9e746b783632ba3` |

PNG hashes are deliberately not cross-platform grading contracts. Both PNGs
must have a valid PNG signature, RGB/RGBA color mode, exact 1,200 by 720 pixel
dimensions from a 10 by 6 inch figure at 120 DPI, nonempty plotted data, and
the exact labels specified below. Independent review must inspect both images
at original detail for clipping, contrast, zero-line visibility, and whether
all points are visible. A valid header and dimensions are only a structural
proxy, not evidence of visual correctness.

The semantic in-memory and schema-aware readback dtypes are:

| output | exact dtypes in column order |
|---|---|
| `inference_summary.csv` | `string`; four `float64` |
| `prediction_intervals.csv` | `string`; seven `float64` |
| `availability_decisions.csv` | `string`; `datetime64[us, UTC]`; `bool`; `string`; `string` |
| `split_manifest.csv` | `string`; two `datetime64[us, UTC]`; `string` |
| `validation_metrics.csv` | `string`; three `float64` |
| `final_test_metrics.csv` | two `string`; three `float64` |
| `final_predictions.csv` | `string`; `datetime64[us, UTC]`; three `float64` |
| `binary_metrics.csv` | `string`; three `float64` |

Do not infer these types from a default read and accept arbitrary object/text
columns. Read strings with explicit pandas string dtype and restore timestamps
with exact-format `pd.to_datetime(..., utc=True)` before comparisons.

## Demo 1: bounded OLS association and uncertainty

Canonical filename: `demo1_bounded_ols.ipynb`.

The first Markdown cell asks the accepted inferential question and states the
input grain (one participant), output grain (one target coefficient and one
new-case interval row), population, estimand, association-only claim, and
outside-scope decisions. Define sample, population, estimand, association,
causation, OLS, conditional mean, response, explanatory variable, intercept,
coefficient, fitted value, residual, and error before the formula fit.

State these assumptions before interpreting output:

- the conditional mean is adequately represented by the stated linear form;
- observations are independent or dependence is handled by the design;
- conventional coefficient intervals assume reasonably stable residual
  variance and appropriate small-sample error shape;
- explanatory variables are not exact linear combinations;
- the sample and measurements are relevant to the intended population; and
- the intended claim is association, not causation.

Fit exactly:

```python
ols_result = smf.ols(
    "assessment_score ~ study_hours + prior_score",
    data=workshop_data,
).fit()
```

Do not display the full model summary, p-values, significance labels, AIC/BIC,
alternate formulas, interactions, categories, or the statsmodels array API.
The exact fitted invariants are:

```text
Intercept coefficient       24.596318298320796
study_hours coefficient      1.6452439696265557
prior_score coefficient      0.6663592593479781
study_hours standard error   0.23585218765558658
study_hours 95% CI lower     1.111709253959844
study_hours 95% CI upper     2.1787786852932673
mean residual               -1.0658141036401503e-14
```

Use `np.isclose` for in-memory floating assertions and the six-decimal output
contract for serialized equality. `inference_summary.csv` has exact columns
`term,estimate,standard_error,ci_lower,ci_upper`, one row for `study_hours`,
and exact serialized values:

```csv
term,estimate,standard_error,ci_lower,ci_upper
study_hours,1.645244,0.235852,1.111709,2.178779
```

Interpret it only as: holding included prior score fixed, one additional study
hour is associated with an estimated 1.645244-point difference in population
mean assessment score under the stated model and sampling assumptions. It is
not an intervention effect.

Define a residual plot as a warning diagnostic before plotting. Save
`ols_residuals.png` with 12 points, title
`Residual check: workshop OLS association`, x label
`Fitted assessment score`, y label
`Residual (observed - fitted)`, and a visible horizontal zero reference. The
text explains that curvature, changing spread, or an isolated residual can
warn about model inadequacy, while a quiet plot cannot prove assumptions.

Define standard error and a 95% confidence-interval procedure before interval
interpretation. Then define a mean-response confidence interval and an
individual prediction interval before using
`ols_result.get_prediction(new_case).summary_frame(alpha=0.05)`. The supplied
new case is `case_id="new-participant"`, `study_hours=5.0`, and
`prior_score=75.0`. `prediction_intervals.csv` has exact columns and values:

```csv
case_id,study_hours,prior_score,predicted_mean,mean_ci_lower,mean_ci_upper,individual_pi_lower,individual_pi_upper
new-participant,5.000000,75.000000,82.799483,82.349511,83.249455,81.338179,84.260786
```

The mean-response width is `0.899944` after serialization and the individual
width is `2.922607`; assert the individual interval is wider before stating
why. Neither interval licenses extrapolation or a causal claim.

## Demo 2: prediction contract, split choice, and leakage rejection

Canonical filename: `demo2_split_availability_leakage.ipynb`.

The opening states the predictive question, exact station contract, fixture
grain, and output grains: one candidate-feature decision and one stable row
split. Define prediction timestamp, target, target timestamp, prediction
horizon, feature, feature cutoff, information availability, and primary metric
before constructing the inventory.

Use the supplied issue `station-a-20260125` at 2026-01-25 00:00 UTC as the
visible availability audit. Define target leakage, temporal leakage,
preprocessing leakage, and test-set leakage before recording any rejection.
Construct, but do not compute the rejected candidates:

```text
candidate                                  latest required UTC  available  decision  leakage type
day_number                                2026-01-25 00:00      True       keep      none
current_temperature_c                     2026-01-25 00:00      True       keep      none
previous_temperature_c                    2026-01-24 00:00      True       keep      none
post_outcome_temperature_review_c         2026-01-26 00:00      False      reject    target/temporal
full_dataset_scaled_current_temperature_c 2026-02-10 00:00      False      reject    preprocessing
```

`availability_decisions.csv` has exact columns
`candidate,latest_required_timestamp,available_by_cutoff,decision,leakage_type`.
The post-outcome review requires information at the target timestamp, after the
feature cutoff. The full-data scaled candidate would learn state from future
validation/test feature rows. Do not create either series and do not call any
scaler, model, `fit`, `fit_transform`, or `predict` in Demo 2.

Define training, validation, test, exchangeable rows, random seed,
chronological split, and split manifest before split code. For the supplied
exchangeable teaching IDs `0` through `29`, use the exact two-stage
`train_test_split` calls from the accepted narrative with `random_state=217`.
The sorted roles are:

```text
train      1, 3, 6, 7, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 28
validation 0, 8, 10, 25, 27, 29
test       2, 4, 5, 15, 23, 26
```

Assert 18/6/6 counts, complete ID conservation, and pairwise disjointness.
State that a seed makes membership reproducible but does not make a split
appropriate.

The station task predicts later dates, so it is not exchangeable. Assign roles
from target timestamps with validation start 2026-01-25 00:00 UTC and test
start 2026-02-01 00:00 UTC. The exact counts are train 22, validation 7, and
test 11. Training target timestamps run from January 3 through January 24,
validation from January 25 through January 31, and test from February 1 through
February 11, all inclusive and UTC. Assert unique IDs, complete row
conservation, pairwise disjointness, and strict target-time separation.

Write `split_manifest.csv` with exact columns
`row_id,prediction_timestamp,target_timestamp,split` in source chronological
order. A final explanation contrasts the two split rules and explicitly rejects
using test results for feature, preprocessing, model, setting, or stopping
choices.

## Demo 3: baseline, one train-only Pipeline, and honest evaluation

Canonical filename: `demo3_honest_prediction_evaluation.ipynb`.

Reconstruct and validate the prediction fixture and chronological 22/7/11
roles independently. Do not read Demo 2 outputs. Define estimator, `fit`,
`predict`, preprocessing, StandardScaler, Pipeline, baseline, and the three
partition roles before fitting.

Use exactly these features and target:

```python
feature_columns = [
    "day_number",
    "current_temperature_c",
    "previous_temperature_c",
]
target_column = "target_next_day_temperature_c"
```

Fit exactly two regression approaches on training rows only:

```python
baseline = DummyRegressor(strategy="mean")
linear_pipeline = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ]
)
```

The baseline prediction is the training-target mean `12.112455318181818`.
The fitted scaler means, in feature order, are
`[11.5, 11.897464409090908, 11.669408363636364]`. Assert these equal the
training-frame means and differ from at least one full-frame mean. Do not fit a
standalone scaler, preprocess before splitting, refit on validation/test, or
fit an additional regression model.

Define MAE, RMSE, and R2 before computing them. Use the inspected scikit-learn
1.9 `root_mean_squared_error` API rather than a guessed keyword or deprecated
shortcut. MAE is the predeclared selection metric. The exact serialized
`validation_metrics.csv` is:

```csv
approach,mae,rmse,r2
training_mean_baseline,2.598597,2.698360,-12.778632
linear_pipeline,0.408166,0.494871,0.536563
```

Select `linear_pipeline` because it has lower validation MAE. Do not create a
test prediction before this choice is fixed.

Define final evaluation before the one test prediction call. Evaluate the
chosen training-fitted Pipeline on test exactly once during a fresh notebook
run. `final_test_metrics.csv` is:

```csv
approach,partition,mae,rmse,r2
linear_pipeline,test,0.450242,0.542081,0.916920
```

`final_predictions.csv` has exact columns
`row_id,target_timestamp,actual_temperature_c,predicted_temperature_c,residual_c`
and exact six-decimal rows:

```csv
row_id,target_timestamp,actual_temperature_c,predicted_temperature_c,residual_c
station-a-20260131,2026-02-01T00:00:00Z,12.768661,13.742458,-0.973797
station-a-20260201,2026-02-02T00:00:00Z,12.688712,12.652152,0.036560
station-a-20260202,2026-02-03T00:00:00Z,13.310430,13.076531,0.233899
station-a-20260203,2026-02-04T00:00:00Z,13.338624,14.019072,-0.680448
station-a-20260204,2026-02-05T00:00:00Z,13.290932,13.740149,-0.449217
station-a-20260205,2026-02-06T00:00:00Z,14.302447,13.679424,0.623023
station-a-20260206,2026-02-07T00:00:00Z,15.487204,15.153987,0.333217
station-a-20260207,2026-02-08T00:00:00Z,15.821827,16.292481,-0.470654
station-a-20260208,2026-02-09T00:00:00Z,16.311473,16.137180,0.174293
station-a-20260209,2026-02-10T00:00:00Z,17.563960,16.666313,0.897647
station-a-20260210,2026-02-11T00:00:00Z,18.266177,18.186271,0.079906
```

Reuse the familiar residual diagnostic from Demo 1 rather than introducing a
new visualization concept. Save `prediction_residuals.png` with these 11 exact
points, title `Final test residuals: selected linear Pipeline`, x label
`Predicted next-day temperature (degrees C)`, y label
`Residual (actual - predicted, degrees C)`, and a visible zero line. Display it
after saving and state that it describes only this held-out period.

Then define binary classification, positive class, true positive, false
positive, false negative, accuracy, precision, and recall before calculating
the supplied table:

```text
actual                    1 0 0 0 1 0 0 0 0 0
supplied model prediction 1 1 0 0 0 0 0 0 0 0
supplied dummy prediction 0 0 0 0 0 0 0 0 0 0
```

Use `pos_label=1` and `zero_division=0` for precision and recall. Do not import,
construct, or fit a classifier. `binary_metrics.csv` is exactly:

```csv
approach,accuracy,precision,recall
supplied_model,0.800000,0.500000,0.500000
supplied_dummy,0.800000,0.000000,0.000000
```

The explanation notes that identical accuracy does not imply identical
positive-class behavior and that the relevant tradeoff depends on decision
consequences. It must not claim that either supplied prediction column is
globally better.

Finish with bounded limitations: synthetic data, one station and date range,
fixed features, conditional model assumptions, possible distribution change,
one held-out period, no causal claim, and no universal performance guarantee.
If a test result prompts redevelopment, it becomes development evidence and a
new untouched test release is required.

## Guide and publication contract

`DEMO_GUIDE.md` must identify the exact three notebook names, their cumulative
questions, the two fixture hashes, all ten generated files, expected visible
tables/plots, supported launch layouts, and the exact candidate environment.
Every step addresses the learner directly and gives an action, observable
outcome, and concise explanation. It contains no instructor talking points,
meta-instructions, lesson-duration estimate, execution-time claim, or
assignment-Colab promise.

The guide must ask learners to predict these outcomes before running the
corresponding code:

- the sign and bounded meaning of the study-hours coefficient;
- why the individual interval is wider than the mean-response interval;
- whether a seeded random split is appropriate for later-date prediction;
- which two candidate features must be rejected and why;
- which approach wins on validation MAE before test is inspected;
- what the sign of `actual - predicted` means in both residual plots; and
- why equal binary accuracy can coexist with different recall.

Likely failures are direct learner recovery steps: wrong working directory,
candidate package mismatch, absent fixture reconstruction, corrupt fixture
checksum stop, read-only output path, stale/binary output replacement, kernel
state/order, and Colab ephemeral-file loss. Do not add a destructive cleanup
command; rerunning a notebook replaces only its owned files.

Provide one development Colab badge per notebook using the official repository
URL form and the current development branch. Label those badges as development
only, say that edits are not automatically saved back to GitHub, and record a
certification table whose local candidate, fresh Colab, and immutable
badge-reference rows begin pending. Before publication, all badges must point
to one confirmed immutable release tag and every notebook must pass a fresh
runtime from that exact reference. A development badge, authored run, or stored
notebook output is not certification.

## API evidence

The design was calculated with `uv 0.11.29` on CPython 3.12.13 using NumPy
2.0.2, pandas 3.0.3, statsmodels 0.14.6, scikit-learn 1.9.0, and Matplotlib
3.11.1. Installed signatures were inspected for the formula API,
`train_test_split`, `DummyRegressor`, `StandardScaler`, `LinearRegression`,
`Pipeline`, regression/classification metrics, and Matplotlib figure creation
and saving.

Primary documentation checked on 2026-07-19:

- statsmodels OLS prediction results and mean/new-observation intervals:
  <https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_prediction.html>
  and
  <https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.PredictionResults.html>;
- scikit-learn `StandardScaler` training-state semantics:
  <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>;
- the scikit-learn metrics API, including the dedicated
  `root_mean_squared_error` available since 1.4:
  <https://scikit-learn.org/stable/api/sklearn.metrics.html> and
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html>;
  and
- accuracy and recall definitions/signatures:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>
  and
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html>.

These checks support the candidate implementation contract. They do not turn
the candidate stack into the final course release lock.

## Independent adversarial QA matrix

After implementation, a reviewer who did not author these notebooks must try to
refute the package with all of the following checks.

### Fresh execution and layouts

- Fresh-execute all three notebooks from the repository root, `10/demo/`, a
  directory nested under `10/demo/`, and a disposable flattened standalone
  package: 12 notebook/layout cases.
- Repeat all 12 cases with notebook-code warnings promoted to errors, for 24
  total fresh candidate-stack executions. Kernel transport/shutdown messages
  outside notebook code are recorded separately and never used to conceal a
  notebook warning.
- Assert each run used the exact candidate versions, resolved the intended demo
  root, wrote only below that root, and finished at its final verification
  cell. A zero process exit without these observable results is not a pass.

### Fixtures, outputs, and repeat behavior

- For each notebook, delete only its needed fixture in a disposable copy and
  verify exact-byte reconstruction and checksum before parsing.
- For each notebook, replace its needed fixture with plausible wrong CSV bytes;
  require a checksum failure before parsing, fixture replacement, output
  directory creation, or owned-output cleanup.
- For each notebook, separately test one deleted owned output, one plausible
  stale text/CSV/PNG output, and one binary-corrupt owned output. Require
  deterministic replacement of all owned files.
- Repeat each successful notebook without cleanup and require identical hashes
  for its CSV outputs and revalidated PNG structure/plot data. Preserve an
  unrelated binary sentinel and outputs owned by the other notebooks in every
  case.
- Verify the exact final ten-file inventory, then remove generated output,
  notebook checkpoints, and Python caches from the repository working tree.

### Numerical and split invariants

- Recompute the fixture byte counts/hashes, parsed dtypes, row grain and key
  uniqueness, and exact one-day horizon.
- Refit the OLS model independently; verify all three coefficients, the
  study-hours standard error and CI, near-zero mean residual, exact new-case
  interval values, and wider individual interval.
- Reproduce the exact 18/6/6 exchangeable membership and 22/7/11 chronological
  roles; verify conservation, pairwise disjointness, UTC order, and strict
  target-time boundaries.
- Verify the availability decisions are exactly keep/keep/keep/reject/reject,
  and statically/runtime-confirm that the two rejected candidates were not
  computed and no Demo 2 scaler/model fit occurred.
- Independently recompute the training target mean and scaler means. Inspect the
  fitted Pipeline to prove scaler and linear-model state came from training
  rows only.
- Recompute both validation rows and verify that selection uses only the
  predeclared validation MAE: require `l10d3-08-validation` to contain no test
  partition access, test prediction, test metric, or final-test artifact write,
  and require `selected_approach_name` to be assigned exactly once in the
  notebook source. Execute through that cell and assert
  `selected_approach_name == "linear_pipeline"` while a prediction-call ledger
  contains no test-row call. Then execute `l10d3-10-test-and-plot` and assert
  the selected name is unchanged, exactly one prediction call received the 11
  exact test row IDs,
  and only then were the one final test-metric row and all 11 final
  predictions/residuals computed. Recompute both binary metric rows and confirm
  no classifier was imported or fit.

### Visual, pedagogy, and scope checks

- Inspect both PNGs at original detail. Verify exact dimensions, titles, axes,
  residual sign definitions, horizontal zero lines, all source points,
  legibility, contrast, no clipping, and no implication that the plots prove
  assumptions or general performance.
- Parse all notebooks for the exact 11/11/13 topology, portable kernelspec,
  null counts, zero outputs, compiling code, and 35 globally unique IDs.
- Build a first-definition/first-use ledger. Terms must precede demanding use,
  especially estimand/association/causation/assumption, both interval meanings,
  timestamp/horizon/cutoff/availability, exchangeability/split roles/leakage,
  baseline/Pipeline/metrics/final evaluation, and positive-class metrics.
- Verify the guide uses direct learner actions and exact expected outcomes and
  contains no instructor/meta language or timing. Check all local links,
  candidate-version statements, fixture/output hashes, privacy/ephemeral-state
  language, and conditional assignment-Colab statement.
- Search required code and guide instruction for network/runtime data, upload,
  Drive, credentials, unseeded randomness or random observation generation,
  mutable dates, p-value grading, interaction
  terms, AIC/BIC selection, regularization, cross-validation, Random Forest,
  feature importance, XGBoost/boosting, deep learning/framework surveys,
  classifier fitting, forecasting, deployment, monitoring, or a second plotting
  library. Expected teaching uses are zero; scope-boundary prose in the guide
  is allowed only when clearly marked as excluded.
- Verify the exact package tree, pins, fixture hashes, no paired Markdown,
  Jupytext, legacy dataset cache, generated output, or unrelated `10/assignment`
  edit. Run scoped whitespace/diff checks and the dependency-free course audit.

### Colab and release lifecycle

- Fresh-run every notebook in the supported Colab runtime with no uploaded
  file, Drive mount, credential, or prior runtime state. Record exact versions,
  warnings, paths, fixture hashes, output hashes/PNG checks, and result.
- Confirm each production badge points to the same approved immutable course
  release and rerun from those exact URLs. Do not mark the lifecycle complete
  from a development-branch badge.
- Keep local implementation acceptance, fresh-Colab acceptance, immutable badge
  acceptance, and the course-wide package freeze as distinct recorded gates.

## Design evidence and unresolved gates

The literal fixtures and eight CSV references were recomputed in repeated
candidate-environment runs and checked for byte/hash stability. The OLS values,
interval distinction, seeded and chronological split memberships, scaler
state, validation/test metrics, predictions, and supplied binary metrics match
the independently verified Lecture 10 narrative evidence.

No pedagogical, fixture, output, or package-layout choice remains unresolved in
this author design. Implementation is still blocked on an independent design
review. After that review, publication separately requires independent local
implementation QA, fresh Colab execution, one canonical immutable release for
all badges, and the course-wide version freeze. This design does not authorize
Assignment 10 work or an unconditional assignment-Colab submission path.
