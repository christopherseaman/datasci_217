# Lecture 10 demonstration guide

These three demonstrations build one bounded modeling workflow: state an
inferential claim and its assumptions, choose splits and features from an
information-availability contract, then select one prediction approach on
validation data and inspect the test partition once. The fixtures are
course-authored, synthetic, and non-identifying.

Required Lecture 10 demos are Colab-first and run equivalently in local Jupyter
or the VS Code notebook interface. Colab storage is ephemeral, and edits made
in a notebook opened from GitHub are not automatically saved back to GitHub.
Assignment use of Colab remains conditional on the repository-save and
Classroom 50 pilot. Restart the kernel and run every cell in order; stored
notebook output is never execution evidence. Do not place credentials, private
records, or sensitive generated files in these shared notebooks.

## Open the notebooks

These badges are development links to `main`, not publication references.
Before publication, replace all three references with the same immutable course
release tag and fresh-run every notebook from those exact URLs.

1. [![Open Demo 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/10/demo/demo1_bounded_ols.ipynb) — ask what one conditional OLS coefficient and its intervals can support.
2. [![Open Demo 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/10/demo/demo2_split_availability_leakage.ipynb) — decide which features and split rule respect a later-date prediction contract.
3. [![Open Demo 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/10/demo/demo3_honest_prediction_evaluation.ipynb) — select between a baseline and one linear Pipeline without spending the test partition during development.

For local use, open the repository with Python 3.12.13 and the exact direct
dependencies in `requirements.txt`. Start Jupyter from the course root,
`10/demo/`, or a directory nested under `10/demo/`. A flattened standalone copy
also works. Each supplied setup cell finds the marked demo root, checks package
versions, and installs only mismatched candidate packages before teaching
imports.

Candidate environment:

```text
Python 3.12.13
NumPy 2.0.2
pandas 3.0.3
statsmodels 0.14.6
scikit-learn 1.9.0
Matplotlib 3.11.1
```

## Fixed inputs and generated outputs

The workshop fixture has grain one hypothetical participant and key
`participant_id`:

- `data/workshop_participants.csv` — 200 bytes, SHA-256
  `eefb5f1023e9b84106f407800fa0db72853b6876d58e61255c346ed2d2d32f05`.

The station fixture has grain one Station A daily prediction issue and key
`row_id`:

- `data/station_next_day.csv` — 3,879 bytes, SHA-256
  `f95330b252c6e0f12026577602c69e21d01dbec232b5e523c6c41b0b62cf85a8`.

Every notebook verifies present fixture bytes before parsing. If a fixture is
absent in an ephemeral standalone or Colab layout, the notebook reconstructs
the exact supplied bytes in memory. A present but corrupt fixture stops the run
and is not replaced. No upload, Drive mount, credential, network data fetch,
random observation generation, or mutable date is needed.

Successful fresh execution creates only these ignored files under `output/`:

| File | Expected content | Bytes | SHA-256 |
|---|---|---:|---|
| `inference_summary.csv` | one study-hours coefficient and interval row | 95 | `feccc3b50dcd46cd3bfb0c73d246940244823d2d9711da91ca1515d7bd9a1066` |
| `prediction_intervals.csv` | one new-participant mean and individual interval row | 200 | `6bbe7d288b39efc870ead781363dc747223d336b1fe521c59eec724927469663` |
| `ols_residuals.png` | 12 OLS fitted-versus-residual points | platform image | inspect at 1,200 by 720 pixels |
| `availability_decisions.csv` | five candidate-feature decisions | 416 | `e765b4426412525b06fcfad9717af158e786e7e0ff32c570341927192d6020f2` |
| `split_manifest.csv` | 40 chronological row roles | 2,755 | `92a1fb047929e318d0f0259634cfa454bad58076779dbc5058992a5b70e3d9d0` |
| `validation_metrics.csv` | baseline and linear validation metrics | 116 | `899d246c84c5116e857e81a4327055819d3bef1d214e4199e6247f82fd69d25f` |
| `final_test_metrics.csv` | one selected-approach test row | 79 | `8c4163c7d9f5de0a49f41646d30a68b4569e96c1e7b69d0d89f7706fbc3ffc6a` |
| `final_predictions.csv` | 11 held-out predictions and residuals | 843 | `eb2b5a5ffa2d62cdfb6f92b1a6e7fea0c2c5c4b1d33e433f4d2cd5c568c08090` |
| `prediction_residuals.png` | 11 final-test prediction residual points | platform image | inspect at 1,200 by 720 pixels |
| `binary_metrics.csv` | two supplied binary-prediction metric rows | 119 | `52180318d2393b626ceeafc07a93ba751bcd7bb2f56907a6a9e746b783632ba3` |

Rerunning a notebook replaces only that notebook's owned files. It preserves
the other notebooks' outputs and unrelated sentinels.

## Demo 1: bounded OLS association and uncertainty

Open `demo1_bounded_ols.ipynb`. Before running its model cell, predict whether
the study-hours coefficient will be positive and write one sentence that keeps
its meaning conditional on prior score. Expect an estimate of `1.645244`: under
the stated model and sampling assumptions, one additional study hour is
associated with an estimated 1.645244-point difference in population mean
assessment score while included prior score is held fixed. This is not an
intervention effect.

Before creating the interval row, predict why an interval for one new
participant should be wider than an interval for the population mean response
at the same feature values. Expect widths `2.922607` and `0.899944`,
respectively: an individual outcome includes person-to-person variation in
addition to uncertainty about the mean.

Inspect the visible coefficient table, interval table, and residual plot. In
the plot, `observed - fitted` above zero means the observed assessment is above
its fitted value. Look for curvature, changing spread, or isolated residuals;
their absence does not prove the assumptions.

## Demo 2: split choice and information availability

Open `demo2_split_availability_leakage.ipynb`. Before running either split,
predict whether a reproducible seeded random split is appropriate when the goal
is to predict later dates. Expect "no": a seed stabilizes membership, but it
does not preserve the future-facing deployment order.

Before writing the candidate inventory, identify the two entries that require
information unavailable at the 2026-01-25 feature cutoff. Expect rejection of
`post_outcome_temperature_review_c` because it uses target-time information and
`full_dataset_scaled_current_temperature_c` because its preprocessing learns
from future validation/test feature rows. The notebook records those decisions
without constructing either rejected series.

Inspect the visible `18/6/6` exchangeable-ID membership and the chronological
`22/7/11` counts. The saved split manifest assigns roles from fixed target-time
cutoffs and preserves strict training-before-validation-before-test order.

## Demo 3: honest prediction evaluation

Open `demo3_honest_prediction_evaluation.ipynb`. Before validation is computed,
predict whether the training-mean baseline or the train-only linear Pipeline
will have lower validation MAE. Expect the Pipeline to win with MAE `0.408166`
versus `2.598597`; only then is the fitted Pipeline evaluated once on test,
where MAE is `0.450242`.

Before viewing the final residual plot, predict the sign meaning. Because the
residual is `actual - predicted`, a positive point means the actual next-day
temperature was warmer than predicted, and a negative point means it was
cooler. Treat the plot as a description of this one held-out period, not a
universal performance guarantee.

Before calculating the supplied binary metrics, predict how two approaches can
both have accuracy `0.8` but different recall. Expect the supplied model to
recover one of two positives (`0.5` recall), while the all-zero supplied dummy
recovers none (`0.0` recall). The decision consequences determine which error
tradeoff matters; the demonstration does not declare either supplied prediction
column globally better.

## Recover from common failures

- If the notebook resolves an unexpected root, start Jupyter from the course
  root, `10/demo/`, a nested demo directory, or a complete flattened copy and
  rerun from a fresh kernel. The setup cell prints the resolved root.
- If a candidate version differs, allow the setup cell to install only the
  mismatched pins, restart the kernel if the package manager requests it, and
  run all cells again. The version table must match before teaching imports.
- If a fixture is absent, run from a fresh kernel and observe exact in-memory
  reconstruction. If a present fixture reports a checksum mismatch, restore
  the committed fixture rather than bypassing the check.
- If `output/` is read-only, choose a writable complete copy and rerun. Do not
  redirect generated work into the fixture directory.
- If an owned file contains stale text or binary bytes, rerun its notebook; the
  notebook replaces only its own named outputs.
- If a name is missing or a result changes with cell order, restart the kernel
  and run all cells from the beginning. A final PASS depends on prior fresh
  state and schema-aware artifact readback.
- If a Colab runtime resets, reopen the development notebook and run all cells
  again. Download anything you need before the ephemeral runtime ends; edits
  and generated files are not automatically saved back to GitHub.

## Scope and certification

The required package does not fit a classifier or introduce regularization,
cross-validation, tree models, boosting, deep learning, forecasting,
deployment, or monitoring. The prediction demonstration is limited to
synthetic data, one station and date range, fixed features, conditional model
assumptions, possible distribution change, and one held-out period. It supports
neither a causal claim nor a universal performance guarantee. If a test result
prompts redevelopment, it becomes development evidence and requires a new
untouched test release.

| Gate | Exact environment | Warnings | Result/reference |
|---|---|---|---|
| Independent local candidate | pending | pending | pending |
| Fresh Colab runtime | pending | pending | pending |
| Immutable release-tag badge | pending | pending | pending |

A development badge, author-run notebook, or stored output does not complete
any pending publication gate.
