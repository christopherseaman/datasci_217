# Assignment 10: bounded modeling and honest evaluation

This assignment is one notebook with three cumulative tasks:

1. fit and interpret one bounded multivariable OLS model;
2. write a prediction contract, audit feature availability, and make a chronological split;
3. compare a mean baseline with one train-only linear pipeline on validation, freeze the choice, and evaluate it once on test.

The records are course-authored synthetic data. They do not describe real people, customers, or operations.

## Work locally

1. Open the `10/assignment` subtree, or its exported standalone assignment repository, with the course-approved Git GUI.
2. Open the supplied notebook in the approved editor. The complete local `data/` directory is supplied.
3. Create the ten committed files in `output/`.
4. From a terminal opened in this assignment folder, run `python check_assignment.py` with the course Python environment.
5. Inspect the ten files in `output/` in your Git GUI. Commit and push them with your completed notebook. Graders read the committed artifacts without rerunning your notebook.

Do not edit protected cells, fixture files, this README, `PLATFORM_CHECK.md`, `requirements.txt`, or `check_assignment.py`. Additional input or diagnostic files are allowed; the required artifacts are what grading checks. The optional Actions workflow is supplied feedback; it is not a submission artifact.

## Required output

Your completed run must retain `output/.gitkeep` and create these required artifacts:

- `inference_summary.csv`
- `inference_case_intervals.csv`
- `inference_residuals.csv` — one row per `run_id`, with `actual`, `fitted`, and `residual`
- `inference_residuals.png`
- `availability_decisions.csv`
- `split_manifest.csv`
- `validation_metrics.csv`
- `final_test_metrics.csv`
- `final_predictions.csv`
- `binary_metrics.csv`

The shared public checker reads the committed artifacts and reports the automated score. CSV rows may be ordered differently and numeric serialization may use reasonable float precision; their schema, IDs, missingness, and values are checked. The residual figure is checked only as a PNG file. It does not execute the notebook or judge explanation quality.

## Assessment

Students, GitHub Actions, and graders use the same public 100-point rubric:

- submission package and fixture integrity: 10
- bounded OLS inference and intervals: 25
- contract, availability, leakage, and chronological split: 30
- train-only comparison, freeze, final test, and binary metrics: 30
- residual figure saved as a PNG: 5

There are no separate human-review points. The notebook and written
interpretation remain required coursework artifacts, but the automated score is
based on the committed artifacts.

Graders run the trusted assignment copy with `python check_assignment.py /path/to/submission --json`. It uses the same `grading.py` rules and points as local student checks, without runner metadata or separate grading settings.

Advanced models, regularization, cross-validation, model search, feature importance, classifier fitting, and test-set model selection are outside this assignment.
