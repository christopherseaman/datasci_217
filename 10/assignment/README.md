# Assignment 10: bounded modeling and honest evaluation

This assignment is one notebook with three cumulative tasks:

1. fit and interpret one bounded multivariable OLS model;
2. write a prediction contract, audit feature availability, and make a chronological split;
3. compare a mean baseline with one train-only linear pipeline on validation, freeze the choice, and evaluate it once on test.

The records are course-authored synthetic data. They do not describe real people, customers, or operations.

## Work locally

1. Open the `10/assignment` subtree, or its exported standalone assignment repository, with the course-approved Git GUI.
2. Open this whole assignment folder in the approved local Jupyter environment. A lone notebook upload is not supported because the notebook requires `data/`.
3. Restart the kernel, clear all outputs, and run every cell in order.
4. From a terminal opened in this assignment folder, run `python check_assignment.py` with the course Python environment.
5. Run all notebook cells a second time. Run the checker again.
6. Inspect the notebook and the nine files in `output/` in your Git GUI. Commit and push the notebook and generated outputs through the GUI.

Do not edit protected cells, fixture files, this README, `PLATFORM_CHECK.md`, `requirements.txt`, or `check_assignment.py`. Do not add files. The optional Actions workflow is supplied feedback; it is not a submission artifact.

## Colab boundary

Colab is not part of this repository contract. If the instructor later publishes a launch route, it must place the entire assignment tree, including `data/`, in the runtime; uploading `assignment.ipynb` by itself is not supported.

## Required output

Your completed run must retain `output/.gitkeep` and create exactly:

- `inference_summary.csv`
- `inference_case_intervals.csv`
- `inference_residuals.png`
- `availability_decisions.csv`
- `split_manifest.csv`
- `validation_metrics.csv`
- `final_test_metrics.csv`
- `final_predictions.csv`
- `binary_metrics.csv`

The public checker checks readiness and structure. It does not execute the notebook, award points, or judge explanation quality. The central grader always clears stored notebook output and executes from a fresh kernel.

## Assessment

The central grader reports 90 automated points:

- template, environment, fixtures, and protected integrity: 10
- bounded OLS inference and intervals: 20
- contract, availability, leakage, and chronological split: 25
- train-only comparison, freeze, final test, and binary metrics: 30
- portability, visible output, repeatability, and resubmission: 5

Human review covers 10 points: Task 1 interpretation (3), Task 2 reasoning (3), and Task 3 evaluation judgment (4).

Advanced models, regularization, cross-validation, model search, feature importance, classifier fitting, and test-set model selection are outside this assignment.
