# Next-Hour Chicago Beach Air Temperature Forecast

## Executive Summary

[State the release period, forecast grain, selected sklearn regressor, and one test result.]

## Data and Cleaning

[Summarize the release audit, station coverage, timestamp handling, and range-rule effects.]

![Release exploration](output/q1_visualizations.png)

## Patterns

[Report one or two training-only monthly or local-hour patterns.]

![Training patterns](output/q5_patterns.png)

## Forecast Design

[Describe the exact next-hour target, persistence baseline, past-only features, pipeline preprocessing, and chronological splits.]

## Model Results

[Replace the four metric rows below with the two Q7 validation rows and two Q8 test rows; keep this six-column layout.]

| Evaluation set | Model | MAE | RMSE | R2 | n |
|---|---|---:|---:|---:|---:|
| Validation | persistence_baseline | [value] | [value] | [value] | [value] |
| Validation | student_model | [value] | [value] | [value] | [value] |
| Test | persistence_baseline | [value] | [value] | [value] | [value] |
| Test | student_model | [value] | [value] | [value] | [value] |

[Compare the model with persistence and note any station-level difference without tuning on test results.]

![Final model results](output/q8_final_visualizations.png)

## Limitations

[Give concrete limitations tied to the two-station release, sensor missingness, predictors, chronological evaluation, or deployment assumptions.]
