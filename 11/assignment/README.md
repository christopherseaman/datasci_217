# Final Project: Chicago Beach Weather Forecasting

**Total:** 100 points
**Data:** Chicago Beach Weather Stations, 2022-2024

## Overview

In this project, you will complete a nine-phase data science workflow with hourly observations from the Foster and Oak Street weather stations. Your goal is practical: for each station and cutoff hour, predict the air temperature one elapsed hour later.

You will audit a frozen release, clean sensor values, construct a complete hourly panel, engineer past-only features, explore training data, make chronological splits, compare one scikit-learn model with a persistence baseline, and communicate the result. There is no performance threshold. Careful, reproducible work matters more than finding a complicated model.

Start with [`assignment.md`](assignment.md), which is the exact artifact contract, then work through the nine notebook pairs in order.

## Quick Start

From `11/assignment`:

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install -r requirements.txt
./download_data.sh
jupyter lab
```

Open the question files in your preferred editor. Notebook execution is optional
local QA; the committed CSV/PNG artifacts and `report.md` are the grading
deliverables.

The release and provenance manifest are committed under `data/`. `download_data.sh` does not download or replace anything; it verifies those committed files.

## Nine Phases

| Question | Points | Notebook | Main result |
|---|---:|---|---|
| Q1 | 7 | [`q1_setup_exploration.ipynb`](q1_setup_exploration.ipynb) | Audit and explore the frozen release |
| Q2 | 9 | [`q2_data_cleaning.ipynb`](q2_data_cleaning.ipynb) | Clean timestamps and sensor values |
| Q3 | 11 | [`q3_data_wrangling.ipynb`](q3_data_wrangling.ipynb) | Build a complete station-hour panel |
| Q4 | 14 | [`q4_feature_engineering.ipynb`](q4_feature_engineering.ipynb) | Build leakage-safe forecast features |
| Q5 | 7 | [`q5_pattern_analysis.ipynb`](q5_pattern_analysis.ipynb) | Describe training-only patterns |
| Q6 | 11 | [`q6_modeling_preparation.ipynb`](q6_modeling_preparation.ipynb) | Create fixed chronological splits |
| Q7 | 13 | [`q7_modeling.ipynb`](q7_modeling.ipynb) | Select and validate one sklearn model |
| Q8 | 13 | [`q8_results.ipynb`](q8_results.ipynb) | Evaluate the untouched test period |
| Q9 | 15 human | [`q9_writeup.ipynb`](q9_writeup.ipynb) | Complete `report.md` |

Lecture 11 demonstrates the workflow. Geographic material is outside this
assignment: do not add maps, coordinates, spatial joins, geographic features,
or geographic dependencies.

## Submission

Commit the nine `.md`/`.ipynb` pairs, `report.md`, and all required artifacts under `output/`. Do not modify files under `data/`. Keep notebook outputs cleared in the submitted notebooks; generated CSV and PNG artifacts remain in `output/`.

Before submitting:

```bash
./download_data.sh
jupytext --to ipynb --test-strict q*.md
uv run check_assignment.py
```

Students, GitHub Actions, and graders use the same public `grading.py` rules through `check_assignment.py`. The checker reads saved artifacts and requires the nine coursework `.md`/`.ipynb` pairs, but does not execute notebooks or refit models. Passing phase points are retained: the automated maximum is 85, with Q9's 15 points awarded by human review of `report.md` reasoning and communication. Automated report-structure checks award no human-review points. The full rubric and tests are public.

Graders run the trusted assignment copy against a submission with `python check_assignment.py /path/to/submission --json`. No grader-only settings or required runner metadata change the score. Use the public diagnostics to complete your own assignment rather than copying example content.

See [`HINTS.md`](HINTS.md) for nudges and [`PLATFORM_CHECK.md`](PLATFORM_CHECK.md) for environment checks.
