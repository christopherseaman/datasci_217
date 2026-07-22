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
uv venv --python 3.12.13
source .venv/bin/activate
uv pip install -r requirements.txt
./download_data.sh
jupyter lab
```

Open `q1_setup_exploration.ipynb`. Local Jupyter is the grading reference. Colab assignment support is pending save-back validation; use local Jupyter unless your instructor announces otherwise.

The release and provenance manifest are committed under `data/`. `download_data.sh` does not download or replace anything; it verifies those committed files.

## Nine Phases

| Question | Points | Notebook | Main result |
|---|---:|---|---|
| Q1 | 8 | [`q1_setup_exploration.ipynb`](q1_setup_exploration.ipynb) | Audit and explore the frozen release |
| Q2 | 10 | [`q2_data_cleaning.ipynb`](q2_data_cleaning.ipynb) | Clean timestamps and sensor values |
| Q3 | 12 | [`q3_data_wrangling.ipynb`](q3_data_wrangling.ipynb) | Build a complete station-hour panel |
| Q4 | 16 | [`q4_feature_engineering.ipynb`](q4_feature_engineering.ipynb) | Build leakage-safe forecast features |
| Q5 | 8 | [`q5_pattern_analysis.ipynb`](q5_pattern_analysis.ipynb) | Describe training-only patterns |
| Q6 | 12 | [`q6_modeling_preparation.ipynb`](q6_modeling_preparation.ipynb) | Create fixed chronological splits |
| Q7 | 14 | [`q7_modeling.ipynb`](q7_modeling.ipynb) | Select and validate one sklearn model |
| Q8 | 14 | [`q8_results.ipynb`](q8_results.ipynb) | Evaluate the untouched test period |
| Q9 | 6 | [`q9_writeup.ipynb`](q9_writeup.ipynb) | Complete `report.md` |

Lecture 11 demonstrates the workflow. The optional NYC Taxi geographic visualization is a lecture demo only; this assignment has no maps, coordinates, spatial joins, geographic features, or geographic dependencies.

## Submission

Commit the nine `.md`/`.ipynb` pairs, `report.md`, and all required artifacts under `output/`. Do not modify files under `data/`. Keep notebook outputs cleared in the submitted notebooks; generated CSV and PNG artifacts remain in `output/`.

Before submitting:

```bash
./download_data.sh
jupytext --to ipynb --test-strict q*.md
uv run check_assignment.py
```

The central grader deterministically evaluates saved artifacts, not notebook source. The local checker is a structural/readiness check. Central grader test names and diagnostics are discoverable in grading feedback; use them to finish your own assignment rather than copying example content. Passing phase points are retained, and failed or dependency-blocked checks receive targeted human review. Q9 checks structure only, not prose quality.

See [`HINTS.md`](HINTS.md) for nudges and [`PLATFORM_CHECK.md`](PLATFORM_CHECK.md) for environment checks.
