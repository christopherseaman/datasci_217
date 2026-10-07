# Assignment 11: Chicago Beach Weather Forecasting (Final Exam)

## Overview

Forecast each Chicago beach weather station's air temperature one hour ahead: clean the sensor data, build a station-by-hour panel and past-only features, split by time, fit one scikit-learn pipeline against a persistence baseline, test once, and write a report. Nine notebooks, Q1 to Q9, each save files in `output/`.

`data/chicago_beach_sensors_2022_2024.csv` is a frozen extract of the City of Chicago's beach weather sensors: 50,895 hourly readings from Foster Weather Station and Oak Street Weather Station, for local 2022-01-01 through 2024-12-31.

```text
station_name,measurement_timestamp,air_temperature_c,wet_bulb_temperature_c,relative_humidity_pct,...
Foster Weather Station,2022-01-01 00:00:00,3.39,,87,...
```

- `measurement_timestamp` is Chicago wall-clock time with no time zone written on it.
- `data/release_manifest.json` records the release's file name, SHA-256, size, row and column counts, columns, stations, and time zone.
- Leave both files exactly as they ship: Q1 checks them against each other.

## Setup

1. Fork the assignment repository on GitHub and clone your fork as in Lecture 01: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open the cloned folder itself.
2. In the integrated terminal, create the environment, activate it, install the packages `pyproject.toml` and `uv.lock` list, and verify the data. Do not run `uv init`: the project files already exist.

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    bash download_data.sh
    ```

    - Expect: `uv sync` lists `+ pandas==3.0.5`; the last command prints `Verified frozen release and manifest: data/chicago_beach_sensors_2022_2024.csv (4731351 bytes)`. On a mismatch, discard your changes to `data/` in Source Control and run it again.
3. Open `q1_setup_exploration.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`. Select the same kernel in each notebook.
4. Run the first code cell.
    - Expect: the Python, NumPy, pandas, scikit-learn, and Matplotlib versions, then the first rows of the release.

## Files

```text
assignment/
├── assignment.md                 # the forecasting question, rules, and every file's columns: read it first
├── q1_setup_exploration.ipynb    # Q1 to Q8: complete every TODO, then run top to bottom
├── ...                           # q2_data_cleaning to q8_results, same pattern
├── q9_writeup.ipynb              # Q9: shows your saved results while you write report.md
├── q1_setup_exploration.md ... q9_writeup.md  # supplied plain-text copies of the notebooks; leave them as they are
├── report.md                     # you complete in Q9
├── HINTS.md                      # supplied nudges for each question
├── CHECKS.md                     # supplied: every output file's first line and line count, and what earns points
├── example_report/               # supplied formatting example with made-up numbers
├── data/                         # supplied frozen release and manifest; never edit them
├── download_data.sh              # supplied: verifies the two data files (it downloads nothing)
├── pyproject.toml, uv.lock       # supplied: the packages and exact versions `uv sync` installs
├── .python-version               # supplied: Python 3.13
├── .gitattributes, .gitignore    # supplied: keep the data byte for byte, keep .venv/ out of Git
└── output/                       # you make 28 files, listed in CHECKS.md
    ├── q1_*  # release audit, station coverage, visualizations
    ├── q2_*  # cleaned observations, cleaning audit, missingness
    ├── q3_*  # hourly panel, panel summary
    ├── q4_*  # features, feature manifest
    ├── q5_*  # monthly station summary, correlations, patterns
    ├── q6_*  # X and y for train, validation, test, and the split summary
    ├── q7_*  # validation predictions, validation metrics, model spec, permutation importance
    └── q8_*  # test predictions, test metrics, station metrics, final visualizations
```

## Tasks

1. Read [`assignment.md`](assignment.md): the forecasting question (each station's air temperature one hour after a cutoff hour), cleaning rules, fixed predictors, train, validation, and test periods, and every output file's columns.
2. Complete the nine notebooks in order. Each starts from the files the previous one saved in `output/`, so you can close one and open the next without rerunning anything.
    - Expect: every section that saves a file ends with a **Checkpoint** naming the file, its first line, and its line count.
3. When stuck, open [`HINTS.md`](HINTS.md): one nudge per question, naming the lecture that taught the tool.

| Question | Notebook | What you build | Points |
| --- | --- | --- | ---: |
| Q1 | [`q1_setup_exploration.ipynb`](q1_setup_exploration.ipynb) | Audit the release against its manifest, measure each station's coverage, and plot a first look | 5 |
| Q2 | [`q2_data_cleaning.ipynb`](q2_data_cleaning.ipynb) | Convert times to UTC, apply the sensor rules, and record what changed | 10 |
| Q3 | [`q3_data_wrangling.ipynb`](q3_data_wrangling.ipynb) | Build the complete station-by-hour panel and summarize its gaps | 10 |
| Q4 | [`q4_feature_engineering.ipynb`](q4_feature_engineering.ipynb) | Build the next-hour target and the past-only predictors | 12 |
| Q5 | [`q5_pattern_analysis.ipynb`](q5_pattern_analysis.ipynb) | Describe monthly and hourly patterns in the training period only | 5 |
| Q6 | [`q6_modeling_preparation.ipynb`](q6_modeling_preparation.ipynb) | Split the eligible rows into train, validation, and test by time | 10 |
| Q7 | [`q7_modeling.ipynb`](q7_modeling.ipynb) | Fit one scikit-learn pipeline and compare it with persistence on validation | 12 |
| Q8 | [`q8_results.ipynb`](q8_results.ipynb) | Refit the frozen choice and evaluate the test period once | 11 |
| Q9 | [`q9_writeup.ipynb`](q9_writeup.ipynb) | Complete `report.md` | 25, human review |

## Check your work

1. Restart each notebook's kernel and **Run All**, Q1 through Q8.
    - Expect: every cell finishes without an error.
2. Compare `output/` and `report.md` with [CHECKS.md](CHECKS.md).
3. Save each notebook after **Run All** so its outputs are committed. In Source Control, stage the nine notebooks, `report.md`, and `output/`, commit (for example `Complete the final exam`), and sync.
    - Expect: `output/` on your fork's `main` branch on GitHub shows all 28 files, `report.md` shows its table and images, and `data/` shows no change.

What earns each point: [CHECKS.md](CHECKS.md#completion-contract)
