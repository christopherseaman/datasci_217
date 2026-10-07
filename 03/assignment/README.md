# Assignment 03: Telemetry Ward Analysis with NumPy

## Overview

Build the project environment, count the dataset from the shell, and answer the ward's questions with NumPy, saving each result for the checks.

`data/bp_readings.csv` is one shift's export from a step-down unit: one row per patient with a patient id, the bedside monitor that recorded them, and 12 hourly automated systolic blood pressure readings in mmHg. Leave it exactly as shipped; the checks compare your answers with what it gives.

```text
patient_id,monitor,sbp_h01,sbp_h02, ... ,sbp_h12
P0001,M06,111,122, ... ,108
```

## Setup

1. Fork the assignment repository on GitHub and clone your fork as in Lecture 01: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open the cloned folder itself, not a folder above it.
2. Open **Terminal → New Terminal** in VS Code.
    - Expect: `ls data` prints `bp_readings.csv`.

## Files

```text
ds217-03/
├── data/bp_readings.csv    # supplied readings; keep exactly as handed out
├── analysis.py             # starter script: the CSV loader is written, the analysis is yours
├── pyproject.toml          # supplied: the project's one direct dependency, numpy
├── uv.lock                 # supplied: the exact versions `uv sync` installs
├── CHECKS.md               # supplied: what each check looks for
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _public_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
├── .python-version         # you create in Task 1
└── output/
    ├── environment.txt             # you generate in Task 1
    ├── record_count.txt            # you generate in Task 2.1
    ├── monitor_counts_<timestamp>.txt  # you generate in Task 2.2, one file per run
    └── vitals_summary.txt          # you generate in Task 3
```

## Task 1: Build and record the environment

### 1.1 Create the environment

1. Pin the course interpreter, create the environment, activate it, and install the packages `pyproject.toml` and `uv.lock` list. Do not run `uv init`: the project files already exist.

    ```bash
    uv python pin 3.13
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```

    - If `uv venv` asks `Do you want to replace it? [y/n]`, answer `n`; `source .venv/bin/activate` and `uv sync` still work.
    - Expect: `uv sync` lists `+ numpy==2.3.3`, and `uv python pin` writes `.python-version`, which you commit.

> **Checkpoint: the environment**
> `python3 -c "import numpy as np; print(np.__version__)"` prints `2.3.3`.

### 1.2 Save an environment probe

1. With the environment active, save three labelled lines to `output/environment.txt`, each a label, a colon, and a value:

    ```text
    python: <the version the active interpreter reports>
    numpy: <the version of numpy installed in it>
    interpreter: <the path to the active interpreter>
    ```

2. Build each line with command substitution (Lecture 03's "Shell Variables and Timestamps" card): `>` starts the file, `>>` adds to it.

    ```bash
    echo "python: $(python3 --version)" > output/environment.txt
    ```

3. Use Lecture 03's one-line Python commands for the NumPy version and the interpreter path. Quotes inside `$( )` belong to the command inside it.

> **Checkpoint: `output/environment.txt`**
> Three lines. The `numpy` line shows a version number and the `interpreter` line a path inside your project's `.venv`.

## Task 2: Count the dataset from the shell

Build both answers with a shell pipeline (`tail`, `cut`, `sort`, `uniq -c`, `wc -l`), not with Python. Demos 1.6 and 1.7 count clinic encounters the same way.

### 2.1 How many patients are in the file?

1. Drop the header line, which is not a patient record, then count the rest.
2. Save the count to `output/record_count.txt`.

> **Checkpoint: `output/record_count.txt`**
> The number of patient rows. Counting the whole CSV counts the header too, one too many.

### 2.2 How many patients did each monitor record?

1. Capture the timestamp once into a shell variable (the card's `date` format string gives `YYYYMMDD_HHMMSS`), as Demo 1.7's `count_clinics.sh` does.
2. Count the patients per monitor and save the result to `output/monitor_counts_<timestamp>.txt`, so a second run keeps the first result.
    - Expect: six lines, one per monitor, each a count and a monitor id as `uniq -c` prints them. Leave off Demo 1.6's `| head -n 5`, or the sixth monitor goes missing.

> **Checkpoint: `output/monitor_counts_<timestamp>.txt`**
> A file in `output/` named `monitor_counts` plus the run timestamp, with one line per monitor.

## Task 3: Answer the ward's questions with NumPy

`analysis.py` already loads the CSV into arrays. Answer the questions below from those arrays and save them to `output/vitals_summary.txt`.

Two definitions:

- A patient's **12-hour mean** is the mean of that patient's twelve readings: `readings.mean(axis=1)`, one number per patient.
- A monitor's **average** is the mean of the 12-hour means of the patients it recorded.

Steps for the monitor questions:

1. Group patients with Lecture 03's "Select One Group by a Label" snippet: `monitors == "M01"` builds a Boolean mask, and indexing the 12-hour means with it keeps that monitor's patients. Mask and values must both hold one entry per patient.
2. Collect each monitor's average in a list in the order of `sorted(set(monitors))`, turn it into an array with `np.array()`, and pick the name with `argmax()`, as the "Find the Highest Values and Who Has Them" snippet does.
3. Flip the mask with `~` to select every other monitor's patients, as Demo 3.4 does.

Write one `key: value` line per answer, ending each write with `"\n"` as Lecture 02's `file.write(f"{result}\n")` does, and open the file with `"w"` so each run replaces it:

```text
patients: <whole number>
mean_sbp: <number>
high_monitor: <monitor id>
```

| Key                     | The question it answers                                                                                   | Value                                | Points |
| ----------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------ | -----: |
| `patients`              | How many patients does the file describe?                                                                 | Whole number                         |      4 |
| `readings`              | How many individual readings does it hold, counting every patient and every hour?                         | Whole number                         |      4 |
| `mean_sbp`              | What is the mean of every reading in the file?                                                            | mmHg                                 |      3 |
| `sd_sbp`                | What is the standard deviation of every reading?                                                          | mmHg                                 |      3 |
| `min_sbp`               | What is the lowest single reading?                                                                        | Whole number of mmHg                 |      3 |
| `max_sbp`               | What is the highest single reading?                                                                       | Whole number of mmHg                 |      3 |
| `stage2_patients`       | How many patients have a 12-hour mean of 140 mmHg or higher?                                              | Whole number                         |      4 |
| `highest_patient`       | Which patient has the highest 12-hour mean?                                                               | `patient_id` as written in the file  |      4 |
| `highest_patient_mean`  | What is that patient's 12-hour mean?                                                                      | mmHg                                 |      4 |
| `peak_hour_column`      | Which hour column has the highest mean across all patients?                                               | Column name as written in the header |      4 |
| `peak_hour_mean`        | What is that column's mean?                                                                               | mmHg                                 |      4 |
| `high_monitor`          | Which monitor's average is highest?                                                                       | Monitor id as written in the file    |      4 |
| `monitor_offset`        | How far above the average of the patients on the _other_ monitors does that monitor's average sit?        | mmHg                                 |      3 |
| `stage2_other_monitors` | Leaving out the patients on that monitor, how many of the rest have a 12-hour mean of 140 mmHg or higher? | Whole number                         |      3 |

Put the answer first and any note after it: `readings: 3600 (300 x 12)` reads 3600. Every number answer is one number, not an array. [CHECKS.md](CHECKS.md) lists how values are read.

> **Checkpoint: `output/vitals_summary.txt`**
> One `key: value` line for each of the 14 keys above.

## Check your work

- Run `python3 analysis.py`, then `python3 check_assignment.py`.
- The checker prints your score and what to fix, using the latest checks from the course repository: the same checks GitHub runs on each push.
- Commit `.python-version`, `analysis.py`, and the `output/` files, then push.

What each check looks for: [CHECKS.md](CHECKS.md)
