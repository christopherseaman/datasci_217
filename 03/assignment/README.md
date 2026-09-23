# Assignment 03: Telemetry Ward Analysis with NumPy

## Files

```text
assignment/
├── data/bp_readings.csv    # supplied readings; keep this file exactly as handed out
├── analysis.py             # starter script: the CSV loader is written, the analysis is yours
├── requirements.txt        # supplied: the one direct dependency this project installs
├── check_assignment.py     # supplied: run it to check the shape of your work; keep unchanged
├── grading.py, _public_checks.py  # supplied: the checks themselves; keep unchanged
├── .python-version         # you create in Task 1
└── output/
    ├── environment.txt             # you generate in Task 1
    ├── record_count.txt            # you generate in Task 2
    ├── monitor_counts_<timestamp>.txt  # you generate in Task 2, one file per run
    └── vitals_summary.txt          # you generate in Task 3
```

## The data

`data/bp_readings.csv` is one shift's export from a step-down unit. Each row is one patient: a patient id, the bedside monitor that recorded them, and 12 hourly automated systolic blood-pressure readings in mmHg.

```text
patient_id,monitor,sbp_h01,sbp_h02, ... ,sbp_h12
P0001,M06,111,122, ... ,108
```

Leave this file exactly as it ships: the checks that grade your answers recompute them from it.

## Where your work is judged

Grading happens in two places, and both read only your committed artifacts. Neither ever runs or reads your Python code, so any way of producing a correct artifact counts.

| Where | What it checks | What it cannot tell you |
| --- | --- | --- |
| `python check_assignment.py`, in your repository | The shape of each artifact: the file is there, it is readable text, it carries the required labels, and each value is a number or a label in a range a clinician would accept. | Whether a value is right. The answers are not in your repository. |
| GitHub Actions, on every push | The same shape checks, plus every answer compared with the value recomputed from `data/bp_readings.csv`. | n/a |

Run the local checks to catch a missing file, a missing key, or a typo before you push; push to find out whether the analysis is right.

Work in `03/assignment` or its standalone repository. In VS Code, sync `main` and use **Git: Create Branch** to create `feature/numpy-analysis`, then open **Terminal → New Terminal** at the assignment directory. If you use a native terminal or WSL Ubuntu instead, `cd` into the assignment directory first.

> **Windows:** work in the **WSL: Ubuntu** window from Lecture 01's setup. Task 2 uses `tail`, `cut`, `sort`, `uniq`, and `wc`, which native PowerShell does not have. Git Bash also provides them; there the environment activates with `source .venv/Scripts/activate` instead.

## Task 1: Build and record the environment

### 1.1 Create the environment

Pin the course interpreter, create the project environment, activate it, and install the supplied requirement:

```bash
uv python pin 3.13
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`uv python pin` writes the `.python-version` file for you; commit it.

> **Checkpoint: `.python-version`**
> Records the course interpreter series, and `requirements.txt` still pins numpy.

### 1.2 Save an environment probe

With the environment active, save three labelled lines to `output/environment.txt`:

```text
python: <the version the active interpreter reports>
numpy: <the version of numpy installed in it>
interpreter: <the path to the active interpreter>
```

Each line is a label, a colon, and the value. Command substitution from the lecture's "Variables and Timestamps" card builds a labelled line from a command's output, and `>` starts the file while `>>` adds to it:

```bash
echo "python: $(python --version)" > output/environment.txt
```

Lecture 03 gives the one-line Python commands that print the installed NumPy version and the interpreter path.

> **Checkpoint: `output/environment.txt`**
> Three lines: the Python version (any version is accepted), the numpy version `requirements.txt` pins, and an interpreter path inside your project's `.venv`.

## Task 2: Count the dataset from the shell

Build both answers with a shell pipeline (`tail`, `cut`, `sort`, `uniq -c`, `wc -l`), not with Python. Demo 1 of the [Lecture 03 demo guide](https://github.com/christopherseaman/datasci_217/blob/main/03/demo/DEMO_GUIDE.md) counts a smaller file the same way.

### 2.1 How many patients are in the file?

Save the count of patient records to `output/record_count.txt`. The header line is not a patient record, so drop it before counting.

> **Checkpoint: `output/record_count.txt`**
> Holds the number of patient rows in `data/bp_readings.csv`. Only the first number in the file is read, so a bare count or a line with a word after it both work. Counting the whole CSV counts the header too, which is one too many.

### 2.2 How many patients did each monitor record?

Count the patients per monitor and save the result under a name carrying the run's timestamp, so a second run keeps the first result instead of overwriting it:

```text
output/monitor_counts_YYYYMMDD_HHMMSS.txt
```

Capture the timestamp once into a shell variable and use it in the filename; the lecture's "Variables and Timestamps" reference card gives the `date` format string that produces `YYYYMMDD_HHMMSS`.

> **Checkpoint: `output/monitor_counts_<timestamp>.txt`**
> One line per monitor with that monitor's count and its id, as `uniq -c` prints them. Spacing, separators such as `M01: 58`, and line order do not matter, and earlier timestamped runs may sit beside it.

## Task 3: Answer the ward's questions with NumPy

Blood pressure on this unit peaks at some point in the monitored shift, a share of patients average into stage 2 hypertension, and the staff suspect one bedside monitor reads high. `analysis.py` already loads the CSV into arrays; answer the questions below from those arrays and save the answers to `output/vitals_summary.txt`.

Two definitions the questions use:

- A patient's **12-hour mean** is the mean of that patient's twelve readings: one number per patient, which is `readings.mean(axis=1)`.
- A monitor's **average** is the mean of the 12-hour means of the patients it recorded. Every patient has twelve readings, so that is the same number as the mean of all of that monitor's readings.

Grouping patients by their monitor is what the optional Demo 3.4 script does with clinics: `systolic[clinics == clinic]` builds a Boolean mask from a text column and keeps the values belonging to one group. The mask and the values it selects have to be the same length, so group an array holding one value per patient, the 12-hour means, with `monitors`, which also holds one value per patient.

Write one line per answer, a key, a colon, and the value:

```text
patients: <whole number>
mean_sbp: <number>
high_monitor: <monitor id>
```

| Key | The question it answers | Value | Points |
| --- | --- | --- | ---: |
| `patients` | How many patients does the file describe? | Whole number | 4 |
| `readings` | How many individual readings does it hold, counting every patient and every hour? | Whole number | 4 |
| `mean_sbp` | What is the mean of every reading in the file? | mmHg | 3 |
| `sd_sbp` | What is the standard deviation of every reading? | mmHg | 3 |
| `min_sbp` | What is the lowest single reading? | Whole number of mmHg | 3 |
| `max_sbp` | What is the highest single reading? | Whole number of mmHg | 3 |
| `stage2_patients` | How many patients have a 12-hour mean of 140 mmHg or higher? | Whole number | 4 |
| `highest_patient` | Which patient has the highest 12-hour mean? | `patient_id` as written in the file | 4 |
| `highest_patient_mean` | What is that patient's 12-hour mean? | mmHg | 4 |
| `peak_hour_column` | Which hour column has the highest mean across all patients? | Column name as written in the header | 4 |
| `peak_hour_mean` | What is that column's mean? | mmHg | 4 |
| `high_monitor` | Which monitor's average is highest? | Monitor id as written in the file | 4 |
| `monitor_offset` | How far above the average of the patients on the *other* monitors does that monitor's average sit? | mmHg | 3 |
| `stage2_other_monitors` | Leaving out the patients on that monitor, how many of the rest have a 12-hour mean of 140 mmHg or higher? | Whole number | 3 |

How the values are read:

- Each answer is scored on its own, so a wrong value costs only its own points.
- mmHg values are accepted within 0.6 of the value recomputed from the data, so a whole number, one decimal, or every digit NumPy prints all pass, and a trailing unit such as `mmHg` is ignored. Counts and whole-number readings must match exactly.
- A NumPy scalar printed as `np.float64(121.5)` or `np.int64(96)` reads as the number inside it.
- Either the population or the sample standard deviation is accepted; at this many readings they agree far inside the tolerance.
- "140 mmHg or higher" includes a mean of exactly 140.
- Keys may appear in any order, spacing is free, and extra lines are ignored.

> **Checkpoint: `output/vitals_summary.txt`**
> One `key: value` line for each of the 14 keys above, holding the answers your analysis computed from `data/bp_readings.csv`.

## Check your work

With the environment active, run your script and then the checks, from the assignment directory:

```bash
python analysis.py
python check_assignment.py
```

These checks read your committed artifacts and confirm that each one is well formed. They do not hold the answers, so a complete run says only that:

```text
19 of 19 shape checks passed.
These checks confirm the shape of your artifacts; your values are checked when you push.
```

It reports a count rather than a score, because it has not looked at a single answer. When something is off it names the artifact to revise. Your answers are compared with the readings when you push, and the GitHub Actions run reports the same nineteen checks, each carrying its own points.

### Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `.python-version`, `requirements.txt` | They record the course interpreter series and a pinned numpy version. | environment records | 5 |
| `output/environment.txt` | Its `python`, `numpy`, and `interpreter` lines agree with those records and name an interpreter inside `.venv`. | environment probe | 8 |
| `output/record_count.txt` | It holds the number of patient records in the supplied CSV. | record count artifact | 10 |
| `output/monitor_counts_<timestamp>.txt` | A timestamped file holds every monitor's patient count. | monitor counts artifact | 15 |
| `output/vitals_summary.txt` | It has a readable `key: value` line for all 14 keys. | summary artifact format | 12 |
| `output/vitals_summary.txt` | Each of the 14 answers matches the supplied readings. | one check per key, named `answer: <key>` | 50 |

Extra files and extra lines are ignored.

## Submit

In VS Code Source Control, stage `.python-version`, `analysis.py`, and everything in `output/`, including every timestamped counts file you kept. Commit with `Analyze telemetry ward readings`. Keep `.venv/` out of the commit; `.gitignore` already lists it.

Publish or sync the branch. With no unfinished changes, switch to `main`, run **Git: Merge...**, and select `feature/numpy-analysis`. Resolve any unexpected conflict, inspect the resolution, and sync. Confirm the committed artifacts on `main` in the repository browser. GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork. Each run downloads the current version of the checks, including the comparison against the supplied readings, from the course checks repository, so a correction made after the assignment was handed out reaches you on your next push. If that download fails, the run says so and falls back to the shape checks in your repository, which do not verify any answer. If a required VS Code control is unavailable, record its message and contact the instructor.
