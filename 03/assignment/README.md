# Assignment 03: Telemetry Ward Analysis with NumPy

## Files

```text
assignment/
├── data/bp_readings.csv    # supplied readings; keep this file exactly as handed out
├── analysis.py             # starter script: the CSV loader is written, the analysis is yours
├── requirements.txt        # supplied: the one direct dependency this project installs
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _public_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
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

## Setup

Fork the assignment repository on GitHub and clone your fork the way Lecture 01 did: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open it. Then open **Terminal → New Terminal** in VS Code at the assignment directory (Ctrl+Shift+backtick, also Control on Mac). If you use a native terminal or WSL Ubuntu instead, `cd` into the assignment directory first. Run `ls data` and expect `bp_readings.csv`. This clone is a new repository, so before your first commit run Lecture 02's two `git config user.name "..."` and `git config user.email "..."` lines in this terminal, with your name and GitHub noreply email.

Switch to `main`, select **Sync Changes** if Source Control shows it, and use **Git: Create Branch** to create `feature/numpy-analysis`. Work on that branch until the Submit section.

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

> **Checkpoint: `requirements.txt`**
> Still pins numpy as `numpy==<version>`.

### 1.2 Save an environment probe

With the environment active, save three labelled lines to `output/environment.txt`:

```text
python: <the version the active interpreter reports>
numpy: <the version of numpy installed in it>
interpreter: <the path to the active interpreter>
```

Each line is a label, a colon, and the value. Command substitution from the lecture's "Shell Variables and Timestamps" card builds a labelled line from a command's output, and `>` starts the file while `>>` adds to it:

```bash
echo "python: $(python --version)" > output/environment.txt
```

Lecture 03 gives the one-line Python commands that print the installed NumPy version and the interpreter path. Quotes inside `$( )` belong to the command inside it, so a `python -c "..."` command goes inside `echo "numpy: $(...)"` unchanged.

> **Checkpoint: `output/environment.txt`**
> Three lines. With the environment active, the `numpy` line shows the version `requirements.txt` pins and the `interpreter` line shows a path inside your project's `.venv`. The check asks only for a version number on the `numpy` line and a path on the `interpreter` line; the `python` line is not graded.

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

Capture the timestamp once into a shell variable and use it in the filename; the lecture's "Shell Variables and Timestamps" reference card gives the `date` format string that produces `YYYYMMDD_HHMMSS`. Demo 1 ends its pipeline with `| head -n 5` to keep its display short, but this file has six monitors, so leave that stage off or the last monitor goes missing.

> **Checkpoint: `output/monitor_counts_<timestamp>.txt`**
> One line per monitor with that monitor's count and its id, as `uniq -c` prints them. Spacing, separators such as `M01: 58`, and line order do not matter, and earlier timestamped runs may sit beside it.

## Task 3: Answer the ward's questions with NumPy

Blood pressure on this unit peaks at some point in the monitored shift, a share of patients average into stage 2 hypertension, and the staff suspect one bedside monitor reads high. `analysis.py` already loads the CSV into arrays; answer the questions below from those arrays and save the answers to `output/vitals_summary.txt`.

Two definitions the questions use:

- A patient's **12-hour mean** is the mean of that patient's twelve readings: one number per patient, which is `readings.mean(axis=1)`.
- A monitor's **average** is the mean of the 12-hour means of the patients it recorded. Every patient has twelve readings, so that is the same number as the mean of all of that monitor's readings.

To group patients by their monitor, use the lecture's "Select One Group by a Label" snippet: comparing a text array with one label, as in `monitors == "M01"`, builds a Boolean mask, and indexing another array with that mask keeps the values belonging to that group. The mask and the values it selects have to be the same length, so group an array holding one value per patient, the 12-hour means, with `monitors`, which also holds one value per patient. To name the monitor with the highest average, collect each monitor's average in a list in the order of `sorted(set(monitors))`, turn that list into an array with `np.array()`, and use its `argmax()` position to pick the name from the sorted list, as the lecture's "Find the Highest Values and Who Has Them" snippet does with `ids[avg_glucose.argmax()]`. Demo 3.4 names its highest-average clinic this way.

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
| `monitor_offset` | How far above the average of the patients on the _other_ monitors does that monitor's average sit? | mmHg | 3 |
| `stage2_other_monitors` | Leaving out the patients on that monitor, how many of the rest have a 12-hour mean of 140 mmHg or higher? | Whole number | 3 |

How the values are read:

- Each answer is scored on its own, so a wrong value costs only its own points.
- mmHg values are accepted within 0.6 of the value recomputed from the data, so one decimal or every digit NumPy prints passes, and so does a whole number, whether rounded with `:.0f` or cut short with `int()`. A trailing unit such as `mmHg` is ignored. Counts and whole-number readings must match exactly.
- A NumPy scalar printed as `np.float64(121.5)` or `np.int64(96)` reads as the number inside it.
- Either the population or the sample standard deviation is accepted; at this many readings they agree far inside the tolerance.
- "140 mmHg or higher" includes a mean of exactly 140.
- A patient id, column name, or monitor id may sit in quotes or brackets, as a one-item list prints it (`['M02']`), and may carry a note before or after it, as in `monitor M02` or `M02 (128.4 mmHg)`, as long as the line names no other id of the same kind.
- Keys may appear in any order, spacing is free, and extra lines are ignored. When a key appears on more than one line, the first is read, so open the file with `"w"`, which replaces it on each run, rather than `"a"`.

> **Checkpoint: `output/vitals_summary.txt`**
> One `key: value` line for each of the 14 keys above, holding the answers your analysis computed from `data/bp_readings.csv`.

## Check your work

With the environment active, run your script and then the checks, from the assignment directory:

```bash
python analysis.py
python check_assignment.py
```

`check_assignment.py` runs the same checks GitHub runs. They read only your files in `output/` and recompute every answer from the supplied `data/bp_readings.csv`. They never run or read your Python code, so any way of producing a correct artifact counts.

Each check prints `PASS` or `FIX` and the points it earned, and a `FIX` says what to fix on the line beneath it. Before Task 1, for example, the first check reports:

```text
[FIX ]   0/13  environment probe
         output/environment.txt is missing; commit it as a regular file.
```

Fix what it names, rerun whatever produces that artifact and then the checks, and repeat until every check passes. A clean local run ends with:

```text
[PASS]   4/4   answer: high_monitor
[PASS]   3/3   answer: monitor_offset
[PASS]   3/3   answer: stage2_other_monitors

Score: 100/100
All checks passed.
```

Every push also runs GitHub Actions, which downloads the course's current copy of the checks and reruns them on the files you committed and pushed. That run is what counts, and a check corrected after handout reaches you there on your next push.

### Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/environment.txt` | Its `numpy` line holds a version number and its `interpreter` line is not empty. Which version and which interpreter are not graded, and neither is the `python` line. | environment probe | 13 |
| `output/record_count.txt` | It holds the number of patient records in the supplied CSV. | record count artifact | 10 |
| `output/monitor_counts_<timestamp>.txt` | A timestamped file holds every monitor's patient count. | monitor counts artifact | 15 |
| `output/vitals_summary.txt` | It has a readable `key: value` line for at least one key in Task 3's table. A missing or unreadable key costs only its own answer check. | summary artifact format | 12 |
| `output/vitals_summary.txt` | Each of the 14 answers matches the supplied readings. | one check per key, named `answer: <key>` | 50 |

Extra files and extra lines are ignored.

## Submit

In VS Code Source Control, stage `.python-version`, `analysis.py`, and everything in `output/`, including every timestamped counts file you kept. Commit with `Analyze telemetry ward readings`. Keep `.venv/` out of the commit; `.gitignore` already lists it.

Publish or sync the branch. With no unfinished changes, switch to `main`, run **Git: Merge...**, and select `feature/numpy-analysis`. Resolve any unexpected conflict, inspect the resolution, and sync. Confirm the committed artifacts on `main` in the repository browser. GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork. If a run cannot download the course's current checks, it grades with the copy in your repository and says so in its log. If your local run and the GitHub run ever disagree, the GitHub run counts, because it uses the course's current checks. If a required VS Code control is unavailable, record its message and contact the instructor.
