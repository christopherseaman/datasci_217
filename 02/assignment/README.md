# Assignment 02: Clinic Encounter Summary

## Overview

Document a small Python project, summarize a week of clinic encounters into one report, and choose a follow-up cutoff, saving each result for the checks.

- `data/clinic_encounters.csv` is synthetic: one row per encounter with a patient ID, a visit date, and the **systolic blood pressure** (the top number, in mmHg) recorded at that visit.
- Like any export, it has rows nobody can use. Lecture 02's Demo 3 reads a file of this shape.
- Keep the file exactly as it ships: the checks compare your answers with the ones it gives.

```text
patient_id,visit_date,systolic
P001,2026-03-02,118
P004,2026-03-02,not recorded
```

## Setup

1. Fork the assignment repository on GitHub and clone your fork as in Lecture 01: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open the cloned folder itself.
2. Command Palette → **Git: Create Branch**, and name it `feature/clinic-report`.
3. In the integrated terminal, run `ls data`.
    - Expect: `clinic_encounters.csv`.

## Files

```text
assignment/
├── README.md                    # these instructions; you complete the two TODO lines below
├── .gitignore                   # you complete the Python cache patterns
├── data/
│   └── clinic_encounters.csv    # supplied: one week of encounters; keep it exactly as handed out
├── vitals_tools.py              # scaffold: the calculations you reuse
├── clinic_report.py             # scaffold: reads the data and writes both artifacts
├── CHECKS.md                    # supplied: what each check looks for
├── check_assignment.py          # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── vitals_report.txt        # you generate in Task 2
    └── followup_list.txt        # you generate in Task 3
```

## Task 1: Document the project

### 1.1 Describe the project

Replace the `TODO` line under `## Project description` below with 30-300 characters of your own text saying what this project reads and what it produces.

### 1.2 Say how to run it

Replace the `TODO` line under `## Run` below with a command that runs your report script, such as `python3 clinic_report.py`.

### 1.3 Keep the Python cache out of Git

In `.gitignore`, replace the two `TODO` comments with the patterns for Python's cache folder and compiled files: `__pycache__/` and `*.pyc`.
- Expect: Source Control lists `README.md` and `.gitignore` under **Changes**.

> **Checkpoint: `README.md` and `.gitignore`**
> Stage both files and commit with `Document the clinic report`.

## Project description

TODO: Replace this line with a 30-300 character description of what this project does.

## Run

TODO: Replace this line with the Python 3.13 terminal command that runs your report script.

## Task 2: Summarize the supplied encounters

### 2.1 Decide which rows you can use

A **data row** is every line after the header, blank lines included. The export has one blank line in the middle, which Demo 3's loop reports as `Skipping a blank row.`

A data row is **usable** when all three hold:

1. It splits into exactly three comma-separated fields.
2. `int()` can read the third field.
3. The reading is from 60 to 250 mmHg inclusive; anything outside is a recording error.

Every other data row is skipped, the blank line included. Print one line per skipped row while you develop.

- Expect: each skipped row printed with its reason.

### 2.2 Split the work across the two scripts

1. Complete the calculations in `vitals_tools.py`.
2. Complete the reading, printing, and saving in `clinic_report.py`, importing from `vitals_tools` as Demo 2 and Demo 3 do.
3. Have `read_encounters()` return two values: the usable encounters and the number of skipped rows.
4. Keep `clinic_report.py` safe to import.
    - Expect: `python3 -c "import clinic_report"` prints nothing and writes nothing.

### 2.3 Write the summary

Write these six lines to `output/vitals_report.txt`, in any order:

```text
Usable encounters: <how many data rows were usable>
Skipped rows: <how many data rows were skipped>
Patients seen: <how many different patient IDs appear among the usable encounters>
Mean systolic: <mean of every usable reading> mmHg
Highest systolic: <largest usable reading> mmHg
Lowest systolic: <smallest usable reading> mmHg
```

- `Patients seen` counts different IDs among usable rows, so a repeat visit counts once and a patient whose only row was skipped is not counted.
- `Mean systolic` averages every usable reading, repeat visits included, with at least one decimal place.
- Write each label as shown, followed by a colon and the number. Case, spacing, the `mmHg` unit, and extra lines do not matter.

Read the file back and print it, as Demo 3 does.

> **Checkpoint: `output/vitals_report.txt`**
> Your six labelled lines.

## Task 3: Choose the follow-up cutoff

The clinic can call back a limited number of patients and asks you for the list. Choose a systolic cutoff from 120 to 180 mmHg inclusive: a lower one calls more borderline patients, a higher one only the most urgent.

### 3.1 Record the decision and the list it produces

Write `output/followup_list.txt`:

```text
Cutoff: <the cutoff you chose> mmHg
Reason: <one line, 20-300 characters, saying why you chose it>
<patient id>
<patient id>
...
```

- List every patient with at least one usable reading at or above your cutoff, one ID per line.
    - Expect: order and repeated IDs do not matter; any other line is ignored.
- The checks recompute the list from your declared cutoff, so every cutoff in range is correct when the list matches it.

> **Checkpoint: `output/followup_list.txt`**
> A `Cutoff:` line, a `Reason:` line, then one patient ID per line.

## Check your work

- Run `python3 clinic_report.py`, then `python3 check_assignment.py`.
- The checker prints your score and what to fix, using the latest checks from the course repository: the same checks GitHub runs on each push.
- Commit `vitals_tools.py`, `clinic_report.py`, and the `output/` files, push, then merge `feature/clinic-report` into `main` and push again.

What each check looks for: [CHECKS.md](CHECKS.md)
