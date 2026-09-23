# Assignment 02: Clinic Encounter Summary

## Project description

TODO: Replace this line with a 30-300 character description of what this project does. (This is part of the assignment, read on and it will make more sense.)

## Run

TODO: Replace this line with the Python 3.13 terminal command that runs your report script.

## Files

```text
assignment/
├── README.md                    # these instructions; you complete the two TODO lines above
├── .gitignore                   # you complete the Python cache patterns
├── data/
│   └── clinic_encounters.csv    # supplied: one week of encounters, exactly as the clinic exported them
├── vitals_tools.py              # scaffold: the calculations you reuse
├── clinic_report.py             # scaffold: reads the data and writes both artifacts
├── check_assignment.py          # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── vitals_report.txt        # you generate in Task 2
    └── followup_list.txt        # you generate in Task 3
```

## Setup

Fork the assignment repository on GitHub and clone your fork the way Lecture 01 did: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open it. Then open **Terminal → New Terminal** in VS Code at the assignment directory (Ctrl+Shift+backtick, also Control on Mac). If you use a native terminal or WSL Ubuntu instead, `cd` into the assignment directory first. Run `ls data` and expect `clinic_encounters.csv`. This clone is a new repository, so before your first commit run Lecture 02's two `git config user.name "..."` and `git config user.email "..."` lines in this terminal, with your name and GitHub noreply email.

Open the repository in VS Code, switch to `main`, select **Sync Changes** if Source Control shows it, and finish any outstanding changes. Open the Command Palette, select **Git: Create Branch**, and name the new branch `feature/clinic-report`. Work on that branch until the Submit section.

## Task 1: Document the project

### 1.1 Describe the project

Replace the `TODO` line under `## Project description` at the top of this file with 30-300 characters of your own text saying what this project reads and what it produces.

### 1.2 Say how to run it

Replace the `TODO` line under `## Run` with a Python 3.13 command that runs your report script: `python3 clinic_report.py` in Bash, `py -3.13 clinic_report.py` in native Windows PowerShell, or `python clinic_report.py` in an activated Python 3.13 environment. The bare command, a bullet, a fenced code block, and a sentence such as "Run `python3 clinic_report.py` from this folder." all count.

### 1.3 Keep the Python cache out of Git

Importing `vitals_tools` creates a `__pycache__/` folder of compiled files. Replace the two `TODO` comments in `.gitignore` with the pattern for that folder and the pattern for the compiled files it holds. The standard pair is `__pycache__/` and `*.pyc`; GitHub's own Python template writes the second one as `*.py[codz]`, which is equally good.

> **Checkpoint: `README.md` and `.gitignore`**
> Confirm in Source Control that both files appear under **Changes**, stage them, and commit with `Document the clinic report`.

## Task 2: Summarize the supplied encounters

`data/clinic_encounters.csv` is a week of encounters from one clinic: a patient ID, a visit date, and the systolic blood pressure recorded at that visit, in mmHg. Like any export, it has rows nobody can use. Demo 3 of the [Lecture 02 demo guide](https://github.com/christopherseaman/datasci_217/blob/main/02/demo/DEMO_GUIDE.md) reads a file of this shape. Keep the file exactly as it ships: the checks recompute the answers from it.

### 2.1 Decide which rows you can use

A **data row** is every line after the header, blank lines included. This export has one in the middle, and Demo 3's loop reports it as `Skipping a blank row.` (The newline that ends the last row is not a row of its own; `readlines()` and `splitlines()` already treat it that way.)

A data row is **usable** when all three of these hold:

1. it splits into exactly three comma-separated fields;
2. `int()` can read the third field; and
3. the reading is from 60 to 250 mmHg inclusive. Anything outside that range is a recording error, not a blood pressure.

Every other data row is **skipped**, the blank line included. Print one line per skipped row while you develop, so you can see which rows dropped out and why.

### 2.2 Split the work across the two scripts

Complete the calculations in `vitals_tools.py` and the reading, printing, and saving in `clinic_report.py`, which imports from `vitals_tools` the way Demo 2 and Demo 3 do. `read_encounters()` gives back two values, the usable encounters and the number of skipped rows, because your report needs both. Keep `clinic_report.py` safe to import: `python3 -c "import clinic_report"` should print nothing and write nothing.

### 2.3 Write the summary

Write these six lines to `output/vitals_report.txt`, one per line, in any order:

```text
Usable encounters: <how many data rows were usable>
Skipped rows: <how many data rows were skipped>
Patients seen: <how many different patient IDs appear among the usable encounters>
Mean systolic: <mean of every usable reading> mmHg
Highest systolic: <largest usable reading> mmHg
Lowest systolic: <smallest usable reading> mmHg
```

- Some patients came in twice, so `Patients seen` is not the same as `Usable encounters`. A patient whose only row was skipped was not seen.
- `Mean systolic` averages every usable reading, including a patient's second visit. Give at least one decimal place. Rounding is not a trap: a value within 0.1 mmHg of the mean passes, and so does the mean rounded to however many decimal places you wrote.
- Write each label exactly as shown, followed by a colon and then the number. Around that, the checks are relaxed: letter case and the spaces between words do not matter, the `mmHg` unit is optional (`mm Hg` is fine too), and words around the number are ignored. Extra lines in the file are ignored. What is not optional is the label wording and the colon, so `Usable encounters = 12` or `usable -> 12` does not count.

Read the file back and print it, the way Demo 3 does, so you can see what landed on disk.

> **Checkpoint: `output/vitals_report.txt`**
> Open the saved file in the Explorer and confirm it holds your six labelled lines.

## Task 3: Choose the follow-up cutoff

The clinic can call back a limited number of patients this week, and asks you for the list. You choose the systolic cutoff: any value from 120 to 180 mmHg inclusive. A lower cutoff calls in more patients with borderline readings; a higher one calls only the most urgent.

### 3.1 Record the decision and the list it produces

Write `output/followup_list.txt` in this shape:

```text
Cutoff: <the cutoff you chose> mmHg
Reason: <one line, 20-300 characters, saying why you chose it>
<patient id>
<patient id>
...
```

List the patient ID of every patient with at least one usable reading at or above your cutoff, one per line. Order does not matter, a repeated ID counts once, the `mmHg` unit on the cutoff is optional, and any other line is ignored, whether it is a heading, a blank line, or a row of dashes.

The checks recompute the list from the cutoff you declared, so every cutoff in range is correct, as long as the patients you list are the ones your cutoff selects.

> **Checkpoint: `output/followup_list.txt`**
> Open the saved file and confirm it starts with your `Cutoff:` and `Reason:` lines, followed by one patient ID per line.

## Check your work

Run your script, then the checks, from the assignment directory:

```bash
python3 clinic_report.py
python3 check_assignment.py
```

`check_assignment.py` runs the same checks GitHub runs. They look only at your artifacts: this `README.md`, `.gitignore`, and the two files in `output/`. They also read the supplied `data/clinic_encounters.csv` to work out what your answers should be, and never run or read your Python code, so any way of producing a correct artifact counts.

Each check prints `PASS` or `FIX` and the points it earned, and a `FIX` says what to fix on the line beneath it. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/5  README project description
         README.md: replace the TODO line under `## Project description` with your own description.
```

Fix what it names, rerun your script and the checks, and repeat until every check passes. A clean local run ends with:

```text
[PASS]  5/5  follow-up cutoff
[PASS]  5/5  follow-up reason
[PASS] 15/15 follow-up patient list

Score: 100/100
All checks passed.
```

Every push also runs GitHub Actions, which downloads the course's current copy of the checks and reruns them on the files you committed and pushed. That run is what counts, and a check corrected after handout reaches you there on your next push.

### Completion contract

Commit these files at the assignment repository root. Grading totals 100 points.

| Artifact | Complete when | Points |
|---|---|---:|
| `README.md` | `## Project description` holds 30-300 characters of your own text. | 5 |
| `README.md` | `## Run` holds a Python command that runs a `.py` script; any Python version counts. | 5 |
| `.gitignore` | It lists a standard pattern for Python's bytecode cache, such as `__pycache__/` or `*.py[cod]`. | 5 |
| `output/vitals_report.txt` | UTF-8 text with all six labelled lines present. | 10 |
| `output/vitals_report.txt` | `Usable encounters` matches the supplied encounters. | 8 |
| `output/vitals_report.txt` | `Skipped rows` matches the supplied encounters. | 7 |
| `output/vitals_report.txt` | `Patients seen` matches the distinct patient IDs among the usable encounters. | 10 |
| `output/vitals_report.txt` | `Mean systolic` matches the mean of the usable readings. | 15 |
| `output/vitals_report.txt` | `Highest systolic` matches the largest usable reading. | 5 |
| `output/vitals_report.txt` | `Lowest systolic` matches the smallest usable reading. | 5 |
| `output/followup_list.txt` | `Cutoff` is a number from 120 to 180 mmHg. | 5 |
| `output/followup_list.txt` | `Reason` is 20-300 characters on one line. | 5 |
| `output/followup_list.txt` | The listed patient IDs are exactly the patients with a usable reading at or above your cutoff. | 15 |

Each row is scored on its own, so a right value earns its points whatever else is wrong. The one exception is the patient list: it is checked against your `Cutoff`, so it scores only when that cutoff is from 120 to 180. Extra files and extra lines are ignored.

## Submit

Inspect your changes in VS Code Source Control, then stage `vitals_tools.py`, `clinic_report.py`, `output/vitals_report.txt`, and `output/followup_list.txt` and commit with `Summarize clinic encounters`. Select **Publish Branch** or **Sync Changes**. With no unfinished changes left, switch to `main`, run **Git: Merge...**, and select `feature/clinic-report`. Resolve any conflict, inspect the result, and sync. Confirm in the repository browser that `main` holds both scripts and both files under `output/`.

GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork. If your local run and the GitHub run ever disagree, the GitHub run counts, because it uses the course's current checks.
