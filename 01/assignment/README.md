# Assignment 01: Terminal and Python Readiness

## Overview

Practice terminal file operations, finish two Python scripts, fix three prepared errors, and generate two output files for the checks. The files in `terminal-practice/` and `output/` are what is graded; your code is never run or read.

## Setup

1. Fork the assignment repository on GitHub and clone your fork as in Lecture 01: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open the cloned folder.
2. Open **Terminal → New Terminal** (Ctrl+Shift+backtick, also Control on Mac).
    - Expect: `pwd` ends in the assignment folder, and `ls` lists this `README.md`.
3. Check the Python version.
    - Expect: `python3 --version` prints `Python 3.13` and a patch number.

## Files

```text
assignment/
├── README.md                     # these instructions
├── CHECKS.md                     # supplied: what each check looks for
├── readiness.py                  # scaffold: you complete it in Task 1.2
├── measurement_summary.py        # scaffold: you complete it in Task 2.1
├── debug_report.py               # scaffold: three prepared errors you fix in Task 3.1
├── make_output.py                # supplied: saves the readiness report in Task 3.2; keep unchanged
├── capture_identity.py           # supplied: saves your identity hash in Task 3.3; keep unchanged
├── process_email.py              # supplied: used by capture_identity.py; keep unchanged
├── check_assignment.py           # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
├── media/                        # supplied: screenshots for the Lecture 01 walkthrough
├── terminal-practice/            # you create in Task 1.1
│   ├── source.txt
│   └── path-check.txt
└── output/
    ├── readiness.txt             # you generate in Task 3.2
    └── student_identity.txt      # you generate in Task 3.3
```

## Task 1: Paths and readiness

### 1.1 Practice file operations

Run these commands from the directory containing this `README.md`:

```bash
pwd
mkdir terminal-practice
touch terminal-practice/source.txt
cp terminal-practice/source.txt terminal-practice/source-copy.txt
mv terminal-practice/source-copy.txt terminal-practice/path-check.txt
touch terminal-practice/remove-me.txt
pwd
ls terminal-practice/remove-me.txt
rm terminal-practice/remove-me.txt
ls terminal-practice
```

> **Checkpoint: `terminal-practice/source.txt` and `terminal-practice/path-check.txt`**
> Both empty files should now exist, and the last `ls` lists only these two. Commit them with your completed work.

### 1.2 Complete `readiness.py`

Do not edit the supplied block at the top of `readiness.py`. It obtains three values for you:

- the current Python major/minor family;
- the supplied project label;
- the current script filename.

Replace the three `TODO` output lines so the script prints exactly these labels and values when run as `readiness.py`:

```text
Python family: 3.13
Project: DataSci 217 Assignment 01
Script: readiness.py
```

Print the supplied variable that belongs on each line. Run the script:

```bash
python3 readiness.py
```

## Task 2: Summarize supplied measurements

### 2.1 Calculate the summary

Complete `measurement_summary.py`. Keep the supplied `measurements` list and `review_threshold_text` while developing your answer. `measurements` is a list, as in Lecture 01: `len()` counts its items, and a `for` loop visits each one in order.

Build the script in these steps:

1. convert `review_threshold_text` to an integer named `review_threshold`;
2. start `total` and `review_count` at zero;
3. use one direct `for` loop written as `for measurement in measurements:` to visit every value;
4. add each value to `total`;
5. use `if` and `else` so a value at or above `review_threshold` is labeled `review`, while a lower value is labeled `within range`;
6. add one to `review_count` only for a value labeled `review`;
7. print one labeled line per measurement from inside the loop;
8. after the loop, calculate the mean using the actual list length; and
9. print the summary labels shown below using `print()` with comma-separated values. The supplied data gives a mean of `20.5`; no rounding or text formatting is needed.

### 2.2 Run and compare

For the supplied data, the output must be:

```text
Measurement: 18 within range
Measurement: 21 review
Measurement: 24 review
Measurement: 19 within range
Count: 4
Total: 82
Mean: 20.5
Review count: 2
```

Run it with:

```bash
python3 measurement_summary.py
```

Try another list or threshold to test your calculations, then restore `[18, 21, 24, 19]` and `"20"` before generating the report.

## Task 3: Read, fix, rerun, and make the output file

### 3.1 Correct `debug_report.py`

`debug_report.py` contains exactly three prepared errors. Run it with `python3 debug_report.py`, read the last line of the error message and the source line it points to, make one small correction, save, and rerun. Repeat until it exits successfully.

The first error is an `IndentationError`. Python finds it before running anything, so it has no `Traceback (most recent call last)` header and nothing prints. The other two appear only when Python reaches the bad line, after the lines above it have printed.

The corrected script prints:

```text
Readiness: complete
Participant count: 4
Next checkpoint: 5
```

Test another participant count if helpful, then restore `participant_count_text = "4"` before saving the report.

### 3.2 Generate the readiness report

After all three student scripts run cleanly, use the supplied wrapper:

```bash
python3 make_output.py
```

> **Checkpoint: `output/readiness.txt`**
> The helper runs the three scripts and saves their combined output: the 3 lines from Task 1.2, 8 from Task 2.2, and 3 from Task 3.1, in that order. Open the file and check all 14 lines.

### 3.3 Generate your identity hash

Then run the supplied identity helper and enter the email address used on the course roster at its prompt:

```bash
python3 capture_identity.py
```

The helper trims whitespace, lowercases the address, requires `@ucsf.edu`, and hashes the username after removing punctuation. It saves only the hash; keep your email address out of files and commits.

> **Checkpoint: `output/student_identity.txt`**
> The file contains one 64-character SHA-256 hash. The checks match it to the course roster. If they report no match, rerun the helper with your roster email or contact the course team.

## Check your work

1. Run the checker from the assignment folder. It uses the latest checks from the course repository, the same checks GitHub runs.

    ```bash
    python3 check_assignment.py
    ```

    - Expect: a `Checks:` line naming which copy ran, one `PASS` or `FIX` line per check, then `Score: 100/100` and `All checks passed.`
2. Fix what `Left to fix` names, rerun `python3 make_output.py` if a report line changed, and check again.
3. Commit the three scripts, `terminal-practice/`, and `output/` (Source Control: stage with **+**, commit, **Sync Changes**). Your fork is the submission; no pull request is needed. GitHub Actions runs the checks on every push; in a new fork, enable Actions once if prompted.
    - Expect: the files appear on GitHub, and the Actions run, which is the one that counts, shows the same score.
    - If the commit or push stops and asks who you are, set your Git identity as in Lecture 01, then commit again:

    ```bash
    git config user.name "Your Name"
    git config user.email "YOUR GITHUB NOREPLY EMAIL"
    ```

What each check looks for: [CHECKS.md](CHECKS.md)
