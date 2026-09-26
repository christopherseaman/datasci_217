# Assignment 01: Terminal and Python Readiness

## Files

```text
assignment/
├── README.md                     # these instructions
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

## Setup

Fork and clone the assignment using the [Lecture 01 instructions](https://app.notion.com/p/271d9fdd1a1a805784e1fe68dc985696). Open your clone in VS Code, then **Terminal → New Terminal** (Ctrl+Shift+backtick, also Control on Mac). Use `pwd` and `ls` to confirm that you are in the folder containing this `README.md`.

Use Python 3.13: `python3 --version` should print `Python 3.13` and a patch number. The commands below use `python3`, as in Lecture 01.

If using a separate terminal app, open Terminal on macOS/Linux or Ubuntu on Windows and `cd` to your cloned assignment folder before running the commands.

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

Run the checks from the assignment directory:

```bash
python3 check_assignment.py
```

These are the same checks GitHub runs. They look only at the four files you commit in `terminal-practice/` and `output/`, and never run or read your Python code. Whitespace and blank lines are ignored, and so is the report's first line, your Python version.

Each check prints `PASS` or `FIX` and the points it earned, and a `FIX` says what to fix on the line beneath it. When the next checks need the same fix, such as a missing file, they say `(same fix as above)`. A report line that differs shows what it should read, what yours reads, and which script prints it.

Below the score, `Left to fix` lists the checks still failing and the points they are worth, as in `Left to fix (5 points): output/readiness.txt: Total.` Fix what they name, rerun `python3 make_output.py` when a report line changed, and check again until every check passes. A clean local run ends with:

```text
[PASS] 15/15 identity hash on the roster

Score: 100/100
All checks passed.
```

GitHub Actions reruns these checks on every push with the course's current copy of them, and that run is what counts. A check corrected after you forked reaches you on your next push. If a run cannot download the course's copy, it uses the copy in your repository and says so in its log.

### Completion contract

Commit these files in your fork. Grading totals 100 points, and each check is scored on its own.

| Artifact | Complete when | Points |
|---|---|---:|
| `terminal-practice/source.txt` and `terminal-practice/path-check.txt` | Each exists as a regular file in a regular `terminal-practice` directory. Their contents are not checked. | 20 (10 each) |
| `output/readiness.txt` | A UTF-8 text file in a regular `output` directory holding the 13 lines this README shows after the Python version in Tasks 1.2, 2.2, and 3.1, in the same order. Whitespace, blank lines, and extra lines are ignored, and the Python version line is not checked. | 65 (5 each) |
| `output/student_identity.txt` | A regular file in a regular `output` directory holding one hash from the course roster; surrounding whitespace and letter case are ignored. | 15 |

A wrong or missing report line costs only its own 5 points, and the identity hash is checked separately from the report. Extra files are ignored, but keep the supplied ones, because `capture_identity.py` needs `process_email.py` and the checks need their own files.

## Submit

Commit your three completed scripts and the four checkpoint files to **your fork**: in VS Code Source Control, stage each file with **+**, commit, and select **Sync Changes**. Follow the [Lecture 01 submission walkthrough](https://app.notion.com/p/271d9fdd1a1a805784e1fe68dc985696) for VS Code or GitHub web upload. On GitHub, open both files under `output/` and confirm their contents. Your fork is the submission; no pull request is needed.

GitHub Actions runs the checks automatically on every push; in a new fork, open **Actions** and enable workflows once if GitHub prompts you. Open the latest run to see which artifacts need attention. If your local run and the GitHub run ever disagree, the GitHub run counts, because it uses the course's current copy of the checks.
