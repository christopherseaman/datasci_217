# Assignment 01: Terminal and Python Readiness

## Setup

Fork and clone the assignment using the [Lecture 01 instructions](https://app.notion.com/p/271d9fdd1a1a805784e1fe68dc985696). Open your clone in VS Code, then **Terminal → New Terminal** (Ctrl+Shift+backtick, also Control on Mac). Use `pwd` and `ls` to confirm that you are in the folder containing this `README.md`. Use Python 3.13; the commands below use `python3`, as in Lecture 01.

```text
assignment/
├── readiness.py               # Q1 scaffold
├── measurement_summary.py     # Q2 scaffold
├── debug_report.py            # Q3 scaffold: three prepared errors
├── make_output.py             # supplied report helper
├── capture_identity.py        # supplied identity helper
├── check_assignment.py        # run to check your artifacts
├── terminal-practice/         # create in Task 1.1
│   ├── source.txt
│   └── path-check.txt
└── output/                    # generated in Tasks 3.2–3.3
    ├── readiness.txt
    └── student_identity.txt
```

If using a separate terminal app, open Terminal on macOS/Linux or Ubuntu on Windows and `cd` to your cloned assignment folder before running the commands.

## Question 1: Paths and readiness

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

> **Checkpoint — `terminal-practice/source.txt` and `terminal-practice/path-check.txt`**
> Both empty files should now exist. Commit them with your completed work.

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

Use the supplied variable names, not repeated or hard-coded values. Run the script:

```bash
python3 readiness.py
```

## Question 2: Summarize supplied measurements

### 2.1 Calculate the summary

Complete `measurement_summary.py`. Keep the supplied `measurements` list and `review_threshold_text` while developing your answer.

Your script must:

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

Here, the list holds values for your loop. Lecture 02 covers lists, indexing, and slicing in more detail; Lecture 03 introduces NumPy arrays.

## Question 3: Read, fix, rerun, and make the output file

### 3.1 Correct `debug_report.py`

`debug_report.py` contains exactly three prepared errors. Run it, read the final traceback line and referenced source line, make one small correction, save, and rerun. Repeat until it exits successfully.

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

> **Checkpoint — `output/readiness.txt`**
> The helper runs the three scripts and saves their combined output: the 3 lines from Task 1.2, 8 from Task 2.2, and 3 from Task 3.1, in that order. Open the file and check all 14 lines.

### 3.3 Generate your identity hash

Then run the supplied identity helper and enter the email address used on the course roster at its prompt:

```bash
python3 capture_identity.py
```

The helper trims whitespace, lowercases the address, requires `@ucsf.edu`, and hashes the username after removing punctuation. It saves only the hash; keep your email address out of files and commits.

> **Checkpoint — `output/student_identity.txt`**
> The file contains one 64-character SHA-256 hash. The checker must match it to the course roster. If it does not, rerun the helper with your roster email or contact the course team.

## Check Your Work

Run the checker from the assignment directory:

```bash
python3 check_assignment.py
```

It reports a result for each artifact group. A complete submission ends with:

```text
Score: 100/100
All checks passed.
```

If the report check fails, inspect your scripts, rerun `python3 make_output.py`, and check again.

GitHub Actions runs the same checks automatically on every push. In a new fork, open **Actions** and enable workflows once if GitHub prompts you. Open the latest run to see which artifacts need attention.

### Completion Contract

| Points | Artifacts | What is checked |
| --- | --- | --- |
| 20 | `terminal-practice/source.txt`, `terminal-practice/path-check.txt` | Both exist as regular files in a regular directory; contents are not checked. |
| 80 | `output/readiness.txt`, `output/student_identity.txt` | Both are regular UTF-8 files in a regular `output` directory. The report matches all 14 expected lines, including spacing and a final newline. The identity file contains one roster hash; surrounding whitespace and hex-letter case are ignored. Both artifacts must pass for these points. |

Extra files are ignored. The checker reads saved artifacts, not your source code or how you produced the results.

## Submit

Commit your three completed scripts and the four checkpoint files to **your fork**. Follow the [Lecture 01 submission walkthrough](https://app.notion.com/p/271d9fdd1a1a805784e1fe68dc985696) for VS Code commit/sync or GitHub web upload. On GitHub, open both files under `output/` and confirm their contents. Your fork is the submission; no pull request is needed.
