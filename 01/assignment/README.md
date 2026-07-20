# Assignment 01: Terminal and Python Readiness

This assignment checks the Lecture 01 skills you will use throughout the course: working from the intended directory, running terminal Python scripts, using values and a list, writing one decision and one loop, and correcting beginner errors from tracebacks.

Complete the work in a POSIX-style terminal with Python 3.12. Do not use a notebook or Google Colab. Use `python3` instead of `python` only if that is the command established during onboarding.

The repository-delivery steps are in [`PLATFORM_CHECK.md`](PLATFORM_CHECK.md). They are required so your work reaches the course system, but Git concepts are not part of this assignment's Python result. Lecture 02 explains the Git model and workflow.

## Start in the assignment directory

Open this repository as a folder in VS Code. Open **Terminal → New Terminal**, then use `pwd` and `ls` to confirm that the terminal is in the repository folder containing this `README.md`.

The starter package contains:

- `readiness.py`: Question 1 starter;
- `measurement_summary.py`: Question 2 starter;
- `debug_report.py`: Question 3 starter with exactly three prepared errors;
- `make_output.py`: supplied output-file wrapper; do not edit it;
- `check_assignment.py`: supplied local checker; do not edit it;
- `test_assignment.py`: the public Classroom 50/pytest contract; do not edit it.

The supplied checker and wrapper use later Python features internally. You only run those files. Their implementation is not assessed and is not a model for your Lecture 01 code.

## Question 1: Paths and readiness

### A. Leave terminal-practice evidence

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

The final `terminal-practice` directory must contain `source.txt` and `path-check.txt`, but not `remove-me.txt`. These named-file operations provide path-practice evidence without recursive removal.

### B. Complete `readiness.py`

Do not edit the supplied block at the top of `readiness.py`. It obtains three values for you:

- the current Python major/minor family;
- the supplied project label;
- the current script filename.

Replace the three `TODO` output lines so the script prints exactly these labels and values when run as `readiness.py`:

```text
Python family: 3.12
Project: DataSci 217 Assignment 01
Script: readiness.py
```

Use the supplied variable names, not repeated or hard-coded values. Run the script:

```bash
python readiness.py
```

## Question 2: Summarize supplied measurements

Complete `measurement_summary.py`. Keep the supplied `measurements` list and `review_threshold_text` while developing your answer.

Your script must:

1. convert `review_threshold_text` to an integer named `review_threshold`;
2. select and print the first measurement with the zero-based index `measurements[0]`;
3. start `total` and `review_count` at zero;
4. use one direct `for` loop written as `for measurement in measurements:` to visit every value;
5. add each value to `total`;
6. use `if` and `else` so a value at or above `review_threshold` is labeled `review`, while a lower value is labeled `within range`;
7. add one to `review_count` only for a value labeled `review`;
8. print one labeled line per measurement from inside the loop;
9. after the loop, calculate the mean using the actual list length; and
10. print the exact summary labels shown below, with the mean rounded to one decimal place.

For the supplied data, the output must be:

```text
First measurement: 18
Measurement 18: within range
Measurement 21: review
Measurement 24: review
Measurement 19: within range
Count: 4
Total: 82
Mean: 20.5
Review count: 2
```

Run it with:

```bash
python measurement_summary.py
```

The grader also runs a temporary copy with different top-level `measurements` and `review_threshold_text` values. Do not assume there are always four measurements, and do not print a prepared answer.

## Question 3: Read, fix, rerun, and make the output file

`debug_report.py` contains exactly three prepared errors. Run it, read the final traceback line and referenced source line, make one small correction, save, and rerun. Repeat until it exits successfully.

The three corrections use only Lecture 01 ideas. Do not replace the program with prepared output. The clean result is:

```text
Readiness: complete
Participant count: 4
Next checkpoint: 5
```

The grader also runs a temporary copy with another supplied participant count. Keep the calculation and printed variables rather than replacing them with the displayed answers.

After all three student scripts run cleanly, use the supplied wrapper:

```bash
python make_output.py
```

It freshly runs all three scripts and writes their combined printed output to `output/readiness.txt`. You are not expected to use `open()`, imports, or Python file-writing code yourself.

## Check your work

Run the dependency-free public checker from the assignment directory:

```bash
python check_assignment.py
```

It prints one actionable result for each public requirement. A complete submission ends with:

```text
All public checks passed.
```

If a check fails, fix the student file named in the message, rerun that script, rerun `python make_output.py` when requested, and then run the checker again. You do not need to install pytest locally.

Classroom 50 may run `test_assignment.py` with pytest as part of grading. Those public tests use the same checks as `check_assignment.py`. Additional production tests, if any, remain in the centrally managed grader bundle rather than the student starter repository, but they enforce this same written contract.

## Completion contract

This is a competence-focused pass/fail assignment. A passing Python result requires all public behavior, structure, terminal-evidence, and fresh-output checks to pass. The exact GUI synchronization checklist is required for delivery but is unassessed and has no Git-concept rubric.

Do not add notebooks, third-party packages, shell pipes, or shell redirection. In the three student scripts, do not add:

- student-defined functions, async functions, classes, or `lambda`;
- dictionaries, sets, comprehensions, or generator expressions;
- imports beyond the two supplied imports in `readiness.py`;
- `open()`, `with`, `.read*()`/`.write*()` methods, or other file I/O;
- `while`, `break`, or `continue` (Question 2 requires the direct `for` loop);
- `try`/`except` (including `except*`), `raise`, or `async with`;
- `match` or the `:=` assignment expression;
- dynamic-code calls such as `exec()`, `eval()`, `compile()`, or `__import__()`; or
- `sum()` in place of the required loop.

The only function calls needed in student-authored lines are `print()`, `type()`, `int()`, `float()`, `len()`, and `range()`. Keep the supplied `Path()` call in `readiness.py` unchanged. Indirect calls, dynamically selected calls, and other method calls are outside the Lecture 01 boundary.

These are scope boundaries, not stylistic traps: Lecture 01 has not introduced those constructs, and none is needed for the written contract.
