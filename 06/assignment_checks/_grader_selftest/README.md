# Assignment 06 grading development self-test

Course-side QA for the midterm checks, not a second grading mode. `run.py` answers the midterm with pandas from `06/assignment/data/` the way the handout README asks, builds submissions in ignored `scratch/`, and confirms that:

- every constant in `grading.py` matches the independent answer, `data/manifest.json` matches the data files, and the README's issue labels, merge-check labels, and decision table match the ones graded;
- the handout README lists all eleven output files in its file tree and in its checklist, each with the first line and line count the answer produces, and prints none of the expected values;
- the handout ships no checks (no `check_assignment.py`, `grading.py`, `.github/`, or self-test), marks `data/*.csv` `-text` in `.gitattributes`, has a notebook with cleared outputs and an empty `output/`, and no checker or handout file reads the Python version;
- an empty directory and the untouched handout score 0 of 75, Task 1's two files alone score 10, a correct submission scores 75, and so does one written with CRLF or CR-only line endings, a byte-order mark, trailing spaces, semicolons or tabs, a space after every comma, reversed columns and rows, index columns, codes and IDs in lower case, `6.0` for `6`, `<NA>` and `NaN` for missing, dates with a time or slashes, `data/` paths, and melt rows kept for blank results;
- no submitted file runs, even when named like the checks or a standard-library module;
- each of 28 single mistakes, from an unmapped gender, a gender left as text, or a kept recheck to a flipped `sbp_change`, costs exactly its own points, a mistake carried into a later file is charged once, and the feedback names the fix.

```bash
uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 python 06/assignment_checks/_grader_selftest/run.py
```
