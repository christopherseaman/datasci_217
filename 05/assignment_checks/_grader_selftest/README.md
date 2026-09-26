# Assignment 05 grading development self-test

Course-side QA for the midterm checks, not a second grading mode. `run.py` answers the midterm with pandas and NumPy from `05/assignment/data/people_raw.csv` the way the handout README asks, builds submissions in ignored `scratch/`, and confirms that:

- every constant in `grading.py` matches the independent answer, and the README's issue labels and decision table match the ones graded;
- the handout README lists all seven output files in its file tree and in its checklist, each with the first line and line count the answer produces, and prints none of the expected values;
- the handout ships no checks (no `check_assignment.py`, `grading.py`, `.github/`, or self-test), marks `data/people_raw.csv` `-text` in `.gitattributes`, and has a notebook with cleared outputs and an empty `output/`;
- an empty directory and the untouched handout score 0 of 75, a correct submission scores 75, and so does one written with CRLF, trailing spaces, no final newline, a byte-order mark, reordered columns, index columns, other letter case, `6.0` for `6`, `yes`/`no` flags, `<NA>` and `NaN` for missing, and dates with a time;
- no submitted file runs, even when named like the checks or a standard-library module;
- NumPy scalar reprs such as `np.int64(12)`, a Series-style unnamed metric column, `YYYY/MM/DD` dates, CR-only line endings, cells separated by semicolons (with a decimal comma) or tabs, and a space after, or padding around, every separator cost nothing, while a name left unstripped inside such padding still costs its one record, a cleaned table without `record_id` loses only that column's 4 points, and a file no CSV reader accepts loses only its own checks;
- each single mistake, from a blank age to a dropped column, costs exactly its own points, and its feedback names the fix.

```bash
uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 python 05/assignment_checks/_grader_selftest/run.py
```
