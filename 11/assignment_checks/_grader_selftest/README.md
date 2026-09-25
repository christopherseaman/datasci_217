# Assignment 11 grading development self-test

Course-side QA for the final exam's checks, not a second grading mode. `run.py` answers the exam with pandas, NumPy, and scikit-learn from `11/assignment/data/` the way `assignment.md` asks, builds submissions in ignored `scratch/`, and confirms that:

- the handout ships no checks (no `check_assignment.py`, `grading.py`, `.github/`, `.badmath.toml`, or self-test), and no Python file in the handout or the checks reads the Python version;
- an empty directory and the untouched handout score 0 of 85, and the independent answer scores 85, even with files named `grading.py` and `pandas.py` in the submission, which never run;
- a copy of that answer rewritten with reversed columns, shuffled rows, upper-case labels, CRLF line endings, a byte-order mark, spaces around header names and after lines, no final newline, index columns, and `yes`/`no` flags still scores 85;
- so does a copy using the alternatives the handout allows: the `column_names` audit row as a printed list, naive local times, `Z` suffixes, `T` separators, rounded values, `missing_pct` as a fraction, plain-English audit results, an unlabeled correlation index, a Python-dict `parameters_json`, comma-separated `feature_columns`, the regressor's name in place of `student_model`, four files separated by semicolons (with decimal commas) or tabs, and the release file name written as an upper-case path; a wrong value in a semicolon-separated file costs only its own check;
- each of 20 single mistakes, from a kept ambiguous row to a flipped error sign, costs only its own check, and every lost point comes with feedback;
- a Q2 mistake carried into the Q3 panel is charged once, in Q2, and a missing test predictions file costs only its own checks, while a metrics file without it still loses points for a model row count that disagrees with the baseline's;
- the handout README names every output file with its first line and mentions no checker, GitHub Actions, or automated feedback.

```bash
uv run --python 3.13 --with-requirements 11/assignment/requirements.txt python 11/assignment_checks/_grader_selftest/run.py
```

It runs the checker about 25 times, so it takes about six minutes.
