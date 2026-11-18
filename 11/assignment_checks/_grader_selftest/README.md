# Assignment 11 grading development self-test

Course-side QA for the final exam's checks, not a second grading mode. `run.py` answers the exam with pandas, NumPy, and scikit-learn from `11/assignment/data/` the way `assignment.md` asks, builds submissions in ignored `scratch/`, and confirms that:

- the handout ships no checks (no `check_assignment.py`, `grading.py`, `.github/`, `.badmath.toml`, or self-test), and no Python file in the handout or the checks reads the Python version;
- the handout README's question table, CHECKS.md's artifact checks and human-review tables, each notebook's points line, the Q9 category table, and `assignment.md` state the points the checks give (75) and the rubric's five 5-point categories (25);
- an empty directory and the untouched handout score 0 of 75, and the independent answer scores 75, even with files named `grading.py` and `pandas.py` in the submission, which never run;
- a copy of that answer rewritten with reversed columns, shuffled rows, upper-case labels, CRLF line endings, a byte-order mark, spaces around header names and after lines, no final newline, index columns, and `yes`/`no` flags still scores 75;
- so does a copy using the alternatives the handout allows: the `column_names` audit row as a printed list, naive local times, `Z` suffixes, `T` separators, rounded values, `missing_pct` as a fraction, plain-English audit results, an unlabeled correlation index, a Python-dict `parameters_json`, comma-separated `feature_columns`, the regressor's name in place of `student_model`, four files separated by semicolons (with decimal commas) or tabs, and the release file name written as an upper-case path; a wrong value in a semicolon-separated file costs only its own check;
- equivalent repeated headers are accepted, including numeric, timestamp, boolean, and missing-value spellings; contradictory duplicate values lose their affected part, and decoded NUL bytes are rejected instead of silently truncated;
- 24 single variations accept an equivalent weekday numbering and charge actual mistakes, from a kept ambiguous row to a flipped error sign, only to their own checks; every lost point comes with feedback;
- a mistake carried downstream is charged once, where it was made: a nonfinite model prediction does not also cost its unrecomputable metrics, a Q2 mistake carried into the Q3 panel, a kept ambiguous row the audit also records, a panel without its unobserved hours carried into Q4 (with feedback naming the join), and a split on cutoff time carried from X into y (with feedback naming the cause);
- a missing test predictions file, or validation predictions overwritten with the test rows, costs only that file's checks (the feedback names the test split), while a metrics file without its predictions still loses points for a model row count that disagrees with the baseline's;
- one mistake repeated in later files is charged in the first file that shows it, and the later checks say "Same cause" and keep their points: the MSE saved as `rmse` in the three metrics files costs 1 point, in the first of them, and UTC-midnight boundaries in Q5 and Q6 cost 1 point, in the Q5 monthly summary, with local or UTC months there; a second mistake beside either still costs, and UTC-midnight boundaries in Q6 alone cost nothing, with the X rows check naming the boundary and the fix;
- squared `rmse` values get feedback naming the MSE, and 2023-12 counts that are short get feedback naming the last training month, while counts that are too high, empty, or text do not;
- each notebook Checkpoint and `assignment.md` name every output file with its first line (or say it is a PNG image) and line count, and no handout page mentions a checker, GitHub Actions, or automated feedback.

```bash
uv run --isolated --project 11/assignment --locked python3 11/assignment_checks/_grader_selftest/run.py
```

It runs the checker 45 times, so it takes about ten minutes.
