# Assignment 03 grading development self-test

Course-side QA for both halves of the checks, not a second grading mode. It
answers the assignment with NumPy from `03/assignment/data/bp_readings.csv`,
builds submissions in ignored `scratch/`, and confirms that:

- the value checks in this directory score a correct submission 100, accept
  loose formatting, rounding, a BOM, CRLF, NumPy scalar reprs and the sample
  standard deviation, and take away exactly the points a wrong, partial,
  miscounted, unactivated or edited-dataset submission should lose;
- the shape-only checks in `03/assignment/` report the *same* thing for a
  correct submission and for a plausible but entirely wrong one, need no
  dataset at all, and still catch a missing, unreadable, or implausible value.

```bash
python 03/assignment_checks/_grader_selftest/run.py
```

`solve()` is the answer key, which is why it lives here and not in the fork.
The supplied dataset is rebuilt by `scripts/make_assignment03_data.py`, which
also prints the SHA-256 that `_public_checks.py` pins.
