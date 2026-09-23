# Assignment 03 grading development self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with NumPy from `03/assignment/data/bp_readings.csv`, builds
submissions in ignored `scratch/`, and confirms that:

- the checks score a correct submission 100, accept loose formatting,
  rounding, a BOM, CRLF, NumPy scalar reprs and the sample standard deviation,
  and take away exactly the points a wrong, partial, miscounted, unactivated or
  edited-dataset submission should lose;
- the handout's own `check_assignment.py` reports the same result as the
  checks here for an empty, a correct, and a partly wrong submission;
- every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `03/assignment/` and here, so students run locally exactly the checks GitHub
  runs; the list names every checker file here, so a download never misses a
  module; and the handout's only other Python file is `analysis.py`, the
  scaffold the student completes.

```bash
uv run --python 3.13 --with numpy==2.3.3 python 03/assignment_checks/_grader_selftest/run.py
```

`solve()` is an independent NumPy answer to the assignment, which is why it
lives here and not in the handout. The supplied dataset is rebuilt by
`scripts/make_assignment03_data.py`, which also prints the SHA-256 that
`_public_checks.py` pins.
