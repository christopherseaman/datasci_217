# Assignment 10 checks self-test

Course-side QA for the checks, not a second grading mode. It builds
submissions in ignored `scratch/a10_selftest/` and confirms that:

- every frozen value in `_value_checks.py`, the 20 common wrong answers
  included, matches a fresh computation from `10/assignment/data/` by
  `scripts/build_assignment10_data.py`;
- a correct submission scores 100, through `grade_submission()` and both
  copies of `check_assignment.py`, and so does the same submission written
  differently: rows and columns reversed, headers in capitals, CRLF and a
  byte-order mark, semicolons with decimal commas, tabs, a row-number column,
  numbers rounded to two decimals, R² and the screening metrics as percents,
  `1`/`0` booleans, `const` for the intercept, `pd.cut()`'s own band names
  such as `(17, 39]`, and the partitions named `Training`, `fit_rows`,
  `valid`, and `Testing`;
- a submission folder with broken data and a Python file in it still scores
  100: the checks read `output/` only and never run student code;
- an empty folder and the untouched handout score 0;
- each of the 47 checks loses its points, and only its points, to one
  mistake: a value column renamed (the columns check), a row left out or a
  one-row key changed (the rows check), one wrong cell (each values check),
  or bytes that are not a PNG; each message says what was found and what was
  expected;
- each of the 20 common wrong answers costs exactly one check, the first it
  reaches, whose message names its cause: a formula without `age` or `bmi`,
  90% intervals, `"size"` counting the patients with no BMI, residuals with
  the sign flipped, the availability rules `< 0`, `== 0`, and no audit (each
  also carried into Tasks 3.1 and 3.2), a split on `visit_time`, ranges read
  from `visit_time`, skipping the fit-row cut, a baseline fitted on the
  validation rows, RMSE without its root and swapped `r2_score` arguments (in
  Tasks 3.1 and 3.2), Task 3.1 scored on the test rows, a refit on the
  validation rows, test metrics scored on the validation rows, mean errors
  with the sign flipped, `>` for `>=` in the LACE rules, and precision and
  recall with their arguments swapped;
- a missing file costs only its own artifact's checks;
- the README's checkpoint header lines, the notebook's, and the CHECKS.md
  contract agree with the checks and `POINTS`; the README's row counts and
  feature list match the frozen values; the README and the notebook number
  their tasks the same way; neither prints a finished answer expression; and
  no em or en dash appears in them or the checks;
- every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `10/assignment/` and here (and in the handout's `.checks/` cache when
  present), the handout has no other Python file, an empty `output/`, the
  four Parquet files in `data/`, and a notebook with its outputs cleared, and
  pytest collects and runs the checks.

```bash
uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 \
    --with statsmodels==0.14.6 --with scikit-learn==1.9.0 --with 'pytest>=8,<9' \
    python 10/assignment_checks/_grader_selftest/run.py
```
