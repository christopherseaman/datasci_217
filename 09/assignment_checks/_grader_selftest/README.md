# Assignment 09 checks self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with pandas from `09/assignment/data/vitals.csv` and
`09/assignment/data/labs.csv`, the way the lecture does, builds submissions in
ignored `scratch/`, and confirms that:

- a correct submission scores 100, and so does one written differently:
  columns and rows in another order, CRLF, a byte-order mark, UTF-16, quoted
  or padded cells, labels and headers in another letter case, `later holdout`
  for `later_holdout`, means rounded to one decimal, `NaN` for an empty cell,
  timestamps written with `T` and `Z`, with a trailing `UTC`, with New York
  offsets, or with no offset (and, read one at a time, with slashes, a month
  name, a 12-hour clock, or `EST`), `true`/`false` and `1`/`0` for the flags, a
  leading row-number column, `source_row` kept or left out where it is
  optional, a summary saved with its index, and every file separated by
  semicolons (with decimal commas) or by tabs, where one wrong value still
  costs only its own check;
- an empty directory and the untouched handout score 0;
- each of 27 single mistakes, one per check, costs exactly that check,
  including a misnamed column; a summary saved with `index=False` without
  `reset_index()`, which drops its key columns, costs only the columns check;
  a prepared table left on the New York clock, with the grid and features built
  from it, costs only the UTC check; a missing file costs only its own
  artifact's checks; the feedback says what was expected and what was found;
- the checks read only `output/`: poisoned data and code in a submission change
  nothing;
- the copies of the data in `_value_checks.py` match `09/assignment/data/`,
  its fixed UTC-5 conversion agrees with pandas' `America/New_York` for every
  supplied time, and the handout README's and notebook's checkpoint header
  lines, the completion contract, and the task numbering match the checks,
  `POINTS`, and each other;
- every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `09/assignment/` and here, the handout has no other Python file, and its
  notebook ships with outputs cleared; both `check_assignment.py` copies and
  pytest report the same result.

```bash
uv run --python 3.13 --with pandas==3.0.5 --with 'pytest>=8,<9' python 09/assignment_checks/_grader_selftest/run.py
```
