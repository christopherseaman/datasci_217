# Assignment 06 checks self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with pandas from `06/assignment/data/`, the way the lecture does,
builds submissions in ignored `scratch/`, and confirms that:

- a correct submission scores 100, and so does one written differently: columns
  and rows in another order, CRLF, a byte-order mark, quoted or padded cells,
  labels and headers in another letter case, `148.0` or `5.00` for a number,
  `NaN` for an empty cell, a leading row-number column, IDs in an index column
  with no header, the round trip saved straight from `pivot()`, and the
  `record_status` column the lecture's merge keeps, and every file separated
  by semicolons (with decimal commas) or by tabs, where one wrong value still
  costs only its own check;
- an empty directory and the untouched handout score 0;
- each of 20 single mistakes, one per check, costs exactly that check, including
  a misnamed column such as melt's default `variable` and `value`; a missing
  file costs only its own artifact's checks; an unfiltered or inner merge costs
  only the checks it gets wrong; a file saved without its ID column costs its
  columns and rows checks, not its values; and the feedback says what to fix;
- the checks read only `output/`: poisoned data and code in a submission change
  nothing;
- the copies of the supplied data in `_value_checks.py` match
  `06/assignment/data/`, and the handout README's checkpoint header lines and
  completion contract match the checks and `POINTS`;
- every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `06/assignment/` and here, the handout has no other Python file, and its
  notebook ships with outputs cleared; both `check_assignment.py` copies and
  pytest report the same result.

```bash
uv run --python 3.13 --with pandas==3.0.5 --with 'pytest>=8,<9' python 06/assignment_checks/_grader_selftest/run.py
```
