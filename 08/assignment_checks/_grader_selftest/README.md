# Assignment 08 checks self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with pandas from `08/assignment/data/clinic_visits.csv`, the way the
lecture does, builds submissions in ignored `scratch/`, and confirms that:

- a correct submission scores 100, and so do one built from plain text keys
  instead of the ordered categorical and one written differently: columns and
  rows in another order, CRLF, a byte-order mark, UTF-16, quoted or padded
  cells, labels and headers in another letter case, `6.00` for `6`, means
  rounded to one decimal, `NaN` for an empty cell, a leading row-number column,
  summaries saved with their index, the zero-visit rows `observed=False`
  adds, and every file separated by semicolons (with decimal commas) or by
  tabs, where one wrong value still costs only its own check;
- an omitted value column costs only the columns check, while an independently wrong present sibling still fails its values check; header-only and unrecognizable partial tables earn no vacuous value credit;
- an empty directory and the untouched handout score 0;
- each of 23 single mistakes, one per check, costs exactly that check,
  including a misnamed column; a summary saved with `index=False` after
  grouping without `as_index=False`, which drops its key column, costs only the
  columns check; a missing file costs only its own artifact's checks; a pivot
  saved after `.dropna()`, which loses Sunset's row and its empty cell, costs
  the rows and empty-cell checks; a file holding only the optional Excelsior
  row earns no value checks; the feedback says what was expected and what was
  found;
- a missing file is never told to take another required file's name; a pivot
  with `index=` and `columns=` swapped costs only its columns check and says
  so; a pivot with `margins=True` costs the columns and rows checks, and both
  name `margins=True`; `wait_vs_clinic` worked out from a wrong
  `clinic_mean_wait` costs only the `clinic_mean_wait` check; a
  `clinic_mean_wait` column assigned from `.mean()` gets a message naming that
  cause;
- the checks read only `output/`: poisoned data and code in a submission change
  nothing;
- the copy of the visit log in `_value_checks.py` matches
  `08/assignment/data/clinic_visits.csv` and the notebook's `CLINIC_ORDER`, and
  the handout README's checkpoint header lines, completion contract, and task
  numbering match the checks, `POINTS`, and the notebook;
- every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `08/assignment/` and here, the handout has no other Python file, and its
  notebook ships with outputs cleared; both `check_assignment.py` copies and
  pytest report the same result.

```bash
uv run --python 3.13 --with pandas==3.0.5 --with 'pytest>=8,<9' python 08/assignment_checks/_grader_selftest/run.py
```
