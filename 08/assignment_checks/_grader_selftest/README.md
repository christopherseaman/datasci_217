# Assignment 08 checks self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with pandas from `08/assignment/data/imaging_exams.parquet`, the
way the lecture does, builds submissions in ignored `scratch/`, and confirms
that:

- the Parquet file equals `scripts/build_assignment08_data.py`'s `build()`,
  and the frozen values in `_value_checks.py` equal its `expected()`;
- a correct submission scores 100, and so do one built from plain text keys
  instead of the ordered categorical, one built with `observed=False` (zero-exam
  rows for Eastport and Valley's MRI), one saved with every row index (apply's
  MultiIndex included), one grouped without `as_index=False` and saved with its
  index, and one written differently: columns and rows in another order, CRLF,
  a byte-order mark, UTF-16, quoted or padded cells, labels and headers in
  another letter case, rounded means and medians, `NaN` for an empty cell, the
  flat `sort_values()` and `head()` route for Task 2.4, and every file
  separated by semicolons (with decimal commas) or by tabs, where one wrong
  value still costs only its own check;
- a mistake is charged once: an omitted value column, a key-only file, a pivot
  that drops MRI, and two columns renamed with the right values cost only the
  columns check; two renamed with wrong values also cost both values checks;
  lost key columns cost only the columns checks; a mistake copied from Task
  2.1 into the pivot is charged at its source;
- each plausible mistake costs only its own check, with a message naming the
  cause: a peer median over every priority, computed after keeping the delayed
  exams, or assigned from `.median()`; `>= 8` for `> 8`; a dose filter grouped
  over every modality; every column saved from the filter; `fill_value=0`,
  `aggfunc="mean"`, swapped axes, `margins=True`, and list arguments in the
  pivot; `.dropna()` on the pivot. A selection that shares no row with the
  expected one (the per-scan row mask, the top reports over every priority,
  `nsmallest`) also costs that artifact's values check, which has nothing to
  compare;
- an empty directory and the untouched handout score 0;
- each of 27 single mistakes, one per check, costs exactly that check; a
  missing file costs only its own artifact's checks and is never told to take
  another required file's name; a file holding only the optional Eastport row
  earns no value checks;
- the checks read only `output/`: poisoned data and code in a submission change
  nothing;
- the handout README's checkpoint header lines and row counts, the completion
  contract, and the task numbering match the checks, `POINTS`, and the
  notebook;
- every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `08/assignment/` and here, the handout has no other Python file, and its
  notebook ships with outputs cleared; both `check_assignment.py` copies and
  pytest report the same result.

```bash
uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 --with 'pytest>=8,<9' \
    python 08/assignment_checks/_grader_selftest/run.py
```
