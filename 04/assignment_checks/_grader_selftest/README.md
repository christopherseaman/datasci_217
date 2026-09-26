# Assignment 04 grading development self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with pandas from the handout's `data/supply_order.csv` and the
notebook's supplied `fridge_readings` array, builds submissions in ignored
`scratch/`, and confirms that:

- an empty directory and the untouched handout score 0, and a correct
  submission scores 100 through both the course-owned and the handout's
  `check_assignment.py`;
- formatting that changes no value costs nothing: CRLF, trailing spaces, a
  missing final newline, quoting every cell, column order, label and item
  case, `2.00`-style numbers, a leading row-number column, UTF-16 or a BOM, a
  file name in another letter case, an index moved into a column with
  `reset_index()`, whether or not `index=False` is left out, and cells
  separated by semicolons (with decimal commas) or tabs;
- each single mistake (index left out or unnamed, a slice one row short or
  long, a missing column, a wrong value, no mask, `>` for `>=`, no tie-break,
  the wrong sort direction or both keys descending, one wrong line total or
  quantity, every total computed wrongly, a missing or misnamed total, a
  misnamed column, an extra column, a missing file) costs exactly the checks
  it gets wrong, with a message that names the fix, and so does a wrong
  reading in a block saved without its index or a wrong value in a semicolon-
  or tab-separated file. The selected supplies are checked per column, per
  order line, and per sort rule, so one wrong line total costs only that
  line's 4 points;
- the printed report gives a fix shared by consecutive checks once, marking
  the rest `(same fix as above)`, and ends with a `Left to fix` line naming
  the failing checks by file and the points they are worth;
- a fresh handout prints the README's "Before Task 2" example, and the
  README's checkpoint order and completion contract agree with the checks;
- the values the checks hold match `data/supply_order.csv` and the notebook,
  every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `04/assignment/` and here, the list names every checker file here, the
  handout ships no other Python and no self-test, and its notebook has no
  saved outputs.

```bash
uv run --python 3.13 --with pandas==3.0.5 python 04/assignment_checks/_grader_selftest/run.py
```

Run it before committing a change to the checks.
