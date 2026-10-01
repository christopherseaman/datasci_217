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
  long, a missing column, a renamed reading column, a wrong value, no mask,
  `>` for `>=`, a mask that keeps nothing, no tie-break, the wrong sort
  direction or both keys descending, `sort_values()` not assigned back, one
  wrong line total or quantity, every total computed wrongly, a missing or
  misnamed total, a misnamed column, an extra column, a missing file, a file
  saved in the assignment folder instead of `output/`) costs exactly the
  checks it gets wrong, with a message that names the fix, and so do wrong
  readings or swapped columns in a two-row block saved without its index, and
  a wrong value in a semicolon- or tab-separated file. Column names, order
  lines, line totals, and sort rules are separate checks, so a renamed column
  costs only its 2-point name check. A missing `line_total_usd` in recognizable data costs only its column check, while wrong present totals lose their totals points; header-only/unknown-key omissions earn no inferred reading or total credit;
- the printed report gives a fix shared by consecutive checks once, marking
  the rest `(same fix as above)`, and ends with a `Left to fix` line naming
  the failing checks by file and the points they are worth;
- a fresh handout prints the README's "Before Task 2" example, and the
  README's checkpoint order and completion contract agree with the checks;
- the values the checks hold match `data/supply_order.csv`, in its order, and
  the notebook,
  every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `04/assignment/` and here, the list names every checker file here, the
  handout ships no other Python and no self-test, and its notebook has no
  saved outputs;
- the handout sets up as Lecture 03 does: `pyproject.toml` pins numpy 2.3.3,
  pandas 3.0.5, and ipykernel 6.29.5, `uv.lock` locks those versions,
  `.python-version` names 3.13, and no `requirements.txt` ships beside them.

```bash
uv run --python 3.13 --with pandas==3.0.5 python 04/assignment_checks/_grader_selftest/run.py
```

Run it before committing a change to the checks.
