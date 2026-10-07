# Assignment 04 grading development self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with pandas from the handout's `data/bp_followup.csv` and
`data/home_bp.csv`, builds submissions in ignored `scratch/`, and confirms
that:

- an empty directory and the untouched handout score 0, and a correct
  submission scores 100 through both the course-owned and the handout's
  `check_assignment.py`;
- formatting and equivalent approaches cost nothing: CRLF, spaces, upper-case
  labels and headers, every cell quoted, two-decimal numbers, cells separated
  by semicolons (with decimal commas) or tabs, column order, UTF-16 or a BOM,
  a file name in another letter case, IDs moved into a column with
  `reset_index()` (with or without the row numbers `to_csv()` then writes),
  means rounded to one decimal, the visit summary saved sideways or as
  `describe()`, and a rank computed over every patient instead of the program
  patients;
- `-999` kept as a reading is one mistake: every later file is judged against
  the student's own `bp_loaded.csv`, so it costs only its own check;
- each other single mistake costs exactly the checks it gets wrong, with a
  message that names the fix: the IDs lost with `index=False`, the units row
  kept or used as the header, the note column kept, `nrows` cutting the file
  short, a column or value wrong, a table written twice with `mode="a"`, a
  file missing or saved outside `output/`, reductions run across rows, `len()`
  for `count()`, a missing statistic, `normalize=True`, counts re-sorted by
  name, `>` for `>=`, a missing `isin()` or baseline condition, `age` kept, a
  derived column renamed, missing, or computed wrongly, the change's sign
  flipped, `sort_values()` not assigned back, a descending sort, ties in the
  wrong order, the default or a descending rank, a Parquet file that is
  missing, is CSV text, lost its index, or was saved before the rank, the home
  readings read without `index_col`, the gap's sign flipped, `fill_value=0`,
  and the gap saved without its IDs;
- the printed report gives a fix shared by consecutive checks once, marking
  the rest `(same fix as above)`, and ends with a `Left to fix` line naming
  the failing checks by file and the points they are worth;
- a fresh handout prints the README's "Before Task 2" example, and the
  README's checkpoint order and completion contract agree with the checks;
- the values the checks hold match `data/bp_followup.csv` (units row
  included) and `data/home_bp.csv`, every file the workflow lists in
  `CHECKS_FILES` is byte-identical in `04/assignment/` and here, the list
  names every checker file here, the handout ships no other Python and no
  self-test, and its notebook has no saved outputs;
- the handout sets up as Lecture 03 does: `pyproject.toml` pins numpy 2.3.3,
  pandas 3.0.5, ipykernel 6.29.5, and pyarrow 25.0.0, `uv.lock` locks those
  versions, `.python-version` names 3.13, and no `requirements.txt` ships
  beside them.

```bash
uv run --python 3.13 --with pandas==3.0.5 --with pyarrow==25.0.0 python 04/assignment_checks/_grader_selftest/run.py
```

Run it before committing a change to the checks.
