# Assignment 07 grading development self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with pandas, matplotlib, and Altair from the handout's `data/`
files, builds submissions in ignored `scratch/`, and confirms that:

- an empty directory and the untouched handout score 0, and a correct
  submission scores 100 through both the course-owned and the handout's
  `check_assignment.py`;
- formatting and reasonable alternatives that change no value cost nothing:
  CRLF, trailing spaces, a BOM or UTF-16, quoting every cell, column and key
  order, label and key case, `79.00`-style numbers, one or two leading
  row-number columns, a file name in another letter case, a spec built from the three
  plotted columns with inferred types and `.interactive()`, rows embedded as
  `data.values`, a layered chart, the critique as a dict keyed by category,
  data types written as notes (`ordered categorical`, `Nominal: ...`,
  `:O`) under `variable_roles` or `variables` or in other words
  (`qualitative`, `categorical (ordered)`, `numeric, continuous`), supporting
  data separated by semicolons or tabs, and a text alternative file in
  another letter case;
- each single mistake (a circle mark, a missing or mistyped encoding, swapped
  axes, a changed or missing patient, data linked by URL, a missing or
  non-PNG chart, a blank or missing critique entry, an extra, misnamed, or
  missing column, a wrong or missing row, a blank or missing contract string,
  a wrong data type, including `categorical (ordered)` for `program` and
  `categorical, not ordinal` for `visit_number`, a wrong value in a
  semicolon-separated file, a text alternative that differs from the JSON, a
  truncated JSON file) costs exactly the checks it gets wrong, with a message
  that names the fix, and a submission that stops after Task 2 keeps its 62
  points;
- a fresh handout prints the README's "Before Task 1" example, and the
  README's clean-run example and completion contract agree with the checks;
- the values the checks hold match `data/rehab_patients.csv` and
  `data/followup_goals.csv`, every file the workflow lists in `CHECKS_FILES`
  is byte-identical in `07/assignment/` and here, the list names every
  checker file here, the handout ships no other Python and no self-test, its
  notebook has no saved outputs, and the notebook's task headings match the
  README's.

```bash
uv run --python 3.13 --with-requirements 07/assignment/requirements.txt \
    python 07/assignment_checks/_grader_selftest/run.py
```

Run it before committing a change to the checks.
