# Assignment 09 checks self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with pandas from the three Parquet files in `09/assignment/data/`,
the way the README's steps do, builds submissions in ignored `scratch/`, and
confirms that:

- the Parquet files equal `scripts/build_assignment09_data.py`'s `build()`,
  the frozen values in `_value_checks.py` equal its `expected()`, the
  notebook's constants and download URL match the generator, and the handout's
  `.gitignore` keeps `data/` committed;
- a correct submission scores 100, and so do one that counts Task 2.1's hours
  from `heart_rate`, one that builds the Task 3.2
  features from time masks instead of `shift()` and `rolling()`, one saved with
  every row index, one grouped without `as_index=False` and saved with its
  index, and one written differently: columns and rows in another order, CRLF,
  a byte-order mark, UTF-16, quoted or padded cells, labels and headers in
  another letter case, US dates, rounded means, `NaN` for an empty cell, and
  every file separated by semicolons (with decimal commas) or by tabs;
- a mistake is charged once: an omitted value column, a key-only file, and two
  columns renamed with the right values cost only the columns check; two
  renamed with wrong values also cost both values checks; lost key columns
  cost only the columns checks;
- each plausible mistake costs only its own check, with a message naming the
  cause: UTC days in Task 1.2 (readings only, not episodes); `count()` for
  `size`; hourly means in Task 2.2 (charged once, though Tasks 3.2 and 3.3 are
  built from them); a window that includes the prediction hour (Task 3.3, built
  from it, still passes); lab clocks localized as UTC (charged in Task 3.1, not
  in `last_lactate`); lactates picked by collection time; episodes with no gap
  dropped or left blank;
- an empty directory and the untouched handout score 0;
- each of 25 single mistakes, one per check, costs exactly that check; a
  missing file costs only its own artifact's checks and is never told to take
  another required file's name;
- the checks read only `output/`: poisoned data and code in a submission change
  nothing;
- the handout README's checkpoint header lines and row counts, the completion
  contract, and the task numbering match the checks, `POINTS`, and the
  notebook; the README prints no finished answer expression and no em or en
  dash;
- every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `09/assignment/` and here, the handout has no other Python file, and its
  notebook ships with outputs cleared; both `check_assignment.py` copies and
  pytest report the same result.

```bash
uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 --with 'pytest>=8,<9' \
    python 09/assignment_checks/_grader_selftest/run.py
```
