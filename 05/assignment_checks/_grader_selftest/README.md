# Assignment 05 grading development self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with pandas from the handout's `data/lab_visits.csv`, builds
submissions in ignored `scratch/`, and confirms that:

- an empty directory and the untouched handout score 0, and a correct
  submission scores 100 through both the course-owned and the handout's
  `check_assignment.py`;
- formatting and equivalent forms cost nothing: CRLF, upper-case cells and
  headers, spaces around report cells, every cell quoted, semicolons, tabs,
  reversed columns, shuffled rows, row numbers, dates with a time, `NaT` for a
  blank, UTF-16 or a BOM, a file name in another letter case, unrounded
  percents, `True`/`False` indicators, `1`/`0` flags, and readable rule names;
- each single mistake costs exactly the checks it gets wrong, with a message
  that names the fix: counts taken after `dropna()`, `notna()` for `isna()`,
  fractions for percents, the report saved with `index=False`, the no-lab
  visit kept or `how="any"`, repeats kept, `keep=False`, a subset that drops the
  repeat visit, `-999` kept, ages or dates left as text, IDs or clinics not
  stripped or not re-cased, the blank clinic not filled, gender upper-cased
  instead of mapped or mapped without `X`, the wrong `pd.cut` edges, no indicators, passes counted
  for failures, a blank failing a range rule, bad IDs kept, every failing row
  dropped, impossible values kept, the outlier flag missing or flagging a
  blank, blanks filled with the median, a row saved twice, a file missing or
  saved outside `output/`, a sideways report, and a report written twice;
- a mistake upstream (repeats kept, `-999` kept, dates as text, IDs or clinics
  not stripped, the clinic not filled) is charged only where it was made, since
  later files are judged against the student's own;
- the printed report gives a fix shared by consecutive checks once, ends with
  a `Left to fix` line, and the notebook's `run_checks()` prints the same text;
- a fresh handout prints CHECKS.md's "Before Task 1" example, and CHECKS.md's
  completion contract agrees with the checks and their points;
- `data/lab_visits.csv` matches `RAW_ROWS` and `scripts/make_assignment05_data.py`,
  every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `05/assignment/` and here, the list names every checker file here, the
  handout ships no other Python and no self-test, and its notebook has no
  saved outputs and ends with the checks;
- the handout sets up as Lecture 03 does: `pyproject.toml` pins numpy 2.3.3,
  pandas 3.0.5, ipykernel 6.29.5, and pyarrow 25.0.0, `uv.lock` locks those
  versions, `.python-version` names 3.13, and no `requirements.txt` ships
  beside them.

```bash
uv run --python 3.13 --with pandas==3.0.5 --with pyarrow==25.0.0 python 05/assignment_checks/_grader_selftest/run.py
```

Run it before committing a change to the checks.
