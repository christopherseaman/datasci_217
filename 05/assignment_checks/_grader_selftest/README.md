# Assignment 05 grading development self-test

Course-side QA for the checks, not a second grading mode. It writes the export
with `generate_data.py`, answers the assignment with pandas, builds submissions
in ignored `scratch/`, and confirms that:

- an empty directory and the untouched handout score 0, a correct submission
  scores 100 through both the course-owned and the handout's
  `check_assignment.py`, and its six files total under 200 KB;
- formatting and equivalent forms cost nothing: CRLF, upper-case cells and
  headers, spaces around report cells, every cell quoted, semicolons, tabs,
  reversed columns, shuffled rows, row numbers, dates day first or with a
  month name, UTF-16 or a BOM, a file name in another letter case, unrounded or
  whole-number percents, `describe()` without `.T`, category counts on their
  side, other step, indicator, and rule labels, and `missing` for a blank;
- each single mistake costs exactly the checks it gets wrong, with a message
  that names the fix: counts taken after `dropna()`, `notna()` for `isna()`,
  fractions for percents, a report saved with `index=False`, the no-lab visits
  kept or `how="any"`, repeats kept, `keep=False`, a subset that drops repeat
  visits, `-999` kept, ages or dates left as text, ages filled with 0, undated
  visits dropped early, IDs or clinics not stripped or not re-cased, blank
  clinics not filled, gender upper-cased, replaced, or mapped without `X`, the
  wrong `pd.cut` edges, counts without `get_dummies()`, passes counted for
  failures, a blank failing a range rule, a renamed rule, 1 * IQR, blanks
  flagged as outliers, an outlier rule not negated, bad IDs kept, every failing visit dropped, impossible values
  counted after blanking or kept (in both columns or one), blanks filled with the median, a missing or
  misplaced file, a sideways report, and a report written twice;
- a mistake upstream is charged only where it was made, since later files are
  judged against the pipeline the student's own earlier files show; Task 5.2's
  assert, which stops some mistakes, is skipped in those cases to show it; a
  stopped assert costs only the impossible-values check, and nothing when kept
  repeats stopped it;
- the printed report gives a fix shared by consecutive checks once, ends with
  a `Left to fix` line, and the notebook's `run_checks()` prints the same text;
- a fresh handout prints CHECKS.md's "Before Task 1" example, and CHECKS.md's
  completion contract agrees with the checks and their points;
- `generate_data.py` writes the bytes `data_manifest.json` names on two runs,
  `data/` is ignored and uncommitted, every file the workflow lists in
  `CHECKS_FILES` is byte-identical in `05/assignment/` and here, the list names
  every checker file here, `check_assignment.py` downloads every module the
  checks import, the handout ships no other Python and no self-test, and its
  notebook has no saved outputs, builds the data in its first cell, saves every
  file but `clean_summary.csv` before Task 5.2's assert, and ends with the
  checks;
- the handout sets up as Lecture 03 does: `pyproject.toml` pins numpy 2.3.3,
  pandas 3.0.5, ipykernel 6.29.5, and pyarrow 25.0.0, `uv.lock` locks those
  versions, `.python-version` names 3.13, and no `requirements.txt` ships
  beside them.

```bash
uv run --python 3.13 --with pandas==3.0.5 --with pyarrow==25.0.0 python 05/assignment_checks/_grader_selftest/run.py
```

Run it before committing a change to the checks. It does not run Python 3.9;
check the generator there with `uv run --python 3.9 --no-project python
generate_data.py` in a scratch copy and compare the checksum with
`data_manifest.json`.
