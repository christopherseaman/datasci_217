# Assignment 04 checks (course-owned)

These are the checks that decide the Assignment 04 grade. They read the five
CSV files and one Parquet file a submission saves in `output/` and compare
them with values recomputed from the supplied blood pressure export and home
readings. The Parquet file is read with the standard library alone: its
`PAR1` markers and the column list pandas writes into its footer.
`04/assignment/.github/workflows/tests.yml` downloads them on every student
push from `christopherseaman/datasci_217@main:04/assignment_checks/` (its
`CHECKS_REPO`, `CHECKS_REF`, and `CHECKS_PATH`), checks that they run, and
grades with them.

The handout ships a byte-identical copy of every file the workflow lists in
`CHECKS_FILES`, so `python3 check_assignment.py` in a student's repository
prints each check's result, what to fix, and the score, exactly as GitHub will. The notebook's last cell calls
`run_checks()` from the same file, so it prints that report too.
The listed files:

```text
.github/test/test_assignment.py
.github/test/requirements.txt
test_assignment.py
_value_checks.py
check_assignment.py
grading.py
```

## Publishing

Pushing this directory to `main` publishes it; there is no separate checks
repository. The workflow downloads the files named in `CHECKS_FILES` over the
committed copies. A fetch that misses any one of them, or a set that does not
run together, is rolled back, and the run grades with the copy committed in the
student's repository and says so in the log.

Correct a check here and copy the changed file over its twin in
`04/assignment/`; the self-test fails while the two differ. Every fork gets the
correction on its next push. The local `check_assignment.py` downloads the
current `_value_checks.py` and `grading.py` from main, falling back to the
bundled copy offline or with `--local-checks` / `DS217_LOCAL_CHECKS=1`, so local
and GitHub runs agree.

`_value_checks.py` holds its own copy of the supplied data (`BP_EXPORT`,
`UNITS_ROW`, and `HOME_SBP`); change it together with `data/bp_followup.csv`
and `data/home_bp.csv`. A change to the check list or to `POINTS`
belongs in `_value_checks.py` and `grading.py` at once, in the same order:
`grading.py` pairs the checks with `POINTS` and raises when their counts
differ. It compares the counts itself rather than using `zip(strict=True)`, so
a student's Python 3.9 still runs the checks. Keep the README's completion
contract in agreement with `POINTS`.

## Checking the checks

```bash
uv run --python 3.13 --with pandas==3.0.5 --with pyarrow==25.0.0 python 04/assignment_checks/_grader_selftest/run.py
```

It grades every kind of submission and confirms the handout's copy matches this
one byte for byte.
