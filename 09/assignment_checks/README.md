# Assignment 09 checks (course-owned)

These are the checks that decide the Assignment 09 grade. They read the six
CSV files a submission saves in `output/` and compare them with values
computed from the supplied step-down unit data, `data/vitals.csv` and
`data/labs.csv`. `09/assignment/.github/workflows/tests.yml` downloads them on
every student push from `christopherseaman/datasci_217@main:09/assignment_checks/`
(its `CHECKS_REPO`, `CHECKS_REF`, and `CHECKS_PATH`), checks that they run,
and grades with them.

The handout ships a byte-identical copy of every file the workflow lists in
`CHECKS_FILES`, so `python check_assignment.py` in a student's repository
prints each check's result, what to fix, and the score, exactly as GitHub will.
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
`09/assignment/`; the self-test fails while the two differ. Every fork gets the
correction on its next push. A student's local copy changes only when the
handout is republished (`scripts/publish_assignment.sh 09 ...`) and the student
syncs the fork, so until then the local run can lag the GitHub run, and the
GitHub run counts.

`_value_checks.py` holds its own copy of the two data files (`VITALS` and
`LABS`) and the prediction time (`PREDICTION_TIME`); change them together with
`09/assignment/data/`, the README, and the notebook. It converts the New York
clock times with a fixed UTC-5 offset, which holds because every time falls in
January; data outside standard time needs a real time zone there. A change to
the check list or to `POINTS` belongs in `_value_checks.py` and `grading.py`
at once, in the same order: `grading.py` zips the checks against `POINTS` with
`strict=True`. Keep the README's checkpoints and completion contract in
agreement with the checks; the self-test compares them.

## Checking the checks

```bash
uv run --python 3.13 --with pandas==3.0.5 --with 'pytest>=8,<9' python 09/assignment_checks/_grader_selftest/run.py
```

It grades every kind of submission and confirms the handout's copy matches this
one byte for byte.
