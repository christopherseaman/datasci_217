# Assignment 10 checks (course-owned)

These are the checks that decide the Assignment 10 grade. They read the nine
CSV files and the one PNG a submission saves in `output/` and compare them
with values computed from the supplied data in `10/assignment/data/`.
`10/assignment/.github/workflows/tests.yml` downloads them on every student
push from `christopherseaman/datasci_217@main:10/assignment_checks/` (its
`CHECKS_REPO`, `CHECKS_REF`, and `CHECKS_PATH`), checks that they run, and
grades with them.

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
`10/assignment/`; the self-test fails while the two differ. Every fork gets the
correction on its next push. A student's local copy changes only when the
handout is republished (`scripts/publish_assignment.sh 10 ...`) and the student
syncs the fork, so until then the local run can lag the GitHub run, and the
GitHub run counts.

`_value_checks.py` holds the expected values as literals, so the checks need
only the standard library. They come from the lecture's own calls on
`10/assignment/data/`, and the test predictions and test metrics hold both
refits the lecture allows (training rows only, or training plus validation
rows). Change them together with the data files and the notebook; the
self-test recomputes every one. A change to the check list or to `POINTS`
belongs in `_value_checks.py` and `grading.py` at once, in the same order:
`grading.py` zips the checks against `POINTS` with `strict=True`. Keep the
README's checkpoints and completion contract in agreement with the checks; the
self-test compares them.

## Checking the checks

```bash
uv run --python 3.13 --with-requirements 10/assignment/requirements.txt --with 'pytest>=8,<9' python 10/assignment_checks/_grader_selftest/run.py
```

It grades every kind of submission and confirms the handout's copy matches this
one byte for byte.
