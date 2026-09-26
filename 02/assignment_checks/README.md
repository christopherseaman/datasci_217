# Assignment 02 checks (course-owned)

These are the checks that decide the Assignment 02 grade.
`02/assignment/.github/workflows/tests.yml` downloads them on every student push
from `christopherseaman/datasci_217@main:02/assignment_checks/` (its
`CHECKS_REPO`, `CHECKS_REF`, and `CHECKS_PATH`), checks that they run, and
grades with them.

The handout ships a byte-identical copy of every file the workflow lists in
`CHECKS_FILES` (`grading.py`, `check_assignment.py`, `test_assignment.py`,
`_value_checks.py`, and the two files under `.github/test/`), so
`python3 check_assignment.py` in a student's repository prints each check's
result, what to fix, and the score, exactly as GitHub will, then a `Left to fix`
line that only the local run prints. The checks recompute every expected value
from their own copy of the supplied `data/clinic_encounters.csv`
(`SUPPLIED_ENCOUNTERS` in `_value_checks.py`), so there is no answer key to hide.
They read only what the student writes: `README.md`, `.gitignore`, and
`output/`. A student's copy of the data file, like the scaffold scripts, is
never read, so changing it cannot change a score.

## Publishing

Pushing this directory to `main` publishes it; there is no separate checks
repository. The workflow downloads the files named in `CHECKS_FILES` over the
committed copies. A fetch that misses any one of them, or a set that does not
run together, is rolled back, and the run grades with the copy committed in the
student's repository and says so in the log.

Correct a check here and copy the changed file over its twin in
`02/assignment/`; the self-test fails while the two differ. Every fork gets the
correction on its next push. A student's local copy changes only when the
handout is republished (`scripts/publish_assignment.sh 02 ...`) and the student
syncs the fork, so until then the local run can lag the GitHub run, and the
GitHub run counts.

## Checking the checks

```bash
python3 02/assignment_checks/_grader_selftest/run.py
```
