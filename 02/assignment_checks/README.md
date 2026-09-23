# Assignment 02 value checks (course-side)

These are the checks that decide the Assignment 02 grade.
`02/assignment/.github/workflows/tests.yml` fetches them on every student push
from `christopherseaman/datasci_217@main:02/assignment_checks/` (its
`CHECKS_REPO`, `CHECKS_REF`, and `CHECKS_PATH`). Nothing here ships in a student
fork: the GitHub Actions run downloads it, checks it runs, and grades with it.

The split it belongs to:

| Half | Lives in | Answers |
|---|---|---|
| Shape checks | `02/assignment/` (`_shape_checks.py`, `grading.py`, `check_assignment.py`) | Is each artifact well formed: present, readable, labelled, a number where a number belongs, inside a clinically plausible band? |
| Value checks | here (`_value_checks.py`, `grading.py`, `check_assignment.py`) | Is each value right, recomputed from `data/clinic_encounters.csv` with the tolerances the assignment documents? |

Both halves expose `grade_submission(path)` returning the same
`datasci217/grading-result/v1` dict, both total 100 points, and
`test_assignment.py` is identical in both, so the same tooling runs either.
The shape half holds no expected value and never opens the encounter file.

## Publishing

Pushing this directory to `main` publishes it; there is no separate checks
repository. The workflow downloads `grading.py`, `check_assignment.py`,
`test_assignment.py`, `_value_checks.py`, and the two files under
`.github/test/` over the committed copies. A fetch that misses any one of them,
or a set that does not run together, is rolled back, and the run falls back to
the shape checks and says in the log that no value was verified.

The two halves parse artifacts with the same regexes, limits, and helpers,
and `test_assignment.py` and `.github/test/` are byte-identical in both; the
self-test fails if they drift. Correct a check here, and change
`02/assignment/_shape_checks.py` too when the edit touches something they share
or the shape of an artifact.

## Checking the checks

```bash
python3 02/assignment_checks/_grader_selftest/run.py
```
