# Assignment 02 value checks (course-side)

This directory is the source for `02/` in the course checks repository that
`02/assignment/.github/workflows/tests.yml` names in `CHECKS_REPO`
(`christopherseaman/datasci_217` by default, path `02/assignment_checks`). Nothing here ships in a student
fork, and nothing here is fetched by a student: the GitHub Actions run
downloads it, checks it runs, and grades with it.

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

Copy this directory to `02/` in the checks repository, keeping the file names:
the workflow downloads them over the committed copies, so `grading.py`,
`check_assignment.py`, `test_assignment.py`, `_value_checks.py`, and the two
files under `.github/test/` must all be served from `<ref>/02/`. A fetch that
misses any one of them is rolled back, and the run falls back to the shape
checks and says in the log that no value was verified.

Correct a check here first, then publish; the fork's shape checks only need
changing when the shape of an artifact changes.

## Checking the checks

```bash
python3 02/assignment_checks/_grader_selftest/run.py
```
