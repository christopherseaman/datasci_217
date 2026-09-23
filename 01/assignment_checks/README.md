# Assignment 01 value checks (course-side)

These are the checks that decide the Assignment 01 grade.
`01/assignment/.github/workflows/tests.yml` fetches them on every student push
from `christopherseaman/datasci_217@main:01/assignment_checks/` (its
`CHECKS_REPO`, `CHECKS_REF`, and `CHECKS_PATH`), and
`scripts/grade_submissions.py 01` grades every fork with them. Nothing here
ships in a student fork: it holds the expected readiness report and the course
roster's identity hashes, which the fork must not.

The split it belongs to:

| Half | Lives in | Answers |
|---|---|---|
| Shape checks | `01/assignment/` (`_shape_checks.py`, `grading.py`, `check_assignment.py`) | Is each artifact well formed: both practice files present as regular files, a UTF-8 report of 14 lines with a final newline (13 without the Python version line), one 64-character hash? |
| Value checks | here (`_value_checks.py`, `grading.py`, `check_assignment.py`) | Does the report match `EXPECTED_READINESS` after its first line (the Python version, never graded), and is the hash in `ROSTER_HASHES`? |

Both halves expose `grade_submission(path)` returning the same
`datasci217/grading-result/v1` dict with the same two checks worth 20 and 80
points, and `test_assignment.py` is identical in both, so the same tooling runs
either. The 80 points need both output files to pass. The shape half holds no
expected report line and no roster hash.

The value checks replaced the vendored `01/assignment/_assignment_checks.py`
and `grading.py` at commit `f39598b`. They give the score and detail the
f39598b checker gives once the report's first line is set aside, and never a
lower score; any first line, or none, is accepted. Points and test names are
unchanged. The self-test replays that checker from the repository history
against every artifact variant it builds and fails on any other difference, so
forks completed against the earlier handout keep at least their scores when
graded here.

## Publishing

Pushing this directory to `main` publishes it; there is no separate checks
repository. The workflow downloads `grading.py`, `check_assignment.py`,
`test_assignment.py`, `_value_checks.py`, and the two files under
`.github/test/` over the committed copies. A fetch that misses any one of them,
or a set that does not run together, is rolled back, and the run falls back to
the shape checks and says in the log that no value was verified.

The two halves read artifacts with the same constants and helpers, and
`test_assignment.py` and `.github/test/` are byte-identical in both; the
self-test fails if they drift. Correct a check here, and change
`01/assignment/_shape_checks.py` too when the edit touches something they share
or the shape of an artifact. Update `ROSTER_HASHES` here when the roster
changes; it never goes in the fork.

## Checking the checks

```bash
python3 01/assignment_checks/_grader_selftest/run.py
```
