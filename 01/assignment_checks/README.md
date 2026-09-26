# Assignment 01 checks (course-side)

These are the checks that grade Assignment 01. The handout,
`01/assignment/`, ships a byte-identical copy of each file the workflow lists,
so a student's `python3 check_assignment.py` gives the same score and the same
advice that GitHub gives. `01/assignment/.github/workflows/tests.yml` still
downloads this directory on every student push, from
`christopherseaman/datasci_217@main:01/assignment_checks/` (its `CHECKS_REPO`,
`CHECKS_REF`, and `CHECKS_PATH`), so a correction reaches every student on
their next push, and `scripts/grade_submissions.py 01` grades every fork with
it.

## What is graded

Each check is scored on its own, and `POINTS` in `grading.py` follows the
order of `CHECKS` in `_value_checks.py`:

| Checks | Passes when | Points |
|---|---|---:|
| `terminal-practice/source.txt`, `terminal-practice/path-check.txt` | The file is a regular file in a regular `terminal-practice` directory. | 10 each |
| `report: Project` through `report: Next checkpoint`, one per line of `EXPECTED_READINESS` after the first | `output/readiness.txt` has that line in its place, with all whitespace ignored. | 5 each |
| `identity hash on the roster` | `output/student_identity.txt` holds a hash in `ROSTER_HASHES`, whatever the report says. | 15 |

The report's lines are compared with every whitespace character removed.
Blank lines are dropped, and the rest are aligned in order with the expected
lines (`difflib.SequenceMatcher`); an expected line passes when the alignment
matches it. So an extra line costs nothing, whether it is the Python version
line (which is never graded, whatever it says), a leftover debug print, or a
blank line, and a missing line costs only itself instead of moving every later
line out of place. A report that is missing, empty, not a regular file, or not
UTF-8 fails every line with that reason. A failing line says what it should
read and names the script that prints it and its task. When the unmatched
stretch of the report around it has a line with the same label, the message
shows that line; otherwise, when the stretch holds exactly one line for each
line left to show and does not start the report, it shows the line in the same
place; otherwise it names the expected lines around it and says whether the
report has the line out of order or not at all. The README already shows every
expected line.

`check_assignment.py` prints a fix shared by consecutive checks once, marking
the rest `(same fix as above)`, and ends a run that is not complete with a
`Left to fix` line naming the failing checks by the file they read, from the
`artifact` and `label` of each check in `CHECKS`.

The checks replaced the vendored `01/assignment/_assignment_checks.py` and
`grading.py` at commit `f39598b`, which gave 20 points for the practice files
and 80 for the report and identity together. This quarter's forks were graded
with that checker, so no submission may score lower here: the self-test
replays it from the repository history against every artifact variant it
builds and fails if any scores lower, with or without the Python line set
aside.

## Publishing

Pushing this directory to `main` publishes it; there is no separate checks
repository. The workflow downloads the files in its `CHECKS_FILES` over the
committed copies: `grading.py`, `check_assignment.py`, `test_assignment.py`,
`_value_checks.py`, and the two files under `.github/test/`. A fetch that
misses any one of them, or a set that does not run together, is rolled back,
and the run grades with the copy committed in the fork and says so in its log.
Forks made before the handout shipped these checks still download them, since
the file names are unchanged.

Correct a check here and copy the same file into `01/assignment/`; the
self-test fails if the two differ. Update `ROSTER_HASHES` here when the roster
changes, and copy `_value_checks.py` into the handout too.

## Checking the checks

```bash
python3 01/assignment_checks/_grader_selftest/run.py
```
