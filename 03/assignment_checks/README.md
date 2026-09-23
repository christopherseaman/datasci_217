# Assignment 03 course-side checks

The checks that compare a submission's answers with the supplied readings.
Nothing in this directory ships in a student fork.

## How the two halves fit together

| Half | Lives in | Answers | Points |
| --- | --- | --- | ---: |
| Shape | `03/assignment/` (the fork students clone) | Is each artifact well formed: present, readable UTF-8, labelled, and inside a plausible range? | 100 |
| Value | here, published to the checks repository | Does each answer match the value recomputed from `data/bp_readings.csv`? | 100 |

Both halves expose `grade_submission(path)` returning the same
`datasci217/grading-result/v1` dict, with the same nineteen check names in the
same order and the same `POINTS` tuple, so the same tooling runs either one.
`grading.py`, `check_assignment.py`, `test_assignment.py` and
`.github/test/` are byte-identical to the fork's copies; only
`_public_checks.py` differs, and it is what `check_assignment.py` reads the
report wording from (`SCOPE_NOTE`, `SCORE_LABEL`, `COMPLETE_NOTE`).

A student's fork can therefore never contain an expected value, and a reviewer
cannot score points by importing the checker.

## Publishing

`03/assignment/.github/workflows/tests.yml` downloads these files at run time
from `CHECKS_REPO` (`UCSF-DataSci/ds217-26f-checks`), at path `03/<file>`:

```text
.github/test/test_assignment.py
.github/test/requirements.txt
test_assignment.py
_public_checks.py
check_assignment.py
grading.py
```

Copy this directory to `03/` in that repository. The repository must exist and
hold one directory per assignment; students do not fork it. The workflow
validates what arrives, smoke-tests that it runs together, rolls back to the
fork's shape-only copy if either fails, and says in the log when a run did not
verify any value.

## Maintaining both halves

- A parsing, environment, or naming rule that both halves share — the summary
  reader, the `count label` reader, the NumPy-scalar unwrapper, the
  environment checks, the timestamped file name — is written in both
  `_public_checks.py` files. Fix it in both; they are deployed separately, so
  neither can import the other.
- A change to the check list or to `POINTS` belongs in both halves at once, in
  the same order: `grading.py` zips the checks against `POINTS` with
  `strict=True`.
- Run `python 03/assignment_checks/_grader_selftest/run.py` after any change;
  it grades both halves.
