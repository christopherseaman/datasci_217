# Assignment 03 grader self-test

This instructor-only directory materializes disposable correct and defective submissions and runs the public checker against them. It is not part of the nine student-work/supplied-data files listed for implementation in the assignment, and it must be excluded when the standalone student assignment repository is created.

Run from `03/assignment` in the candidate Python 3.12.13 and NumPy 2.0.2 environment:

```bash
python _grader_selftest/run.py
python _grader_selftest/run.py --pytest
```

Instructor grading tests should be implemented independently from the written contract and should not trust this editable self-test or `_public_checks.py`. Published checker logic is discoverable; this directory makes no secrecy claim and contains no solution or credential intended for student distribution.
