# Assignment 01 grader self-test

This directory is instructor/grader validation material, not part of the student starter repository. `run.py` materializes temporary fixture submissions for correct, partial, hard-coded, partial-slice-loop, dead-code, dynamic-file-I/O, boundary, divisor, missing-loop, missing-else, working-directory, forbidden-construct, stale-output, and label/rounding cases. It verifies that the shared public checks accept the correct fixture and reject each targeted defect.

Run from `01/assignment` with Python 3.12:

```bash
python _grader_selftest/run.py
```

Production-only tests and fixtures remain in the centrally managed Classroom 50 grader bundle. Public checks are intentionally discoverable and implement the written student contract; this self-test does not claim secrecy.
