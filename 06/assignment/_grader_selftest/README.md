# Assignment 06 instructor grader self-test

This directory is instructor-only, discoverable grading infrastructure. Exclude
it from the Classroom50 student starter and production submission. It contains
no secrets, credentials, or private fixtures.

Classroom50 invokes `autograder.py` with plain Python from the student checkout.
That standard-library bootstrap installs the exact sibling requirements into
the same interpreter before importing the grader; PEP 723 is only local `uv`
tooling. `classroom50_grader.py` independently protects the assignment contract, clears
stored notebook state, removes or replaces artifacts in disposable copies,
starts fresh Jupyter kernels, appends grader-owned checks, and calls all six
student functions on alternate in-memory tables. It writes the official
`classroom50/result/v1` object to `./result.json`; captured student-test failures
still exit zero. The result's automated `max-score` is the provisional 90. The
pending-policy human 10 points remain outside this file.

Run the full adversarial harness from `06/assignment` with the exact recorded
Python, NumPy, and pandas environment plus the pinned instructor dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run _grader_selftest/run.py
```

Production execution requires nonempty `CLASSROOM`, `ASSIGNMENT`,
`SUBMISSION_TAG`, `COMMIT_URL`, and `RELEASE_URL`. `REVIEW_URL` falls back to
`COMMIT_URL`; the grader generates the UTC result `datetime`.

The harness builds disposable correct and defective submissions, tests the
plain-Python production entrypoint and both
flattened and course-root layouts including paths with spaces, and performs no
external Classroom50 configuration or network operation.
