# Assignment 05 grader self-test

This directory is instructor-only repository infrastructure. Exclude it from
the Classroom 50 student starter and from production submissions. Nothing here
is secret: the grading contract is discoverable, contains no credentials or
private data, and depends on behavioral variation rather than hidden answers.

`autograder.py` is the plain-Python production entrypoint. Its standard-library
bootstrap installs the exact sibling requirements into the runner interpreter
before importing `classroom50_grader.py`; PEP 723 remains local tooling, not
production provisioning. `classroom50_grader.py` is the independent central-
grader reference. It never
imports the editable student `check_assignment.py`. It executes code cells from
fresh state, exercises noncanonical tables, and emits a
`classroom50/result/v1` object for the provisional 85 automated points. The
remaining 15 points are pending-policy human review.

Run the adversarial harness from the assignment directory in the exact recorded
environment:

```bash
uv run _grader_selftest/run.py
```

Production execution requires nonempty `CLASSROOM`, `ASSIGNMENT`,
`SUBMISSION_TAG`, `COMMIT_URL`, and `RELEASE_URL`. `REVIEW_URL` falls back to
`COMMIT_URL`; the grader generates the UTC result `datetime`.

The harness materializes disposable correct and defective submissions. It does
not configure Classroom 50 or contact an external service.
