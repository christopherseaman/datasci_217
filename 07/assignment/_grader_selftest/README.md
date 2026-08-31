# Assignment 07 instructor grader candidate

These files are instructor-only and are excluded from student repositories.
They provide an independent grader candidate and adversarial release
harness. The grader never imports or invokes `check_assignment.py`.

The grading runner invokes the standard-library `autograder.py` bootstrap with plain
Python from the student checkout. It installs the exact sibling requirements
into that interpreter before importing `grader.py`; PEP 723 is only
local `uv run` provisioning.

Run the harness from the pinned CPython 3.12.13 environment:

```bash
uv run _grader_selftest/run.py
```

Production execution requires nonempty `ASSIGNMENT`, `SUBMISSION_TAG`,
`COMMIT_URL`, and `RELEASE_URL`. `REVIEW_URL` falls back to
`COMMIT_URL`; the grader generates the UTC result `datetime`. It writes official
`./result.json`; ordinary student-test failures return process status zero,
while missing context or grader/kernel infrastructure failure returns nonzero.

Fresh central execution proves that committed artifacts equal regenerated
artifacts. Human review then uses the runner's `REVIEW_URL` to inspect
the committed notebook Markdown and artifacts; the grader creates no separate
review-storage service or bundle.
