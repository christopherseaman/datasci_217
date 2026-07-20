# Assignment 08 instructor grader self-test

This directory is instructor-only and must be excluded from the Classroom50
student starter and production submission. It contains discoverable release QA,
no credentials, private records, or secret grading rule.

The production runner invokes standard-library `autograder.py` with plain
Python from the student checkout. It installs the exact sibling requirements
into that interpreter before importing `classroom50_grader.py`; PEP 723 is only
local `uv run` provisioning. `classroom50_grader.py` independently protects the assignment contract, removes
submitted outputs from disposable copies, clears stored notebook state, starts
fresh Jupyter kernels, checks canonical behavior, and calls all five public
functions on the disclosed alternate prepared table. It writes the official
`classroom50/result/v1` object to `./result.json`; completed grading exits zero
even when student checks fail. The automated maximum is 90. Human review of the
student-authored Markdown uses Classroom50's context-supplied `review` URL and
remains outside this result.

The grader consumes nonempty `CLASSROOM`, `ASSIGNMENT`, `SUBMISSION_TAG`,
`COMMIT_URL`, and `RELEASE_URL`; `REVIEW_URL` falls back to `COMMIT_URL`, and
the grader generates the UTC result `datetime`. Classroom50's runner may
authoritatively stamp optional `owner`, `assignment_type`, and `submitted_by`
fields; the grader does not require or invent them.

Run the full adversarial author harness from `08/assignment` with the exact
recorded runtime and instructor dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run _grader_selftest/run.py
```

The harness builds only disposable completed/defective submissions. It covers
fresh canonical and alternate behavior, feasible flattened/course-root/nested/
relocated/path-with-spaces layouts, deterministic repeat, fixture failure before
cleanup, unrelated-sentinel preservation separately from extra-file rejection,
official result shapes/exits, corrected resubmission, and representative
operation/scope defects. It performs no network or external Classroom50 write.
