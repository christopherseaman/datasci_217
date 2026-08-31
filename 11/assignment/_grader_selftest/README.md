# Assignment 11 Instructor Grader

This directory is instructor-only packaging. Do not include it in the student
starter or require it from `check_assignment.py`. The public checker applies
structural, invariant, and cross-artifact readiness checks without publishing a
copyable reference pipeline. The trusted grader independently applies exact
deterministic reference checks. Grader discoverability and packaging are
controlled by the course platform rather than by the student starter.

Both graders are artifact-only. They inspect the release, manifest, required CSV
and PNG outputs, and `report.md`; they do not inspect or execute student source or
notebooks. Consequently, exact training-only summaries can be checked, but model
fitting and decision provenance cannot be proven without human source/execution
review.

`autograder.py` is a plain-Python bootstrap. It installs the sibling pinned
`requirements.txt`, invokes the trusted grader, and writes an exact
`datasci217/grading-result/v1` `result.json`. Learner artifact failures exit 0 with all
nine result rows, zero points for failed or blocked rows, and console diagnostics.
Provisioning, context, release, or grader-startup failures exit 2 and remove any
result file. `REVIEW_URL` falls back to `COMMIT_URL`; result datetimes are UTC.

Run the real release harness with CPython 3.12.13:

```bash
uv run --python 3.12.13 --with-requirements 11/assignment/_grader_selftest/requirements.txt python 11/assignment/_grader_selftest/run.py
```
