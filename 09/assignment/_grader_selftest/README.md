# Assignment 09 grading regression checks

The public `grading.py` module is the single source of artifact checks and scoring for students and graders. `check_assignment.py [submission_dir] --json` reports the same 100-point automated result for either audience.

This directory contains development QA fixtures. `run.py` tests fresh saved artifacts, equivalent CSV serialization and row order, missing milestones, and incorrect values; it does not execute student notebooks. `autograder.py` and `grader.py` only forward to the public implementation, without provisioning dependencies or requiring runner metadata.

Run in the assignment's declared environment from the course repository:

```bash
python 09/assignment/_grader_selftest/run.py
```
