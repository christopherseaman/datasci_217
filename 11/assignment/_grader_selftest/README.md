# Assignment 11 grading regression checks

This directory contains development QA examples, not a separate student/grader policy. The public `grading.py` module supplies the same artifact checks and point allocations through `check_assignment.py` for everyone.

The grader reads CSV/PNG outputs and `report.md` from the submission, using the trusted grader's release and manifest to calculate expected results. It does not require submitted starter files, inspect coursework pairs, execute submitted code, or refit models. Each milestone is evaluated independently. It reports 85 automated points. The published rubric assigns the remaining 15 points to human review of the report's reasoning and communication; artifact checks cannot prove how modeling decisions were made.

`autograder.py` and `grader.py` are compatibility entrypoints to the public implementation. They do not install dependencies or require runner metadata. Use `check_assignment.py [submission_dir] --json` for the same machine-readable results available to students.

Run the regression harness in the declared CPython 3.14 environment:

```bash
python 11/assignment/_grader_selftest/run.py
```
