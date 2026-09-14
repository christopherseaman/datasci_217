# Assignment 04 grading development self-test

This development QA directory checks the public grader against committed artifacts and preserves credit for independent milestones. It never executes student code and is not a separate scoring mode.

Run with the course environment:

```bash
python 04/assignment/_grader_selftest/run.py
```

The self-test uses frozen instructor examples and disposable copies under the course repository's ignored `scratch/` directory. It checks the actual public command and central grading function: empty starter, correct artifacts, equivalent CSV quoting/line endings, a missing milestone, and incorrect values.

`autograder.py` is a compatibility launcher for the public checker. Instructors run a trusted copy of that checker against submissions; it does not import or execute student code.
