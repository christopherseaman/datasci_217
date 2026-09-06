# Assignment 04 artifact-grader regression checks

This instructor-only directory is excluded from student assignment repositories. The grader reads committed artifacts and preserves credit for independent milestones. It never executes student code. Notebook explanations are reviewed by a person.

Run with the course environment:

```bash
python 04/assignment/_grader_selftest/run.py
```

The self-test uses frozen instructor examples and disposable copies under the course repository's ignored `scratch/` directory. It checks the actual public command and central grading function: empty starter, correct artifacts, equivalent CSV quoting/line endings, a missing milestone, and incorrect values.

`autograder.py` is the instructor entrypoint and provisions the sibling requirements. The instructor controls this bundle; it does not import a student's checker.
