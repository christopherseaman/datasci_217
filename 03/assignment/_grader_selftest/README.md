# Assignment 03 grading development self-test

This development QA smoke test copies the current assignment into ignored `scratch/`, accepts its saved pipeline results and analysis transcript, then confirms a changed artifact is rejected. It is not a separate scoring mode; graders use the public checker from a trusted copy.

```bash
python 03/assignment/_grader_selftest/run.py
```
