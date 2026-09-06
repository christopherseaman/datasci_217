# Assignment 09 grader maintenance

`autograder.py` is the plain-Python production entrypoint. Its standard-library bootstrap installs the exact sibling requirements into the runner interpreter before importing `grader.py`; PEP 723 is only local `uv run` provisioning. The grader does not import the editable public checker. `run.py` is optional release QA: it may execute disposable notebooks and exercise alternate layouts, but production grading reads committed artifacts.

Run from repository root with the exact candidate environment:

```text
PYTHONDONTWRITEBYTECODE=1 uv run --python 3.12.13 --with-requirements 09/assignment/_grader_selftest/requirements.txt python 09/assignment/_grader_selftest/run.py
```

The student template excludes this directory. The grader bundle is discoverable and contains no solution, credential, private record, or secrecy-dependent test. Human review is separate from the automated 90 points and follows the runner-supplied `review` URL.
