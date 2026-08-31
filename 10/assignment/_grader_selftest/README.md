# Assignment 10 instructor self-test

This directory is instructor-only and must never enter a learner submission.

`run.py` materializes a correct notebook only in disposable directories, executes it through the real Jupyter entry point, checks exact artifacts and live behavior, exercises public alternates and adversarial package boundaries, and validates the instructor grading-result contract.

`autograder.py` provisions the exact versions in the instructor-only `requirements.txt` and then invokes `grader.py`. The learner package retains its existing direct pins and does not receive a separate dependency lock.
