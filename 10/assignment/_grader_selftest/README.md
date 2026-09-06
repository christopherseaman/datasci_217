# Assignment 10 instructor self-test

This directory is instructor-only and must never enter a learner submission.

`run.py` creates fresh temporary learner exports from frozen instructor-owned artifacts embedded in the harness. It checks public and central grading for accepted CSV serialization/row order, a minimal valid PNG, isolated missing artifacts, and wrong numeric values. It never executes learner notebook code or depends on ignored scratch output.

`autograder.py` provisions the exact versions in the instructor-only `requirements.txt` and then invokes `grader.py`. The learner package retains its existing direct pins and does not receive a separate dependency lock.
