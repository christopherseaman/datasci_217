# Assignment 10 grading regression checks

This directory contains development QA fixtures, not a separate grading policy. The public `grading.py` module is the single source of scoring rules for students and graders.

`run.py` creates fresh temporary submissions from frozen examples embedded in the harness. It checks the shared grader for accepted CSV serialization/row order, a minimal valid PNG, isolated missing artifacts, and wrong numeric values. It never executes student notebook code or depends on old scratch output.

`autograder.py` and `grader.py` are compatibility entrypoints to the same public grader; they do not provision packages or require runner metadata. Use the assignment's declared environment and run `python 10/assignment/_grader_selftest/run.py` from the course repository.
