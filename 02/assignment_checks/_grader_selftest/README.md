# Assignment 02 grading development self-test

This development QA smoke test builds submissions in ignored `scratch/` and
confirms both halves of the checks behave.

For the value checks: a complete submission scores 100; formatting differences
(spacing, letter case, line endings, decimal places, `mm Hg`, a UTF-8 BOM,
NumPy scalar reprs, line order, extra lines) still score 100; every cutoff from
120 to 180 mmHg scores 100 when its patient list matches; each wrong value costs
only its own check; the documented `.gitignore` and run-command forms all pass;
and a changed `data/clinic_encounters.csv` fails every recomputed check. It also
confirms the supplied data file still matches `DATA_FINGERPRINT`.

For the shape checks that ship in the fork: a correct submission and a plausibly
wrong one are indistinguishable, no expected answer appears among the module's
constants, the supplied data file is never read (the checks still pass with
`data/` deleted), and implausible artifacts are still caught.

It is not a separate scoring mode; graders use the supplied checker from a
trusted copy.

```bash
python3 02/assignment_checks/_grader_selftest/run.py
```
