# Assignment 02 grading development self-test

This development QA smoke test builds submissions in ignored `scratch/` and
confirms the checks behave.

A complete submission scores 100; formatting differences (spacing, letter case,
line endings, decimal places, `mm Hg`, a UTF-8 BOM, NumPy scalar reprs, line
order, extra lines) still score 100; every cutoff from 120 to 180 mmHg scores
100 when its patient list matches; each wrong value costs only its own check;
the documented `.gitignore` and run-command forms all pass; a repeated patient
ID counts once; a blank line appended to the export is one more skipped row;
and a changed `data/clinic_encounters.csv` fails every recomputed check. It
also confirms the supplied data file still matches `DATA_FINGERPRINT`.

For the handout: every file the workflow lists in `CHECKS_FILES` is
byte-identical in `02/assignment/` and here, so students run locally exactly
the checks GitHub runs; the list names every checker file here, so a download
never misses a module; and the handout's only other Python files are the two
scaffolds the student completes.

It is not a separate scoring mode; graders use the supplied checker from a
trusted copy.

```bash
python3 02/assignment_checks/_grader_selftest/run.py
```
