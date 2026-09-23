# Assignment 01 grading development self-test

This development QA smoke test builds submissions in ignored `scratch/` and
confirms the checks behave.

For the contract: the identity helper hashes each documented address form the
same way and rejects addresses off `@ucsf.edu`; an untouched handout scores 0,
and every check says why; the two practice files alone score 20, and each
costs its own 10; every roster hash, however many students the roster holds,
earns the identity's 15 whatever its case or surrounding whitespace, with or
without a report; a hash off the roster, or no hash, costs those 15 and
nothing else; a missing, non-UTF-8, or folder-shaped report fails every report
line with the same advice, which names make_output.py. Whitespace of any
kind (including a missing space or a no-break space), blank lines, extra lines,
line endings, and the first line (any Python version, other text, or none) are
never graded; a missing line costs only itself, and a wrong value or a letter's
case costs that line's 5 points and says what the line should read. The real
submission that compared the running total instead of each measurement,
printed with double spaces, scores 90.

For the handout: each file in the workflow's `CHECKS_FILES` is byte-identical
in `01/assignment/` and here, and no shape-only checks remain; the README shows
every expected line, and its contract table agrees with `POINTS` row by row;
the handout's `check_assignment.py` prints the same JSON as this directory's
for the same submission; and a clean local run ends with the lines the README
quotes.

For never scoring lower: these checks replaced the checker the fork used to
carry, `01/assignment/_assignment_checks.py` and `grading.py` at commit
`f39598b`. The self-test reads that checker from the repository history and
grades about 1,800 artifact variants with both: each roster hash, other Python
families, CRLF and CR line endings, no final newline, extra whitespace, a
byte-order mark, non-UTF-8 and UTF-16 files, symlinks, directories where files
belong, missing and extra files, and unreadable files. Every variant scores at
least what the f39598b checker gives it, both as built and once its first line
reads the way that checker required, and whatever that checker passed, the
matching checks here pass. Every whitespace, blank-line, and extra-line variant
of the correct report scores 100.

It is not a separate scoring mode; graders use the supplied checker from a
trusted copy. The equivalence run needs a full clone, since it reads commit
`f39598b`.

```bash
python3 01/assignment_checks/_grader_selftest/run.py
```
