# Assignment 01 grading development self-test

This development QA smoke test builds submissions in ignored `scratch/` and
confirms both halves of the checks behave.

For the value checks: the identity helper hashes each documented address form
the same way and rejects addresses off `@ucsf.edu`; an untouched handout scores
0; the two practice files alone score 20; every one of the 40 roster hashes
scores 100 with a correct report, whatever its case or surrounding whitespace;
a hash off the roster, a wrong report line, or a missing report costs the 80
points together; and any first line, or none, earns them.

For equivalence: the value checks replaced the checker the fork used to carry,
`01/assignment/_assignment_checks.py` and `grading.py` at commit `f39598b`. The
self-test reads that checker from the repository history and grades about 1,500
artifact variants with both: each roster hash, other Python families, CRLF and
CR line endings, no final newline, extra whitespace, a byte-order mark,
non-UTF-8 and UTF-16 files, symlinks, directories where files belong, missing
and extra files, and unreadable files. The value checks give the score and
detail the f39598b checker gives once the first line is set aside, and never a
lower score; any first line, or none, is accepted. Forks completed against the
earlier handout keep at least their scores.

For the shape checks that ship in the fork: they share the value checks'
constants, helpers, and byte-identical test entrypoints; no roster hash and no
report line appears in any checker file under `01/assignment/`; a correct
submission and a plausibly wrong one (a wrong total and a hash off the roster)
are indistinguishable; malformed artifacts are still caught; every artifact the
value checks accept passes the shape checks; and the local checker ends with the
two lines the README quotes.

It is not a separate scoring mode; graders use the supplied checker from a
trusted copy. The equivalence run needs a full clone, since it reads commit
`f39598b`.

```bash
python3 01/assignment_checks/_grader_selftest/run.py
```
