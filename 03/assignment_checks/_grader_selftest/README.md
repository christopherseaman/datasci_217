# Assignment 03 grading development self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with NumPy from `03/assignment/data/bp_readings.csv`, builds
submissions in ignored `scratch/`, and confirms that:

- the checks score a correct submission 100, accept loose formatting,
  rounding or truncating mmHg values to whole numbers, a BOM, CRLF, UTF-16,
  NumPy scalar reprs, `_` digit grouping, the sample standard deviation,
  keys decorated with Markdown or JSON, `=` or space separators, a label in
  brackets or with a note before or after it, bulleted or annotated counts
  lines, a Markdown table, numbered rows, or a dict printed on one line, a
  title line above the record count, a counts file named with any timestamp
  layout or extension, trailing whitespace in the dataset, a file name in
  another letter case, any numpy version and any interpreter path, and take
  away exactly the points a wrong, partial, hedged, miscounted or cut-short
  submission should lose; `monitor_offset` and `stage2_other_monitors` right
  for the monitor a wrong `high_monitor` names cost nothing more; the probe's
  `numpy` and `interpreter` lines, the counts file's timestamped name, and
  each monitor's count are separate checks, so one slip costs only its part;
- the checks read only `output/`: editing or deleting the dataset, the
  scaffold, `pyproject.toml`, or `uv.lock` changes neither a score nor a word
  of feedback, and `SUPPLIED_READINGS` at the end of `_public_checks.py` is
  `03/assignment/data/bp_readings.csv` byte for byte;
- the feedback names the fix for a misnamed or misplaced summary, an undated
  or misplaced counts file (saying when its counts are right), unsorted
  `uniq -c` input, the wrong `cut` field, `cut` without `-d','`, an empty
  `$timestamp`, an empty numpy probe, a probe whose lines `>` overwrote,
  a wrong or empty answer on a line that writes without `"\n"` ran
  together (whose right answers each score in full), patient answers
  computed with `axis=0`, a label line naming two ids, a label written as
  its position rather than its id, a stage 2 count of readings rather than
  patients, and a key written twice, says which number it read from a line holding several, and says
  what the data gives for a wrong answer; the printed report gives a fix
  shared by consecutive checks once, marking the
  rest `(same fix as above)`, and ends with a `Left to fix` line naming the
  failing checks by file; and a fresh handout prints the README's "Before
  Task 1" example exactly;
- a number answer written as an array of several numbers fails with a
  message that counts them and names the call without `axis=`: `mean_sbp`
  as the per-hour array `readings.mean(axis=0)` prints, whose first number
  sits within the tolerance, fails printed, repr'd, or as a list, and so does
  `sd_sbp` per hour, while a one-item list still reads as its number;
- no submission loses a check it passed under the checks committed at HEAD,
  except a check it declares tightened (the array rule above, made before the
  assignment opened on 2026-09-30): from then on a change to the checks may
  only raise a score, and each committed change becomes the baseline for the
  next;
- the handout's own `check_assignment.py` reports the same result as the
  checks here for an empty, a correct, and a partly wrong submission;
- every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `03/assignment/` and here, so students run locally exactly the checks GitHub
  runs; the list names every checker file here, so a download never misses a
  module; and the handout's only other Python file is `analysis.py`, the
  scaffold the student completes.

```bash
uv run --python 3.13 --with numpy==2.3.3 python 03/assignment_checks/_grader_selftest/run.py
```

`solve()` is an independent NumPy answer to the assignment, which is why it
lives here and not in the handout. The supplied dataset is rebuilt by
`scripts/make_assignment03_data.py`; a rebuilt file also has to be copied into
`SUPPLIED_READINGS`, which the self-test confirms.
