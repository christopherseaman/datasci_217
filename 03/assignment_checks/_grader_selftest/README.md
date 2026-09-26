# Assignment 03 grading development self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with NumPy from `03/assignment/data/bp_readings.csv`, builds
submissions in ignored `scratch/`, and confirms that:

- the checks score a correct submission 100, accept loose formatting,
  rounding or truncating mmHg values to whole numbers, a BOM, CRLF, UTF-16,
  NumPy scalar reprs, `_` digit grouping, the sample standard deviation,
  keys decorated with Markdown or JSON, `=` or space separators, a label in
  brackets or with a note before or after it, bulleted or annotated counts
  lines, trailing whitespace in the dataset, a file name in another letter
  case, any numpy version and any interpreter path, and take away exactly the
  points a wrong, partial, hedged, miscounted, cut-short or edited-dataset
  submission should lose;
- the feedback names the fix for a misnamed or misplaced summary, unsorted
  `uniq -c` input, the wrong `cut` field, a counts file without `.txt`, and a
  key written twice, and says what the data gives for a wrong answer; the
  printed report gives a fix shared by consecutive checks once, marking the
  rest `(same fix as above)`, and ends with a `Left to fix` line naming the
  failing checks by file; and a fresh handout prints the README's "Before
  Task 1" example exactly;
- no submission loses a check it passed under the checks committed at HEAD:
  the assignment is out with students, so a change to the checks may only
  raise a score, and each committed change becomes the baseline for the next;
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
`scripts/make_assignment03_data.py`, which also prints the SHA-256 that
`_public_checks.py` pins.
