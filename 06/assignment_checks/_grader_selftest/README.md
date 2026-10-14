# Assignment 06 grading development self-test

Course-side QA for the midterm checks, not a second grading mode. `run.py` generates the data with the handout's `generate_data.py`, answers the midterm with pandas the way the handout README asks, builds submissions in ignored `scratch/`, and confirms that:

- `generate_data.py` writes the same bytes twice, in under 3 seconds, with `\n` line endings, 2 to 8 MB in all, matching `data_manifest.json` and `grading.DATA_SHA256`;
- every value `grading.py` derives matches the independent answer, every valid date falls on a weekday, most baseline A1c results are at or above 6.5%, SBP rows sit exactly on 70 and 250, each recheck pair has two readings, an exact patient repeat changes an audit count, the `SAMPLE_IDS` patients cover every planted pattern, and the README's issue, metric, merge-check, and decision labels match the ones graded;
- the handout README lists all eleven output files in its file tree and in its checklist, each with the first line and line count the answer produces, lists `SAMPLE_IDS` in order, and states no graded count; the complete `output/` stays under 8 KB per file and 200 KB in all;
- the handout ships no checks (no `check_assignment.py`, `grading.py`, `.github/`, or self-test) and no data (`data/` ignored), has a notebook with cleared outputs whose first cell runs the generator and checks the manifest, saves every file but `analysis_sample.csv` before the validation assert, has an empty `output/`, and no checker or handout file reads the Python version;
- an empty directory and the untouched handout score 0 of 75, Task 1's two files alone score 10, a correct submission scores 75, and so does one written with CRLF or CR-only line endings, a byte-order mark, trailing spaces, semicolons or tabs, a space after every comma, reversed columns and rows, index columns, labels, codes, and IDs in another case, `6.0` for `6`, `<NA>` and `NaN` for missing, and dates with a time or slashes;
- no submitted file runs, even when named like the checks, the generator, or a standard-library module;
- each of 29 single mistakes, from a gender spelling left out of the dictionary or swapped melt labels to a recheck's earlier reading kept, carried through every later file as the student's own pipeline would carry it, costs exactly its own points, and the feedback names the fix.

```bash
uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 python 06/assignment_checks/_grader_selftest/run.py
```

It takes about two minutes.
