# Assignment 05 checks (course-owned)

These checks grade the midterm. The handout in `05/assignment/` ships none of them, with no checker, workflow, or test, so its README names every output file with the task that makes it, its first line, and its line count instead. After the deadline, `uv run scripts/grade_submissions.py 05` clones each fork and runs `check_assignment.py` from this folder on the fork's committed files. To grade one submission by hand:

```bash
python3 05/assignment_checks/check_assignment.py path/to/submission          # readable report
python3 05/assignment_checks/check_assignment.py path/to/submission --json   # datasci217/grading-result/v1
```

The report covers the 75 points graded from files; the other 25 come from human review of the notebook and the decision reasons, five 5-point categories that the handout README's Completion contract lists with the cell each one reads. The command exits 0 only when every check passes, so judge a run by its JSON, not its exit status.

The checks read only the seven files in the submission's `output/`. They never import, run, or read submitted code, and the notebook is left to human review. The expected values are constants in `grading.py`, fixed by the supplied `05/assignment/data/people_raw.csv`; `_grader_selftest/run.py` recomputes every one of them from that file with pandas.

## What each check scores

| Check | Points | Unit scored |
| --- | ---: | --- |
| `raw_preview.txt` | 4 | 2 for each command label followed by the lines that command prints |
| `pipeline_summary.txt` | 5 | 1 per key |
| `numpy_age_summary.csv` | 5 | 1 per metric; `mean` within 0.05 |
| `pandas_selection.csv` | 4 | 1 per requested record with its raw site and status, less 1 per other row; 1 for exactly the three columns |
| `issue_audit.csv` | 15 | 1 per issue |
| `cleaned_people.csv: <column>`, one check for each of the 7 columns | 28 | 4 each, in proportion to the 11 records right in that column, rounded down |
| `decision_log.csv: decisions` | 8 | 1 per decision whose field, issue, and action match |
| `decision_log.csv: reason`, `source`, `source_sha256`, `rows_before`, `rows_after` | 2, 1, 1, 1, 1 | In proportion to the decision rows with the right value, rounded down |

A check scores only its own file, so no check depends on another passing, and one wrong value costs only its own share. Every failure says what was expected, what was found (for the cleaned table, the first differing `record_id` and column), and which task to fix.

What never costs points: line endings, blank lines, a byte-order mark, spaces at the end of a line or around a header name, a missing final newline, column order, extra columns (except in `pandas_selection.csv`, whose task is choosing three), a leading unnamed index column (or, in `numpy_age_summary.csv` and `issue_audit.csv`, that column holding the metric or issue names when the header lacks them, as a saved Series writes it), number format (`6`, `6.0`, NumPy's `np.int64(6)`), the letter case and spacing of labels (keys, metric names, issue names, decision text, record IDs), boolean spellings (`True`, `true`, `1`, `yes`), missing-value spellings (empty, `NaN`, `<NA>`), a date written `YYYY/MM/DD`, and a time or UTC offset after a date (a zoned timestamp may name the day as written or in UTC). Cleaned text values are compared exactly, because normalizing their case and spacing is what Task 3.3 asks for; `NA` therefore counts as the unconverted status sentinel, not as missing. `needs_review` is also right when it follows the rule from the file's own `age` and `visit_date`, and `rows_after` when it equals the file's own cleaned row count, so one cleaning mistake is not charged twice. A cleaned table without a `record_id` column loses that column's 4 points; when it still has one row per record, its rows are matched to the records in source order, so the other six columns are still graded.

## Changing a check

Edit `grading.py`, keep the handout README's Completion contract and file checklist in agreement with it, and rerun both tests:

```bash
uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 python 05/assignment_checks/_grader_selftest/run.py
uv run scripts/test_assignment_grading.py 5
```
