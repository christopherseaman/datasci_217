# Assignment 06 checks (course-owned)

These checks grade the midterm. The handout in `06/assignment/` ships none of them, with no checker, workflow, or test, so its README names every output file with the task that makes it, its first line, and its line count instead. After the deadline, `uv run scripts/grade_submissions.py 06` clones each fork and runs `check_assignment.py` from this folder on the fork's committed files. It grades with main's `grading.py`; add `--local-checks` to use this folder's copy. To grade one submission by hand:

```bash
python3 06/assignment_checks/check_assignment.py path/to/submission
python3 06/assignment_checks/check_assignment.py path/to/submission --json
```

The first prints a readable report; `--json` prints the `datasci217/grading-result/v1` report. The report covers the 75 points graded from files; the other 25 come from human review of the notebook, the decision reasons, and the commit history, five 5-point categories that the handout's CHECKS.md lists with the cell each one reads. The command exits 0 only when every check passes, so judge a run by its JSON, not its exit status.

The checks read only the ten files in the submission's `output/`. They never import, run, or read submitted code, and the notebook is left to human review. The expected values are constants in `grading.py`, fixed by the supplied files in `06/assignment/data/` (built by `scripts/build_assignment06_data.py`); `_grader_selftest/run.py` recomputes every one of them from those files with pandas.

## What each check scores

| Check | Points | Unit scored |
| --- | ---: | --- |
| `file_inventory.txt` | 5 | 1 per data file whose `wc -l` count is right |
| `table_profile.csv` | 5 | 1 per table with right `rows`, `columns`, and `exact_duplicate_rows` |
| `issue_audit.csv` | 10 | 1 per issue |
| `patients_clean.csv: rows`, `sex`, `birth_year`, `site_code` | 2 each | Rows: the 24 patients once each, less 1 per repeated or unexpected row; columns: in proportion to the 24 patients right, rounded down (`sex` and `site_code` must be the exact stripped, uppercased text) |
| `visits_clean.csv: rows` | 4 | The 46 visits (`patient_id` and `visit`) once each, less 1 per repeated or unexpected row and 1 per `visit` label that is not exactly stripped and lowercased |
| `visits_clean.csv: visit_date`, `sbp_mmhg` | 3 each | In proportion to the 46 visits right, rounded down |
| `visits_clean.csv: weight_kg`, `source_file` | 2 each | The same |
| `a1c_long.csv: rows`, `a1c_pct` | 2, 3 | The 46 results once each (rows with a missing `a1c_pct` are ignored); values in proportion |
| `merge_audit.csv` | 7 | 1 per check |
| `analysis_visits.csv: rows` | 3 | One row per clean visit, against the expected 46 or the file's own `visits_clean.csv` |
| `analysis_visits.csv: site_name`, `a1c_pct` | 3 each | In proportion to the visits right |
| `analysis_visits.csv: sex and birth_year` | 2 | In proportion to the visits with both right |
| `sbp_wide.csv: rows`, `baseline`, `week12`, `sbp_change` | 2, 1, 1, 2 | One row per patient with a visit; values in proportion |
| `decision_log.csv: decisions`, `reason` | 3, 1 | In proportion to the 8 decisions matched by `table` and `decision`; a reason on every row |

Each check reads one artifact without requiring another check to pass. A missing record costs its row points once; the value columns are graded on the records present, and among repeated rows any right one counts. Later artifacts may follow the student's own earlier ones: `analysis_visits.csv` its own `visits_clean.csv` keys, `patients_clean.csv` values, `a1c_long.csv` values, and its own `site_code` mapped through the current site list; the first six `merge_audit.csv` counts the student's own keys; `sbp_wide.csv` its own `visits_clean.csv`; and `sbp_change` its own `baseline` and `week12`. These comparisons use the student's keys and site codes exactly as written, as their merges and pivots did, so a label or code left unnormalized is charged only in `patients_clean.csv` or `visits_clean.csv`. So one mistake is charged once. Every failure says what was expected, what was found (for keyed tables, the first differing record), the rule or likely cause, and which task to fix. The checks use only the standard library and run on Python 3.9 or later.

What never costs points: line endings, blank lines, a byte-order mark, spaces at the end of a line or around a header name, a missing final newline, commas, semicolons (with decimal commas), or tabs between cells, column order, extra columns, a leading unnamed index column (or one holding the labels of `issue_audit.csv` or `merge_audit.csv`, as a Series saved with `to_csv` writes it), number format (`6`, `6.0`, `np.int64(6)`), the letter case and spacing of labels, IDs, and text values (except the normalized `sex`, `site_code`, and `visit` columns of the cleaned tables), missing-value spellings (empty, `NaN`, `<NA>`), table names written with or without `data/` and `.csv`, `source_file` written as a path or as `north`/`south`, and any unambiguous date form with or without a time.

## Changing a check

Edit `grading.py`, keep the handout's CHECKS.md contract and the README file checklist in agreement with it, and rerun both tests:

```bash
uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 python 06/assignment_checks/_grader_selftest/run.py
uv run scripts/test_assignment_grading.py 6
```

To change the data, edit `scripts/build_assignment06_data.py`, run `python3 scripts/build_assignment06_data.py`, then update the constants in `grading.py` and the README's line counts until the self-test passes.
