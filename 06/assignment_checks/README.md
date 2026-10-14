# Assignment 06 checks (course-owned)

These checks grade the midterm. The handout in `06/assignment/` ships none of them, with no checker, workflow, or test, so its README names every output file with the task that makes it, its first line, and its line count instead. After the deadline, `uv run scripts/grade_submissions.py 06` clones each fork and runs `check_assignment.py` from this folder on the fork's committed files. It grades with main's `grading.py`; add `--local-checks` to use this folder's copy. To grade one submission by hand:

```bash
python3 06/assignment_checks/check_assignment.py path/to/submission
python3 06/assignment_checks/check_assignment.py path/to/submission --json
```

The first prints a readable report; `--json` prints the `datasci217/grading-result/v1` report. The report covers the 75 points graded from files; the other 25 come from human review of the notebook, the decision reasons, and the commit history, five 5-point categories that the handout's CHECKS.md lists with the cell each one reads. The command exits 0 only when every check passes, so judge a run by its JSON, not its exit status.

The checks read only the eleven files in the submission's `output/`. They never import, run, or read submitted code, and the notebook is left to human review. The handout ships `generate_data.py` and `data_manifest.json`, never the data (Git ignores `data/`). When `grading.py` loads, it runs the course copy of `06/assignment/generate_data.py` in memory, refuses to load unless each built table's SHA-256 (of its CSV text, which the two Parquet files hold as tables) matches `DATA_SHA256`, and derives every expected value from that text with the standard library, writing nothing to disk and never importing pandas; `_grader_selftest/run.py` recomputes every value with pandas. Loading takes about 2 seconds.

The outputs are analyst-sized, about 5 KB in all: audits (`issue_audit.csv`, `merge_audit.csv`), summaries (`patients_summary.csv`, `gender_categories.csv`, `visits_summary.csv`, `a1c_summary.csv`), the decision log, and two extracts holding only the rows of the 16 patients in `SAMPLE_IDS` (`sbp_wide_sample.csv`, `analysis_sample.csv`). The sample patients cover every planted pattern: a repeated registry row, `9999`, messy gender and site spellings, site `E04`, a visit with no registry row, a registry patient with no visit, rechecks with different readings, SBP `999`, out of range, and exactly 70 and 250, impossible and loosely written dates, blank weights, unmapped visit labels, a missing week 12 visit, and blank A1c results. `SAMPLE_IDS` appears in `grading.py`, the notebook's first cell, and the README; the self-test keeps the three in agreement.

## What each check scores

| Check | Points | Unit scored |
| --- | ---: | --- |
| `file_inventory.txt` | 5 | The three CSV files' `wc -l` counts, in proportion, rounded down |
| `table_profile.csv` | 5 | 1 per table with right `rows`, `columns`, and `exact_duplicate_rows` |
| `issue_audit.csv` | 10 | 1 per issue |
| `patients_summary.csv: patients`, `birth_year`, `site_code` | 2 each | `patients` (30,000 once the repeats are dropped); the three `birth_year` metrics and the four site metrics, each in proportion, rounded down |
| `gender_categories.csv: counts`, `codes` | 2 each | The 4 rows (`F`, `M`, `X`, missing): counts in proportion, rows that differ only in case adding up; codes from `.cat.codes`, each label its own whole number from 0 and missing `-1`. A file without a `code` column, as a text column's counts would be, scores 0 for codes only |
| `visits_summary.csv: rows` | 4 | `visits`, `baseline visits`, `week12 visits`, in proportion |
| `visits_summary.csv: visit_date`, `sbp_mmhg` | 3 each | Missing count, min, and max, in proportion |
| `visits_summary.csv: weight_kg`, `source_file` | 2 each | Missing count and mean; the two source counts |
| `a1c_summary.csv: results`, `a1c_pct_mean` | 2, 3 | Each visit's result count; each visit's mean within 0.006 of the exact mean |
| `merge_audit.csv` | 7 | 1 per check |
| `sbp_wide_sample.csv: rows`, `baseline`, `week12`, `sbp_change` | 2, 1, 1, 2 | One row per sample patient with a visit; values in proportion |
| `analysis_sample.csv: rows` | 3 | One row per visit of a sample patient, in proportion |
| `analysis_sample.csv: site_name`, `a1c_pct`, `gender and birth_year` | 2 each | In proportion to the sample visits right |
| `decision_log.csv: decisions`, `reason` | 3, 1 | In proportion to the 9 decisions matched by `table` and `decision`; a reason on every row |

Each check reads one artifact without requiring another check to pass, and each mistake is charged once, in the file that first shows it:

- Patient and gender counts made with the exact registry repeats left in cost only `patients_summary.csv: patients`.
- When `visits` is off by some rows (charged in `visits_summary.csv: rows`), the summary's other counts may be off by at most that many, and the weight mean by at most that many rows' share; counts made with each recheck's earlier reading are accepted too, because `sbp_wide_sample.csv` charges that mistake.
- An `a1c_pct_mean` is right for the results its row counts, so swapped melt labels cost only `results`.
- When the student's own visit or A1c summary differs from the expected one, the first three `merge_audit.csv` counts are right when `both` plus `left_only` equals their `visits` and `both` plus `right_only` their total `results` (every A1c result when their summary counts none, as unmapped labels leave it); `visits whose patient is not in the registry` and `registry patients with no visit` may differ by as many rows as their `visits` and `patients` do; and `registry patients whose site is not in the site list` is right when it equals their `patients` minus `patients at N01` and `patients at S02`.
- The extracts accept what a charged upstream mistake carries: a gender left unmapped or missing from the dictionary (when `gender_categories.csv` counts are off), `9999` (when the birth year metrics are off), the raw SBP or a blank (when the SBP metrics are off), a missing reading, the earlier reading of a recheck, or an unmapped visit label's row (when the visit counts are off), the other visit's A1c or none (when `a1c_summary.csv` is off), the current `site_name` for the row's own `site_code` as written, repeated rows for rechecked pairs or repeated registry patients, and `sbp_change` from the file's own `baseline` and `week12`.

Every failure says what was expected, what was found, the rule or likely cause, and which task to fix. The checks use only the standard library and run on Python 3.9 or later.

What never costs points: line endings, blank lines, a byte-order mark, spaces at the end of a line or around a header name, a missing final newline, commas, semicolons (with decimal commas), or tabs between cells, column order, extra columns, a leading unnamed index column (or one holding the labels of a summary, as a Series saved with `to_csv` writes it), number format (`6`, `6.0`, `np.int64(6)`), the letter case and spacing of labels, codes, IDs, and text values (`f` reads as `F`; a spelling that still needs its dictionary, such as `woman`, `N-01`, or `wk12`, does cost points), missing-value spellings (empty, `NaN`, `<NA>`), table names written with or without `data/` and `.csv` or `.parquet`, and any unambiguous date form with or without a time.

## Changing a check

Edit `grading.py`, keep the handout's CHECKS.md contract and the README file checklist in agreement with it, and rerun both tests:

```bash
uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 python 06/assignment_checks/_grader_selftest/run.py
uv run scripts/test_assignment_grading.py 6
```

To change the data, edit `06/assignment/generate_data.py` (standard library only, so its bytes match on every Python 3.9 or later), run `python3 scripts/build_assignment06_data.py` to rewrite `data_manifest.json`, copy the SHA-256 values it prints into `DATA_SHA256` in `grading.py`, recheck that `SAMPLE_IDS` still covers every pattern, then update the handout README's line counts until the self-test passes.
