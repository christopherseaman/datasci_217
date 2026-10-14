# Assignment 06 checks

## Completion contract

The midterm totals 100 points: 75 points graded from your committed files after the deadline, 25 by human review. Each file is graded on its own, against the tables `generate_data.py` writes.

| File | What earns the points | Points |
| --- | --- | ---: |
| `output/file_inventory.txt` | The line counts of the three CSV files, in proportion (rounded down) | 5 |
| `output/table_profile.csv` | 1 for each table with its right `rows`, `columns`, and `exact_duplicate_rows` | 5 |
| `output/issue_audit.csv` | 1 for each issue's count | 10 |
| `output/patients_summary.csv` | 2 for `patients`; 2 for the three `birth_year` metrics and 2 for the four site metrics, each in proportion (rounded down) | 6 |
| `output/gender_categories.csv` | 2 for the counts of `F`, `M`, `X`, and missing, and 2 for their category codes (each code its own whole number from 0, missing `-1`), each in proportion | 4 |
| `output/visits_summary.csv` | 4 for the three visit counts; 3 for the `visit_date` metrics and 3 for the `sbp_mmhg` metrics; 2 for the `weight_kg` metrics and 2 for the `source_file` counts; each in proportion (rounded down) | 14 |
| `output/decision_log.csv` | 3 for the nine decisions' `table` and `decision`, in proportion; 1 for a reason on every row | 4 |
| `output/a1c_summary.csv` | 2 for each visit's `results` and 3 for each visit's `a1c_pct_mean`, in proportion | 5 |
| `output/merge_audit.csv` | 1 for each check's count | 7 |
| `output/sbp_wide_sample.csv` | 2 for one row per sample patient with a visit; 1 each for `baseline` and `week12`; 2 for `sbp_change`, in proportion to the right patients | 6 |
| `output/analysis_sample.csv` | 3 for one row per visit of a sample patient; 2 each for `site_name`, `a1c_pct`, and `gender` with `birth_year`, in proportion to the right visits | 9 |

## How the files are read

- Line endings, blank lines, spaces at the end of a line or around a header name, a comma, semicolon, or tab between cells, column order, extra columns, a leading column of row numbers, and number format (`6`, `6.0`) do not matter. Letter case and surrounding spaces in labels, codes, IDs, and text values do not cost points: `f` reads as `F`. A spelling that still needs its dictionary, such as `woman`, `N-01`, or `wk12`, does.
- An empty field, `NaN`, and `<NA>` all read as missing; a date written year first may carry a `00:00:00` time after it or use `/` in place of `-`, while a date with the year last, such as `01/05/2026`, can be read as May 1 or January 5 and costs its point; a mean rounded to two decimals (within 0.006 of the exact one) is right.
- A file missing from `output/` earns 0 points. A file a stopped notebook never writes costs its points on top of the earlier file that shows the mistake:
    - A Task 4.3 `pivot()` `ValueError`, raised when the Task 3.2 step 6 rechecks are left in, leaves `sbp_wide_sample.csv` and `analysis_sample.csv` unwritten (15 points).
    - A `MergeError` in the Task 4.4 join or a failed Task 5.1 check leaves `analysis_sample.csv` unwritten (9 points).
- One mistake is charged once, in the file that first shows it. A later file is also right when it follows from your own earlier ones:
    - `patients_summary.csv` and `gender_categories.csv` counts made with the exact registry repeats left in cost only `patients`.
    - When `visits` in `visits_summary.csv` is off by some rows, its other counts may be off by at most that many; counts made with the earlier reading of each recheck cost only `sbp_wide_sample.csv`.
    - An `a1c_pct_mean` is right for the results its row counts, so swapped labels cost only `results`.
    - The first three `merge_audit.csv` counts are judged against your own `visits` and A1c `results` when those differ from the expected ones (`both` plus `left_only` equals your visits, `both` plus `right_only` your results, or every A1c result when your summary counts none); the patient counts against your own `patients_summary.csv`.
    - The extracts carry what an earlier mistake left without a second charge: an unmapped gender or visit label, `9999`, a raw SBP, a mislabeled A1c result, and the current `site_name` for your own `site_code`. `sbp_change` is judged against your own `baseline` and `week12` columns.

## Human review

5 points for each category:

| Category | What it reads | Full credit (5) | Partial credit |
| --- | --- | --- | --- |
| Lecture 01 to 06 evidence map | The "Cumulative midterm checkpoint" cell | One entry for each of Lectures 01 to 06, each naming a concrete technique or file from that lecture and saying in one sentence where this notebook uses it | 2 to 4 for four or five lectures, or for entries that name a tool without saying where it is used here |
| Data contract | The Task 2.1 cell | States each table's grain and key (single or composite) and the cardinality of each Task 4.4 join, each fitting these files | 2 to 4 when one or two parts are missing or do not fit these files |
| Decision reasons | The `reason` column of `output/decision_log.csv` | Each of the nine reasons names the problem in these files, such as the rows or values it affects, and why its decision fits | 2 to 4 when some reasons are generic, such as "to clean the data", or do not match their decision |
| Join and reshape safeguards | The Task 3.2 step 6 cell and the Task 4.2 and 4.4 cells | Shows the repeated visit pair before keeping the later reading, the caught `MergeError` and the repeated site rows, the `_merge` counts, and a `validate=` on each Task 4.4 merge | 2 to 4 when a part is missing or unclear |
| Validation, read-back, and reproducibility | The Task 1.1 command, the Task 5.1 and 5.3 cells, the notebook as a whole, and the commit history | The Task 1.1 cell shows the command that made `file_inventory.txt`; the checks are named and stop the analysis extract from being saved; the read-back prints `True`; the notebook runs top to bottom with its outputs saved; at least two commits with messages that say what changed; no credentials or personal information appear | 2 to 4 when a part is missing or unclear |

A category whose cell still holds its **TODO** text earns 0.
