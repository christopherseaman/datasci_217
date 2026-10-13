# Assignment 06 checks

## Completion contract

The midterm totals 100 points: 75 points graded from your committed files after the deadline, 25 by human review. Each file is graded on its own, so a wrong value costs only its own points.

| File | What earns the points | Points |
| --- | --- | ---: |
| `output/file_inventory.txt` | 1 for each data file's line count | 5 |
| `output/table_profile.csv` | 1 for each table with its right `rows`, `columns`, and `exact_duplicate_rows` | 5 |
| `output/issue_audit.csv` | 1 for each issue's count | 10 |
| `output/patients_clean.csv` | 2 for one row per patient; 2 each for `sex`, `birth_year`, and `site_code`, in proportion to the patients whose value follows the cleaning rules (rounded down) | 8 |
| `output/visits_clean.csv` | 4 for one row per visit, its `visit` label stripped and lowercased; 3 each for `visit_date` and `sbp_mmhg`, 2 each for `weight_kg` and `source_file`, in proportion to the visits whose value follows the rules (rounded down) | 14 |
| `output/a1c_long.csv` | 2 for one row per result; 3 for `a1c_pct`, in proportion to the right results | 5 |
| `output/merge_audit.csv` | 1 for each check's count | 7 |
| `output/analysis_visits.csv` | 3 for one row per clean visit; 3 each for `site_name` and `a1c_pct`, 2 for `sex` with `birth_year`, in proportion to the right visits | 11 |
| `output/sbp_wide.csv` | 2 for one row per patient with a visit; 1 each for `baseline` and `week12`; 2 for `sbp_change`, in proportion to the right patients | 6 |
| `output/decision_log.csv` | 3 for the eight decisions' `table` and `decision`, in proportion; 1 for a reason on every row | 4 |

## How the files are read

- Line endings, blank lines, spaces at the end of a line or around a header name, a comma, semicolon, or tab between cells, column order, extra columns, a leading column of row numbers, and number format (`6`, `6.0`) do not matter. Letter case and surrounding spaces in labels, IDs, and text values do not cost points, except in the three columns whose rule is that normalization: `sex` and `site_code` in `patients_clean.csv` and `visit` in `visits_clean.csv` must hold the normalized text, such as `F` for ` f`.
- An empty field, `NaN`, and `<NA>` all read as missing; a date may carry a `00:00:00` time after it.
- A later file is also right when it follows from your own earlier one, so one mistake is not charged twice: `analysis_visits.csv` against your `visits_clean.csv`, `patients_clean.csv`, and `a1c_long.csv`; the first six `merge_audit.csv` counts against those same files, matched exactly as your merges match them; `sbp_wide.csv` against your `visits_clean.csv`; and `sbp_change` against your own `baseline` and `week12` columns.

## Human review

Human review reads the notebook, the decision reasons, and your commit history, 5 points for each category:

| Category | What it reads | Full credit (5) | Partial credit |
| --- | --- | --- | --- |
| Lecture 01 to 06 evidence map | The "Cumulative midterm checkpoint" cell | One entry for each of Lectures 01 to 06, each naming a concrete technique or file from that lecture and saying in one sentence where this notebook uses it | 2 to 4 for four or five lectures, or for entries that name a tool without saying where it is used here |
| Data contract | The Task 2.1 cell | States each table's grain and key (single or composite) and the cardinality of each Task 4.3 join, each fitting these files | 2 to 4 when one or two parts are missing or do not fit these files |
| Decision reasons | The `reason` column of `output/decision_log.csv` | Each of the eight reasons names the problem in these files, such as the rows or values it affects, and why its decision fits | 2 to 4 when some reasons are generic, such as "to clean the data", or do not match their decision |
| Join and reshape safeguards | The Task 3.2 step 6 cell and the Task 4.2 and 4.3 cells | Shows the repeated visit pair before keeping the later reading, the caught `MergeError` and the repeated site rows, the `_merge` counts, and a `validate=` on each Task 4.3 merge | 2 to 4 when a part is missing or unclear |
| Validation, read-back, and reproducibility | The Task 1.1 command, the Task 5.1 and 5.3 cells, the notebook as a whole, and the commit history | The Task 1.1 cell shows the command that made `file_inventory.txt`; the checks are named and stop the save; the read-back prints `True`; the notebook runs top to bottom with its outputs saved; at least two commits with messages that say what changed; no credentials or personal information appear | 2 to 4 when a part is missing or unclear |

A category whose cell still holds its **TODO** text earns 0.
