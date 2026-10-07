# Assignment 05 checks

## Completion contract

The midterm totals 100 points: 75 points graded from your committed files after the deadline, 25 by human review. Each file is graded on its own, so a wrong value costs only its own points.

| File | What earns the points | Points |
| --- | --- | ---: |
| `output/raw_preview.txt` | 2 for the `head` label with the four lines it prints, 2 for the `tail` label with the two lines it prints | 4 |
| `output/pipeline_summary.txt` | 1 for each key's count | 5 |
| `output/numpy_age_summary.csv` | 1 for each metric's value | 5 |
| `output/pandas_selection.csv` | 1 for each requested record with its raw `site` and `status`, less 1 for each other row; 1 for exactly the three columns | 4 |
| `output/issue_audit.csv` | 1 for each issue's count | 15 |
| `output/cleaned_people.csv` | 4 for each of the seven columns, in proportion to the records whose value in that column follows the cleaning rules (rounded down) | 28 |
| `output/decision_log.csv` | 1 for each decision with its `field`, `issue`, and `action`; 2 for a reason on every row; 1 each for `source`, `source_sha256`, `rows_before`, and `rows_after` on every row | 14 |

## How the files are read

- Line endings, blank lines, spaces at the end of a line or around a header name, a comma, semicolon, or tab between cells, spaces that pad every separator in the header line and the rows alike, column order, a leading column of row numbers, and number format (`6`, `6.0`) do not matter. Letter case and surrounding spaces in labels and cleaned text values do not cost points: `north`, `North`, and ` north ` represent the same site. Save the normalized forms the cleaning rules describe.
- `True`/`False`, `true`/`false`, `1`/`0`, and `yes`/`no` all read as booleans; an empty field, `NaN`, and `<NA>` all read as missing; a date may carry a `00:00:00` time after it.
- `needs_review` is also right when it follows the rule from your own `age` and `visit_date` columns, and `rows_after` is also right when it equals the rows in your own `cleaned_people.csv`, so one mistake is not charged twice.

## Human review

Human review reads the notebook and the decision reasons, 5 points for each category:

| Category | What it reads | Full credit (5) | Partial credit |
| --- | --- | --- | --- |
| Lecture 01 to 05 evidence map | The "Cumulative midterm checkpoint" cell | One entry for each of Lectures 01 to 05, each naming a concrete technique or file from that lecture and saying in one sentence where this notebook uses it | 2 to 4 for three or four lectures, or for entries that name a tool without saying where it is used here |
| Data contract | The Task 2.1 cell | States the row meaning and the candidate identifier, says how the raw table differs from the cleaned table, and defines all six terms in your own words, each in a way that fits this file | 2 to 4 when one or two parts are missing, or a definition is wrong or does not fit this file |
| Decision reasons | The `reason` column of `output/decision_log.csv` | Each of the eight reasons names the problem in this file, such as the values or records it affects, and why its action fits | 2 to 4 when some reasons are generic, such as "to clean the data", or do not match their decision |
| What you will not do | The Task 3.2 cell | Explains why filling across these person records is wrong, why flagged values stay missing for review, and why clean means meeting the contract, each tied to this file | 2 to 4 when one of the three points is missing or stays general |
| Validation, read-back, and reproducibility | The Task 1.1 commands, the Task 4.1 and 4.3 cells, and the notebook as a whole | The Task 1.1 cell shows the commands that made `raw_preview.txt`; the checks are named and stop the save; each read-back prints `True`; the notebook runs top to bottom with its outputs saved; no credentials or personal information appear | 2 to 4 when a part is missing or unclear |

A category whose cell still holds its **TODO** text earns 0.
