# Assignment 05 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 1, for example, the missing-report checks report:

```text
[FIX ]  0/4  missing report: one row per column
         output/missing_report.csv is missing; run the Task 1.1 cell to write it with missing_report.to_csv(MISSING_PATH), then commit it.
[FIX ]  0/5  missing report: missing_count  (same fix as above)
[FIX ]  0/4  missing report: missing_pct  (same fix as above)
```

A clean run ends with:

```text
[PASS]  4/4  clean summary: statistics

Score: 100/100
All checks passed.
```

How the files are read:

- The checks rebuild `data/lab_visits.csv` from `generate_data.py`, run the README's steps on it, and compare your summaries with the result. They never read your notebook or your `data/` folder.
- Each check is scored on its own, so one mistake costs only the checks it gets wrong.
- If the Task 5.2 `assert` stops the cell, `clean_summary.csv` is not written and only its checks lose points.
- A file is judged against your own other files: if `typed_profile.csv` shows `-999` kept, only the `-999 read as missing` check loses points, and the validation report, decision log, and summary built from that data still count.
- Line endings, quoting, letter case, spaces around a cell, column order, row order, number format (`54`, `54.0`), and date format (`2024-01-02`, `2024/01/02`) never cost points; a percent may be unrounded or rounded to a whole number, a blank may be written as `missing` (`to_csv(na_rep="missing")`), and `clean_summary.csv` may be saved with or without `.T`.
- A leading column of row numbers, which `to_csv()` writes without `index=False`, is ignored.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/missing_report.csv` | It has one row for each of the eight columns of `data/lab_visits.csv`. | missing report: one row per column | 4 |
| `output/missing_report.csv` | Its `missing_count` column counts the empty cells of the file as read. | missing report: missing_count | 5 |
| `output/missing_report.csv` | Its `missing_pct` column is each count as a percent of the rows read. | missing report: missing_pct | 4 |
| `output/typed_profile.csv` | No `min` is `-999`, and `weight_kg` and `glucose_mg_dl` count every `-999` as missing. | typed profile: -999 read as missing | 4 |
| `output/typed_profile.csv` | `age` has a number dtype. | typed profile: age as numbers | 4 |
| `output/typed_profile.csv` | `age` counts every `unknown` as missing, not filled with a number. | typed profile: unreadable ages left blank | 3 |
| `output/typed_profile.csv` | `visit_date` is `datetime64`, with the export's first and last dates, and counts the blank and impossible dates as missing. | typed profile: visit dates parsed | 3 |
| `output/typed_profile.csv` | The visits without a real date are still in `labs`. | typed profile: undated visits kept | 2 |
| `output/typed_profile.csv` | Every other `min`, `max`, and `mean` matches the data. | typed profile: other values unchanged | 3 |
| `output/category_counts.csv` | Each clinic has one spelling, and its count matches. | category counts: clinic names cleaned | 4 |
| `output/category_counts.csv` | `clinic_Unknown` counts the visits the export left without a clinic. | category counts: missing clinic filled | 3 |
| `output/category_counts.csv` | Gender has only `F`, `M`, and `X`, the codes `map()` leaves, with matching counts. | category counts: gender codes mapped | 4 |
| `output/category_counts.csv` | It counts the visits in each HbA1c band. | category counts: a1c_category | 5 |
| `output/category_counts.csv` | Each label is a `get_dummies()` column name, such as `clinic_Mission`. | category counts: indicator columns | 3 |
| `output/validation_report.csv` | It has one row for each of the six rules. | validation report: one row per rule | 3 |
| `output/validation_report.csv` | Each rule keeps its Task 4 name. | validation report: rule names | 2 |
| `output/validation_report.csv` | `patient_id_format` counts the IDs that fail `D` and five digits after Task 3.1 strips and upper-cases them. | validation report: patient IDs cleaned | 4 |
| `output/validation_report.csv` | The date, gender, HbA1c, and weight rules count the visits that break them. | validation report: failing rows | 5 |
| `output/validation_report.csv` | `glucose_within_fences` counts the glucose values outside the 1.5 * IQR fences, blanks not included. | validation report: glucose outliers flagged | 3 |
| `output/decision_log.csv` | `rows_read` is the rows of the file, and `rows_with_labs` the rows left after dropping the visits with neither lab result. | decision log: no-lab visits dropped | 4 |
| `output/decision_log.csv` | `rows_unique` is the rows left after removing exact repeats, keeping one copy of each. | decision log: exact repeats removed | 5 |
| `output/decision_log.csv` | `rows_unique` keeps every patient's repeat visits. | decision log: repeat visits kept | 3 |
| `output/decision_log.csv` | `visits_dropped` counts the visits without a valid ID or a real date. | decision log: failing visits dropped | 5 |
| `output/decision_log.csv` | `hba1c_blanked` and `weight_blanked` count the kept values outside the contract's ranges. | decision log: out-of-range values counted | 3 |
| `output/clean_summary.csv` | Every HbA1c and weight `min` and `max` is inside the contract's range. | clean summary: impossible values blanked | 5 |
| `output/clean_summary.csv` | Each `count` is the values present in the cleaned column, blanks not filled. | clean summary: blanks kept | 3 |
| `output/clean_summary.csv` | Each `mean`, `min`, `50%`, and `max` matches the cleaned table. | clean summary: statistics | 4 |

Extra files are ignored.
