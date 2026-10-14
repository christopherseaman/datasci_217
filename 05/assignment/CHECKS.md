# Assignment 05 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 1, for example, the missing-report checks report:

```text
[FIX ]  0/4  missing report: one row per column
         output/missing_report.csv is missing; run the Task 1.1 cell to write it with missing_report.to_csv(MISSING_PATH), then commit it.
[FIX ]  0/5  missing report: missing_count  (same fix as above)
[FIX ]  0/5  missing report: missing_pct  (same fix as above)
```

A clean run ends with:

```text
[PASS]  3/3  clean summary: statistics

Score: 100/100
All checks passed.
```

How the files are read:

- The checks rebuild `data/lab_visits.csv` from `generate_data.py`, run the README's steps on it, and compare your summaries with the result. They never read your notebook or your `data/` folder.
- Each check is scored on its own, so one mistake costs only the checks it gets wrong.
- When Task 5.2's `assert` stops the cell, so `clean_summary.csv` is missing while `decision_log.csv` is saved, only `clean summary: impossible values blanked` loses points, and none when repeats `row_counts.csv` already shows stopped it.
- A later file is judged against your own earlier ones: if `typed_profile.csv` shows `-999` kept, only the `-999 read as missing` check loses points, and the validation report, fences, and summaries built from that data still count.
- Line endings, quoting, letter case, column order, row order, number format (`54`, `54.0`), and date format (`2024-01-02`, `2024-01-02 00:00:00`, `2024/01/02`, `January 02, 2024`) never cost points; a percent may be unrounded or rounded to a whole number, a blank may be written as `missing` (`to_csv(na_rep="missing")`), an indicator may be `1`/`0` or `True`/`False`, and `clean_summary.csv` may be saved with or without `.T`.
- Spaces around a label cost points only in `clinic_counts.csv` and the `patient_id` of `coded_sample.csv`, where Task 3.1 strips them.
- A leading column of row numbers, which `to_csv()` writes without `index=False`, is ignored.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/missing_report.csv` | It has one row for each of the eight columns of `data/lab_visits.csv`. | missing report: one row per column | 4 |
| `output/missing_report.csv` | Its `missing_count` column counts the empty cells of the file as read. | missing report: missing_count | 5 |
| `output/missing_report.csv` | Its `missing_pct` column is each count as a percent of the rows read. | missing report: missing_pct | 5 |
| `output/row_counts.csv` | `read` is the rows of the file, and `with_labs` the rows left after dropping the visits with neither lab result. | row counts: no-lab visits dropped | 4 |
| `output/row_counts.csv` | `unique` is the rows left after removing exact repeats, keeping one copy of each. | row counts: exact repeats removed | 5 |
| `output/row_counts.csv` | `unique` keeps every patient's repeat visits. | row counts: repeat visits kept | 3 |
| `output/typed_profile.csv` | No `min` is `-999`, and `weight_kg` and `glucose_mg_dl` count every `-999` as missing. | typed profile: -999 read as missing | 4 |
| `output/typed_profile.csv` | `age` has a number dtype. | typed profile: age as numbers | 3 |
| `output/typed_profile.csv` | `age` counts every `unknown` as missing, not filled with a number. | typed profile: unreadable ages left blank | 2 |
| `output/typed_profile.csv` | `visit_date` is `datetime64`, with the export's first and last dates, and counts the blank and impossible dates as missing. | typed profile: visit dates parsed | 3 |
| `output/typed_profile.csv` | The visits without a real date are still in `labs`. | typed profile: undated visits kept | 2 |
| `output/typed_profile.csv` | Every other `min`, `max`, and `mean` matches the data. | typed profile: other values unchanged | 3 |
| `output/clinic_counts.csv` | Each clinic has one spelling without spaces, and its count matches. | clinic counts: clinic names cleaned | 4 |
| `output/clinic_counts.csv` | The blank clinics are counted as `Unknown`. | clinic counts: missing clinic filled | 3 |
| `output/gender_counts.csv` | It counts `F`, `M`, `X`, and blank, the codes `map()` leaves. | gender counts: gender codes mapped | 4 |
| `output/a1c_counts.csv` | It counts each HbA1c band and the visits with none. | a1c counts: a1c_category | 5 |
| `output/coded_sample.csv` | It holds every visit of the sample patients, with IDs in upper case and without spaces. | coded sample: patient IDs cleaned | 4 |
| `output/coded_sample.csv` | It has one `clinic_` column per clinic, 1 for the row's own clinic and 0 for the others. | coded sample: clinic indicators | 4 |
| `output/validation_report.csv` | It has one row for each of the five rules. | validation report: one row per rule | 3 |
| `output/validation_report.csv` | Each rule keeps its Task 4.1 name; letter case and spaces may differ. | validation report: rule names | 2 |
| `output/validation_report.csv` | Its `failing_rows` column counts the visits that break each rule. | validation report: failing rows | 5 |
| `output/glucose_fences.csv` | `q1`, `q3`, `lower`, and `upper` are the glucose quartiles and the 1.5 * IQR fences. | glucose fences: quartiles and fences | 3 |
| `output/glucose_fences.csv` | `outliers` counts the glucose values outside the saved fences, blanks not included. | glucose fences: outliers flagged | 2 |
| `output/decision_log.csv` | `visits_dropped` counts the visits without a valid ID or a real date. | decision log: failing visits dropped | 5 |
| `output/decision_log.csv` | `hba1c_blanked` and `weight_blanked` count the kept values outside the contract's ranges. | decision log: out-of-range values counted | 2 |
| `output/clean_summary.csv` | Every HbA1c and weight `min` and `max` is inside the contract's range. | clean summary: impossible values blanked | 5 |
| `output/clean_summary.csv` | Each `count` is the values present in the cleaned column, blanks not filled. | clean summary: blanks kept | 3 |
| `output/clean_summary.csv` | Each `mean`, `min`, `50%`, and `max` matches the cleaned table. | clean summary: statistics | 3 |

Extra files are ignored.
