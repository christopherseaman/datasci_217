# Assignment 05 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 1, for example, the missing-report checks report:

```text
[FIX ]  0/4  missing report: one row per column
         output/missing_report.csv is missing; run the Task 1.1 cell to write it with missing_report.to_csv(MISSING_PATH), then commit it.
[FIX ]  0/6  missing report: missing_count  (same fix as above)
[FIX ]  0/5  missing report: missing_pct  (same fix as above)
```

A clean run ends with:

```text
[PASS]  5/5  clean table: other values kept

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only the checks it gets wrong.
- A later file is judged against your own earlier one: if `-999` stayed in `labs_typed.csv`, only the `-999 read as missing` check loses points, and the validation report and cleaned table built from it still count.
- Line endings, quoting, letter case, column order, row order, number format (`54`, `54.0`), and a year-first or month-first date (`2026-03-02`, `2026-03-02 00:00:00`, `2026/03/02`, `20260302`, `3/2/2026`) never cost points; a percent may be unrounded, and an indicator or flag may be `1`/`0` or `True`/`False`.
- Spaces around a cell cost points only in `patient_id` and `clinic` of `labs_coded.csv`, where Task 3.1 strips them.
- A leading column of row numbers, which `to_csv()` writes without `index=False`, is ignored.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/missing_report.csv` | It has one row for each of the eight columns of `data/lab_visits.csv`. | missing report: one row per column | 4 |
| `output/missing_report.csv` | Its `missing_count` column counts the empty cells of the file as read. | missing report: missing_count | 6 |
| `output/missing_report.csv` | Its `missing_pct` column is each count as a percent of the 22 rows. | missing report: missing_pct | 5 |
| `output/labs_typed.csv` | D2011, the visit with no lab results, is gone, and every visit with one result stays. | typed table: no-lab visit dropped | 4 |
| `output/labs_typed.csv` | D2006 and D2016, each entered twice, appear once. | typed table: exact repeats removed | 5 |
| `output/labs_typed.csv` | Both of D2001's visits stay. | typed table: repeat visit kept | 3 |
| `output/labs_typed.csv` | The four `-999` cells are blank. | typed table: -999 read as missing | 5 |
| `output/labs_typed.csv` | Every age is a number or blank. | typed table: age as numbers | 4 |
| `output/labs_typed.csv` | Every date is the visit's date, and D2009's `2026-02-30` is blank. | typed table: visit dates parsed | 4 |
| `output/labs_typed.csv` | Every other age, weight, HbA1c, and glucose matches the export. | typed table: other values unchanged | 3 |
| `output/labs_coded.csv` | Every patient ID is upper case, without spaces around it. | coded table: patient IDs cleaned | 4 |
| `output/labs_coded.csv` | Every clinic has one spelling, without spaces around it. | coded table: clinic names cleaned | 4 |
| `output/labs_coded.csv` | D2008's blank clinic is `Unknown`. | coded table: missing clinic filled | 3 |
| `output/labs_coded.csv` | Every sex is `F` or `M`, and D2012's `U` is blank. | coded table: sex codes mapped | 4 |
| `output/labs_coded.csv` | Each `a1c_category` is the HbA1c's band, blank where it has none. | coded table: a1c_category | 5 |
| `output/labs_coded.csv` | It has one `clinic_` column per clinic, 1 for the row's own clinic and 0 for the others. | coded table: clinic indicators | 4 |
| `output/validation_report.csv` | It has one row for each of the five rules. | validation report: one row per rule | 3 |
| `output/validation_report.csv` | Its `failing_rows` column counts the visits that break each rule. | validation report: failing rows | 6 |
| `output/labs_clean.csv` | It holds the 16 visits with a valid ID and a real date. | clean table: failing visits dropped | 5 |
| `output/labs_clean.csv` | Each visit appears once. | clean table: one row per visit | 4 |
| `output/labs_clean.csv` | D2007's HbA1c and D2010's weight are blank. | clean table: impossible values blanked | 5 |
| `output/labs_clean.csv` | Its `glucose_outlier` column is `True` exactly for a glucose outside the IQR fences. | clean table: glucose_outlier flag | 5 |
| `output/labs_clean.csv` | Every other age, weight, HbA1c, and glucose matches the coded table, blanks included. | clean table: other values kept | 5 |

Extra files are ignored.
