# Assignment 08 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/6  clinic counts: columns
         output/clinic_counts.csv is missing; run the Task 1.2 cell to write it, then commit it.
```

A clean run ends with:

```text
[PASS]  4/4  mean wait pivot: empty cell stays empty

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only that check's points.
- A missing column costs the columns check once; values in the remaining columns are still checked. An empty table or one with no recognizable rows earns no value points.
- Spacing, line endings, quoting, column order, and row order never cost points.
- Numbers are compared as numbers, so `6`, `6.0`, and `6.00` are the same value. A mean may keep every digit or be rounded to one or two decimals.
- Clinic names, visit types, IDs, and column names are compared in any letter case.
- An empty cell may be written empty or as `NaN`.
- A leading column of row numbers, which `to_csv()` writes when `index=False` is left out, is ignored.
- A row for Excelsior showing zero visits, which `observed=False` writes, is accepted but not needed; so are Task 2.3's other pairs with zero visits, and an all-empty Excelsior row in the pivot.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/clinic_counts.csv` | Its columns are the four in the Task 1.2 header line. | clinic counts: columns | 6 |
| `output/clinic_counts.csv` | It holds Mission, Sunset, and Bayview, once each. | clinic counts: one row per clinic | 6 |
| `output/clinic_counts.csv` | Each `visit_count` is the clinic's number of visits. | clinic counts: visit_count values | 4 |
| `output/clinic_counts.csv` | Each `satisfaction_count` is the clinic's number of visits with a survey score. | clinic counts: satisfaction_count values | 4 |
| `output/clinic_counts.csv` | Each `patient_count` is the clinic's number of distinct patients. | clinic counts: patient_count values | 4 |
| `output/clinic_summary.csv` | Its columns are the six in the Task 2.1 header line. | clinic summary: columns | 4 |
| `output/clinic_summary.csv` | It holds Mission, Sunset, and Bayview, once each. | clinic summary: one row per clinic | 4 |
| `output/clinic_summary.csv` | Its three counts match Task 1.2's. | clinic summary: count values | 4 |
| `output/clinic_summary.csv` | Each `total_wait_min` is the sum of the clinic's waits. | clinic summary: total_wait_min values | 4 |
| `output/clinic_summary.csv` | Each `mean_wait_min` is the mean of the clinic's waits. | clinic summary: mean_wait_min values | 4 |
| `output/visits_with_context.csv` | Its columns are the eight in the Task 2.2 header line. | visit context: columns | 4 |
| `output/visits_with_context.csv` | It holds V001 to V015, once each. | visit context: one row per visit | 4 |
| `output/visits_with_context.csv` | Each visit's `clinic`, `patient_id`, `visit_type`, `wait_min`, and `satisfaction` match `data/clinic_visits.csv`. | visit context: original visit values | 4 |
| `output/visits_with_context.csv` | Each `clinic_mean_wait` is the mean wait at that visit's clinic. | visit context: clinic_mean_wait values | 4 |
| `output/visits_with_context.csv` | Each `wait_vs_clinic` is `wait_min` minus `clinic_mean_wait`. | visit context: wait_vs_clinic values | 4 |
| `output/clinic_visit_type_summary.csv` | Its columns are the four in the Task 2.3 header line. | clinic and visit type: columns | 4 |
| `output/clinic_visit_type_summary.csv` | It holds each of the eight clinic and visit type pairs with visits once. | clinic and visit type: one row per pair | 5 |
| `output/clinic_visit_type_summary.csv` | Each `visit_count` is the pair's number of visits. | clinic and visit type: visit_count values | 4 |
| `output/clinic_visit_type_summary.csv` | Each `mean_wait_min` is the mean of the pair's waits. | clinic and visit type: mean_wait_min values | 5 |
| `output/mean_wait_pivot.csv` | Its columns are `clinic`, `Follow-up`, `New`, and `Telehealth`. | mean wait pivot: columns | 4 |
| `output/mean_wait_pivot.csv` | It holds Mission, Sunset, and Bayview, once each. | mean wait pivot: one row per clinic | 4 |
| `output/mean_wait_pivot.csv` | Each filled cell is the mean wait for that clinic and visit type. | mean wait pivot: mean waits | 6 |
| `output/mean_wait_pivot.csv` | Sunset's `Telehealth` cell is empty, not 0; a missing row is charged by the row check. | mean wait pivot: empty cell stays empty | 4 |

Extra files are ignored.
