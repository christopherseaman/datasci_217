# Assignment 04 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 2, for example, the loaded-table checks report:

```text
[FIX ]  0/4  bp loaded: patient_id column
         output/bp_loaded.csv is missing; run the Task 2.2 cell to write it with bp.to_csv(LOADED_PATH), then commit it.
[FIX ]  0/4  bp loaded: units row skipped  (same fix as above)
```

A clean run ends with:

```text
[PASS] 10/10 white-coat gap: gaps matched by patient

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only the checks it gets wrong.
- A later file is judged against your own earlier one: if `-999` stayed in `bp_loaded.csv`, only the `-999 read as missing` check loses points, and the values computed from it still count.
- Spacing, line endings, quoting, letter case, column order, and number format (`13`, `13.0`, `13.00`) never cost points; a mean may be rounded to one decimal.
- A leading column of row numbers, which `to_csv()` writes when the index holds only row numbers, is ignored.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/bp_loaded.csv` | It has a `patient_id` column holding the saved index. | bp loaded: patient_id column | 4 |
| `output/bp_loaded.csv` | The units row is not in it. | bp loaded: units row skipped | 4 |
| `output/bp_loaded.csv` | It has no `coordinator_note` column. | bp loaded: coordinator_note left out | 4 |
| `output/bp_loaded.csv` | It holds each of P101 to P114 once. | bp loaded: all 14 patients once | 4 |
| `output/bp_loaded.csv` | The three `-999` readings are blank. | bp loaded: -999 read as missing | 4 |
| `output/bp_loaded.csv` | Every clinic, age, and reading matches the export. | bp loaded: clinic, age, and readings | 4 |
| `output/visit_summary.csv` | It has one row for each of the three visits. | visit summary: one row per visit | 3 |
| `output/visit_summary.csv` | Its `mean` column matches the readings. | visit summary: mean | 5 |
| `output/visit_summary.csv` | Its `median` column matches the readings. | visit summary: median | 5 |
| `output/visit_summary.csv` | Its `count` column counts the readings present. | visit summary: count | 5 |
| `output/clinic_counts.csv` | It has each clinic's number of patients. | clinic counts: patients per clinic | 4 |
| `output/clinic_counts.csv` | The most common clinic comes first. | clinic counts: most common first | 2 |
| `output/followup_priority.csv` | It has a `patient_id` column holding the saved index. | follow-up list: patient_id column | 3 |
| `output/followup_priority.csv` | It holds the seven program patients once each. | follow-up list: program patients | 6 |
| `output/followup_priority.csv` | Its derived columns are named `sbp_mean`, `change_week8`, and `improvement_rank`. | follow-up list: derived column names | 2 |
| `output/followup_priority.csv` | It has no `age` column. | follow-up list: age dropped | 2 |
| `output/followup_priority.csv` | Each `sbp_mean` is the mean of the patient's readings. | follow-up list: sbp_mean values | 4 |
| `output/followup_priority.csv` | Each `change_week8` is week 8 minus baseline. | follow-up list: change_week8 values | 4 |
| `output/followup_priority.csv` | Each `improvement_rank` is the change's rank with `method="min"`. | follow-up list: improvement_rank values | 2 |
| `output/followup_priority.csv` | Its patients run from the largest drop to the smallest. | follow-up list: largest drop first | 3 |
| `output/followup_priority.csv` | Patients with the same change are in `patient_id` order. | follow-up list: ties in patient_id order | 2 |
| `output/followup_priority.parquet` | It is a Parquet file. | follow-up Parquet: Parquet file | 4 |
| `output/followup_priority.parquet` | It has the follow-up list's columns, `patient_id` included. | follow-up Parquet: same columns | 4 |
| `output/white_coat_gap.csv` | It has one row for each of P101 to P115. | white-coat gap: one row per patient in either table | 6 |
| `output/white_coat_gap.csv` | Each gap is that patient's clinic minus home reading, blank where one is missing. | white-coat gap: gaps matched by patient | 10 |

Extra files are ignored.
