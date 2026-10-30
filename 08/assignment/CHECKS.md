# Assignment 08 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/4  site summary: columns
         output/site_summary.csv is missing; run the Task 1.2 cell to write it, then commit it.
```

A clean run ends with:

```text
[PASS]  4/4  median report pivot: empty cell stays empty

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only that check's points.
- A mistake is charged once. A missing column, or one saved under another name with the right values, costs only the columns check; values in the other columns are still checked. An empty table or one with no recognizable rows earns no value points.
- Spacing, line endings, quoting, column order, and row order never cost points.
- Numbers are compared as numbers, so `6`, `6.0`, and `6.00` are the same value. A mean or median may keep every digit or be rounded to one or two decimals.
- Site names, modalities, IDs, and column names are compared in any letter case.
- An empty cell may be written empty or as `NaN`.
- Row-number columns, which `to_csv()` writes when `index=False` is left out and which `apply` keeps as an inner index level, are ignored.
- A row for Eastport showing zero exams, which `observed=False` writes, is accepted but not needed; so are Task 2.1's other pairs with zero exams, and an all-empty Eastport row in the pivot.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/site_summary.csv` | Its columns are the six in the Task 1.2 header line. | site summary: columns | 4 |
| `output/site_summary.csv` | It holds the eight sites with exams, once each. | site summary: one row per site | 4 |
| `output/site_summary.csv` | Each `exam_count` is the site's number of exams. | site summary: exam_count values | 4 |
| `output/site_summary.csv` | Each `report_count` is the site's number of exams with a signed report. | site summary: report_count values | 4 |
| `output/site_summary.csv` | Each `patient_count` is the site's number of distinct patients. | site summary: patient_count values | 4 |
| `output/site_summary.csv` | Each `total_scan_min` is the sum of the site's scan minutes. | site summary: total_scan_min values | 4 |
| `output/site_summary.csv` | Each `mean_scan_min` is the mean of the site's scan minutes. | site summary: mean_scan_min values | 4 |
| `output/site_modality_summary.csv` | Its columns are the five in the Task 2.1 header line. | site and modality: columns | 4 |
| `output/site_modality_summary.csv` | It holds each of the 31 site and modality pairs with exams once. | site and modality: one row per pair | 4 |
| `output/site_modality_summary.csv` | Each `exam_count` is the pair's number of exams. | site and modality: exam_count values | 4 |
| `output/site_modality_summary.csv` | Each `median_wait_hr` is the median of the pair's waits. | site and modality: median_wait_hr values | 4 |
| `output/site_modality_summary.csv` | Each `median_report_hr` is the median of the pair's report times. | site and modality: median_report_hr values | 4 |
| `output/delayed_stat_exams.csv` | Its columns are the six in the Task 2.2 header line. | delayed STAT exams: columns | 4 |
| `output/delayed_stat_exams.csv` | It holds the 190 STAT exams that waited over 8 hours, once each, and no other exam. | delayed STAT exams: one row per delayed exam | 4 |
| `output/delayed_stat_exams.csv` | Each exam's `site`, `modality`, and `wait_hr` match `data/imaging_exams.parquet`. | delayed STAT exams: original exam values | 4 |
| `output/delayed_stat_exams.csv` | Each `peer_median_wait_hr` is the median wait of all STAT exams at that site and modality. | delayed STAT exams: peer_median_wait_hr values | 4 |
| `output/delayed_stat_exams.csv` | Each `wait_vs_peer_hr` is `wait_hr` minus `peer_median_wait_hr`. | delayed STAT exams: wait_vs_peer_hr values | 4 |
| `output/ct_dose_review.csv` | Its columns are the three in the Task 2.3 header line. | CT dose review: columns | 2 |
| `output/ct_dose_review.csv` | It holds the 1,253 CTs of the 159 patients whose CT doses add up to at least 100 mSv, once each, and no other exam. | CT dose review: whole patients kept | 4 |
| `output/ct_dose_review.csv` | Each exam's `patient_id` and `dose_msv` match `data/imaging_exams.parquet`. | CT dose review: original exam values | 2 |
| `output/slowest_stat_reports.csv` | Its columns are the four in the Task 2.4 header line. | slowest STAT reports: columns | 2 |
| `output/slowest_stat_reports.csv` | It holds each site's three STAT exams with the longest `report_hr`, 24 in all, and no other exam. | slowest STAT reports: three per site | 4 |
| `output/slowest_stat_reports.csv` | Each exam's `site`, `modality`, and `report_hr` match `data/imaging_exams.parquet`. | slowest STAT reports: original exam values | 2 |
| `output/median_report_pivot.csv` | Its columns are `site`, `CT`, `MRI`, `Ultrasound`, and `X-ray`. | median report pivot: columns | 4 |
| `output/median_report_pivot.csv` | It holds the eight sites with exams, once each. | median report pivot: one row per site | 4 |
| `output/median_report_pivot.csv` | Each filled cell is the median report time for that site and modality. | median report pivot: median reports | 4 |
| `output/median_report_pivot.csv` | Valley's `MRI` cell is empty, not 0; a missing row is charged by the row check. | median report pivot: empty cell stays empty | 4 |

Extra files are ignored.
