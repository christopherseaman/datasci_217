# Assignment 06 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/8  merge audit: columns
         output/specimen_merge_audit.csv is missing; run the Task 1.2 cell to write it, then commit it.
```

A clean run ends with:

```text
[PASS]  4/4  SBP round trip: followup values

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only that check's points. The round trip also accepts the pivot of your own long table, so a mistake in Task 3.1 is not charged again in Task 3.2.
- Any two nonempty source labels that distinguish batch A from batch B count.
- Spacing, line endings, quoting, column order, and row order never cost points.
- Numbers are compared as numbers, so `5`, `5.0`, and `5.00` are the same value.
- IDs, labels, and column names are compared in any letter case.
- An empty cell may be written empty or as `NaN`.
- A leading column of row numbers, which `to_csv()` writes when `index=False` is left out, is ignored.
- A first column with no header that holds the IDs, which `to_csv()` writes for an index without a name, counts as the ID column.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/specimen_merge_audit.csv` | Its columns are the nine in the Task 1.2 header line; `record_status` may be there too. | merge audit: columns | 8 |
| `output/specimen_merge_audit.csv` | It holds SP101 to SP107, once each. | merge audit: one row per specimen | 8 |
| `output/specimen_merge_audit.csv` | Each specimen's own columns match `data/specimens.csv`. | merge audit: specimen values | 8 |
| `output/specimen_merge_audit.csv` | Each specimen has its clinic's current `clinic_name` and `region`, and SP106's are empty. | merge audit: current clinic names and regions | 8 |
| `output/specimen_merge_audit.csv` | `_merge` is `left_only` for SP106 and `both` for the other six. | merge audit: _merge indicator | 8 |
| `output/combined_specimens.csv` | Its columns are the seven in the Task 2.1 header line. | combined specimens: columns | 3 |
| `output/combined_specimens.csv` | It holds SP101 to SP107, once each. | combined specimens: one row per specimen | 4 |
| `output/combined_specimens.csv` | Each row's values match its batch file. | combined specimens: specimen values | 4 |
| `output/combined_specimens.csv` | `source_partition` consistently distinguishes SP101 to SP104 from SP105 to SP107 with two nonempty labels, such as `batch_a` and `batch_b`. | combined specimens: source_partition labels | 4 |
| `output/aligned_features.csv` | Its columns are `specimen_id`, `volume_ml`, and `transit_min`. | aligned features: columns | 3 |
| `output/aligned_features.csv` | It holds SP101 to SP104 and SP108, once each. | aligned features: one row per specimen | 4 |
| `output/aligned_features.csv` | `volume_ml` matches batch A, and SP108's is empty. | aligned features: volume_ml values | 4 |
| `output/aligned_features.csv` | `transit_min` matches `data/transit_times.csv`, and SP101's and SP104's are empty. | aligned features: transit_min values | 4 |
| `output/sbp_long.csv` | Its columns are `patient_id`, `visit`, and `sbp`. | SBP long: columns | 5 |
| `output/sbp_long.csv` | It holds each of the eight patient and visit pairs once. | SBP long: one row per patient and visit | 5 |
| `output/sbp_long.csv` | Each `sbp` matches that patient's reading at that visit in `data/sbp_wide.csv`. | SBP long: sbp values | 5 |
| `output/sbp_round_trip.csv` | Its columns are `patient_id`, `baseline`, and `followup`. | SBP round trip: columns | 3 |
| `output/sbp_round_trip.csv` | It holds P201 to P204, once each. | SBP round trip: one row per patient | 4 |
| `output/sbp_round_trip.csv` | Each `baseline` matches `data/sbp_wide.csv`. | SBP round trip: baseline values | 4 |
| `output/sbp_round_trip.csv` | Each `followup` matches `data/sbp_wide.csv`. | SBP round trip: followup values | 4 |

Extra files are ignored.
