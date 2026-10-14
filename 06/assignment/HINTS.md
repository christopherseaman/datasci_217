# Hints and Troubleshooting

## Gated Files

- `frame[frame.duplicated(subset=[...], keep=False)].sort_values([...])` shows every row of a repeated key (Lecture 06).
- A row count that differs from the earlier summary points to the step that added or dropped rows. Compare with `visits_summary.csv`, `patients_summary.csv`, and `merge_audit.csv`.
- A raw table that no longer equals its snapshot was changed in place: work on `copy(deep=True)` and rerun from the first cell.

## Task 2: Contract, stack, and audit

- A stack with a column of only missing values means a column name still differs: print `visits_raw.isna().all()` (Lecture 06).

## Task 3: Clean

- After `map()`, `isna().sum()` should equal the empty and `unknown` entries, not more. A larger gap means a spelling is missing from the dictionary, or the values were not stripped and lowercased first (Lecture 05).
- `patients_clean["site_code"].unique()` should hold only codes like `N01`, never `n-01`.
- Repeated visit pairs: fewer than step 6's 240 rows means the `visit` labels were not mapped first, so `wk12` and `week12` looked different.
- After cleaning, `visits_clean.duplicated(subset=["patient_id", "visit"]).sum()` should be 0.
- `visit_date`: compare `pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")` with a text-length check (Lecture 05).

## Task 4: Reshape and join

- Task 4.1: `a1c_long["visit"].unique()` shows only `baseline` and `week12`. Other labels mean the mapping step was skipped.
- Task 4.2: `sites_current["site_code"].is_unique` should be `True`.
- Task 4.3: `pivot()` raises when a pair repeats, so `visits_clean` still holds a recheck. `sbp_wide["patient_id"].is_unique` should be `True`.
- Task 4.4 `MergeError`: the error names the side that repeats.
    - `many_to_one` on `patients_clean` fails when a `patient_id` repeats there: exact registry repeats were not removed in Task 3.1.
    - `many_to_one` on `sites_current` fails when a `site_code` repeats there: a retired record was kept in Task 4.2.
    - `many_to_one` on `a1c_long` fails when a `["patient_id", "visit"]` pair repeats there: the wide table was melted with unmapped labels, or rows were stacked twice.
- A larger `len(analysis)` means the table just merged repeated a key; run the `duplicated()` check on it.
- Dtypes: `analysis.dtypes` should match the table in Task 4.4. `object` where `string` is expected means the `astype()` dictionary missed a column.

## Task 5: Validate, save, and read back

- A repeated `patient_id` and `visit` pair in `analysis` traces to the same repeat in `visits_clean`, a Task 3.2 step 6 miss.
- An `sbp_mmhg` outside 70 through 250 means Task 3.2 step 3 left a value in place.
- A `site_name` that is not current means the join used `sites` instead of `sites_current`.
- The read-back prints `False` when a dtype differs: compare `round_trip.dtypes` with `analysis_sample.dtypes` (Lecture 05).
