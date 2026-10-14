# Hints and Troubleshooting

Use these after trying the task yourself. Each names the lecture that taught the tool and a check that finds the cause. None gives a graded value.

## Gated Files

- A `MergeError` in Task 4.4 or a failed Task 5.1 check stops the notebook before `analysis_sample.csv` is written. Fix the cause upstream and run again; keep `validate=` and the assert.
- `print(checks[~checks])` lists the failed checks (Lecture 05's "Stop before saving a bad table").
- `frame[frame.duplicated(subset=[...], keep=False)].sort_values([...])` shows every row of a repeated key (Lecture 06).
- A row count that differs from the earlier summary points to the step that added or dropped rows. Compare with `visits_summary.csv`, `patients_summary.csv`, and `merge_audit.csv`.

## Task 1: Inventory the files

- Each `rows` in `table_profile.csv` is the `wc -l` count minus 1 for the header (Lecture 03).
- `wc -l data/*.csv > output/file_inventory.txt` sorts by file name and ends with a `total` line.

## Task 2: Contract, stack, and audit

- A stack with a column of only missing values means a column name still differs: print `visits_raw.isna().all()` (Lecture 06).
- Count with `mask.sum()` on the raw columns, before any cleaning. Build `gender` and `site_code` masks from `.str.strip()` and `~isin([...])` (Lecture 05).
- `999` is counted once, as the sentinel. Make the outside-range mask exclude it.
- A `keep=False` count includes the first copy, so it is 0 or at least 2.
- If the notebook's snapshot comparison turns `False`, a cell changed `raw` in place; work on `copy(deep=True)`.

## Task 3: Clean

- `value_counts()` before mapping shows every spelling. After `map()`, `isna().sum()` should equal the empty and `unknown` entries, not more; a larger gap means a spelling is missing from the dictionary (Lecture 05).
- Strip and lowercase before `map()`, or `"M "` becomes missing.
- `patients_clean["site_code"].unique()` should hold only codes like `N01`, never `n-01`. Remove hyphens and spaces after uppercasing.
- `patients_clean["patient_id"].is_unique` should be `True`. If not, the exact duplicate rows were not removed.
- `birth_year` of `9999` must be gone before `min()` and `max()`.
- Repeated visit pairs: step 6 shows 240 rows. Fewer means the `visit` labels were not mapped first, so `wk12` and `week12` looked different. Sort by `visit_time`, then `drop_duplicates(..., keep="last")` (Lecture 06).
- After cleaning, `visits_clean.duplicated(subset=["patient_id", "visit"]).sum()` should be 0.
- `visit_date`: compare `pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")` with a text-length check (Lecture 05).
- `sbp_mmhg` after cleaning: `min()` at least 70, `max()` at most 250.

## Task 4: Reshape and join

- Task 4.1: `a1c_long["visit"].unique()` shows only `baseline` and `week12`. Other labels mean the mapping step was skipped (Lecture 06's `melt()`).
- Task 4.2: the `sites.csv` merge raises `MergeError` while a `site_code` repeats; catch it, show the repeated rows, and keep only `current` records. `sites_current["site_code"].is_unique` should be `True`.
- The first two `merge_audit.csv` counts add up to `len(visits_clean)`; the first and third add up to `len(a1c_long)`.
- Task 4.3: `pivot()` raises when a pair repeats, so `visits_clean` still holds a recheck. `sbp_wide["patient_id"].is_unique` should be `True`.
- Task 4.4 `MergeError`: the error names the side that repeats.
    - `many_to_one` on `patients_clean` fails when a `patient_id` repeats there: exact registry repeats were not removed in Task 3.1.
    - `many_to_one` on `sites_current` fails when a `site_code` repeats there: a retired record was kept in Task 4.2.
    - `many_to_one` on `a1c_long` fails when a `["patient_id", "visit"]` pair repeats there: the wide table was melted with unmapped labels, or rows were stacked twice.
- After each merge, `len(analysis)` equals `len(visits_clean)`. A larger count means that table repeated a key; run the check on that table before merging again.
- Dtypes: `analysis.dtypes` should match the table in Task 4.4. `object` where `string` is expected means the `astype()` dictionary missed a column.

## Task 5: Validate, save, and read back

- Each check is one `True`/`False`. Display the Series and read the name of each `False`; every name points to a task above.
- `analysis` rows differing from `visits_clean` is a Task 4.4 merge that repeated a key.
- A repeated `patient_id` and `visit` pair in `analysis` traces to the same repeat in `visits_clean`, a Task 3.2 step 6 miss.
- An `sbp_mmhg` outside 70 through 250 means Task 3.2 step 3 left a value in place.
- A `site_name` that is not current means the join used `sites` instead of `sites_current`.
- A raw table that no longer equals its snapshot was changed in place; rerun from the first cell and work on copies.
- The read-back prints `False` when a dtype differs: compare `round_trip.dtypes` with `analysis_sample.dtypes` (Lecture 05).
