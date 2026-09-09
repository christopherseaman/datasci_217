# Assignment 05: Documented Cleaning Pipeline

Build one reproducible data pipeline for a small, synthetic table. The midterm
is cumulative through Lecture 05: establish a terminal/Python foundation,
summarize with NumPy, select with pandas, then document and validate cleaning.
Work in this order:

1. establish the terminal/Python, NumPy, and pandas foundation artifacts;
2. define the data contract and audit the untouched raw table;
3. record decisions, then transform a copy;
4. validate, save, and read the artifacts back with explicit schemas.

This is a local-first assignment. Grading reads committed artifacts without executing the notebook. Do not use
Colab, manual uploads, Drive mounts, network access, or `/content` paths. The supplied path finder supports
both a standalone exported assignment repository and this course repository.

## Setup

Use Python 3.14. From this directory, create and activate a virtual
environment, and install the exact dependency records. Open the supplied
notebook through Jupyter or the VS Code notebook interface:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Complete
the checks in [PLATFORM_CHECK.md](PLATFORM_CHECK.md) before editing the
notebook.

## Deliverables

Complete every `TODO` in `assignment.ipynb`, including the cumulative midterm
checkpoint, and commit the notebook source. Create and commit these four
cumulative foundation artifacts before the cleaning artifacts:

- `output/raw_preview.txt` — the exact `head -n 4` and `tail -n 2` terminal
  evidence shown in the notebook contract.
- `output/pipeline_summary.txt` — these exact lines, produced by a small Python
  summary function:

  ```text
  raw_rows=12
  raw_columns=6
  exact_duplicate_rows=1
  candidate_id_duplicate_rows=1
  clean_rows=11
  ```

- `output/numpy_age_summary.csv` — header `metric,value`, followed by the
  `count`, `min`, `max`, `sum`, and `mean` of the valid integer ages in that
  order (`6`, `0`, `52`, `198`, `33.0`).
- `output/pandas_selection.csv` — header `record_id,site,status`, followed by
  the raw rows for `R001`, `R003`, and `R010` in that order.

Then create and commit these three cleaning milestone artifacts:

- `output/issue_audit.csv`
- `output/cleaned_people.csv`
- `output/decision_log.csv`

Do not edit `data/people_raw.csv`, `data/fixture.json`, or the supplied setup
and final-verification notebook cells. The saved artifacts are the automated
Additional input files or diagnostic artifacts are allowed and ignored by the
grader; only the required files and their contents are assessed.

Run the discoverable checks from this directory:

```bash
python check_assignment.py
```

Read every `[FIX]` message, make one focused correction, regenerate the CSV artifacts, then check again.

## Executable contract

Task 1 must define `row_meaning`, `candidate_identifier`, `raw`,
`raw_snapshot`, `audit_person_records`, `issue_audit`, and `issue_counts`. Read
the source with `keep_default_na=False`, preserve a deep raw snapshot, and
return the 15 `issue,count` rows listed in the supplied final cell.
Count exact-row duplication separately from repeated candidate identifiers;
count lexical and calendar date failures together without treating an empty
date as a parse failure.

Task 2 must create these eight decision specifications. Write your own
nonempty, purpose-grounded `reason` for each row.

| `field` | `issue` | `action` |
|---|---|---|
| `full_name` | `empty optional name` | `retain as missing` |
| `full_name, site, status` | `surrounding whitespace and case variants` | `strip surrounding whitespace and normalize bounded field case` |
| `status` | `NA sentinel` | `convert the documented sentinel to missing` |
| `age_text` | `unknown and -9 sentinels` | `convert the documented sentinels to missing` |
| `age_text` | `nonnumeric, fractional, or out-of-range values` | `coerce invalid values to missing without rounding` |
| `visit_date` | `empty, lexically invalid, or calendar-invalid values` | `coerce invalid values to missing after an exact-format check` |
| `all raw columns` | `exact duplicate submissions` | `keep the first exact raw row only` |
| `all fields` | `adjacent-row filling` | `do not forward-fill or backward-fill` |

Implement `clean_person_records(raw_table)` on a deep copy. Derive the exact
duplicate keep mask from untouched raw rows before any normalization. Strip and
title-case `full_name`; strip and lowercase `site` and `status`; convert only
the documented sentinels and empty optional tokens to missing. Keep finite
integer ages from 0 through 120 without rounding. Accept dates only when their
text exactly matches ASCII `YYYY-MM-DD` and the date exists on the calendar.
Remove only exact repeated raw submissions and set `needs_review` exactly when
age or visit date is missing. Required Task 2 names are `decision_table`,
`clean_person_records`, `cleaned`, and `review_queue`.

Task 3 must define `validate_clean_records`, `validation_results`,
`decision_log`, `round_trip`, `audit_round_trip`, and
`decision_round_trip`. The clean columns and in-memory/readback dtypes are:

| Column | dtype |
|---|---|
| `record_id` | `string` |
| `full_name` | `string` |
| `site` | `string` |
| `status` | `string` |
| `age` | `Int64` |
| `visit_date` | `datetime64[us]` |
| `needs_review` | `boolean` |

The decision log columns are `field,issue,action,reason,source,source_sha256,`
`rows_before,rows_after`. Use `source=data/people_raw.csv`, the verified source
checksum, and repeated 12-to-11 row evidence. Assertions must stop export when
any invariant fails. Read every CSV back with an explicit schema and compare it
exactly with the in-memory table.

## Scope

Use the Lecture 01–05 techniques needed for the cumulative foundation and
documented cleaning pipeline. The required outputs are the contract; use any
correct implementation that produces them.
GroupBy and aggregation, `transform`, pivots, plotting, joins, concatenation,
reshaping, encoding, binning, modeling, forward/backward fill, rounding
fractional ages, notebook magics, multi-step pipeline automation, and network
access are out of scope. Do not remove conflicting candidate records
automatically: flag a failed uniqueness invariant for review.

## Midterm assessment

Assignment 05 is the midterm and is cumulative through Lecture 05. Assignments
01–04 provide focused practice in terminal/Python/Git, NumPy, and notebook/pandas
foundations; this assignment applies those skills to the Lecture 05 cleaning
pipeline.

The midterm has an 85-point automated maximum: 25 points for Task 1, 35 for
Task 2, and 25 for Task 3. The remaining 15 points are human review: 5 for the
cumulative Lecture 01–05 evidence map, 5 for cleaning decisions and rationale,
and 5 for validation, provenance, organization, and privacy. The complete
midterm is therefore 100 points; the committed milestones remain the automated
grading contract.

### Artifact comparison

CSV checks compare parsed columns and values, not file hashes or quoting. The
two text artifacts are compared to their exact required content. Rows are
matched by record or issue identity; audit and decision-log row order is not
graded. Each milestone is assessed independently.
