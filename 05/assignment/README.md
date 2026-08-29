# Assignment 05: Documented Cleaning Pipeline

Build one reproducible pandas cleaning pipeline for a small, synthetic table.
The work follows the same sequence used in Lecture 05:

1. define the data contract and audit the untouched raw table;
2. record decisions, then transform a copy;
3. validate, save, and read the artifacts back with explicit schemas.

This is a local Jupyter assignment. Do not use Colab, manual uploads, Drive
mounts, network access, or `/content` paths. The supplied path finder supports
both a standalone exported assignment repository and this course repository.

## Setup

Use Python 3.12.13. From this directory, create and activate a virtual
environment, install the exact dependency records, and open Jupyter or the VS
Code notebook interface:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Complete
the checks in [PLATFORM_CHECK.md](PLATFORM_CHECK.md) before editing the
notebook.

## Deliverables

Complete every `TODO` in `assignment.ipynb`. Restart the kernel and run all
cells from top to bottom. Commit these four files in the assignment repository:

- `assignment.ipynb`
- `output/issue_audit.csv`
- `output/cleaned_people.csv`
- `output/decision_log.csv`

Do not edit `data/people_raw.csv`, `data/fixture.json`, or the two supplied
notebook cells. Generated output must come from a fresh notebook run; stored
cell output is not evidence that the code runs.

Run the discoverable checks from this directory:

```bash
python check_assignment.py
```

Read every `[FIX]` message, make one focused correction, restart and run all,
then check again. The checker intentionally ignores stored notebook output and
re-runs the code in disposable relocated copies.

## Executable contract

Task 1 must define `row_meaning`, `candidate_identifier`, `raw`,
`raw_snapshot`, `audit_person_records`, `issue_audit`, and `issue_counts`. Read
the source with `keep_default_na=False`, preserve a deep raw snapshot, and
return the 15 ordered `issue,count` rows listed in the supplied final cell.
Count exact-row duplication separately from repeated candidate identifiers;
count lexical and calendar date failures together without treating an empty
date as a parse failure.

Task 2 must create these eight ordered decision specifications. Write your own
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

Use the Lecture 05 techniques needed for a documented cleaning pipeline.
GroupBy and aggregation, `transform`, pivots, plotting, joins, concatenation,
reshaping, encoding, binning, modeling, forward/backward fill, rounding
fractional ages, notebook magics, shell automation, and network access are out
of scope. Do not remove conflicting candidate records automatically: flag a
failed uniqueness invariant for review.

## Provisional assessment overlay

The current implementation record is 100 points: Task 1 is 30, Task 2 is 40,
and Task 3 is 30. Of those, 85 points are executable checks and 15 points are
human review of explanations, decision reasoning, organization, and privacy.
This policy overlay is pending syllabus adjudication; the technical assignment
contract is not. This assignment is not labeled as a midterm.
