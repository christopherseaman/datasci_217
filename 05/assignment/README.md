# Assignment 05: Documented Cleaning Pipeline

## Files

```text
assignment/
├── assignment.ipynb                 # notebook scaffold to complete
├── .python-version, requirements.txt # supplied environment records
├── data/people_raw.csv, data/fixture.json # supplied synthetic input/manifest
├── check_assignment.py, grading.py  # supplied checker; keep unchanged
└── output/                         # generate and commit two text files and five CSVs
```

## Setup

Open **Terminal → New Terminal** in VS Code at the assignment directory. If you use a native terminal or WSL Ubuntu instead, first `cd` into the assignment directory.

Use Python 3.13. From this directory, create and activate a virtual
environment, and install the exact dependency records. Open the supplied
notebook through Jupyter or the VS Code notebook interface:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Select this environment as the notebook kernel. Confirm `python --version` reports Python 3.13, then run:

```bash
python -c "import numpy, pandas; print(numpy.__version__, pandas.__version__)"
python -c "from pathlib import Path; print(Path('data/people_raw.csv').is_file())"
```

Expect `2.3.3 3.0.5` and `True`. If the data is missing, return to the assignment directory and restore the supplied files. Use the notebook's supplied portable setup with the local synthetic data. Keep credentials and private information out of source and outputs.

## Foundation artifacts

Complete every `TODO` in `assignment.ipynb`, including the cumulative midterm
checkpoint. Create these four cumulative foundation artifacts before the cleaning artifacts:

> **Checkpoint — `output/raw_preview.txt`**

Save this exact UTF-8 terminal evidence, including the command labels and final newline:

```text
$ head -n 4 data/people_raw.csv
record_id,full_name,site,status,age_text,visit_date
R001, Alice Smith , North ,Active,34,2026-01-15
R002,BOB JONES,north,active,unknown,2026-02-30
R002,BOB JONES,north,active,unknown,2026-02-30
$ tail -n 2 data/people_raw.csv
R010,Jamie Okafor,West,Complete,28,2026-07-15
R011,Kai Patel,south, pending ,0,2026-08-01
```

> **Checkpoint — `output/pipeline_summary.txt`**

Use a small Python summary function to produce these exact lines, with a final newline:

```text
raw_rows=12
raw_columns=6
exact_duplicate_rows=1
candidate_id_duplicate_rows=1
clean_rows=11
```

> **Checkpoint — `output/numpy_age_summary.csv`**

Save header `metric,value`, followed by the `count`, `min`, `max`, `sum`, and `mean` of the valid integer ages in that order (`6`, `0`, `52`, `198`, `33.0`).

> **Checkpoint — `output/pandas_selection.csv`**

Save header `record_id,site,status`, followed by the raw rows for `R001`, `R003`, and `R010` in that order. Preserve their original whitespace and case.

Then create these three cleaning milestone artifacts:

- `output/issue_audit.csv`
- `output/cleaned_people.csv`
- `output/decision_log.csv`

## Task 1: Define and audit

### 1.1 State the contract and audit the raw table

Task 1 must define `row_meaning`, `candidate_identifier`, `raw`,
`raw_snapshot`, `audit_person_records`, `issue_audit`, and `issue_counts`. Read
the source with `keep_default_na=False`, preserve a deep raw snapshot, and
return the 15 `issue,count` rows listed in the supplied final cell.
Count exact-row duplication separately from repeated candidate identifiers;
count lexical and calendar date failures together without treating an empty
date as a parse failure.

## Task 2: Record decisions and clean

### 2.1 Document the decisions

Create these eight decision specifications. Write your own
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

### 2.2 Transform a copy

Implement `clean_person_records(raw_table)` on a deep copy. Derive the exact
duplicate keep mask from untouched raw rows before any normalization. Strip and
title-case `full_name`; strip and lowercase `site` and `status`; convert only
the documented sentinels and empty optional tokens to missing. Keep finite
integer ages from 0 through 120 without rounding. Accept dates only when their
text exactly matches ASCII `YYYY-MM-DD` and the date exists on the calendar.
Remove only exact repeated raw submissions and set `needs_review` exactly when
age or visit date is missing. Required Task 2 names are `decision_table`,
`clean_person_records`, `cleaned`, and `review_queue`.

## Task 3: Validate and save

### 3.1 Validate the cleaned table

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

### 3.2 Save and read back the artifacts

The decision log columns are `field,issue,action,reason,source,source_sha256,`
`rows_before,rows_after`. Use `source=data/people_raw.csv`, the verified source
checksum, and repeated 12-to-11 row evidence. Assertions must stop export when
any invariant fails. Read every CSV back with an explicit schema and compare it
exactly with the in-memory table.

> **Checkpoint — `output/issue_audit.csv`**
> Save the 15-row audit with columns `issue,count`.

> **Checkpoint — `output/cleaned_people.csv`**
> Save the 11 cleaned records with the seven columns listed above, dates as `YYYY-MM-DD`, and missing values as empty fields.

> **Checkpoint — `output/decision_log.csv`**
> Save the eight decisions with nonblank reasons, source checksum, and before/after row counts.

## Check your work

Regenerate the saved artifacts and run from the assignment directory:

```bash
python check_assignment.py
```

Correct the named artifacts and repeat until every check passes.

### Completion contract

The midterm totals 100 points: 85 artifact-check points and 15 human-review points. CSV checks compare parsed columns and values; the two text files must match the specified content, including final newlines.

| Artifacts | Format and completion criteria | Points |
|---|---|---:|
| `output/raw_preview.txt`, `output/pipeline_summary.txt`, `output/numpy_age_summary.csv`, `output/pandas_selection.csv`, `output/issue_audit.csv` | The four foundation outputs specified above plus all 15 unique audit issues and counts in the notebook. Preserve foundation row order and raw selection text, including whitespace and case. Audit order may vary. | 25 |
| `output/cleaned_people.csv` | Columns `record_id,full_name,site,status,age,visit_date,needs_review`; the 11 unique records produced by the stated cleaning rules, including correct normalized values, missingness, valid dates, and review flags. Row order may vary. | 35 |
| `output/decision_log.csv` | Columns `field,issue,action,reason,source,source_sha256,rows_before,rows_after`; exactly the eight documented decisions, nonblank reasons, `data/people_raw.csv`, its verified checksum, and 12-to-11 counts on every row. Row order may vary. | 25 |

Human review awards 5 points for the cumulative Lecture 01–05 evidence map, 5 for cleaning decisions and rationale, and 5 for validation, provenance, organization, and privacy. Additional files are allowed and ignored by the artifact checks.

## Submit

Save the completed notebook. Inspect and commit `assignment.ipynb` and all seven output artifacts through VS Code Source Control, sync, and confirm they appear in your assignment repository.

GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork.
