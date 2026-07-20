# Assignment 05 full-rewrite blueprint

Status: accepted implementation contract for the 2026–27 refresh

## Role and policy boundary

Assignment 05 assesses one documented cleaning pipeline aligned only to Lecture
05's five objectives. Its technical contract is independent of the unresolved
syllabus decision about whether this work is a regular assignment or an exam.

Use the neutral title **Assignment 05: Documented Cleaning Pipeline**. The
provisional policy overlay is 100 points divided 30/40/30 across the three
tasks, with 85 automated and 15 human-reviewed points. The label and weighting
may change after syllabus adjudication; the fixture, tasks, artifacts, and
grader behavior must not.

Classroom 50 applies to Assignment 05. Its grader bundle is discoverable and
must not contain secrets, solutions, credentials, confidential data, or checks
whose value depends on secrecy. Local Jupyter or the VS Code notebook interface
is required. Colab is not an assignment or submission path until the separate
save-to-repository pilot is approved.

## Student package

```text
05/assignment/
├── .gitignore
├── .python-version
├── PLATFORM_CHECK.md
├── README.md
├── assignment.ipynb
├── check_assignment.py
├── requirements.txt
├── data/
│   ├── fixture.json
│   └── people_raw.csv
└── output/
    └── .gitkeep
```

The instructor repository additionally contains `_grader_selftest/`; that
directory is excluded from the student starter and production submissions.

Student submissions contain `assignment.ipynb`,
`output/issue_audit.csv`, `output/cleaned_people.csv`, and
`output/decision_log.csv`.

## Exact three-task sequence

### Task 1 — Define the contract and audit raw data (30 points)

Sequence: `raw → audit`.

Students state that one row is one submitted person record, identify
`record_id` as the candidate identifier, distinguish raw from clean data, and
define schema, sentinel, duplicate, missing value, validation invariant, and
provenance. They load through the supplied portable path with
`keep_default_na=False`, preserve `raw_snapshot = raw.copy(deep=True)`, and
implement `audit_person_records(raw_table)`.

Required public names are `raw`, `raw_snapshot`, `row_meaning`,
`candidate_identifier`, `audit_person_records`, `issue_audit`, and
`issue_counts`.

`issue_audit` has exact columns `issue,count` and these canonical results:

| Issue | Count |
|---|---:|
| schema mismatch | 0 |
| empty full-name tokens | 1 |
| empty date tokens | 1 |
| age sentinel tokens | 3 |
| status sentinel tokens | 1 |
| age parse failures | 1 |
| numeric but noninteger age values | 1 |
| age values outside 0 through 120 | 1 |
| date parse failures | 3 |
| rows in exact duplicate sets | 2 |
| rows with repeated candidate IDs | 2 |
| site values needing format normalization | 4 |
| status values needing format normalization | 3 |
| unexpected site values | 0 |
| unexpected non-sentinel status values | 0 |

The implementation distinguishes lexical date-format failures from calendar
failures before reporting their union as `date parse failures`. It also proves
that auditing did not mutate `raw`.

### Task 2 — Record decisions and transform a copy (40 points)

Sequence: `decide → transform`.

Students create `decision_table` before invoking their cleaning function. It
contains eight ordered, nonempty `field`, `issue`, `action`, and `reason` rows
covering empty optional names; bounded name/site/status normalization; the
`NA` status sentinel; `unknown` and `-9` age sentinels; nonnumeric,
fractional, and out-of-range ages; empty, lexically invalid, and
calendar-invalid dates; exact duplicate submissions; and rejection of
forward/backward fill because entity boundaries and meaningful within-entity
order are absent.

Students implement `clean_person_records(raw_table)` and produce
`decision_table`, `cleaned`, and `review_queue`.

The function must derive its exact-duplicate keep mask from untouched raw rows
before normalization or coercion, deep-copy its input, convert only documented
sentinels to missing, normalize bounded text fields, retain only finite integer
ages from 0 through 120, reject fractional ages without rounding, require exact
ASCII `YYYY-MM-DD` text before calendar parsing, remove only exact repeated raw
submissions, and add `needs_review` when age or visit date is missing. It must
not mutate or fill/invent values.

The canonical result has 11 rows with IDs `R001` through `R011`, and seven
review rows. `R002` has missing age and date; `R004` keeps its missing date;
`R006`, `R007`, and `R008` distinguish nonnumeric, fractional, and
out-of-range age handling; and `R009` has a missing date because
`2026-7-01` violates the lexical contract.

### Task 3 — Validate, save, and read back (30 points)

Sequence: `validate → save`.

Students implement
`validate_clean_records(raw_table, raw_snapshot, cleaned_table)` and produce
`validation_results`, `decision_log`, `round_trip`, `audit_round_trip`, and
`decision_round_trip`.

The validation checks raw immutability, exact clean columns, the raw-derived
row-count relationship, canonical candidate-ID presence and uniqueness,
allowed categories, exact pandas dtypes, numeric range, datetime dtype, review
flag dtype, and the exact review-flag rule. Assertions must stop the pipeline
before export if any invariant fails.

The clean schema after schema-aware readback is:

| Column | dtype | Contract |
|---|---|---|
| `record_id` | `string` | required and unique for the canonical fixture |
| `full_name` | `string` | nullable |
| `site` | `string` | required; `north`, `south`, or `west` |
| `status` | `string` | nullable; `active`, `pending`, or `complete` |
| `age` | `Int64` | nullable; 0 through 120 |
| `visit_date` | `datetime64[us]` | nullable |
| `needs_review` | `boolean` | required |

`decision_log.csv` has exact columns
`field,issue,action,reason,source,source_sha256,rows_before,rows_after`, with
the eight decisions and repeated provenance plus `12 → 11` row evidence.

The notebook writes all three CSV artifacts, reads them back with explicit
dtypes, reparses dates through the same lexical contract, and compares the
round trips exactly. Restart-and-run-all must recreate deleted outputs and
replace stale ones.

## Exact fixture and manifest

`data/people_raw.csv` is this exact 570-byte LF-terminated source:

```csv
record_id,full_name,site,status,age_text,visit_date
R001, Alice Smith , North ,Active,34,2026-01-15
R002,BOB JONES,north,active,unknown,2026-02-30
R002,BOB JONES,north,active,unknown,2026-02-30
R003, Carla Ruiz ,SOUTH,pending,-9,2026-03-01
R004,,south,NA,45,
R005,Evan Li,west,complete,52,2026-02-14
R006,Fatima Noor,north,active,forty,2026-04-01
R007,Grace Chen,south,active,40.5,2026-05-01
R008,Hugo Diaz,west,pending,121,2026-06-01
R009,Inez Park, north ,complete,39,2026-7-01
R010,Jamie Okafor,West,Complete,28,2026-07-15
R011,Kai Patel,south, pending ,0,2026-08-01
```

Its SHA-256 is
`d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b`.

`fixture.json` contains only fixture ID, synthetic provenance, row meaning,
candidate identifier, row count, ordered raw columns, and the checksum. The
supplied setup cell verifies the exact manifest and bytes before pandas reads
the source and supports flattened Classroom 50 and course-repository layouts
without an absolute path.

## Starter and environment

- Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3 are the candidate records.
- The notebook has a portable Python 3 kernelspec, unique cell IDs, cleared
  outputs, and null execution counts.
- One immutable setup cell and one supplied final verification cell are
  complete. Three task sections contain deliberate TODOs.
- The starter contains no generated CSVs beyond `output/.gitkeep`.
- No solution fragments, health/clinical claims, plots, shell scripts,
  downloads, Drive mounts, or hidden-state dependencies are present.
- The starter fails public checks with task-specific messages.

## Public and central checks

The discoverable public checker verifies fixture/manifest integrity, notebook
JSON, supplied-cell integrity, required sections/functions, fresh code-cell
execution in a disposable directory, canonical counts, raw immutability,
clean values/dtypes/order/review flags, decision log, round trips, stale-output
replacement, and scope. It does not trust stored notebook output.

The central Classroom 50 grader independently repeats those checks without
importing the editable public checker, executes canonical and relocated copies,
deletes generated outputs, clears notebook state, appends instructor-owned
function assertions, tests corrupt and missing fixtures, and emits a
`classroom50/result/v1` result.

Discoverable alternate cases cover valid nondefault-index data; different
nonnumeric, fractional, and out-of-range ages; lexical and calendar date
failures; exact and candidate duplicates; normalization collisions; repeat
calls; input mutation; and reordered or extra schema columns. Correctness
depends on behavioral variation rather than secrecy.

## Human-review boundary

Human review covers the row/identifier/raw-clean explanation, decision reasons
grounded in variable meaning and purpose, rejection of adjacent-row filling,
recognition that reviewable missing values remain, the statement that clean
means satisfying a contract rather than perfection, organization, and privacy.

Automation covers executable behavior, files, schemas, values, invariants,
provenance, reproducibility, and scope.

## Scope exclusions

The required assignment excludes GroupBy, aggregation, `transform`,
`pivot_table`, plotting, joins, concatenation, reshape, feature encoding,
binning, modeling, shell notebook automation, notebook magics, network access,
manual uploads, Drive mounts, `/content` paths, universal deletion/imputation
thresholds, executable forward/backward fill, rounding fractional ages,
automatic deletion of conflicting candidate records, and required
MCAR/MAR/MNAR theory.

## Legacy inventory disposition

Production-contract correction (2026-07-19): the teacher bundle includes a
standard-library `autograder.py` because Classroom50 launches it with plain
Python; that entrypoint installs the exact sibling requirements before importing
the grader. Student inventory may additionally contain only delivery-owned
`.classroom50.yaml` and `.github/workflows/autograde.yaml`; only the top-level
`.git/**` repository metadata tree is ignored, while other root/workflow files,
a nested `ordinary/.git/**` tree, and `_grader_selftest/**` are rejected.

- Rewrite and consolidate the old `README.md` and `GRADING_SPEC.md` into one
  student contract and one central rubric source.
- Replace four notebooks with one `assignment.ipynb`.
- Replace the 10,000-row clinical fixture and generator with the exact small
  synthetic fixture and manifest.
- Remove the Q1/Q8 shell scripts, Q2 config exercise, Q3 utility library,
  duplicated standalone analysis, broad tips, completion reports, committed
  generated reports/outputs, old tests, and legacy GitHub Classroom workflow.
- Replace broad requirements with exact direct dependencies.

The old package claimed 100 points but allocated 125. This implementation has
one provisional 100-point source of truth while the assessment policy remains
pending.
