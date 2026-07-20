# Data Cleaning as a Documented, Validated Pipeline

Lecture 05 turns an input table into a documented clean artifact. The goal is not to make every value look convenient. The goal is to identify problems before changing them, choose actions that respect what the rows and variables mean, validate the result, and preserve enough evidence for another person to reproduce the work.

Optional extensions are collected in [BONUS.md](BONUS.md). They are not prerequisites for the required demos, assignment, or Lecture 06.

## Prerequisites

Before starting this lecture, you should be able to:

- restart and run all a notebook;
- load and save CSV data through the course portable-path convention;
- inspect DataFrame shape, columns, dtypes, index, head, and summary;
- select and filter rows and columns, create a column, and sort deterministically; and
- define and call a function when a repeated cleaning step benefits from one.

Lecture 04 introduced these notebook and pandas mechanics. No prior cleaning pipeline, statistical missingness theory, merge, grouping, modeling, or time-series workflow is assumed.

## Learning objectives

By the end of Lecture 05, students should be able to:

1. Distinguish raw and cleaned data and define **schema**, **sentinel value**, **duplicate**, **missing value**, **imputation**, **validation invariant**, and **provenance/audit trail**.
2. Produce a reproducible audit of schema, missingness, sentinel codes, duplicate candidates, category inconsistencies, type failures, and invalid ranges without modifying the raw table.
3. Choose and justify a missing-data action using variable meaning, row meaning, and analysis purpose; identify when forward/backward fill is invalid because entity/order requirements are absent.
4. Standardize strings, categories, numeric/date types, sentinel values, and duplicate records while preserving raw input and recording each decision.
5. Express post-cleaning expectations as executable invariants and produce a clean dataset plus a decision log from a fresh runtime.

## Colab-first execution and evidence

Required demonstrations from this lecture onward are Colab-first and also run in local Jupyter. They use the portable path and conditional package setup established in Lecture 04.

Colab's filesystem is **ephemeral**: runtime-local files can disappear when the runtime is deleted. A required notebook must therefore reacquire its pinned input and create its output directories from code. Course notebooks do not require a manual upload or mount Google Drive by default.

Notebook assignments must remain runnable in clean local Jupyter. Colab becomes an assignment submission path only after the repository-save and Classroom 50 submission pilot is approved.

Before sharing or submitting a notebook:

- remove credentials, private records, and other sensitive output;
- retain ordinary output only when a human rubric needs to inspect it;
- remember that files under `output/` are generated artifacts, not stored cell output; and
- restart and run all from a fresh runtime.

Stored output is never proof that the visible source still runs. A grader executes a fresh copy.

## Define the data contract before cleaning

**Raw data** is the source artifact as received. Preserve it unchanged. **Cleaned data** is a separate, derived artifact that satisfies a documented contract after deliberate transformations and validation. Clean does not mean perfect, complete, or free of unusual values.

**Row meaning** states what one row represents. In the running example, one row is one submitted person record. A **schema** records the expected column names, meanings, data types, allowed or required values, and whether missing values are permitted.

A **candidate identifier** is one column, or a combination of columns, expected to distinguish rows. It is only a candidate until the audit tests its uniqueness and missingness. Here, `record_id` is the candidate identifier.

**Tidy data** uses one column for each variable, one row for each observation, and one table for each type of observational unit. This is a structural description, not proof that the values are valid or clean. Lecture 06 teaches structural reshaping; Lecture 05 only states the expected row and column meanings. See Wickham's [Tidy Data](https://www.jstatsoft.org/article/view/v059i10) for the originating formulation.

**Provenance** records where an artifact came from. An **audit trail** records the source, detected issues, decisions, transformations, validation results, and output. Together they make the path from raw to clean inspectable.

### Preserve raw input and make a working copy

The self-contained examples use one small supplied CSV string. Required notebooks use a pinned file through the Lecture 04 path bootstrap instead of a manual upload.

A **source checksum** is a content fingerprint. Recording it lets the pipeline verify that a later run used the same raw bytes without storing a second mutable copy as evidence.

```python
from hashlib import sha256
from io import StringIO

import pandas as pd

RAW_SOURCE_NAME = "supplied_people_raw.csv"
raw_csv = """record_id,full_name,site,status,age_text,visit_date
R001, Alice Smith , north,Active,34,2026-01-15
R002,BOB JONES,North,active,unknown,2026-02-30
R002,BOB JONES,North,active,unknown,2026-02-30
R003, Carla Ruiz ,SOUTH,pending,-9,2026-03-01
R004,,south,NA,45,
R005,Evan Li,west,complete,52,2026-02-14
"""

source_sha256 = sha256(raw_csv.encode("utf-8")).hexdigest()
raw = pd.read_csv(StringIO(raw_csv), keep_default_na=False)
raw_snapshot = raw.copy(deep=True)
working = raw.copy(deep=True)

print("source:", RAW_SOURCE_NAME)
print("sha256:", source_sha256)
print("shape:", raw.shape)
print(raw.dtypes)
```

`keep_default_na=False` is deliberate for this teaching fixture: it preserves tokens such as an empty string and `NA` so the audit can distinguish source conventions from pandas-recognized missing values. Do not overwrite `raw`; transformations belong in `working` or a later derived object.

The expected schema is:

| Column | Row-level meaning | Intended clean type | Missing allowed? | Other rule |
|---|---|---|---|---|
| `record_id` | submitted record identifier | string | no | unique after resolving duplicate records |
| `full_name` | submitted person name | string | yes | surrounding whitespace removed |
| `site` | collection site | string | no | `north`, `south`, or `west` |
| `status` | record status | string | yes | `active`, `pending`, or `complete` |
| `age_text` → `age` | age in years | nullable integer (`Int64`) | yes | 0 through 120 when present |
| `visit_date` | submitted calendar date | datetime | yes | exact `YYYY-MM-DD` input when present |

## Audit before mutation

An **audit** measures the raw table against the data contract without changing it. The audit should be reproducible: another person running it on the same source should see the same counts.

Inspect structure first:

```python
expected_columns = [
    "record_id",
    "full_name",
    "site",
    "status",
    "age_text",
    "visit_date",
]

schema_matches = list(raw.columns) == expected_columns
print("schema matches:", schema_matches)
print(raw.head(3))
print(raw.isna().sum())
```

The final line reports only missing values pandas already recognizes. It does not detect every source-specific token.

### Missing values, sentinels, and parse failures

A **missing value** represents an absent or unknown value. A **sentinel value** is a source-specific code such as `unknown`, `-9`, `NA`, or an empty field that stands for a special condition. A sentinel is not automatically a valid value and is not automatically missing; the data dictionary determines its meaning.

`pd.NA` is an explicit pandas marker for a missing value. The marker records absence; it does not record why the value is absent or which cleaning action is appropriate.

A **parse failure** occurs when text cannot be converted to its intended type under the stated format. For example, `2026-02-30` is nonempty text but not a valid calendar date. Missing values, sentinels, and parse failures need separate audit counts because they can require different decisions.

A **lexical format check** tests the exact characters in source text before interpreting their meaning. For this schema, valid date text must contain exactly four ASCII digits, a hyphen, two ASCII digits, a hyphen, and two ASCII digits. Passing that check does not prove that the text names a possible calendar date, so calendar parsing remains a separate check.

A **numeric-but-noninteger value** is a finite number with a fractional part, such as `40.5`. It is distinct from nonnumeric text that fails parsing and from an integer that falls outside the allowed range. The age schema does not authorize rounding fractional values into integers.

Use nonmutating conversion probes to locate failures:

```python
EXACT_DATE_PATTERN = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"

age_sentinel_mask = raw["age_text"].isin(["unknown", "-9"])
status_sentinel_mask = raw["status"].eq("NA")
empty_name_mask = raw["full_name"].eq("")
empty_date_mask = raw["visit_date"].eq("")

age_probe_source = raw["age_text"].replace(
    {"unknown": pd.NA, "-9": pd.NA}
)
age_probe = pd.to_numeric(age_probe_source, errors="coerce")
age_parse_failure_mask = age_probe.isna() & age_probe_source.notna()
age_finite_mask = age_probe.notna() & age_probe.abs().lt(float("inf"))
age_noninteger_mask = age_finite_mask & age_probe.mod(1).ne(0)
age_integer_mask = age_finite_mask & ~age_noninteger_mask
age_range_failure_mask = age_integer_mask & ~age_probe.between(0, 120)

date_probe_source = raw["visit_date"].replace({"": pd.NA})
exact_date_text_mask = date_probe_source.str.fullmatch(
    EXACT_DATE_PATTERN,
    na=False,
)
date_format_failure_mask = date_probe_source.notna() & ~exact_date_text_mask
date_probe = pd.to_datetime(
    date_probe_source.where(exact_date_text_mask, pd.NA),
    format="%Y-%m-%d",
    errors="coerce",
)
date_calendar_failure_mask = exact_date_text_mask & date_probe.isna()
date_parse_failure_mask = (
    date_format_failure_mask | date_calendar_failure_mask
)
```

The combined date parse-failure mask counts both nonmissing text that violates the exact lexical format and exact-format text that does not name a possible calendar date. `errors="coerce"` is appropriate for an audit probe because it makes calendar failures visible. It is not a cleaning decision by itself.

### Duplicate records and identity

A **duplicate** is a repeated record under an explicitly stated comparison. An exact duplicate repeats every compared value. A duplicate candidate identifier repeats the proposed identity field, but the rows may contain conflicting information.

Detect both questions before removing anything:

```python
exact_duplicate_mask = raw.duplicated(keep=False)
exact_duplicate_keep_mask = ~raw.duplicated(keep="first")
candidate_duplicate_mask = raw.duplicated(
    subset=["record_id"],
    keep=False,
)

print("rows in exact duplicate sets:", int(exact_duplicate_mask.sum()))
print(
    "rows with repeated candidate IDs:",
    int(candidate_duplicate_mask.sum()),
)
print(raw.loc[candidate_duplicate_mask])
```

`drop_duplicates()` cannot decide which conflicting entity record is correct. Removing an exact repeated submission can be defensible; resolving conflicting records requires source or domain evidence.

### One structured issue audit

```python
normalized_site_probe = raw["site"].str.strip().str.lower()
normalized_status_probe = raw["status"].str.strip().str.lower()
site_format_inconsistency_mask = raw["site"].ne(normalized_site_probe)
status_format_inconsistency_mask = (
    raw["status"].ne(normalized_status_probe)
    & ~status_sentinel_mask
)

issue_audit = pd.DataFrame(
    [
        {"issue": "schema mismatch", "count": int(not schema_matches)},
        {"issue": "empty full-name tokens", "count": int(empty_name_mask.sum())},
        {"issue": "empty date tokens", "count": int(empty_date_mask.sum())},
        {"issue": "age sentinel tokens", "count": int(age_sentinel_mask.sum())},
        {"issue": "status sentinel tokens", "count": int(status_sentinel_mask.sum())},
        {"issue": "age parse failures", "count": int(age_parse_failure_mask.sum())},
        {
            "issue": "numeric but noninteger age values",
            "count": int(age_noninteger_mask.sum()),
        },
        {"issue": "age values outside 0 through 120", "count": int(age_range_failure_mask.sum())},
        {"issue": "date parse failures", "count": int(date_parse_failure_mask.sum())},
        {"issue": "rows in exact duplicate sets", "count": int(exact_duplicate_mask.sum())},
        {"issue": "rows with repeated candidate IDs", "count": int(candidate_duplicate_mask.sum())},
        {
            "issue": "site values needing format normalization",
            "count": int(site_format_inconsistency_mask.sum()),
        },
        {
            "issue": "status values needing format normalization",
            "count": int(status_format_inconsistency_mask.sum()),
        },
        {
            "issue": "unexpected site values",
            "count": int(
                (~normalized_site_probe.isin(["north", "south", "west"])).sum()
            ),
        },
        {
            "issue": "unexpected non-sentinel status values",
            "count": int(
                (
                    ~normalized_status_probe.isin(
                        ["active", "pending", "complete", "na"]
                    )
                ).sum()
            ),
        },
    ]
)

issue_audit
```

This artifact reports affected rows or tokens. Its labels make the counting unit explicit.

## Decide before transforming

**Imputation** replaces a missing value with an estimated or rule-supplied value. An imputed value was not observed, so its rationale and consequences must be documented.

No missingness percentage supplies a universal action. Before dropping, retaining, imputing, or collecting a value again, ask:

1. What does the variable mean?
2. What does removing its row remove from the population represented by the table?
3. Is the field required for the stated purpose?
4. Is there defensible information for an estimate or rule?
5. How will the decision affect later interpretation?
6. Which invariant will verify the intended result?

Record the decision before mutation:

```python
decision_table = pd.DataFrame(
    [
        {
            "field": "full_name",
            "issue": "empty token",
            "action": "convert to missing and retain row",
            "reason": "name is optional for this de-identified exercise",
        },
        {
            "field": "status",
            "issue": "NA sentinel",
            "action": "convert to missing and retain row",
            "reason": "the source dictionary defines NA as unknown status",
        },
        {
            "field": "age_text",
            "issue": "unknown and -9 sentinels",
            "action": "convert to missing; do not impute",
            "reason": "no defensible person-level age estimate is supplied",
        },
        {
            "field": "age_text",
            "issue": "nonnumeric or numeric-but-noninteger value",
            "action": "convert to missing and flag for review; do not round",
            "reason": "the integer-age schema supplies no defensible correction",
        },
        {
            "field": "visit_date",
            "issue": "empty or invalid calendar date",
            "action": "convert to missing and flag for review",
            "reason": "inventing a visit date would change the record meaning",
        },
        {
            "field": "all columns",
            "issue": "one exact repeated submission",
            "action": "retain first exact row",
            "reason": "the repeated rows carry identical information",
        },
    ]
)

decision_table
```

Forward fill with `.ffill()` and backward fill with `.bfill()` copy adjacent values. They are invalid for this table because adjacent rows do not represent ordered observations within the same entity. Do not use either method until both the entity boundary and the order have a documented meaning. Lecture 09 later establishes those time-series requirements.

## LIVE DEMO 1: Audit and decision table

[Open the Lecture 05 demo guide](demo/DEMO_GUIDE.md).

The first required demonstration loads immutable raw data, states row meaning and candidate identifiers, audits schema, sentinels, duplicates, missingness, categories, parse failures, and ranges, then records decisions without mutating the raw table.

## Apply targeted transformations

Transform `working`, never `raw`. Prefer pandas' column operations over a row-by-row custom function when the built-in operation expresses the rule directly. Assign transformed results back to columns; do not use chained `inplace=True` operations.

### Convert documented sentinels

```python
working["full_name"] = working["full_name"].replace({"": pd.NA})
working["status"] = working["status"].replace({"NA": pd.NA})
working["age_text"] = working["age_text"].replace(
    {"unknown": pd.NA, "-9": pd.NA}
)
working["visit_date"] = working["visit_date"].replace({"": pd.NA})
```

These replacements implement the source-specific decisions. They are not a general list of missing tokens for every dataset.

### Normalize strings, categories, and names

**Normalization** maps equivalent representations to one documented form. Bound it to columns whose meaning is known.

```python
working["full_name"] = working["full_name"].str.strip().str.title()
working["site"] = working["site"].str.strip().str.lower()
working["status"] = working["status"].str.strip().str.lower()

print(working[["full_name", "site", "status"]])
```

Keep the source name `age_text` until its values have been converted to the intended numeric type.

### Convert numeric and date values explicitly

```python
working_age_numeric = pd.to_numeric(
    working["age_text"],
    errors="coerce",
)
working_age_finite_mask = (
    working_age_numeric.notna()
    & working_age_numeric.abs().lt(float("inf"))
)
working_age_noninteger_mask = (
    working_age_finite_mask & working_age_numeric.mod(1).ne(0)
)
working_age_integer_mask = (
    working_age_finite_mask & ~working_age_noninteger_mask
)
working["age_text"] = working_age_numeric.where(
    working_age_integer_mask,
    pd.NA,
).astype("Int64")
working = working.rename(columns={"age_text": "age"})

working_date_text_mask = working["visit_date"].str.fullmatch(
    EXACT_DATE_PATTERN,
    na=False,
)
working["visit_date"] = pd.to_datetime(
    working["visit_date"].where(working_date_text_mask, pd.NA),
    format="%Y-%m-%d",
    errors="coerce",
)

date_contract_probe = pd.Series(
    ["2026-01-01", "2026-1-1", "2026-02-30"],
    dtype="string",
)
date_contract_text_mask = date_contract_probe.str.fullmatch(
    EXACT_DATE_PATTERN,
    na=False,
)
date_contract_result = pd.to_datetime(
    date_contract_probe.where(date_contract_text_mask, pd.NA),
    format="%Y-%m-%d",
    errors="coerce",
)
date_contract_format_failure_mask = ~date_contract_text_mask
date_contract_calendar_failure_mask = (
    date_contract_text_mask & date_contract_result.isna()
)
date_contract_parse_failure_mask = (
    date_contract_format_failure_mask
    | date_contract_calendar_failure_mask
)

assert date_contract_text_mask.tolist() == [True, False, True]
assert date_contract_result.notna().tolist() == [True, False, False]
assert date_contract_parse_failure_mask.tolist() == [False, True, True]

print(working.dtypes)
```

The date regression separates two failure modes: non-zero-padded `2026-1-1` fails the lexical contract, while exact-format `2026-02-30` passes that contract but fails calendar parsing. Both become missing during the documented transformation, so the earlier parse-failure audit and decision log must retain evidence that the conversion occurred. Age conversion likewise retains only finite integer-valued parses; nonnumeric and numeric-but-noninteger values become missing without rounding.

### Resolve the documented exact duplicate

```python
working = working.loc[exact_duplicate_keep_mask].copy()
```

The keep mask was derived from untouched raw rows before normalization or coercion. Applying that preserved mask follows the recorded exact-duplicate decision without silently collapsing rows that become equal only after cleaning. It is not authorization to discard every repeated candidate identifier in another dataset.

### Select, filter, and add one review flag

```python
working["needs_review"] = (
    working["age"].isna()
    | working["visit_date"].isna()
)

review_queue = working.loc[
    working["needs_review"],
    ["record_id", "age", "visit_date"],
].copy()

review_queue
```

The flag preserves uncertain rows instead of silently inventing or deleting values.

## LIVE DEMO 2: Targeted transformations

[Open the Lecture 05 demo guide](demo/DEMO_GUIDE.md).

The second required demonstration applies vectorized sentinel, string/category, type/date, and exact-duplicate transformations. It contrasts a defensible missing-value action with invalid adjacent-row filling when entity and order are absent.

## Validate explicit invariants

A **validation invariant** is a condition that must be true after a pipeline stage. Write it before declaring an artifact clean, then make it executable.

Useful invariant categories include:

- exact column presence and order;
- expected row-count relationship;
- required identifiers present and unique;
- allowed category values;
- intended numeric and datetime types;
- permitted missingness by column; and
- valid numeric ranges when values are present.

```python
validation_results = pd.Series(
    {
        "raw preserved": raw.equals(raw_snapshot),
        "expected rows after one exact duplicate removal": len(working) == 5,
        "record ID present": working["record_id"].notna().all(),
        "record ID unique": working["record_id"].is_unique,
        "site allowed": working["site"].isin(["north", "south", "west"]).all(),
        "status allowed when present": working["status"].dropna().isin(
            ["active", "pending", "complete"]
        ).all(),
        "age has nullable integer dtype": str(working["age"].dtype) == "Int64",
        "age in range when present": working["age"].dropna().between(0, 120).all(),
        "visit date has datetime dtype": pd.api.types.is_datetime64_any_dtype(
            working["visit_date"].dtype
        ),
    },
    name="passed",
)

assert validation_results.all(), validation_results[~validation_results]
validation_results
```

Assertions stop the pipeline when its contract is broken. They do not establish that the original decisions were wise; the decision log carries that human reasoning.

## Produce a decision log and restartable pipeline

A useful **decision log** is a structured part of the audit trail. It identifies the field, issue, action, rationale, source, and relevant before/after evidence.

```python
decision_log = decision_table.copy()
decision_log["source"] = RAW_SOURCE_NAME
decision_log["source_sha256"] = source_sha256
decision_log["rows_before"] = len(raw)
decision_log["rows_after"] = len(working)

decision_log
```

The complete transformation can be expressed as one function without hiding its decisions:

```python
def clean_person_records(raw_table):
    exact_duplicate_keep_mask = ~raw_table.duplicated(keep="first")
    result = raw_table.copy(deep=True)

    result["full_name"] = result["full_name"].replace({"": pd.NA})
    result["status"] = result["status"].replace({"NA": pd.NA})
    result["age_text"] = result["age_text"].replace(
        {"unknown": pd.NA, "-9": pd.NA}
    )
    result["visit_date"] = result["visit_date"].replace({"": pd.NA})

    result["full_name"] = result["full_name"].str.strip().str.title()
    result["site"] = result["site"].str.strip().str.lower()
    result["status"] = result["status"].str.strip().str.lower()
    result_age_numeric = pd.to_numeric(
        result["age_text"],
        errors="coerce",
    )
    result_age_finite_mask = (
        result_age_numeric.notna()
        & result_age_numeric.abs().lt(float("inf"))
    )
    result_age_noninteger_mask = (
        result_age_finite_mask & result_age_numeric.mod(1).ne(0)
    )
    result_age_integer_mask = (
        result_age_finite_mask & ~result_age_noninteger_mask
    )
    result["age_text"] = result_age_numeric.where(
        result_age_integer_mask,
        pd.NA,
    ).astype("Int64")
    result = result.rename(columns={"age_text": "age"})

    exact_date_text_mask = result["visit_date"].str.fullmatch(
        EXACT_DATE_PATTERN,
        na=False,
    )
    result["visit_date"] = pd.to_datetime(
        result["visit_date"].where(exact_date_text_mask, pd.NA),
        format="%Y-%m-%d",
        errors="coerce",
    )

    result = result.loc[exact_duplicate_keep_mask].copy()
    result["needs_review"] = result["age"].isna() | result["visit_date"].isna()
    return result


pipeline_result = clean_person_records(raw)
pd.testing.assert_frame_equal(pipeline_result, working)

normalization_distinct_raw = pd.DataFrame(
    {
        "record_id": ["R100", "R100"],
        "full_name": [" Alice Example ", "alice example"],
        "site": ["North", "north"],
        "status": ["Active", "active"],
        "age_text": ["40", "40"],
        "visit_date": ["2026-01-01", "2026-01-01"],
    }
)

assert not normalization_distinct_raw.duplicated(keep=False).any()
normalization_distinct_result = clean_person_records(
    normalization_distinct_raw
)
assert len(normalization_distinct_result) == 2

fractional_age_raw = pd.DataFrame(
    {
        "record_id": ["R200"],
        "full_name": ["Fractional Age"],
        "site": ["north"],
        "status": ["active"],
        "age_text": ["40.5"],
        "visit_date": ["2026-01-01"],
    }
)
fractional_age_probe = pd.to_numeric(
    fractional_age_raw["age_text"],
    errors="coerce",
)
fractional_age_finite_mask = (
    fractional_age_probe.notna()
    & fractional_age_probe.abs().lt(float("inf"))
)
fractional_age_noninteger_mask = (
    fractional_age_finite_mask & fractional_age_probe.mod(1).ne(0)
)
fractional_age_issue_audit = pd.DataFrame(
    [
        {
            "issue": "numeric but noninteger age values",
            "count": int(fractional_age_noninteger_mask.sum()),
        }
    ]
)
fractional_age_result = clean_person_records(fractional_age_raw)

assert fractional_age_issue_audit.loc[0, "count"] == 1
assert pd.isna(fractional_age_result.loc[0, "age"])
assert fractional_age_result.loc[0, "needs_review"]
assert str(fractional_age_result["age"].dtype) == "Int64"
```

The first regression keeps both raw-distinct rows even though normalization makes their cleaned values equal. Their repeated candidate identifier still requires evidence-based resolution; the exact-duplicate rule must not erase that conflict. The second regression counts `40.5` in the numeric-but-noninteger audit category, converts it to missing without rounding or raising, and sends the row to review while preserving the `Int64` schema.

Use the Lecture 04 portable-path bootstrap in required notebooks. This self-contained teaching fixture writes to a runtime-local example directory:

```python
from pathlib import Path

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

working.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)
issue_audit.to_csv(OUTPUT_DIR / "audit.csv", index=False)
decision_log.to_csv(OUTPUT_DIR / "cleaning_log.csv", index=False)

CLEAN_CSV_DTYPES = {
    "record_id": "string",
    "full_name": "string",
    "site": "string",
    "status": "string",
    "age": "Int64",
    "visit_date": "string",
    "needs_review": "boolean",
}

round_trip = pd.read_csv(
    OUTPUT_DIR / "cleaned_data.csv",
    dtype=CLEAN_CSV_DTYPES,
)
round_trip_date_text_mask = round_trip["visit_date"].str.fullmatch(
    EXACT_DATE_PATTERN,
    na=False,
)
assert (
    round_trip["visit_date"].isna() | round_trip_date_text_mask
).all()
round_trip["visit_date"] = pd.to_datetime(
    round_trip["visit_date"].where(round_trip_date_text_mask, pd.NA),
    format="%Y-%m-%d",
    errors="coerce",
)

expected_round_trip = working.reset_index(drop=True).astype(
    {
        "record_id": "string",
        "full_name": "string",
        "site": "string",
        "status": "string",
        "age": "Int64",
        "needs_review": "boolean",
    }
)
pd.testing.assert_frame_equal(
    round_trip,
    expected_round_trip,
    check_exact=True,
)

round_trip_validation_results = pd.Series(
    {
        "expected rows": len(round_trip) == 5,
        "record ID present": round_trip["record_id"].notna().all(),
        "record ID unique": round_trip["record_id"].is_unique,
        "site allowed": round_trip["site"].isin(
            ["north", "south", "west"]
        ).all(),
        "status allowed when present": round_trip["status"].dropna().isin(
            ["active", "pending", "complete"]
        ).all(),
        "age has nullable integer dtype": (
            str(round_trip["age"].dtype) == "Int64"
        ),
        "age in range when present": round_trip["age"].dropna().between(
            0,
            120,
        ).all(),
        "visit date has datetime dtype": (
            pd.api.types.is_datetime64_any_dtype(
                round_trip["visit_date"].dtype
            )
        ),
        "review flag has nullable Boolean dtype": (
            str(round_trip["needs_review"].dtype) == "boolean"
        ),
    },
    name="passed",
)

assert round_trip_validation_results.all(), (
    round_trip_validation_results[~round_trip_validation_results]
)
print("Fresh-run cleaning outputs verified")
```

A formal fresh-runtime check starts from no notebook state, reacquires the pinned raw input, runs raw→audit→decide→clean→validate→save in visual order, and reaches the final assertions without a manual upload or hidden name. The original source checksum must remain unchanged.

## LIVE DEMO 3: Validated end-to-end pipeline

[Open the Lecture 05 demo guide](demo/DEMO_GUIDE.md).

The third required demonstration runs one restartable raw→audit→clean→validate→save notebook, uses executable invariants, and emits a clean artifact plus structured audit and decision logs. It does not aggregate, join, plot, model, or automate notebook execution from a shell.

## Core scope boundary

Lecture 05 owns auditing, documented cleaning decisions, targeted value/type transformations, invariants, and provenance. It does not teach:

- joins, concatenation, or structural wide/long reshaping;
- plotting or visualization design;
- grouped aggregation or aggregating pivot tables;
- time-series ordering, filling, resampling, or rolling analysis;
- feature encoding, train/test splitting, or modeling; or
- command-line notebook automation.

Those concepts have later canonical homes. [BONUS.md](BONUS.md) contains only bounded optional cleaning extensions and is not independently assessed by this lecture.

## Handoff to Lecture 06

Carry these capabilities forward:

- preserve raw and clean artifacts separately;
- state what one row represents and which columns are candidate identifiers;
- identify schema, duplicate, missingness, sentinel, category, type, and range issues;
- make and document a cleaning decision; and
- validate row-count, uniqueness, category, type, missingness, and range invariants.

Lecture 06 consumes the validated clean artifact. It formalizes row grain and keys, then combines and structurally reshapes tables without reopening undocumented cleaning decisions.
