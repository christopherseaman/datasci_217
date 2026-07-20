# Combining and Reshaping Tables with Explicit Contracts

Lecture 06 teaches a safe sequence for combining tables: state what a row represents, identify and test keys, predict the relationship between tables, choose which rows must be preserved, and then run a validated operation. The same attention to row meaning applies when stacking partitions or reshaping between wide and long forms.

Optional index-based combinations and advanced concatenation checks are collected in [BONUS.md](BONUS.md). They are not prerequisites for the required demos, assignment, or Lecture 07.

## Prerequisites

Before starting this lecture, you should be able to:

- run a notebook from a clean runtime and use portable paths;
- inspect DataFrame columns, shape, dtypes, index, missing values, and uniqueness;
- select, filter, sort, and derive columns;
- preserve raw versus cleaned data and validate explicit invariants; and
- state what one row represents and identify candidate identifiers.

Lecture 06 does not assume grouping, aggregating pivot tables, hierarchical-index manipulation, datetime resampling, rolling analysis, or visualization design.

## Learning objectives

By the end of Lecture 06, students should be able to:

1. State the row grain of each input table, identify candidate/foreign keys, and test whether the claimed keys are unique.
2. Predict one-to-one, one-to-many, many-to-one, or many-to-many merge behavior and choose an appropriate join type for a stated preservation goal.
3. Perform a merge with explicit keys, `validate=`, and `indicator=True`; inspect unmatched rows; and verify row-count/key invariants.
4. Concatenate tables vertically when their schemas represent the same row grain and horizontally when index alignment is deliberate, explaining the resulting missing values.
5. Convert a table between wide and long form with `melt()` and structural `pivot()`, explaining the uniqueness condition required for a lossless round trip.

## Colab-first execution and evidence

Required Lecture 06 demonstrations are Colab-first and also run in local Jupyter or the VS Code notebook interface. Colab's runtime is the kernel and temporary filesystem behind the notebook; local Jupyter uses a local kernel and filesystem, but the top-to-bottom execution contract is the same.

The 2026–27 compatibility candidate is Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. This is not the final release lock. In the pin-able Colab 2026.04 runtime, run `%pip install --quiet pandas==3.0.3` only when pandas 3.0.3 is not already installed, and do so before importing pandas. Do not install pandas 3.0.4; that release was yanked. Avoid reinstalling unrelated packages.

The examples below begin with the corresponding version check:

```python
import platform

import numpy as np
import pandas as pd

assert platform.python_version() == "3.12.13"
assert np.__version__ == "2.0.2"
assert pd.__version__ == "3.0.3"

print("Python:", platform.python_version())
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
```

Colab's filesystem is ephemeral. A required notebook must reacquire its pinned prepared input and create output directories from code; manual upload and mounted Drive are not defaults. Changes made in a Colab notebook opened from GitHub are not automatically saved back to the repository.

Assignment notebooks must remain runnable in clean local Jupyter. Colab becomes an assignment submission path only after the repository-save and Classroom 50 pilot is approved. Before sharing or submitting, remove credentials, private records, and sensitive output. Stored cell output is never execution evidence: a grader runs a fresh copy. Files written under `output/` are separate generated artifacts and must be recreated by restart-and-run-all.

## Start with row grain and keys

**Row grain** states exactly what one row represents. It is also called the unit of observation. Two tables can share a column name while representing different kinds of rows, so grain must be stated before choosing any combination operation.

An **identifier** is a value used to distinguish an entity or observation. A **candidate key** is one column, or a combination of columns, that is expected to identify each row uniquely and without missing values. A **primary key** is the candidate key chosen as the table's official row identifier. A **foreign key** is a column, or combination of columns, whose values refer to a key in another table.

The running prepared fixture has two tables:

- `visits` has grain **one row per recorded visit**. Its primary key is `visit_id`. The pair `participant_id` and `visit_number` is another candidate key. `site_code` is a foreign key.
- `sites` has grain **one row per site**. Its primary key is `site_code`.

The supplied CSV text is fixed and verified by checksum. A released demo or assignment uses the same policy with a committed file or immutable HTTPS source rather than a manual upload.

```python
from hashlib import sha256
from io import StringIO

VISITS_SOURCE = """visit_id,participant_id,visit_number,site_code,status,measure
V001,P01,1,N,complete,12.5
V002,P01,2,N,complete,14.0
V003,P02,1,S,complete,9.5
V004,P03,1,W,complete,11.0
V005,P04,1,N,complete,13.5
"""

SITES_SOURCE = """site_code,site_name,region
N,North Clinic,north
S,South Clinic,south
W,West Clinic,west
"""

assert sha256(VISITS_SOURCE.encode("utf-8")).hexdigest() == (
    "510b3946f74dd1b899943b44c08577104a862c9bbd50d674eb74f5b1d0d9d3b5"
)
assert sha256(SITES_SOURCE.encode("utf-8")).hexdigest() == (
    "452893794bb3fcf804485ba7f701ebd657d92aba9657f1469e36aff73fd17cf2"
)

visits = pd.read_csv(
    StringIO(VISITS_SOURCE),
    dtype={
        "visit_id": "string",
        "participant_id": "string",
        "site_code": "string",
        "status": "string",
    },
)
sites = pd.read_csv(
    StringIO(SITES_SOURCE),
    dtype={
        "site_code": "string",
        "site_name": "string",
        "region": "string",
    },
)

print(visits)
print(sites)
```

### Test the key claims

A key claim needs two separate checks: required key values are not missing, and the claimed key values are unique. `Series.is_unique` checks a single-column key. `DataFrame.duplicated(subset=...)` checks a composite key.

```python
assert visits["visit_id"].notna().all()
assert visits["visit_id"].is_unique

assert visits[["participant_id", "visit_number"]].notna().all().all()
assert not visits.duplicated(
    subset=["participant_id", "visit_number"]
).any()

assert sites["site_code"].notna().all()
assert sites["site_code"].is_unique

assert visits["site_code"].notna().all()
```

pandas treats missing merge keys as equal to one another: a missing key on the left can match a missing key on the right. This differs from usual SQL null behavior. For a required key, assert nonmissingness before merging. If missing keys are meaningful, isolate them and document the intended behavior instead of allowing an accidental match. See the pandas [`merge` documentation](https://pandas.pydata.org/docs/reference/api/pandas.merge.html).

## Predict cardinality and preservation before merging

**Cardinality** describes how many rows on each side can share a key:

- **one-to-one:** each key appears at most once on both sides;
- **one-to-many:** each key appears at most once on the left and can repeat on the right;
- **many-to-one:** a key can repeat on the left and appears at most once on the right; and
- **many-to-many:** a key can repeat on both sides.

Cardinality predicts row behavior. For a many-to-many key value `k`, the matching output contains

`n_left(k) * n_right(k)`

rows. Across all shared keys, the number of matched rows is the sum of those products. That multiplication is sometimes intended, but it must never be a surprise.

State a **preservation goal** next: which table's observations must remain? Four common preservation joins answer different questions:

- an **inner join** keeps only matching keys;
- a **left join** keeps every left row;
- a **right join** keeps every right row; and
- an **outer join** keeps rows from both sides.

Here, site codes repeat in `visits` and are unique in `sites`, so the relationship is many-to-one. The goal is to keep every recorded visit, including a visit whose site metadata might be absent, so the intended operation is a left join. With a unique right key, a many-to-one left merge produces exactly one output row for every left row.

```python
visit_site_counts = visits["site_code"].value_counts(dropna=False)
site_key_counts = sites["site_code"].value_counts(dropna=False)

assert visit_site_counts.max() > 1
assert site_key_counts.max() == 1

relationship = "many_to_one"
preservation_goal = "keep every visit row"
expected_output_rows = len(visits)

print("relationship:", relationship)
print("preservation goal:", preservation_goal)
```

## Merge with explicit keys and diagnostics

Always name merge keys with `on=` or `left_on=` and `right_on=`. Do not let pandas infer a join from every same-named column. `validate=` turns the predicted cardinality into an executable contract. `indicator=True` adds a temporary `_merge` column that distinguishes matches from unmatched rows.

```python
visits_with_site_audit = visits.merge(
    sites,
    on="site_code",
    how="left",
    validate="many_to_one",
    indicator=True,
    suffixes=("_visit", "_site"),
)

print(visits_with_site_audit)
```

An **unmatched row** has a key with no partner on the other side. Inspect it before dropping the indicator. Then verify the preservation goal and other invariants directly.

```python
merge_source_counts = (
    visits_with_site_audit["_merge"]
    .value_counts(dropna=False)
    .to_dict()
)
unmatched_visits = visits_with_site_audit.loc[
    visits_with_site_audit["_merge"].eq("left_only"),
    ["visit_id", "site_code"],
]

assert merge_source_counts["both"] == len(visits)
assert merge_source_counts["left_only"] == 0
assert merge_source_counts["right_only"] == 0
assert unmatched_visits.empty
assert len(visits_with_site_audit) == expected_output_rows
assert visits_with_site_audit["visit_id"].is_unique
assert set(visits_with_site_audit["visit_id"]) == set(visits["visit_id"])

visits_with_site = visits_with_site_audit.drop(columns="_merge")
```

### Composite keys and overlapping column names

A **composite key** uses more than one column because no single column identifies a row. Name every component in `on=[...]` and test the combination before merging.

When non-key columns have the same name on both sides, `suffixes=` makes their origins explicit. A suffix resolves a naming collision; it does not prove that the values agree.

```python
visit_reviews = pd.DataFrame(
    {
        "participant_id": ["P01", "P01", "P02", "P03", "P04"],
        "visit_number": [1, 2, 1, 1, 1],
        "status": ["verified", "verified", "pending", "verified", "pending"],
    }
).astype(
    {
        "participant_id": "string",
        "status": "string",
    }
)

assert visit_reviews[
    ["participant_id", "visit_number"]
].notna().all().all()
assert not visit_reviews.duplicated(
    subset=["participant_id", "visit_number"]
).any()

visits_with_review = visits.merge(
    visit_reviews,
    on=["participant_id", "visit_number"],
    how="left",
    validate="one_to_one",
    indicator=True,
    suffixes=("_visit", "_review"),
)

assert {"status_visit", "status_review"} <= set(visits_with_review.columns)
assert visits_with_review["_merge"].eq("both").all()
assert len(visits_with_review) == len(visits)
```

### Make failure cases observable

The next cell tests four cases independently:

1. a duplicate right-side key violates the many-to-one contract;
2. an orphan foreign key remains visible as `left_only`;
3. missing pandas keys match unless they are isolated; and
4. a many-to-many result follows the per-key multiplication rule.

```python
duplicate_sites = pd.concat(
    [sites, sites.iloc[[0]]],
    ignore_index=True,
)

duplicate_contract_failed = False
try:
    visits.merge(
        duplicate_sites,
        on="site_code",
        how="left",
        validate="many_to_one",
    )
except pd.errors.MergeError as error:
    duplicate_contract_failed = True
    print("expected validation failure:", type(error).__name__)

assert duplicate_contract_failed

orphan_row = pd.DataFrame(
    {
        "visit_id": pd.Series(["V006"], dtype="string"),
        "participant_id": pd.Series(["P05"], dtype="string"),
        "visit_number": [1],
        "site_code": pd.Series(["X"], dtype="string"),
        "status": pd.Series(["complete"], dtype="string"),
        "measure": [8.0],
    }
)
visits_with_orphan = pd.concat([visits, orphan_row], ignore_index=True)
orphan_audit = visits_with_orphan.merge(
    sites,
    on="site_code",
    how="left",
    validate="many_to_one",
    indicator=True,
)
assert orphan_audit.loc[
    orphan_audit["_merge"].eq("left_only"),
    "visit_id",
].tolist() == ["V006"]
assert len(orphan_audit) == len(visits_with_orphan)

null_left = pd.DataFrame(
    {
        "key": pd.Series(["A", pd.NA], dtype="string"),
        "left_value": [1, 2],
    }
)
null_right = pd.DataFrame(
    {
        "key": pd.Series(["A", pd.NA], dtype="string"),
        "right_value": [10, 20],
    }
)
null_match = null_left.merge(
    null_right,
    on="key",
    how="inner",
    validate="one_to_one",
)
assert len(null_match) == 2
assert null_match["key"].isna().sum() == 1

safe_left = null_left.loc[null_left["key"].notna()].copy()
safe_right = null_right.loc[null_right["key"].notna()].copy()
assert safe_left["key"].notna().all()
assert safe_right["key"].notna().all()

many_left = pd.DataFrame({"key": ["A", "A", "B"]})
many_right = pd.DataFrame({"key": ["A", "A", "A", "B", "B"]})
many_result = many_left.merge(
    many_right,
    on="key",
    how="inner",
    validate="many_to_many",
)

left_counts = many_left["key"].value_counts()
right_counts = many_right["key"].value_counts()
shared_keys = left_counts.index.intersection(right_counts.index)
expected_many_rows = sum(
    int(left_counts[key]) * int(right_counts[key])
    for key in shared_keys
)

assert expected_many_rows == 8
assert len(many_result) == expected_many_rows
```

## LIVE DEMO 1: Validated merge diagnostics

The first required demonstration follows the [demo guide](demo/DEMO_GUIDE.md): declare grain and keys, predict a many-to-one relationship, trigger `validate=` with a duplicate dimension key, isolate the problem, inspect an orphan with `indicator=True`, perform the intended left merge, and verify preserved IDs and row count. The duplicate is repaired only after the diagnostic identifies it; silent deduplication is not part of the merge.

## Concatenate tables with deliberate alignment

`pd.concat()` combines whole objects along an axis. It does not match foreign keys. Vertical concatenation stacks rows and aligns by column labels. It is appropriate when partitions have the same row grain and compatible schemas.

Add a source column before stacking so every output row retains **provenance**—where that row came from.

```python
january = visits.iloc[:3].copy().assign(source_partition="january")
february = visits.iloc[3:].copy().assign(source_partition="february")

expected_partition_columns = [
    "visit_id",
    "participant_id",
    "visit_number",
    "site_code",
    "status",
    "measure",
    "source_partition",
]
assert list(january.columns) == expected_partition_columns
assert list(february.columns) == expected_partition_columns

all_partitions = pd.concat(
    [january, february],
    axis="index",
    ignore_index=True,
)

assert len(all_partitions) == len(january) + len(february)
assert all_partitions["visit_id"].is_unique
assert all_partitions["source_partition"].value_counts().to_dict() == {
    "january": 3,
    "february": 2,
}
```

### Column alignment can create missing values

Vertical concatenation uses the union of column labels. If one partition lacks a column or introduces another, pandas creates missing positions. That behavior is useful only when the schema difference is expected.

```python
schema_drift_partition = (
    february
    .drop(columns="measure")
    .assign(review_note="late import")
)
alignment_preview = pd.concat(
    [january, schema_drift_partition],
    ignore_index=True,
    sort=False,
)

assert alignment_preview.loc[
    alignment_preview["source_partition"].eq("february"),
    "measure",
].isna().all()
assert alignment_preview.loc[
    alignment_preview["source_partition"].eq("january"),
    "review_note",
].isna().all()
```

This preview diagnoses a schema mismatch; it does not decide whether either column should be filled, dropped, or renamed.

### Horizontal concatenation aligns index labels

Horizontal concatenation uses `axis="columns"`. It aligns rows by index label rather than by row position, so it is deliberate only when both indexes are meaningful keys for the same row grain.

```python
measure_by_visit = (
    visits
    .set_index("visit_id")
    .loc[["V001", "V002", "V003"], ["measure"]]
)
review_by_visit = pd.DataFrame(
    {"review_score": [7.0, 8.0, 9.0]},
    index=pd.Index(
        ["V002", "V003", "V006"],
        dtype="string",
        name="visit_id",
    ),
)

horizontal_features = pd.concat(
    [measure_by_visit, review_by_visit],
    axis="columns",
)

assert set(horizontal_features.index) == {"V001", "V002", "V003", "V006"}
assert horizontal_features.isna().sum().to_dict() == {
    "measure": 1,
    "review_score": 1,
}
assert pd.isna(horizontal_features.loc["V001", "review_score"])
assert pd.isna(horizontal_features.loc["V006", "measure"])
```

The two missing values have exact structural causes: `V001` exists only in the measure table, and `V006` exists only in the review table.

## LIVE DEMO 2: Concat provenance and alignment

The second required demonstration follows the [demo guide](demo/DEMO_GUIDE.md): vertically stack same-grain partitions with explicit source labels, verify the row total and source counts, then horizontally align two feature tables whose index labels differ. Students explain each resulting missing position from the labels that were present on only one side.

## Reshape between wide and long forms

A **wide-form** table stores repeated measurements in separate columns. A **long-form** table stores the measurement name in one column and its value in another, producing more rows.

**Identifier variables** identify the observation across repeated measurements; they remain as columns during a melt. **Measured variables** are the repeated-measure columns whose names and values move into the long representation.

In the next table, `participant_id` and `site_code` are identifier variables. `baseline_score` and `followup_score` are measured variables.

```python
wide_scores = pd.DataFrame(
    {
        "participant_id": ["P01", "P02", "P03"],
        "site_code": ["N", "S", "W"],
        "baseline_score": [10.0, 8.5, 11.0],
        "followup_score": [12.0, 9.5, 13.0],
    }
).astype(
    {
        "participant_id": "string",
        "site_code": "string",
    }
)

assert wide_scores["participant_id"].is_unique
print(wide_scores)
```

### Melt wide measurements into rows

`melt()` keeps `id_vars` and moves named `value_vars` into a variable column and a value column. This is a structural operation; it does not summarize values.

```python
long_scores = wide_scores.melt(
    id_vars=["participant_id", "site_code"],
    value_vars=["baseline_score", "followup_score"],
    var_name="visit_label",
    value_name="score",
)
long_scores["visit_label"] = long_scores["visit_label"].astype("str")

assert len(long_scores) == len(wide_scores) * 2
assert not long_scores.duplicated(
    subset=["participant_id", "site_code", "visit_label"]
).any()

print(long_scores)
```

### Pivot long measurements back to columns

Structural `pivot()` requires at most one value for every identifier-variable combination and output column label. Here that means each (`participant_id`, `site_code`, `visit_label`) combination must be unique. When it is, the wide-to-long-to-wide round trip can be lossless.

```python
round_trip_scores = (
    long_scores
    .pivot(
        index=["participant_id", "site_code"],
        columns="visit_label",
        values="score",
    )
    .reset_index()
)
round_trip_scores.columns.name = None
round_trip_scores = round_trip_scores.loc[:, wide_scores.columns]

expected_scores = (
    wide_scores
    .sort_values(["participant_id", "site_code"])
    .reset_index(drop=True)
)
round_trip_scores = (
    round_trip_scores
    .sort_values(["participant_id", "site_code"])
    .reset_index(drop=True)
)

pd.testing.assert_frame_equal(round_trip_scores, expected_scores)
```

If the required combination is duplicated, `pivot()` refuses to guess which value to keep. That failure is evidence of a violated structural contract.

```python
duplicate_long_scores = pd.concat(
    [long_scores, long_scores.iloc[[0]]],
    ignore_index=True,
)

duplicate_pivot_failed = False
try:
    duplicate_long_scores.pivot(
        index=["participant_id", "site_code"],
        columns="visit_label",
        values="score",
    )
except ValueError as error:
    duplicate_pivot_failed = True
    print("expected pivot failure:", type(error).__name__)

assert duplicate_pivot_failed
```

## LIVE DEMO 3: Structural melt/pivot round trip

The third required demonstration follows the [demo guide](demo/DEMO_GUIDE.md): state the wide and long grains, melt unique repeated-measure columns, verify the expected long row count and key combination, pivot back without aggregation, and compare the reconstructed table with the original. A planted duplicate combination must make `pivot()` fail. Lecture 08 will explain how a justified aggregation changes that question.

## Produce validated fresh outputs

The merged output has grain **one row per recorded visit**. The long score output has grain **one row per participant, site, and measurement occasion**. Save each only after its invariants pass, then read it back with the declared schema. The output files are generated artifacts, not notebook-state evidence.

```python
from pathlib import Path

analysis_ready = visits_with_site.loc[
    :,
    [
        "visit_id",
        "participant_id",
        "visit_number",
        "site_code",
        "status",
        "measure",
        "site_name",
        "region",
    ],
].copy()

assert len(analysis_ready) == len(visits)
assert analysis_ready["visit_id"].is_unique
assert analysis_ready[["site_name", "region"]].notna().all().all()

output_directory = Path("output")
output_directory.mkdir(parents=True, exist_ok=True)
analysis_path = output_directory / "visits_with_site.csv"
long_path = output_directory / "scores_long.csv"

analysis_ready.to_csv(analysis_path, index=False)
long_scores.to_csv(long_path, index=False)

analysis_reloaded = pd.read_csv(
    analysis_path,
    dtype={
        "visit_id": "string",
        "participant_id": "string",
        "visit_number": "int64",
        "site_code": "string",
        "status": "string",
        "measure": "float64",
        "site_name": "string",
        "region": "string",
    },
)
long_reloaded = pd.read_csv(
    long_path,
    dtype={
        "participant_id": "string",
        "site_code": "string",
        "visit_label": "str",
        "score": "float64",
    },
)

pd.testing.assert_frame_equal(analysis_reloaded, analysis_ready)
pd.testing.assert_frame_equal(long_reloaded, long_scores)
```

A final check is procedural: restart the runtime, run every cell from the top, and confirm that the pinned source is reacquired, all assertions pass, and both files are recreated. A stale in-memory table or an old file under `output/` does not satisfy that check.

## Lecture 07 handoff

After this lecture, students should be able to:

- produce one prepared analysis table with a stated row grain;
- convert supplied wide data to long form suitable for seaborn; and
- explain when rows were added or lost through a join or concatenation.

Lecture 07 can therefore begin with supplied prepared data and focus on chart purpose, integrity, accessibility, and plotting interfaces rather than repairing table structure.

## Core scope boundary

Required Lecture 06 work is limited to explicit-key `merge()`, deliberate `concat()`, and nonaggregating `melt()`/`pivot()`. It does not reopen cleaning decisions. GroupBy, aggregating `pivot_table()`, hierarchical-index aggregation, visualization, time series, modeling, databases, and performance engineering belong to later lectures or other courses.
