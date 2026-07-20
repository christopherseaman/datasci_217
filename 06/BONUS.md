# Bonus: Index-Based Combination and Advanced Concatenation Checks

This optional material extends Lecture 06 with index-based merge/join syntax and stricter concatenation provenance. None of it is required by the Lecture 06 demos, assignment, or Lecture 07.

Return to [README.md](README.md) for the required row-grain, key, cardinality, explicit-column merge, concatenation, and structural reshape workflow.

## Index-based merge with an explicit contract

An **index** is the set of row labels on a Series or DataFrame. Index-based combination is appropriate only when those labels are documented keys for the stated row grain. Test requiredness and uniqueness just as you would for key columns.

The following tables each have grain one row per site. Their named indexes are primary keys.

```python
import pandas as pd

site_capacity = pd.DataFrame(
    {
        "capacity": [40, 30, 25],
    },
    index=pd.Index(
        ["N", "S", "W"],
        dtype="string",
        name="site_code",
    ),
)
site_contact = pd.DataFrame(
    {
        "contact": ["A. Ng", "B. Soto", "C. Wells"],
    },
    index=pd.Index(
        ["N", "S", "W"],
        dtype="string",
        name="site_code",
    ),
)

assert site_capacity.index.notna().all()
assert site_capacity.index.is_unique
assert site_contact.index.notna().all()
assert site_contact.index.is_unique

site_details_audit = site_capacity.merge(
    site_contact,
    left_index=True,
    right_index=True,
    how="left",
    validate="one_to_one",
    indicator=True,
)

assert len(site_details_audit) == len(site_capacity)
assert site_details_audit["_merge"].eq("both").all()
site_details = site_details_audit.drop(columns="_merge")
```

`left_index=True` and `right_index=True` are the explicit key specification. `validate=` and the preservation checks remain necessary; moving a key into the index does not make it trustworthy.

## Join index-aligned tables deliberately

`DataFrame.join()` is concise when the receiving table keeps its index and the other table is aligned to that index. Its default is a left join. Supply `how=` and `validate=` anyway so the contract is visible.

```python
site_region = pd.DataFrame(
    {
        "region": ["north", "south", "west"],
    },
    index=pd.Index(
        ["N", "S", "W"],
        dtype="string",
        name="site_code",
    ),
)

assert site_region.index.notna().all()
assert site_region.index.is_unique

site_complete = site_details.join(
    site_region,
    how="left",
    validate="one_to_one",
)

assert len(site_complete) == len(site_details)
assert site_complete.index.equals(site_details.index)
assert site_complete["region"].notna().all()
```

When non-key columns overlap, `join()` requires explicit `lsuffix=` and `rsuffix=`. As in a column merge, suffixes identify origins but do not resolve disagreement.

### Match a column to an index

One side can keep its key as a column while the other uses a documented key index. The explicit pair is `left_on=` and `right_index=True`.

```python
visit_sites = pd.DataFrame(
    {
        "visit_id": ["V001", "V002", "V003", "V004"],
        "site_code": ["N", "N", "S", "W"],
    }
).astype(
    {
        "visit_id": "string",
        "site_code": "string",
    }
)

column_to_index_audit = visit_sites.merge(
    site_complete,
    left_on="site_code",
    right_index=True,
    how="left",
    validate="many_to_one",
    indicator=True,
)

assert len(column_to_index_audit) == len(visit_sites)
assert column_to_index_audit["visit_id"].is_unique
assert column_to_index_audit["_merge"].eq("both").all()
```

This syntax is optional. The required lecture keeps join keys in columns because that representation makes early key inspection and merge diagnostics more visible.

## Preserve advanced concat provenance with keys

The required lecture adds a source column before vertical concatenation. The `keys=` argument offers an alternative: it creates an outer index level containing the supplied source labels. The original indexes become an inner level.

```python
north_visits = pd.DataFrame(
    {
        "visit_id": ["V001", "V002"],
        "measure": [12.5, 14.0],
    },
    index=pd.Index([0, 1], name="source_row"),
).astype({"visit_id": "string"})
south_visits = pd.DataFrame(
    {
        "visit_id": ["V003"],
        "measure": [9.5],
    },
    index=pd.Index([0], name="source_row"),
).astype({"visit_id": "string"})

keyed_partitions = pd.concat(
    {
        "north_file": north_visits,
        "south_file": south_visits,
    },
    names=["source_file", "source_row"],
)

assert keyed_partitions.index.names == ["source_file", "source_row"]
assert keyed_partitions.index.is_unique
assert keyed_partitions.loc["north_file", "visit_id"].tolist() == [
    "V001",
    "V002",
]
assert keyed_partitions.loc["south_file", "visit_id"].tolist() == [
    "V003",
]
```

This hierarchical index is only a provenance container here. No hierarchical-index aggregation is required or implied. Use an ordinary source column when downstream tools expect a flat table.

## Reject overlapping concat labels with verify_integrity

`verify_integrity=True` asks `concat()` to reject duplicate labels on the concatenation axis. It checks index-label integrity, not business-key uniqueness.

```python
first_indexed = pd.DataFrame(
    {"visit_id": ["V001", "V002"]},
    index=pd.Index([0, 1], name="row_id"),
).astype({"visit_id": "string"})
second_indexed = pd.DataFrame(
    {"visit_id": ["V003", "V004"]},
    index=pd.Index([1, 2], name="row_id"),
).astype({"visit_id": "string"})

integrity_check_failed = False
try:
    pd.concat(
        [first_indexed, second_indexed],
        verify_integrity=True,
    )
except ValueError as error:
    integrity_check_failed = True
    print("expected integrity failure:", type(error).__name__)

assert integrity_check_failed
```

If the original row labels carry no meaning, resetting them during concatenation is an explicit alternative:

```python
reset_partitions = pd.concat(
    [first_indexed, second_indexed],
    ignore_index=True,
    verify_integrity=True,
)

assert reset_partitions.index.is_unique
assert reset_partitions["visit_id"].tolist() == [
    "V001",
    "V002",
    "V003",
    "V004",
]
```

If the labels do carry meaning, do not discard them merely to make the check pass. Diagnose whether overlapping labels represent duplicate observations, unrelated local numbering, or another documented condition.

## Bonus scope boundary

This bonus is limited to index-based `merge()`/`join()` and advanced `concat()` provenance/integrity checks. It introduces no required prerequisite for the course. Cleaning decisions, grouped or hierarchical aggregation, aggregating pivots, visualization, time series, modeling, databases, and performance engineering remain outside this document.
