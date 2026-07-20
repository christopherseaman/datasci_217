# Grouping and Aggregation with an Explicit Result Grain

Lecture 08 teaches how to change a table's row grain deliberately. The required workflow starts by stating what one input row represents and what defines a group, predicts the grouped result, and only then computes a summary or adds group context back to the original rows.

Optional categorical-grouping and index-layout extensions are collected in [BONUS.md](BONUS.md). They are not prerequisites for the required demos, assignment, or Lecture 09.

## Prerequisites

Before starting this lecture, students should be able to:

- select DataFrame rows and columns and distinguish the DataFrame index from ordinary columns;
- recognize missing values and decide whether they should contribute to a calculation;
- state row grain, identify keys, validate merge cardinality, and verify post-merge grain;
- distinguish long and wide tabular forms; and
- read a supplied summary and make or critique one honest chart from it.

Lecture 08 does not assume periods, frequency, resampling, rolling operations, statistical tests, modeling, remote-computing workflows, or performance engineering.

## Learning objectives

By the end of Lecture 08, students should be able to:

1. State the input row grain and grouping key, predict the identity and number of groups and the output grain, and verify those predictions before interpreting a grouped result.
2. Choose `size`, `count`, or `nunique` to match a counting question and produce a flat summary with named aggregation.
3. Distinguish aggregation from `transform` by output grain and use `transform` to add one group statistic while preserving the input row count and index alignment.
4. Produce grouped results with deliberate key columns, value columns, ordering, and index placement, including one bounded two-key summary with an explicit output grain.
5. Build and interpret one aggregating `pivot_table` by naming its index, columns, values, aggregation function, observed-category policy, and missing combinations.

## Colab-first execution and evidence

Required Lecture 08 demonstrations are Colab-first and also run in local Jupyter or the VS Code notebook interface. The 2026–27 compatibility candidate is Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. This is not the final release lock.

In the pin-able Colab 2026.04 runtime, a setup cell must conditionally install pandas 3.0.3 before pandas is imported when the installed version differs. Do not install pandas 3.0.4; that release was yanked. Avoid reinstalling unrelated Colab packages. Every required notebook prints the versions actually in use and must pass both in a fresh Colab runtime and in clean local Jupyter before publication.

Colab's filesystem is ephemeral. Required notebooks use fixed in-notebook data or reacquire a pinned source in code; manual upload and mounted Drive are not defaults. Changes made in a Colab notebook opened from GitHub are not automatically saved back to the repository.

Assignment notebooks remain runnable in clean local Jupyter. Colab becomes an assignment submission path only after the repository-save and Classroom 50 pilot is approved. Remove credentials, private records, and sensitive output before sharing. Stored cell output is not execution evidence: restart the runtime and run every cell in order.

The examples begin with the candidate version check:

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

## Start with rows, groups, and result grain

**Input row grain** states what one row in the source table represents. A **grouping key** is the column, or bounded combination of columns, whose values determine which input rows belong together. A **group** is the set of input rows sharing one observed key value or key combination. The **grouping unit** is the real-world category or entity represented by that group.

For example, if one input row represents one healthcare encounter and `facility` is the grouping key, one group contains all encounter rows for one facility. The grouping unit is one facility.

**Output row grain** states what one row in a result represents. An **aggregation** reduces the rows in each group to one or more summary values. A one-key aggregation normally produces one result row per observed group, so it changes the grain from one encounter to one facility.

`DataFrame.groupby()` creates a **GroupBy object**. That object records how rows are split but is not itself a summary table. A later operation such as `size()`, `mean()`, or `agg()` performs a calculation and combines the group results. This is the split–apply–combine pattern:

1. split input rows according to the grouping key;
2. apply a calculation inside each group; and
3. combine the group results with a deliberate index and column layout.

The pandas [`DataFrame.groupby()` reference](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) documents the grouping controls used below.

### Use a deterministic encounter table

**Deterministic data** have fixed, documented values, so a fresh execution produces the same groups and results. The running table has grain **one row per recorded encounter**. `encounter_id` identifies rows; a provider can appear in several encounters.

```python
FACILITY_LEVELS = ["North", "South", "West", "Remote"]
SERVICE_LEVELS = ["Consult", "Follow-up", "Procedure"]

encounters = pd.DataFrame(
    {
        "encounter_id": [
            "E001", "E002", "E003", "E004",
            "E005", "E006", "E007", "E008",
            "E009", "E010", "E011", "E012",
        ],
        "facility": [
            "North", "North", "North", "North",
            "South", "South", "South", "South",
            "West", "West", "West", "West",
        ],
        "provider_id": [
            "P01", "P01", "P02", "P02",
            "P03", "P03", "P04", "P04",
            "P05", "P05", "P06", "P06",
        ],
        "service": [
            "Consult", "Follow-up", "Consult", "Procedure",
            "Consult", "Consult", "Procedure", "Procedure",
            "Consult", "Procedure", "Consult", "Follow-up",
        ],
        "charge": [
            120, 80, 150, 210,
            110, 90, 220, 125,
            130, 200, 140, 75,
        ],
        "wait_minutes": [
            20, 12, 30, 50,
            18, 16, 45, 25,
            25, 40, 35, 15,
        ],
        "rating": [
            4, pd.NA, 5, 5,
            4, pd.NA, pd.NA, 4,
            3, 4, 3, 4,
        ],
    }
).astype(
    {
        "encounter_id": "string",
        "provider_id": "string",
        "rating": "Int64",
    }
)

encounters["facility"] = pd.Categorical(
    encounters["facility"],
    categories=FACILITY_LEVELS,
    ordered=True,
)
encounters["service"] = pd.Categorical(
    encounters["service"],
    categories=SERVICE_LEVELS,
    ordered=True,
)

assert encounters.shape == (12, 7)
assert encounters["encounter_id"].is_unique
assert encounters[["facility", "service"]].notna().all().all()

print(encounters)
```

`facility` declares four possible categorical levels, but only North, South, and West occur in the input rows. `service` declares three levels, and all three occur somewhere in the data. A declared category is not the same thing as an observed group.

### Make categorical-group policy explicit

The `observed=` parameter applies when any grouping key has pandas categorical dtype:

- `observed=True` returns only category values or combinations that occur in the input;
- `observed=False` can materialize declared but unused category values or combinations.

pandas 3.0 changed the `groupby()` default to `observed=True`. Required Lecture 08 code still writes `observed=True` so the result policy remains visible to a reader and stable if code is moved between environments. The same policy is written explicitly in the pivot later.

`dropna=` answers a different question: whether missing grouping-key values form a group. The required fixture has no missing grouping keys, so its calls use `dropna=True`. Optional missing-key and unused-category policies appear in [BONUS.md](BONUS.md).

`sort=True` makes output ordering deliberate. For these ordered categorical keys, results follow the declared category order. With ordinary strings, `sort=True` orders the group labels.

### Predict before computing

Before executing `groupby()`, write the contract:

- input row grain: one recorded encounter;
- grouping key: `facility`;
- grouping unit: one observed facility;
- predicted group identities: North, South, and West;
- predicted number of groups: three; and
- after aggregation, output row grain: one observed facility.

Now create the GroupBy object and verify the prediction with `ngroups` and `size()`.

```python
facility_groups = encounters.groupby(
    "facility",
    observed=True,
    sort=True,
    dropna=True,
)

facility_sizes = facility_groups.size().rename("encounter_count")

assert facility_groups.ngroups == 3
assert facility_sizes.index.astype("string").tolist() == [
    "North", "South", "West"
]
assert facility_sizes.tolist() == [4, 4, 4]
assert int(facility_sizes.sum()) == len(encounters)

print(facility_sizes)
```

The result has three values because it has grain one observed facility. The input still has twelve rows; constructing a GroupBy object did not mutate it.

## Match the count to the question

Three operations called “counting” answer different questions:

| Operation | What it counts | Missing-value behavior | Example question |
|---|---|---|---|
| `GroupBy.size()` | input rows in each group | counts the row even when another field is missing | How many encounters were recorded? |
| `GroupBy["column"].count()` | nonmissing values in one selected column | excludes missing values in that column | How many encounters have a recorded rating? |
| `GroupBy["column"].nunique()` | distinct values in one selected column | excludes missing values by default | How many distinct providers appear? |

Choose the operation from the question and the source column's meaning. Counting `provider_id` with `count()` would count encounters with a provider value, not distinct providers.

The three results below share the same facility-key index, so the already-learned `concat()` operation can align them as columns before `reset_index()` makes the key an ordinary column.

```python
count_comparison = pd.concat(
    [
        facility_groups.size().rename("encounter_count"),
        facility_groups["rating"].count().rename("rating_count"),
        facility_groups["provider_id"]
        .nunique(dropna=True)
        .rename("unique_provider_count"),
    ],
    axis="columns",
).reset_index()

assert count_comparison.columns.tolist() == [
    "facility",
    "encounter_count",
    "rating_count",
    "unique_provider_count",
]
assert count_comparison["encounter_count"].tolist() == [4, 4, 4]
assert count_comparison["rating_count"].tolist() == [3, 2, 4]
assert count_comparison["unique_provider_count"].tolist() == [2, 2, 2]

print(count_comparison)
```

North has four encounter rows but only three nonmissing ratings. It has two distinct providers even though each provider occurs in two rows. Those are three correct answers to three different questions.

## LIVE DEMO 1: Predict grouping grain and counts

[Open the Lecture 08 demo guide](demo/DEMO_GUIDE.md).

The first required demonstration starts from the fixed encounter table. Students state the input grain, grouping key, grouping unit, observed-category policy, group identities, group count, and aggregated output grain before executing code. They then use the missing rating and repeated provider to choose and verify `size`, `count`, and `nunique` from the question each operation answers.

## Name each aggregation output

`agg()` combines one or more aggregations. **Named aggregation** gives every result column a deliberate name while specifying both its source column and calculation:

```text
output_column_name=("input_column_name", "aggregation")
```

The keywords become flat output column names. This avoids ambiguous names and nested value-column labels. See the pandas [`DataFrameGroupBy.aggregate()` reference](https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.aggregate.html).

```python
facility_summary = (
    encounters.groupby(
        "facility",
        as_index=False,
        observed=True,
        sort=True,
        dropna=True,
    )
    .agg(
        encounter_count=("encounter_id", "size"),
        rating_count=("rating", "count"),
        unique_provider_count=("provider_id", "nunique"),
        total_charge=("charge", "sum"),
        mean_wait_minutes=("wait_minutes", "mean"),
    )
)

assert facility_summary.columns.tolist() == [
    "facility",
    "encounter_count",
    "rating_count",
    "unique_provider_count",
    "total_charge",
    "mean_wait_minutes",
]
assert facility_summary["total_charge"].tolist() == [560, 545, 545]
assert np.allclose(
    facility_summary["mean_wait_minutes"],
    [28.0, 26.0, 28.75],
)
assert int(facility_summary["encounter_count"].sum()) == len(encounters)

print(facility_summary)
```

The result is flat because `as_index=False` keeps `facility` as a column and named aggregation creates one level of value-column names. Its output grain is one observed facility. The encounter count conserves the twelve input rows because every encounter belongs to exactly one nonmissing facility group.

## Contrast aggregation with transform

An aggregation produces one result row per group and therefore reduces or changes row grain. A GroupBy **transform** calculates within groups but returns one value aligned to every input row. For a selected Series, pandas defines `transform()` as producing a same-indexed Series; see the [`SeriesGroupBy.transform()` reference](https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.transform.html).

The next operation computes each facility's mean charge and broadcasts that value to all encounter rows from the facility. It then calculates each encounter's difference from its facility mean.

```python
facility_mean_charge = (
    encounters.groupby(
        "facility",
        observed=True,
        sort=True,
        dropna=True,
    )["charge"]
    .transform("mean")
)

encounters_with_context = encounters.assign(
    facility_mean_charge=facility_mean_charge,
    difference_from_facility_mean=(
        encounters["charge"] - facility_mean_charge
    ),
)

assert len(encounters_with_context) == len(encounters)
pd.testing.assert_index_equal(
    encounters_with_context.index,
    encounters.index,
)
assert encounters_with_context.loc[0, "facility_mean_charge"] == 140.0
assert encounters_with_context.loc[0, "difference_from_facility_mean"] == -20.0
assert encounters_with_context.loc[4, "facility_mean_charge"] == 136.25

print(
    encounters_with_context[
        [
            "encounter_id",
            "facility",
            "charge",
            "facility_mean_charge",
            "difference_from_facility_mean",
        ]
    ]
)
```

The result's grain remains one encounter. Repeating a facility mean beside encounter rows is intentional here because the new column provides group context for each encounter. `transform()` is not an aggregation result merely because it uses an aggregation-like calculation internally.

Before assigning any group result back to source rows, verify both `len(result) == len(source)` and `result.index.equals(source.index)`. A three-row facility summary cannot be assigned positionally as though it were a twelve-row encounter-level transform.

## Make columns, index, and ordering deliberate

The DataFrame **index** labels result rows; it is not automatically a data variable. With `as_index=True`, the grouping key becomes the grouped result's index. With `as_index=False`, the key remains an ordinary result column.

Both layouts can be valid. Choose one deliberately based on what consumes the table next, and state the output grain either way.

```python
indexed_charge_summary = (
    encounters.groupby(
        "facility",
        as_index=True,
        observed=True,
        sort=True,
        dropna=True,
    )
    .agg(mean_charge=("charge", "mean"))
)

assert indexed_charge_summary.index.name == "facility"
assert indexed_charge_summary.columns.tolist() == ["mean_charge"]
assert indexed_charge_summary.index.astype("string").tolist() == [
    "North", "South", "West"
]

print(indexed_charge_summary)
```

A two-key grouping uses the combination of both key values to define one group. The next result uses `as_index=False` so both keys are explicit columns and the value columns remain flat. Its output grain is **one observed facility–service combination**.

```python
facility_service_summary = (
    encounters.groupby(
        ["facility", "service"],
        as_index=False,
        observed=True,
        sort=True,
        dropna=True,
    )
    .agg(
        encounter_count=("encounter_id", "size"),
        mean_charge=("charge", "mean"),
    )
)

assert facility_service_summary.columns.tolist() == [
    "facility",
    "service",
    "encounter_count",
    "mean_charge",
]
assert len(facility_service_summary) == 8
assert int(facility_service_summary["encounter_count"].sum()) == len(
    encounters
)
assert not (
    facility_service_summary["facility"].eq("South")
    & facility_service_summary["service"].eq("Follow-up")
).any()

print(facility_service_summary)
```

There are eight output rows, not twelve and not the twelve possible combinations of four declared facilities and three services. `observed=True` omits the unused Remote level and the unobserved South–Follow-up combination. The grouping key, observed policy, output ordering, columns, and result grain are all explicit.

## LIVE DEMO 2: Named aggregation and transform

[Open the Lecture 08 demo guide](demo/DEMO_GUIDE.md).

The second required demonstration uses the same key twice. Students first build a flat, named facility summary with one row per observed group. They then add a facility statistic to every encounter with `transform()`, prove that row count and index are unchanged, and diagnose why an aggregated three-row result has the wrong grain for direct encounter-row assignment. A bounded two-key summary makes key columns and ordering explicit without requiring hierarchical-index manipulation.

## Build one aggregating pivot table

A **pivot table** is an aggregated reshape. It groups long-form input rows and places the grouped result across two display axes. Before calling `pivot_table()`, name five parts:

- `index`: the grouping key whose observed values become result rows;
- `columns`: the grouping key whose observed values become result columns;
- `values`: the numeric column being summarized;
- `aggfunc`: the aggregation applied when several input rows occupy one cell; and
- `observed`: whether unused categorical values or combinations may appear.

For this table:

- `index="facility"` makes one result row per observed facility;
- `columns="service"` makes one result column per observed service;
- `values="charge"` supplies the measurements;
- `aggfunc="mean"` computes mean charge for each facility–service group; and
- `observed=True` excludes unused categorical levels while preserving a missing cell for a combination absent within otherwise observed row and column levels.

This is different from the structural `pivot()` taught in Lecture 06. Structural `pivot()` requires each row/column combination to be unique and does not aggregate. `pivot_table()` deliberately combines repeated combinations. See the pandas [`pivot_table()` reference](https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html).

```python
mean_charge_pivot = pd.pivot_table(
    encounters,
    index="facility",
    columns="service",
    values="charge",
    aggfunc="mean",
    observed=True,
    sort=True,
    dropna=True,
)

assert mean_charge_pivot.index.name == "facility"
assert mean_charge_pivot.columns.name == "service"
assert mean_charge_pivot.index.astype("string").tolist() == [
    "North", "South", "West"
]
assert mean_charge_pivot.columns.astype("string").tolist() == [
    "Consult", "Follow-up", "Procedure"
]
assert mean_charge_pivot.loc["North", "Consult"] == 135.0
assert mean_charge_pivot.loc["South", "Procedure"] == 172.5
assert pd.isna(mean_charge_pivot.loc["South", "Follow-up"])
assert "Remote" not in mean_charge_pivot.index

print(mean_charge_pivot)
```

Every populated pivot cell should agree with the equivalent row in the earlier two-key GroupBy summary. The loop below verifies that invariant without introducing another pivot.

```python
for grouped_row in facility_service_summary.itertuples(index=False):
    pivot_value = mean_charge_pivot.loc[
        grouped_row.facility,
        grouped_row.service,
    ]
    assert np.isclose(pivot_value, grouped_row.mean_charge)
```

The South–Follow-up cell is missing because the input contains no encounter with that key combination. It means **no input row for this combination**, not a measured charge of zero. Replacing that missing value with zero would assert new domain meaning and requires separate justification.

The pivot's displayed row grain is one observed facility, while each populated cell summarizes one observed facility–service group. Reading both levels prevents the wide display from hiding what was aggregated.

### Finish with fresh-runtime invariants

A fresh execution should prove the complete contract rather than trust stored output.

```python
assert encounters.shape == (12, 7)
assert facility_groups.ngroups == 3
assert facility_summary.shape == (3, 6)
assert int(facility_summary["encounter_count"].sum()) == 12
assert encounters_with_context.shape == (12, 9)
pd.testing.assert_index_equal(
    encounters_with_context.index,
    encounters.index,
)
assert facility_service_summary.shape == (8, 4)
assert mean_charge_pivot.shape == (3, 3)
assert pd.isna(mean_charge_pivot.loc["South", "Follow-up"])

print("Lecture 08 core verification passed.")
```

## LIVE DEMO 3: One aggregating pivot

[Open the Lecture 08 demo guide](demo/DEMO_GUIDE.md).

The third required demonstration predicts the pivot specification before execution, builds exactly one mean-charge pivot, compares every populated cell with the equivalent two-key GroupBy result, and interprets South–Follow-up as an absent input combination rather than zero. It may end with at most one already-familiar Lecture 07 chart if that chart clarifies the grouped table; plotting is not a new objective.

## Handoff to Lecture 09

After this lecture, students should be able to:

- define a grouping key and the real-world unit represented by each group;
- distinguish one input row from one aggregated output row;
- distinguish aggregation, which reduces or changes grain, from `transform`, which preserves row count and index;
- choose `size`, `count`, or `nunique` from the question being answered;
- create grouped results with named columns and deliberate index placement; and
- read one aggregating pivot without mistaking an absent combination for zero.

Lecture 09 may use those grouping and result-grain skills after it separately defines timestamps, periods, frequency, entity boundaries, resampling, lags, and rolling windows. None of those time-series concepts is introduced by Lecture 08.

## Core scope boundary

Required Lecture 08 work is limited to grouping unit and key, input/output grain, observed group prediction, `size`/`count`/`nunique`, named aggregation, `transform`, deliberate grouped output columns/index, one bounded two-key summary, and one aggregating pivot table.

Group filtering, categorical edge cases, and hierarchical result indexes are optional bonus material. `GroupBy.apply`, advanced MultiIndex manipulation, crosstabs, custom statistical tests, periods, resampling, rolling operations, time-series analysis, remote-computing tools, chunking, parallelism, performance optimization, and a new plotting objective are not core Lecture 08 requirements.
