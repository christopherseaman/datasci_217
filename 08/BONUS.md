# Bonus: Optional Grouping and Index-Output Policies

This optional material extends Lecture 08 only where a grouping policy or indexed output needs to be made more explicit. It assumes the core grain, counting, named-aggregation, `transform`, and pivot workflow.

Return to [README.md](README.md) for the five required objectives and three demonstrations. Nothing here is required by the Lecture 08 assignment or by Lecture 09.

## Prepare the bounded bonus fixture

The bonus uses the same provisional Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3 candidate as the core. The fixture is self-contained so this file can be executed independently in fresh Colab or local Jupyter.

```python
import platform

import numpy as np
import pandas as pd

assert platform.python_version() == "3.12.13"
assert np.__version__ == "2.0.2"
assert pd.__version__ == "3.0.3"
```

```python
FACILITY_LEVELS = ["North", "South", "West", "Remote"]
SERVICE_LEVELS = ["Consult", "Follow-up", "Procedure"]

bonus_encounters = pd.DataFrame(
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
        "rating": [
            4, pd.NA, 5, 5,
            4, pd.NA, pd.NA, 4,
            3, 4, 3, 4,
        ],
    }
).astype(
    {
        "encounter_id": "string",
        "rating": "Int64",
    }
)

bonus_encounters["facility"] = pd.Categorical(
    bonus_encounters["facility"],
    categories=FACILITY_LEVELS,
    ordered=True,
)
bonus_encounters["service"] = pd.Categorical(
    bonus_encounters["service"],
    categories=SERVICE_LEVELS,
    ordered=True,
)

assert bonus_encounters.shape == (12, 5)
assert bonus_encounters["encounter_id"].is_unique
```

## Contrast observed and declared category levels

The core uses `observed=True` because the question concerns groups represented by input rows. An analysis may instead need every declared category level, including levels with no rows. That is an output-design decision, not a performance switch.

```python
observed_counts = (
    bonus_encounters.groupby(
        "facility",
        observed=True,
        sort=True,
        dropna=True,
    )
    .size()
    .rename("encounter_count")
)

all_level_counts = (
    bonus_encounters.groupby(
        "facility",
        observed=False,
        sort=True,
        dropna=True,
    )
    .size()
    .rename("encounter_count")
)

assert observed_counts.index.astype("string").tolist() == [
    "North", "South", "West"
]
assert all_level_counts.index.astype("string").tolist() == [
    "North", "South", "West", "Remote"
]
assert int(all_level_counts.loc["Remote"]) == 0

print(pd.concat({"observed": observed_counts, "all_levels": all_level_counts}, axis="columns"))
```

The Remote count is zero because the declared category has no input rows. With several categorical keys, `observed=False` can expose many unused combinations, so state why those rows belong in the result before choosing it.

## Decide whether a missing key forms a group

`observed=` controls unused categorical levels. `dropna=` independently controls rows whose grouping key is missing. The next fixture removes one North facility label while preserving the encounter row.

```python
missing_key_encounters = bonus_encounters.copy()
missing_key_encounters.loc[0, "facility"] = pd.NA

drop_missing_key = missing_key_encounters.groupby(
    "facility",
    observed=True,
    sort=True,
    dropna=True,
).size()

keep_missing_key = missing_key_encounters.groupby(
    "facility",
    observed=True,
    sort=True,
    dropna=False,
).size()

assert int(drop_missing_key.sum()) == 11
assert int(keep_missing_key.sum()) == 12
assert keep_missing_key.index.isna().sum() == 1
assert int(keep_missing_key.loc[pd.isna(keep_missing_key.index)].iloc[0]) == 1

print(keep_missing_key)
```

Use `dropna=False` only when “missing facility” is a meaningful grouping unit to report. Otherwise repair, reject, or separately audit the missing key according to the data contract. Never confuse a missing key with a declared but unused category.

## Filter whole groups with a named rule

`GroupBy.filter()` evaluates a condition for each group and keeps or removes the group's input rows as a unit. It does not produce one summary row per group. That output behavior makes it bonus material: students must already be able to predict aggregation and transform shapes.

The named rule below keeps facilities with at least three recorded ratings. North and West pass; every South encounter row is removed even though two South rows have ratings.

```python
def has_at_least_three_recorded_ratings(group):
    return group["rating"].count() >= 3


filtered_encounters = (
    bonus_encounters.groupby(
        "facility",
        observed=True,
        sort=True,
        dropna=True,
    )
    .filter(has_at_least_three_recorded_ratings)
)

assert len(filtered_encounters) == 8
assert filtered_encounters.index.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
assert filtered_encounters["facility"].astype("string").unique().tolist() == [
    "North", "West"
]

print(filtered_encounters)
```

The result retains the original encounter-row grain and original index labels for the groups that pass. A Boolean row filter answers a different question because it can keep only selected rows inside a group.

## Inspect one bounded hierarchical result index

When `as_index=True` is used with two grouping keys, the grouped result has a two-level index. This **hierarchical index**, also called a **MultiIndex**, stores each facility–service key combination in the result's row labels.

The core avoids requiring MultiIndex manipulation by using `as_index=False`. This bonus shows only how to inspect the level names and flatten the result with `reset_index()`.

```python
indexed_two_key_summary = (
    bonus_encounters.groupby(
        ["facility", "service"],
        as_index=True,
        observed=True,
        sort=True,
        dropna=True,
    )
    .agg(
        encounter_count=("encounter_id", "size"),
        mean_charge=("charge", "mean"),
    )
)

assert indexed_two_key_summary.index.nlevels == 2
assert indexed_two_key_summary.index.names == ["facility", "service"]
assert indexed_two_key_summary.index.is_unique
assert indexed_two_key_summary.shape == (8, 2)

flat_two_key_summary = indexed_two_key_summary.reset_index()

assert flat_two_key_summary.columns.tolist() == [
    "facility",
    "service",
    "encounter_count",
    "mean_charge",
]
assert flat_two_key_summary.shape == (8, 4)

print(indexed_two_key_summary)
print(flat_two_key_summary)
```

Neither layout changes the summary's meaning: one row represents one observed facility–service combination. `reset_index()` changes where the keys are stored, not the grain or values.

## Deferred local-terminal lab: SSH and tmux

SSH and tmux are not part of this notebook bonus. A safe remote workflow depends on institution-specific hosts, account authorization, authentication or MFA policy, host-key verification, network restrictions, port-forwarding rules, and installed terminal tools. Generic commands would not constitute a reproducible or supportable course exercise.

If the instructor retains this material, it should be published as a separate optional local-terminal lab with an approved practice host, current security guidance, explicit cleanup/reconnection steps, and platform-tested instructions. It must not run in Colab, must not become a Lecture 08 or Lecture 09 prerequisite, and must not be mixed into an aggregation demonstration.

## Bonus scope boundary

This bonus is limited to categorical `observed` policy, missing grouping-key policy, whole-group filtering, and one bounded two-level result-index layout.

It does not teach `GroupBy.apply`, custom statistical functions, advanced MultiIndex manipulation, advanced pivots or crosstabs, periods, resampling, rolling windows, time-series analysis, hypothesis tests, plotting, chunking, parallelism, remote execution, or performance optimization.
