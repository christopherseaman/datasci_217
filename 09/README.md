# Temporal Structure, Entity Boundaries, and Past-Only Windows

Temporal data are observations whose ordering and spacing carry meaning. Lecture 09 teaches how to represent that structure explicitly before changing frequency or creating time-based columns. The required path keeps station identity visible throughout: a plausible calculation for one station can become wrong if rows from two stations are silently pooled.

Optional window variants, advanced selection, daylight-saving-time cases, decomposition, forecasting, and high-frequency analysis are collected in [BONUS.md](BONUS.md). They are not prerequisites for the required demonstrations, assignment, or Lecture 10.

## Prerequisites

Before starting this lecture, students should be able to:

- select, sort, index, and save pandas data;
- distinguish source missingness from a justified cleaning decision;
- state row grain and identify candidate and grouping keys;
- group and aggregate with an explicit result grain; and
- read or create one clearly labeled line chart when a supplied summary needs visual reinforcement.

Lecture 09 does not assume forecasting, decomposition, model fitting, train/validation/test terminology, or advanced daylight-saving-time policy.

## Learning objectives

By the end of Lecture 09, students should be able to:

1. Classify a dataset as timestamp- or period-based, regular or irregular, and single-series or panel; state the row grain and sort keys.
2. Parse timestamps, distinguish naive from timezone-aware values, localize or convert one series correctly, and create a sorted datetime index within entity.
3. Use `asfreq` or `resample` with an aggregation justified by measurement meaning, preserving entity boundaries and explaining newly introduced missing values.
4. Create a lag, difference, and trailing observation-count or elapsed-time window without crossing entity boundaries or using future observations.
5. State what information is available at a prediction timestamp, reject centered or future-derived features, and construct a plausible chronological holdout for Lecture 10.

## Colab-first execution and evidence

Required Lecture 09 demonstrations are Colab-first and also run in local Jupyter or the VS Code notebook interface. The 2026–27 compatibility candidate is Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. This is not the final release lock.

In the pin-able Colab 2026.04 runtime, a setup cell must conditionally install pandas 3.0.3 before pandas is imported when the installed version differs. Do not install pandas 3.0.4; that release was yanked. Avoid reinstalling unrelated Colab packages. Every required notebook prints the versions actually in use and must pass in both a fresh Colab runtime and clean local Jupyter before publication.

Colab's filesystem is ephemeral. Required notebooks use fixed in-notebook data or reacquire a pinned source in code; manual upload and mounted Drive are not defaults. Changes made in a Colab notebook opened from GitHub are not automatically saved back to the repository.

Assignment notebooks remain runnable in clean local Jupyter. Colab becomes an assignment submission path only after the repository-save and Classroom 50 pilot is approved. Stored cell output is not execution evidence: restart the runtime and run every cell in order.

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

## Why temporal order changes analysis

For an ordinary table, rearranging rows may change presentation without changing a calculation. For temporal data, rearranging rows can change which observation is considered previous, which values fall in a window, and whether a calculation uses information that did not yet exist. Before calling a time-series method, answer four questions:

1. What does one row represent?
2. Which entity does the row belong to?
3. What instant or span does the row describe?
4. In what order must rows be processed within each entity?

Those answers are part of the data contract, not cleanup trivia.

## Describe temporal structure before computing

### Timestamp versus period

A **timestamp** represents an instant on a timeline, such as a sensor reading recorded at 08:00. A **period** represents a span with a start and end, such as the calendar day 2026-01-15 or the month January 2026. A label such as `2026-01-15` is not enough by itself to determine which meaning the source intends.

Use a timestamp when the observation happened at a particular instant. Use a period when the value describes the whole span. For example, “temperature measured at 08:00” is timestamp-based; “total visits during January” is period-based.

```python
example_timestamp = pd.Timestamp("2026-01-15 08:00")
example_period = pd.Period("2026-01-15", freq="D")

assert example_timestamp == pd.Timestamp("2026-01-15 08:00:00")
assert example_period.start_time == pd.Timestamp("2026-01-15 00:00:00")
assert example_period.end_time.date().isoformat() == "2026-01-15"
```

This lecture uses timestamp-based station observations. Advanced period arithmetic and fiscal calendars remain optional.

### Entity, single series, panel, and row grain

An **entity** is the real-world unit whose observations form one ordered history. Its **entity key** identifies that unit in the table. A **single series** contains one entity's history. A **panel** contains parallel histories for multiple entities, such as several stations or patients.

**Row grain** states what one source row represents. In the running example, one row represents one recorded temperature observation for one station at one timestamp. The pair `station` plus `observed_at` is the row key. The required **sort keys** are the entity key first and timestamp second.

Repeated timestamps across stations are expected in a panel. They do not mean the rows are duplicates, and they do not authorize pooling the station histories.

### Regular, irregular, and frequency

A series is **regular** when consecutive observations follow one expected spacing, such as exactly one hour. It is **irregular** when gaps vary or observations occur only when events happen. A panel can contain one regular or irregular series per entity.

A **frequency** is an expected time grid or calendar offset, such as hourly or month-end. It describes spacing or bin boundaries; it does not guarantee that the source has a valid observed value at every grid point.

Common pandas 3 offset aliases include:

| Alias | Meaning |
|---|---|
| `min` | minute |
| `h` | hour |
| `D` | calendar day |
| `W` | week ending Sunday; use an anchored form such as `W-MON` when needed |
| `MS` / `ME` | month start / month end |
| `QS` / `QE` | quarter start / quarter end |
| `YS` / `YE` | year start / year end |

The old uppercase hourly alias `H` and old quarter/year-end aliases `Q` and `A` are not used in the pandas 3 course path. The pandas [time-series user guide](https://pandas.pydata.org/docs/user_guide/timeseries.html) lists current offset aliases.

The deterministic running panel contains two stations, gaps of one and two hours, and one missing source temperature. The `source_row` marker records which rows came from the source so later grid-created rows can be distinguished.

```python
raw = pd.DataFrame(
    {
        "station": ["north"] * 5 + ["south"] * 5,
        "observed_at": [
            "2026-01-15 08:00",
            "2026-01-15 09:00",
            "2026-01-15 11:00",
            "2026-01-15 12:00",
            "2026-01-15 14:00",
            "2026-01-15 08:00",
            "2026-01-15 10:00",
            "2026-01-15 11:00",
            "2026-01-15 13:00",
            "2026-01-15 14:00",
        ],
        "temperature_c": [
            10.0,
            11.0,
            np.nan,
            13.0,
            14.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
        ],
    }
)
raw["source_row"] = 1

assert raw.shape == (10, 4)
assert raw["station"].nunique() == 2
```

## Parse, localize, convert, sort, and index

**Parsing** converts text to pandas datetime values. A **naive timestamp** contains a date and clock time but no time-zone offset. A **timezone-aware timestamp** identifies its offset from Coordinated Universal Time, so it represents an unambiguous instant.

**Localization** attaches the source time zone to naive clock readings without changing those displayed clock readings. **Conversion** expresses already-aware instants in another time zone; the displayed clock values can change, but the instants do not. Do not use conversion on naive values, and do not localize a second time zone onto already-aware values.

The fixture's text is documented as unambiguous Los Angeles local time in January. Advanced daylight-saving-time ambiguity and nonexistent local clock times belong in the bonus material.

After conversion, sort by entity and timestamp. A **DatetimeIndex** is a pandas index whose labels are datetime values and therefore support time-aware operations. In a panel, the same timestamp can occur for several entities, so group by the entity before resampling, shifting, or rolling.

```python
raw["observed_at"] = pd.to_datetime(
    raw["observed_at"],
    format="%Y-%m-%d %H:%M",
)
assert raw["observed_at"].dt.tz is None

raw["observed_at"] = raw["observed_at"].dt.tz_localize(
    "America/Los_Angeles"
)
raw["observed_at"] = raw["observed_at"].dt.tz_convert("UTC")
assert str(raw["observed_at"].dt.tz) == "UTC"

prepared = raw.sort_values(
    ["station", "observed_at"],
    kind="stable",
).reset_index(drop=True)

assert not prepared.duplicated(["station", "observed_at"]).any()
assert all(
    group["observed_at"].is_monotonic_increasing
    for _, group in prepared.groupby("station", sort=False)
)

indexed = prepared.set_index("observed_at")
assert isinstance(indexed.index, pd.DatetimeIndex)

gap_since_previous = prepared.groupby(
    "station",
    sort=False,
)["observed_at"].diff()
assert gap_since_previous.dropna().nunique() == 2
```

The first gap in each station is missing because that entity has no previous row. The one-hour and two-hour gaps prove that both station series are irregular.

A **bounded time interval** has an explicit start and end. For a panel stored with entity and timestamp columns, a boolean timestamp condition keeps the entity key visible and avoids pretending that duplicate timestamps identify rows by themselves.

```python
interval_start = pd.Timestamp("2026-01-15 17:00", tz="UTC")
interval_end = pd.Timestamp("2026-01-15 20:00", tz="UTC")

bounded = prepared.loc[
    prepared["observed_at"].between(
        interval_start,
        interval_end,
        inclusive="both",
    )
].copy()

assert bounded.shape[0] == 5
assert set(bounded["station"]) == {"north", "south"}
```

## LIVE DEMO 1: Classify and prepare temporal structure

[Open the Lecture 09 demo guide](demo/DEMO_GUIDE.md).

The first required demonstration compares timestamp and period meanings, single and panel data, and regular and irregular observations. It states the row grain and sort keys, parses one two-station panel, distinguishes naive from aware timestamps, localizes and converts once, sorts within station, and verifies entity–timestamp uniqueness and within-entity order.

## Change frequency without changing meaning accidentally

Changing frequency is a question about both labels and measurements:

- **Upsampling** creates a finer grid, such as two-hour labels to hourly labels. It can introduce labels for which the source has no row.
- **Downsampling** creates coarser bins, such as hourly observations summarized into two-hour intervals. Several source rows can contribute to one output row.
- `asfreq()` conforms data to a new grid without combining observations.
- `resample()` groups timestamps into bins and requires a summary operation when multiple observations can enter a bin.

With no fill method, [`asfreq()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.asfreq.html) preserves values at exact matching labels and inserts missing values at new labels. It does not aggregate nearby off-grid observations. [`resample()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html) is the appropriate operation when each output bin must combine observations.

### Separate source missingness from grid-created missingness

**Source missingness** means a row existed in the supplied data but its measurement was missing. **Grid-created missingness** means a requested grid introduced a timestamp for which no source row existed. Both can display as `NaN`, but they have different provenance and may require different decisions.

A grouped resample temporarily uses a **two-level index**, an index whose labels contain both station and timestamp. Resetting that index makes both keys ordinary columns again. The entity key must remain in either the index or the columns throughout the operation.

The source observations below occur on hourly clock labels even though some hours are skipped. Therefore an hourly grid can show the skipped labels without dropping off-grid source rows.

```python
hourly_grid = (
    indexed.groupby("station")[["temperature_c", "source_row"]]
    .resample("h")
    .asfreq()
    .reset_index()
)

hourly_grid["grid_created_row"] = hourly_grid["source_row"].isna()
hourly_grid["source_value_missing"] = (
    hourly_grid["source_row"].eq(1)
    & hourly_grid["temperature_c"].isna()
)

assert set(hourly_grid["station"]) == {"north", "south"}
assert int(hourly_grid["grid_created_row"].sum()) == 4
assert int(hourly_grid["source_value_missing"].sum()) == 1
```

No fill is automatic. Forward fill, backward fill, interpolation, and zero each assert a different measurement story. A grid change alone is not evidence that any of those stories is correct.

### Choose a resampling aggregation from measurement meaning

**Measurement meaning** describes what a value represents and how, if at all, values may be combined. Temperature is a state observed at an instant. The mean temperature in a two-hour bin can answer “what was the average of the recorded temperatures in this interval?” The number of source readings is additive and can be summed. A patient identifier, station name, or other label should not be averaged.

The output bins below are left-closed and left-labeled: a label at 16:00 represents the interval from 16:00 up to, but not including, 18:00. Writing `closed=` and `label=` makes that boundary choice visible.

```python
two_hour_summary = (
    indexed.groupby("station")
    .resample("2h", closed="left", label="left")
    .agg(
        mean_temperature_c=("temperature_c", "mean"),
        reading_count=("source_row", "sum"),
    )
    .reset_index()
)

north_first_bin = two_hour_summary.loc[
    two_hour_summary["station"].eq("north")
    & two_hour_summary["observed_at"].eq(
        pd.Timestamp("2026-01-15 16:00", tz="UTC")
    )
].iloc[0]

south_second_bin = two_hour_summary.loc[
    two_hour_summary["station"].eq("south")
    & two_hour_summary["observed_at"].eq(
        pd.Timestamp("2026-01-15 18:00", tz="UTC")
    )
].iloc[0]

assert np.isclose(north_first_bin["mean_temperature_c"], 10.5)
assert north_first_bin["reading_count"] == 2
assert np.isclose(south_second_bin["mean_temperature_c"], 21.5)
assert south_second_bin["reading_count"] == 2
assert int(two_hour_summary["reading_count"].sum()) == len(prepared)
```

The result grain is one station–two-hour interval per row. The station column proves that the two histories were not pooled. The missing mean in North's 18:00 bin comes from a source row whose temperature was missing; it is not an empty station–interval.

## LIVE DEMO 2: Resample with measurement meaning

[Open the Lecture 09 demo guide](demo/DEMO_GUIDE.md).

The second required demonstration contrasts an hourly `asfreq()` grid with a two-hour `resample()` summary. It justifies a state-variable mean and a reading count, retains station identity, labels bin boundaries explicitly, and reports source missingness separately from rows introduced by the new grid.

## Create past-only comparisons within entity

A **lag** attaches an earlier observation from the same entity to the current row. A **lead** attaches a later observation to the current row and is therefore a warning sign for prediction-time work. A **difference** subtracts the previous observation from the current observation within an entity.

`shift(1)` means one previous row, not one hour. On irregular data those are different ideas. Always sort and group first. Calling `shift(1)` or `diff()` on the pooled panel could borrow the last North value for the first South row.

```python
features = prepared[
    ["station", "observed_at", "temperature_c"]
].copy()

by_station = features.groupby(
    "station",
    sort=False,
)["temperature_c"]

features["temperature_lag_1"] = by_station.shift(1)
features["temperature_difference"] = by_station.diff()

first_rows = features.groupby("station", sort=False).head(1)
assert first_rows["temperature_lag_1"].isna().all()
assert first_rows["temperature_difference"].isna().all()
```

The first lag and difference for each station are missing because each station begins a new history. That is evidence that values did not cross the entity boundary. A negative shift such as `shift(-1)` creates a lead; it is not computed in the required path because it would attach future information.

## Distinguish observation-count and elapsed-time windows

A **trailing window** summarizes values at or before a row while moving forward through time. An **observation-count window** contains a fixed number of rows, regardless of the elapsed time between them. An **elapsed-time window** contains observations whose timestamps fall inside a stated duration, so the number of rows can vary.

For a past-only candidate at timestamp `t`, the examples below exclude the current row:

- the observation-count window uses the previous two station rows; and
- the elapsed-time window uses station observations in `[t - 2 hours, t)`.

`min_periods=1` means at least one nonmissing value is required for a mean. It does not fill missing values.

```python
features["mean_previous_2_observations"] = (
    features.groupby("station", sort=False)["temperature_c"]
    .transform(
        lambda values: values.shift(1)
        .rolling(window=2, min_periods=1)
        .mean()
    )
)

elapsed_summary = (
    features.set_index("observed_at")
    .groupby("station")["temperature_c"]
    .rolling("2h", closed="left", min_periods=1)
    .mean()
    .rename("mean_previous_2h")
    .reset_index()
)

features = features.merge(
    elapsed_summary,
    on=["station", "observed_at"],
    how="left",
    validate="one_to_one",
    sort=False,
)

south_at_21 = features.loc[
    features["station"].eq("south")
    & features["observed_at"].eq(
        pd.Timestamp("2026-01-15 21:00", tz="UTC")
    )
].iloc[0]

assert np.isclose(south_at_21["mean_previous_2_observations"], 21.5)
assert np.isclose(south_at_21["mean_previous_2h"], 22.0)
```

At 21:00, South's previous two observations are the 18:00 and 19:00 readings, whose mean is 21.5. Only the 19:00 reading falls in the previous two elapsed hours, so that mean is 22.0. Neither answer is universally “the rolling mean”; the intended window must be named.

## Check information availability at a prediction timestamp

Here, a **candidate feature** is a value that might later be supplied to a prediction procedure. Lecture 10 formalizes features, targets, and horizons. The **prediction timestamp** is the supplied instant at which the prediction would be issued. **Information availability** asks whether every source value required for a candidate was known by that instant.

A **centered window** uses observations on both sides of a row. A **future-derived candidate** requires any value recorded after the prediction timestamp. Using either one as if it were already known creates **future leakage**: the procedure receives information that would not have existed when the prediction was issued.

For South at 21:00 UTC, the next reading is at 22:00. The inventory records the latest timestamp required by each candidate, then keeps only candidates available by 21:00. The centered window is defined here only so it can be rejected; its implementation is optional bonus material.

```python
prediction_timestamp = pd.Timestamp(
    "2026-01-15 21:00",
    tz="UTC",
)

availability = pd.DataFrame(
    {
        "candidate": [
            "calendar hour",
            "previous observed temperature",
            "centered three-observation mean",
            "next observed temperature",
        ],
        "latest_required_timestamp": pd.to_datetime(
            [
                "2026-01-15 21:00Z",
                "2026-01-15 19:00Z",
                "2026-01-15 22:00Z",
                "2026-01-15 22:00Z",
            ],
            utc=True,
        ),
    }
)

availability["available_by_prediction_time"] = availability[
    "latest_required_timestamp"
].le(prediction_timestamp)
availability["decision"] = np.where(
    availability["available_by_prediction_time"],
    "keep",
    "reject",
)

assert availability["available_by_prediction_time"].tolist() == [
    True,
    True,
    False,
    False,
]
```

Availability is a property of the real workflow, not just the final DataFrame. A value can appear in a completed historical dataset and still have been unavailable at the prediction timestamp.

## Construct a chronological holdout

A **chronological holdout** is a later time block set aside while work is developed on an earlier block. It is plausible only when every earlier timestamp precedes every held-out timestamp and the entity coverage fits the intended question. Lecture 10 will assign formal evaluation roles and define targets, horizons, baselines, and model-selection rules.

```python
holdout_start = pd.Timestamp("2026-01-15 21:00", tz="UTC")

earlier_block = features.loc[
    features["observed_at"].lt(holdout_start)
].copy()
later_holdout = features.loc[
    features["observed_at"].ge(holdout_start)
].copy()

assert not earlier_block.empty
assert not later_holdout.empty
assert earlier_block["observed_at"].max() < later_holdout[
    "observed_at"
].min()
assert set(earlier_block["station"]) == {"north", "south"}
assert set(later_holdout["station"]) == {"north", "south"}

assert features.groupby("station", sort=False).head(1)[
    "temperature_lag_1"
].isna().all()
assert int(hourly_grid["grid_created_row"].sum()) == 4
assert int(hourly_grid["source_value_missing"].sum()) == 1

print("Lecture 09 core verification passed.")
```

This split is a handoff, not a completed modeling workflow. It does not decide what should be predicted, how far ahead, which metric matters, or which data block may be used for model choice.

## LIVE DEMO 3: Build past-only features and audit availability

[Open the Lecture 09 demo guide](demo/DEMO_GUIDE.md).

The third required demonstration builds station-scoped lags, differences, and both window meanings on irregular data. It proves that the first value in each station is not borrowed from another entity, marks one supplied prediction timestamp, rejects centered and future-derived candidates, and creates a chronological holdout for Lecture 10. It may add at most one already-familiar Lecture 07 line chart for one station; plotting is not a new Lecture 09 objective.

## Handoff to Lecture 10

After this lecture, students should be able to:

- distinguish timestamp from period, regular from irregular, and single-series from panel structure;
- state the row grain, entity key, timestamp key, and within-entity sort order;
- parse, localize, convert, sort, and index entity-specific timestamp data;
- distinguish `asfreq()` from `resample()` and justify an aggregation from measurement meaning;
- report source missingness separately from grid-created missingness;
- create lags, differences, and trailing observation-count or elapsed-time windows without crossing entities;
- reject candidates requiring observations after a supplied prediction timestamp; and
- make a chronological holdout plausible without claiming that a complete model-evaluation design already exists.

Lecture 10 may use those capabilities after it defines descriptive, inferential, and predictive questions; association and causation; target and horizon; train, validation, and test roles; baselines; evaluation; and the broader meaning of leakage.

## Core scope boundary

Required Lecture 09 work is limited to temporal structure, basic parsing and timezone operations, within-entity ordering, one bounded interval, `asfreq()` versus measurement-aware grouped resampling, source versus grid missingness, lag/difference, two trailing-window meanings, information availability, and a plausible chronological holdout.

Exponentially weighted, centered, expanding, and custom windows; advanced time selection; fiscal-period arithmetic; daylight-saving-time edge cases; decomposition; STL; forecasting; ARIMA; exponential-smoothing forecasts; high-frequency analysis; and broad visualization surveys remain optional bonus material. Lecture 09 does not fit or evaluate a model and adds no new visualization objective.
