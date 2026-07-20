# Optional Extensions for Temporal Data

This bonus material extends Lecture 09 after the required entity-aware, past-only workflow is secure. None of these topics is required by the Lecture 09 demonstrations or assignment, and none is an entry prerequisite for Lecture 10.

## Scope boundary

The core lecture owns timestamp and period meanings, single and panel structure, regularity and frequency, basic localization and conversion, `asfreq()` versus grouped `resample()`, source versus grid missingness, lags, differences, two trailing-window meanings, information availability, and a chronological handoff.

This file contains optional breadth:

- expanding, exponentially weighted, centered, and custom rolling calculations;
- partial-string and clock-time selection;
- advanced calendar and period handling;
- explicit daylight-saving-time policy; and
- an orientation to decomposition, forecasting, ARIMA, STL, and high-frequency data.

Specialized methods do not become safe merely because pandas or another library can compute them. Preserve entity boundaries, state the measurement meaning, and inventory prediction-time availability first.

## Window variants beyond the required trailing means

An **expanding window** begins at the first observation and grows through the current row. An **exponentially weighted window** assigns larger weights to more recent values; `span` or `alpha` controls how quickly old values lose influence. A **centered window** places the current row near the middle and therefore uses later observations except at its trailing edge. A **custom rolling calculation** supplies a named rule instead of a built-in summary such as `mean()`.

The examples use one already sorted, timezone-aware single series. They are descriptive extensions, not automatically approved prediction features.

```python
import platform

import numpy as np
import pandas as pd

assert platform.python_version() == "3.12.13"
assert np.__version__ == "2.0.2"
assert pd.__version__ == "3.0.3"

bonus_index = pd.date_range(
    "2026-02-01",
    periods=6,
    freq="h",
    tz="UTC",
)
bonus_series = pd.Series(
    [18.0, 19.0, 21.0, 26.0, 25.0, 28.0],
    index=bonus_index,
    name="value",
)
```

### Expanding and exponentially weighted summaries

The expanding mean below uses every value from the beginning through the current row. The exponentially weighted mean also uses values through the current row but emphasizes recent observations. Neither one repairs missingness or proves that the inputs were available in a real workflow.

```python
bonus_windows = pd.DataFrame({"value": bonus_series})
bonus_windows["expanding_mean"] = bonus_series.expanding(
    min_periods=1
).mean()
bonus_windows["ewm_mean"] = bonus_series.ewm(
    span=3,
    adjust=False,
).mean()

assert bonus_windows["expanding_mean"].notna().all()
assert bonus_windows["ewm_mean"].notna().all()
```

### Centered and custom calculations

The centered three-observation mean at 02:00 uses the 01:00, 02:00, and 03:00 values. The 03:00 value is in the future relative to 02:00, so this column is suitable only when the full descriptive series is available. It must be rejected at a 02:00 prediction cutoff.

The custom rule returns the range, maximum minus minimum, of each trailing three-row window. Named rules are easier to test and explain than anonymous one-off calculations.

```python
bonus_windows["centered_mean_3"] = bonus_series.rolling(
    window=3,
    center=True,
).mean()


def peak_to_peak(values):
    """Return maximum minus minimum for one rolling window."""
    return values.max() - values.min()


bonus_windows["custom_range_3"] = bonus_series.rolling(
    window=3
).apply(
    peak_to_peak,
    raw=True,
)

assert np.isclose(bonus_windows["centered_mean_3"].iloc[2], 22.0)
assert np.isclose(bonus_windows["custom_range_3"].iloc[2], 3.0)
```

Custom indexers, weighted kernels, and performance tuning add still more API surface. They belong in a specialized project with an explicit need, not in the required Lecture 09 path.

## Advanced time selection

The core lecture selects one bounded interval with an explicit timestamp condition. A sorted `DatetimeIndex` also supports concise **partial-string selection**, where a label such as `2026-02-02` selects that calendar span. `between_time()` selects repeated clock-time ranges across dates, and `at_time()` selects one repeated clock time.

These conveniences can hide date, timezone, and boundary assumptions. Confirm the index timezone, sorting, inclusivity, and entity scope before using them.

```python
selection_index = pd.date_range(
    "2026-02-01",
    periods=72,
    freq="h",
)
selection_data = pd.DataFrame(
    {"reading": np.arange(72)},
    index=selection_index,
)

second_day = selection_data.loc["2026-02-02"]
mornings = selection_data.between_time("08:00", "10:00")
noon = selection_data.at_time("12:00")

assert len(second_day) == 24
assert len(mornings) == 9
assert len(noon) == 3
```

In a panel, perform clock-time selection inside the intended entity history or keep the entity key in the result. A timestamp-only selection does not make repeated entity rows interchangeable.

## Advanced calendar and period handling

Anchored offsets describe calendar boundaries rather than fixed elapsed durations. In pandas 3, use `QE` for quarter end and `YE` for year end; use anchored forms such as `QE-JUN` or `YE-JUN` when a reporting year ends in another month. Hourly grids use lowercase `h`.

```python
quarter_ends = pd.date_range(
    "2026-01-01",
    periods=4,
    freq="QE",
)
year_ends = pd.date_range(
    "2024-01-01",
    periods=3,
    freq="YE",
)

assert quarter_ends.month.tolist() == [3, 6, 9, 12]
assert year_ends.month.tolist() == [12, 12, 12]
```

Period arithmetic can shift calendar spans, convert between span frequencies, or map a span to a start/end timestamp. Fiscal labels require a documented business convention: “2026 Q1” is ambiguous until the reporting year-end rule is known. Do not convert timestamps to periods merely to make grouping convenient when the source meaning is still an instant.

## Daylight-saving-time policy

Basic localization in the core uses unambiguous winter timestamps. Real local clock data can contain two special cases:

- an **ambiguous local time** occurs twice when clocks move backward; and
- a **nonexistent local time** is skipped when clocks move forward.

The correct policy comes from source-system documentation. `tz_localize()` raises by default rather than guessing. The example below shows explicit policies only to expose the choices; it does not prescribe them for an unknown dataset.

```python
spring_naive = pd.DatetimeIndex(
    [
        "2026-03-08 01:30",
        "2026-03-08 02:30",
        "2026-03-08 03:30",
    ]
)
spring_aware = spring_naive.tz_localize(
    "America/New_York",
    nonexistent="shift_forward",
)

fall_naive = pd.DatetimeIndex(
    [
        "2026-11-01 01:30",
        "2026-11-01 01:30",
    ]
)
fall_aware = fall_naive.tz_localize(
    "America/New_York",
    ambiguous=[True, False],
)
fall_utc = fall_aware.tz_convert("UTC")

assert spring_aware[1].hour == 3
assert fall_aware[0].utcoffset() != fall_aware[1].utcoffset()
assert fall_utc.is_unique
```

The two fall clock labels look identical before localization but map to different UTC instants. Silently choosing one offset would change ordering, elapsed-time windows, and event matching.

## Specialized analysis: orientation only

The topics below require more statistical and domain context than the required course handoff supplies. They are listed so students can recognize the vocabulary, not so they can fit or interpret these methods independently.

### Decomposition and STL

**Decomposition** represents an observed series as components such as trend, seasonality, and remainder under stated assumptions. **STL** is a seasonal-trend decomposition method based on locally weighted smoothing. A defensible analysis must choose a meaningful seasonal period, have enough coverage, handle gaps, and avoid treating estimated components as ground truth.

No decomposition API is required in Lecture 09. Synthetic sine waves do not establish that a real seasonal component exists.

### Forecasting, ARIMA, and exponential smoothing

**Forecasting** estimates values beyond a stated cutoff. It requires a horizon, a baseline, chronological evaluation, uncertainty, and an availability contract. **ARIMA** combines autoregressive, differencing, and moving-average terms. Forecasting forms of **exponential smoothing** model level and optionally trend or seasonality.

These methods are not shortcuts around Lecture 10's definitions of target, horizon, split roles, baseline, evaluation, and leakage. They also add a `statsmodels` dependency that is not part of the required Lecture 09 environment.

### High-frequency and tick data

**High-frequency data** arrive at very short, often irregular intervals. **Tick data** record individual events or state changes rather than a pre-existing regular grid. Analysis may require event-time versus clock-time reasoning, market or device calendars, interval conventions, duplicate policies, memory-aware storage, and domain-specific aggregation rules.

No tick-processing, custom-frequency, interactive-plotting, or performance objective is required here. A later specialized project should introduce those capabilities only after its data contract and resource constraints are explicit.

## Bonus completion check

The optional examples should still run from top to bottom without relying on stored output.

```python
assert bonus_windows.shape == (6, 5)
assert selection_data.shape == (72, 1)
assert quarter_ends.is_monotonic_increasing
assert year_ends.is_monotonic_increasing
assert str(fall_utc.tz) == "UTC"

print("Lecture 09 bonus verification passed.")
```
