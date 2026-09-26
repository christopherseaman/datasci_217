---
notion:
  title_line: "# DLC: Advanced Time Series Analysis Topics"
  role: bonus
  status: mapped
  page_id: "3d2d9fdd-1a1a-8164-a3f7-c5c5d3a8617a"
  url: "https://app.notion.com/p/3d2d9fdd1a1a8164a3f7c5c5d3a8617a"
---

# DLC: Advanced Time Series Analysis Topics

Everything in this document is optional for Lecture 09. It collects specialized material on periods, time-series patterns and decomposition, forecasting, more window options, high-frequency data, custom frequencies, advanced time zones, and additional visualization.

The forecasting, stationarity, and temporal-modeling material below previews ideas Lecture 10 covers in depth. Treat it as specialized reference rather than required content.

# Period Arithmetic and Fiscal Year Handling

_Periods represent time spans, not specific moments. Understanding periods is crucial for fiscal year analysis and business reporting._

## Period Basics

### Reference Card: Period Creation and Arithmetic

| Function | Description |
|----------|-------------|
| `pd.Period('2011', freq='Y-DEC')` | Annual period ending December |
| `pd.Period('2011Q4', freq='Q-JAN')` | Quarterly period with fiscal year |
| `pd.period_range(start, end, freq='M')` | Create period range |
| `period.asfreq('M', how='start')` | Convert period frequency |
| `period.to_timestamp()` | Convert period to timestamp |

### Code Snippet: Period Arithmetic and Fiscal Quarters

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

# Create annual period
p = pd.Period('2011', freq='Y-DEC')
print(f"Period: {p}")

# Period arithmetic
print(f"Plus 5 years: {p + 5}")   # Shift forward 5 years
print(f"Minus 2 years: {p - 2}")  # Shift backward 2 years

# Quarterly periods with fiscal year
p = pd.Period('2012Q4', freq='Q-JAN')  # Fiscal year ending in January
print(f"Fiscal Q4: {p}")
print(f"Start date: {p.asfreq('D', how='start')}")
print(f"End date: {p.asfreq('D', how='end')}")

# Create period range
periods = pd.period_range('2000-01-01', '2000-06-30', freq='M')
ts = pd.Series(rng.standard_normal(6), index=periods)
print("\nPeriod-indexed Series:")
print(ts)
```

## Converting Between Timestamps and Periods

### Reference Card: Timestamp-Period Conversion

| Function | Description |
|----------|-------------|
| `ts.to_period()` | Convert timestamp index to periods |
| `ts.to_period('M')` | Convert to monthly periods |
| `pts.to_timestamp()` | Convert periods back to timestamps |
| `pts.to_timestamp(how='end')` | Use end of period as timestamp |

### Code Snippet: Convert Timestamps to Periods and Back

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

# Timestamp-indexed time series
dates = pd.date_range('2000-01-01', periods=3, freq='ME')
ts = pd.Series(rng.standard_normal(3), index=dates)

# Convert to periods
pts = ts.to_period()
print("Period-indexed:")
print(pts)

# Convert back to timestamps
ts_back = pts.to_timestamp()
print("\nBack to timestamps:")
print(ts_back)
```

# Advanced Time Series Decomposition

_Decomposition separates time series into trend, seasonal, and residual components, revealing underlying patterns._

## Patterns in a Time Series

The lecture sorts series by spacing, regular or irregular. A series can also be described by the pattern its values follow, and decomposition pulls those patterns apart.

![Six kinds of time series: regular and irregular spacing from the lecture, then seasonal, trending, stationary, and combined patterns.](media/types_of_time_series.png)

| Pattern | Description | Example |
| --- | --- | --- |
| **Seasonal** | Patterns repeat over time | Monthly flu case counts |
| **Trending** | Long-term direction | Long-term blood pressure trends |
| **Stationary** | Statistical properties don't change | Laboratory control measurements |
| **Combined** | Multiple components (trend + seasonal + noise) | Real-world medical data with all patterns |

## Seasonal Decomposition

### Reference Card: Seasonal Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose a Series `ts` with a DatetimeIndex
decomposition = seasonal_decompose(ts, model='additive', period=7)
# or
decomposition = seasonal_decompose(ts, model='multiplicative', period=12)

# Access components
decomposition.observed  # Original series
decomposition.trend     # Trend component
decomposition.seasonal  # Seasonal component
decomposition.resid     # Residual component
```

### Code Snippet: Decompose a Seasonal Disease-Count Series

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

rng = np.random.default_rng(42)

# Create seasonal time series (daily disease cases over 3 years)
dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
trend = np.linspace(100, 200, len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
noise = rng.normal(0, 5, len(dates))
values = trend + seasonal + noise

ts = pd.Series(values, index=dates)

# Decompose time series
decomposition = seasonal_decompose(ts, model='additive', period=365)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(15, 12))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()
```

## STL Decomposition

### Reference Card: STL Decomposition

```python
from statsmodels.tsa.seasonal import STL

# STL decomposition (more robust to outliers); ts is a Series with a DatetimeIndex
# `period` is the known number of observations per cycle (annual here).
# `seasonal` is the odd length of STL's seasonal smoother, not the period.
stl = STL(ts, period=365, seasonal=13, robust=True)
result = stl.fit()

# Access components
result.observed  # Original series
result.trend     # Trend component
result.seasonal  # Seasonal component
result.resid     # Residual component
```

# Time Series Forecasting

_Forecasting uses historical patterns to predict future values. Always be honest about uncertainty and prediction intervals._

## ARIMA Models

### Reference Card: ARIMA Models and Stationarity Checks

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Check stationarity
def check_stationarity(series):
    """Report the ADF test against the unit-root null hypothesis."""
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    
    if result[1] <= 0.05:
        print("Reject the unit-root null: evidence supports stationarity")
    else:
        print("Fail to reject the unit-root null: result is inconclusive")

# Fit ARIMA model
def fit_arima(series, order=(1, 1, 1)):
    """Fit ARIMA model"""
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    
    # Forecast
    forecast = fitted_model.forecast(steps=30)
    forecast_ci = fitted_model.get_forecast(steps=30).conf_int()
    
    return fitted_model, forecast, forecast_ci
```

## Exponential Smoothing

### Reference Card: Exponential Smoothing Variants

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Simple exponential smoothing
model = ExponentialSmoothing(ts, trend=None, seasonal=None)
fitted = model.fit(smoothing_level=0.3)
forecast = fitted.forecast(steps=30)

# Holt's method (trend)
model = ExponentialSmoothing(ts, trend='add', seasonal=None)
fitted = model.fit(smoothing_level=0.3, smoothing_trend=0.3)
forecast = fitted.forecast(steps=30)

# Holt-Winters (trend + seasonal)
model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=12)
fitted = model.fit(smoothing_level=0.3, smoothing_trend=0.3, smoothing_seasonal=0.3)
forecast = fitted.forecast(steps=30)
```

# Advanced Resampling Operations

_Resampling with periods requires careful handling of period boundaries and conventions._

## Resampling with Periods

### Reference Card: Resampling with Periods

| Function | Description |
|----------|-------------|
| `ts.to_period('M')` | Convert timestamp labels to monthly periods without aggregation |
| `ts.resample('ME').mean().to_period('M')` | Aggregate monthly, then represent labels as periods |
| `period_ts.resample('Q-DEC', convention='start')` | Upsample an annual PeriodIndex from the start of each period |

`QE` and `YE` are timestamp offset aliases used with a `DatetimeIndex`. Period frequencies describe spans and retain aliases such as `Q-DEC` and `Y-DEC`; do not substitute timestamp aliases mechanically.

### Code Snippet: Resample a Period-Indexed DataFrame

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

# Resample with periods
frame = pd.DataFrame(rng.standard_normal((24, 4)),
                     index=pd.period_range('1-2000', '12-2001', freq='M'),
                     columns=['Colorado', 'Texas', 'New York', 'Ohio'])

# Downsample to annual
annual = frame.resample('Y-DEC').mean()

# Upsample annual to quarterly
quarterly = annual.resample('Q-DEC', convention='start').ffill()
```

# More Window and Selection Options

_The lecture's rolling and EWM windows cover daily work; these options cover the rest._

## Expanding Windows, Rolling Quantiles, and Custom Functions

An **expanding window** grows from the first row to the current one, so its mean is a running average of everything so far. A rolling quantile, such as the median, moves less than a mean when one reading is wild.

### Reference Card: More Window Options

- `ts.expanding().mean()`: Running mean from the first row to the current one; no `NaN` at the start.
- `ts.rolling(window=5).quantile(0.5)`: Rolling median.
- `ts.rolling(window=5).apply(custom_func)`: Run your own function on each window's values.
- `a.rolling(30).corr(b)`: Rolling correlation between two aligned series, such as daily heart rate and blood pressure.
- `ts.ewm(alpha=0.3).mean()`: Set the decay directly instead of through `span`; larger `alpha` weights recent observations more.
- `ts.ewm(halflife=2).mean()`: Weighted mean whose weights halve every two observations.
- `ts.shift(1, freq='D')`: Move the timestamps one day later instead of the values, so nothing becomes `NaN`.
- `ts.resample('ME').agg(mean='mean', count='count')`: Named aggregation on a Series, one named column per summary.
- `ts.ewm(span=5).std()`: Exponentially weighted standard deviation.
- `ts.truncate(before='2023-06-01', after='2023-06-30')`: Drop everything outside the range (requires a sorted index).

### Code Snippet: Expanding Mean and Rolling Median

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
temps = pd.Series(98.6 + np.cumsum(rng.standard_normal(100) * 0.1),
                  index=pd.date_range('2023-01-01', periods=100, freq='D'))
print(pd.DataFrame({
    'temperature': temps,
    'expanding_mean': temps.expanding().mean(),
    'rolling_median': temps.rolling(window=5).quantile(0.5),
}).head(6).round(2))
```

```text
            temperature  expanding_mean  rolling_median
2023-01-01        98.63           98.63             NaN
2023-01-02        98.53           98.58             NaN
2023-01-03        98.60           98.59             NaN
2023-01-04        98.70           98.61             NaN
2023-01-05        98.50           98.59           98.60
2023-01-06        98.37           98.55           98.53
```

# High-Frequency Data Analysis

_High-frequency data requires special handling for irregular intervals and tick data._

## Tick Data Processing

### Reference Card: Tick Data Processing

```python
import pandas as pd

# Process high-frequency tick data (e.g., sensor readings)
def process_tick_data(df, freq='1min'):
    """Aggregate a DatetimeIndex table with ``value`` and ``volume`` columns.

    ``n_obs`` counts input rows in each interval; it is not a source
    column. Add another named aggregation when a separate quantity field
    should be summed. Choose and document a naive or timezone-aware index
    before calling; resampling preserves that time basis.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError('df must use a DatetimeIndex')
    resampled = df.sort_index().resample(freq).agg(
        last_value=('value', 'last'),
        total_volume=('volume', 'sum'),
        n_obs=('value', 'size'),
    )
    return resampled
```

# Advanced Time Zone Operations

_Time zones can be complex, especially with daylight saving time transitions and historical data._

## Resolving Clock-Change Times

The lecture sets repeated (fall-back) and skipped (spring-forward) clock times to `NaT` so they can be counted and set aside. When the data carry enough context, pandas can resolve them instead.

### Reference Card: Resolving Clock-Change Times

| Argument | Effect |
|----------|--------|
| `ambiguous='infer'` | For a sorted run of readings that passes through the repeated hour, assign the first pass to daylight time and the second to standard time |
| `ambiguous=[True, False, ...]` | State for each reading whether it is daylight time (`True`) or standard time (`False`) |
| `nonexistent='shift_forward'` | Move a skipped time to the first valid time after the gap |

### Code Snippet: Resolve Repeated and Skipped Clock Times

```python
import pandas as pd

# A sorted run of readings that passes through 01:00-01:59 twice
fall_back = pd.DatetimeIndex(['2024-11-03 00:30', '2024-11-03 01:00', '2024-11-03 01:30',
                              '2024-11-03 01:00', '2024-11-03 01:30', '2024-11-03 02:00'])
print(fall_back.tz_localize('America/New_York', ambiguous='infer'))

# A skipped spring-forward time moved to 03:00
spring = pd.DatetimeIndex(['2024-03-10 02:30'])
print(spring.tz_localize('America/New_York', nonexistent='shift_forward'))
```

```text
DatetimeIndex(['2024-11-03 00:30:00-04:00', '2024-11-03 01:00:00-04:00',
               '2024-11-03 01:30:00-04:00', '2024-11-03 01:00:00-05:00',
               '2024-11-03 01:30:00-05:00', '2024-11-03 02:00:00-05:00'],
              dtype='datetime64[us, America/New_York]', freq=None)
DatetimeIndex(['2024-03-10 03:00:00-04:00'], dtype='datetime64[us, America/New_York]', freq=None)
```

`'infer'` relies on the readings being in recording order: it raises `ValueError` when it cannot see the clock repeat, and it can guess wrong if the rows were shuffled. `ambiguous='NaT'` from the lecture is the safer default.

## Operations Between Different Time Zones

### Reference Card: Combining Series Across Time Zones

```python
import pandas as pd

# Combining time series with different time zones
dates = pd.date_range('2024-03-01 09:00', periods=4, freq='D')
ts = pd.Series(range(4), index=dates)
ts1 = ts.tz_localize('Europe/London')
ts2 = ts1.iloc[2:].tz_convert('Europe/Moscow')
result = ts1 + ts2  # Aligned on the same instants
print(result.index.tz)  # UTC
```

# Custom Frequency Classes

_For specialized time series needs, you can create custom frequency classes, though this is rarely necessary._

## Custom Business Day Frequencies

### Reference Card: Custom Business Day Frequencies

```python
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

# Create custom business day (e.g., excluding specific holidays)
custom_bday = CustomBusinessDay(holidays=['2023-12-25'])
dates = pd.date_range('2023-12-01', '2023-12-31', freq=custom_bday)
```

# Time Series Visualization

## Interactive Time Series Plots

### Reference Card: Interactive Time Series Plot with Plotly

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create interactive plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ts.index,
    y=ts.values,
    mode='lines',
    name='Time Series'
))
fig.show()
```

## Autocorrelation and Partial Autocorrelation

### Reference Card: ACF and PACF Plots

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ts, lags=40, ax=axes[0])
plot_pacf(ts, lags=40, ax=axes[1])
plt.tight_layout()
plt.show()
```

These advanced topics will help you handle complex time series analysis scenarios in specialized applications. For most daily data science work, the content in the main lecture is sufficient.
