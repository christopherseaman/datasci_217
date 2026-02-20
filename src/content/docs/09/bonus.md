---
title: "Bonus: Advanced Time Series"
---


This document covers advanced time series topics that go beyond daily data science practice. These techniques are essential for specialized time series analysis, forecasting, and complex temporal modeling.

## Period Arithmetic and Fiscal Year Handling

*Periods represent time spans, not specific moments. Understanding periods is crucial for fiscal year analysis and business reporting.*

### Period Basics

**Reference:**

| Function | Description |
|----------|-------------|
| `pd.Period('2011', freq='A-DEC')` | Annual period ending December |
| `pd.Period('2011Q4', freq='Q-JAN')` | Quarterly period with fiscal year |
| `pd.period_range(start, end, freq='M')` | Create period range |
| `period.asfreq('M', how='start')` | Convert period frequency |
| `period.to_timestamp()` | Convert period to timestamp |

**Example:**

```python
import pandas as pd
import numpy as np

# Create annual period
p = pd.Period('2011', freq='A-DEC')
print(f"Period: {p}")

# Period arithmetic
p + 5  # Shift forward 5 years
p - 2  # Shift backward 2 years

# Quarterly periods with fiscal year
p = pd.Period('2012Q4', freq='Q-JAN')  # Fiscal year ending in January
print(f"Fiscal Q4: {p}")
print(f"Start date: {p.asfreq('D', how='start')}")
print(f"End date: {p.asfreq('D', how='end')}")

# Create period range
periods = pd.period_range('2000-01-01', '2000-06-30', freq='M')
ts = pd.Series(np.random.randn(6), index=periods)
print("\nPeriod-indexed Series:")
print(ts)
```

### Converting Between Timestamps and Periods

**Reference:**

| Function | Description |
|----------|-------------|
| `ts.to_period()` | Convert timestamp index to periods |
| `ts.to_period('M')` | Convert to monthly periods |
| `pts.to_timestamp()` | Convert periods back to timestamps |
| `pts.to_timestamp(how='end')` | Use end of period as timestamp |

**Example:**

```python
# Timestamp-indexed time series
dates = pd.date_range('2000-01-01', periods=3, freq='M')
ts = pd.Series(np.random.randn(3), index=dates)

# Convert to periods
pts = ts.to_period()
print("Period-indexed:")
print(pts)

# Convert back to timestamps
ts_back = pts.to_timestamp()
print("\nBack to timestamps:")
print(ts_back)
```

## Advanced Time Series Decomposition

*Decomposition separates time series into trend, seasonal, and residual components, revealing underlying patterns.*

### Seasonal Decomposition

**Reference:**

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposition = seasonal_decompose(ts, model='additive', period=7)
# or
decomposition = seasonal_decompose(ts, model='multiplicative', period=12)

# Access components
decomposition.observed  # Original series
decomposition.trend     # Trend component
decomposition.seasonal  # Seasonal component
decomposition.resid     # Residual component
```

**Example:**

```python
# Create seasonal time series (daily disease cases over 3 years)
dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
trend = np.linspace(100, 200, len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
noise = np.random.normal(0, 5, len(dates))
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

### STL Decomposition

**Reference:**

```python
from statsmodels.tsa.seasonal import STL

# STL decomposition (more robust to outliers)
stl = STL(ts, seasonal=365)
result = stl.fit()

# Access components
result.observed  # Original series
result.trend     # Trend component
result.seasonal  # Seasonal component
result.resid     # Residual component
```

## Time Series Forecasting

*Forecasting uses historical patterns to predict future values. Always be honest about uncertainty and prediction intervals.*

### ARIMA Models

**Reference:**

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Check stationarity
def check_stationarity(series):
    """Check if time series is stationary"""
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    
    if result[1] <= 0.05:
        print("Series is stationary")
    else:
        print("Series is not stationary")

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

### Exponential Smoothing

**Reference:**

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

## Advanced Resampling Operations

*Resampling with periods requires careful handling of period boundaries and conventions.*

### Resampling with Periods

**Reference:**

| Function | Description |
|----------|-------------|
| `ts.resample('M', kind='period')` | Resample to periods |
| `ts.resample('M', kind='period').mean()` | Aggregate to periods |
| `ts.resample('Q-DEC', convention='start')` | Period convention for upsampling |

**Example:**

```python
# Resample with periods
frame = pd.DataFrame(np.random.randn(24, 4),
                     index=pd.period_range('1-2000', '12-2001', freq='M'),
                     columns=['Colorado', 'Texas', 'New York', 'Ohio'])

# Downsample to annual
annual = frame.resample('A-DEC').mean()

# Upsample annual to quarterly
quarterly = annual.resample('Q-DEC', convention='start').ffill()
```

### Grouped Time Resampling

**Reference:**

```python
from pandas import Grouper

# Resample multiple time series by group
df = pd.DataFrame({
    'time': times.repeat(3),
    'key': np.tile(['a', 'b', 'c'], N),
    'value': np.arange(N * 3.)
})

# Group by key and resample by time
time_key = Grouper(freq='5min')
resampled = df.set_index('time').groupby(['key', time_key]).sum()
```

## High-Frequency Data Analysis

*High-frequency data requires special handling for irregular intervals and tick data.*

### Tick Data Processing

**Reference:**

```python
# Process high-frequency tick data (e.g., sensor readings)
def process_tick_data(df, freq='1T'):
    """Process tick data into regular intervals"""
    resampled = df.resample(freq).agg({
        'value': 'last',      # Last value in interval
        'volume': 'sum',      # Total volume in interval
        'count': 'count'      # Number of observations in interval
    })
    return resampled
```

## Advanced Time Zone Operations

*Time zones can be complex, especially with daylight saving time transitions and historical data.*

### Time Zone Localization and Conversion

**Reference:**

```python
# Localize naive timestamps
ts_utc = ts.tz_localize('UTC')

# Convert between time zones
ts_eastern = ts_utc.tz_convert('US/Eastern')

# Handle ambiguous times (DST transitions)
ts = ts.tz_localize('US/Eastern', ambiguous='infer')
ts = ts.tz_localize('US/Eastern', nonexistent='NaT')
```

### Operations Between Different Time Zones

**Reference:**

```python
# Combining time series with different time zones
# Result automatically converts to UTC
ts1 = ts[:7].tz_localize('Europe/London')
ts2 = ts1[2:].tz_convert('Europe/Moscow')
result = ts1 + ts2  # Result is in UTC
```

## Custom Frequency Classes

*For specialized time series needs, you can create custom frequency classes, though this is rarely necessary.*

**Reference:**

```python
from pandas.tseries.offsets import CustomBusinessDay

# Create custom business day (e.g., excluding specific holidays)
custom_bday = CustomBusinessDay(holidays=['2023-12-25'])
dates = pd.date_range('2023-12-01', '2023-12-31', freq=custom_bday)
```

## Time Series Visualization

### Interactive Time Series Plots

**Reference:**

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

### Autocorrelation and Partial Autocorrelation

**Reference:**

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ts, lags=40, ax=axes[0])
plot_pacf(ts, lags=40, ax=axes[1])
plt.tight_layout()
plt.show()
```

These advanced topics will help you handle complex time series analysis scenarios in specialized applications. For most daily data science work, the content in the main lecture is sufficient.
