# Time Series Analysis with pandas

Welcome to our exploration of time series analysis! Time series data is fundamental in many fields including finance, economics, ecology, neuroscience, and physics. This lecture covers the essential tools and techniques for working with temporal data using pandas.

By the end of this session, you'll understand how to manipulate dates and times, work with time series indexes, perform resampling operations, and apply moving window functions to analyze temporal patterns in your data.

## Learning Objectives

- Master date and time data types in Python and pandas
- Create and manipulate time series with [`DatetimeIndex`](11/README.md:1)
- Understand frequency codes and date offsets
- Handle time zones and period arithmetic
- Perform resampling for frequency conversion
- Apply moving window functions for time series analysis

## What is Time Series Data?

Time series data consists of observations recorded at many points in time. These can be:

- **Timestamps**: Specific instants in time (2023-01-15 14:30:00)
- **Fixed periods**: Time spans like months or years (January 2023, Q1 2023)
- **Intervals**: Ranges defined by start and end timestamps
- **Elapsed time**: Measurements relative to a start time

Time series can be **fixed frequency** (regular intervals like daily, hourly) or **irregular** (no fixed pattern between observations).

## 11.1 Date and Time Data Types and Tools

### Python's datetime Module

Python's standard library provides foundational date/time functionality:

```python
from datetime import datetime, timedelta

# Current date and time
now = datetime.now()
print(now)  # 2023-04-12 13:09:16.484533

# Extract components
print(now.year, now.month, now.day)  # (2023, 4, 12)

# Time differences
delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
print(delta)  # 926 days, 15:45:00
print(delta.days)  # 926

# Adding/subtracting time
start = datetime(2011, 1, 7)
future = start + timedelta(12)  # 12 days later
past = start - 2 * timedelta(12)  # 24 days earlier
```

### String to Datetime Conversion

pandas provides powerful tools for parsing date strings:

```python
import pandas as pd
import numpy as np

# Parse date strings automatically
datestrs = ['2011-07-06 12:00:00', '2011-08-06 00:00:00']
dates = pd.to_datetime(datestrs)
print(dates)
# DatetimeIndex(['2011-07-06 12:00:00', '2011-08-06 00:00:00'], dtype='datetime64[ns]')

# Handle unparseable dates with errors='coerce'
mixed_dates = ['2011-07-06 12:00:00', 'invalid_date', '2011-08-06']
parsed = pd.to_datetime(mixed_dates, errors='coerce')
print(parsed)  # NaT (Not a Time) for invalid dates
```

## 11.2 Time Series Basics

### Creating Time Series

A time series in pandas is typically a [`Series`](11/README.md:1) with a [`DatetimeIndex`](11/README.md:1):

```python
# Create time series from datetime objects
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), 
         datetime(2011, 1, 7), datetime(2011, 1, 8)]
ts = pd.Series(np.random.randn(4), index=dates)
print(ts)

# The index is automatically converted to DatetimeIndex
print(type(ts.index))  # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
print(ts.index.dtype)  # datetime64[ns]
```

### Indexing and Selection

Time series support flexible indexing and slicing:

```python
# Create longer time series
longer_ts = pd.Series(np.random.randn(1000),
                     index=pd.date_range('2000-01-01', periods=1000))

# Select by year
print(longer_ts['2001'])

# Select by year and month
print(longer_ts['2001-05'])

# Slice by date range
print(longer_ts['2001-05-15':'2001-05-25'])

# Boolean indexing works too
recent_data = longer_ts[longer_ts.index >= '2001-01-01']
```

### Working with DataFrames

Time series operations work seamlessly with DataFrames:

```python
# Create DataFrame with date index
dates = pd.date_range('2000-01-01', periods=100, freq='W-WED')
df = pd.DataFrame(np.random.randn(100, 4),
                  index=dates,
                  columns=['A', 'B', 'C', 'D'])

# Select data for specific years
print(df['2001'])

# Select specific columns and date ranges
print(df.loc['2001-05':'2001-08', ['A', 'C']])
```

## 11.3 Date Ranges, Frequencies, and Shifting

### Generating Date Ranges

The [`pd.date_range()`](11/README.md:1) function creates sequences of dates:

```python
# Basic date range (daily by default)
index = pd.date_range('2012-04-01', '2012-06-01')
print(index)

# Specify number of periods
daily_range = pd.date_range(start='2012-04-01', periods=20)
monthly_range = pd.date_range(end='2012-06-01', periods=20)

# Business month end frequency
business_months = pd.date_range('2000-01-01', '2000-12-01', freq='BM')
print(business_months)
```

### Frequency Codes and Date Offsets

pandas supports many frequency codes for different time intervals:

| Code | Description | Code | Description |
|------|-------------|------|-------------|
| D | Calendar day | B | Business day |
| W | Weekly | M | Month end |
| BM | Business month end | MS | Month start |
| Q | Quarter end | A | Year end |
| H | Hourly | T, min | Minutely |
| S | Secondly | L, ms | Millisecond |

```python
# Different frequencies
hourly = pd.date_range('2000-01-01', periods=10, freq='H')
every_4_hours = pd.date_range('2000-01-01', '2000-01-03 23:59', freq='4H')

# Custom frequency strings
custom_freq = pd.date_range('2000-01-01', periods=10, freq='1h30min')

# Week of month dates (3rd Friday of each month)
third_fridays = pd.date_range('2012-01-01', '2012-09-01', freq='WOM-3FRI')
```

### Shifting Data

Shifting moves data forward or backward in time:

```python
ts = pd.Series(np.random.randn(4),
               index=pd.date_range('2000-01-01', periods=4, freq='M'))

# Shift data by 2 periods (keeps same timestamps)
shifted_data = ts.shift(2)

# Shift timestamps by 2 months (keeps same data alignment)
shifted_time = ts.shift(2, freq='M')

# Calculate period-over-period changes
returns = ts / ts.shift(1) - 1
```

## 11.4 Time Zone Handling

### Time Zone Localization and Conversion

pandas provides comprehensive time zone support:

```python
# Create naive (no timezone) time series
dates = pd.date_range('2012-03-09 09:30', periods=6)
ts = pd.Series(np.random.randn(6), index=dates)

# Localize to a specific timezone
ts_utc = ts.tz_localize('UTC')
ts_eastern = ts.tz_localize('US/Eastern')

# Convert between timezones
ts_tokyo = ts_utc.tz_convert('Asia/Tokyo')

# Create timezone-aware date ranges
utc_range = pd.date_range('2012-03-09 09:30', periods=10, tz='UTC')
```

### Operations with Different Time Zones

```python
# Combining different timezone series results in UTC
ts1 = pd.Series(np.random.randn(3),
                index=pd.date_range('2012-03-07 09:30', periods=3, freq='B', tz='US/Eastern'))
ts2 = pd.Series(np.random.randn(3),
                index=pd.date_range('2012-03-07 09:30', periods=3, freq='B', tz='Europe/Berlin'))

# Result is automatically converted to UTC
result = ts1 + ts2
print(result.index.tz)  # UTC
```

## 11.5 Periods and Period Arithmetic

### Working with Periods

Periods represent time spans rather than specific timestamps:

```python
# Create period objects
p = pd.Period('2011', freq='A-DEC')  # Annual period ending in December
print(p)  # Period('2011', 'A-DEC')

# Period arithmetic
print(p + 5)  # Period('2016', 'A-DEC')
print(p - 2)  # Period('2009', 'A-DEC')

# Create period ranges
periods = pd.period_range('2000-01-01', '2000-06-30', freq='M')
pts = pd.Series(np.random.randn(6), index=periods)
```

### Period Frequency Conversion

Convert periods between different frequencies:

```python
p = pd.Period('2011', freq='A-DEC')

# Convert to monthly periods
print(p.asfreq('M', how='start'))  # Period('2011-01', 'M')
print(p.asfreq('M', how='end'))    # Period('2011-12', 'M')

# Convert entire period series
annual_periods = pd.period_range('2006', '2009', freq='A-DEC')
ts = pd.Series(np.random.randn(4), index=annual_periods)
monthly_ts = ts.asfreq('M', how='start')
```

### Converting Between Timestamps and Periods

```python
# Timestamp to period conversion
dates = pd.date_range('2000-01-01', periods=3, freq='M')
ts = pd.Series(np.random.randn(3), index=dates)
pts = ts.to_period()

# Period to timestamp conversion
back_to_ts = pts.to_timestamp(how='end')
```

## 11.6 Resampling and Frequency Conversion

Resampling converts time series from one frequency to another.

### Downsampling

Aggregating higher frequency data to lower frequency:

```python
# Create minute-frequency data
dates = pd.date_range('2000-01-01', periods=12, freq='T')
ts = pd.Series(np.arange(12), index=dates)

# Downsample to 5-minute intervals
five_min_sum = ts.resample('5min').sum()
five_min_mean = ts.resample('5min').mean()
five_min_ohlc = ts.resample('5min').ohlc()  # Open, High, Low, Close

# Control bin edges and labels
result = ts.resample('5min', closed='right', label='right').sum()
```

### Upsampling and Interpolation

Converting lower frequency to higher frequency:

```python
# Weekly data
frame = pd.DataFrame(np.random.randn(2, 4),
                     index=pd.date_range('2000-01-01', periods=2, freq='W-WED'),
                     columns=['A', 'B', 'C', 'D'])

# Upsample to daily frequency
daily_frame = frame.resample('D').asfreq()  # Introduces NaN values

# Forward fill missing values
daily_filled = frame.resample('D').ffill()

# Limit forward filling
limited_fill = frame.resample('D').ffill(limit=2)
```

### Resampling with Periods

```python
# Monthly period data
frame = pd.DataFrame(np.random.randn(24, 4),
                     index=pd.period_range('2000-01', '2001-12', freq='M'),
                     columns=['A', 'B', 'C', 'D'])

# Downsample to annual
annual_frame = frame.resample('A-DEC').mean()

# Upsample to quarterly
quarterly_frame = annual_frame.resample('Q-DEC').ffill()
```

## 11.7 Moving Window Functions

Moving window functions compute statistics over sliding time windows.

### Rolling Windows

```python
# Load sample stock data and resample to business days
# (In practice, you'd load real data)
np.random.seed(12345)
close_px = pd.Series(np.random.randn(1000).cumsum(),
                     index=pd.date_range('2000-01-01', periods=1000))
close_px = close_px.resample('B').ffill()

# 250-day moving average
ma250 = close_px.rolling(250).mean()

# Rolling standard deviation with minimum periods
std250 = close_px.pct_change().rolling(250, min_periods=10).std()

# Rolling correlations
returns = pd.DataFrame({
    'AAPL': np.random.randn(1000),
    'MSFT': np.random.randn(1000),
    'SPX': np.random.randn(1000)
}, index=pd.date_range('2000-01-01', periods=1000))

# Rolling correlation between AAPL and SPX
corr = returns['AAPL'].rolling(125, min_periods=100).corr(returns['SPX'])
```

### Expanding Windows

```python
# Expanding window (all data from start to current point)
expanding_mean = close_px.expanding().mean()
expanding_std = close_px.expanding().std()
```

### Exponentially Weighted Functions

```python
# Exponentially weighted moving average
ema = close_px.ewm(span=30).mean()

# Compare with simple moving average
sma = close_px.rolling(30, min_periods=20).mean()
```

### Custom Moving Window Functions

```python
from scipy.stats import percentileofscore

def score_at_2percent(x):
    return percentileofscore(x, 0.02)

# Apply custom function over rolling window
result = returns['AAPL'].rolling(250).apply(score_at_2percent)
```

## Practical Time Series Analysis Workflow

### Complete Example: Stock Price Analysis

```python
# 1. Create sample time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
stock_data = pd.Series(prices, index=dates, name='Stock_Price')

# 2. Basic time series operations
print("Data shape:", stock_data.shape)
print("Date range:", stock_data.index.min(), "to", stock_data.index.max())

# 3. Resample to monthly data
monthly_prices = stock_data.resample('M').last()
monthly_returns = monthly_prices.pct_change().dropna()

# 4. Calculate moving averages
stock_data_clean = stock_data.dropna()
sma_20 = stock_data_clean.rolling(20).mean()
sma_50 = stock_data_clean.rolling(50).mean()

# 5. Volatility analysis
daily_returns = stock_data_clean.pct_change().dropna()
rolling_vol = daily_returns.rolling(30).std() * np.sqrt(252)  # Annualized

# 6. Time-based filtering
covid_period = stock_data['2020-03':'2020-06']
recovery_period = stock_data['2020-07':'2021-12']

print("Analysis complete!")
```

## Key Takeaways

1. **DatetimeIndex** is the foundation of time series in pandas
2. **Frequency codes** provide flexible ways to specify time intervals
3. **Resampling** allows conversion between different time frequencies
4. **Moving windows** enable trend and volatility analysis
5. **Time zones** can be handled systematically across operations
6. **Periods** represent time spans and support arithmetic operations

## Next Steps

- Practice with real-world time series datasets
- Explore seasonal decomposition techniques
- Learn about time series forecasting methods
- Investigate financial time series analysis
- Study irregularly-spaced time series handling

Time series analysis is a rich field with applications across many domains. The pandas tools covered here provide a solid foundation for temporal data manipulation and analysis in your data science projects.