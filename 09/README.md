Time Series Analysis: Temporal Data and Trends

See [BONUS.md](BONUS.md) for advanced topics:

- Advanced time series decomposition and seasonal analysis
- Time series forecasting with ARIMA and exponential smoothing
- Time zone handling and international data
- High-frequency data analysis and tick data
- Time series visualization and interactive dashboards

*Fun fact: Time series analysis is like being a detective for data - you're looking for patterns, trends, and clues that reveal the story of how things change over time. It's the difference between knowing what happened and understanding why it happened.*

![xkcd 2048: Time](media/xkcd_2048.png)

*"The data clearly shows that our hypothesis is correct, assuming we ignore all the data that doesn't support our hypothesis."*

Time series analysis is the art of understanding temporal patterns in data. This lecture covers the essential tools for time series analysis: **datetime handling**, **resampling and frequency conversion**, **rolling window operations**, and **automation** for handling time-based data.

**Learning Objectives:**

- Master datetime data types and parsing
- Perform time series indexing and selection
- Use resampling and frequency conversion
- Apply rolling window operations
- Handle time zones and temporal data
- Automate time-based tasks with cron jobs

# Understanding Time Series Data

*Reality check: Time series data is everywhere - stock prices, weather data, website traffic, sensor readings. Understanding how to work with temporal data is essential for any data scientist.*

Time series data is characterized by observations collected over time, where the order and timing of observations matter. Unlike cross-sectional data, time series data has a natural temporal structure that we can exploit for analysis.

**Visual Guide - Time Series Characteristics:**

```
TIME SERIES DATA STRUCTURE
┌─────────────┬─────────────┬─────────────┐
│ Timestamp   │ Value       │ Context     │
├─────────────┼─────────────┼─────────────┤
│ 2023-01-01  │ 100.5       │ Stock Price │
│ 2023-01-02  │ 102.3       │ Stock Price │
│ 2023-01-03  │ 98.7        │ Stock Price │
│ 2023-01-04  │ 105.2       │ Stock Price │
│ 2023-01-05  │ 103.8       │ Stock Price │
└─────────────┴─────────────┴─────────────┘

KEY CHARACTERISTICS:
- Temporal ordering matters
- Observations are dependent
- Patterns may repeat (seasonality)
- Trends may exist
- Noise and outliers are common
```

## Types of Time Series

**Reference:**

- **Regular time series**: Fixed intervals (daily, hourly, monthly)
- **Irregular time series**: Variable intervals (event-based)
- **Seasonal time series**: Patterns repeat over time
- **Trending time series**: Long-term direction
- **Stationary time series**: Statistical properties don't change

**Example:**

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create regular time series
dates = pd.date_range('2023-01-01', periods=10, freq='D')
values = np.cumsum(np.random.randn(10)) + 100
ts_regular = pd.Series(values, index=dates)
print("Regular time series:")
print(ts_regular.head())

# Create irregular time series
irregular_dates = pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-07'])
irregular_values = [100, 105, 98]
ts_irregular = pd.Series(irregular_values, index=irregular_dates)
print("\nIrregular time series:")
print(ts_irregular)
```

# Date and Time Data Types

*Think of datetime objects as the Swiss Army knife of temporal data - they can represent any moment in time with precision down to microseconds, and pandas makes them incredibly powerful for analysis.*

## Python datetime Module

**Reference:**

- `datetime.now()` - Current date and time
- `datetime(year, month, day)` - Create specific date
- `datetime.strptime(string, format)` - Parse string to datetime
- `datetime.strftime(format)` - Format datetime to string
- `timedelta(days=1)` - Time differences

**Example:**

```python
from datetime import datetime, timedelta

# Current time
now = datetime.now()
print(f"Current time: {now}")

# Specific date
birthday = datetime(1990, 5, 15)
print(f"Birthday: {birthday}")

# String parsing
date_str = "2023-12-25 14:30:00"
parsed_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
print(f"Parsed date: {parsed_date}")

# String formatting
formatted = parsed_date.strftime("%B %d, %Y at %I:%M %p")
print(f"Formatted: {formatted}")

# Time differences
time_diff = now - birthday
print(f"Age in days: {time_diff.days}")
```

## pandas DatetimeIndex

**Reference:**

- `pd.to_datetime()` - Convert to datetime
- `pd.date_range()` - Create date range
- `pd.DatetimeIndex()` - Create datetime index
- `df.set_index('date')` - Set datetime index
- `df.index` - Access datetime index

**Example:**

```python
# Convert to datetime
date_strings = ['2023-01-01', '2023-01-02', '2023-01-03']
dates = pd.to_datetime(date_strings)
print("Converted dates:")
print(dates)

# Create date range
date_range = pd.date_range('2023-01-01', periods=10, freq='D')
print("\nDate range:")
print(date_range)

# Create DataFrame with datetime index
df = pd.DataFrame({
    'value': np.random.randn(10)
}, index=date_range)
print("\nDataFrame with datetime index:")
print(df.head())
```

## Date Range Generation

**Reference:**

- `pd.date_range(start, end, freq='D')` - Create date range
- `pd.bdate_range(start, end)` - Business days only
- `pd.date_range(freq='B')` - Business frequency
- `pd.date_range(freq='W-MON')` - Weekly on Monday
- `pd.date_range(freq='MS')` - Month start
- `pd.date_range(freq='QS')` - Quarter start

**Example:**

```python
# Different date range types
print("Daily range:")
daily = pd.date_range('2023-01-01', '2023-01-10', freq='D')
print(daily)

print("\nBusiness days only:")
business = pd.bdate_range('2023-01-01', '2023-01-10')
print(business)

print("\nWeekly range (Mondays):")
weekly = pd.date_range('2023-01-01', '2023-03-01', freq='W-MON')
print(weekly)

print("\nMonthly range:")
monthly = pd.date_range('2023-01-01', '2023-12-01', freq='MS')
print(monthly)
```

## Frequency Inference

**Reference:**

- `pd.infer_freq(ts.index)` - Infer frequency from time series
- `ts.asfreq(freq)` - Convert to specific frequency
- `ts.resample(freq).asfreq()` - Resample and convert frequency

**Example:**

```python
# Create time series with inferred frequency
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)

# Infer frequency
freq = pd.infer_freq(ts.index)
print(f"Inferred frequency: {freq}")

# Convert to different frequency
ts_weekly = ts.asfreq('W')
print(f"Weekly frequency: {pd.infer_freq(ts_weekly.index)}")
```

## Shifting and Lagging

**Reference:**

- `ts.shift(1)` - Shift by 1 period (lag)
- `ts.shift(-1)` - Shift by -1 period (lead)
- `ts.diff()` - First difference
- `ts.pct_change()` - Percentage change
- `ts.shift(1, freq='D')` - Shift by 1 day

**Example:**

```python
# Create sample time series
dates = pd.date_range('2023-01-01', periods=10, freq='D')
ts = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=dates)

# Shifting operations
ts['lag_1'] = ts.shift(1)  # Previous day
ts['lead_1'] = ts.shift(-1)  # Next day
ts['diff'] = ts.diff()  # First difference
ts['pct_change'] = ts.pct_change()  # Percentage change

print("Time series with shifts:")
print(ts)
```

## Exponentially Weighted Functions

**Reference:**

- `ts.ewm(span=5).mean()` - Exponentially weighted moving average
- `ts.ewm(alpha=0.3).mean()` - EWM with alpha parameter
- `ts.ewm(halflife=2).mean()` - EWM with half-life
- `ts.ewm(span=5).std()` - Exponentially weighted standard deviation

**Example:**

```python
# Create sample time series
dates = pd.date_range('2023-01-01', periods=50, freq='D')
ts = pd.Series(np.cumsum(np.random.randn(50)) + 100, index=dates)

# Exponentially weighted functions
ts['ewm_mean'] = ts.ewm(span=5).mean()
ts['ewm_std'] = ts.ewm(span=5).std()
ts['ewm_alpha'] = ts.ewm(alpha=0.3).mean()

print("Time series with EWM functions:")
print(ts[['value', 'ewm_mean', 'ewm_std']].head(10))
```

## Time Zone Handling

**Reference:**

- `pd.to_datetime().dt.tz_localize()` - Add timezone
- `pd.to_datetime().dt.tz_convert()` - Convert timezone
- `pd.Timestamp.now(tz='UTC')` - Current time in timezone
- `df.index.tz_localize('UTC')` - Localize index
- `df.index.tz_convert('US/Eastern')` - Convert timezone

**Example:**

```python
# Create timezone-aware datetime
utc_time = pd.Timestamp.now(tz='UTC')
print(f"UTC time: {utc_time}")

# Convert to different timezone
eastern_time = utc_time.tz_convert('US/Eastern')
print(f"Eastern time: {eastern_time}")

# Create timezone-aware DataFrame
df_tz = pd.DataFrame({
    'value': np.random.randn(3)
}, index=pd.date_range('2023-01-01', periods=3, freq='D'))

# Localize to UTC
df_tz.index = df_tz.index.tz_localize('UTC')
print("\nUTC DataFrame:")
print(df_tz)

# Convert to Eastern time
df_tz.index = df_tz.index.tz_convert('US/Eastern')
print("\nEastern DataFrame:")
print(df_tz)
```

## Business Day Handling

**Reference:**

- `pd.bdate_range(start, end)` - Business date range
- `pd.date_range(freq='B')` - Business frequency
- `ts.resample('B').mean()` - Resample to business days
- `ts.asfreq('B')` - Convert to business frequency

**Example:**

```python
# Business day operations
business_dates = pd.bdate_range('2023-01-01', '2023-01-31')
print("Business days in January 2023:")
print(business_dates)

# Create business day time series
ts_business = pd.Series(np.random.randn(len(business_dates)), index=business_dates)
print(f"\nBusiness day time series shape: {ts_business.shape}")

# Convert daily to business days
daily_ts = pd.Series(np.random.randn(31), index=pd.date_range('2023-01-01', periods=31, freq='D'))
business_ts = daily_ts.asfreq('B')
print(f"Converted to business days: {business_ts.shape}")
```

# LIVE DEMO!

# Time Series Indexing and Selection

*Time series indexing is like having a time machine for your data - you can jump to any point in time, slice through time periods, and even travel backwards to see how things were.*

## Basic Time Series Selection

**Reference:**

- `ts['2023-01-01']` - Select specific date
- `ts['2023-01-01':'2023-01-31']` - Select date range
- `ts['2023']` - Select entire year
- `ts['2023-01']` - Select specific month
- `ts.loc['2023-01-01']` - Label-based selection
- `ts.iloc[0:10]` - Position-based selection

**Example:**

```python
# Create sample time series
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365)) + 100
ts = pd.Series(values, index=dates)

# Select specific date
print("January 1, 2023:")
print(ts['2023-01-01'])

# Select date range
print("\nJanuary 2023:")
print(ts['2023-01-01':'2023-01-31'].head())

# Select entire year
print("\n2023 data shape:")
print(ts['2023'].shape)

# Select specific month
print("\nJanuary 2023:")
print(ts['2023-01'].head())
```

## Advanced Time Series Selection

**Reference:**

- `ts.between_time('09:00', '17:00')` - Select time range
- `ts.at_time('12:00')` - Select specific time
- `ts.first('10D')` - First 10 days
- `ts.last('10D')` - Last 10 days
- `ts.truncate(before='2023-06-01')` - Truncate before date
- `ts.truncate(after='2023-06-30')` - Truncate after date

**Example:**

```python
# Create hourly time series
hourly_dates = pd.date_range('2023-01-01', periods=24*7, freq='H')
hourly_values = np.random.randn(24*7)
ts_hourly = pd.Series(hourly_values, index=hourly_dates)

# Select business hours (9 AM to 5 PM)
business_hours = ts_hourly.between_time('09:00', '17:00')
print("Business hours data:")
print(business_hours.head())

# Select specific time
noon_data = ts_hourly.at_time('12:00')
print("\nNoon data:")
print(noon_data.head())

# Select first and last periods
print("\nFirst 3 days:")
print(ts_hourly.first('3D').head())

print("\nLast 3 days:")
print(ts_hourly.last('3D').head())
```

# Resampling and Frequency Conversion

*Resampling is like changing the lens on your camera - you can zoom in to see more detail (higher frequency) or zoom out to see the big picture (lower frequency).*

**Visual Guide - Frequency Conversion:**

```
HIGH FREQUENCY (Daily)              LOW FREQUENCY (Monthly)
┌─────────────┬─────────┐           ┌─────────────┬─────────┐
│ Date        │ Value   │           │ Date        │ Value   │
├─────────────┼─────────┤           ├─────────────┼─────────┤
│ 2023-01-01  │ 100     │           │ 2023-01-01  │ 105.2   │
│ 2023-01-02  │ 102     │    →      │ 2023-02-01  │ 108.7   │
│ 2023-01-03  │ 98      │           │ 2023-03-01  │ 112.3   │
│ 2023-01-04  │ 105     │           │ 2023-04-01  │ 115.8   │
│ 2023-01-05  │ 110     │           └─────────────┴─────────┘
│ ...         │ ...     │
└─────────────┴─────────┘
```

## Basic Resampling

**Reference:**

- `ts.resample('D')` - Daily resampling
- `ts.resample('W')` - Weekly resampling
- `ts.resample('M')` - Monthly resampling
- `ts.resample('Q')` - Quarterly resampling
- `ts.resample('A')` - Annual resampling
- `ts.resample('H')` - Hourly resampling

**Example:**

```python
# Create daily time series
daily_dates = pd.date_range('2023-01-01', periods=30, freq='D')
daily_values = np.cumsum(np.random.randn(30)) + 100
ts_daily = pd.Series(daily_values, index=daily_dates)

# Resample to different frequencies
print("Original daily data shape:", ts_daily.shape)

# Weekly resampling
weekly = ts_daily.resample('W').mean()
print("Weekly resampled shape:", weekly.shape)
print("Weekly data:")
print(weekly.head())

# Monthly resampling
monthly = ts_daily.resample('M').mean()
print("\nMonthly resampled shape:", monthly.shape)
print("Monthly data:")
print(monthly.head())
```

## Resampling with Different Aggregations

**Reference:**

- `ts.resample('D').mean()` - Mean aggregation
- `ts.resample('D').sum()` - Sum aggregation
- `ts.resample('D').max()` - Maximum aggregation
- `ts.resample('D').min()` - Minimum aggregation
- `ts.resample('D').std()` - Standard deviation
- `ts.resample('D').agg(['mean', 'std', 'min', 'max'])` - Multiple aggregations

**Example:**

```python
# Create sample data with multiple columns
df = pd.DataFrame({
    'value': np.random.randn(365),
    'volume': np.random.randint(100, 1000, 365)
}, index=pd.date_range('2023-01-01', periods=365, freq='D'))

# Different resampling methods
print("Daily to weekly resampling:")
weekly_stats = df.resample('W').agg({
    'value': ['mean', 'std', 'min', 'max'],
    'volume': 'sum'
})
print(weekly_stats.head())

# Custom resampling function
def custom_agg(series):
    return pd.Series({
        'mean': series.mean(),
        'std': series.std(),
        'range': series.max() - series.min(),
        'count': len(series)
    })

print("\nCustom aggregation:")
custom_stats = df['value'].resample('M').apply(custom_agg)
print(custom_stats.head())
```

# LIVE DEMO!

# Rolling Window Operations

*Rolling windows are like looking through a moving frame - you can see how things change over time by examining a sliding window of observations.*

**Visual Guide - Rolling Window:**

```
ROLLING WINDOW (size=3)
┌─────────────┬─────────┬─────────┐
│ Date        │ Value   │ Rolling │
├─────────────┼─────────┼─────────┤
│ 2023-01-01  │ 100     │ NaN     │
│ 2023-01-02  │ 102     │ NaN     │
│ 2023-01-03  │ 98      │ 100.0   │ ← mean(100,102,98)
│ 2023-01-04  │ 105     │ 101.7   │ ← mean(102,98,105)
│ 2023-01-05  │ 110     │ 101.0   │ ← mean(98,105,110)
│ 2023-01-06  │ 108     │ 107.7   │ ← mean(105,110,108)
└─────────────┴─────────┴─────────┘
```

## Basic Rolling Operations

**Reference:**

- `ts.rolling(window=5)` - 5-period rolling window
- `ts.rolling(window=5).mean()` - Rolling mean
- `ts.rolling(window=5).std()` - Rolling standard deviation
- `ts.rolling(window=5).sum()` - Rolling sum
- `ts.rolling(window=5).min()` - Rolling minimum
- `ts.rolling(window=5).max()` - Rolling maximum

**Example:**

```python
# Create sample time series
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100)) + 100
ts = pd.Series(values, index=dates)

# Rolling statistics
ts['rolling_mean'] = ts.rolling(window=7).mean()
ts['rolling_std'] = ts.rolling(window=7).std()
ts['rolling_min'] = ts.rolling(window=7).min()
ts['rolling_max'] = ts.rolling(window=7).max()

print("Time series with rolling statistics:")
print(ts[['value', 'rolling_mean', 'rolling_std']].head(10))
```

## Advanced Rolling Operations

**Reference:**

- `ts.rolling(window=5, center=True)` - Centered rolling window
- `ts.rolling(window=5, min_periods=3)` - Minimum periods required
- `ts.rolling(window=5).quantile(0.5)` - Rolling median
- `ts.rolling(window=5).apply(custom_func)` - Custom rolling function
- `ts.expanding()` - Expanding window
- `ts.ewm(span=5)` - Exponentially weighted moving average

**Example:**

```python
# Advanced rolling operations
ts['centered_mean'] = ts.rolling(window=7, center=True).mean()
ts['expanding_mean'] = ts.expanding().mean()
ts['ewm_mean'] = ts.ewm(span=7).mean()

# Custom rolling function
def rolling_range(series):
    return series.max() - series.min()

ts['rolling_range'] = ts.rolling(window=7).apply(rolling_range)

print("Advanced rolling statistics:")
print(ts[['value', 'centered_mean', 'expanding_mean', 'ewm_mean']].head(10))
```

# Command Line: Automation with Cron Jobs

*When you need to run time-based analysis automatically, cron jobs are your best friend. They're like having a reliable assistant that never forgets to run your analysis at the right time.*

## Cron Job Basics

**Reference:**

- `crontab -e` - Edit crontab
- `crontab -l` - List cron jobs
- `crontab -r` - Remove all cron jobs
- `crontab -u username -e` - Edit user's crontab

**Cron Schedule Format:**
```
* * * * * command
│ │ │ │ │
│ │ │ │ └─── Day of week (0-7, 0=Sunday)
│ │ │ └───── Month (1-12)
│ │ └─────── Day of month (1-31)
│ └───────── Hour (0-23)
└─────────── Minute (0-59)
```

**Example:**

```bash
# Edit crontab
crontab -e

# Add cron jobs
# Run every day at 2 AM
0 2 * * * /path/to/script.sh

# Run every Monday at 9 AM
0 9 * * 1 /path/to/script.sh

# Run every 15 minutes
*/15 * * * * /path/to/script.sh

# Run at specific times
0 9,17 * * * /path/to/script.sh  # 9 AM and 5 PM
```

## Python Scripts for Cron Jobs

**Reference:**

```python
#!/usr/bin/env python3
"""
Time series analysis script for cron job
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    filename='/path/to/logs/analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_daily_analysis():
    """Run daily time series analysis"""
    try:
        # Load data
        df = pd.read_csv('/path/to/data/daily_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Perform analysis
        daily_stats = df.resample('D').agg({
            'value': ['mean', 'std', 'min', 'max'],
            'volume': 'sum'
        })
        
        # Save results
        daily_stats.to_csv('/path/to/results/daily_stats.csv')
        
        logging.info("Daily analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in daily analysis: {e}")

if __name__ == "__main__":
    run_daily_analysis()
```

## Advanced Cron Job Management

**Reference:**

```bash
# Create cron job script
cat > /path/to/scripts/time_series_analysis.sh << 'EOF'
#!/bin/bash

# Set environment variables
export PATH="/path/to/conda/bin:$PATH"
export PYTHONPATH="/path/to/project:$PYTHONPATH"

# Activate conda environment
source /path/to/conda/bin/activate datasci_217

# Run Python script
cd /path/to/project
python scripts/time_series_analysis.py

# Log completion
echo "$(date): Time series analysis completed" >> /path/to/logs/cron.log
EOF

# Make script executable
chmod +x /path/to/scripts/time_series_analysis.sh

# Add to crontab
echo "0 2 * * * /path/to/scripts/time_series_analysis.sh" | crontab -
```

# Time Series Visualization

*Visualizing time series data is like creating a movie of your data - you can see how things change over time, spot patterns, and identify trends that would be invisible in static snapshots.*

# FIXME: Add time series plot examples showing trend, seasonality, and noise components

# FIXME: Add seasonal decomposition visualization

# FIXME: Add correlation heatmap for time series data

## Basic Time Series Plots

**Reference:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Basic time series plot
fig, ax = plt.subplots(figsize=(12, 6))
ts.plot(ax=ax, title='Time Series Plot')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
plt.show()

# Multiple time series
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Line plot
ts.plot(ax=axes[0, 0], title='Line Plot')
axes[0, 0].grid(True, alpha=0.3)

# Rolling mean
ts.rolling(window=7).mean().plot(ax=axes[0, 1], title='Rolling Mean (7 days)')
axes[0, 1].grid(True, alpha=0.3)

# Histogram
ts.hist(ax=axes[1, 0], bins=30, title='Distribution')
axes[1, 0].grid(True, alpha=0.3)

# Box plot by month
ts.groupby(ts.index.month).plot(kind='box', ax=axes[1, 1], title='Monthly Distribution')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Advanced Time Series Visualization

**Reference:**

```python
# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposition = seasonal_decompose(ts, model='additive', period=7)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(15, 12))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()

# Correlation analysis
fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = ts.to_frame().corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Time Series Correlation Matrix')
plt.show()
```

# LIVE DEMO!

# Key Takeaways

1. **Master datetime handling** - the foundation of time series analysis
2. **Use resampling** to change data frequency
3. **Apply rolling windows** to see trends and patterns
4. **Automate analysis** with cron jobs
5. **Visualize time series** to identify patterns
6. **Handle time zones** for global data
7. **Understand temporal patterns** in your data

You now have the skills to analyze temporal data effectively and automate time-based analysis tasks.

Next week: We'll dive into advanced data science topics and project work!

Practice Challenge

Before next class:
1. **Practice time series analysis:**
   - Load time series data
   - Perform resampling operations
   - Calculate rolling statistics
   
2. **Set up automation:**
   - Create a cron job
   - Write a Python script for time series analysis
   - Test the automation
   
3. **Visualize temporal patterns:**
   - Create time series plots
   - Identify trends and seasonality
   - Use appropriate chart types

Remember: Time series analysis is about understanding how things change over time - look for patterns, trends, and cycles!