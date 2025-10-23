# Demo 1: datetime Fundamentals

## Learning Objectives
- Master Python datetime module basics
- Create and manipulate pandas DatetimeIndex
- Generate date ranges with different frequencies
- Handle time zones and business days

## Setup

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
```

## Part 1: Python datetime Module

### Basic datetime Operations

```python
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

### datetime Arithmetic

```python
# Date arithmetic
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

# Calculate difference
duration = end_date - start_date
print(f"Duration: {duration.days} days")

# Add/subtract time
future_date = start_date + timedelta(days=30)
past_date = start_date - timedelta(weeks=2)

print(f"30 days from start: {future_date}")
print(f"2 weeks before start: {past_date}")

# Business days (weekdays only)
business_days = pd.bdate_range(start_date, end_date)
print(f"Business days in 2023: {len(business_days)}")
```

## Part 2: pandas DatetimeIndex

### Creating DatetimeIndex

```python
# Convert strings to datetime
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

### Date Range Generation

```python
# Different date range types
print("=== Date Range Generation ===")

# Daily range
daily = pd.date_range('2023-01-01', '2023-01-10', freq='D')
print("Daily range:")
print(daily)

# Business days only
business = pd.bdate_range('2023-01-01', '2023-01-10')
print("\nBusiness days only:")
print(business)

# Weekly range (Mondays)
weekly = pd.date_range('2023-01-01', '2023-03-01', freq='W-MON')
print("\nWeekly range (Mondays):")
print(weekly)

# Monthly range
monthly = pd.date_range('2023-01-01', '2023-12-01', freq='MS')
print("\nMonthly range:")
print(monthly)

# Quarterly range
quarterly = pd.date_range('2023-01-01', '2023-12-31', freq='QS')
print("\nQuarterly range:")
print(quarterly)
```

### Frequency Inference

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

# Resample to different frequency
ts_monthly = ts.resample('M').mean()
print(f"Monthly frequency: {pd.infer_freq(ts_monthly.index)}")
```

## Part 3: Time Zone Handling

### Basic Time Zone Operations

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

### Multiple Time Zones

```python
# Work with multiple time zones
timezones = ['UTC', 'US/Eastern', 'US/Pacific', 'Europe/London', 'Asia/Tokyo']

print("=== Multiple Time Zones ===")
base_time = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')

for tz in timezones:
    converted_time = base_time.tz_convert(tz)
    print(f"{tz}: {converted_time}")
```

## Part 4: Business Day Operations

### Business Day Calculations

```python
# Business day operations
print("=== Business Day Operations ===")

# Business days in January 2023
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

### Custom Business Day Rules

```python
# Custom business day rules
from pandas.tseries.offsets import CustomBusinessDay

# Define custom business day (exclude weekends and specific holidays)
custom_bd = CustomBusinessDay(holidays=['2023-01-02', '2023-01-16'])  # New Year's Day, MLK Day
custom_dates = pd.date_range('2023-01-01', '2023-01-31', freq=custom_bd)

print("=== Custom Business Days ===")
print("Custom business days (excluding holidays):")
print(custom_dates)
```

## Part 5: Real-world Example

### Stock Market Data Simulation

```python
# Simulate stock market data
print("=== Stock Market Data Simulation ===")

# Create trading days (business days only)
trading_days = pd.bdate_range('2023-01-01', '2023-12-31')
n_days = len(trading_days)

# Generate stock price data (random walk)
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
prices = 100 * np.exp(np.cumsum(returns))  # Stock prices

# Create time series
stock_data = pd.Series(prices, index=trading_days)
print(f"Stock data shape: {stock_data.shape}")
print(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
print(f"Price range: ${stock_data.min():.2f} to ${stock_data.max():.2f}")

# Display sample data
print("\nSample stock data:")
print(stock_data.head())
print(stock_data.tail())
```

### Time Series Analysis

```python
# Basic time series analysis
print("=== Time Series Analysis ===")

# Calculate daily returns
daily_returns = stock_data.pct_change()
print(f"Daily returns mean: {daily_returns.mean():.4f}")
print(f"Daily returns std: {daily_returns.std():.4f}")

# Calculate moving averages
stock_data['MA_5'] = stock_data.rolling(window=5).mean()
stock_data['MA_20'] = stock_data.rolling(window=20).mean()

# Calculate volatility (rolling standard deviation)
stock_data['Volatility'] = daily_returns.rolling(window=20).std()

print("\nStock data with indicators:")
print(stock_data.tail())
```

### Visualization

```python
# Create time series plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Price and moving averages
axes[0].plot(stock_data.index, stock_data.values, label='Stock Price', linewidth=1)
axes[0].plot(stock_data.index, stock_data['MA_5'], label='5-day MA', alpha=0.7)
axes[0].plot(stock_data.index, stock_data['MA_20'], label='20-day MA', alpha=0.7)
axes[0].set_title('Stock Price with Moving Averages')
axes[0].set_ylabel('Price ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Volatility
axes[1].plot(stock_data.index, stock_data['Volatility'], label='20-day Volatility', color='red')
axes[1].set_title('Stock Price Volatility')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Volatility')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Key Takeaways

1. **datetime Module**: Basic date/time operations and arithmetic
2. **DatetimeIndex**: pandas time series indexing and manipulation
3. **Date Ranges**: Generate sequences of dates with different frequencies
4. **Time Zones**: Handle timezone-aware data and conversions
5. **Business Days**: Work with business day calculations and custom rules
6. **Real-world Application**: Apply datetime skills to financial data analysis

## Next Steps

- Practice with your own time series data
- Learn about resampling and frequency conversion
- Explore rolling window operations
- Set up automated time series analysis
