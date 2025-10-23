# Demo 2: Time Series Indexing and Resampling

## Learning Objectives
- Master time series indexing and selection
- Use resampling for frequency conversion
- Apply rolling window operations
- Handle missing data in time series

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
```

## Part 1: Time Series Indexing

### Create Sample Time Series

```python
# Create sample time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365)) + 100
ts = pd.Series(values, index=dates)

print("=== Time Series Data ===")
print(f"Data shape: {ts.shape}")
print(f"Date range: {ts.index.min()} to {ts.index.max()}")
print(f"Value range: {ts.min():.2f} to {ts.max():.2f}")
print("\nSample data:")
print(ts.head())
```

### Basic Time Series Selection

```python
# Select specific date
print("=== Basic Time Series Selection ===")
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

### Advanced Time Series Selection

```python
# Create hourly time series for advanced selection
hourly_dates = pd.date_range('2023-01-01', periods=24*7, freq='H')
hourly_values = np.random.randn(24*7)
ts_hourly = pd.Series(hourly_values, index=hourly_dates)

print("=== Advanced Time Series Selection ===")
print("Hourly data shape:", ts_hourly.shape)

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

## Part 2: Resampling and Frequency Conversion

### Basic Resampling

```python
# Create daily time series
daily_dates = pd.date_range('2023-01-01', periods=30, freq='D')
daily_values = np.cumsum(np.random.randn(30)) + 100
ts_daily = pd.Series(daily_values, index=daily_dates)

print("=== Basic Resampling ===")
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

### Resampling with Different Aggregations

```python
# Create sample data with multiple columns
df = pd.DataFrame({
    'value': np.random.randn(365),
    'volume': np.random.randint(100, 1000, 365)
}, index=pd.date_range('2023-01-01', periods=365, freq='D'))

print("=== Resampling with Different Aggregations ===")
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

### Frequency Conversion

```python
# Convert between frequencies
print("=== Frequency Conversion ===")

# Daily to business days
daily_ts = pd.Series(np.random.randn(31), index=pd.date_range('2023-01-01', periods=31, freq='D'))
business_ts = daily_ts.asfreq('B')
print(f"Daily to business days: {daily_ts.shape} -> {business_ts.shape}")

# Daily to weekly
weekly_ts = daily_ts.asfreq('W')
print(f"Daily to weekly: {daily_ts.shape} -> {weekly_ts.shape}")

# Handle missing values
print("\nHandling missing values:")
print("Original data:")
print(daily_ts.head(10))
print("\nBusiness days (with NaN for weekends):")
print(business_ts.head(10))
```

## Part 3: Rolling Window Operations

### Basic Rolling Operations

```python
# Create sample time series
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100)) + 100
ts = pd.Series(values, index=dates)

print("=== Basic Rolling Operations ===")
print("Original time series:")
print(ts.head())

# Rolling statistics
ts['rolling_mean'] = ts.rolling(window=7).mean()
ts['rolling_std'] = ts.rolling(window=7).std()
ts['rolling_min'] = ts.rolling(window=7).min()
ts['rolling_max'] = ts.rolling(window=7).max()

print("\nTime series with rolling statistics:")
print(ts[['value', 'rolling_mean', 'rolling_std']].head(10))
```

### Advanced Rolling Operations

```python
# Advanced rolling operations
print("=== Advanced Rolling Operations ===")

# Centered rolling window
ts['centered_mean'] = ts.rolling(window=7, center=True).mean()

# Expanding window
ts['expanding_mean'] = ts.expanding().mean()

# Exponentially weighted moving average
ts['ewm_mean'] = ts.ewm(span=7).mean()

# Custom rolling function
def rolling_range(series):
    return series.max() - series.min()

ts['rolling_range'] = ts.rolling(window=7).apply(rolling_range)

print("Advanced rolling statistics:")
print(ts[['value', 'centered_mean', 'expanding_mean', 'ewm_mean']].head(10))
```

### Rolling Window Visualization

```python
# Visualize rolling windows
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Original data with rolling mean
axes[0].plot(ts.index, ts['value'], label='Original', alpha=0.7)
axes[0].plot(ts.index, ts['rolling_mean'], label='7-day Rolling Mean', linewidth=2)
axes[0].plot(ts.index, ts['ewm_mean'], label='7-day EWM', linewidth=2)
axes[0].set_title('Time Series with Rolling Averages')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Rolling standard deviation
axes[1].plot(ts.index, ts['rolling_std'], label='7-day Rolling Std', color='red')
axes[1].set_title('Rolling Standard Deviation')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Standard Deviation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Part 4: Missing Data Handling

### Create Data with Missing Values

```python
# Create time series with missing values
print("=== Missing Data Handling ===")

# Create regular time series
regular_ts = pd.Series(np.random.randn(30), index=pd.date_range('2023-01-01', periods=30, freq='D'))

# Introduce missing values
missing_ts = regular_ts.copy()
missing_ts.iloc[5:8] = np.nan  # Missing values for 3 days
missing_ts.iloc[15:17] = np.nan  # Missing values for 2 days

print("Time series with missing values:")
print(missing_ts.head(20))
print(f"Missing values: {missing_ts.isnull().sum()}")
```

### Missing Data Imputation

```python
# Different imputation methods
print("=== Missing Data Imputation ===")

# Forward fill
forward_fill = missing_ts.fillna(method='ffill')
print("Forward fill:")
print(forward_fill.head(20))

# Backward fill
backward_fill = missing_ts.fillna(method='bfill')
print("\nBackward fill:")
print(backward_fill.head(20))

# Interpolation
interpolated = missing_ts.interpolate()
print("\nInterpolated:")
print(interpolated.head(20))

# Rolling mean imputation
rolling_mean = missing_ts.fillna(missing_ts.rolling(window=5, min_periods=1).mean())
print("\nRolling mean imputation:")
print(rolling_mean.head(20))
```

## Part 5: Real-world Example

### Stock Market Analysis

```python
# Create realistic stock market data
print("=== Stock Market Analysis ===")

# Generate stock price data
np.random.seed(42)
n_days = 252  # Trading days in a year
dates = pd.bdate_range('2023-01-01', periods=n_days)
returns = np.random.normal(0.001, 0.02, n_days)
prices = 100 * np.exp(np.cumsum(returns))

# Create stock data
stock_data = pd.Series(prices, index=dates)
stock_data.name = 'Stock Price'

print(f"Stock data shape: {stock_data.shape}")
print(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
print(f"Price range: ${stock_data.min():.2f} to ${stock_data.max():.2f}")
```

### Technical Analysis

```python
# Technical analysis indicators
print("=== Technical Analysis ===")

# Moving averages
stock_data['MA_5'] = stock_data.rolling(window=5).mean()
stock_data['MA_20'] = stock_data.rolling(window=20).mean()
stock_data['MA_50'] = stock_data.rolling(window=50).mean()

# Volatility (rolling standard deviation of returns)
returns = stock_data.pct_change()
stock_data['Volatility'] = returns.rolling(window=20).std()

# Bollinger Bands
stock_data['BB_Upper'] = stock_data['MA_20'] + 2 * stock_data['Volatility']
stock_data['BB_Lower'] = stock_data['MA_20'] - 2 * stock_data['Volatility']

print("Stock data with technical indicators:")
print(stock_data.tail())
```

### Monthly Analysis

```python
# Monthly analysis
print("=== Monthly Analysis ===")

# Monthly returns
monthly_returns = stock_data.resample('M').last().pct_change()
print("Monthly returns:")
print(monthly_returns.head())

# Monthly volatility
monthly_volatility = returns.resample('M').std()
print("\nMonthly volatility:")
print(monthly_volatility.head())

# Monthly statistics
monthly_stats = stock_data.resample('M').agg({
    'Stock Price': ['mean', 'std', 'min', 'max']
})
print("\nMonthly statistics:")
print(monthly_stats.head())
```

## Key Takeaways

1. **Time Series Indexing**: Select data by date ranges, specific dates, and time periods
2. **Resampling**: Convert between different time frequencies
3. **Rolling Windows**: Calculate moving statistics and trends
4. **Missing Data**: Handle gaps in time series data
5. **Real-world Application**: Apply techniques to financial data analysis
6. **Technical Analysis**: Create indicators for time series analysis

## Next Steps

- Practice with your own time series data
- Learn about seasonal decomposition
- Explore time series visualization techniques
- Set up automated time series analysis
