#!/usr/bin/env python3
"""
Time Series Analysis Demo - DataSci 217 Lecture 11
Hands-on examples covering McKinney Chapter 11 concepts

This demo walks through essential time series operations using pandas,
from basic datetime handling to advanced resampling and moving window functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TIME SERIES ANALYSIS DEMO - LECTURE 11")
print("=" * 60)

# ==============================================================================
# 1. DATE AND TIME BASICS
# ==============================================================================
print("\n1. DATE AND TIME FUNDAMENTALS")
print("-" * 40)

# Python datetime basics
now = datetime.now()
print(f"Current datetime: {now}")
print(f"Year: {now.year}, Month: {now.month}, Day: {now.day}")

# Time deltas
delta = datetime(2023, 12, 31) - datetime(2023, 1, 1)
print(f"Days in 2023: {delta.days}")

# String to datetime conversion
date_strings = ['2023-01-15', '2023-02-20', '2023-03-10']
dates = pd.to_datetime(date_strings)
print(f"Parsed dates: {dates}")

# ==============================================================================
# 2. CREATING TIME SERIES
# ==============================================================================
print("\n2. CREATING TIME SERIES")
print("-" * 40)

# Basic time series creation
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=10, freq='D')
values = np.random.randn(10).cumsum()
ts = pd.Series(values, index=dates, name='Random_Walk')
print("Basic time series:")
print(ts.head())
print(f"Index type: {type(ts.index)}")

# Time series with business days
business_dates = pd.date_range('2023-01-01', periods=20, freq='B')
business_ts = pd.Series(np.random.randn(20), index=business_dates)
print(f"\nBusiness days series (first 5):")
print(business_ts.head())

# ==============================================================================
# 3. TIME SERIES INDEXING AND SELECTION
# ==============================================================================
print("\n3. TIME SERIES INDEXING AND SELECTION")
print("-" * 40)

# Create longer series for slicing examples
long_dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
long_ts = pd.Series(np.random.randn(len(long_dates)).cumsum(), 
                   index=long_dates, name='Stock_Price')

# Year-based selection
year_2022 = long_ts['2022']
print(f"2022 data points: {len(year_2022)}")

# Month-based selection
jan_2022 = long_ts['2022-01']
print(f"January 2022 data points: {len(jan_2022)}")

# Date range slicing
covid_period = long_ts['2020-03-01':'2020-06-30']
print(f"COVID period (Mar-Jun 2020): {len(covid_period)} points")
print(f"Start: {covid_period.index[0]}, End: {covid_period.index[-1]}")

# ==============================================================================
# 4. FREQUENCY CODES AND DATE RANGES
# ==============================================================================
print("\n4. FREQUENCY CODES AND DATE RANGES")
print("-" * 40)

# Different frequency examples
frequencies = {
    'D': 'Daily',
    'B': 'Business Day',
    'W': 'Weekly',
    'M': 'Month End',
    'MS': 'Month Start',
    'Q': 'Quarter End',
    'A': 'Year End',
    'H': 'Hourly',
    '4H': 'Every 4 Hours',
    '30min': 'Every 30 Minutes'
}

print("Date range examples with different frequencies:")
for freq_code, description in frequencies.items():
    try:
        range_example = pd.date_range('2023-01-01', periods=5, freq=freq_code)
        print(f"{freq_code:6} ({description:15}): {range_example[0]} to {range_example[-1]}")
    except:
        continue

# Custom frequency - third Friday of each month
third_fridays = pd.date_range('2023-01-01', '2023-12-31', freq='WOM-3FRI')
print(f"\nThird Fridays in 2023: {len(third_fridays)} dates")
print(f"First few: {third_fridays[:3].tolist()}")

# ==============================================================================
# 5. SHIFTING AND LAG OPERATIONS
# ==============================================================================
print("\n5. SHIFTING AND LAG OPERATIONS")
print("-" * 40)

# Create sample monthly data
monthly_dates = pd.date_range('2023-01-01', periods=12, freq='MS')
monthly_values = [100, 102, 98, 105, 110, 108, 115, 112, 118, 120, 116, 125]
monthly_ts = pd.Series(monthly_values, index=monthly_dates, name='Monthly_Value')

print("Original monthly data:")
print(monthly_ts.head())

# Shift data by periods
shifted_data = monthly_ts.shift(1)
print("\nShifted by 1 period (data moves, dates stay):")
print(pd.DataFrame({'Original': monthly_ts, 'Shifted': shifted_data}).head())

# Shift timestamps
shifted_time = monthly_ts.shift(1, freq='M')
print("\nShifted timestamps by 1 month:")
print(f"Original first date: {monthly_ts.index[0]}")
print(f"Shifted first date: {shifted_time.index[0]}")

# Calculate returns using shift
returns = (monthly_ts / monthly_ts.shift(1) - 1) * 100
print("\nMonthly returns (%):")
print(returns.dropna().round(2))

# ==============================================================================
# 6. TIME ZONE HANDLING
# ==============================================================================
print("\n6. TIME ZONE HANDLING")
print("-" * 40)

# Create timezone-naive series
naive_dates = pd.date_range('2023-06-15 09:00', periods=5, freq='H')
naive_ts = pd.Series(range(5), index=naive_dates)
print("Naive (no timezone) series:")
print(naive_ts)

# Localize to specific timezone
eastern_ts = naive_ts.tz_localize('US/Eastern')
print("\nLocalized to US/Eastern:")
print(eastern_ts)

# Convert to different timezone
tokyo_ts = eastern_ts.tz_convert('Asia/Tokyo')
print("\nConverted to Asia/Tokyo:")
print(tokyo_ts)

# Create timezone-aware date range
utc_range = pd.date_range('2023-06-15 12:00', periods=3, freq='6H', tz='UTC')
print(f"\nUTC timezone-aware range: {utc_range}")

# ==============================================================================
# 7. RESAMPLING OPERATIONS
# ==============================================================================
print("\n7. RESAMPLING OPERATIONS")
print("-" * 40)

# Create high-frequency data for resampling
high_freq_dates = pd.date_range('2023-06-01', '2023-06-30', freq='H')
high_freq_data = np.random.randn(len(high_freq_dates)).cumsum()
high_freq_ts = pd.Series(high_freq_data, index=high_freq_dates)

print(f"High frequency data: {len(high_freq_ts)} hourly observations")

# Downsample to daily
daily_mean = high_freq_ts.resample('D').mean()
daily_max = high_freq_ts.resample('D').max()
daily_min = high_freq_ts.resample('D').min()
daily_std = high_freq_ts.resample('D').std()

print(f"Downsampled to daily: {len(daily_mean)} observations")
print("\nDaily aggregations (first 5 days):")
daily_summary = pd.DataFrame({
    'Mean': daily_mean,
    'Max': daily_max,
    'Min': daily_min,
    'Std': daily_std
}).round(3)
print(daily_summary.head())

# OHLC resampling (financial data style)
ohlc_daily = high_freq_ts.resample('D').ohlc()
print("\nOHLC (Open-High-Low-Close) daily summary:")
print(ohlc_daily.head().round(3))

# Upsampling example
weekly_data = daily_mean.resample('W').mean()
upsampled = weekly_data.resample('D').ffill()  # Forward fill
print(f"\nWeekly data: {len(weekly_data)} points")
print(f"Upsampled back to daily: {len(upsampled)} points")

# ==============================================================================
# 8. MOVING WINDOW FUNCTIONS
# ==============================================================================
print("\n8. MOVING WINDOW FUNCTIONS")
print("-" * 40)

# Create sample stock price data
stock_dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
stock_prices = 100 * np.exp(np.random.randn(len(stock_dates)).cumsum() * 0.01)
stock_ts = pd.Series(stock_prices, index=stock_dates, name='Stock_Price')

print(f"Stock price series: {len(stock_ts)} business days")
print(f"Price range: ${stock_ts.min():.2f} - ${stock_ts.max():.2f}")

# Moving averages
sma_10 = stock_ts.rolling(10).mean()  # 10-day simple moving average
sma_30 = stock_ts.rolling(30).mean()  # 30-day simple moving average

# Exponentially weighted moving average
ema_10 = stock_ts.ewm(span=10).mean()

print("\nMoving averages (last 5 values):")
moving_avg_df = pd.DataFrame({
    'Price': stock_ts,
    'SMA_10': sma_10,
    'SMA_30': sma_30,
    'EMA_10': ema_10
}).round(2)
print(moving_avg_df.tail())

# Rolling volatility (standard deviation)
returns = stock_ts.pct_change()
rolling_vol = returns.rolling(30).std() * np.sqrt(252)  # Annualized volatility
print(f"\nRolling 30-day volatility (annualized):")
print(f"Min: {rolling_vol.min():.3f}, Max: {rolling_vol.max():.3f}")

# Rolling correlations
market_index = pd.Series(np.random.randn(len(stock_dates)).cumsum() * 0.008, 
                        index=stock_dates, name='Market_Index')
market_returns = market_index.pct_change()
stock_returns = stock_ts.pct_change()

rolling_corr = stock_returns.rolling(60).corr(market_returns)
print(f"\n60-day rolling correlation with market:")
print(f"Mean correlation: {rolling_corr.mean():.3f}")

# ==============================================================================
# 9. PRACTICAL TIME SERIES ANALYSIS
# ==============================================================================
print("\n9. PRACTICAL TIME SERIES ANALYSIS")
print("-" * 40)

# Create realistic financial time series
np.random.seed(123)
analysis_dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')
trend = np.linspace(0, 0.5, len(analysis_dates))  # Upward trend
noise = np.random.randn(len(analysis_dates)) * 0.02
seasonal = 0.1 * np.sin(2 * np.pi * np.arange(len(analysis_dates)) / 252)  # Annual cycle
price_series = 100 * np.exp(trend + seasonal + noise.cumsum())

financial_ts = pd.Series(price_series, index=analysis_dates, name='Asset_Price')

print(f"Financial time series analysis:")
print(f"Period: {financial_ts.index[0]} to {financial_ts.index[-1]}")
print(f"Total return: {(financial_ts.iloc[-1] / financial_ts.iloc[0] - 1) * 100:.1f}%")

# Key statistics
daily_returns = financial_ts.pct_change().dropna()
print(f"\nDaily returns statistics:")
print(f"Mean: {daily_returns.mean()*100:.3f}% per day")
print(f"Std: {daily_returns.std()*100:.3f}% per day")
print(f"Annualized Sharpe ratio: {daily_returns.mean() / daily_returns.std() * np.sqrt(252):.2f}")

# Monthly and annual aggregations
monthly_returns = financial_ts.resample('M').last().pct_change().dropna()
annual_returns = financial_ts.resample('A').last().pct_change().dropna()

print(f"\nMonthly returns: Mean {monthly_returns.mean()*100:.2f}%, Std {monthly_returns.std()*100:.2f}%")
print(f"Annual returns: {[f'{r*100:.1f}%' for r in annual_returns]}")

# Identify significant moves
large_moves = daily_returns[abs(daily_returns) > daily_returns.std() * 2]
print(f"\nLarge moves (>2 std dev): {len(large_moves)} days")
if len(large_moves) > 0:
    print(f"Largest daily gain: {large_moves.max()*100:.2f}%")
    print(f"Largest daily loss: {large_moves.min()*100:.2f}%")

# ==============================================================================
# 10. PERIOD-BASED ANALYSIS
# ==============================================================================
print("\n10. PERIOD-BASED ANALYSIS")
print("-" * 40)

# Create quarterly business data
quarterly_periods = pd.period_range('2020Q1', '2023Q4', freq='Q')
quarterly_revenue = pd.Series([100, 110, 105, 120, 125, 135, 130, 145, 
                              150, 165, 160, 175, 180, 195, 190, 205], 
                             index=quarterly_periods, name='Revenue')

print("Quarterly revenue data:")
print(quarterly_revenue)

# Period arithmetic and conversion
print(f"\nPeriod operations:")
print(f"Q1 2023: {quarterly_periods[12]}")
print(f"Next quarter: {quarterly_periods[12] + 1}")
print(f"Year over year: {quarterly_periods[12] - 4}")

# Convert periods to different frequencies
annual_revenue = quarterly_revenue.groupby(quarterly_revenue.index.year).sum()
print(f"\nAnnual revenue totals:")
for year, revenue in annual_revenue.items():
    print(f"{year}: ${revenue}M")

# Growth calculations
qoq_growth = quarterly_revenue.pct_change() * 100
yoy_growth = quarterly_revenue.pct_change(4) * 100  # 4 quarters = 1 year

print(f"\nQuarter-over-quarter growth (last 4 quarters):")
print(qoq_growth.tail(4).round(1))

print(f"\nYear-over-year growth (last 4 quarters):")
print(yoy_growth.tail(4).round(1))

# ==============================================================================
# SUMMARY AND NEXT STEPS
# ==============================================================================
print("\n" + "=" * 60)
print("TIME SERIES ANALYSIS DEMO COMPLETE")
print("=" * 60)

print("\nKey concepts covered:")
print("✓ Date/time data types and conversion")
print("✓ Time series creation and indexing")
print("✓ Frequency codes and date ranges")
print("✓ Shifting and lag operations")
print("✓ Time zone handling")
print("✓ Resampling and aggregation")
print("✓ Moving window functions")
print("✓ Practical financial analysis")
print("✓ Period-based operations")

print("\nNext steps for time series mastery:")
print("• Practice with real-world datasets (financial, weather, sensor data)")
print("• Learn seasonal decomposition and trend analysis")
print("• Explore time series forecasting methods (ARIMA, exponential smoothing)")
print("• Study irregular time series and missing data handling")
print("• Investigate multivariate time series analysis")

print("\nRemember: Time series analysis is fundamental to many data science")
print("applications. Master these pandas tools and you'll be well-equipped")
print("to handle temporal data in your projects!")