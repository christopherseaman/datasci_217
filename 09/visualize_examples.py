#!/usr/bin/env python3
"""
Companion visualization script for Lecture 09: Time Series Analysis
Generates visualizations for code examples in the lecture README.

Run this script to generate all example visualizations:
    python visualize_examples.py

Output images are saved to the media/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# Create media directory if it doesn't exist
os.makedirs('media', exist_ok=True)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("Generating time series visualizations...")

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Visualization 1: Types of Time Series
# ============================================================================
print("1. Creating 'Types of Time Series' visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Types of Time Series Data', fontsize=16, fontweight='bold')

# Regular time series (daily patient measurements)
dates = pd.date_range('2023-01-01', periods=30, freq='D')
regular = pd.Series(np.cumsum(np.random.randn(30)) + 100, index=dates)
axes[0, 0].plot(regular.index, regular.values, marker='o', markersize=4)
axes[0, 0].set_title('Regular (Daily Fixed Intervals)', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Temperature (°F)')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# Irregular time series (clinical visits)
irregular_dates = pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-07', 
                                   '2023-01-15', '2023-01-20', '2023-01-28'])
irregular = pd.Series([98.6, 99.2, 98.4, 99.0, 98.8, 98.5], index=irregular_dates)
axes[0, 1].plot(irregular.index, irregular.values, marker='o', markersize=8, linestyle='--')
axes[0, 1].set_title('Irregular (Event-Based)', fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Temperature (°F)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Seasonal time series (monthly flu cases)
dates = pd.date_range('2020-01-01', periods=36, freq='ME')
seasonal = 100 + 20 * np.sin(2 * np.pi * np.arange(36) / 12) + np.random.randn(36) * 5
axes[0, 2].plot(dates, seasonal, marker='o', markersize=4)
axes[0, 2].set_title('Seasonal (Repeating Patterns)', fontweight='bold')
axes[0, 2].set_xlabel('Date')
axes[0, 2].set_ylabel('Cases per Month')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].grid(True, alpha=0.3)

# Trending time series (long-term blood pressure)
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = 120 + np.linspace(0, 10, 365) + np.random.randn(365) * 2
axes[1, 0].plot(dates, trend, alpha=0.7, linewidth=1)
axes[1, 0].plot(dates, 120 + np.linspace(0, 10, 365), 'r--', linewidth=2, label='Trend')
axes[1, 0].set_title('Trending (Long-Term Direction)', fontweight='bold')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Blood Pressure (mmHg)')
axes[1, 0].legend()
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Stationary time series (lab control measurements)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
stationary = 100 + np.random.randn(100) * 2
axes[1, 1].plot(dates, stationary, marker='o', markersize=3, alpha=0.7)
axes[1, 1].axhline(y=100, color='r', linestyle='--', linewidth=2, label='Mean')
axes[1, 1].set_title('Stationary (Constant Properties)', fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Control Value')
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

# Combined (trend + seasonal + noise)
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend_component = np.linspace(100, 120, 365)
seasonal_component = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25)
noise = np.random.randn(365) * 3
combined = trend_component + seasonal_component + noise
axes[1, 2].plot(dates, combined, alpha=0.7, linewidth=1)
axes[1, 2].plot(dates, trend_component, 'r--', linewidth=2, label='Trend')
axes[1, 2].set_title('Combined (Trend + Seasonal + Noise)', fontweight='bold')
axes[1, 2].set_xlabel('Date')
axes[1, 2].set_ylabel('Patient Metric')
axes[1, 2].legend()
axes[1, 2].tick_params(axis='x', rotation=45)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('media/types_of_time_series.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: media/types_of_time_series.png")
plt.close()

# ============================================================================
# Visualization 2: Resampling Example
# ============================================================================
print("2. Creating 'Resampling Example' visualization...")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Resampling: Daily to Monthly Aggregation', fontsize=16, fontweight='bold')

# Daily data (high frequency)
dates_daily = pd.date_range('2023-01-01', periods=90, freq='D')
daily_values = 98.6 + np.cumsum(np.random.randn(90) * 0.1)
ts_daily = pd.Series(daily_values, index=dates_daily)

axes[0].plot(ts_daily.index, ts_daily.values, alpha=0.6, linewidth=1, marker='o', markersize=2)
axes[0].set_title('High Frequency (Daily) - 90 data points', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Temperature (°F)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3)

# Monthly resampled (low frequency)
ts_monthly = ts_daily.resample('ME').mean()

axes[1].plot(ts_monthly.index, ts_monthly.values, marker='o', markersize=8, 
             linewidth=2, color='red', label='Monthly Mean')
axes[1].set_title('Low Frequency (Monthly) - 3 data points', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Average Temperature (°F)')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('media/resampling_example.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: media/resampling_example.png")
plt.close()

# ============================================================================
# Visualization 3: Rolling Window Operations
# ============================================================================
print("3. Creating 'Rolling Window' visualization...")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Rolling Window Operations', fontsize=16, fontweight='bold')

# Create time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = 98.6 + np.cumsum(np.random.randn(100) * 0.1)
ts = pd.Series(values, index=dates)

# Original data
axes[0].plot(ts.index, ts.values, alpha=0.5, linewidth=1, label='Original Data', color='blue')
axes[0].plot(ts.index, ts.rolling(window=7).mean(), linewidth=2, 
             label='7-Day Rolling Mean', color='red')
axes[0].fill_between(ts.index, 
                     ts.rolling(window=7).mean() - ts.rolling(window=7).std(),
                     ts.rolling(window=7).mean() + ts.rolling(window=7).std(),
                     alpha=0.2, color='red', label='±1 Std Dev')
axes[0].set_title('Rolling Mean and Standard Deviation (Window=7)', fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Temperature (°F)')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3)

# Rolling window comparison
axes[1].plot(ts.index, ts.values, alpha=0.3, linewidth=1, label='Original', color='gray')
axes[1].plot(ts.index, ts.rolling(window=7).mean(), linewidth=2, 
             label='7-Day Window', color='blue')
axes[1].plot(ts.index, ts.rolling(window=14).mean(), linewidth=2, 
             label='14-Day Window', color='green')
axes[1].plot(ts.index, ts.rolling(window=30).mean(), linewidth=2, 
             label='30-Day Window', color='red')
axes[1].set_title('Rolling Mean with Different Window Sizes', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Temperature (°F)')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('media/rolling_window.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: media/rolling_window.png")
plt.close()

# ============================================================================
# Visualization 4: Exponentially Weighted Moving Average
# ============================================================================
print("4. Creating 'EWM Comparison' visualization...")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Exponentially Weighted vs Simple Moving Average', fontsize=16, fontweight='bold')

dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(98.6 + np.cumsum(np.random.randn(100) * 0.1), index=dates)

# Comparison: EWM vs Rolling
axes[0].plot(ts.index, ts.values, alpha=0.4, linewidth=1, label='Original', color='gray')
ma_30 = ts.rolling(window=30, min_periods=20).mean()
ewm_30 = ts.ewm(span=30).mean()
axes[0].plot(ma_30.index, ma_30.values, linewidth=2, 
             label='30-Day Simple MA', color='blue')
axes[0].plot(ewm_30.index, ewm_30.values, linewidth=2, 
             label='30-Day EWM (span=30)', color='red', linestyle='--')
axes[0].set_title('EWM vs Simple Moving Average', fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Temperature (°F)')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3)

# Different EWM spans
axes[1].plot(ts.index, ts.values, alpha=0.3, linewidth=1, label='Original', color='gray')
axes[1].plot(ts.index, ts.ewm(span=7).mean(), linewidth=2, label='EWM span=7', color='blue')
axes[1].plot(ts.index, ts.ewm(span=14).mean(), linewidth=2, label='EWM span=14', color='green')
axes[1].plot(ts.index, ts.ewm(span=30).mean(), linewidth=2, label='EWM span=30', color='red')
axes[1].set_title('Exponentially Weighted Moving Average - Different Spans', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Temperature (°F)')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('media/ewm_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: media/ewm_comparison.png")
plt.close()

# ============================================================================
# Visualization 5: Time Series Indexing
# ============================================================================
print("5. Creating 'Time Series Indexing' visualization...")

fig, axes = plt.subplots(3, 1, figsize=(12, 12))
fig.suptitle('Time Series Indexing and Selection', fontsize=16, fontweight='bold')

dates = pd.date_range('2023-01-01', periods=365, freq='D')
ts = pd.Series(100 + np.cumsum(np.random.randn(365) * 0.5), index=dates)

# Full year
axes[0].plot(ts.index, ts.values, alpha=0.7, linewidth=1)
axes[0].set_title('Full Year (2023)', fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Value')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3)

# January selection
jan_data = ts['2023-01']
axes[1].plot(jan_data.index, jan_data.values, marker='o', markersize=4, linewidth=2, color='red')
axes[1].set_title('January 2023 Selection', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Value')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

# Date range selection
range_data = ts['2023-06-01':'2023-06-30']
axes[2].plot(range_data.index, range_data.values, marker='o', markersize=4, 
              linewidth=2, color='green')
axes[2].set_title('June 2023 Date Range Selection', fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Value')
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('media/time_series_indexing.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: media/time_series_indexing.png")
plt.close()

# ============================================================================
# Visualization 6: Shifting and Lagging
# ============================================================================
print("6. Creating 'Shifting and Lagging' visualization...")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Shifting and Lagging Operations', fontsize=16, fontweight='bold')

dates = pd.date_range('2023-01-01', periods=30, freq='D')
ts = pd.Series(70 + np.cumsum(np.random.randn(30) * 0.2), index=dates)

# Shifting
axes[0].plot(ts.index, ts.values, linewidth=2, label='Original', color='blue')
axes[0].plot(ts.index, ts.shift(1), linewidth=2, label='Lag 1 (shift forward)', 
             color='red', linestyle='--', alpha=0.7)
axes[0].plot(ts.index, ts.shift(-1), linewidth=2, label='Lead 1 (shift backward)', 
             color='green', linestyle='--', alpha=0.7)
axes[0].set_title('Shifting Operations', fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Patient Weight (kg)')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3)

# Difference and percentage change
ax2_twin = axes[1].twinx()
axes[1].plot(ts.index, ts.diff(), marker='o', markersize=4, linewidth=2, 
             label='First Difference', color='blue')
ax2_twin.plot(ts.index, ts.pct_change() * 100, marker='s', markersize=3, 
               linewidth=1, label='Percentage Change (%)', color='red', alpha=0.7)
axes[1].set_title('First Difference and Percentage Change', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Difference', color='blue')
ax2_twin.set_ylabel('Percentage Change (%)', color='red')
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', labelcolor='blue')
ax2_twin.tick_params(axis='y', labelcolor='red')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper left')
ax2_twin.legend(loc='upper right')

plt.tight_layout()
plt.savefig('media/shifting_lagging.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: media/shifting_lagging.png")
plt.close()

print("\n✓ All visualizations generated successfully!")
print("  Images saved in the 'media/' directory")
print("\nTo use these in your lecture, reference them in README.md like:")
print("  ![Types of Time Series](media/types_of_time_series.png)")

