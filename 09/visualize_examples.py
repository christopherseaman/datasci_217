#!/usr/bin/env python3
"""
Companion visualization script for Lecture 09: Time Series Analysis
Generates the BONUS figures (types_of_time_series, ewm_comparison) and the
lecture's two ILI snippet plots (viz_ili_rolling, viz_ili_monthly).
scripts/generate_lecture_figures.py generates every other lecture figure.

Run this script to generate all example visualizations:
    python3 visualize_examples.py

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
# Visualization 2: Exponentially Weighted Moving Average
# ============================================================================
print("2. Creating 'EWM Comparison' visualization...")

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
# Visualization 3: Time Series Plotting (lecture snippet figures)
# ============================================================================
print("3. Creating 'Influenza-like Illness Visits' plots...")

# One year of daily influenza-like-illness (ILI) visits with a winter peak
days = pd.date_range('2023-01-01', '2023-12-31', freq='D')
winter_wave = 20 * np.cos(2 * np.pi * (days.dayofyear - 15) / 365.25)
ts = pd.Series((40 + winter_wave + np.random.default_rng(9).normal(0, 4, len(days))).round(), index=days)

fig, ax = plt.subplots(figsize=(12, 6))
ts.plot(ax=ax, alpha=0.5, label='Daily', color='gray')
ts.rolling(window=30).mean().plot(ax=ax, linewidth=2, label='30-Day Rolling Mean', color='blue')
ax.set(title='Influenza-like Illness Visits with Rolling Mean', xlabel='Date', ylabel='Visits per day')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('media/viz_ili_rolling.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: media/viz_ili_rolling.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
ts.groupby(ts.index.month).mean().plot(kind='bar', ax=ax, rot=0, title='Mean Daily Visits by Calendar Month',
                                       xlabel='Month', ylabel='Visits per day')
plt.savefig('media/viz_ili_monthly.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: media/viz_ili_monthly.png")
plt.close()

print("\n✓ All visualizations generated successfully!")
print("  Images saved in the 'media/' directory")
print("\nTo use these in your lecture, reference them in README.md like:")
print("  ![Types of Time Series](media/types_of_time_series.png)")

