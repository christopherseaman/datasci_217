---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Assignment 9: Time Series Analysis

## Overview
This assignment covers time series analysis using real-world healthcare data including patient monitoring, clinical trials, and disease surveillance data. You'll practice datetime handling, resampling, rolling windows, and time series visualization.

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('default')
sns.set_style('whitegrid')

# Create output directory
os.makedirs('output', exist_ok=True)
```

## Question 1: datetime Fundamentals and Time Series Indexing

### Part 1.1: Load and Explore Data

```python
# Load the healthcare datasets
patient_vitals = pd.read_csv('data/patient_vitals.csv')
icu_monitoring = pd.read_csv('data/icu_monitoring.csv')
disease_surveillance = pd.read_csv('data/disease_surveillance.csv')

print("Patient vitals shape:", patient_vitals.shape)
print("ICU monitoring shape:", icu_monitoring.shape)
print("Disease surveillance shape:", disease_surveillance.shape)

print("\nPatient vitals columns:", patient_vitals.columns.tolist())
print("ICU monitoring columns:", icu_monitoring.columns.tolist())
print("Disease surveillance columns:", disease_surveillance.columns.tolist())

# Display sample data
print("\nPatient vitals sample:")
print(patient_vitals.head())
```

### Part 1.2: datetime Operations

**TODO: Perform datetime operations**

```python
# TODO: Convert date columns to datetime
# patient_vitals['date'] = pd.to_datetime(patient_vitals['date'])
# icu_monitoring['datetime'] = pd.to_datetime(icu_monitoring['datetime'])
# disease_surveillance['date'] = pd.to_datetime(disease_surveillance['date'])

# TODO: Set datetime columns as index
# patient_vitals = patient_vitals.set_index('date')
# icu_monitoring = icu_monitoring.set_index('datetime')
# disease_surveillance = disease_surveillance.set_index('date')

# TODO: Extract year, month, day, hour components from datetime index
# patient_vitals['year'] = None  # Extract from index
# patient_vitals['month'] = None  # Extract from index
# patient_vitals['day'] = None  # Extract from index

# TODO: Calculate time differences (e.g., days between measurements)
# patient_vitals['days_since_start'] = None  # Calculate from start date

# TODO: Create business day ranges for clinic visit schedules
# clinic_dates = None  # Use pd.bdate_range() for clinic visits

# TODO: Create date ranges with different frequencies
# daily_range = None  # Daily monitoring schedule
# weekly_range = None  # Weekly lab test schedule (Mondays)
# monthly_range = None  # Monthly checkup schedule

# TODO: Save results as 'output/q1_datetime_analysis.csv'
# datetime_analysis = None  # Combine your datetime operations results
# datetime_analysis.to_csv('output/q1_datetime_analysis.csv')
```

### Part 1.3: Time Zone Handling

**TODO: Handle time zones**

```python
# TODO: Create timezone-aware datetime (for multi-site clinical trials)
# utc_time = None  # Current time in UTC
# eastern_time = None  # Convert to US Eastern

# TODO: Convert between different timezones
# Create timezone-aware DataFrame from patient_vitals
# patient_vitals_tz = None  # Localize to UTC
# patient_vitals_tz_eastern = None  # Convert to Eastern time

# TODO: Handle daylight saving time transitions
# Create datetime that spans DST transition
# dst_date = None  # Date around DST transition
# dst_time_utc = None  # Localize and convert

# TODO: Document timezone operations
# Create a report string
timezone_report = """
TODO: Document your timezone operations:
- What timezone was your original data in?
- How did you localize the data?
- What timezone did you convert to?
- What issues did you encounter with DST?
"""

# TODO: Save results as 'output/q1_timezone_report.txt'
# with open('output/q1_timezone_report.txt', 'w') as f:
#     f.write(timezone_report)
```

## Question 2: Time Series Indexing and Resampling

### Part 2.1: Time Series Selection

**TODO: Perform time series indexing and selection**

```python
# Ensure patient_vitals has datetime index (from Q1)
# patient_vitals = patient_vitals.set_index('date')

# TODO: Select data by specific dates
# january_first = None  # Select January 1, 2023
# print("January 1, 2023 data:", january_first)

# TODO: Select data by date ranges
# january_data = None  # Select entire January 2023
# print("January 2023 shape:", january_data.shape)

# TODO: Select data by time periods
# first_quarter = None  # Select Q1 2023
# entire_year = None  # Select all of 2023

# TODO: Use first() and last() methods
# first_week = None  # First 7 days
# last_week = None  # Last 7 days

# TODO: Use truncate() method
# data_after_june = None  # Truncate before June 1, 2023
# data_before_september = None  # Truncate after August 31, 2023

# For ICU data with time components:
# TODO: Select business hours (9 AM to 5 PM)
# business_hours = None  # Use between_time()
# print("Business hours data shape:", business_hours.shape)

# TODO: Select specific time (noon readings)
# noon_data = None  # Use at_time('12:00')

# TODO: Demonstrate different selection methods
# selection_results = None  # Combine your selection results

# TODO: Save results as 'output/q2_resampling_analysis.csv'
# selection_results.to_csv('output/q2_resampling_analysis.csv')
```

### Part 2.2: Resampling Operations

**TODO: Perform resampling and frequency conversion**

```python
# Use patient_vitals (daily) and icu_monitoring (hourly) from Q1

# TODO: Resample daily data to weekly
# patient_vitals_weekly = None  # Resample to weekly with mean aggregation
# print("Weekly resampled shape:", patient_vitals_weekly.shape)

# TODO: Resample daily data to monthly
# patient_vitals_monthly = None  # Resample to monthly with mean aggregation
# print("Monthly resampled shape:", patient_vitals_monthly.shape)

# TODO: Resample hourly data to daily
# icu_daily = None  # Resample ICU hourly data to daily
# print("ICU daily shape:", icu_daily.shape)

# TODO: Use different aggregation functions (mean, sum, max, min)
# icu_daily_stats = None  # Resample with multiple aggregations
# Example: resample('D').agg({'heart_rate': ['mean', 'max', 'min'], 
#                             'temperature': 'mean'})

# TODO: Handle missing values during resampling
# Demonstrate upsampling (monthly to daily) creates missing values
# monthly_to_daily = None  # Upsample monthly data to daily
# print("Missing values after upsampling:", monthly_to_daily.isna().sum())

# TODO: Compare different resampling frequencies
# Compare daily, weekly, monthly, quarterly resampling
# resampling_comparison = None  # DataFrame with different frequencies

# TODO: Save results as 'output/q2_resampling_analysis.csv'
# resampling_comparison.to_csv('output/q2_resampling_analysis.csv')
```

### Part 2.3: Missing Data Handling

**TODO: Handle missing data in time series**

```python
# TODO: Identify missing values in time series
# Create a time series with missing values (e.g., upsample monthly to daily)
# ts_with_missing = None  # Time series with missing values
# print("Missing value count:", ts_with_missing.isna().sum())
# print("Missing value percentage:", ts_with_missing.isna().sum() / len(ts_with_missing) * 100)

# TODO: Use forward fill and backward fill
# ts_ffill = None  # Forward fill missing values
# ts_bfill = None  # Backward fill missing values

# TODO: Use interpolation methods
# ts_interpolated = None  # Interpolate missing values
# ts_interpolated_linear = None  # Linear interpolation
# ts_interpolated_time = None  # Time-based interpolation

# TODO: Use rolling mean for imputation
# ts_rolling_imputed = None  # Fill missing with rolling mean

# TODO: Create missing data report
missing_data_report = """
TODO: Document your missing data handling:
- How many missing values did you find?
- What percentage of data was missing?
- Which method did you use to fill missing values?
- Why did you choose that method?
- What are the pros/cons of your approach?
"""

# TODO: Document missing data patterns
# missing_patterns = None  # Analyze when/why data is missing
# missing_by_month = None  # Missing values by month
# missing_by_day = None  # Missing values by day of week

# TODO: Save results as 'output/q2_missing_data_report.txt'
# with open('output/q2_missing_data_report.txt', 'w') as f:
#     f.write(missing_data_report)
#     f.write(f"\nMissing patterns:\n{missing_patterns}")
```

## Question 3: Rolling Window Operations and Visualization

### Part 3.1: Basic Rolling Operations

**TODO: Apply rolling window operations**

```python
# Use patient_vitals from previous questions (daily data with datetime index)

# TODO: Calculate 7-day rolling mean
# patient_vitals['rolling_7d_mean'] = None  # 7-day rolling mean
# patient_vitals['rolling_7d_std'] = None  # 7-day rolling standard deviation

# TODO: Calculate 30-day rolling statistics
# patient_vitals['rolling_30d_mean'] = None  # 30-day rolling mean
# patient_vitals['rolling_30d_min'] = None  # 30-day rolling minimum
# patient_vitals['rolling_30d_max'] = None  # 30-day rolling maximum

# TODO: Calculate rolling sum
# patient_vitals['rolling_7d_sum'] = None  # 7-day rolling sum

# TODO: Use different window sizes
# Compare 7-day, 14-day, and 30-day rolling windows
# patient_vitals['rolling_14d_mean'] = None  # 14-day rolling mean

# TODO: Create rolling statistics dataframe
# rolling_stats = None  # DataFrame with rolling statistics

# TODO: Save results as 'output/q3_rolling_analysis.csv'
# rolling_stats.to_csv('output/q3_rolling_analysis.csv')
```

### Part 3.2: Advanced Rolling Operations

**TODO: Apply advanced rolling operations**

```python
# TODO: Use centered rolling windows
# patient_vitals['rolling_7d_centered'] = None  # Centered 7-day window

# TODO: Use expanding windows
# patient_vitals['expanding_mean'] = None  # Expanding mean from start

# TODO: Calculate exponentially weighted moving averages
# patient_vitals['ewm_span_7'] = None  # EWM with span=7
# patient_vitals['ewm_span_30'] = None  # EWM with span=30
# patient_vitals['ewm_alpha_0.3'] = None  # EWM with alpha=0.3

# TODO: Create custom rolling function
# Example: rolling range (max - min)
def rolling_range(series):
    """Calculate rolling range (max - min)"""
    return None  # TODO: Implement

# patient_vitals['rolling_7d_range'] = None  # Apply custom function

# TODO: Handle minimum periods requirement
# patient_vitals['rolling_7d_min_periods'] = None  # Rolling with min_periods=3

# TODO: Compare different rolling methods
# rolling_comparison = None  # DataFrame comparing different methods

# TODO: Save results as 'output/q3_rolling_analysis.csv'
# rolling_comparison.to_csv('output/q3_rolling_analysis.csv')
```

### Part 3.3: Trend Analysis Visualization

**TODO: Create trend analysis visualization**

```python
# Use patient_vitals with rolling statistics from Part 3.1

# TODO: Create time series plot with original data
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# TODO: Plot original data (e.g., temperature or heart_rate)
# axes[0].plot(None, None, alpha=0.5, label='Daily', color='gray')  # Original data

# TODO: Add rolling mean overlay
# axes[0].plot(None, None, linewidth=2, label='7-Day Rolling Mean', color='blue')

# TODO: Add rolling standard deviation bands
# axes[0].fill_between(None, None, None, alpha=0.2, color='blue', 
#                     label='±1 Std Dev')  # Rolling mean ± std

# TODO: Add exponentially weighted moving average
# axes[0].plot(None, None, linewidth=2, label='7-Day EWM', color='red', linestyle='--')

# TODO: Customize colors and styling
# axes[0].set_title('Patient Temperature with Rolling Statistics', fontsize=14, fontweight='bold')
# axes[0].set_xlabel('Date')
# axes[0].set_ylabel('Temperature (°F)')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
# axes[0].tick_params(axis='x', rotation=45)

# TODO: Create second subplot showing rolling statistics comparison
# axes[1].plot(None, None, label='7-Day Rolling', color='blue')
# axes[1].plot(None, None, label='30-Day Rolling', color='green')
# axes[1].plot(None, None, label='EWM (span=7)', color='red', linestyle='--')
# axes[1].set_title('Rolling Window Comparison', fontsize=14, fontweight='bold')
# axes[1].set_xlabel('Date')
# axes[1].set_ylabel('Value')
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)
# axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()

# TODO: Save the plot as 'output/q3_trend_analysis.png'
# plt.savefig('output/q3_trend_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Part 3.4: Comprehensive Visualization (Bonus)

**TODO: Create comprehensive multi-variable time series visualization**

```python
# TODO: Create visualization with multiple variables
# Use patient_vitals DataFrame with multiple columns (temperature, heart_rate, etc.)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# TODO: Plot temperature with rolling mean
# TODO: Plot heart rate with rolling mean
# TODO: Plot weight with rolling mean

# Customize each subplot with titles, labels, legends, grids

plt.tight_layout()

# TODO: Save the plot as 'output/q3_visualization.png'
# plt.savefig('output/q3_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_datetime_analysis.csv` - datetime analysis
- [ ] `output/q1_timezone_report.txt` - timezone report
- [ ] `output/q2_resampling_analysis.csv` - resampling analysis
- [ ] `output/q2_missing_data_report.txt` - missing data report
- [ ] `output/q3_rolling_analysis.csv` - rolling analysis
- [ ] `output/q3_trend_analysis.png` - trend analysis plot
- [ ] `output/q3_visualization.png` - comprehensive visualization (bonus)

## Key Learning Objectives

- Master datetime handling and parsing with health data
- Perform time series indexing and selection
- Use resampling and frequency conversion for clinical data
- Apply rolling window operations for trend detection
- Handle time zones for multi-site clinical trials
- Create publication-quality time series visualizations
