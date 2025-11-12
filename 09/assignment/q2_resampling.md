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

# Question 2: Resampling and Frequency Conversion

This question focuses on resampling operations and frequency conversion using ICU monitoring data (hourly) and patient vital signs data (daily).

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

## Part 2.1: Load and Prepare Data

**Note:** These datasets have realistic characteristics:
- **ICU Monitoring**: 75 patients with variable stay lengths (2-30 days). Not all patients are present for the entire 6-month period - patients are admitted and discharged at different times.
- **Patient Vitals**: Already contains some missing visits (~5% missing data). This is realistic and will be useful for practicing missing data handling.

```python
# Load ICU monitoring data (hourly)
icu_monitoring = pd.read_csv('data/icu_monitoring.csv')

# Load patient vitals data (daily) - for comparison
patient_vitals = pd.read_csv('data/patient_vitals.csv')

print("ICU monitoring shape:", icu_monitoring.shape)
print("Patient vitals shape:", patient_vitals.shape)

# Convert datetime columns and set as index
icu_monitoring['datetime'] = pd.to_datetime(icu_monitoring['datetime'])
icu_monitoring = icu_monitoring.set_index('datetime')

patient_vitals['date'] = pd.to_datetime(patient_vitals['date'])
patient_vitals = patient_vitals.set_index('date')

print("\nICU monitoring sample:")
print(icu_monitoring.head())
print("\nPatient vitals sample:")
print(patient_vitals.head())

# Check data characteristics
print(f"\nICU patients: {icu_monitoring['patient_id'].nunique()}")
print(f"ICU date range: {icu_monitoring.index.min()} to {icu_monitoring.index.max()}")
print(f"\nPatient vitals patients: {patient_vitals['patient_id'].nunique()}")
print(f"Patient vitals date range: {patient_vitals.index.min()} to {patient_vitals.index.max()}")
```

## Part 2.2: Time Series Selection

**TODO: Perform time series indexing and selection**

```python
# TODO: Select data by specific dates
# Note: Not all patients may have data on January 1, 2023 (some start later)
# january_first = None  # Select January 1, 2023 from patient_vitals
# print("January 1, 2023 data:", january_first)
# print(f"Records on Jan 1: {len(january_first)} (some patients may start later)")

# TODO: Select data by date ranges
# january_data = None  # Select entire January 2023
# print("January 2023 shape:", january_data.shape)

# TODO: Select data by time periods
# first_quarter = None  # Select Q1 2023
# entire_year = None  # Select all of 2023 (will include patients with partial year data)

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
```

## Part 2.3: Resampling Operations

**TODO: Perform resampling and frequency conversion**

```python
# TODO: Resample hourly ICU data to daily
# icu_daily = None  # Resample ICU hourly data to daily
# print("ICU daily shape:", icu_daily.shape)

# TODO: Resample daily patient data to weekly
# patient_vitals_weekly = None  # Resample to weekly with mean aggregation
# print("Weekly resampled shape:", patient_vitals_weekly.shape)

# TODO: Resample daily patient data to monthly
# patient_vitals_monthly = None  # Resample to monthly with mean aggregation
# print("Monthly resampled shape:", patient_vitals_monthly.shape)

# TODO: Use different aggregation functions (mean, sum, max, min)
# icu_daily_stats = None  # Resample with multiple aggregations
# Example: resample('D').agg({'heart_rate': ['mean', 'max', 'min'], 
#                             'temperature': 'mean'})

# TODO: Handle missing values during resampling
# Demonstrate upsampling (monthly to daily) creates missing values
# monthly_to_daily = None  # Upsample monthly data to daily
# print("Missing values after upsampling:", monthly_to_daily.isna().sum())

# TODO: Compare different resampling frequencies
# Create a DataFrame comparing resampling results at different frequencies
# Note: ICU data has variable patient stay lengths, so aggregate across all patients
# Include columns: frequency, date_range, row_count, mean_value, std_value
# Use the same metric (e.g., temperature or heart_rate) across all frequencies
# For ICU data, you may want to aggregate by patient first, then resample
# Example structure:
# resampling_comparison = pd.DataFrame({
#     'frequency': ['hourly', 'daily', 'weekly', 'monthly'],
#     'date_range': [str(icu_monitoring.index.min()) + ' to ' + str(icu_monitoring.index.max()), ...],
#     'row_count': [len(icu_monitoring), len(icu_daily), len(patient_vitals_weekly), len(patient_vitals_monthly)],
#     'mean_temperature': [icu_monitoring['temperature'].mean(), icu_daily['temperature'].mean(), ...],
#     'std_temperature': [icu_monitoring['temperature'].std(), icu_daily['temperature'].std(), ...]
# })

# TODO: Save results as 'output/q2_resampling_analysis.csv'
# resampling_comparison.to_csv('output/q2_resampling_analysis.csv', index=False)
```

## Part 2.4: Missing Data Handling

**Note:** The patient_vitals dataset already contains some naturally occurring missing data (~5% missing visits). You can also create additional missing data by upsampling (e.g., monthly to daily) to practice different imputation methods.

**TODO: Handle missing data in time series**

```python
# TODO: Identify missing values in time series
# Option 1: Use naturally occurring missing data from patient_vitals
# Option 2: Create missing values by upsampling (e.g., monthly to daily)
# For patient_vitals, you can aggregate by date first, then upsample to see missing patterns
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
# Document your missing data handling with the following sections:
# 1. Missing value summary: Total count and percentage
# 2. Missing data patterns: When/why data is missing (by month, day of week, etc.)
# 3. Imputation method: Which method you used (forward fill, backward fill, interpolation, rolling mean)
# 4. Rationale: Why you chose that method
# 5. Pros and cons: Advantages and limitations of your approach
# 6. Example: Show at least one example of missing data before and after imputation
# Minimum length: 300 words
missing_data_report = """
TODO: Document your missing data handling:
- How many missing values did you find?
- What percentage of data was missing?
- Which method did you use to fill missing values?
- Why did you choose that method?
- What are the pros/cons of your approach?
- Include examples showing missing data patterns
"""

# TODO: Document missing data patterns
# Analyze when/why data is missing
# missing_by_month = ts_with_missing.groupby(ts_with_missing.index.month).apply(lambda x: x.isna().sum())
# missing_by_day = ts_with_missing.groupby(ts_with_missing.index.dayofweek).apply(lambda x: x.isna().sum())
# missing_patterns = f"Missing by month:\n{missing_by_month}\n\nMissing by day of week:\n{missing_by_day}"

# TODO: Save results as 'output/q2_missing_data_report.txt'
# with open('output/q2_missing_data_report.txt', 'w') as f:
#     f.write(missing_data_report)
#     f.write(f"\n\nMissing patterns:\n{missing_patterns}")
```

## Submission Checklist

Before moving to Question 3, verify you've created:

- [ ] `output/q2_resampling_analysis.csv` - resampling analysis results
- [ ] `output/q2_missing_data_report.txt` - missing data handling report

