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

**Important Note:** Since multiple patients share the same date, the `patient_vitals` index is non-monotonic (not strictly increasing). For reliable date-based selection with `.loc`, you should sort the index first: `patient_vitals = patient_vitals.sort_index()`. This ensures pandas can properly handle date range selections.

```python
# TODO: Select data by specific dates
# Note: Not all patients may have data on January 1, 2023 (some start later)
# Important: Sort the index first since multiple patients share the same date
# patient_vitals = patient_vitals.sort_index()  # Sort for reliable date-based selection
# january_first = None  # Select January 1, 2023 from patient_vitals
# print("January 1, 2023 data:", january_first)
# print(f"Records on Jan 1: {len(january_first)} (some patients may start later)")

# TODO: Select data by date ranges
# january_data = None  # Select entire January 2023
# print("January 2023 shape:", january_data.shape)

# TODO: Select data by time periods
# first_quarter = None  # Select Q1 2023
# entire_year = None  # Select all of 2023 (will include patients with partial year data)

# TODO: Select first and last periods using .loc
# first_week = patient_vitals.loc[:patient_vitals.index.min() + pd.Timedelta(days=6)]  # First 7 days
# last_week = patient_vitals.loc[patient_vitals.index.max() - pd.Timedelta(days=6):]  # Last 7 days

# TODO: Use truncate() method
# Note: truncate() requires a sorted index. Sort first if needed: patient_vitals = patient_vitals.sort_index()
# data_after_june = None  # Truncate before June 1, 2023
# data_before_september = None  # Truncate after August 31, 2023

# TODO: Use selected data for analysis
# Compare average temperature between first quarter and data after June
# print(f"\nFirst quarter average temperature: {first_quarter['temperature'].mean():.2f}°F")
# print(f"After June average temperature: {data_after_june['temperature'].mean():.2f}°F")
# print(f"First week average temperature: {first_week['temperature'].mean():.2f}°F")
# print(f"Last week average temperature: {last_week['temperature'].mean():.2f}°F")

# For ICU data with time components:
# TODO: Select business hours (9 AM to 5 PM)
# business_hours = None  # Use between_time()
# print("Business hours data shape:", business_hours.shape)

# TODO: Select specific time (noon readings)
# noon_data = None  # Use at_time('12:00')

# TODO: Use time-based selection for analysis
# Compare vital signs during business hours vs other times
# all_hours_avg = icu_monitoring.select_dtypes(include=[np.number]).mean()
# business_hours_avg = business_hours.select_dtypes(include=[np.number]).mean()
# print(f"\nAverage heart rate - All hours: {all_hours_avg['heart_rate']:.1f} bpm")
# print(f"Average heart rate - Business hours: {business_hours_avg['heart_rate']:.1f} bpm")
# print(f"Average temperature - All hours: {all_hours_avg['temperature']:.1f}°F")
# print(f"Average temperature - Business hours: {business_hours_avg['temperature']:.1f}°F")
```

## Part 2.3: Resampling Operations

**TODO: Perform resampling and frequency conversion**

**Important Note:** When resampling DataFrames that contain non-numeric columns (like `patient_id`), you'll get an error if you try to aggregate them with numeric functions like `mean()`. Use `df.select_dtypes(include=[np.number])` to select only numeric columns before resampling, or specify which columns to aggregate in `.agg()`.

```python
# TODO: Resample hourly ICU data to daily
# Note: Exclude non-numeric columns like 'patient_id' when resampling
# Select only numeric columns before resampling
# numeric_cols = icu_monitoring.select_dtypes(include=[np.number]).columns
# icu_daily = icu_monitoring[numeric_cols].resample('D').mean()
# print("ICU daily shape:", icu_daily.shape)

# TODO: Resample daily patient data to weekly
# Note: Exclude 'patient_id' column when resampling
# Select only numeric columns before resampling
# numeric_cols_pv = patient_vitals.select_dtypes(include=[np.number]).columns
# patient_vitals_weekly = patient_vitals[numeric_cols_pv].resample('W').mean()
# print("Weekly resampled shape:", patient_vitals_weekly.shape)

# TODO: Resample daily patient data to monthly
# patient_vitals_monthly = None  # Resample to monthly with mean aggregation (use freq='ME' for Month End)
# print("Monthly resampled shape:", patient_vitals_monthly.shape)

# TODO: Use different aggregation functions (mean, sum, max, min)
# icu_daily_stats = None  # Resample with multiple aggregations
# Example: resample('D').agg({'heart_rate': ['mean', 'max', 'min'], 
#                             'temperature': 'mean'})

# TODO: Handle missing values during resampling
# Demonstrate upsampling (monthly to daily) creates missing values
# Note: When upsampling, use .asfreq() to create missing values, or use .resample() with aggregation
# monthly_to_daily = None  # Upsample monthly data to daily (use .asfreq() or .resample('D'))
# print("Missing values after upsampling:", monthly_to_daily.isna().sum())

# TODO: Compare different resampling frequencies
# Create a DataFrame comparing resampling results at different frequencies
# Important: Since patient_vitals contains multiple patients per date, you need to aggregate by date first
# to create a single daily time series for comparison.
# Why aggregation is needed: The patient_vitals DataFrame has multiple rows per date (one for each patient),
# so we need to average across patients for each date to create a single daily time series that can be
# meaningfully compared with the weekly and monthly resampled data. Without aggregation, resampling would
# operate on each patient's time series separately, making it difficult to compare frequencies meaningfully.
# Steps:
# 1. Since 'date' is currently the index, reset it to a column first, then aggregate by date
#    Note: groupby('date').mean() automatically sets 'date' as the index in the result, so you don't need
#    to call set_index('date') again after groupby.
#    patient_vitals_reset = patient_vitals[numeric_cols_pv].reset_index()
#    patient_vitals_daily_agg = patient_vitals_reset.groupby('date').mean()
#    # The date is already the index after groupby, so no need to set_index again
# 2. Compare the aggregated daily data with weekly and monthly resampled data
# Use patient_vitals data resampled to different frequencies:
# - Original daily data (aggregated by date): patient_vitals_daily_agg
# - Weekly resampled (patient_vitals_weekly) 
# - Monthly resampled (patient_vitals_monthly)
# Include columns: frequency, date_range, row_count, mean_temperature, std_temperature
# Use the 'temperature' column from each resampled dataset
# Example structure:
# resampling_comparison = pd.DataFrame({
#     'frequency': ['daily', 'weekly', 'monthly'],
#     'date_range': [...],  # Use index.min() and index.max() for each dataset
#     'row_count': [...],  # Use len() for each dataset
#     'mean_temperature': [...],  # Use .mean() on 'temperature' column for each dataset
#     'std_temperature': [...]   # Use .std() on 'temperature' column for each dataset
# })

# TODO: Save results as 'output/q2_resampling_analysis.csv'
# resampling_comparison.to_csv('output/q2_resampling_analysis.csv', index=False)
```

## Part 2.4: Missing Data Handling

**Note:** We'll use upsampling to create missing values for practice. The patient_vitals dataset also contains some naturally occurring missing data (~5% missing visits), which you could use as an alternative approach in practice.

**Important:** See the "Missing Data Handling" section in `assignment/README.md` for detailed guidance on creating time series with missing values and choosing imputation methods.

**TODO: Handle missing data in time series**

```python
# TODO: Identify missing values in time series
# Recommended approach: Create missing values by upsampling (e.g., monthly to daily)
#   - Use the monthly resampled data from Part 2.3: patient_vitals_monthly['temperature']
#   - Upsample to daily frequency using .resample('D').asfreq()
#   - This creates missing values for all days except month-end dates
# Alternative approach (for practice): You could also use naturally occurring missing data from patient_vitals
#   by aggregating by date and using groupby('date'), but upsampling provides a clearer pattern for demonstration
# ts_with_missing = None  # Time series with missing values
# print("Missing value count:", ts_with_missing.isna().sum())
# print("Missing value percentage:", ts_with_missing.isna().sum() / len(ts_with_missing) * 100)

# TODO: Use forward fill and backward fill
# ts_ffill = None  # Forward fill missing values (use .ffill() method)
# ts_bfill = None  # Backward fill missing values (use .bfill() method)

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

