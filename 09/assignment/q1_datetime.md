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

# Question 1: datetime Fundamentals and Time Series Indexing

This question focuses on datetime handling and time series indexing using patient vital signs data.

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

## Part 1.1: Load and Explore Data

**Note:** This dataset contains realistic healthcare data characteristics:
- **200 patients** with daily vital signs over 1 year
- **Missing visits**: Patients miss approximately 5% of scheduled visits (realistic!)
- **Different start dates**: Not all patients start monitoring on January 1st (some join later)
- When selecting data by date ranges, you may find that some patients don't have data for certain periods - this is expected and realistic

```python
# Load patient vital signs data
patient_vitals = pd.read_csv('data/patient_vitals.csv')

print("Patient vitals shape:", patient_vitals.shape)
print("Patient vitals columns:", patient_vitals.columns.tolist())

# Display sample data
print("\nPatient vitals sample:")
print(patient_vitals.head())
print("\nData summary:")
print(patient_vitals.describe())

# Check date range and missing data patterns
print(f"\nDate range: {patient_vitals['date'].min()} to {patient_vitals['date'].max()}")
print(f"Unique patients: {patient_vitals['patient_id'].nunique()}")
print(f"Total records: {len(patient_vitals)}")
print(f"Expected records (200 patients × 365 days): {200 * 365:,}")
print(f"Missing visits: ~{200 * 365 - len(patient_vitals):,} records")
```

## Part 1.2: datetime Operations

**TODO: Perform datetime operations**

```python
# TODO: Convert date column to datetime
# patient_vitals['date'] = pd.to_datetime(patient_vitals['date'])

# TODO: Set datetime column as index
# patient_vitals = patient_vitals.set_index('date')

# TODO: Extract year, month, day components from datetime index
# patient_vitals['year'] = None  # Extract from index
# patient_vitals['month'] = None  # Extract from index
# patient_vitals['day'] = None  # Extract from index

# TODO: Calculate time differences (e.g., days since first measurement)
# Note: Since patients start at different times, calculate days_since_start per patient
# Hint: To use groupby on the 'date' column, temporarily reset the index, then set it back
# Example: patient_vitals_reset = patient_vitals.reset_index()
#          Use groupby('patient_id')['date'].transform(lambda x: (x - x.min()).dt.days)
#          Or use groupby('patient_id').apply() to calculate days from each patient's first date
#          Then: patient_vitals = patient_vitals_reset.set_index('date')
# patient_vitals['days_since_start'] = None  # Calculate from each patient's start date

# TODO: Create business day ranges for clinic visit schedules
# clinic_dates = None  # Use pd.bdate_range() for clinic visits

# TODO: Create date ranges with different frequencies
# daily_range = None  # Daily monitoring schedule
# weekly_range = None  # Weekly lab test schedule (Mondays)
# monthly_range = None  # Monthly checkup schedule

# TODO: Use date ranges to analyze visit patterns
# Check how many patient visits occurred on clinic business days vs weekends
# patient_dates_set = set(patient_vitals.index.date)
# clinic_dates_set = set(clinic_dates.date)
# visits_on_clinic_days = len(patient_dates_set & clinic_dates_set)
# visits_on_weekends = len(patient_dates_set) - visits_on_clinic_days
# print(f"Visits on clinic business days: {visits_on_clinic_days}")
# print(f"Visits on weekends: {visits_on_weekends}")
# print(f"Total unique visit dates: {len(patient_dates_set)}")

# TODO: Save results as 'output/q1_datetime_analysis.csv'
# Create a DataFrame with datetime analysis results including:
# - date (datetime index or column)
# - year, month, day (extracted from datetime)
# - days_since_start (calculated time differences)
# - patient_id
# - At least one original column (e.g., temperature, heart_rate)
# Note: When saving to CSV with index=False, you'll need to convert the index to a column first
# Example structure:
# datetime_analysis = patient_vitals[['patient_id', 'year', 'month', 'day', 'days_since_start', 'temperature']].copy()
# datetime_analysis.to_csv('output/q1_datetime_analysis.csv', index=False)
```

## Part 1.3: Time Zone Handling

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
# Note: Using UTC avoids DST ambiguity issues - UTC has no daylight saving time
# Best practice: Store data in UTC, convert to local timezones only when needed
# dst_date_utc = pd.Timestamp('2023-03-12 10:00:00', tz='UTC')  # UTC time avoids DST issues
# dst_time_eastern = dst_date_utc.tz_convert('US/Eastern')  # Convert UTC to Eastern

# TODO: Document timezone operations
# Create a report string with the following sections:
# 1. Original timezone: Describe what timezone your original data was in (or if it was naive)
# 2. Localization method: Explain how you localized the data (e.g., tz_localize('UTC'))
# 3. Conversion: Describe what timezone you converted to (e.g., 'US/Eastern')
# 4. DST handling: Document any issues or observations about daylight saving time transitions
#    Note: Explain why using UTC as the base timezone avoids DST ambiguity issues
# 5. Example: Show at least one example of a datetime before and after conversion
# Minimum length: 200 words
timezone_report = """
TODO: Document your timezone operations:
- What timezone was your original data in?
- How did you localize the data?
- What timezone did you convert to?
- What issues did you encounter with DST? (Note: Using UTC avoids DST ambiguity)
- Include at least one example showing a datetime before and after conversion
- Explain why UTC is recommended as the base timezone for storing temporal data
"""

# TODO: Save results as 'output/q1_timezone_report.txt'
# with open('output/q1_timezone_report.txt', 'w') as f:
#     f.write(timezone_report)
```

## Submission Checklist

Before moving to Question 2, verify you've created:

- [ ] `output/q1_datetime_analysis.csv` - datetime analysis results
- [ ] `output/q1_timezone_report.txt` - timezone handling report

