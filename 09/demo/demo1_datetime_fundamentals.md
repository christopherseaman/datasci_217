# Demo 1: datetime Fundamentals and Time Series Indexing

**Placement**: After "Shifting and Lagging" section (~1/3 through lecture)  
**Duration**: 25 minutes  
**Focus**: Python datetime module, pandas DatetimeIndex, and time series indexing with health data

## Learning Objectives
- Master Python datetime module basics with clinical timestamps
- Create and manipulate pandas DatetimeIndex for patient data
- Generate date ranges for medical monitoring schedules
- Perform time series indexing and selection with health data
- Handle time zones for multi-site clinical trials

## Setup

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
```

## Part 1: Python datetime Module with Clinical Data

### Basic datetime Operations

```python
# Current time (for timestamping clinical records)
now = datetime.now()
print(f"Current time: {now}")

# Patient birth date
birth_date = datetime(1990, 5, 15)
print(f"Patient birth date: {birth_date}")

# Lab result timestamp (string parsing)
lab_timestamp = "2023-12-25 14:30:00"
parsed_timestamp = datetime.strptime(lab_timestamp, "%Y-%m-%d %H:%M:%S")
print(f"Lab timestamp: {parsed_timestamp}")

# Format for medical records
formatted = parsed_timestamp.strftime("%B %d, %Y at %I:%M %p")
print(f"Formatted for records: {formatted}")

# Calculate patient age (time differences)
age_days = now - birth_date
print(f"Patient age in days: {age_days.days}")
print(f"Patient age in years: {age_days.days / 365.25:.1f}")
```

### datetime Arithmetic for Clinical Schedules

```python
# Clinical trial timeline
trial_start = datetime(2023, 1, 1)
trial_end = datetime(2023, 12, 31)

# Calculate trial duration
duration = trial_end - trial_start
print(f"Trial duration: {duration.days} days ({duration.days / 30:.1f} months)")

# Calculate follow-up visit dates
visit_1 = trial_start + timedelta(days=30)  # 30-day follow-up
visit_2 = trial_start + timedelta(days=90)  # 90-day follow-up
visit_3 = trial_start + timedelta(weeks=26)  # 6-month follow-up

print(f"\nFollow-up visits:")
print(f"Visit 1 (30 days): {visit_1}")
print(f"Visit 2 (90 days): {visit_2}")
print(f"Visit 3 (6 months): {visit_3}")

# Business days for clinic visits (weekdays only)
clinic_days = pd.bdate_range(trial_start, trial_end)
print(f"\nClinic days in 2023: {len(clinic_days)}")
```

## Part 2: pandas DatetimeIndex with Patient Data

### Creating DatetimeIndex from Lab Results

```python
# Lab test dates (from clinical records)
lab_test_dates = ['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22', '2023-01-29']
dates = pd.to_datetime(lab_test_dates)
print("Lab test dates:")
print(dates)

# Create date range for daily patient monitoring
monitoring_dates = pd.date_range('2023-01-01', periods=30, freq='D')
print(f"\nDaily monitoring dates (30 days):")
print(monitoring_dates[:5])
print("...")
print(monitoring_dates[-5:])

# Create DataFrame with daily vital signs
vital_signs = pd.DataFrame({
    'temperature': np.random.normal(98.6, 0.5, 30),
    'heart_rate': np.random.randint(60, 100, 30),
    'blood_pressure_systolic': np.random.randint(110, 140, 30),
    'blood_pressure_diastolic': np.random.randint(70, 90, 30)
}, index=monitoring_dates)

print("\nVital signs DataFrame:")
print(vital_signs.head())
print(f"\nDate range: {vital_signs.index.min()} to {vital_signs.index.max()}")
```

### Date Range Generation for Medical Schedules

```python
# Different date range types for clinical data
print("=== Medical Schedule Date Ranges ===\n")

# Daily monitoring (vital signs)
daily_monitoring = pd.date_range('2023-01-01', '2023-01-10', freq='D')
print(f"Daily monitoring: {len(daily_monitoring)} days")
print(daily_monitoring)

# Weekly lab tests (Mondays)
weekly_labs = pd.date_range('2023-01-01', '2023-03-01', freq='W-MON')
print(f"\nWeekly lab tests (Mondays): {len(weekly_labs)} dates")
print(weekly_labs)

# Monthly checkups (first of month)
monthly_checkups = pd.date_range('2023-01-01', '2023-12-01', freq='MS')
print(f"\nMonthly checkups: {len(monthly_checkups)} dates")
print(monthly_checkups)

# Quarterly assessments (quarter start)
quarterly_assessments = pd.date_range('2023-01-01', '2023-12-31', freq='QS')
print(f"\nQuarterly assessments: {len(quarterly_assessments)} dates")
print(quarterly_assessments)

# Business days only (clinic visits)
clinic_visits = pd.bdate_range('2023-01-01', '2023-01-31')
print(f"\nClinic visits (business days in January): {len(clinic_visits)} days")
print(clinic_visits[:5])
```

### Frequency Inference

```python
# Create time series with inferred frequency
dates = pd.date_range('2023-01-01', periods=100, freq='D')
daily_temperature = pd.Series(np.random.normal(98.6, 0.5, 100), index=dates)

# Infer frequency
freq = pd.infer_freq(daily_temperature.index)
print(f"Inferred frequency: {freq}")

# Convert to different frequency (daily to weekly)
weekly_temperature = daily_temperature.asfreq('W')
print(f"Weekly frequency: {pd.infer_freq(weekly_temperature.index)}")
print(f"\nWeekly data shape: {weekly_temperature.shape}")

# Resample to monthly (average monthly temperature)
monthly_temperature = daily_temperature.resample('ME').mean()
print(f"Monthly frequency: {pd.infer_freq(monthly_temperature.index)}")
print(f"\nMonthly data shape: {monthly_temperature.shape}")
```

## Part 3: Time Series Indexing with Patient Data

### Basic Time Series Selection

```python
# Create a year of patient data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
patient_weight = pd.Series(70 + np.cumsum(np.random.randn(365) * 0.1), index=dates)

# Select specific date
print("Weight on January 1, 2023:")
print(f"{patient_weight['2023-01-01']:.2f} kg")

# Select date range (first week)
print("\nFirst week of January:")
print(patient_weight['2023-01-01':'2023-01-07'])

# Select entire month
january_data = patient_weight['2023-01']
print(f"\nJanuary 2023 data shape: {january_data.shape}")
print(f"January average weight: {january_data.mean():.2f} kg")

# Select entire year
year_data = patient_weight['2023']
print(f"\n2023 data shape: {year_data.shape}")
print(f"2023 average weight: {year_data.mean():.2f} kg")
print(f"2023 weight range: {year_data.min():.2f} - {year_data.max():.2f} kg")
```

### Advanced Time Series Selection

```python
# Create hourly ICU monitoring data
hourly_dates = pd.date_range('2023-01-01', periods=24*7, freq='H')
icu_data = pd.DataFrame({
    'heart_rate': np.random.randint(60, 100, 24*7),
    'blood_pressure': np.random.randint(90, 140, 24*7),
    'oxygen_saturation': np.random.randint(95, 100, 24*7)
}, index=hourly_dates)

# Select business hours (9 AM to 5 PM)
business_hours = icu_data.between_time('09:00', '17:00')
print(f"Business hours data shape: {business_hours.shape}")
print("\nBusiness hours sample:")
print(business_hours.head())

# Select specific time (noon readings)
noon_readings = icu_data.at_time('12:00')
print(f"\nNoon readings (7 days):")
print(noon_readings)

# Select first and last periods
print(f"\nFirst 3 days:")
print(icu_data.first('3D').head())

print(f"\nLast 3 days:")
print(icu_data.last('3D').tail())
```

## Part 4: Time Zone Handling for Multi-Site Studies

### Basic Time Zone Operations

```python
# Create timezone-aware datetime (clinical trial with multiple sites)
utc_time = pd.Timestamp.now(tz='UTC')
print(f"UTC time: {utc_time}")

# Convert to different timezone (US Eastern)
eastern_time = utc_time.tz_convert('US/Eastern')
print(f"Eastern time: {eastern_time}")

# Create timezone-aware DataFrame (multi-site clinical trial)
site_data = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003'],
    'value': np.random.randn(3)
}, index=pd.date_range('2023-01-01', periods=3, freq='D'))

# Localize to UTC (standardize all sites)
site_data.index = site_data.index.tz_localize('UTC')
print("\nSite data (UTC):")
print(site_data)

# Convert to Eastern time (site location)
site_data.index = site_data.index.tz_convert('US/Eastern')
print("\nSite data (Eastern):")
print(site_data)
```

### Multiple Time Zones

```python
# Clinical trial sites in different time zones
sites = {
    'UTC': 'UTC',
    'New York': 'US/Eastern',
    'London': 'Europe/London',
    'Tokyo': 'Asia/Tokyo',
    'Sydney': 'Australia/Sydney'
}

print("=== Multi-Site Clinical Trial Time Zones ===")
base_time = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')

for site_name, tz in sites.items():
    converted_time = base_time.tz_convert(tz)
    print(f"{site_name:12} ({tz:20}): {converted_time}")
```

## Part 5: Real-World Example - Patient Monitoring Analysis

### Load and Prepare Patient Data

```python
# Simulate patient monitoring data (daily vital signs for 1 year)
print("=== Patient Monitoring Data Analysis ===\n")

# Create monitoring schedule (daily for 1 year)
monitoring_dates = pd.date_range('2023-01-01', periods=365, freq='D')

# Generate realistic patient data
np.random.seed(42)
patient_data = pd.DataFrame({
    'temperature': np.random.normal(98.6, 0.5, 365),
    'heart_rate': np.random.randint(60, 100, 365),
    'blood_pressure_systolic': np.random.randint(110, 140, 365),
    'blood_pressure_diastolic': np.random.randint(70, 90, 365),
    'weight': 70 + np.cumsum(np.random.randn(365) * 0.1)
}, index=monitoring_dates)

print(f"Patient data shape: {patient_data.shape}")
print(f"Date range: {patient_data.index.min()} to {patient_data.index.max()}")
print(f"\nSample data:")
print(patient_data.head())
```

### Time Series Analysis

```python
# Basic time series analysis
print("=== Time Series Analysis ===\n")

# Select specific periods
january = patient_data['2023-01']
print(f"January statistics:")
print(january[['temperature', 'heart_rate']].describe())

# Monthly averages
monthly_avg = patient_data.resample('ME').mean()
print(f"\nMonthly averages:")
print(monthly_avg[['temperature', 'heart_rate', 'weight']].head())

# Calculate changes
patient_data['weight_change'] = patient_data['weight'].diff()
patient_data['heart_rate_change'] = patient_data['heart_rate'].diff()

# Identify significant changes
significant_weight_changes = patient_data[abs(patient_data['weight_change']) > 1.0]
print(f"\nSignificant weight changes (>1 kg): {len(significant_weight_changes)} days")
print(significant_weight_changes[['weight', 'weight_change']].head())
```

### Visualization

```python
# Create comprehensive time series visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Temperature over time
axes[0].plot(patient_data.index, patient_data['temperature'], 
             alpha=0.7, linewidth=1, label='Daily Temperature')
axes[0].axhline(y=98.6, color='red', linestyle='--', label='Normal (98.6°F)')
axes[0].set_title('Patient Temperature Over Time', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Temperature (°F)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Heart rate over time
axes[1].plot(patient_data.index, patient_data['heart_rate'], 
             alpha=0.7, linewidth=1, color='green', label='Daily Heart Rate')
axes[1].axhline(y=80, color='red', linestyle='--', label='Reference (80 bpm)')
axes[1].set_title('Patient Heart Rate Over Time', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Heart Rate (bpm)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Weight over time
axes[2].plot(patient_data.index, patient_data['weight'], 
             alpha=0.7, linewidth=1, color='blue', label='Daily Weight')
axes[2].set_title('Patient Weight Over Time', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Weight (kg)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Key Takeaways

1. **datetime Module**: Essential for parsing clinical timestamps and calculating time differences
2. **DatetimeIndex**: Powerful pandas feature for time series indexing and manipulation
3. **Date Ranges**: Generate medical schedules (daily, weekly, monthly, quarterly)
4. **Time Zones**: Critical for multi-site clinical trials and global health data
5. **Time Series Indexing**: Intuitive selection of data by date ranges
6. **Real-world Application**: Apply datetime skills to patient monitoring and clinical analysis

## Next Steps

- Practice with your own health/medical time series data
- Learn about resampling and frequency conversion (Demo 2)
- Explore rolling window operations for trend detection
- Integrate with visualization tools from Lecture 07
