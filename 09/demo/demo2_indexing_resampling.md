# Demo 2: Resampling and Rolling Window Operations

**Placement**: After "Resampling" section (~2/3 through lecture)  
**Duration**: 25 minutes  
**Focus**: Resampling operations, rolling windows, and exponentially weighted functions with ICU monitoring data

## Learning Objectives
- Use resampling for frequency conversion with ICU data
- Apply rolling window operations for trend detection
- Calculate exponentially weighted moving averages
- Handle missing data in time series
- Combine resampling with visualization

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
```

## Part 1: ICU Monitoring Data - Resampling

### Load and Prepare ICU Data

```python
# Simulate hourly ICU patient monitoring data (6 months)
print("=== ICU Patient Monitoring Data ===\n")

# Create hourly dates for 6 months
hourly_dates = pd.date_range('2023-01-01', periods=24*30*6, freq='H')

# Generate realistic ICU monitoring data
np.random.seed(42)
icu_data = pd.DataFrame({
    'heart_rate': np.random.randint(60, 100, len(hourly_dates)),
    'blood_pressure_systolic': np.random.randint(110, 140, len(hourly_dates)),
    'blood_pressure_diastolic': np.random.randint(70, 90, len(hourly_dates)),
    'oxygen_saturation': np.random.randint(95, 100, len(hourly_dates)),
    'temperature': np.random.normal(98.6, 0.5, len(hourly_dates))
}, index=hourly_dates)

print(f"ICU data shape: {icu_data.shape}")
print(f"Date range: {icu_data.index.min()} to {icu_data.index.max()}")
print(f"\nSample data:")
print(icu_data.head())
print(f"\nData summary:")
print(icu_data.describe())
```

### Basic Resampling - Hourly to Daily

```python
# Resample hourly data to daily (aggregate to daily summaries)
print("=== Resampling: Hourly to Daily ===\n")

# Daily resampling with different aggregations
daily_icu = pd.DataFrame({
    'heart_rate_mean': icu_data['heart_rate'].resample('D').mean(),
    'heart_rate_max': icu_data['heart_rate'].resample('D').max(),
    'heart_rate_min': icu_data['heart_rate'].resample('D').min(),
    'blood_pressure_systolic_mean': icu_data['blood_pressure_systolic'].resample('D').mean(),
    'oxygen_saturation_mean': icu_data['oxygen_saturation'].resample('D').mean(),
    'temperature_mean': icu_data['temperature'].resample('D').mean(),
    'observation_count': icu_data['heart_rate'].resample('D').count()  # Count of hourly readings
})

print(f"Daily resampled shape: {daily_icu.shape}")
print(f"Original hourly shape: {icu_data.shape}")
print(f"Reduction: {icu_data.shape[0] / daily_icu.shape[0]:.1f}x fewer rows")
print(f"\nDaily summary:")
print(daily_icu.head())
```

### Resampling with Multiple Aggregations

```python
# Resample with multiple aggregations (like groupby from Lecture 5)
print("=== Resampling with Multiple Aggregations ===\n")

# Resample to weekly with multiple statistics
weekly_stats = icu_data.resample('W').agg({
    'heart_rate': ['mean', 'std', 'min', 'max'],
    'blood_pressure_systolic': ['mean', 'std'],
    'oxygen_saturation': ['mean', 'min'],
    'temperature': ['mean', 'std']
})

print(f"Weekly stats shape: {weekly_stats.shape}")
print(f"\nWeekly statistics:")
print(weekly_stats.head())

# Resample to monthly
monthly_stats = icu_data.resample('ME').agg({
    'heart_rate': 'mean',
    'blood_pressure_systolic': 'mean',
    'oxygen_saturation': 'mean',
    'temperature': 'mean'
})

print(f"\nMonthly averages:")
print(monthly_stats.head())
```

### Handling Missing Data in Resampling

```python
# Demonstrate handling missing data (upsampling)
print("=== Handling Missing Data in Resampling ===\n")

# Create daily summary (some days missing)
daily_summary = icu_data['heart_rate'].resample('D').mean()

# Upsample to hourly (creates missing values)
hourly_upsampled = daily_summary.resample('H').asfreq()
print(f"Upsampled data - missing values: {hourly_upsampled.isna().sum()}")
print(f"Missing percentage: {hourly_upsampled.isna().sum() / len(hourly_upsampled) * 100:.1f}%")

# Forward fill missing values
hourly_filled = daily_summary.resample('H').ffill()
print(f"\nAfter forward fill - missing values: {hourly_filled.isna().sum()}")

# Interpolate missing values
hourly_interpolated = daily_summary.resample('H').interpolate(method='linear')
print(f"After interpolation - missing values: {hourly_interpolated.isna().sum()}")
```

## Part 2: Rolling Window Operations

### Basic Rolling Window Statistics

```python
# Apply rolling windows to ICU data
print("=== Rolling Window Operations ===\n")

# Calculate rolling statistics for heart rate
daily_heart_rate = icu_data['heart_rate'].resample('D').mean()

# 7-day rolling window (1 week)
daily_heart_rate['rolling_7d_mean'] = daily_heart_rate.rolling(window=7).mean()
daily_heart_rate['rolling_7d_std'] = daily_heart_rate.rolling(window=7).std()
daily_heart_rate['rolling_7d_min'] = daily_heart_rate.rolling(window=7).min()
daily_heart_rate['rolling_7d_max'] = daily_heart_rate.rolling(window=7).max()

# 30-day rolling window (1 month)
daily_heart_rate['rolling_30d_mean'] = daily_heart_rate.rolling(window=30).mean()

print("Daily heart rate with rolling statistics:")
print(daily_heart_rate.head(10))
print(f"\nRolling statistics summary:")
print(daily_heart_rate[['rolling_7d_mean', 'rolling_30d_mean']].describe())
```

### Advanced Rolling Operations

```python
# Advanced rolling operations
print("=== Advanced Rolling Operations ===\n")

# Centered rolling window (looks both forward and backward)
daily_heart_rate['rolling_7d_centered'] = daily_heart_rate['heart_rate'].rolling(
    window=7, center=True
).mean()

# Expanding window (from start to current)
daily_heart_rate['expanding_mean'] = daily_heart_rate['heart_rate'].expanding().mean()

# Rolling with minimum periods (starts calculating earlier)
daily_heart_rate['rolling_7d_min_periods'] = daily_heart_rate['heart_rate'].rolling(
    window=7, min_periods=3
).mean()

print("Advanced rolling operations:")
print(daily_heart_rate[['heart_rate', 'rolling_7d_centered', 'expanding_mean']].head(10))
```

### Exponentially Weighted Moving Average

```python
# Exponentially weighted moving average (more responsive to recent changes)
print("=== Exponentially Weighted Moving Average ===\n")

# Calculate EWM with different spans
daily_heart_rate['ewm_span_7'] = daily_heart_rate['heart_rate'].ewm(span=7).mean()
daily_heart_rate['ewm_span_30'] = daily_heart_rate['heart_rate'].ewm(span=30).mean()

# Compare with simple moving average
print("Comparison: Simple MA vs EWM")
print(daily_heart_rate[['heart_rate', 'rolling_7d_mean', 'ewm_span_7']].head(10))

# EWM with different parameters
daily_heart_rate['ewm_alpha_0.3'] = daily_heart_rate['heart_rate'].ewm(alpha=0.3).mean()
daily_heart_rate['ewm_halflife_7'] = daily_heart_rate['heart_rate'].ewm(halflife=7).mean()

print(f"\nEWM comparison:")
print(daily_heart_rate[['ewm_span_7', 'ewm_alpha_0.3', 'ewm_halflife_7']].head(10))
```

## Part 3: Visualization with Resampling and Rolling Windows

### Visualizing Resampling Effects

```python
# Visualize resampling effects
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Original hourly data (sample of first week)
hourly_sample = icu_data['heart_rate']['2023-01-01':'2023-01-07']
axes[0].plot(hourly_sample.index, hourly_sample.values, 
             marker='o', markersize=3, alpha=0.7, linewidth=1, label='Hourly')
axes[0].set_title('High Frequency: Hourly Heart Rate (First Week)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Heart Rate (bpm)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Daily resampled
daily_sample = daily_heart_rate['2023-01-01':'2023-01-31']
axes[1].plot(daily_sample.index, daily_sample['heart_rate'], 
             marker='o', markersize=5, linewidth=2, label='Daily Mean', color='red')
axes[1].set_title('Medium Frequency: Daily Heart Rate (January)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Heart Rate (bpm)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Weekly resampled
weekly_sample = daily_heart_rate['heart_rate'].resample('W').mean()['2023-01-01':'2023-01-31']
axes[2].plot(weekly_sample.index, weekly_sample.values, 
             marker='o', markersize=8, linewidth=2, label='Weekly Mean', color='green')
axes[2].set_title('Low Frequency: Weekly Heart Rate (January)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Heart Rate (bpm)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Visualizing Rolling Windows

```python
# Visualize rolling window effects
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Rolling mean comparison
sample_data = daily_heart_rate['2023-01-01':'2023-03-31']
axes[0].plot(sample_data.index, sample_data['heart_rate'], 
             alpha=0.5, linewidth=1, label='Daily Heart Rate', color='gray')
axes[0].plot(sample_data.index, sample_data['rolling_7d_mean'], 
             linewidth=2, label='7-Day Rolling Mean', color='blue')
axes[0].plot(sample_data.index, sample_data['rolling_30d_mean'], 
             linewidth=2, label='30-Day Rolling Mean', color='red')
axes[0].fill_between(sample_data.index,
                     sample_data['rolling_7d_mean'] - sample_data['rolling_7d_std'],
                     sample_data['rolling_7d_mean'] + sample_data['rolling_7d_std'],
                     alpha=0.2, color='blue', label='±1 Std Dev')
axes[0].set_title('Rolling Window Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Heart Rate (bpm)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# EWM vs Simple MA
axes[1].plot(sample_data.index, sample_data['heart_rate'], 
             alpha=0.3, linewidth=1, label='Daily Heart Rate', color='gray')
axes[1].plot(sample_data.index, sample_data['rolling_7d_mean'], 
             linewidth=2, label='7-Day Simple MA', color='blue', linestyle='--')
axes[1].plot(sample_data.index, sample_data['ewm_span_7'], 
             linewidth=2, label='7-Day EWM', color='red')
axes[1].set_title('Simple Moving Average vs Exponentially Weighted', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Heart Rate (bpm)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Part 4: Combining Concepts - Multi-Variable Analysis

### Multi-Variable Rolling Analysis

```python
# Analyze multiple variables with rolling windows
print("=== Multi-Variable Rolling Analysis ===\n")

# Create daily summary for all variables
daily_summary = icu_data.resample('D').agg({
    'heart_rate': 'mean',
    'blood_pressure_systolic': 'mean',
    'oxygen_saturation': 'mean',
    'temperature': 'mean'
})

# Calculate rolling correlations (7-day window)
rolling_corr = daily_summary['heart_rate'].rolling(window=7).corr(
    daily_summary['blood_pressure_systolic']
)

print("Rolling correlation (7-day window) between heart rate and blood pressure:")
print(rolling_corr.head(10))

# Multiple rolling statistics
daily_summary['hr_rolling_mean'] = daily_summary['heart_rate'].rolling(window=7).mean()
daily_summary['bp_rolling_mean'] = daily_summary['blood_pressure_systolic'].rolling(window=7).mean()
daily_summary['temp_rolling_mean'] = daily_summary['temperature'].rolling(window=7).mean()

print(f"\nDaily summary with rolling statistics:")
print(daily_summary.head(10))
```

### Visualization with Multiple Variables

```python
# Create comprehensive multi-variable visualization
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# Heart rate
axes[0].plot(daily_summary.index, daily_summary['heart_rate'], 
             alpha=0.5, linewidth=1, label='Daily', color='gray')
axes[0].plot(daily_summary.index, daily_summary['hr_rolling_mean'], 
             linewidth=2, label='7-Day Rolling Mean', color='blue')
axes[0].set_title('Heart Rate with Rolling Mean', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Heart Rate (bpm)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Blood pressure
axes[1].plot(daily_summary.index, daily_summary['blood_pressure_systolic'], 
             alpha=0.5, linewidth=1, label='Daily', color='gray')
axes[1].plot(daily_summary.index, daily_summary['bp_rolling_mean'], 
             linewidth=2, label='7-Day Rolling Mean', color='red')
axes[1].set_title('Blood Pressure (Systolic) with Rolling Mean', fontsize=12, fontweight='bold')
axes[1].set_ylabel('BP Systolic (mmHg)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Oxygen saturation
axes[2].plot(daily_summary.index, daily_summary['oxygen_saturation'], 
             alpha=0.5, linewidth=1, label='Daily', color='gray')
axes[2].axhline(y=95, color='red', linestyle='--', label='Critical Threshold (95%)')
axes[2].set_title('Oxygen Saturation', fontsize=12, fontweight='bold')
axes[2].set_ylabel('O2 Sat (%)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Temperature
axes[3].plot(daily_summary.index, daily_summary['temperature'], 
             alpha=0.5, linewidth=1, label='Daily', color='gray')
axes[3].plot(daily_summary.index, daily_summary['temp_rolling_mean'], 
             linewidth=2, label='7-Day Rolling Mean', color='orange')
axes[3].axhline(y=98.6, color='red', linestyle='--', label='Normal (98.6°F)')
axes[3].set_title('Temperature with Rolling Mean', fontsize=12, fontweight='bold')
axes[3].set_xlabel('Date')
axes[3].set_ylabel('Temperature (°F)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Key Takeaways

1. **Resampling**: Convert between frequencies (hourly → daily → weekly → monthly)
2. **Rolling Windows**: Smooth noisy data and identify trends
3. **Exponentially Weighted**: More responsive to recent changes
4. **Missing Data**: Handle gaps in time series with forward fill, interpolation
5. **Multi-Variable Analysis**: Apply rolling operations to multiple variables simultaneously
6. **Visualization**: Combine resampling and rolling windows with plots

## Next Steps

- Practice with your own time series data
- Integrate with visualization tools from Lecture 07 (Demo 3)
- Explore seasonal decomposition for pattern identification
- Set up automated analysis workflows
