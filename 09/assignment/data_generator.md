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

# Healthcare Time Series Data Generator for Assignment 9

## Overview
This notebook generates healthcare time series datasets for the assignment: daily patient vital signs, hourly ICU monitoring data, and monthly disease surveillance data.

## Setup

```python
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create data directory
os.makedirs('data', exist_ok=True)

print("Healthcare Time Series Data Generator - Setup Complete")
```

## Generate Patient Vitals Data (Daily, 1 Year)

```python
# Generate daily patient vital signs data for 1 year
print("=== Generating Patient Vitals Data ===\n")

# Create date range for 1 year (daily)
dates = pd.date_range('2023-01-01', periods=365, freq='D')

# Generate patient IDs - larger cohort for realistic analysis
n_patients = 200
patient_ids = [f'P{i:04d}' for i in range(1, n_patients + 1)]

# Generate daily vital signs for each patient
vitals_data = []

for patient_id in patient_ids:
    # Patient-specific baseline values (more realistic variation)
    age = np.random.randint(25, 85)
    gender = np.random.choice(['M', 'F'])
    
    # Age and gender affect baselines
    if gender == 'M':
        baseline_temp = np.random.normal(98.6, 0.3)
        baseline_hr = np.random.randint(60, 85)
        baseline_bp_sys = np.random.randint(110, 135)
        baseline_weight = np.random.normal(80, 15)
    else:
        baseline_temp = np.random.normal(98.4, 0.3)
        baseline_hr = np.random.randint(65, 90)
        baseline_bp_sys = np.random.randint(105, 130)
        baseline_weight = np.random.normal(65, 12)
    
    # Age affects heart rate and blood pressure
    baseline_hr += int((age - 50) * 0.2)
    baseline_bp_sys += int((age - 50) * 0.3)
    baseline_bp_dia = baseline_bp_sys - np.random.randint(35, 45)
    
    # Patient start date (not all patients start on Jan 1)
    start_offset = np.random.randint(0, 60)  # Some patients join later
    patient_dates = dates[start_offset:]
    
    # Missing data pattern (patients miss ~5% of visits)
    missing_probability = 0.05
    
    # Generate time series with trend and seasonality
    for i, date in enumerate(patient_dates):
        # Skip some dates (missing visits)
        if np.random.random() < missing_probability:
            continue
            
        # Seasonal component (slight variation)
        day_of_year = date.timetuple().tm_yday
        seasonal = 0.5 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Trend component (weight may change over time)
        trend = (i / len(patient_dates)) * np.random.choice([-0.2, 0, 0.2], p=[0.3, 0.4, 0.3])
        
        # Day of week effect (weekend vs weekday)
        day_of_week = date.weekday()
        weekday_effect = 0.3 if day_of_week < 5 else -0.2
        
        # Correlated noise (temperature and heart rate correlated)
        noise_base = np.random.normal(0, 0.3)
        noise_temp = noise_base + np.random.normal(0, 0.2)
        noise_hr = noise_base * 2 + np.random.randint(-5, 5)
        noise_bp = np.random.randint(-5, 5)
        
        # Calculate values with realistic correlations
        temp = max(96.0, min(102.0, baseline_temp + seasonal + noise_temp))
        hr = max(50, min(120, baseline_hr + weekday_effect + noise_hr))
        bp_sys = max(90, min(160, baseline_bp_sys + noise_bp))
        bp_dia = max(60, min(100, baseline_bp_dia + noise_bp))
        weight = max(40, min(150, baseline_weight + trend + np.random.normal(0, 0.5)))
        
        vitals_data.append({
            'date': date,
            'patient_id': patient_id,
            'temperature': round(temp, 1),
            'heart_rate': hr,
            'blood_pressure_systolic': bp_sys,
            'blood_pressure_diastolic': bp_dia,
            'weight': round(weight, 1)
        })

# Create DataFrame
patient_vitals = pd.DataFrame(vitals_data)

print(f"Generated {len(patient_vitals):,} patient vital signs records")
print(f"Date range: {patient_vitals['date'].min()} to {patient_vitals['date'].max()}")
print(f"Number of patients: {patient_vitals['patient_id'].nunique()}")
print(f"Average records per patient: {len(patient_vitals) / patient_vitals['patient_id'].nunique():.1f}")
print(f"\nSample data:")
print(patient_vitals.head(10))
print(f"\nData summary:")
print(patient_vitals.describe())

# Save to CSV
patient_vitals.to_csv('data/patient_vitals.csv', index=False)
print(f"\n✅ Saved: data/patient_vitals.csv")
```

## Generate ICU Monitoring Data (Hourly, 6 Months)

```python
# Generate hourly ICU monitoring data for 6 months
print("\n=== Generating ICU Monitoring Data ===\n")

# Create date range for 6 months (hourly)
icu_start_date = pd.Timestamp('2023-01-01')
icu_end_date = pd.Timestamp('2023-06-30 23:00:00')
icu_dates = pd.date_range(icu_start_date, icu_end_date, freq='H')

# Generate ICU patient IDs - larger cohort
n_icu_patients = 75
icu_patient_ids = [f'ICU{i:03d}' for i in range(1, n_icu_patients + 1)]

# Generate hourly ICU data
icu_data = []

for patient_id in icu_patient_ids:
    # Patient-specific baseline values (more realistic ICU scenarios)
    # Some patients are more critical than others
    severity = np.random.choice(['critical', 'moderate', 'stable'], p=[0.2, 0.4, 0.4])
    
    if severity == 'critical':
        baseline_hr = np.random.randint(90, 120)
        baseline_bp_sys = np.random.randint(90, 110)
        baseline_o2 = np.random.randint(88, 94)
        baseline_temp = np.random.normal(99.5, 0.8)
    elif severity == 'moderate':
        baseline_hr = np.random.randint(75, 95)
        baseline_bp_sys = np.random.randint(100, 130)
        baseline_o2 = np.random.randint(93, 97)
        baseline_temp = np.random.normal(98.8, 0.6)
    else:  # stable
        baseline_hr = np.random.randint(60, 80)
        baseline_bp_sys = np.random.randint(110, 140)
        baseline_o2 = np.random.randint(96, 100)
        baseline_temp = np.random.normal(98.6, 0.4)
    
    baseline_bp_dia = baseline_bp_sys - np.random.randint(35, 50)
    
    # Patient admission and discharge dates (not all patients stay entire period)
    admission_date = icu_start_date + pd.Timedelta(days=np.random.randint(0, 120))
    length_of_stay = np.random.randint(2, 30)  # Days in ICU
    discharge_date = min(admission_date + pd.Timedelta(days=length_of_stay), icu_end_date)
    
    # Only generate data for patient's stay period
    patient_dates = icu_dates[(icu_dates >= admission_date) & (icu_dates <= discharge_date)]
    
    # Generate hourly time series with realistic patterns
    for i, datetime in enumerate(patient_dates):
        # Daily circadian rhythm (stronger for stable patients)
        hour = datetime.hour
        circadian_strength = 8 if severity == 'stable' else 5
        circadian = circadian_strength * np.sin(2 * np.pi * hour / 24)
        
        # Day of stay effect (patients may improve or worsen)
        days_in_stay = (datetime - admission_date).days
        recovery_trend = -0.5 * days_in_stay if severity != 'critical' else 0.2 * days_in_stay
        
        # Correlated noise (vital signs are correlated)
        noise_base = np.random.normal(0, 0.5)
        noise_hr = noise_base * 3 + np.random.randint(-8, 8)
        noise_bp = noise_base * 2 + np.random.randint(-6, 6)
        noise_o2 = -noise_base * 0.5 + np.random.randint(-2, 2)  # Inverse correlation
        noise_temp = noise_base + np.random.normal(0, 0.3)
        
        # Calculate values
        hr = max(50, min(150, baseline_hr + circadian + recovery_trend + noise_hr))
        bp_sys = max(80, min(180, baseline_bp_sys + recovery_trend + noise_bp))
        bp_dia = max(50, min(110, baseline_bp_dia + recovery_trend + noise_bp * 0.7))
        o2 = max(85, min(100, baseline_o2 + recovery_trend * 0.1 + noise_o2))
        temp = max(95.0, min(103.0, baseline_temp + noise_temp))
        
        icu_data.append({
            'datetime': datetime,
            'patient_id': patient_id,
            'heart_rate': hr,
            'blood_pressure_systolic': bp_sys,
            'blood_pressure_diastolic': bp_dia,
            'oxygen_saturation': o2,
            'temperature': round(temp, 1)
        })

# Create DataFrame
icu_monitoring = pd.DataFrame(icu_data)

print(f"Generated {len(icu_monitoring):,} ICU monitoring records")
print(f"Date range: {icu_monitoring['datetime'].min()} to {icu_monitoring['datetime'].max()}")
print(f"Number of ICU patients: {icu_monitoring['patient_id'].nunique()}")
print(f"Average records per patient: {len(icu_monitoring) / icu_monitoring['patient_id'].nunique():.1f}")
print(f"\nSample data:")
print(icu_monitoring.head(10))
print(f"\nData summary:")
print(icu_monitoring.describe())

# Save to CSV
icu_monitoring.to_csv('data/icu_monitoring.csv', index=False)
print(f"\n✅ Saved: data/icu_monitoring.csv")
```

## Generate Disease Surveillance Data (Monthly, 5 Years, Multiple Sites)

```python
# Generate monthly disease surveillance data for 5 years (multiple sites)
print("\n=== Generating Disease Surveillance Data ===\n")

# Create date range for 5 years (monthly) - longer period for better analysis
surveillance_dates = pd.date_range('2020-01-01', periods=60, freq='ME')  # Month-end

# Surveillance sites - more sites for realistic multi-site analysis
sites = ['Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E', 'Site_F']

# Generate disease surveillance data
surveillance_data = []

for site in sites:
    # Site-specific baseline case counts (different population sizes)
    baseline_cases = np.random.randint(50, 200)
    
    # Site-specific characteristics (urban vs rural, climate zone)
    site_type = np.random.choice(['urban', 'suburban', 'rural'], p=[0.3, 0.4, 0.3])
    climate_zone = np.random.choice(['temperate', 'tropical', 'arid'], p=[0.5, 0.3, 0.2])
    
    # Adjust baseline based on site characteristics
    if site_type == 'urban':
        baseline_cases *= 1.3  # Higher population density
    elif site_type == 'rural':
        baseline_cases *= 0.7
    
    # Generate monthly time series with seasonality
    for i, date in enumerate(surveillance_dates):
        # Strong seasonal component (flu season peaks in winter)
        month = date.month
        
        # Seasonal pattern varies by climate zone
        if climate_zone == 'temperate':
            seasonal = 40 * np.sin(2 * np.pi * (month - 1) / 12) + 20  # Strong winter peak
        elif climate_zone == 'tropical':
            seasonal = 15 * np.sin(2 * np.pi * (month - 1) / 12) + 10  # Milder seasonality
        else:  # arid
            seasonal = 25 * np.sin(2 * np.pi * (month - 1) / 12) + 15
        
        # Trend component (slight increase over time, with some variation)
        trend = (i / 60) * np.random.choice([-3, 0, 5, 8], p=[0.1, 0.3, 0.4, 0.2])
        
        # Outbreak events (random spikes)
        outbreak = 0
        if np.random.random() < 0.05:  # 5% chance of outbreak
            outbreak = np.random.randint(30, 80)
        
        # Random noise
        noise = np.random.normal(0, 10)
        
        # Temperature (correlated with seasonality and climate zone)
        if climate_zone == 'temperate':
            temp = 55 + 35 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 4)
        elif climate_zone == 'tropical':
            temp = 75 + 8 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 2)
        else:  # arid
            temp = 70 + 25 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 5)
        
        # Humidity (inverse of temperature, varies by climate)
        if climate_zone == 'tropical':
            humidity = 75 + 10 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 5)
        else:
            humidity = 60 - 30 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 6)
        humidity = max(15, min(95, humidity))
        
        # Calculate cases (temperature and humidity affect disease spread)
        temp_effect = -0.5 * (temp - 65) if temp > 65 else 0  # Higher temp reduces some diseases
        humidity_effect = 0.3 * (humidity - 50) if humidity > 50 else 0
        
        cases = max(0, int(baseline_cases + seasonal + trend + outbreak + temp_effect + humidity_effect + noise))
        
        surveillance_data.append({
            'date': date,
            'site': site,
            'cases': cases,
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1)
        })

# Create DataFrame
disease_surveillance = pd.DataFrame(surveillance_data)

print(f"Generated {len(disease_surveillance):,} disease surveillance records")
print(f"Date range: {disease_surveillance['date'].min()} to {disease_surveillance['date'].max()}")
print(f"Number of sites: {disease_surveillance['site'].nunique()}")
print(f"Records per site: {len(disease_surveillance) / disease_surveillance['site'].nunique():.0f}")
print(f"\nSample data:")
print(disease_surveillance.head(10))
print(f"\nData summary by site:")
print(disease_surveillance.groupby('site')['cases'].describe())

# Save to CSV
disease_surveillance.to_csv('data/disease_surveillance.csv', index=False)
print(f"\n✅ Saved: data/disease_surveillance.csv")
```

## Summary

```python
print("\n" + "="*60)
print("Healthcare Time Series Data Generation Complete")
print("="*60)
print("\nGenerated datasets:")
print(f"1. patient_vitals.csv: {len(patient_vitals):,} records ({patient_vitals['patient_id'].nunique()} patients, daily, 1 year)")
print(f"2. icu_monitoring.csv: {len(icu_monitoring):,} records ({icu_monitoring['patient_id'].nunique()} patients, hourly, 6 months)")
print(f"3. disease_surveillance.csv: {len(disease_surveillance):,} records ({disease_surveillance['site'].nunique()} sites, monthly, 5 years)")
print("\nAll datasets saved to the 'data/' directory")
print("Ready for time series analysis!")
```
