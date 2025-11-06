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

# Generate patient IDs
n_patients = 50
patient_ids = [f'P{i:04d}' for i in range(1, n_patients + 1)]

# Generate daily vital signs for each patient
vitals_data = []

for patient_id in patient_ids:
    # Patient-specific baseline values
    baseline_temp = np.random.normal(98.6, 0.3)
    baseline_hr = np.random.randint(60, 90)
    baseline_bp_sys = np.random.randint(110, 130)
    baseline_bp_dia = np.random.randint(70, 85)
    baseline_weight = np.random.normal(70, 15)
    
    # Generate time series with trend and seasonality
    for i, date in enumerate(dates):
        # Seasonal component (slight variation)
        seasonal = 0.5 * np.sin(2 * np.pi * i / 365.25)
        
        # Trend component (weight may change over time)
        trend = (i / 365) * 0.1  # Slight weight gain/loss
        
        # Random noise
        noise_temp = np.random.normal(0, 0.3)
        noise_hr = np.random.randint(-5, 5)
        noise_bp = np.random.randint(-5, 5)
        
        vitals_data.append({
            'date': date,
            'patient_id': patient_id,
            'temperature': max(96.0, min(102.0, baseline_temp + seasonal + noise_temp)),
            'heart_rate': max(50, min(120, baseline_hr + noise_hr)),
            'blood_pressure_systolic': max(90, min(160, baseline_bp_sys + noise_bp)),
            'blood_pressure_diastolic': max(60, min(100, baseline_bp_dia + noise_bp)),
            'weight': max(40, min(150, baseline_weight + trend + np.random.normal(0, 0.5)))
        })

# Create DataFrame
patient_vitals = pd.DataFrame(vitals_data)

print(f"Generated {len(patient_vitals):,} patient vital signs records")
print(f"Date range: {patient_vitals['date'].min()} to {patient_vitals['date'].max()}")
print(f"Number of patients: {patient_vitals['patient_id'].nunique()}")
print(f"\nSample data:")
print(patient_vitals.head())
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
icu_dates = pd.date_range('2023-01-01', periods=24 * 30 * 6, freq='H')

# Generate ICU patient IDs
n_icu_patients = 20
icu_patient_ids = [f'ICU{i:03d}' for i in range(1, n_icu_patients + 1)]

# Generate hourly ICU data
icu_data = []

for patient_id in icu_patient_ids:
    # Patient-specific baseline values
    baseline_hr = np.random.randint(60, 100)
    baseline_bp_sys = np.random.randint(100, 140)
    baseline_bp_dia = np.random.randint(60, 90)
    baseline_o2 = np.random.randint(95, 100)
    baseline_temp = np.random.normal(98.6, 0.5)
    
    # Generate hourly time series
    for i, datetime in enumerate(icu_dates):
        # Daily circadian rhythm
        hour = datetime.hour
        circadian = 5 * np.sin(2 * np.pi * hour / 24)
        
        # Random noise
        noise_hr = np.random.randint(-10, 10)
        noise_bp = np.random.randint(-8, 8)
        noise_o2 = np.random.randint(-2, 2)
        noise_temp = np.random.normal(0, 0.3)
        
        icu_data.append({
            'datetime': datetime,
            'patient_id': patient_id,
            'heart_rate': max(50, min(150, baseline_hr + circadian + noise_hr)),
            'blood_pressure_systolic': max(90, min(160, baseline_bp_sys + noise_bp)),
            'blood_pressure_diastolic': max(60, min(100, baseline_bp_dia + noise_bp)),
            'oxygen_saturation': max(90, min(100, baseline_o2 + noise_o2)),
            'temperature': max(95.0, min(103.0, baseline_temp + noise_temp))
        })

# Create DataFrame
icu_monitoring = pd.DataFrame(icu_data)

print(f"Generated {len(icu_monitoring):,} ICU monitoring records")
print(f"Date range: {icu_monitoring['datetime'].min()} to {icu_monitoring['datetime'].max()}")
print(f"Number of ICU patients: {icu_monitoring['patient_id'].nunique()}")
print(f"\nSample data:")
print(icu_monitoring.head())
print(f"\nData summary:")
print(icu_monitoring.describe())

# Save to CSV
icu_monitoring.to_csv('data/icu_monitoring.csv', index=False)
print(f"\n✅ Saved: data/icu_monitoring.csv")
```

## Generate Disease Surveillance Data (Monthly, 3 Years, Multiple Sites)

```python
# Generate monthly disease surveillance data for 3 years (multiple sites)
print("\n=== Generating Disease Surveillance Data ===\n")

# Create date range for 3 years (monthly)
surveillance_dates = pd.date_range('2021-01-01', periods=36, freq='ME')  # Month-end

# Surveillance sites
sites = ['Site_A', 'Site_B', 'Site_C']

# Generate disease surveillance data
surveillance_data = []

for site in sites:
    # Site-specific baseline case counts
    baseline_cases = np.random.randint(80, 150)
    
    # Generate monthly time series with seasonality
    for i, date in enumerate(surveillance_dates):
        # Strong seasonal component (flu season peaks in winter)
        month = date.month
        seasonal = 30 * np.sin(2 * np.pi * (month - 1) / 12) + 15
        
        # Trend component (slight increase over time)
        trend = (i / 36) * 5
        
        # Random noise
        noise = np.random.normal(0, 8)
        
        # Temperature (correlated with seasonality)
        temp = 60 + 30 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 3)
        
        # Humidity (inverse of temperature)
        humidity = 70 - 30 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 5)
        humidity = max(20, min(90, humidity))
        
        # Calculate cases
        cases = max(0, int(baseline_cases + seasonal + trend + noise))
        
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
print(f"\nSample data:")
print(disease_surveillance.head())
print(f"\nData summary:")
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
print(f"1. patient_vitals.csv: {len(patient_vitals):,} records (daily, 1 year)")
print(f"2. icu_monitoring.csv: {len(icu_monitoring):,} records (hourly, 6 months)")
print(f"3. disease_surveillance.csv: {len(disease_surveillance):,} records (monthly, 3 years)")
print("\nAll datasets saved to the 'data/' directory")
print("Ready for time series analysis!")
```
