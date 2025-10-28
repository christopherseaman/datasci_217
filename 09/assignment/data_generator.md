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
This notebook generates comprehensive healthcare time series datasets for analysis including patient monitoring, vital signs, and medication schedules.

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

## Generate Patient Monitoring Data

```python
# Generate continuous patient monitoring data
n_patients = 100
n_hours = 24 * 30  # 30 days of hourly data

# Create patient IDs
patient_ids = [f'P{i:06d}' for i in range(1, n_patients + 1)]

# Generate hourly timestamps for the last 30 days
end_time = datetime.now()
start_time = end_time - timedelta(days=30)
timestamps = pd.date_range(start_time, end_time, freq='H')

# Generate monitoring data for each patient
monitoring_data = []

for patient_id in patient_ids:
    # Generate patient-specific baseline values
    baseline_hr = np.random.normal(75, 10)
    baseline_bp = np.random.normal(120, 15)
    baseline_temp = np.random.normal(98.6, 0.5)
    baseline_oxygen = np.random.normal(98, 2)
    
    # Generate time series with trends and seasonality
    for timestamp in timestamps:
        # Add daily seasonality (lower HR at night)
        hour = timestamp.hour
        daily_factor = 1 + 0.1 * np.sin(2 * np.pi * hour / 24)
        
        # Add weekly seasonality (weekend effect)
        day_of_week = timestamp.weekday()
        weekly_factor = 1 + 0.05 * np.sin(2 * np.pi * day_of_week / 7)
        
        # Add random noise
        noise_factor = np.random.normal(1, 0.1)
        
        # Generate vital signs
        heart_rate = max(40, min(200, baseline_hr * daily_factor * weekly_factor * noise_factor))
        blood_pressure = max(80, min(200, baseline_bp * daily_factor * weekly_factor * noise_factor))
        temperature = max(95, min(105, baseline_temp + np.random.normal(0, 0.5)))
        oxygen_saturation = max(85, min(100, baseline_oxygen + np.random.normal(0, 2)))
        
        monitoring_data.append({
            'patient_id': patient_id,
            'timestamp': timestamp,
            'heart_rate': heart_rate,
            'blood_pressure': blood_pressure,
            'temperature': temperature,
            'oxygen_saturation': oxygen_saturation,
            'room_number': np.random.randint(100, 500),
            'nurse_id': f'N{np.random.randint(1, 50):03d}'
        })

# Create DataFrame
patient_monitoring = pd.DataFrame(monitoring_data)

print(f"Generated {len(patient_monitoring)} patient monitoring records")
print(patient_monitoring.head())
```

## Generate Continuous Vital Signs Data

```python
# Generate high-frequency vital signs data (every 15 minutes)
n_patients_vitals = 20  # Subset of patients with continuous monitoring
vitals_patients = np.random.choice(patient_ids, n_patients_vitals, replace=False)

# Generate 15-minute timestamps
vitals_timestamps = pd.date_range(start_time, end_time, freq='15min')

vitals_data = []

for patient_id in vitals_patients:
    # Get baseline from monitoring data
    patient_baseline = patient_monitoring[patient_monitoring['patient_id'] == patient_id].iloc[0]
    
    for timestamp in vitals_timestamps:
        # Add more granular patterns
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Circadian rhythm
        circadian_factor = 1 + 0.15 * np.sin(2 * np.pi * (hour + minute/60) / 24)
        
        # Add medication effects (simulate)
        medication_effect = 1 + 0.05 * np.sin(2 * np.pi * timestamp.hour / 6)  # Every 6 hours
        
        # Generate vital signs with higher frequency
        heart_rate = max(40, min(200, patient_baseline['heart_rate'] * circadian_factor * medication_effect + np.random.normal(0, 5)))
        blood_pressure = max(80, min(200, patient_baseline['blood_pressure'] * circadian_factor + np.random.normal(0, 8)))
        temperature = max(95, min(105, patient_baseline['temperature'] + np.random.normal(0, 0.3)))
        oxygen_saturation = max(85, min(100, patient_baseline['oxygen_saturation'] + np.random.normal(0, 1.5)))
        
        vitals_data.append({
            'patient_id': patient_id,
            'timestamp': timestamp,
            'heart_rate': heart_rate,
            'blood_pressure': blood_pressure,
            'temperature': temperature,
            'oxygen_saturation': oxygen_saturation,
            'sensor_id': f'S{np.random.randint(1, 20):03d}',
            'quality_score': np.random.uniform(0.8, 1.0)
        })

# Create DataFrame
vital_signs_continuous = pd.DataFrame(vitals_data)

print(f"Generated {len(vital_signs_continuous)} continuous vital signs records")
print(vital_signs_continuous.head())
```

## Generate Medication Schedule Data

```python
# Generate medication schedule data
medications = [
    'Metformin', 'Lisinopril', 'Atorvastatin', 'Metoprolol', 'Amlodipine',
    'Omeprazole', 'Losartan', 'Simvastatin', 'Hydrochlorothiazide', 'Sertraline'
]

medication_data = []

for patient_id in patient_ids:
    # Random number of medications per patient
    n_meds = np.random.randint(1, 5)
    patient_meds = np.random.choice(medications, n_meds, replace=False)
    
    for med in patient_meds:
        # Medication start date
        start_date = start_time + timedelta(days=np.random.randint(0, 25))
        
        # Generate medication schedule (daily, twice daily, etc.)
        frequency = np.random.choice(['daily', 'twice_daily', 'three_times_daily'], p=[0.6, 0.3, 0.1])
        
        if frequency == 'daily':
            times = [8]  # 8 AM
        elif frequency == 'twice_daily':
            times = [8, 20]  # 8 AM, 8 PM
        else:  # three_times_daily
            times = [8, 14, 20]  # 8 AM, 2 PM, 8 PM
        
        # Generate medication records
        current_date = start_date
        while current_date <= end_time:
            for time_hour in times:
                med_timestamp = current_date.replace(hour=time_hour, minute=0, second=0)
                
                # Add some randomness to timing
                med_timestamp += timedelta(minutes=np.random.randint(-30, 30))
                
                medication_data.append({
                    'patient_id': patient_id,
                    'medication': med,
                    'timestamp': med_timestamp,
                    'dosage': f"{np.random.randint(1, 4)}mg",
                    'frequency': frequency,
                    'administered': np.random.choice([True, False], p=[0.95, 0.05])  # 95% compliance
                })
            
            current_date += timedelta(days=1)

# Create DataFrame
medication_schedule = pd.DataFrame(medication_data)

print(f"Generated {len(medication_schedule)} medication records")
print(medication_schedule.head())
```

## Generate Clinical Trial Time Series Data

```python
# Generate clinical trial time series data
n_trials = 10
trial_ids = [f'T{i:03d}' for i in range(1, n_trials + 1)]

# Generate trial participants
trial_participants = []
for trial_id in trial_ids:
    n_participants = np.random.randint(20, 50)
    trial_patients = np.random.choice(patient_ids, n_participants, replace=False)
    
    for patient_id in trial_patients:
        # Trial start date
        trial_start = start_time + timedelta(days=np.random.randint(0, 20))
        
        # Generate weekly measurements
        current_date = trial_start
        measurement_count = 0
        
        while current_date <= end_time and measurement_count < 12:  # Max 12 weeks
            # Generate measurement data
            measurement_data = {
                'trial_id': trial_id,
                'patient_id': patient_id,
                'measurement_date': current_date,
                'weight': np.random.normal(70, 15) + np.random.normal(0, 2),  # Weight with trend
                'blood_pressure_systolic': np.random.normal(120, 15),
                'blood_pressure_diastolic': np.random.normal(80, 10),
                'heart_rate': np.random.normal(75, 10),
                'cholesterol': np.random.normal(200, 30),
                'glucose': np.random.normal(100, 20),
                'side_effects': np.random.choice(['None', 'Mild', 'Moderate', 'Severe'], p=[0.6, 0.25, 0.1, 0.05]),
                'adherence_score': np.random.uniform(0.7, 1.0)
            }
            
            trial_participants.append(measurement_data)
            current_date += timedelta(weeks=1)
            measurement_count += 1

# Create DataFrame
clinical_trial_data = pd.DataFrame(trial_participants)

print(f"Generated {len(clinical_trial_data)} clinical trial measurements")
print(clinical_trial_data.head())
```

## Generate Medical Device Sensor Data

```python
# Generate medical device sensor data
device_types = ['ECG', 'Pulse Oximeter', 'Blood Pressure Monitor', 'Temperature Sensor', 'Respiratory Monitor']
n_devices = 50

device_data = []
for device_id in range(1, n_devices + 1):
    device_type = np.random.choice(device_types)
    
    # Generate device readings every 5 minutes
    device_timestamps = pd.date_range(start_time, end_time, freq='5min')
    
    for timestamp in device_timestamps:
        # Generate device-specific readings
        if device_type == 'ECG':
            reading = np.random.normal(1.0, 0.1)  # ECG amplitude
        elif device_type == 'Pulse Oximeter':
            reading = np.random.normal(98, 2)  # Oxygen saturation
        elif device_type == 'Blood Pressure Monitor':
            reading = np.random.normal(120, 15)  # Systolic pressure
        elif device_type == 'Temperature Sensor':
            reading = np.random.normal(98.6, 0.5)  # Temperature
        else:  # Respiratory Monitor
            reading = np.random.normal(16, 3)  # Respiratory rate
        
        device_data.append({
            'device_id': f'D{device_id:03d}',
            'device_type': device_type,
            'timestamp': timestamp,
            'reading': reading,
            'patient_id': np.random.choice(patient_ids),
            'room_id': f'R{np.random.randint(100, 500)}',
            'battery_level': np.random.uniform(0.2, 1.0),
            'signal_quality': np.random.uniform(0.7, 1.0)
        })

# Create DataFrame
medical_devices = pd.DataFrame(device_data)

print(f"Generated {len(medical_devices)} medical device readings")
print(medical_devices.head())
```

## Generate Patient Outcomes Data

```python
# Generate patient outcomes data
outcomes_data = []

for patient_id in patient_ids:
    # Patient admission date
    admission_date = start_time + timedelta(days=np.random.randint(0, 25))
    
    # Length of stay
    length_of_stay = np.random.exponential(5)  # Average 5 days
    discharge_date = admission_date + timedelta(days=int(length_of_stay))
    
    # Generate daily outcomes
    current_date = admission_date
    while current_date <= min(discharge_date, end_time):
        # Generate outcome metrics
        pain_score = np.random.randint(0, 11)  # 0-10 pain scale
        mobility_score = np.random.randint(1, 6)  # 1-5 mobility scale
        cognitive_score = np.random.randint(15, 30)  # Mini-mental state exam
        
        # Generate complications
        complications = []
        if np.random.random() < 0.1:  # 10% chance of complication
            complications.append(np.random.choice(['Infection', 'Bleeding', 'Allergic Reaction', 'Medication Error']))
        
        outcomes_data.append({
            'patient_id': patient_id,
            'date': current_date,
            'pain_score': pain_score,
            'mobility_score': mobility_score,
            'cognitive_score': cognitive_score,
            'complications': '; '.join(complications) if complications else 'None',
            'length_of_stay': int(length_of_stay),
            'discharge_date': discharge_date,
            'readmission_risk': np.random.uniform(0, 1)
        })
        
        current_date += timedelta(days=1)

# Create DataFrame
patient_outcomes = pd.DataFrame(outcomes_data)

print(f"Generated {len(patient_outcomes)} patient outcome records")
print(patient_outcomes.head())
```

## Save All Datasets

```python
# Save all datasets to CSV files
datasets = {
    'patient_monitoring.csv': patient_monitoring,
    'vital_signs_continuous.csv': vital_signs_continuous,
    'medication_schedule.csv': medication_schedule,
    'clinical_trial_data.csv': clinical_trial_data,
    'medical_devices.csv': medical_devices,
    'patient_outcomes.csv': patient_outcomes
}

print("Saving healthcare time series datasets...")
for filename, df in datasets.items():
    filepath = f'data/{filename}'
    df.to_csv(filepath, index=False)
    print(f"✅ Saved {filename}: {len(df)} records")

print("\n=== Healthcare Time Series Data Generation Complete ===")
print("Generated datasets:")
for filename, df in datasets.items():
    print(f"- {filename}: {len(df):,} records")

print("\nAll healthcare time series datasets ready for analysis!")
print("Data includes:")
print("- Patient monitoring (hourly vital signs)")
print("- Continuous vital signs (15-minute intervals)")
print("- Medication schedules")
print("- Clinical trial measurements")
print("- Medical device sensor data")
print("- Patient outcomes and complications")
```
