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

# Healthcare Data Generator for Assignment 8

## Overview
This notebook generates comprehensive healthcare datasets for data aggregation analysis including Electronic Health Records (EHR), clinical trials, and medical sensor data.

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

print("Healthcare Data Generator - Setup Complete")
```

## Generate Patient Data

```python
# Generate patient demographics
n_patients = 5000
patient_ids = [f'P{i:06d}' for i in range(1, n_patients + 1)]

# Generate ages with realistic distribution
ages = np.random.choice(
    np.concatenate([
        np.random.randint(0, 18, 800),      # Pediatric
        np.random.randint(18, 35, 1200),    # Young Adult
        np.random.randint(35, 50, 1500),    # Middle Age
        np.random.randint(50, 65, 1000),    # Senior
        np.random.randint(65, 100, 500)     # Elderly
    ])
)

# Generate other demographics
genders = np.random.choice(['M', 'F'], n_patients, p=[0.48, 0.52])
insurance_types = np.random.choice(['Private', 'Medicare', 'Medicaid', 'Uninsured'], 
                                 n_patients, p=[0.4, 0.3, 0.2, 0.1])

# Create patient data
patient_data = pd.DataFrame({
    'patient_id': patient_ids,
    'age': ages,
    'gender': genders,
    'insurance_type': insurance_types,
    'registration_date': pd.date_range('2020-01-01', periods=n_patients, freq='D')
})

print(f"Generated {len(patient_data)} patient records")
print(patient_data.head())
```

## Generate Admission Data

```python
# Generate hospital admissions
n_admissions = 8000
admission_ids = [f'A{i:06d}' for i in range(1, n_admissions + 1)]

# Randomly assign patients to admissions
admission_patients = np.random.choice(patient_data['patient_id'], n_admissions)

# Generate admission dates (last 2 years)
admission_dates = pd.date_range('2023-01-01', periods=n_admissions, freq='h')
admission_dates = np.random.choice(admission_dates, n_admissions)

# Generate admission types and departments
admission_types = np.random.choice(['Emergency', 'Elective', 'Urgent'], n_admissions, p=[0.4, 0.4, 0.2])
departments = np.random.choice(['Cardiology', 'Neurology', 'Oncology', 'Orthopedics', 'Pediatrics', 'ICU'], n_admissions)

# Generate length of stay (days)
length_of_stay = np.random.exponential(3, n_admissions)
length_of_stay = np.maximum(1, np.round(length_of_stay)).astype(int)

# Create admission data
admission_data = pd.DataFrame({
    'admission_id': admission_ids,
    'patient_id': admission_patients,
    'admission_date': admission_dates,
    'admission_type': admission_types,
    'department': departments,
    'length_of_stay': length_of_stay,
    'discharge_date': [pd.Timestamp(admission_dates[i]) + timedelta(days=int(length_of_stay[i])) for i in range(n_admissions)]
})

print(f"Generated {len(admission_data)} admission records")
print(admission_data.head())
```

## Generate Diagnosis Data

```python
# Generate diagnosis codes and descriptions
diagnosis_codes = [
    'I10', 'E11', 'M79', 'K21', 'J44', 'N18', 'F32', 'G43', 'M25', 'H52',
    'I25', 'E78', 'M17', 'K59', 'J45', 'N39', 'F41', 'G47', 'M79', 'H25'
]

diagnosis_descriptions = [
    'Essential hypertension', 'Type 2 diabetes mellitus', 'Fibromyalgia', 'Gastro-esophageal reflux disease',
    'Chronic obstructive pulmonary disease', 'Chronic kidney disease', 'Major depressive disorder',
    'Migraine', 'Osteoarthritis', 'Refractive errors', 'Ischemic heart disease', 'Disorders of lipoprotein metabolism',
    'Osteoarthritis of knee', 'Constipation', 'Asthma', 'Other disorders of urinary system',
    'Generalized anxiety disorder', 'Sleep disorders', 'Other soft tissue disorders', 'Age-related cataract'
]

# Generate diagnosis records
n_diagnoses = 12000
diagnosis_ids = [f'D{i:06d}' for i in range(1, n_diagnoses + 1)]

# Randomly assign admissions to diagnoses
diagnosis_admissions = np.random.choice(admission_data['admission_id'], n_diagnoses)

# Generate diagnosis codes and descriptions
diagnosis_codes_list = np.random.choice(diagnosis_codes, n_diagnoses)
diagnosis_descriptions_list = [diagnosis_descriptions[diagnosis_codes.index(code)] for code in diagnosis_codes_list]

# Generate treatment plans
treatment_plans = np.random.choice(['Medication', 'Surgery', 'Physical Therapy', 'Counseling', 'Monitoring'], n_diagnoses)

# Create diagnosis data
diagnosis_data = pd.DataFrame({
    'diagnosis_id': diagnosis_ids,
    'admission_id': diagnosis_admissions,
    'diagnosis_code': diagnosis_codes_list,
    'diagnosis_description': diagnosis_descriptions_list,
    'treatment_plan': treatment_plans,
    'severity': np.random.choice(['Mild', 'Moderate', 'Severe'], n_diagnoses, p=[0.5, 0.3, 0.2])
})

print(f"Generated {len(diagnosis_data)} diagnosis records")
print(diagnosis_data.head())
```

## Generate Vital Signs Data

```python
# Generate vital signs readings
n_vitals = 15000
vital_ids = [f'V{i:06d}' for i in range(1, n_vitals + 1)]

# Randomly assign patients to vital signs
vital_patients = np.random.choice(patient_data['patient_id'], n_vitals)

# Generate vital signs types and values
vital_types = np.random.choice(['Heart Rate', 'Blood Pressure', 'Temperature', 'Oxygen Saturation', 'Respiratory Rate'], n_vitals)

# Generate realistic vital signs values
vital_values = []
for vital_type in vital_types:
    if vital_type == 'Heart Rate':
        vital_values.append(np.random.normal(75, 15))
    elif vital_type == 'Blood Pressure':
        vital_values.append(np.random.normal(120, 20))
    elif vital_type == 'Temperature':
        vital_values.append(np.random.normal(98.6, 1.5))
    elif vital_type == 'Oxygen Saturation':
        vital_values.append(np.random.normal(98, 2))
    else:  # Respiratory Rate
        vital_values.append(np.random.normal(16, 4))

# Generate timestamps
vital_timestamps = pd.date_range('2024-01-01', periods=n_vitals, freq='min')
vital_timestamps = np.random.choice(vital_timestamps, n_vitals)

# Create vital signs data
vital_signs = pd.DataFrame({
    'vital_id': vital_ids,
    'patient_id': vital_patients,
    'vital_type': vital_types,
    'vital_value': vital_values,
    'timestamp': vital_timestamps,
    'unit': ['bpm' if vt == 'Heart Rate' else 'mmHg' if vt == 'Blood Pressure' else '°F' if vt == 'Temperature' else '%' if vt == 'Oxygen Saturation' else 'breaths/min' for vt in vital_types]
})

print(f"Generated {len(vital_signs)} vital signs records")
print(vital_signs.head())
```

## Generate Clinical Trial Data

```python
# Generate clinical trials
n_trials = 50
trial_ids = [f'T{i:03d}' for i in range(1, n_trials + 1)]

# Generate trial information
trial_phases = np.random.choice(['Phase I', 'Phase II', 'Phase III', 'Phase IV'], n_trials, p=[0.1, 0.3, 0.4, 0.2])
trial_types = np.random.choice(['Drug Trial', 'Device Trial', 'Behavioral Trial', 'Surgical Trial'], n_trials)
trial_statuses = np.random.choice(['Recruiting', 'Active', 'Completed', 'Suspended'], n_trials, p=[0.2, 0.4, 0.3, 0.1])

# Create clinical trials data
clinical_trials = pd.DataFrame({
    'trial_id': trial_ids,
    'trial_name': [f'Clinical Trial {i}' for i in range(1, n_trials + 1)],
    'trial_phase': trial_phases,
    'trial_type': trial_types,
    'status': trial_statuses,
    'start_date': pd.date_range('2023-01-01', periods=n_trials, freq='M'),
    'estimated_duration_days': np.random.randint(30, 365, n_trials)
})

print(f"Generated {len(clinical_trials)} clinical trials")
print(clinical_trials.head())
```

## Generate Trial Participants Data

```python
# Generate trial participants
n_participants = 2000
participant_ids = [f'TP{i:06d}' for i in range(1, n_participants + 1)]

# Randomly assign participants to trials
participant_trials = np.random.choice(clinical_trials['trial_id'], n_participants)
participant_patients = np.random.choice(patient_data['patient_id'], n_participants)

# Generate participant information
treatment_groups = np.random.choice(['Control', 'Treatment A', 'Treatment B', 'Placebo'], n_participants, p=[0.25, 0.3, 0.3, 0.15])
enrollment_dates = pd.date_range('2023-01-01', periods=n_participants, freq='D')
enrollment_dates = np.random.choice(enrollment_dates, n_participants)

# Create trial participants data
trial_participants = pd.DataFrame({
    'participant_id': participant_ids,
    'trial_id': participant_trials,
    'patient_id': participant_patients,
    'treatment_group': treatment_groups,
    'enrollment_date': enrollment_dates,
    'follow_up_days': np.random.randint(30, 180, n_participants)
})

print(f"Generated {len(trial_participants)} trial participants")
print(trial_participants.head())
```

## Generate Trial Outcomes Data

```python
# Generate trial outcomes
n_outcomes = 2000
outcome_ids = [f'O{i:06d}' for i in range(1, n_outcomes + 1)]

# Generate outcome information
treatment_responses = np.random.choice(['Positive', 'Negative', 'Partial'], n_outcomes, p=[0.4, 0.3, 0.3])
adverse_events = np.random.choice(['Yes', 'No'], n_outcomes, p=[0.2, 0.8])
outcome_dates = pd.date_range('2023-01-01', periods=n_outcomes, freq='D')
outcome_dates = np.random.choice(outcome_dates, n_outcomes)

# Create trial outcomes data
trial_outcomes = pd.DataFrame({
    'outcome_id': outcome_ids,
    'participant_id': trial_participants['participant_id'],
    'treatment_response': treatment_responses,
    'adverse_events': adverse_events,
    'outcome_date': outcome_dates,
    'quality_of_life_score': np.random.uniform(1, 10, n_outcomes)
})

print(f"Generated {len(trial_outcomes)} trial outcomes")
print(trial_outcomes.head())
```

## Generate Medical Sensor Data

```python
# Generate medical sensors
n_sensors = 200
sensor_ids = [f'S{i:04d}' for i in range(1, n_sensors + 1)]

# Generate sensor information
sensor_types = np.random.choice(['Heart Rate Monitor', 'Blood Pressure Cuff', 'Temperature Sensor', 'Pulse Oximeter', 'ECG Monitor'], n_sensors)
sensor_locations = np.random.choice(['ICU', 'General Ward', 'Emergency', 'Operating Room', 'Outpatient'], n_sensors)

# Create medical sensors data
medical_sensors = pd.DataFrame({
    'sensor_id': sensor_ids,
    'sensor_type': sensor_types,
    'location': sensor_locations,
    'manufacturer': np.random.choice(['MedTech Inc', 'HealthCorp', 'BioSensors Ltd', 'MediDevice'], n_sensors),
    'installation_date': pd.date_range('2023-01-01', periods=n_sensors, freq='D')
})

print(f"Generated {len(medical_sensors)} medical sensors")
print(medical_sensors.head())
```

## Generate Sensor Readings Data

```python
# Generate sensor readings
n_readings = 50000
reading_ids = [f'R{i:07d}' for i in range(1, n_readings + 1)]

# Randomly assign sensors to readings
reading_sensors = np.random.choice(medical_sensors['sensor_id'], n_readings)
reading_patients = np.random.choice(patient_data['patient_id'], n_readings)

# Generate reading values based on sensor type
reading_values = []
for sensor_id in reading_sensors:
    sensor_type = medical_sensors[medical_sensors['sensor_id'] == sensor_id]['sensor_type'].iloc[0]
    if sensor_type == 'Heart Rate Monitor':
        reading_values.append(np.random.normal(75, 15))
    elif sensor_type == 'Blood Pressure Cuff':
        reading_values.append(np.random.normal(120, 20))
    elif sensor_type == 'Temperature Sensor':
        reading_values.append(np.random.normal(98.6, 1.5))
    elif sensor_type == 'Pulse Oximeter':
        reading_values.append(np.random.normal(98, 2))
    else:  # ECG Monitor
        reading_values.append(np.random.normal(1.0, 0.2))

# Generate timestamps
reading_timestamps = pd.date_range('2024-01-01', periods=n_readings, freq='min')
reading_timestamps = np.random.choice(reading_timestamps, n_readings)

# Create sensor readings data
sensor_readings = pd.DataFrame({
    'reading_id': reading_ids,
    'sensor_id': reading_sensors,
    'patient_id': reading_patients,
    'reading_value': reading_values,
    'timestamp': reading_timestamps,
    'quality_score': np.random.uniform(0.8, 1.0, n_readings)
})

print(f"Generated {len(sensor_readings)} sensor readings")
print(sensor_readings.head())
```

## Save All Datasets

```python
# Save all datasets to CSV files
datasets = {
    'patient_data.csv': patient_data,
    'admission_data.csv': admission_data,
    'diagnosis_data.csv': diagnosis_data,
    'vital_signs.csv': vital_signs,
    'clinical_trials.csv': clinical_trials,
    'trial_participants.csv': trial_participants,
    'trial_outcomes.csv': trial_outcomes,
    'medical_sensors.csv': medical_sensors,
    'sensor_readings.csv': sensor_readings
}

print("Saving datasets...")
for filename, df in datasets.items():
    filepath = f'data/{filename}'
    df.to_csv(filepath, index=False)
    print(f"✅ Saved {filename}: {len(df)} records")

print("\n=== Healthcare Data Generation Complete ===")
print("Generated datasets:")
for filename, df in datasets.items():
    print(f"- {filename}: {len(df):,} records")

print("\nAll healthcare datasets ready for analysis!")
```
