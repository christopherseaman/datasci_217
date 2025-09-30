#!/usr/bin/env python3
"""
Health Sensor Data Generator

Generates realistic health sensor data for Assignment 03.
Creates a CSV file with 50,000 health readings from 10,000 patients.
"""

import random
import csv
from datetime import datetime, timedelta

def generate_patient_id(num):
    """Generate patient ID in format P00001-P10000."""
    return f"P{num:05d}"

def generate_sensor_id(num):
    """Generate sensor ID in format S001-S050."""
    return f"S{num:03d}"

def generate_timestamp(base_date, offset_hours):
    """Generate timestamp for a reading."""
    timestamp = base_date + timedelta(hours=offset_hours)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S")

def generate_health_reading(patient_num, reading_num, base_date):
    """Generate a single health reading with realistic values."""
    # Set seed for reproducibility but vary by reading
    random.seed(42 + reading_num)
    
    patient_id = generate_patient_id(random.randint(1, 10000))
    sensor_id = generate_sensor_id(random.randint(1, 50))
    
    # Timestamps distributed over a week (168 hours)
    timestamp = generate_timestamp(base_date, random.uniform(0, 168))
    
    # Generate realistic health metrics
    heart_rate = random.randint(60, 100)  # Normal range: 60-100 bpm
    bp_systolic = random.randint(110, 140)  # Systolic BP: 110-140 mmHg
    bp_diastolic = random.randint(70, 90)   # Diastolic BP: 70-90 mmHg
    temperature = round(random.uniform(97.0, 99.5), 1)  # Temperature: 97.0-99.5°F
    glucose = random.randint(70, 130)  # Glucose: 70-130 mg/dL
    
    return {
        'patient_id': patient_id,
        'timestamp': timestamp,
        'heart_rate': heart_rate,
        'blood_pressure_systolic': bp_systolic,
        'blood_pressure_diastolic': bp_diastolic,
        'temperature': temperature,
        'glucose_level': glucose,
        'sensor_id': sensor_id
    }

def main():
    """Generate health sensor data and save to CSV."""
    print("Generating health sensor data...")
    
    # Configuration
    NUM_READINGS = 50000
    OUTPUT_FILE = 'health_data.csv'
    BASE_DATE = datetime(2024, 1, 15, 8, 0, 0)
    
    # CSV columns
    fieldnames = [
        'patient_id',
        'timestamp',
        'heart_rate',
        'blood_pressure_systolic',
        'blood_pressure_diastolic',
        'temperature',
        'glucose_level',
        'sensor_id'
    ]
    
    # Generate data and write to CSV
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(NUM_READINGS):
            reading = generate_health_reading(i % 10000, i, BASE_DATE)
            writer.writerow(reading)
            
            # Progress indicator
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1:,} readings...")
    
    print(f"\n✓ Successfully generated {NUM_READINGS:,} health readings")
    print(f"✓ Data saved to: {OUTPUT_FILE}")
    print(f"\nFile contains:")
    print(f"  - 50,000 health sensor readings")
    print(f"  - 10,000 unique patients (P00001-P10000)")
    print(f"  - 50 sensors (S001-S050)")
    print(f"  - Timestamps over a 7-day period")
    print(f"\nYou can now proceed with Part 1 of the assignment.")

if __name__ == "__main__":
    main()