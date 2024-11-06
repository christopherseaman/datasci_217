import random
from datetime import datetime, timedelta
import numpy as np

# Constants
EDUCATION_LEVELS = ['High School', 'Some College', 'Bachelors', 'Graduate']
START_DATE = datetime(2020, 1, 1)  # Study starts
END_DATE = datetime(2024, 1, 1)    # Study ends
NUM_PATIENTS = 1000

# Extra metadata
DATA_ENTRY_STAFF = ['JSmith', 'MJones', 'RWilson', 'KLee']
SITE = 'UCSF Medical Center'
FORM_VERSION = 'v2.1'

# Effects on T25FW speed (feet/second)
AGE_EFFECT = -0.03        # Speed decreases 0.03 f/s per year of age
EDUCATION_EFFECT = {
    'High School': 0,     # baseline
    'Some College': 0.4,  # better outcomes
    'Bachelors': 0.8,    # even better outcomes
    'Graduate': 1.2       # best outcomes
}
BASE_SPEED = 5.0          # baseline walking speed for young person with basic education

# Generate patients (not output yet)
patients = []
for i in range(NUM_PATIENTS):
    patient_id = f'P{str(i+1).zfill(4)}'
    
    # Generate birthday (between 1940 and 2000)
    birth_year = random.randint(1940, 2000)
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)  # avoid edge cases with month lengths
    birthday = datetime(birth_year, birth_month, birth_day)
    
    # Assign persistent education level
    education = random.choice(EDUCATION_LEVELS)
    
    patients.append({
        'id': patient_id,
        'birthday': birthday,
        'education': education
    })

# Generate visits and measurements
visits = []
for patient in patients:
    # Start with a random first visit date
    current_date = START_DATE + timedelta(days=random.randint(0, 90))
    
    while current_date < END_DATE:
        # Calculate age at visit
        age = (current_date - patient['birthday']).days / 365.25
        
        # Calculate walking speed with systematic variation and noise
        speed_age_effect = age * AGE_EFFECT
        speed_education_effect = EDUCATION_EFFECT[patient['education']]
        
        # Add disease progression effect (slight decline over time)
        years_in_study = (current_date - START_DATE).days / 365.25
        progression_effect = -0.1 * years_in_study  # -0.1 f/s per year in study
        
        # Add seasonal variation (slightly better in summer)
        month = current_date.month
        seasonal_effect = 0.2 * np.sin(2 * np.pi * (month - 6) / 12)  # peak in July
        
        # Random noise includes day-to-day variation
        speed_random_noise = np.random.normal(0, 0.3)  # random variation ±0.3 f/s
        
        speed = (BASE_SPEED + 
                speed_age_effect + 
                speed_education_effect + 
                progression_effect + 
                seasonal_effect + 
                speed_random_noise)
        
        speed = max(2.0, min(8.0, speed))  # clamp to realistic range
        
        # Generate extra metadata
        room = f"Exam-{random.randint(1,12):02d}"
        hour = random.randint(9, 16)  # 9 AM to 4 PM
        minute = random.randint(0, 59)
        time = f"{hour:02d}:{minute:02d}"
        staff = random.choice(DATA_ENTRY_STAFF)
        
        visits.append({
            'patient_id': patient['id'],
            'date': current_date.strftime('%Y-%m-%d'),
            'time': time,
            'age': round(age, 2),
            'education': patient['education'],
            'walking_speed': round(speed, 2),
            'site': SITE,
            'room': room,
            'staff': staff,
            'form_version': FORM_VERSION
        })
        
        # Next visit in roughly 90 ±15 days
        days_to_next = random.randint(75, 105)
        current_date += timedelta(days=days_to_next)

# Write dirty data file
with open('ms_data_dirty.csv', 'w') as f:
    # Add comments and headers with extra commas
    f.write('# Multiple Sclerosis Study Data Export\n')
    f.write('# Generated: 2024-01-15\n')
    f.write('# Timed 25-Foot Walk Test measurements across patient visits\n\n')
    f.write('patient_id,visit_date,time,age,education_level,walking_speed,site,room,staff_id,form_version\n')
    
    # Write visits with intentional noise
    for visit in sorted(visits, key=lambda x: (x['patient_id'], x['date'])):
        # Oops, they skipped a visit
        if random.random() < 0.05:
            continue

        # Sometimes add empty lines
        if random.random() < 0.05:
            f.write('\n')
            
        # Sometimes add comments
        if random.random() < 0.05:
            f.write('# Quality check: measurement verified\n')
        
        # Sometimes add extra commas
        if random.random() < 0.1:
            line = f"{visit['patient_id']},{visit['date']},{visit['time']},,{visit['age']},{visit['education']},{visit['walking_speed']},{visit['site']},{visit['room']},{visit['staff']},{visit['form_version']}\n"
        else:
            line = f"{visit['patient_id']},{visit['date']},{visit['time']},{visit['age']},{visit['education']},{visit['walking_speed']},{visit['site']},{visit['room']},{visit['staff']},{visit['form_version']}\n"
            
        f.write(line)
