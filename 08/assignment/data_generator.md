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
This notebook generates healthcare/insurance EHR data for Assignment 8. **Run all cells** to create the three required datasets:
- `provider_data.csv` - Healthcare providers (doctors, nurses, etc.)
- `facility_data.csv` - Healthcare facilities (hospitals, clinics, etc.)
- `encounter_data.csv` - Patient encounters/insurance claims

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

## Generate Assignment Healthcare/Insurance Data

```python
# Generate healthcare/insurance EHR data for aggregation analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Healthcare facilities (hospitals, clinics, etc.)
facility_types = ["Hospital", "Clinic", "Urgent Care", "Specialty Center"]
facility_names = [
    "City General Hospital",
    "Community Health Clinic",
    "Regional Medical Center",
    "Northside Urgent Care",
    "Cardiology Specialty Center",
    "Oncology Treatment Center",
    "Pediatric Care Center",
    "Emergency Medical Center",
    "Primary Care Clinic",
    "Surgical Specialty Center",
]
regions = ["North", "South", "East", "West"]

facilities = pd.DataFrame({
    "facility_id": [f"FAC{i:03d}" for i in range(1, len(facility_names) + 1)],
    "facility_name": facility_names,
    "facility_type": np.random.choice(facility_types, len(facility_names)),
    "region": np.random.choice(regions, len(facility_names)),
    "beds": np.random.randint(0, 500, len(facility_names)),
    "established_date": [
        (datetime(1980, 1, 1) + timedelta(days=int(np.random.uniform(0, 365 * 44))))
        .date()
        .isoformat()
        for _ in range(len(facility_names))
    ],
})

# Healthcare providers (doctors, nurses, etc.)
num_providers = 500
provider_types = ["Physician", "Nurse Practitioner", "Physician Assistant", "Nurse", "Therapist"]
specialties = [
    "Cardiology",
    "Oncology",
    "Neurology",
    "Pediatrics",
    "Orthopedics",
    "Dermatology",
    "Emergency Medicine",
    "Radiology",
    "Psychiatry",
    "Endocrinology",
    "Primary Care",
    "General Practice",
]
first_names = [
    "Alex", "Jordan", "Taylor", "Casey", "Riley", "Morgan",
    "Jamie", "Avery", "Cameron", "Drew", "Sam", "Quinn",
]
last_names = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller",
    "Davis", "Garcia", "Rodriguez", "Wilson", "Martinez", "Anderson",
]

provider_ids = [f"PR{i:04d}" for i in range(1, num_providers + 1)]
providers = pd.DataFrame({
    "provider_id": provider_ids,
    "provider_name": [
        f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
        for _ in range(num_providers)
    ],
    "provider_type": np.random.choice(
        provider_types, num_providers, p=[0.3, 0.2, 0.15, 0.25, 0.1]
    ),
    "facility_id": np.random.choice(facilities["facility_id"], num_providers),
    "specialty": np.random.choice(specialties, num_providers),
    "years_experience": np.random.randint(1, 40, num_providers),
    "license_number": [f"LIC{i:06d}" for i in range(1, num_providers + 1)],
})

# Patient encounters (insurance claims/EHR records)
num_encounters = 5000
encounter_types = ["Office Visit", "Emergency", "Inpatient", "Outpatient Procedure", "Lab"]

# Common ICD-10 diagnosis codes with realistic procedure mappings
# Each diagnosis can have associated procedures
diagnosis_procedure_map = {
    "E11.9": ["99213", "99214", "80053", "85025", "36415"],  # Type 2 diabetes - office visits, labs
    "I10": ["99213", "99214", "80053", "93000"],  # Hypertension - office visits, EKG, labs
    "K21.9": ["99213", "99214", "80053"],  # GERD - office visits, labs
    "J44.9": ["99213", "99214", "72040", "80053", "36415"],  # COPD - office visits, chest X-ray, labs
    "N18.9": ["99213", "99214", "80053", "85025"],  # Chronic kidney disease - office visits, labs
    "F32.9": ["99214", "99213"],  # Major depressive disorder - office visits
    "G43.9": ["99213", "99214", "70450"],  # Migraine - office visits, CT head
    "M25.5": ["99213", "99214", "72040"],  # Joint pain - office visits, X-ray
    "H52.9": ["99213", "99214"],  # Refractive disorder - office visits
    "M79.3": ["99213", "99214"],  # Panniculitis - office visits
}

# Flatten for random selection, but keep realistic pairs
diagnosis_codes = list(diagnosis_procedure_map.keys())
all_procedure_codes = [
    "99213", "99214", "99283", "36415", "80053", "85025", 
    "93000", "72040", "70450", "70551"
]

start_date = datetime(2023, 1, 1)

# Generate encounters with realistic diagnosis-procedure pairs
encounter_list = []
for i in range(1, num_encounters + 1):
    diagnosis = np.random.choice(diagnosis_codes)
    # 70% chance of using realistic procedure, 30% random
    if np.random.random() < 0.7 and diagnosis in diagnosis_procedure_map:
        procedure = np.random.choice(diagnosis_procedure_map[diagnosis])
    else:
        procedure = np.random.choice(all_procedure_codes)
    
    encounter_type = np.random.choice(encounter_types)
    
    # Adjust charge based on encounter type and procedure
    if procedure.startswith("99"):  # Office/Emergency visits
        base_charge = np.random.uniform(100, 500)
    elif procedure in ["70450", "70551"]:  # Imaging
        base_charge = np.random.uniform(500, 3000)
    elif procedure in ["80053", "85025"]:  # Labs
        base_charge = np.random.uniform(50, 300)
    else:
        base_charge = np.random.uniform(50, 500)
    
    # Adjust for encounter type
    if encounter_type == "Emergency":
        base_charge *= 2
    elif encounter_type == "Inpatient":
        base_charge *= 3
    
    encounter_list.append({
        "encounter_id": f"ENC{i:05d}",
        "patient_id": f"PAT{np.random.randint(1, 2000):05d}",
        "provider_id": np.random.choice(providers["provider_id"]),
        "facility_id": np.random.choice(facilities["facility_id"]),
        "encounter_date": (start_date + timedelta(days=int(np.random.uniform(0, 365)))).date().isoformat(),
        "encounter_type": encounter_type,
        "diagnosis_code": diagnosis,
        "procedure_code": procedure,
        "service_charge": round(base_charge, 2),
    })

encounters = pd.DataFrame(encounter_list)

# Calculate insurance and patient payments (insurance typically covers 70-90%)
coverage_rate = np.random.uniform(0.70, 0.90, num_encounters)
encounters["insurance_paid"] = (encounters["service_charge"] * coverage_rate).round(2)
encounters["patient_paid"] = (encounters["service_charge"] - encounters["insurance_paid"]).round(2)

# Derive region from facility
facility_region_map = dict(zip(facilities["facility_id"], facilities["region"]))
encounters["region"] = encounters["facility_id"].map(facility_region_map)

# Write assignment datasets
out_dir = "data"
providers.to_csv(f"{out_dir}/provider_data.csv", index=False)
facilities.to_csv(f"{out_dir}/facility_data.csv", index=False)
encounters.to_csv(f"{out_dir}/encounter_data.csv", index=False)

print("\n=== Healthcare/Insurance EHR Data Generated ===")
print(f"✅ Saved provider_data.csv: {len(providers)} records")
print(f"✅ Saved facility_data.csv: {len(facilities)} records")
print(f"✅ Saved encounter_data.csv: {len(encounters)} records")
print("\nAssignment datasets ready!")
```
