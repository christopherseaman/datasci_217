# Data Generator for Assignment 10

This notebook generates the patient dataset used in the assignment.

```python
import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('data', exist_ok=True)

# Generate patient data
n_patients = 2000

data = {
    'patient_id': [f'PAT_{i:04d}' for i in range(1, n_patients + 1)],
    'age': np.random.normal(65, 15, n_patients).astype(int),
    'bmi': np.random.normal(28, 6, n_patients),
    'chronic_conditions': np.random.poisson(2, n_patients),
    'medication_count': np.random.poisson(5, n_patients),
    'hospital_stay_days': np.random.gamma(3, 2, n_patients).astype(int) + 1
}

df = pd.DataFrame(data)

# Create target variable: readmission risk score
# Model: risk = 20 + 0.5*age + 1.2*bmi + 3*chronic_conditions + 0.8*medication_count + 1.5*hospital_stay_days + noise
df['readmission_risk'] = (
    20 +
    0.5 * df['age'] +
    1.2 * df['bmi'] +
    3.0 * df['chronic_conditions'] +
    0.8 * df['medication_count'] +
    1.5 * df['hospital_stay_days'] +
    np.random.normal(0, 10, n_patients)
)

# Clip to reasonable range
df['readmission_risk'] = df['readmission_risk'].clip(0, 100)

# Save to CSV
df.to_csv('data/patient_data.csv', index=False)

print(f"Generated {len(df)} patient records")
print(f"\nDataset saved to: data/patient_data.csv")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nSummary statistics:")
print(df.describe())
```

