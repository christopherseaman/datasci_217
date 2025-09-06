"""Generate synthetic health data for modeling assignment"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

# Define column names in snake_case
BASELINE_COLS = [
    'patient_id', 'age', 'sex', 'bmi', 'smoking', 'diabetes',
    'bp_systolic', 'cholesterol', 'heart_rate'
]

LONGITUDINAL_COLS = [
    'patient_id', 'visit_date', 'bp_systolic', 'heart_rate',
    'adverse_event', 'age', 'sex', 'bmi', 'smoking', 'diabetes'
]

TREATMENT_COLS = [
    'patient_id', 'age', 'sex', 'bmi', 'smoking', 'diabetes',
    'bp_systolic', 'cholesterol', 'heart_rate', 'treatment',
    'adherence', 'outcome'
]

def generate_patient_cohort(n_patients=10000, random_seed=42):
    """Generate a cohort of patients with baseline characteristics."""
    np.random.seed(random_seed)
    
    # Generate baseline characteristics
    age = np.random.normal(50, 15, n_patients)
    age = np.clip(age, 18, 90)  # Restrict to adult patients
    
    sex = np.random.binomial(1, 0.5, n_patients)
    
    # BMI correlated with age
    bmi_mean = 25 + 0.04 * age
    bmi = np.random.normal(bmi_mean, 5, n_patients)
    bmi = np.clip(bmi, 15, 50)
    
    # Smoking more common in middle age
    smoking_prob = 0.2 + 0.2 * np.exp(-(age - 45)**2 / 200)
    smoking = np.random.binomial(1, smoking_prob, n_patients)
    
    # Diabetes risk increases with age and BMI
    diabetes_prob = 0.05 + 0.3 * (age > 50) + 0.3 * (bmi > 30)
    diabetes_prob = np.clip(diabetes_prob, 0, 1)
    diabetes = np.random.binomial(1, diabetes_prob, n_patients)
    
    # Blood pressure affected by age, BMI, smoking, diabetes
    bp_mean = 110 + 0.3 * age + 0.2 * bmi + 5 * smoking + 10 * diabetes
    bp_systolic = np.random.normal(bp_mean, 10, n_patients)
    
    # Cholesterol affected by age, BMI, smoking
    chol_mean = 170 + 0.5 * age + 0.3 * bmi + 10 * smoking
    cholesterol = np.random.normal(chol_mean, 20, n_patients)
    
    # Heart rate affected by age, smoking
    hr_mean = 70 + 0.1 * age + 10 * smoking
    heart_rate = np.random.normal(hr_mean, 10, n_patients)
    
    # Create DataFrame with snake_case columns
    data = {
        'patient_id': range(n_patients),
        'age': age.round(1),
        'sex': sex.astype(int),
        'bmi': bmi.round(1),
        'smoking': smoking.astype(int),
        'diabetes': diabetes.astype(int),
        'bp_systolic': bp_systolic.round(1),
        'cholesterol': cholesterol.round(1),
        'heart_rate': heart_rate.round(1)
    }
    return pd.DataFrame(data, columns=BASELINE_COLS)

def generate_longitudinal_data(baseline_df, n_visits=61, max_years=5):
    """Generate longitudinal measurements for each patient."""
    n_patients = len(baseline_df)
    start_date = datetime(2023, 1, 1)
    
    # Generate visit times (not exactly evenly spaced)
    visit_times = np.linspace(0, max_years, n_visits) 
    visit_times = visit_times + np.random.normal(0, 0.1, n_visits)
    visit_times = np.clip(visit_times, 0, max_years)
    
    # Create long format DataFrame
    visits = []
    for patient_id in range(n_patients):
        patient = baseline_df.iloc[patient_id]
        
        # Get baseline values
        base_bp = patient['bp_systolic']
        base_hr = patient['heart_rate']
        
        # Generate longitudinal measurements with autocorrelation
        bp_changes = np.random.normal(0, 5, n_visits)  # Random changes
        bp_changes = np.cumsum(bp_changes) * 0.5  # Cumulative but dampened
        bp_measurements = base_bp + bp_changes
        
        hr_changes = np.random.normal(0, 5, n_visits)
        hr_changes = np.cumsum(hr_changes) * 0.5
        hr_measurements = base_hr + hr_changes
        
        # Risk of adverse event increases with age, bp, smoking
        event_risk = 0.01 + 0.1 * (patient['age'] > 60) + \
                    0.1 * (base_bp > 140) + 0.1 * patient['smoking']
        event_risk = np.clip(event_risk, 0, 1)
        adverse_events = np.random.binomial(1, event_risk, n_visits)
        
        # Create patient visits with dates
        for visit in range(n_visits):
            visit_date = start_date + timedelta(days=int(visit_times[visit] * 365))
            visits.append({
                'patient_id': patient.name,
                'visit_date': visit_date.strftime('%Y-%m-%d'),
                'bp_systolic': bp_measurements[visit].round(1),
                'heart_rate': hr_measurements[visit].round(1),
                'adverse_event': adverse_events[visit],
                'age': patient['age'],
                'sex': patient['sex'],
                'bmi': patient['bmi'],
                'smoking': patient['smoking'],
                'diabetes': patient['diabetes']
            })
    
    # Convert to DataFrame with proper column order
    return pd.DataFrame(visits, columns=LONGITUDINAL_COLS)

def generate_treatment_data(baseline_df, n_months=12):
    """Generate treatment and outcome data."""
    n_patients = len(baseline_df)
    
    # Assign treatment (more likely if bp > 140 or cholesterol > 200)
    treatment_prob = 0.3 + 0.4 * (baseline_df['bp_systolic'] > 140) + \
                    0.3 * (baseline_df['cholesterol'] > 200)
    treatment_prob = np.clip(treatment_prob, 0, 1)
    treatment = np.random.binomial(1, treatment_prob, n_patients)
    
    # Generate adherence (better if not smoking, no diabetes)
    adherence_mean = 0.8 - 0.2 * baseline_df['smoking'] - 0.1 * baseline_df['diabetes']
    adherence = np.random.normal(adherence_mean, 0.1, n_patients)
    adherence = np.clip(adherence, 0, 1)
    
    # Outcome improves with treatment * adherence, worsens with risk factors
    outcome_prob = 0.3 + 0.4 * treatment * adherence - \
                  0.1 * baseline_df['smoking'] - \
                  0.1 * (baseline_df['age'] > 60) - \
                  0.1 * baseline_df['diabetes']
    outcome_prob = np.clip(outcome_prob, 0, 1)
    outcomes = np.random.binomial(1, outcome_prob, n_patients)
    
    # Create DataFrame with all columns
    result_df = baseline_df.copy()
    result_df['treatment'] = treatment
    result_df['adherence'] = adherence.round(2)
    result_df['outcome'] = outcomes
    
    # Ensure proper column order
    return result_df[TREATMENT_COLS]

if __name__ == "__main__":
    # Generate example datasets
    baseline = generate_patient_cohort(n_patients=1000)
    longitudinal = generate_longitudinal_data(baseline)
    treatment = generate_treatment_data(baseline)
    
    # Save to TSV files with proper formatting
    baseline.to_csv('patient_baseline.csv', index=False, sep='\t', float_format='%.1f')
    longitudinal.to_csv('patient_longitudinal.csv', index=False, sep='\t', float_format='%.1f')
    treatment.to_csv('patient_treatment.csv', index=False, sep='\t', float_format='%.2f')
