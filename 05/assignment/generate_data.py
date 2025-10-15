#!/usr/bin/env python3
"""
Generate realistic clinical trial data with hidden variables and realistic correlations.

Hidden variables drive the realistic patterns:
- Underlying cardiovascular health status
- Treatment response propensity
- Site quality
- Patient engagement level
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# Number of patients
N = 10000

print(f"Generating clinical trial data for {N} patients...")

# ============================================================================
# HIDDEN VARIABLES (drive realistic correlations, not observed in final data)
# ============================================================================

# Hidden: Underlying CV health (0-1, lower is worse)
cv_health = np.random.beta(5, 2, N)  # Skewed toward healthier

# Hidden: Treatment response propensity (0-1, higher responds better)
treatment_response = np.random.beta(2, 2, N)  # Centered distribution

# Hidden: Site quality (affects data completeness and accuracy)
site_quality = {
    'Site A': 0.95,  # Excellent
    'Site B': 0.85,  # Good
    'Site C': 0.75,  # Average
    'Site D': 0.70,  # Below average
    'Site E': 0.60   # Poor (more missing data)
}

# Hidden: Patient engagement (affects adherence and follow-up)
engagement = np.random.beta(3, 2, N)  # Skewed toward engaged

# ============================================================================
# PATIENT DEMOGRAPHICS
# ============================================================================

# Age: realistic distribution for CVD trial (0-100, peak ~60)
# Students will filter to 18-85 using filter_data() utility
age = np.random.gamma(8, 6, N) + 35
age = np.clip(age, 0, 100).astype(int)

# Sex: roughly balanced
sex = np.random.choice(['M', 'F'], N, p=[0.48, 0.52])

# BMI: correlated with CV health (worse health -> higher BMI)
bmi_base = 22 + (1 - cv_health) * 15 + np.random.normal(0, 3, N)
bmi = np.clip(bmi_base, 16, 45).round(1)

# Patient ID
patient_id = [f'P{i:05d}' for i in range(1, N+1)]

# ============================================================================
# TRIAL INFORMATION
# ============================================================================

# Site: unequal enrollment (Site A enrolled most, Site E least)
site_probs = [0.30, 0.25, 0.20, 0.15, 0.10]
site = np.random.choice(['Site A', 'Site B', 'Site C', 'Site D', 'Site E'],
                        N, p=site_probs)

# Get site quality for each patient
site_quality_values = np.array([site_quality[s] for s in site])

# Enrollment date: spread over 2 years
start_date = datetime(2022, 1, 1)
enrollment_date = [start_date + timedelta(days=int(x))
                   for x in np.random.uniform(0, 730, N)]
enrollment_date_str = [d.strftime('%Y-%m-%d') for d in enrollment_date]

# Intervention group: balanced randomization
intervention_group = np.random.choice(['Control', 'Treatment A', 'Treatment B'],
                                     N, p=[0.33, 0.33, 0.34])

# Follow-up months: more recent enrollees have less follow-up
days_since_enrollment = [(datetime(2024, 1, 1) - d).days for d in enrollment_date]
follow_up_months = np.clip(np.array(days_since_enrollment) / 30, 0, 24).round(0).astype(int)

# ============================================================================
# CLINICAL MEASUREMENTS (correlated with CV health)
# ============================================================================

# Systolic BP: higher when CV health is poor
systolic_bp_base = 110 + (1 - cv_health) * 40 + np.random.normal(0, 12, N)
systolic_bp = np.clip(systolic_bp_base, 90, 200).round(0).astype(int)

# Diastolic BP: correlated with systolic (roughly 60% of systolic)
diastolic_bp_base = systolic_bp * 0.6 + np.random.normal(0, 8, N)
diastolic_bp = np.clip(diastolic_bp_base, 60, 120).round(0).astype(int)

# Total cholesterol: higher when CV health is poor
cholesterol_total_base = 160 + (1 - cv_health) * 80 + np.random.normal(0, 30, N)
cholesterol_total = np.clip(cholesterol_total_base, 120, 350).round(0).astype(int)

# HDL cholesterol: higher when CV health is good (protective)
cholesterol_hdl_base = 40 + cv_health * 30 + np.random.normal(0, 10, N)
cholesterol_hdl = np.clip(cholesterol_hdl_base, 25, 100).round(0).astype(int)

# LDL cholesterol: roughly total - HDL - 20% (VLDL estimate)
cholesterol_ldl = np.clip(cholesterol_total - cholesterol_hdl - cholesterol_total * 0.2,
                          40, 250).round(0).astype(int)

# Fasting glucose: higher when CV health is poor (metabolic syndrome)
glucose_fasting_base = 85 + (1 - cv_health) * 50 + np.random.normal(0, 15, N)
glucose_fasting = np.clip(glucose_fasting_base, 70, 250).round(0).astype(int)

# ============================================================================
# TREATMENT EFFECTS (for Treatment A and Treatment B)
# ============================================================================

# Treatment A: reduces BP and cholesterol (if patient responds)
treatment_a_effect = (intervention_group == 'Treatment A') * treatment_response
systolic_bp = (systolic_bp - treatment_a_effect * 15).round(0).astype(int)
cholesterol_total = (cholesterol_total - treatment_a_effect * 30).round(0).astype(int)

# Treatment B: reduces glucose primarily (if patient responds)
treatment_b_effect = (intervention_group == 'Treatment B') * treatment_response
glucose_fasting = (glucose_fasting - treatment_b_effect * 20).round(0).astype(int)
systolic_bp = (systolic_bp - treatment_b_effect * 8).round(0).astype(int)

# ============================================================================
# OUTCOMES (driven by CV health and treatment effects)
# ============================================================================

# CVD event probability: driven by CV health, age, and treatment
cvd_risk = (1 - cv_health) * 0.3 + (age - 40) / 200 - treatment_a_effect * 0.1 - treatment_b_effect * 0.05
cvd_risk = np.clip(cvd_risk, 0, 0.4)
outcome_cvd = np.random.random(N) < cvd_risk
outcome_cvd_str = ['Yes' if x else 'No' for x in outcome_cvd]

# Adherence: driven by engagement, site quality, and side effects
adherence_pct_base = engagement * 85 + site_quality_values * 10 + np.random.normal(0, 10, N)
adherence_pct = np.clip(adherence_pct_base, 20, 100).round(0).astype(int)

# Adverse events: higher with poor CV health and in treatment groups
adverse_events_rate = (1 - cv_health) * 0.02 + (intervention_group != 'Control') * 0.01
adverse_events = np.random.poisson(adverse_events_rate * follow_up_months)

# Dropout: more likely with low engagement, long follow-up, and adverse events
dropout_risk = (1 - engagement) * 0.3 + (adverse_events > 2) * 0.2 + (follow_up_months > 18) * 0.1
dropout = np.random.random(N) < dropout_risk
dropout_str = ['Yes' if x else 'No' for x in dropout]

# ============================================================================
# INJECT REALISTIC DATA QUALITY ISSUES
# ============================================================================

# Make a DataFrame
df = pd.DataFrame({
    'patient_id': patient_id,
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'enrollment_date': enrollment_date_str,
    'systolic_bp': systolic_bp,
    'diastolic_bp': diastolic_bp,
    'cholesterol_total': cholesterol_total,
    'cholesterol_hdl': cholesterol_hdl,
    'cholesterol_ldl': cholesterol_ldl,
    'glucose_fasting': glucose_fasting,
    'site': site,
    'intervention_group': intervention_group,
    'follow_up_months': follow_up_months,
    'adverse_events': adverse_events,
    'outcome_cvd': outcome_cvd_str,
    'adherence_pct': adherence_pct,
    'dropout': dropout_str
})

print(f"Generated clean data: {df.shape}")

# ============================================================================
# DATA QUALITY ISSUES (realistic clinical data problems)
# ============================================================================

# 1. MISSING DATA (more missing at lower-quality sites)
print("\nInjecting missing data...")

# BMI missing (5-15% depending on site quality)
missing_bmi_prob = 0.15 - site_quality_values * 0.1
missing_bmi = np.random.random(N) < missing_bmi_prob
df.loc[missing_bmi, 'bmi'] = np.nan

# Blood pressure missing (equipment issues, skipped visits)
missing_bp_prob = 0.08 - site_quality_values * 0.05
missing_bp = np.random.random(N) < missing_bp_prob
df.loc[missing_bp, 'systolic_bp'] = np.nan
df.loc[missing_bp, 'diastolic_bp'] = np.nan

# Cholesterol panel missing (expensive test, not always done)
missing_chol_prob = 0.12 - site_quality_values * 0.08
missing_chol = np.random.random(N) < missing_chol_prob
df.loc[missing_chol, 'cholesterol_total'] = np.nan
df.loc[missing_chol, 'cholesterol_hdl'] = np.nan
df.loc[missing_chol, 'cholesterol_ldl'] = np.nan

# Glucose missing (fasting requirement not always met)
missing_glucose_prob = 0.06 - site_quality_values * 0.03
missing_glucose = np.random.random(N) < missing_glucose_prob
df.loc[missing_glucose, 'glucose_fasting'] = np.nan

# Follow-up data missing for dropouts
df.loc[df['dropout'] == 'Yes', 'adherence_pct'] = np.nan

# 2. SENTINEL VALUES (data entry system codes)
print("Injecting sentinel values...")

# Age: -999 for missing (old data entry system)
sentinel_age = np.random.choice(df.index, size=int(N * 0.02), replace=False)
df.loc[sentinel_age, 'age'] = -999

# BMI: -1 sometimes used instead of NaN
sentinel_bmi = df[df['bmi'].isna()].sample(frac=0.3).index
df.loc[sentinel_bmi, 'bmi'] = -1

# 3. DATA ENTRY ERRORS (removed - not covered in assignment)
# Note: BP swaps and BMI as weight errors removed as they're not part of the assignment scope

# 4. TEXT INCONSISTENCIES
print("Injecting text inconsistencies...")

# Site names: inconsistent capitalization
site_variations = {
    'Site A': ['Site A', 'SITE A', 'site a', 'Site  A'],
    'Site B': ['Site B', 'SITE B', 'site b'],
    'Site C': ['Site C', 'SITE C', 'site c'],
    'Site D': ['Site D', 'SITE D', 'site d', 'Site_D'],
    'Site E': ['Site E', 'SITE E', 'site e']
}

for site_name, variations in site_variations.items():
    site_mask = df['site'] == site_name
    site_indices = df[site_mask].index
    for idx in site_indices:
        df.loc[idx, 'site'] = np.random.choice(variations)

# Sex: M/F vs Male/Female
male_mask = df['sex'] == 'M'
female_mask = df['sex'] == 'F'
df.loc[male_mask.sample(frac=0.3).index, 'sex'] = 'Male'
df.loc[female_mask.sample(frac=0.3).index, 'sex'] = 'Female'

# Intervention group: typos and spacing
intervention_variations = {
    'Control': ['Control', 'control', 'CONTROL', 'Contrl'],
    'Treatment A': ['Treatment A', 'TREATMENT A', 'treatment a', 'Treatmen A', 'TreatmentA'],
    'Treatment B': ['Treatment B', 'TREATMENT B', 'treatment b', 'Treatment  B']
}

for group_name, variations in intervention_variations.items():
    group_mask = df['intervention_group'] == group_name
    group_indices = df[group_mask].index
    for idx in group_indices:
        df.loc[idx, 'intervention_group'] = np.random.choice(variations)

# Outcome CVD: Yes/No variations
yes_mask = df['outcome_cvd'] == 'Yes'
no_mask = df['outcome_cvd'] == 'No'
df.loc[yes_mask.sample(frac=0.2).index, 'outcome_cvd'] = 'yes'
df.loc[no_mask.sample(frac=0.2).index, 'outcome_cvd'] = 'no'

# 5. DATE FORMAT INCONSISTENCIES
print("Injecting date format issues...")

# Some dates in different formats
date_sample = df.sample(frac=0.15).index
for idx in date_sample:
    original_date = enrollment_date[idx]
    # Random format: MM/DD/YYYY or DD-MM-YYYY
    if np.random.random() < 0.5:
        df.loc[idx, 'enrollment_date'] = original_date.strftime('%m/%d/%Y')
    else:
        df.loc[idx, 'enrollment_date'] = original_date.strftime('%d-%m-%Y')

# 6. WHITESPACE IN TEXT FIELDS
print("Injecting whitespace issues...")

# Random leading/trailing spaces
text_cols = ['site', 'intervention_group', 'sex']
for col in text_cols:
    space_sample = df.sample(frac=0.1).index
    df.loc[space_sample, col] = '  ' + df.loc[space_sample, col] + '  '

# ============================================================================
# SAVE DATA
# ============================================================================

output_file = 'data/clinical_trial_raw.csv'
df.to_csv(output_file, index=False)

print(f"\n✓ Generated clinical trial data: {output_file}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")
print(f"\nData quality issues injected:")
print(f"  - Missing data: varies by site (5-15%)")
print(f"  - Sentinel values: ~2% of age as -999")
print(f"  - Text inconsistencies: capitalization, typos, spacing")
print(f"  - Date formats: 3 different formats")
print(f"  - Whitespace: ~10% of text fields")

# Print summary statistics
print(f"\nSummary statistics:")
print(f"  Sites: {df['site'].nunique()} unique sites")
print(f"  Intervention groups: {df['intervention_group'].nunique()} groups")
print(f"  CVD events: {(df['outcome_cvd'].str.lower() == 'yes').sum()} patients")
print(f"  Dropouts: {(df['dropout'].str.lower() == 'yes').sum()} patients")
print(f"  Missing BMI: {df['bmi'].isna().sum()} ({df['bmi'].isna().mean()*100:.1f}%)")
print(f"  Missing cholesterol: {df['cholesterol_total'].isna().sum()} ({df['cholesterol_total'].isna().mean()*100:.1f}%)")
