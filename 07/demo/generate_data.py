"""Build the synthetic clinic tables for the Lecture 07 demos.

build() writes data/patients.parquet (2,400 adult patients) and data/clinic_monthly.parquet
(12 months of visits for three clinics). The seed is fixed, so every run writes the same values.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CLINIC_SIZES = {'North': 900, 'South': 700, 'East': 500, 'West': 300}
# South serves an older population with slightly higher blood pressure; West is younger.
CLINIC_AGE_SHIFT = {'North': 0, 'South': 5, 'East': 0, 'West': -4}
CLINIC_BP_SHIFT = {'North': 0, 'South': 4, 'East': -1, 'West': -2}


def build_patients(rng):
    clinic = np.repeat(list(CLINIC_SIZES), list(CLINIC_SIZES.values()))
    rng.shuffle(clinic)
    n = len(clinic)
    age_shift = pd.Series(clinic).map(CLINIC_AGE_SHIFT).to_numpy()
    bp_shift = pd.Series(clinic).map(CLINIC_BP_SHIFT).to_numpy()

    age = np.clip(18 + 72 * rng.beta(2.2, 2.0, n) + age_shift, 18, 90).round()
    bmi = np.clip(28 + 0.04 * (age - 55) + rng.normal(0, 5.4, n), 16, 55).round(1)
    sbp = 122 + 0.45 * (age - 55) + 0.7 * (bmi - 28) + bp_shift + rng.normal(0, 15, n)
    # A few untreated hypertensive readings give the box plots real high outliers.
    high = rng.random(n) < 0.015
    sbp[high] = rng.uniform(170, 200, high.sum())
    # Reflect the rare very low draws above 90 mmHg so no readings pile up at a clip floor.
    sbp = np.where(sbp < 90, 180 - sbp, sbp)
    steps = 8000 - 60 * (age - 55) - 160 * (bmi - 28) + rng.normal(0, 2200, n)

    # Diabetes risk rises with age and BMI; diabetic and non-diabetic glucose form two peaks.
    risk = 1 / (1 + np.exp(-(-1.55 + 0.035 * (age - 55) + 0.08 * (bmi - 28))))
    diabetic = rng.random(n) < risk
    glucose = np.where(diabetic, rng.normal(165, 28, n), rng.normal(94, 9, n))

    return pd.DataFrame({
        'patient_id': [f'P{i:04d}' for i in range(1, n + 1)],
        'clinic': clinic,
        'sex': rng.choice(['F', 'M'], n, p=[0.54, 0.46]),
        'age': age.astype(int),
        'bmi': bmi,
        'systolic_bp': np.clip(sbp, 85, 210).round().astype(int),
        'steps_per_day': np.clip(steps, 300, 22000).round().astype(int),
        'fasting_glucose': np.clip(glucose, 65, 320).round().astype(int),
    })


def build_clinic_monthly(rng):
    month = np.arange(1, 13)
    winter = np.cos(2 * np.pi * (month - 1) / 12)  # 1 in January, -1 in July
    table = {'month': month}
    for clinic, base, swing, noise in [('North', 230, 0.10, 4), ('South', 122, 0.07, 2), ('East', 30, 0.18, 1.5)]:
        table[clinic] = (base * (1 + swing * winter) + rng.normal(0, noise, 12)).round().astype(int)
    return pd.DataFrame(table)


def check(patients, monthly):
    assert len(patients) == 2400 and patients['patient_id'].is_unique
    assert patients['clinic'].value_counts().to_dict() == CLINIC_SIZES
    assert patients['age'].between(18, 90).all() and patients['bmi'].between(16, 55).all()
    assert 27 <= patients['bmi'].mean() <= 29 and 5 <= patients['bmi'].std() <= 6
    corr = patients[['age', 'bmi', 'systolic_bp', 'steps_per_day']].corr()
    assert 0.35 <= corr.loc['age', 'systolic_bp'] <= 0.5
    assert -0.5 <= corr.loc['age', 'steps_per_day'] <= -0.3
    assert -0.5 <= corr.loc['bmi', 'steps_per_day'] <= -0.3
    assert 17 <= patients['systolic_bp'].std() <= 20
    assert (patients['systolic_bp'] >= 170).mean() >= 0.015
    low = patients['systolic_bp'].value_counts()
    assert low[90] < 2 * low[93], 'readings pile up at the low end'
    # Two glucose peaks: the 120-130 trough holds fewer readings than the 150-170 band.
    glucose = patients['fasting_glucose']
    assert 0.15 <= (glucose >= 126).mean() <= 0.22
    assert glucose.between(120, 129).sum() < glucose.between(150, 169).sum() / 2
    by_clinic = {c: patients.loc[patients['clinic'] == c, ['age', 'systolic_bp']].mean() for c in CLINIC_SIZES}
    assert by_clinic['South']['age'] > by_clinic['North']['age'] + 2
    assert by_clinic['South']['systolic_bp'] > by_clinic['North']['systolic_bp'] + 2
    assert monthly['North'].between(200, 260).all() and monthly['South'].between(110, 135).all()
    assert monthly['East'].between(22, 38).all()
    for clinic in ['North', 'South', 'East']:
        assert monthly[clinic].iloc[[0, 11]].mean() > monthly[clinic].iloc[[5, 6]].mean()


def build(data_dir='data'):
    """Write both tables as Parquet under data_dir and return their paths."""
    rng = np.random.default_rng(217)
    patients = build_patients(rng)
    monthly = build_clinic_monthly(rng)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {'patients': data_dir / 'patients.parquet', 'clinic_monthly': data_dir / 'clinic_monthly.parquet'}
    patients.to_parquet(paths['patients'], index=False)
    monthly.to_parquet(paths['clinic_monthly'], index=False)
    check(pd.read_parquet(paths['patients']), pd.read_parquet(paths['clinic_monthly']))
    return paths


if __name__ == '__main__':
    for name, path in build().items():
        print(f'Wrote {path}')
