"""Build the synthetic health-system extracts for the Lecture 10 demos.

build() writes four Parquet tables under data/:

- pcp_visits: about 520,000 primary-care visits by 160,000 adults in 2024 and 2025 (Demo 1).
- rpm_patients and rpm_readings: 5,000 patients in a remote blood-pressure program and about
  590,000 home cuff readings from January to September 2025 (Demos 1 and 2).
- discharges: about 230,000 inpatient discharges from four hospitals, 2023 to mid-2025 (Demo 3).

Each table follows a causal diagram (arrows point from cause to effect), and its columns are
generated in that order. The seed is fixed, so every run writes the same values.
All patients and values are synthetic.

Primary care (pcp_visits):

    clinic -> deprivation (ADI decile) -> insurance
    age, sex, deprivation -> smoking, physical activity
    age, deprivation, activity -> BMI -> diabetes;  age, diabetes -> eGFR
    age (curved), sex x age, BMI, activity, smoking, deprivation, diabetes, eGFR,
        patient intercept -> untreated SBP
    untreated SBP, age, diabetes, insurance -> on_bp_med  (confounding by indication)
    untreated SBP, on_bp_med -> treated SBP
    treated SBP + clinic offset + season + visit noise growing with age -> recorded SBP
    age, on_bp_med, diabetes -> number of visits
    clinic, BMI over 40, age over 80 -> BMI missing;  deprivation -> activity missing

Remote monitoring (rpm_patients, rpm_readings), one simulated day at a time:

    clinic -> deprivation -> BMI, salt intake, adherence
    age, BMI -> diabetes, sleep apnea;  age, diabetes -> eGFR
    age, sex, BMI, apnea, NSAIDs, salt, eGFR, clinic -> untreated level
    untreated level -> drugs at enrollment (indication)
    adherence -> refill lapses -> drug effect falls within days, returns on refill
    pharmacist review every 4 weeks: high 2-week average -> drug added (60% of the time)
    untreated level - drug effect + slow drift + season -> daily level
    daily level + time of day + cuff size and device bias + noise -> reading
    symptom days (more often in anxious patients) -> extra off-routine reading, BP up that hour
    phone sync -> upload delay;  cuff errors and re-synced copies

Inpatient discharges (discharges), one admission at a time per patient:

    hospital -> deprivation -> insurance, unstable housing;  age -> living alone
    latent frailty, age -> comorbidities -> diagnosis group (winter adds respiratory)
    frailty, diagnosis -> length of stay, labs, medication count, ED visits
    age, length of stay -> disposition;  hospital, housing, insurance -> follow-up scheduled
    prior admissions, ED visits, comorbidities, diagnosis, sodium (U-shaped), anemia, kidney injury,
        age under 35, social factors, hospital, and interactions (heart failure without follow-up,
        living alone at 75+ sent home, 10+ medications at 75+, unstable housing with frequent
        ED visits) -> 30-day readmission
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 217


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mapped(values, table):
    return pd.Series(values).map(table).to_numpy()


# ---------------------------------------------------------------- primary-care visits

# patients, mean ADI decile of the catchment, BP offset (protocol and staff), manual cuff
PCP_CLINICS = {
    'Mission': (42000, 6.0, 1.0, False),
    'Sunset': (36000, 4.0, -0.5, False),
    'Richmond': (30000, 3.5, 0.0, False),
    'Bayview': (24000, 8.0, 2.0, True),
    'Excelsior': (18000, 7.0, -1.0, True),
    'Marina': (10000, 2.0, -1.5, False),
}
# Marina's intake skips the scale more often, so its BMI is missing more often.
PCP_BMI_MISSING = {'Mission': 0.05, 'Sunset': 0.05, 'Richmond': 0.06,
                   'Bayview': 0.07, 'Excelsior': 0.06, 'Marina': 0.20}


def build_pcp_visits(rng):
    names = list(PCP_CLINICS)
    clinic = np.repeat(names, [PCP_CLINICS[c][0] for c in names])
    rng.shuffle(clinic)
    n = len(clinic)
    adi_mean = mapped(clinic, {c: v[1] for c, v in PCP_CLINICS.items()})

    # demographics
    male = rng.random(n) < 0.45
    age0 = np.round(18 + 72 * rng.beta(2.0, 2.2, n))
    adi = np.clip(np.round(adi_mean + rng.normal(0, 1.8, n)), 1, 10)
    medicaid = rng.random(n) < sigmoid(-2.0 + 0.45 * (adi - 5))
    uninsured = ~medicaid & (rng.random(n) < 0.05 + 0.01 * (adi - 5))
    insurance = np.where(age0 >= 65, 'medicare',
                         np.where(medicaid, 'medicaid', np.where(uninsured, 'uninsured', 'commercial')))
    # behaviors
    smoker = rng.random(n) < sigmoid(-2.1 + 0.2 * (adi - 5) + 0.35 * male - 0.015 * (age0 - 45))
    active = rng.random(n) < sigmoid(1.0 - 0.03 * (age0 - 45) - 0.15 * (adi - 5))
    activity = np.where(active, np.exp(rng.normal(np.log(150), 0.6, n)), rng.uniform(0, 30, n))
    # physiology
    bmi0 = np.clip(np.exp(rng.normal(np.log(27.0), 0.18, n)) + 0.4 * (adi - 5)
                   - 0.006 * np.minimum(activity, 400) - 0.002 * (age0 - 55) ** 2 + 1.5, 15, 65)
    diabetes = rng.random(n) < sigmoid(-2.7 + 0.17 * (bmi0 - 28) + 0.045 * (age0 - 50) + 0.15 * (adi - 5))
    egfr = np.clip(105 - 0.9 * np.clip(age0 - 40, 0, None) - 8 * diabetes + rng.normal(0, 13, n), 10, 130)
    # Untreated SBP: about 1 mmHg per kg/m2 of BMI, curved upward with age; men run higher until
    # their 60s; 300 min/week of activity lowers it about 4 mmHg (exercise trials: 4 to 8).
    untreated = (116 + 0.30 * (age0 - 50) + 0.007 * (age0 - 50) ** 2 + 0.9 * (bmi0 - 27)
                 + male * np.clip(7 - 0.2 * (age0 - 45), -2, 7) - 0.012 * np.minimum(activity, 300)
                 + 1.5 * smoker + 0.6 * (adi - 5) + 3 * diabetes + 0.1 * np.clip(60 - egfr, 0, None)
                 + rng.normal(0, 9, n))
    # Confounding by indication: high untreated SBP gets medication, which lowers it about
    # 10 mmHg (single-drug trials: 8 to 10); uninsured patients are treated less often.
    on_med = rng.random(n) < sigmoid((untreated - 130) / 6 + 0.03 * (age0 - 60) + 0.5 * diabetes
                                     - 0.8 * (insurance == 'uninsured') - 0.1 * (adi - 5))
    patient_sbp = untreated - on_med * np.clip(rng.normal(10, 4, n), 2, 20)

    # utilization: overdispersed visit counts, more for older, treated, and diabetic patients
    mean_visits = 2.8 * np.exp(0.25 * on_med + 0.3 * diabetes + 0.008 * (age0 - 50))
    visits_per = 1 + rng.negative_binomial(2, 2 / (2 + mean_visits - 1).clip(0.05))
    idx = np.repeat(np.arange(n), visits_per)
    m = len(idx)
    day = rng.integers(0, 731, m)
    visit_date = pd.Timestamp('2024-01-01') + pd.to_timedelta(day, unit='D')
    age = age0[idx] + (day >= 365)
    vc = clinic[idx]
    manual = mapped(vc, {c: v[3] for c, v in PCP_CLINICS.items()})
    season = 2 * np.cos(2 * np.pi * (day - 15) / 365.25)          # winter readings about 4 mmHg above summer
    noise_sd = 7 + 0.15 * (age - 18)                               # visit-to-visit spread grows with age
    sbp = (patient_sbp[idx] + mapped(vc, {c: v[2] for c, v in PCP_CLINICS.items()}) + 3 * manual
           + season + rng.normal(0, 1, m) * noise_sd)
    dbp = 30 + 0.40 * sbp - 0.12 * np.clip(age - 50, 0, None) + rng.normal(0, 6, m)
    # manual cuffs: terminal-digit preference, about a third of readings rounded to the nearest 10
    rounded = manual & (rng.random(m) < 0.35)
    sbp = np.where(rounded, np.round(sbp, -1), np.round(sbp))
    bmi = bmi0[idx] + rng.normal(0, 0.6, m)
    activity_reported = activity[idx] * np.exp(rng.normal(0.15, 0.4, m))   # self-report runs high

    table = pd.DataFrame({
        'patient_id': [f'PC{i:06d}' for i in idx + 1],
        'clinic': vc,
        'visit_date': visit_date,
        'sex': np.where(male[idx], 'M', 'F'),
        'age': age.astype(int),
        'insurance': insurance[idx],
        'adi_decile': adi[idx].astype(int),
        'bmi': bmi.round(1),
        'activity_min_wk': np.round(activity_reported, -1),
        'smoker': smoker[idx],
        'diabetes': diabetes[idx],
        'sbp': sbp,
        'dbp': dbp.round(),
        'on_bp_med': on_med[idx],
    })
    # Missing BMI depends on the clinic and on the BMI itself (scale limits, declined weighing).
    p_missing = mapped(vc, PCP_BMI_MISSING) + 0.08 * (bmi0[idx] > 40) + 0.05 * (age > 80)
    table.loc[rng.random(m) < p_missing, 'bmi'] = np.nan
    table.loc[rng.random(m) < 0.12 + 0.02 * (adi[idx] - 5), 'activity_min_wk'] = np.nan
    # Messiness of a real extract: decimal slips, swapped BP fields, typos, double-fed rows.
    slip = rng.random(m) < 0.001
    table.loc[slip, 'bmi'] = (table.loc[slip, 'bmi'] * 10).round()
    swap = rng.random(m) < 0.0015
    table.loc[swap, ['sbp', 'dbp']] = table.loc[swap, ['dbp', 'sbp']].to_numpy()
    typo = rng.random(m) < 0.0005
    table.loc[typo, 'sbp'] = table.loc[typo, 'sbp'] * 10
    table['sbp'] = table['sbp'].astype(int)
    table['dbp'] = table['dbp'].astype(int)
    # Double-fed rows repeat every field except the feed's own visit_id.
    dupes = table.sample(frac=0.005, random_state=SEED)
    table = pd.concat([table, dupes]).sort_values(['visit_date', 'patient_id'], kind='stable')
    table.insert(0, 'visit_id', [f'V{i:07d}' for i in range(1, len(table) + 1)])
    return table.reset_index(drop=True)


# ---------------------------------------------------------------- remote BP monitoring

RPM_START, RPM_DAYS = pd.Timestamp('2025-01-01'), 273   # through 2025-09-30
# share of patients, mean ADI decile, level offset
RPM_CLINICS = {'Mission': (0.34, 6.0, 1.0), 'Sunset': (0.26, 4.0, -1.0), 'Richmond': (0.2, 3.5, 0.0),
               'Bayview': (0.13, 8.0, 2.5), 'Marina': (0.07, 2.0, -1.5)}
INTAKE_ITEMS = 30


def build_rpm(rng, n=5000):
    names = list(RPM_CLINICS)
    clinic = rng.choice(names, n, p=[RPM_CLINICS[c][0] for c in names])
    male = rng.random(n) < 0.47
    age = np.round(30 + 55 * rng.beta(2.6, 2.0, n)).astype(int)
    adi = np.clip(np.round(mapped(clinic, {c: v[1] for c, v in RPM_CLINICS.items()})
                           + rng.normal(0, 1.8, n)), 1, 10)
    bmi = np.clip(np.exp(rng.normal(np.log(30.0), 0.17, n)) + 0.4 * (adi - 5), 18, 60)
    diabetes = rng.random(n) < sigmoid(-1.6 + 0.12 * (bmi - 30) + 0.03 * (age - 60))
    egfr = np.clip(105 - 0.8 * (age - 40) - 10 * diabetes + rng.normal(0, 14, n), 8, 130)
    sleep_apnea = rng.random(n) < sigmoid(-2.2 + 0.15 * (bmi - 30) + 0.6 * male)
    nsaid = rng.random(n) < 0.12
    salt_score = np.clip(np.round(rng.normal(5 + 0.3 * (adi - 5), 2)), 0, 10)
    # Untreated level: patients enroll because their BP is uncontrolled; untreated apnea adds about
    # 4 mmHg, NSAIDs about 3, and low eGFR adds resistant hypertension.
    untreated = (152 + 0.35 * (age - 60) + 0.5 * (bmi - 30) + 2 * male + 4 * sleep_apnea + 3 * nsaid
                 + 0.8 * (salt_score - 5) + 0.12 * np.clip(60 - egfr, 0, None)
                 + mapped(clinic, {c: v[2] for c, v in RPM_CLINICS.items()}) + rng.normal(0, 10, n))
    meds0 = np.clip(rng.poisson(np.clip((untreated - 130) / 12, 0.1, None)), 0, 4)
    step = np.clip(rng.normal(6, 2, (n, 8)), 2, 12)     # mmHg per added drug or dose step
    adherence = np.clip(rng.beta(6, 2, n) - 0.03 * (adi - 5) - 0.003 * np.clip(60 - age, 0, None), 0.2, 0.99)
    # proportion of days covered by fills in the year before enrollment, from claims
    pdc = np.clip(adherence + rng.normal(0, 0.08, n), 0.1, 1.0)

    cuff = np.where((bmi > 33) & (rng.random(n) < 0.7), 'large', 'standard')
    device = rng.choice(['cuff_v1', 'cuff_v2'], n, p=[0.6, 0.4])
    # a standard cuff on a large arm reads about 6 mmHg high
    bias = 6 * ((cuff == 'standard') & (bmi > 33)) + 2.5 * (device == 'cuff_v2')
    enroll = rng.integers(0, 90, n)
    leave = np.where(rng.random(n) < 0.18, enroll + rng.integers(30, 200, n), RPM_DAYS)
    meas_rate = rng.beta(4, 2.5, n)
    routine = rng.choice(['morning', 'evening', 'night'], n, p=[0.68, 0.24, 0.08])
    usual_hour = np.select([routine == 'morning', routine == 'evening'],
                           [rng.uniform(6, 9.5, n), rng.uniform(18, 22, n)], rng.uniform(23, 25.5, n))
    anxious = rng.random(n) < 0.3

    days = np.arange(RPM_DAYS)
    season = 3 * np.cos(2 * np.pi * (days - 15) / 365)   # winter about 6 mmHg above late summer
    meds = meds0.copy()
    effect = np.array([step[i, :meds0[i]].sum() for i in range(n)])  # full effect of current drugs
    realized = effect.copy()                                          # effect acting today
    supply_end = enroll + rng.integers(0, 30, n)
    last_change = np.full(n, -1)
    last_dir = np.zeros(n, dtype=np.int8)
    level = untreated - realized + rng.normal(0, 3, n)
    drift = np.zeros(n)
    hist = np.tile(level[:, None], (1, 14))
    rec_level = np.empty((n, RPM_DAYS))
    rec_meds = np.empty((n, RPM_DAYS), dtype=np.int8)
    rec_change = np.empty((n, RPM_DAYS), dtype=np.int32)
    rec_dir = np.empty((n, RPM_DAYS), dtype=np.int8)
    rec_overdue = np.empty((n, RPM_DAYS), dtype=np.int16)
    rec_due = np.empty((n, RPM_DAYS), dtype=np.int32)
    for d in days:
        active = (d >= enroll) & (d < leave)
        # 30-day fills: on time with probability `adherence`, otherwise a lapse ending 10% a day
        due = active & (d >= supply_end)
        refill = due & (rng.random(n) < np.where(d == supply_end, adherence, 0.10))
        supply_end = np.where(refill, d + 30, supply_end)
        overdue = np.where(active, np.clip(d - supply_end, 0, None), 0)
        taking = overdue <= 1
        target = np.where(taking, effect, 0.1 * effect)
        # BP rebounds within days of stopping; a new drug takes one to three weeks
        realized += (target - realized) / np.where(target < realized, 4.0, 10.0)
        drift += -0.03 * drift + rng.normal(0, 0.6, n)
        level += 0.25 * (untreated - realized + drift + season[d] - level) + rng.normal(0, 0.8, n)
        hist[:, d % 14] = level
        # pharmacist review at the end of every 4 weeks enrolled
        review = active & ((d - enroll) % 28 == 27)
        avg14 = hist.mean(1) + rng.normal(0, 2, n)
        up = review & (avg14 >= 140) & (meds < 5) & taking & (rng.random(n) < 0.6)
        down = review & (avg14 < 112) & (meds > 0) & (rng.random(n) < 0.5)
        rows = np.nonzero(up)[0]
        effect[rows] += step[rows, meds[rows]]
        rows = np.nonzero(down)[0]
        effect[rows] -= step[rows, meds[rows] - 1]
        meds += up.astype(int) - down.astype(int)
        last_change = np.where(up | down, d, last_change)
        last_dir = np.where(up, 1, np.where(down, -1, last_dir)).astype(np.int8)
        rec_level[:, d], rec_meds[:, d], rec_change[:, d] = level, meds, last_change
        rec_dir[:, d], rec_overdue[:, d], rec_due[:, d] = last_dir, overdue, supply_end

    enrolled = (days >= enroll[:, None]) & (days < leave[:, None])
    fade = 1 - 0.35 * np.clip(days - enroll[:, None], 0, None) / RPM_DAYS
    pid, day = np.nonzero(enrolled & (rng.random((n, RPM_DAYS)) < meas_rate[:, None] * fade))
    # symptom days: a headache or dizziness prompts an extra reading off the usual routine
    spid, sday = np.nonzero(enrolled & (rng.random((n, RPM_DAYS)) < 0.0125 + 0.05 * anxious[:, None]))
    symptom = np.r_[np.zeros(len(pid), bool), np.ones(len(spid), bool)]
    pid, day = np.r_[pid, spid], np.r_[day, sday]
    k = len(pid)
    odd = rng.random(k) < 0.06
    hour = np.where(symptom, usual_hour[pid] + rng.uniform(3, 21, k),
                    np.where(odd, rng.uniform(0, 24, k), usual_hour[pid] + rng.normal(0, 0.7, k))) % 24
    reading_time = (RPM_START + pd.to_timedelta(day, unit='D')
                    + pd.to_timedelta(np.round(hour * 60), unit='min'))
    rhythm = 5 * np.cos(2 * np.pi * (hour - 8) / 24)     # morning surge, evening dip
    sbp = (rec_level[pid, day] + rhythm + bias[pid] + rng.normal(0, 5.5, k)
           + symptom * rng.normal(16, 6, k))
    dbp = 0.45 * sbp + 20 + rng.normal(0, 5, k)
    pulse = rng.normal(72, 9, k) - 1.5 * rec_meds[pid, day]
    # most cuffs sync within minutes; some wait hours or days for the phone
    lag_min = np.select([rng.random(k) < 0.82, rng.random(k) < 0.78],
                        [rng.exponential(3, k), rng.uniform(30, 720, k)], rng.uniform(1440, 8640, k))
    change_day = rec_change[pid, day]
    readings = pd.DataFrame({
        'patient_id': [f'RP{i:05d}' for i in pid + 1],
        'reading_time': reading_time,
        'upload_time': reading_time + pd.to_timedelta(np.round(lag_min), unit='min'),
        'reason': np.where(symptom, 'symptoms', 'scheduled'),
        'sbp': sbp.round().astype(int),
        'dbp': dbp.round().astype(int),
        'pulse': pulse.round().astype(int),
        # the platform attaches the EHR medication list and the pharmacy fill feed to each reading
        'bp_med_count': rec_meds[pid, day],
        'last_med_change': (RPM_START + pd.to_timedelta(np.maximum(change_day, 0), unit='D')).where(change_day >= 0),
        'last_change_type': pd.Series(rec_dir[pid, day]).map({1: 'added', -1: 'reduced'}).to_numpy(),
        'refill_due_date': RPM_START + pd.to_timedelta(rec_due[pid, day], unit='D'),
        'refill_days_overdue': rec_overdue[pid, day],
    })
    # cuff error codes and re-synced copies
    err = rng.random(k) < 0.003
    readings.loc[err, 'sbp'] = rng.choice([0, 300, 310, 999], err.sum())
    resync = readings.sample(frac=0.008, random_state=SEED)
    resync['upload_time'] = resync['upload_time'] + pd.to_timedelta(rng.integers(60, 2880, len(resync)), unit='min')
    readings = pd.concat([readings, resync]).sort_values('upload_time', kind='stable').reset_index(drop=True)
    readings.insert(0, 'reading_id', [f'R{i:07d}' for i in range(1, len(readings) + 1)])

    patients = pd.DataFrame({
        'patient_id': [f'RP{i:05d}' for i in range(1, n + 1)],
        'enroll_date': RPM_START + pd.to_timedelta(enroll, unit='D'),
        'clinic': clinic, 'sex': np.where(male, 'M', 'F'), 'age': age,
        'bmi': np.where(rng.random(n) < 0.07, np.nan, bmi.round(1)),
        'adi_decile': adi.astype(int), 'diabetes': diabetes, 'egfr': egfr.round(),
        'sleep_apnea': sleep_apnea, 'nsaid_use': nsaid, 'salt_score': salt_score.astype(int),
        'pdc_prior_year': pdc.round(2), 'cuff_size': cuff, 'device': device,
    })
    # 1-to-5 answers on the intake questionnaire (sleep, stress, diet, mood): none moves BP;
    # every seventh item tracks deprivation
    for j in range(1, INTAKE_ITEMS + 1):
        shift = 0.2 * (adi - 5) if j % 7 == 0 else 0
        patients[f'intake_q{j:02d}'] = np.clip(np.round(3 + shift + rng.normal(0, 1, n)), 1, 5).astype(int)
    return patients, readings


# ---------------------------------------------------------------- inpatient discharges

# share of patients, mean ADI decile, readmission offset (logit), share of follow-ups scheduled
HOSPITALS = {'General': (0.42, 6.0, 0.05, 0.62), 'Bayside': (0.27, 4.0, -0.12, 0.75),
             'Northgate': (0.19, 3.0, -0.05, 0.70), 'Valley': (0.12, 7.0, 0.15, 0.48)}
DX = ['heart_failure', 'copd', 'pneumonia', 'sepsis', 'ami', 'gi_bleed', 'orthopedic', 'other']
DX_LOS = {'heart_failure': 4.5, 'copd': 3.8, 'pneumonia': 4.2, 'sepsis': 6.0, 'ami': 3.5,
          'gi_bleed': 3.6, 'orthopedic': 2.8, 'other': 3.2}
DX_RISK = {'heart_failure': 0.45, 'copd': 0.35, 'pneumonia': 0.1, 'sepsis': 0.25, 'ami': 0.15,
           'gi_bleed': 0.2, 'orthopedic': -0.7, 'other': 0.0}
COMORBIDITIES = {  # base logit, per log-frailty, per decade over 60
    'chf': (-1.8, 0.9, 0.35), 'copd': (-1.6, 0.7, 0.15), 'diabetes': (-0.6, 0.5, 0.05),
    'ckd': (-1.5, 0.8, 0.35), 'cancer': (-1.8, 0.5, 0.2), 'dementia': (-3.0, 0.6, 0.8),
    'liver': (-2.6, 0.6, -0.1), 'depression': (-1.1, 0.4, -0.1),
}
EXTRACT_END = pd.Timestamp('2025-06-30')
WINDOW_DAYS = (EXTRACT_END - pd.Timestamp('2023-01-01')).days


def build_discharges(rng, n=150000):
    names = list(HOSPITALS)
    hospital = rng.choice(names, n, p=[HOSPITALS[h][0] for h in names])
    frail = rng.gamma(1.3, 1.0, n)
    log_frail = np.log(frail)
    age0 = 18 + 82 * rng.beta(3.0, 1.9, n)
    male = rng.random(n) < 0.48
    adi = np.clip(np.round(mapped(hospital, {h: v[1] for h, v in HOSPITALS.items()}) + rng.normal(0, 2, n)), 1, 10)
    medicaid = (age0 < 65) & (rng.random(n) < sigmoid(-1.5 + 0.5 * (adi - 5)))
    insurance = np.where(age0 >= 65, 'medicare', np.where(medicaid, 'medicaid', 'commercial'))
    lives_alone = rng.random(n) < sigmoid(-1.6 + 0.03 * (age0 - 60))
    housing_unstable = rng.random(n) < sigmoid(-4.0 + 0.45 * (adi - 5) - 0.02 * (age0 - 50))
    comorb = {c: rng.random(n) < sigmoid(b + f * log_frail + a * (age0 - 60) / 10)
              for c, (b, f, a) in COMORBIDITIES.items()}
    n_comorb = sum(comorb.values()).astype(int)
    ed_rate = rng.gamma(0.8, 0.6 * (0.5 + frail) * (1 + 0.5 * housing_unstable))  # ED visits per 6 months
    hosp_offset = mapped(hospital, {h: v[2] for h, v in HOSPITALS.items()})
    hosp_followup = mapped(hospital, {h: v[3] for h, v in HOSPITALS.items()})

    admit = rng.exponential(1100 / (0.4 + frail))
    alive = np.ones(n, dtype=bool)
    rows = []
    while True:
        cur = np.nonzero(alive & (admit <= WINDOW_DAYS))[0]
        if len(cur) == 0:
            break
        m = len(cur)
        f, lf, a = frail[cur], log_frail[cur], age0[cur] + admit[cur] / 365
        winter = np.cos(2 * np.pi * (admit[cur] % 365.25 - 15) / 365.25) > 0.5
        c = {k: v[cur] for k, v in comorb.items()}
        p_dx = np.column_stack([0.03 + 0.25 * c['chf'], 0.03 + 0.2 * c['copd'] + 0.05 * winter,
                                0.10 + 0.06 * winter, np.full(m, 0.08), np.full(m, 0.06),
                                0.05 + 0.05 * c['liver'], np.full(m, 0.13), np.full(m, 0.3)])
        p_dx /= p_dx.sum(1, keepdims=True)
        dx = np.array(DX)[(p_dx.cumsum(1) > rng.random(m)[:, None]).argmax(1)]
        hf = dx == 'heart_failure'
        elective = (dx == 'orthopedic') & (rng.random(m) < 0.75)
        source = np.where(elective, 'elective', np.where(rng.random(m) < 0.08, 'transfer', 'emergency'))
        # lognormal length of stay by diagnosis, longer for frail patients
        los = np.maximum(0, np.round(np.exp(rng.normal(np.log(mapped(dx, DX_LOS)), 0.6)) + 0.3 * f - 1)).astype(int)
        sodium = (rng.normal(138.5, 3.0, m) - 1.5 * hf - 2.0 * c['liver'] - 0.6 * f
                  + rng.normal(0, 3, m) * (rng.random(m) < 0.1))
        hgb = rng.normal(12.8, 1.7, m) - 0.4 * f - 2.2 * (dx == 'gi_bleed') - 1.0 * c['ckd'] - 0.8 * c['cancer']
        creat = np.exp(rng.normal(np.log(0.95), 0.3, m) + 0.6 * c['ckd'] + 0.004 * (a - 60) + 0.1 * hf)
        n_meds = rng.poisson(4 + 1.5 * n_comorb[cur] + 0.03 * np.clip(a - 50, 0, None))
        ed_6m = rng.poisson(ed_rate[cur])
        died = rng.random(m) < sigmoid(-5.2 + 0.035 * (a - 65) + 1.4 * (dx == 'sepsis') + 0.3 * f)
        snf = rng.random(m) < sigmoid(-2.8 + 0.06 * (a - 75) + 0.18 * los + 0.6 * c['dementia'])
        hh = ~snf & (rng.random(m) < 0.22)
        ama = ~snf & ~hh & (rng.random(m) < 0.015 + 0.04 * (a < 45) + 0.04 * housing_unstable[cur])
        dispo = np.select([died, snf, hh, ama], ['expired', 'snf', 'home_health', 'left_ama'], 'home')
        followup = ~died & (rng.random(m) < hosp_followup[cur] + 0.1 * (dispo == 'home_health')
                            - 0.2 * housing_unstable[cur] - 0.08 * (insurance[cur] == 'medicaid') - 0.3 * ama)
        rows.append(pd.DataFrame({
            'pid': cur, 'admit_day': admit[cur], 'age': a, 'dx_group': dx, 'admit_source': source,
            'los_days': los, 'n_meds': n_meds, 'ed_visits_6m': ed_6m, 'sodium': sodium,
            'hemoglobin': hgb, 'creatinine': creat, 'disposition': dispo,
            'followup_scheduled': followup, 'winter': winter,
        }))
        discharge = admit[cur] + los
        # prior admissions in the 365 days before this one, for the readmission risk
        prior = count_prior(rows, cur, admit[cur])
        home = np.isin(dispo, ['home', 'home_health'])
        # Effects on the log-odds scale; prior utilization and comorbidity dominate, as in LACE
        # and HOSPITAL scores (odds ratios about 1.3 to 2 per factor).
        logit = (-3.5 + 0.35 * lf + mapped(dx, DX_RISK) + hosp_offset[cur]
                 + 0.35 * np.log1p(prior) + 0.10 * np.minimum(ed_6m, 6) + 0.10 * n_comorb[cur]
                 + 0.6 * (sodium < 133) + 0.35 * (sodium > 145) + 0.5 * (hgb < 10)
                 + 0.45 * (creat > 2) + 0.25 * np.log1p(los) + 0.02 * n_meds + 0.3 * c['cancer']
                 + 0.3 * housing_unstable[cur] + 0.15 * (insurance[cur] == 'medicaid') + 0.35 * (a < 35)
                 - 0.2 * followup + 0.8 * ama + 0.2 * (dispo == 'snf') - 0.5 * elective
                 # interactions, odds ratios about 1.8 to 2.7
                 + 1.0 * (hf & ~followup)
                 + 0.8 * (lives_alone[cur] & (a >= 75) & home)
                 + 0.6 * ((n_meds >= 10) & (a >= 75))
                 + 0.8 * (housing_unstable[cur] & (ed_6m >= 3)))
        readmit = ~died & (rng.random(m) < sigmoid(logit))
        rows[-1]['followup_completed'] = followup & (rng.random(m) < np.where(readmit, 0.45, 0.8))
        gap = np.where(readmit, 1 + np.floor(rng.beta(1.1, 2.4, m) * 29),
                       31 + rng.exponential(700 / (0.3 + f)))
        admit[cur] = discharge + gap
        alive[cur] = ~died & (rng.random(m) > 0.04 * f)   # some die outside the hospital
    t = pd.concat(rows, ignore_index=True)
    t = t[t['admit_day'] + t['los_days'] <= WINDOW_DAYS]
    t = t.sort_values(['admit_day', 'pid'], kind='stable').reset_index(drop=True)
    pid = t['pid'].to_numpy()

    start = pd.Timestamp('2023-01-01')
    admit_date = start + pd.to_timedelta(np.floor(t['admit_day']), unit='D')
    out = pd.DataFrame({
        'discharge_id': [f'D{i:07d}' for i in range(1, len(t) + 1)],
        'patient_id': [f'IP{i:06d}' for i in pid + 1],
        'hospital': hospital[pid],
        'admit_date': admit_date,
        'discharge_date': admit_date + pd.to_timedelta(t['los_days'], unit='D'),
        'age': np.minimum(t['age'], 100).astype(int),
        'sex': np.where(male[pid], 'M', 'F'),
        'insurance': insurance[pid],
        'adi_decile': adi[pid].astype(int),
        'lives_alone': lives_alone[pid],
        'housing_unstable': housing_unstable[pid],
        'admit_source': t['admit_source'],
        'dx_group': t['dx_group'],
        'n_comorbidities': n_comorb[pid],
        'los_days': t['los_days'],
        'ed_visits_6m': t['ed_visits_6m'],
        'n_meds': t['n_meds'],
        'sodium': t['sodium'].round(),
        'hemoglobin': t['hemoglobin'].round(1),
        'creatinine': t['creatinine'].round(2),
        'disposition': t['disposition'],
        'followup_scheduled': t['followup_scheduled'],
        'followup_completed': t['followup_completed'],
    })
    # Prior admissions in the 365 days before this admission, as the EHR shows at discharge.
    out = out.sort_values(['patient_id', 'admit_date'], kind='stable')
    prior = np.zeros(len(out), dtype=int)
    same = out['patient_id'].to_numpy()
    dates = out['admit_date'].to_numpy()
    for lag in range(1, 40):
        hit = np.zeros(len(out), dtype=bool)
        hit[lag:] = (same[lag:] == same[:-lag]) & ((dates[lag:] - dates[:-lag]) <= np.timedelta64(365, 'D'))
        if not hit.any():
            break
        prior += hit
    out.insert(out.columns.get_loc('ed_visits_6m'), 'n_prior_admits_12m', prior)
    out = out.sort_values('discharge_id').reset_index(drop=True)
    # Labs missing when not drawn: more often before planned surgery and for short, healthy stays.
    elective = (out['admit_source'] == 'elective').to_numpy()
    short = (out['los_days'] <= 1).to_numpy()
    for col, rate in [('sodium', 0.03), ('hemoglobin', 0.04), ('creatinine', 0.06)]:
        miss = rng.random(len(out)) < rate * np.where(elective, 5, 1) * np.where(short, 1.5, 1)
        out.loc[miss, col] = np.nan
    return out


def count_prior(rows, cur, admit_now):
    """Admissions of each current patient in the 365 days before admit_now, from earlier rounds."""
    if len(rows) == 1:
        return np.zeros(len(cur))
    past = pd.concat([r[['pid', 'admit_day']] for r in rows[:-1]], ignore_index=True)
    past = past[past['pid'].isin(cur)]
    now = pd.Series(admit_now, index=cur)
    recent = past['admit_day'].to_numpy() >= now.loc[past['pid']].to_numpy() - 365
    counts = past[recent].groupby('pid').size()
    return counts.reindex(cur, fill_value=0).to_numpy()


# ---------------------------------------------------------------- checks and build

def check(t):
    v, p, r, d = t['pcp_visits'], t['rpm_patients'], t['rpm_readings'], t['discharges']
    assert 430_000 <= len(v) <= 540_000 and v['patient_id'].nunique() == 160_000
    assert 0.05 <= v['bmi'].isna().mean() <= 0.12
    assert v.duplicated(['patient_id', 'visit_date', 'sbp', 'dbp', 'bmi']).sum() > 1500
    assert len(p) == 5000 and p['patient_id'].is_unique
    assert 500_000 <= len(r) <= 700_000
    clean = r[r['sbp'].between(60, 260)]
    assert 130 <= clean['sbp'].mean() <= 150
    assert (r['upload_time'] >= r['reading_time']).all()
    assert 180_000 <= len(d) <= 280_000
    assert d['patient_id'].value_counts().max() >= 10
    assert (d['discharge_date'] <= EXTRACT_END).all() and (d['discharge_date'] >= d['admit_date']).all()


def build(data_dir='data'):
    """Write the four demo tables as Parquet under data_dir and return their paths."""
    # One random stream per table, so changing one table's code leaves the others' values unchanged.
    tables = {'pcp_visits': build_pcp_visits(np.random.default_rng([SEED, 1]))}
    tables['rpm_patients'], tables['rpm_readings'] = build_rpm(np.random.default_rng([SEED, 2]))
    tables['discharges'] = build_discharges(np.random.default_rng([SEED, 3]))
    check(tables)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        paths[name] = data_dir / f'{name}.parquet'
        table.to_parquet(paths[name], index=False)
    return paths


if __name__ == '__main__':
    for name, path in build().items():
        print(f'Wrote {path}')
