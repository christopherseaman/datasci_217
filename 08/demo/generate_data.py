"""Build the synthetic health-system tables for the Lecture 08 demos.

build() writes three Parquet files under data/:

- clinics.parquet: 9 clinics in the board's reporting order, including Oceanview, which opens in 2026.
- visits.parquet: 1,200,000 outpatient visits in 2025, one row per visit.
- labs.parquet: 2,000,000 lab results in 2025, one row per result.

Columns are generated in this causal order (an arrow means "is an input to"):

    home clinic -> deprivation (area index 1-10) -> insurance
    age, sex, deprivation -> smoking, activity -> BMI -> systolic BP
    age, BMI, deprivation -> diabetes, baseline A1c; BP -> hypertension
    age, sex, diabetes, hypertension -> kidney function (creatinine)
    diabetes, age -> statin -> LDL               (sicker patients get the drug)
    A1c >= 8 -> treatment intensified -> A1c falls through the year
    conditions, age -> visit frequency (gamma frailty, so a few patients visit weekly)
    conditions, age -> department; clinic, provider, weekday, season, visit type -> wait
    age, deprivation, wait -> survey returned; wait, provider -> satisfaction score
    diabetes, A1c control, engagement -> A1c test spacing (about 3 to 6 months)

Operational events planted on top: Parkside opens in July and takes Sunset's patients,
Bayview loses an orthopedic surgeon in September, Excelsior's check-in kiosk fails in
October, an Excelsior analyzer writes 9999 for 23 failed glucose runs, and Ridgecrest
runs a monthly cardiology outreach clinic for 10 regulars.

The seed is fixed, so every run writes the same values.
"""

from pathlib import Path

import numpy as np
import pandas as pd

N_PATIENTS = 260_000
N_VISITS = 1_200_000
N_LABS = 2_000_000

CLINICS = pd.DataFrame({
    'clinic': ['Mission', 'Sunset', 'Richmond', 'Bayview', 'Excelsior', 'Marina', 'Parkside', 'Ridgecrest', 'Oceanview'],
    'region': ['Central', 'West', 'West', 'South', 'South', 'North', 'West', 'Rural', 'West'],
    'opened': pd.to_datetime(['2009-03-02', '2012-06-04', '2011-09-12', '2015-01-05', '2018-04-02',
                              '2016-02-01', '2025-07-07', '2019-10-07', '2026-01-12']),
})
# Share of patients whose home clinic this is (Oceanview has none yet).
HOME_SHARE = [0.22, 0.17, 0.14, 0.15, 0.12, 0.10, 0.07, 0.03]
# Mean area deprivation decile of each clinic's neighborhood (1 least, 10 most deprived).
CLINIC_DEPRIVATION = [5.5, 4.0, 4.0, 7.5, 7.0, 2.5, 4.5, 6.5]
# Log-scale wait shift per clinic: Mission is short-staffed, Ridgecrest is small and quiet.
CLINIC_WAIT_LOG = [0.20, 0.04, 0.00, 0.06, 0.08, -0.05, -0.08, -0.30]
PARKSIDE_OPENS = pd.Timestamp('2025-07-07')
BAYVIEW_LOSS = pd.Timestamp('2025-09-01')

DEPTS = ['Family Medicine', 'Pediatrics', 'Behavioral Health', 'Cardiology', 'Orthopedics', 'Endocrinology', 'Dermatology']
# Median wait in minutes and log-scale spread; walk-in Family Medicine varies most,
# scheduled therapy in Behavioral Health least. Outpatient waits in US surveys run 15 to 30 minutes.
WAIT_MEDIAN = {'Family Medicine': 16, 'Pediatrics': 12, 'Behavioral Health': 6, 'Cardiology': 20,
               'Orthopedics': 27, 'Endocrinology': 18, 'Dermatology': 14}
WAIT_LOG_SD = {'Family Medicine': 0.55, 'Pediatrics': 0.40, 'Behavioral Health': 0.35, 'Cardiology': 0.40,
               'Orthopedics': 0.42, 'Endocrinology': 0.38, 'Dermatology': 0.40}
# Departments a clinic does not run; those visits go to Family Medicine instead.
NOT_OFFERED = {
    'Marina': ['Orthopedics'],
    'Parkside': ['Dermatology', 'Endocrinology'],
    'Ridgecrest': ['Cardiology', 'Orthopedics', 'Endocrinology', 'Dermatology'],
}
HOLIDAYS = pd.to_datetime(['2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', '2025-07-04',
                           '2025-09-01', '2025-11-27', '2025-11-28', '2025-12-25'])
DAYS = pd.date_range('2025-01-01', '2025-12-31', freq='D')


def ids(prefix, numbers, width):
    return prefix + pd.Series(numbers).astype(str).str.zfill(width)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def day_weights(winter):
    """Probability of each 2025 day: weekdays, a light Saturday, closed Sundays and holidays."""
    weekday = DAYS.dayofweek.to_numpy()
    weight = 1 + winter * np.cos(2 * np.pi * (DAYS.dayofyear.to_numpy() - 15) / 365)
    weight = np.where(weekday == 5, 0.3 * weight, weight)
    weight = np.where((weekday == 6) | DAYS.isin(HOLIDAYS), 0, weight)
    return weight / weight.sum()


def build_patients(rng):
    n = N_PATIENTS
    home = rng.choice(8, n, p=HOME_SHARE)
    deprivation = np.clip(np.round(np.array(CLINIC_DEPRIVATION)[home] + rng.normal(0, 1.8, n)), 1, 10)
    child = rng.random(n) < 0.19
    age = np.where(child, rng.integers(0, 18, n), (18 + 77 * rng.beta(2.0, 1.8, n)).astype(int))
    male = rng.random(n) < 0.47
    adult = ~child

    # Adult smoking runs about 8% in the least and 20% in the most deprived areas.
    smoker = adult & (rng.random(n) < 0.06 + 0.015 * deprivation)
    activity = np.clip(rng.normal(150 - 1.2 * (age - 45) - 6 * (deprivation - 5), 70, n), 0, None)
    bmi = np.where(child, rng.normal(19, 3, n),
                   27.5 + 0.35 * (deprivation - 5) - 0.012 * (activity - 150) + 0.03 * (age - 45) + rng.normal(0, 5, n))
    bmi = np.clip(bmi, 14, 60)
    # Each BMI point adds about 1 mmHg; each decade of age about 5 mmHg.
    sbp = 118 + 0.5 * (age - 45) + 1.0 * (bmi - 27) + 2 * smoker + rng.normal(0, 13, n)
    hypertension = adult & (sbp >= 140)

    # Diagnosed diabetes reaches about 12% of adults, rising steeply with age and BMI.
    dm_risk = logistic(-2.7 + 0.045 * (age - 50) + 0.11 * (bmi - 28) + 0.12 * (deprivation - 5))
    diabetic = adult & (rng.random(n) < dm_risk)
    a1c_base = np.where(
        diabetic,
        # Younger and more deprived patients with diabetes are less well controlled.
        6.0 + rng.gamma(2.0, 0.8, n) - 0.012 * (age - 55) + 0.09 * (deprivation - 5),
        5.25 + 0.006 * (age - 45) + 0.015 * (bmi - 27) + rng.normal(0, 0.28, n),
    )
    # eGFR falls about 1 mL/min per year of age; diabetes and hypertension lower it further.
    egfr = np.clip(135 - 0.9 * age - 9 * diabetic - 6 * hypertension + rng.normal(0, 13, n), 8, 140)
    # Statins: most adults with diabetes over 40 and about a third of others over 60.
    statin = adult & (rng.random(n) < np.where(diabetic & (age >= 40), 0.70, np.where(age >= 60, 0.33, 0.04)))
    depression = rng.random(n) < np.where(child, 0.05, 0.07 + 0.012 * deprivation)

    conditions = diabetic.astype(int) + hypertension + depression + (egfr < 60)
    # Visit frequency: gamma frailty gives a long tail, so a few patients come every week.
    rate = np.where(child, 1.1, 0.8 * np.exp(0.35 * conditions + 0.008 * (age - 45)))
    frailty = np.minimum(rng.gamma(0.7, 1 / 0.7, n), 6)
    return pd.DataFrame({
        'home': home, 'deprivation': deprivation, 'age': age, 'male': male, 'smoker': smoker,
        'bmi': bmi, 'sbp': sbp, 'hypertension': hypertension, 'diabetic': diabetic,
        'a1c_base': a1c_base, 'egfr': egfr, 'statin': statin, 'depression': depression,
        'rate': rate * frailty,
        # Engagement: patients who come in less often also test less often (missing not at random).
        'engagement': frailty,
    })


def choose_department(rng, p, who):
    """One department per visit, with each patient's conditions steering the specialty mix."""
    n = len(who)
    age = p['age'].to_numpy()[who]
    weights = np.column_stack([
        np.full(n, 4.0),                                                    # Family Medicine
        np.zeros(n),                                                        # Pediatrics
        0.8 + 3.0 * p['depression'].to_numpy()[who],                        # Behavioral Health
        0.4 + 2.0 * p['hypertension'].to_numpy()[who] + 0.02 * np.clip(age - 40, 0, None),  # Cardiology
        0.6 + 0.025 * np.clip(age - 30, 0, None) + 0.05 * np.clip(p['bmi'].to_numpy()[who] - 30, 0, None),  # Orthopedics
        0.15 + 2.2 * p['diabetic'].to_numpy()[who],                         # Endocrinology
        np.full(n, 0.9),                                                    # Dermatology
    ])
    child = age < 18
    weights[child] = [0, 9.3, 0.7, 0, 0, 0, 0]
    cumulative = weights.cumsum(axis=1)
    pick = rng.random(n) * cumulative[:, -1]
    return np.array(DEPTS, dtype=object)[(cumulative < pick[:, None]).sum(axis=1)]


def build_visits(rng, p):
    n = N_VISITS
    who = rng.choice(N_PATIENTS, n, p=(p['rate'] / p['rate'].sum()).to_numpy())
    # Winter respiratory season: about 25% more visits in January than in July.
    date = DAYS[rng.choice(len(DAYS), n, p=day_weights(0.12))].to_numpy().copy()

    names = CLINICS['clinic'].to_numpy()
    home = p['home'].to_numpy()[who]
    clinic = names[np.where(rng.random(n) < 0.88, home, rng.choice(8, n, p=HOME_SHARE))].astype(object)
    # Parkside's patients went to Sunset before it opened.
    clinic[(clinic == 'Parkside') & (date < PARKSIDE_OPENS.to_datetime64())] = 'Sunset'

    # A patient in ongoing care (therapy, fracture follow-up) returns to the same department:
    # about half of visits go to the patient's main department, drawn once per patient.
    main_dept = choose_department(rng, p, np.arange(N_PATIENTS))
    dept = np.where(rng.random(n) < 0.5, main_dept[who], choose_department(rng, p, who))
    for name, missing in NOT_OFFERED.items():
        dept[(clinic == name) & np.isin(dept, missing)] = 'Family Medicine'

    # Ridgecrest's monthly cardiology outreach: 10 hypertensive regulars, one or two visits a month each.
    outreach = np.flatnonzero((clinic == 'Ridgecrest') & (dept == 'Family Medicine')
                              & (p['age'].to_numpy()[who] >= 55) & p['hypertension'].to_numpy()[who])
    pool = rng.choice(np.unique(who[outreach]), 10, replace=False)
    first_tuesdays = [d for d in DAYS if d.dayofweek == 1 and d.day <= 7]
    rows = rng.choice(outreach, 10 * 12 + 60, replace=False)
    who[rows] = np.concatenate([np.repeat(pool, 12), rng.choice(pool, 60)])
    date[rows] = np.array(first_tuesdays * 10 + list(rng.choice(first_tuesdays, 60)), dtype='datetime64[ns]')
    dept[rows] = 'Cardiology'
    age = p['age'].to_numpy()[who]

    order = np.lexsort((who, date))
    who, date, clinic, dept, age = who[order], date[order], clinic[order], dept[order], age[order]
    # A patient's first visit of the year is often new to the clinic; later ones are mostly follow-ups.
    first = ~pd.Series(who).duplicated().to_numpy()
    visit_type = np.where(rng.random(n) < np.where(first, 0.45, 0.12), 'New', 'Follow-up')

    # Each provider has a typical pace; about one provider per 2,500 visits in each clinic and department.
    cell = pd.Series(clinic).fillna('') + '|' + pd.Series(dept)
    cell_code, cell_names = pd.factorize(cell)
    cell_size = np.bincount(cell_code)
    n_providers = np.maximum(1, np.round(cell_size / 2500)).astype(int)
    provider = (rng.random(n) * n_providers[cell_code]).astype(int)
    offset = np.concatenate([[0], np.cumsum(n_providers)[:-1]])
    provider_id = offset[cell_code] + provider
    provider_pace = rng.normal(0, 0.15, n_providers.sum())
    provider_manner = rng.normal(0, 0.25, n_providers.sum())

    d = pd.DatetimeIndex(date)
    month, weekday = d.month.to_numpy(), d.dayofweek.to_numpy()
    clinic_code = pd.Series(clinic).map(dict(zip(names, range(9)))).to_numpy()
    log_wait = (np.log(pd.Series(dept).map(WAIT_MEDIAN).to_numpy())
                + np.array(CLINIC_WAIT_LOG + [0.0])[clinic_code]
                + provider_pace[provider_id]
                + 0.15 * (visit_type == 'New')
                + 0.08 * (weekday == 0)                        # Mondays carry the weekend backlog
                + 0.10 * np.isin(month, [1, 2]) + 0.05 * (month == 12)
                + rng.normal(0, 1, n) * pd.Series(dept).map(WAIT_LOG_SD).to_numpy())
    # Bayview lost an orthopedic surgeon in September; the rest of its orthopedic waits nearly double.
    bayview_ortho = (clinic == 'Bayview') & (dept == 'Orthopedics') & (date >= BAYVIEW_LOSS.to_datetime64())
    log_wait = log_wait + 0.62 * bayview_ortho
    wait = np.clip(np.exp(log_wait), 0, 240).round()

    # Check-in kiosks miss some waits, more often for new patients filling in paper forms;
    # Excelsior's kiosk was down for most of October.
    lost = rng.random(n) < np.where(visit_type == 'New', 0.03, 0.015)
    lost |= (clinic == 'Excelsior') & (month == 10) & (rng.random(n) < 0.6)
    recorded_wait = np.where(lost, np.nan, wait)

    # About one survey in three comes back: older patients return more, deprived areas and long waits fewer.
    dep = p['deprivation'].to_numpy()[who]
    returned = rng.random(n) < logistic(-0.75 + 0.018 * (age - 45) - 0.08 * (dep - 5) - 0.01 * (wait - 20))
    latent = 4.55 - 0.03 * wait + provider_manner[provider_id] + rng.normal(0, 0.7, n)
    satisfaction = np.where(returned, np.clip(latent.round(), 1, 5), np.nan)

    # A few registrations never recorded the clinic.
    clinic[rng.random(n) < 0.0025] = None

    visits = pd.DataFrame({
        'visit_id': ids('V', np.arange(1, n + 1), 7),
        'visit_date': d.as_unit('us'),
        'visit_month': d.strftime('%Y-%m'),
        'patient_id': ids('P', who + 1, 6),
        'clinic': pd.array(clinic, dtype='str'),
        'department': pd.array(dept, dtype='str'),
        'visit_type': visit_type,
        'age': age,
        'wait_min': recorded_wait,
        'satisfaction': satisfaction,
    })
    return visits


def a1c_schedule(rng, p):
    """A1c test dates: every 3 to 6 months for diabetes, one yearly screen for some others."""
    diabetic = p['diabetic'].to_numpy()
    adult = p['age'].to_numpy() >= 18
    dm = np.flatnonzero(diabetic)
    # ADA: test quarterly when above goal, twice a year when at goal; disengaged patients stretch it.
    above_goal = p['a1c_base'].to_numpy()[dm] >= 7
    mean_gap = np.where(above_goal, 100, 165) * np.clip(1.25 / np.sqrt(p['engagement'].to_numpy()[dm]), 0.9, 2.5)
    # A1c reflects about 3 months of glucose, so a retest comes no sooner than 80 days.
    gaps = 80 + rng.gamma(4, (mean_gap[:, None] - 80) / 4, (len(dm), 6))
    start = rng.uniform(-200, 10, len(dm))[:, None] + gaps.cumsum(axis=1)
    keep = (start >= 0) & (start < 364)
    dm_who = np.repeat(dm, keep.sum(axis=1))
    dm_day = start[keep]
    # USPSTF screening for adults 35 to 70 with overweight; about half of eligible patients get it.
    age, bmi = p['age'].to_numpy(), p['bmi'].to_numpy()
    eligible = adult & ~diabetic & (((age >= 35) & (age <= 70) & (bmi >= 25)) | (age > 70))
    screened = np.flatnonzero(rng.random(len(p)) < np.where(eligible, 0.5, np.where(adult & ~diabetic, 0.06, 0)))
    who = np.concatenate([dm_who, screened])
    day = np.concatenate([dm_day, draw_open_days(rng, len(screened))])
    return who, day


def draw_open_days(rng, size):
    """Draw unscheduled lab days with the lab's weekday, Saturday, and closure weights."""
    return rng.choice(len(DAYS), size, p=day_weights(0.05)).astype(float)


def to_open_day(day):
    """Move a scheduled fractional day of 2025 to the next day the lab is open."""
    weights = day_weights(0.05)
    open_days = np.flatnonzero(weights > 0)
    index = np.searchsorted(open_days, np.floor(day).astype(int))
    return open_days[np.minimum(index, len(open_days) - 1)]


def build_labs(rng, p):
    adult = np.flatnonzero(p['age'].to_numpy() >= 18)
    a1c_who, a1c_day = a1c_schedule(rng, p)

    # Basic metabolic panels (glucose and creatinine together) follow visit frequency;
    # most A1c draws add one. Lipid panels are yearly for adults over 40 and for diabetes.
    bmp_n = rng.poisson(3.15 * p['rate'].to_numpy()[adult])
    bmp_who = np.repeat(adult, bmp_n)
    bmp_day = draw_open_days(rng, len(bmp_who))
    with_a1c = rng.random(len(a1c_who)) < 0.85
    age = p['age'].to_numpy()
    lipid_p = np.where(p['diabetic'].to_numpy(), 0.85, np.where(age >= 40, 0.55, 0.12))
    ldl_who = np.flatnonzero((age >= 18) & (rng.random(len(p)) < lipid_p))
    ldl_day = draw_open_days(rng, len(ldl_who))

    draw_who = np.concatenate([a1c_who, a1c_who[with_a1c], bmp_who, ldl_who])
    draw_day = np.concatenate([a1c_day, a1c_day[with_a1c], bmp_day, ldl_day])
    kind = np.concatenate([np.full(len(a1c_who), 'A1c'), np.full(with_a1c.sum(), 'BMP'),
                           np.full(len(bmp_who), 'BMP'), np.full(len(ldl_who), 'LDL')])
    open_day = to_open_day(draw_day)
    # About 60% of draws are fasting and cluster before 10:00; the rest spread across the day.
    # A panel drawn with an A1c shares its collection time.
    morning = np.minimum(7 * 60 + rng.gamma(2.0, 45, len(draw_who)), 16 * 60 + 59)
    minute = np.where(rng.random(len(draw_who)) < 0.6, morning, rng.uniform(7 * 60, 17 * 60, len(draw_who))).astype(int)
    minute[len(a1c_who):len(a1c_who) + with_a1c.sum()] = minute[:len(a1c_who)][with_a1c]
    when = DAYS[open_day].to_numpy() + minute.astype('timedelta64[m]')

    # A panel gives two rows: glucose and creatinine at the same time.
    bmp = kind == 'BMP'
    who = np.concatenate([draw_who, draw_who[bmp]])
    when = np.concatenate([when, when[bmp]])
    minute = np.concatenate([minute, minute[bmp]])
    test = np.concatenate([np.where(bmp, 'Glucose', kind), np.full(bmp.sum(), 'Creatinine')])

    # Trim to the extract size by dropping whole random rows (a fixed count makes the demos easy to check).
    assert len(who) >= N_LABS, len(who)
    keep = np.sort(rng.choice(len(who), N_LABS, replace=False))
    who, when, minute, test = who[keep], when[keep], minute[keep], test[keep]
    n = N_LABS
    month_frac = (pd.DatetimeIndex(when).dayofyear.to_numpy() - 1) / 365

    diabetic, base = p['diabetic'].to_numpy()[who], p['a1c_base'].to_numpy()[who]
    # Intensified treatment lowers A1c about 0.8 points over the year for patients at 8% or more;
    # others drift up about 0.1 point a year.
    trend = np.where(diabetic & (base >= 8), -0.8, 0.1 * diabetic)
    a1c_now = base + trend * month_frac
    result = np.empty(n)
    a1c = test == 'A1c'
    # Assay error: about 2% coefficient of variation.
    result[a1c] = np.clip(a1c_now[a1c] * (1 + rng.normal(0, 0.02, a1c.sum())), 4.0, 15.0).round(1)
    glu = test == 'Glucose'
    # ADAG: average glucose = 28.7 x A1c - 46.7 mg/dL. Draws after 10:00 are often not fasting.
    average = 28.7 * a1c_now[glu] - 46.7
    fasting = np.where(minute[glu] < 10 * 60, 0.88, 1.0) * np.where(diabetic[glu], 1.0, 0.92)
    spread = np.where(diabetic[glu], 0.22, 0.09)
    # The analyzer reads 45 to 900 mg/dL; values above 600 are rare emergencies.
    result[glu] = np.clip(average * fasting * rng.lognormal(0, spread), 45, 900).round()
    ldl = test == 'LDL'
    # Statins lower LDL about 35%; men and middle age run higher.
    ldl_mean = (112 + 8 * p['male'].to_numpy()[who[ldl]] + 0.3 * (np.clip(age[who[ldl]], 18, 65) - 45)
                ) * np.where(p['statin'].to_numpy()[who[ldl]], 0.65, 1.0)
    result[ldl] = np.clip(ldl_mean * rng.lognormal(0, 0.25, ldl.sum()), 25, 330).round()
    cre = test == 'Creatinine'
    # Creatinine from the patient's kidney function (simplified CKD-EPI inverse), 6% visit-to-visit noise.
    sex_base = np.where(p['male'].to_numpy()[who[cre]], 0.9, 0.7)
    egfr = p['egfr'].to_numpy()[who[cre]]
    result[cre] = np.clip(sex_base * (90 / egfr) ** 1.1 * rng.lognormal(0, 0.06, cre.sum()), 0.3, 12).round(2)

    names = CLINICS['clinic'].to_numpy()
    clinic = names[p['home'].to_numpy()[who]].astype(object)
    clinic[(clinic == 'Parkside') & (when < PARKSIDE_OPENS.to_datetime64())] = 'Sunset'
    # Cancelled or hemolyzed samples have no result; Ridgecrest's courier delays spoil more.
    spoiled = rng.random(n) < np.where(clinic == 'Ridgecrest', 0.02, 0.005)
    result[spoiled] = np.nan
    # One Excelsior analyzer reports a failed glucose run as 9999, a code that is not a result.
    error = np.flatnonzero(glu & (clinic == 'Excelsior') & ~spoiled)
    result[rng.choice(error, 23, replace=False)] = 9999

    unit = pd.Series(test).map({'Glucose': 'mg/dL', 'A1c': '%', 'LDL': 'mg/dL', 'Creatinine': 'mg/dL'})
    labs = pd.DataFrame({
        'patient_id': ids('P', who + 1, 6),
        'clinic': pd.array(clinic, dtype='str'),
        'collected_at': pd.DatetimeIndex(when).as_unit('us'),
        'test_name': test,
        'result': result,
        'unit': unit,
    })
    return labs.sort_values(['collected_at', 'patient_id', 'test_name'], kind='stable', ignore_index=True)


def check(clinics, visits, labs):
    assert len(clinics) == 9 and len(visits) == N_VISITS and len(labs) == N_LABS
    assert visits['visit_id'].is_unique
    assert set(visits['clinic'].dropna()) == set(clinics['clinic']) - {'Oceanview'}
    assert 2500 <= visits['clinic'].isna().sum() <= 3500
    assert visits.loc[visits['clinic'] == 'Parkside', 'visit_month'].min() == '2025-07'
    assert visits['wait_min'].dropna().between(0, 240).all()
    assert 0.2 <= visits['satisfaction'].notna().mean() <= 0.4
    per_patient = visits['patient_id'].value_counts()
    assert per_patient.max() >= 40 and (per_patient == 1).mean() >= 0.2
    combo = visits.groupby(['clinic', 'department']).size()
    assert ('Marina', 'Orthopedics') not in combo.index and ('Ridgecrest', 'Dermatology') not in combo.index
    ridge = visits[(visits['clinic'] == 'Ridgecrest') & (visits['department'] == 'Cardiology')]
    assert ridge['patient_id'].nunique() == 10
    october = visits[(visits['visit_month'] == '2025-10') & (visits['clinic'] == 'Excelsior')]
    assert october['wait_min'].isna().mean() > 0.4
    assert (labs['result'] == 9999).sum() == 23
    a1c = labs[labs['test_name'] == 'A1c'].dropna(subset=['result'])
    assert a1c['result'].between(4, 15).all()
    gaps = a1c.sort_values('collected_at').groupby('patient_id')['collected_at'].diff().dropna()
    assert gaps.min() >= pd.Timedelta(days=75), gaps.min()


def build(data_dir='data'):
    """Write the three tables as Parquet under data_dir and return their paths."""
    rng = np.random.default_rng(217)
    patients = build_patients(rng)
    tables = {'clinics': CLINICS, 'visits': build_visits(rng, patients), 'labs': build_labs(rng, patients)}
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: data_dir / f'{name}.parquet' for name in tables}
    for name, table in tables.items():
        table.to_parquet(paths[name], index=False)
    check(*(pd.read_parquet(paths[name]) for name in tables))
    return paths


if __name__ == '__main__':
    for name, path in build().items():
        print(f'Wrote {path}')
