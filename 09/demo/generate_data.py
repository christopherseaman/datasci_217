"""Build the synthetic health-system extracts for the Lecture 09 demos.

build() writes seven Parquet tables under data/:

- ed_visits: 2023 to 2024 emergency department (ED) visits at four sites, timestamps as text (Demos 1 and 3).
- stays, telemetry, charted_temps: a 24-bed step-down unit, January to March 2024, with a bedside
  monitor reading every 5 minutes and nurse-charted temperatures every 4 hours (Demo 2).
- encounters, ward_vitals, labs: 2024 ward encounters at three hospitals in three time zones,
  vitals on each hospital's wall clock and lab times in UTC (Demo 3).

Columns are generated in this causal order (an arrow means "is an input to"):

    ED
    site catchment -> age, sex, deprivation (area index 1-10) -> insurance, smoking
    age, deprivation, smoking -> chronic conditions (count, COPD) -> how often a person visits
    hour of day, weekday, flu season, site -> arrivals per hour -> crowding (arrivals in the last 6 hours)
    flu season, COPD, age -> respiratory complaint
    age, conditions, hour -> ambulance arrival
    age, conditions, ambulance, respiratory -> acuity -> triage level (ESI 1 to 5)
    crowding, ESI, site -> wait for a provider -> left without being seen (LWBS)
    ESI, age, conditions, site -> admitted -> boarding time (longer in flu season and on weekends)
    ESI, wait, boarding -> length of stay; LWBS -> departure often never recorded
    crowding -> triage level left blank; plus typed dates, a feed outage, and impossible text dates

    Step-down unit
    age -> COPD, heart failure on a beta-blocker, atrial fibrillation; admission severity
    age, severity, COPD -> deterioration (ICU transfer) -> 6 to 18 hour ramp in HR and SpO2
    beta-blocker, AF, severity, time of day, fever -> heart rate; COPD, severity -> SpO2
    severity, daytime -> time off the monitor (imaging, walks, showers)
    daytime movement -> motion artifact spikes in HR; probe slipping -> false low SpO2, then blanks
    fever episodes (more with COPD; sepsis before transfer) -> charted temperature and heart rate

    Ward
    season, hour, weekday -> admission time; season -> diagnosis (pneumonia and COPD in winter)
    age, conditions, diagnosis -> severity at admission
    severity, site -> ICU transfer and its timing; severity -> length of stay
    severity, COPD, diagnosis, deterioration ramp -> heart rate and SpO2
    worry (severity, ramp) -> how often vitals are charted; night round skipped for stable patients
    severity, diagnosis -> which labs are ordered (sicker patients get lactates) -> lab values
    test, site, night collection -> lab turnaround; wall-clock typing on clock-change nights

The seed is fixed, so every run writes the same values. All patients and values are synthetic.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 909
TEXT_FORMAT = '%Y-%m-%d %H:%M'
HOUR_US = 3_600_000_000
MINUTE_US = 60_000_000


def logistic(x):
    return 1 / (1 + np.exp(-x))


def patient_ids(rng, n):
    """Unique synthetic medical record numbers in random order."""
    return np.array([f'MRN{v:07d}' for v in rng.choice(9_000_000, n, replace=False) + 1_000_000])


def flu_intensity(times):
    """0 to 1 influenza activity: the 2022-23 tail, the 2023-24 peak near New Year, and the 2024-25 rise."""
    flu = np.zeros(len(times))
    for peak, width_days in [('2022-12-12', 22), ('2024-01-01', 20), ('2025-01-28', 24)]:
        days_from = ((times - pd.Timestamp(peak)) / pd.Timedelta(days=1)).to_numpy()
        flu = np.maximum(flu, np.exp(-0.5 * (days_from / width_days) ** 2))
    return flu


# ---------------------------------------------------------------- ED visits (Demos 1 and 3)

ED_SITES = {'Central': 330, 'Northside': 240, 'Eastside': 170, 'Harbor': 100}  # mean visits per day
ED_DEPRIVATION = {'Central': 6.5, 'Northside': 3.0, 'Eastside': 7.5, 'Harbor': 5.0}  # catchment mean decile
ED_AGE_SHIFT = {'Central': 0, 'Northside': 3, 'Eastside': -4, 'Harbor': 8}  # Harbor serves a retirement area
ED_WAIT_LOG = {'Central': 0.15, 'Northside': -0.10, 'Eastside': 0.25, 'Harbor': -0.20}  # staffing per visit
ED_ADMIT_LOG = {'Central': 0.10, 'Northside': -0.05, 'Eastside': -0.15, 'Harbor': 0.20}
ED_OUTAGE = ('2024-08-12', '2024-08-21')  # Eastside sent no rows for these 9 days (end excluded)
# Relative arrivals by hour of day: quiet before dawn, busiest late morning to evening
# (US EDs see about 4 to 5 times as many arrivals at 11:00 as at 04:00).
ED_HOURLY = np.array([0.55, 0.45, 0.38, 0.33, 0.30, 0.32, 0.45, 0.70, 0.98, 1.22, 1.38, 1.45,
                      1.42, 1.38, 1.35, 1.33, 1.32, 1.34, 1.36, 1.30, 1.18, 1.02, 0.85, 0.68])
ED_WEEKDAY = np.array([1.12, 1.03, 1.00, 0.98, 0.97, 0.93, 0.97])  # Monday first
ESI_P = [0.008, 0.14, 0.44, 0.33, 0.082]  # national triage mix, ESI 1 to 5
BAD_ARRIVALS = ['', 'UNKNOWN', '2023-02-29 08:15', '2024-06-31 14:40', '2024-11-05 24:10']
BAD_EACH = 30  # each bad pattern appears this many times
# Catchment residents per ED visit, and the gamma shape of how often each visits. Together they give about 1.6
# visits per patient a year, with the 4-or-more group about 7% of patients making about a fifth of visits,
# within the ranges US ED studies report.
ED_CATCHMENT = 2.0
ED_FRAILTY_SHAPE = 0.25


def build_ed_population(rng, site, n):
    """One site's catchment: who lives there, how sick they are, and how often they use the ED."""
    age = np.clip(1 + 92 * rng.beta(1.5, 2.0, n) + ED_AGE_SHIFT[site], 0, 99).round()
    deprivation = np.clip(rng.normal(ED_DEPRIVATION[site], 1.8, n), 1, 10).round()
    adult = age >= 18
    smoker = adult & (rng.random(n) < logistic(-1.7 + 0.22 * (deprivation - 5)))  # about 14%, more where deprived
    conditions = np.minimum(rng.poisson(np.exp(-2.4 + 0.035 * age + 0.07 * (deprivation - 5) + 0.3 * smoker)), 7)
    copd = adult & (rng.random(n) < logistic(-6.0 + 0.055 * age + 1.5 * smoker))  # about 6% of adults
    medicare = (age >= 65) & (rng.random(n) < 0.93)
    medicaid = ~medicare & (rng.random(n) < logistic(-1.8 + 0.45 * (deprivation - 5) + 1.0 * (age < 18)))
    uninsured = ~medicare & ~medicaid & (rng.random(n) < 0.12)
    insurance = np.select([medicare, medicaid, uninsured], ['Medicare', 'Medicaid', 'uninsured'], 'commercial')
    # Gamma frailty gives the long tail of frequent visitors seen in every ED.
    propensity = rng.gamma(ED_FRAILTY_SHAPE, 1, n) * np.exp(0.22 * conditions + 0.06 * (deprivation - 5)
                                                + 0.35 * (insurance == 'uninsured') + 0.6 * copd)
    return pd.DataFrame({'patient_id': patient_ids(rng, n), 'age': age, 'conditions': conditions,
                         'copd': copd, 'propensity': propensity})


def build_ed_visits(rng):
    hours = pd.date_range('2023-01-01', '2024-12-31 23:00', freq='h')
    flu_h = flu_intensity(hours)
    doy = hours.dayofyear.to_numpy()
    season = (1 + 0.03 * np.cos(2 * np.pi * (doy - 200) / 365.25)) * (1 + 0.30 * flu_h)  # summer injuries; flu peaks add up to 30%
    shape = ED_HOURLY[hours.hour] / ED_HOURLY.mean() * ED_WEEKDAY[hours.dayofweek] * season
    day_of = ((hours - hours[0]) // pd.Timedelta(days=1)).to_numpy()
    starts = hours.as_unit('us').asi8

    parts = []
    for site, per_day in ED_SITES.items():
        day_effect = rng.gamma(250, 1 / 250, day_of.max() + 1)  # day-to-day spread beyond Poisson, about 6%
        counts = rng.poisson(per_day / 24 * shape * day_effect[day_of])
        recent = np.convolve(counts, np.ones(6))[:len(counts)]  # arrivals in this hour and the 5 before
        crowd = recent / recent.mean()
        rows = np.repeat(np.arange(len(hours)), counts)
        parts.append(pd.DataFrame({
            'site': site,
            'arrival_us': starts[rows] + rng.integers(0, 60, len(rows)) * MINUTE_US,
            'crowd': crowd[rows], 'flu': flu_h[rows],
        }))
    visits = pd.concat(parts, ignore_index=True)
    visits['arrival'] = pd.to_datetime(visits['arrival_us'], unit='us')
    outage = (visits['site'] == 'Eastside') & visits['arrival'].between(*ED_OUTAGE, inclusive='left')
    visits = visits[~outage].sort_values(['site', 'arrival']).reset_index(drop=True)
    n = len(visits)

    # Each visit belongs to a resident of the site's catchment, weighted by how often they visit;
    # respiratory visits in flu season come more often from older adults and people with COPD.
    hour = visits['arrival'].dt.hour.to_numpy()
    respiratory = rng.random(n) < 0.07 + 0.22 * visits['flu'].to_numpy()
    cols = {k: np.empty(n, dtype=object if k == 'patient_id' else float) for k in ['patient_id', 'age', 'conditions', 'copd']}
    for site in ED_SITES:
        rows = np.flatnonzero(visits['site'].to_numpy() == site)
        pop = build_ed_population(rng, site, int(len(rows) * ED_CATCHMENT))
        weight = pop['propensity'].to_numpy()
        resp_weight = weight * (1 + 2.5 * pop['copd'] + 0.8 * (pop['age'] >= 65) + 0.6 * (pop['age'] < 5)).to_numpy()
        pick = np.empty(len(rows), dtype=np.int64)
        r = respiratory[rows]
        pick[~r] = rng.choice(len(pop), (~r).sum(), p=weight / weight.sum())
        pick[r] = rng.choice(len(pop), r.sum(), p=resp_weight / resp_weight.sum())
        for k in cols:
            cols[k][rows] = pop[k].to_numpy()[pick]
    age, conditions, copd = cols['age'], cols['conditions'], cols['copd'].astype(bool)
    night = (hour < 7) | (hour >= 23)
    crowd = visits['crowd'].to_numpy()
    site = visits['site'].to_numpy()

    # About 17% arrive by ambulance, more often older, sicker, and at night.
    ambulance = rng.random(n) < logistic(-2.3 + 0.03 * (age - 45) + 0.25 * conditions + 0.5 * night)
    acuity = (0.022 * (age - 45) + 0.22 * conditions + 1.0 * ambulance + 0.5 * (respiratory & copd)
              + 0.25 * night + rng.normal(0, 1, n))
    cuts = np.quantile(acuity, 1 - np.cumsum(ESI_P)[:-1])  # the most acute 0.8% are ESI 1, and so on
    esi = 1 + sum((acuity < c).astype(int) for c in cuts)

    # Door-to-provider wait: median about 25 minutes, longer when crowded (exponent about 1.5).
    wait_min = (24 * crowd ** 1.5 * np.array([0, 0.05, 0.35, 1.0, 1.25, 1.3])[esi]
                * np.exp(pd.Series(site).map(ED_WAIT_LOG).to_numpy() + rng.normal(0, 0.7, n)))
    p_lwbs = np.where(esi >= 3, logistic(-4.6 + 1.1 * np.log(wait_min / 30) + 0.35 * (esi - 3) - 0.01 * (age - 45)), 0)
    lwbs = rng.random(n) < p_lwbs
    p_admit = logistic(np.array([0, 1.6, 0.0, -1.7, -3.6, -4.8])[esi] + 0.022 * (age - 45) + 0.18 * conditions
                       + 0.4 * (respiratory & (age >= 65)) + pd.Series(site).map(ED_ADMIT_LOG).to_numpy())
    u = rng.random(n)
    admitted = ~lwbs & (u < p_admit)
    transferred = ~lwbs & ~admitted & (u < p_admit + 0.02 * (esi <= 3))
    disposition = np.select([lwbs, admitted, transferred],
                            ['left without being seen', 'admitted', 'transferred'], 'discharged')

    treat_h = (np.array([0, 4.0, 3.6, 3.0, 1.5, 1.0])[esi] * (1 + 0.004 * (age - 45)).clip(0.8)
               * np.exp(rng.normal(0, 0.5, n)))
    weekend = visits['arrival'].dt.dayofweek.to_numpy() >= 5
    # Admitted patients wait for an inpatient bed ("boarding"): median about 2 hours, doubling at the flu peak.
    board_h = 2.0 * (1 + 1.4 * visits['flu'].to_numpy()) * np.where(weekend, 1.2, 1.0) * np.exp(rng.normal(0, 0.8, n))
    los_h = wait_min / 60 + treat_h + np.where(admitted, board_h, 0) + np.where(transferred, 1.5, 0)
    los_h = np.where(lwbs, wait_min / 60 * rng.uniform(0.5, 1.0, n), los_h)
    los_min = np.clip(los_h, 0.2, 48) * 60
    departure = visits['arrival'] + pd.to_timedelta(los_min.round(), unit='min')

    arrival_text = visits['arrival'].dt.strftime(TEXT_FORMAT).to_numpy(dtype=object)
    departure_text = departure.dt.strftime(TEXT_FORMAT).to_numpy(dtype=object)
    no_departure = (lwbs & (rng.random(n) < 0.45)) | (rng.random(n) < 0.003)  # LWBS patients are rarely checked out
    departure_text[no_departure] = None
    typo = rng.choice(np.flatnonzero(~no_departure), 75, replace=False)  # departure typed with the previous day's date
    departure_text[typo] = (departure.iloc[typo] - pd.Timedelta(days=1)).dt.strftime(TEXT_FORMAT).to_numpy()
    bad = rng.choice(n, BAD_EACH * len(BAD_ARRIVALS), replace=False)
    arrival_text[bad] = np.repeat(BAD_ARRIVALS, BAD_EACH)

    esi_col = pd.array(esi, dtype='Int64')
    esi_col[rng.random(n) < 0.002 + 0.02 * np.clip(crowd - 1.2, 0, 1)] = pd.NA  # skipped when the ED is crowded
    return pd.DataFrame({
        'visit_id': [f'V{i:08d}' for i in range(1, n + 1)],
        'site': site,
        'patient_id': cols['patient_id'],
        'age': age.astype(int),
        'arrival_mode': np.where(ambulance, 'ambulance', 'walk-in'),
        'arrival_time': pd.array(arrival_text, dtype='str'),
        'departure_time': pd.array(departure_text, dtype='str'),
        'esi': esi_col,
        'disposition': disposition,
    })


# ---------------------------------------------------------------- Step-down unit (Demo 2)

def smooth_noise(rng, n, sd, width=6):
    """Slowly varying noise: a moving average of white noise, rescaled to standard deviation sd."""
    raw = rng.normal(0, 1, n + width - 1)
    return np.convolve(raw, np.ones(width) / width, mode='valid') * sd * np.sqrt(width)


def build_step_down(rng):
    start, end = pd.Timestamp('2024-01-01'), pd.Timestamp('2024-03-31 23:55')
    stays = []
    for bed in range(1, 25):
        admit = start + pd.Timedelta(minutes=int(rng.integers(0, 12 * 60)))
        while admit < end:
            severity = rng.normal(0, 1)
            # Length of stay: lognormal, median about 2.5 days, longer for sicker patients.
            los_min = int(np.clip(60 * 60 * np.exp(0.15 * severity + rng.normal(0, 0.55)), 8 * 60, 14 * 24 * 60))
            stays.append({'bed': f'4W-{bed:02d}', 'admitted_at': admit,
                          'left_at': min(admit + pd.Timedelta(minutes=los_min), end), 'severity': severity})
            admit = admit + pd.Timedelta(minutes=los_min) + pd.Timedelta(
                minutes=int(np.clip(240 * np.exp(rng.normal(0, 0.7)), 45, 2880)))  # bed cleaned and refilled
    stays = pd.DataFrame(stays).sort_values('admitted_at').reset_index(drop=True)
    n = len(stays)
    stays.insert(0, 'stay_id', [f'S{i:04d}' for i in range(1, n + 1)])
    ids = patient_ids(rng, n)
    readmit = np.flatnonzero(rng.random(n) < 0.10)
    readmit = readmit[readmit > 20]
    ids[readmit] = [ids[rng.integers(0, i - 5)] for i in readmit]  # same patient as an earlier stay
    stays.insert(1, 'patient_id', ids)
    age = np.clip(rng.normal(68, 13, n), 25, 97).round()
    copd = rng.random(n) < logistic(-3.2 + 0.035 * (age - 40))  # about 20% on a step-down unit
    beta_blocker = rng.random(n) < logistic(-1.6 + 0.04 * (age - 60))  # heart failure or rate control
    afib = rng.random(n) < logistic(-3.0 + 0.06 * (age - 60))
    severity = stays['severity'].to_numpy()
    # About 6% are transferred to the ICU; odds roughly double per SD of admission severity.
    deteriorate = (rng.random(n) < logistic(-3.4 + 0.85 * severity + 0.02 * (age - 68) + 0.4 * copd)) & (stays['left_at'] < end)
    stays['age'] = age.astype(int)
    stays['outcome'] = np.where(deteriorate, 'ICU transfer',
                                np.where(stays['left_at'] >= end, 'still on unit', 'discharged'))

    tele_parts, temp_parts = [], []
    for i, row in enumerate(stays.itertuples()):
        det = deteriorate[i]
        times = pd.date_range(row.admitted_at.ceil('5min'), row.left_at.floor('5min'), freq='5min')
        m = len(times)
        clock = times.hour.to_numpy() + times.minute.to_numpy() / 60
        awake = (clock >= 8) & (clock < 21)
        hours_to_end = ((row.left_at - times) / pd.Timedelta(hours=1)).to_numpy()
        hours_in = ((times - row.admitted_at) / pd.Timedelta(hours=1)).to_numpy()

        # Fever episodes on the 5-minute grid: 12% of stays (20% with COPD), plus sepsis before a transfer.
        fever = np.zeros(m)
        if rng.random() < (0.20 if copd[i] else 0.12) and m > 200:
            onset, length = rng.integers(0, m - 100), rng.integers(100, 600)
            bump = np.zeros(m)
            bump[onset:onset + length] = rng.uniform(1.8, 3.0)
            fever = np.convolve(bump, np.ones(36) / 36, mode='same')  # rises and falls over about 3 hours
        ramp = np.zeros(m)
        if det:
            ramp = np.clip(1 - hours_to_end / rng.uniform(6, 18), 0, 1)
            fever = fever + rng.uniform(0.5, 2.2) * ramp

        # Heart rate: about 4.5 bpm per degree F of fever; 6 bpm day-night swing with a trough near 04:00.
        hr = (np.clip(rng.normal(84, 10), 58, 115) - 9 * beta_blocker[i] + 3 * severity[i]
              + 6 * np.cos(2 * np.pi * (clock - 16) / 24) + 4.5 * fever
              + smooth_noise(rng, m, 7.0 if afib[i] else 3.5) - 0.03 * hours_in.clip(0, 96))
        spo2 = (np.clip(rng.normal(96.5, 1.0), 93, 99) - 2.5 * copd[i] - 0.5 * severity[i]
                + smooth_noise(rng, m, 0.8))
        if det:
            hr = hr + rng.uniform(22, 38) * ramp
            spo2 = spo2 - rng.uniform(3, 6) * (1.3 if copd[i] else 1.0) * ramp
        hr = hr + rng.normal(0, 1.5, m)  # beat-to-beat measurement noise
        spikes = np.flatnonzero(rng.random(m) < np.where(awake, 0.0015, 0.0004))  # motion artifact
        for first in spikes:
            k = len(hr[first:first + rng.integers(1, 4)])
            hr[first:first + k] = rng.uniform(140, 185, k)
        hr = np.clip(hr, 35, 190).round()
        hr[rng.random(m) < 0.008] = np.nan  # lead off for one reading

        spo2 = np.clip(spo2, 82, 100).round()
        for first in np.flatnonzero(rng.random(m) < 0.006):  # probe slips: a false low, then blanks
            spo2[first] = rng.integers(70, 86)
            spo2[first + 1:first + rng.integers(3, 13)] = np.nan

        # Off the monitor: about 1.3 breaks a day, mostly in the daytime, more for sicker patients.
        keep = np.ones(m, dtype=bool)
        rate = 1.3 * np.exp(0.25 * severity[i]) / 288 * np.where(awake, 1.7, 0.3)
        for first in np.flatnonzero(rng.random(m) < rate):
            kind = rng.random()
            length = (rng.integers(2, 10) if kind < 0.70 else rng.integers(12, 31) if kind < 0.95
                      else rng.integers(36, 97))  # 5-minute steps: walks and showers, imaging, procedures
            keep[first:first + length] = False
        tele_parts.append(pd.DataFrame({'stay_id': row.stay_id, 'recorded_at': times[keep],
                                        'heart_rate': hr[keep], 'spo2': spo2[keep]}))

        rounds = pd.date_range(row.admitted_at.ceil('4h'), row.left_at, freq='4h')
        k = len(rounds)
        at = np.clip(np.searchsorted(times, rounds), 0, m - 1)
        temp = 98.2 + 0.3 * np.cos(2 * np.pi * (rounds.hour.to_numpy() - 18) / 24) + fever[at] + rng.normal(0, 0.25, k)
        # The 04:00 round is skipped most often, for stable patients who are asleep.
        skip = np.where(rounds.hour == 4, 0.25 if fever.max() < 0.5 else 0.08, 0.05)
        charted = rng.random(k) >= skip
        temp = temp.round(1)
        temp[rng.random(k) < 0.015] = np.nan  # row started, value never entered
        temp_parts.append(pd.DataFrame({'stay_id': row.stay_id, 'charted_at': rounds[charted],
                                        'temperature_f': temp[charted]}))

    telemetry = pd.concat(tele_parts, ignore_index=True)
    telemetry.insert(1, 'patient_id', telemetry['stay_id'].map(stays.set_index('stay_id')['patient_id']))
    temps = pd.concat(temp_parts, ignore_index=True)
    stays = stays[['stay_id', 'patient_id', 'bed', 'age', 'admitted_at', 'left_at', 'outcome']]
    return stays, telemetry, temps


# ---------------------------------------------------------------- Ward encounters (Demo 3)

WARD_SITES = {'New York': 'America/New_York', 'Chicago': 'America/Chicago', 'Phoenix': 'America/Phoenix'}
WARD_ENCOUNTERS = {'New York': 4200, 'Chicago': 3300, 'Phoenix': 2500}
WARD_TRANSFER_LOG = {'New York': 0.10, 'Chicago': 0.0, 'Phoenix': -0.15}  # ICU bed policy differs by hospital
LAB_SLOWER = {'New York': 1.0, 'Chicago': 1.1, 'Phoenix': 1.25}  # Phoenix sends chemistry to a central lab
STANDARD_OFFSET_H = {'New York': -5, 'Chicago': -6, 'Phoenix': -7}
DIAGNOSES = ['pneumonia', 'sepsis', 'heart failure', 'COPD', 'GI bleed', 'other']
DX_SEVERITY = np.array([0.4, 0.9, 0.3, 0.3, 0.2, -0.2])
# Hospital admissions by hour of day: most come up from the ED in the afternoon and evening.
ADMIT_HOURLY = np.array([0.6, 0.5, 0.4, 0.35, 0.3, 0.3, 0.35, 0.5, 0.7, 0.9, 1.1, 1.3,
                         1.45, 1.55, 1.6, 1.6, 1.55, 1.5, 1.45, 1.4, 1.3, 1.1, 0.9, 0.75])
SPRING_FORWARD_ENTRIES = 4  # manual chart entries typed at 02:xx on March 10, 2024, a time that never happened


def build_ward(rng):
    # Admission days weighted by season and weekday, hours by the ED pattern above.
    days = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    flu_d = flu_intensity(days)
    day_w = (1 + 0.25 * flu_d) * np.where(days.dayofweek >= 5, 0.85, 1.0)
    enc_parts = []
    for site, n in WARD_ENCOUNTERS.items():
        d = rng.choice(len(days), n, p=day_w / day_w.sum())
        h = rng.choice(24, n, p=ADMIT_HOURLY / ADMIT_HOURLY.sum())
        local = days[d] + pd.to_timedelta(h * 60 + rng.integers(0, 60, n), unit='min')
        admitted = local.tz_localize(WARD_SITES[site], ambiguous=True, nonexistent='shift_forward').tz_convert('UTC')
        enc_parts.append(pd.DataFrame({'site': site, 'admitted_at': admitted, 'flu': flu_d[d]}))
    enc = pd.concat(enc_parts, ignore_index=True).sort_values('admitted_at').reset_index(drop=True)
    n = len(enc)
    enc.insert(0, 'encounter_id', [f'E{i:05d}' for i in range(1, n + 1)])
    ids = patient_ids(rng, n)
    readmit = np.flatnonzero(rng.random(n) < 0.12)
    readmit = readmit[readmit > 50]
    ids[readmit] = [ids[rng.integers(max(0, i - 3000), i - 10)] for i in readmit]
    enc.insert(1, 'patient_id', ids)

    age = np.clip(rng.normal(66, 16, n), 18, 99).round()
    conditions = np.minimum(rng.poisson(np.exp(-0.9 + 0.025 * age)), 8)
    flu = enc['flu'].to_numpy()
    dx_p = np.array([0.12, 0.08, 0.12, 0.07, 0.06, 0.55])
    dx_w = np.tile(dx_p, (n, 1))
    dx_w[:, 0] *= 1 + 1.2 * flu  # pneumonia and COPD exacerbations rise in flu season
    dx_w[:, 3] *= 1 + 1.0 * flu
    dx = (rng.random(n)[:, None] > np.cumsum(dx_w / dx_w.sum(1, keepdims=True), axis=1)).sum(1)
    copd = (dx == 3) | (rng.random(n) < 0.10)
    severity = DX_SEVERITY[dx] + 0.02 * (age - 66) + 0.15 * (conditions - 2) + rng.normal(0, 1, n)
    site = enc['site'].to_numpy()
    # About 6% go to the ICU; odds rise about 2.5-fold per unit of severity, which no single vital captures.
    deteriorate = rng.random(n) < logistic(-3.6 + 0.95 * severity + pd.Series(site).map(WARD_TRANSFER_LOG).to_numpy())
    los_h = np.clip(80 * np.exp(0.18 * severity + rng.normal(0, 0.65, n)), 10, 21 * 24)
    # Transfers come 4 hours to about a week in, most in the first 3 days.
    transfer_h = 4 + rng.gamma(1.3, 40, n)
    los_h = np.where(deteriorate, np.minimum(transfer_h, 21 * 24), los_h)
    enc['left_at'] = (enc['admitted_at'] + pd.to_timedelta((los_h * 60).round(), unit='min')).dt.floor('min')
    enc['outcome'] = np.where(deteriorate, 'ICU transfer', 'discharged')
    enc['age'] = age.astype(int)
    enc['diagnosis'] = np.array(DIAGNOSES)[dx]

    rounds_by_site = {}
    for s, zone in WARD_SITES.items():
        local_rounds = pd.date_range('2023-12-31', '2025-01-02', freq='4h')  # 00:00, 04:00, ... on the wall clock
        utc = local_rounds.tz_localize(zone, ambiguous=True, nonexistent='shift_forward').tz_convert('UTC')
        rounds_by_site[s] = (utc.as_unit('us').asi8, local_rounds.hour.to_numpy())

    enc_rows, time_parts, hr_parts, spo2_parts = [], [], [], []
    for i, row in enumerate(enc.itertuples()):
        det, s_i = deteriorate[i], severity[i]
        a, b = row.admitted_at.value // 1000, row.left_at.value // 1000  # microseconds
        grid, grid_hour = rounds_by_site[row.site]
        inside = (grid >= a) & (grid < b)
        times, hours = grid[inside], grid_hour[inside]
        skip = np.where(hours == 4, 0.30 * logistic(-1.5 * s_i), 0.05)  # stable patients are left to sleep
        times = times[rng.random(len(times)) >= skip]
        times = times + (np.clip(rng.normal(0, 12, len(times)), -40, 40).round() * MINUTE_US).astype(np.int64)
        extra = a + (rng.random(rng.poisson((0.5 + 0.4 * max(s_i, 0)) * (b - a) / (24 * HOUR_US))) * (b - a)).astype(np.int64)
        times = np.concatenate([times, extra])
        ramp_h = rng.uniform(10, 30)
        if det:  # hourly checks once the nurse is worried, over the last 12 to 24 hours
            span = int(rng.uniform(12, 24) * HOUR_US)
            times = np.concatenate([times, np.arange(max(a, b - span), b, HOUR_US) + rng.integers(-10, 11) * MINUTE_US])
        times = np.unique((times // MINUTE_US) * MINUTE_US)
        times = times[(times >= a) & (times < b)]
        m = len(times)
        hours_to_end = (b - times) / HOUR_US
        ramp = np.clip(1 - hours_to_end / ramp_h, 0, 1) if det else np.zeros(m)
        infection = dx[i] in (0, 1)
        hr = (np.clip(rng.normal(80, 9), 55, 115) + 5 * s_i + 6 * infection
              - 2.5 * np.minimum((times - a) / (24 * HOUR_US), 4)  # settles as treatment works
              + rng.normal(0, 6, m) + rng.uniform(20, 35) * ramp)
        spo2 = (np.clip(rng.normal(96.5, 1.0), 93, 99) - 0.8 * s_i - 2.5 * copd[i] - 1.0 * (dx[i] == 0)
                + rng.normal(0, 1, m) - rng.uniform(3, 6) * ramp)
        hr = hr.round()
        hr[rng.random(m) < 0.015] = np.nan
        enc_rows.append(np.full(m, i))
        time_parts.append(times)
        hr_parts.append(hr)
        spo2_parts.append(np.clip(spo2, 78, 100).round())
    which = np.concatenate(enc_rows)
    vitals = pd.DataFrame({
        'encounter_id': enc['encounter_id'].to_numpy()[which],
        'patient_id': enc['patient_id'].to_numpy()[which],
        'site': site[which],
        't': np.concatenate(time_parts),
        'heart_rate': np.concatenate(hr_parts),
        'spo2': np.concatenate(spo2_parts),
    })
    local = pd.Series(pd.NaT, index=vitals.index, dtype='datetime64[us]')
    for s, zone in WARD_SITES.items():
        rows = vitals['site'] == s
        utc = pd.to_datetime(vitals.loc[rows, 't'], unit='us', utc=True)
        local[rows] = utc.dt.tz_convert(zone).dt.tz_localize(None).astype('datetime64[us]')
    text = local.dt.strftime(TEXT_FORMAT).to_numpy(dtype=object)
    # Manual entries typed on the spring-forward night at 02:xx, a clock time that never happened.
    spring_hour = local.between(pd.Timestamp('2024-03-10 03:00'), pd.Timestamp('2024-03-10 03:59'))
    spring = np.flatnonzero((spring_hour & vitals['site'].isin(['New York', 'Chicago'])).to_numpy())
    for j in rng.choice(spring, SPRING_FORWARD_ENTRIES, replace=False):
        text[j] = text[j].replace(' 03:', ' 02:')
    vitals['recorded_local'] = pd.array(text, dtype='str')
    vitals = (vitals.sample(frac=1, random_state=SEED)  # the export arrives in no particular order
              [['encounter_id', 'patient_id', 'site', 'recorded_local', 'heart_rate', 'spo2']]
              .reset_index(drop=True))

    lab_rows = []
    for i, row in enumerate(enc.itertuples()):
        det, s_i, d = deteriorate[i], severity[i], dx[i]
        a, b = row.admitted_at.value // 1000, row.left_at.value // 1000  # microseconds

        def add(test, t, value):
            lab_rows.append((row.encounter_id, row.site, test, t, value))
        first = a + int(rng.uniform(0.3, 2.0) * HOUR_US)
        # Confounding by indication: lactate is drawn for patients who look sick or septic.
        if rng.random() < logistic(-1.3 + 1.0 * s_i + 2.2 * (d == 1) + 0.6 * (d == 0)):
            add('lactate', first, rng.lognormal(np.log(1.3) + 0.22 * s_i + 0.3 * (d == 1), 0.35))
        if rng.random() < (0.6 if d == 2 else 0.2):
            add('troponin', first, rng.lognormal(np.log(8) + 0.4 * (d == 2) + 0.2 * s_i, 0.9))
            add('troponin', first + 3 * HOUR_US, rng.lognormal(np.log(9) + 0.4 * (d == 2) + 0.2 * s_i, 0.9))
        if rng.random() < (0.95 if d == 1 else 0.6 if d == 0 else 0.1 + 0.3 * det):
            add('blood culture', first + int(0.2 * HOUR_US), np.nan)
        base_cr = rng.lognormal(np.log(0.95) + 0.008 * (age[i] - 66) + 0.1 * conditions[i] / 2, 0.25)
        t = first
        while t < b:
            add('creatinine', t, base_cr * np.exp(0.08 * s_i + rng.normal(0, 0.08)))
            t += 24 * HOUR_US + int(rng.normal(0, 1.5) * HOUR_US)
        if det:
            t = b - int(rng.uniform(4, 12) * HOUR_US)
            while t < b:
                add('lactate', t, rng.lognormal(np.log(2.8) + 0.15 * s_i, 0.3))
                t += int(rng.uniform(2, 6) * HOUR_US)
    labs = pd.DataFrame(lab_rows, columns=['encounter_id', 'site', 'test', 'collected_us', 'value'])
    local_hour = ((labs['collected_us'] // HOUR_US) + labs['site'].map(STANDARD_OFFSET_H)) % 24
    median_min = labs['test'].map({'lactate': 35, 'troponin': 60, 'creatinine': 95, 'blood culture': 2400})
    # Lognormal turnaround; the night shift (00:00 to 06:59 local) runs about 25% slower.
    turnaround = (median_min * labs['site'].map(LAB_SLOWER) * np.where(local_hour < 7, 1.25, 1.0)
                  * np.exp(rng.normal(0, 0.45, len(labs)))).round()
    collected = pd.to_datetime(labs['collected_us'], unit='us', utc=True).dt.floor('min')
    resulted = collected + pd.to_timedelta(turnaround, unit='min')
    labs['collected_at'] = collected.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    resulted_text = resulted.dt.strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy(dtype=object)
    resulted_text[rng.random(len(labs)) < 0.012] = ''  # cancelled or never resulted
    labs['resulted_at'] = pd.array(resulted_text, dtype='str')
    labs['value'] = labs['value'].round(2)
    labs = labs[['encounter_id', 'test', 'value', 'collected_at', 'resulted_at']]
    enc = enc[['encounter_id', 'patient_id', 'site', 'age', 'diagnosis', 'admitted_at', 'left_at', 'outcome']]
    return enc, vitals, labs


def check(t):
    ed = t['ed_visits']
    assert 590_000 <= len(ed) <= 660_000 and ed['visit_id'].is_unique
    assert ed['arrival_time'].isin(BAD_ARRIVALS).sum() == BAD_EACH * len(BAD_ARRIVALS)
    arrival = pd.to_datetime(ed['arrival_time'], format=TEXT_FORMAT, errors='coerce')
    assert arrival.isna().sum() == BAD_EACH * len(BAD_ARRIVALS)
    east_days = arrival[ed['site'] == 'Eastside'].dt.floor('D')
    assert not east_days.between(*ED_OUTAGE, inclusive='left').any()
    assert ed['patient_id'].value_counts().max() >= 20  # some frequent visitors
    by_hour = arrival.dt.hour.value_counts()
    assert by_hour[11] > 3.5 * by_hour[4]
    share = ed['disposition'].value_counts(normalize=True)
    assert 0.13 <= share['admitted'] <= 0.22 and 0.01 <= share['left without being seen'] <= 0.05

    stays, tele = t['stays'], t['telemetry']
    assert 600 <= len(stays) <= 900 and stays['stay_id'].is_unique
    assert 500_000 <= len(tele) <= 900_000
    assert (tele['recorded_at'].dt.minute % 5 == 0).all()
    assert tele['heart_rate'].dropna().between(35, 190).all() and tele['heart_rate'].isna().any()
    assert (stays['outcome'] == 'ICU transfer').sum() >= 25

    enc, vit, labs = t['encounters'], t['ward_vitals'], t['labs']
    assert len(enc) == sum(WARD_ENCOUNTERS.values()) and enc['encounter_id'].is_unique
    assert enc['patient_id'].duplicated().sum() > 500  # readmissions
    assert 0.04 <= (enc['outcome'] == 'ICU transfer').mean() <= 0.09
    assert 200_000 <= len(vit) <= 400_000
    local = pd.to_datetime(vit['recorded_local'], format=TEXT_FORMAT)
    flagged = 0
    for site in ['New York', 'Chicago']:
        rows = vit['site'] == site
        aware = local[rows].dt.tz_localize(WARD_SITES[site], ambiguous='NaT', nonexistent='NaT')
        flagged += aware.isna().sum()
    assert flagged > SPRING_FORWARD_ENTRIES  # the typed 02:xx entries plus the repeated 01:xx hour
    phoenix = local[vit['site'] == 'Phoenix'].dt.tz_localize('America/Phoenix', ambiguous='raise', nonexistent='raise')
    assert phoenix.notna().all()
    assert pd.to_datetime(labs['collected_at'], utc=True).notna().all()


def build(data_dir='data'):
    """Write the seven tables as Parquet under data_dir and return their paths."""
    # One random stream per builder, so a change to one extract leaves the others unchanged.
    tables = {'ed_visits': build_ed_visits(np.random.default_rng([SEED, 1]))}
    tables['stays'], tables['telemetry'], tables['charted_temps'] = build_step_down(np.random.default_rng([SEED, 2]))
    tables['encounters'], tables['ward_vitals'], tables['labs'] = build_ward(np.random.default_rng([SEED, 3]))
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        paths[name] = data_dir / f'{name}.parquet'
        table.to_parquet(paths[name], index=False)
    check({name: pd.read_parquet(path) for name, path in paths.items()})
    return paths


if __name__ == '__main__':
    for name, path in build().items():
        print(f'Wrote {path}')
