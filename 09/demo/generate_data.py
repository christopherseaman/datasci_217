"""Build the synthetic health-system extracts for the Lecture 09 demos.

build() writes seven Parquet tables under data/:

- ed_visits: 2023 to 2024 emergency department (ED) visits at four sites, timestamps as text (Demos 1 and 3).
- stays, telemetry, charted_temps: a 24-bed step-down unit, January to March 2024, with a bedside
  monitor reading every 5 minutes and nurse-charted temperatures every 4 hours (Demo 2).
- encounters, ward_vitals, labs: 2024 ward encounters at three hospitals in three time zones,
  vitals on each hospital's wall clock and lab times in UTC (Demo 3).

Each table's people and visits come from a causal graph declared once below (the *_EDGES strings; an
arrow means "is an input to"). Every node has a formula plus noise, run by dowhy.gcm as a structural
causal model, so confounding such as "sicker patients get a lactate and also go to the ICU" comes from
the edges. Four graphs:

    ED_POPULATION_EDGES   catchment residents: site -> age, deprivation -> smoking, conditions, COPD,
                          insurance -> how often each resident visits
    ED_VISIT_EDGES        visits: flu season, crowding, night, the patient -> respiratory complaint,
                          ambulance, triage level (ESI) -> wait -> left without being seen (LWBS)
                          -> disposition -> length of stay; LWBS -> departure never recorded
    STAY_EDGES            step-down stays: age, severity -> COPD, beta-blocker, AF, deterioration,
                          fever, baseline heart rate and SpO2
    WARD_EDGES            ward encounters: flu season -> diagnosis -> severity -> ICU transfer, length
                          of stay, which labs are ordered and their values, baseline vitals

Two documented gcm calls do the sampling. gcm.draw_samples draws the step-down stays, whose roots are
all random. Where a root comes from a calendar process (the hour a visit arrived, how crowded the ED
was, an admission's flu season) or from the resident a visit belongs to, sample() passes those values
as observed data, and gcm.interventional_samples with an identity intervention on the roots draws every
other node from its mechanism.

Readings are expanded from each unit with numpy, because a gcm row is one unit: arrivals per hour and
crowding (arrivals in the last 6 hours); the 5-minute monitor feed (day-night swing, fever episodes,
a 6 to 18 hour deterioration ramp, motion artifacts, probe slips, time off the monitor that is longer
for sicker patients and in the daytime); 4-hourly temperatures (high readings rechecked under the same
round time; the 04:00 round skipped for stable patients); ward vitals charted more often when a
patient is sicker or deteriorating; repeat lactates before an ICU transfer, rising from the patient's
own lactate; lab collection times and turnaround (slower at night).

Small documented post-steps add record-keeping events tied to the calendar or a feed: Eastside's
9-day feed outage, impossible arrival text, departures typed with the previous day's date, manual
entries typed in the spring-forward hour, readmissions, and the export's row order.

The seeds are fixed, so every run writes the same bytes. All patients and values are synthetic.
"""

import inspect
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # dowhy 0.14 warns about its own regex strings and, in Jupyter, about tqdm widgets
    from dowhy import gcm
    from dowhy.gcm.causal_models import PARENTS_DURING_FIT

SEED = 909
TEXT_FORMAT = '%Y-%m-%d %H:%M'
HOUR_US = 3_600_000_000
MINUTE_US = 60_000_000


def logistic(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- gcm plumbing

class Formula(gcm.PredictionModel):
    """The f in an additive noise model Y = f(parents) + N, written as a function of named parents."""

    def __init__(self, fn, parents):
        self.fn, self.parents = fn, parents

    def fit(self, X, Y):
        pass  # the formula is declared, not learned

    def predict(self, X):
        return np.asarray(self.fn(**dict(zip(self.parents, X.T.astype(float)))), dtype=float).reshape(-1, 1)

    def clone(self):
        return self


class Probabilities(gcm.ClassificationModel):
    """Class probabilities as a function of named parents, for gcm.ClassifierFCM."""

    def __init__(self, fn, parents, classes):
        self.fn, self.parents, self._classes = fn, parents, list(classes)

    def fit(self, X, Y):
        pass

    def predict_probabilities(self, X):
        p = np.asarray(self.fn(**dict(zip(self.parents, X.T.astype(float)))), dtype=float)
        return np.column_stack([1 - p, p]) if p.ndim == 1 else p

    def predict(self, X):
        return np.array(self._classes)[self.predict_probabilities(X).argmax(1)]

    @property
    def classes(self):
        return self._classes

    def clone(self):
        return self


def mechanism(fn, make):
    """A node's formula (kept for the parent check in causal_model) and how to build its gcm mechanism."""
    make.fn = fn
    return make


def normal(fn, sd):
    """Y = fn(parents) + Normal(0, sd)."""
    return mechanism(fn, lambda parents: gcm.AdditiveNoiseModel(
        Formula(fn, parents), gcm.ScipyDistribution(stats.norm, loc=0, scale=sd)))


def additive(fn, distribution, **parameters):
    """Y = fn(parents) + noise from any scipy distribution."""
    return mechanism(fn, lambda parents: gcm.AdditiveNoiseModel(
        Formula(fn, parents), gcm.ScipyDistribution(distribution, **parameters)))


def bernoulli(fn):
    """Y is 1 with probability fn(parents), else 0."""
    return mechanism(fn, lambda parents: gcm.ClassifierFCM(Probabilities(fn, parents, [0, 1])))


def categorical(fn, classes):
    """Y is one of classes, with fn(parents) returning one probability column per class."""
    return mechanism(fn, lambda parents: gcm.ClassifierFCM(Probabilities(fn, parents, classes)))


def root(distribution, **parameters):
    """A root node drawn from a scipy distribution."""
    return mechanism(None, lambda parents: gcm.ScipyDistribution(distribution, **parameters))


def poisson_probs(mean, top):
    """Poisson probabilities for 0 to top, with the tail folded into top."""
    p = stats.poisson.pmf(np.arange(top + 1)[None, :], np.asarray(mean)[:, None])
    p[:, -1] += 1 - p.sum(1)
    return p


def graph(edges):
    """A DiGraph from lines such as 'age, smoker -> copd'."""
    g = nx.DiGraph()
    for line in edges.strip().splitlines():
        sources, target = line.split('->')
        g.add_node(target.strip())
        g.add_edges_from((s.strip(), target.strip()) for s in sources.split(','))
    return g


def causal_model(edges, mechanisms, given=()):
    """A gcm.StructuralCausalModel whose formulas take exactly each node's parents in the graph."""
    g = graph(edges)
    g.add_nodes_from(mechanisms)
    scm = gcm.StructuralCausalModel(g)
    for node in g.nodes:
        parents = sorted(g.predecessors(node))  # gcm passes parents in sorted order
        if node in given:  # observed values replace this root; gcm only needs a placeholder mechanism
            assert not parents, f'{node} is given but has parents'
            scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        else:
            make = mechanisms[node]
            inputs = set(inspect.signature(make.fn).parameters) if make.fn else set()
            assert inputs == set(parents), f'{node}: formula inputs {sorted(inputs)} differ from its graph parents {parents}'
            scm.set_causal_mechanism(node, make(parents))
        # The formulas are declared, not fitted, so record each node's parents as gcm.fit and
        # gcm.create_causal_model_from_equations do; gcm checks this before it samples.
        scm.graph.nodes[node][PARENTS_DURING_FIT] = parents
    return scm


def sample(scm, seed, n=None, given=None):
    """gcm.draw_samples when every root is random; with given, those roots are fixed and the rest is drawn."""
    gcm.util.general.set_random_seed(seed)
    if given is None:
        return gcm.draw_samples(scm, n)
    data = given.reset_index(drop=True).copy()
    roots = [v for v in scm.graph.nodes if scm.graph.in_degree(v) == 0]
    for node in roots:
        if node not in data:
            data[node] = scm.causal_mechanism(node).draw_samples(len(data)).reshape(-1)
    return gcm.interventional_samples(scm, {node: (lambda x: x) for node in roots}, observed_data=data)


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

ED_SITES = ['Central', 'Northside', 'Eastside', 'Harbor']  # site codes 0 to 3
ED_PER_DAY = np.array([330, 240, 170, 100])  # mean visits per day
ED_DEPRIVATION = np.array([6.5, 3.0, 7.5, 5.0])  # catchment mean area deprivation decile
ED_AGE_SHIFT = np.array([0, 3, -4, 8])  # Harbor serves a retirement area
ED_WAIT_LOG = np.array([0.15, -0.10, 0.25, -0.20])  # staffing per visit
ED_ADMIT_LOG = np.array([0.10, -0.05, -0.15, 0.20])
ED_OUTAGE = ('2024-08-12', '2024-08-21')  # Eastside sent no rows for these 9 days (end excluded)
# Relative arrivals by hour of day: quiet before dawn, busiest late morning to evening
# (US EDs see about 4 to 5 times as many arrivals at 11:00 as at 04:00).
ED_HOURLY = np.array([0.55, 0.45, 0.38, 0.33, 0.30, 0.32, 0.45, 0.70, 0.98, 1.22, 1.38, 1.45,
                      1.42, 1.38, 1.35, 1.33, 1.32, 1.34, 1.36, 1.30, 1.18, 1.02, 0.85, 0.68])
ED_WEEKDAY = np.array([1.12, 1.03, 1.00, 0.98, 0.97, 0.93, 0.97])  # Monday first
# Cut points on the acuity scale that give the national triage mix (ESI 1 to 5: 0.8%, 14%, 44%, 33%, 8%).
ESI_CUTS = np.array([3.88, 1.79, 0.04, -1.39])
ESI_WAIT = np.array([0, 0.02, 0.35, 1.0, 1.25, 1.3])  # door-to-provider wait multiplier by ESI
ESI_ADMIT_LOG = np.array([0, 1.6, 0.0, -1.7, -3.6, -4.8])
ESI_TREAT_H = np.array([0, 4.0, 3.6, 3.0, 1.5, 1.0])  # median treatment hours by ESI
DISPOSITIONS = ['discharged', 'admitted', 'transferred', 'left without being seen']  # codes 0 to 3
BAD_ARRIVALS = ['', 'UNKNOWN', '2023-02-29 08:15', '2024-06-31 14:40', '2024-11-05 24:10']
BAD_EACH = 30  # each bad pattern appears this many times
# Catchment residents per ED visit, and the gamma shape of how often each visits. Together they give about 1.6
# visits per patient a year, with the 4-or-more group about 7% of patients making about a fifth of visits,
# within the ranges US ED studies report.
ED_CATCHMENT = 2.0
ED_FRAILTY_SHAPE = 0.25

ED_POPULATION_EDGES = """
site -> age
site -> deprivation
age, deprivation -> smoker
age, deprivation, smoker -> conditions
age, smoker -> copd
age, deprivation -> insurance
conditions, copd, deprivation, insurance -> log_propensity
"""


def _insurance(age, deprivation):
    medicare = (age >= 65) * 0.93
    medicaid = (1 - medicare) * logistic(-1.8 + 0.45 * (deprivation - 5) + 1.0 * (age < 18))
    uninsured = (1 - medicare - medicaid) * 0.12
    return np.column_stack([medicare, medicaid, uninsured, 1 - medicare - medicaid - uninsured])


ED_POPULATION = {
    # Age 0 to 93 from a right-skewed beta, shifted by catchment; deprivation decile around the catchment mean.
    'age': additive(lambda site: 1 + ED_AGE_SHIFT[site.astype(int)], stats.beta, a=1.5, b=2.0, scale=92),
    'deprivation': normal(lambda site: ED_DEPRIVATION[site.astype(int)], 1.8),
    'smoker': bernoulli(lambda age, deprivation: (age >= 18) * logistic(-1.7 + 0.22 * (deprivation - 5))),  # about 14%
    # Chronic conditions: about 3.5% more per year of age, more where deprived and among smokers.
    'conditions': categorical(lambda age, deprivation, smoker: poisson_probs(
        np.exp(-2.4 + 0.035 * age + 0.07 * (deprivation - 5) + 0.3 * smoker), 7), range(8)),
    'copd': bernoulli(lambda age, smoker: (age >= 18) * logistic(-6.0 + 0.055 * age + 1.5 * smoker)),  # about 6% of adults
    'insurance': categorical(_insurance, range(4)),  # Medicare, Medicaid, uninsured, commercial
    # How often a resident visits: gamma frailty (the log of a gamma variable is loggamma) times exp(risk),
    # which gives the long tail of frequent visitors seen in every ED.
    'log_propensity': additive(lambda conditions, copd, deprivation, insurance: 0.22 * conditions + 0.06 * (deprivation - 5)
                               + 0.35 * (insurance == 2) + 0.6 * copd, stats.loggamma, c=ED_FRAILTY_SHAPE),
}

ED_VISIT_EDGES = """
age, copd, flu -> respiratory
age, conditions, night -> ambulance
age, ambulance, conditions, copd, night, respiratory -> esi
crowd, esi, site -> log_wait
age, esi, log_wait -> lwbs
age, conditions, esi, lwbs, respiratory, site -> disposition
age, esi -> log_treat
flu, weekend -> log_board
disposition, log_board, log_treat, log_wait -> los_hours
disposition -> departure_recorded
crowd -> esi_recorded
"""


def _esi(age, ambulance, conditions, copd, night, respiratory):
    # Latent acuity: older, sicker, by ambulance, COPD with a respiratory complaint, and at night.
    acuity = (0.022 * (age - 45) + 0.22 * conditions + 1.0 * ambulance + 0.5 * respiratory * copd + 0.25 * night)
    at_least = stats.norm.cdf(acuity[:, None] - ESI_CUTS[None, :])  # P(ESI 1), P(ESI 1 or 2), ...
    return np.diff(np.column_stack([np.zeros(len(acuity)), at_least, np.ones(len(acuity))]), axis=1)


def _disposition(age, conditions, esi, lwbs, respiratory, site):
    p_admit = logistic(ESI_ADMIT_LOG[esi.astype(int)] + 0.022 * (age - 45) + 0.18 * conditions
                       + 0.4 * respiratory * (age >= 65) + ED_ADMIT_LOG[site.astype(int)])
    p_transfer = np.minimum(0.02 * (esi <= 3), 1 - p_admit)
    seen = 1 - lwbs
    return np.column_stack([seen * (1 - p_admit - p_transfer), seen * p_admit, seen * p_transfer, lwbs])


def _los_hours(disposition, log_board, log_treat, log_wait):
    wait_h = np.exp(log_wait) / 60
    stay = wait_h + np.exp(log_treat) + (disposition == 1) * np.exp(log_board) + (disposition == 2) * 1.5
    return np.where(disposition == 3, 0.75 * wait_h, stay)  # LWBS patients leave partway through the wait


ED_VISIT = {
    # About 7% of visits are respiratory, up to about 30% at the flu peak; more with COPD, over 65, and under 5.
    'respiratory': bernoulli(lambda age, copd, flu: logistic(-2.75 + 1.9 * flu + 1.3 * copd + 0.5 * (age >= 65) + 0.4 * (age < 5))),
    # About 15% arrive by ambulance, more often older, sicker, and at night.
    'ambulance': bernoulli(lambda age, conditions, night: logistic(-2.3 + 0.03 * (age - 45) + 0.25 * conditions + 0.5 * night)),
    'esi': categorical(_esi, [1, 2, 3, 4, 5]),
    # Door-to-provider wait in minutes: median about 25, rising with crowding (exponent 1.5), noise sd 0.7 on the log scale.
    'log_wait': normal(lambda crowd, esi, site: np.log(24) + 1.5 * np.log(crowd) + np.log(ESI_WAIT[esi.astype(int)])
                       + ED_WAIT_LOG[site.astype(int)], 0.7),
    # Leaving without being seen: odds triple per tripling of the wait; ESI 1 and 2 are seen at once.
    'lwbs': bernoulli(lambda age, esi, log_wait: (esi >= 3) * logistic(-4.6 + 1.1 * (log_wait - np.log(30))
                                                                        + 0.35 * (esi - 3) - 0.01 * (age - 45))),
    'disposition': categorical(_disposition, range(4)),
    'log_treat': normal(lambda age, esi: np.log(ESI_TREAT_H[esi.astype(int)]) + np.log(np.clip(1 + 0.004 * (age - 45), 0.8, None)), 0.5),
    # Admitted patients wait for an inpatient bed ("boarding"): median about 2 hours, more than doubling at the flu peak.
    'log_board': normal(lambda flu, weekend: np.log(2.0) + np.log(1 + 1.4 * flu) + np.log(1.2) * weekend, 0.8),
    'los_hours': normal(_los_hours, 0.1),
    'departure_recorded': bernoulli(lambda disposition: np.where(disposition == 3, 0.55, 0.997)),  # nobody checks out a walkout
    'esi_recorded': bernoulli(lambda crowd: 1 - 0.002 - 0.02 * np.clip(crowd - 1.2, 0, 1)),  # skipped when crowded
}


def ed_arrivals(rng):
    """Visits per hour for each site, and how crowded the ED was when each visit arrived."""
    hours = pd.date_range('2023-01-01', '2024-12-31 23:00', freq='h')
    flu_h = flu_intensity(hours)
    doy = hours.dayofyear.to_numpy()
    season = (1 + 0.03 * np.cos(2 * np.pi * (doy - 200) / 365.25)) * (1 + 0.30 * flu_h)  # summer injuries; flu peaks add up to 30%
    shape = ED_HOURLY[hours.hour] / ED_HOURLY.mean() * ED_WEEKDAY[hours.dayofweek] * season
    day_of = ((hours - hours[0]) // pd.Timedelta(days=1)).to_numpy()
    starts = hours.as_unit('us').asi8
    parts = []
    for code, per_day in enumerate(ED_PER_DAY):
        day_effect = rng.gamma(250, 1 / 250, day_of.max() + 1)  # day-to-day spread beyond Poisson, about 6%
        counts = rng.poisson(per_day / 24 * shape * day_effect[day_of])
        recent = np.convolve(counts, np.ones(6))[:len(counts)]  # arrivals in this hour and the 5 before
        rows = np.repeat(np.arange(len(hours)), counts)
        parts.append(pd.DataFrame({'site': code, 'arrival_us': starts[rows] + rng.integers(0, 60, len(rows)) * MINUTE_US,
                                   'crowd': (recent / recent.mean())[rows], 'flu': flu_h[rows]}))
    visits = pd.concat(parts, ignore_index=True)
    visits['arrival'] = pd.to_datetime(visits['arrival_us'], unit='us')
    outage = (visits['site'] == ED_SITES.index('Eastside')) & visits['arrival'].between(*ED_OUTAGE, inclusive='left')
    return visits[~outage].sort_values(['site', 'arrival']).reset_index(drop=True)


def build_ed_visits(rng):
    visits = ed_arrivals(rng)
    n = len(visits)
    site = visits['site'].to_numpy()

    # Each visit belongs to a resident of its site's catchment, drawn in proportion to how often they visit.
    population = sample(causal_model(ED_POPULATION_EDGES, ED_POPULATION, given={'site'}), SEED + 11,
                        given=pd.DataFrame({'site': np.repeat(np.arange(4), (np.bincount(site) * ED_CATCHMENT).astype(int))}))
    population['patient_id'] = patient_ids(rng, len(population))
    pick = np.empty(n, dtype=np.int64)
    for code in range(4):
        residents = np.flatnonzero(population['site'].to_numpy() == code)
        weight = np.exp(population['log_propensity'].to_numpy()[residents])
        rows = np.flatnonzero(site == code)
        pick[rows] = rng.choice(residents, len(rows), p=weight / weight.sum())
    patient = population.iloc[pick].reset_index(drop=True)

    hour = visits['arrival'].dt.hour.to_numpy()
    context = pd.DataFrame({'site': site, 'flu': visits['flu'], 'crowd': visits['crowd'],
                            'night': ((hour < 7) | (hour >= 23)).astype(int),
                            'weekend': (visits['arrival'].dt.dayofweek.to_numpy() >= 5).astype(int),
                            'age': patient['age'], 'conditions': patient['conditions'], 'copd': patient['copd']})
    v = sample(causal_model(ED_VISIT_EDGES, ED_VISIT, given=set(context)), SEED + 12, given=context)

    age = np.clip(v['age'].to_numpy(), 0, 99).round()
    los_min = np.clip(v['los_hours'].to_numpy(), 0.2, 48) * 60
    departure = visits['arrival'] + pd.to_timedelta(los_min.round(), unit='min')
    arrival_text = visits['arrival'].dt.strftime(TEXT_FORMAT).to_numpy(dtype=object)
    departure_text = departure.dt.strftime(TEXT_FORMAT).to_numpy(dtype=object)
    no_departure = v['departure_recorded'].to_numpy() == 0
    departure_text[no_departure] = None

    # Post-steps: departures typed with the previous day's date, and arrival text no parser accepts.
    typo = rng.choice(np.flatnonzero(~no_departure), 75, replace=False)
    departure_text[typo] = (departure.iloc[typo] - pd.Timedelta(days=1)).dt.strftime(TEXT_FORMAT).to_numpy()
    bad = rng.choice(n, BAD_EACH * len(BAD_ARRIVALS), replace=False)
    arrival_text[bad] = np.repeat(BAD_ARRIVALS, BAD_EACH)

    esi = pd.array(v['esi'].to_numpy().astype(int), dtype='Int64')
    esi[v['esi_recorded'].to_numpy() == 0] = pd.NA
    return pd.DataFrame({
        'visit_id': [f'V{i:08d}' for i in range(1, n + 1)],
        'site': np.array(ED_SITES)[site],
        'patient_id': patient['patient_id'].to_numpy(),
        'age': age.astype(int),
        'arrival_mode': np.where(v['ambulance'].to_numpy() == 1, 'ambulance', 'walk-in'),
        'arrival_time': pd.array(arrival_text, dtype='str'),
        'departure_time': pd.array(departure_text, dtype='str'),
        'esi': esi,
        'disposition': np.array(DISPOSITIONS)[v['disposition'].to_numpy().astype(int)],
    })


# ---------------------------------------------------------------- Step-down unit (Demo 2)

STAY_EDGES = """
age -> copd
age -> beta_blocker
age -> afib
severity -> log_los_min
age, copd, severity -> deteriorate
copd -> fever_episode
beta_blocker, severity -> hr_base
copd, severity -> spo2_base
"""

STAY = {
    'age': root(stats.norm, loc=68, scale=13),
    'severity': root(stats.norm, loc=0, scale=1),  # admission severity, in standard deviations
    'copd': bernoulli(lambda age: logistic(-3.2 + 0.035 * (age - 40))),  # about 20% on a step-down unit
    'beta_blocker': bernoulli(lambda age: logistic(-1.6 + 0.04 * (age - 60))),  # heart failure or rate control
    'afib': bernoulli(lambda age: logistic(-3.0 + 0.06 * (age - 60))),
    # Length of stay: lognormal, median about 2.5 days, longer for sicker patients.
    'log_los_min': normal(lambda severity: np.log(60 * 60) + 0.15 * severity, 0.55),
    # About 6% are transferred to the ICU; odds roughly double per SD of admission severity.
    'deteriorate': bernoulli(lambda age, copd, severity: logistic(-3.4 + 0.85 * severity + 0.02 * (age - 68) + 0.4 * copd)),
    'fever_episode': bernoulli(lambda copd: np.where(copd == 1, 0.20, 0.12)),
    # Resting heart rate: 9 bpm lower on a beta-blocker, 3 bpm higher per SD of severity.
    'hr_base': normal(lambda beta_blocker, severity: 84 - 9 * beta_blocker + 3 * severity, 10),
    'spo2_base': normal(lambda copd, severity: 96.5 - 2.5 * copd - 0.5 * severity, 1.0),
}


def smooth_noise(rng, n, sd, width=6):
    """Slowly varying noise: a moving average of white noise, rescaled to standard deviation sd."""
    raw = rng.normal(0, 1, n + width - 1)
    return np.convolve(raw, np.ones(width) / width, mode='valid') * sd * np.sqrt(width)


def build_step_down(rng):
    start, end = pd.Timestamp('2024-01-01'), pd.Timestamp('2024-03-31 23:55')
    pool = sample(causal_model(STAY_EDGES, STAY), SEED + 21, n=1200)  # more patients than the beds can take
    los_min = np.clip(np.exp(pool['log_los_min'].to_numpy()), 8 * 60, 14 * 24 * 60).astype(int)
    stays, k = [], 0
    for bed in range(1, 25):
        admit = start + pd.Timedelta(minutes=int(rng.integers(0, 12 * 60)))
        while admit < end:
            stays.append({'bed': f'4W-{bed:02d}', 'admitted_at': admit, 'pool_row': k,
                          'left_at': min(admit + pd.Timedelta(minutes=int(los_min[k])), end)})
            admit = admit + pd.Timedelta(minutes=int(los_min[k])) + pd.Timedelta(
                minutes=int(np.clip(240 * np.exp(rng.normal(0, 0.7)), 45, 2880)))  # bed cleaned and refilled
            k += 1
    stays = pd.DataFrame(stays).sort_values('admitted_at').reset_index(drop=True)
    p = pool.iloc[stays['pool_row']].reset_index(drop=True)
    n = len(stays)
    stays.insert(0, 'stay_id', [f'S{i:04d}' for i in range(1, n + 1)])
    ids = patient_ids(rng, n)
    readmit = np.flatnonzero(rng.random(n) < 0.10)
    readmit = readmit[readmit > 20]
    ids[readmit] = [ids[rng.integers(0, i - 5)] for i in readmit]  # same patient as an earlier stay
    stays.insert(1, 'patient_id', ids)
    severity, copd = p['severity'].to_numpy(), p['copd'].to_numpy() == 1
    afib, fever_episode = p['afib'].to_numpy() == 1, p['fever_episode'].to_numpy() == 1
    hr_base, spo2_base = p['hr_base'].to_numpy(), p['spo2_base'].to_numpy()
    deteriorate = (p['deteriorate'].to_numpy() == 1) & (stays['left_at'] < end).to_numpy()  # still on unit at the archive end
    stays['age'] = np.clip(p['age'].to_numpy(), 25, 97).round().astype(int)
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

        # A fever episode rises and falls over about 3 hours; sepsis adds fever before a transfer.
        fever = np.zeros(m)
        if fever_episode[i] and m > 200:
            onset, length = rng.integers(0, m - 100), rng.integers(100, 600)
            bump = np.zeros(m)
            bump[onset:onset + length] = rng.uniform(1.8, 3.0)
            fever = np.convolve(bump, np.ones(36) / 36, mode='same')
        ramp = np.zeros(m)
        if det:
            ramp = np.clip(1 - hours_to_end / rng.uniform(6, 18), 0, 1)
            fever = fever + rng.uniform(0.5, 2.2) * ramp

        # Heart rate: about 4.5 bpm per degree F of fever; 6 bpm day-night swing with a trough near 04:00;
        # AF doubles the slow wander.
        hr = (hr_base[i] + 6 * np.cos(2 * np.pi * (clock - 16) / 24) + 4.5 * fever
              + smooth_noise(rng, m, 7.0 if afib[i] else 3.5) - 0.03 * hours_in.clip(0, 96))
        spo2 = spo2_base[i] + smooth_noise(rng, m, 0.8)
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
    # A high reading is often rechecked and charted again under the same round time; a few others are entered twice.
    recheck = rng.random(len(temps)) < np.where(temps['temperature_f'] >= 99.5, 0.4, 0.003)
    again = temps[recheck].copy()
    again['temperature_f'] = (again['temperature_f'] + rng.normal(0, 0.3, len(again))).round(1)
    temps = pd.concat([temps, again]).sort_values(['stay_id', 'charted_at'], kind='stable').reset_index(drop=True)
    stays = stays[['stay_id', 'patient_id', 'bed', 'age', 'admitted_at', 'left_at', 'outcome']]
    return stays, telemetry, temps


# ---------------------------------------------------------------- Ward encounters (Demo 3)

WARD_SITES = {'New York': 'America/New_York', 'Chicago': 'America/Chicago', 'Phoenix': 'America/Phoenix'}  # codes 0 to 2
WARD_ENCOUNTERS = {'New York': 4200, 'Chicago': 3300, 'Phoenix': 2500}
WARD_TRANSFER_LOG = np.array([0.10, 0.0, -0.15])  # ICU bed policy differs by hospital
LAB_SLOWER = {'New York': 1.0, 'Chicago': 1.1, 'Phoenix': 1.25}  # Phoenix sends chemistry to a central lab
STANDARD_OFFSET_H = {'New York': -5, 'Chicago': -6, 'Phoenix': -7}
DIAGNOSES = ['pneumonia', 'sepsis', 'heart failure', 'COPD', 'GI bleed', 'other']  # codes 0 to 5
DX_P = np.array([0.12, 0.08, 0.12, 0.07, 0.06, 0.55])
DX_SEVERITY = np.array([0.4, 0.9, 0.3, 0.3, 0.2, -0.2])
# Hospital admissions by hour of day: most come up from the ED in the afternoon and evening.
ADMIT_HOURLY = np.array([0.6, 0.5, 0.4, 0.35, 0.3, 0.3, 0.35, 0.5, 0.7, 0.9, 1.1, 1.3,
                         1.45, 1.55, 1.6, 1.6, 1.55, 1.5, 1.45, 1.4, 1.3, 1.1, 0.9, 0.75])
SPRING_FORWARD_ENTRIES = 4  # manual chart entries typed at 02:xx on March 10, 2024, a time that never happened

WARD_EDGES = """
age -> conditions
flu -> diagnosis
diagnosis -> copd
age, conditions, diagnosis -> severity
severity, site -> deteriorate
severity -> log_los_h
deteriorate, log_los_h, transfer_h -> stay_h
diagnosis, severity -> lactate_ordered
diagnosis, severity -> log_lactate
diagnosis -> troponin_ordered
diagnosis, severity -> log_troponin
deteriorate, diagnosis -> culture_ordered
age, conditions -> log_creatinine
diagnosis, severity -> hr_base
copd, diagnosis, severity -> spo2_base
"""


def _diagnosis(flu):
    w = np.tile(DX_P, (len(flu), 1))
    w[:, 0] *= 1 + 1.2 * flu  # pneumonia and COPD exacerbations rise in flu season
    w[:, 3] *= 1 + 1.0 * flu
    return w / w.sum(1, keepdims=True)


WARD = {
    'age': root(stats.norm, loc=66, scale=16),
    # Hours from admission to an ICU transfer: 4 hours to about a week, most in the first 3 days.
    'transfer_h': root(stats.gamma, a=1.3, loc=4, scale=40),
    'conditions': categorical(lambda age: poisson_probs(np.exp(-0.9 + 0.025 * age), 8), range(9)),
    'diagnosis': categorical(_diagnosis, range(6)),
    'copd': bernoulli(lambda diagnosis: np.where(diagnosis == 3, 1.0, 0.10)),
    'severity': normal(lambda age, conditions, diagnosis: DX_SEVERITY[diagnosis.astype(int)] + 0.02 * (age - 66)
                       + 0.15 * (conditions - 2), 1.0),
    # About 6% go to the ICU; odds rise about 2.6-fold per unit of severity, which no single vital captures.
    'deteriorate': bernoulli(lambda severity, site: logistic(-3.6 + 0.95 * severity + WARD_TRANSFER_LOG[site.astype(int)])),
    'log_los_h': normal(lambda severity: np.log(80) + 0.18 * severity, 0.65),  # median about 3.3 days
    # A transfer ends the ward stay; otherwise the stay runs 10 hours to 21 days. Noise is under a minute.
    'stay_h': additive(lambda deteriorate, log_los_h, transfer_h: np.where(
        deteriorate == 1, np.minimum(transfer_h, 21 * 24), np.clip(np.exp(log_los_h), 10, 21 * 24)),
        stats.uniform, loc=0, scale=1 / 60),
    # Confounding by indication: lactate is drawn for patients who look sick or septic.
    'lactate_ordered': bernoulli(lambda diagnosis, severity: logistic(-1.3 + 1.0 * severity + 2.2 * (diagnosis == 1) + 0.6 * (diagnosis == 0))),
    'log_lactate': normal(lambda diagnosis, severity: np.log(1.3) + 0.22 * severity + 0.3 * (diagnosis == 1), 0.35),  # mmol/L
    'troponin_ordered': bernoulli(lambda diagnosis: np.where(diagnosis == 2, 0.6, 0.2)),
    'log_troponin': normal(lambda diagnosis, severity: np.log(8) + 0.4 * (diagnosis == 2) + 0.2 * severity, 0.9),  # ng/L
    'culture_ordered': bernoulli(lambda deteriorate, diagnosis: np.select(
        [diagnosis == 1, diagnosis == 0], [0.95, 0.6], 0.1 + 0.3 * deteriorate)),
    'log_creatinine': normal(lambda age, conditions: np.log(0.95) + 0.008 * (age - 66) + 0.05 * conditions, 0.25),  # mg/dL
    # Ward heart rate: 5 bpm per unit of severity, 6 bpm higher with an infection.
    'hr_base': normal(lambda diagnosis, severity: 80 + 5 * severity + 6 * np.isin(diagnosis, [0, 1]), 9),
    'spo2_base': normal(lambda copd, diagnosis, severity: 96.5 - 0.8 * severity - 2.5 * copd - 1.0 * (diagnosis == 0), 1.0),
}


def build_ward(rng):
    # Admission days weighted by season and weekday, hours by the ED pattern above.
    days = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    flu_d = flu_intensity(days)
    day_w = (1 + 0.25 * flu_d) * np.where(days.dayofweek >= 5, 0.85, 1.0)
    enc_parts = []
    for code, (site, n) in enumerate(WARD_ENCOUNTERS.items()):
        d = rng.choice(len(days), n, p=day_w / day_w.sum())
        h = rng.choice(24, n, p=ADMIT_HOURLY / ADMIT_HOURLY.sum())
        local = days[d] + pd.to_timedelta(h * 60 + rng.integers(0, 60, n), unit='min')
        admitted = local.tz_localize(WARD_SITES[site], ambiguous=True, nonexistent='shift_forward').tz_convert('UTC')
        enc_parts.append(pd.DataFrame({'site': site, 'site_code': code, 'admitted_at': admitted, 'flu': flu_d[d]}))
    enc = pd.concat(enc_parts, ignore_index=True).sort_values('admitted_at').reset_index(drop=True)
    n = len(enc)
    enc.insert(0, 'encounter_id', [f'E{i:05d}' for i in range(1, n + 1)])
    ids = patient_ids(rng, n)
    readmit = np.flatnonzero(rng.random(n) < 0.12)
    readmit = readmit[readmit > 50]
    ids[readmit] = [ids[rng.integers(max(0, i - 3000), i - 10)] for i in readmit]
    enc.insert(1, 'patient_id', ids)

    w = sample(causal_model(WARD_EDGES, WARD, given={'site', 'flu'}), SEED + 31,
               given=pd.DataFrame({'site': enc['site_code'], 'flu': enc['flu']}))
    age, conditions = w['age'].to_numpy(), w['conditions'].to_numpy()
    dx, severity = w['diagnosis'].to_numpy().astype(int), w['severity'].to_numpy()
    copd, deteriorate = w['copd'].to_numpy() == 1, w['deteriorate'].to_numpy() == 1
    site = enc['site'].to_numpy()
    enc['left_at'] = (enc['admitted_at'] + pd.to_timedelta((w['stay_h'].to_numpy() * 60).round(), unit='min')).dt.floor('min')
    enc['outcome'] = np.where(deteriorate, 'ICU transfer', 'discharged')
    enc['age'] = np.clip(age, 18, 99).round().astype(int)
    enc['diagnosis'] = np.array(DIAGNOSES)[dx]

    rounds_by_site = {}
    for s, zone in WARD_SITES.items():
        local_rounds = pd.date_range('2023-12-31', '2025-01-02', freq='4h')  # 00:00, 04:00, ... on the wall clock
        utc = local_rounds.tz_localize(zone, ambiguous=True, nonexistent='shift_forward').tz_convert('UTC')
        rounds_by_site[s] = (utc.as_unit('us').asi8, local_rounds.hour.to_numpy())

    hr_base, spo2_base = w['hr_base'].to_numpy(), w['spo2_base'].to_numpy()
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
        hr = (hr_base[i] - 2.5 * np.minimum((times - a) / (24 * HOUR_US), 4)  # settles as treatment works
              + rng.normal(0, 6, m) + rng.uniform(20, 35) * ramp)
        spo2 = spo2_base[i] + rng.normal(0, 1, m) - rng.uniform(3, 6) * ramp
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

    # Which labs each encounter had come from the graph; their collection times come from the stay.
    lactate_ordered, troponin_ordered = w['lactate_ordered'].to_numpy() == 1, w['troponin_ordered'].to_numpy() == 1
    culture_ordered = w['culture_ordered'].to_numpy() == 1
    lactate, troponin = np.exp(w['log_lactate'].to_numpy()), np.exp(w['log_troponin'].to_numpy())
    creatinine = np.exp(w['log_creatinine'].to_numpy())
    lab_rows = []
    for i, row in enumerate(enc.itertuples()):
        det, s_i = deteriorate[i], severity[i]
        a, b = row.admitted_at.value // 1000, row.left_at.value // 1000  # microseconds

        def add(test, t, value):
            lab_rows.append((row.encounter_id, row.site, test, t, value))
        first = a + int(rng.uniform(0.3, 2.0) * HOUR_US)
        if lactate_ordered[i]:
            add('lactate', first, lactate[i])
        if troponin_ordered[i]:
            add('troponin', first, troponin[i])
            add('troponin', first + 3 * HOUR_US, troponin[i] * np.exp(rng.normal(0.1, 0.3)))  # 3-hour repeat
        if culture_ordered[i]:
            add('blood culture', first + int(0.2 * HOUR_US), np.nan)
        t = first
        while t < b:
            add('creatinine', t, creatinine[i] * np.exp(0.08 * s_i + rng.normal(0, 0.08)))
            t += 24 * HOUR_US + int(rng.normal(0, 1.5) * HOUR_US)
        if det:  # repeat lactates as the patient deteriorates
            t = b - int(rng.uniform(4, 12) * HOUR_US)
            while t < b:
                rise = 1 - (b - t) / (12 * HOUR_US)  # 0 twelve hours before transfer, 1 at transfer
                add('lactate', t, lactate[i] * np.exp(rng.uniform(0.4, 1.0) * rise + rng.normal(0, 0.15)))
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
    esi_share = ed['esi'].value_counts(normalize=True)
    assert 0.003 <= esi_share[1] <= 0.015 and 0.38 <= esi_share[3] <= 0.50  # national triage mix

    stays, tele = t['stays'], t['telemetry']
    assert 600 <= len(stays) <= 900 and stays['stay_id'].is_unique
    assert 500_000 <= len(tele) <= 900_000
    assert (tele['recorded_at'].dt.minute % 5 == 0).all()
    assert tele['heart_rate'].dropna().between(35, 190).all() and tele['heart_rate'].isna().any()
    assert (stays['outcome'] == 'ICU transfer').sum() >= 25
    assert t['charted_temps'].duplicated(['stay_id', 'charted_at']).sum() > 0  # rechecked rounds

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
    # One numpy stream per builder and one gcm seed per graph, so a change to one extract leaves the others unchanged.
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
