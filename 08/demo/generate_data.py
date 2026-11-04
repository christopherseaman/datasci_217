"""Build the synthetic health-system tables for the Lecture 08 demos.

build() writes three Parquet files under data/:

- clinics.parquet: 9 clinics in the board's reporting order, including Oceanview, which opens in 2026.
- visits.parquet: 1,200,000 outpatient visits in 2025, one row per visit.
- labs.parquet: 2,000,000 lab results in 2025, one row per result.

The data come from one causal graph, declared in MODEL below with dowhy.gcm (a structural
causal model). Each node is a formula of its parents plus noise; the parents are the
formula's argument names. An arrow means "is an input to":

    home clinic -> deprivation (area index 1-10)
    age, deprivation -> smoking, activity, depression; age, deprivation, activity -> BMI
    age, BMI, smoking -> systolic BP -> hypertension
    age, BMI, deprivation -> diabetes -> baseline A1c -> treatment intensified
    age, diabetes, hypertension -> kidney function (eGFR); age, sex, statin -> LDL
    age, diabetes -> statin                      (sicker patients get the drug)
    conditions, age, frailty -> visit rate; conditions, age, BMI -> main department
    home clinic, date -> clinic of the visit; main department, conditions, clinic -> department
    first visit of the year, first visit at a clinic opened in 2025 -> visit type
    department, clinic, provider, visit type, weekday, season -> wait
    visit type, kiosk outage -> wait not recorded
    age, deprivation, wait -> survey returned; wait, provider -> satisfaction score

Confounding comes from these edges, not from patches. Parkside opened in July, so most of its
visits are first visits there and new-patient visits, and it runs Orthopedics, which Marina does
not; both lengthen its waits, while its own staffing is a little faster than Marina's.

gcm.draw_samples draws the 260,000 patients. Visits and lab results are rows of those patients,
so numpy draws which patient and which day, then each visit-level node is drawn from its dowhy
mechanism in graph order. Lab results apply measurement steps (assay error, fasting, the A1c
trend through the year) to each patient's sampled values with numpy.

Events tied to the calendar are inputs to the graph or small post-steps: Parkside opens in July
and takes Sunset's patients; Bayview loses an orthopedic surgeon in September; Excelsior's
check-in kiosk fails in October; an Excelsior analyzer writes 9999 for 23 failed glucose runs;
Ridgecrest runs a monthly cardiology outreach clinic for 10 regulars; and 0.25% of registrations
never record the clinic.

The seed is fixed, so every run writes the same values.
"""

import inspect
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

# dowhy 0.14 prints escape-sequence warnings the first time Python compiles it, and a progress-bar
# notice in notebooks without ipywidgets; neither affects the data.
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    warnings.filterwarnings('ignore', message='IProgress not found')
    from dowhy import gcm
    from dowhy.gcm.causal_models import PARENTS_DURING_FIT
    from dowhy.gcm.ml.regression import InvertibleExponentialFunction
    from dowhy.graph import get_ordered_predecessors

N_PATIENTS = 260_000
N_VISITS = 1_200_000
N_LABS = 2_000_000

CLINICS = pd.DataFrame({
    'clinic': ['Mission', 'Sunset', 'Richmond', 'Bayview', 'Excelsior', 'Marina', 'Parkside', 'Ridgecrest', 'Oceanview'],
    'region': ['Central', 'West', 'West', 'South', 'South', 'North', 'West', 'Rural', 'West'],
    'opened': pd.to_datetime(['2009-03-02', '2012-06-04', '2011-09-12', '2015-01-05', '2018-04-02',
                              '2016-02-01', '2025-07-07', '2019-10-07', '2026-01-12']),
})
MISSION, SUNSET, RICHMOND, BAYVIEW, EXCELSIOR, MARINA, PARKSIDE, RIDGECREST = range(8)
# Share of patients whose home clinic this is, in percent (Oceanview has none yet).
HOME_SHARE = np.array([22, 17, 14, 15, 12, 10, 7, 3]) / 100
# Mean area deprivation decile of each clinic's neighborhood (1 least, 10 most deprived).
CLINIC_DEPRIVATION = np.array([5.5, 4.0, 4.0, 7.5, 7.0, 2.5, 4.5, 6.5])
# Log-scale wait shift per clinic: Mission is short-staffed, Ridgecrest is small and quiet,
# and Parkside is new and fully staffed, about 10% faster than Marina.
CLINIC_WAIT_LOG = np.array([0.20, 0.04, 0.00, 0.06, 0.08, -0.05, -0.15, -0.30])
PARKSIDE_OPENS = pd.Timestamp('2025-07-07')
BAYVIEW_LOSS = pd.Timestamp('2025-09-01')

DEPTS = ['Family Medicine', 'Pediatrics', 'Behavioral Health', 'Cardiology', 'Orthopedics', 'Endocrinology', 'Dermatology']
FAMILY, PEDIATRICS, BEHAVIORAL, CARDIOLOGY, ORTHOPEDICS, ENDOCRINOLOGY, DERMATOLOGY = range(7)
# Median wait in minutes and log-scale spread; walk-in Family Medicine varies most,
# scheduled therapy in Behavioral Health least. Outpatient waits in US surveys run 15 to 30 minutes.
WAIT_MEDIAN = np.array([16, 12, 6, 20, 27, 18, 14])
WAIT_LOG_SD = np.array([0.55, 0.40, 0.35, 0.40, 0.42, 0.38, 0.40])
# Departments a clinic does not run; those visits go to Family Medicine instead.
NOT_OFFERED = {MARINA: [ORTHOPEDICS], PARKSIDE: [DERMATOLOGY, ENDOCRINOLOGY],
               RIDGECREST: [CARDIOLOGY, ORTHOPEDICS, ENDOCRINOLOGY, DERMATOLOGY]}
HOLIDAYS = pd.to_datetime(['2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', '2025-07-04',
                           '2025-09-01', '2025-11-27', '2025-11-28', '2025-12-25'])
DAYS = pd.date_range('2025-01-01', '2025-12-31', freq='D')


def logistic(x):
    return 1 / (1 + np.exp(-x))


def binned_normal(mean, sd, levels):
    """Probability of each integer level for round(clip(Normal(mean, sd))): one row per sample."""
    edges = np.concatenate([[-np.inf], levels[:-1] + 0.5, [np.inf]])
    return np.diff(stats.norm.cdf((edges[None, :] - mean[:, None]) / sd), axis=1)


class Formula(gcm.ml.PredictionModel):
    """A fixed formula; dowhy passes the parent columns in sorted name order, matched here by name."""

    def __init__(self, fn):
        self.fn = fn
        self.parents = sorted(inspect.signature(fn).parameters)

    def fit(self, X, Y):
        pass

    def predict(self, X):
        return np.asarray(self.fn(**dict(zip(self.parents, X.T.astype(float))))).reshape(-1, 1)

    def clone(self):
        return Formula(self.fn)


class Chance(Formula, gcm.ml.ClassificationModel):
    """Fixed class probabilities: fn returns P(class 1) for a yes/no node, or one column per class."""

    def __init__(self, fn, classes=(0, 1)):
        super().__init__(fn)
        self._classes = np.array(classes)

    @property
    def classes(self):
        return self._classes

    def predict_probabilities(self, X):
        p = np.asarray(self.fn(**dict(zip(self.parents, X.T.astype(float)))), dtype=float)
        return np.column_stack([1 - p, p]) if p.ndim == 1 else p

    def predict(self, X):
        return self._classes[self.predict_probabilities(X).argmax(axis=1)]

    def clone(self):
        return Chance(self.fn, self._classes)


def normal(sd):
    return gcm.ScipyDistribution(stats.norm, loc=0, scale=sd)


def additive(fn, sd):
    """node = fn(parents) + Normal(0, sd)."""
    return gcm.AdditiveNoiseModel(Formula(fn), normal(sd))


def lognormal(fn, sd):
    """node = exp(fn(parents) + Normal(0, sd)): a right-skewed positive value such as BMI or a wait."""
    return gcm.PostNonlinearModel(Formula(fn), normal(sd), InvertibleExponentialFunction())


def chance(fn, classes=(0, 1)):
    """A category drawn from fixed probabilities of the parents."""
    return gcm.ClassifierFCM(Chance(fn, classes))


def empirical(values, counts):
    """A root node that takes each value in proportion to its count."""
    model = gcm.EmpiricalDistribution()
    model.fit(np.repeat(values, counts))
    return model


def age_counts():
    """Ages 0 to 95: 19% children spread evenly, adults 18 + 77 x Beta(2, 1.8), in parts per million."""
    adult = np.diff(stats.beta.cdf((np.arange(18, 97) - 18) / 77, 2.0, 1.8))
    pmf = np.concatenate([np.full(18, 0.19 / 18), 0.81 * adult])
    return np.round(pmf * 1e6).astype(int)


def conditions(diabetic, hypertension, depression, egfr):
    return diabetic + hypertension + depression + (egfr < 60)


def department_weights(age, bmi, depression, diabetic, hypertension):
    """Relative chance of each department for a visit, steered by the patient's conditions."""
    weights = np.column_stack([
        np.full(len(age), 4.0),                                      # Family Medicine
        np.zeros(len(age)),                                          # Pediatrics
        0.8 + 3.0 * depression,                                      # Behavioral Health
        0.4 + 2.0 * hypertension + 0.02 * np.clip(age - 40, 0, None),  # Cardiology
        0.6 + 0.025 * np.clip(age - 30, 0, None) + 0.05 * np.clip(bmi - 30, 0, None),  # Orthopedics
        0.15 + 2.2 * diabetic,                                       # Endocrinology
        np.full(len(age), 0.9),                                      # Dermatology
    ])
    weights[age < 18] = [0, 9.3, 0.7, 0, 0, 0, 0]
    return weights / weights.sum(axis=1, keepdims=True)


def visit_department(age, bmi, depression, diabetic, hypertension, main_dept, clinic):
    """Half of visits return to the patient's main department (ongoing care); the clinic must run it."""
    p = 0.5 * department_weights(age, bmi, depression, diabetic, hypertension)
    p[np.arange(len(p)), main_dept.astype(int)] += 0.5
    for code, missing in NOT_OFFERED.items():
        rows = clinic == code
        p[np.ix_(rows, [FAMILY])] += p[np.ix_(rows, missing)].sum(axis=1, keepdims=True)
        p[np.ix_(rows, missing)] = 0
    return p


def visit_clinic(home, day):
    """88% of visits at the home clinic; Parkside's patients went to Sunset before it opened."""
    p = 0.12 * np.tile(HOME_SHARE, (len(home), 1))
    p[np.arange(len(home)), home.astype(int)] += 0.88
    before = day < (PARKSIDE_OPENS - DAYS[0]).days
    p[before, SUNSET] += p[before, PARKSIDE]
    p[before, PARKSIDE] = 0
    return p


MODEL = {
    # Patients: roots
    'home': empirical(np.arange(8), np.round(HOME_SHARE * 100).astype(int)),
    'age': empirical(np.arange(96), age_counts()),
    'male': gcm.ScipyDistribution(stats.bernoulli, p=0.47),
    # Frailty: how much more often than expected a patient comes in (gamma, mean 1, a long tail).
    'frailty': gcm.ScipyDistribution(stats.gamma, a=0.7, scale=1 / 0.7),
    # Unmeasured glycemic tendency (standard normal); it sets where a patient's A1c sits.
    'glycemia': normal(1.0),

    # Patients: health
    'deprivation': chance(lambda home: binned_normal(CLINIC_DEPRIVATION[home.astype(int)], 1.8, np.arange(1, 11)),
                          classes=np.arange(1, 11)),
    # Adult smoking runs about 8% in the least and 20% in the most deprived areas.
    'smoker': chance(lambda age, deprivation: (age >= 18) * (0.06 + 0.015 * deprivation)),
    # Weekly minutes of activity fall with age and deprivation.
    'activity': additive(lambda age, deprivation: 150 - 1.2 * (age - 45) - 6 * (deprivation - 5), 70),
    'bmi': lognormal(lambda age, deprivation, activity: np.log(np.where(
        age < 18, 19,
        27.5 + 0.35 * (deprivation - 5) - 0.012 * (np.clip(activity, 0, None) - 150) + 0.03 * (age - 45))), 0.17),
    # Each BMI point adds about 1 mmHg; each decade of age about 5 mmHg.
    'sbp': additive(lambda age, bmi, smoker: 118 + 0.5 * (age - 45) + 1.0 * (bmi - 27) + 2 * smoker, 13),
    'hypertension': chance(lambda age, sbp: (age >= 18) & (sbp >= 140)),
    # Diagnosed diabetes reaches about 12% of adults, rising steeply with age and BMI.
    'diabetic': chance(lambda age, bmi, deprivation:
                       (age >= 18) * logistic(-2.7 + 0.045 * (age - 50) + 0.11 * (bmi - 28) + 0.12 * (deprivation - 5))),
    # Diabetes: 6% plus a skewed excess (mean 1.6 points); younger and more deprived patients are
    # less well controlled. Without diabetes, about 5.3% with a 0.27-point spread.
    'a1c_base': additive(lambda age, bmi, deprivation, diabetic, glycemia: np.where(
        diabetic == 1,
        6.0 + np.exp(0.27 + 0.64 * glycemia) - 0.012 * (age - 55) + 0.09 * (deprivation - 5),
        5.25 + 0.006 * (age - 45) + 0.015 * (bmi - 27) + 0.27 * glycemia), 0.05),
    # Treatment is intensified for 90% of patients with diabetes at an A1c of 8% or more.
    'intensified': chance(lambda diabetic, a1c_base: 0.9 * ((diabetic == 1) & (a1c_base >= 8))),
    # eGFR falls about 1 mL/min per year of age; diabetes and hypertension lower it further.
    'egfr': additive(lambda age, diabetic, hypertension: 135 - 0.9 * age - 9 * diabetic - 6 * hypertension, 13),
    # Statins: most adults with diabetes over 40 and about a third of others over 60.
    'statin': chance(lambda age, diabetic: (age >= 18) * np.where((diabetic == 1) & (age >= 40), 0.70,
                                                                   np.where(age >= 60, 0.33, 0.04))),
    # Statins lower LDL about 35%; men and middle age run higher.
    'ldl': lognormal(lambda age, male, statin: np.log((112 + 8 * male + 0.3 * (np.clip(age, 18, 65) - 45))
                                                      * np.where(statin == 1, 0.65, 1.0)), 0.22),
    'depression': chance(lambda age, deprivation: np.where(age < 18, 0.05, 0.07 + 0.012 * deprivation)),
    # Visits a year: each condition multiplies the rate by about 1.4, then frailty (capped at 6)
    # gives a long tail, so a few patients come every week.
    'rate': lognormal(lambda age, diabetic, hypertension, depression, egfr, frailty: np.log(np.where(
        age < 18, 1.1, 0.8 * np.exp(0.35 * conditions(diabetic, hypertension, depression, egfr) + 0.008 * (age - 45)))
        * np.clip(frailty, 1e-6, 6)), 0.1),
    # The department of a patient's ongoing care (therapy, fracture follow-up).
    'main_dept': chance(department_weights, classes=np.arange(7)),

    # Visits: one row per visit; patient columns come along with the patient.
    'clinic': chance(visit_clinic, classes=np.arange(8)),
    'department': chance(visit_department, classes=np.arange(7)),
    # A patient's first visit of the year is often new to the clinic; later ones are mostly follow-ups.
    # A first visit at a clinic opened this year is almost always a new-patient visit.
    'new_patient': chance(lambda first_visit, first_at_new_clinic:
                          np.where(first_at_new_clinic == 1, 0.90, np.where(first_visit == 1, 0.45, 0.12))),
    # The visit's luck (standard normal), scaled by its department's spread of waits.
    'wait_luck': normal(1.0),
    'wait': lognormal(lambda department, clinic, provider_pace, new_patient, weekday, month, short_staffed, wait_luck:
                      np.log(WAIT_MEDIAN[department.astype(int)])
                      + CLINIC_WAIT_LOG[clinic.astype(int)]
                      + provider_pace
                      + 0.15 * new_patient
                      + 0.08 * (weekday == 0)                         # Mondays carry the weekend backlog
                      + 0.10 * np.isin(month, [1, 2]) + 0.05 * (month == 12)
                      + 0.62 * short_staffed                          # Bayview Orthopedics after September
                      + WAIT_LOG_SD[department.astype(int)] * wait_luck, 0.02),
    # Check-in kiosks miss some waits, more often for new patients filling in paper forms.
    'wait_lost': chance(lambda new_patient, kiosk_down: np.where(kiosk_down == 1, 0.6, 0.015 + 0.015 * new_patient)),
    # About one survey in three comes back: older patients return more, deprived areas and long waits fewer.
    'returned': chance(lambda age, deprivation, wait:
                       logistic(-0.75 + 0.018 * (age - 45) - 0.08 * (deprivation - 5) - 0.01 * (wait - 20))),
    'satisfaction': chance(lambda wait, provider_manner: binned_normal(4.55 - 0.03 * wait + provider_manner, 0.7,
                                                                       np.arange(1, 6)), classes=np.arange(1, 6)),
}
# Inputs that numpy builds from the calendar and the visit rows, with the nodes they come from.
INPUTS = {
    'day': [], 'weekday': ['day'], 'month': ['day'],
    'first_visit': ['day'], 'first_at_new_clinic': ['day', 'clinic'],
    'provider_pace': ['clinic', 'department'], 'provider_manner': ['clinic', 'department'],
    'short_staffed': ['day', 'clinic', 'department'], 'kiosk_down': ['day', 'clinic'],
}
PATIENT_NODES = list(MODEL)[:list(MODEL).index('clinic')]


def causal_model():
    """The structural causal model: every node in MODEL with an edge from each argument of its formula."""
    graph = nx.DiGraph()
    for node, parents in INPUTS.items():
        graph.add_edges_from((parent, node) for parent in parents)
    for node, mechanism in MODEL.items():
        graph.add_node(node)
        if isinstance(mechanism, gcm.ClassifierFCM):
            graph.add_edges_from((parent, node) for parent in mechanism.classifier_model.parents)
        elif not isinstance(mechanism, gcm.causal_mechanisms.StochasticModel):
            graph.add_edges_from((parent, node) for parent in mechanism.prediction_model.parents)
    scm = gcm.StructuralCausalModel(graph)
    for node, mechanism in MODEL.items():
        scm.set_causal_mechanism(node, mechanism)
        # The formulas are fixed, so there is nothing to fit; record the parents the way
        # gcm.create_causal_model_from_equations does for fully specified equations.
        scm.graph.nodes[node][PARENTS_DURING_FIT] = get_ordered_predecessors(scm.graph, node)
    return scm


def draw(scm, rows, nodes):
    """Add each node to rows from its mechanism and its parents' columns, in the given (graph) order."""
    for node in nodes:
        parents = get_ordered_predecessors(scm.graph, node)
        mechanism = scm.causal_mechanism(node)
        values = mechanism.draw_samples(rows[parents].to_numpy(float)) if parents else mechanism.draw_samples(len(rows))
        rows[node] = np.asarray(values).ravel()
    return rows


def day_weights(winter):
    """Probability of each 2025 day: weekdays, a light Saturday, closed Sundays and holidays."""
    weekday = DAYS.dayofweek.to_numpy()
    weight = 1 + winter * np.cos(2 * np.pi * (DAYS.dayofyear.to_numpy() - 15) / 365)
    weight = np.where(weekday == 5, 0.3 * weight, weight)
    weight = np.where((weekday == 6) | DAYS.isin(HOLIDAYS), 0, weight)
    return weight / weight.sum()


def ids(prefix, numbers, width):
    return prefix + pd.Series(numbers).astype(str).str.zfill(width)


def build_patients(scm):
    patient_scm = gcm.StructuralCausalModel(nx.DiGraph(scm.graph.subgraph(PATIENT_NODES)))
    return gcm.draw_samples(patient_scm, N_PATIENTS)


def build_visits(rng, scm, p):
    n = N_VISITS
    who = rng.choice(N_PATIENTS, n, p=(p['rate'] / p['rate'].sum()).to_numpy())
    # Winter respiratory season: about 25% more visits in January than in July.
    day = rng.choice(len(DAYS), n, p=day_weights(0.12))
    v = p.iloc[who].reset_index(drop=True)
    v['who'], v['day'] = who, day
    v = draw(scm, v, ['clinic', 'department'])

    # Ridgecrest's monthly cardiology outreach: 10 hypertensive regulars, one or two visits a month each.
    outreach = np.flatnonzero((v['clinic'] == RIDGECREST) & (v['department'] == FAMILY)
                              & (v['age'] >= 55) & (v['hypertension'] == 1))
    pool = rng.choice(np.unique(who[outreach]), 10, replace=False)
    first_tuesdays = [i for i, d in enumerate(DAYS) if d.dayofweek == 1 and d.day <= 7]
    rows = rng.choice(outreach, 10 * 12 + 60, replace=False)
    regulars = np.concatenate([np.repeat(pool, 12), rng.choice(pool, 60)])
    patient_cols = PATIENT_NODES + ['who']
    v.loc[rows, patient_cols] = p.iloc[regulars][PATIENT_NODES].assign(who=regulars).to_numpy()
    v.loc[rows, 'day'] = first_tuesdays * 10 + list(rng.choice(first_tuesdays, 60))
    v.loc[rows, 'department'] = CARDIOLOGY

    v = v.sort_values(['day', 'who'], kind='stable', ignore_index=True)
    date = DAYS[v['day'].to_numpy()]
    v['weekday'], v['month'] = date.dayofweek, date.month
    v['first_visit'] = ~v['who'].duplicated()
    v['first_at_new_clinic'] = (v['clinic'] == PARKSIDE) & ~v.duplicated(['who', 'clinic'])
    v['short_staffed'] = (v['clinic'] == BAYVIEW) & (v['department'] == ORTHOPEDICS) & (date >= BAYVIEW_LOSS)
    v['kiosk_down'] = (v['clinic'] == EXCELSIOR) & (v['month'] == 10)

    # Each provider has a typical pace and manner; about one provider per 2,500 visits in each clinic and department.
    cell = v['clinic'] * 7 + v['department']
    cell_code, _ = pd.factorize(cell)
    n_providers = np.maximum(1, np.round(np.bincount(cell_code) / 2500)).astype(int)
    offset = np.concatenate([[0], np.cumsum(n_providers)[:-1]])
    provider_id = offset[cell_code] + (rng.random(n) * n_providers[cell_code]).astype(int)
    v['provider_pace'] = rng.normal(0, 0.15, n_providers.sum())[provider_id]
    v['provider_manner'] = rng.normal(0, 0.25, n_providers.sum())[provider_id]
    v = draw(scm, v, ['new_patient', 'wait_luck', 'wait', 'wait_lost', 'returned', 'satisfaction'])

    clinic = CLINICS['clinic'].to_numpy()[v['clinic'].astype(int)].astype(object)
    # A few registrations never recorded the clinic.
    clinic[rng.random(n) < 0.0025] = None
    wait = np.clip(v['wait'], 0, 240).round()
    return pd.DataFrame({
        'visit_id': ids('V', np.arange(1, n + 1), 7),
        'visit_date': date.as_unit('us'),
        'visit_month': date.strftime('%Y-%m'),
        'patient_id': ids('P', v['who'].astype(int) + 1, 6),
        'clinic': pd.array(clinic, dtype='str'),
        'department': pd.array(np.array(DEPTS)[v['department'].astype(int)], dtype='str'),
        'visit_type': np.where(v['new_patient'] == 1, 'New', 'Follow-up'),
        'age': v['age'].astype(int),
        'wait_min': np.where(v['wait_lost'] == 1, np.nan, wait),
        'satisfaction': np.where(v['returned'] == 1, v['satisfaction'], np.nan),
    })


def a1c_schedule(rng, p):
    """A1c test dates: every 3 to 6 months for diabetes, one yearly screen for some others."""
    diabetic = p['diabetic'].to_numpy() == 1
    adult = p['age'].to_numpy() >= 18
    dm = np.flatnonzero(diabetic)
    # ADA: test quarterly when above goal, twice a year when at goal; disengaged patients stretch it.
    above_goal = p['a1c_base'].to_numpy()[dm] >= 7
    mean_gap = np.where(above_goal, 100, 165) * np.clip(1.25 / np.sqrt(np.clip(p['frailty'].to_numpy()[dm], 1e-6, 6)), 0.9, 2.5)
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
    open_days = np.flatnonzero(day_weights(0.05) > 0)
    index = np.searchsorted(open_days, np.floor(day).astype(int))
    return open_days[np.minimum(index, len(open_days) - 1)]


def build_labs(rng, p):
    age = p['age'].to_numpy()
    adult = np.flatnonzero(age >= 18)
    diabetic_all = p['diabetic'].to_numpy() == 1
    a1c_who, a1c_day = a1c_schedule(rng, p)

    # Basic metabolic panels (glucose and creatinine together) follow visit frequency;
    # most A1c draws add one. Lipid panels are yearly for adults over 40 and for diabetes.
    bmp_n = rng.poisson(3.15 * p['rate'].to_numpy()[adult])
    bmp_who = np.repeat(adult, bmp_n)
    bmp_day = draw_open_days(rng, len(bmp_who))
    with_a1c = rng.random(len(a1c_who)) < 0.85
    lipid_p = np.where(diabetic_all, 0.85, np.where(age >= 40, 0.55, 0.12))
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
    year_frac = (pd.DatetimeIndex(when).dayofyear.to_numpy() - 1) / 365

    diabetic, base = diabetic_all[who], p['a1c_base'].to_numpy()[who]
    # Intensified treatment lowers A1c about 0.8 points over the year; other patients with
    # diabetes drift up about 0.1 point a year.
    trend = np.where(p['intensified'].to_numpy()[who] == 1, -0.8, 0.1 * diabetic)
    a1c_now = base + trend * year_frac
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
    # Each LDL result varies about 12% around the patient's level (fasting, assay).
    result[ldl] = np.clip(p['ldl'].to_numpy()[who[ldl]] * rng.lognormal(0, 0.12, ldl.sum()), 25, 330).round()
    cre = test == 'Creatinine'
    # Creatinine from the patient's kidney function (simplified CKD-EPI inverse), 6% visit-to-visit noise.
    sex_base = np.where(p['male'].to_numpy()[who[cre]] == 1, 0.9, 0.7)
    egfr = np.clip(p['egfr'].to_numpy()[who[cre]], 15, 140)
    result[cre] = np.clip(sex_base * (90 / egfr) ** 1.1 * rng.lognormal(0, 0.06, cre.sum()), 0.3, 12).round(2)

    names = CLINICS['clinic'].to_numpy()
    clinic = names[p['home'].to_numpy()[who].astype(int)].astype(object)
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
    gcm.util.general.set_random_seed(217)  # dowhy draws its noise from numpy's global generator
    rng = np.random.default_rng(217)
    scm = causal_model()
    patients = build_patients(scm)
    tables = {'clinics': CLINICS, 'visits': build_visits(rng, scm, patients), 'labs': build_labs(rng, patients)}
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
