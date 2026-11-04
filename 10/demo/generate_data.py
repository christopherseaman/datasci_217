"""Build the synthetic health-system extracts for the Lecture 10 demos.

build() writes four Parquet tables under data/:

- pcp_visits: about 525,000 primary-care visits by 160,000 adults in 2024 and 2025 (Demo 1).
- rpm_patients and rpm_readings: 5,000 patients in a remote blood-pressure program and about
  580,000 home cuff readings from January to September 2025 (Demos 1 and 2).
- discharges: about 230,000 inpatient discharges from four hospitals, 2023 to mid-2025 (Demo 3).

Each table comes from a causal graph declared with dowhy.gcm (a structural causal model): one
dictionary per table maps each node to its mechanism, a formula of its parents plus noise, and
the formula's argument names are the node's parents. An arrow means "is an input to".

Primary care (PCP_MODEL):

    clinic -> deprivation (ADI decile) -> insurance
    age, sex, deprivation -> smoking, physical activity
    age, deprivation, activity -> BMI -> diabetes;  age, diabetes -> eGFR
    age (curved), sex x age, BMI, activity, smoking, deprivation, diabetes, eGFR -> untreated SBP
    untreated SBP, age, diabetes, insurance -> on_bp_med  (confounding by indication)
    on_bp_med -> drug response;  age, on_bp_med, diabetes -> visit rate
    untreated SBP - drug response + clinic offset + season + visit noise growing with age -> SBP
    clinic, BMI, age -> BMI missing;  deprivation -> activity missing

Remote monitoring (RPM_MODEL):

    clinic -> deprivation -> BMI, salt intake, adherence
    age, BMI -> diabetes, sleep apnea;  age, diabetes -> eGFR
    age, sex, BMI, apnea, NSAIDs, salt, eGFR, clinic -> untreated level -> drugs at enrollment
    adherence -> prior-year refill record (PDC);  BMI -> cuff size
    daily level, hour, cuff size on BMI, device, symptoms -> reading -> DBP;  drugs today -> pulse

Inpatient discharges (DISCHARGE_MODEL):

    hospital -> deprivation -> insurance, unstable housing;  age -> living alone
    latent frailty, age -> 8 comorbidities;  frailty, housing -> ED visit rate
    comorbidities, winter -> diagnosis group -> planned surgery, length of stay, labs
    age, length of stay, dementia -> disposition;  hospital, housing, insurance -> follow-up scheduled
    prior admissions, ED visits, comorbidities, diagnosis, sodium (U-shaped), anemia, kidney injury,
        age under 35, social factors, hospital, and interactions (heart failure without follow-up,
        living alone at 75+ sent home, 10+ medications at 75+, unstable housing with frequent
        ED visits) -> 30-day readmission -> follow-up kept

gcm.draw_samples draws the patients of each table. Rows below the patient (visits, readings,
admissions) are drawn node by node from the same mechanisms, in graph order, on the patient
columns. Three parts depend on time and run in numpy: the visit count and dates per patient;
the remote program's day-by-day loop (refill lapses, drug effects that fade and return, the
pharmacist adding a drug at 4-weekly reviews, slow drift and season); and the admission loop
(each readmission sets the next admission date, and prior admissions count earlier rows).
Extract faults (typos, swapped fields, duplicate feeds, cuff error codes, upload delays) are
numpy post-steps. The seed is fixed, so every run writes the same values. All patients and
values are synthetic.
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

SEED = 217


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- mechanisms

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


def yes(p):
    """A yes/no root node, 1 with probability p."""
    return gcm.ScipyDistribution(stats.bernoulli, p=p)


def additive(fn, noise):
    """node = fn(parents) + noise; a number is the SD of normal noise."""
    return gcm.AdditiveNoiseModel(Formula(fn), normal(noise) if np.isscalar(noise) else noise)


def lognormal(fn, sd):
    """node = exp(fn(parents) + Normal(0, sd)): a right-skewed positive value such as BMI."""
    return gcm.PostNonlinearModel(Formula(fn), normal(sd), InvertibleExponentialFunction())


def gamma_scaled(fn, shape):
    """node = exp(fn(parents)) x Gamma(shape, 1): a rate whose spread across patients is gamma."""
    return gcm.PostNonlinearModel(Formula(fn), gcm.ScipyDistribution(stats.loggamma, c=shape, loc=0, scale=1),
                                  InvertibleExponentialFunction())


def chance(fn, classes=(0, 1)):
    """A category drawn from fixed probabilities of the parents."""
    return gcm.ClassifierFCM(Chance(fn, classes))


def empirical(values, counts):
    """A root node that takes each value in proportion to its count."""
    model = gcm.EmpiricalDistribution()
    model.fit(np.repeat(values, counts))
    return model


def binned_normal(mean, sd, levels):
    """Probability of each integer level for round(clip(Normal(mean, sd))): one row per sample."""
    edges = np.concatenate([[-np.inf], levels[:-1] + 0.5, [np.inf]])
    return np.diff(stats.norm.cdf((edges[None, :] - np.asarray(mean, float)[:, None]) / sd), axis=1)


def poisson_probs(rate, top):
    """Probability of 0, 1, ..., top for a Poisson count, with everything above top counted as top."""
    p = stats.poisson.pmf(np.arange(top + 1)[None, :], np.asarray(rate, float)[:, None])
    p[:, -1] += 1 - p.sum(axis=1)
    return p


def beta_ages(low, span, a, b):
    """Counts per integer age for round(low + span x Beta(a, b)), in parts per million."""
    ages = np.arange(low, low + span + 1)
    edges = np.clip((np.r_[ages - 0.5, ages[-1] + 0.5] - low) / span, 0, 1)
    return ages, np.round(np.diff(stats.beta.cdf(edges, a, b)) * 1e6).astype(int)


def causal_model(model, inputs):
    """The structural causal model: every node in model with an edge from each argument of its formula.

    inputs maps the nodes numpy computes (dates, the daily loop, counts of earlier rows) to their parents.
    """
    graph = nx.DiGraph()
    for node, parents in inputs.items():
        graph.add_node(node)
        graph.add_edges_from((parent, node) for parent in parents)
    for node, mechanism in model.items():
        graph.add_node(node)
        if isinstance(mechanism, gcm.ClassifierFCM):
            graph.add_edges_from((parent, node) for parent in mechanism.classifier_model.parents)
        elif not isinstance(mechanism, gcm.causal_mechanisms.StochasticModel):
            graph.add_edges_from((parent, node) for parent in mechanism.prediction_model.parents)
    scm = gcm.StructuralCausalModel(graph)
    for node, mechanism in model.items():
        scm.set_causal_mechanism(node, mechanism)
        # The formulas are fixed, so there is nothing to fit; record the parents the way
        # gcm.create_causal_model_from_equations does for fully specified equations.
        scm.graph.nodes[node][PARENTS_DURING_FIT] = get_ordered_predecessors(scm.graph, node)
    return scm


def draw_roots_down(scm, nodes, n):
    """gcm.draw_samples on the subgraph of the given nodes (the patient level).

    The subgraph is built in the declared node order: networkx's subgraph view iterates a set of
    names, whose order changes between Python runs, and the order of draws sets the values.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from((node, scm.graph.nodes[node]) for node in nodes)
    graph.add_edges_from((parent, node) for node in nodes for parent in get_ordered_predecessors(scm.graph, node))
    return gcm.draw_samples(gcm.StructuralCausalModel(graph), n)


def draw(scm, rows, nodes):
    """Add each node to rows from its mechanism and its parents' columns, in the given (graph) order."""
    for node in nodes:
        parents = get_ordered_predecessors(scm.graph, node)
        mechanism = scm.causal_mechanism(node)
        values = mechanism.draw_samples(rows[parents].to_numpy(float)) if parents else mechanism.draw_samples(len(rows))
        rows[node] = np.asarray(values).ravel()
    return rows


def ids(prefix, numbers, width):
    return [f'{prefix}{i:0{width}d}' for i in numbers]


# ---------------------------------------------------------------- primary-care visits

PCP_CLINICS = ['Mission', 'Sunset', 'Richmond', 'Bayview', 'Excelsior', 'Marina']
PCP_PATIENTS = 160_000
PCP_SHARE = [42, 36, 30, 24, 18, 10]                     # thousands of patients per clinic
PCP_ADI = np.array([6.0, 4.0, 3.5, 8.0, 7.0, 2.0])       # mean ADI decile of the catchment
PCP_OFFSET = np.array([1.0, -0.5, 0.0, 2.0, -1.0, -1.5])  # BP offset from protocol and staff, mmHg
PCP_MANUAL = np.array([0, 0, 0, 1, 1, 0])                # Bayview and Excelsior use manual cuffs
# Marina's intake skips the scale more often, so its BMI is missing more often.
PCP_BMI_MISSING = np.array([0.05, 0.05, 0.06, 0.07, 0.06, 0.20])
PCP_INSURANCE = np.array(['commercial', 'medicaid', 'medicare', 'uninsured'])
UNINSURED = 3


def pcp_insurance(age, adi):
    """Medicare from 65; below that, Medicaid rises with deprivation and a few are uninsured."""
    medicaid = sigmoid(-2.0 + 0.45 * (adi - 5))
    uninsured = (1 - medicaid) * np.clip(0.05 + 0.01 * (adi - 5), 0, 1)
    p = np.column_stack([1 - medicaid - uninsured, medicaid, np.zeros(len(age)), uninsured])
    p[age >= 65] = [0, 0, 1, 0]
    return p


PCP_MODEL = {
    # Patients
    'clinic': empirical(np.arange(6), PCP_SHARE),
    'male': yes(0.45),
    'age': empirical(*beta_ages(18, 72, 2.0, 2.2)),
    'adi': chance(lambda clinic: binned_normal(PCP_ADI[clinic.astype(int)], 1.8, np.arange(1, 11)),
                  classes=np.arange(1, 11)),
    'insurance': chance(pcp_insurance, classes=np.arange(4)),
    # Smoking about 12% overall: more in deprived areas and men, less with age.
    'smoker': chance(lambda age, adi, male: sigmoid(-2.1 + 0.2 * (adi - 5) + 0.35 * male - 0.015 * (age - 45))),
    'active': chance(lambda age, adi: sigmoid(1.0 - 0.03 * (age - 45) - 0.15 * (adi - 5))),
    # Minutes a week: active patients around 150, inactive around 12.
    'activity': lognormal(lambda active: np.log(np.where(active == 1, 150, 12)), 0.6),
    'bmi': lognormal(lambda age, adi, activity: np.log(np.clip(
        28.5 + 0.4 * (adi - 5) - 0.006 * np.minimum(activity, 400) - 0.002 * (age - 55) ** 2, 16, None)), 0.18),
    'diabetes': chance(lambda age, adi, bmi: sigmoid(-2.7 + 0.17 * (bmi - 28) + 0.045 * (age - 50) + 0.15 * (adi - 5))),
    'egfr': additive(lambda age, diabetes: 105 - 0.9 * np.clip(age - 40, 0, None) - 8 * diabetes, 13),
    # Untreated SBP: about 1 mmHg per kg/m2 of BMI, curved upward with age; men run higher until
    # their 60s; 300 min/week of activity lowers it about 4 mmHg (exercise trials: 4 to 8).
    'untreated_sbp': additive(lambda age, bmi, male, activity, smoker, adi, diabetes, egfr:
                              116 + 0.30 * (age - 50) + 0.007 * (age - 50) ** 2 + 0.9 * (bmi - 27)
                              + male * np.clip(7 - 0.2 * (age - 45), -2, 7) - 0.012 * np.minimum(activity, 300)
                              + 1.5 * smoker + 0.6 * (adi - 5) + 3 * diabetes + 0.1 * np.clip(60 - egfr, 0, None), 9),
    # Confounding by indication: high untreated SBP gets medication; uninsured patients less often.
    'on_bp_med': chance(lambda untreated_sbp, age, diabetes, insurance, adi:
                        sigmoid((untreated_sbp - 130) / 6 + 0.03 * (age - 60) + 0.5 * diabetes
                                - 0.8 * (insurance == UNINSURED) - 0.1 * (adi - 5))),
    # Medication lowers SBP about 10 mmHg (single-drug trials: 8 to 10), from 5 to 19 in most patients.
    'drug_response': lognormal(lambda on_bp_med: np.log(np.where(on_bp_med == 1, 9.5, 1e-3)), 0.35),
    # Visits a year: about 2.8, more for treated, diabetic, and older patients; gamma spread gives
    # the long tail of frequent attenders.
    'visit_rate': gamma_scaled(lambda age, on_bp_med, diabetes:
                               np.log(np.clip(2.8 * np.exp(0.25 * on_bp_med + 0.3 * diabetes + 0.008 * (age - 50)) - 1,
                                              0.05, None) / 2), 2.0),

    # Visits: one row per visit; patient columns come along with the patient.
    'visit_luck': normal(1.0),
    # Recorded SBP: winter about 4 mmHg above summer, manual cuffs about 3 high, and visit-to-visit
    # spread growing with age (7 mmHg at 18, 17 at 85); 1 mmHg of cuff error on top.
    'sbp': additive(lambda untreated_sbp, drug_response, clinic, day, visit_age, visit_luck:
                    untreated_sbp - drug_response + PCP_OFFSET[clinic.astype(int)] + 3 * PCP_MANUAL[clinic.astype(int)]
                    + 2 * np.cos(2 * np.pi * (day - 15) / 365.25) + (7 + 0.15 * (visit_age - 18)) * visit_luck, 1),
    'dbp': additive(lambda sbp, visit_age: 30 + 0.40 * sbp - 0.12 * np.clip(visit_age - 50, 0, None), 6),
    # Manual cuffs: terminal-digit preference, about a third of readings rounded to the nearest 10.
    'rounded': chance(lambda clinic: 0.35 * PCP_MANUAL[clinic.astype(int)]),
    'bmi_measured': additive(lambda bmi: bmi, 0.6),
    'activity_reported': lognormal(lambda activity: np.log(activity) + 0.15, 0.4),   # self-report runs high
    # Missing BMI depends on the clinic and on the BMI itself (scale limits, declined weighing).
    'bmi_missing': chance(lambda clinic, bmi, visit_age:
                          PCP_BMI_MISSING[clinic.astype(int)] + 0.08 * (bmi > 40) + 0.05 * (visit_age > 80)),
    'activity_missing': chance(lambda adi: 0.12 + 0.02 * (adi - 5)),
}
PCP_INPUTS = {'day': [], 'visit_age': ['age', 'day']}
PCP_PATIENT_NODES = list(PCP_MODEL)[:list(PCP_MODEL).index('visit_luck')]


def build_pcp_visits(rng):
    scm = causal_model(PCP_MODEL, PCP_INPUTS)
    p = draw_roots_down(scm, PCP_PATIENT_NODES, PCP_PATIENTS)
    visits_per = 1 + rng.poisson(p['visit_rate'].to_numpy())
    idx = np.repeat(np.arange(PCP_PATIENTS), visits_per)
    v = p.iloc[idx].reset_index(drop=True)
    v['day'] = rng.integers(0, 731, len(v))
    v['visit_age'] = v['age'] + (v['day'] >= 365)
    v = draw(scm, v, list(PCP_MODEL)[len(PCP_PATIENT_NODES):])
    m = len(v)

    clinic = np.array(PCP_CLINICS)[v['clinic'].astype(int)]
    sbp = np.where(v['rounded'] == 1, np.round(v['sbp'], -1), np.round(v['sbp']))
    table = pd.DataFrame({
        'patient_id': ids('PC', idx + 1, 6),
        'clinic': clinic,
        'visit_date': pd.Timestamp('2024-01-01') + pd.to_timedelta(v['day'], unit='D'),
        'sex': np.where(v['male'] == 1, 'M', 'F'),
        'age': v['visit_age'].astype(int),
        'insurance': PCP_INSURANCE[v['insurance'].astype(int)],
        'adi_decile': v['adi'].astype(int),
        'bmi': np.clip(v['bmi_measured'], 15, 65).round(1),
        'activity_min_wk': np.round(v['activity_reported'], -1),
        'smoker': v['smoker'] == 1,
        'diabetes': v['diabetes'] == 1,
        'sbp': sbp,
        'dbp': v['dbp'].round(),
        'on_bp_med': v['on_bp_med'] == 1,
    })
    table.loc[v['bmi_missing'] == 1, 'bmi'] = np.nan
    table.loc[v['activity_missing'] == 1, 'activity_min_wk'] = np.nan
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
    table.insert(0, 'visit_id', ids('V', range(1, len(table) + 1), 7))
    return table.reset_index(drop=True)


# ---------------------------------------------------------------- remote BP monitoring

RPM_START, RPM_DAYS = pd.Timestamp('2025-01-01'), 273   # through 2025-09-30
RPM_PATIENTS = 5000
RPM_CLINICS = ['Mission', 'Sunset', 'Richmond', 'Bayview', 'Marina']
RPM_SHARE = [34, 26, 20, 13, 7]                          # percent of patients
RPM_ADI = np.array([6.0, 4.0, 3.5, 8.0, 2.0])
RPM_OFFSET = np.array([1.0, -1.0, 0.0, 2.5, -1.5])        # level offset, mmHg
INTAKE_ITEMS = 30
ROUTINE_START = np.array([6.0, 18.0, 23.0])               # morning, evening, night measurers
ROUTINE_SPAN = np.array([3.5, 4.0, 2.5])


def intake_item(j):
    """1-to-5 answers on the intake questionnaire (sleep, stress, diet, mood): none moves BP;
    every seventh item tracks deprivation."""
    levels = np.arange(1, 6)
    if j % 7 == 0:
        return chance(lambda adi: binned_normal(3 + 0.2 * (adi - 5), 1, levels), classes=levels)
    return empirical(levels, np.round(binned_normal([3.0], 1, levels)[0] * 1e6).astype(int))


RPM_MODEL = {
    # Patients
    'clinic': empirical(np.arange(5), RPM_SHARE),
    'male': yes(0.47),
    'age': empirical(*beta_ages(30, 55, 2.6, 2.0)),
    'adi': chance(lambda clinic: binned_normal(RPM_ADI[clinic.astype(int)], 1.8, np.arange(1, 11)),
                  classes=np.arange(1, 11)),
    'bmi': lognormal(lambda adi: np.log(30 + 0.4 * (adi - 5)), 0.17),
    'diabetes': chance(lambda age, bmi: sigmoid(-1.6 + 0.12 * (bmi - 30) + 0.03 * (age - 60))),
    'egfr': additive(lambda age, diabetes: 105 - 0.8 * (age - 40) - 10 * diabetes, 14),
    'sleep_apnea': chance(lambda bmi, male: sigmoid(-2.2 + 0.15 * (bmi - 30) + 0.6 * male)),
    'nsaid_use': yes(0.12),
    'salt_score': chance(lambda adi: binned_normal(5 + 0.3 * (adi - 5), 2, np.arange(11)), classes=np.arange(11)),
    # Untreated level: patients enroll because their BP is uncontrolled; untreated apnea adds about
    # 4 mmHg, NSAIDs about 3, and low eGFR adds resistant hypertension.
    'untreated': additive(lambda age, bmi, male, sleep_apnea, nsaid_use, salt_score, egfr, clinic:
                          152 + 0.35 * (age - 60) + 0.5 * (bmi - 30) + 2 * male + 4 * sleep_apnea + 3 * nsaid_use
                          + 0.8 * (salt_score - 5) + 0.12 * np.clip(60 - egfr, 0, None)
                          + RPM_OFFSET[clinic.astype(int)], 10),
    # Drugs at enrollment rise with the untreated level (indication), 0 to 4.
    'meds0': chance(lambda untreated: poisson_probs(np.clip((untreated - 130) / 12, 0.1, None), 4), classes=np.arange(5)),
    # Share of fills on time: most patients 0.75 (Beta(6, 2)), lower in deprived areas and under 60.
    'adherence': additive(lambda adi, age: -0.03 * (adi - 5) - 0.003 * np.clip(60 - age, 0, None),
                          gcm.ScipyDistribution(stats.beta, a=6, b=2)),
    # Proportion of days covered by fills in the year before enrollment, from claims.
    'pdc': additive(lambda adherence: np.clip(adherence, 0.2, 0.99), 0.08),
    'large_cuff': chance(lambda bmi: 0.7 * (bmi > 33)),
    'cuff_v2': yes(0.4),
    'anxious': yes(0.3),
    'routine': empirical(np.arange(3), [68, 24, 8]),
    'clock_draw': gcm.ScipyDistribution(stats.uniform, loc=0, scale=1),
    'usual_hour': additive(lambda routine, clock_draw: ROUTINE_START[routine.astype(int)]
                           + ROUTINE_SPAN[routine.astype(int)] * clock_draw, 0.05),
    'meas_rate': gcm.ScipyDistribution(stats.beta, a=4, b=2.5),
    'enroll_day': empirical(np.arange(90), np.ones(90, dtype=int)),
    **{f'intake_q{j:02d}': intake_item(j) for j in range(1, INTAKE_ITEMS + 1)},

    # Readings: one row per reading.
    'surge_luck': normal(1.0),
    # Reading: morning surge and evening dip of 5 mmHg, a standard cuff on a large arm about 6 high,
    # the v2 device 2.5 high, symptom readings about 16 high (SD 6), and 5.5 mmHg of reading noise.
    'sbp': additive(lambda level, hour, large_cuff, bmi, cuff_v2, symptom, surge_luck:
                    level + 5 * np.cos(2 * np.pi * (hour - 8) / 24) + 6 * ((large_cuff == 0) & (bmi > 33))
                    + 2.5 * cuff_v2 + symptom * (16 + 6 * surge_luck), 5.5),
    'dbp': additive(lambda sbp: 0.45 * sbp + 20, 5),
    'pulse': additive(lambda meds_today: 72 - 1.5 * meds_today, 9),
}
# level and meds_today come from the daily loop; symptom and hour from the reading schedule.
RPM_INPUTS = {'level': ['untreated', 'meds0', 'adherence', 'enroll_day'], 'meds_today': ['meds0', 'level'],
              'symptom': ['anxious', 'enroll_day'], 'hour': ['usual_hour', 'symptom']}
RPM_PATIENT_NODES = list(RPM_MODEL)[:list(RPM_MODEL).index('surge_luck')]


def build_rpm(rng):
    scm = causal_model(RPM_MODEL, RPM_INPUTS)
    p = draw_roots_down(scm, RPM_PATIENT_NODES, RPM_PATIENTS)
    n = RPM_PATIENTS
    untreated = p['untreated'].to_numpy()
    meds0 = p['meds0'].to_numpy().astype(int)
    adherence = np.clip(p['adherence'].to_numpy(), 0.2, 0.99)
    enroll = p['enroll_day'].to_numpy().astype(int)
    anxious = p['anxious'].to_numpy() == 1
    meas_rate = p['meas_rate'].to_numpy()
    usual_hour = p['usual_hour'].to_numpy()

    # Daily loop: each day's level follows the drugs actually taken.
    step = np.clip(rng.normal(6, 2, (n, 8)), 2, 12)     # mmHg per added drug or dose step
    leave = np.where(rng.random(n) < 0.18, enroll + rng.integers(30, 200, n), RPM_DAYS)
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

    # Reading schedule: routine days fade over the program; symptom days (a headache or dizziness)
    # prompt an extra reading off the usual routine, more often in anxious patients.
    enrolled = (days >= enroll[:, None]) & (days < leave[:, None])
    fade = 1 - 0.35 * np.clip(days - enroll[:, None], 0, None) / RPM_DAYS
    pid, day = np.nonzero(enrolled & (rng.random((n, RPM_DAYS)) < meas_rate[:, None] * fade))
    spid, sday = np.nonzero(enrolled & (rng.random((n, RPM_DAYS)) < 0.0125 + 0.05 * anxious[:, None]))
    symptom = np.r_[np.zeros(len(pid), bool), np.ones(len(spid), bool)]
    pid, day = np.r_[pid, spid], np.r_[day, sday]
    k = len(pid)
    odd = rng.random(k) < 0.06
    hour = np.where(symptom, usual_hour[pid] + rng.uniform(3, 21, k),
                    np.where(odd, rng.uniform(0, 24, k), usual_hour[pid] + rng.normal(0, 0.7, k))) % 24
    r = p.iloc[pid].reset_index(drop=True)
    r['level'], r['meds_today'], r['symptom'], r['hour'] = rec_level[pid, day], rec_meds[pid, day], symptom, hour
    r = draw(scm, r, ['surge_luck', 'sbp', 'dbp', 'pulse'])

    reading_time = (RPM_START + pd.to_timedelta(day, unit='D')
                    + pd.to_timedelta(np.round(hour * 60), unit='min'))
    # most cuffs sync within minutes; some wait hours or days for the phone
    lag_min = np.select([rng.random(k) < 0.82, rng.random(k) < 0.78],
                        [rng.exponential(3, k), rng.uniform(30, 720, k)], rng.uniform(1440, 8640, k))
    change_day = rec_change[pid, day]
    readings = pd.DataFrame({
        'patient_id': ids('RP', pid + 1, 5),
        'reading_time': reading_time,
        'upload_time': reading_time + pd.to_timedelta(np.round(lag_min), unit='min'),
        'reason': np.where(symptom, 'symptoms', 'scheduled'),
        'sbp': r['sbp'].round().astype(int),
        'dbp': r['dbp'].round().astype(int),
        'pulse': r['pulse'].round().astype(int),
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
    readings.insert(0, 'reading_id', ids('R', range(1, len(readings) + 1), 7))

    bmi = np.clip(p['bmi'].to_numpy(), 18, 60)
    patients = pd.DataFrame({
        'patient_id': ids('RP', range(1, n + 1), 5),
        'enroll_date': RPM_START + pd.to_timedelta(enroll, unit='D'),
        'clinic': np.array(RPM_CLINICS)[p['clinic'].astype(int)],
        'sex': np.where(p['male'] == 1, 'M', 'F'),
        'age': p['age'].astype(int).to_numpy(),
        'bmi': np.where(rng.random(n) < 0.07, np.nan, bmi.round(1)),
        'adi_decile': p['adi'].astype(int).to_numpy(),
        'diabetes': p['diabetes'].to_numpy() == 1,
        'egfr': np.clip(p['egfr'].to_numpy(), 8, 130).round(),
        'sleep_apnea': p['sleep_apnea'].to_numpy() == 1,
        'nsaid_use': p['nsaid_use'].to_numpy() == 1,
        'salt_score': p['salt_score'].astype(int).to_numpy(),
        'pdc_prior_year': np.clip(p['pdc'].to_numpy(), 0.1, 1.0).round(2),
        'cuff_size': np.where(p['large_cuff'] == 1, 'large', 'standard'),
        'device': np.where(p['cuff_v2'] == 1, 'cuff_v2', 'cuff_v1'),
    })
    for j in range(1, INTAKE_ITEMS + 1):
        patients[f'intake_q{j:02d}'] = p[f'intake_q{j:02d}'].astype(int).to_numpy()
    return patients, readings


# ---------------------------------------------------------------- inpatient discharges

DISCHARGE_PATIENTS = 150_000
HOSPITALS = ['General', 'Bayside', 'Northgate', 'Valley']
HOSPITAL_SHARE = [42, 27, 19, 12]
HOSPITAL_ADI = np.array([6.0, 4.0, 3.0, 7.0])
HOSPITAL_RISK = np.array([0.05, -0.12, -0.05, 0.15])      # readmission offset (log-odds)
HOSPITAL_FOLLOWUP = np.array([0.62, 0.75, 0.70, 0.48])    # share of follow-ups scheduled
DISCHARGE_INSURANCE = np.array(['commercial', 'medicaid', 'medicare'])
MEDICAID = 1
DX = np.array(['heart_failure', 'copd', 'pneumonia', 'sepsis', 'ami', 'gi_bleed', 'orthopedic', 'other'])
HF, COPD, PNEUMONIA, SEPSIS, AMI, GI_BLEED, ORTHOPEDIC, OTHER = range(8)
DX_LOS = np.array([4.5, 3.8, 4.2, 6.0, 3.5, 3.6, 2.8, 3.2])            # median stay, days
DX_RISK = np.array([0.45, 0.35, 0.1, 0.25, 0.15, 0.2, -0.7, 0.0])     # readmission log-odds
COMORBIDITIES = {  # base logit, per log-frailty, per decade over 60
    'chf': (-1.8, 0.9, 0.35), 'copd': (-1.6, 0.7, 0.15), 'dm': (-0.6, 0.5, 0.05),
    'ckd': (-1.5, 0.8, 0.35), 'cancer': (-1.8, 0.5, 0.2), 'dementia': (-3.0, 0.6, 0.8),
    'liver': (-2.6, 0.6, -0.1), 'depression': (-1.1, 0.4, -0.1),
}
EXTRACT_END = pd.Timestamp('2025-06-30')
WINDOW_DAYS = (EXTRACT_END - pd.Timestamp('2023-01-01')).days


def comorbidity(base, per_log_frailty, per_decade):
    return chance(lambda frailty, age0: sigmoid(base + per_log_frailty * np.log(frailty)
                                                + per_decade * (age0 - 60) / 10))


def discharge_insurance(age0, adi):
    medicaid = sigmoid(-1.5 + 0.5 * (adi - 5))
    return np.where((age0 >= 65)[:, None], [0, 0, 1], np.column_stack([1 - medicaid, medicaid, 0 * medicaid]))


def diagnosis(chf, copd, liver, winter):
    p = np.column_stack([0.03 + 0.25 * chf, 0.03 + 0.2 * copd + 0.05 * winter, 0.10 + 0.06 * winter,
                         np.full(len(chf), 0.08), np.full(len(chf), 0.06), 0.05 + 0.05 * liver,
                         np.full(len(chf), 0.13), np.full(len(chf), 0.3)])
    return p / p.sum(axis=1, keepdims=True)


def readmission(frailty, dx, hospital, prior_admits, ed_visits_6m, n_comorb, sodium, hemoglobin, creatinine,
                los_days, n_meds, cancer, housing_unstable, insurance, age, followup_scheduled, ama, snf,
                home_health, elective, lives_alone, died):
    """Effects on the log-odds scale; prior utilization and comorbidity dominate, as in LACE and
    HOSPITAL scores (odds ratios about 1.3 to 2 per factor)."""
    hf = dx == HF
    followup = followup_scheduled == 1
    home = (snf == 0) & (ama == 0)
    logit = (-3.5 + 0.35 * np.log(frailty) + DX_RISK[dx.astype(int)] + HOSPITAL_RISK[hospital.astype(int)]
             + 0.35 * np.log1p(prior_admits) + 0.10 * np.minimum(ed_visits_6m, 6) + 0.10 * n_comorb
             + 0.6 * (sodium < 133) + 0.35 * (sodium > 145) + 0.5 * (hemoglobin < 10)
             + 0.45 * (creatinine > 2) + 0.25 * np.log1p(np.floor(los_days)) + 0.02 * n_meds + 0.3 * cancer
             + 0.3 * housing_unstable + 0.15 * (insurance == MEDICAID) + 0.35 * (age < 35)
             - 0.2 * followup + 0.8 * ama + 0.2 * snf - 0.5 * elective
             # interactions, odds ratios about 1.8 to 2.7
             + 1.0 * (hf & ~followup)
             + 0.8 * ((lives_alone == 1) & (age >= 75) & home)
             + 0.6 * ((n_meds >= 10) & (age >= 75))
             + 0.8 * ((housing_unstable == 1) & (ed_visits_6m >= 3)))
    return (1 - died) * sigmoid(logit)


DISCHARGE_MODEL = {
    # Patients
    'hospital': empirical(np.arange(4), HOSPITAL_SHARE),
    # Latent frailty (gamma, mean 1.3): more comorbidities, admissions, ED visits, and longer stays.
    'frailty': gcm.ScipyDistribution(stats.gamma, a=1.3, scale=1.0),
    'age0': gcm.ScipyDistribution(stats.beta, a=3.0, b=1.9, loc=18, scale=82),
    'male': yes(0.48),
    'adi': chance(lambda hospital: binned_normal(HOSPITAL_ADI[hospital.astype(int)], 2, np.arange(1, 11)),
                  classes=np.arange(1, 11)),
    'insurance': chance(discharge_insurance, classes=np.arange(3)),
    'lives_alone': chance(lambda age0: sigmoid(-1.6 + 0.03 * (age0 - 60))),
    'housing_unstable': chance(lambda adi, age0: sigmoid(-4.0 + 0.45 * (adi - 5) - 0.02 * (age0 - 50))),
    **{name: comorbidity(*terms) for name, terms in COMORBIDITIES.items()},
    # ED visits per 6 months: gamma across patients (shape 0.8), higher with frailty and unstable housing.
    'ed_rate': gamma_scaled(lambda frailty, housing_unstable:
                            np.log(0.6 * (0.5 + frailty) * (1 + 0.5 * housing_unstable)), 0.8),

    # Admissions: one row per admission.
    'dx': chance(diagnosis, classes=np.arange(8)),
    'elective': chance(lambda dx: 0.75 * (dx == ORTHOPEDIC)),
    'transfer': chance(lambda elective: 0.08 * (1 - elective)),
    # Length of stay: lognormal by diagnosis, about 8% longer per unit of frailty; whole days.
    'los_days': lognormal(lambda dx, frailty: np.log(DX_LOS[dx.astype(int)] * (1 + 0.08 * frailty)), 0.6),
    # Sodium: heart failure, liver disease, and frailty lower it; heavy tails (t, 4 df) give the
    # few patients far from 140 mmol/L.
    'sodium': additive(lambda dx, liver, frailty: 138.5 - 1.5 * (dx == HF) - 2.0 * liver - 0.6 * frailty,
                       gcm.ScipyDistribution(stats.t, df=4, loc=0, scale=2.6)),
    'hemoglobin': additive(lambda frailty, dx, ckd, cancer:
                           12.8 - 0.4 * frailty - 2.2 * (dx == GI_BLEED) - 1.0 * ckd - 0.8 * cancer, 1.7),
    'creatinine': lognormal(lambda ckd, age, dx: np.log(0.95) + 0.6 * ckd + 0.004 * (age - 60) + 0.1 * (dx == HF), 0.3),
    'n_meds': chance(lambda n_comorb, age: poisson_probs(4 + 1.5 * n_comorb + 0.03 * np.clip(age - 50, 0, None), 40),
                     classes=np.arange(41)),
    'ed_visits_6m': chance(lambda ed_rate: poisson_probs(ed_rate, 60), classes=np.arange(61)),
    'died': chance(lambda age, dx, frailty: sigmoid(-5.2 + 0.035 * (age - 65) + 1.4 * (dx == SEPSIS) + 0.3 * frailty)),
    'snf': chance(lambda age, los_days, dementia: sigmoid(-2.8 + 0.06 * (age - 75) + 0.18 * np.floor(los_days)
                                                          + 0.6 * dementia)),
    'home_health': chance(lambda snf: 0.22 * (1 - snf)),
    'ama': chance(lambda snf, home_health, age, housing_unstable:
                  (1 - snf) * (1 - home_health) * (0.015 + 0.04 * (age < 45) + 0.04 * housing_unstable)),
    'followup_scheduled': chance(lambda died, hospital, home_health, housing_unstable, insurance, ama:
                                 (1 - died) * np.clip(HOSPITAL_FOLLOWUP[hospital.astype(int)] + 0.1 * home_health
                                                      - 0.2 * housing_unstable - 0.08 * (insurance == MEDICAID)
                                                      - 0.3 * ama, 0, 1)),
    'readmit': chance(readmission),
    # Readmitted patients often miss the visit because they are back in the hospital.
    'followup_completed': chance(lambda followup_scheduled, readmit: followup_scheduled * np.where(readmit == 1, 0.45, 0.8)),
}
# Computed by the admission loop: the admission date (set by frailty and, for a later admission, by
# whether the previous one was readmitted), age and season on that date, the comorbidity count, and
# admissions in the 365 days before.
DISCHARGE_INPUTS = {'admit_day': ['frailty'], 'age': ['age0', 'admit_day'], 'winter': ['admit_day'],
                    'n_comorb': list(COMORBIDITIES), 'prior_admits': ['admit_day']}
DISCHARGE_PATIENT_NODES = list(DISCHARGE_MODEL)[:list(DISCHARGE_MODEL).index('dx')]
ADMISSION_NODES = list(DISCHARGE_MODEL)[len(DISCHARGE_PATIENT_NODES):]


def build_discharges(rng):
    scm = causal_model(DISCHARGE_MODEL, DISCHARGE_INPUTS)
    n = DISCHARGE_PATIENTS
    p = draw_roots_down(scm, DISCHARGE_PATIENT_NODES, n)
    p['n_comorb'] = p[list(COMORBIDITIES)].sum(axis=1)
    frail = p['frailty'].to_numpy()

    admit = rng.exponential(1100 / (0.4 + frail))
    alive = np.ones(n, dtype=bool)
    rounds = []
    while True:
        cur = np.nonzero(alive & (admit <= WINDOW_DAYS))[0]
        if len(cur) == 0:
            break
        a = p.iloc[cur].reset_index(drop=True)
        a['pid'], a['admit_day'] = cur, admit[cur]
        a['age'] = a['age0'] + a['admit_day'] / 365
        a['winter'] = (np.cos(2 * np.pi * (a['admit_day'] % 365.25 - 15) / 365.25) > 0.5).astype(int)
        a['prior_admits'] = count_prior(rounds, cur, admit[cur])
        a = draw(scm, a, ADMISSION_NODES)
        a['los_days'] = np.floor(a['los_days']).astype(int)
        rounds.append(a)
        readmit = a['readmit'].to_numpy() == 1
        died = a['died'].to_numpy() == 1
        f = frail[cur]
        gap = np.where(readmit, 1 + np.floor(rng.beta(1.1, 2.4, len(cur)) * 29),
                       31 + rng.exponential(700 / (0.3 + f)))
        admit[cur] = admit[cur] + a['los_days'].to_numpy() + gap
        alive[cur] = ~died & (rng.random(len(cur)) > 0.04 * f)   # some die outside the hospital
    t = pd.concat(rounds, ignore_index=True)
    t = t[t['admit_day'] + t['los_days'] <= WINDOW_DAYS]
    t = t.sort_values(['admit_day', 'pid'], kind='stable').reset_index(drop=True)

    dispo = np.select([t['died'] == 1, t['snf'] == 1, t['home_health'] == 1, t['ama'] == 1],
                      ['expired', 'snf', 'home_health', 'left_ama'], 'home')
    source = np.where(t['elective'] == 1, 'elective', np.where(t['transfer'] == 1, 'transfer', 'emergency'))
    admit_date = pd.Timestamp('2023-01-01') + pd.to_timedelta(np.floor(t['admit_day']), unit='D')
    out = pd.DataFrame({
        'discharge_id': ids('D', range(1, len(t) + 1), 7),
        'patient_id': ids('IP', t['pid'] + 1, 6),
        'hospital': np.array(HOSPITALS)[t['hospital'].astype(int)],
        'admit_date': admit_date,
        'discharge_date': admit_date + pd.to_timedelta(t['los_days'], unit='D'),
        'age': np.minimum(t['age'], 100).astype(int),
        'sex': np.where(t['male'] == 1, 'M', 'F'),
        'insurance': DISCHARGE_INSURANCE[t['insurance'].astype(int)],
        'adi_decile': t['adi'].astype(int),
        'lives_alone': t['lives_alone'] == 1,
        'housing_unstable': t['housing_unstable'] == 1,
        'admit_source': source,
        'dx_group': DX[t['dx'].astype(int)],
        'n_comorbidities': t['n_comorb'].astype(int),
        'los_days': t['los_days'],
        'ed_visits_6m': t['ed_visits_6m'].astype(int),
        'n_meds': t['n_meds'].astype(int),
        # Clip the heavy tails to values a living patient can have; the readmission cutoffs are unchanged.
        'sodium': t['sodium'].clip(110, 170).round(),
        'hemoglobin': t['hemoglobin'].clip(4, 19).round(1),
        'creatinine': t['creatinine'].round(2),
        'disposition': dispo,
        'followup_scheduled': t['followup_scheduled'] == 1,
        'followup_completed': t['followup_completed'] == 1,
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


def count_prior(rounds, cur, admit_now):
    """Admissions of each current patient in the 365 days before admit_now, from earlier rounds."""
    if not rounds:
        return np.zeros(len(cur))
    past = pd.concat([r[['pid', 'admit_day']] for r in rounds], ignore_index=True)
    past = past[past['pid'].isin(cur)]
    now = pd.Series(admit_now, index=cur)
    recent = past['admit_day'].to_numpy() >= now.loc[past['pid']].to_numpy() - 365
    counts = past[recent].groupby('pid').size()
    return counts.reindex(cur, fill_value=0).to_numpy()


# ---------------------------------------------------------------- checks and build

def check(t):
    v, p, r, d = t['pcp_visits'], t['rpm_patients'], t['rpm_readings'], t['discharges']
    assert 430_000 <= len(v) <= 560_000 and v['patient_id'].nunique() == PCP_PATIENTS
    assert 0.05 <= v['bmi'].isna().mean() <= 0.12
    assert v.duplicated(['patient_id', 'visit_date', 'sbp', 'dbp', 'bmi']).sum() > 1500
    assert len(p) == RPM_PATIENTS and p['patient_id'].is_unique
    assert 500_000 <= len(r) <= 700_000
    clean = r[r['sbp'].between(60, 260)]
    assert 130 <= clean['sbp'].mean() <= 150
    assert (r['upload_time'] >= r['reading_time']).all()
    assert 180_000 <= len(d) <= 280_000
    assert d['patient_id'].value_counts().max() >= 10
    assert (d['discharge_date'] <= EXTRACT_END).all() and (d['discharge_date'] >= d['admit_date']).all()


def build(data_dir='data'):
    """Write the four demo tables as Parquet under data_dir and return their paths."""
    # One seed per table (dowhy draws from numpy's global generator), so changing one table's
    # code leaves the others' values unchanged.
    gcm.util.general.set_random_seed(SEED * 10 + 1)
    tables = {'pcp_visits': build_pcp_visits(np.random.default_rng([SEED, 1]))}
    gcm.util.general.set_random_seed(SEED * 10 + 2)
    tables['rpm_patients'], tables['rpm_readings'] = build_rpm(np.random.default_rng([SEED, 2]))
    gcm.util.general.set_random_seed(SEED * 10 + 3)
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
