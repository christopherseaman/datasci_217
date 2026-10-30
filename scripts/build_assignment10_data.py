"""Build Assignment 10's datasets and freeze the values its checks expect.

    uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 \\
        --with statsmodels==0.14.6 --with scikit-learn==1.9.0 python scripts/build_assignment10_data.py

Writes four Parquet files to `10/assignment/data/` and rewrites the block
between the GENERATED markers in `10/assignment_checks/_value_checks.py` with
the values the checks compare against, worked out the way the README's steps
do. It then copies that file over its handout twin (and the handout's
`.checks/` cache when present), so the two stay byte-identical. The seed is
fixed, so every run writes the same bytes. All patients and values are
synthetic.

bp_patients.parquet, one row per adult primary-care patient (Task 1). Columns
are generated in this causal order (an arrow means "is an input to"):

    clinic -> deprivation decile (ADI) -> insurance;  age -> insurance (Medicare at 65)
    age, sex, deprivation -> smoking, weekly activity minutes
    age, deprivation, activity -> BMI -> diabetes
    age (curved upward), sex x age, BMI, activity, deprivation, diabetes,
        patient intercept -> untreated SBP
    untreated SBP, age, diabetes, insurance -> on_bp_med (confounding by indication)
    untreated SBP - drug effect -> treated SBP
    treated SBP + clinic cuff offset + season + noise growing with age -> recorded SBP
    manual cuffs at two clinics -> terminal-digit rounding
    clinic, BMI over 40, age 80+ -> BMI missing (not at random)

followup_visits.parquet, one row per hypertension-program visit that has a
completed follow-up visit (Tasks 2 and 3), simulated visit by visit:

    clinic -> deprivation -> adherence;  age, sex, BMI, diabetes -> untreated level
    untreated level, patient intercept -> drugs at enrollment
    visit: true level + clinic cuff offset + season + white-coat noise -> sbp_today
    sbp_today, drugs, clinician inertia -> med_change_today (a drug added)
    drug added, adherence -> refill gap over the next 30 days -> drug effect actually taken
    true level - new drug effect + slow drift -> next true level
    next true level -> callback_sbp (home reading 3 days later), sbp_followup (next visit)
    diabetes -> a1c_result (resulted the next day)

discharges.parquet, one row per inpatient discharge (Task 3.3):

    hospital -> deprivation -> unstable housing;  age -> living alone
    age, deprivation -> latent frailty (gamma) -> Charlson index, prior ED visits (overdispersed)
    frailty, season -> emergency admission, length of stay (lognormal)
    LOS, acuity, Charlson, ED visits -> lace_score (the published LACE point rules)
    frailty, ED visits, Charlson, LOS, emergency admission, deprivation, housing,
        living alone at 75+, hospital, and two interactions -> readmitted_30d

feature_candidates.parquet: the nine candidate features of Task 2.1 and when
each becomes known, relative to the end of the visit.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HANDOUT = ROOT / "10" / "assignment"
CHECKS = ROOT / "10" / "assignment_checks"
DATA = HANDOUT / "data"

SEED = 20251010

# Primary-care clinics: patients, mean ADI decile, cuff offset (mmHg), manual cuff, share of BMI missing.
CLINICS = {
    "Alameda": (26000, 3.5, -0.5, False, 0.04),
    "Berkeley": (24000, 3.0, 0.0, False, 0.05),
    "Eastmont": (22000, 7.5, 2.5, True, 0.07),
    "Fremont": (20000, 3.5, -1.0, False, 0.04),
    "Fruitvale": (18000, 7.0, 1.5, True, 0.06),
    "Hayward": (16000, 6.0, 0.5, False, 0.05),
    "San Leandro": (14000, 5.0, 0.0, False, 0.12),
    "Union City": (10000, 4.0, -1.5, False, 0.05),
}
CLINIC_NAMES = list(CLINICS)
HOSPITALS = {"Bayside": (0.34, 6.0, 0.10), "Lakeshore": (0.28, 4.0, -0.10),
             "Summit": (0.22, 3.5, -0.20), "Valley": (0.16, 7.0, 0.20)}

PROGRAM_START = pd.Timestamp("2024-01-01")
EXTRACT_AT = pd.Timestamp("2026-01-01")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mapped(values, table):
    return pd.Series(values).map(table).to_numpy()


def season(day_of_year):
    """Winter readings run about 5 mmHg above summer ones (cohort studies: 3 to 6)."""
    return 2.5 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)


# ---------------------------------------------------------------- Task 1: primary-care patients


def bp_patients(rng) -> pd.DataFrame:
    clinic = np.repeat(CLINIC_NAMES, [CLINICS[c][0] for c in CLINIC_NAMES])
    rng.shuffle(clinic)
    n = len(clinic)
    adi = np.clip(np.round(mapped(clinic, {c: v[1] for c, v in CLINICS.items()}) + rng.normal(0, 1.8, n)), 1, 10)
    male = rng.random(n) < 0.46
    # Primary-care adults skew older than the population.
    age = np.round(18 + 77 * rng.beta(2.1, 2.0, n)).astype(int)
    medicaid = rng.random(n) < sigmoid(-2.0 + 0.45 * (adi - 5))
    uninsured = ~medicaid & (rng.random(n) < 0.04 + 0.01 * (adi - 5))
    insurance = np.where(age >= 65, "Medicare", np.where(medicaid, "Medicaid",
                                                         np.where(uninsured, "Uninsured", "Commercial")))
    smoker = rng.random(n) < sigmoid(-2.2 + 0.22 * (adi - 5) + 0.3 * male - 0.015 * (age - 45))
    active = rng.random(n) < sigmoid(0.9 - 0.03 * (age - 45) - 0.15 * (adi - 5))
    activity = np.where(active, np.exp(rng.normal(np.log(160), 0.6, n)), rng.uniform(0, 40, n))
    bmi_true = np.clip(np.exp(rng.normal(np.log(27.5), 0.18, n)) + 0.4 * (adi - 5)
                       - 0.006 * np.minimum(activity, 400) - 0.0015 * (age - 55) ** 2 + 1.3, 15, 65)
    diabetes = rng.random(n) < sigmoid(-3.3 + 0.16 * (bmi_true - 28) + 0.045 * (age - 50) + 0.15 * (adi - 5))
    # Untreated SBP rises faster after 50 (Framingham curves), about 0.9 mmHg per kg/m2 of BMI
    # (meta-analyses: about 1), men higher until their 60s, 300 min/week of activity about -4 mmHg.
    untreated = (117 + 0.32 * (age - 50) + 0.008 * (age - 50) ** 2 + 0.9 * (bmi_true - 27)
                 + male * np.clip(7 - 0.2 * (age - 45), -3, 7) - 0.013 * np.minimum(activity, 300)
                 + 0.6 * (adi - 5) + 3 * diabetes + rng.normal(0, 9, n))
    # Treated by indication: one drug lowers SBP about 10 mmHg (single-drug trials: 8 to 10).
    on_med = rng.random(n) < sigmoid((untreated - 140) / 6 + 0.03 * (age - 60) + 0.5 * diabetes
                                     - 0.8 * (insurance == "Uninsured"))
    treated = untreated - on_med * np.clip(rng.normal(10, 4, n), 2, 20)
    day = rng.integers(0, 365, n)
    visit_date = pd.Timestamp("2025-01-01") + pd.to_timedelta(day, unit="D")
    manual = mapped(clinic, {c: v[3] for c, v in CLINICS.items()})
    # Office readings vary more from visit to visit in older patients.
    noise = rng.normal(0, 1, n) * (6 + 0.12 * (age - 18))
    sbp = treated + mapped(clinic, {c: v[2] for c, v in CLINICS.items()}) + 2 * manual + season(day) + noise
    # Manual cuffs: terminal-digit preference rounds about a third of readings to the nearest 10.
    sbp = np.where(manual & (rng.random(n) < 0.35), np.round(sbp, -1), np.round(sbp)).astype(int)
    bmi = np.round(bmi_true + rng.normal(0, 0.5, n), 1)
    missing = rng.random(n) < mapped(clinic, {c: v[4] for c, v in CLINICS.items()}) + 0.10 * (bmi_true > 40) \
        + 0.08 * (age >= 80)
    bmi = np.where(missing, np.nan, bmi)
    table = pd.DataFrame({
        "clinic": clinic, "visit_date": visit_date.astype("datetime64[ms]"), "age": age,
        "sex": np.where(male, "M", "F"), "insurance": insurance, "adi_decile": adi.astype(int),
        "smoker": smoker, "activity_min_wk": np.round(activity, -1).astype(int), "diabetes": diabetes,
        "bmi": bmi, "on_bp_med": on_med, "sbp": sbp,
    })
    table = table.sort_values(["visit_date", "clinic"], kind="stable").reset_index(drop=True)
    table.insert(0, "patient_id", [f"P{i:06d}" for i in rng.permutation(n) + 1])
    return table


# ---------------------------------------------------------------- Tasks 2 and 3: follow-up visits


def office_reading(rng, level, clinic, when):
    """One office SBP: the true level plus the clinic's cuff, season, and white-coat and visit noise (about 9 mmHg SD)."""
    manual = mapped(clinic, {c: v[3] for c, v in CLINICS.items()})
    cuff = mapped(clinic, {c: v[2] for c, v in CLINICS.items()}) + 2 * manual
    reading = level + cuff + season(when.dayofyear.to_numpy()) + rng.normal(0, 9, len(level)) \
        + rng.gamma(1.5, 2.0, len(level))
    # Manual cuffs: terminal-digit preference rounds about 30% of readings to the nearest 10.
    return np.where(manual & (rng.random(len(level)) < 0.3), np.round(reading, -1), np.round(reading))


def clinic_time(rng, day: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """A local visit time on that day: clinics see patients 8:00 to 17:00."""
    return day + pd.to_timedelta(np.round(np.clip(rng.normal(12.5, 2.4, len(day)), 8, 16.9) * 60), unit="min")


def followup_visits(rng, n=62000) -> pd.DataFrame:
    clinic = rng.choice(CLINIC_NAMES, n, p=np.array([CLINICS[c][0] for c in CLINIC_NAMES]) / 150000)
    adi = np.clip(np.round(mapped(clinic, {c: v[1] for c, v in CLINICS.items()}) + rng.normal(0, 1.8, n)), 1, 10)
    male = rng.random(n) < 0.48
    age = np.round(30 + 58 * rng.beta(2.5, 2.0, n)).astype(int)
    bmi_true = np.clip(np.exp(rng.normal(np.log(30.0), 0.17, n)) + 0.4 * (adi - 5), 18, 60)
    diabetes = rng.random(n) < sigmoid(-1.5 + 0.12 * (bmi_true - 30) + 0.03 * (age - 60))
    # Patients enroll with uncontrolled blood pressure.
    untreated = (150 + 0.25 * (age - 60) + 0.6 * (bmi_true - 30) + 4 * diabetes + 0.5 * (adi - 5)
                 + rng.gamma(2.0, 5.0, n))
    drugs = np.clip(rng.poisson(np.clip((untreated - 145) / 12, 0.2, None)), 0, 3)
    # Each drug lowers SBP about 8 mmHg at standard dose (Law 2009: 9); the first works best.
    effect_per_drug = np.clip(rng.normal(8.5, 3, n), 2, 16)
    adherence = np.clip(rng.beta(6, 1.6, n) - 0.02 * (adi - 5), 0.1, 1)
    # Clinicians differ in therapeutic inertia: about half of high readings get a drug added.
    clinic_inertia = {c: v for c, v in zip(CLINIC_NAMES, rng.normal(0, 0.35, len(CLINIC_NAMES)))}
    weekdays = pd.bdate_range(PROGRAM_START, "2025-09-30")
    enrolled = clinic_time(rng, weekdays[rng.integers(0, len(weekdays), n)])

    rows = {key: [] for key in ("patient", "visit_time", "followup_time", "age", "bmi", "n_bp_meds",
                                 "med_change_today", "sbp_today", "prior_sbp", "a1c_result",
                                 "callback_sbp", "refill_gap_days", "sbp_followup")}
    level = untreated - drugs * effect_per_drug * adherence + rng.normal(0, 4, n)
    time = enrolled
    prior = np.full(n, np.nan)
    active = np.ones(n, bool)
    bmi_now = bmi_true.copy()
    a1c_base = np.where(diabetes, rng.normal(7.6, 1.1, n), rng.normal(5.5, 0.35, n))
    sbp_today = office_reading(rng, level, clinic, time)
    for _ in range(18):
        doy = time.dayofyear.to_numpy()
        high = sbp_today >= 140
        p_add = sigmoid(-0.6 + 0.07 * (sbp_today - 145) - 0.8 * drugs + mapped(clinic, clinic_inertia))
        change = high & (drugs < 4) & (rng.random(n) < p_add)
        drugs_after = drugs + change
        # Next visit: two to twelve weeks later on the same weekday, sooner after a change or a very high reading.
        gap = np.clip(np.exp(rng.normal(np.log(32) - 0.15 * change - 0.004 * (sbp_today - 140), 0.35, n)), 14, 84)
        nxt = clinic_time(rng, time.normalize() + pd.to_timedelta(7 * np.round(gap / 7), unit="D"))
        # Refill gap over the next 30 days: poor adherers and new prescriptions lapse more often.
        lapse = rng.random(n) < sigmoid(-1.6 - 4 * (adherence - 0.8) + 0.6 * change)
        refill_gap = np.where(lapse, np.round(rng.gamma(1.6, 6, n)), 0).clip(0, 30)
        taken = 1 - refill_gap / 45
        new_level = (level - change * effect_per_drug * np.where(drugs == 0, 1.2, 0.85) * adherence * taken
                     + 3.5 * (refill_gap > 7) + 0.4 * rng.normal(0, 1, n) * np.sqrt(gap / 7)
                     + 0.02 * (level - 150) * -1)
        followup = office_reading(rng, new_level, clinic, nxt)
        # Home readings run about 5 mmHg below office ones (no white coat).
        callback = np.round(new_level + season(doy) - 5 + rng.normal(0, 6, n))
        a1c = np.round(a1c_base + rng.normal(0, 0.25, n), 1)
        bmi_rec = np.where(rng.random(n) < 0.06 + 0.10 * (bmi_now > 40), np.nan, np.round(bmi_now + rng.normal(0, 0.4, n), 1))
        # About 70% of program patients get an A1c with each visit; the rest have none ordered.
        a1c = np.where(rng.random(n) < 0.7, a1c, np.nan)
        done = active & (nxt < EXTRACT_AT)
        for key, value in (("patient", np.arange(n)), ("visit_time", time), ("followup_time", nxt),
                           ("age", age + ((time - enrolled).days.to_numpy() // 365)), ("bmi", bmi_rec),
                           ("n_bp_meds", drugs), ("med_change_today", change), ("sbp_today", sbp_today),
                           ("prior_sbp", prior), ("a1c_result", a1c), ("callback_sbp", callback),
                           ("refill_gap_days", refill_gap), ("sbp_followup", followup)):
            rows[key].append(np.asarray(value)[done])
        # About 6% of patients leave the program after each visit.
        active = done & (rng.random(n) > 0.06)
        prior = sbp_today
        sbp_today = followup
        level = new_level
        drugs = drugs_after
        bmi_now = bmi_now + rng.normal(0, 0.3, n)
        time = nxt
    data = {key: np.concatenate(parts) for key, parts in rows.items()}
    patient = data.pop("patient")
    table = pd.DataFrame(data)
    table.insert(0, "clinic", clinic[patient])
    order = np.lexsort((patient, table["visit_time"].to_numpy()))
    table = table.iloc[order].reset_index(drop=True)
    patient = patient[order]
    labels = rng.permutation(n) + 1
    table.insert(0, "patient_id", [f"H{labels[p]:05d}" for p in patient])
    table.insert(0, "visit_id", [f"FV{i:06d}" for i in range(1, len(table) + 1)])
    for column in ("visit_time", "followup_time"):
        # Clinic hours are Pacific time; the extract stores UTC.
        local = pd.DatetimeIndex(table[column]).floor("min").tz_localize("America/Los_Angeles",
                                                                          nonexistent="shift_forward",
                                                                          ambiguous=False)
        table[column] = local.tz_convert("UTC").astype("datetime64[ms, UTC]")
    for column in ("sbp_today", "sbp_followup", "callback_sbp", "n_bp_meds", "refill_gap_days"):
        table[column] = table[column].astype(int)
    table["med_change_today"] = table["med_change_today"].astype(int)
    return table


def feature_candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_feature": ["age", "bmi", "sbp_today", "prior_sbp", "n_bp_meds", "med_change_today",
                              "a1c_result", "callback_sbp", "refill_gap_days"],
        "becomes_known": ["registration", "intake", "intake", "previous visit", "medication list",
                          "end of visit", "lab report the next day", "nurse phone call 3 days later",
                          "pharmacy claims 30 days later"],
        "hours_after_visit": [0, 0, 0, -672, 0, 0, 24, 72, 720],
    })


# ---------------------------------------------------------------- Task 3.3: discharges


def lace(los, emergency, charlson, ed_visits):
    """The LACE index (van Walraven 2010): Length of stay, Acuity, Comorbidity, ED visits; 0 to 19."""
    l_points = np.select([los < 1, los < 4, los < 7, los < 14], [0, np.floor(los), 4, 5], 7)
    c_points = np.where(charlson >= 4, 5, charlson)
    e_points = np.minimum(ed_visits, 4)
    return (l_points + 3 * emergency + c_points + e_points).astype(int)


def discharges(rng, n=210000) -> pd.DataFrame:
    names = list(HOSPITALS)
    hospital = rng.choice(names, n, p=[HOSPITALS[h][0] for h in names])
    adi = np.clip(np.round(mapped(hospital, {h: v[1] for h, v in HOSPITALS.items()}) + rng.normal(0, 1.8, n)), 1, 10)
    age = np.round(18 + 80 * rng.beta(2.8, 1.7, n)).astype(int)
    male = rng.random(n) < 0.48
    housing = rng.random(n) < sigmoid(-3.6 + 0.35 * (adi - 5))
    alone = rng.random(n) < sigmoid(-1.4 + 0.03 * (age - 60))
    frailty = rng.gamma(2.0, 0.5, n) * np.exp(0.015 * (age - 65) + 0.04 * (adi - 5))
    charlson = np.minimum(rng.poisson(1.3 * frailty), 10)
    # Prior ED visits are overdispersed: most patients have none, a few have many.
    ed_visits = rng.negative_binomial(0.8, 0.8 / (0.8 + 0.7 * frailty * (1 + 0.6 * housing)))
    day = rng.integers(0, 731, n)
    discharge_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(day, unit="D")
    winter = np.cos(2 * np.pi * (discharge_date.dayofyear.to_numpy() - 15) / 365.25)
    emergency = rng.random(n) < sigmoid(0.9 + 0.5 * (frailty - 1) + 0.3 * winter)
    # Length of stay is lognormal: median about 3.5 days, longer for frail and emergency patients.
    los = np.round(np.exp(rng.normal(np.log(3.2) + 0.18 * frailty + 0.25 * emergency, 0.65, n)))
    los = np.clip(los, 0, 60)
    score = lace(los, emergency, charlson, ed_visits)
    # Readmission: about 14% overall (US Medicare: 15 to 17). LACE carries part of the signal; frailty,
    # social factors, and the hospital carry the rest. Effects are on the log-odds scale.
    logit = (-2.75 + 0.45 * (frailty - 1) + 0.16 * np.minimum(ed_visits, 6) + 0.08 * np.minimum(charlson, 6)
             + 0.025 * np.minimum(los, 20) + 0.25 * emergency + 0.05 * (adi - 5) + 0.45 * housing
             + 0.35 * (alone & (age >= 75)) + mapped(hospital, {h: v[2] for h, v in HOSPITALS.items()})
             + 0.5 * (housing & (ed_visits >= 3)) + 0.3 * ((charlson >= 4) & (los >= 7)))
    readmitted = rng.random(n) < sigmoid(logit)
    table = pd.DataFrame({
        "hospital": hospital, "discharge_date": discharge_date.astype("datetime64[ms]"), "age": age,
        "sex": np.where(male, "M", "F"), "admission_type": np.where(emergency, "Emergency", "Elective"),
        "los_days": los.astype(int), "charlson_index": charlson.astype(int), "ed_visits_6mo": ed_visits.astype(int),
        "lace_score": score, "readmitted_30d": readmitted.astype(int),
    })
    table = table.sort_values(["discharge_date", "hospital"], kind="stable").reset_index(drop=True)
    table.insert(0, "discharge_id", [f"D{i:06d}" for i in range(1, n + 1)])
    return table


def build() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    return {"bp_patients": bp_patients(rng), "followup_visits": followup_visits(rng),
            "feature_candidates": feature_candidates(), "discharges": discharges(rng)}




# ---------------------------------------------------------------- expected values

# The assignment's settings; the notebook scaffold sets the same ones.
FORMULA = "sbp ~ age + bmi"
NEW_PATIENT = {"age": 60, "bmi": 31.0}
AGE_BINS = [17, 39, 49, 59, 69, 79, 120]
AGE_LABELS = ["18-39", "40-49", "50-59", "60-69", "70-79", "80+"]
TARGET = "sbp_followup"
VALIDATION_START = pd.Timestamp("2025-07-01", tz="UTC")
TEST_START = pd.Timestamp("2025-10-01", tz="UTC")
LACE_RULES = {"flag_no_one": None, "lace_13_plus": 13, "lace_10_plus": 10}


def _round(value) -> float:
    value = float(value)
    return int(value) if value.is_integer() else round(value, 6)


def _instant(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def ols_tables(patients: pd.DataFrame, formula: str = FORMULA, alpha: float = 0.05, size: bool = False):
    """The coefficient rows, new-patient intervals, and residual bands Task 1 saves."""
    import statsmodels.formula.api as smf

    results = smf.ols(formula, data=patients).fit()
    bounds = results.conf_int(alpha=alpha)
    coefficients = {(term,): {"coef": _round(results.params[term]), "std_err": _round(results.bse[term]),
                              "ci_lower": _round(bounds.loc[term, 0]), "ci_upper": _round(bounds.loc[term, 1])}
                    for term in results.params.index}
    frame = results.get_prediction(pd.DataFrame({k: [v] for k, v in NEW_PATIENT.items()})).summary_frame(alpha=alpha)
    intervals = {(NEW_PATIENT["age"], NEW_PATIENT["bmi"]): {
        column: _round(frame[column].iloc[0])
        for column in ("mean", "mean_ci_lower", "mean_ci_upper", "obs_ci_lower", "obs_ci_upper")}}
    diagnostics = pd.DataFrame({"age": patients["age"], "residual": results.resid})
    bands = pd.cut(diagnostics["age"], AGE_BINS, labels=AGE_LABELS)
    summary = diagnostics.groupby(bands, observed=True)["residual"].agg(["count", "size", "mean", "std"])
    residual_bands = {(band,): {"patients": int(row["size" if size else "count"]),
                                "mean_residual": _round(row["mean"]), "sd_residual": _round(row["std"])}
                      for band, row in summary.iterrows()}
    return results, coefficients, intervals, residual_bands


def split(visits: pd.DataFrame, on: str = "followup_time", trim: bool = True, ranges: str = "followup_time"):
    """The four partitions of Task 2.2 and their summary rows."""
    train = visits[visits[on] < VALIDATION_START]
    valid = visits[(visits[on] >= VALIDATION_START) & (visits[on] < TEST_START)]
    test = visits[visits[on] >= TEST_START]
    fit = train[train["followup_time"] <= valid["visit_time"].min()] if trim else train
    parts = {"train": train, "fit": fit, "validation": valid, "test": test}
    summary = {(name,): {"row_count": len(part), "first_target_time": _instant(part[ranges].min()),
                         "last_target_time": _instant(part[ranges].max())} for name, part in parts.items()}
    return parts, summary


def _pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", LinearRegression())])


def metrics(actual, predicted, rooted: bool = True, swapped: bool = False) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(actual, predicted)
    return {"mae": _round(mean_absolute_error(actual, predicted)), "rmse": _round(np.sqrt(mse) if rooted else mse),
            "r2": _round(r2_score(predicted, actual) if swapped else r2_score(actual, predicted))}


def modeling(parts, features, baseline_rows: str = "fit", scored: str = "validation", rooted: bool = True,
             swapped: bool = False, refit: bool = False, test_scored: str = "test"):
    """Task 3.1's validation rows, and Task 3.2's test metrics and clinic errors."""
    fit, valid, test = parts["fit"], parts["validation"], parts["test"]
    rows = parts[scored]
    baseline_mean = parts[baseline_rows][TARGET].mean()
    pipeline = _pipeline().fit(fit[features], fit[TARGET])
    validation = {
        ("mean_baseline",): metrics(rows[TARGET], np.full(len(rows), baseline_mean), rooted, swapped),
        ("persistence",): metrics(rows[TARGET], rows["sbp_today"], rooted, swapped),
        ("linear_pipeline",): metrics(rows[TARGET], pipeline.predict(rows[features]), rooted, swapped),
    }
    final = _pipeline().fit(pd.concat([fit, valid])[features], pd.concat([fit, valid])[TARGET]) if refit else pipeline
    scored_rows = parts[test_scored]
    predicted = final.predict(scored_rows[features])
    test_metrics = {("linear_pipeline",): metrics(scored_rows[TARGET], predicted, rooted, swapped)}
    errors = pd.DataFrame({"clinic": scored_rows["clinic"], "error": scored_rows[TARGET] - predicted})
    by_clinic = errors.groupby("clinic")["error"].agg(
        test_visits="size", mae=lambda e: e.abs().mean(), mean_error="mean")
    clinics = {(clinic,): {"test_visits": int(row["test_visits"]), "mae": _round(row["mae"]),
                           "mean_error": _round(row["mean_error"])} for clinic, row in by_clinic.iterrows()}
    return validation, test_metrics, clinics


def readmission(table: pd.DataFrame, strict: bool = False, swapped: bool = False):
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    actual = table["readmitted_30d"]
    rows = {}
    for name, cutoff in LACE_RULES.items():
        if cutoff is None:
            flag = pd.Series(0, index=table.index)
        else:
            flag = ((table["lace_score"] > cutoff) if strict else (table["lace_score"] >= cutoff)).astype(int)
        first, second = (flag, actual) if swapped else (actual, flag)
        rows[(name,)] = {"flagged": int(flag.sum()), "accuracy": _round(accuracy_score(actual, flag)),
                         "precision": _round(precision_score(first, second, zero_division=0)),
                         "recall": _round(recall_score(first, second, zero_division=0))}
    return rows


def _changed(slip: dict, correct: dict) -> dict:
    """Only the rows and cells where a wrong answer differs from the right one."""
    out = {}
    for key, values in slip.items():
        diff = {column: value for column, value in values.items()
                if key not in correct or column not in correct[key] or value != correct[key][column]}
        if diff:
            out[key] = diff
    return out


def expected(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    patients, visits = tables["bp_patients"], tables["followup_visits"]
    candidates, stays = tables["feature_candidates"], tables["discharges"]
    results, coefficients, intervals, bands = ols_tables(patients)
    hours = dict(zip(candidates["candidate_feature"], candidates["hours_after_visit"].astype(int)))
    features = [name for name, value in hours.items() if value <= 0]
    parts, split_summary = split(visits)
    validation, test_metrics, clinics = modeling(parts, features)
    flags = readmission(stays)
    winner = min(validation, key=lambda key: validation[key]["mae"])
    assert winner == ("linear_pipeline",), validation

    correct = {"coefficients": coefficients, "intervals": intervals, "bands": bands, "split": split_summary,
               "validation": validation, "test": test_metrics, "clinics": clinics, "readmission": flags}
    variants: dict[str, dict] = {}

    def add(name, **artifacts):
        variants[name] = {artifact: _changed(rows, correct[artifact]) for artifact, rows in artifacts.items()}

    for name, formula in (("formula_without_age", "sbp ~ bmi"), ("formula_without_bmi", "sbp ~ age")):
        _, c, i, b = ols_tables(patients, formula)
        add(name, coefficients=c, intervals=i, bands=b)
    _, c, i, _ = ols_tables(patients, alpha=0.10)
    add("intervals_at_90", intervals=i)
    _, _, _, b = ols_tables(patients, size=True)
    add("bands_size_with_missing", bands=b)
    add("bands_sign_flipped", bands={key: {**row, "mean_residual": -row["mean_residual"]} for key, row in bands.items()})
    # Feature sets from a wrong availability rule, carried into Tasks 3.1 and 3.2.
    for name, rule in (("strict_zero", lambda h: h < 0), ("equals_zero", lambda h: h == 0),
                       ("no_audit", lambda h: True)):
        chosen = [feature for feature, value in hours.items() if rule(value)]
        v, t, k = modeling(parts, chosen)
        add(f"features_{name}", validation=v, test=t, clinics=k)
    visit_parts, s = split(visits, on="visit_time")
    v, t, k = modeling(visit_parts, features)
    add("split_on_visit_time", split=s, validation=v, test=t, clinics=k)
    _, s = split(visits, ranges="visit_time")
    add("ranges_from_visit_time", split=s)
    untrimmed, s = split(visits, trim=False)
    v, t, k = modeling(untrimmed, features, baseline_rows="train")
    add("fit_not_trimmed", split=s, validation=v, test=t, clinics=k)
    v, _, _ = modeling(parts, features, baseline_rows="validation")
    add("baseline_from_validation", validation=v)
    v, t, _ = modeling(parts, features, rooted=False)
    add("rmse_not_rooted", validation=v, test=t)
    v, t, _ = modeling(parts, features, swapped=True)
    add("r2_args_swapped", validation=v, test=t)
    v, _, _ = modeling(parts, features, scored="test")
    add("validation_on_test_rows", validation=v)
    _, t, k = modeling(parts, features, refit=True)
    add("refit_on_validation", test=t, clinics=k)
    _, t, _ = modeling(parts, features, test_scored="validation")
    add("test_scored_on_validation", test=t)
    add("mean_error_sign_flipped",
        clinics={key: {**row, "mean_error": -row["mean_error"]} for key, row in clinics.items()})
    add("lace_strictly_greater", readmission=readmission(stays, strict=True))
    add("flag_passed_first", readmission=readmission(stays, swapped=True))
    for name, artifacts in variants.items():
        assert any(artifacts.values()), f"{name} changes nothing"

    return {
        "PATIENT_ROWS": len(patients),
        "ROWS_USED": int(results.nobs),
        "VISIT_ROWS": len(visits),
        "DISCHARGE_ROWS": len(stays),
        "NEW_PATIENT": (NEW_PATIENT["age"], NEW_PATIENT["bmi"]),
        "AGE_BINS": tuple(AGE_BINS),
        "AGE_BANDS": tuple(AGE_LABELS),
        "FEATURES": tuple(features),
        "CANDIDATE_HOURS": hours,
        "COEFFICIENT_VALUES": coefficients,
        "NEW_PATIENT_VALUES": intervals,
        "RESIDUAL_BAND_VALUES": bands,
        "SPLIT_VALUES": split_summary,
        "VALIDATION_VALUES": validation,
        "TEST_METRIC_VALUES": test_metrics,
        "CLINIC_TEST_VALUES": clinics,
        "READMISSION_VALUES": flags,
        "VARIANTS": variants,
    }


def frozen_block(values: dict[str, object]) -> str:
    """Python source for the GENERATED block: one assignment per name, dict entries one per line."""
    lines = []
    for name, value in values.items():
        if isinstance(value, dict) and name == "VARIANTS":
            lines.append(f"{name} = {{")
            for variant, artifacts in value.items():
                lines.append(f"    {variant!r}: {{")
                for artifact, rows in artifacts.items():
                    if rows:
                        lines.append(f"        {artifact!r}: {rows!r},")
                lines.append("    },")
            lines.append("}")
        elif isinstance(value, dict):
            lines.append(f"{name} = {{")
            lines += [f"    {key!r}: {item!r}," for key, item in value.items()]
            lines.append("}")
        else:
            lines.append(f"{name} = {value!r}")
    return "\n".join(lines) + "\n"


def write_checks(values: dict[str, object]) -> None:
    source = CHECKS / "_value_checks.py"
    text = source.read_text(encoding="utf-8")
    pattern = re.compile(r"(# BEGIN GENERATED[^\n]*\n).*?(# END GENERATED\n)", re.S)
    assert pattern.search(text), f"{source} has no GENERATED block"
    text = pattern.sub(lambda match: match.group(1) + frozen_block(values) + match.group(2), text)
    source.write_text(text, encoding="utf-8")
    for twin in (HANDOUT / "_value_checks.py", HANDOUT / ".checks" / "_value_checks.py"):
        if twin.parent.is_dir():
            shutil.copyfile(source, twin)


def main() -> None:
    tables = build()
    DATA.mkdir(exist_ok=True)
    for name, table in tables.items():
        path = DATA / f"{name}.parquet"
        table.to_parquet(path, index=False)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        print(f"Wrote {path.relative_to(ROOT)}: {table.shape}, {path.stat().st_size / 1e6:.1f} MB, md5 {digest}")
    values = expected(tables)
    write_checks(values)
    print(f"Froze {len(values)} names and {len(values['VARIANTS'])} wrong-answer variants in the checks.")


if __name__ == "__main__":
    main()
