"""Build Assignment 08's imaging dataset and freeze the values its checks expect.

    uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 \\
        python scripts/build_assignment08_data.py

Writes `08/assignment/data/imaging_exams.parquet`, one row per imaging exam
in 2025 at a regional network's eight sites, and rewrites the block between
the GENERATED markers in `08/assignment_checks/_value_checks.py` with the
expected values the checks compare against. It then copies that file over its
handout twin (and the handout's `.checks/` cache when present), so the two
stay byte-identical. The seed is fixed, so every run writes the same bytes.

Columns are generated in this causal order (an arrow means "is an input to"):

    home site -> deprivation decile (1 least, 10 most deprived)
    age, sex, deprivation -> smoking -> BMI
    age, smoking, BMI -> cancer, heart failure, kidney disease
    conditions, age -> imaging frailty (gamma) -> exams per patient
    active cancer -> staging CT of chest/abdomen/pelvis every few months
    age, conditions, deprivation -> care setting of each exam (ED, inpatient, outpatient)
    setting, home site -> site (ED and inpatient exams go to the patient's hospital)
    setting, age -> modality -> body region; setting -> priority (STAT, Urgent, Routine)
    setting, season, weekday -> exam date and hour
    priority, modality, site backlog, winter ED surge, scanner outage -> wait_hr (lognormal)
    modality, region, BMI -> scan_min; CT region, BMI, age, site protocol -> dose_msv
    priority, modality, reader pace, night or weekend -> report_hr
    report still unsigned at the year-end extract (long reads late in December) -> blank report_hr

Planted on top: Eastport opens in 2026 and has no exams yet, Valley has no MRI
scanner, and Northgate's MRI is replaced in March and April, so its routine MRI
waits run long those two months.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HANDOUT = ROOT / "08" / "assignment"
CHECKS = ROOT / "08" / "assignment_checks"
DATA_PATH = HANDOUT / "data" / "imaging_exams.parquet"

SEED = 20250808
N_PATIENTS = 210_000

# Reporting order: the three hospitals, the five outpatient imaging centers, then Eastport (opens 2026).
SITES = ["Downtown", "Harbor", "Northgate", "Lakeside", "Hillcrest", "Riverside", "Westfield", "Valley", "Eastport"]
HOME_SHARE = [0.20, 0.15, 0.14, 0.13, 0.11, 0.11, 0.10, 0.06]
SITE_DEPRIVATION = [6.5, 7.0, 4.5, 3.0, 2.5, 5.5, 4.0, 6.0]
# Each home site's hospital for ED and inpatient exams; Valley runs a small rural ED but sends inpatients to Harbor.
ED_SITE = [0, 1, 2, 2, 0, 1, 2, 7]
INPATIENT_SITE = [0, 1, 2, 2, 0, 1, 2, 1]
# Log-scale backlog on routine waits (staffing and scanner hours); STAT waits feel 40% of it.
SITE_BACKLOG = [0.15, 0.25, 0.00, -0.10, -0.15, 0.10, 0.05, 0.30]
# Site CT protocol: newer scanners with iterative reconstruction give lower doses (about 10 to 20%).
SITE_DOSE = [0.00, 0.08, -0.12, -0.10, -0.05, 0.05, 0.00, 0.15]
VALLEY, WESTFIELD, NORTHGATE = 7, 6, 2

MODALITIES = ["X-ray", "CT", "Ultrasound", "MRI"]
SETTINGS = ["ED", "Inpatient", "Outpatient"]
PRIORITIES = ["STAT", "Urgent", "Routine"]
# Modality mix by setting (rows follow SETTINGS, columns MODALITIES).
MODALITY_MIX = np.array([[0.45, 0.40, 0.13, 0.02],
                         [0.40, 0.30, 0.18, 0.12],
                         [0.40, 0.13, 0.26, 0.21]])
# Priority mix by setting: nearly every ED exam is STAT, outpatient exams are mostly routine.
PRIORITY_MIX = np.array([[0.85, 0.15, 0.00],
                         [0.15, 0.50, 0.35],
                         [0.02, 0.12, 0.86]])
REGIONS = {
    "X-ray": (["Chest", "Extremity", "Spine", "Abdomen/Pelvis"], [0.50, 0.33, 0.12, 0.05]),
    "CT": (["Head", "Chest", "Abdomen/Pelvis", "Spine", "Neck"], [0.34, 0.22, 0.32, 0.08, 0.04]),
    "Ultrasound": (["Abdomen/Pelvis", "Extremity", "Neck"], [0.68, 0.20, 0.12]),
    "MRI": (["Head", "Spine", "Extremity", "Abdomen/Pelvis"], [0.30, 0.36, 0.24, 0.10]),
}
# Median effective dose in mSv for a 27 kg/m2 adult (Mettler 2008, Smith-Bindman 2009). MRI and ultrasound have none.
DOSE = {("CT", "Head"): 2.0, ("CT", "Neck"): 3.0, ("CT", "Chest"): 7.0, ("CT", "Abdomen/Pelvis"): 10.0,
        ("CT", "Spine"): 6.0, ("CT", "Chest/Abdomen/Pelvis"): 14.0, ("X-ray", "Chest"): 0.02,
        ("X-ray", "Extremity"): 0.001, ("X-ray", "Spine"): 1.0, ("X-ray", "Abdomen/Pelvis"): 0.7}
# Median order-to-scan wait in hours by priority and modality. Routine outpatient MRI waits run about two weeks in
# US surveys; ED STAT imaging targets under an hour.
WAIT_MEDIAN = {"STAT": {"X-ray": 0.4, "CT": 0.6, "Ultrasound": 0.9, "MRI": 1.5},
               "Urgent": {"X-ray": 4.0, "CT": 10.0, "Ultrasound": 14.0, "MRI": 30.0},
               "Routine": {"X-ray": 20.0, "CT": 120.0, "Ultrasound": 96.0, "MRI": 300.0}}
WAIT_LOG_SD = {"STAT": 0.75, "Urgent": 0.70, "Routine": 0.60}
# Median scan-to-signed-report hours by priority (ACR targets about 1 hour for ED STAT reads), and modality factors.
REPORT_MEDIAN = {"STAT": 0.5, "Urgent": 3.0, "Routine": 10.0}
REPORT_MODALITY = {"X-ray": 0.8, "CT": 1.0, "Ultrasound": 0.9, "MRI": 1.4}
SCAN_MEDIAN = {"X-ray": 8, "CT": 12, "Ultrasound": 25, "MRI": 40}

YEAR_DAYS = pd.date_range("2025-01-01", "2025-12-31", freq="D")
HOLIDAYS = pd.to_datetime(["2025-01-01", "2025-05-26", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"])
EXTRACT_AT = pd.Timestamp("2026-01-01")


def logistic(x):
    return 1 / (1 + np.exp(-x))


def pick(rng, probabilities):
    """One column index per row of a (rows, choices) probability matrix."""
    cumulative = np.cumsum(probabilities / probabilities.sum(axis=1, keepdims=True), axis=1)
    return (rng.random(len(cumulative))[:, None] > cumulative).sum(axis=1).clip(0, probabilities.shape[1] - 1)


def patients(rng):
    n = N_PATIENTS
    home = rng.choice(8, size=n, p=HOME_SHARE)
    child = rng.random(n) < 0.10
    age = np.where(child, rng.integers(1, 18, n), np.clip(rng.normal(58, 17, n), 18, 97)).round().astype(int)
    female = rng.random(n) < 0.54
    deprivation = np.clip(np.round(np.array(SITE_DEPRIVATION)[home] + rng.normal(0, 1.8, n)), 1, 10)
    # Adult smoking runs about 12% in the US, higher in deprived areas.
    smoker = ~child & (rng.random(n) < logistic(-2.3 + 0.20 * (deprivation - 5)))
    bmi = np.where(child, rng.normal(19, 3, n),
                   np.exp(rng.normal(np.log(27.8) + 0.012 * (deprivation - 5) - 0.02 * smoker, 0.17, n)))
    # Imaging users are older and sicker than the population; prevalences here are among them.
    cancer = rng.random(n) < logistic(-3.4 + 0.045 * (age - 55) + 0.7 * smoker)
    heart_failure = rng.random(n) < logistic(-4.0 + 0.065 * (age - 55) + 0.5 * smoker + 0.04 * (bmi - 28))
    kidney = rng.random(n) < logistic(-3.3 + 0.055 * (age - 55) + 0.03 * (bmi - 28))
    frailty = rng.gamma(1.3, 1 / 1.3, n) * np.exp(0.8 * cancer + 0.6 * heart_failure + 0.4 * kidney
                                                  + 0.006 * (age - 55))
    exams = 1 + rng.poisson(1.35 * frailty)
    # About 35% of cancer patients are in active treatment, restaged by CT every two to four months.
    active = cancer & (rng.random(n) < 0.35)
    staging = np.where(active, rng.poisson(2.6, n), rng.poisson(0.4, n) * cancer)
    return pd.DataFrame({"home": home, "age": age, "female": female, "deprivation": deprivation, "bmi": bmi,
                         "cancer": cancer, "heart_failure": heart_failure, "kidney": kidney,
                         "exams": exams, "staging": staging})


def exam_dates(rng, setting):
    """A date and hour per exam: ED and inpatient exams every day with a winter surge, outpatient ones on weekdays."""
    doy = np.arange(len(YEAR_DAYS))
    weekday = YEAR_DAYS.dayofweek.to_numpy()
    winter = np.cos(2 * np.pi * (doy - 15) / 365)
    acute = 1 + 0.12 * winter
    clinic = np.where(weekday >= 5, 0.08, 1.0) * np.where(YEAR_DAYS.isin(HOLIDAYS), 0.1, 1.0)
    day = np.empty(len(setting), dtype=int)
    hour = np.empty(len(setting))
    for code, weights in ((0, acute), (1, acute), (2, clinic)):
        rows = setting == code
        day[rows] = rng.choice(len(YEAR_DAYS), size=rows.sum(), p=weights / weights.sum())
        if code == 2:
            hour[rows] = rng.uniform(7, 19, rows.sum())
        else:
            # Acute imaging peaks in the late afternoon and evening.
            hour[rows] = (rng.normal(15, 5.5, rows.sum())) % 24
    return day, hour


def exams(rng, people):
    n_regular = people["exams"].to_numpy()
    n_staging = people["staging"].to_numpy()
    owner = np.concatenate([np.repeat(np.arange(len(people)), n_regular),
                            np.repeat(np.arange(len(people)), n_staging)])
    staging = np.concatenate([np.zeros(n_regular.sum(), bool), np.ones(n_staging.sum(), bool)])
    p = people.iloc[owner].reset_index(drop=True)
    n = len(p)
    age, home = p["age"].to_numpy(), p["home"].to_numpy()
    child = age < 18

    ed = logistic(-0.9 + 0.010 * (age - 55) + 0.5 * p["heart_failure"] + 0.10 * (p["deprivation"] - 5)
                  + 0.5 * child)
    inpatient = logistic(-2.1 + 0.015 * (age - 55) + 0.7 * p["cancer"] + 0.9 * p["heart_failure"]
                         + 0.4 * p["kidney"])
    setting = pick(rng, np.column_stack([ed, inpatient * (1 - ed), (1 - ed) * (1 - inpatient)]))
    setting[staging] = 2

    mix = MODALITY_MIX[setting].copy()
    # Children get ultrasound in place of about half their CTs, to spare them radiation.
    mix[child, 2] += mix[child, 1] * 0.5
    mix[child, 1] *= 0.5
    modality = pick(rng, mix)
    modality[staging] = 1
    priority = pick(rng, PRIORITY_MIX[setting])
    priority[staging] = 2

    outpatient_site = np.where(rng.random(n) < 0.85, home, rng.choice(8, size=n, p=HOME_SHARE))
    site = np.where(setting == 0, np.array(ED_SITE)[home],
                    np.where(setting == 1, np.array(INPATIENT_SITE)[home], outpatient_site))
    # Valley has no MRI scanner: its ED sends MRIs to Harbor, and its outpatients book at Westfield.
    valley_mri = (site == VALLEY) & (modality == 3)
    site[valley_mri] = np.where(setting[valley_mri] == 2, WESTFIELD, INPATIENT_SITE[VALLEY])

    modality_name = np.array(MODALITIES)[modality]
    region = np.empty(n, dtype=object)
    for name, (labels, weights) in REGIONS.items():
        rows = modality_name == name
        region[rows] = rng.choice(labels, size=rows.sum(), p=weights)
    region[staging] = "Chest/Abdomen/Pelvis"
    # ED CTs are mostly heads (falls, strokes, headaches).
    ed_ct = (setting == 0) & (modality == 1) & ~staging
    region[ed_ct & (rng.random(n) < 0.25)] = "Head"

    day, hour = exam_dates(rng, setting)
    date = YEAR_DAYS[day]
    weekday = date.dayofweek.to_numpy()
    month = date.month.to_numpy()
    winter = np.isin(month, [12, 1, 2])

    priority_name = np.array(PRIORITIES)[priority]
    median = np.array([WAIT_MEDIAN[pr][mo] for pr, mo in zip(priority_name, modality_name)])
    spread = np.array([WAIT_LOG_SD[pr] for pr in priority_name])
    backlog = np.array(SITE_BACKLOG)[site] * np.where(priority == 0, 0.4, 1.0)
    surge = 0.15 * (winter & (setting == 0))  # winter respiratory season crowds the ED
    outage = 0.6 * ((site == NORTHGATE) & (modality == 3) & np.isin(month, [3, 4]) & (priority == 2))
    wait = np.exp(np.log(median) + backlog + surge + outage + rng.normal(0, 1, n) * spread)
    wait = np.maximum(wait.round(1), 0.1)

    bmi = p["bmi"].to_numpy()
    scan_base = np.array([SCAN_MEDIAN[mo] for mo in modality_name]) * np.where(staging, 1.5, 1.0)
    scan_min = np.maximum(np.round(scan_base * np.exp(0.01 * (bmi - 28) + rng.normal(0, 0.3, n))), 2).astype(int)

    dose_base = np.array([DOSE.get((mo, rg), np.nan) for mo, rg in zip(modality_name, region)])
    # Automatic exposure control raises dose with body size, about 3% per kg/m2; children get pediatric protocols.
    dose = dose_base * np.exp(0.03 * (bmi - 27) + np.array(SITE_DOSE)[site] + rng.normal(0, 0.25, n))
    dose = np.where(child, dose * 0.5, dose).round(3)

    # Readers: each hospital has its own radiologists; the centers share a pool; nights go to teleradiology.
    night = (hour >= 22) | (hour < 7)
    pools = {0: range(1, 9), 1: range(9, 15), 2: range(15, 21)}
    # One draw per exam, so moving exams between sites leaves every later random draw unchanged.
    draw = rng.random(n)
    reader = np.empty(n, dtype=object)
    for code in range(8):
        rows = (site == code) & ~night
        names = np.array([f"R{i:02d}" for i in pools.get(code, range(21, 33))])
        reader[rows] = names[(draw[rows] * len(names)).astype(int)]
    reader[night] = np.array(["T01", "T02", "T03", "T04"])[(draw[night] * 4).astype(int)]
    readers = sorted(set(reader))
    pace = dict(zip(readers, np.exp(rng.normal(0, 0.25, len(readers)))))
    report_median = np.array([REPORT_MEDIAN[pr] for pr in priority_name]) * np.array(
        [REPORT_MODALITY[mo] for mo in modality_name]) * np.array([pace[r] for r in reader])
    report = np.exp(np.log(report_median) + 0.2 * night + rng.normal(0, 0.65, n))
    # Non-STAT exams scanned on a weekend wait for Monday's readers.
    weekend = (weekday >= 5) & (priority != 0)
    report += np.where(weekend, (7 - weekday) * 24 - hour + 8, 0) * rng.uniform(0.6, 1.0, n)
    report = np.maximum(report.round(2), 0.05)
    scanned = date + pd.to_timedelta(hour, unit="h")
    unsigned = (scanned + pd.to_timedelta(report, unit="h") >= EXTRACT_AT) | (rng.random(n) < 0.003)
    report = np.where(unsigned, np.nan, report)

    table = pd.DataFrame({
        "patient": owner, "site": np.array(SITES)[site], "exam_date": date, "hour": hour, "age": age,
        "sex": np.where(p["female"], "F", "M"), "setting": np.array(SETTINGS)[setting],
        "priority": priority_name, "modality": modality_name, "body_region": region.astype(str),
        "scan_min": scan_min, "dose_msv": dose, "wait_hr": wait, "report_hr": report, "reader_id": reader,
    })
    table = table.sort_values(["exam_date", "hour", "patient"], kind="stable").reset_index(drop=True)
    table.insert(0, "exam_id", [f"E{i:07d}" for i in range(1, len(table) + 1)])
    # Patient IDs in a random order, so they say nothing about the patient.
    labels = rng.permutation(len(people)) + 1
    table.insert(1, "patient_id", [f"P{labels[i]:06d}" for i in table["patient"]])
    table = table.drop(columns=["patient", "hour"])
    table["exam_date"] = table["exam_date"].astype("datetime64[ms]")
    for column in ("site", "sex", "setting", "priority", "modality", "body_region", "reader_id"):
        table[column] = table[column].astype(str)
    return table


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    return exams(rng, patients(rng))


def _number(value):
    """A plain int or float for the frozen block, or None for a missing value."""
    if pd.isna(value):
        return None
    value = float(value)
    return int(value) if value.is_integer() else round(value, 4)


def expected(data: pd.DataFrame) -> dict[str, object]:
    """The values each graded artifact should hold, worked out the way the README's steps do."""
    data = data.copy()
    data["site"] = pd.Categorical(data["site"], categories=SITES, ordered=True)
    by_site = data.groupby("site", observed=True).agg(
        exam_count=("exam_id", "size"), report_count=("report_hr", "count"), patient_count=("patient_id", "nunique"),
        total_scan_min=("scan_min", "sum"), mean_scan_min=("scan_min", "mean"))
    pairs = data.groupby(["site", "modality"], observed=True).agg(
        exam_count=("exam_id", "size"), median_wait_hr=("wait_hr", "median"),
        median_report_hr=("report_hr", "median"))

    stat = data[data["priority"] == "STAT"].copy()
    stat["peer_median_wait_hr"] = stat.groupby(["site", "modality"], observed=True)["wait_hr"].transform("median")
    stat["wait_vs_peer_hr"] = stat["wait_hr"] - stat["peer_median_wait_hr"]
    delayed = stat[stat["wait_hr"] > 8]

    ct = data[data["modality"] == "CT"]
    dose = ct[ct.groupby("patient_id")["dose_msv"].transform("sum") >= 100]
    assert (ct["dose_msv"] < 100).all()

    slowest = stat.groupby("site", observed=True).apply(lambda g: g.nlargest(3, "report_hr"), include_groups=False)
    # The checks expect one answer, so no site's third and fourth slowest reports may tie.
    for site, group in stat.groupby("site", observed=True):
        top = group["report_hr"].nlargest(4).tolist()
        assert len(top) == 4 and top[2] != top[3], (site, top)

    return {
        "EXAM_ROWS": len(data),
        "SITE_ORDER": tuple(SITES),
        "MODALITIES": tuple(sorted(MODALITIES)),
        "SITE_SUMMARY": {site: tuple(_number(v) for v in row) for site, row in by_site.iterrows()},
        "SITE_MODALITY": {key: tuple(_number(v) for v in row) for key, row in pairs.iterrows()},
        "DELAYED_STAT": {row.exam_id: (row.site, row.modality, _number(row.wait_hr), _number(row.peer_median_wait_hr),
                                       _number(row.wait_vs_peer_hr)) for row in delayed.itertuples()},
        "CT_DOSE_REVIEW": {row.exam_id: (row.patient_id, _number(row.dose_msv)) for row in dose.itertuples()},
        "SLOWEST_STAT_REPORTS": {row.exam_id: (site, row.modality, _number(row.report_hr))
                                 for (site, _), row in zip(slowest.index, slowest.itertuples())},
    }


def frozen_block(values: dict[str, object]) -> str:
    """Python source for the GENERATED block: one assignment per name, dict entries one per line."""
    lines = []
    for name, value in values.items():
        if isinstance(value, dict):
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
    data = build()
    DATA_PATH.parent.mkdir(exist_ok=True)
    data.to_parquet(DATA_PATH, index=False)
    values = expected(data)
    write_checks(values)
    digest = hashlib.md5(DATA_PATH.read_bytes()).hexdigest()
    print(f"Wrote {DATA_PATH.relative_to(ROOT)}: {data.shape}, {DATA_PATH.stat().st_size / 1e6:.1f} MB, md5 {digest}")
    for name in ("SITE_MODALITY", "DELAYED_STAT", "CT_DOSE_REVIEW", "SLOWEST_STAT_REPORTS"):
        print(f"{name}: {len(values[name])} rows")


if __name__ == "__main__":
    main()
