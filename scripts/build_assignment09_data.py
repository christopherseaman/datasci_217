"""Build Assignment 09's hospital-at-home dataset and freeze the values its checks expect.

    uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 \\
        python scripts/build_assignment09_data.py

Writes three Parquet files to `09/assignment/data/` for a New York health
system's hospital-at-home program, January 5 to March 27, 2026: patients
treated at home for an acute illness, with a wearable patch that sends a heart
rate every 10 minutes and a visiting nurse who draws blood.

- episodes.parquet: one row per episode (admission to discharge or escalation back to hospital).
- vitals.parquet: one row per patch reading, `recorded_at` as UTC text from the vendor's feed.
- labs.parquet: one row per blood test, collection and result times typed on the New York wall clock.

It then rewrites the block between the GENERATED markers in
`09/assignment_checks/_value_checks.py` with the values the checks compare
against, and copies that file over its handout twin (and the handout's
`.checks/` cache when present), so the two stay byte-identical. The seed is
fixed, so every run writes the same bytes.

Columns are generated in this causal order (an arrow means "is an input to"):

    patient: neighborhood -> deprivation decile (1 least, 10 most deprived)
    age, deprivation -> smoking -> COPD; age -> heart failure, kidney disease, atrial fibrillation (AF)
    heart failure, AF -> beta-blocker (slows the heart); AF -> warfarin (INR checks)
    week of the year (flu season peaks late January), weekday -> admission time
    flu season, COPD, heart failure -> diagnosis
    age, conditions, diagnosis, deprivation -> severity (latent) -> escalation, and when it happens
    severity, diagnosis -> length of a completed home stay (lognormal)
    patient random intercept, age, beta-blocker, AF, diagnosis, severity -> baseline heart rate
    local hour (lower at night), recovery over the stay, deterioration ramp before escalation -> heart rate
    diagnosis (pneumonia, cellulitis) -> fever bouts; AF -> rapid-rate bouts; both resolve at home -> heart rate
    daytime movement, younger age -> motion-artifact spikes; AF -> beat-to-beat scatter
    deprivation (home broadband) -> hub dropouts; age (skin) -> patch off; day 3 patch change; vendor outage
    severity, diagnosis, deterioration -> which labs are drawn (lactate mostly for the sicker) -> values
    test, courier, evening collection (overnight batch) -> result time; analyzer clock on the spring-forward night
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HANDOUT = ROOT / "09" / "assignment"
CHECKS = ROOT / "09" / "assignment_checks"
DATA = HANDOUT / "data"

SEED = 20260309
N_PATIENTS = 1_250
N_EPISODES = 1_400
ZONE = "America/New_York"
FIRST_DAY, LAST_DAY = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-03-27")
CADENCE_MIN = 10
# The vendor's cloud was down for six hours; no patch reading from then reached the program.
OUTAGE = (pd.Timestamp("2026-02-17 13:00", tz="UTC"), pd.Timestamp("2026-02-17 19:00", tz="UTC"))
DIAGNOSES = ["Heart failure", "COPD", "Pneumonia", "Cellulitis", "Urinary infection"]

# Task constants the README, notebook, and checks share.
ALERT_BPM = 120
SCORE_AFTER_H = 24
HOLDOUT_FROM = pd.Timestamp("2026-03-01", tz="UTC")
FLAG_BPM = 100


def logistic(x):
    return 1 / (1 + np.exp(-x))


def pick(rng, probabilities):
    """One column index per row of a (rows, choices) probability matrix."""
    cumulative = np.cumsum(probabilities / probabilities.sum(axis=1, keepdims=True), axis=1)
    return (rng.random(len(cumulative))[:, None] > cumulative).sum(axis=1).clip(0, probabilities.shape[1] - 1)


def patients(rng):
    n = N_PATIENTS
    age = np.clip(rng.normal(73, 11, n), 38, 99).round().astype(int)
    female = rng.random(n) < 0.55
    deprivation = rng.integers(1, 11, n)
    smoker = rng.random(n) < logistic(-1.0 + 0.18 * (deprivation - 5) - 0.02 * (age - 70))
    # Hospital-at-home patients are older and sicker than the population; prevalences are among them.
    copd = rng.random(n) < logistic(-1.6 + 1.3 * smoker + 0.02 * (age - 70))
    heart_failure = rng.random(n) < logistic(-1.3 + 0.05 * (age - 70))
    kidney = rng.random(n) < logistic(-1.5 + 0.05 * (age - 70) + 0.4 * heart_failure)
    af = rng.random(n) < logistic(-1.9 + 0.06 * (age - 70) + 0.5 * heart_failure)
    beta_blocker = (heart_failure | af) & (rng.random(n) < 0.75)
    warfarin = af & (rng.random(n) < 0.35)
    # A patient's own resting heart rate differs from the next patient's by about 8 bpm (SD).
    intercept = rng.normal(0, 8, n)
    # Some people return to the program; their visits cluster (gamma frailty).
    frailty = rng.gamma(1.5, 1 / 1.5, n) * np.exp(0.4 * copd + 0.4 * heart_failure)
    return pd.DataFrame({"age": age, "female": female, "deprivation": deprivation, "smoker": smoker, "copd": copd,
                         "heart_failure": heart_failure, "kidney": kidney, "af": af, "beta_blocker": beta_blocker,
                         "warfarin": warfarin, "intercept": intercept, "frailty": frailty})


def episodes(rng, people):
    n = N_EPISODES
    who = rng.choice(len(people), size=n, p=(people["frailty"] / people["frailty"].sum()).to_numpy())
    p = people.iloc[who].reset_index(drop=True)
    days = pd.date_range(FIRST_DAY, LAST_DAY, freq="D")
    # Flu season peaks in late January; programs admit fewer patients on weekends.
    flu = np.exp(-0.5 * ((days - pd.Timestamp("2026-01-27")).days.to_numpy() / 16.0) ** 2)
    weight = (1 + 0.8 * flu) * np.where(days.dayofweek >= 5, 0.55, 1.0)
    day = rng.choice(len(days), size=n, p=weight / weight.sum())
    # Admitted from the ED or clinic, mostly between late morning and evening on the local clock.
    local_hour = np.clip(rng.normal(15.5, 3.2, n), 8, 23.9)
    local = days[day] + pd.to_timedelta(local_hour, unit="h")
    admitted = local.floor("min").tz_localize(ZONE, nonexistent="shift_forward", ambiguous=False).tz_convert("UTC")
    season = flu[day]

    copd, hf = p["copd"].to_numpy(), p["heart_failure"].to_numpy()
    scores = np.column_stack([
        0.3 + 1.8 * hf,                      # heart failure
        0.2 + 1.6 * copd * (0.6 + season),   # COPD exacerbation, more in flu season
        0.6 + 1.2 * season,                  # pneumonia
        0.55 + 0 * season,                   # cellulitis
        0.6 + 0 * season,                    # urinary infection
    ])
    diagnosis = pick(rng, scores)
    age = p["age"].to_numpy()
    conditions = (p[["copd", "heart_failure", "kidney", "af"]].sum(axis=1)).to_numpy()
    diagnosis_effect = np.array([0.3, 0.3, 0.45, -0.5, -0.3])[diagnosis]
    severity = (rng.normal(0, 1, n) + 0.25 * (conditions - 1) + diagnosis_effect + 0.02 * (age - 73)
                + 0.04 * (p["deprivation"].to_numpy() - 5) + 0.35 * season)
    # Escalation back to hospital runs about 7 to 12% in published hospital-at-home programs.
    escalated = rng.random(n) < logistic(-2.75 + 0.95 * severity)
    hours_to_escalation = np.exp(rng.normal(np.log(38), 0.65, n))
    stay_hours = np.clip(np.exp(rng.normal(np.log(78) + 0.08 * severity, 0.3, n)), 30, 240)
    length = np.where(escalated, np.clip(hours_to_escalation, 3, 200), stay_hours)
    ended = (admitted + pd.to_timedelta(np.round(length * 60), unit="min"))

    table = pd.DataFrame({
        "patient": who, "admitted_at": admitted, "ended_at": ended,
        "outcome": np.where(escalated, "escalated", "discharged home"),
        "diagnosis": np.array(DIAGNOSES)[diagnosis], "severity": severity,
    })
    table = pd.concat([table, p.drop(columns=["frailty"])], axis=1)
    table = table.sort_values(["admitted_at", "patient"], kind="stable").reset_index(drop=True)
    table.insert(0, "episode_id", [f"H{i:05d}" for i in range(1, n + 1)])
    labels = rng.permutation(len(people)) + 1
    table["patient_id"] = [f"P{labels[i]:05d}" for i in table["patient"]]
    return table


def gap_mask(rng, ep, minutes):
    """True for the readings of one episode that never arrive (`minutes` since patch on)."""
    hours = minutes / 60
    lost = np.zeros(len(minutes), bool)

    def drop(start_h, length_h):
        lost[(hours >= start_h) & (hours < start_h + length_h)] = True

    total = hours[-1] if len(hours) else 0
    # Hub dropouts: home broadband is patchier in deprived neighborhoods (about 1 a day at decile 10).
    rate = 0.012 + 0.0035 * (ep.deprivation - 1) + rng.gamma(2, 0.004)
    for _ in range(rng.poisson(rate * total)):
        drop(rng.uniform(0, total), np.exp(rng.normal(np.log(1.6), 0.7)))
    # Patch taken off for skin care, more often in older skin; most of these are under an hour.
    for _ in range(rng.poisson(total / 24 * max(0.1, 0.4 + 0.02 * (ep.age - 70)))):
        drop(rng.uniform(0, total), rng.uniform(0.3, 1.6))
    # The patch battery lasts three days; the nurse swaps it at the next visit.
    if total > 74:
        drop(72 + rng.uniform(0, 3), rng.uniform(1, 5))
    return lost


def vitals(rng, eps):
    rows = []
    for ep in eps.itertuples():
        start = ep.admitted_at + pd.Timedelta(minutes=int(rng.integers(20, 150)))
        stop = ep.ended_at - pd.Timedelta(minutes=int(rng.integers(0, 40)))
        if stop <= start:
            continue
        count = int((stop - start) / pd.Timedelta(minutes=CADENCE_MIN)) + 1
        minutes = np.arange(count) * CADENCE_MIN
        keep = ~gap_mask(rng, ep, minutes)
        rows.append(pd.DataFrame({"row": ep.Index, "at": start + pd.to_timedelta(minutes[keep], unit="min"),
                                  "into": minutes[keep] / 60, "left": (stop - start) / pd.Timedelta(hours=1)
                                  - minutes[keep] / 60}))
    readings = pd.concat(rows, ignore_index=True)
    out = readings["at"]
    readings = readings[~((out >= OUTAGE[0]) & (out < OUTAGE[1]))].reset_index(drop=True)

    e = eps.iloc[readings["row"]].reset_index(drop=True)
    n = len(readings)
    diag = e["diagnosis"].to_numpy()
    # Baselines: beta-blockers lower resting rate by about 10 bpm; pneumonia and fever raise it.
    base = (76 + e["intercept"].to_numpy() - 0.12 * (e["age"].to_numpy() - 73) - 10 * e["beta_blocker"].to_numpy()
            + 7 * e["af"].to_numpy() + np.where(diag == "Pneumonia", 7, 0) + np.where(diag == "COPD", 4, 0)
            + 3.0 * e["severity"].to_numpy())
    local = pd.DatetimeIndex(readings["at"]).tz_convert(ZONE)
    clock = local.hour.to_numpy() + local.minute.to_numpy() / 60
    # Heart rate dips 6 to 10 bpm in the small hours and peaks mid-afternoon.
    circadian = 4.5 * np.cos(2 * np.pi * (clock - 15) / 24)
    into, left = readings["into"].to_numpy(), readings["left"].to_numpy()
    escalated = (e["outcome"] == "escalated").to_numpy()
    # A completed stay recovers by about 6 bpm over the first two days as the infection or fluid clears.
    recovery = -6 * (1 - np.exp(-into / 30)) * ~escalated
    # Before escalation, heart rate climbs by about 20 to 35 bpm over the last 6 to 24 hours (sepsis, decompensation).
    ramp_hours = 6 + 18 * logistic(rng.normal(0, 1, len(eps)))[readings["row"]]
    rise = rng.normal(27, 6, len(eps))
    # A second stream for the effects below, so reading times, gaps, and labs keep the main stream's values.
    extra = np.random.default_rng(SEED + 1)
    # About 30% of escalations barely change the heart rate: a fall, a bleed on a beta-blocker, no one at home to help.
    rise = np.where(extra.random(len(eps)) < 0.30, extra.normal(3, 3, len(eps)), rise)[readings["row"]]
    ramp = np.where(escalated, rise * np.clip(1 - left / ramp_hours, 0, 1) ** 1.5, 0)
    # Fast heart rates that resolve at home: a fever spike in pneumonia or cellulitis (about 15% of those episodes),
    # and AF with a rapid rate (about 20% of AF episodes). Each is a plateau with one-hour edges at a random time.
    stay = ((eps["ended_at"] - eps["admitted_at"]) / pd.Timedelta(hours=1)).to_numpy()
    infection = eps["diagnosis"].isin(["Pneumonia", "Cellulitis"]).to_numpy()
    bouts = np.zeros(n)
    for chance, eligible, height, length in ((0.15, infection, (15, 30), (4, 12)),
                                             (0.20, eps["af"].to_numpy(), (30, 45), (2, 8))):
        has = eligible & (extra.random(len(eps)) < chance)
        size = extra.uniform(*height, len(eps))
        hours = extra.uniform(*length, len(eps))
        start = extra.uniform(0, 1, len(eps)) * np.maximum(stay - hours, 0)
        rows = readings["row"].to_numpy()
        edge = np.minimum(into - start[rows], start[rows] + hours[rows] - into)
        bouts += np.where(has[rows], size[rows] * np.clip(edge, 0, 1), 0)
    # Each hour has its own level (activity, meals, sleep) on top of beat-to-beat noise; AF scatters more.
    hour_key = readings["row"].to_numpy() * 1000 + np.floor(into).astype(int)
    _, hour_index = np.unique(hour_key, return_inverse=True)
    hour_level = rng.normal(0, 3.0, hour_index.max() + 1)[hour_index]
    noise = rng.normal(0, 2.5, n) * np.where(e["af"].to_numpy(), 2.6, 1.0)
    rate = base + circadian + recovery + ramp + bouts + hour_level + noise
    # Motion artifacts: a single implausible reading while the patient moves, mostly by day and in younger patients.
    awake = (clock >= 7) & (clock < 22)
    p_spike = np.where(awake, 0.009, 0.0015) * np.where(e["age"].to_numpy() < 70, 1.6, 1.0)
    spike = rng.random(n) < p_spike
    rate = np.where(spike, rate + rng.uniform(35, 85, n), rate)
    rate = np.clip(np.round(rate), 35, 210)
    # The patch reports no heart rate for about 1% of readings (poor skin contact).
    rate = np.where(rng.random(n) < 0.01, np.nan, rate)

    table = pd.DataFrame({
        "episode_id": e["episode_id"].to_numpy(),
        "recorded_at": pd.DatetimeIndex(readings["at"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "heart_rate": rate,
    })
    return table


def lab_times(rng, eps):
    """(row, test, collected UTC, kind) for every draw: admission panels, morning BMPs, BNP, INR, and lactates."""
    draws = []
    for ep in eps.itertuples():
        admit = ep.admitted_at
        end = ep.ended_at
        sick = ep.severity
        # The admission panel is drawn at the first visit, within a few hours of arrival.
        first = admit + pd.Timedelta(minutes=int(rng.integers(30, 240)))
        if first < end:
            draws += [(ep.Index, "BMP", first), (ep.Index, "CBC", first)]
            if ep.diagnosis == "Heart failure":
                draws.append((ep.Index, "BNP", first))
            # Lactate is drawn mostly for patients who look sick (confounding by indication).
            if rng.random() < logistic(-1.0 + 1.3 * sick):
                draws.append((ep.Index, "Lactate", first))
        # Morning visits between 08:00 and 12:00 local: a BMP on most days, INR every other day on warfarin.
        local_day = admit.tz_convert(ZONE).normalize().tz_localize(None)
        for k in range(1, 12):
            visit_local = local_day + pd.Timedelta(days=k, hours=float(rng.uniform(8, 12)))
            visit = visit_local.tz_localize(ZONE).tz_convert("UTC")
            if visit >= end:
                break
            if rng.random() < 0.55 + 0.25 * (ep.kidney or ep.heart_failure):
                draws.append((ep.Index, "BMP", visit))
            if ep.warfarin and k % 2 == 1:
                draws.append((ep.Index, "INR", visit))
            if rng.random() < logistic(-2.6 + 1.0 * sick):
                draws.append((ep.Index, "Lactate", visit))
        # A deteriorating patient gets an evening or overnight urgent visit with a lactate and a CBC.
        if ep.outcome == "escalated" and rng.random() < 0.6:
            urgent = end - pd.Timedelta(minutes=int(rng.integers(60, 420)))
            if urgent > first:
                draws += [(ep.Index, "Lactate", urgent), (ep.Index, "CBC", urgent)]
    return draws


# Median minutes from collection to result once the sample is at the lab, and the log SD.
LAB_MINUTES = {"BMP": (70, 0.45), "CBC": (50, 0.45), "BNP": (120, 0.5), "INR": (60, 0.45), "Lactate": (40, 0.5)}


def labs(rng, eps):
    draws = lab_times(rng, eps)
    row = np.array([d[0] for d in draws])
    test = np.array([d[1] for d in draws])
    collected = pd.DatetimeIndex([d[2] for d in draws]).floor("min")
    n = len(draws)
    # The courier takes about 2 to 3 hours to reach the central lab.
    courier = np.exp(rng.normal(np.log(150), 0.35, n))
    lab_median = np.array([LAB_MINUTES[t][0] for t in test])
    lab_sd = np.array([LAB_MINUTES[t][1] for t in test])
    lab = np.exp(np.log(lab_median) + rng.normal(0, 1, n) * lab_sd)
    arrived = collected + pd.to_timedelta(np.round(courier), unit="min")
    resulted = arrived + pd.to_timedelta(np.round(lab), unit="min")
    # Samples that reach the lab after 21:00 local wait for the overnight batch, posted 01:30 to 03:00.
    arrived_local = arrived.tz_convert(ZONE)
    late = (arrived_local.hour >= 21) | (arrived_local.hour < 1)
    batch_day = (arrived_local + pd.Timedelta(hours=3)).normalize().tz_localize(None)
    batch_clock = batch_day + pd.to_timedelta(np.round(rng.uniform(90, 180, n)), unit="min")
    collected_text = collected.tz_convert(ZONE).strftime("%Y-%m-%d %H:%M")
    resulted_text = resulted.tz_convert(ZONE).strftime("%Y-%m-%d %H:%M")
    # The analyzer writes its own wall clock; on the spring-forward night it still read 02:xx when the batch posted.
    batch_text = pd.DatetimeIndex(batch_clock).strftime("%Y-%m-%d %H:%M")
    resulted_text = np.where(late, batch_text, resulted_text)

    e = eps.iloc[row].reset_index(drop=True)
    sick = e["severity"].to_numpy()
    escalated = (e["outcome"] == "escalated").to_numpy()
    near_end = ((e["ended_at"] - collected).dt.total_seconds().to_numpy() / 3600 < 10) & escalated
    value = np.full(n, np.nan)
    rows = test == "Lactate"
    # Lactate in mmol/L: about 1.2 when well, above 2 in sepsis or shock.
    value[rows] = np.exp(np.log(1.2) + 0.18 * sick[rows] + 0.55 * near_end[rows] + rng.normal(0, 0.25, rows.sum()))
    rows = test == "BMP"  # reported as creatinine, mg/dL
    value[rows] = np.exp(np.log(1.0) + 0.35 * e["kidney"].to_numpy()[rows] + 0.05 * sick[rows]
                         + rng.normal(0, 0.15, rows.sum()))
    rows = test == "CBC"  # reported as white cell count, 10^9/L
    value[rows] = np.exp(np.log(8.5) + 0.12 * sick[rows] + 0.15 * (e["diagnosis"].to_numpy()[rows] == "Pneumonia")
                         + rng.normal(0, 0.25, rows.sum()))
    rows = test == "BNP"  # pg/mL
    value[rows] = np.exp(np.log(650) + 0.25 * sick[rows] + rng.normal(0, 0.6, rows.sum()))
    rows = test == "INR"
    value[rows] = np.clip(rng.normal(2.5, 0.5, rows.sum()), 1.0, 6.0)

    table = pd.DataFrame({"episode_id": e["episode_id"].to_numpy(), "test": test,
                          "collected_at": np.asarray(collected_text), "resulted_at": resulted_text,
                          "value": np.round(value, 2)})
    order = np.lexsort((test, collected.asi8, row))
    return table.iloc[order].reset_index(drop=True)


def build() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    people = patients(rng)
    eps = episodes(rng, people)
    vital_rows = vitals(rng, eps)
    lab_rows = labs(rng, eps)
    sex = np.where(eps["female"], "F", "M")
    episode_table = pd.DataFrame({
        "episode_id": eps["episode_id"], "patient_id": eps["patient_id"], "age": eps["age"], "sex": sex,
        "deprivation_decile": eps["deprivation"],
        "diagnosis": eps["diagnosis"], "admitted_at": eps["admitted_at"].astype("datetime64[us, UTC]"),
        "ended_at": eps["ended_at"].astype("datetime64[us, UTC]"), "outcome": eps["outcome"],
    })
    return {"episodes": episode_table, "vitals": vital_rows, "labs": lab_rows}


def _number(value):
    """A plain int or float for the frozen block, or None for a missing value."""
    if pd.isna(value):
        return None
    value = float(value)
    return int(value) if value.is_integer() else round(value, 4)


def answers(tables: dict[str, pd.DataFrame], hourly_stat: str = "median", day_zone: str = ZONE,
            lab_zone: str = ZONE, count_column: str = "source_row") -> dict[str, pd.DataFrame]:
    """Every graded table, worked out the way the README's steps do.

    The options give the tables a plausible slip produces, which the checks accept downstream of the check that
    charges it: hourly means for medians, UTC days for New York days, and lab clocks localized as UTC. Hours
    counted from `heart_rate` (blank heart rates skipped) are accepted outright in Task 2.1.
    """
    episodes = tables["episodes"]
    vitals = tables["vitals"].copy()
    vitals["recorded_at"] = pd.to_datetime(vitals["recorded_at"], utc=True)
    vitals = vitals.sort_values(["episode_id", "recorded_at"]).reset_index(drop=True)

    # 1.2: readings and episodes per New York calendar day.
    local = vitals.assign(recorded_at=vitals["recorded_at"].dt.tz_convert(day_zone)).set_index("recorded_at")
    daily = local.resample("D").agg(readings=("heart_rate", "size"), episodes=("episode_id", "nunique"),
                                    heart_rates=("heart_rate", "count"))
    daily = daily.reset_index().rename(columns={"recorded_at": "date"})

    # 2.1: hours with no reading, and runs of them, per episode.
    # Every reading counts, with or without a heart rate; `count_column="heart_rate"` gives the slip that skips blanks.
    counts = (vitals.assign(source_row=1).set_index("recorded_at").groupby("episode_id")[count_column]
              .resample("h").count().rename("readings").reset_index())
    counts["source_observed"] = counts["readings"] > 0
    counts["run_id"] = counts.groupby("episode_id")["source_observed"].cumsum()
    gaps = counts[~counts["source_observed"]]
    runs = gaps.groupby(["episode_id", "run_id"]).size().rename("gap_hours").reset_index()
    per_run = runs.groupby("episode_id")["gap_hours"].agg(gap_runs="count", longest_gap_h="max")
    coverage = counts.groupby("episode_id").agg(grid_hours=("readings", "size"),
                                                gap_hours=("source_observed", lambda s: (~s).sum()))
    coverage = coverage.join(per_run).fillna({"gap_runs": 0, "longest_gap_h": 0}).astype(int).reset_index()

    # 2.2: three alert rules over each whole episode.
    grouped = vitals.set_index("recorded_at").groupby("episode_id")["heart_rate"].resample("h")
    hourly = getattr(grouped, hourly_stat)().reset_index()
    hourly["rolling_3h"] = hourly.groupby("episode_id")["heart_rate"].transform(lambda s: s.rolling(3).mean())
    escalated = episodes.set_index("episode_id")["outcome"].eq("escalated")
    flags = pd.DataFrame({
        "any_reading": vitals.groupby("episode_id")["heart_rate"].max() >= ALERT_BPM,
        "hourly_median": hourly.groupby("episode_id")["heart_rate"].max() >= ALERT_BPM,
        "rolling_3h": hourly.groupby("episode_id")["rolling_3h"].max() >= ALERT_BPM,
    })
    outcome = escalated.loc[flags.index]
    rules = pd.DataFrame({"rule": flags.columns, "episodes_flagged": flags.sum().to_numpy(),
                          "escalations_flagged": flags[outcome].sum().to_numpy()})

    # 3.1: lab turnaround after putting both clocks in UTC.
    labs = tables["labs"].copy()
    for column in ("collected_at", "resulted_at"):
        labs[column] = (pd.to_datetime(labs[column], format="%Y-%m-%d %H:%M")
                        .dt.tz_localize(lab_zone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC"))
    labs["turnaround_h"] = (labs["resulted_at"] - labs["collected_at"]).dt.total_seconds() / 3600
    turnaround = labs.groupby("test").agg(orders=("test", "size"), timed_results=("resulted_at", "count"),
                                          median_turnaround_h=("turnaround_h", "median")).reset_index()

    # 3.2: past-only features at each episode's prediction time.
    scored = episodes[["episode_id", "admitted_at", "outcome"]].copy()
    scored["prediction_time"] = scored["admitted_at"].dt.floor("h") + pd.Timedelta(hours=SCORE_AFTER_H)
    features = hourly[["episode_id", "recorded_at", "heart_rate"]].copy()
    by_episode = features.groupby("episode_id")["heart_rate"]
    features["prev_hour_hr"] = by_episode.shift(1)
    features["mean_prev_6h"] = by_episode.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    features = features.merge(scored, on="episode_id", validate="many_to_one")
    features = features[features["recorded_at"] == features["prediction_time"]]
    lactate = labs[labs["test"] == "Lactate"].merge(scored[["episode_id", "prediction_time"]], on="episode_id")
    known = lactate[lactate["resulted_at"] <= lactate["prediction_time"]].sort_values("resulted_at")
    last = known.groupby("episode_id")["value"].last().rename("last_lactate").reset_index()
    features = features.merge(last, on="episode_id", how="left", validate="one_to_one")
    features = features[["episode_id", "prediction_time", "outcome", "prev_hour_hr", "mean_prev_6h", "last_lactate"]]

    # 3.3: chronological blocks of the scored episodes.
    blocks = features.copy()
    blocks["block"] = np.where(blocks["prediction_time"] < HOLDOUT_FROM, "earlier", "later_holdout")
    blocks["escalated"] = blocks["outcome"] == "escalated"
    blocks["flagged"] = blocks["mean_prev_6h"] >= FLAG_BPM
    blocks["flagged_escalation"] = blocks["flagged"] & blocks["escalated"]
    holdout = blocks.groupby("block").agg(episodes=("episode_id", "size"), escalations=("escalated", "sum"),
                                          flagged=("flagged", "sum"),
                                          flagged_escalations=("flagged_escalation", "sum")).reset_index()
    # Lactate results whose time was set aside as NaT are unknown, so the checks skip those episodes' last_lactate.
    unknown = set(labs.loc[(labs["test"] == "Lactate") & labs["resulted_at"].isna(), "episode_id"])
    return {"daily": daily, "coverage": coverage, "rules": rules, "turnaround": turnaround, "features": features,
            "blocks": blocks, "holdout": holdout, "unknown_lactate": unknown, "labs": labs}


def expected(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    """The frozen values: one dict per graded table, keyed the way the checks key it."""
    main, mean = answers(tables), answers(tables, hourly_stat="mean")
    utc_days, utc_labs = answers(tables, day_zone="UTC"), answers(tables, lab_zone="UTC")
    heart_rate_hours = answers(tables, count_column="heart_rate")["coverage"]
    features = main["features"].set_index("episode_id")
    mean_features = mean["features"].set_index("episode_id")
    utc_features = utc_labs["features"].set_index("episode_id")
    assert features.index.equals(mean_features.index) and features.index.equals(utc_features.index)
    utc_daily = utc_days["daily"].set_index(utc_days["daily"]["date"].dt.strftime("%Y-%m-%d"))
    assert list(utc_daily.index) == [day.strftime("%Y-%m-%d") for day in main["daily"]["date"]]
    blocks = main["blocks"].set_index("episode_id")
    return {
        "VITALS_ROWS": len(tables["vitals"]),
        "ALERT_BPM": ALERT_BPM,
        "FLAG_BPM": FLAG_BPM,
        # Per New York day: readings (rows), episodes, and the heart rates a count() would give instead of size.
        "DAILY_FEED": {row.date.strftime("%Y-%m-%d"): (int(row.readings), int(row.episodes), int(row.heart_rates))
                       for row in main["daily"].itertuples()},
        # The same days binned in UTC: readings and episodes, which a missing tz_convert() gives.
        "DAILY_FEED_UTC": {day: (int(row.readings), int(row.episodes)) for day, row in utc_daily.iterrows()},
        "COVERAGE": {row.episode_id: (int(row.grid_hours), int(row.gap_hours), int(row.gap_runs),
                                      int(row.longest_gap_h)) for row in main["coverage"].itertuples()},
        # The same, counting only readings with a heart rate, which the checks accept as well.
        "COVERAGE_HEART_RATE": {row.episode_id: (int(row.grid_hours), int(row.gap_hours), int(row.gap_runs),
                                                 int(row.longest_gap_h)) for row in heart_rate_hours.itertuples()},
        "ALERT_RULES": {row.rule: (int(row.episodes_flagged), int(row.escalations_flagged))
                        for row in main["rules"].itertuples()},
        "ALERT_RULES_MEAN": {row.rule: (int(row.episodes_flagged), int(row.escalations_flagged))
                             for row in mean["rules"].itertuples()},
        "TURNAROUND": {row.test: (int(row.orders), int(row.timed_results), _number(row.median_turnaround_h))
                       for row in main["turnaround"].itertuples()},
        # Per scored episode: block, escalated, prev_hour_hr and mean_prev_6h from hourly medians, the same two from
        # hourly means, last_lactate, whether a lactate result time was set aside (its last_lactate is not checked),
        # and last_lactate when the lab clocks are localized as UTC.
        "FEATURES": {
            episode: (blocks.at[episode, "block"], bool(blocks.at[episode, "escalated"]),
                      _number(features.at[episode, "prev_hour_hr"]), _number(features.at[episode, "mean_prev_6h"]),
                      _number(mean_features.at[episode, "prev_hour_hr"]),
                      _number(mean_features.at[episode, "mean_prev_6h"]),
                      _number(features.at[episode, "last_lactate"]), episode in main["unknown_lactate"],
                      _number(utc_features.at[episode, "last_lactate"]))
            for episode in features.index
        },
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
    tables = build()
    DATA.mkdir(exist_ok=True)
    for name, table in tables.items():
        path = DATA / f"{name}.parquet"
        table.to_parquet(path, index=False)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        print(f"Wrote {path.relative_to(ROOT)}: {table.shape}, {path.stat().st_size / 1e6:.1f} MB, md5 {digest}")
    write_checks(expected(tables))


if __name__ == "__main__":
    main()
