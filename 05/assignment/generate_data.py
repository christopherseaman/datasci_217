"""Write data/lab_visits.csv, the synthetic lab export for Assignment 05.

    python3 generate_data.py

One row is one clinic visit of a diabetes clinic network: patient_id, visit_date, clinic, gender, age,
weight_kg, hba1c_pct, and glucose_mg_dl. Each patient is drawn first (sex, birth date, height, BMI, home
clinic, diabetes status and control), and every visit follows from the patient and the visit before it:

- patients enroll from January 2024 to June 2026 and get IDs in enrollment order, with numbers skipped for
  patients of other services;
- the next visit is a few weeks or months later, by status and control, on a weekday drawn evenly from Monday
  to Friday, and a patient may miss a visit or leave the clinic;
- age follows the birth date; weight drifts from the patient's baseline by a small step each visit;
- HbA1c starts from the status and control, comes down toward a target under treatment, drifts slowly, rises
  with weight gain, and varies a little from visit to visit;
- fasting glucose follows HbA1c through the ADAG relation, estimated average glucose (mg/dL) = 28.7 * HbA1c - 46.7,
  times the patient's own ratio and visit noise.

Out-of-range draws are drawn again rather than clipped, so no value piles up at a limit. Then the export's
problems are planted at the rates a cleaning script has to find with code: blanks, rows sent twice, -999 for
a measurement not taken, ages typed as text, dates that do not exist, IDs and labels spelled several ways,
dropped decimal points, and values exactly on, and just past, the data contract's limits.

The seed fixes every value, so each run writes the same bytes on Python 3.9 to 3.13, macOS, Linux, and WSL:
only random.random() draws numbers, normal draws add uniforms instead of calling math functions, nothing
iterates over a set, and the file is written with "\\n" line endings. The course checks import this file
and rebuild the same rows in memory, so they need neither the data folder nor pandas.
"""

from __future__ import annotations

import csv
import datetime
import random
from pathlib import Path

SEED = 217
COLUMNS = ("patient_id", "visit_date", "clinic", "gender", "age", "weight_kg", "hba1c_pct", "glucose_mg_dl")
OUTPUT = Path(__file__).resolve().parent / "data" / "lab_visits.csv"
PATIENTS = 16500
FIRST_ID = 10001
# Patients enroll from 2021 on; the export holds the visits from FIRST_DAY to LAST_DAY of those still in care.
FIRST_MONDAY = datetime.date(2021, 1, 4)
FIRST_DAY = datetime.date(2024, 1, 2)
LAST_DAY = datetime.date(2026, 6, 30)
# (clinic, share of patients, added chance of diabetes): Bayview's neighborhood has the most diabetes.
CLINICS = (("Mission", 0.35, 0.0), ("Sunset", 0.28, -0.05), ("Excelsior", 0.22, 0.04), ("Bayview", 0.15, 0.08))
GENDERS = (("F", 0.52), ("M", 0.46), ("X", 0.02))
# How the export spells each code; a later visit may spell it another way.
SPELLINGS = {
    "F": (("F", 0.55), ("f", 0.15), ("female", 0.15), ("Female", 0.10), (" F", 0.05)),
    "M": (("M", 0.55), ("m", 0.15), ("male", 0.15), ("Male", 0.10), ("MALE", 0.05)),
    "X": (("X", 0.40), ("x", 0.10), ("nb", 0.20), ("NB", 0.10), ("nonbinary", 0.20)),
}
UNMAPPED_GENDERS = ("U", "unknown", "?")
# Weeks to the next visit, (shortest, spread), by status and control.
CADENCE = {"normal": (24, 28), "prediabetes": (22, 8), "good": (11, 6), "moderate": (10, 5), "poor": (7, 4)}
LEAVE = 0.07  # the chance a patient leaves the clinic after a visit
MISS = 0.10  # the chance a patient misses the next visit and comes 4 to 12 weeks later

# Sample patients, by place in enrollment order, carry only the problems listed for them in PLANTED.
# (patient, visit number from 1, column, cell): "duplicate" sends the visit twice, "no labs" blanks both labs,
# "slip" drops the decimal point, "impossible" writes a date that does not exist, and "lower", "trailing",
# and "leading" misspell the ID; any other cell replaces the value.
PLANTED = (
    (7, 2, "patient_id", "lower"), (7, 3, "clinic", "mission "),
    (112, 1, "patient_id", "trailing"), (112, 1, "gender", "female"), (112, 2, "clinic", ""),
    (112, 3, "gender", "Female"),
    (245, 1, "gender", "nb"), (245, 3, "gender", "nonbinary"),
    (391, 1, "clinic", "SUNSET"), (391, 2, "gender", "U"),
    (533, 1, "duplicate", ""), (533, 3, "hba1c_pct", "slip"),
    (678, 1, "weight_kg", "-999"), (678, 2, "glucose_mg_dl", "-999"),
    (804, 2, "visit_date", "impossible"),
    # Two undated visits of one patient, which drop_duplicates(subset=["patient_id", "visit_date"]) treats as one.
    (950, 3, "visit_date", ""), (950, 5, "visit_date", ""),
    (1126, 2, "clinic", " Excelsior"), (1126, 3, "gender", "Male"), (1126, 4, "patient_id", "leading"),
    (1307, 1, "age", "unknown"), (1307, 2, "hba1c_pct", ""),
    (1489, 2, "weight_kg", "slip"), (1489, 3, "clinic", "bayview"),
    (1652, 2, "no labs", ""),
    # Values exactly on, and just past, the contract's ranges.
    (2001, 1, "hba1c_pct", "4.0"), (2002, 1, "hba1c_pct", "20.0"),
    (2003, 1, "hba1c_pct", "3.9"), (2004, 1, "hba1c_pct", "20.1"),
    (2005, 1, "weight_kg", "30.0"), (2006, 1, "weight_kg", "250.0"),
    (2007, 1, "weight_kg", "29.9"), (2008, 1, "weight_kg", "250.1"),
)
# A sample patient's gender agrees with the spelling planted in its visits.
PLANTED_GENDERS = {112: "F", 245: "X", 1126: "M"}


def pick(rng: random.Random, weighted) -> object:
    """One choice from (value, weight, ...) tuples whose weights sum to 1."""
    draw, total = rng.random(), 0.0
    for option in weighted:
        total += option[1]
        if draw < total:
            return option[0]
    return weighted[-1][0]


def normal(rng: random.Random, mean: float, sd: float) -> float:
    """A near-normal draw: six uniforms sum to mean 3 and variance 1/2, so scale their sum by the square root of 2."""
    return mean + sd * 1.4142135623730951 * (sum(rng.random() for _ in range(6)) - 3.0)


def within(rng: random.Random, mean: float, sd: float, low: float, high: float) -> float:
    """A normal draw from low to high: a draw outside is drawn again, so nothing piles up at a limit."""
    while True:
        value = normal(rng, mean, sd)
        if low <= value <= high:
            return value


def weekday_after(day: datetime.date, weeks: int, rng: random.Random) -> datetime.date:
    """A day about `weeks` weeks after `day`, Monday to Friday drawn evenly."""
    monday = day - datetime.timedelta(days=day.weekday())
    return monday + datetime.timedelta(days=7 * weeks + int(rng.random() * 5))


def age_on(birth: datetime.date, day: datetime.date) -> int:
    return day.year - birth.year - ((day.month, day.day) < (birth.month, birth.day))


def enrollment_days(rng: random.Random) -> list[datetime.date]:
    """Every patient's first visit, earliest first: a week and a weekday drawn evenly from 2021 to LAST_DAY."""
    weeks = (LAST_DAY - FIRST_MONDAY).days // 7 + 1
    days = []
    while len(days) < PATIENTS:
        day = FIRST_MONDAY + datetime.timedelta(days=7 * int(rng.random() * weeks) + int(rng.random() * 5))
        if day <= LAST_DAY:
            days.append(day)
    return sorted(days)


def patient_visits(rng: random.Random, index: int, patient_id: str, first: datetime.date, sample: bool) -> list[dict]:
    """The patient's visits from FIRST_DAY on as clean cells, with the labs not drawn left blank.

    Visits before FIRST_DAY still move age, weight, and HbA1c along. A sample patient never leaves."""
    sex = PLANTED_GENDERS.get(index, pick(rng, GENDERS))
    clinic = pick(rng, CLINICS)
    clinic_risk = next(risk for name, _, risk in CLINICS if name == clinic)
    first_age = within(rng, 57.0, 13.0, 21.0, 90.0)
    birth = first - datetime.timedelta(days=int(first_age * 365.25))
    height = within(rng, {"F": 1.62, "M": 1.76, "X": 1.69}[sex], 0.07, 1.47, 2.02)
    bmi = within(rng, 30.5 if sex == "F" else 29.5, 6.0, 18.0, 60.0)
    weight = bmi * height * height
    # Older and heavier patients more often have diabetes; younger ones with diabetes more often control it poorly.
    diabetes = min(max(0.42 + 0.006 * (first_age - 57.0) + 0.02 * (bmi - 30.0) + clinic_risk, 0.05), 0.9)
    draw = rng.random()
    if draw < diabetes:
        poor = min(max(0.22 + 0.006 * (50.0 - first_age), 0.08), 0.45)
        control = "poor" if rng.random() < poor else "good" if rng.random() < 0.55 else "moderate"
        status = "diabetes"
    else:
        control = status = "prediabetes" if draw < diabetes + 0.6 * (1 - diabetes) else "normal"
    if control == "poor":
        start = within(rng, 9.8, 1.4, 8.0, 15.5)
    elif control == "moderate":
        start = within(rng, 7.8, 0.4, 7.0, 9.0)
    elif control == "good":
        start = within(rng, 6.8, 0.35, 6.0, 7.5)
    elif control == "prediabetes":
        start = within(rng, 6.0, 0.25, 5.6, 6.6)
    else:
        start = within(rng, 5.2, 0.22, 4.5, 5.7)
    # Treatment brings a raised HbA1c toward the patient's target; the rest drifts slowly.
    target = min(start, within(rng, 7.2, 0.5, 6.2, 8.5)) if status == "diabetes" else start
    drift = normal(rng, 0.08, 0.2) if status == "diabetes" else normal(rng, 0.04, 0.08)
    per_kg = 0.04 if status == "diabetes" else 0.01
    weight_trend = normal(rng, -1.0 if status == "diabetes" else 0.3, 2.0)  # kg per year
    glucose_ratio = within(rng, 1.0, 0.06, 0.85, 1.15)  # some patients run higher fasting glucose than others
    shortest, spread = CADENCE[control]

    visits = []
    day, previous, base_weight = first, first, weight
    while day <= LAST_DAY:
        years = (day - first).days / 365.25
        weight = max(weight + normal(rng, weight_trend * (day - previous).days / 365.25, 0.8), 0.85 * base_weight)
        a1c = target + (start - target) / (1.0 + 2.0 * years) + drift * years + per_kg * (weight - base_weight)
        visit_a1c = within(rng, a1c, 0.15, 4.1, 19.0)
        eag = 28.7 * visit_a1c - 46.7
        glucose = within(rng, eag * glucose_ratio, 0.11 * eag, 55.0, 600.0)
        visit = {
            "patient_id": patient_id, "visit_date": day.isoformat(),
            "clinic": clinic if rng.random() < 0.95 else pick(rng, CLINICS),
            "gender": pick(rng, SPELLINGS[sex]), "age": str(age_on(birth, day)),
            "weight_kg": f"{weight + normal(rng, 0.0, 0.3):.1f}",
            "hba1c_pct": f"{visit_a1c:.1f}", "glucose_mg_dl": str(int(glucose + 0.5)),
        }
        if not sample:
            lab_draws(rng, visit, status == "diabetes")
        if day >= FIRST_DAY:
            visits.append(visit)
        if not sample and rng.random() < LEAVE:
            break
        weeks = shortest + int(rng.random() * spread)
        if rng.random() < MISS:
            weeks += 4 + int(rng.random() * 9)
        previous, day = day, weekday_after(day, weeks, rng)
    return visits


def impossible_date(rng: random.Random, cell: str) -> str:
    """A date in the same month that does not exist, such as 2025-02-30, 2025-04-31, or month 13."""
    year, month = cell[:4], int(cell[5:7])
    if month == 2:
        return f"{year}-02-{29 + int(rng.random() * 2) + (0 if int(year) % 4 else 1)}"
    if month in (4, 6, 9, 11):
        return f"{year}-{month:02d}-31"
    return f"{year}-13-{cell[8:10]}"


def lab_draws(rng: random.Random, visit: dict, diabetes: bool) -> None:
    """Blank the labs not drawn: HbA1c is skipped more often for a patient without diabetes."""
    if rng.random() < (0.05 if diabetes else 0.09):
        visit["hba1c_pct"] = ""
    if rng.random() < 0.05:
        visit["glucose_mg_dl"] = ""  # not fasting


def plant_problems(rng: random.Random, visit: dict) -> bool:
    """Plant the export's data problems in one visit at their rates; True when the export sends it twice."""
    roll = rng.random
    pid = visit["patient_id"]
    draw = roll()
    if draw < 0.006:
        visit["patient_id"] = pid.lower()
    elif draw < 0.010:
        visit["patient_id"] = pid + " "
    elif draw < 0.012:
        visit["patient_id"] = " " + pid
    elif draw < 0.0135:
        visit["patient_id"] = pid[:-1]  # a digit left off
    elif draw < 0.0140:
        visit["patient_id"] = pid + pid[-1]  # a digit typed twice
    draw = roll()
    if draw < 0.003:
        visit["visit_date"] = ""
    elif draw < 0.0045:
        visit["visit_date"] = impossible_date(rng, visit["visit_date"])
    draw = roll()
    if draw < 0.004:
        visit["clinic"] = ""
    elif draw < 0.054:
        visit["clinic"] = visit["clinic"].lower()
    elif draw < 0.084:
        visit["clinic"] = visit["clinic"].upper()
    elif draw < 0.104:
        visit["clinic"] = visit["clinic"] + " "
    elif draw < 0.124:
        visit["clinic"] = " " + visit["clinic"]
    draw = roll()
    if draw < 0.004:
        visit["gender"] = UNMAPPED_GENDERS[int(roll() * len(UNMAPPED_GENDERS))]
    elif draw < 0.006:
        visit["gender"] = ""
    draw = roll()
    if draw < 0.0025:
        visit["age"] = "unknown"
    elif draw < 0.004:
        visit["age"] = ""
    draw = roll()
    if draw < 0.006:
        visit["weight_kg"] = ""
    elif draw < 0.016:
        visit["weight_kg"] = "-999"
    elif draw < 0.017:
        visit["weight_kg"] = visit["weight_kg"].replace(".", "")  # the decimal point left out: 95.0 becomes 950
    if visit["hba1c_pct"] and roll() < 0.001:
        visit["hba1c_pct"] = visit["hba1c_pct"].replace(".", "")
    if visit["glucose_mg_dl"] and roll() < 0.008:
        visit["glucose_mg_dl"] = "-999"
    if roll() < 0.012:
        visit["hba1c_pct"] = visit["glucose_mg_dl"] = ""  # a telehealth visit: no labs drawn
    return roll() < 0.008


def plant(visit: dict, column: str, cell: str, rng: random.Random) -> None:
    """Apply one PLANTED problem to a sample patient's visit."""
    if column == "no labs":
        visit["hba1c_pct"] = visit["glucose_mg_dl"] = ""
    elif cell == "slip":
        visit[column] = visit[column].replace(".", "")
    elif cell == "impossible":
        visit[column] = impossible_date(rng, visit[column])
    elif cell in ("lower", "trailing", "leading"):
        pid = visit[column]
        visit[column] = pid.lower() if cell == "lower" else pid + " " if cell == "trailing" else " " + pid
    else:
        visit[column] = cell


def generate_rows() -> list[tuple[str, ...]]:
    """Every row of data/lab_visits.csv in file order, as the tuples of cells csv.writer writes."""
    rng = random.Random(SEED)
    planted: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for patient, number, column, cell in PLANTED:
        planted.setdefault((patient, number), []).append((column, cell))
    sample_patients = {patient: True for patient, _, _, _ in PLANTED}
    keyed = []
    number = FIRST_ID
    for index, first in enumerate(enrollment_days(rng)):
        # IDs follow enrollment, skipping the numbers given to patients of the network's other services.
        if index:
            number += 1 + int(rng.random() * rng.random() * 6)
        visits = patient_visits(rng, index, f"D{number}", first, index in sample_patients)
        for visit_number, visit in enumerate(visits, start=1):
            day = visit["visit_date"]
            twice = False
            if index in sample_patients:
                for column, cell in planted.pop((index, visit_number), []):
                    if column == "duplicate":
                        twice = True
                    else:
                        plant(visit, column, cell, rng)
            else:
                twice = plant_problems(rng, visit)
            row = tuple(visit[column] for column in COLUMNS)
            keyed.append((day, number, row))
            if twice:
                keyed.append((day, number, row))
    if planted:
        raise ValueError(f"PLANTED names visits the sample patients do not have: {sorted(planted)}")
    # The export lists visits by their real day, then by patient; a row sent twice follows itself.
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in keyed]


def main() -> None:
    rows = generate_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"wrote data/{OUTPUT.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
