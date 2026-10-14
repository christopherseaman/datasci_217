"""Write data/lab_visits.csv, the synthetic lab export for Assignment 05.

    python3 generate_data.py

One row is one clinic visit of a diabetes clinic network: patient_id, visit_date, clinic, gender, age,
weight_kg, hba1c_pct, and glucose_mg_dl. Patients come back every few months, so most have several visits.
Like a real export, the file has blanks, rows sent twice, -999 for a measurement not taken, ages typed as
text, dates that do not exist, IDs and labels spelled several ways, and decimal slips, at rates a cleaning
script has to find with code.

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
PATIENTS = 18000
FIRST_ID = 10001
FIRST_DAY = datetime.date(2024, 1, 2)
LAST_FIRST_VISIT = datetime.date(2025, 9, 30)
LAST_DAY = datetime.date(2026, 6, 30)
CLINICS = (("Mission", 0.35), ("Sunset", 0.28), ("Excelsior", 0.22), ("Bayview", 0.15))
GENDERS = (("F", 0.51), ("M", 0.46), ("X", 0.03))
# How the export spells each code; a later visit may spell it another way.
SPELLINGS = {
    "F": (("F", 0.55), ("f", 0.15), ("female", 0.15), ("Female", 0.10), (" F", 0.05)),
    "M": (("M", 0.55), ("m", 0.15), ("male", 0.15), ("Male", 0.10), ("MALE", 0.05)),
    "X": (("X", 0.40), ("x", 0.10), ("nb", 0.20), ("NB", 0.10), ("nonbinary", 0.20)),
}
UNMAPPED_GENDERS = ("U", "unknown", "?")
VISITS = ((1, 0.15), (2, 0.20), (3, 0.20), (4, 0.18), (5, 0.14), (6, 0.08), (7, 0.05))

# Patients that carry one planted problem of each kind, listed in PLANTED.
SAMPLE_IDS = ("D10007", "D10112", "D10245", "D10391", "D10533", "D10678",
              "D10804", "D10950", "D11126", "D11307", "D11489", "D11652")
# (patient, visit number from 1, column, raw cell); "duplicate" sends the visit twice and "no labs" blanks both labs.
PLANTED = (
    ("D10007", 1, "hba1c_pct", "5.6"), ("D10007", 2, "patient_id", "d10007"),
    ("D10007", 3, "clinic", "mission "), ("D10007", 4, "hba1c_pct", "5.7"),
    ("D10112", 1, "patient_id", "D10112 "), ("D10112", 1, "gender", "female"),
    ("D10112", 2, "clinic", ""), ("D10112", 3, "gender", "Female"),
    ("D10245", 1, "gender", "nb"), ("D10245", 2, "hba1c_pct", "6.4"),
    ("D10245", 3, "gender", "nonbinary"), ("D10245", 3, "hba1c_pct", "6.5"),
    ("D10391", 1, "clinic", "SUNSET"), ("D10391", 2, "gender", "U"),
    ("D10533", 1, "duplicate", ""), ("D10533", 3, "hba1c_pct", "64"),
    ("D10678", 1, "weight_kg", "-999"), ("D10678", 2, "glucose_mg_dl", "-999"),
    ("D10804", 2, "visit_date", "2025-02-30"),
    ("D10950", 3, "visit_date", ""),
    ("D11126", 2, "clinic", " Excelsior"), ("D11126", 3, "gender", "Male"), ("D11126", 4, "patient_id", " D11126"),
    ("D11307", 1, "age", "unknown"), ("D11307", 2, "hba1c_pct", ""),
    ("D11489", 2, "weight_kg", "950"), ("D11489", 3, "clinic", "bayview"),
    ("D11652", 1, "glucose_mg_dl", "412"), ("D11652", 2, "no labs", ""),
    # Values exactly on, and just past, the contract's ranges.
    ("D12001", 1, "hba1c_pct", "4.0"), ("D12002", 1, "hba1c_pct", "20.0"),
    ("D12003", 1, "hba1c_pct", "3.9"), ("D12004", 1, "hba1c_pct", "20.1"),
    ("D12005", 1, "weight_kg", "30.0"), ("D12006", 1, "weight_kg", "250.0"),
    ("D12007", 1, "weight_kg", "29.9"), ("D12008", 1, "weight_kg", "250.1"),
)
# The sample patients have at least this many visits, so every planted visit exists.
SAMPLE_VISITS = {"D10007": 4, "D10112": 3, "D10245": 5, "D10391": 3, "D10533": 4, "D10678": 2,
                 "D10804": 3, "D10950": 3, "D11126": 4, "D11307": 2, "D11489": 3, "D11652": 2}

# A sample patient's gender agrees with the spelling planted in its visits.
PLANTED_GENDERS = {"D10112": "F", "D10245": "X", "D11126": "M"}


def pick(rng: random.Random, weighted) -> object:
    """One choice from (value, weight) pairs whose weights sum to 1."""
    draw, total = rng.random(), 0.0
    for value, weight in weighted:
        total += weight
        if draw < total:
            return value
    return weighted[-1][0]


def normal(rng: random.Random, mean: float, sd: float) -> float:
    """A near-normal draw: the sum of 12 uniforms minus 6 has mean 0 and standard deviation 1."""
    return mean + sd * (sum(rng.random() for _ in range(12)) - 6.0)


def weekday(day: datetime.date) -> datetime.date:
    """The day itself, or the Monday after when it falls on a weekend: the clinics see patients Monday to Friday."""
    return day + datetime.timedelta(days=(7 - day.weekday()) % 7 if day.weekday() >= 5 else 0)


def clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def patient_visits(rng: random.Random, number: int) -> list[dict[str, str]]:
    """Every visit of one patient as clean cells, before any data problem is planted."""
    patient_id = f"D{number}"
    gender = pick(rng, GENDERS)
    gender = PLANTED_GENDERS.get(patient_id, gender)
    home = pick(rng, CLINICS)
    first_age = 25 + int(rng.random() * 61)
    weight = clip(normal(rng, 86.0, 17.0), 45.0, 180.0)
    group = rng.random()
    if group < 0.25:
        a1c = 4.8 + rng.random() * 0.8
    elif group < 0.55:
        a1c = 5.7 + rng.random() * 0.7
    else:
        a1c = 6.5 + abs(normal(rng, 0.0, 1.6))
    count = max(pick(rng, VISITS), SAMPLE_VISITS.get(patient_id, 0))
    # A sample patient starts in the first half of 2024, so all of its planted visits fit before LAST_DAY.
    span = 180 if patient_id in SAMPLE_VISITS else (LAST_FIRST_VISIT - FIRST_DAY).days
    first = FIRST_DAY + datetime.timedelta(days=int(rng.random() * span))
    day = weekday(first)
    visits = []
    for _ in range(count):
        if day > LAST_DAY:
            break
        clinic = home if rng.random() < 0.95 else pick(rng, CLINICS)
        age = first_age + (day - first).days // 365
        weight = clip(weight + normal(rng, 0.0, 1.5), 40.0, 200.0)
        visit_a1c = clip(a1c + normal(rng, 0.0, 0.25), 4.2, 16.0)
        # Fasting glucose follows HbA1c: the estimated average glucose is 28.7 * HbA1c - 46.7 mg/dL. Its spread
        # is 10% of that level, so the lowest values taper off near 50 mg/dL with no floor to pile up against.
        level = 0.9 * (28.7 * visit_a1c - 46.7)
        glucose = normal(rng, level, 0.10 * level)
        visits.append({
            "patient_id": patient_id, "visit_date": day.isoformat(), "clinic": clinic,
            "gender": pick(rng, SPELLINGS[gender]), "age": str(age), "weight_kg": f"{weight:.1f}",
            "hba1c_pct": f"{visit_a1c:.1f}", "glucose_mg_dl": str(int(glucose + 0.5)),
        })
        day = weekday(day + datetime.timedelta(days=70 + int(rng.random() * 51)))
    return visits


def impossible_date(rng: random.Random, cell: str) -> str:
    """A date in the same month that does not exist, such as 2025-02-30, 2025-04-31, or month 13."""
    year, month = cell[:4], int(cell[5:7])
    if month == 2:
        return f"{year}-02-{29 + int(rng.random() * 2) + (0 if int(year) % 4 else 1)}"
    if month in (4, 6, 9, 11):
        return f"{year}-{month:02d}-31"
    return f"{year}-13-{cell[8:10]}"


def plant_problems(rng: random.Random, visit: dict[str, str]) -> bool:
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
    draw = roll()
    if draw < 0.06:
        visit["hba1c_pct"] = ""
    elif draw < 0.061:
        visit["hba1c_pct"] = visit["hba1c_pct"].replace(".", "")
    draw = roll()
    if draw < 0.05:
        visit["glucose_mg_dl"] = ""
    elif draw < 0.058:
        visit["glucose_mg_dl"] = "-999"
    if roll() < 0.012:
        visit["hba1c_pct"] = visit["glucose_mg_dl"] = ""  # a no-show: no labs drawn
    return roll() < 0.008


def generate_rows() -> list[tuple[str, ...]]:
    """Every row of data/lab_visits.csv in file order, as the tuples of cells csv.writer writes."""
    rng = random.Random(SEED)
    planted: dict[tuple[str, int], list[tuple[str, str]]] = {}
    for patient, number, column, cell in PLANTED:
        planted.setdefault((patient, number), []).append((column, cell))
    planted_patients = {patient: True for patient, _, _, _ in PLANTED}
    keyed = []
    for number in range(FIRST_ID, FIRST_ID + PATIENTS):
        for visit_number, visit in enumerate(patient_visits(rng, number), start=1):
            day = visit["visit_date"]
            twice = False
            # The sample and boundary patients carry only their planted problems.
            if visit["patient_id"] not in planted_patients:
                twice = plant_problems(rng, visit)
            for column, cell in planted.get((visit["patient_id"], visit_number), []):
                if column == "duplicate":
                    twice = True
                elif column == "no labs":
                    visit["hba1c_pct"] = visit["glucose_mg_dl"] = ""
                else:
                    visit[column] = cell
            row = tuple(visit[column] for column in COLUMNS)
            keyed.append((day, number, row))
            if twice:
                keyed.append((day, number, row))
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
