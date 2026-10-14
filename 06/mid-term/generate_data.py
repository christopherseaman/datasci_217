"""Write the five synthetic clinic tables for Assignment 06 into data/.

    python3 generate_data.py

The rows come from the standard library, seeded, so every run on every computer
builds the same tables. The registry and the A1c results are saved as Parquet
with pandas and pyarrow (Lecture 04); the site list and the two visit exports
stay CSV, as clinic systems export them. The notebook's first cell checks each
table against data_manifest.json by the SHA-256 of its CSV text, which does not
depend on the pyarrow version. The data is course-authored and synthetic: no
real patients. Age and body size influence baseline SBP and A1c; follow-up
measurements depend on baseline and weight change. Rechecks share the first
reading with small measurement noise. Out-of-range draws are rejected, not
clipped. Only random.random() supplies randomness, and CSV uses LF endings.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
import io
from pathlib import Path
import random

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 2026_06

PATIENTS = 30_000
VISIT_ONLY_IDS = range(30_001, 30_041)
LAB_ONLY_IDS = range(90_001, 90_031)

# Parquet keeps every registry column as text, so the messy spellings and the 9999 sentinel stay as
# written; the A1c results are numbers, with an empty cell stored as missing.
PARQUET_FILES = ("patients.parquet", "a1c_wide.parquet")
A1C_COLUMNS = ("a1c_baseline", "a1c_week12")

# Each spelling as staff typed it, the code it stands for, and its weight in the draw.
GENDER_SPELLINGS = (
    ("F", "F", 340), ("M", "M", 330), ("X", "X", 12),
    ("Female ", "F", 18), ("female", "F", 30), ("f", "F", 20), ("woman", "F", 12), ("FEMALE", "F", 8),
    (" F", "F", 6), ("Woman", "F", 4),
    ("male", "M", 30), ("Male", "M", 18), ("m ", "M", 16), ("man", "M", 10), ("MALE", "M", 8), (" M", "M", 6),
    ("nonbinary", "X", 3), ("non-binary", "X", 3), ("NB", "X", 3), ("x", "X", 2), ("Non-Binary", "X", 1),
    ("", None, 12), (" ", None, 3), ("unknown", None, 8), ("Unknown", None, 3),
)
SITE_SPELLINGS = {"N01": ("N-01", "n01", "N 01", " N01"), "S02": (" S 02", "s02", "S-02", "S02 "),
                  "E04": ("e04", "E-04")}
VISIT_SPELLINGS = {"baseline": ("BL", "Baseline", "baseline ", "BASELINE", "bl"),
                   "week12": ("Week 12", "wk12", " week12", "WEEK12", "week 12", "Wk12")}
SITES = (("N01", "North Clinic", "current"), ("N01", "North Street Clinic", "retired"),
         ("S02", "Southgate Clinic", "retired"), ("S02", "South Valley Clinic", "current"),
         ("W03", "West Clinic", "current"))
IMPOSSIBLE_DATES = ("2026-02-29", "2026-02-30", "2026-02-31", "2026-04-31", "2026-06-31", "2026-09-31")
OUT_OF_RANGE_SBP = ("0", "7", "18", "45", "69", "251", "260", "300", "1450")


def normal(rng, mean, sd):
    """Near-normal noise using only stable random() draws, as in Assignment 05."""
    return mean + sd * 1.4142135623730951 * (sum(rng.random() for _ in range(6)) - 3)


def within(rng, mean, sd, low, high):
    """Reject out-of-range draws rather than piling values onto a limit."""
    while True:
        value = normal(rng, mean, sd)
        if low <= value <= high:
            return value


def choice(rng, values):
    return values[int(rng.random() * len(values))]


def sample(rng, values, count):
    return [value for _, value in sorted((rng.random(), value) for value in values)[:count]]


def patient(rng, gender):
    age = within(rng, 59, 12, 30, 86)
    height = within(rng, {"F": 1.63, "M": 1.76, "X": 1.69}[gender], .07, 1.45, 2.02)
    bmi = within(rng, 31 + .035 * (age - 59), 5, 19, 52)
    weight = bmi * height * height
    sbp = within(rng, 143 + .45 * (age - 59) + .8 * (bmi - 31), 12, 95, 215)
    a1c = within(rng, 8.1 + .065 * (bmi - 31) - .018 * (age - 59), 1.1, 5.8, 14)
    change = within(rng, -1.5 - .06 * (bmi - 31), 1.4, -7, 4)
    follow_sbp = within(rng, sbp - .28 * (sbp - 125) + .6 * change, 5, 85, 215)
    follow_a1c = within(rng, a1c - .24 * (a1c - 6.4) + .07 * change, .25, 5.1, 14)
    return age, weight, sbp, a1c, change, follow_sbp, follow_a1c


def appointment(rng):
    # Morning and afternoon sessions, with a lunch gap; five-minute slots.
    center = 10 * 60 if rng.random() < .6 else 15 * 60
    low, high = (8 * 60, 12 * 60 - 5) if center == 600 else (13 * 60, 17 * 60 - 5)
    return 5 * int(within(rng, center, 50, low, high) / 5)


def clinic_days(first: date, count: int) -> list[date]:
    """The clinic days (Monday to Friday) among `count` days from `first`."""
    return [day for day in (first + timedelta(days=n) for n in range(count)) if day.weekday() < 5]


def clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def build(rng: random.Random) -> dict[str, list[list[str]]]:
    """Every file's rows, header first, as text cells."""
    weights = [weight for _, _, weight in GENDER_SPELLINGS]
    patients, homes, traits = [], {}, {}
    for number in range(1, PATIENTS + 1):
        pid = f"P{number:05d}"
        draw = rng.random()
        site = "N01" if draw < .46 else "S02" if draw < .98 else "E04"
        homes[pid] = site
        written = choice(rng, SITE_SPELLINGS[site]) if rng.random() < 0.06 else site
        draw, total = rng.random() * sum(weights), 0
        for spelling, gender, weight in GENDER_SPELLINGS:
            total += weight
            if draw < total:
                break
        trait_gender = gender or choice(rng, ("F", "M"))
        traits[pid] = patient(rng, "F" if number == 1 else trait_gender)
        year = "9999" if rng.random() < 0.012 else str(2026 - int(traits[pid][0]))
        patients.append([pid, spelling, year, written])
    # Stable teaching examples: the extracts exercise the same rules after a rebuild.
    patients[0][1] = "woman"
    patients[3][2] = "9999"
    patients[87][3] = "N-01"
    homes["P00088"] = "N01"
    patients[259][3] = homes["P00260"] = "E04"
    # A registry re-export repeated some rows exactly, later in the file.
    for original in [0, *sample(rng, range(1, PATIENTS), 44)]:
        patients.append(list(patients[original]))

    def site_file(pid: str) -> str:
        return "north" if homes.get(pid, "S02") == "N01" else "south"

    with_visits = [f"P{n:05d}" for n in range(1, PATIENTS + 1) if rng.random() >= 0.015 or n < 1000]
    with_visits.remove("P00631")
    with_visits += [f"P{n:05d}" for n in VISIT_ONLY_IDS]
    for number in VISIT_ONLY_IDS:
        homes[f"P{number:05d}"] = choice(rng, ("N01", "S02"))
        traits[f"P{number:05d}"] = patient(rng, choice(rng, ("F", "M")))
    enrollment_days = clinic_days(date(2026, 1, 5), 81)
    files: dict[str, list[list[str]]] = {"north": [], "south": []}
    for pid in with_visits:
        baseline = choice(rng, enrollment_days)
        _, weight, sbp, _, change, follow_sbp, _ = traits[pid]
        # The last cell, the visit as coded, is dropped before writing.
        rows = [[pid, "baseline", baseline, appointment(rng), str(round(sbp)), f"{weight:.1f}",
                 "baseline"]]
        if rng.random() >= 0.07:
            follow = baseline + timedelta(days=round(within(rng, 84, 6, 63, 112)))
            while follow.weekday() >= 5:
                follow += timedelta(days=1)
            rows.append([pid, "week12", follow, appointment(rng),
                         str(round(follow_sbp)), f"{weight + change:.1f}",
                         "week12"])
        files[site_file(pid)].extend(rows)

    for rows in files.values():
        for row in rows:
            draw = rng.random()
            if draw < 0.006:
                row[4] = "999"
            elif draw < 0.008:
                row[4] = choice(rng, OUT_OF_RANGE_SBP)
            if rng.random() < 0.015:
                row[5] = ""
            if rng.random() < 0.04:
                row[1] = choice(rng, VISIT_SPELLINGS[row[6]])
        # A second reading 10 to 30 minutes later, listed before or after the first.
        candidates = [i for i, row in enumerate(rows) if 90 <= int(row[4]) <= 215]
        planted = [i for i in candidates if rows[i][0] in ("P00001", "P00004", "P00088") and rows[i][6] == "baseline"]
        for index in sorted(planted + sample(rng, [i for i in candidates if i not in planted], 60 - len(planted)), reverse=True):
            first = rows[index]
            label = choice(rng, (first[6], first[6], choice(rng, VISIT_SPELLINGS[first[6]])))
            repeat = round(within(rng, int(first[4]) - 2, 4, 85, 215))
            if repeat == int(first[4]):
                repeat -= 1
            recheck = [first[0], label, first[2], first[3] + choice(rng, (10, 15, 20, 25, 30)), str(repeat),
                       first[5], first[6]]
            rows.insert(index + choice(rng, (0, 1)), recheck)
        for row in rows:
            day = row[2]
            draw = rng.random()
            if draw < 0.0015:
                row[2] = choice(rng, IMPOSSIBLE_DATES)
            elif draw < 0.003:
                row[2] = choice(rng, (f"{day.year}-{day.month}-{day.day:02d}",
                                     f"{day.year}-{day.month:02d}-{day.day}",
                                     f"{day.year}/{day.month:02d}/{day.day:02d}"))
            else:
                row[2] = day.isoformat()
            row[3] = clock(row[3])
            if row[6] == "baseline":
                if row[0] in ("P00260", "P00446", "P00487", "P00851"):
                    row[4] = {"P00260": "70", "P00446": "250", "P00487": "999", "P00851": "69"}[row[0]]
                if row[0] == "P00260":
                    row[2] = "2026-02-30"
                if row[0] == "P00446":
                    row[2], row[5] = "2026/02/02", ""
                if row[0] == "P00487":
                    row[1] = "BL"
            del row[6]
        # An export run twice repeated some rows exactly.
        for original in sample(rng, range(len(rows)), 35):
            rows.append(list(rows[original]))

    a1c = []
    lab_ids = [f"P{n:05d}" for n in range(1, PATIENTS + 1) if rng.random() >= 0.02]
    lab_ids += [f"P{n:05d}" for n in VISIT_ONLY_IDS if rng.random() < 0.5]
    lab_ids += [f"P{n:05d}" for n in LAB_ONLY_IDS]
    for pid in lab_ids:
        # Labs use the same patient and weight trajectory as the visit exports.
        if pid not in traits:
            traits[pid] = patient(rng, choice(rng, ("F", "M")))
        _, _, _, baseline, _, _, week12 = traits[pid]
        a1c.append([pid, "" if rng.random() < 0.01 else f"{baseline:.1f}",
                    "" if rng.random() < 0.06 or pid == "P00001" else f"{week12:.1f}"])

    header = ["patient_id", "visit", "visit_date", "visit_time", "sbp_mmhg"]
    return {
        "patients.parquet": [["patient_id", "gender", "birth_year", "site_code"], *patients],
        "sites.csv": [["site_code", "site_name", "record_status"], *map(list, SITES)],
        "visits_north.csv": [[*header, "weight_kg"], *files["north"]],
        "visits_south.csv": [[*header, "wt_kg"], *files["south"]],
        "a1c_wide.parquet": [["patient_id", "a1c_baseline", "a1c_week12"], *a1c],
    }


def to_csv(rows: list[list[str]]) -> bytes:
    """CSV bytes with `\\n` line endings on every operating system."""
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue().encode("utf-8")


def files() -> dict[str, bytes]:
    """Each data file's name and its table as CSV text, the bytes data_manifest.json hashes.

    A CSV file holds exactly these bytes; a Parquet file holds the same table, which
    `pd.read_parquet(path).to_csv(index=False, lineterminator="\\n")` turns back into them.
    """
    return {name: to_csv(rows) for name, rows in build(random.Random(SEED)).items()}


def write(data_dir: Path) -> None:
    """Write every table into `data_dir`; only this step needs pandas and pyarrow."""
    import pandas as pd

    data_dir.mkdir(exist_ok=True)
    for name, rows in build(random.Random(SEED)).items():
        if name not in PARQUET_FILES:
            (data_dir / name).write_bytes(to_csv(rows))
            continue
        header, *body = rows
        table = pd.DataFrame(body, columns=header, dtype="string")
        for column in A1C_COLUMNS:
            if column in table:
                table[column] = table[column].replace("", pd.NA).astype("float64")
        table.to_parquet(data_dir / name, index=False)


def main() -> None:
    write(DATA_DIR)
    print(f"Wrote 5 data files to {DATA_DIR}")


if __name__ == "__main__":
    main()
