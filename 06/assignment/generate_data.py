"""Write the five synthetic clinic tables for Assignment 06 into data/.

    python3 generate_data.py

The rows come from the standard library, seeded, so every run on every computer
builds the same tables. The registry and the A1c results are saved as Parquet
with pandas and pyarrow (Lecture 04); the site list and the two visit exports
stay CSV, as clinic systems export them. The notebook's first cell checks each
table against data_manifest.json by the SHA-256 of its CSV text, which does not
depend on the pyarrow version. The data is course-authored and synthetic: no
real patients.
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


def clinic_days(first: date, count: int) -> list[date]:
    """The clinic days (Monday to Friday) among `count` days from `first`."""
    return [day for day in (first + timedelta(days=n) for n in range(count)) if day.weekday() < 5]


def clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def build(rng: random.Random) -> dict[str, list[list[str]]]:
    """Every file's rows, header first, as text cells."""
    spellings, _, weights = zip(*GENDER_SPELLINGS)
    patients, homes = [], {}
    for number in range(1, PATIENTS + 1):
        pid = f"P{number:05d}"
        site = rng.choices(("N01", "S02", "E04"), weights=(46, 52, 2))[0]
        homes[pid] = site
        written = rng.choice(SITE_SPELLINGS[site]) if rng.random() < 0.06 else site
        year = "9999" if rng.random() < 0.012 else str(rng.randint(1940, 1996))
        patients.append([pid, rng.choices(spellings, weights=weights)[0], year, written])
    # A registry re-export repeated some rows exactly, later in the file.
    for original in rng.sample(range(PATIENTS), 45):
        patients.insert(rng.randint(original + 1, len(patients)), list(patients[original]))

    def site_file(pid: str) -> str:
        return "north" if homes.get(pid, "S02") == "N01" else "south"

    with_visits = [f"P{n:05d}" for n in range(1, PATIENTS + 1) if rng.random() >= 0.015]
    with_visits += [f"P{n:05d}" for n in VISIT_ONLY_IDS]
    for number in VISIT_ONLY_IDS:
        homes[f"P{number:05d}"] = rng.choice(("N01", "S02"))
    enrollment_days = clinic_days(date(2026, 1, 5), 81)
    files: dict[str, list[list[str]]] = {"north": [], "south": []}
    for pid in with_visits:
        baseline = rng.choice(enrollment_days)
        sbp = min(215, max(95, round(rng.gauss(148, 16))))
        weight = min(170.0, max(48.0, rng.gauss(92, 18)))
        # The last cell, the visit as coded, is dropped before writing.
        rows = [[pid, "baseline", baseline, rng.randrange(8 * 60, 17 * 60, 5), str(sbp), f"{weight:.1f}",
                 "baseline"]]
        if rng.random() >= 0.07:
            follow = rng.choice(clinic_days(baseline + timedelta(days=81), 7))  # 12 weeks, give or take 3 days
            rows.append([pid, "week12", follow, rng.randrange(8 * 60, 17 * 60, 5),
                         str(min(215, max(90, sbp - rng.randint(-6, 22)))), f"{weight + rng.uniform(-4, 1):.1f}",
                         "week12"])
        files[site_file(pid)].extend(rows)

    for rows in files.values():
        for row in rows:
            draw = rng.random()
            if draw < 0.006:
                row[4] = "999"
            elif draw < 0.008:
                row[4] = rng.choice(OUT_OF_RANGE_SBP)
            elif draw < 0.0085:
                row[4] = rng.choice(("70", "250"))
            if rng.random() < 0.015:
                row[5] = ""
            if rng.random() < 0.04:
                row[1] = rng.choice(VISIT_SPELLINGS[row[6]])
        for row, limit in zip(rng.sample(rows, 4), ("70", "70", "250", "250")):
            row[4] = limit
        # A second reading 10 to 30 minutes later, listed before or after the first.
        for index in sorted(rng.sample(range(len(rows)), 60), reverse=True):
            first = rows[index]
            label = rng.choice((first[6], first[6], rng.choice(VISIT_SPELLINGS[first[6]])))
            recheck = [first[0], label, first[2], first[3] + rng.randrange(10, 35, 5), str(rng.randint(118, 176)),
                       first[5], first[6]]
            rows.insert(index + rng.choice((0, 1)), recheck)
        for row in rows:
            day = row[2]
            draw = rng.random()
            if draw < 0.0015:
                row[2] = rng.choice(IMPOSSIBLE_DATES)
            elif draw < 0.003:
                row[2] = rng.choice((f"{day.year}-{day.month}-{day.day:02d}",
                                     f"{day.year}-{day.month:02d}-{day.day}",
                                     f"{day.year}/{day.month:02d}/{day.day:02d}"))
            else:
                row[2] = day.isoformat()
            row[3] = clock(row[3])
            del row[6]
        # An export run twice repeated some rows exactly.
        for original in rng.sample(range(len(rows)), 35):
            rows.insert(rng.randint(original + 1, len(rows)), list(rows[original]))

    a1c = []
    lab_ids = [f"P{n:05d}" for n in range(1, PATIENTS + 1) if rng.random() >= 0.02]
    lab_ids += [f"P{n:05d}" for n in VISIT_ONLY_IDS if rng.random() < 0.5]
    lab_ids += [f"P{n:05d}" for n in LAB_ONLY_IDS]
    for pid in lab_ids:
        # A diabetes follow-up cohort: most baseline A1c results at or above 6.5%.
        baseline = min(13.5, max(5.9, rng.gauss(8.1, 1.1)))
        week12 = max(5.0, baseline - rng.uniform(-0.3, 1.2))
        a1c.append([pid, "" if rng.random() < 0.01 else f"{baseline:.1f}",
                    "" if rng.random() < 0.06 else f"{week12:.1f}"])

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
