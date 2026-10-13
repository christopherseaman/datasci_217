#!/usr/bin/env python3
"""Build the supplied dataset for Assignment 05.

Course tooling, not part of the student handout: the assignment ships the CSV
this writes, and `05/assignment_checks/_value_checks.py` carries a copy of it
(`RAW_ROWS`), so a submission is scored against the file students were given.
Rebuilding the data means updating that copy; the self-test fails while the two
differ.

    python3 scripts/make_assignment05_data.py

The file is a synthetic diabetes clinic lab export: one row per visit with the
patient's HbA1c (%), fasting glucose (mg/dL), and weight (kg). Each row carries
at most one planted problem, listed beside it, so every cleaning step has one
visible effect.
"""

from __future__ import annotations

import csv
from pathlib import Path


OUTPUT = Path(__file__).resolve().parents[1] / "05" / "assignment" / "data" / "lab_visits.csv"

COLUMNS = ("patient_id", "visit_date", "clinic", "sex", "age", "weight_kg", "hba1c_pct", "glucose_mg_dl")
ROWS = (
    ("D2001", "2026-03-02", "Mission", "F", "54", "82.5", "7.8", "162"),
    ("D2002", "2026-03-02", "mission ", "f", "61", "95.1", "6.1", "118"),       # clinic case and space
    ("D2003", "2026-03-03", "Sunset", "M", "47", "-999", "8.9", "201"),         # weight sentinel
    ("D2004", "2026-03-03", "SUNSET", "male", "66", "77.0", "", "135"),         # HbA1c blank
    ("D2005", "2026-03-04", "Excelsior", "F", "unknown", "68.4", "5.4", "96"),  # age as text
    ("D2006", "2026-03-04", "excelsior", "M", "58", "101.3", "9.6", "244"),
    ("D2006", "2026-03-04", "excelsior", "M", "58", "101.3", "9.6", "244"),     # exact repeat
    ("D2007", "2026-03-05", " Mission", "female", "72", "64.8", "64", "140"),   # HbA1c 64 for 6.4
    ("D2008", "2026-03-05", "", "M", "39", "88.0", "5.7", "104"),               # clinic blank; HbA1c on a bin edge
    ("D2009", "2026-02-30", "Sunset", "F", "50", "73.2", "6.8", "128"),         # impossible date
    ("D2010", "2026-03-06", "Mission", "M", "63", "950", "7.2", "-999"),        # weight 950, glucose sentinel
    ("D2011", "2026-03-06", "Excelsior", "F", "45", "59.7", "", ""),            # no labs: a no-show
    ("D2001", "2026-04-06", "Mission", "F", "54", "81.9", "7.4", "151"),        # repeat visit, kept
    ("D2012", "2026-04-07", "Sunset", "U", "57", "70.5", "6.3", "122"),         # sex code outside F/M
    ("D204", "2026-04-07", "Excelsior", "M", "68", "84.2", "8.1", "188"),       # ID missing a digit
    ("D2013", "", "Mission", "F", "49", "66.1", "5.6", "99"),                   # visit date blank
    ("d2014 ", "2026-04-08", "mission", "F", "70", "79.4", "10.8", "412"),      # ID case and space; glucose outlier
    ("D2015", "2026-04-08", "Sunset", "m", "52", "92.7", "6.5", "131"),         # HbA1c on a bin edge
    ("D2016", "2026-04-09", "Excelsior", "F", "44", "61.0", "5.2", "90"),
    ("D2016", "2026-04-09", "Excelsior", "F", "44", "61.0", "5.2", "90"),       # exact repeat
    ("D2017", "2026-04-09", "MISSION", "M", "59", "-999", "7.0", ""),           # weight sentinel, glucose blank
    ("D2018", "2026-04-10", "Sunset", "F", "65", "71.8", "6.9", "145"),
)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(ROWS)
    print(f"wrote {OUTPUT} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
