"""Build the synthetic clinic data for Assignment 06 (midterm).

    python3 scripts/build_assignment06_data.py [--output-dir DIR]

Standard library only, seeded, so every run writes the same bytes. The files are
course-authored synthetic teaching data: no real patients. Each planted issue is
listed in PLANTED below; the course checks' self-test recomputes every expected
value from the written files.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "06" / "assignment" / "data"
SEED = 2026_06

# Every planted issue, for the instructor; the handout never lists them.
PLANTED = {
    "patients.csv": "P009 repeated exactly; sex ' f', 'm ', 'f', and empty (P020); site_code 'n01' and ' S02'; "
                    "birth_year 9999 sentinel (P007); P023 at site E04, which sites.csv lacks; P012 has no visits",
    "sites.csv": "S02 listed twice, one retired and one current",
    "visits_north.csv": "P003 week12 repeated exactly; 'Baseline' and ' week12' labels; sbp 999 sentinel (P006); "
                        "a single-digit month date (P010 week12); P005 has no week12 visit",
    "visits_south.csv": "weight column named wt_kg; 'WEEK12' and 'Baseline ' labels; P015 baseline rechecked "
                        "(later reading first in the file); sbp 999 (P017) and 18 (P022); 2026-02-30 (P019); "
                        "blank weight (P024 week12); P030 not in the registry; P018 has no week12 visit",
    "a1c_wide.csv": "P099 not in the registry and without visits; blank week12 for P018 and P021; "
                    "P005 has a week12 result without a week12 visit",
}


def build(rng: random.Random) -> dict[str, list[list[str]]]:
    """Every file's rows, header first, as text cells."""
    patients = []
    for number in range(1, 25):
        pid = f"P{number:03d}"
        site = "N01" if number <= 12 else "E04" if number == 23 else "S02"
        patients.append([pid, rng.choice(["F", "M"]), str(rng.randint(1942, 1994)), site])
    by_id = {row[0]: row for row in patients}
    by_id["P002"][1] = " f"
    by_id["P010"][1] = "m "
    by_id["P019"][1] = "f"
    by_id["P020"][1] = ""
    by_id["P007"][2] = "9999"
    by_id["P004"][3] = "n01"
    by_id["P016"][3] = " S02"
    position = next(index for index, row in enumerate(patients) if row[0] == "P009")
    patients.insert(position + 1, list(patients[position]))

    sites = [["N01", "North Clinic", "current"], ["S02", "Southgate Clinic", "retired"],
             ["S02", "South Valley Clinic", "current"], ["W03", "West Clinic", "current"]]

    def visit_rows(pid: str, week12: bool) -> list[list[str]]:
        baseline = date(2026, 1, 5) + timedelta(days=rng.randint(0, 60))
        sbp = rng.randint(128, 172)
        weight = round(rng.uniform(55, 115), 1)
        rows = [[pid, "baseline", baseline.isoformat(), f"{rng.randint(8, 15):02d}:{rng.choice(range(0, 60, 5)):02d}",
                 str(sbp), f"{weight:.1f}"]]
        follow_sbp = sbp - rng.randint(-4, 22)
        follow_weight = round(weight + rng.uniform(-3, 1), 1)
        follow = baseline + timedelta(days=84 + rng.randint(-3, 3))
        if week12:
            rows.append([pid, "week12", follow.isoformat(),
                         f"{rng.randint(8, 15):02d}:{rng.choice(range(0, 60, 5)):02d}",
                         str(follow_sbp), f"{follow_weight:.1f}"])
        return rows

    north, south = [], []
    for number in range(1, 25):
        pid = f"P{number:03d}"
        if pid == "P012":
            continue
        target = north if number <= 12 else south
        target.extend(visit_rows(pid, week12=pid not in ("P005", "P018")))
    south.extend(visit_rows("P030", week12=True))

    def row(rows: list[list[str]], pid: str, visit: str) -> list[str]:
        return next(r for r in rows if r[0] == pid and r[1] == visit)

    row(north, "P002", "baseline")[1] = "Baseline"
    row(north, "P008", "week12")[1] = " week12"
    row(north, "P006", "baseline")[4] = "999"
    p010 = date.fromisoformat(row(north, "P010", "week12")[2])
    row(north, "P010", "week12")[2] = f"{p010.year}-{p010.month}-{p010.day:02d}"
    duplicate = row(north, "P003", "week12")
    north.insert(north.index(duplicate) + 1, list(duplicate))

    row(south, "P014", "week12")[1] = "WEEK12"
    row(south, "P021", "baseline")[1] = "Baseline "
    row(south, "P017", "week12")[4] = "999"
    row(south, "P022", "baseline")[4] = "18"
    row(south, "P019", "baseline")[2] = "2026-02-30"
    row(south, "P024", "week12")[5] = ""
    # P015's baseline was rechecked 15 minutes later; the export lists the later reading first.
    first = row(south, "P015", "baseline")
    first[3], first[4] = "09:10", "168"
    recheck = [first[0], first[1], first[2], "09:25", "152", first[5]]
    south.insert(south.index(first), recheck)

    a1c = []
    for number in [*range(1, 12), *range(13, 25), 99]:
        pid = f"P{number:03d}"
        baseline = round(rng.uniform(5.6, 10.4), 1)
        week12 = round(baseline - rng.uniform(-0.3, 1.4), 1)
        a1c.append([pid, f"{baseline:.1f}", "" if pid in ("P018", "P021") else f"{week12:.1f}"])

    header = ["patient_id", "visit", "visit_date", "visit_time", "sbp_mmhg"]
    return {
        "patients.csv": [["patient_id", "sex", "birth_year", "site_code"], *patients],
        "sites.csv": [["site_code", "site_name", "record_status"], *sites],
        "visits_north.csv": [[*header, "weight_kg"], *north],
        "visits_south.csv": [[*header, "wt_kg"], *south],
        "a1c_wide.csv": [["patient_id", "a1c_baseline", "a1c_week12"], *a1c],
    }


def to_csv(rows: list[list[str]]) -> bytes:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"fixture_id": "a06-clinic-midterm-v1",
                "provenance": "course-authored synthetic teaching data; no real patients",
                "files": {}}
    for name, rows in build(random.Random(SEED)).items():
        data = to_csv(rows)
        (args.output_dir / name).write_bytes(data)
        manifest["files"][name] = {"rows": len(rows) - 1, "columns": rows[0],
                                   "sha256": hashlib.sha256(data).hexdigest()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['files'])} files and manifest.json to {args.output_dir}")


if __name__ == "__main__":
    main()
