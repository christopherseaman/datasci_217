"""Course-owned grading rules for Assignment 06, the midterm.

The midterm handout ships no checks. After the deadline the course grades each
fork's committed files in `output/` with this module, through
`check_assignment.py` and `scripts/grade_submissions.py`. Student code is never
imported, executed, or read, and the notebook is left to human review.

The handout ships `generate_data.py`, never the data. Every expected value is
derived, when this module loads, from the bytes the course copy of that
generator (`06/mid-term/generate_data.py`, found above these checks) builds
in memory as CSV text with the standard library, accepted only when each table's
SHA-256 matches `DATA_SHA256` (the values the handout's `data_manifest.json`
lists). Nothing is written to disk, and pandas is never imported: the Parquet
files hold the same tables, which only the handout's copy writes.
`_grader_selftest/run.py` recomputes each value with pandas.

The outputs are analyst-sized: audits and summaries of the whole tables, plus
small extracts of the rows for the patients in `SAMPLE_IDS`. Each check reads
one artifact and compares parsed values rather than text, so line endings,
trailing whitespace, a missing final newline, header spacing, a comma,
semicolon, or tab between cells, column order, a leading unnamed index column,
number formatting, the letter case and spacing of labels and text values, any
unambiguous way of writing a date, and the common spellings of missing values
never cost points. Each check scores the units it can verify on its own, so one
wrong value costs only its own share, and no check depends on another passing.
A later artifact is also judged against the student's own earlier summaries
(the summaries against the site code, date, and visit label rules their own
audit counts show; the merge counts against their own visit, A1c, and patient summaries; the
extracts accept the values an upstream mistake already charged would carry),
so one mistake is charged once.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import combinations
import hashlib
import io
import math
from pathlib import Path
import re


SCHEMA = "datasci217/grading-result/v1"
OUTPUT = Path("output")

DATA_FILES = ("patients.parquet", "sites.csv", "visits_north.csv", "visits_south.csv", "a1c_wide.parquet")
# The generated files' SHA-256 values, as the handout's data_manifest.json lists them. Every expected value
# below is derived from tables with exactly this CSV text (built by 06/mid-term/generate_data.py).
DATA_SHA256 = {
    "patients.parquet": "aa795b05777565cce6af68602e3fc7c286a2483f53f9a6d99a44da410c1214dd",
    "sites.csv": "daaa4f3fc475900db905590faf6524f34143e78b73633b1794de4c6cf4b56a77",
    "visits_north.csv": "f05014a1d281ee5e7732559d31fcf7a0e6e2b44f027b66e31229097484226a79",
    "visits_south.csv": "c19644241228ba7ef007d1aa7386516d80af5340b633401da67e9f59f3383214",
    "a1c_wide.parquet": "934dc120b78d492666b7fbc7872d5fc4d0f20ab9a678bdb7559f00536a2cfa6a",
}
# The patients whose rows the extracts hold, as the README and the notebook's first cell list them. They
# cover a repeated registry row, the 9999 sentinel, messy gender and site spellings, a site sites.csv lacks,
# a visit with no registry row, rechecked visits, SBP sentinels and limits, bad dates, blank weights,
# unmapped visit labels, a missing week 12 visit, and blank A1c results.
SAMPLE_IDS = ("P00001", "P00004", "P00088", "P00260", "P00446", "P00487", "P00631", "P00851", "P02354",
              "P03836", "P05763", "P07510", "P14287", "P16486", "P17390", "P30001")
# Every spelling in the files, stripped and lowercased, mapped to its code; `unknown` and blanks stay missing.
GENDER_MAP = {"f": "F", "female": "F", "woman": "F", "m": "M", "male": "M", "man": "M",
              "x": "X", "nb": "X", "nonbinary": "X", "non-binary": "X"}
VISIT_MAP = {"baseline": "baseline", "bl": "baseline", "week12": "week12", "week 12": "week12", "wk12": "week12"}
EXACT_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
SBP_RANGE = (70, 250)
# A mean written to two decimals is right within this of the exact mean.
MEAN_TOLERANCE = 0.006
PROFILE_COLUMNS = ("rows", "columns", "exact_duplicate_rows")
PROFILE_HINTS = {
    "rows": "len(table) counts data rows; the header is not a row (wc -l counts it, so its numbers are one higher)",
    "columns": "table.shape[1] counts the columns",
    "exact_duplicate_rows": "table.duplicated().sum() counts each repeat once, without keep=False",
}

ISSUE_HINTS = {
    "patients: gender values that are not a code": "leave out empty values and count every other gender, as "
                                                   "written, that is not exactly F, M, or X",
    "patients: empty gender": "compare raw[\"patients\"][\"gender\"].str.strip() with \"\"",
    "patients: birth_year sentinel 9999": "count birth_year values equal to '9999'",
    "patients: site codes needing normalization": "count every site_code that .str.upper() and removing every "
                                                  "space and hyphen change",
    "sites: rows with a repeated site_code": "use duplicated(subset=['site_code'], keep=False), which counts the "
                                             "first copy too",
    "visits: visit labels that are not baseline or week12": "count, in both visit files stacked, every visit "
                                                            "label that is not exactly baseline or week12",
    "visits: sbp sentinel 999": "count sbp_mmhg values equal to '999' in both visit files",
    "visits: sbp outside 70 through 250": "convert with pd.to_numeric, leave out the 999 sentinel, and keep 70 "
                                          "and 250 inside the range",
    "visits: date parse failures": "check the text with .str.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}') first, "
                                   "because format='%Y-%m-%d' alone accepts 2026-5-10, then let "
                                   "to_datetime(..., errors='coerce') catch impossible dates",
    "visits: rows in repeated patient_id and visit pairs": "map the visit labels first, then use "
                                                           "duplicated(subset=['patient_id', 'visit'], keep=False) "
                                                           "on both visit files stacked",
}


def generated_data() -> dict[str, bytes]:
    """Each data table as the CSV text the course copy of generate_data.py builds, checked against DATA_SHA256.

    The generator is course-owned code beside these checks (06/mid-term), never a submitted file. It runs
    from its source text, so no bytecode cache is written next to it.
    """
    here = Path(__file__).resolve().parent
    for script in (parent / "mid-term" / "generate_data.py" for parent in (here, *here.parents)):
        if not script.is_file():
            continue
        namespace = {"__name__": "generate_data", "__file__": str(script)}
        exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)
        built = namespace["files"]()
        if all(name in built and hashlib.sha256(built[name]).hexdigest() == digest
               for name, digest in DATA_SHA256.items()):
            return built
    raise FileNotFoundError("no mid-term/generate_data.py above the checks builds the data tables whose "
                            "SHA-256 values DATA_SHA256 lists")


def read_data(data: bytes) -> list[dict[str, str]]:
    """A table's rows as text, every cell kept as written, as read_csv(dtype="string", keep_default_na=False)."""
    header, *rows = csv.reader(io.StringIO(data.decode("utf-8")))
    return [dict(zip(header, row)) for row in rows]


def whole_number(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def exact_day(text: str) -> date | None:
    """The date when the text is exact YYYY-MM-DD for a day on the calendar, else None."""
    text = text.strip()
    if not EXACT_DATE.fullmatch(text):
        return None
    try:
        return date(int(text[:4]), int(text[5:7]), int(text[8:]))
    except ValueError:
        return None


def repeats(keys: Sequence[object]) -> int:
    """How many items share their value with another item: duplicated(keep=False).sum()."""
    counts: dict[object, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return sum(count for count in counts.values() if count > 1)


def exact_repeats(rows: Sequence[dict[str, str]]) -> int:
    """Rows identical to an earlier row: duplicated().sum()."""
    return len(rows) - len({tuple(row.values()) for row in rows})


def site_code(text: str) -> str:
    return text.upper().replace("-", "").replace(" ", "")


def _first_copies(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """drop_duplicates(keep="first")."""
    seen: set[tuple[str, ...]] = set()
    kept = []
    for row in rows:
        if tuple(row.values()) not in seen:
            seen.add(tuple(row.values()))
            kept.append(row)
    return kept


_DATA = generated_data()
RAW = {name: read_data(_DATA[name]) for name in DATA_FILES}
# `wc -l data/*.csv` counts newline characters, so each count includes the header line. Parquet is
# binary, so the inventory lists only the CSV files.
INVENTORY = {name: _DATA[name].count(b"\n") for name in DATA_FILES if name.endswith(".csv")}
# Each raw table's data rows, columns, and rows identical to an earlier row.
TABLE_PROFILE = {name: (len(rows), len(rows[0]), exact_repeats(rows)) for name, rows in RAW.items()}

_patients, _sites = RAW["patients.parquet"], RAW["sites.csv"]
_raw_visits = [{**row, "weight_kg": row.get("weight_kg", row.get("wt_kg", "")), "source_file": site}
               for site in ("north", "south") for row in RAW[f"visits_{site}.csv"]]
for _row in _raw_visits:
    _row.pop("wt_kg", None)
_sbp_raw = [whole_number(row["sbp_mmhg"]) for row in _raw_visits]
ISSUE_AUDIT = (
    ("patients: gender values that are not a code",
     sum(row["gender"].strip() != "" and row["gender"] not in ("F", "M", "X") for row in _patients)),
    ("patients: empty gender", sum(row["gender"].strip() == "" for row in _patients)),
    ("patients: birth_year sentinel 9999", sum(row["birth_year"].strip() == "9999" for row in _patients)),
    ("patients: site codes needing normalization",
     sum(site_code(row["site_code"]) != row["site_code"] for row in _patients)),
    ("sites: rows with a repeated site_code", repeats([row["site_code"] for row in _sites])),
    ("visits: visit labels that are not baseline or week12",
     sum(row["visit"] not in ("baseline", "week12") for row in _raw_visits)),
    ("visits: sbp sentinel 999", sum(row["sbp_mmhg"].strip() == "999" for row in _raw_visits)),
    ("visits: sbp outside 70 through 250",
     sum(value is not None and value != 999 and not SBP_RANGE[0] <= value <= SBP_RANGE[1] for value in _sbp_raw)),
    ("visits: date parse failures",
     sum(row["visit_date"].strip() != "" and exact_day(row["visit_date"]) is None for row in _raw_visits)),
    ("visits: rows in repeated patient_id and visit pairs",
     repeats([(row["patient_id"], VISIT_MAP.get(row["visit"].strip().lower())) for row in _raw_visits])),
)


def _clean_patient(row: dict[str, str]) -> dict[str, object]:
    year = whole_number(row["birth_year"])
    return {"gender": GENDER_MAP.get(row["gender"].strip().lower()),
            "birth_year": None if year in (None, 9999) else int(year), "site_code": site_code(row["site_code"])}


# The expected cleaned patients, as the README's rules make them from the raw rows, and the same rows
# cleaned without removing the exact repeats (a mistake the patients count charges once).
PATIENTS = {row["patient_id"]: _clean_patient(row) for row in _first_copies(_patients)}
_PATIENT_ROWS_KEPT = [_clean_patient(row) for row in _patients]
# Each registry patient's raw values, which an extract may carry when an upstream rule was skipped.
RAW_GENDER: dict[str, set[str]] = {}
RAW_YEAR: dict[str, set[float]] = {}
for _row in _patients:
    RAW_GENDER.setdefault(_row["patient_id"], set()).add(_row["gender"])
    RAW_YEAR.setdefault(_row["patient_id"], set()).add(whole_number(_row["birth_year"]))



def _latest_by(visit_map: dict[str, str], normal: Callable[[str], str] = lambda text: text.strip().lower()
               ) -> dict[tuple[str, str | None], dict[str, str]]:
    """The exact repeats dropped, each label mapped (unmapped ones missing), and each pair's last row by visit_time."""
    latest: dict[tuple[str, str | None], dict[str, str]] = {}
    for row in sorted(_first_copies(_raw_visits), key=lambda row: row["visit_time"]):
        latest[(row["patient_id"], visit_map.get(normal(row["visit"])))] = row
    return latest


# A recheck keeps the later reading: the last row of each pair once sorted by visit_time.
_latest = _latest_by(VISIT_MAP)
# Pairs the raw files list more than once (rechecks and exact repeats), whose rows a skipped step repeats.
REPEATED_PAIRS = frozenset(key for key, count in Counter(
    (row["patient_id"], VISIT_MAP[row["visit"].strip().lower()]) for row in _raw_visits).items() if count > 1)

# Pairs with a raw label other than exactly baseline or week12, which an unmapped label leaves without a visit.
RELABELED_PAIRS = frozenset((row["patient_id"], VISIT_MAP[row["visit"].strip().lower()]) for row in _raw_visits
                            if row["visit"] not in ("baseline", "week12"))
# Registry patients listed more than once, whose joined rows repeat when the exact repeats are kept.
REPEATED_PATIENTS = frozenset(pid for pid, count in Counter(row["patient_id"] for row in _patients).items()
                              if count > 1)

def _clean_visit(row: dict[str, str], day: Callable[[str], date | None] = exact_day) -> dict[str, object]:
    sbp = whole_number(row["sbp_mmhg"])
    return {"visit_date": day(row["visit_date"]),
            "sbp_mmhg": int(sbp) if sbp is not None and SBP_RANGE[0] <= sbp <= SBP_RANGE[1] else None,
            "weight_kg": whole_number(row["weight_kg"]), "source_file": row["source_file"]}


VISITS = {key: _clean_visit(_latest[key]) for key in sorted(_latest)}
# The kept reading's SBP as written, which an extract carries when the 70 through 250 rule was skipped.
RAW_SBP = {key: whole_number(_latest[key]["sbp_mmhg"]) for key in VISITS}
# melt() puts every baseline before every week 12 result; rows with no result are dropped.
A1C = {(row["patient_id"], visit): whole_number(row[f"a1c_{visit}"])
       for visit in ("baseline", "week12") for row in RAW["a1c_wide.parquet"]
       if whole_number(row[f"a1c_{visit}"]) is not None}
CURRENT_SITES = {row["site_code"]: row["site_name"] for row in _sites if row["record_status"] == "current"}
SITE_CODES = frozenset(row["site_code"] for row in _sites)
VISIT_LABELS = ("baseline", "week12")
GENDER_CODES = ("F", "M", "X")


def _gender_counts(rows: Sequence[dict[str, object]]) -> dict[str | None, int]:
    """value_counts(dropna=False) of the categorical gender: each code's count, and the missing values' (None)."""
    return {code: sum(row["gender"] == code for row in rows) for code in (*GENDER_CODES, None)}


GENDER_COUNTS = _gender_counts(list(PATIENTS.values()))
GENDER_COUNTS_KEPT = _gender_counts(_PATIENT_ROWS_KEPT)


def _patient_summary(rows: Sequence[dict[str, object]]) -> dict[str, float]:
    years = [row["birth_year"] for row in rows if row["birth_year"] is not None]
    summary = {"patients": len(rows), "birth_year missing": len(rows) - len(years), "birth_year min": min(years),
               "birth_year max": max(years), "site_code values": len({row["site_code"] for row in rows})}
    for code in ("N01", "S02", "E04"):
        summary[f"patients at {code}"] = sum(row["site_code"] == code for row in rows)
    return summary


PATIENT_SUMMARY = _patient_summary(list(PATIENTS.values()))
# The same summary with the exact repeats left in, accepted where the patients count already charged them.
PATIENT_SUMMARY_KEPT = _patient_summary(_PATIENT_ROWS_KEPT)
PATIENT_PARTS = {
    "rows": ("patients",),
    "birth_year": ("birth_year missing", "birth_year min", "birth_year max"),
    "site_code": ("site_code values", "patients at N01", "patients at S02", "patients at E04"),
}



def _visit_summary(visits: dict[tuple[str, str], dict[str, object]]) -> dict[str, object]:
    dates = [values["visit_date"] for values in visits.values() if values["visit_date"] is not None]
    sbps = [values["sbp_mmhg"] for values in visits.values() if values["sbp_mmhg"] is not None]
    weights = [values["weight_kg"] for values in visits.values() if values["weight_kg"] is not None]
    return {
        "visits": len(visits),
        "baseline visits": sum(visit == "baseline" for _, visit in visits),
        "week12 visits": sum(visit == "week12" for _, visit in visits),
        "visit_date missing": len(visits) - len(dates), "visit_date min": min(dates), "visit_date max": max(dates),
        "sbp_mmhg missing": len(visits) - len(sbps), "sbp_mmhg min": min(sbps), "sbp_mmhg max": max(sbps),
        "weight_kg missing": len(visits) - len(weights), "weight_kg mean": sum(weights) / len(weights),
        "visits from visits_north.csv": sum(values["source_file"] == "north" for values in visits.values()),
        "visits from visits_south.csv": sum(values["source_file"] == "south" for values in visits.values()),
    }


VISIT_SUMMARY = _visit_summary(VISITS)
# The same summary when each recheck keeps its earlier reading, a mistake sbp_wide_sample.csv charges once.
_earlier: dict[tuple[str, str], dict[str, str]] = {}
for _row in sorted(_first_copies(_raw_visits), key=lambda row: row["visit_time"]):
    _earlier.setdefault((_row["patient_id"], VISIT_MAP[_row["visit"].strip().lower()]), _row)
VISIT_SUMMARY_EARLIER = _visit_summary({key: _clean_visit(_earlier[key]) for key in sorted(_earlier)})


# Task 2.3 counts three issues with the rule Task 3.1 or 3.2 cleans with, so a wrong rule shows first in
# issue_audit.csv. Each common wrong rule is listed with the audit count it gives and the summaries it leads
# to; a summary part accepts those values when the student's own audit row holds that count.
def _loose_day(text: str) -> date | None:
    """The date format='%Y-%m-%d' alone gives: 2026-5-10 parses, impossible and slashed dates do not."""
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _site_rule(steps: Sequence[Callable[[str], str]]) -> Callable[[str], str]:
    def rule(text: str) -> str:
        for step in steps:
            text = step(text)
        return text
    return rule


# Every combination of uppercasing, stripping, and removing hyphens or every space.
_SITE_STEPS = (str.upper, str.strip, lambda text: text.replace("-", ""), lambda text: text.replace(" ", ""))
_SITE_RULES = [_site_rule([step for bit, step in enumerate(_SITE_STEPS) if mask >> bit & 1]) for mask in range(16)]
# Every way of leaving spellings out of the visit dictionary.
_VISIT_MAPS = [{key: value for key, value in VISIT_MAP.items() if key not in left_out}
               for size in (0, 1, 2, 3) for left_out in combinations(("bl", "week 12", "wk12"), size)]
# Every way of skipping .str.strip() or .str.lower() before the map.
_VISIT_NORMALS = (lambda text: text.strip().lower(), str.strip, str.lower, lambda text: text)
_audit = dict(ISSUE_AUDIT)
OWN_RULES: dict[tuple[str, str], tuple[str, list[tuple[int, list[dict[str, object]]]]]] = {
    ("patients_summary.csv", "site_code"): ("patients: site codes needing normalization", [
        (sum(rule(row["site_code"]) != row["site_code"] for row in _patients),
         [_patient_summary([{**_clean_patient(row), "site_code": rule(row["site_code"])} for row in rows])
          for rows in (_first_copies(_patients), _patients)])
        for rule in _SITE_RULES]),
    ("visits_summary.csv", "rows"): ("visits: rows in repeated patient_id and visit pairs", [
        (repeats([(row["patient_id"], visit_map.get(normal(row["visit"]))) for row in _raw_visits]),
         [_visit_summary({key: _clean_visit(row) for key, row in _latest_by(visit_map, normal).items()})])
        for visit_map in _VISIT_MAPS for normal in _VISIT_NORMALS] + [
        # No dictionary at all: each label kept as normalized, so only exact baseline and week12 count as visits.
        (repeats([(row["patient_id"], normal(row["visit"])) for row in _raw_visits]),
         [_visit_summary({key: _clean_visit(row) for key, row in _latest_by(
             {normal(row["visit"]): normal(row["visit"]) for row in _raw_visits}, normal).items()})])
        for normal in _VISIT_NORMALS]),
    ("visits_summary.csv", "visit_date"): ("visits: date parse failures", [
        (sum(row["visit_date"].strip() != "" and _loose_day(row["visit_date"]) is None for row in _raw_visits),
         [_visit_summary({key: _clean_visit(row, _loose_day) for key, row in _latest.items()})])]),
}
# A wrong rule whose count equals the right one's cannot be told apart from it, so it earns nothing extra.
OWN_RULES = {part: (issue, [option for option in options if option[0] != _audit[issue]])
             for part, (issue, options) in OWN_RULES.items()}
# Each pair's earlier clean SBP, which a pivot carries when an unmapped label kept a recheck apart.
EARLIER_SBP = {key: _clean_visit(row)["sbp_mmhg"] for key, row in _earlier.items()}
VISIT_PARTS = {
    "rows": ("visits", "baseline visits", "week12 visits"),
    "visit_date": ("visit_date missing", "visit_date min", "visit_date max"),
    "sbp_mmhg": ("sbp_mmhg missing", "sbp_mmhg min", "sbp_mmhg max"),
    "weight_kg": ("weight_kg missing", "weight_kg mean"),
    "source_file": ("visits from visits_north.csv", "visits from visits_south.csv"),
}
# The widest a weight can move a mean when one row is added or dropped, per row, in kg.
_weights = [float(row["weight_kg"]) for row in _raw_visits if row["weight_kg"].strip()]
WEIGHT_SPAN = max(_weights) - min(_weights)

# Each visit label's A1c results in a1c_long and their mean, as the per-visit summary of the melted table.
A1C_SUMMARY = {visit: (sum(key[1] == visit for key in A1C),
                       sum(value for key, value in A1C.items() if key[1] == visit) / sum(key[1] == visit
                                                                                        for key in A1C))
               for visit in VISIT_LABELS}

# One row per clean visit, as the left merges onto visits_clean give it.
ANALYSIS = {key: {"a1c_pct": A1C.get(key), **PATIENTS.get(key[0], {"gender": None, "birth_year": None,
                                                                   "site_code": None})}
            for key in VISITS}
for _row in ANALYSIS.values():
    _row["site_name"] = CURRENT_SITES.get(_row["site_code"] or "")

SBP_WIDE = {pid: {visit: VISITS.get((pid, visit), {}).get("sbp_mmhg") for visit in VISIT_LABELS}
            for pid in dict.fromkeys(pid for pid, _ in VISITS)}
SAMPLE_VISITS = [key for key in VISITS if key[0] in SAMPLE_IDS]
SAMPLE_WIDE = [pid for pid in SBP_WIDE if pid in SAMPLE_IDS]

_visit_patients = {pid for pid, _ in VISITS}
MERGE_AUDIT = (
    ("visits with an A1c result", sum(key in A1C for key in VISITS)),
    ("visits without an A1c result", sum(key not in A1C for key in VISITS)),
    ("A1c results without a visit", sum(key not in VISITS for key in A1C)),
    ("visits whose patient is not in the registry", sum(pid not in PATIENTS for pid, _ in VISITS)),
    ("registry patients with no visit", sum(pid not in _visit_patients for pid in PATIENTS)),
    ("registry patients whose site is not in the site list",
     sum(row["site_code"] not in SITE_CODES for row in PATIENTS.values())),
    ("site list rows removed by the current-record rule", len(_sites) - len(CURRENT_SITES)),
)
MERGE_HINTS = {
    "visits with an A1c result": "merge visits_clean and a1c_long on ['patient_id', 'visit'] with how='outer' "
                                 "and indicator=True, then count _merge == 'both'",
    "visits without an A1c result": "count _merge == 'left_only' in the same outer merge",
    "A1c results without a visit": "count _merge == 'right_only' in the same outer merge",
    "visits whose patient is not in the registry": "count visits_clean rows whose patient_id is not in "
                                                   "patients_clean (~isin)",
    "registry patients with no visit": "count patients_clean rows whose patient_id is not in visits_clean",
    "registry patients whose site is not in the site list": "count patients_clean rows whose site_code is not "
                                                            "in sites.csv",
    "site list rows removed by the current-record rule": "count the sites.csv rows whose record_status is not "
                                                         "current",
}

# The issues the decision log covers, one row each; students write each decision and reason themselves.
DECISION_ISSUES = (
    ("patients", "exact duplicate rows"),
    ("patients", "gender spellings that are not a code"),
    ("patients", "site codes needing normalization"),
    ("patients", "birth_year sentinel 9999"),
    ("visits", "weight column named wt_kg in visits_south.csv"),
    ("visits", "sbp sentinel 999 and readings outside 70 through 250"),
    ("visits", "impossible and loosely written visit dates"),
    ("visits", "repeated patient_id and visit pairs"),
    ("sites", "repeated site_code rows"),
)

# Spellings of a missing cell. `NA` is left out for text columns, where it could be a value.
TEXT_MISSING = frozenset({"", "nan", "<na>", "nat", "none", "null"})
VALUE_MISSING = TEXT_MISSING | {"na", "n/a"}
LEADING_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
# NumPy 2 prints a scalar's repr as `np.int64(12)`; the number inside is the value.
NUMPY_SCALAR = re.compile(r"^(?:np|numpy)\.\w+\((.*)\)$")
# Two labels at least this similar are the same label with a typo or a small rewording.
SIMILAR = 0.8
# Dates are read with these patterns and datetime() alone, never strptime or fromisoformat, whose
# accepted forms depend on the locale and the Python version: a date reads the same on every Python.
# Each pattern may end with a time (any fraction of a second) and `Z`, `UTC`, or an offset.
TIME = (r"(?:(?:[T ]|,\s*)(?P<hour>\d{1,2})(?::(?P<minute>\d{2})(?::(?P<second>\d{2})(?:[.,]\d+)?)?)?)?"
        r"\s*(?P<zone>Z|UTC|GMT|[+-]\d{2}(?::?\d{2})?)?")
MONTH_NAMES = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
               "november", "december")
MONTHS = {**{name: number for number, name in enumerate(MONTH_NAMES, start=1)},
          **{name[:3]: number for number, name in enumerate(MONTH_NAMES, start=1)}, "sept": 9}
WEEKDAY = re.compile(r"^(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?,?\s+", re.IGNORECASE)
DATE_FORMS = tuple(re.compile(pattern + TIME + "$", re.IGNORECASE) for pattern in (
    r"^(?P<year>\d{4})(?P<mark>[-/.])(?P<month>\d{1,2})(?P=mark)(?P<day>\d{1,2})",  # 2026-01-15, 2026/1/15
    r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})",  # 20260115
    r"^(?P<first>\d{1,2})(?P<mark>[-/.])(?P<second_part>\d{1,2})(?P=mark)(?P<year>\d{4})",  # 01/15/2026, 15.01.2026
    r"^(?P<day>\d{1,2})(?:st|nd|rd|th)?[\s-]+(?P<name>[a-z]+)\.?,?[\s-]+(?P<year>\d{4})",  # 15 January 2026
    r"^(?P<name>[a-z]+)\.?[\s-]+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?[\s-]+(?P<year>\d{4})",  # Jan 15, 2026
    r"^(?P<year>\d{4})[\s-]+(?P<name>[a-z]+)\.?[\s-]+(?P<day>\d{1,2})",  # 2026-Jan-15
))
DASHES = re.compile(r"[-_]")
COMMA_SPACES = re.compile(r"\s*,\s*")
UNNAMED_INDEX = re.compile(r"^(?:unnamed:_?0|index)?$")
# A CSV may separate its cells with commas, semicolons, or tabs.
DELIMITERS = (",", ";", "\t")
# A number written with a decimal comma, as a spreadsheet set to a European locale saves it: 33,0 is 33.0.
DECIMAL_COMMA = re.compile(r"^(\s*[+-]?\d*),(\d+\s*)$")


class ArtifactProblem(Exception):
    """An artifact that cannot be graded at all, with the reason in plain words."""


@dataclass
class Outcome:
    """What one check verified: `right` of `total` units, and what was wrong."""

    total: int
    right: int = 0
    problems: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Check:
    name: str
    points: int
    task: str
    run: Callable[[Path], Outcome]


@dataclass(frozen=True)
class Table:
    """A CSV artifact: normalized column names, original header text, and one dict per data row."""

    path: str
    header: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]

    def require(self, *columns: str) -> None:
        missing = [column for column in columns if column not in self.columns]
        if missing:
            names = ", ".join(f"`{column}`" for column in missing)
            raise ArtifactProblem(f"{self.path} has no {names} column; its header reads `{self.header}` "
                                  "(name each column exactly as the task lists it)")


# Reading artifacts

def read_text(root: Path, name: str) -> str:
    """The artifact's text; a byte-order mark, UTF-16, or Windows encoding is accepted."""
    path = root / OUTPUT / name
    if path.is_symlink() or not path.is_file():
        raise ArtifactProblem(f"output/{name} is missing; expected it committed in the output/ folder on main "
                              "(it was never saved there under this name, it was saved but not committed and pushed, or the "
                              "notebook stopped at an earlier error or failed assert; find the first red cell)")
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def trim_end(cell: str, width: int) -> str:
    """The cell without up to `width` of the spaces it ends with."""
    spaces = min(width, len(cell) - len(cell.rstrip()))
    return cell[:len(cell) - spaces]


def column_name(text: str) -> str:
    """A header cell for comparison: case is ignored, and spaces or hyphens read as underscores."""
    return re.sub(r"[\s-]+", "_", text.strip().strip('"').strip().casefold())


# Each artifact is parsed once per grading run: grade_submission clears these caches before it starts.
@lru_cache(maxsize=None)
def read_table(root: Path, name: str, index_name: str | None = None, value_name: str | None = None) -> Table:
    """Parse a CSV artifact, keyed by column name so column order never matters.

    Blank lines are skipped, whitespace after a row's last field is dropped, and
    a leading unnamed index column (`to_csv` without `index=False`) is ignored.
    With `index_name`, that column is read as `index_name` instead when the
    header lacks it, as when a Series of labeled values is saved with `to_csv`.
    With `value_name`, the second of exactly two columns is read as
    `value_name` when its header is blank or `0`, as a Series saved without a
    name writes it.

    Cells may be separated by commas, semicolons, or tabs: whichever splits the
    header line into the most cells is used, and a tie keeps commas. A file
    separated by semicolons may write decimal commas. Spaces the header line
    puts after every separator, as `", ".join(...)` writes, or before every
    separator, as columns padded to line up do, are part of the separator, so
    the data rows lose them too; a cell's other spaces are its own and count.
    """
    text = read_text(root, name).replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in text:
        raise ArtifactProblem(f"output/{name} contains embedded NUL bytes; expected CSV text "
                              "(save the table again with to_csv(path, index=False))")
    try:
        first = next((line for line in text.split("\n") if line.strip()), "")
        delimiter = max(DELIMITERS, key=lambda mark: len(next(csv.reader([first], delimiter=mark), [])))
        labels = next(csv.reader([first], delimiter=delimiter), [])
        padded = len(labels) > 1 and all(cell[:1].isspace() for cell in labels[1:])
        # The spaces every header cell but the last ends with, which padded columns put before each separator.
        trailing = min(len(cell) - len(cell.rstrip()) for cell in labels[:-1]) if len(labels) > 1 else 0
        rows = [[trim_end(cell, trailing) for cell in row] if trailing else row
                for row in csv.reader(io.StringIO(text), delimiter=delimiter, skipinitialspace=padded)
                if any(cell.strip() for cell in row)]
    except csv.Error as error:
        raise ArtifactProblem(f"output/{name} cannot be read as a CSV file ({error}); "
                              "save it again with to_csv(path, index=False)") from None
    if not rows:
        raise ArtifactProblem(f"output/{name} is empty; expected a header line and data rows")
    if delimiter == ";":
        rows = [[DECIMAL_COMMA.sub(r"\1.\2", cell) for cell in row] for row in rows]
    header = [column_name(cell) for cell in rows[0]]
    header_text = ",".join(cell.strip() for cell in rows[0])
    raw_header = list(rows[0])
    body = rows[1:]
    if value_name and len(header) == 2 and header[1] in ("", "0") and value_name != header[0]:
        header[1] = value_name
    if len(header) > 1 and UNNAMED_INDEX.match(header[0]):
        if index_name and index_name not in header[1:]:
            header[0] = index_name
        else:
            header, raw_header = header[1:], raw_header[1:]
            body = [row[1:] for row in body]
    # Headers that differ only in case or spacing, as pivot() writes for unnormalized labels such as
    # `Baseline` beside `baseline`: the column written exactly as the name is the one read.
    for name in {column for column in header if column and header.count(column) > 1}:
        exact = [index for index, column in enumerate(header) if column == name and raw_header[index] == name]
        if len(exact) == 1:
            header = ["" if column == name and index != exact[0] else column for index, column in enumerate(header)]
    records = []
    for row in body:
        if row:
            row = row[:-1] + [row[-1].rstrip()]
        cells = row + [""] * (len(header) - len(row))
        record = {}
        for column, cell in zip(header, cells):
            if not column:
                continue
            if column in record and not same_cell(record[column], cell, column):
                cell = f"conflicting repeated {column} columns ({record[column]!r} and {cell!r}; remove the extra column)"
            record[column] = cell
        records.append(record)
    return Table(f"output/{name}", header_text, tuple(dict.fromkeys(column for column in header if column)),
                 tuple(records))


# Comparing values

def same_cell(left: str, right: str, column: str = "") -> bool:
    """Whether repeated evidence agrees after the accepted formatting variations."""
    return (label(left) == label(right) or
            (number(left) is not None and number(left) == number(right)) or
            (is_missing(left) and is_missing(right)) or
            bool(calendar_dates(left) & calendar_dates(right)))


@lru_cache(maxsize=None)
def label(text: str) -> str:
    """A label for comparison: case, spacing, `-` or `_` for a space, and a final full stop are ignored."""
    text = DASHES.sub(" ", text.casefold())
    text = COMMA_SPACES.sub(",", text)
    return " ".join(text.split()).strip(" .")


def similarity(first: str, second: str) -> float:
    """How alike two labels are, from 0 to 1 (difflib's ratio of the normalized texts)."""
    return SequenceMatcher(None, label(first), label(second)).ratio()


def match_labels(rows: Sequence[dict[str, str]], column: str, expected: Sequence[str],
                 aliases: dict[str, tuple[str, ...]] | None = None) -> dict[str, dict[str, str]]:
    """Each expected label's row, found by its label or an alias, then by a close spelling, then by position.

    A row is used once, contradictory repeats remain incorrect evidence, and position is used only when the file has
    exactly one row per expected label, as when every label is in the task's order but one is reworded.
    """
    aliases = aliases or {}
    names = [label(row.get(column, "")) for row in rows]
    found: dict[str, dict[str, str]] = {}
    used: set[int] = set()
    for wanted in expected:
        spellings = {label(wanted), *(label(alias) for alias in aliases.get(wanted, ()))}
        index = next((index for index, name in enumerate(names) if name in spellings and index not in used), None)
        if index is not None:
            found[wanted] = dict(rows[index])
            for other, name in enumerate(names):
                if other == index or name not in spellings:
                    continue
                for key, cell in rows[other].items():
                    if key != column and key in found[wanted] and not same_cell(found[wanted][key], cell):
                        found[wanted][key] = (f"conflicting repeated {wanted} rows "
                                              f"({found[wanted][key]!r} and {cell!r}; keep one value)")
            used.add(index)
    pairs = sorted(((SequenceMatcher(None, label(wanted), name).ratio(), position, index)
                    for position, wanted in enumerate(expected) if wanted not in found
                    for index, name in enumerate(names) if index not in used),
                   key=lambda pair: (-pair[0], pair[1], pair[2]))
    for ratio, position, index in pairs:
        if ratio >= SIMILAR and expected[position] not in found and index not in used:
            found[expected[position]] = rows[index]
            used.add(index)
    if len(rows) == len(expected):
        for position, wanted in enumerate(expected):
            if wanted not in found and position not in used:
                found[wanted] = rows[position]
                used.add(position)
    return found


def unwrap(text: str) -> str:
    """The text of a number, without a NumPy scalar repr around it."""
    text = text.strip()
    wrapped = NUMPY_SCALAR.match(text)
    return wrapped.group(1).strip() if wrapped else text


def number(text: str) -> float | None:
    try:
        value = float(unwrap(text))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def leading_number(text: str) -> float | None:
    match = LEADING_NUMBER.match(unwrap(text))
    return number(match.group()) if match else None


def is_missing(cell: str, spellings: frozenset[str] = VALUE_MISSING) -> bool:
    return cell.strip().casefold() in spellings




def calendar_dates(cell: str) -> set[date]:
    """The day a written date or timestamp names, as written and, when zoned, in UTC.

    Any unambiguous way of writing the day is read: 2026-01-15, 2026/1/15,
    20260115, 15 January 2026, Jan 15, 2026, and 01/15/2026 or 15/01/2026
    (a numeric date with the year last only when its parts settle which is the
    day). `visit_date` is a calendar day, so either
    reading of a zoned timestamp is accepted rather than guessing which one the
    student meant.
    """
    text = WEEKDAY.sub("", cell.strip())
    match = next((found for found in (form.match(text) for form in DATE_FORMS) if found), None)
    if not match:
        return set()
    parts = match.groupdict()
    if parts.get("name") is not None:
        month = MONTHS.get(parts["name"].casefold())
        day = int(parts["day"])
    elif parts.get("first") is not None:
        first, second = int(parts["first"]), int(parts["second_part"])
        if first == second or first > 12:
            day, month = first, second
        elif second > 12:
            month, day = first, second
        else:
            return set()
    else:
        month, day = int(parts["month"]), int(parts["day"])
    if month is None:
        return set()
    try:
        moment = datetime(int(parts["year"]), month, day, int(parts["hour"] or 0), int(parts["minute"] or 0),
                          int(parts["second"] or 0))
    except ValueError:
        return set()
    days = {moment.date()}
    zone = parts["zone"]
    if zone:
        digits = "" if zone.upper() in ("Z", "UTC", "GMT") else zone[1:].replace(":", "")
        offset = timedelta(hours=int(digits[:2] or 0), minutes=int(digits[2:] or 0))
        days.add((moment + offset if zone.startswith("-") else moment - offset).date())
    return days


def shown(value: object) -> str:
    """A value as the feedback quotes it."""
    if value is None:
        return "blank (missing)"
    if isinstance(value, str):
        return "blank" if not value.strip() else f"'{value}'"
    return str(value)




# Task 1: the file inventory and the table profile

def file_key(text: str) -> str:
    """A data file's name for comparison: `data/patients.parquet`, `patients.csv`, and `Patients` are the same."""
    name = label(text).replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"\.(csv|parquet)$", "", name)


def check_file_inventory(root: Path) -> Outcome:
    text = read_text(root, "file_inventory.txt")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    outcome = Outcome(total=len(INVENTORY))
    for name, expected in INVENTORY.items():
        found = next((line for line in lines if re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])",
                                                           line.casefold())), None)
        if found is None:
            outcome.problems.append(f"no line names {name}; expected `{expected} data/{name}` as `wc -l data/*.csv` "
                                    "prints it (a later `>` may have replaced the file: write it with one command)")
            continue
        value = leading_number(found)
        if value == expected:
            outcome.right += 1
        else:
            cause = (" (that is the data rows without the header; wc -l counts every line)" if value == expected - 1
                     else "")
            outcome.problems.append(f"{name} should show {expected} lines, found {shown(found)}{cause}; save "
                                    "exactly what `wc -l data/*.csv` prints")
    return outcome


def keyed(rows: Sequence[dict[str, str]], column: str) -> dict[str, dict[str, str]]:
    """The first row for each normalized label in `column`."""
    found: dict[str, dict[str, str]] = {}
    for row in rows:
        found.setdefault(file_key(row.get(column, "")), row)
    return found


def check_table_profile(root: Path) -> Outcome:
    table = read_table(root, "table_profile.csv", index_name="table")
    table.require("table", *PROFILE_COLUMNS)
    rows = keyed(table.rows, "table")
    outcome = Outcome(total=len(TABLE_PROFILE))
    for name, expected in TABLE_PROFILE.items():
        row = rows.get(file_key(name))
        if row is None:
            outcome.problems.append(f"no row for {name} (name each table by its file name, such as {name})")
            continue
        wrong = [(column, want, row[column]) for column, want in zip(PROFILE_COLUMNS, expected)
                 if number(row[column]) != want]
        if not wrong:
            outcome.right += 1
            continue
        column, want, found = wrong[0]
        outcome.problems.append(f"{name} {column} should be {want}, found {shown(found)} ({PROFILE_HINTS[column]})")
    return outcome


# Task 2: the audit

def audit_rows(root: Path) -> dict[str, dict[str, str]]:
    table = read_table(root, "issue_audit.csv", index_name="issue", value_name="count")
    table.require("issue", "count")
    return match_labels(table.rows, "issue", tuple(issue for issue, _ in ISSUE_AUDIT))


def own_rule_summaries(root: Path, name: str, part: str) -> list[dict[str, object]]:
    """The summaries the wrong rule behind the student's own issue_audit.csv count gives (see OWN_RULES)."""
    issue, options = OWN_RULES.get((name, part), ("", []))
    if not options:
        return []
    try:
        row = audit_rows(root).get(issue)
    except ArtifactProblem:
        return []
    count = number(row["count"]) if row else None
    return [summary for found, summaries in options if found == count for summary in summaries]


def check_issue_audit(root: Path) -> Outcome:
    rows = audit_rows(root)
    outcome = Outcome(total=len(ISSUE_AUDIT))
    for issue, expected in ISSUE_AUDIT:
        row = rows.get(issue)
        if row is None:
            outcome.problems.append(f"no row for the issue '{issue}' (copy each label from Task 2.3's block)")
        elif number(row["count"]) == expected:
            outcome.right += 1
        else:
            outcome.problems.append(
                f"'{issue}' should count {expected}, found {shown(row['count'])} ({ISSUE_HINTS[issue]}; count it "
                "in the raw tables, before duplicates are removed)")
    return outcome


# Tasks 3 to 5: keyed tables

Key = tuple


def text_right(expected: object, cell: str) -> bool:
    if expected is None:
        return is_missing(cell, TEXT_MISSING)
    return label(cell) == label(str(expected))


def number_right(expected: object, cell: str) -> bool:
    if expected is None:
        return is_missing(cell)
    value = number(cell)
    return value is not None and abs(value - float(expected)) < 1e-6










def normal_key(key: Key) -> Key:
    return tuple(label(part) for part in key)


def score_rows(grouped: dict[Key, list[dict[str, str]]], expected: Sequence[Key], columns: Sequence[str],
               ignore: Callable[[dict[str, str]], bool] = lambda row: False,
               hints: dict[str, str] | None = None, charge_missing: bool = True) -> Outcome:
    """Each expected key present as often as `expected` lists it (once, or as often as the student's own earlier
    table holds it); a repeated or unexpected row cancels a matched one. With `charge_missing` false, an earlier
    summary already charged the dropped rows, so missing keys cost nothing. `hints` adds advice to the `missing`,
    `repeated`, or `unexpected` problem."""
    wanted: dict[Key, Key] = {}
    times: dict[Key, int] = {}
    for key in expected:
        wanted.setdefault(normal_key(key), key)
        times[normal_key(key)] = times.get(normal_key(key), 0) + 1
    missing = [key for norm, key in wanted.items() if charge_missing and norm not in grouped]
    repeated = [key for norm, key in wanted.items() if len(grouped.get(norm, ())) > times[norm]]
    # Unexpected rows are listed by their key cells exactly as the file writes them.
    unexpected = [tuple(row[column] for column in columns) for key, rows in grouped.items() if key not in wanted
                  for row in rows if not ignore(row)]
    outcome = Outcome(total=len(wanted), right=max(0, len(wanted) - len(missing) - len(repeated)
                                                   - len(unexpected)))
    hints = hints or {}

    def listed(keys: Sequence[Key]) -> str:
        names = [" ".join(part if part.strip() else "(blank)" for part in key) for key in keys[:4]]
        return ", ".join(names) + (f" and {len(keys) - 4} more" if len(keys) > 4 else "")

    for kind, keys, text in (("missing", missing, f"{len(missing)} expected row(s) missing, first {listed(missing)}"),
                             ("repeated", repeated, f"{listed(repeated)} appear(s) more than once"),
                             ("unexpected", unexpected,
                              f"{len(unexpected)} row(s) not expected, first {listed(unexpected)}")):
        if keys:
            outcome.problems.append(text + (f" ({hints[kind]})" if kind in hints else ""))
    return outcome


def best_rows(grouped: dict[Key, list[dict[str, str]]], options: Sequence[Sequence[Key]],
              hint: str | dict[str, str], columns: Sequence[str], **kwargs) -> Outcome:
    """The row score against whichever accepted key set (the expected one, or the student's own) fits best.

    A text `hint` follows the last problem; a dict gives each kind of problem its own (see score_rows).
    """
    hints = hint if isinstance(hint, dict) else None
    outcomes = [score_rows(grouped, keys, columns, hints=hints, **kwargs) for keys in options if keys]
    best = max(outcomes, key=lambda outcome: (outcome.right / max(outcome.total, 1), -len(outcome.problems)))
    if best.problems and hints is None:
        best.problems[-1] += f" ({hint})"
    return best










VISIT_KEYS = ("patient_id", "visit")


# Labeled summaries: one row per metric, read by its label

def read_metrics(root: Path, name: str, labels: Sequence[str]) -> dict[str, str]:
    """Each expected metric's value cell, found by its label (see match_labels)."""
    table = read_table(root, name, index_name="metric", value_name="value")
    table.require("metric", "value")
    return {wanted: row["value"] for wanted, row in match_labels(table.rows, "metric", labels).items()}


def own_metrics(root: Path, name: str, labels: Sequence[str]) -> dict[str, float]:
    """The student's own summary numbers, by label, or {} when the file cannot be read."""
    try:
        cells = read_metrics(root, name, labels)
    except ArtifactProblem:
        return {}
    return {wanted: value for wanted, value in ((wanted, number(cell)) for wanted, cell in cells.items())
            if value is not None}


def score_metrics(cells: dict[str, str], labels: Sequence[str], right: Callable[[str, str], bool],
                  expected: Callable[[str], object], hint: Callable[[str], str], task: str) -> Outcome:
    outcome = Outcome(total=len(labels))
    for wanted in labels:
        cell = cells.get(wanted)
        if cell is None:
            outcome.problems.append(f"no row for '{wanted}' (copy each label from {task}'s block)")
        elif right(wanted, cell):
            outcome.right += 1
        else:
            want = expected(wanted)
            want = want.isoformat() if isinstance(want, date) else f"{want:.2f}" if isinstance(want, float) else want
            outcome.problems.append(f"'{wanted}' should be {want}, found {shown(cell)} ({hint(wanted)})")
    return outcome


def close(value: float | None, want: float, tolerance: float) -> bool:
    return value is not None and abs(value - want) <= tolerance


# Task 3.1: patients_summary.csv and gender_categories.csv

PATIENT_HINTS = {
    "patients": "len(patients_clean) after drop_duplicates() removes the exact repeats in patients.parquet",
    "birth_year missing": "patients_clean['birth_year'].isna().sum() once 9999 is missing",
    "birth_year min": "patients_clean['birth_year'].min()",
    "birth_year max": "patients_clean['birth_year'].max(); 9999 here means the sentinel was never made missing",
    "site_code values": "patients_clean['site_code'].nunique(); every spelling left over, such as N-01, counts "
                        "as its own code (uppercase, then remove every space and hyphen)",
    "patients at N01": "(patients_clean['site_code'] == 'N01').sum() once every spelling is normalized",
    "patients at S02": "(patients_clean['site_code'] == 'S02').sum() once every spelling is normalized",
    "patients at E04": "(patients_clean['site_code'] == 'E04').sum() once every spelling is normalized",
}


def patients_summary(part: str) -> Callable[[Path], Outcome]:
    def check(root: Path) -> Outcome:
        labels = PATIENT_PARTS[part]
        cells = read_metrics(root, "patients_summary.csv", labels)
        # A site rule the audit already charged is right here too.
        own = own_rule_summaries(root, "patients_summary.csv", part)

        def right(wanted: str, cell: str) -> bool:
            # Counted with the exact repeats left in is right too: the patients count charges that once.
            accepted = {PATIENT_SUMMARY[wanted]} if part == "rows" else {PATIENT_SUMMARY[wanted],
                                                                         PATIENT_SUMMARY_KEPT[wanted]}
            return number(cell) in accepted | {summary[wanted] for summary in own}

        return score_metrics(cells, labels, right, PATIENT_SUMMARY.get, PATIENT_HINTS.get, "Task 3.1")
    return check


def gender_table(root: Path) -> Table:
    table = read_table(root, "gender_categories.csv", index_name="gender", value_name="count")
    table.require("gender", "count")
    return table


def code_of(cell: str) -> str | None:
    return None if is_missing(cell, TEXT_MISSING) else label(cell)


def check_gender_counts(root: Path) -> Outcome:
    """Each code's count, from the deduplicated registry or (charged once in the patients count) the full one."""
    table = gender_table(root)
    outcome = Outcome(total=len(GENDER_COUNTS))
    for code, expected in GENDER_COUNTS.items():
        key = code_of(code or "")
        # Rows that differ only in case, such as F and f, are one code: their counts add up.
        found = [number(row["count"]) for row in table.rows if code_of(row["gender"]) == key]
        count = None if None in found else sum(found)
        if found and count in {expected, GENDER_COUNTS_KEPT[code]}:
            outcome.right += 1
            continue
        others = sorted({row["gender"].strip() for row in table.rows
                         if code_of(row["gender"]) not in (*(label(c) for c in GENDER_CODES), None)})
        cause = (f"; rows for {', '.join(repr(other) for other in others[:4])} mean some spellings never mapped "
                 "to F, M, or X (strip and lowercase, then Series.map with a dictionary)" if others else
                 "; a spelling left out of the dictionary, or one with spaces left on because .str.strip() was "
                 "skipped, becomes missing, so strip and lowercase, then list every spelling with value_counts(): "
                 "female ones are F, male M, nonbinary X, and empty or unknown missing")
        outcome.problems.append(
            f"gender {code or 'missing'} should count {expected}, found "
            f"{'no row' if not found else 'blank' if count is None else f'{count:g}'} "
            f"(save value_counts(dropna=False) of the categorical gender column; dropna=False keeps the missing "
            f"count{cause})")
    return outcome


def check_gender_codes(root: Path) -> Outcome:
    """Only a categorical column has `.cat.codes`: each code its own whole number from 0, missing -1."""
    table = gender_table(root)
    if "code" not in table.columns:
        raise ArtifactProblem(f"{table.path} has no `code` column; its header reads `{table.header}` (convert gender "
                              "with astype(\"category\"), then add gender_counts[\"code\"] = "
                              "gender_counts[\"gender\"].cat.codes)")
    codes: dict[str | None, set[float | None]] = {}
    for row in table.rows:
        codes.setdefault(code_of(row["gender"]), set()).add(number(row["code"]))
    shared = [code for key, found in codes.items() if key is not None for code in found]
    outcome = Outcome(total=len(GENDER_COUNTS))
    for code in GENDER_COUNTS:
        key = code_of(code or "")
        numbers = codes.get(key, set())
        if key is None:
            right = numbers <= {-1}
        else:
            right = all(n is not None and n >= 0 and n == int(n) and shared.count(n) == 1 for n in numbers)
        if right:
            outcome.right += 1
            continue
        want = "-1" if key is None else "a whole number from 0 that no other code shares"
        seen = ", ".join("blank" if n is None else f"{n:g}" for n in sorted(numbers, key=str))
        outcome.problems.append(f"gender {code or 'missing'} should have code {want}, found {seen} (fill the code "
                                "column from .cat.codes of the categorical gender column)")
    return outcome


# Task 3.2: visits_summary.csv

VISIT_HINTS = {
    "visits": "len(visits_clean): drop the exact repeats, map every label, then keep one reading per patient_id "
              "and visit pair with drop_duplicates(subset=['patient_id', 'visit'], keep='last')",
    "baseline visits": "strip and lowercase each visit label, then map it to baseline or week12 with a dictionary; "
                       "baseline visits + week12 visits equals visits when every label maps",
    "week12 visits": "map every week 12 spelling, such as wk12 or ' week12', to week12",
    "visit_date missing": "keep only exact YYYY-MM-DD text for a date on the calendar: .str.fullmatch() first, "
                          "then to_datetime(..., format='%Y-%m-%d', errors='coerce')",
    "visit_date min": "visits_clean['visit_date'].min() after parsing",
    "visit_date max": "visits_clean['visit_date'].max() after parsing",
    "sbp_mmhg missing": "values outside 70 through 250, 999 included, become missing",
    "sbp_mmhg min": "70 is inside the range and stays",
    "sbp_mmhg max": "250 is inside the range and stays",
    "weight_kg missing": "rename wt_kg to weight_kg before concat, or every south weight is missing",
    "weight_kg mean": "visits_clean['weight_kg'].mean() with both files' weights converted to numbers",
    "visits from visits_north.csv": "(visits_clean['source_file'] == 'visits_north.csv').sum(), the column added "
                                    "before stacking",
    "visits from visits_south.csv": "(visits_clean['source_file'] == 'visits_south.csv').sum()",
}
VISIT_TASKS = {"rows": "Task 2.2 (stacking) and Task 3.2 (cleaning)", "visit_date": "Task 3.2",
               "sbp_mmhg": "Task 3.2", "weight_kg": "Task 2.2 (stacking) and Task 3.2",
               "source_file": "Task 2.2 (stacking) and Task 3.2"}


def visit_row_gap(root: Path) -> float:
    """How far the student's own `visits` count is from the expected one, 0 when it matches or is unreadable."""
    own = own_metrics(root, "visits_summary.csv", ("visits",)).get("visits")
    return 0 if own is None else abs(own - VISIT_SUMMARY["visits"])


def visit_rows_charged(root: Path) -> bool:
    """True when visits_summary.csv: rows already charges dropped visits (`visits` below the expected count, or the
    file unreadable), so later extracts may miss them; an over-count explains no missing row."""
    try:
        outcome = visits_summary("rows")(root)
    except ArtifactProblem:
        return True
    own = own_metrics(root, "visits_summary.csv", ("visits",)).get("visits")
    return outcome.right < outcome.total and (own is None or own < VISIT_SUMMARY["visits"])


def visits_summary(part: str) -> Callable[[Path], Outcome]:
    def check(root: Path) -> Outcome:
        labels = VISIT_PARTS[part]
        cells = read_metrics(root, "visits_summary.csv", labels)
        # Rows the student kept or dropped by mistake are charged once, in the rows part: each one can move a
        # later count by at most one and a mean by at most one row's share.
        gap = 0 if part == "rows" else visit_row_gap(root)
        dropped = own_metrics(root, "visits_summary.csv", ("visits",)).get("visits", VISIT_SUMMARY["visits"]) \
            < VISIT_SUMMARY["visits"]
        # A visit label or date rule the audit already charged is right here too.
        own = own_rule_summaries(root, "visits_summary.csv", part)

        def right(wanted: str, cell: str) -> bool:
            if part == "rows":
                return number(cell) in {summary[wanted] for summary in (VISIT_SUMMARY, *own)}
            return any(fits(wanted, cell, summary[wanted])
                       for summary in (VISIT_SUMMARY, VISIT_SUMMARY_EARLIER, *own))

        def fits(wanted: str, cell: str, want: object) -> bool:
            if isinstance(want, date):
                return want in calendar_dates(cell)
            if wanted == "weight_kg mean":
                return close(number(cell), want, MEAN_TOLERANCE + gap * WEIGHT_SPAN / VISIT_SUMMARY["visits"])
            if wanted in ("sbp_mmhg min", "sbp_mmhg max") and dropped:
                # Dropped rows may take the boundary readings with them; any value inside 70 through 250 fits.
                value = number(cell)
                return value is not None and SBP_RANGE[0] <= value <= SBP_RANGE[1] and (
                    value >= want if wanted.endswith("min") else value <= want)
            if wanted.endswith(("min", "max")):
                return number(cell) == want
            return close(number(cell), want, gap)

        hints = dict(VISIT_HINTS)
        counts = {name: number(cell) for name, cell in cells.items()}
        if part == "rows" and all(counts.get(name) is not None and counts[name] < VISIT_SUMMARY[name]
                                  for name in labels) \
                and counts["baseline visits"] + counts["week12 visits"] == counts["visits"]:
            # Every label maps, yet every count is low: rows were dropped, not left unmapped.
            for name in ("baseline visits", "week12 visits"):
                hints[name] = ("every label maps but visits is low too, so rows were dropped: "
                               "drop_duplicates(subset=['patient_id', 'visit'], keep='last') keeps one reading of each "
                               "rechecked pair where keep=False drops both, and dropna() on visits_clean drops every "
                               "visit with a missing value")
        return score_metrics(cells, labels, right, VISIT_SUMMARY.get, hints.get, "Task 3.2")
    return check


# Task 4.1: a1c_summary.csv

def a1c_summary_rows(root: Path) -> dict[str, dict[str, str]]:
    """The summary row for each visit label, matched by the label itself: a1c_baseline is not baseline."""
    table = read_table(root, "a1c_summary.csv", index_name="visit")
    table.require("visit", "results", "a1c_pct_mean")
    found: dict[str, dict[str, str]] = {}
    for row in table.rows:
        found.setdefault(label(row["visit"]), row)
    return {visit: found[visit] for visit in VISIT_LABELS if visit in found}


def check_a1c_results(root: Path) -> Outcome:
    rows = a1c_summary_rows(root)
    outcome = Outcome(total=len(VISIT_LABELS))
    for visit in VISIT_LABELS:
        want = A1C_SUMMARY[visit][0]
        row = rows.get(visit)
        if row is None:
            outcome.problems.append(f"no row for visit {visit} (melt a1c_baseline and a1c_week12, then map the "
                                    "labels to baseline and week12)")
        elif number(row["results"]) == want:
            outcome.right += 1
        else:
            outcome.problems.append(f"{visit} should count {want} results, found {shown(row['results'])} (melt with "
                                    "var_name='visit', map each label to its own visit, and drop the rows with no "
                                    "result before counting)")
    return outcome


def check_a1c_means(root: Path) -> Outcome:
    """Each visit's mean, or the mean of the results the row counts when the labels were swapped (charged once)."""
    rows = a1c_summary_rows(root)
    outcome = Outcome(total=len(VISIT_LABELS))
    for visit in VISIT_LABELS:
        row = rows.get(visit)
        if row is None:
            outcome.right += 1  # the results check charges a missing row
            continue
        mean = number(row["a1c_pct_mean"])
        other = VISIT_LABELS[1 - VISIT_LABELS.index(visit)]
        if close(mean, A1C_SUMMARY[visit][1], MEAN_TOLERANCE) or (
                number(row["results"]) == A1C_SUMMARY[other][0]
                and close(mean, A1C_SUMMARY[other][1], MEAN_TOLERANCE)):
            outcome.right += 1
        else:
            outcome.problems.append(f"{visit} a1c_pct_mean should be {A1C_SUMMARY[visit][1]:.2f}, found "
                                    f"{shown(row['a1c_pct_mean'])} (the mean of the {visit} a1c_pct values, "
                                    "rounded to two decimals)")
    return outcome


def a1c_summary_differs(root: Path) -> bool:
    """Whether the student's own A1c summary already shows a charged mistake."""
    try:
        rows = a1c_summary_rows(root)
    except ArtifactProblem:
        return False
    return any(number(rows[visit]["results"]) != A1C_SUMMARY[visit][0] if visit in rows else True
               for visit in VISIT_LABELS)


def own_a1c_total(root: Path) -> float | None:
    """Every result the student's own A1c summary counts, whatever its labels."""
    try:
        table = read_table(root, "a1c_summary.csv", index_name="visit")
    except ArtifactProblem:
        return None
    values = [number(row.get("results", "")) for row in table.rows]
    return None if not values or None in values else sum(values)


# Task 4.2: merge_audit.csv

def check_merge_audit(root: Path) -> Outcome:
    """Each count against the reference, or, where an earlier summary already shows a charged mistake, against
    the student's own summaries: both + left_only is their visits, both + right_only their A1c results."""
    table = read_table(root, "merge_audit.csv", index_name="check", value_name="count")
    table.require("check", "count")
    rows = match_labels(table.rows, "check", tuple(name for name, _ in MERGE_AUDIT))
    found = {name: number(row["count"]) for name, row in rows.items()}
    visits = own_metrics(root, "visits_summary.csv", VISIT_PARTS["rows"])
    patients = own_metrics(root, "patients_summary.csv", ("patients", "patients at N01", "patients at S02"))
    own_visits, own_labs = visits.get("visits"), own_a1c_total(root)
    if own_labs == 0:  # labels left unmapped count no baseline or week12 result, so every result is right_only
        own_labs = len(A1C)
    upstream = (any(visits.get(name) != VISIT_SUMMARY[name] for name in VISIT_PARTS["rows"])
                or a1c_summary_differs(root)) and None not in (own_visits, own_labs)
    both, left, right = (found.get(name) for name, _ in MERGE_AUDIT[:3])
    visit_gap = abs(own_visits - VISIT_SUMMARY["visits"]) if own_visits is not None else 0
    patient_gap = abs(patients["patients"] - PATIENT_SUMMARY["patients"]) if "patients" in patients else 0
    own: dict[str, bool] = {}
    if upstream:
        # Repeated visits (rechecks kept) each match the same A1c row, so both + right_only exceeds the results.
        own["visits with an A1c result"] = both is not None and None not in (left, right) and (
            both + left == own_visits and (both + right == own_labs or own_visits > VISIT_SUMMARY["visits"]))
        own["visits without an A1c result"] = None not in (both, left) and both + left == own_visits
        own["A1c results without a visit"] = None not in (both, right) and both + right == own_labs
    reference = dict(MERGE_AUDIT)
    value = found.get("visits whose patient is not in the registry")
    own["visits whose patient is not in the registry"] = close(
        value, reference["visits whose patient is not in the registry"], visit_gap)
    value = found.get("registry patients with no visit")
    own["registry patients with no visit"] = close(value, reference["registry patients with no visit"],
                                                   visit_gap + patient_gap)
    if all(name in patients for name in ("patients", "patients at N01", "patients at S02")):
        own["registry patients whose site is not in the site list"] = found.get(
            "registry patients whose site is not in the site list") == (
            patients["patients"] - patients["patients at N01"] - patients["patients at S02"])
    outcome = Outcome(total=len(MERGE_AUDIT))
    for name, expected in MERGE_AUDIT:
        row = rows.get(name)
        if row is None:
            outcome.problems.append(f"no row for '{name}' (copy each label from Task 4.2's block)")
        elif found[name] == expected or own.get(name):
            outcome.right += 1
        else:
            outcome.problems.append(f"'{name}' should count {expected}, found {shown(row['count'])} "
                                    f"({MERGE_HINTS[name]})")
    return outcome


# Task 5.2: analysis_sample.csv, the joined rows for SAMPLE_IDS

def sample_key(row: dict[str, str]) -> Key:
    """A joined row's key; a visit label left unmapped is read as its visit, charged once in visits_summary."""
    visit = label(row["visit"])
    return (label(row["patient_id"]), VISIT_MAP.get(visit, visit))


@lru_cache(maxsize=None)
def read_sample(root: Path, name: str, keys: Sequence[str]) -> tuple[Table, dict[Key, list[dict[str, str]]]]:
    table = read_table(root, name)
    table.require(*keys)
    if not table.rows:
        raise ArtifactProblem(f"{table.path} has a header but no data rows")
    grouped: dict[Key, list[dict[str, str]]] = {}
    for row in table.rows:
        grouped.setdefault(sample_key(row) if len(keys) == 2 else (label(row[keys[0]]),), []).append(row)
    return table, grouped


def visit_rows_differ(root: Path) -> bool:
    visits = own_metrics(root, "visits_summary.csv", VISIT_PARTS["rows"])
    return bool(visits) and any(visits.get(name) != VISIT_SUMMARY[name] for name in VISIT_PARTS["rows"])


def analysis_rows(root: Path) -> Outcome:
    _, grouped = read_sample(root, "analysis_sample.csv", VISIT_KEYS)
    expected = SAMPLE_VISITS
    ignore: Callable[[dict[str, str]], bool] = lambda row: False  # noqa: E731
    if visit_rows_differ(root):
        # A recheck or exact repeat kept in visits_clean is charged there; its pair may repeat here. A label
        # left unmapped is charged there too: its row has no visit here, and its pair is not expected when the
        # extract holds that patient's row with an unmapped label.
        repeated = {normal_key(pair) for pair in REPEATED_PAIRS}
        grouped = {key: rows[:1] if key in repeated else rows for key, rows in grouped.items()}
        unmapped = {key[0] for key, rows in grouped.items() for row in rows if label(row["visit"]) not in VISIT_MAP}
        expected = [key for key in SAMPLE_VISITS if key not in RELABELED_PAIRS or label(key[0]) not in unmapped]
        relabeled = {normal_key(pair) for pair in RELABELED_PAIRS}
        ignore = lambda row: label(row["visit"]) not in VISIT_MAP or sample_key(row) in relabeled  # noqa: E731
    if own_metrics(root, "patients_summary.csv", ("patients",)).get("patients", PATIENT_SUMMARY["patients"]) != \
            PATIENT_SUMMARY["patients"]:
        # Exact registry repeats kept are charged once in patients_summary.csv; their visits may repeat here.
        repeated = {label(pid) for pid in REPEATED_PATIENTS}
        grouped = {key: rows[:1] if key[0] in repeated else rows for key, rows in grouped.items()}
    # Visits dropped from visits_clean are charged once, in visits_summary.csv: rows.
    return best_rows(grouped, [expected], {
        "missing": "keep every visit of the SAMPLE_IDS patients: an inner or right merge drops the visits with no "
                   "match, so merge onto visits_clean with how='left', then filter with "
                   "analysis['patient_id'].isin(SAMPLE_IDS); rows dropped from visits_clean itself are missing "
                   "too: drop_duplicates(keep='last') keeps one reading of each recheck where keep=False drops "
                   "both, and dropna() drops every visit with a missing value",
        "repeated": "a lookup table still repeats a key, such as an exact repeat left in patients_clean or a "
                    "site_code listed twice in sites.csv; keep only current sites and pass validate= to each merge",
        "unexpected": "every row should be a visit of a SAMPLE_IDS patient from visits_clean; filter with "
                      "analysis['patient_id'].isin(SAMPLE_IDS) and merge onto visits_clean with how='left'",
    }, VISIT_KEYS, ignore=ignore, charge_missing=not visit_rows_charged(root))


def score_sample(grouped: dict[Key, list[dict[str, str]]], column: str, keys: Sequence[Key],
                 accepted: Callable[[Key, dict[str, str]], Sequence[object]], right: Callable[[object, str], bool],
                 rule: str, total_label: str = "records") -> Outcome:
    """The share of expected keys whose `column` holds an accepted value (a missing key is charged by rows)."""
    outcome = Outcome(total=len(keys))
    wrong: list[str] = []
    first = ""
    for key in keys:
        rows = grouped.get(normal_key(key))
        if not rows or any(right(value, row.get(column, "")) for row in rows for value in accepted(key, row)):
            outcome.right += 1
            continue
        wrong.append(" ".join(key))
        if not first:
            want = accepted(key, rows[0])[0]
            first = f"first {' '.join(key)}: expected {shown(want)}, found {shown(rows[0].get(column, ''))}"
    if wrong:
        others = f" (also {', '.join(wrong[1:4])}{' and more' if len(wrong) > 4 else ''})" if len(wrong) > 1 else ""
        outcome.problems.append(f"column {column}: {len(wrong)} of {len(keys)} {total_label} differ; {first}{others}; "
                                f"the rule: {rule}")
    return outcome


def check_analysis_sites(root: Path) -> Outcome:
    table, grouped = read_sample(root, "analysis_sample.csv", VISIT_KEYS)
    table.require("site_name")

    def accepted(key: Key, row: dict[str, str]) -> list[object]:
        # The name the current site list gives the row's own site_code, exactly as written, is right too.
        return [ANALYSIS[key]["site_name"], CURRENT_SITES.get(row.get("site_code", ""))]

    return score_sample(grouped, "site_name", SAMPLE_VISITS, accepted, text_right,
                        "the current site_name for the patient's site_code (keep record_status == 'current' "
                        "before merging, so a retired name never joins); missing when the patient or site is unknown",
                        "visits")


def check_analysis_a1c(root: Path) -> Outcome:
    table, grouped = read_sample(root, "analysis_sample.csv", VISIT_KEYS)
    table.require("a1c_pct")
    upstream = a1c_summary_differs(root)

    def accepted(key: Key, row: dict[str, str]) -> list[object]:
        values: list[object] = [ANALYSIS[key]["a1c_pct"]]
        if row.get("visit", "").strip() not in VISIT_LABELS:
            values.append(None)  # an unmapped visit label finds no result; visits_summary charges it
        if upstream:
            # The A1c summary already charges mislabeled or unmapped results; the join may carry them.
            other = (key[0], VISIT_LABELS[1 - VISIT_LABELS.index(key[1])])
            values += [A1C.get(other), None]
        return values

    return score_sample(grouped, "a1c_pct", SAMPLE_VISITS, accepted, number_right,
                        "the a1c_long value for the same patient_id and visit, merged on both columns; missing "
                        "when there is no result", "visits")


def check_analysis_patient(root: Path) -> Outcome:
    table, grouped = read_sample(root, "analysis_sample.csv", VISIT_KEYS)
    table.require("gender", "birth_year")
    gender_off = own_rows_differ_gender(root)
    years_off = bool(own_metrics(root, "patients_summary.csv", PATIENT_PARTS["birth_year"])) and any(
        own_metrics(root, "patients_summary.csv", PATIENT_PARTS["birth_year"]).get(name) not in (
            PATIENT_SUMMARY[name], PATIENT_SUMMARY_KEPT[name]) for name in PATIENT_PARTS["birth_year"])

    def genders(key: Key) -> list[object]:
        # A spelling left unmapped, or missing from the dictionary, is charged once in gender_categories.csv.
        values: list[object] = [ANALYSIS[key]["gender"]]
        if gender_off:
            values += [*RAW_GENDER.get(key[0], ()), None]
        return values

    def years(key: Key) -> list[object]:
        values: list[object] = [ANALYSIS[key]["birth_year"]]
        if years_off:
            values += [year for year in RAW_YEAR.get(key[0], ()) if year is not None]
        return values

    outcome = Outcome(total=len(SAMPLE_VISITS))
    wrong = []
    for key in SAMPLE_VISITS:
        rows = grouped.get(normal_key(key))
        if not rows or any(any(text_right(gender, row["gender"]) for gender in genders(key))
                           and any(number_right(year, row["birth_year"]) for year in years(key)) for row in rows):
            outcome.right += 1
        else:
            wrong.append(key)
    if wrong:
        key = wrong[0]
        row = grouped[normal_key(key)][0]
        outcome.problems.append(
            f"gender and birth_year: {len(wrong)} of {len(SAMPLE_VISITS)} visits differ; first {' '.join(key)}: "
            f"expected {shown(ANALYSIS[key]['gender'])} and {shown(ANALYSIS[key]['birth_year'])}, found "
            f"{shown(row['gender'])} and {shown(row['birth_year'])}; the rule: the patient's values from "
            "patients_clean, merged on patient_id with how='left' (missing for a patient not in the registry)")
    return outcome


def own_rows_differ_gender(root: Path) -> bool:
    """Whether the student's own gender counts already show a charged mapping mistake."""
    try:
        return check_gender_counts(root).right < len(GENDER_COUNTS)
    except ArtifactProblem:
        return False


# Task 4.3: sbp_wide_sample.csv, the pivoted rows for SAMPLE_IDS

def sbp_wide_rows(root: Path) -> Outcome:
    _, grouped = read_sample(root, "sbp_wide_sample.csv", ("patient_id",))
    return best_rows(grouped, [[(pid,) for pid in SAMPLE_WIDE]],
                     "pivot visits_clean with index='patient_id', so each patient with a visit gets one row, then "
                     "keep the rows whose patient_id is in SAMPLE_IDS; a patient whose visits were all dropped "
                     "from visits_clean has no row", ("patient_id",),
                     charge_missing=not visit_rows_charged(root))


def sbp_wide_visit(visit: str) -> Callable[[Path], Outcome]:
    def check(root: Path) -> Outcome:
        table, grouped = read_sample(root, "sbp_wide_sample.csv", ("patient_id",))
        table.require(visit)
        sbp_off = bool(own_metrics(root, "visits_summary.csv", VISIT_PARTS["sbp_mmhg"])) and any(
            own_metrics(root, "visits_summary.csv", VISIT_PARTS["sbp_mmhg"]).get(name) != VISIT_SUMMARY[name]
            for name in VISIT_PARTS["sbp_mmhg"])
        rows_off = visit_rows_differ(root)

        def accepted(key: Key, row: dict[str, str]) -> list[object]:
            values: list[object] = [SBP_WIDE[key[0]][visit]]
            if sbp_off:  # a 70 through 250 rule skipped, misapplied, or clipped is charged once in visits_summary.csv
                raw = RAW_SBP.get((key[0], visit))
                values += [raw, None, None if raw is None else min(max(raw, SBP_RANGE[0]), SBP_RANGE[1])]
            if rows_off:  # an unmapped label pivots into its own column, charged once in visits_summary.csv
                values += [None, EARLIER_SBP.get((key[0], visit))]
            return values

        return score_sample(grouped, visit, [(pid,) for pid in SAMPLE_WIDE], accepted, number_right,
                            f"the {visit} sbp_mmhg from visits_clean, the later reading of a rechecked pair "
                            "(sort_values('visit_time'), then drop_duplicates(subset=['patient_id', 'visit'], "
                            "keep='last')); missing when that visit or reading is missing" + (
                                "; visits_summary.csv's sbp_mmhg counts also differ, so check Task 3.2 step 3 "
                                "(keep 70 through 250)" if sbp_off else ""), "patients")
    return check


def check_sbp_change(root: Path) -> Outcome:
    table, grouped = read_sample(root, "sbp_wide_sample.csv", ("patient_id",))
    table.require("sbp_change")

    def accepted(key: Key, row: dict[str, str]) -> list[object]:
        values = SBP_WIDE[key[0]]
        reference = None if None in values.values() else values["week12"] - values["baseline"]
        # The change computed from the file's own baseline and week12 columns is right too.
        base, follow = (number(row.get(visit, "")) for visit in VISIT_LABELS)
        return [reference, None if base is None or follow is None else follow - base]

    return score_sample(grouped, "sbp_change", [(pid,) for pid in SAMPLE_WIDE], accepted, number_right,
                        "week12 minus baseline; missing when either is missing", "patients")

# Task 5.2: decision_log.csv

def decision_table(root: Path) -> Table:
    table = read_table(root, "decision_log.csv")
    if not table.rows:
        raise ArtifactProblem(f"{table.path} has a header but no decision rows")
    return table


def check_issues(root: Path) -> Outcome:
    table = decision_table(root)
    table.require("table", "issue")
    outcome = Outcome(total=len(DECISION_ISSUES))
    pairs = sorted(((similarity(" | ".join(wanted), " | ".join((row["table"], row["issue"]))), position, index)
                    for position, wanted in enumerate(DECISION_ISSUES) for index, row in enumerate(table.rows)),
                   key=lambda pair: (-pair[0], pair[1], pair[2]))
    chosen: dict[int, int] = {}
    for ratio, position, index in pairs:
        if ratio >= SIMILAR and position not in chosen and index not in chosen.values():
            chosen[position] = index
    for position, (table_name, issue) in enumerate(DECISION_ISSUES):
        if position in chosen:
            outcome.right += 1
            continue
        closest = next((table.rows[index] for _, wanted, index in pairs
                        if wanted == position and index not in chosen.values()), None)
        found = "" if closest is None else (f"; the closest row has table {shown(closest['table'])} and "
                                            f"issue {shown(closest['issue'])}")
        outcome.problems.append(f"no row with table '{table_name}' and issue '{issue}'{found} (add one row per "
                                "Task 3.3 issue, with table and issue copied)")
    return outcome


def filled_column(column: str, fix: str) -> Callable[[Path], Outcome]:
    """Every decision row has text in `column`; judged over the rows written, so a missing row is not charged."""
    def check(root: Path) -> Outcome:
        table = decision_table(root)
        table.require(column)
        blank = [position for position, row in enumerate(table.rows, start=1)
                 if is_missing(row[column], TEXT_MISSING)]
        outcome = Outcome(total=len(table.rows), right=len(table.rows) - len(blank))
        if blank:
            outcome.problems.append(f"{len(blank)} of {len(table.rows)} decision rows have an empty {column}, "
                                    f"first row {blank[0]} ({fix})")
        return outcome
    return check


T_PATIENTS = "Task 3.1"
T_A1C = "Task 4.1"
T_ANALYSIS = "Task 4.4 (joining) and Task 5.2 (saving the extract)"
T_WIDE = "Task 3.2 step 6 (rechecks) and Task 4.3 (pivoting and saving the extract)"
T_DECISIONS = "Task 3.3"

CHECKS = (
    Check("file_inventory.txt", 5, "Task 1.1", check_file_inventory),
    Check("table_profile.csv", 5, "Task 1.2", check_table_profile),
    Check("issue_audit.csv", 10, "Task 2.3", check_issue_audit),
    Check("patients_summary.csv: patients", 2, T_PATIENTS, patients_summary("rows")),
    Check("patients_summary.csv: birth_year", 2, T_PATIENTS, patients_summary("birth_year")),
    Check("patients_summary.csv: site_code", 2, T_PATIENTS, patients_summary("site_code")),
    Check("gender_categories.csv: counts", 2, T_PATIENTS, check_gender_counts),
    Check("gender_categories.csv: codes", 2, T_PATIENTS, check_gender_codes),
    Check("visits_summary.csv: rows", 4, VISIT_TASKS["rows"], visits_summary("rows")),
    Check("visits_summary.csv: visit_date", 3, VISIT_TASKS["visit_date"], visits_summary("visit_date")),
    Check("visits_summary.csv: sbp_mmhg", 3, VISIT_TASKS["sbp_mmhg"], visits_summary("sbp_mmhg")),
    Check("visits_summary.csv: weight_kg", 2, VISIT_TASKS["weight_kg"], visits_summary("weight_kg")),
    Check("visits_summary.csv: source_file", 2, VISIT_TASKS["source_file"], visits_summary("source_file")),
    Check("a1c_summary.csv: results", 2, T_A1C, check_a1c_results),
    Check("a1c_summary.csv: a1c_pct_mean", 3, T_A1C, check_a1c_means),
    Check("merge_audit.csv", 7, "Task 4.2", check_merge_audit),
    Check("analysis_sample.csv: rows", 3, T_ANALYSIS, analysis_rows),
    Check("analysis_sample.csv: site_name", 2, T_ANALYSIS, check_analysis_sites),
    Check("analysis_sample.csv: a1c_pct", 2, T_ANALYSIS, check_analysis_a1c),
    Check("analysis_sample.csv: gender and birth_year", 2, T_ANALYSIS, check_analysis_patient),
    Check("sbp_wide_sample.csv: rows", 2, T_WIDE, sbp_wide_rows),
    Check("sbp_wide_sample.csv: baseline", 1, T_WIDE, sbp_wide_visit("baseline")),
    Check("sbp_wide_sample.csv: week12", 1, T_WIDE, sbp_wide_visit("week12")),
    Check("sbp_wide_sample.csv: sbp_change", 2, "Task 4.3", check_sbp_change),
    Check("decision_log.csv: issues", 2, T_DECISIONS, check_issues),
    Check("decision_log.csv: decision", 1, T_DECISIONS,
          filled_column("decision", "write what you did about the issue")),
    Check("decision_log.csv: reason", 1, T_DECISIONS,
          filled_column("reason", "write one sentence on why the decision fits")),
)
MAX_SCORE = sum(check.points for check in CHECKS)
if MAX_SCORE != 75:
    raise ValueError(f"the checks total {MAX_SCORE} points; the midterm grades 75 from files")


def run_check(check: Check, root: Path) -> dict:
    try:
        outcome = check.run(root)
    except ArtifactProblem as problem:
        outcome = Outcome(total=1, problems=[str(problem)])
    total = max(outcome.total, 1)
    right = max(0, min(outcome.right, total))
    passed = right == total and outcome.total > 0
    score = check.points if passed else check.points * right // total
    detail = ""
    if not passed:
        problems = outcome.problems or ["no rows to check"]
        more = f"; and {len(problems) - 3} more" if len(problems) > 3 else ""
        detail = f"{'; '.join(problems[:3])}{more}. Fix: {check.task}."
        artifact = f"output/{check.name.split(':')[0]}"
        if artifact not in detail:
            detail = f"{artifact}: {detail}"
    return {"test-name": check.name, "passed": passed, "score": score, "max-score": check.points,
            "detail": detail}


def grade_submission(submission_dir: str | Path) -> dict:
    """Grade the committed artifacts in `submission_dir` without running any submitted code."""
    root = Path(submission_dir).resolve()
    for cache in (read_table, read_sample):
        cache.cache_clear()
    tests = [run_check(check, root) for check in CHECKS]
    return {"schema": SCHEMA, "score": sum(test["score"] for test in tests), "max-score": MAX_SCORE,
            "tests": tests}

