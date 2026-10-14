"""Checks for Assignment 05.

The course keeps these checks in 05/assignment_checks/, which the assignment workflow downloads on every
push, and the handout ships a byte-identical copy so a local run reports exactly what GitHub will. They
read the eleven small summaries a submission saves in output/ and compare them with values recomputed from
the export that generate_data.py writes: generate_data.generate_rows() rebuilds the same rows in memory, so
the checks need neither the data folder nor pandas.

Each check scores one thing, so one mistake costs only the checks it gets wrong. Values are compared after
parsing: line endings, quoting, column order, row order, a leading row-number column, number formatting
(54 == 54.0), date formatting (2024-01-02 == 2024-01-02 00:00:00), the letter case of labels and headers,
and a placeholder for a missing value (-, NA, NaN, missing) never cost points, and cells may be separated by
commas, semicolons, or tabs. Spaces around a label cost points only in the clinic counts and the coded
sample's patient_id, where removing them is the skill being checked.

A later summary is judged against the student's own earlier ones too: the checks read which mistake an
earlier file shows (no-lab visits kept, -999 kept, dates left as text, and so on), rebuild the pipeline
with that mistake, and accept a later file that is right for the student's own data, so a mistake is
charged in the file where it was made and not again downstream.

Nothing here imports, runs, or inspects student code.
"""

from __future__ import annotations

import codecs
import csv
import dataclasses
import datetime
import difflib
import functools
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import generate_data


MISSING_FILE = "output/missing_report.csv"
ROWS_FILE = "output/row_counts.csv"
PROFILE_FILE = "output/typed_profile.csv"
CLINIC_FILE = "output/clinic_counts.csv"
GENDER_FILE = "output/gender_counts.csv"
A1C_FILE = "output/a1c_counts.csv"
SAMPLE_FILE = "output/coded_sample.csv"
VALIDATION_FILE = "output/validation_report.csv"
FENCES_FILE = "output/glucose_fences.csv"
DECISION_FILE = "output/decision_log.csv"
SUMMARY_FILE = "output/clean_summary.csv"
# The notebook step and the call that save each artifact.
SAVED_BY = {
    MISSING_FILE: ("Task 1.1", "missing_report.to_csv(MISSING_PATH)"),
    ROWS_FILE: ("Task 2.1", "row_counts.to_csv(ROWS_PATH)"),
    PROFILE_FILE: ("Task 2.3", "typed_profile.to_csv(PROFILE_PATH)"),
    CLINIC_FILE: ("Task 3.1", "clinic_counts.to_csv(CLINIC_PATH)"),
    GENDER_FILE: ("Task 3.1", "gender_counts.to_csv(GENDER_PATH)"),
    A1C_FILE: ("Task 3.2", "a1c_counts.to_csv(A1C_PATH)"),
    SAMPLE_FILE: ("Task 3.3", "coded_sample.to_csv(SAMPLE_PATH, index=False)"),
    VALIDATION_FILE: ("Task 4.1", "validation_report.to_csv(VALIDATION_PATH)"),
    FENCES_FILE: ("Task 4.2", "glucose_fences.to_csv(FENCES_PATH)"),
    DECISION_FILE: ("Task 5.1", "decision_log.to_csv(DECISION_PATH)"),
    SUMMARY_FILE: ("Task 5.2", "clean_summary.to_csv(SUMMARY_PATH)"),
}

RAW_COLUMNS = generate_data.COLUMNS
SAMPLE_IDS = generate_data.SAMPLE_IDS
SENTINEL = -999.0
GENDER_CODES = {"f": "F", "female": "F", "m": "M", "male": "M", "x": "X", "nb": "X", "nonbinary": "X"}
UNKNOWN_CLINIC = "Unknown"
# pd.cut(bins=[0, 5.6, 6.4, 20]): each bin is (left, right], and a value outside every bin has no category.
A1C_BINS = (0.0, 5.6, 6.4, 20.0)
WRONG_A1C_BINS = (0.0, 5.7, 6.5, 20.0)
A1C_LABELS = ("normal", "prediabetes", "diabetes")
ID_PATTERN = re.compile(r"D[0-9]{5}")
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
HBA1C_RANGE = (4.0, 20.0)
WEIGHT_RANGE = (30.0, 250.0)
RULES = ("patient_id_format", "visit_date_present", "gender_valid", "hba1c_in_range", "weight_in_range")
# A rule saved under another name is recognized by a word in it.
RULE_WORDS = {"patient_id_format": ("patient", "id"), "visit_date_present": ("date",), "gender_valid": ("gender",),
              "hba1c_in_range": ("hba1c", "a1c"), "weight_in_range": ("weight",)}
PROFILE_COLUMNS = ("visit_date", "age", "weight_kg", "hba1c_pct", "glucose_mg_dl")
NUMBER_COLUMNS = ("age", "weight_kg", "hba1c_pct", "glucose_mg_dl")
STATS = ("count", "mean", "std", "min", "25%", "50%", "75%", "max")

# Supplied values have at most one decimal; a percent may be saved rounded to one decimal or a whole number.
TOLERANCE = 0.051
PCT_TOLERANCE = 0.51
# A multiplier other than Task 4.2's 1.5 that wrong fences may follow.
OTHER_MULTIPLIERS = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


# The pipeline, rebuilt without pandas.


@dataclass(frozen=True)
class Flags:
    """One way to run Tasks 1 to 5: each field is the README's step or a mistake an earlier file shows."""

    dropna: str = "all"  # Task 1.2: "all", "any" (how="any"), or "none" (not dropped)
    dedupe: str = "exact"  # Task 2.1: "exact", "none", "keep_false", or "patient" (subset=["patient_id"])
    sentinel: str = "nan"  # Task 2.2: "nan" or "kept"
    age: str = "coerce"  # Task 2.3: "coerce", "text" (not converted), or "zero" (filled with 0)
    dates: str = "parsed"  # Task 2.3: "parsed" or "text"
    undated: str = "kept"  # "kept", or "dropped" with dropna(subset=["visit_date"]) before Task 5
    ids: str = "clean"  # Task 3.1: "clean", "unstripped" (upper only), "cased" (stripped only), or "raw"
    keep: str = "identified"  # Task 5.1: "identified", "all_rules" (every failing visit dropped), or "dated"
    blank_ranges: bool = True  # Task 5.1: out-of-range HbA1c and weight blanked with where()


CORRECT = Flags()


@functools.lru_cache(maxsize=None)
def raw_rows() -> tuple[tuple[str, ...], ...]:
    """data/lab_visits.csv as csv.reader reads it, header left out."""
    return tuple(generate_data.generate_rows())


def _float(cell: str) -> float | None:
    try:
        number = float(cell)
    except ValueError:
        return None
    return None if number != number else number


def _real_date(cell: str) -> str | None:
    """pd.to_datetime(format="%Y-%m-%d", errors="coerce"): the date, or None for a blank or a day that does not exist."""
    if not DATE_PATTERN.fullmatch(cell):
        return None
    try:
        return datetime.date(int(cell[:4]), int(cell[5:7]), int(cell[8:10])).isoformat()
    except ValueError:
        return None


@dataclass(frozen=True)
class Parsed:
    """Each column of data/lab_visits.csv as read_csv and Task 2 read it, one value per row."""

    weight: tuple
    hba1c: tuple
    glucose: tuple
    age: tuple  # pd.to_numeric(errors="coerce"): a whole number, or None for a blank or text
    real_date: tuple  # pd.to_datetime(format="%Y-%m-%d", errors="coerce"): an ISO date or None
    clinic: tuple  # strip(), title(), and fillna("Unknown")
    gender: tuple  # strip(), lower(), and map(GENDER_CODES): F, M, X, or None


@functools.lru_cache(maxsize=None)
def parsed() -> Parsed:
    rows = raw_rows()

    def numbers(index: int) -> tuple:
        return tuple(_float(row[index]) if row[index] else None for row in rows)

    return Parsed(
        weight=numbers(5), hba1c=numbers(6), glucose=numbers(7),
        age=tuple(float(row[4]) if row[4].isdigit() else None for row in rows),
        real_date=tuple(_real_date(row[1]) for row in rows),
        clinic=tuple(row[2].strip().title() or UNKNOWN_CLINIC for row in rows),
        gender=tuple(GENDER_CODES.get(row[3].strip().lower()) for row in rows),
    )


def _date(i: int, flags: Flags) -> str | None:
    """visit_date as Task 2.3 leaves it: an ISO date or None, or the text itself when it was not converted."""
    if flags.dates == "text":
        return raw_rows()[i][1] or None
    return parsed().real_date[i]


def _patient_id(cell: str, flags: Flags) -> str:
    if flags.ids in ("clean", "cased"):
        cell = cell.strip()
    return cell.upper() if flags.ids in ("clean", "unstripped") else cell


@functools.lru_cache(maxsize=None)
def _with_labs(dropna: str) -> tuple[int, ...]:
    """The rows left after Task 1.2."""
    rows = raw_rows()
    if dropna == "none":
        return tuple(range(len(rows)))
    if dropna == "any":
        return tuple(i for i, row in enumerate(rows) if row[6] and row[7])
    return tuple(i for i, row in enumerate(rows) if row[6] or row[7])


@functools.lru_cache(maxsize=None)
def _unique(dropna: str, dedupe: str) -> tuple[int, ...]:
    """The rows left after Task 1.2 and Task 2.1."""
    rows = raw_rows()
    kept = _with_labs(dropna)
    if dedupe == "none":
        return tuple(kept)
    if dedupe == "keep_false":
        times: dict[tuple[str, ...], int] = {}
        for i in kept:
            times[rows[i]] = times.get(rows[i], 0) + 1
        return tuple(i for i in kept if times[rows[i]] == 1)
    seen: dict[object, bool] = {}
    unique = []
    for i in kept:
        key = rows[i][0] if dedupe == "patient" else rows[i]
        if key not in seen:
            seen[key] = True
            unique.append(i)
    return tuple(unique)


@functools.lru_cache(maxsize=None)
def labs_rows(flags: Flags) -> tuple[int, ...]:
    """The rows of labs through Task 4: after Tasks 1.2 and 2.1, and any undated visits dropped early."""
    kept = _unique(flags.dropna, flags.dedupe)
    if flags.undated == "dropped":
        kept = tuple(i for i in kept if _date(i, flags) is not None)
    return kept


def quantile(values: list[float], q: float) -> float:
    """Series.quantile(q): linear interpolation between the closest sorted values."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    below = int(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def describe(values: list[float]) -> dict[str, float]:
    """Series.describe() of the values present: count, mean, sample std, min, quartiles, and max."""
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    std = (sum((value - mean) ** 2 for value in ordered) / (count - 1)) ** 0.5

    def at(q: float) -> float:
        position = (count - 1) * q
        below = int(position)
        above = min(below + 1, count - 1)
        return ordered[below] + (ordered[above] - ordered[below]) * (position - below)

    return {"count": float(count), "mean": mean, "std": std, "min": ordered[0], "25%": at(0.25),
            "50%": at(0.5), "75%": at(0.75), "max": ordered[-1]}


@dataclass(frozen=True)
class Labs:
    """labs at the end of Task 4.2, one list per column, for the rows labs_rows() keeps."""

    rows: tuple
    patient_id: list
    date: list
    age: list
    weight: list
    hba1c: list
    glucose: list
    clinic: list
    gender: list
    passes: list  # one {rule: passed} dict per visit


@functools.lru_cache(maxsize=None)
def labs_table(flags: Flags) -> Labs:
    rows, columns = raw_rows(), parsed()
    kept = labs_rows(flags)

    def measure(values: tuple, sentinel: bool) -> list:
        found = [values[i] for i in kept]
        if sentinel and flags.sentinel == "nan":
            found = [None if value == SENTINEL else value for value in found]
        return found

    age = [columns.age[i] for i in kept]
    if flags.age == "zero":
        age = [0.0 if value is None else value for value in age]
    table = Labs(
        rows=kept, patient_id=[_patient_id(rows[i][0], flags) for i in kept], date=[_date(i, flags) for i in kept],
        age=age, weight=measure(columns.weight, True), hba1c=measure(columns.hba1c, False),
        glucose=measure(columns.glucose, True), clinic=[columns.clinic[i] for i in kept],
        gender=[columns.gender[i] for i in kept], passes=[],
    )
    for pid, date, gender, hba1c, weight in zip(table.patient_id, table.date, table.gender, table.hba1c, table.weight):
        table.passes.append({
            "patient_id_format": ID_PATTERN.fullmatch(pid) is not None,
            "visit_date_present": date is not None,
            "gender_valid": gender is not None,
            "hba1c_in_range": hba1c is None or HBA1C_RANGE[0] <= hba1c <= HBA1C_RANGE[1],
            "weight_in_range": weight is None or WEIGHT_RANGE[0] <= weight <= WEIGHT_RANGE[1],
        })
    return table


@dataclass(frozen=True)
class Reference:
    """Every graded value for one way of running the pipeline."""

    rows_read: int
    rows_with_labs: int
    rows_unique: int
    missing: dict  # column -> missing count in typed_profile
    low: dict  # column -> min (a number, or an ISO date for visit_date)
    high: dict
    mean: dict
    clinics: dict  # clinic -> visits
    genders: dict  # code or None -> visits
    a1c: dict  # category or None -> visits
    a1c_wrong_edges: dict
    failing: dict  # rule -> failing rows
    blank_lab: dict  # range rule -> blank values that would fail if a blank failed
    q1: float
    q3: float
    glucose: tuple  # the glucose values present in labs
    glucose_blank: int
    dropped: int
    outside: dict  # "hba1c" or "weight" -> values outside the range among the kept visits
    clean: dict  # column -> describe() of the cleaned column
    clean_rows: int
    sample: tuple  # ((patient_id, iso date or None), ...) for the sample patients' visits


def _bin(value: float | None, bins: tuple[float, ...]) -> str | None:
    if value is not None:
        for left, right, name in zip(bins, bins[1:], A1C_LABELS):
            if left < value <= right:
                return name
    return None


def _tally(values) -> dict:
    counts: dict = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


@functools.lru_cache(maxsize=None)
def _labs_values(flags: Flags) -> dict:
    """The values Tasks 2 to 4 grade, which Task 5's choices do not change."""
    rows = raw_rows()
    labs = labs_table(flags)
    missing, low, high, mean = {}, {}, {}, {}
    present = [d for d in labs.date if d is not None]
    missing["visit_date"] = len(labs.date) - len(present)
    low["visit_date"], high["visit_date"] = min(present), max(present)
    for column, values in (("age", labs.age), ("weight_kg", labs.weight), ("hba1c_pct", labs.hba1c),
                           ("glucose_mg_dl", labs.glucose)):
        found = [value for value in values if value is not None]
        missing[column] = len(values) - len(found)
        if column == "age" and flags.age == "text":
            missing[column] = sum(1 for i in labs.rows if not rows[i][4])
        low[column], high[column], mean[column] = min(found), max(found), sum(found) / len(found)
    glucose = tuple(value for value in labs.glucose if value is not None)
    sample = sorted(((pid, date) for pid, date in zip(labs.patient_id, labs.date) if pid in SAMPLE_IDS),
                    key=lambda visit: (visit[0], visit[1] is None, visit[1] or ""))
    return dict(
        rows_read=len(rows), rows_with_labs=len(_with_labs(flags.dropna)),
        rows_unique=len(_unique(flags.dropna, flags.dedupe)), missing=missing, low=low, high=high, mean=mean,
        clinics=_tally(labs.clinic), genders=_tally(labs.gender),
        a1c=_tally(_bin(value, A1C_BINS) for value in labs.hba1c),
        a1c_wrong_edges=_tally(_bin(value, WRONG_A1C_BINS) for value in labs.hba1c),
        failing={rule: sum(1 for p in labs.passes if not p[rule]) for rule in RULES},
        blank_lab={"hba1c_in_range": sum(1 for value in labs.hba1c if value is None),
                   "weight_in_range": sum(1 for value in labs.weight if value is None)},
        q1=quantile(list(glucose), 0.25), q3=quantile(list(glucose), 0.75), glucose=glucose,
        glucose_blank=len(labs.glucose) - len(glucose), sample=tuple(sample),
    )


@functools.lru_cache(maxsize=None)
def reference(flags: Flags = CORRECT) -> Reference:
    """Every graded value when the pipeline runs with `flags`."""
    labs_flags = dataclasses.replace(flags, keep=CORRECT.keep, blank_ranges=CORRECT.blank_ranges)
    labs = labs_table(labs_flags)
    if flags.keep == "all_rules":
        keep = [all(p.values()) for p in labs.passes]
    elif flags.keep == "dated":
        keep = [p["visit_date_present"] for p in labs.passes]
    else:
        keep = [p["patient_id_format"] and p["visit_date_present"] for p in labs.passes]
    outside = {"hba1c": 0, "weight": 0}
    clean = {}
    for column, values, bounds in (("age", labs.age, None), ("weight_kg", labs.weight, WEIGHT_RANGE),
                                   ("hba1c_pct", labs.hba1c, HBA1C_RANGE), ("glucose_mg_dl", labs.glucose, None)):
        values = [value for value, k in zip(values, keep) if k and value is not None]
        if bounds is not None:
            inside = [value for value in values if bounds[0] <= value <= bounds[1]]
            outside["weight" if column == "weight_kg" else "hba1c"] = len(values) - len(inside)
            values = inside if flags.blank_ranges else values
        clean[column] = describe(values)
    return Reference(**_labs_values(labs_flags), dropped=len(keep) - sum(keep), outside=outside, clean=clean,
                     clean_rows=sum(keep))


@functools.lru_cache(maxsize=None)
def missing_counts() -> dict[str, int]:
    """isna().sum() of the table as read: read_csv makes an empty cell missing, and -999 or 'unknown' are values."""
    rows = raw_rows()
    return {column: sum(1 for row in rows if not row[i]) for i, column in enumerate(RAW_COLUMNS)}


# Reading saved CSV files.


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _decode(raw: bytes) -> str:
    """An artifact's text without a byte-order mark; UTF-16 is read through its mark."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace").lstrip("﻿")


# A CSV may separate its cells with commas, semicolons, or tabs.
DELIMITERS = (",", ";", "\t")
# A number written with a decimal comma, as a spreadsheet set to a European locale saves it: 12,5 is 12.5.
DECIMAL_COMMA = re.compile(r"^(\s*[+-]?\d*),(\d+\s*)$")


def _csv_rows(text: str) -> list[list[str]]:
    """The rows of a saved CSV, blank lines left out, split on the separator its header line uses."""
    if "\x00" in text:
        raise csv.Error("embedded NUL bytes; save the table again as CSV text")
    lines = text.splitlines()
    header = next((line for line in lines if line.strip()), "")
    delimiter = max(DELIMITERS, key=lambda mark: len(next(csv.reader([header], delimiter=mark), [])))
    rows = [row for row in csv.reader(lines, delimiter=delimiter) if any(cell.strip() for cell in row)]
    if delimiter == ",":
        return rows
    return [[DECIMAL_COMMA.sub(r"\1.\2", cell) for cell in row] for row in rows]


def _artifact(root: Path, name: str) -> Path | None:
    """The artifact's path, matching its file name in any letter case, or None.

    macOS and Windows disks ignore letter case, so matching in any case makes a local run there agree with
    GitHub's Linux runner.
    """
    path = root / name
    if path.is_file() and not path.is_symlink():
        return path
    if not path.parent.is_dir():
        return None
    same_name = [
        other
        for other in path.parent.iterdir()
        if other.name.casefold() == path.name.casefold() and other.is_file() and not other.is_symlink()
    ]
    return same_name[0] if len(same_name) == 1 else None


def _look_alikes(root: Path, folder: Path, name: str) -> list[str]:
    """Files in folder whose names are close to name, as paths relative to root."""
    if not folder.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in folder.iterdir()
        if path.is_file() and difflib.SequenceMatcher(None, name.casefold(), path.name.casefold()).ratio() >= 0.8
    )


def _missing(root: Path, name: str) -> str:
    """Say the artifact is missing and why it may be: a file of a similar name, or one saved outside output/."""
    wanted = root / name
    step, call = SAVED_BY[name]
    beside = [other for other in _look_alikes(root, wanted.parent, wanted.name)
              if other != name and other not in SAVED_BY]
    if beside:
        return f"{name} is missing; found {_join(beside)}. In {step}, save with {call}, which writes {name}, then commit it."
    misplaced = _look_alikes(root, root, wanted.name)
    if misplaced:
        return (
            f"{name} is missing, but the assignment folder itself has {_join(misplaced)}; "
            f"in {step}, save with {call}, which writes {name}, then commit it."
        )
    if name == SUMMARY_FILE and _artifact(root, DECISION_FILE):
        return (
            f"{name} is missing, while {DECISION_FILE} is saved, so Task 5.2's assert probably stopped the cell "
            f"before {call}. Read that cell's output: it names the check that failed. Fix that step, run the cell "
            "again, then commit the file."
        )
    return f"{name} is missing; run the {step} cell to write it with {call}, then commit it."


def _clean(cell: str) -> str:
    return " ".join(cell.split())


def _header(cell: str) -> str:
    """A column name as snake_case: Patient ID, patient-id, PatientId, and patient_id are one name."""
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", _clean(cell))
    # HbA1c is one word, though its capitals look like CamelCase.
    return re.sub(r"[\s\-]+", "_", name).casefold().replace("hb_a1c", "hba1c")


def _number(cell: str) -> float | None:
    text = cell.strip().replace(",", "")
    return _float(text[:-1] if text.endswith("%") else text)


def _same(given: str, expected: float, tolerance: float = TOLERANCE) -> bool:
    number = _number(given)
    return number is not None and abs(number - expected) <= tolerance


def _near(value: float | None, expected: float, tolerance: float = TOLERANCE) -> bool:
    return value is not None and abs(value - expected) <= tolerance


def _blank(cell: str) -> bool:
    """A missing value as pandas, a spreadsheet, or to_csv(na_rep="missing") writes it."""
    return cell.strip().casefold() in ("", "nan", "na", "n/a", "<na>", "nat", "none", "null", "missing", "-", ".")


def _flag(cell: str) -> bool | None:
    """A True/False flag as pandas, a spreadsheet, or a 0/1 column writes it; None for anything else."""
    text = cell.strip().casefold()
    if text in ("true", "1", "1.0", "yes"):
        return True
    if text in ("false", "0", "0.0", "no"):
        return False
    return None


def _show(value) -> str:
    if value is None:
        return "blank (missing)"
    if isinstance(value, str):
        return value
    return f"{value:g}" if float(value).is_integer() else f"{value:.2f}".rstrip("0")


def _join(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _first(items: list[str], limit: int = 4) -> str:
    return _join(items[:limit] + ([f"{len(items) - limit} more"] if len(items) > limit else []))


def _columns(table: Table) -> str:
    """The file's column names for a message, naming a blank header rather than printing nothing."""
    return _join(column or "a column with no header" for column in table.columns) or "none"


MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
MIDNIGHT = r"(?:[ T]00:00(?::00(?:\.0+)?)?)?$"


def _dates(cell: str) -> list[str]:
    """Every calendar date a cell can mean, as YYYY-MM-DD, the month-first reading first.

    A year written first is read as year, month, day. Two numbers before the year (03/02/2026, 02-03-26)
    can be month and day or day and month, so both readings are kept. A month written as a word
    (March 02, 2026, 02-Mar-2026) has one reading. A two-digit year is in the 2000s. A blank or a day that
    does not exist gives an empty list.
    """
    text = cell.strip()
    found = []
    match = (re.match(r"^(\d{4})([-/.])(\d{1,2})\2(\d{1,2})" + MIDNIGHT, text)
             or re.match(r"^(\d{4})()(\d{2})(\d{2})$", text))
    if match:
        found.append((int(match.group(1)), int(match.group(3)), int(match.group(4))))
    match = re.match(r"^(\d{1,2})([-/.])(\d{1,2})\2(\d{4}|\d{2})" + MIDNIGHT, text)
    if match:
        first, second, year = int(match.group(1)), int(match.group(3)), int(match.group(4))
        year += 2000 if year < 100 else 0
        found += [(year, first, second), (year, second, first)]
    match = (re.match(r"^(?P<month>[A-Za-z]{3,})\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})$", text)
             or re.match(r"^(?P<day>\d{1,2})[\s-]+(?P<month>[A-Za-z]{3,})\.?[\s-]+(?P<year>\d{4})$", text))
    if match and match.group("month")[:3].casefold() in MONTHS:
        month = MONTHS.index(match.group("month")[:3].casefold()) + 1
        found.append((int(match.group("year")), month, int(match.group("day"))))
    dates = []
    for year, month, day in found:
        try:
            dates.append(datetime.date(year, month, day).isoformat())
        except ValueError:
            continue
    return list(dict.fromkeys(dates))


@dataclass(frozen=True)
class Table:
    """A saved CSV: its snake_case column names, one {column: cell} dict per data row, and the same rows unstripped."""

    name: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    raw: tuple[dict[str, str], ...]
    # How many times the table is in the file: to_csv(mode="a") writes it again below the first copy.
    copies: int = 1


def read_table(root: Path, name: str) -> Table:
    """Parse a saved CSV, however it is separated, quoted, or ended.

    A leading column headed by nothing, `Unnamed: 0`, or `index` that holds only whole numbers is the row
    numbers pandas writes when `index=False` is left out, so it is set aside rather than counted as a column.
    """
    step, _ = SAVED_BY[name]
    path = _artifact(root, name)
    _assert(path is not None, _missing(root, name))
    try:
        lines = _csv_rows(_decode(path.read_bytes()))
    except csv.Error as error:
        raise AssertionError(
            f"{name} cannot be read as a CSV table ({error}); run the {step} cell again so to_csv() writes it."
        ) from None
    _assert(bool(lines), f"{name} is empty, with no header line and no rows; run the {step} cell again, then commit it.")
    header = [_header(cell) for cell in lines[0]]
    # A table written again with to_csv(mode="a") repeats its header line; the last copy is the latest write.
    starts = [i for i, row in enumerate(lines) if [_header(cell) for cell in row] == header]
    copies = len(starts)
    raw_body = lines[starts[-1] + 1:]
    while (
        len(header) > 1
        and (header[0] in ("", "index") or header[0].startswith("unnamed"))
        and raw_body
        and all(row and _number(row[0]) is not None and _number(row[0]).is_integer() for row in raw_body)
    ):
        header = header[1:]
        raw_body = [row[1:] for row in raw_body]
    raw = tuple({column: (row[i] if i < len(row) else "") for i, column in reversed(list(enumerate(header)))}
                for row in raw_body)
    rows = tuple({column: _clean(cell) for column, cell in row.items()} for row in raw)
    return Table(name=name, columns=tuple(dict.fromkeys(header)), rows=rows, raw=raw, copies=copies)


def _try_read(root: Path, name: str) -> Table | None:
    try:
        return read_table(root, name)
    except (AssertionError, OSError, csv.Error, UnicodeDecodeError):
        return None


def _written_again(table: Table) -> str:
    step, call = SAVED_BY[table.name]
    return (
        f"{table.name} holds the same table {table.copies} times: to_csv() with mode=\"a\" adds to the end of the "
        f"file instead of replacing it. The checks graded the last copy; in {step}, save once with {call}."
    )


@dataclass(frozen=True)
class Report:
    """A saved report of labeled values: {label: {column: cell}}, the raw label cells, and any layout problem."""

    table: Table
    cells: dict
    labels: tuple = ()
    raw_labels: tuple = ()
    key: str | None = None
    problem: str | None = None


def _report(root: Path, name: str, label: Callable[[str], str | None], expected: tuple[str, ...],
            sideways_ok: bool = True) -> Report:
    """A summary saved with its labels in the index: one row per label, the labels read by `label`.

    A report written with its labels across the top, as describe() leaves it without .T, is read the same
    way. A report saved with index=False has no labels; when its rows are in the README's order they are
    read in that order, and `problem` says what to fix.
    """
    table = read_table(root, name)
    step, call = SAVED_BY[name]
    key = next((c for c in table.columns if any(label(row[c]) for row in table.rows)), None)
    if key is not None:
        cells: dict[str, dict[str, str]] = {}
        labels = []
        for row in table.rows:
            found = label(row[key])
            labels.append(found)
            if found is not None:
                cells.setdefault(found, {c: v for c, v in row.items() if c != key})
        return Report(table, cells, tuple(labels), tuple(raw[key] for raw in table.raw), key)
    across = [c for c in table.columns if label(c)]
    if across and sideways_ok:
        first = table.columns[0]
        cells = {label(c): {(_header(row[first]) if row[first] else ""): row[c] for row in table.rows} for c in across}
        return Report(table, cells, tuple(label(c) for c in across), tuple(across), None)
    problem = (f"{name} names none of {_first(list(expected), 3)} (its columns are {_columns(table)}): the labels "
               f"are the index, and index=False leaves them out. In {step}, save with {call}, keeping the index.")
    _assert(len(table.rows) == len(expected), problem)
    cells = {wanted: dict(row) for wanted, row in zip(expected, table.rows)}
    return Report(table, cells, tuple(expected), tuple(expected), None, problem)


def _value(cells: dict[str, str], *words: str) -> str | None:
    """The cell of the first column whose name is one of words or contains one; None when there is none."""
    for word in words:
        if word in cells:
            return cells[word]
    for word in words:
        for column, cell in cells.items():
            if word in column:
                return cell
    numbers = [cell for cell in cells.values() if _number(cell) is not None]
    return numbers[0] if len(cells) == 1 and numbers else None


# Detecting the student's own earlier mistakes.


def _row_counts(root: Path) -> Report | None:
    try:
        return _report(root, ROWS_FILE, _row_label, ("read", "with_labs", "unique"))
    except (AssertionError, OSError, csv.Error, UnicodeDecodeError):
        return None


def _row_label(cell: str) -> str | None:
    name = _header(cell)
    if any(word in name for word in ("dup", "uniq", "repeat")):
        return "unique"
    if any(word in name for word in ("lab", "dropna", "result")):
        return "with_labs"
    if any(word in name for word in ("read", "raw", "load", "start", "original", "export")):
        return "read"
    return None


def _count_cell(report: Report, label: str) -> float | None:
    cells = report.cells.get(label)
    return None if cells is None else _number(_value(cells, "rows", "count", "n") or "")


def _detected_rows(root: Path) -> tuple[str, str]:
    """(dropna, dedupe) as row_counts.csv shows them, the README's steps when it cannot tell."""
    report = _row_counts(root)
    if report is None:
        return CORRECT.dropna, CORRECT.dedupe
    with_labs, unique = _count_cell(report, "with_labs"), _count_cell(report, "unique")
    dropna = next((option for option in ("all", "any", "none")
                   if with_labs == len(_with_labs(option))), CORRECT.dropna)
    dedupe = next((option for option in ("exact", "none", "keep_false", "patient")
                   if unique == len(_unique(dropna, option))), CORRECT.dedupe)
    return dropna, dedupe


def _profile(root: Path) -> Report | None:
    try:
        return _report(root, PROFILE_FILE, _column_label, PROFILE_COLUMNS)
    except (AssertionError, OSError, csv.Error, UnicodeDecodeError):
        return None


def _column_label(cell: str) -> str | None:
    name = _header(cell)
    return name if name in PROFILE_COLUMNS else None


def _dtype(cells: dict[str, str]) -> str:
    return (_value(cells, "dtype", "type") or "").casefold()


def _numeric_dtype(cells: dict[str, str]) -> bool:
    """Whether a profile row describes numbers: its dtype names a number, or its min and max are numbers."""
    dtype = _dtype(cells)
    if dtype:
        return any(word in dtype for word in ("int", "float", "number"))
    return all(_number(_value(cells, word) or "") is not None for word in ("min", "max"))


def _detected_typed(root: Path, dropna: str, dedupe: str) -> dict[str, str]:
    """The Task 2.2 and 2.3 choices typed_profile.csv shows, the README's steps when it cannot tell."""
    found = {"sentinel": CORRECT.sentinel, "age": CORRECT.age, "dates": CORRECT.dates, "undated": CORRECT.undated}
    report = _profile(root)
    if report is None:
        return found
    for column in ("weight_kg", "glucose_mg_dl"):
        if _same(_value(report.cells.get(column, {}), "min") or "", SENTINEL):
            found["sentinel"] = "kept"
    age = report.cells.get("age", {})
    if age and not _numeric_dtype(age):
        found["age"] = "text"
    elif age and _same(_value(age, "min") or "", 0.0):
        found["age"] = "zero"
    date = report.cells.get("visit_date", {})
    dtype = _dtype(date)
    if date and dtype and "date" not in dtype:
        found["dates"] = "text"
    missing = _number(_value(date, "missing", "na", "null") or "") if date else None
    flags = Flags(dropna=dropna, dedupe=dedupe, dates=found["dates"])
    if missing == 0 and reference(flags).missing["visit_date"] > 0:
        found["undated"] = "dropped"
    return found


def _sample(root: Path) -> Table | None:
    table = _try_read(root, SAMPLE_FILE)
    return table if table is not None and "patient_id" in table.columns else None


def _detected_ids(root: Path) -> tuple[str, ...]:
    """The Task 3.1 ID cleaning the coded sample shows; every way when the sample cannot tell."""
    table = _sample(root)
    if table is None:
        return ("clean", "unstripped", "cased", "raw")
    cells = [raw["patient_id"] for raw in table.raw]
    padded = any(cell != cell.strip() for cell in cells)
    lower = any(cell != cell.upper() for cell in cells)
    if padded or lower:
        return ({(True, True): "raw", (True, False): "unstripped", (False, True): "cased"}[(padded, lower)],)
    expected = len(reference().sample)
    return ("clean",) if len(table.rows) >= expected else ("clean", "unstripped", "cased", "raw")


def upstream(root: Path, stage: int) -> list[Flags]:
    """The README's pipeline, then the student's own: the mistakes the files before `stage` show.

    Stage 2 is typed_profile.csv, 3 the Task 3 files, 4 the Task 4 files, and 5 the Task 5 files.
    """
    dropna, dedupe = _detected_rows(root)
    own = {"dropna": dropna, "dedupe": dedupe}
    if stage >= 3:
        own.update(_detected_typed(root, dropna, dedupe))
    options = [CORRECT]
    for ids in (_detected_ids(root) if stage >= 4 else ("clean",)):
        flags = Flags(**own, ids=ids)
        if flags not in options:
            options.append(flags)
    return options


# Task 1.1: output/missing_report.csv


def _missing_report(root: Path) -> tuple[Report, str | None, str | None]:
    table = read_table(root, MISSING_FILE)
    on_top = [column for column in RAW_COLUMNS if column in table.columns]
    if not any(row[c].casefold() in RAW_COLUMNS for c in table.columns for row in table.rows):
        _assert(
            not on_top,
            f"{MISSING_FILE} has the data's columns ({_first(on_top, 3)}) across the top, so it was saved sideways "
            "or before the counts were lined up. In Task 1.1, build missing_report = pd.DataFrame({\"missing_count\": "
            "labs.isna().sum(), \"missing_pct\": ...}), one row per column of labs.",
        )
    report = _report(root, MISSING_FILE, lambda cell: cell.casefold() if cell.casefold() in RAW_COLUMNS else None,
                     RAW_COLUMNS, sideways_ok=False)
    others = [c for c in report.table.columns if c != report.key]
    count = next((c for c in others if "count" in c or c in ("sum", "missing", "n_missing")), None)
    pct = next((c for c in others if "pct" in c or "percent" in c or "%" in c or "share" in c), None)
    if count is None and pct is None and len(others) == 1:
        count = others[0]
    return report, count, pct


def check_missing_rows(root: Path) -> None:
    report, _, _ = _missing_report(root)
    _assert(report.problem is None, report.problem or "")
    listed = [label for label in report.labels if label is not None]
    absent = [column for column in RAW_COLUMNS if column not in listed]
    repeated = [column for column in RAW_COLUMNS if listed.count(column) > 1]
    problems = ([f"it has no row for {_join(absent)}"] if absent else []) + (
        [f"it lists {_join(repeated)} more than once"] if repeated else [])
    if report.table.copies > 1 and not problems:
        raise AssertionError(_written_again(report.table))
    _assert(
        not problems,
        f"{MISSING_FILE} should have one row for each of the {len(RAW_COLUMNS)} columns of data/lab_visits.csv, but "
        + "; ".join(problems) + ". In Task 1.1, count with labs.isna().sum() on the whole table as read.",
    )


def check_missing_count(root: Path) -> None:
    report, count, _ = _missing_report(root)
    _assert(count is not None,
            f"{MISSING_FILE} has no missing_count column (its columns are {_columns(report.table)}); Task 1.1 "
            'builds it with "missing_count": labs.isna().sum().')
    cells = {label: row[count] for label, row in report.cells.items()}
    expected = missing_counts()
    wrong = [f"{column} has {cell or 'blank'}, expected {expected[column]}"
             for column, cell in cells.items() if not _same(cell, expected[column])]
    if not wrong:
        return
    rows = len(raw_rows())
    if all(_same(cell, rows - expected[column]) for column, cell in cells.items()):
        cause = "Each count is of the values present: isna().sum() counts the gaps, and notna() or count() the values."
    elif all(_same(cell, expected[column] / rows * 100, PCT_TOLERANCE) for column, cell in cells.items()):
        cause = "The counts are percents; missing_count is labs.isna().sum(), and missing_pct holds the percent."
    else:
        cause = ("Count on labs exactly as pd.read_csv(DATA_PATH) reads it, before Task 1.2's dropna(): only empty "
                 "cells are missing so far, while -999 and 'unknown' are values until Task 2.")
    raise AssertionError(f"{MISSING_FILE}, missing_count: {_first(wrong, 3)}. {cause}")


def check_missing_pct(root: Path) -> None:
    report, count, pct = _missing_report(root)
    _assert(pct is not None,
            f"{MISSING_FILE} has no missing_pct column (its columns are {_columns(report.table)}); Task 1.1 builds "
            'it with "missing_pct": (labs.isna().mean() * 100).round(1).')
    cells = {label: row[pct] for label, row in report.cells.items()}
    rows = len(raw_rows())
    expected = {column: n / rows * 100 for column, n in missing_counts().items()}
    if all(_same(cell, expected[column], PCT_TOLERANCE) for column, cell in cells.items()):
        return
    # Percents that follow from the student's own counts, of the table as read or after a later step's rows,
    # are charged once, by the missing_count check.
    own = {label: _number(row[count]) for label, row in report.cells.items()} if count else {}
    totals = (rows, reference().rows_with_labs, reference().rows_unique)
    for total in totals:
        if own and all(own.get(column) is not None and _same(cell, own[column] / total * 100, PCT_TOLERANCE)
                       for column, cell in cells.items()):
            return
    wrong = [f"{column} has {cell or 'blank'}, expected {expected[column]:.1f}"
             for column, cell in cells.items() if not _same(cell, expected[column], PCT_TOLERANCE)]
    if all(_same(cell, expected[column] / 100) for column, cell in cells.items()):
        cause = "These are fractions: isna().mean() gives the share missing, so multiply it by 100 for a percent."
    else:
        cause = (f"missing_pct is (labs.isna().mean() * 100).round(1), the percent of the {rows} rows that are "
                 "missing.")
    raise AssertionError(f"{MISSING_FILE}, missing_pct: {_first(wrong, 3)}. {cause}")


# Task 2.1: output/row_counts.csv


def _rows_report(root: Path) -> Report:
    report = _report(root, ROWS_FILE, _row_label, ("read", "with_labs", "unique"))
    absent = [label for label in ("read", "with_labs", "unique") if _count_cell(report, label) is None]
    _assert(
        not absent,
        f"{ROWS_FILE} has no row count for {_join(absent)} (its rows are "
        f"{_join(raw or 'blank' for raw in report.raw_labels) or 'none'}). In Task 2.1, build row_counts = "
        'pd.Series({"read": rows_read, "with_labs": rows_with_labs, "unique": len(labs)}, name="rows").',
    )
    if report.table.copies > 1:
        raise AssertionError(_written_again(report.table))
    return report


def check_rows_no_lab(root: Path) -> None:
    report = _rows_report(root)
    expected = reference()
    read, with_labs = _count_cell(report, "read"), _count_cell(report, "with_labs")
    _assert(read == expected.rows_read,
            f"{ROWS_FILE}: read is {_show(read)}, expected {expected.rows_read}, the rows of data/lab_visits.csv. "
            "Set rows_read = len(labs) right after pd.read_csv(DATA_PATH), and keep the data file as "
            "generate_data.py wrote it.")
    if with_labs == expected.rows_with_labs:
        return
    if with_labs == expected.rows_read:
        cause = ('nothing was dropped. In Task 1.2, drop the no-show visits with labs = labs.dropna(subset=["hba1c_pct", '
                 '"glucose_mg_dl"], how="all"), then set rows_with_labs = len(labs).')
    elif with_labs == len(_with_labs("any")):
        cause = ('that is how="any", which also drops every visit with one of the two results. how="all" drops a '
                 "visit only when both are missing.")
    else:
        cause = ('In Task 1.2, drop with labs.dropna(subset=["hba1c_pct", "glucose_mg_dl"], how="all"); a plain '
                 "dropna() drops every row with any gap.")
    raise AssertionError(f"{ROWS_FILE}: with_labs is {_show(with_labs)}, expected {expected.rows_with_labs}; {cause}")


def check_rows_repeats(root: Path) -> None:
    report = _rows_report(root)
    dropna, _ = _detected_rows(root)
    unique = _count_cell(report, "unique")
    expected = len(_unique(dropna, "exact"))
    if unique == expected or unique == reference().rows_unique:
        return
    if unique is not None and unique <= len(_unique(dropna, "patient")):
        return  # Visits collapsed to one per patient are charged once, by the repeat-visit check.
    with_labs = len(_with_labs(dropna))
    if unique == with_labs:
        cause = "nothing was removed. In Task 2.1, remove the exact repeats with labs = labs.drop_duplicates()."
    elif unique == len(_unique(dropna, "keep_false")):
        cause = ("keep=False drops every copy of a repeated row, the first one too. drop_duplicates() with no "
                 "arguments keeps the first copy.")
    else:
        cause = "In Task 2.1, remove the exact repeats with labs = labs.drop_duplicates(), then save len(labs)."
    raise AssertionError(f"{ROWS_FILE}: unique is {_show(unique)}, expected {expected} ({with_labs - expected} exact "
                         f"repeats removed); {cause}")


def check_rows_repeat_visits(root: Path) -> None:
    report = _rows_report(root)
    dropna, _ = _detected_rows(root)
    unique = _count_cell(report, "unique")
    patients = len(_unique(dropna, "patient"))
    _assert(
        unique is None or unique > patients,
        f"{ROWS_FILE}: unique is {_show(unique)}, but the export has {patients} patient IDs and most patients came "
        f"back several times, so about {len(_unique(dropna, 'exact'))} visits should stay. drop_duplicates() with no "
        'subset removes only rows identical in every column; subset=["patient_id"] keeps one visit per patient.',
    )


# Task 2.3: output/typed_profile.csv


def _typed_report(root: Path) -> Report:
    report = _report(root, PROFILE_FILE, _column_label, PROFILE_COLUMNS)
    absent = [column for column in PROFILE_COLUMNS if column not in report.cells]
    _assert(
        not absent,
        f"{PROFILE_FILE} has no row for {_join(absent)}. In Task 2.3, profile "
        'typed = labs[["visit_date", "age", "weight_kg", "hba1c_pct", "glucose_mg_dl"]], one row per column.',
    )
    return report


def _typed_options(root: Path) -> list[Reference]:
    """The typed profile's expected values: the README's rows, then the rows the student's row_counts.csv shows."""
    found = _detected_typed(root, *_detected_rows(root))
    options = []
    for flags in upstream(root, 2):
        for undated in dict.fromkeys(("kept", found["undated"])):
            options.append(reference(dataclasses.replace(flags, undated=undated, dates=found["dates"]
                                                         if undated == "dropped" else flags.dates)))
    return options


def _stat(report: Report, column: str, *words: str) -> str:
    return _value(report.cells.get(column, {}), *words) or ""


def check_typed_sentinel(root: Path) -> None:
    report = _typed_report(root)
    kept = [f"{column}'s min is {_stat(report, column, 'min')}" for column in ("weight_kg", "glucose_mg_dl")
            if _same(_stat(report, column, "min"), SENTINEL)]
    _assert(
        not kept,
        f"{PROFILE_FILE}: {_join(kept)}, so -999, the export's code for not measured, is still a value. In Task 2.2, "
        'replace it with labs.replace({"weight_kg": {-999: np.nan}, "glucose_mg_dl": {-999: np.nan}}) and assign '
        "the result back to labs.",
    )
    options = _typed_options(root)
    for expected in options:
        if all(_same(_stat(report, column, "missing", "na", "null"), expected.missing[column])
               for column in ("weight_kg", "glucose_mg_dl")):
            return
    expected = options[-1]
    wrong = [f"{column} has {_stat(report, column, 'missing', 'na', 'null') or 'blank'} missing, expected "
             f"{expected.missing[column]}" for column in ("weight_kg", "glucose_mg_dl")]
    raise AssertionError(
        f"{PROFILE_FILE}: {_join(wrong)}: the blanks of the export plus every -999. Replace -999 in weight_kg and "
        "glucose_mg_dl only, after Task 2.1, and count with typed.isna().sum()."
    )


def check_typed_age(root: Path) -> None:
    report = _typed_report(root)
    cells = report.cells["age"]
    _assert(
        _numeric_dtype(cells),
        f"{PROFILE_FILE}: age has dtype {_dtype(cells) or 'blank'} and runs from {_stat(report, 'age', 'min') or 'blank'} "
        f"to {_stat(report, 'age', 'max') or 'blank'}, so it is still text; expected Int64. In Task 2.3, convert with "
        'pd.to_numeric(labs["age"], errors="coerce").astype("Int64"), which turns text that is not a number into <NA>.',
    )


def check_typed_age_blank(root: Path) -> None:
    report = _typed_report(root)
    if not _numeric_dtype(report.cells["age"]):
        return  # Ages left as text are charged once, by the age-as-numbers check.
    options = _typed_options(root)
    missing = _stat(report, "age", "missing", "na", "null")
    if any(_same(missing, expected.missing["age"]) and _same(_stat(report, "age", "min"), expected.low["age"])
           for expected in options):
        return
    expected = options[-1]
    texts = sum(1 for i in labs_rows(CORRECT) if raw_rows()[i][4] == "unknown")
    raise AssertionError(
        f"{PROFILE_FILE}: age has {missing or 'blank'} missing and a min of {_stat(report, 'age', 'min') or 'blank'}, "
        f"expected {expected.missing['age']} missing (the blank ages plus about {texts} written 'unknown') and a min of "
        f"{_show(expected.low['age'])}. pd.to_numeric(labs[\"age\"], errors=\"coerce\") leaves <NA>; keep it, since "
        "fillna() with 0 or another number records an age nobody measured."
    )


def check_typed_dates(root: Path) -> None:
    report = _typed_report(root)
    cells = report.cells["visit_date"]
    dtype = _dtype(cells)
    low, high = _stat(report, "visit_date", "min"), _stat(report, "visit_date", "max")
    _assert(
        "date" in dtype or (not dtype and _dates(low) and _dates(high)),
        f"{PROFILE_FILE}: visit_date has dtype {dtype or 'blank'}, expected datetime64. In Task 2.3, convert with "
        'pd.to_datetime(labs["visit_date"], format="%Y-%m-%d", errors="coerce"), which turns a date that does not '
        "exist into NaT.",
    )
    options = _typed_options(root)
    for expected in options:
        if (expected.low["visit_date"] in _dates(low) and expected.high["visit_date"] in _dates(high)
                and _same(_stat(report, "visit_date", "missing", "na", "null"), expected.missing["visit_date"])):
            return
    expected = options[0]
    raise AssertionError(
        f"{PROFILE_FILE}: visit_date runs from {low or 'blank'} to {high or 'blank'} with "
        f"{_stat(report, 'visit_date', 'missing', 'na', 'null') or 'blank'} missing; expected "
        f"{expected.low['visit_date']} to {expected.high['visit_date']} with {expected.missing['visit_date']} missing, "
        "the blank dates plus the ones that do not exist, such as 2025-02-30. Use format=\"%Y-%m-%d\" and "
        'errors="coerce".'
    )


def check_typed_undated(root: Path) -> None:
    _typed_report(root)
    found = _detected_typed(root, *_detected_rows(root))
    _assert(
        found["undated"] == "kept",
        f"{PROFILE_FILE}: visit_date has 0 missing, but about {reference().missing['visit_date']} visits have no real "
        "date. Keep them through Task 4: Task 4.1's visit_date_present rule counts them, and Task 5.1 drops them. "
        'Leave out dropna(subset=["visit_date"]) before Task 5.',
    )


def check_typed_values(root: Path) -> None:
    report = _typed_report(root)
    found = _detected_typed(root, *_detected_rows(root))
    wanted = [("hba1c_pct", ("min", "max", "mean"))]
    if found["age"] == "coerce":
        wanted.append(("age", ("max", "mean")))
    for column in ("weight_kg", "glucose_mg_dl"):
        wanted.append((column, ("max",) if found["sentinel"] == "kept" else ("min", "max", "mean")))
    stats = {"min": "low", "max": "high", "mean": "mean"}
    first: list[str] = []
    for expected in _typed_options(root):
        wrong = [f"{column}'s {stat} is {_stat(report, column, stat) or 'blank'}, expected "
                 f"{_show(getattr(expected, stats[stat])[column])}"
                 for column, names in wanted for stat in names
                 if not _same(_stat(report, column, stat), getattr(expected, stats[stat])[column])]
        if not wrong:
            return
        first = first or wrong
    raise AssertionError(
        f"{PROFILE_FILE}: {_first(first, 3)}. Tasks 1 and 2 change only no-show visits, repeats, -999, and types; "
        "keep data/lab_visits.csv as generate_data.py wrote it, and profile with typed.min(), typed.max(), and "
        "typed.mean(numeric_only=True).round(2)."
    )


# Task 3: the counts and the coded sample


def _counts(root: Path, name: str) -> Report:
    table = read_table(root, name)
    step, call = SAVED_BY[name]
    _assert(len(table.columns) >= 2,
            f"{name} has one column ({_columns(table)}), so the labels are gone: value_counts() keeps them in the "
            f"index, and index=False leaves them out. In {step}, save with {call}, keeping the index.")
    # The labels are the column holding text; the counts, the column named count, or else a column of numbers.
    key = next((c for c in table.columns if any(_number(row[c]) is None and not _blank(row[c]) for row in table.rows)),
               table.columns[0])
    others = [c for c in table.columns if c != key]
    count = next((c for c in others if c in ("count", "visits", "n", "rows")),
                 next((c for c in others if any(_number(row[c]) is not None for row in table.rows)), None))
    _assert(count is not None, f"{name} has no counts beside its labels (its columns are {_columns(table)}); "
                               f"save {call.split('.to_csv')[0]}, a value_counts(dropna=False).")
    cells = {}
    for row in table.rows:
        cells.setdefault(row[key], {"count": row[count]})
    if table.copies > 1:
        raise AssertionError(_written_again(table))
    return Report(table, cells, tuple(row[key] for row in table.rows), tuple(raw[key] for raw in table.raw), key)


def _counted(report: Report) -> dict[str | None, float]:
    """{label: count}, a blank label (NaN) as None and labels compared in lower case."""
    found: dict[str | None, float] = {}
    for label, cells in report.cells.items():
        name = None if _blank(label) else label.casefold()
        found[name] = found.get(name, 0.0) + (_number(cells["count"]) or 0.0)
    return found


def _counts_match(found: dict, expected: dict) -> bool:
    wanted = {(None if key is None else key.casefold()): value for key, value in expected.items() if value}
    return set(found) == set(wanted) and all(abs(found[key] - wanted[key]) < 0.5 for key in wanted)


def _show_counts(counts: dict) -> str:
    return ", ".join(f"{'blank' if key is None else key} {_show(value)}"
                     for key, value in sorted(counts.items(), key=lambda item: -item[1]))


def _proportions(found: dict) -> bool:
    return bool(found) and all(value <= 1 for value in found.values()) and abs(sum(found.values()) - 1) < 0.02


def check_clinic_clean(root: Path) -> None:
    report = _counts(root, CLINIC_FILE)
    padded = [repr(raw) for raw in report.raw_labels if raw != raw.strip()]
    _assert(not padded,
            f"{CLINIC_FILE} lists clinic names with spaces around them ({_first(padded)}), which value_counts() and "
            "get_dummies() count as other clinics. In Task 3.1, clean the column with "
            'labs["clinic"].str.strip().str.title().')
    spelled: dict[str, list[str]] = {}
    for label in report.labels:
        if not _blank(label):
            spelled.setdefault(label.casefold(), []).append(label)
    mixed = {key: names for key, names in spelled.items() if len(names) > 1}
    _assert(not mixed,
            f"{CLINIC_FILE} spells " + "; ".join(f"{key} as {_join(names)}" for key, names in mixed.items())
            + '. In Task 3.1, give each clinic one spelling with labs["clinic"].str.strip().str.title().')
    found = {key: value for key, value in _counted(report).items()
             if key is not None and key != UNKNOWN_CLINIC.casefold()}
    _assert(not _proportions(found),
            f"{CLINIC_FILE} holds shares, not counts; use value_counts(dropna=False) without normalize=True.")
    for flags in upstream(root, 3):
        expected = {key: value for key, value in reference(flags).clinics.items() if key != UNKNOWN_CLINIC}
        if _counts_match(found, expected):
            return
    expected = {key: value for key, value in reference().clinics.items() if key != UNKNOWN_CLINIC}
    raise AssertionError(f"{CLINIC_FILE} counts {_show_counts(found)}, expected {_show_counts(expected)}. Count the "
                         'cleaned column with labs["clinic"].value_counts(dropna=False) after Task 3.1.')


def check_clinic_filled(root: Path) -> None:
    report = _counts(root, CLINIC_FILE)
    found = _counted(report)
    unknown = found.get(UNKNOWN_CLINIC.casefold())
    options = [reference(flags).clinics.get(UNKNOWN_CLINIC, 0) for flags in upstream(root, 3)]
    if None not in found and unknown is not None and any(abs(unknown - value) < 0.5 for value in options):
        return
    if None in found:
        cause = f"it has {_show(found[None])} visits with a blank clinic"
    elif unknown is None:
        cause = "it has no Unknown row"
    else:
        cause = f"Unknown has {_show(unknown)} visits"
    raise AssertionError(
        f"{CLINIC_FILE}: {cause}, expected Unknown with {options[0]} visits, the visits the export left without a "
        f'clinic. In Task 3.1, fill them with .fillna("{UNKNOWN_CLINIC}") after strip() and title(), and count with '
        "value_counts(dropna=False) so a blank would show."
    )


def check_gender(root: Path) -> None:
    report = _counts(root, GENDER_FILE)
    found = _counted(report)
    options = [reference(flags).genders for flags in upstream(root, 3)]
    if any(_counts_match(found, expected) for expected in options):
        return
    expected = options[0]
    others = sorted(str(key) for key in found if key is not None and key not in ("f", "m", "x"))
    if others:
        cause = (f"it lists {_first(others)}, which are spellings, not codes. Map every spelling with "
                 'labs["gender"].str.strip().str.lower().map(gender_codes); map() turns a spelling the dictionary '
                 "does not list into NaN, while replace() leaves it.")
    elif "x" not in found:
        cause = 'there is no X: the dictionary also maps "x", "nb", and "nonbinary" to X.'
    else:
        cause = ("count after mapping, with labs[\"gender\"].value_counts(dropna=False), so the blank codes show "
                 "too.")
    raise AssertionError(f"{GENDER_FILE} counts {_show_counts(found)}, expected {_show_counts(expected)}; {cause}")


def check_a1c(root: Path) -> None:
    report = _counts(root, A1C_FILE)
    found = _counted(report)
    options = [reference(flags) for flags in upstream(root, 3)]
    if any(_counts_match(found, expected.a1c) for expected in options):
        return
    expected = options[0].a1c
    if any(_counts_match(found, option.a1c_wrong_edges) for option in options):
        cause = ("these follow the edges 5.7 and 6.5. Each pd.cut() bin includes its right edge, so "
                 "bins=[0, 5.6, 6.4, 20] puts 5.6 in normal, 5.7 to 6.4 in prediabetes, and 6.5 or more in diabetes.")
    elif None not in found:
        cause = "there is no blank row: count with value_counts(dropna=False), so the visits with no category show."
    else:
        cause = ('bin with pd.cut(labs["hba1c_pct"], bins=[0, 5.6, 6.4, 20], labels=["normal", "prediabetes", '
                 '"diabetes"]); a blank HbA1c, or one above 20, gets no category.')
    raise AssertionError(f"{A1C_FILE} counts {_show_counts(found)}, expected {_show_counts(expected)}; {cause}")


def _sample_rows(root: Path) -> Table:
    table = read_table(root, SAMPLE_FILE)
    _assert("patient_id" in table.columns,
            f"{SAMPLE_FILE} has no patient_id column (its columns are {_columns(table)}). In Task 3.3, save "
            'coded_sample = labs[labs["patient_id"].isin(SAMPLE_IDS)] with index=False.')
    if table.copies > 1:
        raise AssertionError(_written_again(table))
    return table


def check_sample_ids(root: Path) -> None:
    table = _sample_rows(root)
    cells = [raw["patient_id"] for raw in table.raw]
    padded = [repr(cell) for cell in cells if cell != cell.strip()]
    _assert(not padded, f"{SAMPLE_FILE} has patient IDs with spaces around them ({_first(padded)}). In Task 3.1, "
                        'clean them with labs["patient_id"].str.strip().str.upper().')
    lower = [cell for cell in cells if cell != cell.upper()]
    _assert(not lower, f"{SAMPLE_FILE} has lower-case patient IDs ({_first(lower)}), so one patient appears under two "
                       "IDs. In Task 3.1, add .str.upper() after .str.strip().")
    found: dict[str, int] = {}
    for cell in cells:
        found[cell.strip().upper()] = found.get(cell.strip().upper(), 0) + 1
    for flags in upstream(root, 3):
        expected: dict[str, int] = {}
        for visit in reference(flags).sample:
            expected[visit[0]] = expected.get(visit[0], 0) + 1
        if found == expected:
            return
        if flags == CORRECT:
            wanted = expected
    wrong = [f"{pid} has {found.get(pid, 0)} visits, expected {n}" for pid, n in wanted.items() if found.get(pid, 0) != n]
    wrong += [f"{pid} is not a sample patient" for pid in found if pid not in wanted]
    raise AssertionError(
        f"{SAMPLE_FILE}: {_first(wrong, 3)}. A visit whose ID was saved as 'd10007' or ' D11126' matches isin() only "
        'after labs["patient_id"].str.strip().str.upper() in Task 3.1; select with '
        'labs[labs["patient_id"].isin(SAMPLE_IDS)].'
    )


def check_sample_dummies(root: Path) -> None:
    table = _sample_rows(root)
    clinics = _try_read(root, CLINIC_FILE)
    labels = sorted(reference().clinics)
    if clinics is not None and len(clinics.columns) >= 2:
        own = [row[clinics.columns[0]] for row in clinics.raw if not _blank(row[clinics.columns[0]])]
        if any(cell != cell.strip() for cell in own) or len({c.casefold() for c in own}) != len(own):
            return  # Unclean clinic names are charged once, by the clinic names check.
        labels = own
    expected = {_header(f"clinic_{label}"): label.casefold() for label in labels}
    saved = [c for c in table.columns if c.startswith("clinic_")]
    missing = [column for column in expected if column not in saved]
    extra = [column for column in saved if column not in expected]
    _assert(
        not missing and not extra,
        f"{SAMPLE_FILE} has the indicator columns {_join(saved) or 'none'}, expected one per clinic: "
        f"{_join(expected)}" + (f"; extra {_join(extra)}" if extra else "") + ". In Task 3.3, build them with "
        'pd.get_dummies(labs["clinic"], prefix="clinic", dtype="int64"), add them with labs = labs.join(...), '
        "and then take the sample.",
    )
    _assert("clinic" in table.columns, f"{SAMPLE_FILE} has no clinic column; save every column of labs.")
    wrong = []
    for row in table.rows:
        for column, label in expected.items():
            flag = _flag(row[column])
            if flag is None or flag != (row["clinic"].casefold() == label):
                wrong.append(f"{row.get('patient_id') or 'a row'} at {row['clinic'] or 'no clinic'} has "
                             f"{column} {row[column] or 'blank'}")
    _assert(
        not wrong,
        f"{SAMPLE_FILE}: {_first(wrong, 3)}; each row should have 1 in its own clinic's column and 0 in the others. "
        "Join the get_dummies() result to the same labs it was built from.",
    )


# Task 4.1: output/validation_report.csv


def _rule_label(cell: str) -> str | None:
    """The Task 4.1 rule a label names: its own name, or the one rule whose words it contains."""
    name = _header(cell)
    if name in RULES:
        return name
    named = [rule for rule, words in RULE_WORDS.items() if any(word in name.split("_") or word in name for word in words)]
    if len(named) > 1 and "patient_id_format" in named:
        named.remove("patient_id_format")  # "visit_date_present" never names an ID, but "id" hides in other words
    return named[0] if len(named) == 1 else None


def _validation(root: Path) -> Report:
    report = _report(root, VALIDATION_FILE, _rule_label, RULES)
    if report.key is None and report.problem is None:
        problem = (f"{VALIDATION_FILE} has the rules across the top ({_first(list(report.raw_labels), 3)}), so it "
                   "holds the rules table itself or a sideways report. In Task 4.1, save validation_report = "
                   "(~rules).sum(), one row per rule.")
        _assert(len(report.table.rows) == 1, problem)
        return dataclasses.replace(report, problem=problem)
    return report


def _failing_cell(cells: dict[str, str]) -> str:
    return _value(cells, "failing_rows", "fail", "count", "rows") or next(iter(cells.values()), "")


def check_validation_rows(root: Path) -> None:
    report = _validation(root)
    _assert(report.problem is None, report.problem or "")
    labels = list(report.labels)
    absent = [rule for rule in RULES if rule not in labels]
    repeated = [rule for rule in RULES if labels.count(rule) > 1]
    problems = ([f"it has no row for {_join(absent)}"] if absent else []) + (
        [f"it lists {_join(repeated)} more than once"] if repeated else [])
    if report.table.copies > 1 and not problems:
        raise AssertionError(_written_again(report.table))
    _assert(
        not problems,
        f"{VALIDATION_FILE} should have one row for each of the {len(RULES)} rules in Task 4.1, but "
        + "; ".join(problems) + ". Build rules = pd.DataFrame({...}) with those five column names, then "
        "validation_report = (~rules).sum().",
    )


def check_validation_names(root: Path) -> None:
    report = _validation(root)
    if report.problem is not None:
        return  # A report without rule names, or turned on its side, is charged once, by the one-row-per-rule check.
    renamed = [f"{raw} for {rule}" for rule, raw in zip(report.labels, report.raw_labels)
               if rule is not None and _header(raw) != rule]
    _assert(
        not renamed,
        f"{VALIDATION_FILE} names a rule differently from Task 4.1: it has {_join(renamed)}. Keep the five column "
        f"names of rules, {_join(RULES)}, so the report reads the same as the data contract.",
    )


def check_validation_counts(root: Path) -> None:
    report = _validation(root)
    cells = {rule: _failing_cell(row) for rule, row in report.cells.items()}
    _assert(bool(cells), f"{VALIDATION_FILE} has no failing_rows values beside its rules; save (~rules).sum().")
    options = [reference(flags) for flags in upstream(root, 4)]
    own_gender = None
    genders = _try_read(root, GENDER_FILE)
    if genders is not None:
        try:
            counted = _counted(_counts(root, GENDER_FILE))
            own_gender = sum(value for key, value in counted.items() if key not in ("f", "m", "x"))
        except AssertionError:
            pass
    wrong = []
    for rule, cell in cells.items():
        accepted = [expected.failing[rule] for expected in options] + (
            [own_gender] if rule == "gender_valid" and own_gender is not None else [])
        if not any(_same(cell, value) for value in accepted):
            wrong.append(f"{rule} has {cell or 'blank'}, expected {options[0].failing[rule]}")
    if not wrong:
        return
    truth = options[0]
    rows = len(labs_rows(CORRECT))
    if all(_same(cell, rows - truth.failing[rule]) for rule, cell in cells.items()):
        cause = "These count the rows that pass: rules.sum() counts True, so count the failures with (~rules).sum()."
    elif any(rule in cells and _same(cells[rule], truth.failing[rule] + truth.blank_lab[rule])
             for rule in ("hba1c_in_range", "weight_in_range")):
        cause = ("A blank result counts as a failure: between() is False for NaN, so the range rules add "
                 '| labs["hba1c_pct"].isna() (and the same for weight_kg) to let a missing value pass.')
    else:
        cause = ("Run the rules on the Task 3 table: patient_id with .str.fullmatch(r\"D[0-9]{5}\", na=False), "
                 'visit_date with notna(), gender with isin(["F", "M", "X"]), and the two ranges with between(low, high).')
    raise AssertionError(f"{VALIDATION_FILE}, failing_rows: {_first(wrong, 3)}. {cause}")


# Task 4.2: output/glucose_fences.csv


def _fence_label(cell: str) -> str | None:
    name = _header(cell)
    for label, words in (("outliers", ("outlier", "flag")), ("lower", ("lower", "low")),
                         ("upper", ("upper", "high")), ("q1", ("q1", "25", "first")), ("q3", ("q3", "75", "third")),
                         ("iqr", ("iqr",))):
        if any(word in name for word in words):
            return label
    return None


def _fences(root: Path, *needed: str) -> dict[str, float | None]:
    report = _report(root, FENCES_FILE, _fence_label, ("q1", "q3", "lower", "upper", "outliers"))
    if report.table.copies > 1:
        raise AssertionError(_written_again(report.table))
    values = {label: _number(_value(cells, "value", "glucose") or next(iter(cells.values()), ""))
              for label, cells in report.cells.items()}
    absent = [label for label in needed if values.get(label) is None]
    _assert(not absent, f"{FENCES_FILE} has no value for {_join(absent)}. In Task 4.2, save glucose_fences = "
                        'pd.Series({"q1": q1, "q3": q3, "lower": lower, "upper": upper, "outliers": '
                        'labs["glucose_outlier"].sum()}, name="value").')
    return values


def check_fences(root: Path) -> None:
    values = _fences(root, "q1", "q3", "lower", "upper")
    options = [reference(flags) for flags in upstream(root, 4)]
    for expected in options:
        iqr = expected.q3 - expected.q1
        if all(_near(values[label], value, 0.01) for label, value in (
                ("q1", expected.q1), ("q3", expected.q3), ("lower", expected.q1 - 1.5 * iqr),
                ("upper", expected.q3 + 1.5 * iqr))):
            return
    expected = options[0]
    iqr = expected.q3 - expected.q1
    cause = ('Compute q1 = labs["glucose_mg_dl"].quantile(0.25) and q3 = .quantile(0.75) after Task 2.2 blanks -999, '
             "then iqr = q3 - q1, lower = q1 - 1.5 * iqr, and upper = q3 + 1.5 * iqr.")
    for k in OTHER_MULTIPLIERS:
        if _near(values["lower"], values["q1"] - k * (values["q3"] - values["q1"]), 0.01) and _near(
                values["upper"], values["q3"] + k * (values["q3"] - values["q1"]), 0.01):
            cause = f"These fences are {k:g} * IQR from the quartiles; Task 4.2 multiplies the IQR by 1.5."
            break
    raise AssertionError(
        f"{FENCES_FILE}: q1 {_show(values['q1'])}, q3 {_show(values['q3'])}, fences {_show(values['lower'])} to "
        f"{_show(values['upper'])}; expected q1 {_show(expected.q1)}, q3 {_show(expected.q3)}, fences "
        f"{_show(expected.q1 - 1.5 * iqr)} to {_show(expected.q3 + 1.5 * iqr)} mg/dL. {cause}"
    )


def check_outliers(root: Path) -> None:
    values = _fences(root, "outliers")
    options = [reference(flags) for flags in upstream(root, 4)]
    counts = []
    for expected in options:
        iqr = expected.q3 - expected.q1
        if values.get("lower") is None or values.get("upper") is None:
            values["lower"], values["upper"] = expected.q1 - 1.5 * iqr, expected.q3 + 1.5 * iqr
        own = sum(1 for g in expected.glucose if g < values["lower"] or g > values["upper"])
        iqr = expected.q3 - expected.q1
        truth = sum(1 for g in expected.glucose if g < expected.q1 - 1.5 * iqr or g > expected.q3 + 1.5 * iqr)
        counts.append((own, truth, expected.glucose_blank))
    if any(values["outliers"] in (own, truth) for own, truth, _ in counts):
        return
    own, truth, blanks = counts[0]
    if any(values["outliers"] in (o + b, t + b) for o, t, b in counts):
        cause = (f"that adds the {blanks} blank glucose values: ~between() is True for NaN. Flag with "
                 '(labs["glucose_mg_dl"] < lower) | (labs["glucose_mg_dl"] > upper), which is False for a blank.')
    elif values["outliers"] < 2 or values["outliers"] < truth / 50:
        cause = 'save the number of flagged visits, labs["glucose_outlier"].sum().'
    else:
        cause = ('flag with (labs["glucose_mg_dl"] < lower) | (labs["glucose_mg_dl"] > upper) and save '
                 'labs["glucose_outlier"].sum().')
    raise AssertionError(f"{FENCES_FILE}: outliers is {_show(values['outliers'])}, expected {own}, the glucose values "
                         f"outside the saved fences {_show(values['lower'])} to {_show(values['upper'])}; {cause}")


# Task 5.1: output/decision_log.csv


def _decision_label(cell: str) -> str | None:
    name = _header(cell)
    if "hba1c" in name or "a1c" in name:
        return "hba1c"
    if "weight" in name:
        return "weight"
    if any(word in name for word in ("drop", "visit", "remov", "id", "date")):
        return "dropped"
    return None


def _decisions(root: Path) -> dict[str, float | None]:
    report = _report(root, DECISION_FILE, _decision_label, ("dropped", "hba1c", "weight"))
    if report.table.copies > 1:
        raise AssertionError(_written_again(report.table))
    values = {label: _number(_value(cells, "rows", "count", "value") or next(iter(cells.values()), ""))
              for label, cells in report.cells.items()}
    absent = [name for label, name in (("dropped", "visits_dropped"), ("hba1c", "hba1c_blanked"),
                                       ("weight", "weight_blanked")) if values.get(label) is None]
    _assert(not absent, f"{DECISION_FILE} has no count for {_join(absent)}. In Task 5.1, save decision_log = "
                        'pd.Series({"visits_dropped": ..., "hba1c_blanked": ..., "weight_blanked": ...}, name="rows").')
    return values


def _task5_options(root: Path):
    """upstream() with the README's Task 5.1 rows, then with the rows the decision log's visits_dropped shows."""
    try:
        dropped = _decisions(root).get("dropped")
    except (AssertionError, OSError, csv.Error, UnicodeDecodeError):
        dropped = None
    for keep in ("identified", "all_rules", "dated"):
        for flags in upstream(root, 5):
            flags = dataclasses.replace(flags, keep=keep)
            if keep == "identified" or _near(dropped, reference(flags).dropped):
                yield flags


def check_decision_dropped(root: Path) -> None:
    values = _decisions(root)
    options = upstream(root, 5)
    if any(_near(values["dropped"], reference(flags).dropped) for flags in options):
        return
    expected = reference().dropped
    if any(_near(values["dropped"], reference(dataclasses.replace(flags, keep="all_rules")).dropped)
           for flags in options):
        cause = ("that drops every visit that fails any rule. Keep a visit that breaks only a range or gender rule; "
                 'Task 5.1 keeps clean[rules["patient_id_format"] & rules["visit_date_present"]].')
    elif any(_near(values["dropped"], reference(dataclasses.replace(flags, keep="dated")).dropped) for flags in options):
        cause = ('that drops only the undated visits; also drop the visits whose patient_id fails r"D[0-9]{5}", with '
                 'clean[rules["patient_id_format"] & rules["visit_date_present"]].')
    else:
        cause = ('keep clean[rules["patient_id_format"] & rules["visit_date_present"]] and save '
                 "len(labs) - len(clean).")
    raise AssertionError(f"{DECISION_FILE}: visits_dropped is {_show(values['dropped'])}, expected {expected}, the "
                         f"visits without a valid ID or a real date; {cause}")


def check_decision_outside(root: Path) -> None:
    values = _decisions(root)
    if any(_near(values["hba1c"], reference(flags).outside["hba1c"])
           and _near(values["weight"], reference(flags).outside["weight"]) for flags in _task5_options(root)):
        return
    expected = reference()
    raise AssertionError(
        f"{DECISION_FILE}: hba1c_blanked is {_show(values['hba1c'])} and weight_blanked {_show(values['weight'])}, "
        f"expected {expected.outside['hba1c']} and {expected.outside['weight']}, the values outside 4 to 20 % and "
        '30 to 250 kg among the kept visits. Count them before blanking, as ((clean["hba1c_pct"] < 4) | '
        '(clean["hba1c_pct"] > 20)).sum(), and the same for weight_kg.'
    )


# Task 5.2: output/clean_summary.csv


def _summary_label(cell: str) -> str | None:
    name = _header(cell)
    return name if name in NUMBER_COLUMNS else None


def _summary(root: Path) -> dict[str, dict[str, float | None]]:
    report = _report(root, SUMMARY_FILE, _summary_label, NUMBER_COLUMNS)
    if report.table.copies > 1:
        raise AssertionError(_written_again(report.table))
    found = {}
    for column, cells in report.cells.items():
        stats = {}
        for stat in STATS:
            words = (stat, "median") if stat == "50%" else (stat,)
            cell = next((cells[c] for c in cells if c in words), "")
            stats[stat] = _number(cell)
        found[column] = stats
    absent = [column for column in NUMBER_COLUMNS if column not in found]
    texts = _detected_typed(root, *_detected_rows(root))["age"] == "text"
    if absent == ["age"] and texts:
        return found  # Ages left as text, which describe() leaves out, are charged once, by the typed profile.
    _assert(not absent, f"{SUMMARY_FILE} has no row for {_join(absent)}. In Task 5.2, save clean_summary = "
                        'clean[["age", "weight_kg", "hba1c_pct", "glucose_mg_dl"]].describe().round(2).T.')
    return found


def _summary_options(root: Path):
    """The clean summary's expected values, the README's first, then the student's own earlier choices."""
    for blank in (True, False):
        for flags in _task5_options(root):
            yield reference(dataclasses.replace(flags, blank_ranges=blank))


def _summary_stopped(root: Path) -> bool:
    """Whether clean_summary.csv is missing while decision_log.csv is saved: Task 5.2's assert stopped the cell."""
    return _artifact(root, SUMMARY_FILE) is None and _artifact(root, DECISION_FILE) is not None


def _repeats_left(flags: Flags) -> bool:
    """Whether Task 5.1's clean still has two visits with one patient_id and visit_date, which Task 5.2's assert stops."""
    labs = labs_table(flags)
    seen: dict = {}
    for pid, date, passes in zip(labs.patient_id, labs.date, labs.passes):
        if passes["patient_id_format"] and passes["visit_date_present"]:
            if (pid, date) in seen:
                return True
            seen[(pid, date)] = True
    return False


def check_summary_blanked(root: Path) -> None:
    # A stopped assert is charged once, here, unless repeats an earlier file already charged stopped it.
    if _summary_stopped(root) and any(_repeats_left(flags) for flags in upstream(root, 5)):
        return
    found = _summary(root)
    wrong = []
    for column, (low, high) in (("hba1c_pct", HBA1C_RANGE), ("weight_kg", WEIGHT_RANGE)):
        stats = found.get(column, {})
        for stat in ("min", "max"):
            value = stats.get(stat)
            if value is None or not low - TOLERANCE <= value <= high + TOLERANCE:
                wrong.append(f"{column}'s {stat} is {_show(value)}")
    _assert(
        not wrong,
        f"{SUMMARY_FILE}: {_join(wrong)}, outside the contract's 4 to 20 % for HbA1c or 30 to 250 kg for weight. In "
        'Task 5.1, blank them with clean["hba1c_pct"] = clean["hba1c_pct"].where(clean["hba1c_pct"].between(4, 20)), '
        "and the same for weight_kg with 30 and 250.",
    )


def check_summary_counts(root: Path) -> None:
    if _summary_stopped(root):
        return  # The stopped assert is charged once, by the impossible-values check.
    found = _summary(root)
    for expected in _summary_options(root):
        if all(stats.get("count") is not None and abs(stats["count"] - expected.clean[column]["count"]) < 0.5
               for column, stats in found.items()):
            return
    expected = reference()
    wrong = [f"{column} has {_show(stats.get('count'))}, expected {_show(expected.clean[column]['count'])}"
             for column, stats in found.items()
             if stats.get("count") is None or abs(stats["count"] - expected.clean[column]["count"]) >= 0.5]
    full = all(stats.get("count") is not None and abs(stats["count"] - expected.clean_rows) < 0.5
               for stats in found.values())
    cause = ("Every column counts every visit, so the blanks were filled in; Task 5 keeps blanks as blanks, since "
             "nothing says what they were." if full else
             "count is the values present in each column of clean after Task 5.1; describe() counts them.")
    raise AssertionError(f"{SUMMARY_FILE}, count: {_first(wrong, 3)}. {cause}")


def check_summary_stats(root: Path) -> None:
    if _summary_stopped(root):
        return  # The stopped assert is charged once, by the impossible-values check.
    found = _summary(root)
    best: list[str] | None = None
    for expected in _summary_options(root):
        if all(stats.get("count") is not None and abs(stats["count"] - expected.clean_rows) < 0.5
               for stats in found.values()):
            return  # Blanks filled in are charged once, by the blanks-kept check.
        wrong = []
        for column, stats in found.items():
            for stat in ("mean", "min", "50%", "max"):
                value, want = stats.get(stat), expected.clean[column][stat]
                if value is None or abs(value - want) > max(TOLERANCE, abs(want) * 1e-4):
                    wrong.append(f"{column}'s {stat} is {_show(value)}, expected {_show(want)}")
        if not wrong:
            return
        if best is None:
            best = wrong
    raise AssertionError(
        f"{SUMMARY_FILE}: {_first(best or [], 3)}. Task 5.1 changes only the out-of-range HbA1c and weight values; "
        'summarize clean with clean[["age", "weight_kg", "hba1c_pct", "glucose_mg_dl"]].describe().round(2).T.'
    )


CHECKS = (
    Check("missing report: one row per column", check_missing_rows),
    Check("missing report: missing_count", check_missing_count),
    Check("missing report: missing_pct", check_missing_pct),
    Check("row counts: no-lab visits dropped", check_rows_no_lab),
    Check("row counts: exact repeats removed", check_rows_repeats),
    Check("row counts: repeat visits kept", check_rows_repeat_visits),
    Check("typed profile: -999 read as missing", check_typed_sentinel),
    Check("typed profile: age as numbers", check_typed_age),
    Check("typed profile: unreadable ages left blank", check_typed_age_blank),
    Check("typed profile: visit dates parsed", check_typed_dates),
    Check("typed profile: undated visits kept", check_typed_undated),
    Check("typed profile: other values unchanged", check_typed_values),
    Check("clinic counts: clinic names cleaned", check_clinic_clean),
    Check("clinic counts: missing clinic filled", check_clinic_filled),
    Check("gender counts: gender codes mapped", check_gender),
    Check("a1c counts: a1c_category", check_a1c),
    Check("coded sample: patient IDs cleaned", check_sample_ids),
    Check("coded sample: clinic indicators", check_sample_dummies),
    Check("validation report: one row per rule", check_validation_rows),
    Check("validation report: rule names", check_validation_names),
    Check("validation report: failing rows", check_validation_counts),
    Check("glucose fences: quartiles and fences", check_fences),
    Check("glucose fences: outliers flagged", check_outliers),
    Check("decision log: failing visits dropped", check_decision_dropped),
    Check("decision log: out-of-range values counted", check_decision_outside),
    Check("clean summary: impossible values blanked", check_summary_blanked),
    Check("clean summary: blanks kept", check_summary_counts),
    Check("clean summary: statistics", check_summary_stats),
)


def run_checks(root: Path) -> list[tuple[str, str | None]]:
    """Return one (check name, problem or None) pair per check; every check runs."""
    results = []
    for check in CHECKS:
        try:
            check.action(Path(root))
        except (AssertionError, OSError, ValueError, UnicodeDecodeError, csv.Error) as error:
            results.append((check.name, str(error)))
        else:
            results.append((check.name, None))
    return results
