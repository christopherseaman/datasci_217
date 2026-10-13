"""Checks for Assignment 05.

The course keeps these checks in 05/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They read the five CSV files a
submission saves in output/ and compare them with values recomputed from the
supplied data, copied below as RAW_ROWS. The self-test confirms that copy
matches data/lab_visits.csv.

Each check scores one thing, so one mistake costs only the checks it gets
wrong. Values are compared after parsing: line endings, quoting, column order,
row order, a leading row-number column, number formatting (54 == 54.0), date
formatting (2026-03-02 == 2026-03-02 00:00:00), the letter case of labels and
headers, and a placeholder for a missing value (-, NA, NaT) never cost points,
and cells may be separated by commas, semicolons, or tabs. Spaces around a
cell cost points only in the two columns Task 3.1 strips, patient_id and
clinic, where removing them is the skill being checked.

A later file is judged against the student's own earlier file too: the
validation report against labs_coded.csv, and the cleaned table against
labs_typed.csv and labs_coded.csv, so a mistake is charged in the file where
it was made and not again downstream.

Nothing here imports, runs, or inspects student code.
"""

from __future__ import annotations

import codecs
import csv
import datetime
import difflib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


MISSING_FILE = "output/missing_report.csv"
TYPED_FILE = "output/labs_typed.csv"
CODED_FILE = "output/labs_coded.csv"
VALIDATION_FILE = "output/validation_report.csv"
CLEAN_FILE = "output/labs_clean.csv"
# The notebook step and the call that save each artifact.
SAVED_BY = {
    MISSING_FILE: ("Task 1.1", "missing_report.to_csv(MISSING_PATH)"),
    TYPED_FILE: ("Task 2.3", "labs.to_csv(TYPED_PATH, index=False)"),
    CODED_FILE: ("Task 3.3", "labs.to_csv(CODED_PATH, index=False)"),
    VALIDATION_FILE: ("Task 4.1", "validation_report.to_csv(VALIDATION_PATH)"),
    CLEAN_FILE: ("Task 5.2", "clean.to_csv(CLEAN_PATH, index=False)"),
}

# data/lab_visits.csv, cell for cell, spaces included.
RAW_COLUMNS = ("patient_id", "visit_date", "clinic", "gender", "age", "weight_kg", "hba1c_pct", "glucose_mg_dl")
RAW_ROWS = (
    ("D2001", "2026-03-02", "Mission", "F", "54", "82.5", "7.8", "162"),
    ("D2002", "2026-03-02", "mission ", "f", "61", "95.1", "6.1", "118"),
    ("D2003", "2026-03-03", "Sunset", "NB", "47", "-999", "8.9", "201"),
    ("D2004", "2026-03-03", "SUNSET", "male", "66", "77.0", "", "135"),
    ("D2005", "2026-03-04", "Excelsior", "F", "unknown", "68.4", "5.4", "96"),
    ("D2006", "2026-03-04", "excelsior", "M", "58", "101.3", "9.6", "244"),
    ("D2006", "2026-03-04", "excelsior", "M", "58", "101.3", "9.6", "244"),
    ("D2007", "2026-03-05", " Mission", "female", "72", "64.8", "64", "140"),
    ("D2008", "2026-03-05", "", "M", "39", "88.0", "5.7", "104"),
    ("D2009", "2026-02-30", "Sunset", "F", "50", "73.2", "6.8", "128"),
    ("D2010", "2026-03-06", "Mission", "M", "63", "950", "7.2", "-999"),
    ("D2011", "2026-03-06", "Excelsior", "F", "45", "59.7", "", ""),
    ("D2001", "2026-04-06", "Mission", "F", "54", "81.9", "7.4", "151"),
    ("D2012", "2026-04-07", "Sunset", "U", "57", "70.5", "6.3", "122"),
    ("D204", "2026-04-07", "Excelsior", "M", "68", "84.2", "8.1", "188"),
    ("D2013", "", "Mission", "F", "49", "66.1", "5.6", "99"),
    ("d2014 ", "2026-04-08", "mission", "F", "70", "79.4", "10.8", "412"),
    ("D2015", "2026-04-08", "Sunset", "m", "52", "92.7", "6.5", "131"),
    ("D2016", "2026-04-09", "Excelsior", "F", "44", "61.0", "5.2", "90"),
    ("D2016", "2026-04-09", "Excelsior", "F", "44", "61.0", "5.2", "90"),
    ("D2017", "2026-04-09", "MISSION", "M", "59", "-999", "7.0", ""),
    ("D2018", "2026-04-10", "Sunset", "nonbinary", "65", "71.8", "6.9", "145"),
)
SENTINEL = -999.0
SENTINEL_COLUMNS = ("weight_kg", "glucose_mg_dl")
LAB_COLUMNS = ("hba1c_pct", "glucose_mg_dl")
NUMBER_COLUMNS = ("age", "weight_kg", "hba1c_pct", "glucose_mg_dl")
GENDER_CODES = {"f": "F", "female": "F", "m": "M", "male": "M", "x": "X", "nb": "X", "nonbinary": "X"}
UNKNOWN_CLINIC = "Unknown"
# pd.cut(bins=[0, 5.6, 6.4, 20]): each bin is (left, right], and a value outside every bin has no category.
A1C_BINS = (0.0, 5.6, 6.4, 20.0)
A1C_LABELS = ("normal", "prediabetes", "diabetes")
ID_PATTERN = re.compile(r"D[0-9]{4}")
HBA1C_RANGE = (4.0, 20.0)
WEIGHT_RANGE = (30.0, 250.0)
RULES = ("patient_id_format", "visit_date_present", "gender_valid", "hba1c_in_range", "weight_in_range")
# A rule saved under another name is recognized by a word in it.
RULE_WORDS = {"patient_id_format": ("patient", "id"), "visit_date_present": ("date",), "gender_valid": ("gender",),
              "hba1c_in_range": ("hba1c", "a1c"), "weight_in_range": ("weight",)}

# Supplied values have at most one decimal; a percent may be saved rounded to one decimal.
TOLERANCE = 0.051


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


@dataclass(frozen=True)
class Table:
    """A saved CSV: its casefolded column names, one {column: cell} dict per data row, and the same rows unstripped."""

    name: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    raw: tuple[dict[str, str], ...]
    # How many times the table is in the file: to_csv(mode="a") writes it again below the first copy.
    copies: int = 1


@dataclass(frozen=True)
class Visit:
    """One visit as the data contract describes it at each stage; None marks a missing value."""

    patient_id: str  # stripped and upper case, as Task 3.1 leaves it
    date: str | None  # ISO date, or None when blank or not a real date
    raw_id: str
    raw_clinic: str
    raw_gender: str
    age: float | None
    weight_kg: float | None
    hba1c_pct: float | None
    glucose_mg_dl: float | None

    @property
    def key(self) -> tuple[str, str | None]:
        return (self.patient_id, self.date)

    @property
    def clinic(self) -> str:
        return self.raw_clinic.strip().title() or UNKNOWN_CLINIC

    @property
    def gender(self) -> str | None:
        return GENDER_CODES.get(self.raw_gender.strip().lower())

    def value(self, column: str) -> float | None:
        return getattr(self, column)


# Expected values, recomputed from the supplied data.


def _iso(cell: str) -> str | None:
    """The calendar date a cell holds as YYYY-MM-DD, or None for a blank or a date that does not exist."""
    text = cell.strip()
    match = (re.match(r"^(\d{4})([-/])(\d{1,2})\2(\d{1,2})(?:[ T]00:00(?::00(?:\.0+)?)?)?$", text)
             or re.match(r"^(\d{4})()(\d{2})(\d{2})$", text))
    if match is None:
        match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", text)
        if match is None:
            return None
        month, day, year = (int(part) for part in match.groups())
    else:
        year, month, day = (int(match.group(i)) for i in (1, 3, 4))
    try:
        return datetime.date(year, month, day).isoformat()
    except ValueError:
        return None


def _float(cell: str) -> float | None:
    try:
        number = float(cell)
    except ValueError:
        return None
    return None if number != number else number


def _raw_visits() -> list[Visit]:
    """Every row of data/lab_visits.csv, with -999 and text ages still in place as numbers or None."""
    visits = []
    for row in RAW_ROWS:
        cells = dict(zip(RAW_COLUMNS, row))
        visits.append(Visit(
            patient_id=cells["patient_id"].strip().upper(), date=_iso(cells["visit_date"]),
            raw_id=cells["patient_id"], raw_clinic=cells["clinic"], raw_gender=cells["gender"],
            **{column: _float(cells[column]) if cells[column] else None for column in NUMBER_COLUMNS},
        ))
    return visits


def _typed_visits() -> list[Visit]:
    """The visits after Task 1.2's dropna, Task 2.1's drop_duplicates, and Task 2.2's replace."""
    kept, seen = [], set()
    for row, visit in zip(RAW_ROWS, _raw_visits()):
        cells = dict(zip(RAW_COLUMNS, row))
        if all(not cells[column] for column in LAB_COLUMNS) or row in seen:
            continue
        seen.add(row)
        changes = {column: None if visit.value(column) == SENTINEL else visit.value(column)
                   for column in SENTINEL_COLUMNS}
        kept.append(Visit(**{**visit.__dict__, **changes}))
    return kept


RAW_VISITS = _raw_visits()
TYPED = _typed_visits()
TYPED_BY_KEY = {visit.key: visit for visit in TYPED}
NO_LAB_KEYS = [visit.key for row, visit in zip(RAW_ROWS, RAW_VISITS) if not row[6] and not row[7]]
REPEATED_KEYS = sorted({v.key for v in RAW_VISITS if sum(1 for o in RAW_VISITS if o.key == v.key) > 1})
REPEAT_VISIT_ID = next(pid for pid in sorted({v.patient_id for v in TYPED})
                       if sum(1 for v in TYPED if v.patient_id == pid) > 1)


def missing_counts() -> dict[str, int]:
    """isna().sum() of the table as read: read_csv makes an empty cell missing, and -999 or 'unknown' are values."""
    return {column: sum(1 for row in RAW_ROWS if not row[i]) for i, column in enumerate(RAW_COLUMNS)}


def a1c_category(value: float | None) -> str | None:
    if value is None:
        return None
    for low, high, label in zip(A1C_BINS, A1C_BINS[1:], A1C_LABELS):
        if low < value <= high:
            return label
    return None


def _in(value: float | None, bounds: tuple[float, float]) -> bool:
    return value is not None and bounds[0] <= value <= bounds[1]


def rule_passes(visit: Visit) -> dict[str, bool]:
    """Task 4.1's rules for one visit of the coded table."""
    return {
        "patient_id_format": ID_PATTERN.fullmatch(visit.patient_id) is not None,
        "visit_date_present": visit.date is not None,
        "gender_valid": visit.gender in ("F", "M", "X"),
        "hba1c_in_range": visit.hba1c_pct is None or _in(visit.hba1c_pct, HBA1C_RANGE),
        "weight_in_range": visit.weight_kg is None or _in(visit.weight_kg, WEIGHT_RANGE),
    }


def failing_counts() -> dict[str, int]:
    return {rule: sum(1 for visit in TYPED if not rule_passes(visit)[rule]) for rule in RULES}


def quantile(values: list[float], q: float) -> float:
    """Series.quantile(q): linear interpolation between the closest sorted values."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    below = int(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def fences(values: list[float]) -> tuple[float, float]:
    q1, q3 = quantile(values, 0.25), quantile(values, 0.75)
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


GLUCOSE_FENCES = fences([v.glucose_mg_dl for v in TYPED if v.glucose_mg_dl is not None])
CLEAN_KEYS = [v.key for v in TYPED if rule_passes(v)["patient_id_format"] and rule_passes(v)["visit_date_present"]]
DROPPED_KEYS = [v.key for v in TYPED if v.key not in CLEAN_KEYS]


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

    macOS and Windows disks ignore letter case, so matching in any case makes a
    local run there agree with GitHub's Linux runner.
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
    beside = [other for other in _look_alikes(root, wanted.parent, wanted.name) if other != name]
    if beside:
        return f"{name} is missing; found {_join(beside)}. In {step}, save with {call}, which writes {name}, then commit it."
    misplaced = _look_alikes(root, root, wanted.name)
    if misplaced:
        return (
            f"{name} is missing, but the assignment folder itself has {_join(misplaced)}; "
            f"in {step}, save with {call}, which writes {name}, then commit it."
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
    return _float(text)


def _same(given: str, expected: float) -> bool:
    number = _number(given)
    return number is not None and abs(number - expected) <= TOLERANCE


def _blank(cell: str) -> bool:
    """A missing value as pandas or a spreadsheet writes it."""
    return cell.strip().casefold() in ("", "nan", "na", "n/a", "<na>", "nat", "none", "null", "-", ".")


def _matches(cell: str, expected: float | None) -> bool:
    return _blank(cell) if expected is None else _same(cell, expected)


def _flag(cell: str) -> bool | None:
    """A True/False flag as pandas, a spreadsheet, or a 0/1 column writes it; None for anything else."""
    text = cell.strip().casefold()
    if text in ("true", "1", "1.0", "yes"):
        return True
    if text in ("false", "0", "0.0", "no"):
        return False
    return None


def _show(value: float | None) -> str:
    if value is None:
        return "blank (missing)"
    return f"{value:g}" if float(value).is_integer() else f"{value:.1f}"


def _given(cell: str) -> str:
    return repr(cell) if cell != cell.strip() else (cell or "blank")


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


def _visit_name(key: tuple[str, str | None]) -> str:
    patient, date = key
    if sum(1 for other in TYPED if other.patient_id == patient) > 1:
        return f"{patient} ({date or 'no date'})"
    return patient


def read_table(root: Path, name: str) -> Table:
    """Parse a saved CSV, however it is separated, quoted, or ended.

    A leading column headed by nothing, `Unnamed: 0`, or `index` that holds only
    whole numbers is the row numbers pandas writes when `index=False` is left
    out, so it is set aside rather than counted as a column.
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


def _written_again(table: Table) -> str:
    step, call = SAVED_BY[table.name]
    return (
        f"{table.name} holds the same table {table.copies} times: to_csv() with mode=\"a\" adds to the end of the "
        f"file instead of replacing it. The checks graded the last copy; in {step}, save once with {call}."
    )


def _need(table: Table, columns: tuple[str, ...]) -> None:
    absent = [column for column in columns if column not in table.columns]
    step, call = SAVED_BY[table.name]
    _assert(
        not absent,
        f"{table.name} has no {_join(absent)} column (its columns are {_columns(table)}). In {step}, save the whole "
        f"table with {call}; index=False is right because these columns are ordinary columns, not the index.",
    )


def _match(row: dict[str, str], known: dict[tuple[str, str | None], Visit]) -> Visit | None:
    """The visit a saved row is: by patient ID and date, or by ID alone when the patient has one visit."""
    patient = row.get("patient_id", "").strip().upper()
    date = _iso(row.get("visit_date", ""))
    if (patient, date) in known:
        return known[(patient, date)]
    same_patient = [visit for key, visit in known.items() if key[0] == patient]
    return same_patient[0] if len(same_patient) == 1 else None


def _visits(table: Table, known: dict | None = None) -> list[tuple[Visit | None, dict[str, str], dict[str, str]]]:
    """(visit or None, cleaned row, raw row) for each data row, in file order."""
    known = TYPED_BY_KEY if known is None else known
    return [(_match(row, known), row, raw) for row, raw in zip(table.rows, table.raw)]


def _try_read(root: Path, name: str) -> Table | None:
    try:
        return read_table(root, name)
    except (AssertionError, OSError, csv.Error, UnicodeDecodeError):
        return None


def _own_visits(table: Table | None) -> list[Visit] | None:
    """A saved table's rows as visits, read the way pandas would see each cell, or None when it cannot be read."""
    if table is None or not {"patient_id", *NUMBER_COLUMNS} <= set(table.columns):
        return None
    own = []
    for row, raw in zip(table.rows, table.raw):
        own.append(Visit(
            patient_id=raw["patient_id"], date=_iso(row.get("visit_date", "")) if "visit_date" in row else None,
            raw_id=raw["patient_id"], raw_clinic=raw.get("clinic", ""), raw_gender=raw.get("gender", ""),
            **{column: _number(row[column]) for column in NUMBER_COLUMNS},
        ))
    return own


def _date_kept(table: Table | None) -> list[bool]:
    """For each row of a saved table, whether its visit_date cell holds anything: notna() after a round trip."""
    if table is None or "visit_date" not in table.columns:
        return []
    return [not _blank(row["visit_date"]) for row in table.rows]


# Task 1.1: output/missing_report.csv


def _report_cells(table: Table, labels: dict[str, str], value_words: tuple[str, ...]) -> tuple[str | None, dict[str, str]]:
    """(the value column, {label: cell}) for a two-column report whose labels are known names."""
    key = next((c for c in table.columns if any(row[c].casefold() in labels for row in table.rows)), None)
    if key is None:
        return None, {}
    others = [c for c in table.columns if c != key]
    value = next((c for c in others if any(word in c for word in value_words)), others[0] if len(others) == 1 else None)
    cells = {}
    for row in table.rows:
        label = labels.get(row[key].casefold())
        if label is not None and value is not None:
            cells.setdefault(label, row[value])
    return value, cells


def _missing_report(root: Path) -> tuple[Table, str | None, str | None, list[str]]:
    """(table, count column, percent column, the raw column names it lists in file order)."""
    table = read_table(root, MISSING_FILE)
    names = {column: column for column in RAW_COLUMNS}
    key = next((c for c in table.columns if any(row[c].casefold() in names for row in table.rows)), None)
    if key is None:
        on_top = [column for column in RAW_COLUMNS if column in table.columns]
        _assert(
            not on_top,
            f"{MISSING_FILE} has the data's columns ({_first(on_top, 3)}) across the top, so it was saved sideways "
            "or before the counts were lined up. In Task 1.1, build missing_report = pd.DataFrame({\"missing_count\": "
            "labs.isna().sum(), \"missing_pct\": ...}), one row per column of labs.",
        )
        raise AssertionError(
            f"{MISSING_FILE} names none of the columns of labs (its columns are {_columns(table)}): the column names "
            "are the index, and index=False leaves them out. In Task 1.1, save with missing_report.to_csv(MISSING_PATH), "
            "keeping the index."
        )
    others = [c for c in table.columns if c != key]
    count = next((c for c in others if "count" in c or c in ("sum", "missing", "n_missing")), None)
    pct = next((c for c in others if "pct" in c or "percent" in c or "%" in c or "share" in c), None)
    if count is None and pct is None and len(others) == 1:
        count = others[0]
    listed = [row[key].casefold() for row in table.rows if row[key].casefold() in names]
    return table, count, pct, listed


def check_missing_rows(root: Path) -> None:
    table, _, _, listed = _missing_report(root)
    absent = [column for column in RAW_COLUMNS if column not in listed]
    repeated = [column for column in RAW_COLUMNS if listed.count(column) > 1]
    problems = ([f"it has no row for {_join(absent)}"] if absent else []) + (
        [f"it lists {_join(repeated)} more than once"] if repeated else [])
    if table.copies > 1 and not problems:
        raise AssertionError(_written_again(table))
    _assert(
        not problems,
        f"{MISSING_FILE} should have one row for each of the {len(RAW_COLUMNS)} columns of data/lab_visits.csv, but "
        + "; ".join(problems) + ". In Task 1.1, count with labs.isna().sum() on the whole table as read.",
    )


def check_missing_count(root: Path) -> None:
    table, count, _, _ = _missing_report(root)
    _assert(count is not None,
            f"{MISSING_FILE} has no missing_count column (its columns are {_columns(table)}); Task 1.1 builds it "
            'with "missing_count": labs.isna().sum().')
    _, cells = _report_cells(table, {c: c for c in RAW_COLUMNS}, (count,))
    expected = missing_counts()
    wrong = [f"{column} has {cell or 'blank'}, expected {expected[column]}"
             for column, cell in cells.items() if not _same(cell, expected[column])]
    if not wrong:
        return
    rows = len(RAW_ROWS)
    if all(_same(cell, rows - expected[column]) for column, cell in cells.items()):
        cause = "Each count is of the values present: isna().sum() counts the gaps, and notna() or count() the values."
    elif all(_same(cell, expected[column] / rows * 100) for column, cell in cells.items()):
        cause = "The counts are percents; missing_count is labs.isna().sum(), and missing_pct holds the percent."
    else:
        cause = ("Count on labs exactly as pd.read_csv(DATA_PATH) reads it, before Task 1.2's dropna(): only empty "
                 "cells are missing so far, while -999 and 'unknown' are values until Task 2.")
    raise AssertionError(f"{MISSING_FILE}, missing_count: {_first(wrong, 3)}. {cause}")


def check_missing_pct(root: Path) -> None:
    table, count, pct, _ = _missing_report(root)
    _assert(pct is not None,
            f"{MISSING_FILE} has no missing_pct column (its columns are {_columns(table)}); Task 1.1 builds it "
            'with "missing_pct": (labs.isna().mean() * 100).round(1).')
    _, cells = _report_cells(table, {c: c for c in RAW_COLUMNS}, (pct,))
    _, counts = _report_cells(table, {c: c for c in RAW_COLUMNS}, (count,)) if count else (None, {})
    rows = len(RAW_ROWS)
    expected = {column: n / rows * 100 for column, n in missing_counts().items()}
    if all(_same(cell, expected[column]) for column, cell in cells.items()):
        return
    # Percents that follow from the student's own counts, of the table as read or after a later step's rows,
    # are charged once, by the missing_count check.
    own = {column: _number(cell) for column, cell in counts.items()}
    for total in (rows, rows - len(NO_LAB_KEYS), len(TYPED)):
        if own and all(column in own and own[column] is not None and _same(cell, own[column] / total * 100)
                       for column, cell in cells.items()):
            return
    wrong = [f"{column} has {cell or 'blank'}, expected {expected[column]:.1f}"
             for column, cell in cells.items() if not _same(cell, expected[column])]
    if all(_same(cell, expected[column] / 100) for column, cell in cells.items()):
        cause = "These are fractions: isna().mean() gives the share missing, so multiply it by 100 for a percent."
    else:
        cause = "missing_pct is (labs.isna().mean() * 100).round(1), the percent of the 22 rows that are missing."
    raise AssertionError(f"{MISSING_FILE}, missing_pct: {_first(wrong, 3)}. {cause}")


# Task 2: output/labs_typed.csv


def _typed(root: Path) -> list[tuple[Visit | None, dict[str, str], dict[str, str]]]:
    table = read_table(root, TYPED_FILE)
    _need(table, ("patient_id",))
    return _visits(table, {**{v.key: v for v in RAW_VISITS}, **TYPED_BY_KEY})


def _keys(rows) -> list[tuple[str, str | None]]:
    return [visit.key for visit, _, _ in rows if visit is not None]


def check_typed_no_lab(root: Path) -> None:
    keys = _keys(_typed(root))
    kept = [key for key in NO_LAB_KEYS if key in keys]
    one_lab = [v.key for v in TYPED if (v.hba1c_pct is None) != (v.glucose_mg_dl is None) and v.key not in keys]
    _assert(
        not kept,
        f"{TYPED_FILE} still has {_join(_visit_name(key) for key in kept)}, a visit with neither an HbA1c nor a "
        'glucose result. In Task 1.2, drop it with labs.dropna(subset=["hba1c_pct", "glucose_mg_dl"], how="all").',
    )
    _assert(
        not one_lab,
        f"{TYPED_FILE} is missing {_join(_visit_name(key) for key in one_lab)}, which has one of the two lab results. "
        'how="all" drops a visit only when both are missing; leave out how="any" and plain dropna(), which drop more.',
    )


def check_typed_repeats(root: Path) -> None:
    rows = _typed(root)
    keys = _keys(rows)
    repeated = [key for key in dict.fromkeys(keys) if keys.count(key) > 1]
    gone = [key for key in REPEATED_KEYS if key not in keys]
    _assert(
        not repeated,
        f"{TYPED_FILE} lists " + _join(f"{_visit_name(key)} {keys.count(key)} times" for key in repeated)
        + ", but each was entered twice by mistake. In Task 2.1, keep one copy with labs = labs.drop_duplicates().",
    )
    _assert(
        not gone,
        f"{TYPED_FILE} has no row for {_join(_visit_name(key) for key in gone)}, whose exact repeat should leave one "
        "copy. drop_duplicates() keeps the first copy by default; keep=False drops every copy.",
    )


def check_typed_repeat_visit(root: Path) -> None:
    keys = _keys(_typed(root))
    visits = [v.key for v in TYPED if v.patient_id == REPEAT_VISIT_ID]
    gone = [key for key in visits if key not in keys]
    _assert(
        not gone,
        f"{TYPED_FILE} has {len(visits) - len(gone)} of {REPEAT_VISIT_ID}'s {len(visits)} visits; missing "
        f"{_join(_visit_name(key) for key in gone)}. They are on different dates, so both are real visits. "
        'drop_duplicates() with no subset removes only exact repeats; subset=["patient_id"] removes the repeat visit too.',
    )


def check_typed_sentinel(root: Path) -> None:
    rows = _typed(root)
    table = read_table(root, TYPED_FILE)
    _need(table, SENTINEL_COLUMNS)
    kept = []
    for visit, row, _ in rows:
        if visit is None:
            continue
        raw = next(r for r in RAW_VISITS if r.key == visit.key)
        for column in SENTINEL_COLUMNS:
            if raw.value(column) == SENTINEL and not _blank(row[column]):
                kept.append(f"{_visit_name(visit.key)}'s {column} is {row[column]}")
    _assert(
        not kept,
        f"{TYPED_FILE}: {_first(kept, 3)}, where -999 is the export's code for not measured; expected a blank "
        "(missing) cell. In Task 2.2, replace it with labs.replace({\"weight_kg\": {-999: np.nan}, "
        "\"glucose_mg_dl\": {-999: np.nan}}) and assign the result back to labs.",
    )


def check_typed_age(root: Path) -> None:
    rows = _typed(root)
    _need(read_table(root, TYPED_FILE), ("age",))
    text = [f"{_visit_name(visit.key)}'s age is {row['age']}" for visit, row, _ in rows
            if visit is not None and not _blank(row["age"]) and _number(row["age"]) is None]
    _assert(
        not text,
        f"{TYPED_FILE}: {_first(text, 3)}, expected a number or a blank. In Task 2.3, convert with "
        'pd.to_numeric(labs["age"], errors="coerce").astype("Int64"), which turns text that is not a number into <NA>.',
    )


def check_typed_dates(root: Path) -> None:
    rows = _typed(root)
    _need(read_table(root, TYPED_FILE), ("visit_date",))
    wrong = []
    for visit, row, _ in rows:
        if visit is None:
            continue
        cell = row["visit_date"]
        if visit.date is None and not _blank(cell):
            wrong.append(f"{_visit_name(visit.key)}'s visit_date is {cell}, expected a blank (no such day)")
        elif visit.date is not None and _iso(cell) != visit.date:
            wrong.append(f"{_visit_name(visit.key)}'s visit_date is {cell or 'blank'}, expected {visit.date}")
    _assert(
        not wrong,
        f"{TYPED_FILE}: {_first(wrong, 3)}. In Task 2.3, convert with pd.to_datetime(labs[\"visit_date\"], "
        'format="%Y-%m-%d", errors="coerce"), which turns a date that does not exist into NaT, saved as a blank.',
    )


def check_typed_values(root: Path) -> None:
    rows = _typed(root)
    _need(read_table(root, TYPED_FILE), NUMBER_COLUMNS)
    wrong = []
    for visit, row, _ in rows:
        if visit is None:
            continue
        raw = next(r for r in RAW_VISITS if r.key == visit.key)
        for column in NUMBER_COLUMNS:
            expected = raw.value(column)
            if expected is None or expected == SENTINEL:
                continue  # Blanks, -999, and the text age are graded by their own checks.
            if not _same(row[column], expected):
                wrong.append(f"{_visit_name(visit.key)}'s {column} is {row[column] or 'blank'}, expected {_show(expected)}")
    unknown = sum(1 for visit, _, _ in rows if visit is None)
    _assert(
        not wrong,
        f"{TYPED_FILE}: {_first(wrong, 3)}, as in data/lab_visits.csv. Tasks 1 and 2 change only blanks, repeats, "
        "-999, and types; keep the data file unchanged and leave other values as read.",
    )
    _assert(
        not unknown,
        f"{TYPED_FILE} has {unknown} row{'s' if unknown > 1 else ''} whose patient_id and visit_date match no visit in "
        "data/lab_visits.csv; keep the data file unchanged and save labs itself.",
    )


# Task 3: output/labs_coded.csv


def _coded(root: Path, *columns: str) -> tuple[Table, list[tuple[Visit | None, dict[str, str], dict[str, str]]]]:
    table = read_table(root, CODED_FILE)
    _need(table, ("patient_id", *columns))
    return table, [(visit, row, raw) for visit, row, raw in _visits(table) if visit is not None]


def _padded(table: Table, column: str) -> list[str]:
    return [raw[column] for raw in table.raw if raw[column] != raw[column].strip()]


def _spellings(cells: list[str]) -> dict[str, list[str]]:
    """{label in lower case: its spellings} for labels written more than one way."""
    found: dict[str, list[str]] = {}
    for cell in cells:
        if cell.strip():
            found.setdefault(cell.strip().casefold(), [])
            if cell.strip() not in found[cell.strip().casefold()]:
                found[cell.strip().casefold()].append(cell.strip())
    return {label: spelled for label, spelled in found.items() if len(spelled) > 1}


def check_coded_ids(root: Path) -> None:
    table, rows = _coded(root)
    padded = _padded(table, "patient_id")
    _assert(
        not padded,
        f"{CODED_FILE} has patient IDs with spaces around them ({_first([repr(cell) for cell in padded])}). In Task "
        '3.1, clean them with labs["patient_id"].str.strip().str.upper().',
    )
    lower = [raw["patient_id"] for _, _, raw in rows if raw["patient_id"] != raw["patient_id"].upper()]
    _assert(
        not lower or len(lower) == len(rows),
        f"{CODED_FILE} mixes letter cases in patient_id ({_first(lower)} beside upper-case IDs), so one patient could "
        'appear under two IDs. In Task 3.1, add .str.upper() after .str.strip().',
    )


def _clinic_problem(table: Table) -> str | None:
    """Why the clinic column is not clean, or None: spaces around a name, or a name spelled two ways."""
    padded = _padded(table, "clinic")
    if padded:
        return (f"{CODED_FILE} has clinic names with spaces around them ({_first([repr(cell) for cell in padded])}), "
                "which value_counts() and get_dummies() treat as other clinics")
    mixed = _spellings([raw["clinic"] for raw in table.raw])
    if mixed:
        return (f"{CODED_FILE} spells " + "; ".join(f"{label} as {_join(spelled)}" for label, spelled in mixed.items())
                + ". value_counts() and get_dummies() count each spelling as its own clinic")
    return None


def check_coded_clinic(root: Path) -> None:
    table, rows = _coded(root, "clinic")
    problem = _clinic_problem(table)
    _assert(problem is None,
            f"{problem}. In Task 3.1, clean the column with labs[\"clinic\"].str.strip().str.title().")
    known = {visit.clinic.casefold() for visit in TYPED}
    other = [f"{_visit_name(visit.key)} is at {row['clinic']}" for visit, row, _ in rows
             if row["clinic"] and row["clinic"].casefold() not in known]
    _assert(not other, f"{CODED_FILE}: {_first(other, 3)}, which is not a clinic in data/lab_visits.csv; "
                       "strip() and title() change only spaces and letter case.")


def check_coded_clinic_filled(root: Path) -> None:
    _, rows = _coded(root, "clinic")
    blanks = [v for v in TYPED if not v.raw_clinic.strip()]
    wrong = [f"{_visit_name(visit.key)}'s clinic is {row['clinic'] or 'blank'}" for visit, row, _ in rows
             if visit in blanks and row["clinic"].casefold() != UNKNOWN_CLINIC.casefold()]
    _assert(
        not wrong,
        f"{CODED_FILE}: {_join(wrong)}, expected {UNKNOWN_CLINIC}, since the export left it blank. In Task 3.1, "
        f'fill it with labs["clinic"] = labs["clinic"].fillna("{UNKNOWN_CLINIC}"), after strip() and title().',
    )


def check_coded_gender(root: Path) -> None:
    _, rows = _coded(root, "gender")
    wrong = []
    for visit, row, _ in rows:
        cell, expected = row["gender"], visit.gender
        if expected is None:
            if not _blank(cell):
                wrong.append(f"{_visit_name(visit.key)} has {cell}, expected a blank for the code {visit.raw_gender}, "
                             "which the dictionary does not list (map() turns it into NaN; replace() leaves it)")
        elif cell.casefold() != expected.casefold():
            wrong.append(f"{_visit_name(visit.key)} has {cell or 'blank'}, expected {expected}")
    _assert(
        not wrong,
        f"{CODED_FILE}, gender: {_first(wrong, 3)}. In Task 3.1, map every spelling to one code with "
        'labs["gender"].str.strip().str.lower().map({"f": "F", "female": "F", "m": "M", "male": "M", "x": "X", "nb": "X", "nonbinary": "X"}).',
    )
    mixed = sorted({row["gender"] for _, row, _ in rows if row["gender"] and not _blank(row["gender"])})
    _assert(
        len({code.casefold() for code in mixed}) == len(mixed),
        f"{CODED_FILE}, gender: the codes are written as {_join(mixed)}; map each spelling to F, M, or X with the "
        "dictionary in Task 3.1.",
    )


def check_coded_category(root: Path) -> None:
    table, rows = _coded(root, "a1c_category")
    wrong = []
    for visit, row, _ in rows:
        own = _number(row.get("hba1c_pct", ""))
        accepted = {a1c_category(visit.hba1c_pct), a1c_category(own) if own is not None else None}
        cell = row["a1c_category"]
        if not any((label is None and _blank(cell)) or (label is not None and cell.casefold() == label)
                   for label in accepted):
            expected = a1c_category(visit.hba1c_pct)
            wrong.append(f"{_visit_name(visit.key)} (HbA1c {_show(visit.hba1c_pct)}) has {cell or 'blank'}, "
                         f"expected {expected or 'a blank'}")
    _assert(
        not wrong,
        f"{CODED_FILE}, a1c_category: {_first(wrong, 3)}. In Task 3.2, bin with pd.cut(labs[\"hba1c_pct\"], "
        'bins=[0, 5.6, 6.4, 20], labels=["normal", "prediabetes", "diabetes"]): each bin includes its right edge, '
        "and a value outside every bin, or a blank, gets no category.",
    )


def check_coded_dummies(root: Path) -> None:
    table, rows = _coded(root, "clinic")
    if _clinic_problem(table) is not None:
        return  # Unclean clinic names are charged once, by the clinic names check.
    labels = sorted({row["clinic"].casefold() for row in table.rows if not _blank(row["clinic"])})
    expected = {_header(f"clinic_{label}"): label for label in labels}
    saved = [c for c in table.columns if c.startswith("clinic_")]
    missing = [column for column in expected if column not in saved]
    extra = [column for column in saved if column not in expected]
    _assert(
        not missing and not extra,
        f"{CODED_FILE} has the indicator columns {_join(saved) or 'none'}, expected one per clinic: "
        f"{_join(expected)}" + (f"; extra {_join(extra)}" if extra else "") + ". In Task 3.3, build them with "
        'pd.get_dummies(labs["clinic"], prefix="clinic", dtype="int64") and add them with labs = labs.join(...).',
    )
    wrong = []
    for row in table.rows:
        for column, label in expected.items():
            flag = _flag(row[column])
            if flag is None or flag != (row["clinic"].casefold() == label):
                wrong.append(f"{row.get('patient_id') or 'a row'} at {row['clinic'] or 'no clinic'} has "
                             f"{column} {row[column] or 'blank'}")
    _assert(
        not wrong,
        f"{CODED_FILE}: {_first(wrong, 3)}; each row should have 1 in its own clinic's column and 0 in the others. "
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


def _validation(root: Path) -> tuple[Table, list[str | None], dict[str, str]]:
    """(table, the rule each row names in file order, {rule: failing_rows cell})."""
    table = read_table(root, VALIDATION_FILE)
    key = next((c for c in table.columns if any(_rule_label(row[c]) for row in table.rows)), None)
    if key is None:
        across = [c for c in table.columns if c in RULES]
        _assert(not across,
                f"{VALIDATION_FILE} has the rules across the top ({_first(across, 3)}), so it holds the rules table "
                "itself or a sideways report. In Task 4.1, save validation_report = (~rules).sum(), one row per rule.")
        raise AssertionError(
            f"{VALIDATION_FILE} names none of the rules {_join(RULES)} (its columns are {_columns(table)}): the rule "
            "names are the index, and index=False leaves them out. In Task 4.1, save with "
            "validation_report.to_csv(VALIDATION_PATH), keeping the index.")
    others = [c for c in table.columns if c != key]
    value = next((c for c in others if "fail" in c), others[0] if len(others) == 1 else None)
    labels = [_rule_label(row[key]) for row in table.rows]
    cells: dict[str, str] = {}
    for rule, row in zip(labels, table.rows):
        if rule is not None and value is not None:
            cells.setdefault(rule, row[value])
    return table, labels, cells


def check_validation_rows(root: Path) -> None:
    table, labels, _ = _validation(root)
    absent = [rule for rule in RULES if rule not in labels]
    repeated = [rule for rule in RULES if labels.count(rule) > 1]
    problems = ([f"it has no row for {_join(absent)}"] if absent else []) + (
        [f"it lists {_join(repeated)} more than once"] if repeated else [])
    if table.copies > 1 and not problems:
        raise AssertionError(_written_again(table))
    _assert(
        not problems,
        f"{VALIDATION_FILE} should have one row for each of the {len(RULES)} rules in Task 4.1, but "
        + "; ".join(problems) + ". Build rules = pd.DataFrame({...}) with those five column names, then "
        "validation_report = (~rules).sum().",
    )


def _own_failing(root: Path) -> dict[str, int] | None:
    """The rule failures of the student's own labs_coded.csv, as Task 4.1's code would count them."""
    table = _try_read(root, CODED_FILE)
    own = _own_visits(table)
    if own is None:
        return None
    dates = _date_kept(table)
    counts = {rule: 0 for rule in RULES}
    for visit, raw, has_date in zip(own, table.raw, dates or [False] * len(own)):
        passes = {
            "patient_id_format": ID_PATTERN.fullmatch(raw["patient_id"]) is not None,
            "visit_date_present": has_date,
            "gender_valid": raw.get("gender", "") in ("F", "M", "X"),
            "hba1c_in_range": visit.hba1c_pct is None or _in(visit.hba1c_pct, HBA1C_RANGE),
            "weight_in_range": visit.weight_kg is None or _in(visit.weight_kg, WEIGHT_RANGE),
        }
        for rule, passed in passes.items():
            counts[rule] += 0 if passed else 1
    return counts


def check_validation_counts(root: Path) -> None:
    _, _, cells = _validation(root)
    _assert(bool(cells), f"{VALIDATION_FILE} has no failing_rows values beside its rules; save (~rules).sum().")
    truth = failing_counts()
    own = _own_failing(root)
    for expected in [truth] + ([own] if own else []):
        if all(_same(cell, expected[rule]) for rule, cell in cells.items()):
            return
    wrong = [f"{rule} has {cell or 'blank'}, expected {truth[rule]}" for rule, cell in cells.items()
             if not _same(cell, truth[rule])]
    rows = len(TYPED)
    if all(_same(cell, rows - truth[rule]) for rule, cell in cells.items()):
        cause = "These count the rows that pass: rules.sum() counts True, so count the failures with (~rules).sum()."
    elif any(rule in cells and _same(cells[rule], truth[rule] + sum(1 for v in TYPED if v.value(column) is None))
             for rule, column in (("hba1c_in_range", "hba1c_pct"), ("weight_in_range", "weight_kg"))):
        cause = ("A blank result counts as a failure: between() is False for NaN, so the range rules add "
                 '| labs["hba1c_pct"].isna() (and the same for weight_kg) to let a missing value pass.')
    else:
        cause = ("Run the rules on the Task 3 table: patient_id with .str.fullmatch(r\"D[0-9]{4}\", na=False), "
                 'visit_date with notna(), gender with isin(["F", "M", "X"]), and the two ranges with between(low, high).')
    raise AssertionError(f"{VALIDATION_FILE}, failing_rows: {_first(wrong, 3)}. {cause}")


# Task 5: output/labs_clean.csv


def _clean_rows(root: Path, *columns: str) -> tuple[Table, list[tuple[Visit | None, dict[str, str], dict[str, str]]]]:
    table = read_table(root, CLEAN_FILE)
    _need(table, ("patient_id", *columns))
    return table, _visits(table)


def _own_clean_keys(root: Path) -> set[tuple[str, str | None]] | None:
    """The visits the student's own labs_coded.csv leaves after Task 5.1 keeps valid IDs and dates."""
    table = _try_read(root, CODED_FILE)
    if _own_visits(table) is None:
        return None
    dates = _date_kept(table)
    kept = set()
    for (visit, row, raw), has_date in zip(_visits(table), dates or [False] * len(table.rows)):
        if visit is not None and ID_PATTERN.fullmatch(raw["patient_id"]) and has_date:
            kept.add(visit.key)
    return kept


def check_clean_dropped(root: Path) -> None:
    table, rows = _clean_rows(root)
    keys = set(_keys(rows))
    if keys == set(CLEAN_KEYS) or keys == _own_clean_keys(root):
        if table.copies > 1:
            raise AssertionError(_written_again(table))
        return
    kept = [key for key in DROPPED_KEYS if key in keys]
    gone = [key for key in CLEAN_KEYS if key not in keys]
    problems = []
    if kept:
        problems.append(f"it still has {_join(_visit_name(key) for key in kept)}, which fail the patient_id_format "
                        "or visit_date_present rule")
    if gone:
        problems.append(f"it is missing {_first([_visit_name(key) for key in gone])}")
    out_of_range = [v.key for v in TYPED if v.key in gone and not all(rule_passes(v)[r] for r in RULES[2:])]
    hint = (" A visit that breaks only a range or gender rule stays, with the bad value blanked." if out_of_range else "")
    raise AssertionError(
        f"{CLEAN_FILE} should hold the {len(CLEAN_KEYS)} visits that pass both identity rules, but "
        + "; ".join(problems) + '. In Task 5.1, keep clean[rules["patient_id_format"] & '
        f'rules["visit_date_present"]].{hint}'
    )


def check_clean_unique(root: Path) -> None:
    _, rows = _clean_rows(root)
    keys = _keys(rows)
    own_typed = _try_read(root, TYPED_FILE)
    if own_typed is not None and "patient_id" in own_typed.columns:
        typed_keys = [visit.key for visit, _, _ in _visits(own_typed) if visit is not None]
        if len(typed_keys) != len(set(typed_keys)):
            return  # Repeats kept in Task 2.1 are charged once, by the typed table's repeats check.
    repeated = [key for key in dict.fromkeys(keys) if keys.count(key) > 1]
    _assert(
        not repeated,
        f"{CLEAN_FILE} lists " + _join(f"{_visit_name(key)} {keys.count(key)} times" for key in repeated)
        + "; each visit should appear once. Task 5.2's one row per visit check, "
        'not clean.duplicated(subset=["patient_id", "visit_date"]).any(), should stop the save.',
    )


def check_clean_blanked(root: Path) -> None:
    _, rows = _clean_rows(root, "hba1c_pct", "weight_kg")
    wrong = []
    for visit, row, _ in rows:
        if visit is None:
            continue
        for column, bounds in (("hba1c_pct", HBA1C_RANGE), ("weight_kg", WEIGHT_RANGE)):
            value = visit.value(column)
            # A blanked value filled again later is charged once, by the other-values check.
            if value is not None and not _in(value, bounds) and _same(row[column], value):
                wrong.append(f"{_visit_name(visit.key)}'s {column} is {row[column]}, outside "
                             f"{_show(bounds[0])} to {_show(bounds[1])}")
    _assert(
        not wrong,
        f"{CLEAN_FILE}: {_join(wrong)}; expected a blank, because the value is impossible and the visit is kept. "
        'In Task 5.1, blank it with clean["hba1c_pct"] = clean["hba1c_pct"].where(clean["hba1c_pct"].between(4, 20)), '
        "and the same for weight_kg with 30 and 250.",
    )


def _own_fences(root: Path) -> tuple[float, float] | None:
    own = _own_visits(_try_read(root, CODED_FILE))
    if own is None:
        return None
    values = [v.glucose_mg_dl for v in own if v.glucose_mg_dl is not None and v.glucose_mg_dl != SENTINEL]
    return fences(values) if len(values) >= 4 else None


def check_clean_outliers(root: Path) -> None:
    table, rows = _clean_rows(root)
    column = "glucose_outlier" if "glucose_outlier" in table.columns else next(
        (c for c in table.columns if "outlier" in c), None)
    _assert(column is not None,
            f"{CLEAN_FILE} has no glucose_outlier column (its columns are {_columns(table)}). Task 4.2 adds "
            'labs["glucose_outlier"] = (labs["glucose_mg_dl"] < lower) | (labs["glucose_mg_dl"] > upper) before '
            "Task 5.1 copies labs.")
    options = [GLUCOSE_FENCES] + ([_own_fences(root)] if _own_fences(root) else [])
    for low, high in options:
        if all(_flag(row[column]) == (_number(row.get("glucose_mg_dl", "")) is not None
                                      and not low <= _number(row["glucose_mg_dl"]) <= high)
               for visit, row, _ in rows if visit is not None):
            return
    low, high = GLUCOSE_FENCES
    wrong = []
    for visit, row, _ in rows:
        if visit is None:
            continue
        glucose = visit.glucose_mg_dl
        expected = glucose is not None and not low <= glucose <= high
        if _flag(row[column]) != expected:
            reading = "no glucose" if glucose is None else f"glucose {_show(glucose)}"
            wrong.append(f"{_visit_name(visit.key)} ({reading}) has {row[column] or 'blank'}, "
                         f"expected {expected}")
    raise AssertionError(
        f"{CLEAN_FILE}, {column}: {_first(wrong, 3)}. The fences are Q1 - 1.5 * IQR = {_show(low)} and "
        f"Q3 + 1.5 * IQR = {_show(high)} mg/dL from labs[\"glucose_mg_dl\"].quantile(0.25) and .quantile(0.75); "
        "a blank glucose is not an outlier, which the < and > comparisons give."
    )


def check_clean_values(root: Path) -> None:
    table, rows = _clean_rows(root, *NUMBER_COLUMNS)
    coded = _try_read(root, CODED_FILE)
    own = {}
    if _own_visits(coded) is not None:
        own = {visit.key: row for visit, row, _ in _visits(coded) if visit is not None}
    wrong = []
    for visit, row, _ in rows:
        if visit is None:
            continue
        for column in NUMBER_COLUMNS:
            expected = visit.value(column)
            if column == "hba1c_pct" and expected is not None and not _in(expected, HBA1C_RANGE):
                continue  # Graded by the out-of-range check.
            if column == "weight_kg" and expected is not None and not _in(expected, WEIGHT_RANGE):
                continue
            mine = own.get(visit.key, {}).get(column)
            if not (_matches(row[column], expected) or (mine is not None and _clean(row[column]) == mine)):
                wrong.append(f"{_visit_name(visit.key)}'s {column} is {row[column] or 'blank'}, "
                             f"expected {_show(expected)}")
    if not wrong:
        return
    filled = all(word.endswith("blank (missing)") for word in wrong)
    cause = ("A missing value was filled in; Task 5 keeps blanks as blanks, since nothing says what they were."
             if filled else "Task 5.1 changes only the out-of-range HbA1c and weight values; copy every other value.")
    raise AssertionError(f"{CLEAN_FILE}: {_first(wrong, 3)}. {cause}")


CHECKS = (
    Check("missing report: one row per column", check_missing_rows),
    Check("missing report: missing_count", check_missing_count),
    Check("missing report: missing_pct", check_missing_pct),
    Check("typed table: no-lab visit dropped", check_typed_no_lab),
    Check("typed table: exact repeats removed", check_typed_repeats),
    Check("typed table: repeat visit kept", check_typed_repeat_visit),
    Check("typed table: -999 read as missing", check_typed_sentinel),
    Check("typed table: age as numbers", check_typed_age),
    Check("typed table: visit dates parsed", check_typed_dates),
    Check("typed table: other values unchanged", check_typed_values),
    Check("coded table: patient IDs cleaned", check_coded_ids),
    Check("coded table: clinic names cleaned", check_coded_clinic),
    Check("coded table: missing clinic filled", check_coded_clinic_filled),
    Check("coded table: gender codes mapped", check_coded_gender),
    Check("coded table: a1c_category", check_coded_category),
    Check("coded table: clinic indicators", check_coded_dummies),
    Check("validation report: one row per rule", check_validation_rows),
    Check("validation report: failing rows", check_validation_counts),
    Check("clean table: failing visits dropped", check_clean_dropped),
    Check("clean table: one row per visit", check_clean_unique),
    Check("clean table: impossible values blanked", check_clean_blanked),
    Check("clean table: glucose_outlier flag", check_clean_outliers),
    Check("clean table: other values kept", check_clean_values),
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
