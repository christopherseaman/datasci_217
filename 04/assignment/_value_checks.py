"""Checks for Assignment 04.

The course keeps these checks in 04/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They read the files a
submission saves in output/ and compare them with values recomputed from the
supplied data below. The self-test confirms those copies match
data/bp_followup.csv and data/home_bp.csv.

Each check scores one thing, so one mistake costs only the checks it gets
wrong, and a later file is judged against the student's own earlier one where
an earlier mistake carries into it. Values are compared after parsing:
spacing, line endings, quoting, column order, a leading row-number column,
number formatting (2 == 2.0 == 2.00, 12,5 == 12.5), the letter case of labels,
spaces or hyphens in a header (Patient ID is patient_id), and a placeholder for
a missing value (-, ., NA) never cost points, and cells may be separated by
commas, semicolons, or tabs. A later file that follows from a mistake in the
student's own output/bp_loaded.csv is judged against that table too.

Nothing here imports, runs, or inspects student code.
"""

from __future__ import annotations

import codecs
import csv
import difflib
import json
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


LOADED_FILE = "output/bp_loaded.csv"
SUMMARY_FILE = "output/visit_summary.csv"
COUNTS_FILE = "output/clinic_counts.csv"
FOLLOWUP_FILE = "output/followup_priority.csv"
PARQUET_FILE = "output/followup_priority.parquet"
GAP_FILE = "output/white_coat_gap.csv"
# The notebook step and the call that save each artifact.
SAVED_BY = {
    LOADED_FILE: ("Task 2.2", "bp.to_csv(LOADED_PATH)"),
    SUMMARY_FILE: ("Task 3.1", "visit_summary.to_csv(SUMMARY_PATH)"),
    COUNTS_FILE: ("Task 3.2", "clinic_counts.to_csv(COUNTS_PATH)"),
    FOLLOWUP_FILE: ("Task 4.3", "followup.to_csv(FOLLOWUP_PATH)"),
    PARQUET_FILE: ("Task 4.3", "followup.to_parquet(PARQUET_PATH)"),
    GAP_FILE: ("Task 5", "white_coat_gap.to_csv(GAP_PATH)"),
}

# data/bp_followup.csv without its units row and coordinator_note column:
# patient_id -> (clinic, age, sbp_baseline, sbp_week4, sbp_week8). None marks the
# export's -999 code for a reading that was not taken.
BP_EXPORT = {
    "P101": ("North", 58, 152, 146, 138),
    "P102": ("South", 64, 146, 140, 135),
    "P103": ("West", 47, 138, 134, 131),
    "P104": ("North", 71, 164, None, 150),
    "P105": ("West", 55, 158, 150, 147),
    "P106": ("East", 62, 149, 141, 136),
    "P107": ("North", 49, 140, 137, 133),
    "P108": ("South", 68, 171, 160, 152),
    "P109": ("West", 60, 155, 149, None),
    "P110": ("East", 53, 160, 152, 145),
    "P111": ("East", 66, None, 145, 139),
    "P112": ("North", 45, 132, 130, 128),
    "P113": ("East", 70, 168, 158, 155),
    "P114": ("North", 57, 145, 143, 137),
}
SENTINEL = -999
LOADED_COLUMNS = ("patient_id", "clinic", "age", "sbp_baseline", "sbp_week4", "sbp_week8")
READINGS = LOADED_COLUMNS[3:]
NOTE_COLUMN = "coordinator_note"
UNITS_ROW = ("id", "text", "years", "mmHg", "mmHg", "mmHg", "text")
# data/home_bp.csv: patient_id -> home_sbp_week8, in the file's order.
HOME_SBP = {"P113": 148, "P101": 131, "P110": 140, "P104": 146, "P115": 139, "P106": 133, "P109": 150}

PROGRAM_CLINICS = ("North", "East")
PROGRAM_MIN_BASELINE = 140
STATS = ("mean", "median", "count")
DERIVED = ("sbp_mean", "change_week8", "improvement_rank")
# A derived column saved under another name is recognized by a word in it.
DERIVED_WORDS = {"sbp_mean": "mean", "change_week8": "change", "improvement_rank": "rank"}
FOLLOWUP_COLUMNS = ("patient_id", "clinic", *READINGS, *DERIVED)

# Readings are whole mmHg; means may be saved unrounded or rounded to one decimal.
TOLERANCE = 0.051


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


@dataclass(frozen=True)
class Table:
    """A saved CSV: its casefolded column names and one {column: cell} dict per data row."""

    name: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    # How many times the table is in the file: to_csv(mode="a") writes it again below the first copy.
    copies: int = 1


# Expected values, recomputed from the supplied data.


# Each function below takes the table it computes from: the supplied BP_EXPORT, or the student's own
# bp_loaded.csv (see _worlds), so a later file is judged by the table the student actually saved.
Export = dict[str, tuple]


def reading(patient: str, column: str, export: Export | None = None) -> float | None:
    """A patient's value in the loaded table."""
    value = (export or BP_EXPORT)[patient][LOADED_COLUMNS.index(column) - 1]
    return None if value is None else float(value)


def visit_stats(export: Export | None = None) -> dict[str, dict[str, float]]:
    """{stat: {visit: value}} as pandas computes them, skipping missing readings."""
    stats: dict[str, dict[str, float]] = {stat: {} for stat in STATS}
    for visit in READINGS:
        values = [reading(p, visit, export) for p in (export or BP_EXPORT)]
        present = [value for value in values if value is not None]
        stats["mean"][visit] = statistics.mean(present) if present else float("nan")
        stats["median"][visit] = statistics.median(present) if present else float("nan")
        stats["count"][visit] = float(len(present))
    return stats


def clinic_counts(export: Export | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for clinic, *_ in (export or BP_EXPORT).values():
        counts[clinic] = counts.get(clinic, 0) + 1
    return counts


def in_program(patient: str, export: Export | None = None) -> bool:
    baseline = reading(patient, "sbp_baseline", export)
    clinic = (export or BP_EXPORT)[patient][0]
    return clinic in PROGRAM_CLINICS and baseline is not None and baseline >= PROGRAM_MIN_BASELINE


def change(patient: str, export: Export | None = None) -> float | None:
    week8, baseline = reading(patient, "sbp_week8", export), reading(patient, "sbp_baseline", export)
    return None if week8 is None or baseline is None else week8 - baseline


def sbp_mean(patient: str) -> float:
    present = [reading(patient, visit) for visit in READINGS]
    return statistics.mean(value for value in present if value is not None)


def min_rank(values: dict[str, float | None], descending: bool = False) -> dict[str, float | None]:
    """rank(method="min"): ties share the best place; a missing value has no rank."""
    present = [value for value in values.values() if value is not None]
    sign = -1 if descending else 1
    return {
        key: None if value is None else 1.0 + sum(1 for other in present if sign * other < sign * value - 1e-9)
        for key, value in values.items()
    }


def average_rank(values: dict[str, float | None]) -> dict[str, float | None]:
    """rank() with its default method: ties share the mean of their places."""
    present = [value for value in values.values() if value is not None]
    ranks: dict[str, float | None] = {}
    for key, value in values.items():
        if value is None:
            ranks[key] = None
            continue
        below = sum(1 for other in present if other < value - 1e-9)
        tied = sum(1 for other in present if abs(other - value) <= 1e-9)
        ranks[key] = below + (tied + 1) / 2
    return ranks


def program_order(export: Export | None = None) -> list[str]:
    """The follow-up patients as Task 4.3 sorts them: largest drop first, ties by patient_id."""
    patients = [p for p in (export or BP_EXPORT) if in_program(p, export)]
    return sorted(patients, key=lambda p: (change(p, export), p))


def gap_expected(export: Export | None = None) -> dict[str, float | None]:
    """Clinic week-8 minus home reading for every patient in either table; None where one side is missing."""
    patients = sorted(set(export or BP_EXPORT) | set(HOME_SBP))
    expected = {}
    for patient in patients:
        clinic = reading(patient, "sbp_week8", export) if patient in (export or BP_EXPORT) else None
        home = HOME_SBP.get(patient)
        expected[patient] = None if clinic is None or home is None else clinic - home
    return expected


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
    """The rows of a saved CSV, blank lines left out, split on the separator its header line uses.

    Whichever of commas, semicolons, and tabs splits the header line into the
    most cells is used, and a tie keeps commas. A cell holding a decimal comma
    (a semicolon- or tab-separated file from a European locale, or a quoted cell)
    reads 12,5 as 12.5. In a one-column file a missing value is the row "", kept.
    """
    if "\x00" in text:
        raise csv.Error("embedded NUL bytes; save the table again as CSV text")
    lines = text.splitlines()
    header = next((line for line in lines if line.strip()), "")
    delimiter = max(DELIMITERS, key=lambda mark: len(next(csv.reader([header], delimiter=mark), [])))
    one_column = len(next(csv.reader([header], delimiter=delimiter), [])) == 1
    rows = [row for row in csv.reader(lines, delimiter=delimiter)
            if any(cell.strip() for cell in row) or (one_column and row == [""])]
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
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", _clean(cell))
    return re.sub(r"[\s\-]+", "_", name).casefold()


def _number(cell: str) -> float | None:
    text = cell.strip().replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return None if number != number else number  # NaN is a missing value, not a number


def _same(given: str, expected: float) -> bool:
    number = _number(given)
    return number is not None and abs(number - expected) <= TOLERANCE


def _blank(cell: str) -> bool:
    """A missing value as pandas or a spreadsheet writes it."""
    return cell.strip().casefold() in ("", "nan", "na", "n/a", "<na>", "none", "null", "-", ".")


def _matches(cell: str, expected: float | None) -> bool:
    return _blank(cell) if expected is None else _same(cell, expected)


def _show(value: float | None) -> str:
    if value is None:
        return "blank (missing)"
    return f"{value:g}" if float(value).is_integer() else f"{value:.2f}"


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


def read_table(root: Path, name: str) -> Table:
    """Parse a saved CSV, however it is separated, spaced, quoted, or ended.

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
    body = [[_clean(cell) for cell in row] for row in lines[starts[-1] + 1:]]
    # mode="a" with header=False repeats only the rows: a body that is one block of rows over and over.
    for size in range(1, len(body) // 2 + 1):
        if len(body) % size == 0 and body == body[:size] * (len(body) // size):
            copies *= len(body) // size
            body = body[:size]
            break
    while (
        len(header) > 1
        and (header[0] in ("", "index") or header[0].startswith("unnamed"))
        and body
        and all(row and _number(row[0]) is not None and _number(row[0]).is_integer() for row in body)
    ):
        header = header[1:]
        body = [row[1:] for row in body]
    rows = tuple({column: (row[i] if i < len(row) else "") for i, column in enumerate(header)} for row in body)
    return Table(name=name, columns=tuple(dict.fromkeys(header)), rows=rows, copies=copies)


def _written_again(table: Table) -> str:
    """The fix when the file holds its table more than once, as to_csv(mode="a") leaves it."""
    step, call = SAVED_BY[table.name]
    return (
        f"{table.name} holds the same table {table.copies} times: to_csv() with mode=\"a\" adds to the end of the "
        f"file instead of replacing it. The checks graded the last copy; in {step}, save once with {call}."
    )


def _key_column(table: Table, known: set[str]) -> str | None:
    """The column holding patient IDs: patient_id, or else the first column naming a known patient."""
    if "patient_id" in table.columns:
        return "patient_id"
    for column in table.columns:
        if any(row[column].casefold() in known for row in table.rows):
            return column
    return None


def _patients(table: Table, known: set[str], by_readings: bool = False) -> list[str | None]:
    """The patient each row is, in file order, or None for a row that names no known patient.

    Rows are named by their patient IDs. With by_readings and no ID column, a
    row is the one patient whose age and readings match at least two of its
    cells, so leaving the IDs out costs only the patient_id check.
    """
    known_cf = {patient.casefold(): patient for patient in known}
    key = _key_column(table, set(known_cf))
    if key is not None:
        return [known_cf.get(row[key].casefold()) for row in table.rows]
    if not by_readings:
        return [None for _ in table.rows]
    named = []
    for row in table.rows:
        matches = [
            patient for patient in BP_EXPORT
            if sum(
                1 for column in ("age", *READINGS)
                if column in row and reading(patient, column) is not None
                and _same(row[column], reading(patient, column))
            ) >= 2
        ]
        named.append(matches[0] if len(matches) == 1 else None)
    return named


def _rows_by_patient(table: Table, known: set[str], by_readings: bool = False) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for patient, row in zip(_patients(table, known, by_readings), table.rows):
        if patient is not None:
            found.setdefault(patient, row)
    return found


def _id_check(name: str, step_hint: str, read: Callable[[Path], Table]) -> Callable[[Path], None]:
    """The saved table keeps the patient IDs in a patient_id column."""

    def check(root: Path) -> None:
        table = read(root)
        if "patient_id" in table.columns:
            return
        key = _key_column(table, {patient.casefold() for patient in BP_EXPORT})
        if key is not None:
            where = f"a column headed {key}" if key else "a first column with no header"
            raise AssertionError(
                f"{name} has the patient IDs in {where}, not patient_id. {step_hint}"
            )
        raise AssertionError(
            f"{name} has no patient_id column (its columns are {_columns(table)}), so the patient IDs were lost. "
            f"{step_hint}"
        )

    return check


# Task 2: output/bp_loaded.csv


LOAD_CALL = (
    'bp = pd.read_csv(DATA_PATH, sep=";", skiprows=[1], usecols=[...], na_values=["-999"], index_col="patient_id")'
)


def _is_units_row(row: dict[str, str]) -> bool:
    cells = [cell.casefold() for cell in row.values()]
    return "mmhg" in cells or (bool(cells) and cells[0] == "id")


UNITS_HEADER = (
    f"{LOADED_FILE} uses the units row (id, text, years, mmHg, ...) as its header, so the column names were "
    "skipped instead: skiprows counts the header line as 0, so skiprows=[1] skips the units row below it."
)


def _loaded(root: Path) -> Table:
    """The saved table, for the checks that need its column names: a header of units has none."""
    table = read_table(root, LOADED_FILE)
    _assert("mm_hg" not in table.columns or "patient_id" in table.columns, UNITS_HEADER)
    return table


def check_loaded_units(root: Path) -> None:
    table = _loaded(root)
    units = [number for number, row in enumerate(table.rows, start=1) if _is_units_row(row)]
    _assert(
        not units,
        f"{LOADED_FILE} still has the units row ({', '.join(UNITS_ROW[:4])}, ...) as record {_join(map(str, units))}, "
        "which also makes every reading column text. In Task 2.2, skip it with skiprows=[1]: line 1, counting the "
        "header line as 0.",
    )


def check_loaded_note(root: Path) -> None:
    table = read_table(root, LOADED_FILE)
    _assert(
        NOTE_COLUMN not in table.columns,
        f"{LOADED_FILE} also has {NOTE_COLUMN}, the coordinator's free text, which the analysis does not use. "
        f"In Task 2.2, read only the six columns {_join(LOADED_COLUMNS)} with usecols=[...].",
    )


def check_loaded_patients(root: Path) -> None:
    table = read_table(root, LOADED_FILE)
    rows = [row for row in table.rows if not _is_units_row(row)]
    named = _patients(Table(table.name, table.columns, tuple(rows)), set(BP_EXPORT), by_readings=True)
    missing = [patient for patient in BP_EXPORT if patient not in named]
    repeated = [patient for patient in dict.fromkeys(named) if patient is not None and named.count(patient) > 1]
    unknown = named.count(None)
    problems = []
    if missing:
        problems.append(f"it is missing {_first(missing)}")
    if repeated:
        problems.append("it lists " + _join(f"{p} {named.count(p)} times" for p in repeated))
    if unknown:
        problems.append(f"{unknown} of its rows match{'es' if unknown == 1 else ''} no patient in data/bp_followup.csv")
    if table.copies > 1 and not problems:
        raise AssertionError(_written_again(table))
    _assert(
        not problems,
        f"{LOADED_FILE} should hold each of the {len(BP_EXPORT)} patients P101 to P114 once, but "
        + "; ".join(problems) + f". In Task 2.2, read the whole file (leave out nrows) and save bp itself.",
    )


def check_loaded_sentinel(root: Path) -> None:
    table = read_table(root, LOADED_FILE)
    found = _rows_by_patient(table, set(BP_EXPORT), by_readings=True)
    kept = []
    for patient, row in found.items():
        for column in READINGS:
            if reading(patient, column) is None and column in row and not _blank(row[column]):
                kept.append(f"{patient}'s {column} is {row[column]}")
    _assert(
        not kept,
        f"{LOADED_FILE}: {_join(kept)}, where the export's -999 means the reading was not taken; expected a blank "
        '(missing) cell. In Task 2.2, add na_values=["-999"] so pandas reads the code as NaN and every mean skips it.',
    )


def check_loaded_values(root: Path) -> None:
    table = _loaded(root)
    absent = [column for column in LOADED_COLUMNS[1:] if column not in table.columns]
    _assert(
        not absent,
        f"{LOADED_FILE} has no {_join(absent)} column (its columns are {_columns(table)}); "
        f"keep all of {_join(LOADED_COLUMNS)} in usecols.",
    )
    found = _rows_by_patient(table, set(BP_EXPORT), by_readings=True)
    _assert(bool(found), f"{LOADED_FILE} names no patient from data/bp_followup.csv, so its values cannot be compared.")
    wrong = []
    for patient, row in found.items():
        clinic = BP_EXPORT[patient][0]
        if row["clinic"].casefold() != clinic.casefold():
            wrong.append(f"{patient}'s clinic is {row['clinic'] or 'blank'}, expected {clinic}")
        for column in ("age", *READINGS):
            expected = reading(patient, column)
            if expected is not None and not _same(row[column], expected):
                wrong.append(f"{patient}'s {column} is {row[column] or 'blank'}, expected {_show(expected)}")
    _assert(
        not wrong,
        f"{LOADED_FILE}: {_first(wrong, 3)}, as in data/bp_followup.csv. Keep the data file unchanged and save the "
        "table as read.",
    )


def _own_export(root: Path) -> Export | None:
    """The student's own bp_loaded.csv as an export, or None when it cannot be read as one."""
    try:
        table = read_table(root, LOADED_FILE)
    except (AssertionError, OSError, csv.Error):
        return None
    if not {"clinic", *READINGS} <= set(table.columns):
        return None
    own: Export = {}
    for patient, row in _rows_by_patient(table, set(BP_EXPORT), by_readings=True).items():
        clinic = BP_EXPORT[patient][0]
        if row["clinic"].casefold() != clinic.casefold():
            clinic = row["clinic"]
        age = _number(row.get("age", ""))
        own[patient] = (clinic, age, *(_number(row[column]) for column in READINGS))
    return own or None


def _worlds(root: Path) -> list[Export]:
    """The tables a later file may be right for: the supplied data, then the student's own bp_loaded.csv.

    A mistake in bp_loaded.csv (keeping -999, nrows=10) carries into every later
    file, so those are judged against it too and the mistake is charged once.
    """
    own = _own_export(root)
    return [BP_EXPORT] + ([own] if own is not None and own != BP_EXPORT else [])


# Task 3.1: output/visit_summary.csv


STAT_NAMES = {"mean": "mean", "median": "median", "50%": "median", "count": "count"}


def _unlabeled_summary(table: Table) -> bool:
    """Three rows of mean, median, or count with no visit names: the summary saved with index=False."""
    visits = set(READINGS)
    named = any(row[c].casefold() in visits for c in table.columns for row in table.rows)
    return (not named and len(table.rows) == len(READINGS)
            and any(c in STAT_NAMES for c in table.columns) and not any(c in visits for c in table.columns))


def _summary_grid(table: Table) -> tuple[dict[str, dict[str, str]], list[str]]:
    """({stat: {visit: cell}}, the visit labels in file order), whichever way round the table was saved.

    Visits as rows is the Task 3.1 layout; visits as columns is the same summary
    saved with .T, or describe(), whose 50% row is the median.
    """
    visits = set(READINGS)
    label = next((c for c in table.columns if any(row[c].casefold() in visits for row in table.rows)), None)
    grid: dict[str, dict[str, str]] = {}
    if label is not None:
        for column in table.columns:
            if column != label and column in STAT_NAMES:
                grid[STAT_NAMES[column]] = {row[label].casefold(): row[column] for row in table.rows}
        return grid, [row[label].casefold() for row in table.rows]
    if _unlabeled_summary(table):
        for column in table.columns:
            if column in STAT_NAMES:
                grid[STAT_NAMES[column]] = {visit: row[column] for visit, row in zip(READINGS, table.rows)}
        return grid, list(READINGS)
    columns = [c for c in table.columns if c in visits]
    if columns:
        first = table.columns[0]
        for row in table.rows:
            stat = STAT_NAMES.get(row[first].casefold())
            if stat is not None:
                grid[stat] = {c: row[c] for c in columns}
        return grid, columns
    return grid, []


def _no_visits(table: Table) -> str:
    """The one fix when the summary names no visit, shared by every summary check it fails."""
    if any(_patients(table, set(BP_EXPORT))):
        return (
            f"{SUMMARY_FILE} has one row per patient, so its reductions ran across each row, as "
            'axis="columns" does. Leave axis out in Task 3.1: the default runs down each column, one value per visit.'
        )
    return (
        f"{SUMMARY_FILE} names none of {_join(READINGS)} (its columns are {_columns(table)}); in Task 3.1, "
        "summarize readings = bp[[" + ", ".join(f'"{v}"' for v in READINGS) + "]], one row per visit."
    )


def check_summary_visits(root: Path) -> None:
    table = read_table(root, SUMMARY_FILE)
    _assert(
        not _unlabeled_summary(table),
        f"{SUMMARY_FILE} has no visit names (its columns are {_columns(table)}): they are the index, and "
        "index=False leaves them out. In Task 3.1, save with visit_summary.to_csv(SUMMARY_PATH), keeping the index.",
    )
    _, labels = _summary_grid(table)
    _assert(bool(labels), _no_visits(table))
    missing = [visit for visit in READINGS if visit not in labels]
    repeated = [visit for visit in READINGS if labels.count(visit) > 1]
    problems = ([f"it has no {_join(missing)}"] if missing else []) + (
        [f"it lists {_join(repeated)} more than once"] if repeated else [])
    if table.copies > 1 and not problems:
        raise AssertionError(_written_again(table))
    _assert(
        not problems,
        f"{SUMMARY_FILE} should have one row for each of {_join(READINGS)}, but " + "; ".join(problems)
        + ". In Task 3.1, summarize readings = bp[[" + ", ".join(f'"{v}"' for v in READINGS) + "]].",
    )


def _summary_stat(stat: str) -> Callable[[Path], None]:
    call = {"mean": "readings.mean()", "median": "readings.median()", "count": "readings.count()"}[stat]

    def check(root: Path) -> None:
        table = read_table(root, SUMMARY_FILE)
        grid, labels = _summary_grid(table)
        if not labels and any(_patients(table, set(BP_EXPORT))):
            return  # Reductions run across rows are charged once, by the one-row-per-visit check.
        _assert(bool(labels), _no_visits(table))
        _assert(
            stat in grid,
            f"{SUMMARY_FILE} has no {stat} column (its columns are {_columns(table)}); in Task 3.1, "
            f'build visit_summary with "{stat}": {call}.',
        )
        saved = grid[stat]
        for expected in (visit_stats(world) for world in _worlds(root)):
            if all(visit not in saved or _same(saved[visit], expected[stat][visit]) for visit in READINGS):
                return
        truth = visit_stats()[stat]
        wrong = [
            f"{visit} has {saved[visit] or 'blank'}, expected {_show(truth[visit])}"
            for visit in READINGS if visit in saved and not _same(saved[visit], truth[visit])
        ]
        hint = (" count() counts readings present, so each visit's one missing reading is left out."
                if stat == "count" else f" {call} skips the missing readings and runs down each column, one value per visit.")
        raise AssertionError(f"{SUMMARY_FILE}, {stat}: {'; '.join(wrong)}.{hint}")

    return check


# Task 3.2: output/clinic_counts.csv


def _counts(table: Table) -> list[tuple[str, str]]:
    """(clinic, count cell) for each row naming a clinic, in file order."""
    clinics = {clinic.casefold() for clinic in clinic_counts()}
    label = next((c for c in table.columns if any(row[c].casefold() in clinics for row in table.rows)), None)
    others = [c for c in table.columns if c != label]
    value = "count" if "count" in others else (others[0] if others else None)
    if label is None or value is None:
        return []
    return [(row[label], row[value]) for row in table.rows if row[label].casefold() in clinics]


def check_counts_values(root: Path) -> None:
    table = read_table(root, COUNTS_FILE)
    saved = _counts(table)
    _assert(
        bool(saved) or _count_column(table) is None,
        f"{COUNTS_FILE} has counts but no clinic names (its columns are {_columns(table)}): the clinics are the "
        "index, and index=False leaves them out. In Task 3.2, save with clinic_counts.to_csv(COUNTS_PATH), keeping "
        "the index.",
    )
    _assert(
        bool(saved),
        f"{COUNTS_FILE} names no clinic (its columns are {_columns(table)}); in Task 3.2, save "
        'clinic_counts = bp["clinic"].value_counts().',
    )

    def problems_for(expected: dict[str, int]) -> list[str]:
        by_name = {clinic.casefold(): clinic for clinic in expected}
        named = [by_name.get(clinic.casefold()) for clinic, _ in saved]
        found = [f"{clinic} has {count or 'blank'}, expected {expected[by_name[clinic.casefold()]]}"
                 for clinic, count in saved
                 if clinic.casefold() in by_name and not _same(count, expected[by_name[clinic.casefold()]])]
        found += [f"{clinic} is not a clinic in bp" for clinic, _ in saved if clinic.casefold() not in by_name]
        found += [f"it has no row for {clinic}" for clinic in expected if clinic not in named]
        found += [f"it lists {clinic} {named.count(clinic)} times" for clinic in expected if named.count(clinic) > 1]
        return found

    options = [problems_for(clinic_counts(world)) for world in _worlds(root)]
    if not any(not found for found in options):
        problems = options[0]
        raise AssertionError(
            f"{COUNTS_FILE}: {'; '.join(problems)}. value_counts() counts each clinic's patients in bp; "
            "a fraction instead of a count means normalize=True was added."
        )
    if table.copies > 1 or len(table.rows) > len(saved):
        raise AssertionError(_written_again(table) if table.copies > 1 else
                             f"{COUNTS_FILE} has rows that name no clinic; save only the value_counts() result.")


def _count_column(table: Table) -> str | None:
    """The first column whose cells are all numbers, for a counts file that lost its clinic names."""
    return next((c for c in table.columns if table.rows and all(_number(row[c]) is not None for row in table.rows)),
                None)


def check_counts_order(root: Path) -> None:
    table = read_table(root, COUNTS_FILE)
    saved = [(clinic, _number(count)) for clinic, count in _counts(table)]
    column = _count_column(table)
    if not saved and column is not None:
        # The clinic names are charged once, by the patients-per-clinic check; the counts still have an order.
        numbers = [_number(row[column]) for row in table.rows]
        rise = next(((a, b) for a, b in zip(numbers, numbers[1:]) if a < b), None)
        _assert(
            rise is None,
            f"{COUNTS_FILE} lists a count of {_show(rise[0] if rise else 0)} before {_show(rise[1] if rise else 0)}; "
            "value_counts() puts the most common clinic first, so save its result as it is.",
        )
        return
    _assert(len(saved) >= 2, f"{COUNTS_FILE} has fewer than two clinic counts, so their order cannot be checked.")
    rise = next(((a, b) for a, b in zip(saved, saved[1:])
                 if a[1] is not None and b[1] is not None and a[1] < b[1]), None)
    if rise is not None:
        raise AssertionError(
            f"{COUNTS_FILE} lists {rise[0][0]} ({_show(rise[0][1])} patients) before {rise[1][0]} "
            f"({_show(rise[1][1])}); value_counts() puts the most common clinic first, so save its result as it is."
        )


# Task 4: output/followup_priority.csv and output/followup_priority.parquet


def _followup_column(table: Table, column: str) -> str | None:
    """The derived column under its own name, or else the one extra column whose name says what it holds."""
    if column in table.columns:
        return column
    stand_ins = [c for c in table.columns if c not in FOLLOWUP_COLUMNS and DERIVED_WORDS[column] in c]
    return stand_ins[0] if len(stand_ins) == 1 else None


def _followup_rows(table: Table) -> list[tuple[str, dict[str, str]]]:
    named = _patients(table, set(BP_EXPORT), by_readings=True)
    return [(patient, row) for patient, row in zip(named, table.rows) if patient is not None]


def _own_mean(row: dict[str, str]) -> float | None:
    values = [_number(row[c]) for c in READINGS if c in row]
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def _own_change(row: dict[str, str]) -> float | None:
    week8, baseline = _number(row.get("sbp_week8", "")), _number(row.get("sbp_baseline", ""))
    return None if week8 is None or baseline is None else week8 - baseline


def check_followup_rows(root: Path) -> None:
    table = read_table(root, FOLLOWUP_FILE)
    named = [patient for patient, _ in _followup_rows(table)]
    repeated = [p for p in dict.fromkeys(named) if named.count(p) > 1]
    unknown = len(table.rows) - len(named)

    def differs(expected: list[str]) -> tuple[list[str], list[str]]:
        return [p for p in expected if p not in named], [p for p in dict.fromkeys(named) if p not in expected]

    expected = program_order()
    missing, extra = differs(expected)
    if any(not (repeated or unknown or any(differs(program_order(world)))) for world in _worlds(root)):
        missing = extra = []
    if not (missing or extra or repeated or unknown):
        if table.copies > 1:
            raise AssertionError(_written_again(table))
        return
    if missing == ["P107"] and not (extra or repeated or unknown):
        raise AssertionError(
            f"{FOLLOWUP_FILE} is missing P107, whose baseline is exactly {PROGRAM_MIN_BASELINE} mmHg: "
            f'bp["sbp_baseline"] >= {PROGRAM_MIN_BASELINE} keeps it, while > {PROGRAM_MIN_BASELINE} drops it.'
        )
    causes = []
    other_clinic = [p for p in extra if BP_EXPORT[p][0] not in PROGRAM_CLINICS]
    low = [p for p in extra if BP_EXPORT[p][0] in PROGRAM_CLINICS]
    if other_clinic:
        causes.append(f"{_first(other_clinic)} {'is' if len(other_clinic) == 1 else 'are'} not at North or East, "
                      'which bp["clinic"].isin(["North", "East"]) keeps')
    if low:
        causes.append(f"{_first(low)} {'has' if len(low) == 1 else 'have'} a baseline below "
                      f"{PROGRAM_MIN_BASELINE} mmHg or none at all")
    if missing:
        causes.append(f"it is missing {_first(missing)}")
    if repeated:
        causes.append("it lists " + _join(f"{p} {named.count(p)} times" for p in repeated))
    if unknown:
        causes.append(f"{unknown} of its rows name no patient in data/bp_followup.csv")
    raise AssertionError(
        f"{FOLLOWUP_FILE} should hold the {len(expected)} program patients ({_join(sorted(expected))}), but "
        + "; ".join(causes) + ". In Task 4.1, keep the rows where both conditions hold: "
        f'bp["clinic"].isin(["North", "East"]) & (bp["sbp_baseline"] >= {PROGRAM_MIN_BASELINE}).'
    )


def check_followup_names(root: Path) -> None:
    table = read_table(root, FOLLOWUP_FILE)
    renamed = [(column, _followup_column(table, column)) for column in DERIVED
               if column not in table.columns and _followup_column(table, column) is not None]
    _assert(
        not renamed,
        f"{FOLLOWUP_FILE} has " + _join(f"{saved} where Task 4.2 names it {column}" for column, saved in renamed)
        + "; rename it to match, so later work can find it by name.",
    )


def check_followup_age(root: Path) -> None:
    table = read_table(root, FOLLOWUP_FILE)
    _assert(
        "age" not in table.columns,
        f'{FOLLOWUP_FILE} still has the age column; in Task 4.1, follow the selection with .drop(columns=["age"]).',
    )


def _derived_values(column: str) -> Callable[[Path], None]:
    """The derived column's values: right by the supplied readings or by the row's own saved ones."""
    truth = {"sbp_mean": sbp_mean, "change_week8": change}[column]
    own = {"sbp_mean": _own_mean, "change_week8": _own_change}[column]
    how = {
        "sbp_mean": 'followup[["sbp_baseline", "sbp_week4", "sbp_week8"]].mean(axis="columns"), which skips '
                    "a missing reading",
        "change_week8": 'followup["sbp_week8"] - followup["sbp_baseline"], negative when pressure fell',
    }[column]

    def check(root: Path) -> None:
        table = read_table(root, FOLLOWUP_FILE)
        saved = _followup_column(table, column)
        _assert(saved is not None,
                f"{FOLLOWUP_FILE} has no {column} column (its columns are {_columns(table)}); Task 4.2 adds "
                f"{column} = {how}.")
        rows = [(p, row) for p, row in _followup_rows(table) if in_program(p)]
        _assert(bool(rows), f"{FOLLOWUP_FILE} has none of the program patients, so its {column} values cannot "
                            "be compared; the program patients check says what to fix.")
        wrong = []
        for patient, row in rows:
            expected, recomputed = truth(patient), own(row)
            if not (_same(row[saved], expected) or (recomputed is not None and _same(row[saved], recomputed))):
                wrong.append(f"{patient} has {row[saved] or 'blank'}, expected {_show(expected)}")
        _assert(not wrong, f"{FOLLOWUP_FILE}, {column}: {_first(wrong, 3)}. Task 4.2 computes it as {how}.")

    return check


def check_followup_rank(root: Path) -> None:
    table = read_table(root, FOLLOWUP_FILE)
    saved = _followup_column(table, "improvement_rank")
    _assert(saved is not None,
            f"{FOLLOWUP_FILE} has no improvement_rank column; Task 4.3 adds "
            'improvement_rank = followup["change_week8"].rank(method="min").')
    rows = [(p, row) for p, row in _followup_rows(table) if in_program(p)]
    _assert(bool(rows), f"{FOLLOWUP_FILE} has none of the program patients, so its ranks cannot be compared.")
    program = {p: change(p) for p in program_order()}
    everyone = {p: change(p) for p in BP_EXPORT}
    own_change = _followup_column(table, "change_week8")
    accepted = [min_rank(program), min_rank(everyone)]
    if own_change is not None:
        accepted.append(min_rank({p: _number(row[own_change]) for p, row in rows}))
        every_row = {p: _number(row[own_change]) for p, row in _followup_rows(table)}
        accepted.append(min_rank(every_row))
    if any(all(_matches(row[saved], ranks.get(p)) for p, row in rows) for ranks in accepted):
        return
    truth = min_rank(program)
    wrong = [f"{p} has {row[saved] or 'blank'}, expected {_show(truth[p])}" for p, row in rows
             if not _matches(row[saved], truth[p])]
    if all(_matches(row[saved], average_rank(program)[p]) for p, row in rows):
        cause = 'Tied patients share the mean of their places, as rank() does by default; add method="min".'
    elif all(_matches(row[saved], min_rank(program, descending=True)[p]) for p, row in rows):
        cause = "The smallest drop is ranked 1; leave out ascending=False, so the most negative change ranks first."
    else:
        cause = 'Task 4.3 ranks with followup["change_week8"].rank(method="min"): the largest drop is 1.'
    raise AssertionError(f"{FOLLOWUP_FILE}, improvement_rank: {_first(wrong, 3)}. {cause}")


def _order_bases(table: Table, rows: list[tuple[str, dict[str, str]]]) -> list[dict[str, float]]:
    """The changes the order is judged by: the true ones, and the file's own when every one is a number."""
    bases = [{p: change(p) for p, _ in rows}]
    saved = _followup_column(table, "change_week8")
    if saved is not None:
        own = {p: _number(row[saved]) for p, row in rows}
        if all(value is not None for value in own.values()):
            bases.append(own)
    return bases


def _sorted_rows(root: Path) -> tuple[list[str], list[dict[str, float]]]:
    table = read_table(root, FOLLOWUP_FILE)
    rows = [(p, row) for p, row in _followup_rows(table) if in_program(p)]
    _assert(len(rows) >= 2, f"{FOLLOWUP_FILE} names fewer than two program patients, so its order cannot be "
                            "checked; the program patients check says what to fix.")
    return [p for p, _ in rows], _order_bases(table, rows)


def _first_rise(ids: list[str], changes: dict[str, float]) -> tuple[str, str] | None:
    return next(((a, b) for a, b in zip(ids, ids[1:]) if changes[a] > changes[b] + TOLERANCE), None)


def check_followup_sorted(root: Path) -> None:
    ids, bases = _sorted_rows(root)
    if any(_first_rise(ids, changes) is None for changes in bases):
        return
    if ids == [p for p in BP_EXPORT if p in ids]:
        raise AssertionError(
            f"{FOLLOWUP_FILE} lists its patients in data order ({', '.join(ids[:3])}, ...), so they were saved "
            "unsorted. sort_values() returns a new table, so assign it back: followup = followup.sort_values("
            'by=["change_week8", "patient_id"]), then save again.'
        )
    first, second = _first_rise(ids, bases[0])
    raise AssertionError(
        f"{FOLLOWUP_FILE} is not sorted from the largest drop to the smallest: {first} (change "
        f"{_show(change(first))}) comes before {second} ({_show(change(second))}). The largest drop is the most "
        'negative change, so in Task 4.3 sort ascending: by=["change_week8", "patient_id"].'
    )


def check_followup_ties(root: Path) -> None:
    ids, bases = _sorted_rows(root)
    if not any(_first_rise(ids, changes) is None for changes in bases):
        return  # A file not sorted by change is charged once, by the largest-drop-first check.
    for changes in bases:
        tie = next(((a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
                    if abs(changes[a] - changes[b]) <= TOLERANCE and a > b), None)
        if tie is None:
            return
    first, second = next((a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
                         if abs(change(a) - change(b)) <= TOLERANCE and a > b)
    raise AssertionError(
        f"{FOLLOWUP_FILE} lists {first} before {second}, but both changed by {_show(change(first))} mmHg, so the "
        f"tie goes to the smaller patient_id, {second}. In Task 4.3, sort with by=[\"change_week8\", \"patient_id\"]."
    )


def _parquet(root: Path) -> bytes:
    path = _artifact(root, PARQUET_FILE)
    _assert(path is not None, _missing(root, PARQUET_FILE))
    data = path.read_bytes()
    if len(data) >= 12 and data[:4] == b"PAR1" and data[-4:] == b"PAR1":
        return data
    kind = "CSV text, as to_csv() writes" if b"," in data[:200] else "not Parquet"
    raise AssertionError(
        f"{PARQUET_FILE} is {kind}: a Parquet file starts and ends with the bytes PAR1. "
        f"In {SAVED_BY[PARQUET_FILE][0]}, save with {SAVED_BY[PARQUET_FILE][1]}."
    )


def check_parquet_file(root: Path) -> None:
    _parquet(root)


def _parquet_columns(data: bytes) -> list[str] | None:
    """The column names pandas recorded in the file's footer, index included, or None without them."""
    size = int.from_bytes(data[-8:-4], "little")
    footer = data[-8 - size:-8] if 0 < size <= len(data) - 12 else data
    start = footer.find(b'{"index_columns"')
    if start < 0:
        return None
    try:
        metadata, _ = json.JSONDecoder().raw_decode(footer[start:].decode("utf-8", errors="replace"))
    except ValueError:
        return None
    return [str(column.get("name") or column.get("field_name") or "") for column in metadata.get("columns", [])]


def check_parquet_columns(root: Path) -> None:
    columns = _parquet_columns(_parquet(root))
    _assert(columns is not None,
            f"{PARQUET_FILE} records no pandas column list; save it from pandas with {SAVED_BY[PARQUET_FILE][1]}.")
    # A row-number index pandas saved for a table without index_col is not a column of followup.
    columns = [column for column in columns if not re.fullmatch(r"__index_level_\d+__", column)]
    saved = {_header(column) for column in columns}
    accepted = [set(FOLLOWUP_COLUMNS)]
    try:
        own = read_table(root, FOLLOWUP_FILE)
        accepted.append(set(own.columns))
    except (AssertionError, OSError, csv.Error):
        pass
    if saved in accepted:
        return
    if "patient_id" not in saved and saved | {"patient_id"} in accepted:
        raise AssertionError(
            f"{PARQUET_FILE} has no patient IDs: the index holds them, and index=False leaves it out. "
            f"Save with {SAVED_BY[PARQUET_FILE][1]}, keeping the index."
        )
    expected = accepted[-1]
    missing = sorted(expected - saved)
    extra = sorted(saved - expected)
    if len(missing) == 1 and len(extra) == 1:
        raise AssertionError(
            f"{PARQUET_FILE} has a column named {extra[0]} where {FOLLOWUP_FILE} has {missing[0]}; rename it to "
            f"{missing[0]} (followup.rename(columns=...)), then save again with {SAVED_BY[PARQUET_FILE][1]}."
        )
    raise AssertionError(
        f"{PARQUET_FILE} has the columns {_join(columns) or 'none'}"
        + (f"; it is missing {_join(missing)}" if missing else "") + (f"; it also has {_join(extra)}" if extra else "")
        + f". Save the same followup table as {FOLLOWUP_FILE}, after Task 4.3 adds improvement_rank."
    )


# Task 5: output/white_coat_gap.csv


def _gap(table: Table) -> tuple[str | None, str | None]:
    """(the patient ID column, the gap column)."""
    key = _key_column(table, {p.casefold() for p in gap_expected()})
    others = [c for c in table.columns if c != key]
    gaps = [c for c in others if "gap" in c]
    value = gaps[0] if len(gaps) == 1 else (others[0] if len(others) == 1 else None)
    return key, value


def check_gap_rows(root: Path) -> None:
    table = read_table(root, GAP_FILE)
    key, _ = _gap(table)
    expected = list(gap_expected())
    _assert(
        key is not None,
        f"{GAP_FILE} has no patient IDs (its columns are {_columns(table)}); in Task 5, read home_bp.csv with "
        'index_col="patient_id" and save the result with its index, so leave out index=False.',
    )
    labels = [row[key] for row in table.rows]
    known = {p.casefold(): p for p in expected}
    named = [known.get(label.casefold()) for label in labels]
    unknown = [label for label, p in zip(labels, named) if p is None]
    missing = [p for p in expected if p not in named]
    repeated = [p for p in expected if named.count(p) > 1]
    if not unknown and not repeated and any(set(named) == set(gap_expected(world)) for world in _worlds(root)):
        missing = []
    if not (unknown or missing or repeated):
        if table.copies > 1:
            raise AssertionError(_written_again(table))
        return
    if unknown and all(_number(label) is not None for label in unknown):
        raise AssertionError(
            f"{GAP_FILE} has the row labels {_first(unknown)} beside the patient IDs: bp or home was read with "
            'row numbers as its index, so no label matched. Read both with index_col="patient_id" (Task 2.2 and Task 5).'
        )
    problems = ([f"it is missing {_first(missing)}"] if missing else []) + (
        [f"it also has {_first(unknown)}"] if unknown else []) + (
        ["it lists " + _join(f"{p} {named.count(p)} times" for p in repeated)] if repeated else [])
    raise AssertionError(
        f"{GAP_FILE} should have one row for each of the {len(expected)} patients in either table, P101 to P115, "
        f"but {'; '.join(problems)}. Subtracting two Series keeps every label from both, with NaN where one side "
        "has no reading; save that whole result."
    )


def check_gap_values(root: Path) -> None:
    table = read_table(root, GAP_FILE)
    key, value = _gap(table)
    worlds = [gap_expected(world) for world in _worlds(root)]
    if key is None and value is not None:
        # index=False loses the patient IDs, which the one-row-per-patient check charges. The rows are still the
        # gaps in the order Series alignment writes them, sorted by patient, so the values can be graded.
        candidates = [[(p, row[value]) for p, row in zip(sorted(expected), table.rows)]
                      for expected in worlds if len(table.rows) == len(expected)]
        _assert(
            bool(candidates),
            f"{GAP_FILE} has {len(table.rows)} rows and no patient IDs, so its gaps cannot be matched to patients; "
            f"save the whole white_coat_gap Series, one row for each of the {len(worlds[0])} patients, with its index.",
        )
        worlds = [expected for expected in worlds if len(table.rows) == len(expected)]
    else:
        _assert(key is not None and value is not None,
                f"{GAP_FILE} needs a patient_id column and one column of gaps (its columns are {_columns(table)}); "
                "save the white_coat_gap Series with its index.")
        known = {p.casefold(): p for p in gap_expected()}
        if any(row[key].casefold() not in known and _number(row[key]) is not None for row in table.rows):
            return  # Row numbers beside the patient IDs are charged once, by the one-row-per-patient check.
        rows = [(known[row[key].casefold()], row[value]) for row in table.rows if row[key].casefold() in known]
        _assert(bool(rows), f"{GAP_FILE} names no patient, so its gaps cannot be compared.")
        candidates = [rows] * len(worlds)
    for expected, rows in zip(worlds, candidates):
        if all(_matches(cell, expected[p]) if p in expected else _blank(cell) for p, cell in rows):
            return
    truth = worlds[0]
    rows = candidates[0]
    matched = [(p, cell) for p, cell in rows if truth.get(p) is not None]
    unmatched = [(p, cell) for p, cell in rows if truth.get(p) is None and not _blank(cell)]
    if matched and all(_same(cell, -truth[p]) for p, cell in matched):
        cause = ("Every gap has the wrong sign: Task 5 subtracts the home reading from the clinic one, "
                 'bp["sbp_week8"] - home["home_sbp_week8"].')
    elif unmatched and all(_matches(cell, truth[p]) for p, cell in matched):
        if all(_number(cell) is not None for _, cell in unmatched):
            cause = ("A patient with a reading on only one side has a number instead of a blank: fill_value=0 "
                     "subtracts 0 for the missing side. Use the plain - so those gaps stay NaN.")
        else:
            cause = ("A patient with a reading on only one side must have a blank gap, which to_csv() writes for "
                     f"NaN; remove the replacement text such as {unmatched[0][1]}.")
    else:
        cause = ('Task 5 computes bp["sbp_week8"] - home["home_sbp_week8"], which pairs readings by patient_id; '
                 "both tables need patient_id as their index.")
    wrong = [f"{p} has {cell or 'blank'}, expected {_show(truth[p])}" for p, cell in rows
             if p in truth and not _matches(cell, truth[p])]
    raise AssertionError(f"{GAP_FILE}: {_first(wrong, 3)}. {cause}")


LOADED_HINT = f"In Task 2.2, read with {LOAD_CALL}, then save with bp.to_csv(LOADED_PATH), keeping the index."
FOLLOWUP_HINT = "Task 4.1 keeps bp's patient_id index; save with followup.to_csv(FOLLOWUP_PATH), keeping the index."
CHECKS = (
    Check("bp loaded: patient_id column", _id_check(LOADED_FILE, LOADED_HINT, _loaded)),
    Check("bp loaded: units row skipped", check_loaded_units),
    Check("bp loaded: coordinator_note left out", check_loaded_note),
    Check("bp loaded: all 14 patients once", check_loaded_patients),
    Check("bp loaded: -999 read as missing", check_loaded_sentinel),
    Check("bp loaded: clinic, age, and readings", check_loaded_values),
    Check("visit summary: one row per visit", check_summary_visits),
    *(Check(f"visit summary: {stat}", _summary_stat(stat)) for stat in STATS),
    Check("clinic counts: patients per clinic", check_counts_values),
    Check("clinic counts: most common first", check_counts_order),
    Check("follow-up list: patient_id column", _id_check(FOLLOWUP_FILE, FOLLOWUP_HINT, lambda root: read_table(root, FOLLOWUP_FILE))),
    Check("follow-up list: program patients", check_followup_rows),
    Check("follow-up list: derived column names", check_followup_names),
    Check("follow-up list: age dropped", check_followup_age),
    Check("follow-up list: sbp_mean values", _derived_values("sbp_mean")),
    Check("follow-up list: change_week8 values", _derived_values("change_week8")),
    Check("follow-up list: improvement_rank values", check_followup_rank),
    Check("follow-up list: largest drop first", check_followup_sorted),
    Check("follow-up list: ties in patient_id order", check_followup_ties),
    Check("follow-up Parquet: Parquet file", check_parquet_file),
    Check("follow-up Parquet: same columns", check_parquet_columns),
    Check("white-coat gap: one row per patient in either table", check_gap_rows),
    Check("white-coat gap: gaps matched by patient", check_gap_values),
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
