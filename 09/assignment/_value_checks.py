"""Checks for Assignment 09.

The course keeps these checks in 09/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They read the six CSV files a
submission saves in output/ and compare them with values computed from the
supplied vitals and lab log, copied below. The self-test confirms those copies
match 09/assignment/data/vitals.csv and 09/assignment/data/labs.csv.

Each check scores one thing, so one mistake costs only its own points, and no
check waits for another to pass. Values are compared after parsing: spacing,
line endings, quoting, a byte-order mark, column order, row order, a leading
row-number column, number formatting (2 == 2.0 == 2.00), the spelling of
True and False, and the letter case of labels and headers never cost points,
and cells may be separated by commas, semicolons, or tabs.
Timestamps are compared as instants, so any unambiguous way of writing one
passes: `2026-01-20 18:00:00+00:00`, `2026-01-20T18:00:00Z`,
`2026-01-20 18:00 UTC`, or the same instant in New York time,
`2026-01-20 13:00:00-05:00`. A timestamp with no offset is read as UTC.

Only Task 1.1 grades the conversion of `recorded_at` to UTC; Task 3.2 grades
its own lab times. A later table whose `recorded_at` times are all off by the
same number of hours, as when Task 1.1 skipped the conversion, is matched row
by row after that shift, so the earlier mistake is not charged again.

Nothing here imports, runs, or inspects student code.
"""

from __future__ import annotations

import codecs
import csv
import difflib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


UTC = timezone.utc
# The unit's clocks are in New York, which keeps Eastern Standard Time (UTC-5) all January.
NEW_YORK_IN_JANUARY = timezone(timedelta(hours=-5))
CLOCK_FORMAT = "%Y-%m-%d %H:%M"

# data/vitals.csv: one charted heart rate (beats per minute) per row, on the
# unit's New York clock; None where a row was started but no value was charted.
VITALS_COLUMNS = ("patient_id", "recorded_at", "heart_rate")
VITALS = (
    ("P02", "2026-01-20 12:00", 76),
    ("P01", "2026-01-20 07:00", 84),
    ("P02", "2026-01-20 08:00", 72),
    ("P01", "2026-01-20 13:00", 104),
    ("P02", "2026-01-20 09:00", 74),
    ("P01", "2026-01-20 10:00", None),
    ("P02", "2026-01-20 15:00", 75),
    ("P01", "2026-01-20 08:00", 88),
    ("P02", "2026-01-20 11:00", 70),
    ("P01", "2026-01-20 11:00", 96),
    ("P02", "2026-01-20 13:00", 73),
    ("P01", "2026-01-20 14:00", 110),
)
# data/labs.csv: one lab order per row, collected and resulted on the same New York clock.
LABS_COLUMNS = ("patient_id", "test", "collected_at", "resulted_at")
LABS = (
    ("P01", "lactate", "2026-01-20 11:05", "2026-01-20 11:50"),
    ("P01", "procalcitonin", "2026-01-20 10:40", "2026-01-21 09:30"),
    ("P01", "creatinine", "2026-01-20 12:20", "2026-01-20 13:25"),
    ("P02", "potassium", "2026-01-20 09:15", "2026-01-20 10:05"),
    ("P02", "troponin", "2026-01-20 12:30", "2026-01-20 13:00"),
    ("P02", "hemoglobin", "2026-01-20 13:10", "2026-01-20 13:40"),
)
# The early-warning score runs at 13:00 in New York; it is also the holdout cutoff.
PREDICTION_TIME = datetime(2026, 1, 20, 18, 0, tzinfo=UTC)

# Means may be saved with every digit or rounded to one decimal.
TOLERANCE = 0.06
# Ways pandas and people write an empty cell.
MISSING_SPELLINGS = frozenset({"", "nan", "na", "n/a", "none", "null", "<na>", "nat"})
TRUE_SPELLINGS = frozenset({"true", "t", "yes", "y", "1", "1.0"})
FALSE_SPELLINGS = frozenset({"false", "f", "no", "n", "0", "0.0"})
# Zone words a timestamp may end with; EST and EDT are what strftime("%Z") writes on the New York clock.
ZONE_WORDS = {"utc": timedelta(0), "gmt": timedelta(0), "est": timedelta(hours=-5), "edt": timedelta(hours=-4)}
# Other ways to write a date and a time that ISO 8601 does not cover. Month first is
# tried before day first, so day first only matches where the day is over 12.
DATE_PATTERNS = ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d %Y", "%B %d %Y", "%d %b %Y", "%d %B %Y")
TIME_PATTERNS = ("%H:%M", "%H:%M:%S", "%H:%M:%S.%f", "%I:%M %p", "%I:%M:%S %p")


def to_utc(clock: str) -> datetime:
    """A New York clock reading from the supplied files as a UTC instant."""
    return datetime.strptime(clock, CLOCK_FORMAT).replace(tzinfo=NEW_YORK_IN_JANUARY).astimezone(UTC)


def _mean(values: list) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _readings() -> dict[str, list[tuple[datetime, float | None]]]:
    """Each patient's readings in time order: [(UTC instant, heart rate or None)]."""
    by_patient: dict[str, list[tuple[datetime, float | None]]] = {}
    for patient, clock, heart_rate in VITALS:
        by_patient.setdefault(patient, []).append((to_utc(clock), None if heart_rate is None else float(heart_rate)))
    return {patient: sorted(by_patient[patient]) for patient in sorted(by_patient)}


READINGS = _readings()
HOUR = timedelta(hours=1)


def _prepared_rows() -> dict[tuple, dict[str, object]]:
    return {
        (patient, instant): {"heart_rate": heart_rate, "source_row": 1}
        for patient, readings in READINGS.items() for instant, heart_rate in readings
    }


def _grid_rows() -> dict[tuple, dict[str, object]]:
    """Each patient's hourly grid, from their first reading's hour to their last."""
    rows = {}
    for patient, readings in READINGS.items():
        charted = dict(readings)
        hour, last = readings[0][0], readings[-1][0]
        while hour <= last:
            found = hour in charted
            rows[(patient, hour)] = {
                "heart_rate": charted.get(hour),
                "source_row": 1 if found else None,
                "grid_created": not found,
                "value_missing": found and charted[hour] is None,
            }
            hour += HOUR
    return rows


def _bin_start(instant: datetime) -> datetime:
    return instant.replace(hour=instant.hour - instant.hour % 2, minute=0, second=0, microsecond=0)


def _two_hour_rows() -> dict[tuple, dict[str, object]]:
    """Left-closed, left-labeled two-hour bins per patient, as resample("2h") draws them."""
    rows = {}
    for patient, readings in READINGS.items():
        start, last = _bin_start(readings[0][0]), _bin_start(readings[-1][0])
        while start <= last:
            inside = [heart_rate for instant, heart_rate in readings if _bin_start(instant) == start]
            rows[(patient, start)] = {"mean_hr": _mean(inside), "n_rows": len(inside)}
            start += 2 * HOUR
    return rows


def _feature_rows() -> dict[tuple, dict[str, object]]:
    """Same-patient lag, change, and two past-only means for every reading."""
    rows = {}
    for patient, readings in READINGS.items():
        for position, (instant, heart_rate) in enumerate(readings):
            previous = readings[position - 1][1] if position else None
            rows[(patient, instant)] = {
                "heart_rate": heart_rate,
                "previous_hr": previous,
                "hr_change": None if previous is None or heart_rate is None else heart_rate - previous,
                "mean_prev_2": _mean([value for _, value in readings[max(0, position - 2):position]]),
                "mean_prev_2h": _mean(
                    [value for earlier, value in readings if instant - 2 * HOUR <= earlier < instant]),
            }
    return rows


def _lab_rows() -> dict[tuple, dict[str, object]]:
    rows = {}
    for patient, test, collected, resulted in LABS:
        rows[(patient, test)] = {
            "collected_at": to_utc(collected),
            "resulted_at": to_utc(resulted),
            "available": to_utc(resulted) <= PREDICTION_TIME,
        }
    return rows


def _block_rows() -> dict[tuple, dict[str, object]]:
    return {
        (patient, instant): {"block": "earlier" if instant < PREDICTION_TIME else "later_holdout"}
        for patient, readings in READINGS.items() for instant, _ in readings
    }


@dataclass(frozen=True)
class Artifact:
    """One saved CSV: where it lives, which task writes it, and what it should hold.

    `columns` is the header line the notebook writes. `rows` maps each key, a
    tuple of the key columns' values, to the expected value of the other
    checked columns; None means an empty cell. `optional` columns may be saved
    or left out and are never compared. `time_key` names the key column that
    holds a timestamp, if one does.
    """

    path: str
    task: str
    columns: tuple[str, ...]
    key: tuple[str, ...]
    rows: dict[tuple, dict[str, object]]
    optional: tuple[str, ...] = ()
    time_key: str | None = None

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(column for column in self.columns if column not in self.optional)

    @property
    def allowed(self) -> tuple[str, ...]:
        return (*self.columns, *(column for column in self.optional if column not in self.columns))


@dataclass(frozen=True)
class Table:
    """A saved CSV: its casefolded column names and one {column: cell} dict per data row."""

    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class Match:
    """Saved rows grouped by the expected key they hold, and a description of each row that holds none."""

    grouped: dict[tuple, list[dict[str, str]]]
    unknown: list[str]


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


PREPARED = Artifact(
    path="output/prepared_vitals.csv",
    task="Task 1.1",
    columns=("patient_id", "recorded_at", "heart_rate", "source_row"),
    key=("patient_id", "recorded_at"),
    rows=_prepared_rows(),
    time_key="recorded_at",
)

HOURLY_GRID = Artifact(
    path="output/hourly_grid.csv",
    task="Task 2.1",
    columns=("patient_id", "recorded_at", "heart_rate", "source_row", "grid_created", "value_missing"),
    key=("patient_id", "recorded_at"),
    rows=_grid_rows(),
    optional=("source_row",),
    time_key="recorded_at",
)

TWO_HOUR = Artifact(
    path="output/two_hour_summary.csv",
    task="Task 2.2",
    columns=("patient_id", "recorded_at", "mean_hr", "n_rows"),
    key=("patient_id", "recorded_at"),
    rows=_two_hour_rows(),
    time_key="recorded_at",
)

FEATURES = Artifact(
    path="output/past_features.csv",
    task="Task 3.1",
    columns=("patient_id", "recorded_at", "heart_rate", "previous_hr", "hr_change", "mean_prev_2", "mean_prev_2h"),
    key=("patient_id", "recorded_at"),
    rows=_feature_rows(),
    optional=("source_row",),
    time_key="recorded_at",
)

LAB_AVAILABILITY = Artifact(
    path="output/lab_availability.csv",
    task="Task 3.2",
    columns=("patient_id", "test", "collected_at", "resulted_at", "available"),
    key=("patient_id", "test"),
    rows=_lab_rows(),
)

BLOCKS = Artifact(
    path="output/chronological_blocks.csv",
    task="Task 3.3",
    columns=("patient_id", "recorded_at", "heart_rate", "source_row", "block"),
    key=("patient_id", "recorded_at"),
    rows=_block_rows(),
    optional=("source_row",),
    time_key="recorded_at",
)

ARTIFACTS = (PREPARED, HOURLY_GRID, TWO_HOUR, FEATURES, LAB_AVAILABILITY, BLOCKS)


def _assert(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _join(items) -> str:
    items = [str(item) for item in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _listed(problems: list[str], limit: int = 4) -> str:
    shown = "; ".join(problems[:limit])
    if len(problems) > limit:
        shown += f"; and {len(problems) - limit} more like {'it' if len(problems) - limit == 1 else 'them'}"
    return shown


def _clean(cell: str) -> str:
    return " ".join(cell.split())


def _fold(column: str) -> str:
    return _clean(column).casefold()


def _label(cell: str) -> str:
    """A label compared in any letter case, with spaces or hyphens for underscores."""
    return re.sub(r"[\s_-]+", "_", _clean(cell).casefold())


def _number(cell: str) -> float | None:
    try:
        return float(_clean(cell).replace(",", ""))
    except ValueError:
        return None


def _is_whole_number(cell: str) -> bool:
    number = _number(cell)
    return number is not None and number.is_integer()


def _boolean(cell: str) -> bool | None:
    text = _clean(cell).casefold()
    if text in TRUE_SPELLINGS:
        return True
    if text in FALSE_SPELLINGS:
        return False
    return None


def instant(cell: str) -> datetime | None:
    """The UTC instant a timestamp cell names, or None when it names none.

    Accepts ISO 8601 in its common spellings (a space or `T`, an offset, `Z`),
    a trailing zone word (`UTC`, `GMT`, or New York's `EST` and `EDT`), dates
    written with slashes or a month name, and 12-hour clock times. A timestamp
    with no offset and no zone word is read as UTC.
    """
    text = _clean(cell)
    if not text:
        return None
    zone = UTC
    word = re.search(r"\s*\(?(?<![a-z])(utc|gmt|est|edt)\)?$", text, flags=re.I)
    if word:
        zone = timezone(ZONE_WORDS[word.group(1).casefold()])
        text = text[:word.start()]
    # "UTC+00:00" names the zone twice; the offset is enough.
    text = re.sub(r"\s*(?<![a-z])(utc|gmt)(?=[+-]\d)", "", text, flags=re.I)
    text = re.sub(r"\s+([+-]\d\d:?\d\d)$", r"\1", text)
    moment = _parse_moment(text)
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=zone)
    return moment.astimezone(UTC)


def _parse_moment(text: str) -> datetime | None:
    """A datetime from ISO 8601 text or one of DATE_PATTERNS and TIME_PATTERNS, with its offset if it has one."""
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    offset = None
    written = re.fullmatch(r"(.*(?::\d\d(?:\.\d+)?|[ap]m))([+-])(\d\d):?(\d\d)?", text, flags=re.I)
    if written:
        text = written.group(1)
        hours, minutes = int(written.group(3)), int(written.group(4) or 0)
        offset = timezone((1 if written.group(2) == "+" else -1) * timedelta(hours=hours, minutes=minutes))
    text = " ".join(re.sub(r"(?<=\d)T(?=\d)", " ", text).replace(",", " ").split())
    for date in DATE_PATTERNS:
        for time in TIME_PATTERNS:
            try:
                moment = datetime.strptime(text, f"{date} {time}")
            except ValueError:
                continue
            return moment if offset is None else moment.replace(tzinfo=offset)
    return None


def _matches(cell: str, expected: object) -> bool:
    text = _clean(cell)
    if expected is None:
        return text.casefold() in MISSING_SPELLINGS
    if isinstance(expected, bool):
        return _boolean(text) is expected
    if isinstance(expected, (int, float)):
        number = _number(text)
        return number is not None and abs(number - expected) <= TOLERANCE
    if isinstance(expected, datetime):
        return instant(text) == expected
    return _label(text) == _label(str(expected))


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S+00:00")


def _show(value: object) -> str:
    if value is None:
        return "blank"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{round(value, 2):g}"
    if isinstance(value, datetime):
        return _stamp(value)
    return str(value)


def _given(cell: str) -> str:
    text = _clean(cell)
    return text if text else "blank"


def _key_name(key: tuple) -> str:
    """A row key in words: `P01 at 15:00 UTC` or `P01 lactate`. Every reading falls on 2026-01-20."""
    return " ".join(
        part.astimezone(UTC).strftime("at %H:%M UTC") if isinstance(part, datetime) else str(part) for part in key
    )


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

    Commas, semicolons, and tabs are all accepted: whichever splits the header
    line into the most cells is used, and a tie keeps commas. A file separated
    by semicolons may write decimal commas, so there 12,5 reads as 12.5.
    """
    lines = text.splitlines()
    header = next((line for line in lines if line.strip()), "")
    delimiter = max(DELIMITERS, key=lambda mark: len(next(csv.reader([header], delimiter=mark), [])))
    rows = [row for row in csv.reader(lines, delimiter=delimiter) if any(cell.strip() for cell in row)]
    if delimiter == ";":
        rows = [[DECIMAL_COMMA.sub(r"\1.\2", cell) for cell in row] for row in rows]
    return rows


def _artifact_path(root: Path, name: str) -> Path | None:
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


def _missing(root: Path, artifact: Artifact) -> str:
    message = f"{artifact.path} is missing; run the {artifact.task} cell to write it, then commit it."
    wanted = root / artifact.path
    if wanted.parent.is_dir():
        look_alikes = sorted(
            path.relative_to(root).as_posix()
            for path in wanted.parent.iterdir()
            if path.is_file()
            and difflib.SequenceMatcher(None, wanted.name.casefold(), path.name.casefold()).ratio() >= 0.75
        )
        if look_alikes:
            message += f" Found {_join(look_alikes)}; save it as {artifact.path} instead."
    return message


def _is_index_header(cell: str) -> bool:
    return cell in ("", "index") or cell.startswith("unnamed")


def read_table(root: Path, artifact: Artifact) -> Table:
    """Parse a saved CSV, however it is separated, spaced, quoted, or ended.

    A leading column headed by nothing, `Unnamed: 0`, or `index` that holds only
    whole numbers is the row numbers pandas writes when `index=False` is left
    out, so it is set aside rather than counted as a column. So is an empty
    column with no header, which a trailing comma on every line makes.
    """
    path = _artifact_path(root, artifact.path)
    _assert(path is not None, _missing(root, artifact))
    lines = _csv_rows(_decode(path.read_bytes()))
    _assert(lines, f"{artifact.path} is empty; run the {artifact.task} cell again to write it.")
    header = [_fold(cell) for cell in lines[0]]
    body = [[_clean(cell) for cell in row] for row in lines[1:]]
    width = max(len(header), *(len(row) for row in body)) if body else len(header)
    header += [""] * (width - len(header))
    body = [row + [""] * (width - len(row)) for row in body]

    if len(header) > 1 and _is_index_header(header[0]) and body and all(_is_whole_number(row[0]) for row in body):
        header = header[1:]
        body = [row[1:] for row in body]
    keep = [i for i, column in enumerate(header) if column or any(row[i] for row in body)]
    header = [header[i] for i in keep]
    body = [[row[i] for i in keep] for row in body]

    rows = tuple({column: row[i] for i, column in enumerate(header)} for row in body)
    return Table(columns=tuple(header), rows=rows)


def _resolve(table: Table, artifact: Artifact) -> dict[str, str]:
    """Map each expected column to the saved column that holds it.

    A column saved under its own name, in any letter case, is found by name. A
    key column saved under another name, such as a blank header over saved
    index labels, is found by its values. If exactly one expected value column
    is still unplaced and exactly one unexpected column is left, that column is
    taken to hold it. A misnamed column then costs only the columns check.
    """
    known = artifact.allowed
    found = {column: _fold(column) for column in known if _fold(column) in table.columns}
    spare = [column for column in table.columns if column not in {_fold(name) for name in known}]
    for position, column in enumerate(artifact.key):
        if column in found:
            continue
        if column == artifact.time_key:
            fits = lambda value: instant(value) is not None  # noqa: E731
        else:
            wanted = {_label(str(key[position])) for key in artifact.rows}
            fits = lambda value: _label(value) in wanted  # noqa: E731
        candidates = []
        for other in spare:
            values = [row[other] for row in table.rows if row[other]]
            if values and sum(bool(fits(value)) for value in values) * 2 > len(values):
                candidates.append(other)
        if len(candidates) == 1:
            found[column] = candidates[0]
            spare.remove(candidates[0])
    # A key column is only ever found by name or by its values, never by elimination.
    unplaced = [column for column in artifact.required if column not in found and column not in artifact.key]
    if len(unplaced) == 1 and len(spare) == 1:
        found[unplaced[0]] = spare[0]
    return found


def _saved_key(row: dict[str, str], artifact: Artifact, found: dict[str, str]) -> tuple:
    return tuple(
        instant(row[found[column]]) if column == artifact.time_key else _label(row[found[column]])
        for column in artifact.key
    )


def _match(table: Table, artifact: Artifact, found: dict[str, str]) -> Match:
    """Group the saved rows by the expected key each one holds.

    When every key column is saved, each row is matched by its key. Times are
    compared as instants, and when every saved time sits the same number of
    hours from the expected one, the rows are matched after that shift, so a
    table built from unconverted clock times is judged on its own values. When
    a key column is not saved, as when the keys stayed in the index and were
    dropped by `index=False`, a table with exactly the expected number of rows
    is matched in the notebook's sorted order.
    """
    expected = list(artifact.rows)
    if not all(column in found for column in artifact.key):
        absent = [column for column in artifact.key if column not in found]
        _assert(
            len(table.rows) == len(expected),
            f"{artifact.path} has no {_join(absent)} column (its columns are {_join(table.columns) or 'none'}), "
            f"so its rows cannot be matched; {artifact.task} saves the header line {','.join(artifact.columns)}.",
        )
        return Match({key: [row] for key, row in zip(expected, table.rows)}, [])

    saved = [(_saved_key(row, artifact, found), row) for row in table.rows]
    lookup = {tuple(_label(str(part)) if not isinstance(part, datetime) else part for part in key): key
              for key in expected}
    time_position = artifact.key.index(artifact.time_key) if artifact.time_key else None

    def placed(shift: timedelta) -> dict[tuple, tuple]:
        """{saved key: expected key} for every saved key that names an expected row after `shift`."""
        result = {}
        for key, _ in saved:
            if time_position is not None:
                if key[time_position] is None:
                    continue
                key = (*key[:time_position], key[time_position] + shift, *key[time_position + 1:])
            if key in lookup:
                result[key] = lookup[key]
        return result

    shift = timedelta(0)
    if time_position is not None:
        shifts = {timedelta(0)}
        for key, _ in saved:
            if key[time_position] is None:
                continue
            for other in expected:
                if _label(str(other[0])) == key[0]:
                    shifts.add(other[time_position] - key[time_position])
        best = len(set(placed(shift).values()))
        for candidate in sorted(shifts, key=lambda value: (abs(value), value)):
            count = len(set(placed(candidate).values()))
            if count > best:
                shift, best = candidate, count

    grouped: dict[tuple, list[dict[str, str]]] = {}
    unknown = []
    for key, row in saved:
        moved = key
        if time_position is not None and key[time_position] is not None:
            moved = (*key[:time_position], key[time_position] + shift, *key[time_position + 1:])
        target = lookup.get(moved)
        if target is None:
            shown = " ".join(row[found[column]] or "blank" for column in artifact.key)
            unknown.append(f"a row for {shown}")
        else:
            grouped.setdefault(target, []).append(row)
    return Match(grouped, unknown)


def columns_check(artifact: Artifact, hint: str) -> Callable[[Path], None]:
    """The saved header holds exactly the expected columns, in any order."""

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact)
        # A saved index with no name leaves its column unheaded; holding the keys, it is the key column.
        first = table.columns[0] if table.columns else None
        unheaded = {
            column: found[column]
            for column in artifact.key
            if _fold(column) not in table.columns and first is not None and found.get(column) == first
            and _is_index_header(first)
        }
        allowed = {_fold(column) for column in artifact.allowed}
        missing = [
            column for column in artifact.required
            if _fold(column) not in table.columns and column not in unheaded
        ]
        extra = [
            column or "a column with no header"
            for column in table.columns
            if column not in allowed and column not in unheaded.values()
        ]
        problems = []
        if missing:
            problems.append(f"is missing {_join(missing)}")
        if extra:
            problems.append(f"also has {_join(extra)}")
        _assert(
            not problems,
            f"{artifact.path} " + " and ".join(problems) + f"; {artifact.task} saves the header line "
            f"{','.join(artifact.columns)} (any column order). {hint}",
        )

    return check


def rows_check(artifact: Artifact, description: str, hint: str) -> Callable[[Path], None]:
    """Each expected key appears on exactly one row, and no unexpected key appears."""

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        match = _match(table, artifact, _resolve(table, artifact))
        missing = [key for key in artifact.rows if key not in match.grouped]
        repeated = [key for key, rows in match.grouped.items() if len(rows) > 1]
        problems = []
        if missing:
            problems.append(f"is missing {_listed([_key_name(key) for key in missing], 6)}")
        if repeated:
            problems.append("lists " + _join(f"{_key_name(key)} {len(match.grouped[key])} times" for key in repeated))
        if match.unknown:
            shown = _join(match.unknown[:4]) + (f" and {len(match.unknown) - 4} more" if len(match.unknown) > 4 else "")
            problems.append(f"also has {shown}")
        _assert(
            not problems,
            f"{artifact.path} should hold {len(artifact.rows)} rows, {description}, but it "
            + "; ".join(problems) + f". Fix it in {artifact.task}: {hint}",
        )

    return check


def values_check(artifact: Artifact, columns: tuple[str, ...], hint: str) -> Callable[[Path], None]:
    """Every saved row for an expected key holds the expected values in `columns`.

    A missing or repeated row costs the rows check, not this one: this check
    compares whatever rows the file does hold for the expected keys.
    """

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact)
        absent = [column for column in columns if column not in found]
        _assert(
            not absent,
            f"{artifact.path} has no {_join(absent)} column (its columns are {_join(table.columns) or 'none'}), "
            f"so those values cannot be compared. Fix it in {artifact.task}: {hint}",
        )
        match = _match(table, artifact, found)
        _assert(
            match.grouped,
            f"{artifact.path} has no row for any expected {' and '.join(artifact.key)}, so its "
            f"{_join(columns)} values cannot be compared; the rows check says what to fix in {artifact.task}.",
        )
        wrong = []
        for key, rows in match.grouped.items():
            for column in columns:
                expected = artifact.rows[key][column]
                given = sorted({_given(row[found[column]]) for row in rows if not _matches(row[found[column]], expected)})
                if given:
                    wrong.append(f"{_key_name(key)} has {column} {_join(given)}, expected {_show(expected)}")
        _assert(not wrong, f"{artifact.path}: " + _listed(wrong) + f". Fix it in {artifact.task}: {hint}")

    return check


def utc_check(artifact: Artifact, hint: str) -> Callable[[Path], None]:
    """Every saved time parses, and together they name the readings' UTC instants, not their New York clock times.

    Compares the times alone, whatever patient they sit beside, so a missing or
    extra row costs the rows check and not this one.
    """

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact)
        column = artifact.time_key
        _assert(
            column in found,
            f"{artifact.path} has no {column} column (its columns are {_join(table.columns) or 'none'}), so its "
            f"times cannot be checked. Fix it in {artifact.task}: {hint}",
        )
        unreadable = sorted({row[found[column]] or "blank" for row in table.rows if instant(row[found[column]]) is None})
        _assert(
            not unreadable,
            f"{artifact.path} has {column} values that are not timestamps: {_join(unreadable[:4])}. "
            f"Fix it in {artifact.task}: {hint}",
        )
        saved = {instant(row[found[column]]) for row in table.rows}
        expected = {key[artifact.key.index(column)] for key in artifact.rows}
        shifts = sorted({wanted - moment for moment in saved for wanted in expected}, key=lambda value: (abs(value), value))
        best = max(shifts, key=lambda shift: len({moment + shift for moment in saved} & expected), default=timedelta(0))
        _assert(
            best or saved & expected,
            f"{artifact.path} has no {column} that names a supplied reading, in UTC or otherwise. "
            f"Fix it in {artifact.task}: {hint}",
        )
        if len({moment + best for moment in saved} & expected) > len(saved & expected):
            earliest = min((row[found[column]] for row in table.rows), key=instant)
            hours = best / timedelta(hours=1)
            direction = "behind" if hours > 0 else "ahead of"
            _assert(
                False,
                f"{artifact.path} has each {column} {abs(hours):g} hour{'' if abs(hours) == 1 else 's'} {direction} "
                f"its UTC instant: the earliest "
                f"reads {earliest}, expected {_stamp(instant(earliest) + best)}. Fix it in {artifact.task}: {hint}",
            )

    return check


UTC_HINT = (
    'parse the clock text with pd.to_datetime(..., format="%Y-%m-%d %H:%M"), attach the zone it was charted in '
    'with .dt.tz_localize("America/New_York"), then convert with .dt.tz_convert("UTC").'
)

CHECKS = (
    # Task 1.1: output/prepared_vitals.csv
    Check(
        "prepared vitals: columns",
        columns_check(PREPARED, 'Keep the three supplied columns and add source_row, a column of 1s.'),
    ),
    Check(
        "prepared vitals: one row per reading",
        rows_check(PREPARED, "one per charted reading", "keep every row of data/vitals.csv, including the one with "
                   "no heart rate, and do not add or drop rows."),
    ),
    Check("prepared vitals: recorded_at in UTC", utc_check(PREPARED, UTC_HINT)),
    Check(
        "prepared vitals: heart_rate values",
        values_check(PREPARED, ("heart_rate",), "keep each reading's heart_rate from data/vitals.csv unchanged, "
                     "and leave P01's missing value empty rather than filling it."),
    ),
    Check(
        "prepared vitals: source_row values",
        values_check(PREPARED, ("source_row",), 'source_row is 1 on every row: vitals["source_row"] = 1.'),
    ),
    # Task 2.1: output/hourly_grid.csv
    Check(
        "hourly grid: columns",
        columns_check(HOURLY_GRID, "Keep heart_rate and source_row through .asfreq(), reset_index(), and add the "
                      "two flag columns."),
    ),
    Check(
        "hourly grid: one row per patient-hour",
        rows_check(HOURLY_GRID, "one per hour from each patient's first reading to their last",
                   'start from the prepared vitals of Task 1.1, group by patient_id so each patient gets their own '
                   'grid, then .resample("h").asfreq().'),
    ),
    Check(
        "hourly grid: heart_rate values",
        values_check(HOURLY_GRID, ("heart_rate",), "a charted hour keeps its heart_rate; an hour the grid created "
                     "stays empty rather than filled."),
    ),
    Check(
        "hourly grid: grid_created flags",
        values_check(HOURLY_GRID, ("grid_created",), 'grid_created is grid["source_row"].isna(): True only on '
                     "hours with no charted row."),
    ),
    Check(
        "hourly grid: value_missing flags",
        values_check(HOURLY_GRID, ("value_missing",), 'value_missing is grid["source_row"].notna() & '
                     'grid["heart_rate"].isna(): True only on a charted row with no heart rate.'),
    ),
    # Task 2.2: output/two_hour_summary.csv
    Check(
        "two-hour summary: columns",
        columns_check(TWO_HOUR, 'Name both summaries in .agg(): mean_hr=("heart_rate", "mean") and '
                      'n_rows=("source_row", "count"), then reset_index().'),
    ),
    Check(
        "two-hour summary: one row per patient and bin",
        rows_check(TWO_HOUR, "one per two-hour bin from each patient's first bin to their last",
                   'start from the prepared vitals of Task 1.1, group by patient_id, then .resample("2h"), whose '
                   "bins start on even UTC hours."),
    ),
    Check(
        "two-hour summary: mean_hr values",
        values_check(TWO_HOUR, ("mean_hr",), 'mean_hr is ("heart_rate", "mean") within each patient\'s bin; '
                     "a bin whose only row has no heart rate stays empty."),
    ),
    Check(
        "two-hour summary: n_rows values",
        values_check(TWO_HOUR, ("n_rows",), 'n_rows is ("source_row", "count"), which counts the charted row with '
                     "no heart rate too; counting heart_rate skips it."),
    ),
    # Task 3.1: output/past_features.csv
    Check(
        "past features: columns",
        columns_check(FEATURES, 'Start from vitals[["patient_id", "recorded_at", "heart_rate"]].copy() and add the '
                      "four feature columns."),
    ),
    Check(
        "past features: one row per reading",
        rows_check(FEATURES, "one per charted reading", 'merge the two-hour means back with on=["patient_id", '
                   '"recorded_at"] and validate="one_to_one", which keeps one row per reading.'),
    ),
    Check(
        "past features: previous_hr values",
        values_check(FEATURES, ("previous_hr",), 'previous_hr is groupby("patient_id")["heart_rate"].shift(1): '
                     "the same patient's previous row, empty on each patient's first reading."),
    ),
    Check(
        "past features: hr_change values",
        values_check(FEATURES, ("hr_change",), 'hr_change is groupby("patient_id")["heart_rate"].diff(): '
                     "current minus previous, within one patient."),
    ),
    Check(
        "past features: mean_prev_2 values",
        values_check(FEATURES, ("mean_prev_2",), "mean_prev_2 is the grouped transform of "
                     "lambda s: s.shift(1).rolling(2, min_periods=1).mean(), which leaves out the current reading."),
    ),
    Check(
        "past features: mean_prev_2h values",
        values_check(FEATURES, ("mean_prev_2h",), 'mean_prev_2h is the grouped .rolling("2h", closed="left").mean() '
                     "on recorded_at: readings in the two hours before this one, not counting it."),
    ),
    # Task 3.2: output/lab_availability.csv
    Check(
        "lab availability: columns",
        columns_check(LAB_AVAILABILITY, "Keep the four supplied columns and add available."),
    ),
    Check(
        "lab availability: one row per lab",
        rows_check(LAB_AVAILABILITY, "one per lab order in data/labs.csv", "keep every lab order; "
                   "availability is a new column, not a filter."),
    ),
    Check(
        "lab availability: collected_at and resulted_at in UTC",
        values_check(LAB_AVAILABILITY, ("collected_at", "resulted_at"), "convert both columns the way Task 1.1 "
                     'converts recorded_at: tz_localize("America/New_York"), then tz_convert("UTC").'),
    ),
    Check(
        "lab availability: available flags",
        values_check(LAB_AVAILABILITY, ("available",), 'available is labs["resulted_at"] <= prediction_time, with '
                     "prediction_time 2026-01-20 18:00:00+00:00: a result reported exactly then counts."),
    ),
    # Task 3.3: output/chronological_blocks.csv
    Check(
        "chronological blocks: columns",
        columns_check(BLOCKS, "Start from a .copy() of the prepared vitals and add block."),
    ),
    Check(
        "chronological blocks: one row per reading",
        rows_check(BLOCKS, "one per charted reading", "label every prepared reading; the holdout is a label, not "
                   "a filter."),
    ),
    Check(
        "chronological blocks: block labels",
        values_check(BLOCKS, ("block",), 'block is np.where(recorded_at < prediction_time, "earlier", '
                     '"later_holdout"): a reading at exactly 2026-01-20 18:00:00+00:00 is later_holdout.'),
    ),
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
