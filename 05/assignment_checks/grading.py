"""Course-owned grading rules for Assignment 05, the midterm.

The midterm handout ships no checks. After the deadline the course grades each
fork's committed files in `output/` with this module, through
`check_assignment.py` and `scripts/grade_submissions.py`. Student code is never
imported, executed, or read, and the notebook is left to human review.

Every expected value follows from the supplied `data/people_raw.csv` (its
SHA-256 is `SOURCE_SHA256`); `_grader_selftest/run.py` recomputes each one from
that file with pandas and confirms it matches the constants below.

Each check reads one artifact and compares parsed values rather than text, so
line endings, trailing whitespace, a missing final newline, header spacing, a
comma, semicolon, or tab between cells and spaces padding every separator,
column order, a leading unnamed index column, the `0` header of a Series saved
without a name, number formatting, the letter case and spacing of labels, a
label with a typo, any unambiguous way of writing a date, and the common
spellings of booleans and missing values never cost points. Each check scores
the lines, rows, or records it can verify on its own, so one wrong value costs
only its own share, and no check depends on another passing. A value computed
from an earlier one (a summary of the student's own age array, `needs_review`
from their own `age` and `visit_date`, `rows_before` and `rows_after` from
their own counts) is judged against that earlier value, so one mistake is
charged once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from itertools import combinations
import io
import math
from pathlib import Path
import re


SCHEMA = "datasci217/grading-result/v1"
OUTPUT = Path("output")

SOURCE = "data/people_raw.csv"
SOURCE_SHA256 = "d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b"
RAW_ROWS = 12
CLEAN_ROWS = 11

PREVIEW_COMMANDS = {
    "head": "head -n 4 data/people_raw.csv",
    "tail": "tail -n 2 data/people_raw.csv",
}
PREVIEW_LINES = {
    "head": (
        "record_id,full_name,site,status,age_text,visit_date",
        "R001, Alice Smith , North ,Active,34,2026-01-15",
        "R002,BOB JONES,north,active,unknown,2026-02-30",
        "R002,BOB JONES,north,active,unknown,2026-02-30",
    ),
    "tail": (
        "R010,Jamie Okafor,West,Complete,28,2026-07-15",
        "R011,Kai Patel,south, pending ,0,2026-08-01",
    ),
}

PIPELINE_SUMMARY = {
    "raw_rows": 12,
    "raw_columns": 6,
    "exact_duplicate_rows": 1,
    "candidate_id_duplicate_rows": 1,
    "clean_rows": 11,
}
# How each count is made, for the feedback on a wrong one.
PIPELINE_HINTS = {
    "raw_rows": "len(raw) counts data rows; the header is not a row",
    "raw_columns": "raw.shape[1] counts the columns",
    "exact_duplicate_rows": "raw.duplicated().sum() counts each repeat once, without keep=False",
    "candidate_id_duplicate_rows": 'raw.duplicated(subset=["record_id"]).sum() counts each repeated ID once, '
                                   "without keep=False",
    "clean_rows": "len(raw.drop_duplicates()) counts the rows left after removing exact repeats",
}

NUMPY_AGE_SUMMARY = {"count": 6, "min": 0, "max": 52, "sum": 198, "mean": 33.0}
# Other names a student may give a metric, such as the "size" Task 1.3's text uses.
METRIC_ALIASES = {
    "count": ("size", "n", "len", "length"),
    "min": ("minimum",),
    "max": ("maximum",),
    "sum": ("total",),
    "mean": ("average", "avg"),
}
# Every numeric age_text value in the source, one per row, and the valid ages among them. A summary that
# misses these ages is compared with every other choice of ages, so the metrics computed from the
# student's own array still earn their points and the wrong choice costs one.
AGE_CANDIDATES = (34, -9, 45, 52, 40.5, 121, 39, 28, 0)
VALID_AGES = (34, 45, 52, 39, 28, 0)
VALID_AGE_HINT = ("a valid age is a whole number from 0 through 120, so leave out the sentinels, words, "
                  "fractions such as 40.5, and ages above 120, and keep an age of 0")
MEAN_TOLERANCE = 0.05

# The raw site and status of each selected record, as written in the source.
PANDAS_SELECTION = {
    "R001": (" North ", "Active"),
    "R003": ("SOUTH", "pending"),
    "R010": ("West", "Complete"),
}
SELECTION_COLUMNS = ("record_id", "site", "status")

ISSUE_AUDIT = (
    ("schema mismatch", 0),
    ("empty full-name tokens", 1),
    ("empty date tokens", 1),
    ("age sentinel tokens", 3),
    ("status sentinel tokens", 1),
    ("age parse failures", 1),
    ("numeric but noninteger age values", 1),
    ("age values outside 0 through 120", 1),
    ("date parse failures", 3),
    ("rows in exact duplicate sets", 2),
    ("rows with repeated candidate IDs", 2),
    ("site values needing format normalization", 4),
    ("status values needing format normalization", 3),
    ("unexpected site values", 0),
    ("unexpected non-sentinel status values", 0),
)

CLEANED_COLUMNS = ("record_id", "full_name", "site", "status", "age", "visit_date", "needs_review")
# A column the cleaned table may hold under another name, used when the task's name is absent:
# age_text cleaned in place instead of copied to a new age column.
CLEANED_ALIASES = {"age": "age_text"}
# Each column's cleaning rule, for the feedback on a wrong value.
CLEANED_RULES = {
    "full_name": "strip the spaces around a name and title-case it; an empty name becomes missing",
    "site": "strip and lowercase every site",
    "status": "strip and lowercase every status; the NA sentinel becomes missing",
    "age": "keep whole numbers from 0 through 120; sentinels, words, fractions, and ages above 120 become "
           "missing, never rounded (pd.to_numeric on string text gives Float64, whose astype('Int64') silently "
           "turns 40.5 into 40, so blank the values where .mod(1).eq(0) is False before casting)",
    "visit_date": "keep only exact YYYY-MM-DD text for a date on the calendar; anything else becomes missing",
    "needs_review": "True exactly where age or visit_date is missing, otherwise False",
}
# One row per record: full_name, site, status, age, visit_date, needs_review; None is missing.
_CLEANED_ROWS = {
    "R001": ("Alice Smith", "north", "active", 34, "2026-01-15", False),
    "R002": ("Bob Jones", "north", "active", None, None, True),
    "R003": ("Carla Ruiz", "south", "pending", None, "2026-03-01", True),
    "R004": (None, "south", None, 45, None, True),
    "R005": ("Evan Li", "west", "complete", 52, "2026-02-14", False),
    "R006": ("Fatima Noor", "north", "active", None, "2026-04-01", True),
    "R007": ("Grace Chen", "south", "active", None, "2026-05-01", True),
    "R008": ("Hugo Diaz", "west", "pending", None, "2026-06-01", True),
    "R009": ("Inez Park", "north", "complete", 39, None, True),
    "R010": ("Jamie Okafor", "west", "complete", 28, "2026-07-15", False),
    "R011": ("Kai Patel", "south", "pending", 0, "2026-08-01", False),
}
# Compared by hand rather than with zip(strict=True), which Python 3.9 lacks, so a
# column added to one table and not the other still stops the run.
if any(len(values) != len(CLEANED_COLUMNS) - 1 for values in _CLEANED_ROWS.values()):
    raise ValueError("_CLEANED_ROWS and CLEANED_COLUMNS disagree on the columns; update both together.")
CLEANED = {
    record_id: dict(zip(CLEANED_COLUMNS[1:], (
        name, site, status, age, None if visit is None else date.fromisoformat(visit), review,
    )))
    for record_id, (name, site, status, age, visit, review) in _CLEANED_ROWS.items()
}

DECISION_COLUMNS = ("field", "issue", "action", "reason", "source", "source_sha256", "rows_before", "rows_after")
DECISIONS = (
    ("full_name", "empty optional name", "retain as missing"),
    ("full_name, site, status", "surrounding whitespace and case variants",
     "strip surrounding whitespace and normalize bounded field case"),
    ("status", "NA sentinel", "convert the documented sentinel to missing"),
    ("age_text", "unknown and -9 sentinels", "convert the documented sentinels to missing"),
    ("age_text", "nonnumeric, fractional, or out-of-range values",
     "coerce invalid values to missing without rounding"),
    ("visit_date", "empty, lexically invalid, or calendar-invalid values",
     "coerce invalid values to missing after an exact-format check"),
    ("all raw columns", "exact duplicate submissions", "keep the first exact raw row only"),
    ("all fields", "adjacent-row filling", "do not forward-fill or backward-fill"),
)

# Spellings of a missing cell. `NA` is left out for text columns, where it is
# the status sentinel the cleaning must convert, not a way of writing missing.
TEXT_MISSING = frozenset({"", "nan", "<na>", "nat", "none", "null"})
VALUE_MISSING = TEXT_MISSING | {"na", "n/a"}
TRUE_WORDS = frozenset({"true", "t", "yes", "y", "1", "1.0"})
FALSE_WORDS = frozenset({"false", "f", "no", "n", "0", "0.0"})
KEY_VALUE = re.compile(r"^\s*(?:[-*]\s+)?([A-Za-z][\w \t-]*?)\s*[=:,]\s*(.*?)\s*$")
LEADING_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
# NumPy 2 prints a scalar's repr as `np.int64(12)`; the number inside is the value.
NUMPY_SCALAR = re.compile(r"^(?:np|numpy)\.\w+\((.*)\)$")
# The number right after a key, as `=12`, `: 12`, `": 12` in a dict, or `  12` in a printed Series.
NUMBER_TEXT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
VALUE_AFTER_KEY = re.compile(rf"[ \t\"']*[=:,]?[ \t\"']*((?:np|numpy)\.\w+\(\s*{NUMBER_TEXT}\s*\)|{NUMBER_TEXT})")
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
DAY_OR_MONTH_FIRST = DATE_FORMS[2]
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
                              "(it was never saved there under this name, or it was saved but not committed and pushed)")
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
        rows = [[trim_end(cell, trailing) for cell in row]
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
    body = rows[1:]
    if value_name and len(header) == 2 and header[1] in ("", "0") and value_name != header[0]:
        header[1] = value_name
    if len(header) > 1 and UNNAMED_INDEX.match(header[0]):
        if index_name and index_name not in header[1:]:
            header[0] = index_name
        else:
            header = header[1:]
            body = [row[1:] for row in body]
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
            (column == "needs_review" and boolean(left) is not None and boolean(left) == boolean(right)) or
            (column == "visit_date" and bool(calendar_dates(left) & calendar_dates(right))))

def label(text: str) -> str:
    """A label for comparison: case, spacing, `-` or `_` for a space, and a final full stop are ignored."""
    text = re.sub(r"[-_]", " ", text.casefold())
    text = re.sub(r"\s*,\s*", ",", text)
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


def squeeze(text: str) -> str:
    return "".join(text.split())


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


def boolean(cell: str) -> bool | None:
    word = cell.strip().casefold()
    return True if word in TRUE_WORDS else False if word in FALSE_WORDS else None


def day_month_order(cells: Sequence[str]) -> str | None:
    """Whether a column writes numeric dates day first ("dmy") or month first ("mdy").

    A date such as 15/01/2026 can only be day first and 01/15/2026 only month
    first; the column's unambiguous dates settle how its others, such as
    03/01/2026, are read. None when the column never settles it.
    """
    day_first = month_first = False
    for cell in cells:
        match = DAY_OR_MONTH_FIRST.match(WEEKDAY.sub("", cell.strip()))
        if match:
            day_first |= int(match["first"]) > 12
            month_first |= int(match["second_part"]) > 12
    if day_first != month_first:
        return "dmy" if day_first else "mdy"
    return None


def calendar_dates(cell: str, order: str | None = None) -> set[date]:
    """The day a written date or timestamp names, as written and, when zoned, in UTC.

    Any unambiguous way of writing the day is read: 2026-01-15, 2026/1/15,
    20260115, 15 January 2026, Jan 15, 2026, and 01/15/2026 or 15/01/2026
    (a numeric date with the year last uses `order` from day_month_order when
    its own parts do not settle it). `visit_date` is a calendar day, so either
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
        if first == second or first > 12 or (second <= 12 and order == "dmy"):
            day, month = first, second
        elif second > 12 or order == "mdy":
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


# Task 1: foundation artifacts

def check_raw_preview(root: Path) -> Outcome:
    text = read_text(root, "raw_preview.txt")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    squeezed = [squeeze(line) for line in lines]
    every_label = [index for index, line in enumerate(lines) if "people_raw.csv" in line.casefold()
                   and any(command in line.casefold() for command in PREVIEW_COMMANDS)]

    def block_end(label_index: int) -> int:
        """Where the lines under a label stop: the next label or the end of the file."""
        return next((index for index in every_label if index > label_index), len(lines))

    # Two units per command: the label and the lines it prints. The lines without their label earn one.
    outcome = Outcome(total=2 * len(PREVIEW_COMMANDS))
    for command, expected in PREVIEW_LINES.items():
        wanted = [squeeze(line) for line in expected]
        labels = [index for index in every_label if command in lines[index].casefold()]
        starts = [index for index in labels if squeezed[index + 1:index + 1 + len(wanted)] == wanted]
        if any(block_end(index) == index + 1 + len(wanted) for index in starts):
            outcome.right += 2
            continue
        if not labels and any(squeezed[index:index + len(wanted)] == wanted for index in range(len(squeezed))):
            outcome.right += 1
            outcome.problems.append(
                f"the {len(expected)} lines `{PREVIEW_COMMANDS[command].rsplit(' ', 1)[0]}` prints are there, but "
                f"no `$ {PREVIEW_COMMANDS[command]}` label line above them, so they earn half (write the label "
                f"with `echo '$ {PREVIEW_COMMANDS[command]}'` before adding the output with `>>`)")
            continue
        if starts:
            printed = PREVIEW_COMMANDS[command].rsplit(" ", 1)[0]
            outcome.problems.append(
                f"found {block_end(starts[0]) - starts[0] - 1} lines under `$ {PREVIEW_COMMANDS[command]}`, but "
                f"`{printed}` prints {len(expected)} (run the command exactly as the label shows; plain `{command}` "
                "prints 10 lines and `cat` prints the whole file)")
            continue
        if not labels:
            outcome.problems.append(
                f"no `$ {PREVIEW_COMMANDS[command]}` line; expected that label line with the "
                f"{len(expected)} lines the command prints beneath it (if the label was written, a later `>` "
                "replaced it: write the first part with `>` and add every later part with `>>`)")
            continue
        under = lines[labels[0] + 1:labels[0] + 1 + len(expected)]
        position = next((index for index, line in enumerate(wanted)
                         if index >= len(under) or squeeze(under[index]) != line), 0)
        found = shown(under[position]) if position < len(under) else "nothing"
        outcome.problems.append(
            f"under `$ {PREVIEW_COMMANDS[command]}`, line {position + 1} of {len(expected)} should be "
            f"'{expected[position]}' as that command prints it, found {found} (run the command exactly as "
            "the label shows and add its output with `>>` right after the label)")
    return outcome


def key_anywhere(text: str, key: str) -> tuple[float | None, str | None]:
    """The number after `key` anywhere in the text, and what follows its first mention.

    Finds `raw_rows=12` even when the writes left out the newlines and the keys
    run together, and `raw rows: 12`, `"raw_rows": 12`, or `rawRows 12` too.
    """
    pattern = re.compile(r"(?<![A-Za-z])" + r"[\s_-]*".join(map(re.escape, key.split("_"))), re.IGNORECASE)
    first_mention = None
    for mention in pattern.finditer(text):
        value = VALUE_AFTER_KEY.match(text, mention.end())
        if value:
            return number(value.group(1)), value.group(1)
        if first_mention is None:
            first_mention = text[mention.end():].split("\n", 1)[0].strip(" \t=:,\"'")[:40]
    return None, first_mention


def summary_counts(root: Path) -> tuple[dict[str, float | None], dict[str, str | None]]:
    """Each key's number in pipeline_summary.txt, and the text found for it (None when the key is absent).

    Lines are matched to keys by name, a close spelling, or position; a key no
    line names is then looked for anywhere in the text.
    """
    text = read_text(root, "pipeline_summary.txt")
    pairs = []
    for line in text.splitlines():
        match = KEY_VALUE.match(line)
        if match:
            pairs.append({"key": match.group(1), "value": match.group(2)})
    rows = match_labels(pairs, "key", tuple(PIPELINE_SUMMARY))
    values: dict[str, float | None] = {}
    found: dict[str, str | None] = {}
    for key in PIPELINE_SUMMARY:
        value = leading_number(rows[key]["value"]) if key in rows else None
        found[key] = rows[key]["value"] if key in rows else None
        if value is None and not (found[key] or "").startswith("conflicting repeated"):
            value, text_found = key_anywhere(text, key)
            found[key] = text_found if text_found is not None else found[key]
        values[key] = value
    return values, found


def check_pipeline_summary(root: Path) -> Outcome:
    values, found = summary_counts(root)
    outcome = Outcome(total=len(PIPELINE_SUMMARY))
    absent = [key for key in PIPELINE_SUMMARY if found[key] is None]
    if absent:
        outcome.problems.append(
            f"no {', '.join(f'`{key}`' for key in absent)} followed by a number (write each key as Task 1.2 "
            "lists it, then `=` and the count)")
    # clean_rows follows from raw_rows and exact_duplicate_rows, so it is also right when it follows from the
    # file's own two counts: a wrong raw_rows is charged there, not again here.
    own_clean = (None if values["raw_rows"] is None or values["exact_duplicate_rows"] is None
                 else values["raw_rows"] - values["exact_duplicate_rows"])
    for key, expected in PIPELINE_SUMMARY.items():
        value = values[key]
        if value == expected or (key == "clean_rows" and value is not None and value == own_clean):
            outcome.right += 1
        elif found[key] is not None:
            outcome.problems.append(f"`{key}` should be {expected}, found {shown(found[key])} "
                                    f"({PIPELINE_HINTS[key]})")
    return outcome


def age_summary(ages: Sequence[float]) -> dict[str, float]:
    return {"count": len(ages), "min": min(ages), "max": max(ages), "sum": sum(ages), "mean": sum(ages) / len(ages)}


def summary_matches(values: dict[str, float | None], summary: dict[str, float]) -> set[str]:
    """The metrics whose value is the summary's, the mean within MEAN_TOLERANCE."""
    return {metric for metric, value in values.items()
            if value is not None and abs(value - summary[metric]) <= (MEAN_TOLERANCE if metric == "mean" else 1e-9)}


def check_numpy_age_summary(root: Path) -> Outcome:
    table = read_table(root, "numpy_age_summary.csv", index_name="metric", value_name="value")
    if not {"metric", "value"} <= set(table.columns) and len(table.rows) == 1:
        # One row with a column per metric, as pd.DataFrame([summary]).to_csv(index=False) writes it.
        rows = match_labels([{"metric": column, "value": table.rows[0][column]} for column in table.columns],
                            "metric", tuple(NUMPY_AGE_SUMMARY), METRIC_ALIASES)
    else:
        table.require("metric", "value")
        rows = match_labels(table.rows, "metric", tuple(NUMPY_AGE_SUMMARY), METRIC_ALIASES)
    values = {metric: number(rows[metric]["value"]) if metric in rows else None for metric in NUMPY_AGE_SUMMARY}
    right = summary_matches(values, NUMPY_AGE_SUMMARY)
    outcome = Outcome(total=len(NUMPY_AGE_SUMMARY))
    # The choice of ages that explains the most metrics, preferring the valid ages on a tie. When another
    # choice explains four or five of them, the array holds the wrong ages but the NumPy summaries of it are
    # right: those earn their points and the choice of ages costs one.
    own_ages = max((ages for size in range(1, len(AGE_CANDIDATES) + 1) for ages in combinations(AGE_CANDIDATES, size)),
                   key=lambda ages: (len(summary_matches(values, age_summary(ages))), sorted(ages) == sorted(VALID_AGES)))
    own = summary_matches(values, age_summary(own_ages))
    if len(right) < len(NUMPY_AGE_SUMMARY) and sorted(own_ages) != sorted(VALID_AGES) and len(own) >= 4:
        extra = sorted(set(own_ages) - set(VALID_AGES))
        left_out = sorted(set(VALID_AGES) - set(own_ages))
        change = "; ".join(part for part in (f"it includes {', '.join(map(str, extra))}" if extra else "",
                                              f"it leaves out {', '.join(map(str, left_out))}" if left_out else "")
                           if part)
        outcome.problems.append(
            f"the metrics summarize the ages {', '.join(map(str, own_ages))}, not the valid ages ({change}), so the "
            f"choice of ages costs 1 point and the metrics computed from that array earn theirs ({VALID_AGE_HINT})")
        right = right | own
        outcome.right = min(len(right), len(NUMPY_AGE_SUMMARY) - 1)
    else:
        outcome.right = len(right)
    for metric, expected in NUMPY_AGE_SUMMARY.items():
        if metric in right:
            continue
        if metric not in rows:
            outcome.problems.append(f"no `{metric}` row; expected `{metric},{expected}` "
                                    "(name each metric as Task 1.3 lists it)")
        else:
            outcome.problems.append(f"`{metric}` should be {expected}, found {shown(rows[metric]['value'])} "
                                    f"({VALID_AGE_HINT})")
    return outcome


def check_pandas_selection(root: Path) -> Outcome:
    table = read_table(root, "pandas_selection.csv")
    outcome = Outcome(total=len(PANDAS_SELECTION) + 1)
    if set(table.columns) == set(SELECTION_COLUMNS):
        outcome.right += 1
    missing = [column for column in SELECTION_COLUMNS if column not in table.columns]
    if missing or set(table.columns) != set(SELECTION_COLUMNS):
        outcome.problems.append(
            f"the columns should be exactly {', '.join(SELECTION_COLUMNS)}; the header reads `{table.header}`"
            + (", so without record_id no row can be checked" if "record_id" in missing else "")
            + ' (name them in raw.loc[mask, ["record_id", "site", "status"]])')
    if "record_id" in missing:
        return outcome
    # A row is checked on the value columns it has; a missing column is charged once, above.
    value_columns = [column for column in SELECTION_COLUMNS[1:] if column in table.columns]
    matched = 0
    extra_rows = 0
    seen: set[str] = set()
    wanted = {label(record_id): record_id for record_id in PANDAS_SELECTION}
    for row in table.rows:
        key = label(row["record_id"])
        if key not in wanted or key in seen:
            extra_rows += 1
            continue
        seen.add(key)
        record_id = wanted[key]
        expected = dict(zip(SELECTION_COLUMNS[1:], PANDAS_SELECTION[record_id]))
        # The rows are the skill here, so the raw values or their cleaned forms both count: case and the
        # spaces around a value are ignored.
        if all(row[column].strip().casefold() == expected[column].strip().casefold() for column in value_columns):
            matched += 1
        else:
            outcome.problems.append(
                f"{record_id} should have site '{expected['site']}' and status '{expected['status']}' as the raw "
                f"file has them, found "
                + " and ".join(shown(row[column]) for column in value_columns)
                + " (select from raw, and select whole rows so each value stays with its record_id)")
    for key, record_id in wanted.items():
        if key not in seen:
            outcome.problems.append(f"no row for {record_id} (the mask should keep it)")
    if extra_rows:
        outcome.problems.append(
            f"the selection should hold only {', '.join(PANDAS_SELECTION)}, but {extra_rows} other "
            f"row(s) are in the file, and each one cancels a matched row (build the mask with "
            f"raw['record_id'].isin([...]) and save only the rows it keeps)")
    outcome.right += max(0, matched - extra_rows)
    return outcome


# Task 2: the audit

def check_issue_audit(root: Path) -> Outcome:
    table = read_table(root, "issue_audit.csv", index_name="issue", value_name="count")
    table.require("issue", "count")
    rows = match_labels(table.rows, "issue", tuple(issue for issue, _ in ISSUE_AUDIT))
    outcome = Outcome(total=len(ISSUE_AUDIT))
    for issue, expected in ISSUE_AUDIT:
        row = rows.get(issue)
        if row is None:
            outcome.problems.append(f"no row for the issue '{issue}' (copy each label from Task 2.2's block)")
        elif number(row["count"]) == expected:
            outcome.right += 1
        else:
            outcome.problems.append(
                f"'{issue}' should count {expected}, found {shown(row['count'])} (count it in raw, every row "
                "before duplicates are removed, comparing values after .str.strip(), as Task 2.2's table defines it)")
    return outcome


# Tasks 3 and 4: the cleaned table

CLEANED_TASK = "Task 3.3 (cleaning) and Task 4.2 (saving)"
DECISIONS_TASK = "Task 3.1 (decisions) and Task 4.2 (saving)"


def cleaned_table(root: Path) -> Table:
    """cleaned_people.csv, with a column held under an alias (CLEANED_ALIASES) read under the task's name."""
    table = read_table(root, "cleaned_people.csv")
    renames = {alias: column for column, alias in CLEANED_ALIASES.items()
               if column not in table.columns and alias in table.columns}
    if not renames:
        return table
    return Table(table.path, table.header, tuple(renames.get(column, column) for column in table.columns),
                 tuple({renames.get(column, column): cell for column, cell in row.items()} for row in table.rows))


def cleaned_records(root: Path) -> tuple[Table, dict[str, list[dict[str, str]]]]:
    """The cleaned rows keyed by record_id.

    Without a record_id column, a table with one row per record is matched in
    source order, so the other columns are still graded; the record_id check
    charges the missing column once.
    """
    table = cleaned_table(root)
    if "record_id" not in table.columns and len(table.rows) == len(CLEANED):
        return table, {label(record_id): [row] for record_id, row in zip(CLEANED, table.rows)}
    table.require("record_id")
    records: dict[str, list[dict[str, str]]] = {}
    for row in table.rows:
        records.setdefault(label(row["record_id"]), []).append(row)
    return table, records


def check_cleaned_record_ids(root: Path) -> Outcome:
    cleaned_table(root).require("record_id")
    _, records = cleaned_records(root)
    outcome = Outcome(total=len(CLEANED))
    wanted = {label(record_id): record_id for record_id in CLEANED}
    missing = [record_id for key, record_id in wanted.items() if key not in records]
    repeated = [record_id for key, record_id in wanted.items() if len(records.get(key, ())) > 1]
    unexpected = sum(len(rows) for key, rows in records.items() if key not in wanted)
    once = len(CLEANED) - len(missing) - len(repeated)
    if missing:
        outcome.problems.append(f"missing record(s) {', '.join(missing)}; keep every record and flag it instead")
    if repeated:
        outcome.problems.append(f"{', '.join(repeated)} appear(s) more than once; keep one row per record")
    if unexpected:
        outcome.problems.append(f"{unexpected} row(s) have a record_id that is not in the source")
    outcome.right = max(0, once - unexpected)
    return outcome


def text_missing_spellings(table: Table) -> frozenset[str]:
    """How a text column in this file may write a missing value.

    `NA` in a text column usually is the status sentinel left unconverted, but
    a file that also writes a missing age or date as `NA` (to_csv with
    na_rep="NA") uses it to mean missing everywhere, so there it counts as
    missing.
    """
    written = {row.get(column, "").strip().casefold() for row in table.rows for column in ("age", "visit_date")}
    return TEXT_MISSING | (written & {"na", "n/a"})


def cleaned_value_is_right(column: str, expected: object, cell: str, row: dict[str, str],
                           text_missing: frozenset[str] = TEXT_MISSING, order: str | None = None) -> bool:
    if column in ("full_name", "site", "status"):
        return is_missing(cell, text_missing) if expected is None else cell.strip().casefold() == str(expected).strip().casefold()
    if expected is None:
        return is_missing(cell)
    if column == "age":
        return number(cell) == expected
    if column == "visit_date":
        return expected in calendar_dates(cell, order)
    flag = boolean(cell)
    if flag == expected:
        return True
    # A flag that follows the rule from this file's own age and visit_date is right:
    # a mistake in either column is charged there, not a second time here.
    if flag is not None and "age" in row and "visit_date" in row:
        return flag == (is_missing(row["age"]) or is_missing(row["visit_date"]))
    return False


def cleaned_column_check(column: str) -> Callable[[Path], Outcome]:
    def check(root: Path) -> Outcome:
        table, records = cleaned_records(root)
        table.require(column)
        if not table.rows:
            raise ArtifactProblem("output/cleaned_people.csv has a header but no data rows; save the cleaned table")
        if not any(label(record_id) in records for record_id in CLEANED):
            raise ArtifactProblem("output/cleaned_people.csv has no recognizable source record_id, so its values "
                                  "cannot be checked; expected R001 through R011 (keep the source record IDs "
                                  "and save the cleaned table again)")
        text_missing = text_missing_spellings(table)
        order = day_month_order([row.get("visit_date", "") for row in table.rows])
        outcome = Outcome(total=len(CLEANED))
        wrong: list[str] = []
        first = ""
        for record_id, expected in CLEANED.items():
            rows = records.get(label(record_id))
            if not rows:
                # The identifier check already charges a missing record.
                outcome.right += 1
                continue
            bad = next((row for row in rows if not cleaned_value_is_right(column, expected[column], row[column],
                                                                          row, text_missing, order)), None)
            if bad is None:
                outcome.right += 1
                continue
            wrong.append(record_id)
            if not first:
                wanted = expected[column]
                if isinstance(wanted, date):
                    wanted = wanted.isoformat()
                first = f"first {record_id}: expected {shown(wanted)}, found {shown(bad[column])}"
        if wrong:
            others = f" (also {', '.join(wrong[1:])})" if len(wrong) > 1 else ""
            outcome.problems.append(
                f"column {column}: {len(wrong)} of {len(CLEANED)} records differ; {first}{others}; "
                f"the rule: {CLEANED_RULES[column]}")
        return outcome
    return check


# Tasks 3 and 4: the decision log

def decision_table(root: Path) -> Table:
    table = read_table(root, "decision_log.csv")
    if not table.rows:
        raise ArtifactProblem(f"{table.path} has a header but no decision rows")
    return table


def check_decisions(root: Path) -> Outcome:
    table = decision_table(root)
    table.require("field", "issue", "action")
    outcome = Outcome(total=len(DECISIONS))
    rows = table.rows
    # Each decision's row: the first with its field and issue, then, for a field or issue with a typo, the
    # unused row most like the whole decision. A row is used once.
    chosen: dict[int, int] = {}
    for position, (field_name, issue, _) in enumerate(DECISIONS):
        index = next((index for index, row in enumerate(rows) if index not in chosen.values()
                      and (label(row["field"]), label(row["issue"])) == (label(field_name), label(issue))), None)
        if index is not None:
            chosen[position] = index
    pairs = sorted(((similarity(" | ".join(DECISIONS[position]),
                                " | ".join((row["field"], row["issue"], row["action"]))), position, index)
                    for position in range(len(DECISIONS)) if position not in chosen
                    for index, row in enumerate(rows) if index not in chosen.values()),
                   key=lambda pair: (-pair[0], pair[1], pair[2]))
    for ratio, position, index in pairs:
        if ratio >= SIMILAR and position not in chosen and index not in chosen.values():
            chosen[position] = index
    for position, (field_name, issue, action) in enumerate(DECISIONS):
        if position not in chosen:
            outcome.problems.append(f"no row with field '{field_name}' and issue '{issue}' (copy the field "
                                    "and issue of each decision from Task 3.1's table)")
            continue
        found = rows[chosen[position]]["action"]
        # A typo or a dropped word still names the action; a different action does not.
        if similarity(found, action) >= SIMILAR:
            outcome.right += 1
        else:
            outcome.problems.append(
                f"the '{issue}' decision's action should be '{action}', found {shown(found)} (copy it "
                "from Task 3.1's table)")
    return outcome


def same_source(cell: str) -> bool:
    """Whether the cell names the source file: data/people_raw.csv, or any path or Path repr ending in it."""
    path = cell.strip().replace("\\", "/").casefold()
    return re.search(r"(?<![\w.-])people_raw\.csv(?![\w.-])", path) is not None


def cleaned_row_count(root: Path) -> int | None:
    try:
        return len(read_table(root, "cleaned_people.csv").rows)
    except ArtifactProblem:
        return None


def provenance_check(column: str, describe: str, cause: str,
                     accept: Callable[[Path], Callable[[str], bool]]) -> Callable[[Path], Outcome]:
    """Score one column of the decision log by the share of decision rows that carry the right value."""
    def check(root: Path) -> Outcome:
        table = decision_table(root)
        table.require(column)
        is_right = accept(root)
        wrong = [(position, row) for position, row in enumerate(table.rows, start=1)
                 if not is_right(row[column])]
        outcome = Outcome(total=len(table.rows), right=len(table.rows) - len(wrong))
        if wrong:
            position, row = wrong[0]
            issue = row.get("issue", "").strip()
            which = f"row {position}" + (f" ('{issue}')" if issue else "")
            outcome.problems.append(
                f"{column} should be {describe} on every decision row; {len(wrong)} of {len(table.rows)} "
                f"rows differ, first {which} has {shown(row[column])} ({cause})")
        return outcome
    return check


def _reason(root: Path) -> Callable[[str], bool]:
    return lambda cell: not is_missing(cell, TEXT_MISSING)


def _source(root: Path) -> Callable[[str], bool]:
    return same_source


def _sha(root: Path) -> Callable[[str], bool]:
    # The hash with a label around it, such as `sha256:<hash>`, still is the hash.
    return lambda cell: SOURCE_SHA256 in cell.casefold()


def _rows_before(root: Path) -> Callable[[str], bool]:
    # rows_before is the raw_rows count again, so it is also right when it equals the raw_rows in the
    # file's own pipeline_summary.txt: a wrong count is charged there, not again here.
    try:
        own = summary_counts(root)[0]["raw_rows"]
    except ArtifactProblem:
        own = None
    accepted = {RAW_ROWS, own} - {None}
    return lambda cell: number(cell) in accepted


def _rows_after(root: Path) -> Callable[[str], bool]:
    # The count of the committed cleaned table is right too, so a cleaning
    # mistake is charged to cleaned_people.csv and not again here.
    accepted = {CLEAN_ROWS, cleaned_row_count(root)} - {None}
    return lambda cell: number(cell) in accepted


CHECKS = (
    Check("raw_preview.txt", 4, "Task 1.1", check_raw_preview),
    Check("pipeline_summary.txt", 5, "Task 1.2", check_pipeline_summary),
    Check("numpy_age_summary.csv", 5, "Task 1.3", check_numpy_age_summary),
    Check("pandas_selection.csv", 4, "Task 1.4", check_pandas_selection),
    Check("issue_audit.csv", 15, "Task 2.2", check_issue_audit),
    Check("cleaned_people.csv: record_id", 4, CLEANED_TASK, check_cleaned_record_ids),
    *(Check(f"cleaned_people.csv: {column}", 4, CLEANED_TASK, cleaned_column_check(column))
      for column in CLEANED_COLUMNS[1:]),
    Check("decision_log.csv: decisions", 8, DECISIONS_TASK, check_decisions),
    Check("decision_log.csv: reason", 2, DECISIONS_TASK,
          provenance_check("reason", "a written reason", "write one sentence for every decision", _reason)),
    Check("decision_log.csv: source", 1, "Task 4.2",
          provenance_check("source", f"'{SOURCE}'", "write the source path as Task 4.2 gives it", _source)),
    Check("decision_log.csv: source_sha256", 1, "Task 4.2",
          provenance_check("source_sha256", "the sha256 value from data/fixture.json",
                           'copy manifest["sha256"], which the first code cell loads from data/fixture.json',
                           _sha)),
    Check("decision_log.csv: rows_before", 1, "Task 4.2",
          provenance_check("rows_before", f"the raw table's row count, {RAW_ROWS}",
                           "use len(raw), counted before duplicates are removed", _rows_before)),
    Check("decision_log.csv: rows_after", 1, "Task 4.2",
          provenance_check("rows_after", f"the cleaned table's row count, {CLEAN_ROWS}",
                           "use len(cleaned), the table saved as cleaned_people.csv", _rows_after)),
)
MAX_SCORE = sum(check.points for check in CHECKS)


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
    tests = [run_check(check, root) for check in CHECKS]
    return {"schema": SCHEMA, "score": sum(test["score"] for test in tests), "max-score": MAX_SCORE,
            "tests": tests}
