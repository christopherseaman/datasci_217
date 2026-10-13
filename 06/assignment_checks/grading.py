"""Course-owned grading rules for Assignment 06, the midterm.

The midterm handout ships no checks. After the deadline the course grades each
fork's committed files in `output/` with this module, through
`check_assignment.py` and `scripts/grade_submissions.py`. Student code is never
imported, executed, or read, and the notebook is left to human review.

Every expected value follows from the five supplied files in `data/` (their
SHA-256 values are in `data/manifest.json`); `_grader_selftest/run.py`
recomputes each one from those files with pandas and confirms it matches the
constants below.

Each check reads one artifact and compares parsed values rather than text, so
line endings, trailing whitespace, a missing final newline, header spacing, a
comma, semicolon, or tab between cells, column order, a leading unnamed index
column, number formatting, the letter case and spacing of labels and text
values, any unambiguous way of writing a date, and the common spellings of
missing values never cost points. The exception is the three columns whose
rule is the normalization itself (`sex` and `site_code` in patients_clean.csv,
`visit` in visits_clean.csv): their exact text counts there, once, and later
checks follow the student's exact text as their own merges and pivots did. Each check scores the rows or records it can
verify on its own, so one wrong value costs only its own share, and no check
depends on another passing. A later artifact is also judged against the
student's own earlier one (the joined table against their cleaned tables, the
merge counts against their own keys, the SBP change against their own
baseline and week 12 columns), so one mistake is charged once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
import io
import math
from pathlib import Path
import re


SCHEMA = "datasci217/grading-result/v1"
OUTPUT = Path("output")

DATA_FILES = ("patients.csv", "sites.csv", "visits_north.csv", "visits_south.csv", "a1c_wide.csv")
# `wc -l data/*.csv` counts newline characters, so each count includes the header line.
INVENTORY = {"patients.csv": 26, "sites.csv": 5, "visits_north.csv": 23, "visits_south.csv": 27,
             "a1c_wide.csv": 25}
INVENTORY_TOTAL = 106

# Each raw table's data rows, columns, and rows identical to an earlier row.
TABLE_PROFILE = {
    "patients.csv": (25, 4, 1),
    "sites.csv": (4, 3, 0),
    "visits_north.csv": (22, 6, 1),
    "visits_south.csv": (26, 6, 0),
    "a1c_wide.csv": (24, 3, 0),
}
PROFILE_COLUMNS = ("rows", "columns", "exact_duplicate_rows")
PROFILE_HINTS = {
    "rows": "len(table) counts data rows; the header is not a row (wc -l counts it, so its numbers are one higher)",
    "columns": "table.shape[1] counts the columns",
    "exact_duplicate_rows": "table.duplicated().sum() counts each repeat once, without keep=False",
}

ISSUE_AUDIT = (
    ("patients: sex values needing normalization", 3),
    ("patients: empty sex", 1),
    ("patients: birth_year sentinel 9999", 1),
    ("patients: site codes needing normalization", 2),
    ("sites: rows with a repeated site_code", 2),
    ("visits: visit labels needing normalization", 4),
    ("visits: sbp sentinel 999", 2),
    ("visits: sbp outside 70 through 250", 1),
    ("visits: date parse failures", 2),
    ("visits: rows in repeated patient_id and visit pairs", 4),
)
ISSUE_HINTS = {
    "patients: sex values needing normalization": "leave out empty values and count every other sex that "
                                                  ".str.strip().str.upper() changes",
    "patients: empty sex": "compare the stripped value with '' in the table read with keep_default_na=False",
    "patients: birth_year sentinel 9999": "count birth_year values equal to '9999'",
    "patients: site codes needing normalization": "count every site_code that .str.strip().str.upper() changes",
    "sites: rows with a repeated site_code": "use duplicated(subset=['site_code'], keep=False), which counts the "
                                             "first copy too",
    "visits: visit labels needing normalization": "count, in both visit files stacked, every visit label that "
                                                  ".str.strip().str.lower() changes",
    "visits: sbp sentinel 999": "count sbp_mmhg values equal to '999' in both visit files",
    "visits: sbp outside 70 through 250": "convert with pd.to_numeric, leave out the 999 sentinel, and keep 70 "
                                          "and 250 inside the range",
    "visits: date parse failures": "check the text with .str.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}') first, "
                                   "because format='%Y-%m-%d' alone accepts 2026-5-10, then let "
                                   "to_datetime(..., errors='coerce') catch impossible dates",
    "visits: rows in repeated patient_id and visit pairs": "normalize the visit labels first, then use "
                                                           "duplicated(subset=['patient_id', 'visit'], keep=False) "
                                                           "on both visit files stacked",
}

# The expected cleaned tables, as the reference solution writes them (an empty cell is missing).
_PATIENTS = """\
P001,F,1991,N01
P002,F,1984,N01
P003,F,1949,N01
P004,M,1966,N01
P005,M,1943,N01
P006,M,1955,N01
P007,M,,N01
P008,F,1954,N01
P009,F,1985,N01
P010,M,1991,N01
P011,F,1964,N01
P012,F,1965,N01
P013,M,1946,S02
P014,M,1944,S02
P015,M,1944,S02
P016,M,1963,S02
P017,F,1948,S02
P018,M,1966,S02
P019,F,1951,S02
P020,,1953,S02
P021,F,1971,S02
P022,M,1993,S02
P023,M,1958,E04
P024,F,1981,S02
"""
_VISITS = """\
P001,baseline,2026-01-28,134,72.3,north
P001,week12,2026-04-21,131,69.7,north
P002,baseline,2026-01-22,161,86.7,north
P002,week12,2026-04-14,141,84.0,north
P003,baseline,2026-02-22,161,101.4,north
P003,week12,2026-05-16,160,101.3,north
P004,baseline,2026-01-05,163,97.3,north
P004,week12,2026-03-30,163,94.4,north
P005,baseline,2026-02-25,135,101.0,north
P006,baseline,2026-01-14,,72.1,north
P006,week12,2026-04-09,132,71.3,north
P007,baseline,2026-02-09,172,111.1,north
P007,week12,2026-05-02,154,111.2,north
P008,baseline,2026-01-27,133,94.6,north
P008,week12,2026-04-21,128,93.5,north
P009,baseline,2026-01-13,151,64.6,north
P009,week12,2026-04-07,130,65.0,north
P010,baseline,2026-02-12,147,69.0,north
P010,week12,,151,68.6,north
P011,baseline,2026-02-12,133,87.8,north
P011,week12,2026-05-10,115,85.8,north
P013,baseline,2026-02-27,144,95.6,south
P013,week12,2026-05-23,130,94.0,south
P014,baseline,2026-01-29,137,63.9,south
P014,week12,2026-04-22,125,62.0,south
P015,baseline,2026-01-16,152,59.8,south
P015,week12,2026-04-10,151,58.3,south
P016,baseline,2026-02-20,137,100.7,south
P016,week12,2026-05-12,124,98.7,south
P017,baseline,2026-01-14,142,61.7,south
P017,week12,2026-04-11,,62.5,south
P018,baseline,2026-02-24,149,55.5,south
P019,baseline,,138,113.2,south
P019,week12,2026-04-21,133,111.4,south
P020,baseline,2026-02-01,172,106.7,south
P020,week12,2026-04-29,173,104.9,south
P021,baseline,2026-03-02,165,62.1,south
P021,week12,2026-05-22,163,60.6,south
P022,baseline,2026-01-11,,73.8,south
P022,week12,2026-04-08,168,71.0,south
P023,baseline,2026-01-11,137,101.3,south
P023,week12,2026-04-04,134,101.5,south
P024,baseline,2026-02-17,141,61.1,south
P024,week12,2026-05-12,123,,south
P030,baseline,2026-03-05,135,83.7,south
P030,week12,2026-05-25,137,81.8,south
"""
_A1C = """\
P001,7.7,7.2
P002,10.4,9.9
P003,9.4,8.5
P004,9.8,9.5
P005,8.6,7.3
P006,7.2,6.1
P007,6.1,5.2
P008,6.5,5.4
P009,8.7,8.9
P010,6.6,6.0
P011,9.4,8.4
P013,7.7,8.0
P014,6.4,6.3
P015,9.1,8.4
P016,5.6,4.2
P017,7.3,6.6
P018,9.7,
P019,8.1,6.9
P020,5.9,6.1
P021,6.7,
P022,6.0,5.2
P023,6.5,6.8
P024,10.3,9.7
P099,9.8,9.2
"""


def _cell(text: str, kind: Callable[[str], object]) -> object:
    return kind(text) if text else None


PATIENTS = {pid: {"sex": _cell(sex, str), "birth_year": _cell(year, int), "site_code": site}
            for pid, sex, year, site in (line.split(",") for line in _PATIENTS.splitlines())}
VISITS = {(pid, visit): {"visit_date": _cell(day, date.fromisoformat), "sbp_mmhg": _cell(sbp, int),
                         "weight_kg": _cell(weight, float), "source_file": source}
          for pid, visit, day, sbp, weight, source in (line.split(",") for line in _VISITS.splitlines())}
A1C = {(pid, visit): float(value)
       for pid, *values in (line.split(",") for line in _A1C.splitlines())
       for visit, value in zip(("baseline", "week12"), values) if value}
CURRENT_SITES = {"N01": "North Clinic", "S02": "South Valley Clinic", "W03": "West Clinic"}
RETIRED_SITES = {"S02": "Southgate Clinic"}
SITE_CODES = frozenset(CURRENT_SITES)
VISIT_LABELS = ("baseline", "week12")

# One row per clean visit, as the left merges onto visits_clean give it.
ANALYSIS = {key: {"a1c_pct": A1C.get(key), **PATIENTS.get(key[0], {"sex": None, "birth_year": None,
                                                                   "site_code": None})}
            for key in VISITS}
for _row in ANALYSIS.values():
    _row["site_name"] = CURRENT_SITES.get(_row["site_code"] or "")

SBP_WIDE = {pid: {visit: VISITS.get((pid, visit), {}).get("sbp_mmhg") for visit in VISIT_LABELS}
            for pid in dict.fromkeys(pid for pid, _ in VISITS)}

MERGE_AUDIT = (
    ("visits with an A1c result", 43),
    ("visits without an A1c result", 3),
    ("A1c results without a visit", 3),
    ("visits whose patient is not in the registry", 2),
    ("registry patients with no visit", 1),
    ("registry patients whose site is not in the site list", 1),
    ("site list rows removed by the current-record rule", 1),
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

DECISION_COLUMNS = ("table", "decision", "reason")
DECISIONS = (
    ("patients", "remove the exact duplicate row, keeping the first"),
    ("patients", "strip and uppercase sex and site_code; an empty sex becomes missing"),
    ("patients", "convert the birth_year sentinel 9999 to missing"),
    ("visits", "rename wt_kg to weight_kg before stacking"),
    ("visits", "convert sbp_mmhg outside 70 through 250, including 999, to missing"),
    ("visits", "keep only exact YYYY-MM-DD calendar dates"),
    ("visits", "keep the later reading of a repeated patient_id and visit pair"),
    ("sites", "keep only current site records before joining"),
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




# Task 1: the file inventory and the table profile

def file_key(text: str) -> str:
    """A data file's name for comparison: `data/patients.csv`, `patients.csv`, and `Patients` are the same."""
    name = label(text).replace("\\", "/").rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".csv") else name


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

def check_issue_audit(root: Path) -> Outcome:
    table = read_table(root, "issue_audit.csv", index_name="issue", value_name="count")
    table.require("issue", "count")
    rows = match_labels(table.rows, "issue", tuple(issue for issue, _ in ISSUE_AUDIT))
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


def exact_right(expected: object, cell: str) -> bool:
    """For a column whose rule is stripping and changing case: the cell must be that normalized text."""
    if expected is None:
        return is_missing(cell, TEXT_MISSING)
    return cell == expected


def number_right(expected: object, cell: str) -> bool:
    if expected is None:
        return is_missing(cell)
    value = number(cell)
    return value is not None and abs(value - float(expected)) < 1e-6


def date_right(expected: object, cell: str) -> bool:
    return is_missing(cell) if expected is None else expected in calendar_dates(cell)


def source_right(expected: object, cell: str) -> bool:
    """`visits_north.csv`, `data/visits_north.csv`, and `north` all name the north file."""
    words = set(re.split(r"[^a-z]+", cell.casefold()))
    other = "south" if expected == "north" else "north"
    return expected in words and other not in words


def read_keyed(root: Path, name: str, keys: Sequence[str],
               exact: bool = False) -> tuple[Table, dict[Key, list[dict[str, str]]]]:
    """A table's rows grouped by their normalized key cells, or by the cells as written with `exact`."""
    table = read_table(root, name)
    table.require(*keys)
    if not table.rows:
        raise ArtifactProblem(f"{table.path} has a header but no data rows")
    grouped: dict[Key, list[dict[str, str]]] = {}
    for row in table.rows:
        grouped.setdefault(tuple(row[key] if exact else label(row[key]) for key in keys), []).append(row)
    return table, grouped


def own_rows(root: Path, name: str, keys: Sequence[str],
             exact: bool = False) -> dict[Key, list[dict[str, str]]] | None:
    """The student's own earlier table, grouped by key, or None when it cannot be read."""
    try:
        return read_keyed(root, name, keys, exact)[1]
    except ArtifactProblem:
        return None


def normal_key(key: Key) -> Key:
    return tuple(label(part) for part in key)


def score_rows(grouped: dict[Key, list[dict[str, str]]], expected: Sequence[Key],
               ignore: Callable[[dict[str, str]], bool] = lambda row: False) -> Outcome:
    """Each expected key present exactly once; a repeated or unexpected row cancels a matched one."""
    wanted = {normal_key(key): key for key in expected}
    missing = [key for norm, key in wanted.items() if norm not in grouped]
    repeated = [key for norm, key in wanted.items() if len(grouped.get(norm, ())) > 1]
    unexpected = [key for key, rows in grouped.items() if key not in wanted for row in rows if not ignore(row)]
    outcome = Outcome(total=len(wanted), right=max(0, len(wanted) - len(missing) - len(repeated) - len(unexpected)))

    def listed(keys: Sequence[Key]) -> str:
        names = [" ".join(key) for key in keys[:4]]
        return ", ".join(names) + (f" and {len(keys) - 4} more" if len(keys) > 4 else "")

    if missing:
        outcome.problems.append(f"{len(missing)} expected row(s) missing, first {listed(missing)}")
    if repeated:
        outcome.problems.append(f"{listed(repeated)} appear(s) more than once")
    if unexpected:
        outcome.problems.append(f"{len(unexpected)} row(s) not expected, first {listed(unexpected)}")
    return outcome


def best_rows(grouped: dict[Key, list[dict[str, str]]], options: Sequence[Sequence[Key]],
              hint: str, **kwargs) -> Outcome:
    """The row score against whichever accepted key set (the expected one, or the student's own) fits best."""
    outcomes = [score_rows(grouped, keys, **kwargs) for keys in options if keys]
    best = max(outcomes, key=lambda outcome: (outcome.right / max(outcome.total, 1), -len(outcome.problems)))
    if best.problems:
        best.problems[-1] += f" ({hint})"
    return best


def score_column(table: Table, grouped: dict[Key, list[dict[str, str]]], column: str,
                 expected: dict[Key, Sequence[object]], right: Callable[[object, str], bool], rule: str) -> Outcome:
    """The share of expected records whose value in `column` is right.

    `expected` maps each key to the values accepted for it (the reference value
    and, where an earlier artifact feeds this one, the value the student's own
    earlier artifact implies). A record the file lacks counts as right here,
    because the row check charges it; among repeated rows any right one counts.
    """
    table.require(column)
    outcome = Outcome(total=len(expected))
    wrong: list[str] = []
    first = ""
    for key, accepted in expected.items():
        rows = grouped.get(normal_key(key))
        if not rows or any(right(value, row[column]) for row in rows for value in accepted):
            outcome.right += 1
            continue
        wrong.append(" ".join(key))
        if not first:
            want = accepted[0].isoformat() if isinstance(accepted[0], date) else accepted[0]
            first = f"first {' '.join(key)}: expected {shown(want)}, found {shown(rows[0][column])}"
    if wrong:
        others = f" (also {', '.join(wrong[1:4])}{' and more' if len(wrong) > 4 else ''})" if len(wrong) > 1 else ""
        outcome.problems.append(f"column {column}: {len(wrong)} of {len(expected)} records differ; {first}{others}; "
                                f"the rule: {rule}")
    return outcome


def value_of(rows: dict[Key, list[dict[str, str]]] | None, key: Key, column: str,
             parse: Callable[[str], object], exact: bool = False) -> list[object]:
    """The parsed value the student's own earlier table holds for `key`, as a one-item list, or []."""
    found = (rows or {}).get(key if exact else normal_key(key))
    if not found or column not in found[0]:
        return []
    cell = found[0][column]
    if is_missing(cell, TEXT_MISSING if parse is str else VALUE_MISSING):
        return [None]
    value = parse(cell)
    return [] if value is None else [value]


def parse_text(cell: str) -> str:
    return cell.strip()


def parse_number(cell: str) -> float | None:
    return number(cell)


# Task 3.1: patients_clean.csv

PATIENT_RULES = {
    "sex": "strip and uppercase; an empty sex becomes missing",
    "birth_year": "a whole year; the 9999 sentinel becomes missing",
    "site_code": "strip and uppercase",
}


def patients_rows(root: Path) -> Outcome:
    _, grouped = read_keyed(root, "patients_clean.csv", ("patient_id",))
    return best_rows(grouped, [[(pid,) for pid in PATIENTS]],
                     "keep one row per patient: drop_duplicates() removes the exact repeat of P009")


def patients_column(column: str, right: Callable[[object, str], bool]) -> Callable[[Path], Outcome]:
    def check(root: Path) -> Outcome:
        table, grouped = read_keyed(root, "patients_clean.csv", ("patient_id",))
        expected = {(pid,): [values[column]] for pid, values in PATIENTS.items()}
        return score_column(table, grouped, column, expected, right, PATIENT_RULES[column])
    return check


# Task 3.2: visits_clean.csv

VISIT_KEYS = ("patient_id", "visit")
VISIT_RULES = {
    "visit_date": "keep only exact YYYY-MM-DD text for a date on the calendar; anything else becomes missing",
    "sbp_mmhg": "a whole number from 70 through 250; 999 and other values outside the range become missing, and "
                "a repeated patient_id and visit pair keeps the later reading by visit_time",
    "weight_kg": "the number from weight_kg in visits_north.csv or wt_kg in visits_south.csv; a blank stays missing",
    "source_file": "the file each row came from, visits_north.csv or visits_south.csv, added before stacking",
}
VISIT_CHECKS = {"visit_date": date_right, "sbp_mmhg": number_right, "weight_kg": number_right,
                "source_file": source_right}


def visits_rows(root: Path) -> Outcome:
    """One row per visit, each with its visit label stripped and lowercased; an unnormalized label costs its row."""
    table, grouped = read_keyed(root, "visits_clean.csv", VISIT_KEYS)
    outcome = best_rows(grouped, [list(VISITS)],
                        "stack both files, lowercase and strip the visit labels, drop the exact repeat, then keep one "
                        "row per patient_id and visit")
    unnormalized = [row for row in table.rows if row["visit"] != row["visit"].strip().lower()]
    if unnormalized:
        outcome.right = max(0, outcome.right - len(unnormalized))
        first = unnormalized[0]
        outcome.problems.append(f"{len(unnormalized)} visit label(s) not stripped and lowercased, first "
                                f"{first['patient_id']} {shown(first['visit'])} (apply .str.strip().str.lower() "
                                "to visit before removing repeats)")
    return outcome


def visits_column(column: str) -> Callable[[Path], Outcome]:
    def check(root: Path) -> Outcome:
        table, grouped = read_keyed(root, "visits_clean.csv", VISIT_KEYS)
        if column == "weight_kg" and "weight_kg" not in table.columns and "wt_kg" in table.columns:
            raise ArtifactProblem("output/visits_clean.csv has wt_kg but no weight_kg column, so the south "
                                  "weights never joined the north ones (rename wt_kg to weight_kg before concat)")
        expected = {key: [values[column]] for key, values in VISITS.items()}
        right = VISIT_CHECKS[column]
        if column == "visit_date":
            # A column's unambiguous dates settle whether its others, such as 03/01/2026, are day or month first.
            order = day_month_order([row.get(column, "") for row in table.rows])
            right = lambda want, cell: is_missing(cell) if want is None else want in calendar_dates(cell, order)  # noqa: E731
        return score_column(table, grouped, column, expected, right, VISIT_RULES[column])
    return check


# Task 4.1: a1c_long.csv

def a1c_rows(root: Path) -> Outcome:
    table, grouped = read_keyed(root, "a1c_long.csv", VISIT_KEYS)
    value = "a1c_pct" if "a1c_pct" in table.columns else None
    # Rows melt() makes for a blank result may stay; they hold no value to check.
    ignore = (lambda row: is_missing(row[value])) if value else (lambda row: False)
    return best_rows(grouped, [list(A1C)],
                     "melt a1c_baseline and a1c_week12, map the labels to baseline and week12, and drop the rows "
                     "with no result", ignore=ignore)


def a1c_values(root: Path) -> Outcome:
    table, grouped = read_keyed(root, "a1c_long.csv", VISIT_KEYS)
    return score_column(table, grouped, "a1c_pct", {key: [value] for key, value in A1C.items()}, number_right,
                        "the A1c percentage from the matching a1c_wide.csv cell")


# Task 4.2: merge_audit.csv

def own_keys(rows: dict[Key, list[dict[str, str]]] | None, value: str | None = None) -> list[Key] | None:
    """The student's own keys, one per row, leaving out rows whose `value` is missing."""
    if rows is None:
        return None
    return [key for key, group in rows.items() for row in group
            if value is None or not is_missing(row.get(value, ""))]


def merge_counts(root: Path) -> dict[str, int]:
    """The merge counts the student's own visits_clean, a1c_long, and patients_clean imply."""
    # Keys and site codes compare exactly as written, as the student's own merge compared them.
    visits = own_keys(own_rows(root, "visits_clean.csv", VISIT_KEYS, exact=True))
    labs = own_keys(own_rows(root, "a1c_long.csv", VISIT_KEYS, exact=True), "a1c_pct")
    patients = own_rows(root, "patients_clean.csv", ("patient_id",), exact=True)
    counts: dict[str, int] = {}
    if visits is not None and labs is not None:
        lab_set, visit_set = set(labs), set(visits)
        counts["visits with an A1c result"] = sum(key in lab_set for key in visits)
        counts["visits without an A1c result"] = sum(key not in lab_set for key in visits)
        counts["A1c results without a visit"] = sum(key not in visit_set for key in labs)
    if visits is not None and patients is not None:
        registry = {key[0] for key in patients}
        counts["visits whose patient is not in the registry"] = sum(key[0] not in registry for key in visits)
        counts["registry patients with no visit"] = len(registry - {key[0] for key in visits})
    if patients is not None:
        counts["registry patients whose site is not in the site list"] = sum(
            row.get("site_code", "") not in SITE_CODES
            for group in patients.values() for row in group)
    return counts


def check_merge_audit(root: Path) -> Outcome:
    table = read_table(root, "merge_audit.csv", index_name="check", value_name="count")
    table.require("check", "count")
    rows = match_labels(table.rows, "check", tuple(name for name, _ in MERGE_AUDIT))
    own = merge_counts(root)
    outcome = Outcome(total=len(MERGE_AUDIT))
    for name, expected in MERGE_AUDIT:
        row = rows.get(name)
        if row is None:
            outcome.problems.append(f"no row for '{name}' (copy each label from Task 4.2's block)")
        elif number(row["count"]) in {expected, own.get(name)}:
            outcome.right += 1
        else:
            yours = f" (or {own[name]} from your own saved tables)" if own.get(name, expected) != expected else ""
            outcome.problems.append(f"'{name}' should count {expected}{yours}, found {shown(row['count'])} "
                                    f"({MERGE_HINTS[name]})")
    return outcome


# Task 4.3: analysis_visits.csv

def analysis_rows(root: Path) -> Outcome:
    _, grouped = read_keyed(root, "analysis_visits.csv", VISIT_KEYS)
    options = [list(VISITS), own_keys(own_rows(root, "visits_clean.csv", VISIT_KEYS)) or []]
    return best_rows(grouped, options,
                     "left-merge onto visits_clean so every clean visit keeps exactly one row; a repeated row "
                     "usually means a lookup table still repeats a key, such as P009 in patients or S02 in sites")


def analysis_expected(root: Path, column: str) -> dict[Key, list[object]]:
    """Each clean visit's accepted values: the reference, and what the student's own tables imply."""
    patients = own_rows(root, "patients_clean.csv", ("patient_id",))
    # The A1c the student's own merge finds: their visit key, exactly as written, looked up in their a1c_long.
    labs = own_rows(root, "a1c_long.csv", VISIT_KEYS, exact=True)
    own_visits: dict[Key, list[Key]] = {}
    for written in own_rows(root, "visits_clean.csv", VISIT_KEYS, exact=True) or {}:
        own_visits.setdefault(normal_key(written), []).append(written)
    expected: dict[Key, list[object]] = {}
    for key, values in ANALYSIS.items():
        accepted = [values[column]]
        if column == "a1c_pct" and labs is not None:
            for written in own_visits.get(normal_key(key), [key]):
                accepted += value_of(labs, written, "a1c_pct", parse_number, exact=True) or [None]
        if column in ("sex", "birth_year"):
            accepted += value_of(patients, (key[0],), column, parse_text if column == "sex" else parse_number)
        expected[key] = accepted
    return expected


def check_analysis_sites(root: Path) -> Outcome:
    table, grouped = read_keyed(root, "analysis_visits.csv", VISIT_KEYS)
    table.require("site_name")
    expected = {key: [values["site_name"]] for key, values in ANALYSIS.items()}
    if "site_code" in table.columns:
        # The name the current site list gives the row's own site_code, exactly as written, is right too.
        for key in expected:
            for row in grouped.get(normal_key(key), ()):
                expected[key].append(CURRENT_SITES.get(row["site_code"]))
    return score_column(table, grouped, "site_name", expected, text_right,
                        "the current site_name for the patient's site_code (keep record_status == 'current' "
                        "before merging; Southgate Clinic is retired); missing when the patient or site is unknown")


def check_analysis_a1c(root: Path) -> Outcome:
    table, grouped = read_keyed(root, "analysis_visits.csv", VISIT_KEYS)
    return score_column(table, grouped, "a1c_pct", analysis_expected(root, "a1c_pct"), number_right,
                        "the a1c_long value for the same patient_id and visit, merged on both columns; missing "
                        "when there is no result")


def check_analysis_patient(root: Path) -> Outcome:
    table, grouped = read_keyed(root, "analysis_visits.csv", VISIT_KEYS)
    table.require("sex", "birth_year")
    sexes, years = analysis_expected(root, "sex"), analysis_expected(root, "birth_year")
    outcome = Outcome(total=len(ANALYSIS))
    wrong = []
    for key in ANALYSIS:
        rows = grouped.get(normal_key(key))
        if not rows or any(any(text_right(sex, row["sex"]) for sex in sexes[key])
                           and any(number_right(year, row["birth_year"]) for year in years[key]) for row in rows):
            outcome.right += 1
        else:
            wrong.append(" ".join(key))
    if wrong:
        key = tuple(wrong[0].split(" "))
        row = grouped[normal_key(key)][0]
        outcome.problems.append(
            f"sex and birth_year: {len(wrong)} of {len(ANALYSIS)} visits differ; first {wrong[0]}: expected "
            f"{shown(sexes[key][0])} and {shown(years[key][0])}, found {shown(row['sex'])} and "
            f"{shown(row['birth_year'])}; the rule: the patient's values from patients_clean, merged on patient_id "
            "with how='left' (missing for a patient not in the registry)")
    return outcome


# Task 4.4: sbp_wide.csv

def sbp_wide_rows(root: Path) -> Outcome:
    _, grouped = read_keyed(root, "sbp_wide.csv", ("patient_id",))
    own = own_keys(own_rows(root, "visits_clean.csv", VISIT_KEYS))
    options = [[(pid,) for pid in SBP_WIDE], list(dict.fromkeys((key[0],) for key in own or []))]
    return best_rows(grouped, options,
                     "pivot visits_clean with index='patient_id', so each patient with a visit gets one row")


def sbp_expected(root: Path, visit: str) -> dict[Key, list[object]]:
    """Each patient's accepted reading: the reference, and the one the student's own visits_clean holds."""
    # pivot() makes a column per visit label exactly as written, so the reading comes from that exact key.
    visits = own_rows(root, "visits_clean.csv", VISIT_KEYS, exact=True)
    expected: dict[Key, list[object]] = {}
    for pid, values in SBP_WIDE.items():
        own = ([] if visits is None else
               value_of(visits, (pid, visit), "sbp_mmhg", parse_number, exact=True) or [None])
        expected[(pid,)] = [values[visit], *own]
    return expected


def sbp_wide_visit(visit: str) -> Callable[[Path], Outcome]:
    def check(root: Path) -> Outcome:
        table, grouped = read_keyed(root, "sbp_wide.csv", ("patient_id",))
        return score_column(table, grouped, visit, sbp_expected(root, visit), number_right,
                            f"the {visit} sbp_mmhg from visits_clean; missing when that visit or reading is missing")
    return check


def check_sbp_change(root: Path) -> Outcome:
    table, grouped = read_keyed(root, "sbp_wide.csv", ("patient_id",))
    table.require("sbp_change")
    expected: dict[Key, list[object]] = {}
    for pid, values in SBP_WIDE.items():
        reference = (None if None in values.values() else values["week12"] - values["baseline"])
        accepted: list[object] = [reference]
        for row in grouped.get(normal_key((pid,)), ()):
            # The change computed from the file's own baseline and week12 columns is right too.
            base, follow = (number(row.get(visit, "")) for visit in VISIT_LABELS)
            accepted.append(None if base is None or follow is None else follow - base)
        expected[(pid,)] = accepted
    return score_column(table, grouped, "sbp_change", expected, number_right,
                        "week12 minus baseline; missing when either is missing")


# Task 5.2: decision_log.csv

def decision_table(root: Path) -> Table:
    table = read_table(root, "decision_log.csv")
    if not table.rows:
        raise ArtifactProblem(f"{table.path} has a header but no decision rows")
    return table


def check_decisions(root: Path) -> Outcome:
    table = decision_table(root)
    table.require("table", "decision")
    outcome = Outcome(total=len(DECISIONS))
    pairs = sorted(((similarity(" | ".join(wanted), " | ".join((row["table"], row["decision"]))), position, index)
                    for position, wanted in enumerate(DECISIONS) for index, row in enumerate(table.rows)),
                   key=lambda pair: (-pair[0], pair[1], pair[2]))
    chosen: dict[int, int] = {}
    for ratio, position, index in pairs:
        if ratio >= SIMILAR and position not in chosen and index not in chosen.values():
            chosen[position] = index
    for position, (table_name, decision) in enumerate(DECISIONS):
        if position in chosen:
            outcome.right += 1
        else:
            outcome.problems.append(f"no row with table '{table_name}' and decision '{decision}' (copy both from "
                                    "Task 3.3's table)")
    return outcome


def check_reasons(root: Path) -> Outcome:
    table = decision_table(root)
    table.require("reason")
    blank = [position for position, row in enumerate(table.rows, start=1) if is_missing(row["reason"], TEXT_MISSING)]
    outcome = Outcome(total=len(table.rows), right=len(table.rows) - len(blank))
    if blank:
        outcome.problems.append(f"{len(blank)} of {len(table.rows)} decision rows have no reason, first row "
                                f"{blank[0]} (write one sentence for every decision)")
    return outcome


T_PATIENTS = "Task 3.1"
T_VISITS = "Task 2.2 (stacking) and Task 3.2 (cleaning)"
T_ANALYSIS = "Task 4.3 (joining) and Task 5.2 (saving)"
T_WIDE = "Task 4.4 (pivoting) and Task 5.2 (saving)"
T_DECISIONS = "Task 3.3 (decisions) and Task 5.2 (saving)"

CHECKS = (
    Check("file_inventory.txt", 5, "Task 1.1", check_file_inventory),
    Check("table_profile.csv", 5, "Task 1.2", check_table_profile),
    Check("issue_audit.csv", 10, "Task 2.3", check_issue_audit),
    Check("patients_clean.csv: rows", 2, T_PATIENTS, patients_rows),
    Check("patients_clean.csv: sex", 2, T_PATIENTS, patients_column("sex", exact_right)),
    Check("patients_clean.csv: birth_year", 2, T_PATIENTS, patients_column("birth_year", number_right)),
    Check("patients_clean.csv: site_code", 2, T_PATIENTS, patients_column("site_code", exact_right)),
    Check("visits_clean.csv: rows", 4, T_VISITS, visits_rows),
    Check("visits_clean.csv: visit_date", 3, T_VISITS, visits_column("visit_date")),
    Check("visits_clean.csv: sbp_mmhg", 3, T_VISITS, visits_column("sbp_mmhg")),
    Check("visits_clean.csv: weight_kg", 2, T_VISITS, visits_column("weight_kg")),
    Check("visits_clean.csv: source_file", 2, T_VISITS, visits_column("source_file")),
    Check("a1c_long.csv: rows", 2, "Task 4.1", a1c_rows),
    Check("a1c_long.csv: a1c_pct", 3, "Task 4.1", a1c_values),
    Check("merge_audit.csv", 7, "Task 4.2", check_merge_audit),
    Check("analysis_visits.csv: rows", 3, T_ANALYSIS, analysis_rows),
    Check("analysis_visits.csv: site_name", 3, T_ANALYSIS, check_analysis_sites),
    Check("analysis_visits.csv: a1c_pct", 3, T_ANALYSIS, check_analysis_a1c),
    Check("analysis_visits.csv: sex and birth_year", 2, T_ANALYSIS, check_analysis_patient),
    Check("sbp_wide.csv: rows", 2, T_WIDE, sbp_wide_rows),
    Check("sbp_wide.csv: baseline", 1, T_WIDE, sbp_wide_visit("baseline")),
    Check("sbp_wide.csv: week12", 1, T_WIDE, sbp_wide_visit("week12")),
    Check("sbp_wide.csv: sbp_change", 2, T_WIDE, check_sbp_change),
    Check("decision_log.csv: decisions", 3, T_DECISIONS, check_decisions),
    Check("decision_log.csv: reason", 1, T_DECISIONS, check_reasons),
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
    tests = [run_check(check, root) for check in CHECKS]
    return {"schema": SCHEMA, "score": sum(test["score"] for test in tests), "max-score": MAX_SCORE,
            "tests": tests}
