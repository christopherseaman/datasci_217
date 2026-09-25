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
column order, a leading unnamed index column, number formatting, and the common
spellings of booleans and missing values never cost points. Each check scores
the lines, rows, or records it can verify on its own, so one wrong value costs
only its own share, and no check depends on another passing.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
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

NUMPY_AGE_SUMMARY = {"count": 6, "min": 0, "max": 52, "sum": 198, "mean": 33.0}
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
CLEANED = {
    record_id: dict(zip(CLEANED_COLUMNS[1:], (
        name, site, status, age, None if visit is None else date.fromisoformat(visit), review,
    ), strict=True))
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
SLASHED_DATE = re.compile(r"^\d{4}/\d{2}/\d{2}(?!\d)")
UNNAMED_INDEX = re.compile(r"^(?:unnamed: ?0|index)?$")
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
            raise ArtifactProblem(f"{self.path} has no {names} column; its header reads `{self.header}`")


# Reading artifacts

def read_text(root: Path, name: str) -> str:
    """The artifact's text; a byte-order mark, UTF-16, or Windows encoding is accepted."""
    path = root / OUTPUT / name
    if path.is_symlink() or not path.is_file():
        raise ArtifactProblem(f"output/{name} is missing; expected it committed in the output/ folder")
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
    return text.strip().strip('"').strip().casefold()


def read_table(root: Path, name: str, index_name: str | None = None) -> Table:
    """Parse a CSV artifact, keyed by column name so column order never matters.

    Blank lines are skipped, whitespace after a row's last field is dropped, and
    a leading unnamed index column (`to_csv` without `index=False`) is ignored.
    With `index_name`, that column is read as `index_name` instead when the
    header lacks it, as when a Series of labeled values is saved with `to_csv`.

    Cells may be separated by commas, semicolons, or tabs: whichever splits the
    header line into the most cells is used, and a tie keeps commas. A file
    separated by semicolons may write decimal commas. Spaces the header line
    puts after every separator, as `", ".join(...)` writes, or before every
    separator, as columns padded to line up do, are part of the separator, so
    the data rows lose them too; a cell's other spaces are its own and count.
    """
    text = read_text(root, name).replace("\r\n", "\n").replace("\r", "\n")
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
        records.append({column: cell for column, cell in zip(header, cells) if column})
    return Table(f"output/{name}", header_text, tuple(column for column in header if column), tuple(records))


# Comparing values

def label(text: str) -> str:
    """A label for comparison: case, spacing, `-` or `_` for a space, and a final full stop are ignored."""
    text = re.sub(r"[-_]", " ", text.casefold())
    text = re.sub(r"\s*,\s*", ",", text)
    return " ".join(text.split()).strip(" .")


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


def calendar_dates(cell: str) -> set[date]:
    """The day an ISO date or timestamp names, as written and, when zoned, in UTC.

    `visit_date` is a calendar day, so either reading of a zoned timestamp is
    accepted rather than guessing which one the student meant. `YYYY/MM/DD`
    reads as `YYYY-MM-DD`.
    """
    text = cell.strip()
    if SLASHED_DATE.match(text):
        text = text[:10].replace("/", "-") + text[10:]
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return set()
    days = {moment.date()}
    if moment.tzinfo is not None:
        days.add(moment.astimezone(timezone.utc).date())
    return days


def shown(value: object) -> str:
    """A value as the feedback quotes it."""
    if value is None:
        return "blank (missing)"
    if isinstance(value, str):
        return "blank" if not value.strip() else f"'{value}'"
    return str(value)


def first_rows_by(table: Table, column: str) -> dict[str, dict[str, str]]:
    """The first row for each label in `column`; a later repeat is ignored."""
    found: dict[str, dict[str, str]] = {}
    for row in table.rows:
        found.setdefault(label(row.get(column, "")), row)
    return found


# Task 1: foundation artifacts

def check_raw_preview(root: Path) -> Outcome:
    text = read_text(root, "raw_preview.txt")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    squeezed = [squeeze(line) for line in lines]
    outcome = Outcome(total=len(PREVIEW_COMMANDS))
    for command, expected in PREVIEW_LINES.items():
        wanted = [squeeze(line) for line in expected]
        labels = [index for index, line in enumerate(lines)
                  if command in line.casefold() and "people_raw.csv" in line.casefold()]
        if any(squeezed[index + 1:index + 1 + len(wanted)] == wanted for index in labels):
            outcome.right += 1
            continue
        if not labels:
            outcome.problems.append(
                f"no `$ {PREVIEW_COMMANDS[command]}` line; expected that label line with the "
                f"{len(expected)} lines the command prints beneath it")
            continue
        under = lines[labels[0] + 1:labels[0] + 1 + len(expected)]
        position = next((index for index, line in enumerate(wanted)
                         if index >= len(under) or squeeze(under[index]) != line), 0)
        found = shown(under[position]) if position < len(under) else "nothing"
        outcome.problems.append(
            f"under `$ {PREVIEW_COMMANDS[command]}`, line {position + 1} of {len(expected)} should be "
            f"'{expected[position]}' as that command prints it, found {found}")
    return outcome


def check_pipeline_summary(root: Path) -> Outcome:
    text = read_text(root, "pipeline_summary.txt")
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = KEY_VALUE.match(line)
        if match:
            key = re.sub(r"[\s-]+", "_", match.group(1).strip().casefold())
            found.setdefault(key, match.group(2))
    outcome = Outcome(total=len(PIPELINE_SUMMARY))
    for key, expected in PIPELINE_SUMMARY.items():
        if key not in found:
            outcome.problems.append(f"no `{key}=` line; expected `{key}={expected}`")
        elif leading_number(found[key]) == expected:
            outcome.right += 1
        else:
            outcome.problems.append(f"`{key}` should be {expected}, found {shown(found[key])}")
    return outcome


def check_numpy_age_summary(root: Path) -> Outcome:
    table = read_table(root, "numpy_age_summary.csv", index_name="metric")
    table.require("metric", "value")
    rows = first_rows_by(table, "metric")
    outcome = Outcome(total=len(NUMPY_AGE_SUMMARY))
    for metric, expected in NUMPY_AGE_SUMMARY.items():
        row = rows.get(metric)
        tolerance = MEAN_TOLERANCE if metric == "mean" else 1e-9
        if row is None:
            outcome.problems.append(f"no `{metric}` row; expected `{metric},{expected}`")
        elif (value := number(row["value"])) is not None and abs(value - expected) <= tolerance:
            outcome.right += 1
        else:
            outcome.problems.append(f"`{metric}` should be {expected}, found {shown(row['value'])}")
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
            + (f", so without {' and '.join(missing)} no row can be checked" if missing else ""))
    if missing:
        return outcome
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
        site, status = PANDAS_SELECTION[record_id]
        if label(row["site"]) == label(site) and label(row["status"]) == label(status):
            matched += 1
        else:
            outcome.problems.append(
                f"{record_id} should have site '{site}' and status '{status}' as the raw file has them, "
                f"found {shown(row['site'])} and {shown(row['status'])}")
    for key, record_id in wanted.items():
        if key not in seen:
            outcome.problems.append(f"no row for {record_id}")
    if extra_rows:
        outcome.problems.append(
            f"the selection should hold only {', '.join(PANDAS_SELECTION)}, but {extra_rows} other "
            f"row(s) are in the file; each one cancels a matched row")
    outcome.right += max(0, matched - extra_rows)
    return outcome


# Task 2: the audit

def check_issue_audit(root: Path) -> Outcome:
    table = read_table(root, "issue_audit.csv", index_name="issue")
    table.require("issue", "count")
    rows = first_rows_by(table, "issue")
    outcome = Outcome(total=len(ISSUE_AUDIT))
    for issue, expected in ISSUE_AUDIT:
        row = rows.get(label(issue))
        if row is None:
            outcome.problems.append(f"no row for the issue '{issue}'")
        elif number(row["count"]) == expected:
            outcome.right += 1
        else:
            outcome.problems.append(f"'{issue}' should count {expected}, found {shown(row['count'])}")
    return outcome


# Tasks 3 and 4: the cleaned table

CLEANED_TASK = "Task 3.3 (cleaning) and Task 4.2 (saving)"


def cleaned_records(root: Path) -> tuple[Table, dict[str, list[dict[str, str]]]]:
    """The cleaned rows keyed by record_id.

    Without a record_id column, a table with one row per record is matched in
    source order, so the other columns are still graded; the record_id check
    charges the missing column once.
    """
    table = read_table(root, "cleaned_people.csv")
    if "record_id" not in table.columns and len(table.rows) == len(CLEANED):
        return table, {label(record_id): [row] for record_id, row in zip(CLEANED, table.rows)}
    table.require("record_id")
    records: dict[str, list[dict[str, str]]] = {}
    for row in table.rows:
        records.setdefault(label(row["record_id"]), []).append(row)
    return table, records


def check_cleaned_record_ids(root: Path) -> Outcome:
    read_table(root, "cleaned_people.csv").require("record_id")
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


def cleaned_value_is_right(column: str, expected: object, cell: str, row: dict[str, str]) -> bool:
    if column in ("full_name", "site", "status"):
        return is_missing(cell, TEXT_MISSING) if expected is None else cell == expected
    if expected is None:
        return is_missing(cell)
    if column == "age":
        return number(cell) == expected
    if column == "visit_date":
        return expected in calendar_dates(cell)
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
        outcome = Outcome(total=len(CLEANED))
        wrong: list[str] = []
        first = ""
        for record_id, expected in CLEANED.items():
            rows = records.get(label(record_id))
            if not rows:
                wrong.append(record_id)
                first = first or f"first {record_id}: the record is missing from the file"
                continue
            bad = next((row for row in rows if not cleaned_value_is_right(column, expected[column],
                                                                          row[column], row)), None)
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
                f"column {column}: {len(wrong)} of {len(CLEANED)} records differ; {first}{others}")
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
    recorded = {(label(row["field"]), label(row["issue"])): row for row in reversed(table.rows)}
    for field_name, issue, action in DECISIONS:
        row = recorded.get((label(field_name), label(issue)))
        if row is None:
            outcome.problems.append(f"no row with field '{field_name}' and issue '{issue}'")
        elif label(row["action"]) == label(action):
            outcome.right += 1
        else:
            outcome.problems.append(
                f"the '{issue}' decision's action should be '{action}', found {shown(row['action'])}")
    return outcome


def same_source(cell: str) -> bool:
    path = cell.strip().replace("\\", "/").casefold()
    path = path.removeprefix("./")
    return path == SOURCE or path.endswith("/" + SOURCE)


def cleaned_row_count(root: Path) -> int | None:
    try:
        return len(read_table(root, "cleaned_people.csv").rows)
    except ArtifactProblem:
        return None


def provenance_check(column: str, describe: str,
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
                f"rows differ, first {which} has {shown(row[column])}")
        return outcome
    return check


def _reason(root: Path) -> Callable[[str], bool]:
    return lambda cell: not is_missing(cell, TEXT_MISSING)


def _source(root: Path) -> Callable[[str], bool]:
    return same_source


def _sha(root: Path) -> Callable[[str], bool]:
    return lambda cell: cell.strip().casefold() == SOURCE_SHA256


def _rows_before(root: Path) -> Callable[[str], bool]:
    return lambda cell: number(cell) == RAW_ROWS


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
    Check("cleaned_people.csv: record_id", 5, CLEANED_TASK, check_cleaned_record_ids),
    *(Check(f"cleaned_people.csv: {column}", 5, CLEANED_TASK, cleaned_column_check(column))
      for column in CLEANED_COLUMNS[1:]),
    Check("decision_log.csv: decisions", 8, "Task 3.1", check_decisions),
    Check("decision_log.csv: reason", 2, "Task 3.1",
          provenance_check("reason", "a written reason", _reason)),
    Check("decision_log.csv: source", 1, "Task 4.2",
          provenance_check("source", f"'{SOURCE}'", _source)),
    Check("decision_log.csv: source_sha256", 2, "Task 4.2",
          provenance_check("source_sha256", "the sha256 value from data/fixture.json", _sha)),
    Check("decision_log.csv: rows_before", 2, "Task 4.2",
          provenance_check("rows_before", f"the raw table's row count, {RAW_ROWS}", _rows_before)),
    Check("decision_log.csv: rows_after", 2, "Task 4.2",
          provenance_check("rows_after", f"the cleaned table's row count, {CLEAN_ROWS}", _rows_after)),
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
    return {"test-name": check.name, "passed": passed, "score": score, "max-score": check.points,
            "detail": detail}


def grade_submission(submission_dir: str | Path) -> dict:
    """Grade the committed artifacts in `submission_dir` without running any submitted code."""
    root = Path(submission_dir).resolve()
    tests = [run_check(check, root) for check in CHECKS]
    return {"schema": SCHEMA, "score": sum(test["score"] for test in tests), "max-score": MAX_SCORE,
            "tests": tests}
