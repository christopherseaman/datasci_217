"""Checks for Assignment 08.

The course keeps these checks in 08/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They read the five CSV files a
submission saves in output/ and compare them with values computed from the
supplied visit log, copied below. The self-test confirms that copy matches
08/assignment/data/clinic_visits.csv.

Each check scores one thing, so one mistake costs only its own points, and no
check waits for another to pass. Values are compared after parsing: spacing,
line endings, quoting, a byte-order mark, column order, row order, a leading
row-number column, number formatting (2 == 2.0 == 2.00), and the letter case
of labels and headers never cost points, and cells may be separated by
commas, semicolons, or tabs. A row for Excelsior, the clinic with
no visits, is accepted when it shows zero visits, as observed=False writes it.
A pivot saved with its rows and columns swapped is read the right way round,
so the swap costs only its columns check.

Nothing here imports, runs, or inspects student code.
"""

from __future__ import annotations

import codecs
import csv
import difflib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


# data/clinic_visits.csv: one row per visit. wait_min is minutes from check-in to
# being seen; satisfaction is the 1 to 5 survey score, None where none came back.
VISIT_COLUMNS = ("visit_id", "clinic", "patient_id", "visit_type", "wait_min", "satisfaction")
VISITS = {
    "V001": {"clinic": "Mission", "patient_id": "P101", "visit_type": "New", "wait_min": 18, "satisfaction": 4},
    "V002": {"clinic": "Mission", "patient_id": "P102", "visit_type": "Follow-up", "wait_min": 12, "satisfaction": 5},
    "V003": {"clinic": "Mission", "patient_id": "P101", "visit_type": "Follow-up", "wait_min": 9, "satisfaction": None},
    "V004": {"clinic": "Mission", "patient_id": "P103", "visit_type": "Telehealth", "wait_min": 5, "satisfaction": 4},
    "V005": {"clinic": "Mission", "patient_id": "P104", "visit_type": "New", "wait_min": 26, "satisfaction": 3},
    "V006": {"clinic": "Mission", "patient_id": "P102", "visit_type": "Follow-up", "wait_min": 16, "satisfaction": 4},
    "V007": {"clinic": "Sunset", "patient_id": "P105", "visit_type": "New", "wait_min": 32, "satisfaction": 3},
    "V008": {"clinic": "Sunset", "patient_id": "P106", "visit_type": "Follow-up", "wait_min": 21, "satisfaction": None},
    "V009": {"clinic": "Sunset", "patient_id": "P105", "visit_type": "Follow-up", "wait_min": 17, "satisfaction": 4},
    "V010": {"clinic": "Sunset", "patient_id": "P107", "visit_type": "New", "wait_min": 41, "satisfaction": 2},
    "V011": {"clinic": "Sunset", "patient_id": "P106", "visit_type": "Follow-up", "wait_min": 15, "satisfaction": None},
    "V012": {"clinic": "Bayview", "patient_id": "P108", "visit_type": "Telehealth", "wait_min": 6, "satisfaction": 5},
    "V013": {"clinic": "Bayview", "patient_id": "P109", "visit_type": "New", "wait_min": 28, "satisfaction": 3},
    "V014": {"clinic": "Bayview", "patient_id": "P110", "visit_type": "Follow-up", "wait_min": 13, "satisfaction": 4},
    "V015": {"clinic": "Bayview", "patient_id": "P108", "visit_type": "Telehealth", "wait_min": 8, "satisfaction": 5},
}
# Task 1.1's reporting order. Excelsior opened this month and has no visits yet.
CLINIC_ORDER = ("Mission", "Sunset", "Bayview", "Excelsior")
VISIT_TYPES = ("Follow-up", "New", "Telehealth")

# Means may be saved with every digit or rounded to one or two decimals.
TOLERANCE = 0.06
# Ways pandas and people write an empty cell.
MISSING_SPELLINGS = frozenset({"", "nan", "na", "n/a", "none", "null", "<na>", "nat"})


@dataclass(frozen=True)
class Artifact:
    """One saved CSV: where it lives, which task writes it, and what it should hold.

    `rows` maps each required key, a tuple of the key columns' values, to the
    expected value of every other column; None means an empty cell. `optional`
    holds rows that may appear but need not, checked like the others when they do.
    `across` names the pivot's columns= key for a pivot table, whose value
    columns are that key's labels.
    """

    path: str
    task: str
    columns: tuple[str, ...]
    key: tuple[str, ...]
    rows: dict[tuple[str, ...], dict[str, object]]
    optional: dict[tuple[str, ...], dict[str, object]] = field(default_factory=dict)
    across: str = ""

    def expected(self, key: tuple[str, ...]) -> dict[str, object]:
        return self.rows[key] if key in self.rows else self.optional[key]

    def labels(self) -> set[str]:
        """The casefolded values of a one-column key, such as the clinic names."""
        return {key[0].casefold() for key in (*self.rows, *self.optional)} if len(self.key) == 1 else set()


@dataclass(frozen=True)
class Table:
    """A saved CSV: its casefolded column names and one {column: cell} dict per data row.

    `written` maps each casefolded column name to the header as the file wrote
    it, for messages. `transposed` is True when a pivot was saved with its rows
    and columns swapped and has been read the right way round.
    """

    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    written: dict[str, str] = field(default_factory=dict)
    transposed: bool = False
    conflicting_columns: tuple[str, ...] = ()

    def header(self, columns=None) -> str:
        """The named columns (all of them by default) as the file wrote them: 'Mission, Sunset and Total'."""
        shown = [self.written.get(column, column) or "a column with no header"
                 for column in (self.columns if columns is None else columns)]
        return _join(shown) or "none"


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


def _visits_at(clinic: str, visit_type: str | None = None) -> list[dict[str, object]]:
    return [
        visit for visit in VISITS.values()
        if visit["clinic"] == clinic and visit_type in (None, visit["visit_type"])
    ]


def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def _counts(visits: list[dict[str, object]]) -> dict[str, object]:
    return {
        "visit_count": len(visits),
        "satisfaction_count": sum(visit["satisfaction"] is not None for visit in visits),
        "patient_count": len({visit["patient_id"] for visit in visits}),
    }


def _summary(visits: list[dict[str, object]]) -> dict[str, object]:
    waits = [visit["wait_min"] for visit in visits]
    return {**_counts(visits), "total_wait_min": sum(waits), "mean_wait_min": _mean(waits)}


def _pair(visits: list[dict[str, object]]) -> dict[str, object]:
    return {"visit_count": len(visits), "mean_wait_min": _mean([visit["wait_min"] for visit in visits])}


def _pivot_row(clinic: str) -> dict[str, object]:
    return {visit_type: _mean([visit["wait_min"] for visit in _visits_at(clinic, visit_type)]) for visit_type in VISIT_TYPES}


def _context_rows() -> dict[tuple[str, ...], dict[str, object]]:
    rows = {}
    for visit_id, visit in VISITS.items():
        clinic_mean = _mean([other["wait_min"] for other in _visits_at(str(visit["clinic"]))])
        rows[(visit_id,)] = {**visit, "clinic_mean_wait": clinic_mean, "wait_vs_clinic": visit["wait_min"] - clinic_mean}
    return rows


OBSERVED = tuple(clinic for clinic in CLINIC_ORDER if _visits_at(clinic))
UNUSED = tuple(clinic for clinic in CLINIC_ORDER if clinic not in OBSERVED)
PAIRS = tuple((clinic, visit_type) for clinic in CLINIC_ORDER for visit_type in VISIT_TYPES)

CLINIC_COUNTS = Artifact(
    path="output/clinic_counts.csv",
    task="Task 1.2",
    columns=("clinic", "visit_count", "satisfaction_count", "patient_count"),
    key=("clinic",),
    rows={(clinic,): _counts(_visits_at(clinic)) for clinic in OBSERVED},
    optional={(clinic,): _counts([]) for clinic in UNUSED},
)

CLINIC_SUMMARY = Artifact(
    path="output/clinic_summary.csv",
    task="Task 2.1",
    columns=("clinic", "visit_count", "satisfaction_count", "patient_count", "total_wait_min", "mean_wait_min"),
    key=("clinic",),
    rows={(clinic,): _summary(_visits_at(clinic)) for clinic in OBSERVED},
    optional={(clinic,): _summary([]) for clinic in UNUSED},
)

VISIT_CONTEXT = Artifact(
    path="output/visits_with_context.csv",
    task="Task 2.2",
    columns=(*VISIT_COLUMNS, "clinic_mean_wait", "wait_vs_clinic"),
    key=("visit_id",),
    rows=_context_rows(),
)

CLINIC_VISIT_TYPE = Artifact(
    path="output/clinic_visit_type_summary.csv",
    task="Task 2.3",
    columns=("clinic", "visit_type", "visit_count", "mean_wait_min"),
    key=("clinic", "visit_type"),
    rows={pair: _pair(_visits_at(*pair)) for pair in PAIRS if _visits_at(*pair)},
    # observed=False also lists the pairs with no visits: a count of 0 and no mean.
    optional={pair: _pair([]) for pair in PAIRS if not _visits_at(*pair)},
)

MEAN_WAIT_PIVOT = Artifact(
    path="output/mean_wait_pivot.csv",
    task="Task 3.1",
    columns=("clinic", *VISIT_TYPES),
    key=("clinic",),
    rows={(clinic,): _pivot_row(clinic) for clinic in OBSERVED},
    optional={(clinic,): _pivot_row(clinic) for clinic in UNUSED},
    across="visit_type",
)

ARTIFACTS = (CLINIC_COUNTS, CLINIC_SUMMARY, VISIT_CONTEXT, CLINIC_VISIT_TYPE, MEAN_WAIT_PIVOT)


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


def _number(cell: str) -> float | None:
    try:
        return float(_clean(cell).replace(",", ""))
    except ValueError:
        return None


def _is_whole_number(cell: str) -> bool:
    number = _number(cell)
    return number is not None and number.is_integer()


def _matches(cell: str, expected: object) -> bool:
    text = _clean(cell)
    if expected is None:
        return text.casefold() in MISSING_SPELLINGS
    if isinstance(expected, (int, float)):
        number = _number(text)
        return number is not None and abs(number - expected) <= TOLERANCE
    return text.casefold() == str(expected).casefold()


def _show(value: object) -> str:
    if value is None:
        return "blank"
    if isinstance(value, float):
        return f"{round(value, 2):g}"
    return str(value)


def _given(cell: str) -> str:
    text = _clean(cell)
    return text if text else "blank"


def _key_name(key: tuple[str, ...]) -> str:
    return " ".join(key)


def _expected_keys(names: list[str]) -> str:
    """The expected row keys, shortened past six: 'V001, V002, V003, ... and V015 (15 in all)'."""
    if len(names) <= 6:
        return _join(names)
    return f"{', '.join(names[:3])}, ... and {names[-1]} ({len(names)} in all)"


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
    if "\x00" in text:
        raise csv.Error("file contains a NUL byte instead of CSV text")
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
    """Say the artifact is missing, naming any saved file whose name looks like a typo of its name.

    Another artifact's file is never suggested, however alike the names:
    clinic_summary.csv and clinic_visit_type_summary.csv are both required.
    """
    message = f"{artifact.path} is missing; run the {artifact.task} cell to write it, then commit it."
    wanted = root / artifact.path
    required = {Path(other.path).name.casefold() for other in ARTIFACTS}
    if wanted.parent.is_dir():
        look_alikes = sorted(
            path.relative_to(root).as_posix()
            for path in wanted.parent.iterdir()
            if path.is_file()
            and path.name.casefold() not in required
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
    try:
        lines = _csv_rows(_decode(path.read_bytes()))
    except csv.Error as error:
        raise AssertionError(
            f"{artifact.path} cannot be read as a CSV table ({error}); run the {artifact.task} cell again so "
            "to_csv() writes it, then compare it with the checkpoint in README.md."
        ) from None
    _assert(lines, f"{artifact.path} is empty; run the {artifact.task} cell again to write it.")
    written = [_clean(cell) for cell in lines[0]]
    body = [[_clean(cell) for cell in row] for row in lines[1:]]
    width = max(len(written), *(len(row) for row in body)) if body else len(written)
    written += [""] * (width - len(written))
    body = [row + [""] * (width - len(row)) for row in body]

    header = [cell.casefold() for cell in written]
    while len(header) > 1 and _is_index_header(header[0]) and body and all(_is_whole_number(row[0]) for row in body):
        written, header = written[1:], header[1:]
        body = [row[1:] for row in body]
    keep = [i for i, column in enumerate(header) if column or any(row[i] for row in body)]
    written, header = [written[i] for i in keep], [header[i] for i in keep]
    body = [[row[i] for i in keep] for row in body]

    if _is_transposed(artifact, header, body):
        key = _fold(artifact.key[0])
        down = [row[0] for row in body]
        rows = tuple(
            {key: label, **{name.casefold(): row[i] for name, row in zip(down, body)}}
            for i, label in enumerate(written[1:], start=1)
        )
        return Table(
            columns=(key, *(name.casefold() for name in down)),
            rows=rows,
            written={key: artifact.key[0], **{name.casefold(): name for name in down}},
            transposed=True,
        )
    rows = []
    conflicting = set()
    for row in body:
        values = {}
        for i, column in enumerate(header):
            cell = row[i] if i < len(row) else ""
            if values.get(column) and cell:
                first_number, next_number = _number(values[column]), _number(cell)
                equal = (
                    abs(first_number - next_number) <= TOLERANCE
                    if first_number is not None and next_number is not None
                    else values[column].casefold() == cell.casefold()
                )
                if not equal:
                    conflicting.add(column)
            if not values.get(column):
                values[column] = cell
        rows.append(values)
    rows = tuple(rows)
    names: dict[str, str] = {}
    for column, as_written in zip(header, written):
        names.setdefault(column, as_written)
    header = list(dict.fromkeys(header))
    return Table(columns=tuple(header), rows=rows, written=names, conflicting_columns=tuple(sorted(conflicting)))


def _is_transposed(artifact: Artifact, header: list[str], body: list[list[str]]) -> bool:
    """True when a pivot's key labels head its columns and its column labels run down the first column.

    That is what swapping index= and columns= saves: the clinics across the
    top and the visit types down the side.
    """
    if not artifact.across or len(header) < 2 or not body:
        return False
    across = {_fold(column) for column in artifact.columns if column not in artifact.key}
    return all(column in artifact.labels() for column in header[1:]) and all(
        row[0].casefold() in across for row in body
    )


def _resolve(table: Table, artifact: Artifact) -> dict[str, str]:
    """Map each expected column to the saved column that holds it.

    A column saved under its own name, in any letter case, is found by name. A
    key column saved under another name, such as a blank header over saved
    index labels, is found by its values. If exactly one expected value column
    is still unplaced and exactly one unexpected column is left, that column is
    taken to hold it. A misnamed column then costs only the columns check.
    """
    expected = [_fold(column) for column in artifact.columns]
    found = {column: _fold(column) for column in artifact.columns if _fold(column) in table.columns}
    spare = [column for column in table.columns if column not in expected]
    for position, column in enumerate(artifact.key):
        if column in found:
            continue
        wanted = {key[position].casefold() for key in (*artifact.rows, *artifact.optional)}
        candidates = []
        for other in spare:
            values = [row[other].casefold() for row in table.rows if row[other]]
            if values and sum(value in wanted for value in values) * 2 > len(values):
                candidates.append(other)
        if len(candidates) == 1:
            found[column] = candidates[0]
            spare.remove(candidates[0])
    # A key column is only ever found by name or by its values, never by elimination.
    unplaced = [column for column in artifact.columns if column not in found and column not in artifact.key]
    if len(unplaced) == 1 and len(spare) == 1:
        found[unplaced[0]] = spare[0]
    return found


def _describe(row: dict[str, str], columns: list[str]) -> str:
    return "a row reading " + ", ".join(_given(row[column]) for column in columns)


def _group(table: Table, artifact: Artifact, found: dict[str, str]):
    """({expected key: [rows]}, [rows naming no expected key]) for the saved table.

    When every key column is saved, each row is matched by its key. When one is
    not, as when the clinic names stayed in the index and were dropped by
    `index=False`, each row is matched by the values it does hold, so the lost
    column costs the columns check alone.
    """
    everyone = {**artifact.rows, **artifact.optional}
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    unknown = []
    if all(column in found for column in artifact.key):
        canonical = {tuple(part.casefold() for part in key): key for key in everyone}
        for row in table.rows:
            cells = tuple(row[found[column]].casefold() for column in artifact.key)
            key = canonical.get(cells)
            if key is None:
                unknown.append("a row for " + " ".join(row[found[column]] or "blank" for column in artifact.key))
            else:
                grouped.setdefault(key, []).append(row)
        return grouped, unknown

    values = [column for column in artifact.columns if column not in artifact.key and column in found]
    absent = [column for column in artifact.key if column not in found]
    _assert(
        values,
        f"{artifact.path} has no {_join(absent)} column (its columns are {table.header()}), "
        f"so its rows cannot be matched; {artifact.task} saves the header line {','.join(artifact.columns)}."
        + _swap_note(artifact, table.columns),
    )
    for row in table.rows:
        candidates = [
            key
            for key, expected in everyone.items()
            if all(
                row[found[column]].casefold() == key[position].casefold()
                for position, column in enumerate(artifact.key)
                if column in found
            )
            and all(_matches(row[found[column]], expected[column]) for column in values)
        ]
        if len(candidates) == 1:
            grouped.setdefault(candidates[0], []).append(row)
        elif not candidates or any(key in artifact.rows for key in candidates):
            unknown.append(_describe(row, [found[column] for column in artifact.columns if column in found]))
    return grouped, unknown


# The labels margins=True gives its total row and column: its default, and the lecture's margins_name.
MARGIN_NAMES = frozenset({"all", "total"})


def _swapped(artifact: Artifact) -> str:
    key = artifact.key[0]
    return f'index= and columns= are swapped: use index="{key}" and columns="{artifact.across}".'


def _swap_note(artifact: Artifact, columns) -> str:
    """For a pivot whose header holds key labels, such as clinic names: say index= and columns= are swapped."""
    if not artifact.across or not any(column in artifact.labels() for column in columns):
        return ""
    return f" Its header holds {artifact.key[0]} labels, so {_swapped(artifact)}"


def _margins_note(artifact: Artifact, labels) -> str:
    """For a pivot with a Total or All row or column: say it comes from margins=True."""
    totals = sorted({label for label in labels if label.casefold() in MARGIN_NAMES})
    if not artifact.across or not totals:
        return ""
    return f" {_join(totals)} comes from margins=True; leave margins out of {artifact.task}."


def columns_check(artifact: Artifact, hint: str) -> Callable[[Path], None]:
    """The saved header holds exactly the expected columns, in any order."""

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        _assert(
            not table.transposed,
            f"{artifact.path} has the {artifact.key[0]} labels across the top and the {artifact.across} labels "
            f"down the side, so {_swapped(artifact)} {artifact.task} saves the header line "
            f"{','.join(artifact.columns)}.",
        )
        found = _resolve(table, artifact)
        # A saved index with no name leaves its column unheaded; holding the IDs, it is the key column.
        first = table.columns[0] if table.columns else None
        unheaded = {
            column: found[column]
            for column in artifact.key
            if _fold(column) not in table.columns and first is not None and found.get(column) == first
            and _is_index_header(first)
        }
        expected = [_fold(column) for column in artifact.columns]
        missing = [
            column for column in artifact.columns
            if _fold(column) not in table.columns and column not in unheaded
        ]
        extra = [
            column
            for column in table.columns
            if column not in expected and column not in unheaded.values()
        ]
        problems = []
        if missing:
            problems.append(f"is missing {_join(missing)}")
        if extra:
            problems.append(f"also has {table.header(extra)}")
        notes = _swap_note(artifact, extra) or _margins_note(artifact, [table.written.get(c, c) for c in extra])
        _assert(
            not problems,
            f"{artifact.path} " + " and ".join(problems) + f"; {artifact.task} saves the header line "
            f"{','.join(artifact.columns)} (any column order).{notes} {hint}",
        )

    return check


def rows_check(artifact: Artifact, hint: str) -> Callable[[Path], None]:
    """Each required key appears on exactly one row, and no unexpected key appears."""

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact)
        _assert(
            not any(found.get(column) in table.conflicting_columns for column in artifact.key),
            f"{artifact.path} has repeated key headers with conflicting labels; expected one key value per "
            f"row. Remove the extra key column and save {artifact.task}'s table again.",
        )
        grouped, unknown = _group(table, artifact, found)
        first_key = found.get(artifact.key[0])
        notes = _margins_note(artifact, [row[first_key] for row in table.rows] if first_key else [])
        missing = [key for key in artifact.rows if key not in grouped]
        repeated = [key for key, rows in grouped.items() if len(rows) > 1]
        problems = []
        if missing:
            problems.append(f"is missing {_join(_key_name(key) for key in missing)}")
        if repeated:
            problems.append("lists " + _join(f"{_key_name(key)} {len(grouped[key])} times" for key in repeated))
        if unknown:
            shown = _join(unknown[:4]) + (f" and {len(unknown) - 4} more" if len(unknown) > 4 else "")
            problems.append(f"also has {shown}")
        _assert(
            not problems,
            f"{artifact.path} " + "; ".join(problems) + "; it should hold one row for each of "
            f"{_expected_keys([_key_name(key) for key in artifact.rows])}.{notes} Fix it in {artifact.task}: {hint}",
        )

    return check


def values_check(
    artifact: Artifact,
    columns: tuple[str, ...],
    hint: str,
    cells: str = "all",
    also: Callable[[dict[str, str], dict[str, str]], tuple[float, str] | None] | None = None,
    cause: Callable[[Table, dict[str, str]], str] | None = None,
    prior: Artifact | None = None,
) -> Callable[[Path], None]:
    """Every saved row for an expected key holds the expected values in `columns`.

    `cells` narrows the comparison to the cells expected to hold a value
    ("filled") or to be empty ("empty"). A missing or repeated row costs the
    rows check, not this one: this check compares whatever rows the file does
    hold for the expected keys; absent rows are charged only by the rows check.

    `also(row, found)`, for a column worked out from other saved columns, gives
    the value the row's own saved cells imply and how, or None. A cell that
    matches it passes too, so a mistake in the other column costs only that
    column's check. `prior` accepts values copied from the named earlier artifact,
    so that an earlier mistake costs points only once. `cause(table, found)`
    names a likely cause, or returns "".
    """

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact)
        # Schema charges omitted columns; judge the present values without skipping sibling mistakes.
        present = tuple(column for column in columns if column in found)
        conflicts = [column for column in present if found.get(column) in table.conflicting_columns]
        _assert(
            not conflicts,
            f"{artifact.path} has repeated {_join(conflicts)} headers with conflicting values; expected one "
            f"value per column on each row. Remove the extra columns and save {artifact.task}'s table again.",
        )
        grouped, unknown = _group(table, artifact, found)
        held = [f"a row for {_key_name(key)}" for key in grouped] + unknown
        if len(held) > 3:
            holds = ", ".join(held[:3]) + f" and {len(held) - 3} more"
        else:
            holds = _join(held) or "no data rows"
        _assert(
            any(key in artifact.rows for key in grouped),
            f"{artifact.path} has no row for any of {_join(_key_name(key) for key in artifact.rows)} (it holds "
            f"{holds}), so its {_join(columns)} values cannot be compared; the rows check says what to fix in "
            f"{artifact.task}.",
        )
        previous = {}
        if prior is not None:
            try:
                earlier = read_table(root, prior)
                earlier_found = _resolve(earlier, prior)
                previous, _ = _group(earlier, prior, earlier_found)
                previous = {tuple(part.casefold() for part in key): rows for key, rows in previous.items()}
            except AssertionError:
                earlier_found = {}
        wrong = []
        for key, rows in grouped.items():
            expected_row = artifact.expected(key)
            for column in present:
                expected = expected_row[column]
                if (cells == "filled" and expected is None) or (cells == "empty" and expected is not None):
                    continue
                given, implied = set(), set()
                for row in rows:
                    cell = row[found[column]]
                    derived = also(row, found) if also else None
                    if prior is not None:
                        prior_key = tuple(row.get(found.get(name, ""), "").casefold() for name in prior.key)
                        prior_column = "mean_wait_min" if column == "clinic_mean_wait" or artifact is MEAN_WAIT_PIVOT else column
                        if artifact is MEAN_WAIT_PIVOT:
                            prior_key = (key[0].casefold(), column.casefold())
                        saved = previous.get(prior_key, [])
                        if len(saved) == 1 and prior_column in earlier_found:
                            if _matches(cell, _number(saved[0][earlier_found[prior_column]])):
                                continue
                    if also and derived is None and "clinic_mean_wait" in found and _number(row[found["clinic_mean_wait"]]) is None and _matches(cell, None):
                        continue
                    if _matches(cell, expected) or (derived is not None and _matches(cell, derived[0])):
                        continue
                    given.add(_given(cell))
                    if derived is not None and not _matches(_show(derived[0]), expected):
                        implied.add(f"{_show(derived[0])}, {derived[1]}")
                if given:
                    wrong.append(
                        f"{_key_name(key)} has {column} {_join(sorted(given))}, expected {_show(expected)}"
                        + (f" (or {_join(sorted(implied))})" if implied else "")
                    )
        note = cause(table, found) if cause else ""
        _assert(
            not wrong,
            f"{artifact.path}: " + _listed(wrong) + "." + (f" {note}" if note else "")
            + f" Fix it in {artifact.task}: {hint}",
        )

    return check


def _finite(cell: str) -> float | None:
    number = _number(cell)
    return number if number is not None and math.isfinite(number) else None


def _saved_wait_vs_clinic(row: dict[str, str], found: dict[str, str]) -> tuple[float, str] | None:
    """wait_min minus clinic_mean_wait from the row's own saved cells, or None when either is not a number."""
    wait, mean = (
        _finite(row[found[column]]) if column in found else None for column in ("wait_min", "clinic_mean_wait")
    )
    if wait is None or mean is None:
        return None
    return wait - mean, "wait_min minus the clinic_mean_wait saved on that row"


def _when_clinic_means_blank(note: str) -> Callable[[Table, dict[str, str]], str]:
    """A cause that applies when every saved clinic_mean_wait is empty, as assigning a .mean() summary leaves it."""

    def cause(table: Table, found: dict[str, str]) -> str:
        column = found.get("clinic_mean_wait")
        cells = [row[column] for row in table.rows] if column else []
        return note if cells and all(_clean(cell).casefold() in MISSING_SPELLINGS for cell in cells) else ""

    return cause


COUNT_HINT = (
    "size counts every visit, satisfaction's count skips the blank scores, and patient_id's nunique counts "
    "each patient once."
)

CHECKS = (
    # Task 1.2: output/clinic_counts.csv
    Check(
        "clinic counts: columns",
        columns_check(
            CLINIC_COUNTS,
            "Group with as_index=False so clinic stays a column, and name each count in .agg(): "
            'visit_count=("visit_id", "size") and so on.',
        ),
    ),
    Check(
        "clinic counts: one row per clinic",
        rows_check(CLINIC_COUNTS, 'Group visits by "clinic": one row per clinic that has visits.'),
    ),
    Check(
        "clinic counts: visit_count values",
        values_check(CLINIC_COUNTS, ("visit_count",), 'visit_count is ("visit_id", "size"): every visit at the clinic.'),
    ),
    Check(
        "clinic counts: satisfaction_count values",
        values_check(
            CLINIC_COUNTS,
            ("satisfaction_count",),
            'satisfaction_count is ("satisfaction", "count"), which skips the visits with no survey score.',
        ),
    ),
    Check(
        "clinic counts: patient_count values",
        values_check(
            CLINIC_COUNTS,
            ("patient_count",),
            'patient_count is ("patient_id", "nunique"): a patient seen twice at a clinic counts once.',
        ),
    ),
    # Task 2.1: output/clinic_summary.csv
    Check(
        "clinic summary: columns",
        columns_check(
            CLINIC_SUMMARY,
            "Keep Task 1.2's three counts and add total_wait_min and mean_wait_min, with as_index=False.",
        ),
    ),
    Check(
        "clinic summary: one row per clinic",
        rows_check(CLINIC_SUMMARY, 'Group visits by "clinic": one row per clinic that has visits.'),
    ),
    Check(
        "clinic summary: count values",
        values_check(CLINIC_SUMMARY, ("visit_count", "satisfaction_count", "patient_count"), COUNT_HINT, prior=CLINIC_COUNTS),
    ),
    Check(
        "clinic summary: total_wait_min values",
        values_check(CLINIC_SUMMARY, ("total_wait_min",), 'total_wait_min is ("wait_min", "sum").'),
    ),
    Check(
        "clinic summary: mean_wait_min values",
        values_check(
            CLINIC_SUMMARY,
            ("mean_wait_min",),
            'mean_wait_min is ("wait_min", "mean"); save it unrounded or to at least one decimal.',
        ),
    ),
    # Task 2.2: output/visits_with_context.csv
    Check(
        "visit context: columns",
        columns_check(
            VISIT_CONTEXT,
            "Start from a .copy() of visits, so every original column stays, and add clinic_mean_wait and "
            "wait_vs_clinic.",
        ),
    ),
    Check(
        "visit context: one row per visit",
        rows_check(
            VISIT_CONTEXT,
            "transform keeps one row per visit, V001 to V015; a reducing summary such as .mean() does not.",
        ),
    ),
    Check(
        "visit context: original visit values",
        values_check(
            VISIT_CONTEXT,
            VISIT_COLUMNS[1:],
            "The copy keeps each visit's own values from data/clinic_visits.csv unchanged.",
        ),
    ),
    Check(
        "visit context: clinic_mean_wait values",
        values_check(
            VISIT_CONTEXT,
            ("clinic_mean_wait",),
            'clinic_mean_wait is groupby("clinic")["wait_min"].transform("mean"): every visit gets its own '
            "clinic's mean wait.",
            prior=CLINIC_SUMMARY,
            cause=_when_clinic_means_blank(
                "Every clinic_mean_wait is blank, as happens when the column is assigned from .mean(): that "
                "summary is labeled by clinic, not by row, so it lines up with no row (Lecture 08's callout)."
            ),
        ),
    ),
    Check(
        "visit context: wait_vs_clinic values",
        values_check(
            VISIT_CONTEXT,
            ("wait_vs_clinic",),
            "wait_vs_clinic is wait_min minus clinic_mean_wait, so a visit that waited longer than its "
            "clinic's mean is positive.",
            also=_saved_wait_vs_clinic,
            cause=_when_clinic_means_blank(
                "Every clinic_mean_wait is blank, so wait_min minus clinic_mean_wait is blank too; fix "
                "clinic_mean_wait first."
            ),
        ),
    ),
    # Task 2.3: output/clinic_visit_type_summary.csv
    Check(
        "clinic and visit type: columns",
        columns_check(
            CLINIC_VISIT_TYPE,
            'Group by ["clinic", "visit_type"] with as_index=False so both keys stay columns, and name '
            "visit_count and mean_wait_min in .agg().",
        ),
    ),
    Check(
        "clinic and visit type: one row per pair",
        rows_check(
            CLINIC_VISIT_TYPE,
            'Group by both keys, ["clinic", "visit_type"]: one row per clinic and visit type pair that has '
            "visits, eight in all.",
        ),
    ),
    Check(
        "clinic and visit type: visit_count values",
        values_check(CLINIC_VISIT_TYPE, ("visit_count",), 'visit_count is ("visit_id", "size") within each pair.'),
    ),
    Check(
        "clinic and visit type: mean_wait_min values",
        values_check(
            CLINIC_VISIT_TYPE,
            ("mean_wait_min",),
            'mean_wait_min is ("wait_min", "mean") within each pair; save it unrounded or to at least one decimal.',
        ),
    ),
    # Task 3.1: output/mean_wait_pivot.csv
    Check(
        "mean wait pivot: columns",
        columns_check(
            MEAN_WAIT_PIVOT,
            'Use index="clinic", columns="visit_type", and values="wait_min", then save with the index '
            "(leave out index=False) so clinic is the first column.",
        ),
    ),
    Check(
        "mean wait pivot: one row per clinic",
        rows_check(MEAN_WAIT_PIVOT, 'index="clinic" gives one row per clinic that has visits.'),
    ),
    Check(
        "mean wait pivot: mean waits",
        values_check(
            MEAN_WAIT_PIVOT,
            VISIT_TYPES,
            'Each cell is the mean wait_min of that clinic\'s visits of that type: aggfunc="mean".',
            cells="filled",
            prior=CLINIC_VISIT_TYPE,
        ),
    ),
    Check(
        "mean wait pivot: empty cell stays empty",
        values_check(
            MEAN_WAIT_PIVOT,
            VISIT_TYPES,
            "Sunset had no telehealth visits, so there is no wait to average: keep Sunset's row and leave that cell "
            "empty rather than filling it with 0 or dropping the row.",
            cells="empty",
        ),
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
