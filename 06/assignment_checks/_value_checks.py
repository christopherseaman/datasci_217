"""Checks for Assignment 06.

The course keeps these checks in 06/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They read the five CSV files a
submission saves in output/ and compare them with values taken from the
supplied data, copied below. The self-test confirms those copies match the
files in 06/assignment/data/.

Each check scores one thing, so one mistake costs only its own points, and no
check waits for another to pass. Values are compared after parsing: spacing,
line endings, quoting, a byte-order mark, column order, row order, a leading
row-number column, number formatting (2 == 2.0 == 2.00), and the letter case,
spaces, underscores, and hyphens of labels and headers never cost points, and
cells may be separated by commas, semicolons, or tabs.

A mistake is charged once. A column saved twice with the same values counts
once; a column name used twice for different values costs the columns check,
and the other checks read the copy that fits best. An ID column named twice,
which putting tables side by side with pd.concat(..., axis=1) makes, costs
only the columns check, and aligned_features.csv lined up by row position
rather than by specimen_id costs only one check. A row listed twice costs
the rows check, and the values checks accept it when one copy is right. The
batch labels may be any two labels that tell the batches apart, and the round
trip is also judged against the student's own sbp_long.csv, so a mistake in
Task 3.1 is not charged again in Task 3.2.

Nothing here imports, runs, or inspects student code.
"""

from __future__ import annotations

import codecs
import csv
import difflib
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path


# data/specimens.csv: one row per specimen. data/specimens_batch_a.csv holds the
# first four rows and data/specimens_batch_b.csv the other three.
SPECIMEN_COLUMNS = ("specimen_id", "patient_id", "collection_number", "clinic_id", "specimen_type", "volume_ml")
SPECIMENS = {
    "SP101": {"patient_id": "P201", "collection_number": 1, "clinic_id": "K01", "specimen_type": "blood", "volume_ml": 5.0},
    "SP102": {"patient_id": "P201", "collection_number": 2, "clinic_id": "K01", "specimen_type": "urine", "volume_ml": 30.0},
    "SP103": {"patient_id": "P202", "collection_number": 1, "clinic_id": "K02", "specimen_type": "blood", "volume_ml": 4.0},
    "SP104": {"patient_id": "P203", "collection_number": 1, "clinic_id": "K03", "specimen_type": "urine", "volume_ml": 25.0},
    "SP105": {"patient_id": "P204", "collection_number": 1, "clinic_id": "K01", "specimen_type": "blood", "volume_ml": 6.0},
    "SP106": {"patient_id": "P205", "collection_number": 1, "clinic_id": "K09", "specimen_type": "swab", "volume_ml": 3.0},
    "SP107": {"patient_id": "P206", "collection_number": 1, "clinic_id": "K02", "specimen_type": "urine", "volume_ml": 40.0},
}
BATCH_A = ("SP101", "SP102", "SP103", "SP104")

# data/clinics_history.csv: the current record of each clinic, and K01's retired one.
CURRENT_CLINICS = {
    "K01": {"clinic_name": "Bayview Clinic", "region": "southeast"},
    "K02": {"clinic_name": "Castro Clinic", "region": "central"},
    "K03": {"clinic_name": "Richmond Clinic", "region": "west"},
    "K04": {"clinic_name": "Excelsior Clinic", "region": "south"},
}
RETIRED_CLINICS = {"K01": {"clinic_name": "Bayview Annex", "region": "southeast"}}

# data/transit_times.csv: minutes from collection to lab receipt.
TRANSIT_MIN = {"SP102": 45, "SP103": 30, "SP108": 60}

# data/sbp_wide.csv: systolic blood pressure in mmHg at each visit.
SBP_WIDE = {
    "P201": {"baseline": 148, "followup": 136},
    "P202": {"baseline": 162, "followup": 151},
    "P203": {"baseline": 139, "followup": 141},
    "P204": {"baseline": 155, "followup": 142},
}
VISITS = ("baseline", "followup")

# Every supplied value has at most one decimal, so anything closer than this is the same number.
TOLERANCE = 0.005
# Ways pandas and people write an empty cell.
MISSING_SPELLINGS = frozenset({"", "nan", "na", "n/a", "none", "null", "<na>", "nat"})


@dataclass(frozen=True)
class Artifact:
    """One saved CSV: where it lives, which task writes it, and what it should hold.

    `path_name` is the notebook's name for `path`, such as ALIGNED_PATH.
    `rows` maps each expected key, a tuple of the key columns' values, to the
    expected value of every other column; None means an empty cell.
    `key_hint` says how the key column usually goes missing from this file, and
    `repeated_hint` how a column comes to be named twice in it.
    """

    path: str
    path_name: str
    task: str
    columns: tuple[str, ...]
    key: tuple[str, ...]
    rows: dict[tuple[str, ...], dict[str, object]]
    allowed_extra: tuple[str, ...] = ()
    key_hint: str = ""
    repeated_hint: str = ""


@dataclass(frozen=True)
class Table:
    """A saved CSV: its column names in file order, one tuple of cells per data row
    in the same order, and how many times each column name used more than once is used.

    Checks find a column by its position, so every copy of a repeated name stays readable.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    repeated: dict[str, int]


@dataclass(frozen=True)
class RowProblems:
    """What a rows check found: expected keys with no row, expected keys on several
    rows, the keys of rows that match no expected key, and rows with an empty key."""

    missing: tuple[tuple[str, ...], ...]
    repeated: tuple[tuple[str, ...], ...]
    unknown: tuple[tuple[str, ...], ...]
    empty: int


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


def _merge_audit_rows() -> dict[tuple[str, ...], dict[str, object]]:
    rows = {}
    for specimen_id, values in SPECIMENS.items():
        clinic = CURRENT_CLINICS.get(str(values["clinic_id"]))
        rows[(specimen_id,)] = {
            **values,
            "clinic_name": clinic["clinic_name"] if clinic else None,
            "region": clinic["region"] if clinic else None,
            "_merge": "both" if clinic else "left_only",
        }
    return rows


MERGE_AUDIT = Artifact(
    path="output/specimen_merge_audit.csv",
    path_name="MERGE_AUDIT_PATH",
    task="Task 1.2",
    columns=(*SPECIMEN_COLUMNS, "clinic_name", "region", "_merge"),
    key=("specimen_id",),
    rows=_merge_audit_rows(),
    # The lecture's snippet merges the current rows with record_status still attached.
    allowed_extra=("record_status",),
)

COMBINED = Artifact(
    path="output/combined_specimens.csv",
    path_name="COMBINED_PATH",
    task="Task 2.1",
    columns=(*SPECIMEN_COLUMNS, "source_partition"),
    key=("specimen_id",),
    rows={
        (specimen_id,): {**values, "source_partition": "batch_a" if specimen_id in BATCH_A else "batch_b"}
        for specimen_id, values in SPECIMENS.items()
    },
    repeated_hint="Putting the batches side by side with pd.concat(..., axis=1) repeats every column name; stack "
    "them with the default axis=0 instead.",
)

ALIGNED = Artifact(
    path="output/aligned_features.csv",
    path_name="ALIGNED_PATH",
    task="Task 2.3",
    columns=("specimen_id", "volume_ml", "transit_min"),
    key=("specimen_id",),
    rows={
        (specimen_id,): {
            "volume_ml": SPECIMENS[specimen_id]["volume_ml"] if specimen_id in BATCH_A else None,
            "transit_min": TRANSIT_MIN.get(specimen_id),
        }
        for specimen_id in dict.fromkeys([*BATCH_A, *TRANSIT_MIN])
    },
    key_hint="specimen_id is the index of aligned_features, so save it with the index: leave out index=False.",
    repeated_hint="Both tables kept specimen_id as a column, so pd.concat(..., axis=1) lined their rows up by row "
    'number: set_index("specimen_id") on both tables before pd.concat(..., axis=1).',
)

SBP_LONG = Artifact(
    path="output/sbp_long.csv",
    path_name="SBP_LONG_PATH",
    task="Task 3.1",
    columns=("patient_id", "visit", "sbp"),
    key=("patient_id", "visit"),
    rows={(patient_id, visit): {"sbp": SBP_WIDE[patient_id][visit]} for visit in VISITS for patient_id in SBP_WIDE},
)

SBP_ROUND_TRIP = Artifact(
    path="output/sbp_round_trip.csv",
    path_name="SBP_ROUND_TRIP_PATH",
    task="Task 3.2",
    columns=("patient_id", *VISITS),
    key=("patient_id",),
    rows={(patient_id,): dict(readings) for patient_id, readings in SBP_WIDE.items()},
    key_hint="pivot() moves patient_id into the index, so call reset_index() before saving with index=False.",
)


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


def _number(cell: str) -> float | None:
    try:
        return float(_clean(cell).replace(",", ""))
    except ValueError:
        return None


def _is_whole_number(cell: str) -> bool:
    number = _number(cell)
    return number is not None and number.is_integer()


def _label(text: str) -> str:
    """A label or header as it is compared: letter case, spaces, underscores, and hyphens set aside."""
    return re.sub(r"[\W_]+", "", text.casefold())


def _matches(cell: str, expected: object) -> bool:
    text = _clean(cell)
    if expected is None:
        return text.casefold() in MISSING_SPELLINGS
    if isinstance(expected, (int, float)):
        number = _number(text)
        return number is not None and abs(number - expected) <= TOLERANCE
    return _label(text) == _label(str(expected))


def _same_cell(left: str, right: str) -> bool:
    """Equivalent saved cells, including formatting differences between duplicated columns."""
    if left.casefold() in MISSING_SPELLINGS and right.casefold() in MISSING_SPELLINGS:
        return True
    number = _number(right)
    if number is not None:
        return _matches(left, number)
    return _label(left) == _label(right)


def _show(value: object) -> str:
    if value is None:
        return "blank"
    if isinstance(value, float):
        return f"{value:g}"
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


def _in_words(number: int) -> str:
    """'a', 'two', ... 'nine', then digits: how many columns a message counts."""
    words = ("no", "a", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    return words[number] if number < len(words) else str(number)


def _count(number: int, noun: str) -> str:
    """'1 row', '4 rows'."""
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _no_columns(names: list[str]) -> str:
    """'no sbp column' or 'no clinic_name and region columns'."""
    return f"no {_join(names)} column{'s' if len(names) > 1 else ''}"


def _named_more_than_once(table: Table, names) -> str:
    """'specimen_id twice' or 'visit and sbp twice', with the count past two."""
    by_count: dict[int, list[str]] = {}
    for name in names:
        by_count.setdefault(table.repeated[name], []).append(name)
    parts = []
    for count, group in by_count.items():
        times = "twice" if count == 2 else f"{count} times"
        parts.append(f"{_join(group)} {times}")
    return " and ".join(parts)


# The fix for a column named twice, when the artifact names no likelier cause.
REPEATED_HINT = "Save each column once; pd.concat(..., axis=1) repeats every column name the two tables share."


def _repeated_cause(table: Table, artifact: Artifact, found: dict[str, int], columns) -> str:
    """The likely cause when a column this check reads is named more than once, or ""."""
    twice = list(dict.fromkeys(table.columns[found[column]] for column in columns if column in found))
    twice = [name for name in twice if name in table.repeated]
    if not twice:
        return ""
    return (
        f"Its header line names {_named_more_than_once(table, twice)}, so the checks read the copy that fits best: "
        f"{artifact.repeated_hint or REPEATED_HINT}"
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
    if "\x00" in text:
        raise csv.Error("embedded NUL bytes; save the table again as CSV text")
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


def _look_alikes(folder: Path, name: str) -> list[str]:
    """Names of the files in `folder` whose names match or nearly match `name`."""
    if not folder.is_dir():
        return []
    return sorted(
        path.name
        for path in folder.iterdir()
        if path.is_file() and difflib.SequenceMatcher(None, name.casefold(), path.name.casefold()).ratio() >= 0.75
    )


def _missing(root: Path, artifact: Artifact) -> str:
    """Say the artifact is missing, and where a file that looks like it was saved instead."""
    message = f"{artifact.path} is missing; run the {artifact.task} cell to write it, then commit it."
    wanted = root / artifact.path
    folder = wanted.parent.relative_to(root).as_posix()
    in_output = [f"{folder}/{name}" for name in _look_alikes(wanted.parent, wanted.name)]
    in_root = _look_alikes(root, wanted.name)
    if in_output:
        message += f" Found {_join(in_output)}; save it as {artifact.path} instead."
    elif in_root:
        message += (
            f" Found {_join(in_root)} in the assignment folder, outside {folder}/; save it to "
            f"{artifact.path_name}, which is {artifact.path}."
        )
    return message


def _is_index_header(cell: str) -> bool:
    return cell in ("", "index") or cell.startswith("unnamed")


def read_table(root: Path, artifact: Artifact) -> Table:
    """Parse a saved CSV, however it is separated, spaced, quoted, or ended.

    A leading column headed by nothing, `Unnamed: 0`, or `index` that holds only
    whole numbers is the row numbers pandas writes when `index=False` is left
    out, so it is set aside rather than counted as a column. So is an empty
    column with no header, which a trailing comma on every line makes. A
    header that differs from an expected name only in letter case, spaces,
    underscores, or hyphens is read as that name. A column that repeats an
    earlier column of the same name cell for cell, as saving an index kept
    with set_index(..., drop=False) does, is read once; a name used for
    columns that differ is recorded in `repeated`, and every copy stays in
    `rows`.
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
    header = [_clean(cell).casefold() for cell in lines[0]]
    body = [[_clean(cell) for cell in row] for row in lines[1:]]
    width = max(len(header), *(len(row) for row in body)) if body else len(header)
    header += [""] * (width - len(header))
    body = [row + [""] * (width - len(row)) for row in body]

    if len(header) > 1 and _is_index_header(header[0]) and body and all(_is_whole_number(row[0]) for row in body):
        header = header[1:]
        body = [row[1:] for row in body]
    keep = [i for i, column in enumerate(header) if column or any(row[i] for row in body)]
    by_label = {_label(name): name for name in (*artifact.columns, *artifact.allowed_extra)}
    header = [by_label.get(_label(header[i]), header[i]) if header[i] else "" for i in keep]
    body = [[row[i] for i in keep] for row in body]

    # A named column identical to an earlier one of the same name adds nothing, so it is read once.
    unique = []
    for i, column in enumerate(header):
        same = [j for j in unique if header[j] == column]
        if column and same and all(_same_cell(row[i], row[same[0]]) for row in body):
            continue
        unique.append(i)
    header = [header[i] for i in unique]
    body = [[row[i] for i in unique] for row in body]

    counts: dict[str, int] = {}
    for column in header:
        if column:
            counts[column] = counts.get(column, 0) + 1
    return Table(
        columns=tuple(header),
        rows=tuple(tuple(row) for row in body),
        repeated={column: count for column, count in counts.items() if count > 1},
    )


def _positions(table: Table, name: str) -> list[int]:
    return [i for i, column in enumerate(table.columns) if column == name]


def _key_labels(artifact: Artifact, position: int) -> set[str]:
    """The expected values of the key column at `position`, as they are compared."""
    return {_label(key[position]) for key in artifact.rows}


def _canonical_key(table_row: tuple[str, ...], artifact: Artifact, found: dict[str, int]):
    """The expected key a saved row names, or None."""
    canonical = {tuple(_label(part) for part in key): key for key in artifact.rows}
    return canonical.get(tuple(_label(table_row[found[column]]) for column in artifact.key))


def _resolve(table: Table, artifact: Artifact) -> dict[str, int]:
    """Map each expected column to the position of the saved column that holds it.

    A column saved under its own name is found by name; when the name is used
    for several columns, a key column is read from the copy that holds the
    most expected IDs, and any other column from its first copy (the values
    checks then pick the copy that fits best). A key column saved under
    another name, such as a blank header over saved index labels or melt's
    default `variable`, is found by its values, and so is any other column
    when exactly one unexpected column holds its expected value on every row.
    If exactly one expected column is still unplaced and exactly one
    unexpected column is left, such as melt's default `value`, that column is
    taken to hold it. A misnamed column then costs only the columns check.
    """
    found: dict[str, int] = {}
    for column in artifact.columns:
        places = _positions(table, column)
        if len(places) > 1 and column in artifact.key:
            wanted = _key_labels(artifact, artifact.key.index(column))
            # max() keeps the first of equally good copies.
            found[column] = max(places, key=lambda i: len({_label(row[i]) for row in table.rows} & wanted))
        elif places:
            found[column] = places[0]
    spare = [
        i for i, column in enumerate(table.columns)
        if column not in artifact.columns and column not in artifact.allowed_extra
    ]
    for position, column in enumerate(artifact.key):
        if column in found:
            continue
        wanted = _key_labels(artifact, position)
        candidates = []
        for other in spare:
            values = [_label(row[other]) for row in table.rows if row[other]]
            if values and sum(value in wanted for value in values) * 2 > len(values):
                candidates.append(other)
        if len(candidates) == 1:
            found[column] = candidates[0]
            spare.remove(candidates[0])
    if all(column in found for column in artifact.key):
        keyed = [(key, row) for key, row in ((_canonical_key(row, artifact, found), row) for row in table.rows) if key]
        for column in artifact.columns:
            if column in found or not keyed:
                continue
            candidates = [
                other for other in spare
                if all(_matches(row[other], artifact.rows[key][column]) for key, row in keyed)
            ]
            if len(candidates) == 1:
                found[column] = candidates[0]
                spare.remove(candidates[0])
    unplaced = [column for column in artifact.columns if column not in found]
    if len(unplaced) == 1 and len(spare) == 1:
        found[unplaced[0]] = spare[0]
    _assert(
        found,
        f"{artifact.path} has no recognizable columns or values for {','.join(artifact.columns)}; "
        f"found {_join(table.columns)}. In {artifact.task}, save the requested table again with to_csv().",
    )
    return found


def _group(table: Table, artifact: Artifact, found: dict[str, int]):
    """({expected key: [rows]}, [key cells of each row naming no expected key]) for the saved table."""
    absent = [column for column in artifact.key if column not in found]
    _assert(
        not absent,
        f"{artifact.path} has {_no_columns(absent)} (its columns are {_join(table.columns) or 'none'}), "
        f"so its rows cannot be matched; {artifact.task} saves the header line {','.join(artifact.columns)}. "
        f"{artifact.key_hint}".rstrip(),
    )
    grouped: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    unknown = []
    for row in table.rows:
        key = _canonical_key(row, artifact, found)
        if key is None:
            unknown.append(tuple(row[found[column]] for column in artifact.key))
        else:
            grouped.setdefault(key, []).append(row)
    return grouped, unknown


def _best_copy(table: Table, artifact: Artifact, grouped, found: dict[str, int], column: str) -> int:
    """The position of the copy of `column` that matches the expected value for the most keys."""
    places = _positions(table, table.columns[found[column]])
    if len(places) < 2:
        return found[column]

    def fits(i: int) -> int:
        return sum(any(_matches(row[i], artifact.rows[key][column]) for row in rows) for key, rows in grouped.items())

    return max(places, key=fits)


def columns_check(artifact: Artifact, hint: str) -> Callable[[Path], None]:
    """The saved header holds exactly the expected columns, in any order."""

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact)
        # A saved index with no name leaves its column unheaded; holding the IDs, it is the key column.
        unheaded = {
            column: found[column]
            for column in artifact.key
            if column not in table.columns and found.get(column) == 0 and _is_index_header(table.columns[0])
        }
        missing = [column for column in artifact.columns if column not in table.columns and column not in unheaded]
        extra = [
            column
            for i, column in enumerate(table.columns)
            if column not in artifact.columns and column not in artifact.allowed_extra and i not in unheaded.values()
        ]
        unnamed = extra.count("")
        extra = [column for column in extra if column]
        if unnamed:
            extra.append(f"{_in_words(unnamed)} column{'' if unnamed == 1 else 's'} with no header")
        problems = []
        if missing:
            problems.append(f"is missing {_join(missing)}")
        if extra:
            problems.append(f"also has {_join(extra)}")
        if table.repeated:
            problems.append(f"names {_named_more_than_once(table, table.repeated)}")
        # A header whose only fault is a repeated name gets the cause of the repeat in place of the general hint.
        advice = (artifact.repeated_hint or REPEATED_HINT) if table.repeated and not (missing or extra) else hint
        _assert(
            not problems,
            f"{artifact.path} " + " and ".join(problems) + f"; {artifact.task} saves the header line "
            f"{','.join(artifact.columns)} (any column order). {advice}",
        )

    return check


def rows_check(
    artifact: Artifact, hint: str, cause: Callable[[RowProblems], str] | None = None
) -> Callable[[Path], None]:
    """Each expected key appears on exactly one row, and no other key appears.

    `cause`, given what the check found, returns the likely cause of it, or "".
    """

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact)
        if any(column not in found for column in artifact.key):
            return  # The columns check charges missing keys once.
        if any(table.columns[found[column]] in table.repeated for column in artifact.key):
            return  # The columns check charges an ID column named twice, with its cause, once.
        grouped, unknown = _group(table, artifact, found)
        named = [cells for cells in unknown if any(cells)]
        seen = RowProblems(
            missing=tuple(key for key in artifact.rows if key not in grouped),
            repeated=tuple(key for key, rows in grouped.items() if len(rows) > 1),
            unknown=tuple(named),
            empty=len(unknown) - len(named),
        )
        problems = []
        if seen.missing:
            problems.append(f"is missing {_join(_key_name(key) for key in seen.missing)}")
        if seen.repeated:
            problems.append(
                "lists " + _join(f"{_key_name(key)} {len(grouped[key])} times" for key in seen.repeated)
            )
        if seen.unknown:
            names = [_key_name(tuple(cell or "blank" for cell in cells)) for cells in seen.unknown]
            shown = _join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
            problems.append(f"also has {_count(len(names), 'row')} for {shown}")
        if seen.empty:
            problems.append(f"also has {_count(seen.empty, 'row')} with an empty {' and '.join(artifact.key)}")
        likely = cause(seen) if cause else ""
        _assert(
            not problems,
            f"{artifact.path} " + "; ".join(problems) + "; it should hold one row for each of "
            f"{_expected_keys([_key_name(key) for key in artifact.rows])}. Fix it in {artifact.task}: {hint}"
            + (f" {likely}" if likely else ""),
        )

    return check


def _same_values_any_order(saved: list[str], expected: list[object]) -> bool:
    unmatched = list(expected)
    for cell in saved:
        match = next((i for i, value in enumerate(unmatched) if _matches(cell, value)), None)
        if match is None:
            return False
        unmatched.pop(match)
    return not unmatched


def _unkeyed_values(table: Table, artifact: Artifact, found: dict[str, int], columns: tuple[str, ...], hint: str) -> None:
    """With the key column missing, each column's values match the expected ones in some row order.

    The missing key already costs the columns and rows checks; this keeps a
    correct column from costing its values check too. Any copy of a repeated
    column may be the one that matches.
    """
    wrong = []
    for column in columns:
        expected = [values[column] for values in artifact.rows.values()]
        copies = _positions(table, table.columns[found[column]])
        if any(_same_values_any_order([row[i] for row in table.rows], expected) for i in copies):
            continue
        saved = [row[found[column]] for row in table.rows]
        wrong.append(
            f"its {column} values are {_join(_given(cell) for cell in saved) or 'none'}, "
            f"expected {_join(_show(value) for value in expected)} in any order"
        )
    _assert(
        not wrong,
        f"{artifact.path} has {_no_columns([column for column in artifact.key if column not in found])}, "
        f"so its values were compared without matching rows, and " + "; ".join(wrong)
        + f". Fix it in {artifact.task}: {hint}",
    )


def values_check(
    artifact: Artifact,
    columns: tuple[str, ...],
    hint: str,
    cause: Callable[[dict[tuple[str, ...], list[tuple[str, ...]]], dict[str, int]], str] | None = None,
    own: Callable[[dict[tuple[str, ...], list[tuple[str, ...]]], int], dict[tuple[str, ...], object] | None] | None = None,
) -> Callable[[Path], None]:
    """Each expected key has a saved row holding the expected values in `columns`.

    A missing or repeated row costs the rows check, not this one: this check
    compares whatever rows the file does hold for the expected keys, and a key
    on several rows passes when one of them is right. A file without its key
    column is compared column by column in any row order. A file without one
    of `columns` pays its columns check once; present columns are still assessed. `cause`, given
    the saved rows by key and the column map, returns the likely cause of
    wrong values, or "". `own`, given the saved rows by key and the position
    of a column, returns values the student chose that also count, by key, or
    None.
    """

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact)
        present = tuple(column for column in columns if column in found)
        absent = [column for column in columns if column not in found]
        if absent:
            recognizable = bool(_group(table, artifact, found)[0]) if all(
                column in found for column in artifact.key
            ) else any(column in found for column in columns)
            _assert(
                recognizable,
                f"{artifact.path} has no recognizable rows or {_join(columns)} values; expected source "
                f"records with the columns {','.join(artifact.columns)}. In {artifact.task}, save the requested table again.",
            )
            if not present:
                return  # The columns check charges missing values once.
        if any(column not in found for column in artifact.key):
            _unkeyed_values(table, artifact, found, present, hint)
            return
        grouped, _ = _group(table, artifact, found)
        _assert(
            grouped,
            f"{artifact.path} has no row for any of {_join(_key_name(key) for key in artifact.rows)}, so its "
            f"{_join(columns)} values cannot be compared; the rows check says what to fix in {artifact.task}.",
        )
        places = {column: _best_copy(table, artifact, grouped, found, column) for column in present}
        chosen = {column: (own(grouped, places[column]) if own else None) or {} for column in present}
        wrong = []
        for key, rows in grouped.items():
            for column in present:
                expected = artifact.rows[key][column]
                cells = [row[places[column]] for row in rows]
                if any(_matches(cell, expected) or (key in chosen[column] and _matches(cell, chosen[column][key]))
                       for cell in cells):
                    continue
                wrong.append(f"{_key_name(key)} has {column} {_join(sorted({_given(cell) for cell in cells}))}, "
                             f"expected {_show(expected)}")
        likely = ""
        if wrong:
            likely = _repeated_cause(table, artifact, found, (*artifact.key, *present)) or (
                cause(grouped, found) if cause else ""
            )
        _assert(
            not wrong,
            f"{artifact.path}: " + _listed(wrong) + f". Fix it in {artifact.task}: {hint}" + (f" {likely}" if likely else ""),
        )

    return check


# Likely causes, each returned only when the check found what that cause produces.


def _merge_audit_rows_cause(seen: RowProblems) -> str:
    causes = []
    if seen.empty:
        causes.append(
            'A row with an empty specimen_id is a clinic with no specimens, such as K04, which how="outer" and '
            'how="right" add; how="left" keeps only the specimens.'
        )
    if seen.repeated:
        causes.append(
            "K01 has a retired and a current record in clinics_history.csv, so merging every record repeats its "
            'specimens: keep only the rows whose record_status is "current".'
        )
    if ("SP106",) in seen.missing and not seen.empty:
        causes.append('Without how="left", merge() is an inner join and drops SP106, whose clinic K09 has no record.')
    return " ".join(causes)


def _combined_rows_cause(seen: RowProblems) -> str:
    if seen.repeated:
        return "A repeated specimen went into pd.concat twice; list each batch once."
    return ""


def _aligned_rows_cause(seen: RowProblems) -> str:
    if seen.empty:
        return (
            "Rows with an empty specimen_id mean one table kept its row numbers as its index: "
            'set_index("specimen_id") on both tables before pd.concat(..., axis=1).'
        )
    if seen.repeated:
        return "A specimen listed twice means the tables were stacked: pass axis=1 to put them side by side."
    if seen.missing:
        return 'join="inner", or a merge other than how="outer", drops the labels that only one table has.'
    return ""


# The fix for aligned_features.csv lined up by row position rather than by specimen_id.
BY_POSITION_CAUSE = (
    "Its rows were lined up by row number, not by specimen_id: pd.concat(..., axis=1) matches row labels, and "
    'a table that keeps specimen_id as a column still has row numbers as its labels. set_index("specimen_id") '
    "on both tables first."
)


def _in_file_order(table: Table, position: int | None, expected: list[object]) -> bool:
    filled = [row[position] for row in table.rows if row[position]] if position is not None else []
    return len(filled) == len(expected) and all(_matches(cell, value) for cell, value in zip(filled, expected))


def _lined_up_by_position(root: Path) -> bool:
    """aligned_features.csv holds every source value in file order, but not beside its own specimen_id.

    That is pd.concat(..., axis=1) with row numbers still the index of one or
    both tables, in either order: one mistake, charged once.
    """
    try:
        table = read_table(root, ALIGNED)
        found = _resolve(table, ALIGNED)
    except AssertionError:
        return False
    volumes = [SPECIMENS[specimen_id]["volume_ml"] for specimen_id in BATCH_A]
    return (
        _in_file_order(table, found.get("volume_ml"), volumes)
        and _in_file_order(table, found.get("transit_min"), list(TRANSIT_MIN.values()))
        and not all(_passes(values_check(ALIGNED, (column,), ""), root) for column in ("volume_ml", "transit_min"))
    )


def _aligned_columns(hint: str) -> Callable[[Path], None]:
    def check(root: Path) -> None:
        columns_check(ALIGNED, BY_POSITION_CAUSE if _lined_up_by_position(root) else hint)(root)

    return check


def _aligned_rows(hint: str) -> Callable[[Path], None]:
    def check(root: Path) -> None:
        by_position = _lined_up_by_position(root)
        if by_position and not _passes(columns_check(ALIGNED, ""), root):
            return  # The columns check charges the misaligned rows once, naming the cause.
        rows_check(ALIGNED, hint, (lambda seen: BY_POSITION_CAUSE) if by_position else _aligned_rows_cause)(root)

    return check


def _aligned_values(column: str, hint: str) -> Callable[[Path], None]:
    def check(root: Path) -> None:
        if _lined_up_by_position(root) and not (
            _passes(columns_check(ALIGNED, ""), root) and _passes(rows_check(ALIGNED, ""), root)
        ):
            return  # The columns or rows check charges the shifted values once, naming the cause.
        values_check(ALIGNED, (column,), hint)(root)

    return check


def _round_trip_rows_cause(seen: RowProblems) -> str:
    if seen.repeated:
        return "A patient on two rows, one per visit, means sbp_long was saved here; save the pivoted table."
    return ""


MISSING_LABELS = frozenset(_label(spelling) for spelling in MISSING_SPELLINGS)


def _own_batch_labels(grouped: dict[tuple[str, ...], list[tuple[str, ...]]], place: int):
    """The student's own batch labels by key, or None.

    Any two labels tell the batches apart, such as A and B or the file names,
    when every batch A row has one, every batch B row the other, and neither
    names the other batch.
    """
    labels: dict[bool, set[str]] = {True: set(), False: set()}
    for key, rows in grouped.items():
        labels[key[0] in BATCH_A].update(_label(row[place]) for row in rows)
    in_a, in_b = labels[True], labels[False]
    if len(in_a) != 1 or len(in_b) != 1 or in_a == in_b or (in_a | in_b) & MISSING_LABELS:
        return None
    (label_a,), (label_b,) = in_a, in_b
    if label_a == "b" or "batchb" in label_a.replace("_", "") or label_b == "a" or "batcha" in label_b.replace("_", ""):
        return None
    return {key: label_a if key[0] in BATCH_A else label_b for key in grouped}


def _own_round_trip(root: Path) -> Artifact | None:
    """The round trip that pivoting the student's own sbp_long.csv gives, or None.

    The round-trip checks also accept this, so a mistake already charged in
    Task 3.1 is not charged again in Task 3.2. None when that file is
    unreadable, has no patient_id, visit, or sbp column, repeats a patient and
    visit pair (pivot() refuses it), or matches sbp_wide.csv.
    """
    try:
        table = read_table(root, SBP_LONG)
        found = _resolve(table, SBP_LONG)
    except AssertionError:
        return None
    if any(column not in found for column in SBP_LONG.columns):
        return None
    patients = {_label(key[0]): key[0] for key in SBP_ROUND_TRIP.rows}
    visits = {_label(visit): visit for visit in VISITS}
    readings: dict[tuple[str, ...], dict[str, object]] = {}
    columns: list[str] = []
    for row in table.rows:
        patient, visit, sbp = (_clean(row[found[column]]) for column in SBP_LONG.columns)
        if not patient or not visit:
            continue
        patient = patients.get(_label(patient), patient)
        visit = visits.get(_label(visit), visit.casefold())
        cells = readings.setdefault((patient,), {})
        if visit in cells:
            return None
        number = _number(sbp)
        cells[visit] = None if sbp.casefold() in MISSING_SPELLINGS else (sbp if number is None else number)
        if visit not in columns:
            columns.append(visit)
    if not readings:
        return None
    rows = {key: {visit: cells.get(visit) for visit in columns} for key, cells in readings.items()}
    if sorted(columns) == sorted(VISITS) and rows == SBP_ROUND_TRIP.rows:
        return None
    return replace(SBP_ROUND_TRIP, columns=("patient_id", *columns), rows=rows)


def _passes(check: Callable[[Path], None], root: Path) -> bool:
    try:
        check(root)
    except AssertionError:
        return False
    return True


def _or_own_long(check_for: Callable[[Artifact], Callable[[Path], None]]) -> Callable[[Path], None]:
    """A round-trip check that also passes against the pivot of the student's own sbp_long.csv."""

    def check(root: Path) -> None:
        try:
            check_for(SBP_ROUND_TRIP)(root)
        except AssertionError as error:
            own = _own_round_trip(root)
            if own is None or not _passes(check_for(own), root):
                raise error from None

    return check


def _round_trip_values(visit: str, hint: str) -> Callable[[Path], None]:
    """One visit's round-trip values; nothing to compare when the student's own sbp_long has no such visit."""
    return _or_own_long(
        lambda artifact: values_check(
            artifact, tuple(column for column in (visit,) if column in artifact.columns), hint, _round_trip_swapped
        )
    )


def _round_trip_swapped(grouped: dict[tuple[str, ...], list[tuple[str, ...]]], found: dict[str, int]) -> str:
    def holds(column: str, other: str) -> bool:
        return column in found and all(
            _matches(row[found[column]], SBP_WIDE[key[0]][other]) for key, rows in grouped.items() for row in rows
        )

    if holds("baseline", "followup") and holds("followup", "baseline"):
        return (
            "The baseline and followup columns hold each other's readings, as when columns are renamed by "
            "position: keep the names pivot() gives."
        )
    return ""


CHECKS = (
    # Task 1.2: output/specimen_merge_audit.csv
    Check(
        "merge audit: columns",
        columns_check(
            MERGE_AUDIT,
            "Select clinic_id, clinic_name, and region from the current clinic records, then merge with "
            "indicator=True, which adds _merge. Keeping record_status as well is fine.",
        ),
    ),
    Check(
        "merge audit: one row per specimen",
        rows_check(
            MERGE_AUDIT,
            "The merge audit keeps every row of specimens.csv, the left table, once.",
            _merge_audit_rows_cause,
        ),
    ),
    Check(
        "merge audit: specimen values",
        values_check(
            MERGE_AUDIT,
            SPECIMEN_COLUMNS[1:],
            "The merge copies each specimen's own columns unchanged from specimens.csv, the left table, so a "
            "different value means a table changed before the merge: click Restart, then Run All.",
        ),
    ),
    Check(
        "merge audit: current clinic names and regions",
        values_check(
            MERGE_AUDIT,
            ("clinic_name", "region"),
            "Each specimen takes its clinic's current record (Bayview Annex is K01's retired name); SP106's "
            "clinic K09 has no record, so its clinic_name and region stay blank.",
        ),
    ),
    Check(
        "merge audit: _merge indicator",
        values_check(
            MERGE_AUDIT,
            ("_merge",),
            "merge(..., indicator=True) labels each row both or left_only; SP106's clinic K09 is not in "
            "clinics_history.csv, so it is left_only.",
        ),
    ),
    # Task 2.1: output/combined_specimens.csv
    Check(
        "combined specimens: columns",
        columns_check(COMBINED, "Add source_partition to each batch, then stack batch_a above batch_b with pd.concat."),
    ),
    Check(
        "combined specimens: one row per specimen",
        rows_check(
            COMBINED,
            "batch_a holds SP101 to SP104 and batch_b holds SP105 to SP107; stack both, once each.",
            _combined_rows_cause,
        ),
    ),
    Check(
        "combined specimens: specimen values",
        values_check(
            COMBINED,
            SPECIMEN_COLUMNS[1:],
            "Stacking copies each row unchanged from its batch file, so a different value means a batch "
            "changed before stacking: click Restart, then Run All.",
        ),
    ),
    Check(
        "combined specimens: source_partition labels",
        values_check(
            COMBINED,
            ("source_partition",),
            'Set source_partition to "batch_a" on every batch_a row and "batch_b" on every batch_b row before '
            "stacking; any two labels that tell the batches apart also count.",
            own=_own_batch_labels,
        ),
    ),
    # Task 2.3: output/aligned_features.csv
    Check(
        "aligned features: columns",
        _aligned_columns(
            "Move specimen_id into the index of both tables, put batch_a's volume_ml beside transit_times with "
            "pd.concat(..., axis=1), and save with the index, so leave out index=False.",
        ),
    ),
    Check(
        "aligned features: one row per specimen",
        _aligned_rows(
            "pd.concat(..., axis=1) keeps every specimen_id label from both tables: SP101 to SP104 from batch_a "
            "and SP108 from transit_times.",
        ),
    ),
    Check(
        "aligned features: volume_ml values",
        _aligned_values(
            "volume_ml",
            "volume_ml comes from batch_a; SP108 is not in batch_a, so its volume_ml stays blank.",
        ),
    ),
    Check(
        "aligned features: transit_min values",
        _aligned_values(
            "transit_min",
            "transit_min comes from transit_times.csv; SP101 and SP104 have no transit time, so theirs stay blank.",
        ),
    ),
    # Task 3.1: output/sbp_long.csv
    Check(
        "SBP long: columns",
        columns_check(
            SBP_LONG,
            'Melt sbp_wide with var_name="visit" and value_name="sbp"; without them, melt() names the columns '
            "variable and value.",
        ),
    ),
    Check(
        "SBP long: one row per patient and visit",
        rows_check(
            SBP_LONG,
            "Melt the baseline and followup columns for all four patients: eight rows, one per patient and visit.",
        ),
    ),
    Check(
        "SBP long: sbp values",
        values_check(
            SBP_LONG,
            ("sbp",),
            "Each sbp is the reading in that patient's baseline or followup column of sbp_wide.csv.",
        ),
    ),
    # Task 3.2: output/sbp_round_trip.csv, which may also match the pivot of the student's own sbp_long.csv.
    Check(
        "SBP round trip: columns",
        _or_own_long(
            lambda artifact: columns_check(
                artifact,
                'Pivot with index="patient_id", columns="visit", values="sbp", then reset_index() so patient_id '
                "is a column again.",
            )
        ),
    ),
    Check(
        "SBP round trip: one row per patient",
        _or_own_long(
            lambda artifact: rows_check(
                artifact,
                "Pivoting sbp_long gives one row per patient, P201 to P204.",
                _round_trip_rows_cause,
            )
        ),
    ),
    Check(
        "SBP round trip: baseline values",
        _round_trip_values("baseline", "The round trip puts every reading back in the cell it came from in sbp_wide.csv."),
    ),
    Check(
        "SBP round trip: followup values",
        _round_trip_values("followup", "The round trip puts every reading back in the cell it came from in sbp_wide.csv."),
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
