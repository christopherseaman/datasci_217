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
row-number column, number formatting (2 == 2.0 == 2.00), and the letter case
of labels and headers never cost points, and cells may be separated by
commas, semicolons, or tabs.

Nothing here imports, runs, or inspects student code.
"""

from __future__ import annotations

import codecs
import csv
import difflib
import re
from collections.abc import Callable
from dataclasses import dataclass
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

    `rows` maps each expected key, a tuple of the key columns' values, to the
    expected value of every other column; None means an empty cell.
    `key_hint` says how the key column usually goes missing from this file.
    """

    path: str
    task: str
    columns: tuple[str, ...]
    key: tuple[str, ...]
    rows: dict[tuple[str, ...], dict[str, object]]
    allowed_extra: tuple[str, ...] = ()
    key_hint: str = ""


@dataclass(frozen=True)
class Table:
    """A saved CSV: its casefolded column names and one {column: cell} dict per data row."""

    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


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
    task="Task 1.2",
    columns=(*SPECIMEN_COLUMNS, "clinic_name", "region", "_merge"),
    key=("specimen_id",),
    rows=_merge_audit_rows(),
    # The lecture's snippet merges the current rows with record_status still attached.
    allowed_extra=("record_status",),
)

COMBINED = Artifact(
    path="output/combined_specimens.csv",
    task="Task 2.1",
    columns=(*SPECIMEN_COLUMNS, "source_partition"),
    key=("specimen_id",),
    rows={
        (specimen_id,): {**values, "source_partition": "batch_a" if specimen_id in BATCH_A else "batch_b"}
        for specimen_id, values in SPECIMENS.items()
    },
)

ALIGNED = Artifact(
    path="output/aligned_features.csv",
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
)

SBP_LONG = Artifact(
    path="output/sbp_long.csv",
    task="Task 3.1",
    columns=("patient_id", "visit", "sbp"),
    key=("patient_id", "visit"),
    rows={(patient_id, visit): {"sbp": SBP_WIDE[patient_id][visit]} for visit in VISITS for patient_id in SBP_WIDE},
)

SBP_ROUND_TRIP = Artifact(
    path="output/sbp_round_trip.csv",
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
    header = [header[i] for i in keep]
    body = [[row[i] for i in keep] for row in body]

    rows = tuple({column: row[i] for i, column in enumerate(header)} for row in body)
    return Table(columns=tuple(header), rows=rows)


def _resolve(table: Table, artifact: Artifact) -> dict[str, str]:
    """Map each expected column to the saved column that holds it.

    A column saved under its own name is found by name. A key column saved
    under another name, such as a blank header over saved index labels or
    melt's default `variable`, is found by its values. If exactly one expected
    column is still unplaced and exactly one unexpected column is left, such as
    melt's default `value`, that column is taken to hold it. A misnamed column
    then costs only the columns check.
    """
    found = {column: column for column in artifact.columns if column in table.columns}
    spare = [column for column in table.columns if column not in artifact.columns and column not in artifact.allowed_extra]
    for position, column in enumerate(artifact.key):
        if column in found:
            continue
        wanted = {key[position].casefold() for key in artifact.rows}
        candidates = []
        for other in spare:
            values = [row[other].casefold() for row in table.rows if row[other]]
            if values and sum(value in wanted for value in values) * 2 > len(values):
                candidates.append(other)
        if len(candidates) == 1:
            found[column] = candidates[0]
            spare.remove(candidates[0])
    unplaced = [column for column in artifact.columns if column not in found]
    if len(unplaced) == 1 and len(spare) == 1:
        found[unplaced[0]] = spare[0]
    return found


def _group(table: Table, artifact: Artifact, found: dict[str, str]):
    """({expected key: [rows]}, [rows naming no expected key]) for the saved table."""
    absent = [column for column in artifact.key if column not in found]
    _assert(
        not absent,
        f"{artifact.path} has no {_join(absent)} column (its columns are {_join(table.columns) or 'none'}), "
        f"so its rows cannot be matched; {artifact.task} saves the header line {','.join(artifact.columns)}. "
        f"{artifact.key_hint}".rstrip(),
    )
    canonical = {tuple(part.casefold() for part in key): key for key in artifact.rows}
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    unknown = []
    for row in table.rows:
        cells = tuple(row[found[column]].casefold() for column in artifact.key)
        key = canonical.get(cells)
        if key is None:
            unknown.append(tuple(row[found[column]] or "blank" for column in artifact.key))
        else:
            grouped.setdefault(key, []).append(row)
    return grouped, unknown


def columns_check(artifact: Artifact, hint: str) -> Callable[[Path], None]:
    """The saved header holds exactly the expected columns, in any order."""

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact)
        # A saved index with no name leaves its column unheaded; holding the IDs, it is the key column.
        unheaded = {
            column: found[column]
            for column in artifact.key
            if column not in table.columns and found.get(column) == table.columns[0] and _is_index_header(table.columns[0])
        }
        missing = [column for column in artifact.columns if column not in table.columns and column not in unheaded]
        extra = [
            column or "a column with no header"
            for column in table.columns
            if column not in artifact.columns and column not in artifact.allowed_extra and column not in unheaded.values()
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


def rows_check(artifact: Artifact, hint: str) -> Callable[[Path], None]:
    """Each expected key appears on exactly one row, and no other key appears."""

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        grouped, unknown = _group(table, artifact, _resolve(table, artifact))
        missing = [key for key in artifact.rows if key not in grouped]
        repeated = [key for key, rows in grouped.items() if len(rows) > 1]
        problems = []
        if missing:
            problems.append(f"is missing {_join(_key_name(key) for key in missing)}")
        if repeated:
            problems.append(
                "lists " + _join(f"{_key_name(key)} {len(grouped[key])} times" for key in repeated)
            )
        if unknown:
            names = [_key_name(key) for key in unknown]
            shown = _join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
            problems.append(f"also has {len(unknown)} row{'s' if len(unknown) != 1 else ''} for {shown}")
        _assert(
            not problems,
            f"{artifact.path} " + "; ".join(problems) + "; it should hold one row for each of "
            f"{_expected_keys([_key_name(key) for key in artifact.rows])}. Fix it in {artifact.task}: {hint}",
        )

    return check


def _unkeyed_values(table: Table, artifact: Artifact, found: dict[str, str], columns: tuple[str, ...], hint: str) -> None:
    """With the key column missing, each column's values match the expected ones in some row order.

    The missing key already costs the columns and rows checks; this keeps a
    correct column from costing its values check too.
    """
    wrong = []
    for column in columns:
        saved = [row[found[column]] for row in table.rows]
        expected = [values[column] for values in artifact.rows.values()]
        unmatched = list(expected)
        all_matched = True
        for cell in saved:
            match = next((i for i, value in enumerate(unmatched) if _matches(cell, value)), None)
            if match is None:
                all_matched = False
                break
            unmatched.pop(match)
        if all_matched and not unmatched:
            continue
        wrong.append(
            f"its {column} values are {_join(_given(cell) for cell in saved) or 'none'}, "
            f"expected {_join(_show(value) for value in expected)} in any order"
        )
    _assert(
        not wrong,
        f"{artifact.path} has no {_join(column for column in artifact.key if column not in found)} column, "
        f"so its values were compared without matching rows, and " + "; ".join(wrong)
        + f". Fix it in {artifact.task}: {hint}",
    )


def values_check(artifact: Artifact, columns: tuple[str, ...], hint: str) -> Callable[[Path], None]:
    """Every saved row for an expected key holds the expected values in `columns`.

    A missing or repeated row costs the rows check, not this one: this check
    compares whatever rows the file does hold for the expected keys. A file
    without its key column is compared column by column in any row order.
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
        if any(column not in found for column in artifact.key):
            _unkeyed_values(table, artifact, found, columns, hint)
            return
        grouped, _ = _group(table, artifact, found)
        _assert(
            grouped,
            f"{artifact.path} has no row for any of {_join(_key_name(key) for key in artifact.rows)}, so its "
            f"{_join(columns)} values cannot be compared; the rows check says what to fix in {artifact.task}.",
        )
        wrong = []
        for key, rows in grouped.items():
            for column in columns:
                expected = artifact.rows[key][column]
                given = sorted({_given(row[found[column]]) for row in rows if not _matches(row[found[column]], expected)})
                if given:
                    wrong.append(f"{_key_name(key)} has {column} {_join(given)}, expected {_show(expected)}")
        _assert(not wrong, f"{artifact.path}: " + _listed(wrong) + f". Fix it in {artifact.task}: {hint}")

    return check


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
            "K01 has a retired and a current record in clinics_history.csv, so merging every record repeats its "
            'specimens: keep only the rows whose record_status is "current", then merge with how="left" so '
            "SP106 stays.",
        ),
    ),
    Check(
        "merge audit: specimen values",
        values_check(
            MERGE_AUDIT,
            SPECIMEN_COLUMNS[1:],
            "The merge copies each specimen's own columns unchanged from specimens.csv, the left table.",
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
        rows_check(COMBINED, "batch_a holds SP101 to SP104 and batch_b holds SP105 to SP107; stack both, once each."),
    ),
    Check(
        "combined specimens: specimen values",
        values_check(COMBINED, SPECIMEN_COLUMNS[1:], "Stacking copies each row unchanged from its batch file."),
    ),
    Check(
        "combined specimens: source_partition labels",
        values_check(
            COMBINED,
            ("source_partition",),
            'Set source_partition to "batch_a" on every batch_a row and "batch_b" on every batch_b row before stacking.',
        ),
    ),
    # Task 2.3: output/aligned_features.csv
    Check(
        "aligned features: columns",
        columns_check(
            ALIGNED,
            "Move specimen_id into the index of both tables, put batch_a's volume_ml beside transit_times with "
            "pd.concat(..., axis=1), and save with the index, so leave out index=False.",
        ),
    ),
    Check(
        "aligned features: one row per specimen",
        rows_check(
            ALIGNED,
            "pd.concat(..., axis=1) keeps every specimen_id label from both tables: SP101 to SP104 from batch_a "
            "and SP108 from transit_times.",
        ),
    ),
    Check(
        "aligned features: volume_ml values",
        values_check(
            ALIGNED,
            ("volume_ml",),
            "volume_ml comes from batch_a; SP108 is not in batch_a, so its volume_ml stays blank.",
        ),
    ),
    Check(
        "aligned features: transit_min values",
        values_check(
            ALIGNED,
            ("transit_min",),
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
    # Task 3.2: output/sbp_round_trip.csv
    Check(
        "SBP round trip: columns",
        columns_check(
            SBP_ROUND_TRIP,
            'Pivot with index="patient_id", columns="visit", values="sbp", then reset_index() so patient_id '
            "is a column again.",
        ),
    ),
    Check(
        "SBP round trip: one row per patient",
        rows_check(SBP_ROUND_TRIP, "Pivoting sbp_long gives one row per patient, P201 to P204."),
    ),
    Check(
        "SBP round trip: baseline values",
        values_check(
            SBP_ROUND_TRIP,
            ("baseline",),
            "The round trip puts every reading back in the cell it came from in sbp_wide.csv.",
        ),
    ),
    Check(
        "SBP round trip: followup values",
        values_check(
            SBP_ROUND_TRIP,
            ("followup",),
            "The round trip puts every reading back in the cell it came from in sbp_wide.csv.",
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
