"""Checks for Assignment 04.

The course keeps these checks in 04/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They read the two CSV files a
submission saves in output/ and compare them with values recomputed from the
supplied fridge readings and supply order below. The self-test confirms those
copies match assignment.ipynb and data/supply_order.csv.

Each check scores one thing, so one mistake costs only the checks it gets
wrong. A column saved under another name costs only its name check: its values
are still graded under the name it has. Values are compared after parsing:
spacing, line endings, quoting, column order, a leading row-number column,
number formatting (2 == 2.0 == 2.00), and the letter case of labels never cost
points, and cells may be separated by commas, semicolons, or tabs.

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


FRIDGE_FILE = "output/fridge_block.csv"
SUPPLIES_FILE = "output/selected_supplies.csv"
# The notebook step and the call that save each artifact.
SAVED_BY = {
    FRIDGE_FILE: ("Task 2.2", "label_block.to_csv(FRIDGE_OUTPUT_PATH)"),
    SUPPLIES_FILE: ("Task 3.2", "selected_supplies.to_csv(SUPPLIES_OUTPUT_PATH, index=False)"),
}
# The fix when one task's save call writes to the other task's path.
SAVE_BOTH = (
    f"{SAVED_BY[SUPPLIES_FILE][0]} saves with {SAVED_BY[SUPPLIES_FILE][1]}, and {SAVED_BY[FRIDGE_FILE][0]} with "
    f"{SAVED_BY[FRIDGE_FILE][1]}; correct the path in the call that differs, then Restart and Run All."
)

# The supplied fridge_readings array in assignment.ipynb, with its row labels, in °C.
FRIDGE_READINGS = {
    "FRG-101": {"am_temp_c": 4.1, "pm_temp_c": 5.6},
    "FRG-102": {"am_temp_c": 3.8, "pm_temp_c": 6.2},
    "FRG-103": {"am_temp_c": 5.0, "pm_temp_c": 7.4},
    "FRG-104": {"am_temp_c": 2.9, "pm_temp_c": 4.4},
}
FRIDGE_ID = "fridge_id"
TEMP_COLUMNS = ("am_temp_c", "pm_temp_c")
BLOCK_IDS = ("FRG-102", "FRG-103")

# data/supply_order.csv: item_id -> (item, quantity, unit_price_usd).
SUPPLY_ORDER = {
    "C3150": ("Nitrile exam gloves (box of 100)", 6, 9.50),
    "C1022": ("Blood pressure cuff (adult)", 1, 24.00),
    "C2210": ("Gauze pads 4x4 in (pack of 25)", 4, 6.00),
    "C4105": ("Syringes 3 mL (box of 100)", 2, 18.00),
    "C1407": ("Digital thermometer", 1, 35.00),
    "C2318": ("Alcohol prep pads (box of 200)", 3, 4.25),
    "C2904": ("Adhesive bandages (box of 100)", 4, 3.25),
    "C3012": ("Surgical masks (box of 50)", 8, 4.50),
    "C2877": ("Exam table paper (roll)", 3, 8.00),
    "C1560": ("Pulse oximeter", 1, 42.00),
    "C2655": ("Tongue depressors (box of 500)", 2, 6.50),
    "C1833": ("Specimen cups (case of 100)", 2, 28.50),
}
SUPPLY_COLUMNS = ("item_id", "item", "quantity", "unit_price_usd", "line_total_usd")
MIN_QUANTITY = 2

# Temperatures have one decimal and prices whole cents, so anything within half
# of the last digit is the same value.
TOLERANCE = 0.005


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
    conflicting_columns: tuple[str, ...] = ()
    # How many times the table is in the file: to_csv(mode="a") writes it again below the first copy.
    copies: int = 1
    # The fridge block saved sideways, with .T: rows are reading columns and columns are fridge IDs.
    transposed: bool = False


def line_total(item_id: str) -> float:
    _, quantity, unit_price = SUPPLY_ORDER[item_id]
    return quantity * unit_price


def expected_selection() -> list[str]:
    """The item_ids Task 3 selects, in the order it sorts them."""
    selected = [item_id for item_id, (_, quantity, _) in SUPPLY_ORDER.items() if quantity >= MIN_QUANTITY]
    return sorted(selected, key=lambda item_id: (-line_total(item_id), item_id))


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
    return raw.decode("utf-8", errors="replace").lstrip("\ufeff")


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
        if path.is_file() and difflib.SequenceMatcher(None, name.casefold(), path.name.casefold()).ratio() >= 0.75
    )


def _missing(root: Path, name: str, task: str) -> str:
    """Say the artifact is missing and why it may be: a file of a similar name, or the other artifact's path."""
    wanted = root / name
    step, call = SAVED_BY[name]
    beside = _look_alikes(root, wanted.parent, wanted.name)
    if beside:
        return (
            f"{name} is missing; run the {task} cells to write it, then commit it. "
            f"Found {', '.join(beside)}; save it as {name} instead."
        )
    misplaced = _look_alikes(root, root, wanted.name) if wanted.parent != root else []
    if misplaced:
        return (
            f"{name} is missing, but the assignment folder itself has {_join(misplaced)}; "
            f"in {step}, save with {call}, which writes {name}, then commit it."
        )
    if name == SUPPLIES_FILE and _artifact(root, FRIDGE_FILE) is not None:
        try:
            fridge = read_table(root, FRIDGE_FILE, "Task 2")
        except (AssertionError, OSError, csv.Error):
            fridge = None
        if fridge is not None and _holds_supply_lines(fridge):
            return f"{name} is missing, and {FRIDGE_FILE} holds the supply order lines instead. {SAVE_BOTH}"
    return f"{name} is missing; run the {task} cells to write it, then commit it. {step} saves it with {call}."


def _clean(cell: str) -> str:
    return " ".join(cell.split())


def _is_whole_number(cell: str) -> bool:
    number = _number(cell)
    return number is not None and number.is_integer()


def read_table(root: Path, name: str, task: str) -> Table:
    """Parse a saved CSV, however it is separated, spaced, quoted, or ended.

    A leading column headed by nothing, `Unnamed: 0`, or `index` that holds only
    whole numbers is the row numbers pandas writes when `index=False` is left
    out, so it is set aside rather than counted as a column. `reset_index()`
    followed by `to_csv()` writes two such columns, so each one is set aside.
    """
    path = _artifact(root, name)
    _assert(path is not None, _missing(root, name, task))
    try:
        lines = _csv_rows(_decode(path.read_bytes()))
    except csv.Error as error:
        raise AssertionError(
            f"{name} cannot be read as a CSV table ({error}); run the {task} cells again so to_csv() "
            "writes it, then compare it with the checkpoint in README.md."
        ) from None
    _assert(
        bool(lines),
        f"{name} is empty, with no header line and no rows; run the {task} cells again so to_csv() writes it, "
        "then commit it.",
    )
    header = [_clean(cell).casefold() for cell in lines[0]]
    # A table written again with to_csv(mode="a") repeats its header line; the last copy is the latest write.
    starts = [i for i, row in enumerate(lines) if [_clean(cell).casefold() for cell in row] == header]
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
        and all(row and _is_whole_number(row[0]) for row in body)
    ):
        header = header[1:]
        body = [row[1:] for row in body]

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
    header = list(dict.fromkeys(header))
    return Table(name=name, columns=tuple(header), rows=rows, conflicting_columns=tuple(sorted(conflicting)),
                 copies=copies)


def _written_again(table: Table) -> str:
    """The fix when the file holds its table more than once, as to_csv(mode="a") leaves it."""
    step, call = SAVED_BY[table.name]
    return (
        f"{table.name} holds the same table {table.copies} times, so it was written into the file "
        f'{table.copies} times: to_csv() with mode="a" adds to the end of the file instead of replacing it. '
        f"The checks graded the last copy; in {step}, save once with {call}, which replaces the file."
    )


def _number(cell: str) -> float | None:
    text = cell.strip().removeprefix("$").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _same(given: str, expected: float) -> bool:
    number = _number(given)
    return number is not None and abs(number - expected) <= TOLERANCE


def _join(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _columns(table: Table) -> str:
    """The file's column names for a message, naming a blank header rather than printing nothing."""
    return _join(column or "a column with no header" for column in table.columns) or "none"


# Task 2: output/fridge_block.csv


def _text_columns(table: Table) -> list[str]:
    """The columns whose every cell is text rather than a number, such as a saved index of labels."""
    return [
        column
        for column in table.columns
        if table.rows and all(row[column] and _number(row[column]) is None for row in table.rows)
    ]


def _fridge_label_column(table: Table) -> str | None:
    """The column holding fridge IDs: fridge_id, or else a column of text that names a supplied fridge."""
    if FRIDGE_ID in table.columns:
        return FRIDGE_ID
    known = {fridge.casefold() for fridge in FRIDGE_READINGS}
    for column in _text_columns(table):
        if any(row[column].casefold() in known for row in table.rows):
            return column
    return None


def _holds_supply_lines(table: Table) -> bool:
    """The file has Task 3's columns and nothing of the fridge block: the supply lines saved to the wrong path."""
    fridge_columns = (FRIDGE_ID, *TEMP_COLUMNS)
    return (
        any(column in table.columns for column in SUPPLY_COLUMNS)
        and not any(column in table.columns for column in fridge_columns)
        and _fridge_label_column(table) is None
    )


def _other_table(table: Table) -> str | None:
    """The one fix when the fridge block's file holds another table, or None when it may be the block.

    A file with no fridge IDs and no reading columns, under their own names or
    standing in for them, holds another table: the supply lines saved to the
    fridge block's path, or another object, such as latest_by_area, saved in
    Task 2.2. It replaces the message of each fridge check that fails, never
    whether a check passes.
    """
    if _holds_supply_lines(table):
        return f"{FRIDGE_FILE} holds the supply order lines ({_columns(table)}), not the fridge block. {SAVE_BOTH}"
    if _fridge_label_column(table) is None and all(saved is None for saved in _temperature_columns(table).values()):
        step, call = SAVED_BY[FRIDGE_FILE]
        return (
            f"{FRIDGE_FILE} holds another table: its columns are {_columns(table)}, with no fridge IDs and no "
            f"{' or '.join(TEMP_COLUMNS)} readings. In {step}, save the fridge block with {call}."
        )
    return None


def _fridge_fix(table: Table, message: str) -> str:
    """The check's own message, unless the file holds another table than the fridge block."""
    return _other_table(table) or message


def _column_label(column: str) -> str:
    return f"a column named {column}" if column else "a column with no header"


def _temperature_columns(table: Table) -> dict[str, str | None]:
    """Where each reading is saved: {expected column: its column in the file, or None}.

    A reading column saved under another name, such as am_temp for am_temp_c,
    stands in for its expected name when the missing names and the unexpected
    columns of numbers pair up one to one, in the order Task 2.1 gives. The
    rename then costs only that column's name check, and its values are still
    graded.
    """
    saved: dict[str, str | None] = {column: column for column in TEMP_COLUMNS if column in table.columns}
    missing = [column for column in TEMP_COLUMNS if column not in saved]
    label = _fridge_label_column(table)
    numbers = [
        column
        for column in table.columns
        if column not in TEMP_COLUMNS
        and column not in (FRIDGE_ID, label)
        and all(_number(row[column]) is not None for row in table.rows)
    ]
    if missing and len(numbers) == len(missing):
        saved.update(zip(missing, numbers))
    return {column: saved.get(column) for column in TEMP_COLUMNS}


def _fridge_rows(table: Table) -> dict[str, dict[str, str]]:
    """{fridge id: row} for every row that names or, without fridge IDs, matches a supplied fridge.

    Without a column of fridge IDs, a row is identified by its readings. No two
    supplied readings are the same, so a row that matches exactly one fridge on
    either reading is that fridge. When the file has exactly two rows, the
    block's size, and no column of text labels, a row that matches no fridge is
    the block's fridge at its position, FRG-102 first. Leaving the index out
    then costs only the fridge_id check, and wrong readings cost only their
    columns' value checks. Rows labeled with something else, such as another
    Series' labels, are never taken by position.
    """
    label = _fridge_label_column(table)
    if label is not None:
        known = {fridge.casefold(): fridge for fridge in FRIDGE_READINGS}
        identified = [known.get(row[label].casefold()) for row in table.rows]
    else:
        saved = _temperature_columns(table)
        identified = []
        for row in table.rows:
            matches = [
                fridge
                for fridge, readings in FRIDGE_READINGS.items()
                if any(saved[column] is not None and _same(row[saved[column]], readings[column]) for column in TEMP_COLUMNS)
            ]
            identified.append(matches[0] if len(matches) == 1 else None)
        if len(identified) == len(BLOCK_IDS) and not _text_columns(table):
            identified = [
                fridge if fridge is not None or BLOCK_IDS[position] in identified else BLOCK_IDS[position]
                for position, fridge in enumerate(identified)
            ]
    found: dict[str, dict[str, str]] = {}
    for fridge, row in zip(identified, table.rows):
        if fridge is not None:
            found.setdefault(fridge, row)
    return found


def read_fridge(root: Path) -> Table:
    """The saved fridge block, turned back the right way when it was saved sideways with .T.

    The transposed block holds the right readings, so only the fridge_id check
    charges it, and the other checks grade the readings as if it were upright.
    """
    table = read_table(root, FRIDGE_FILE, "Task 2")
    if not table.columns or not table.rows:
        return table
    label, fridges = table.columns[0], table.columns[1:]
    known = {fridge.casefold() for fridge in FRIDGE_READINGS}
    if not (all(row[label].casefold() in TEMP_COLUMNS for row in table.rows) and any(f in known for f in fridges)):
        return table
    readings = [row[label].casefold() for row in table.rows]
    rows = tuple(
        {FRIDGE_ID: fridge, **{column: row[fridge] for column, row in zip(readings, table.rows)}} for fridge in fridges
    )
    return replace(table, columns=(FRIDGE_ID, *dict.fromkeys(readings)), rows=rows, conflicting_columns=(),
                   transposed=True)


def check_fridge_id_column(root: Path) -> None:
    """The saved block keeps its named index as a fridge_id column."""
    table = read_fridge(root)
    if table.transposed:
        step, call = SAVED_BY[FRIDGE_FILE]
        raise AssertionError(
            f"{FRIDGE_FILE} is saved sideways: its rows are {_join(TEMP_COLUMNS)} and its columns are the fridge "
            f"IDs, as .T leaves the block. Expected one row per fridge under the header fridge_id,"
            f"{','.join(TEMP_COLUMNS)}; in {step}, save the block itself with {call}, without .T."
        )
    if FRIDGE_ID in table.columns:
        return
    label = _fridge_label_column(table)
    text = _text_columns(table)
    if label is not None:
        where = f"a column headed {label}" if label else "a first column with no header"
        message = (
            f"{FRIDGE_FILE} saves the fridge IDs in {where}, not fridge_id; "
            'name the index with fridge_log.index.name = "fridge_id" in Task 2.1, then save again.'
        )
    elif text and text[0] == table.columns[0]:
        shown = list(dict.fromkeys(row[text[0]] for row in table.rows))
        message = (
            f"{FRIDGE_FILE} has no fridge_id column, and its row labels ({_join(shown[:4])}) are not fridge IDs; "
            "in Task 2.1, build fridge_log with the index FRG-101, FRG-102, FRG-103, FRG-104 and name it with "
            'fridge_log.index.name = "fridge_id", then save the Task 2.2 block again.'
        )
    else:
        message = (
            f"{FRIDGE_FILE} has no fridge_id column (its columns are {_columns(table)}); "
            "in Task 2.2, save label_block with its index, so leave out index=False."
        )
    raise AssertionError(_fridge_fix(table, message))


def check_fridge_rows(root: Path) -> None:
    """The block holds exactly FRG-102 and FRG-103."""
    table = read_fridge(root)
    found = _fridge_rows(table)
    missing = [fridge for fridge in BLOCK_IDS if fridge not in found]
    extra = [fridge for fridge in found if fridge not in BLOCK_IDS]
    problems = []
    if missing:
        problems.append(f"it is missing {_join(missing)}")
    if extra:
        problems.append(f"it also holds {_join(sorted(extra))}")
    # Too few rows already show as missing fridges; only extra or repeated rows need the count.
    if len(table.rows) > len(BLOCK_IDS) and not extra:
        problems.append(f"it holds {len(table.rows)} rows, expected exactly {len(BLOCK_IDS)}; remove the extra or repeated rows")
    if not found:
        problems = [f"none of its {len(table.rows)} rows is one of the fridges FRG-101 to FRG-104"]
    if table.copies > 1 and not problems:
        raise AssertionError(_fridge_fix(table, _written_again(table)))
    if not missing and {"FRG-101", "FRG-104"} <= set(extra):
        # Extras on both sides of the block are not a slice endpoint: the whole table was saved.
        raise AssertionError(_fridge_fix(
            table,
            f"{FRIDGE_FILE} should hold the rows FRG-102 and FRG-103, but it holds all four fridges, "
            f"so it looks like fridge_log was saved; {SAVED_BY[FRIDGE_FILE][0]} saves the selected block with "
            f"{SAVED_BY[FRIDGE_FILE][1]}.",
        ))
    if problems:
        raise AssertionError(_fridge_fix(
            table,
            f"{FRIDGE_FILE} should hold the rows FRG-102 and FRG-103, but " + "; ".join(problems) + ". "
            'In Task 2.2, .loc["FRG-102":"FRG-103"] includes its end label, while .iloc[1:3] stops before position 3.',
        ))


def _missing_temperature(table: Table, column: str) -> str:
    return (
        f"{FRIDGE_FILE} has no {column} column (its columns are {_columns(table)}); "
        f"keep both {_join(TEMP_COLUMNS)} in the Task 2.2 selection."
    )


def _temperature_name_check(column: str) -> Callable[[Path], None]:
    """The saved block has this reading column, under its own name."""

    def check(root: Path) -> None:
        table = read_fridge(root)
        if column in table.columns:
            return
        stand_in = _temperature_columns(table)[column]
        if stand_in is not None:
            names = ", ".join(f'"{name}"' for name in TEMP_COLUMNS)
            raise AssertionError(_fridge_fix(
                table,
                f"{FRIDGE_FILE} has {_column_label(stand_in)} where Task 2.1 names it {column}; build fridge_log "
                f"with columns=[{names}], then run Task 2.2 again to save it.",
            ))
        raise AssertionError(_fridge_fix(table, _missing_temperature(table, column)))

    return check


def _swapped_readings(table: Table) -> bool:
    """The two reading columns hold each other's block values, one mistake charged on am_temp_c values alone."""
    saved = _temperature_columns(table)
    if None in saved.values():
        return False
    found = _fridge_rows(table)
    if found:
        rows = [found.get(fridge) for fridge in BLOCK_IDS]
    elif _fridge_label_column(table) is None:
        rows = list(table.rows)
    else:
        return False
    if len(rows) != len(BLOCK_IDS) or None in rows:
        return False
    am, pm = TEMP_COLUMNS
    return all(
        _same(row[saved[am]], FRIDGE_READINGS[fridge][pm]) and _same(row[saved[pm]], FRIDGE_READINGS[fridge][am])
        for fridge, row in zip(BLOCK_IDS, rows)
    )


def _check_temperature(root: Path, column: str) -> None:
    table = read_fridge(root)
    if _swapped_readings(table):
        if column != TEMP_COLUMNS[0]:
            return  # Charged once, on the am_temp_c values check.
        am, pm = TEMP_COLUMNS
        names = ", ".join(f'"{name}"' for name in TEMP_COLUMNS)
        raise AssertionError(_fridge_fix(
            table,
            f"{FRIDGE_FILE} has the column labels swapped: its {am} column holds "
            f"{_join(f'{FRIDGE_READINGS[f][pm]:.1f}' for f in BLOCK_IDS)}, the {pm} readings, and its {pm} column "
            f"holds {_join(f'{FRIDGE_READINGS[f][am]:.1f}' for f in BLOCK_IDS)}, the {am} readings. The supplied "
            f"array's first column is the morning reading, so build fridge_log with columns=[{names}] in that order, "
            "then run Task 2.2 again to save it.",
        ))
    saved = _temperature_columns(table)[column]
    if column in table.conflicting_columns:
        raise AssertionError(
            f"{FRIDGE_FILE} has repeated {column} headers with conflicting readings; expected one reading "
            "column with one value per fridge. Remove the extra column and save the selected block again."
        )
    if saved is None:
        if _other_table(table) is None and any(fridge in _fridge_rows(table) for fridge in BLOCK_IDS):
            return  # The column-name check charges an omission in recognizable data.
        raise AssertionError(_fridge_fix(table, _missing_temperature(table, column)))
    where = column if saved == column else f"{saved or '(no header)'} (saved for {column})"
    found = {fridge: row for fridge, row in _fridge_rows(table).items() if fridge in BLOCK_IDS}
    if not found and _fridge_label_column(table) is None:
        # No labels and no row matches a fridge on either reading: compare this column alone, in order.
        given = [row[saved] for row in table.rows]
        expected = [FRIDGE_READINGS[fridge][column] for fridge in BLOCK_IDS]
        _assert(
            len(given) == len(expected) and all(_same(value, want) for value, want in zip(given, expected)),
            _fridge_fix(
                table,
                f"{FRIDGE_FILE} lists {where} as {_join(given) or 'nothing'}; "
                f"FRG-102 and FRG-103 read {_join(f'{value:.1f}' for value in expected)} °C in the supplied array. "
                "Build fridge_log from the supplied fridge_readings array with the columns in the order Task 2.1 "
                "gives, keep the array unchanged, and save the Task 2.2 block with its index.",
            ),
        )
        return
    if not found:
        raise AssertionError(_fridge_fix(
            table,
            f"{FRIDGE_FILE} has no row for FRG-102 or FRG-103, so its {column} values cannot be compared; "
            'in Task 2.2, select the block with fridge_log.loc["FRG-102":"FRG-103", ...] and save it again.',
        ))
    wrong = [
        f"{fridge} has {row[saved] or 'nothing'} where the supplied array has {FRIDGE_READINGS[fridge][column]:.1f} °C"
        for fridge, row in found.items()
        if not _same(row[saved], FRIDGE_READINGS[fridge][column])
    ]
    if wrong:
        raise AssertionError(_fridge_fix(
            table,
            f"{FRIDGE_FILE}, column {where}: {'; '.join(wrong)}. Build fridge_log from the supplied fridge_readings "
            "array with the columns in the order Task 2.1 gives, and keep the array unchanged.",
        ))


# Task 3: output/selected_supplies.csv


def _total_column(table: Table) -> str | None:
    """The line-total column: line_total_usd, or else the one extra column standing in for it.

    A stand-in is the one unexpected column named like a total or, when
    line_total_usd is the only name missing, the one unexpected column, as the
    column checks read it. A misnamed total then costs only the line_total_usd
    column check, and its values are still graded by the line totals check.
    """
    if "line_total_usd" in table.columns:
        return "line_total_usd"
    extra = [column for column in table.columns if column not in SUPPLY_COLUMNS]
    totals = [column for column in extra if "total" in column]
    if len(totals) == 1:
        return totals[0]
    missing = [column for column in SUPPLY_COLUMNS if column not in table.columns]
    return extra[0] if missing == ["line_total_usd"] and len(extra) == 1 else None


def _extra_columns(table: Table) -> list[str]:
    return [column or "a column with no header" for column in table.columns if column not in SUPPLY_COLUMNS]


def _supply_rows(table: Table) -> list[tuple[str, dict[str, str]]]:
    """(item_id, row) for every row naming a supplied item, in file order.

    Rows are named by item_id, or by the item's description when the file has
    no item_id column.
    """
    by_id = {item_id.casefold(): item_id for item_id in SUPPLY_ORDER}
    by_item = {_clean(item).casefold(): item_id for item_id, (item, _, _) in SUPPLY_ORDER.items()}
    if "item_id" in table.columns:
        column, lookup = "item_id", by_id
    elif "item" in table.columns:
        column, lookup = "item", by_item
    else:
        return []
    return [(lookup[row[column].casefold()], row) for row in table.rows if row[column].casefold() in lookup]


def _require_line_key(table: Table) -> None:
    _assert(
        "item_id" in table.columns or "item" in table.columns,
        f"{SUPPLIES_FILE} has no item_id column (its columns are {_columns(table)}), so its lines "
        "cannot be matched to data/supply_order.csv; keep item_id in the Task 3.1 .loc selection.",
    )


def _column_check(column: str, hint: str) -> Callable[[Path], None]:
    """The saved selection has this one Task 3 column, under its own name."""

    def check(root: Path) -> None:
        table = read_table(root, SUPPLIES_FILE, "Task 3")
        if column in table.columns:
            return
        stand_in = _total_column(table) if column == "line_total_usd" else None
        missing = [name for name in SUPPLY_COLUMNS if name not in table.columns]
        extra = [name for name in table.columns if name not in SUPPLY_COLUMNS]
        if stand_in is None and len(missing) == 1 and len(extra) == 1:
            stand_in = extra[0]
        if stand_in is not None:
            raise AssertionError(
                f"{SUPPLIES_FILE} has {_column_label(stand_in)} where Task 3.1 names it {column}; "
                f"rename it to {column}. {hint}"
            )
        raise AssertionError(
            f"{SUPPLIES_FILE} has no {column} column (its columns are {_columns(table)}). {hint}"
        )

    return check


def check_supply_extra_columns(root: Path) -> None:
    """No column beyond the five, other than one standing in for a missing, misnamed column."""
    table = read_table(root, SUPPLIES_FILE, "Task 3")
    missing = [column for column in SUPPLY_COLUMNS if column not in table.columns]
    extra = _extra_columns(table)
    _assert(
        len(extra) <= len(missing),
        f"{SUPPLIES_FILE} also has {_join(extra)}; Task 3.1 saves only the five columns "
        f"{_join(SUPPLY_COLUMNS)}, so leave {'it' if len(extra) == 1 else 'them'} out of the selection.",
    )


def _no_lines(table: Table) -> str | None:
    """The shared fix when the file names no supplied order line at all, or None when it names some."""
    if _supply_rows(table):
        return None
    key = "item_id" if "item_id" in table.columns else "item"
    found = "a header but no order lines" if not table.rows else (
        f"{len(table.rows)} rows, but none has an {key} from data/supply_order.csv"
    )
    kept = len(expected_selection())
    return (
        f"{SUPPLIES_FILE} has {found}; Task 3.1 keeps the {kept} lines with quantity {MIN_QUANTITY} or more. "
        f'Check the mask quantity_at_least_two = supplies["quantity"] >= {MIN_QUANTITY} (the cell prints '
        f"selected lines: {kept}), select with supplies.loc[quantity_at_least_two, [...]], and keep "
        "data/supply_order.csv unchanged."
    )


def _inverted(lines: list[tuple[str, dict[str, str]]]) -> bool:
    """The file holds exactly the lines Task 3.1 drops, as a mask written < 2 for >= 2 selects."""
    named = {item_id for item_id, _ in lines}
    return bool(named) and named == set(SUPPLY_ORDER) - set(expected_selection())


def _line_check(item_id: str) -> Callable[[Path], None]:
    """This selected order line is in the file with its supplied item, quantity, and unit price.

    Its line total is graded by the line totals check, so a missing or wrong
    total never costs a line check.
    """
    item, quantity, unit_price = SUPPLY_ORDER[item_id]

    def check(root: Path) -> None:
        table = read_table(root, SUPPLIES_FILE, "Task 3")
        _require_line_key(table)
        nothing = _no_lines(table)
        if nothing:
            raise AssertionError(nothing)
        lines = _supply_rows(table)
        if _inverted(lines):
            return  # One mistake, charged once by the no-other-lines check.
        rows = [row for found, row in lines if found == item_id]
        if not rows:
            # Only a mask written with > keeps exactly the larger lines and drops every quantity-2 line;
            # a truncated table also lacks larger lines, so each of its missing lines is charged.
            above = {found for found in expected_selection() if SUPPLY_ORDER[found][1] > MIN_QUANTITY}
            greater_than = quantity == MIN_QUANTITY and {found for found, _ in lines} == above
            # That one mistake drops every quantity-2 line, so it is charged once, to the first of them.
            at_minimum = [found for found in expected_selection() if SUPPLY_ORDER[found][1] == MIN_QUANTITY]
            if greater_than and item_id != at_minimum[0]:
                return
            why = (
                f'supplies["quantity"] >= {MIN_QUANTITY} keeps it, while > {MIN_QUANTITY} drops it, '
                f"along with {_join(at_minimum[1:])}"
                if greater_than
                else f'build the mask quantity_at_least_two = supplies["quantity"] >= {MIN_QUANTITY} and select with it'
            )
            raise AssertionError(
                f"{SUPPLIES_FILE} has no line for {item_id} ({item}, quantity {quantity}); "
                f"Task 3.1 keeps every line with quantity {MIN_QUANTITY} or more: {why}."
            )
        wrong = []
        for row in rows:
            if "item" in row and row["item"].casefold() != _clean(item).casefold():
                wrong.append(f"item is {row['item'] or 'blank'}, expected {item}")
            if "quantity" in row and not _same(row["quantity"], quantity):
                wrong.append(f"quantity is {row['quantity'] or 'blank'}, expected {quantity}")
            if "unit_price_usd" in row and not _same(row["unit_price_usd"], unit_price):
                wrong.append(f"unit_price_usd is {row['unit_price_usd'] or 'blank'}, expected {unit_price:.2f}")
        _assert(
            not wrong,
            f"{SUPPLIES_FILE}: {item_id}'s " + "; its ".join(dict.fromkeys(wrong)) + ", as in "
            "data/supply_order.csv. Keep the data file unchanged and copy the selected rows as they are in Task 3.1.",
        )

    return check


def check_supply_line_totals(root: Path) -> None:
    """Every saved order line's total is its quantity times its unit price.

    A total is right when it matches the supplied quantity and price or the
    line's own saved ones, so a wrong quantity costs only its line check.
    """
    table = read_table(root, SUPPLIES_FILE, "Task 3")
    _require_line_key(table)
    nothing = _no_lines(table)
    if nothing:
        raise AssertionError(nothing)
    total = _total_column(table)
    if total is None:
        return  # The column check already charges the absent total column.
    wrong = []
    for item_id, row in _supply_rows(table):
        _, quantity, unit_price = SUPPLY_ORDER[item_id]
        expected = quantity * unit_price
        saved_quantity, saved_price = _number(row.get("quantity", "")), _number(row.get("unit_price_usd", ""))
        own = saved_quantity * saved_price if saved_quantity is not None and saved_price is not None else expected
        if not (_same(row[total], expected) or _same(row[total], own)):
            wrong.append(
                f"{item_id}'s {total} is {row[total] or 'blank'}, expected {quantity} * {unit_price:.2f} = {expected:.2f}"
            )
    wrong = list(dict.fromkeys(wrong))
    shown = "; ".join(wrong[:3]) + (f"; and {len(wrong) - 3} more lines" if len(wrong) > 3 else "")
    _assert(
        not wrong,
        f"{SUPPLIES_FILE}: {shown}. In Task 3.1, line_total_usd = quantity * unit_price_usd for each line.",
    )


def check_supply_other_lines(root: Path) -> None:
    """No line with quantity 1, no line twice, and no row naming an item outside the supply order."""
    table = read_table(root, SUPPLIES_FILE, "Task 3")
    _require_line_key(table)
    rows = _supply_rows(table)
    named = [item_id for item_id, _ in rows]
    expected = expected_selection()
    extra = [item_id for item_id in SUPPLY_ORDER if item_id in named and item_id not in expected]
    repeated = [item_id for item_id in dict.fromkeys(named) if named.count(item_id) > 1]
    key = "item_id" if "item_id" in table.columns else "item"
    matched = {id(row) for _, row in rows}
    unknown = [row[key] or "blank" for row in table.rows if id(row) not in matched]
    if _inverted(rows):
        kept = expected_selection()
        raise AssertionError(
            f"{SUPPLIES_FILE} holds only {_join(extra)}, the lines with quantity 1, and none of the {len(kept)} "
            f"lines with quantity {MIN_QUANTITY} or more, so the mask is inverted: a comparison such as "
            f'supplies["quantity"] < {MIN_QUANTITY} keeps exactly the lines Task 3.1 drops. Build it as '
            f'quantity_at_least_two = supplies["quantity"] >= {MIN_QUANTITY} (the cell prints selected lines: '
            f"{len(kept)}), then save again."
        )
    problems = []
    if extra:
        verb = "has" if len(extra) == 1 else "have"
        problems.append(f"also holds {_join(extra)}, which {verb} quantity 1")
    if repeated:
        problems.append("lists " + _join(f"{item_id} {named.count(item_id)} times" for item_id in repeated))
    if unknown:
        shown = _join(unknown[:4]) + (f" and {len(unknown) - 4} more" if len(unknown) > 4 else "")
        problems.append(f"has {key} {shown}, which {'is' if len(unknown) == 1 else 'are'} not in data/supply_order.csv")
    again = _written_again(table) if table.copies > 1 else ""
    if again and not problems:
        raise AssertionError(again)
    _assert(
        not problems,
        (again + " In that copy, the file " if again else f"{SUPPLIES_FILE} ") + "; ".join(problems)
        + f". Task 3.1 keeps only the lines with quantity {MIN_QUANTITY} or more, once each: select them with the "
        f'mask quantity_at_least_two = supplies["quantity"] >= {MIN_QUANTITY}.',
    )


def _order_bases(table: Table, rows: list[tuple[str, dict[str, str]]]) -> list[dict[str, float]]:
    """The totals the order is judged by: the true line totals, and the file's own when every one is a number.

    Judging by the file's own totals too means a wrong total costs only the line totals check.
    """
    bases = [{item_id: line_total(item_id) for item_id, _ in rows}]
    total = _total_column(table)
    if total is not None:
        saved = {item_id: _number(row[total]) for item_id, row in rows}
        if all(value is not None for value in saved.values()):
            bases.append(saved)
    return bases


def _unsorted(ids: list[str]) -> str | None:
    """The fix when the lines are still in data/supply_order.csv order, which sort_values() not assigned back leaves."""
    if ids != [item_id for item_id in SUPPLY_ORDER if item_id in ids]:
        return None
    first = ", ".join(ids[:3]) + (", ..." if len(ids) > 3 else "")
    return (
        f"{SUPPLIES_FILE} lists its lines in the order data/supply_order.csv has them ({first}), so "
        "they were saved unsorted. In Task 3.2, sort_values() returns a new, sorted table and leaves "
        "selected_supplies as it was, so assign the result back: selected_supplies = selected_supplies.sort_values("
        'by=["line_total_usd", "item_id"], ascending=[False, True]), then save again.'
    )


def _ordered_rows(root: Path) -> tuple[list[str], list[dict[str, float]]]:
    table = read_table(root, SUPPLIES_FILE, "Task 3")
    _require_line_key(table)
    rows = _supply_rows(table)
    _assert(
        len(rows) >= 2,
        f"{SUPPLIES_FILE} names {len(rows)} line{'' if len(rows) == 1 else 's'} from data/supply_order.csv, so "
        f"its order cannot be checked; Task 3.1 keeps {len(expected_selection())} lines, and the line checks say "
        "what to fix.",
    )
    return [item_id for item_id, _ in rows], _order_bases(table, rows)


def _first_rise(ids: list[str], totals: dict[str, float]) -> tuple[str, str] | None:
    return next(((a, b) for a, b in zip(ids, ids[1:]) if totals[a] < totals[b] - TOLERANCE), None)


def _descending(ids: list[str], bases: list[dict[str, float]]) -> bool:
    """True when the lines run from the highest line total down, by recomputed or saved totals."""
    return any(_first_rise(ids, totals) is None for totals in bases)


def check_supply_descending(root: Path) -> None:
    """Each line's total is at least the next line's: the highest line total comes first."""
    ids, bases = _ordered_rows(root)
    if _descending(ids, bases):
        return
    rises = [_first_rise(ids, totals) for totals in bases]
    unsorted = _unsorted(ids)
    if unsorted:
        raise AssertionError(unsorted)
    first, second = rises[0]
    raise AssertionError(
        f"{SUPPLIES_FILE} is not sorted from the highest line_total_usd to the lowest: {first} "
        f"(line total {line_total(first):.2f}) comes before {second} (line total {line_total(second):.2f}). "
        'In Task 3.2, sort with by=["line_total_usd", "item_id"] and ascending=[False, True].'
    )


def check_supply_ties(root: Path) -> None:
    """Lines with the same total appear in item_id order, A to Z."""
    ids, bases = _ordered_rows(root)

    def first_tie_out_of_order(totals: dict[str, float]) -> tuple[str, str] | None:
        for i, first in enumerate(ids):
            for second in ids[i + 1:]:
                if abs(totals[first] - totals[second]) <= TOLERANCE and first > second:
                    return first, second
        return None

    found = [first_tie_out_of_order(totals) for totals in bases]
    if None in found:
        return
    if not _descending(ids, bases):
        return  # A file not sorted by line total is charged once, by the descending check.
    first, second = found[0]
    raise AssertionError(
        f"{SUPPLIES_FILE} lists {first} before {second}, but both have the line total {line_total(first):.2f}, "
        f"so the tie goes to the smaller item_id, {second}. In Task 3.2, break ties with "
        'by=["line_total_usd", "item_id"] and ascending=[False, True].'
    )


SELECT_HINT = "Task 3.1 selects item_id, item, quantity, and unit_price_usd with one .loc."
CHECKS = (
    Check("fridge block: fridge_id index column", check_fridge_id_column),
    Check("fridge block: rows FRG-102 and FRG-103", check_fridge_rows),
    *(Check(f"fridge block: {column} column", _temperature_name_check(column)) for column in TEMP_COLUMNS),
    Check("fridge block: am_temp_c values", lambda root: _check_temperature(root, "am_temp_c")),
    Check("fridge block: pm_temp_c values", lambda root: _check_temperature(root, "pm_temp_c")),
    *(Check(f"selected supplies: {column} column", _column_check(column, SELECT_HINT)) for column in SUPPLY_COLUMNS[:4]),
    Check(
        "selected supplies: line_total_usd column",
        _column_check("line_total_usd", "Task 3.1 adds line_total_usd = quantity * unit_price_usd."),
    ),
    Check("selected supplies: no extra columns", check_supply_extra_columns),
    *(Check(f"selected supplies: line {item_id}", _line_check(item_id)) for item_id in expected_selection()),
    Check("selected supplies: line totals", check_supply_line_totals),
    Check("selected supplies: no other lines", check_supply_other_lines),
    Check("selected supplies: highest line total first", check_supply_descending),
    Check("selected supplies: ties in item_id order", check_supply_ties),
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
