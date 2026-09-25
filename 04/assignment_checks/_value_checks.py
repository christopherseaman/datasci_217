"""Checks for Assignment 04.

The course keeps these checks in 04/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They read the two CSV files a
submission saves in output/ and compare them with values recomputed from the
supplied fridge readings and supply order below. The self-test confirms those
copies match assignment.ipynb and data/supply_order.csv.

Each check scores one thing, so one mistake costs only its own points. Values
are compared after parsing: spacing, line endings, quoting, column order, a
leading row-number column, number formatting (2 == 2.0 == 2.00), and the
letter case of labels never cost points, and cells may be separated by
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


FRIDGE_FILE = "output/fridge_block.csv"
SUPPLIES_FILE = "output/selected_supplies.csv"

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


def _missing(root: Path, name: str, task: str) -> str:
    message = f"{name} is missing; run the {task} cells to write it, then commit it."
    wanted = root / name
    if wanted.parent.is_dir():
        look_alikes = sorted(
            path.relative_to(root).as_posix()
            for path in wanted.parent.iterdir()
            if path.is_file()
            and difflib.SequenceMatcher(None, wanted.name.casefold(), path.name.casefold()).ratio() >= 0.75
        )
        if look_alikes:
            message += f" Found {', '.join(look_alikes)}; save it as {name} instead."
    return message


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
    lines = _csv_rows(_decode(path.read_bytes()))
    _assert(bool(lines), f"{name} is empty; run the {task} cells again to write it.")
    header = [_clean(cell).casefold() for cell in lines[0]]
    body = [[_clean(cell) for cell in row] for row in lines[1:]]

    while (
        len(header) > 1
        and (header[0] in ("", "index") or header[0].startswith("unnamed"))
        and body
        and all(row and _is_whole_number(row[0]) for row in body)
    ):
        header = header[1:]
        body = [row[1:] for row in body]

    rows = tuple({column: (row[i] if i < len(row) else "") for i, column in enumerate(header)} for row in body)
    return Table(name=name, columns=tuple(header), rows=rows)


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


# Task 2: output/fridge_block.csv


def _fridge_label_column(table: Table) -> str | None:
    """The column holding fridge IDs: fridge_id, or else any column of text labels."""
    if FRIDGE_ID in table.columns:
        return FRIDGE_ID
    for column in table.columns:
        values = [row[column] for row in table.rows]
        if values and all(value and _number(value) is None for value in values):
            return column
    return None


def _fridge_rows(table: Table) -> dict[str, dict[str, str]]:
    """{fridge id: row} for every row that names or, without labels, matches a supplied fridge.

    Without a label column, a row is identified by its readings. No two
    supplied readings are the same, so a row that matches exactly one fridge on
    either reading is that fridge. Leaving the index out then costs only the
    fridge_id check, and one wrong reading costs only its own column's check.
    """
    label = _fridge_label_column(table)
    known = {fridge.casefold(): fridge for fridge in FRIDGE_READINGS}
    found: dict[str, dict[str, str]] = {}
    for row in table.rows:
        if label is not None:
            fridge = known.get(row[label].casefold())
        else:
            matches = [
                fridge
                for fridge, readings in FRIDGE_READINGS.items()
                if any(column in row and _same(row[column], readings[column]) for column in TEMP_COLUMNS)
            ]
            fridge = matches[0] if len(matches) == 1 else None
        if fridge is not None:
            found.setdefault(fridge, row)
    return found


def check_fridge_id_column(root: Path) -> None:
    """The saved block keeps its named index as a fridge_id column."""
    table = read_table(root, FRIDGE_FILE, "Task 2")
    if FRIDGE_ID in table.columns:
        return
    label = _fridge_label_column(table)
    if label is not None:
        where = f"a column headed {label}" if label else "a first column with no header"
        raise AssertionError(
            f"{FRIDGE_FILE} saves the fridge IDs in {where}, not fridge_id; "
            'name the index with fridge_log.index.name = "fridge_id" in Task 2.1, then save again.'
        )
    raise AssertionError(
        f"{FRIDGE_FILE} has no fridge_id column (its columns are {_join(table.columns) or 'none'}); "
        "in Task 2.2, save label_block with its index, so leave out index=False."
    )


def check_fridge_rows(root: Path) -> None:
    """The block holds exactly FRG-102 and FRG-103."""
    table = read_table(root, FRIDGE_FILE, "Task 2")
    found = _fridge_rows(table)
    missing = [fridge for fridge in BLOCK_IDS if fridge not in found]
    extra = [fridge for fridge in found if fridge not in BLOCK_IDS]
    problems = []
    if missing:
        problems.append(f"it is missing {_join(missing)}")
    if extra:
        problems.append(f"it also holds {_join(sorted(extra))}")
    if not found:
        problems = [f"none of its {len(table.rows)} rows is one of the fridges FRG-101 to FRG-104"]
    _assert(
        not problems,
        f"{FRIDGE_FILE} should hold the rows FRG-102 and FRG-103, but " + "; ".join(problems) + ". "
        'In Task 2.2, .loc["FRG-102":"FRG-103"] includes its end label, while .iloc[1:3] stops before position 3.',
    )


def _check_temperature(root: Path, column: str) -> None:
    table = read_table(root, FRIDGE_FILE, "Task 2")
    _assert(
        column in table.columns,
        f"{FRIDGE_FILE} has no {column} column (its columns are {_join(table.columns) or 'none'}); "
        f"keep both {_join(TEMP_COLUMNS)} in the Task 2.2 selection.",
    )
    found = {fridge: row for fridge, row in _fridge_rows(table).items() if fridge in BLOCK_IDS}
    if not found and _fridge_label_column(table) is None:
        # No labels and no row matches a fridge on either reading: compare this column alone, in order.
        given = [row[column] for row in table.rows]
        expected = [FRIDGE_READINGS[fridge][column] for fridge in BLOCK_IDS]
        _assert(
            len(given) == len(expected) and all(_same(value, want) for value, want in zip(given, expected)),
            f"{FRIDGE_FILE} lists {column} as {_join(given) or 'nothing'}; "
            f"FRG-102 and FRG-103 read {_join(f'{value:.1f}' for value in expected)} °C in the supplied array.",
        )
        return
    _assert(
        bool(found),
        f"{FRIDGE_FILE} has no row for FRG-102 or FRG-103, so its {column} values cannot be compared; "
        "the fridge rows check says what to fix.",
    )
    wrong = [
        f"{fridge} has {row[column] or 'nothing'} where the supplied array has {FRIDGE_READINGS[fridge][column]:.1f}"
        for fridge, row in found.items()
        if not _same(row[column], FRIDGE_READINGS[fridge][column])
    ]
    _assert(
        not wrong,
        f"{FRIDGE_FILE}: {'; '.join(wrong)} °C. Build fridge_log from the supplied fridge_readings array "
        "with the columns in the order Task 2.1 gives, and keep the array unchanged.",
    )


# Task 3: output/selected_supplies.csv


def _total_column(table: Table) -> str | None:
    """The line-total column: line_total_usd, or else the one unexpected column named like a total.

    A misnamed total then costs only the columns check, not the values check too.
    """
    if "line_total_usd" in table.columns:
        return "line_total_usd"
    totals = [column for column in table.columns if "total" in column and column not in SUPPLY_COLUMNS]
    return totals[0] if len(totals) == 1 else None


def _listed(problems: list[str], limit: int = 3) -> str:
    shown = "; ".join(problems[:limit])
    if len(problems) > limit:
        shown += f"; and {len(problems) - limit} more like {'it' if len(problems) - limit == 1 else 'them'}"
    return shown


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


def check_supply_columns(root: Path) -> None:
    """The saved selection has exactly the five Task 3 columns, in any order."""
    table = read_table(root, SUPPLIES_FILE, "Task 3")
    missing = [column for column in SUPPLY_COLUMNS if column not in table.columns]
    extra = [column or "(blank)" for column in table.columns if column not in SUPPLY_COLUMNS]
    problems = []
    if missing:
        problems.append(f"is missing {_join(missing)}")
    if extra:
        problems.append(f"also has {_join(extra)}")
    hints = []
    misnamed_total = _total_column(table) if "line_total_usd" in missing else None
    if misnamed_total is not None:
        hints.append(f"In Task 3.1, name the derived column line_total_usd rather than {misnamed_total}.")
    elif "line_total_usd" in missing:
        hints.append("Task 3.1 adds line_total_usd = quantity * unit_price_usd.")
    if [column for column in missing if column != "line_total_usd"] or [
        column for column in extra if column != misnamed_total
    ]:
        hints.append("Task 3.1 selects item_id, item, quantity, and unit_price_usd with .loc.")
    _assert(
        not problems,
        f"{SUPPLIES_FILE} " + " and ".join(problems) + "; "
        f"it should have the columns {_join(SUPPLY_COLUMNS)}. " + " ".join(hints),
    )


def check_supply_rows(root: Path) -> None:
    """The saved selection holds exactly the order lines with quantity 2 or more, once each."""
    table = read_table(root, SUPPLIES_FILE, "Task 3")
    _assert(
        "item_id" in table.columns or "item" in table.columns,
        f"{SUPPLIES_FILE} has no item_id column, so its rows cannot be matched to the supply order; "
        "keep item_id in the Task 3.1 selection.",
    )
    named = [item_id for item_id, _ in _supply_rows(table)]
    expected = expected_selection()
    missing = [item_id for item_id in expected if item_id not in named]
    extra = sorted(item_id for item_id in set(named) if item_id not in expected)
    repeated = sorted(item_id for item_id in set(named) if named.count(item_id) > 1)
    unknown = len(table.rows) - len(named)
    problems = []
    if missing:
        problems.append(f"is missing {_join(missing)}")
    if extra:
        problems.append(
            f"also holds {_join(f'{item_id} (quantity {SUPPLY_ORDER[item_id][1]})' for item_id in extra)}"
        )
    if repeated:
        problems.append(f"lists {_join(repeated)} more than once")
    if unknown:
        problems.append(f"has {unknown} row{'s' if unknown != 1 else ''} naming no item in data/supply_order.csv")
    _assert(
        not problems,
        f"{SUPPLIES_FILE} " + "; ".join(problems) + f". It should hold the {len(expected)} order lines with "
        f"quantity {MIN_QUANTITY} or more, once each: build the mask quantity_at_least_two = "
        f'supplies["quantity"] >= {MIN_QUANTITY} in Task 3.1 and select with it.',
    )


def check_supply_values(root: Path) -> None:
    """Each selected line keeps its supplied values, and line_total_usd = quantity * unit_price_usd."""
    table = read_table(root, SUPPLIES_FILE, "Task 3")
    total = _total_column(table)
    _assert(
        total is not None,
        f"{SUPPLIES_FILE} has no line_total_usd column to compare; "
        "Task 3.1 adds line_total_usd = quantity * unit_price_usd.",
    )
    selected = set(expected_selection())
    rows = [(item_id, row) for item_id, row in _supply_rows(table) if item_id in selected]
    _assert(
        bool(rows),
        f"{SUPPLIES_FILE} has no row naming one of the selected order lines, so its values cannot be "
        "compared; the rows check says what to fix.",
    )
    wrong = []
    for item_id, row in rows:
        item, quantity, unit_price = SUPPLY_ORDER[item_id]
        if "item" in row and row["item"].casefold() != _clean(item).casefold():
            wrong.append(f"{item_id}'s item is {row['item'] or 'blank'}, not {item}")
        if "quantity" in row and not _same(row["quantity"], quantity):
            wrong.append(f"{item_id}'s quantity is {row['quantity'] or 'blank'}, not {quantity}")
        if "unit_price_usd" in row and not _same(row["unit_price_usd"], unit_price):
            wrong.append(f"{item_id}'s unit_price_usd is {row['unit_price_usd'] or 'blank'}, not {unit_price:.2f}")
        if not _same(row[total], quantity * unit_price):
            wrong.append(
                f"{item_id}'s {total} is {row[total] or 'blank'}, not "
                f"{quantity} * {unit_price:.2f} = {quantity * unit_price:.2f}"
            )
    _assert(
        not wrong,
        f"{SUPPLIES_FILE}: " + _listed(wrong) + ". Copy the selected rows unchanged, then compute "
        'line_total_usd from the quantity and unit_price_usd columns in Task 3.1.',
    )


def _sorted_by(ids: list[str], totals: dict[str, float]) -> tuple[str, str] | None:
    """The first adjacent pair out of order by total (highest first), then item_id (A to Z), or None."""
    for first, second in zip(ids, ids[1:]):
        if (-totals[first], first) > (-totals[second], second):
            return first, second
    return None


def check_supply_order(root: Path) -> None:
    """Rows run from the highest line total to the lowest, with ties in item_id order.

    The order is judged by the true line totals, and also by the totals the
    file itself holds, so a wrong total or a wrong row costs its own check
    rather than this one too.
    """
    table = read_table(root, SUPPLIES_FILE, "Task 3")
    rows = _supply_rows(table)
    ids = [item_id for item_id, _ in rows]
    _assert(
        len(ids) >= 2,
        f"{SUPPLIES_FILE} names fewer than two items from data/supply_order.csv, so its order cannot be "
        "checked; the rows check says what to fix.",
    )
    true_totals = {item_id: line_total(item_id) for item_id in ids}
    out_of_order = _sorted_by(ids, true_totals)
    if out_of_order is None:
        return
    total = _total_column(table)
    saved = {item_id: _number(row[total]) if total else None for item_id, row in rows}
    if all(value is not None for value in saved.values()) and _sorted_by(ids, saved) is None:
        return
    first, second = out_of_order
    if true_totals[first] == true_totals[second]:
        reason = (
            f"{first} comes before {second}, but both total {true_totals[first]:.2f}, "
            f"so the tie goes to the smaller item_id, {second}"
        )
    else:
        reason = (
            f"{first} (total {true_totals[first]:.2f}) comes before {second} (total {true_totals[second]:.2f})"
        )
    raise AssertionError(
        f"{SUPPLIES_FILE} is not sorted by line_total_usd from highest to lowest with ties broken by item_id: "
        f"{reason}. In Task 3.2, sort with by=[\"line_total_usd\", \"item_id\"] and ascending=[False, True]."
    )


CHECKS = (
    Check("fridge block: fridge_id index column", check_fridge_id_column),
    Check("fridge block: rows FRG-102 and FRG-103", check_fridge_rows),
    Check("fridge block: am_temp_c values", lambda root: _check_temperature(root, "am_temp_c")),
    Check("fridge block: pm_temp_c values", lambda root: _check_temperature(root, "pm_temp_c")),
    Check("selected supplies: columns", check_supply_columns),
    Check("selected supplies: rows with quantity 2 or more", check_supply_rows),
    Check("selected supplies: values and line totals", check_supply_values),
    Check("selected supplies: sort order", check_supply_order),
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
