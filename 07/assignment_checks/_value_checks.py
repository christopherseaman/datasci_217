"""Checks for Assignment 07.

The course keeps these checks in 07/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They read the six files a
submission saves in output/ and compare them with the supplied data, whose
values are copied below; the self-test confirms those copies match data/.

Each check scores one thing, so one mistake costs only its own points. Values
are compared after parsing: spacing, line endings, quoting, key and column
order, a leading row-number column, number formatting (79 == 79.0 == 79.00),
and the letter case of labels never cost points, and a CSV's cells may be
separated by commas, semicolons, or tabs. A chart saved as a PNG is
checked for being a PNG image; how it looks is the student's to inspect.

Nothing here imports, runs, or inspects student code.
"""

from __future__ import annotations

import codecs
import csv
import difflib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path


SPEC_FILE = "output/exploratory_spec.json"
EVIDENCE_FILE = "output/visualization_evidence.json"
REDESIGN_FILE = "output/critique_redesign.png"
SUPPORTING_FILE = "output/explanatory_supporting_data.csv"
EXPLANATORY_FILE = "output/explanatory_chart.png"
TEXT_ALTERNATIVE_FILE = "output/explanatory_text_alternative.txt"

# data/rehab_patients.csv: patient_id -> (program, sessions_attended, walk_distance_m).
REHAB_PATIENTS = {
    "R01": ("Home-based", 14, 371),
    "R02": ("Center-based", 20, 402),
    "R03": ("Home-based", 32, 431),
    "R04": ("Center-based", 11, 360),
    "R05": ("Center-based", 28, 436),
    "R06": ("Home-based", 10, 352),
    "R07": ("Home-based", 21, 398),
    "R08": ("Center-based", 34, 458),
    "R09": ("Home-based", 16, 380),
    "R10": ("Center-based", 15, 384),
    "R11": ("Home-based", 27, 415),
    "R12": ("Center-based", 22, 413),
}
# The columns the Task 1 scatter plot draws, in the order of a REHAB_PATIENTS value.
PLOTTED_FIELDS = ("program", "sessions_attended", "walk_distance_m")

# data/followup_goals.csv without patients_seen: (program, visit_number, goal_met_pct).
FOLLOWUP_GOALS = (
    ("Home-based", 1, 58),
    ("Home-based", 2, 63),
    ("Home-based", 3, 67),
    ("Home-based", 4, 70),
    ("Center-based", 1, 57),
    ("Center-based", 2, 65),
    ("Center-based", 3, 72),
    ("Center-based", 4, 79),
)
SUPPORTING_COLUMNS = ("program", "visit_number", "goal_met_pct")

CRITIQUE_CATEGORIES = (
    "unsupported claim",
    "truncated baseline",
    "missing unit",
    "color-only encoding",
    "distracting decoration",
)
# Each contract string Task 3 saves, and the task that writes it.
EVIDENCE_TEXT = {
    "question": "Task 3.1",
    "audience": "Task 3.1",
    "intended_claim": "Task 3.1",
    "displayed_unit": "Task 3.1",
    "grain": "Task 3.1",
    "text_alternative": "Task 3.3",
}
DATA_TYPES = {"program": "categorical", "visit_number": "ordinal", "goal_met_pct": "quantitative"}
DATA_TYPE_KEYS = ("data_types", "data_type", "variable_types", "variable_roles", "variables")
# Lecture 07's data-type words, the synonyms people use beside them, and Altair's type letters.
TYPE_WORDS = {
    "categorical": "categorical",
    "nominal": "categorical",
    "qualitative": "categorical",
    "ordinal": "ordinal",
    "ordered": "ordinal",
    "quantitative": "quantitative",
    "numeric": "quantitative",
    "numerical": "quantitative",
    "continuous": "quantitative",
    "temporal": "temporal",
}
TYPE_LETTERS = {"n": "categorical", "o": "ordinal", "q": "quantitative", "t": "temporal"}
# A word right after one of these is ruled out, as in "not ordinal" or "non-numeric".
NEGATION = re.compile(r"(?:\bnot|\bnon|\bno|n't|\brather than|\binstead of)[\s-]*(?:an?\s+)?$")

# The data are whole numbers, so numbers are compared rounded to two decimals: 79 == 79.0 == 79.00.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _join(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _rows_phrase(count: int, verb: str, plural_verb: str) -> str:
    """'1 row matches' or '3 rows match': a count of rows with its verb in agreement."""
    return f"{count} row {verb}" if count == 1 else f"{count} rows {plural_verb}"


def _clean(text: str) -> str:
    return " ".join(str(text).split())


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
    message = f"{name} is missing; run the {task} cell to write it, then commit it."
    wanted = root / name
    if wanted.parent.is_dir():
        look_alikes = sorted(
            path.relative_to(root).as_posix()
            for path in wanted.parent.iterdir()
            if path.is_file()
            and path.name.casefold() != wanted.name.casefold()
            and difflib.SequenceMatcher(None, wanted.name.casefold(), path.name.casefold()).ratio() >= 0.75
        )
        if look_alikes:
            message += f" Found {_join(look_alikes)}; save it as {name} instead."
    return message


def _read_text(root: Path, name: str, task: str) -> str:
    path = _artifact(root, name)
    _assert(path is not None, _missing(root, name, task))
    return _decode(path.read_bytes())


def _number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().removesuffix("%").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _key(text) -> str:
    """A dictionary key or column name, compared in any case and with spaces or hyphens for underscores."""
    return re.sub(r"[\s\-]+", "_", str(text).strip().casefold())


def _label(value) -> str:
    return _clean(value).casefold()


def _cell(value, numeric: bool):
    """A value ready to compare: numbers rounded to two decimals, text in any case and spacing."""
    if numeric:
        number = _number(value)
        if number is not None:
            return round(number, 2)
    return _label(value)


def _load_json(root: Path, name: str, task: str):
    text = _read_text(root, name, task)
    _assert(bool(text.strip()), f"{name} is empty; run the {task} cell again to write it.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"{name} is not valid JSON ({error.msg}, line {error.lineno}, column {error.colno}); "
            f"run the {task} cell again so json.dump or .save() writes the whole file."
        ) from None


# Task 1: output/exploratory_spec.json


def _spec(root: Path) -> dict:
    spec = _load_json(root, SPEC_FILE, "Task 1.1")
    _assert(
        isinstance(spec, dict),
        f"{SPEC_FILE} holds a JSON {type(spec).__name__}, not a chart specification; "
        "in Task 1.1, save the chart itself with exploratory_chart.save(EXPLORATORY_SPEC_PATH).",
    )
    return spec


def _views(spec: dict, inherited: dict | None = None) -> Iterator[tuple[dict, dict]]:
    """Every view in the specification with the encodings it inherits, the outermost first."""
    own = spec.get("encoding") if isinstance(spec.get("encoding"), dict) else {}
    encoding = {**(inherited or {}), **own}
    yield spec, encoding
    for key in ("layer", "hconcat", "vconcat", "concat"):
        children = spec.get(key)
        for child in children if isinstance(children, list) else []:
            if isinstance(child, dict):
                yield from _views(child, encoding)
    if isinstance(spec.get("spec"), dict):
        yield from _views(spec["spec"], encoding)


def _mark_type(view: dict) -> str:
    mark = view.get("mark")
    if isinstance(mark, dict):
        mark = mark.get("type")
    return str(mark).casefold() if mark is not None else ""


def _plotted_view(spec: dict) -> tuple[dict, dict] | None:
    """The view to grade and its encodings: the first with a point mark, else the first with any mark."""
    marked = [(view, encoding) for view, encoding in _views(spec) if "mark" in view]
    points = [(view, encoding) for view, encoding in marked if _mark_type(view) == "point"]
    return (points or marked or [None])[0]


def check_spec_mark(root: Path) -> None:
    """The exploratory chart draws points."""
    spec = _spec(root)
    found = _plotted_view(spec)
    _assert(
        found is not None,
        f"{SPEC_FILE} has no mark, so it draws nothing; in Task 1.1, build the chart with "
        "alt.Chart(patients).mark_point(filled=True, size=90).",
    )
    view, _ = found
    mark = _mark_type(view)
    _assert(
        mark == "point",
        f"{SPEC_FILE} draws {mark or 'an unnamed'} marks, not points; in Task 1.1, use .mark_point(), "
        "the one mark that shows the shape encoding.",
    )


def _embedded_rows(spec: dict) -> list:
    """The rows the chart draws, from the graded view's data or the specification's."""
    found = _plotted_view(spec)
    candidates = [found[0]] if found else []
    candidates.append(spec)
    datasets = spec.get("datasets") if isinstance(spec.get("datasets"), dict) else {}
    for source in candidates:
        data = source.get("data")
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("values"), list):
            return data["values"]
        if isinstance(data.get("name"), str) and isinstance(datasets.get(data["name"]), list):
            return datasets[data["name"]]
        if "url" in data:
            raise AssertionError(
                f"{SPEC_FILE} links to {data['url']} instead of embedding the rows; in Task 1.1, "
                "build the chart from the patients DataFrame, alt.Chart(patients), and save it with .save()."
            )
    if len(datasets) == 1:
        rows = next(iter(datasets.values()))
        if isinstance(rows, list):
            return rows
    raise AssertionError(
        f"{SPEC_FILE} embeds no data rows; in Task 1.1, build the chart from the patients DataFrame, "
        "alt.Chart(patients), and save it with exploratory_chart.save(EXPLORATORY_SPEC_PATH)."
    )


def check_spec_rows(root: Path) -> None:
    """The specification embeds the twelve patients' plotted values, and no other rows.

    Only the three plotted columns are compared, so leaving out patient_id or
    adding a column costs nothing.
    """
    rows = _embedded_rows(_spec(root))
    described = {
        (_label(program), float(sessions), float(distance)): f"{patient_id}: {program}, {sessions} sessions, {distance} m"
        for patient_id, (program, sessions, distance) in REHAB_PATIENTS.items()
    }
    expected = Counter(list(described))
    found: Counter = Counter()
    incomplete = 0
    for row in rows:
        cells = {_key(key): value for key, value in row.items()} if isinstance(row, dict) else {}
        if not all(field in cells for field in PLOTTED_FIELDS):
            incomplete += 1
            continue
        program, sessions, distance = (cells[field] for field in PLOTTED_FIELDS)
        found[(_cell(program, False), _cell(sessions, True), _cell(distance, True))] += 1
    if found == expected and not incomplete:
        return
    if not found:
        columns = sorted({_key(key) for row in rows if isinstance(row, dict) for key in row})
        raise AssertionError(
            f"{SPEC_FILE} embeds {len(rows)} rows, but none has all of {_join(PLOTTED_FIELDS)} "
            f"(its rows have {_join(columns) or 'no columns'}); in Task 1.1, build the chart from "
            "patients, read unchanged from data/rehab_patients.csv."
        )
    missing = list((expected - found).elements())
    extra = list((found - expected).elements())
    problems = []
    if missing:
        shown = "; ".join(described[values] for values in missing[:3])
        verb = "is" if len(missing) == 1 else "are"
        problems.append(f"{len(missing)} of the 12 patients {verb} missing (such as {shown})")
    if extra:
        shown = "; ".join(
            f"{program}, {_shown(sessions)} sessions, {_shown(distance)} m" for program, sessions, distance in extra[:3]
        )
        problems.append(f"{_rows_phrase(len(extra), 'matches', 'match')} no patient (such as {shown})")
    if incomplete:
        problems.append(f"{_rows_phrase(incomplete, 'lacks', 'lack')} one of {_join(PLOTTED_FIELDS)}")
    embedded = "1 row" if len(rows) == 1 else f"{len(rows)} rows"
    raise AssertionError(
        f"{SPEC_FILE} embeds {embedded} where it should embed the 12 patients: " + "; ".join(problems) + ". "
        "In Task 1.1, chart every row of patients, read unchanged from data/rehab_patients.csv."
    )


def _check_encoding(root: Path, channel: str, field: str, data_type: str, how: str) -> None:
    spec = _spec(root)
    found = _plotted_view(spec)
    _assert(
        found is not None,
        f"{SPEC_FILE} has no mark, so its {channel} encoding cannot be read; in Task 1.1, build the chart with "
        f".mark_point() and encode {how}.",
    )
    _, encoding = found
    entry = encoding.get(channel)
    _assert(
        entry is not None,
        f"{SPEC_FILE} has no {channel} encoding; in Task 1.1, encode {how}.",
    )
    _assert(
        isinstance(entry, dict) and "field" in entry,
        f"{SPEC_FILE}'s {channel} encoding names no column (it holds {json.dumps(entry)[:80]}); "
        f"in Task 1.1, encode {how}.",
    )
    given = str(entry.get("field"))
    _assert(
        _key(given) == field,
        f"{SPEC_FILE} encodes {given} as {channel}, not {field}; in Task 1.1, encode {how}.",
    )
    given_type = str(entry.get("type", "")).strip().casefold()
    accepted = {data_type, data_type[0]}
    _assert(
        given_type in accepted,
        f"{SPEC_FILE} gives {field} the {channel} type {given_type or 'none'}, not {data_type}; "
        f"in Task 1.1, encode {how}.",
    )


# Task 2 and Task 3: output/visualization_evidence.json


def _evidence(root: Path, task: str) -> dict:
    evidence = _load_json(root, EVIDENCE_FILE, task)
    _assert(
        isinstance(evidence, dict),
        f"{EVIDENCE_FILE} holds a JSON {type(evidence).__name__}, not an object with named keys; "
        "save a dict with json.dump in Tasks 2.2 and 3.3.",
    )
    keyed: dict = {}
    for key, value in evidence.items():
        keyed.setdefault(_key(key), value)
    return keyed


def _category(text) -> str:
    return re.sub(r"[\s\-_]+", " ", str(text).casefold().replace("colour", "color")).strip()


def _critique_entries(evidence: dict) -> dict[str, dict]:
    """{normalized category: entry with normalized keys} from a list of entries or a dict keyed by category."""
    critique = evidence.get("critique")
    _assert(
        critique is not None,
        f"{EVIDENCE_FILE} has no critique key (its keys are {_join(sorted(evidence)) or 'none'}); "
        'Task 2.2 saves {"critique": critique}, and Task 3.3 keeps critique when it saves the file again.',
    )
    if isinstance(critique, dict):
        critique = [{"category": category, **entry} for category, entry in critique.items() if isinstance(entry, dict)]
    _assert(
        isinstance(critique, list),
        f"{EVIDENCE_FILE}'s critique is a {type(critique).__name__}, not a list of entries; "
        "keep the list of five dicts Task 2.2 supplies.",
    )
    entries: dict[str, dict] = {}
    for entry in critique:
        if isinstance(entry, dict):
            keyed = {_key(key): value for key, value in entry.items()}
            entries.setdefault(_category(keyed.get("category", "")), keyed)
    return entries


def _check_critique(root: Path, category: str) -> None:
    entries = _critique_entries(_evidence(root, "Task 2.2"))
    entry = entries.get(_category(category))
    found = _join(sorted(name for name in entries if name)) or "none"
    _assert(
        entry is not None,
        f"{EVIDENCE_FILE}'s critique has no {category} entry (its categories are {found}); "
        "in Task 2.2, keep the five category values the cell supplies.",
    )
    blank = [
        part for part in ("problem", "repair") if not (isinstance(entry.get(part), str) and entry[part].strip())
    ]
    _assert(
        not blank,
        f"{EVIDENCE_FILE}'s {category} entry has no {' and no '.join(blank)} text; in Task 2.2, "
        "say what is wrong with the draft chart (problem) and what the redesign changes (repair).",
    )


def _check_evidence_text(root: Path, key: str) -> None:
    task = EVIDENCE_TEXT[key]
    evidence = _evidence(root, "Task 3.3")
    _assert(
        key in evidence,
        f"{EVIDENCE_FILE} has no {key} key (its keys are {_join(sorted(evidence)) or 'none'}); "
        f"in Task 3.3, add {key} to visualization_evidence and save it again.",
    )
    value = evidence[key]
    _assert(
        isinstance(value, str) and value.strip() != "",
        f"{EVIDENCE_FILE}'s {key} is {json.dumps(value)[:40]}, where it should be your own text; "
        f"write {key} in {task}, then save the file again in Task 3.3.",
    )


def _type_of(value) -> str | None:
    """The data type a value names, or an Altair type letter alone.

    A word ruled out, as in "not ordinal", is skipped. Ordinal data are
    categorical data with an order, so an answer that names both, such as
    "categorical (ordered)" or "ordered categorical", names ordinal. Otherwise
    the first type named is the answer.
    """
    text = value if isinstance(value, str) else json.dumps(value)
    text = text.casefold()
    letter = text.strip().strip(":").strip()
    if letter in TYPE_LETTERS:
        return TYPE_LETTERS[letter]
    named = [
        TYPE_WORDS[match.group(1)]
        for match in re.finditer(r"\b(" + "|".join(TYPE_WORDS) + r")\b", text)
        if not NEGATION.search(text[:match.start()])
    ]
    if set(named) == {"categorical", "ordinal"}:
        return "ordinal"
    return named[0] if named else None


def _check_data_type(root: Path, column: str) -> None:
    expected = DATA_TYPES[column]
    evidence = _evidence(root, "Task 3.3")
    key = next((key for key in DATA_TYPE_KEYS if key in evidence), None)
    _assert(
        key is not None,
        f"{EVIDENCE_FILE} has no data_types key; in Task 3.3, add the data_types dict from Task 3.1 "
        "to visualization_evidence and save it again.",
    )
    types = evidence[key]
    _assert(
        isinstance(types, dict),
        f"{EVIDENCE_FILE}'s {key} is a {type(types).__name__}, not a dict of column names and data types; "
        "keep the data_types dict Task 3.1 supplies.",
    )
    by_column = {_key(name): value for name, value in types.items()}
    _assert(
        column in by_column,
        f"{EVIDENCE_FILE}'s {key} has no {column} entry (it names {_join(sorted(by_column)) or 'nothing'}); "
        "in Task 3.1, give a data type for each of the three plotted columns.",
    )
    given = by_column[column]
    found = _type_of(given)
    hint = " visit_number is the visits' order, not a date." if column == "visit_number" else ""
    _assert(
        found == expected,
        f"{EVIDENCE_FILE} gives {column} the data type {json.dumps(given)[:60]}, not {expected}; in Task 3.1, "
        f"use one of Lecture 07's words: categorical, quantitative, ordinal, or temporal.{hint}",
    )


def check_text_alternative_file(root: Path) -> None:
    """The text alternative is saved as its own file, matching the JSON's text_alternative in any spacing and case."""
    text = _read_text(root, TEXT_ALTERNATIVE_FILE, "Task 3.3")
    _assert(
        text.strip() != "",
        f"{TEXT_ALTERNATIVE_FILE} is empty; in Task 3.3, write text_alternative to it.",
    )
    try:
        evidence = _evidence(root, "Task 3.3")
    except AssertionError:
        return  # The JSON's own checks report what is wrong with it.
    saved = evidence.get("text_alternative")
    if not (isinstance(saved, str) and saved.strip()):
        return
    _assert(
        _label(text) == _label(saved),
        f"{TEXT_ALTERNATIVE_FILE} differs from text_alternative in {EVIDENCE_FILE}; in Task 3.3, "
        "write the same text_alternative string to both files, then run the cell again.",
    )


# Task 2 and Task 3: the saved PNG charts


def _check_png(root: Path, name: str, task: str, figure: str) -> None:
    path = _artifact(root, name)
    _assert(path is not None, _missing(root, name, task))
    head = path.read_bytes()[:8]
    _assert(bool(head), f"{name} is empty; run the {task} cell again to save the chart.")
    _assert(
        head == PNG_SIGNATURE,
        f"{name} is not a PNG image; in {task}, save the chart with "
        f'{figure}.savefig(..., dpi=150, bbox_inches="tight") to the .png path the notebook supplies.',
    )


# Task 3: output/explanatory_supporting_data.csv


def _is_whole_number(cell: str) -> bool:
    number = _number(cell)
    return number is not None and number.is_integer()


def read_table(root: Path, name: str, task: str) -> Table:
    """Parse a saved CSV, however it is separated, spaced, quoted, or ended.

    A leading column headed by nothing, `Unnamed: 0`, or `index` that holds only
    whole numbers is the row numbers pandas writes when `index=False` is left
    out, so it is set aside rather than counted as a column. Saving a reread file
    that way again adds a second such column, which is set aside too.
    """
    text = _read_text(root, name, task)
    lines = _csv_rows(text)
    _assert(bool(lines), f"{name} is empty; run the {task} cell again to write it.")
    header = [_key(cell) for cell in lines[0]]
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


def check_supporting_columns(root: Path) -> None:
    """The supporting data has exactly the three plotted columns, in any order."""
    table = read_table(root, SUPPORTING_FILE, "Task 3.1")
    missing = [column for column in SUPPORTING_COLUMNS if column not in table.columns]
    extra = [column or "(blank)" for column in table.columns if column not in SUPPORTING_COLUMNS]
    problems = []
    if missing:
        problems.append(f"is missing {_join(missing)}")
    if extra:
        problems.append(f"also has {_join(extra)}")
    _assert(
        not problems,
        f"{SUPPORTING_FILE} " + " and ".join(problems) + f"; it should have exactly the three plotted columns "
        f"{_join(SUPPORTING_COLUMNS)}. In Task 3.1, select those three columns of followup, then save "
        "with index=False.",
    )


def _shown(value) -> str:
    """A compared value as a reader would write it: 14.0 as 14."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def _describe_goal(columns: tuple[str, ...], values: tuple) -> str:
    parts = []
    for column, value in zip(columns, values):
        if column == "visit_number":
            parts.append(f"visit {_shown(value)}")
        elif column == "goal_met_pct":
            parts.append(f"{_shown(value)}%")
        else:
            parts.append(str(value))
    return ", ".join(parts)


def check_supporting_rows(root: Path) -> None:
    """The supporting data holds the eight program-and-visit rows with their goal percentages.

    Rows are compared on whichever of the three plotted columns the file has,
    so a missing or misnamed column costs only the columns check.
    """
    table = read_table(root, SUPPORTING_FILE, "Task 3.1")
    present = tuple(column for column in SUPPORTING_COLUMNS if column in table.columns)
    _assert(
        bool(present),
        f"{SUPPORTING_FILE} has none of the columns {_join(SUPPORTING_COLUMNS)}, so its rows cannot be "
        "compared; in Task 3.1, select those three columns of followup, then save with index=False.",
    )
    positions = [SUPPORTING_COLUMNS.index(column) for column in present]
    numeric = {"visit_number", "goal_met_pct"}
    expected = Counter(
        tuple(_cell(row[i], SUPPORTING_COLUMNS[i] in numeric) for i in positions) for row in FOLLOWUP_GOALS
    )
    found = Counter(tuple(_cell(row[column], column in numeric) for column in present) for row in table.rows)
    if found == expected:
        return
    missing = list((expected - found).elements())
    extra = list((found - expected).elements())
    problems = []
    if missing:
        shown = "; ".join(_describe_goal(present, values) for values in missing[:3])
        verb = "is" if len(missing) == 1 else "are"
        problems.append(f"{len(missing)} of the 8 rows {verb} missing (such as {shown})")
    if extra:
        shown = "; ".join(_describe_goal(present, values) for values in extra[:3])
        problems.append(f"{_rows_phrase(len(extra), 'matches', 'match')} no row of data/followup_goals.csv "
                        f"(such as {shown})")
    saved = "1 row" if len(table.rows) == 1 else f"{len(table.rows)} rows"
    raise AssertionError(
        f"{SUPPORTING_FILE} has {saved} where it should have the 8 rows of "
        "data/followup_goals.csv: " + "; ".join(problems) + ". In Task 3.1, save the three plotted columns "
        "of every row of followup, unchanged."
    )


CHECKS = (
    Check("exploratory spec: point mark", check_spec_mark),
    Check("exploratory spec: embedded patient rows", check_spec_rows),
    Check(
        "exploratory spec: x encoding",
        lambda root: _check_encoding(root, "x", "sessions_attended", "quantitative", "x='sessions_attended:Q'"),
    ),
    Check(
        "exploratory spec: y encoding",
        lambda root: _check_encoding(root, "y", "walk_distance_m", "quantitative", "y='walk_distance_m:Q'"),
    ),
    Check(
        "exploratory spec: color encoding",
        lambda root: _check_encoding(root, "color", "program", "nominal", "color='program:N'"),
    ),
    Check(
        "exploratory spec: shape encoding",
        lambda root: _check_encoding(root, "shape", "program", "nominal", "shape='program:N'"),
    ),
    Check(
        "critique redesign: PNG image",
        lambda root: _check_png(root, REDESIGN_FILE, "Task 2.3", "redesign_fig"),
    ),
    *(
        Check(f"critique: {category}", lambda root, category=category: _check_critique(root, category))
        for category in CRITIQUE_CATEGORIES
    ),
    Check(
        "explanatory chart: PNG image",
        lambda root: _check_png(root, EXPLANATORY_FILE, "Task 3.2", "explanatory_fig"),
    ),
    Check("supporting data: columns", check_supporting_columns),
    Check("supporting data: rows and values", check_supporting_rows),
    *(Check(f"evidence: {key}", lambda root, key=key: _check_evidence_text(root, key)) for key in EVIDENCE_TEXT),
    *(
        Check(f"data type: {column}", lambda root, column=column: _check_data_type(root, column))
        for column in DATA_TYPES
    ),
    Check("text alternative file", check_text_alternative_file),
)


def run_checks(root: Path) -> list[tuple[str, str | None]]:
    """Return one (check name, problem or None) pair per check; every check runs."""
    results = []
    for check in CHECKS:
        try:
            check.action(Path(root))
        except (AssertionError, OSError, ValueError, UnicodeDecodeError, csv.Error,
                AttributeError, KeyError, TypeError, RecursionError) as error:
            results.append((check.name, str(error)))
        else:
            results.append((check.name, None))
    return results
