"""Checks for Assignment 07.

The course keeps these checks in 07/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They read the six files a
submission saves in output/ and compare them with the supplied data, whose
values are copied below; the self-test confirms those copies match data/.

Each check scores one thing, so one mistake costs only its own points, and an
answer that shows the skill in another form earns them. Values are compared
after parsing: spacing, line endings, quoting, key and column order, a leading
row-number column, number formatting (79 == 79.0 == 79.00), and the letter case
of labels never cost points, and a CSV's cells may be separated by commas,
semicolons, or tabs. A mistake that several checks could see is charged once:
a file saved outside output/ by the first check that reads it, swapped axes by
the x check, one data type given to both color and shape by the color check,
and a column named twice by the columns check. A chart saved as a PNG is
checked for being a PNG image that is not blank; how it looks is the student's
to inspect.

Nothing here imports, runs, or inspects student code.
"""

from __future__ import annotations

import codecs
import csv
import difflib
import itertools
import json
import re
import struct
import zlib
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
# The notebook's path constant for each file, named when a file lands outside output/.
PATH_CONSTANTS = {
    SPEC_FILE: "EXPLORATORY_SPEC_PATH",
    EVIDENCE_FILE: "EVIDENCE_PATH",
    REDESIGN_FILE: "REDESIGN_PATH",
    SUPPORTING_FILE: "SUPPORTING_DATA_PATH",
    EXPLANATORY_FILE: "EXPLANATORY_PATH",
    TEXT_ALTERNATIVE_FILE: "TEXT_ALTERNATIVE_PATH",
}
# The files more than one check reads, so a copy saved outside output/ can still be graded by the rest.
MULTI_CHECK_FILES = (SPEC_FILE, EVIDENCE_FILE, SUPPORTING_FILE)

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
    "y_measure": "Task 3.1",
    "grain": "Task 3.1",
    "text_alternative": "Task 3.3",
}
# Other keys that hold the same answer: the key's earlier name (displayed_unit, renamed so it is not
# confused with the lecture's "unit displayed"), short forms, and the combined keys of the Lecture 07
# demos' contracts (audience_and_claim, unit_and_grain).
EVIDENCE_ALIASES = {
    "audience": ("audience_and_claim",),
    "intended_claim": ("claim", "audience_and_claim"),
    "y_measure": ("displayed_unit", "measure", "y_axis"),
    "grain": ("unit_and_grain",),
    "text_alternative": ("text_alt", "alt_text", "alternative_text"),
}
# Every Task 3 evidence check gives this one message for a file holding only the critique, so the
# report prints it once: Task 3 is not done yet, or the Task 2.2 cell ran again after Task 3.3.
ONLY_CRITIQUE = (
    f"{EVIDENCE_FILE} holds only the critique that Task 2.2 saves. Task 3.3 saves the file again with "
    "question, audience, intended_claim, y_measure, grain, data_types, and text_alternative added, so "
    "run the Task 3.3 cell after the Task 2.2 cell: Restart, then Run All runs them in order."
)
# The data types each plotted column may be given. The visits come one after another in time, so
# temporal counts for visit_number beside ordinal, as Lecture 07's Demo 1 calls its week numbers temporal.
DATA_TYPES = {
    "program": ("categorical",),
    "visit_number": ("ordinal", "temporal"),
    "goal_met_pct": ("quantitative",),
}
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
# An answer that names how pandas stores a column (Lecture 04's dtypes) rather than what its values mean.
PANDAS_DTYPE = re.compile(
    r"(?:u?int|float|complex)(?:8|16|32|64)?|object|str|string|category|bool(?:ean)?|datetime64(?:\[\w+\])?"
)
# A word right after one of these is ruled out, as in "not ordinal" or "non-numeric".
NEGATION = re.compile(r"(?:\bnot|\bnon|\bno|n't|\brather than|\binstead of)[\s-]*(?:an?\s+)?$")

# The data are whole numbers, so numbers are compared rounded to two decimals: 79 == 79.0 == 79.00.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Samples per pixel for each PNG color type: gray, RGB, palette index, gray and alpha, RGBA.
PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


@dataclass(frozen=True)
class Table:
    """A saved CSV: its casefolded column names in file order, each data row's cells, and how
    many times each column name used more than once appears."""

    name: str
    columns: tuple[str, ...]
    cells: tuple[tuple[str, ...], ...]
    repeated: dict[str, int]

    def readings(self) -> Iterator[tuple[dict[str, str], ...]]:
        """The rows as {column: cell} dicts: once, or, when a column is named more than once, once
        for each way of taking one copy of every such column, so no copy is silently dropped."""
        positions: dict[str, list[int]] = {}
        for i, column in enumerate(self.columns):
            positions.setdefault(column, []).append(i)
        names = list(positions)
        for choice in itertools.product(*(positions[name] for name in names)):
            yield tuple({name: row[i] for name, i in zip(names, choice)} for row in self.cells)


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


# The artifacts found in the assignment folder instead of output/ during one run_checks() call.
_REPORTED_OUTSIDE: set[str] = set()


def _locate(root: Path, name: str, task: str) -> Path:
    """The artifact's path, or its copy in the assignment folder itself.

    A bare file name, as in savefig("critique_redesign.png"), saves into the
    assignment folder rather than output/. The first check that reads such a file
    says where it belongs, and the checks after it read it where it is, so the
    slip costs one check.
    """
    path = _artifact(root, name)
    if path is not None:
        return path
    wanted = root / name
    outside = _artifact(root, wanted.name) if wanted.parent != root else None
    _assert(outside is not None, _missing(root, name, task))
    if name not in _REPORTED_OUTSIDE:
        _REPORTED_OUTSIDE.add(name)
        rest = " The checks after this one read it where it is." if name in MULTI_CHECK_FILES else ""
        raise AssertionError(
            f"{name} is missing, but the assignment folder has {outside.name}; in {task}, save to "
            f"{PATH_CONSTANTS[name]}, which puts the file in output/, then commit it.{rest}"
        )
    return outside


def _missing(root: Path, name: str, task: str) -> str:
    wanted = root / name
    message = f"{name} is missing; run the {task} cell to write it, then commit it."
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
    return _decode(_locate(root, name, task).read_bytes())


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


def _has_text(value) -> bool:
    """Whether a JSON value holds written text: a non-blank string, or a list or object holding one."""
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, dict):
        value = list(value.values())
    return isinstance(value, list) and any(_has_text(item) for item in value)


def _cell(value, numeric: bool):
    """A value ready to compare: numbers rounded to two decimals, text in any case and spacing."""
    if numeric:
        number = _number(value)
        if number is not None:
            return round(number, 2)
    return _label(value)


def _json_kind(value) -> str:
    """A JSON value's kind as JSON names it: an object, a list, a string, a number, true or false, or null."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true or false"
    if isinstance(value, dict):
        return "an object"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, str):
        return "a string"
    return "a number"


def _brief(value, limit: int = 60) -> str:
    """A JSON value as saved, shortened with ... past `limit` characters."""
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _load_json(root: Path, name: str, task: str):
    text = _read_text(root, name, task)
    _assert(bool(text.strip()), f"{name} is empty; run the {task} cell again to write it.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        if text.lstrip()[:2] in ("{'", "['"):
            # Python's printed form of a dict or list: str(), print(), or file.write(str(...)).
            how = (
                "in Task 1.1, save the chart itself with exploratory_chart.save(EXPLORATORY_SPEC_PATH)"
                if name == SPEC_FILE
                else "in Task 3.3, save visualization_evidence with json.dump(visualization_evidence, file, "
                "indent=2, ensure_ascii=False), as the Task 2.2 cell saves the critique"
            )
            raise AssertionError(
                f"{name} holds a Python dict written with str() or print(), not JSON, which puts every key and "
                f"text in double quotes; {how}."
            ) from None
        raise AssertionError(
            f"{name} is not valid JSON ({error.msg}, line {error.lineno}, column {error.colno}); "
            f"run the {task} cell again so json.dump or .save() writes the whole file."
        ) from None


# Task 1: output/exploratory_spec.json


def _spec(root: Path) -> dict:
    spec = _load_json(root, SPEC_FILE, "Task 1.1")
    _assert(
        isinstance(spec, dict),
        f"{SPEC_FILE} holds {_json_kind(spec)}, not a chart specification (a JSON object); "
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
    mark = view.get("mark")
    if isinstance(mark, dict):
        mark = mark.get("type")
    shown = f"{mark} marks" if isinstance(mark, str) and mark.strip() else f"the mark {_brief(mark, 40)}"
    _assert(
        _mark_type(view) == "point",
        f"{SPEC_FILE} draws {shown}, not points; in Task 1.1, use .mark_point(), "
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


def _records(spec: dict) -> list[dict]:
    """The embedded rows with their column names compared in any case and spacing."""
    return [
        {_key(key): value for key, value in row.items()} if isinstance(row, dict) else {}
        for row in _embedded_rows(spec)
    ]


def _resolve_fields(records: list[dict]) -> dict[str, str]:
    """Each plotted field and the embedded column holding it.

    That is the column of the same name, or else the one column, renamed as in
    "Walk distance (m)", whose values are that field's twelve values. A renamed
    column still shows the data, so the checks follow it.
    """
    columns = {key for record in records for key in record}
    resolved = {}
    for i, field in enumerate(PLOTTED_FIELDS):
        if field in columns:
            resolved[field] = field
            continue
        numeric = field != "program"
        expected = Counter(_cell(values[i], numeric) for values in REHAB_PATIENTS.values())
        matches = [
            column
            for column in sorted(columns - set(PLOTTED_FIELDS))
            if Counter(_cell(record[column], numeric) for record in records if column in record) == expected
        ]
        if len(matches) == 1:
            resolved[field] = matches[0]
    return resolved


def check_spec_rows(root: Path) -> None:
    """The specification embeds the twelve patients' plotted values, and no other rows.

    Only the three plotted columns are compared, so leaving out patient_id,
    adding a column, or renaming a plotted one costs nothing.
    """
    records = _records(_spec(root))
    for field, column in _resolve_fields(records).items():
        for record in records:
            if column in record:
                record[field] = record[column]
    if records and all("patient_id" in record for record in records):
        _check_spec_rows_by_patient(records)
        return
    described = {
        (_label(program), float(sessions), float(distance)): f"{patient_id}: {program}, {sessions} sessions, {distance} m"
        for patient_id, (program, sessions, distance) in REHAB_PATIENTS.items()
    }
    expected = Counter(list(described))
    found: Counter = Counter()
    incomplete = 0
    for cells in records:
        if not all(field in cells for field in PLOTTED_FIELDS):
            incomplete += 1
            continue
        program, sessions, distance = (cells[field] for field in PLOTTED_FIELDS)
        found[(_cell(program, False), _cell(sessions, True), _cell(distance, True))] += 1
    if found == expected and not incomplete:
        return
    if not found:
        columns = sorted({key for record in records for key in record})
        raise AssertionError(
            f"{SPEC_FILE} embeds {'1 row' if len(records) == 1 else f'{len(records)} rows'}, but none has all of "
            f"{_join(PLOTTED_FIELDS)} "
            f"(its rows have {_join(columns) or 'no column names'}); in Task 1.1, build the chart from "
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
    embedded = "1 row" if len(records) == 1 else f"{len(records)} rows"
    raise AssertionError(
        f"{SPEC_FILE} embeds {embedded} where it should embed the 12 patients: " + "; ".join(problems) + ". "
        "In Task 1.1, chart every row of patients, read unchanged from data/rehab_patients.csv."
    )


def _listed(problems: list[str], limit: int = 4) -> str:
    shown = "; ".join(problems[:limit])
    if len(problems) > limit:
        shown += f"; and {len(problems) - limit} more like {'it' if len(problems) - limit == 1 else 'them'}"
    return shown


def _times(count: int) -> str:
    return "twice" if count == 2 else f"{count} times"


def _problems(name: str, row_problems: list[str], value_problems: list[str]) -> str:
    """'{name} is missing ...; lists ...; R03 has ...' or, with only wrong values, '{name}: R03 has ...'."""
    if row_problems:
        return f"{name} " + "; ".join(row_problems) + ("; " + _listed(value_problems) if value_problems else "")
    return f"{name}: " + _listed(value_problems)


def _check_spec_rows_by_patient(records: list[dict]) -> None:
    """Match embedded rows to patients by patient_id and name each wrong or missing value."""
    by_id = {patient_id.casefold(): patient_id for patient_id in REHAB_PATIENTS}
    seen: dict[str, list[dict]] = {}
    unknown = []
    for record in records:
        patient_id = by_id.get(_label(record["patient_id"]))
        if patient_id is None:
            unknown.append(_clean(record["patient_id"]) or "blank")
        else:
            seen.setdefault(patient_id, []).append(record)
    problems, wrong = [], []
    missing = [patient_id for patient_id in REHAB_PATIENTS if patient_id not in seen]
    if missing:
        problems.append(f"is missing {len(missing)} of the 12 patients: {_join(missing)}")
    repeated = [patient_id for patient_id, found in seen.items() if len(found) > 1]
    if repeated:
        problems.append("embeds " + _join(f"{patient_id} {_times(len(seen[patient_id]))}" for patient_id in repeated))
    if unknown:
        problems.append(f"also embeds patient_id {_join(unknown[:4])}, which data/rehab_patients.csv does not have")
    for patient_id, found in seen.items():
        for field, expected in zip(PLOTTED_FIELDS, REHAB_PATIENTS[patient_id]):
            numeric = field != "program"
            given = sorted({
                _shown(record[field]) if field in record else "nothing"
                for record in found
                if field not in record or _cell(record[field], numeric) != _cell(expected, numeric)
            })
            if given:
                wrong.append(f"{patient_id} has {field} {_join(given)}, expected {expected}")
    _assert(
        not (problems or wrong),
        _problems(SPEC_FILE, problems, wrong) + ". In Task 1.1, chart every row of patients, read unchanged "
        "from data/rehab_patients.csv.",
    )


# Each encoding Task 1.1 asks for: the column it shows, its data type, and how the README writes it.
ENCODINGS = {
    "x": ("sessions_attended", "quantitative", "x='sessions_attended:Q'"),
    "y": ("walk_distance_m", "quantitative", "y='walk_distance_m:Q'"),
    "color": ("program", "nominal", "color='program:N'"),
    "shape": ("program", "nominal", "shape='program:N'"),
}


def _definition(entry):
    """An encoding's field definition: the entry itself or, for a conditional encoding such as
    alt.condition(selection, 'program:N', alt.value('gray')), the definition inside its condition."""
    if isinstance(entry, dict) and "field" not in entry:
        condition = entry.get("condition")
        for option in condition if isinstance(condition, list) else [condition]:
            if isinstance(option, dict) and "field" in option:
                return option
    return entry


def _plotted_field(entry, resolved: dict[str, str]) -> str | None:
    """The plotted field an encoding shows, following a renamed column; otherwise the name it gives."""
    if not (isinstance(entry, dict) and "field" in entry):
        return None
    given = _key(entry["field"])
    return next((field for field, column in resolved.items() if given in (field, column)), given)


def _check_encoding(root: Path, channel: str) -> None:
    field, data_type, how = ENCODINGS[channel]
    spec = _spec(root)
    found = _plotted_view(spec)
    _assert(
        found is not None,
        f"{SPEC_FILE} has no mark, so its {channel} encoding cannot be read; in Task 1.1, build the chart with "
        f".mark_point() and encode {how}.",
    )
    _, encoding = found
    entry = _definition(encoding.get(channel))
    _assert(
        entry is not None,
        f"{SPEC_FILE} has no {channel} encoding; in Task 1.1, encode {how}.",
    )
    _assert(
        isinstance(entry, dict) and "field" in entry,
        f"{SPEC_FILE}'s {channel} encoding names no column (it holds {_brief(entry)}); "
        f"in Task 1.1, encode {how}.",
    )
    try:
        resolved = _resolve_fields(_records(spec))
    except AssertionError:
        resolved = {}  # The embedded rows check reports what is wrong with the data.
    shown = _plotted_field(entry, resolved)
    if channel in ("x", "y"):
        # Swapped axes are one mistake, charged by the x check; the y check grades the column y shows.
        other = "y" if channel == "x" else "x"
        swapped = shown == ENCODINGS[other][0] and _plotted_field(_definition(encoding.get(other)), resolved) == field
        _assert(
            not (swapped and channel == "x"),
            f"{SPEC_FILE} puts {ENCODINGS['y'][0]} on x and {field} on y, the other way round; the question asks "
            f"whether attending more sessions goes with walking farther, so in Task 1.1, encode {how} and "
            f"{ENCODINGS['y'][2]}.",
        )
        if swapped:
            field = shown
    _assert(
        shown == field,
        f"{SPEC_FILE} encodes {entry['field']} as {channel}, not {field}; in Task 1.1, encode {how}.",
    )
    given_type = str(entry.get("type", "")).strip().casefold()
    accepted = {data_type, data_type[0]}
    if channel == "shape":
        # One data type given to program for both color and shape is one mistake, charged by the color check.
        color = _definition(encoding.get("color"))
        if _plotted_field(color, resolved) == field:
            accepted.add(str(color.get("type", "")).strip().casefold())
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
        f"{EVIDENCE_FILE} holds {_json_kind(evidence)}, not a JSON object with named keys; "
        "save a dict with json.dump in Tasks 2.2 and 3.3.",
    )
    keyed: dict = {}
    for key, value in evidence.items():
        keyed.setdefault(_key(key), value)
    for canonical, aliases in EVIDENCE_ALIASES.items():
        if canonical not in keyed:
            for alias in aliases:
                if alias in keyed:
                    keyed[canonical] = keyed[alias]
                    break
    return keyed


def _category(text) -> str:
    return re.sub(r"[\s\-_]+", " ", str(text).casefold().replace("colour", "color")).strip()


def _critique_entries(evidence: dict) -> dict[str, dict]:
    """{normalized category: entry with normalized keys} from a list of entries or a dict keyed by category."""
    _assert(
        "critique" in evidence,
        f"{EVIDENCE_FILE} has no critique key (its keys are {_join(sorted(evidence)) or 'none'}); "
        'Task 2.2 saves {"critique": critique}, and Task 3.3 keeps critique when it saves the file again.',
    )
    critique = evidence["critique"]
    if isinstance(critique, dict):
        critique = [{"category": category, **entry} for category, entry in critique.items() if isinstance(entry, dict)]
    _assert(
        isinstance(critique, list),
        f"{EVIDENCE_FILE}'s critique is {_json_kind(critique)}, not a list of entries; "
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
        part for part in ("problem", "repair") if not _has_text(entry.get(part))
    ]
    _assert(
        not blank,
        f"{EVIDENCE_FILE}'s {category} entry has no {' and no '.join(blank)} text; in Task 2.2, "
        "say what is wrong with the draft chart (problem) and what the redesign changes (repair).",
    )


def _check_evidence_text(root: Path, key: str) -> None:
    task = EVIDENCE_TEXT[key]
    evidence = _evidence(root, "Task 3.3")
    _assert(set(evidence) != {"critique"}, ONLY_CRITIQUE)
    _assert(
        key in evidence,
        f"{EVIDENCE_FILE} has no {key} key (its keys are {_join(sorted(evidence)) or 'none'}); "
        f"in Task 3.3, add {key} to visualization_evidence and save it again.",
    )
    value = evidence[key]
    where = f"write {key} in Task 3.3 and save the file again" if task == "Task 3.3" else (
        f"write {key} in {task}, then save the file again in Task 3.3"
    )
    _assert(
        _has_text(value),
        f"{EVIDENCE_FILE}'s {key} is {_brief(value, 40)}, where it should be your own text; {where}.",
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
    _assert(set(evidence) != {"critique"}, ONLY_CRITIQUE)
    key = next((key for key in DATA_TYPE_KEYS if key in evidence), None)
    _assert(
        key is not None,
        f"{EVIDENCE_FILE} has no data_types key; in Task 3.3, add the data_types dict from Task 3.1 "
        "to visualization_evidence and save it again.",
    )
    types = evidence[key]
    _assert(
        isinstance(types, dict),
        f"{EVIDENCE_FILE}'s {key} is {_json_kind(types)}, not an object of column names and data types; "
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
    if isinstance(given, str) and PANDAS_DTYPE.fullmatch(_label(given)):
        hint = (
            " That is how pandas stores the column (Lecture 07's \"Storage dtype Is Not Visualization Type\"), "
            "not what its values mean."
        )
    elif column == "visit_number" and found == "quantitative":
        hint = " visit_number's numbers give the visits' order; the gaps between visits are not measured amounts."
    elif column == "visit_number":
        hint = " visit_number is the visits' order, not a date."
    else:
        hint = ""
    _assert(
        found in expected,
        f"{EVIDENCE_FILE} gives {column} the data type {_brief(given)}, not {' or '.join(expected)}; in Task 3.1, "
        f"use {expected[0]}{''.join(f' (or {other})' for other in expected[1:])}.{hint}",
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
    if _label(text) == _label(saved):
        return
    stripped = text.strip()
    _assert(
        not (len(stripped) > 1 and stripped[0] == stripped[-1] == '"'),
        f"{TEXT_ALTERNATIVE_FILE} holds text_alternative inside quotation marks, as json.dump writes a string; "
        "in Task 3.3, write the plain text with file.write(text_alternative).",
    )
    _assert(
        not stripped.startswith("{"),
        f"{TEXT_ALTERNATIVE_FILE} holds a whole dict rather than the text_alternative paragraph; in Task 3.3, "
        "write only the paragraph with file.write(text_alternative).",
    )
    in_file, in_json = _clean(text), _clean(saved)
    start = next(
        (i for i, (a, b) in enumerate(zip(in_file, in_json)) if a.casefold() != b.casefold()),
        min(len(in_file), len(in_json)),
    )
    raise AssertionError(
        f"{TEXT_ALTERNATIVE_FILE} differs from text_alternative in {EVIDENCE_FILE}: from character {start + 1}, "
        f"the file has {_excerpt(in_file, start)} where the JSON has {_excerpt(in_json, start)}. In Task 3.3, "
        "write text_alternative itself with file.write(text_alternative), then run the cell again so both files "
        "are saved from the same text."
    )


def _excerpt(text: str, start: int) -> str:
    """The text around position `start`, quoted, with ... where it is cut; 'nothing more' past its end."""
    if start >= len(text):
        return "nothing more"
    begin = max(0, start - 20)
    end = start + 40
    return '"' + ("..." if begin else "") + text[begin:end] + ("..." if end < len(text) else "") + '"'


# Task 2 and Task 3: the saved PNG charts


# What a saved chart holds instead of a PNG, told by how the file starts.
OTHER_FORMATS = (
    (b"\xff\xd8\xff", "a JPEG image"),
    (b"%PDF", "a PDF"),
    (b"<?xml", "an SVG drawing"),
    (b"<svg", "an SVG drawing"),
)


def _png_is_blank(data: bytes) -> bool:
    """Whether every pixel of a PNG is the same color, as in a figure saved with nothing drawn on it.

    Each stored row is compared with the bytes a one-color image stores for that
    row's filter (PNG filter types 0 to 4), so no pixel needs decoding. A PNG this
    cannot read (interlaced, under 8 bits per sample, or damaged) counts as not blank.
    """
    try:
        position, header, compressed = 8, None, []
        while position + 8 <= len(data):
            length, kind = struct.unpack(">I4s", data[position:position + 8])
            body = data[position + 8:position + 8 + length]
            if kind == b"IHDR":
                header = struct.unpack(">IIBBBBB", body)
            elif kind == b"IDAT":
                compressed.append(body)
            elif kind == b"IEND":
                break
            position += 12 + length
        if header is None:
            return False
        width, height, depth, color_type, _, _, interlaced = header
        if interlaced or depth not in (8, 16) or color_type not in PNG_CHANNELS or (color_type == 3 and depth != 8):
            return False
        raw = zlib.decompress(b"".join(compressed))
    except (struct.error, zlib.error):
        return False
    size = PNG_CHANNELS[color_type] * depth // 8
    stride = width * size
    if not width or not height or len(raw) < height * (stride + 1):
        return False
    pixel = raw[1:1 + size]  # the first pixel, stored unchanged whatever the first row's filter
    half = bytes((value - value // 2) % 256 for value in pixel)
    rest = bytes(stride - size)
    # filter type -> (the first row's stored bytes, every later row's) when every pixel equals `pixel`
    one_color = {
        0: (pixel * width, pixel * width),
        1: (pixel + rest, pixel + rest),
        2: (pixel * width, bytes(stride)),
        3: (pixel + half * (width - 1), half + rest),
        4: (pixel + rest, bytes(stride)),
    }
    for row in range(height):
        start = row * (stride + 1)
        stored = one_color.get(raw[start])
        if stored is None or raw[start + 1:start + 1 + stride] != stored[1 if row else 0]:
            return False
    return True


def _png_complete(data: bytes) -> bool:
    """Check PNG chunks and compressed pixels without an optional image library."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    position, chunks, compressed = 8, [], []
    palette = None
    while position + 12 <= len(data):
        size = int.from_bytes(data[position:position + 4], "big")
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + size]
        end = position + 12 + size
        if end > len(data) or zlib.crc32(kind + payload) != int.from_bytes(data[end - 4:end], "big"):
            return False
        chunks.append(kind)
        if kind == b"IHDR":
            if len(chunks) != 1 or size != 13:
                return False
            header = payload
        if kind == b"PLTE":
            if palette is not None or compressed or not size or size % 3 or size > 768:
                return False
            palette = payload
        if kind == b"IDAT":
            compressed.append(payload)
        position = end
        if kind == b"IEND":
            if size != 0 or position != len(data):
                return False
            break
    if not chunks or chunks[0] != b"IHDR" or chunks[-1] != b"IEND" or not compressed:
        return False
    width, height = int.from_bytes(header[:4], "big"), int.from_bytes(header[4:8], "big")
    depth, color, compression, filtering, interlace = header[8:]
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    allowed_depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
    if not width or not height or depth not in allowed_depths.get(color, ()) or compression or filtering or interlace > 1:
        return False
    if color == 3 and (palette is None or len(palette) // 3 > 2 ** depth):
        return False
    try:
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(b"".join(compressed)) + decoder.flush()
        if not decoder.eof or decoder.unused_data:
            return False
    except zlib.error:
        return False
    # Adam7 interlacing stores seven smaller images, each with its own filtered rows.
    passes = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
              (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)) if interlace else ((0, 0, 1, 1),)
    position = 0
    for x, y, dx, dy in passes:
        across, down = max(0, (width - x + dx - 1) // dx), max(0, (height - y + dy - 1) // dy)
        if not across or not down:
            continue
        stride = 1 + (across * channels[color] * depth + 7) // 8
        end = position + down * stride
        if end > len(pixels) or any(pixels[offset] > 4 for offset in range(position, end, stride)):
            return False
        position = end
    return position == len(pixels)



def _check_png(root: Path, name: str, task: str, figure: str) -> None:
    path = _locate(root, name, task)
    data = path.read_bytes()
    head = data[:8]
    _assert(bool(head), f"{name} is empty; run the {task} cell again to save the chart.")
    held = next((kind for start, kind in OTHER_FORMATS if head.lstrip().startswith(start)), "something else")
    _assert(
        head == PNG_SIGNATURE,
        f"{name} is not a PNG image (it holds {held}; renaming a file does not convert it); in {task}, save "
        f'the chart with {figure}.savefig(..., dpi=150, bbox_inches="tight") to the .png path the notebook supplies.',
    )
    _assert(
        _png_complete(data),
        f"{name} is damaged or incomplete; expected a readable PNG chart, not just its header. "
        f"In {task}, rerun {figure}.savefig(...) and commit the complete file.",
    )
    _assert(
        not _png_is_blank(data),
        f"{name} is blank: every pixel is the same color, so the figure it saved had nothing drawn on it. "
        f"plt.savefig() after plt.show() saves a new, empty figure; in {task}, draw the chart first, then call "
        f"{figure}.savefig(...) before plt.show().",
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
    try:
        lines = _csv_rows(text)
    except csv.Error as error:
        raise AssertionError(
            f"{name} cannot be read as a CSV table ({error}); run the {task} cell again so to_csv() writes it, "
            "then compare it with the checkpoint in README.md."
        ) from None
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
    cells = tuple(tuple(row[i] if i < len(row) else "" for i in range(len(header))) for row in body)
    repeated = {column: count for column, count in Counter(header).items() if count > 1}
    return Table(name=name, columns=tuple(header), cells=cells, repeated=repeated)


def check_supporting_columns(root: Path) -> None:
    """The supporting data has exactly the three plotted columns, in any order."""
    table = read_table(root, SUPPORTING_FILE, "Task 3.1")
    missing = [column for column in SUPPORTING_COLUMNS if column not in table.columns]
    extra = [column or "(blank)" for column in table.columns if column not in SUPPORTING_COLUMNS]
    problems = []
    if table.repeated:
        problems.append("names columns more than once: " + _join(table.repeated))
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


def _check_supporting_rows_by_visit(rows: tuple[dict[str, str], ...], with_values: bool) -> None:
    """Match rows on program and visit_number, then compare each goal_met_pct with the supplied one."""
    expected = {(_label(program), float(visit)): (program, visit, pct) for program, visit, pct in FOLLOWUP_GOALS}
    seen: dict[tuple, list[dict[str, str]]] = {}
    unknown = []
    for row in rows:
        visit = _number(row["visit_number"])
        key = (_label(row["program"]), round(visit, 2) if visit is not None else None)
        if key in expected:
            seen.setdefault(key, []).append(row)
        else:
            unknown.append(f"{row['program'] or 'blank'}, visit {row['visit_number'] or 'blank'}")
    problems, wrong = [], []
    missing: dict[str, list[str]] = {}
    for key, (program, visit, _) in expected.items():
        if key not in seen:
            missing.setdefault(program, []).append(str(visit))
    per_program = Counter(program for program, _, _ in FOLLOWUP_GOALS)
    whole = [program for program, visits in missing.items() if len(visits) == per_program[program]]
    for program, visits in missing.items():
        if program in whole:
            problems.append(f"has no {program} rows at all")
        else:
            problems.append(f"has no {program} row for visit{'s' if len(visits) > 1 else ''} {_join(visits)}")
    repeated = [key for key, rows in seen.items() if len(rows) > 1]
    if repeated:
        problems.append("lists " + _join(
            f"{expected[key][0]}, visit {expected[key][1]} {_times(len(seen[key]))}" for key in repeated))
    if unknown:
        shown = _join(unknown[:3]) + (f" and {len(unknown) - 3} more" if len(unknown) > 3 else "")
        problems.append(f"also has {shown}, which data/followup_goals.csv does not")
    if with_values:
        for key, rows in seen.items():
            program, visit, pct = expected[key]
            given = sorted({
                row["goal_met_pct"] or "blank" for row in rows
                if _cell(row["goal_met_pct"], True) != _cell(pct, True)
            })
            if given:
                wrong.append(f"{program}, visit {visit} has goal_met_pct {_join(given)}, expected {pct}")
    scope = "all eight rows of followup, for both programs" if whole else "every row of followup"
    _assert(
        not (problems or wrong),
        _problems(SUPPORTING_FILE, problems, wrong) + f". In Task 3.1, save the three plotted columns of {scope}, "
        "unchanged.",
    )


def _check_supporting_reading(table: Table, rows: tuple[dict[str, str], ...]) -> None:
    """The supporting data holds the eight program-and-visit rows with their goal percentages.

    Rows are matched on program and visit_number, so a wrong percentage is named
    with its row. Without one of those columns, rows are compared on whichever
    of the three plotted columns the file has, so a missing or misnamed column
    costs only the columns check.
    """
    present = tuple(column for column in SUPPORTING_COLUMNS if column in table.columns)
    _assert(
        bool(present),
        f"{SUPPORTING_FILE} has none of the columns {_join(SUPPORTING_COLUMNS)}, so its rows cannot be "
        "compared; in Task 3.1, select those three columns of followup, then save with index=False.",
    )
    if "program" in present and "visit_number" in present:
        _check_supporting_rows_by_visit(rows, "goal_met_pct" in present)
        return
    positions = [SUPPORTING_COLUMNS.index(column) for column in present]
    numeric = {"visit_number", "goal_met_pct"}
    expected = Counter(
        tuple(_cell(row[i], SUPPORTING_COLUMNS[i] in numeric) for i in positions) for row in FOLLOWUP_GOALS
    )
    found = Counter(tuple(_cell(row[column], column in numeric) for column in present) for row in rows)
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
    saved = "1 row" if len(rows) == 1 else f"{len(rows)} rows"
    raise AssertionError(
        f"{SUPPORTING_FILE} has {saved} where it should have the 8 rows of "
        "data/followup_goals.csv: " + "; ".join(problems) + ". In Task 3.1, save the three plotted columns "
        "of every row of followup, unchanged."
    )


def check_supporting_rows(root: Path) -> None:
    table = read_table(root, SUPPORTING_FILE, "Task 3.1")
    problem = None
    for rows in table.readings():
        try:
            _check_supporting_reading(table, rows)
            return
        except AssertionError as error:
            problem = error
    raise problem


CHECKS = (
    Check("exploratory spec: point mark", check_spec_mark),
    Check("exploratory spec: embedded patient rows", check_spec_rows),
    *(
        Check(f"exploratory spec: {channel} encoding", lambda root, channel=channel: _check_encoding(root, channel))
        for channel in ENCODINGS
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
    _REPORTED_OUTSIDE.clear()
    results = []
    for check in CHECKS:
        try:
            check.action(Path(root))
        except (AssertionError, OSError, ValueError, UnicodeDecodeError, csv.Error) as error:
            results.append((check.name, str(error)))
        except (AttributeError, KeyError, TypeError, RecursionError) as error:
            # JSON can nest anything anywhere; a shape no check anticipated fails this check alone.
            results.append((check.name, (
                f"a file this check reads is not laid out the way its task saves it ({type(error).__name__}: "
                f"{error}); compare it with its checkpoint in README.md, run that cell again, and rerun the checks."
            )))
        else:
            results.append((check.name, None))
    return results
