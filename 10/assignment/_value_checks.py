"""Checks for Assignment 10.

The course keeps these checks in 10/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They read the nine CSV files
and the one PNG a submission saves in output/ and compare them with values
computed from the supplied data files. The expected values below come from
the lecture's own calls (statsmodels OLS, a DummyRegressor mean baseline, and a
StandardScaler and LinearRegression pipeline); the self-test recomputes them
from 10/assignment/data/ and confirms they match.

Each check scores one thing, so one mistake costs only its own points, and no
check waits for another to pass. Values are compared after parsing: spacing,
line endings, quoting, a byte-order mark, column order, row order, a leading
row-number column, number formatting (2 == 2.0 == 2.00), the letter case of
labels and headers, and the way a timestamp is written never cost points, and
cells may be separated by commas, semicolons, or tabs. A number may be
rounded to as few as one decimal, and accuracy, precision, recall, and R² may
be written as percents (85% for 0.85). Timestamps are compared as instants, in
UTC when they carry a zone and read as UTC when they do not. A row label the
checks do not recognize costs the rows check alone: the values checks still
find the row by its other values. Booleans may be written True/False,
true/false, yes/no, or 1/0. Test predictions and test metrics are accepted
from either refit the lecture allows: the pipeline fitted on training rows
only, or refitted on training plus validation rows.

Nothing here imports, runs, or inspects student code.
"""

from __future__ import annotations

import codecs
import csv
import difflib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


# Two decimals are enough for coefficients and metrics; mmHg values may be rounded to one decimal.
FINE = 0.006
MMHG = 0.06
# Any number may also be rounded to fewer decimals, down to one: 3.3 stands for 3.34 (see _close).
# These columns hold proportions, which may also be written as a percent: 85% is 0.85.
PERCENT_COLUMNS = frozenset({"r2", "accuracy", "precision", "recall"})
# Ways pandas and people write an empty cell.
MISSING_SPELLINGS = frozenset({"", "nan", "na", "n/a", "none", "null", "<na>", "nat"})
TRUE_SPELLINGS = frozenset({"true", "t", "yes", "y", "1", "1.0"})
FALSE_SPELLINGS = frozenset({"false", "f", "no", "n", "0", "0.0"})


@dataclass(frozen=True)
class Instant:
    """An expected timestamp, written as UTC text such as 2026-05-09T18:30:00Z."""

    text: str


@dataclass(frozen=True)
class Flag:
    """An expected boolean."""

    value: bool


@dataclass(frozen=True)
class Starts:
    """An expected label: any cell that starts with this word, in any letter case."""

    word: str


@dataclass(frozen=True)
class Either:
    """An expected number with more than one accepted value."""

    values: tuple[float, ...]


@dataclass(frozen=True)
class Artifact:
    """One saved CSV: where it lives, which task writes it, and what it should hold.

    `rows` maps each required key, a tuple of the key columns' values, to the
    expected value of every other column. `optional_columns` may appear without
    counting as extra. `aliases` maps a key column to {other spelling: key value}.
    """

    path: str
    task: str
    columns: tuple[str, ...]
    key: tuple[str, ...]
    rows: dict[tuple, dict[str, object]]
    tolerance: float
    optional_columns: tuple[str, ...] = ()
    aliases: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Table:
    """A saved CSV: its casefolded column names and one {column: cell} dict per data row."""

    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


# Task 1: data/clinic_bp.csv, the model sbp ~ age + bmi, and the new patient aged 60 with BMI 31.0.
COEFFICIENT_VALUES = {
    "Intercept": {"coef": 70.288692, "std_err": 10.668834, "ci_lower": 47.77942, "ci_upper": 92.797964},
    "age": {"coef": 0.499716, "std_err": 0.121105, "ci_lower": 0.244208, "ci_upper": 0.755225},
    "bmi": {"coef": 1.608931, "std_err": 0.318119, "ci_lower": 0.937759, "ci_upper": 2.280104},
}
NEW_PATIENT = (60, 31.0)
NEW_PATIENT_VALUES = {
    "mean": 150.14854, "mean_ci_lower": 146.130215, "mean_ci_upper": 154.166864,
    "obs_ci_lower": 136.328789, "obs_ci_upper": 163.968291,
}
RESIDUAL_VALUES = {
    "P01": {"observed": 149, "fitted": 142.705821, "residual": 6.294179},
    "P02": {"observed": 164, "fitted": 163.224433, "residual": 0.775567},
    "P03": {"observed": 137, "fitted": 143.5368, "residual": -6.5368},
    "P04": {"observed": 162, "fitted": 160.708814, "residual": 1.291186},
    "P05": {"observed": 141, "fitted": 150.72588, "residual": -9.72588},
    "P06": {"observed": 137, "fitted": 139.149136, "residual": -2.149136},
    "P07": {"observed": 150, "fitted": 148.234859, "residual": 1.765141},
    "P08": {"observed": 147, "fitted": 142.637673, "residual": 4.362327},
    "P09": {"observed": 138, "fitted": 140.027394, "residual": -2.027394},
    "P10": {"observed": 150, "fitted": 139.824866, "residual": 10.175134},
    "P11": {"observed": 130, "fitted": 129.578817, "residual": 0.421183},
    "P12": {"observed": 160, "fitted": 147.330087, "residual": 12.669913},
    "P13": {"observed": 131, "fitted": 136.682712, "residual": -5.682712},
    "P14": {"observed": 147, "fitted": 156.169733, "residual": -9.169733},
    "P15": {"observed": 141, "fitted": 147.032899, "residual": -6.032899},
    "P16": {"observed": 136, "fitted": 139.149136, "residual": -3.149136},
    "P17": {"observed": 150, "fitted": 148.10804, "residual": 1.89196},
    "P18": {"observed": 159, "fitted": 154.409385, "residual": 4.590615},
    "P19": {"observed": 145, "fitted": 143.14495, "residual": 1.85505},
    "P20": {"observed": 118, "fitted": 119.618565, "residual": -1.618565},
}

# Task 2: data/feature_availability.csv, and data/followup_visits.csv split on followup_time
# at 2026-05-01 and 2026-05-09 UTC.
HOURS_AFTER_VISIT = {"age": 0, "bmi": 0, "sbp_today": 0, "a1c_result": 24, "callback_sbp": 72}
SPLIT_VALUES = {
    "train": {"row_count": 30, "first_target_time": "2026-04-01T18:18:00Z", "last_target_time": "2026-04-30T17:07:00Z"},
    "validation": {"row_count": 8, "first_target_time": "2026-05-01T15:52:00Z", "last_target_time": "2026-05-08T16:25:00Z"},
    "test": {"row_count": 10, "first_target_time": "2026-05-09T18:30:00Z", "last_target_time": "2026-05-18T21:51:00Z"},
}

# Task 3: the features age, bmi, and sbp_today, the target sbp_followup, and data/readmission_flags.csv.
VALIDATION_VALUES = {
    "mean_baseline": {"mae": 7.033333, "rmse": 8.931281, "r2": -0.233125},
    "linear_pipeline": {"mae": 3.337899, "rmse": 3.679567, "r2": 0.790698},
}
# The frozen pipeline fitted on training rows only, or refitted on training plus validation rows.
REFITS = ("train_only", "train_plus_validation")
TEST_METRIC_VALUES = {
    "train_only": {"mae": 2.640593, "rmse": 3.618139, "r2": 0.810934},
    "train_plus_validation": {"mae": 2.539112, "rmse": 3.558349, "r2": 0.817131},
}
TEST_PREDICTION_VALUES = {
    "V39": {"followup_time": "2026-05-09T18:30:00Z", "sbp_followup": 155, "train_only": 154.811131, "train_plus_validation": 154.935364},
    "V40": {"followup_time": "2026-05-10T16:50:00Z", "sbp_followup": 142, "train_only": 142.965453, "train_plus_validation": 143.804302},
    "V41": {"followup_time": "2026-05-11T15:08:00Z", "sbp_followup": 149, "train_only": 139.91313, "train_plus_validation": 140.132336},
    "V42": {"followup_time": "2026-05-12T20:15:00Z", "sbp_followup": 152, "train_only": 149.241207, "train_plus_validation": 149.364313},
    "V43": {"followup_time": "2026-05-13T15:26:00Z", "sbp_followup": 139, "train_only": 142.550643, "train_plus_validation": 142.460547},
    "V44": {"followup_time": "2026-05-14T16:57:00Z", "sbp_followup": 148, "train_only": 150.4136, "train_plus_validation": 150.285056},
    "V45": {"followup_time": "2026-05-15T18:29:00Z", "sbp_followup": 141, "train_only": 141.101165, "train_plus_validation": 141.111434},
    "V46": {"followup_time": "2026-05-16T18:23:00Z", "sbp_followup": 135, "train_only": 138.785459, "train_plus_validation": 139.017202},
    "V47": {"followup_time": "2026-05-17T15:49:00Z", "sbp_followup": 165, "train_only": 163.80024, "train_plus_validation": 164.959612},
    "V48": {"followup_time": "2026-05-18T21:51:00Z", "sbp_followup": 150, "train_only": 152.355318, "train_plus_validation": 152.104206},
}
READMISSION_VALUES = {
    "model_flag": {"accuracy": 0.85, "precision": 0.6, "recall": 0.75},
    "never_flag": {"accuracy": 0.8, "precision": 0.0, "recall": 0.0},
}


COEFFICIENTS = Artifact(
    path="output/ols_coefficients.csv",
    task="Task 1.1",
    columns=("term", "coef", "std_err", "ci_lower", "ci_upper"),
    key=("term",),
    rows={(term,): values for term, values in COEFFICIENT_VALUES.items()},
    tolerance=FINE,
    # The array interface, sm.OLS with sm.add_constant(), names the intercept const.
    aliases={"term": {"const": "Intercept"}},
)

NEW_PATIENT_INTERVALS = Artifact(
    path="output/new_patient_intervals.csv",
    task="Task 1.2",
    columns=("age", "bmi", *NEW_PATIENT_VALUES),
    key=("age", "bmi"),
    rows={NEW_PATIENT: NEW_PATIENT_VALUES},
    tolerance=MMHG,
    # summary_frame() also returns mean_se; keeping it is fine.
    optional_columns=("mean_se",),
)

RESIDUALS = Artifact(
    path="output/ols_residuals.csv",
    task="Task 1.3",
    columns=("patient_id", "observed", "fitted", "residual"),
    key=("patient_id",),
    rows={(patient,): values for patient, values in RESIDUAL_VALUES.items()},
    tolerance=MMHG,
)

AVAILABILITY = Artifact(
    path="output/availability_decisions.csv",
    task="Task 2.1",
    columns=("candidate_feature", "hours_after_visit", "available", "decision"),
    key=("candidate_feature",),
    rows={
        (feature,): {
            "hours_after_visit": hours,
            "available": Flag(hours <= 0),
            "decision": Starts("keep" if hours <= 0 else "exclude"),
        }
        for feature, hours in HOURS_AFTER_VISIT.items()
    },
    tolerance=FINE,
)

SPLIT = Artifact(
    path="output/split_summary.csv",
    task="Task 2.2",
    columns=("partition", "row_count", "first_target_time", "last_target_time"),
    key=("partition",),
    rows={
        (partition,): {
            "row_count": values["row_count"],
            "first_target_time": Instant(values["first_target_time"]),
            "last_target_time": Instant(values["last_target_time"]),
        }
        for partition, values in SPLIT_VALUES.items()
    },
    tolerance=FINE,
    aliases={"partition": {"training": "train", "valid": "validation", "val": "validation", "testing": "test"}},
)

VALIDATION = Artifact(
    path="output/validation_metrics.csv",
    task="Task 3.1",
    columns=("approach", "mae", "rmse", "r2"),
    key=("approach",),
    rows={(approach,): values for approach, values in VALIDATION_VALUES.items()},
    tolerance=FINE,
)

TEST_METRICS = Artifact(
    path="output/test_metrics.csv",
    task="Task 3.2",
    columns=("approach", "mae", "rmse", "r2"),
    key=("approach",),
    rows={
        ("linear_pipeline",): {
            metric: Either(tuple(TEST_METRIC_VALUES[refit][metric] for refit in REFITS))
            for metric in ("mae", "rmse", "r2")
        }
    },
    tolerance=FINE,
)

TEST_PREDICTIONS = Artifact(
    path="output/test_predictions.csv",
    task="Task 3.2",
    columns=("visit_id", "followup_time", "sbp_followup", "predicted_sbp"),
    key=("visit_id",),
    rows={
        (visit,): {
            "followup_time": Instant(values["followup_time"]),
            "sbp_followup": values["sbp_followup"],
            "predicted_sbp": Either(tuple(values[refit] for refit in REFITS)),
        }
        for visit, values in TEST_PREDICTION_VALUES.items()
    },
    tolerance=MMHG,
)

READMISSION = Artifact(
    path="output/readmission_metrics.csv",
    task="Task 3.3",
    columns=("approach", "accuracy", "precision", "recall"),
    key=("approach",),
    rows={(approach,): values for approach, values in READMISSION_VALUES.items()},
    tolerance=FINE,
)

ARTIFACTS = (
    COEFFICIENTS, NEW_PATIENT_INTERVALS, RESIDUALS, AVAILABILITY, SPLIT,
    VALIDATION, TEST_METRICS, TEST_PREDICTIONS, READMISSION,
)
RESIDUAL_PLOT = "output/residuals_vs_fitted.png"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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


def _label(cell: str) -> str:
    """A label compared in any letter case, with spaces, hyphens, and underscores alike."""
    return re.sub(r"[\s_-]+", "_", cell.strip().casefold())


def _fold(column: str) -> str:
    """A column name compared the way labels are."""
    return _label(column)


def _number(cell: str) -> float | None:
    try:
        return float(_clean(cell).replace(",", ""))
    except ValueError:
        return None


def _is_whole_number(cell: str) -> bool:
    number = _number(cell)
    return number is not None and number.is_integer()


def _instant(cell: str) -> datetime | None:
    """A timestamp as a UTC instant; one written without a zone is read as UTC.

    A trailing Z, UTC, or GMT, as strftime("%Z") writes it, names UTC.
    """
    text = _clean(cell)
    zone = re.search(r"\s*(?:z|utc|gmt)$", text, re.IGNORECASE)
    if zone:
        text = text[:zone.start()] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _flag(cell: str) -> bool | None:
    text = _clean(cell).casefold()
    if text in TRUE_SPELLINGS:
        return True
    if text in FALSE_SPELLINGS:
        return False
    return None


def _close(cell: str, expected: float, tolerance: float, percent: bool = False) -> bool:
    """Whether a number cell holds `expected`, within `tolerance` or the rounding its decimals show.

    A number written with one or more decimals is read as rounded there, so it
    passes within half of its last decimal: 3.3 for 3.34, or 0.8 for 0.79. With
    `percent`, a cell ending in % is a percent of the proportion: 85% is 0.85.
    """
    text = _clean(cell).replace(",", "")
    scale = 1
    if percent and text.endswith("%"):
        text, scale = text[:-1].strip(), 100
    try:
        written = Decimal(text)
    except InvalidOperation:
        return False
    if not written.is_finite():
        return False
    decimals = -written.as_tuple().exponent + (2 if scale == 100 else 0)
    allowed = max(tolerance, 0.5 * 10.0 ** -decimals + 1e-9) if decimals >= 1 else tolerance
    return abs(float(written) / scale - expected) <= allowed


def _matches(cell: str, expected: object, tolerance: float, percent: bool = False) -> bool:
    text = _clean(cell)
    if expected is None:
        return text.casefold() in MISSING_SPELLINGS
    if isinstance(expected, Instant):
        return _instant(text) is not None and _instant(text) == _instant(expected.text)
    if isinstance(expected, Flag):
        return _flag(text) is expected.value
    if isinstance(expected, Starts):
        return _label(text).startswith(expected.word)
    if isinstance(expected, Either):
        return any(_close(text, value, tolerance, percent) for value in expected.values)
    if isinstance(expected, (int, float)):
        return _close(text, expected, tolerance, percent)
    return _label(text) == _label(str(expected))


def _key_matches(cell: str, part: object, artifact: Artifact, column: str) -> bool:
    """Whether a saved key cell names this expected key value, allowing the column's other spellings."""
    if isinstance(part, (int, float)):
        return _close(cell, part, artifact.tolerance)
    spelled = _label(cell)
    spelled = {_label(alias): _label(value) for alias, value in artifact.aliases.get(column, {}).items()}.get(spelled, spelled)
    return spelled == _label(str(part))


def _show(value: object) -> str:
    if value is None:
        return "blank"
    if isinstance(value, Instant):
        return value.text
    if isinstance(value, Flag):
        return str(value.value)
    if isinstance(value, Starts):
        return f'a label starting with "{value.word}"'
    if isinstance(value, Either):
        return " or ".join(_show(item) for item in value.values)
    if isinstance(value, float):
        return f"{round(value, 2):g}"
    return str(value)


def _given(cell: str) -> str:
    text = _clean(cell)
    return text if text else "blank"


def _key_name(key: tuple, artifact: Artifact) -> str:
    if len(key) == 1:
        return str(key[0])
    return " and ".join(f"{column} {part}" for column, part in zip(artifact.key, key))


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


def _missing(root: Path, path: str, task: str) -> str:
    message = f"{path} is missing; run the {task} cell to write it, then commit it."
    wanted = root / path
    if wanted.parent.is_dir():
        # Another required file, such as validation_metrics.csv beside readmission_metrics.csv, is never a look-alike.
        required = {artifact.path.casefold() for artifact in ARTIFACTS} | {RESIDUAL_PLOT.casefold()}
        look_alikes = sorted(
            other.relative_to(root).as_posix()
            for other in wanted.parent.iterdir()
            if other.is_file()
            and other.relative_to(root).as_posix().casefold() not in required
            and difflib.SequenceMatcher(None, wanted.name.casefold(), other.name.casefold()).ratio() >= 0.75
        )
        if look_alikes:
            message += f" Found {_join(look_alikes)}; save it as {path} instead."
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
    _assert(path is not None, _missing(root, artifact.path, artifact.task))
    try:
        lines = _csv_rows(_decode(path.read_bytes()))
    except csv.Error as error:
        raise AssertionError(
            f"{artifact.path} cannot be read as a CSV table ({error}); run the {artifact.task} cell again so "
            "to_csv() writes it, then compare it with the checkpoint in README.md."
        ) from None
    _assert(lines, f"{artifact.path} is empty; run the {artifact.task} cell again to write it.")
    header = [_fold(cell) for cell in lines[0]]
    body = [[_clean(cell) for cell in row] for row in lines[1:]]
    width = max(len(header), *(len(row) for row in body)) if body else len(header)
    header += [""] * (width - len(header))
    body = [row + [""] * (width - len(row)) for row in body]

    # Unless the file has no column named for its first key and this column holds those keys instead.
    first_is_key = _fold(artifact.key[0]) not in header and body and all(
        any(_key_matches(row[0], key[0], artifact, artifact.key[0]) for key in artifact.rows) for row in body
    )
    if (len(header) > 1 and _is_index_header(header[0]) and body and not first_is_key
            and all(_is_whole_number(row[0]) for row in body)):
        header = header[1:]
        body = [row[1:] for row in body]
    keep = [i for i, column in enumerate(header) if column or any(row[i] for row in body)]
    header = [header[i] for i in keep]
    body = [[row[i] for i in keep] for row in body]

    rows = tuple({column: row[i] for i, column in enumerate(header)} for row in body)
    return Table(columns=tuple(header), rows=rows)


def _resolve(table: Table, artifact: Artifact) -> dict[str, str]:
    """Map each expected column to the saved column that holds it.

    A column saved under its own name, in any letter case, is found by name. A
    key column saved under another name, such as a blank header over saved
    index labels, is found by its values. If exactly one expected value column
    is still unplaced and exactly one unexpected column is left, that column is
    taken to hold it. A misnamed column then costs only the columns check.
    """
    expected = [_fold(column) for column in (*artifact.columns, *artifact.optional_columns)]
    found = {column: _fold(column) for column in artifact.columns if _fold(column) in table.columns}
    spare = [column for column in table.columns if column not in expected]
    for position, column in enumerate(artifact.key):
        if column in found:
            continue
        candidates = []
        for other in spare:
            values = [row[other] for row in table.rows if row[other]]
            hits = sum(any(_key_matches(value, key[position], artifact, column) for key in artifact.rows) for value in values)
            if values and hits * 2 > len(values):
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


def _by_values(row: dict[str, str], artifact: Artifact, found: dict[str, str], claimed, checked: tuple[str, ...]):
    """The expected key whose values this row holds, for a row its key cells do not name, or None.

    The row is recognized by its value columns other than `checked`, so a wrong
    value in a checked column is still compared instead of hiding the row. When
    those values fit several keys, as three kept features do, the row goes to a
    key it fits in every column, one not yet claimed if there is one.
    """
    values = [column for column in artifact.columns if column not in artifact.key and column in found]
    others = [column for column in values if column not in checked] or values

    def fits(key: tuple, columns: list[str]) -> bool:
        return all(
            _key_matches(row[found[column]], key[position], artifact, column)
            for position, column in enumerate(artifact.key)
            if column in found
        ) and all(
            _matches(row[found[column]], artifact.rows[key][column], artifact.tolerance, column in PERCENT_COLUMNS)
            for column in columns
        )

    candidates = [key for key in artifact.rows if fits(key, others)]
    if len(candidates) == 1:
        return candidates[0]
    exact = [key for key in candidates if fits(key, values)]
    unclaimed = [key for key in exact if key not in claimed]
    return (unclaimed or exact or [None])[0]


def _group(table: Table, artifact: Artifact, found: dict[str, str], checked: tuple[str, ...] | None = None):
    """({expected key: [rows]}, [rows naming no expected key]) for the saved table.

    When every key column is saved, each row is matched by its key. When one is
    not, each row is matched by the values it does hold, so the lost column
    costs the columns check alone. A values check passes the columns it
    compares as `checked`; rows whose key cells name no expected key are then
    matched by their other values too, so a wrong or unexpected label costs the
    rows check alone.
    """
    grouped: dict[tuple, list[dict[str, str]]] = {}
    unknown = []
    if len(artifact.rows) == 1 and len(table.rows) == 1:
        # A one-row answer saved as one row is compared as it stands; a wrong or missing
        # key value there costs the rows check, not the values checks.
        only, row = next(iter(artifact.rows)), table.rows[0]
        named = all(
            _key_matches(row[found[column]], only[position], artifact, column)
            for position, column in enumerate(artifact.key)
            if column in found
        )
        if not named:
            unknown.append("a row for " + " ".join(row[found[column]] or "blank" for column in artifact.key if column in found))
        return {only: [row]}, unknown
    if all(column in found for column in artifact.key):
        unnamed = []
        for row in table.rows:
            match = next(
                (
                    key for key in artifact.rows
                    if all(
                        _key_matches(row[found[column]], key[position], artifact, column)
                        for position, column in enumerate(artifact.key)
                    )
                ),
                None,
            )
            if match is None:
                unnamed.append(row)
            else:
                grouped.setdefault(match, []).append(row)
        if checked is None:
            unknown = ["a row for " + " ".join(row[found[column]] or "blank" for column in artifact.key)
                       for row in unnamed]
            return grouped, unknown
        # The rows check has already charged for these labels; read the values beneath them anyway.
        found = {column: saved for column, saved in found.items() if column not in artifact.key}
        rows = unnamed
    else:
        rows = list(table.rows)

    values = [column for column in artifact.columns if column not in artifact.key and column in found]
    absent = [column for column in artifact.key if column not in found]
    _assert(
        values,
        f"{artifact.path} has no {_join(absent)} column (its columns are {_join(table.columns) or 'none'}), "
        f"so its rows cannot be matched; {artifact.task} saves the header line {','.join(artifact.columns)}.",
    )
    for row in rows:
        match = _by_values(row, artifact, found, grouped, checked or ())
        if match is None:
            unknown.append(_describe(row, [found[column] for column in artifact.columns if column in found]))
        else:
            grouped.setdefault(match, []).append(row)
    return grouped, unknown


def columns_check(artifact: Artifact, hint: str) -> Callable[[Path], None]:
    """The saved header holds exactly the expected columns, in any order."""

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact)
        # A saved index with no name leaves its column unheaded; holding the keys, it is the key column.
        first = table.columns[0] if table.columns else None
        unheaded = {
            column: found[column]
            for column in artifact.key
            if _fold(column) not in table.columns and first is not None and found.get(column) == first
            and _is_index_header(first)
        }
        expected = [_fold(column) for column in (*artifact.columns, *artifact.optional_columns)]
        missing = [
            column for column in artifact.columns
            if _fold(column) not in table.columns and column not in unheaded
        ]
        extra = [
            column or "a column with no header"
            for column in table.columns
            if column not in expected and column not in unheaded.values()
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
    """Each required key appears on exactly one row, and no unexpected key appears."""

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        grouped, unknown = _group(table, artifact, _resolve(table, artifact))
        missing = [key for key in artifact.rows if key not in grouped]
        repeated = [key for key, rows in grouped.items() if len(rows) > 1]
        problems = []
        if missing:
            problems.append(f"is missing {_join(_key_name(key, artifact) for key in missing)}")
        if repeated:
            problems.append("lists " + _join(f"{_key_name(key, artifact)} {len(grouped[key])} times" for key in repeated))
        if unknown:
            shown = _join(unknown[:4]) + (f" and {len(unknown) - 4} more" if len(unknown) > 4 else "")
            problems.append(f"also has {shown}")
        names = [_key_name(key, artifact) for key in artifact.rows]
        wanted = f"one row for each of {_expected_keys(names)}" if len(names) > 1 else f"one row, for {names[0]}"
        _assert(
            not problems,
            f"{artifact.path} " + "; ".join(problems) + f"; it should hold {wanted}. Fix it in {artifact.task}: {hint}",
        )

    return check


def values_check(artifact: Artifact, columns: tuple[str, ...], hint: str) -> Callable[[Path], None]:
    """Every saved row for an expected key holds the expected values in `columns`.

    A missing or repeated row costs the rows check, not this one: this check
    compares whatever rows the file does hold for the expected keys.
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
        grouped, _ = _group(table, artifact, found, columns)
        _assert(
            grouped,
            f"{artifact.path} has no row for {_join(_key_name(key, artifact) for key in artifact.rows)}, so its "
            f"{_join(columns)} values cannot be compared; the rows check says what to fix in {artifact.task}.",
        )
        wrong = []
        for key, rows in grouped.items():
            for column in columns:
                expected = artifact.rows[key][column]
                given = sorted({
                    _given(row[found[column]]) for row in rows
                    if not _matches(row[found[column]], expected, artifact.tolerance, column in PERCENT_COLUMNS)
                })
                if given:
                    wrong.append(f"{_key_name(key, artifact)} has {column} {_join(given)}, expected {_show(expected)}")
        _assert(not wrong, f"{artifact.path}: " + _listed(wrong) + f". Fix it in {artifact.task}: {hint}")

    return check


def png_check(root: Path) -> None:
    """The residual plot is saved as a PNG image."""
    path = _artifact_path(root, RESIDUAL_PLOT)
    _assert(path is not None, _missing(root, RESIDUAL_PLOT, "Task 1.3"))
    start = path.read_bytes()[:8]
    _assert(
        start == PNG_SIGNATURE,
        f"{RESIDUAL_PLOT} is not a PNG image (its first bytes are {start!r}); in Task 1.3 save the figure with "
        f'fig.savefig(RESIDUAL_PLOT_PATH), keeping the .png ending so matplotlib writes a PNG.',
    )


FIT_HINT = 'fit smf.ols("sbp ~ age + bmi", data=patients) on every row of data/clinic_bp.csv.'
PIPELINE_HINT = (
    "fit each approach on the training rows only, with the features age, bmi, and sbp_today and the target "
    "sbp_followup, then score its predictions for the validation rows."
)
TEST_HINT = (
    "predict the test rows once with the frozen linear pipeline, fitted on the training rows or refitted on "
    "training plus validation rows, using the features age, bmi, and sbp_today."
)

CHECKS = (
    # Task 1.1: output/ols_coefficients.csv
    Check(
        "coefficients: columns",
        columns_check(
            COEFFICIENTS,
            "Build the table from results.params, results.bse, and the two columns of results.conf_int(), name "
            'its index with coefficients.index.name = "term", and save it without index=False so the terms '
            "become the first column.",
        ),
    ),
    Check(
        "coefficients: one row per term",
        rows_check(COEFFICIENTS, "the model sbp ~ age + bmi has three terms: Intercept, age, and bmi."),
    ),
    Check("coefficients: coef values", values_check(COEFFICIENTS, ("coef",), f"coef is results.params; {FIT_HINT}")),
    Check("coefficients: std_err values", values_check(COEFFICIENTS, ("std_err",), f"std_err is results.bse; {FIT_HINT}")),
    Check(
        "coefficients: confidence interval values",
        values_check(
            COEFFICIENTS,
            ("ci_lower", "ci_upper"),
            "ci_lower and ci_upper are columns 0 and 1 of results.conf_int(alpha=0.05), the 95% interval.",
        ),
    ),
    # Task 1.2: output/new_patient_intervals.csv
    Check(
        "new-patient intervals: columns",
        columns_check(
            NEW_PATIENT_INTERVALS,
            "Put the new patient's age and bmi beside summary_frame(alpha=0.05) with pd.concat([...], axis=1); "
            "keeping mean_se is fine.",
        ),
    ),
    Check(
        "new-patient intervals: one row for the new patient",
        rows_check(NEW_PATIENT_INTERVALS, "predict for one new patient, age 60 with BMI 31.0."),
    ),
    Check(
        "new-patient intervals: mean",
        values_check(
            NEW_PATIENT_INTERVALS, ("mean",),
            "mean is the fitted SBP from results.get_prediction(new_patient).summary_frame(alpha=0.05).",
        ),
    ),
    Check(
        "new-patient intervals: mean-response interval",
        values_check(
            NEW_PATIENT_INTERVALS, ("mean_ci_lower", "mean_ci_upper"),
            "mean_ci_lower and mean_ci_upper bound the average SBP of patients like this one.",
        ),
    ),
    Check(
        "new-patient intervals: prediction interval",
        values_check(
            NEW_PATIENT_INTERVALS, ("obs_ci_lower", "obs_ci_upper"),
            "obs_ci_lower and obs_ci_upper bound one such patient's SBP; this interval is the wider one.",
        ),
    ),
    # Task 1.3: output/ols_residuals.csv and output/residuals_vs_fitted.png
    Check(
        "residuals: columns",
        columns_check(
            RESIDUALS,
            "Build the table from patients['patient_id'], patients['sbp'], results.fittedvalues, and results.resid.",
        ),
    ),
    Check("residuals: one row per patient", rows_check(RESIDUALS, "keep one row per patient, P01 to P20.")),
    Check(
        "residuals: observed values",
        values_check(RESIDUALS, ("observed",), "observed is each patient's sbp from data/clinic_bp.csv, unchanged."),
    ),
    Check("residuals: fitted values", values_check(RESIDUALS, ("fitted",), f"fitted is results.fittedvalues; {FIT_HINT}")),
    Check(
        "residuals: residual values",
        values_check(RESIDUALS, ("residual",), "residual is results.resid, observed minus fitted."),
    ),
    Check("residual plot: PNG image", png_check),
    # Task 2.1: output/availability_decisions.csv
    Check(
        "availability: columns",
        columns_check(
            AVAILABILITY,
            "Start from a .copy() of the supplied candidates table and add the columns available and decision.",
        ),
    ),
    Check(
        "availability: one row per candidate",
        rows_check(AVAILABILITY, "keep one row for each candidate in data/feature_availability.csv."),
    ),
    Check(
        "availability: hours_after_visit values",
        values_check(
            AVAILABILITY, ("hours_after_visit",),
            "hours_after_visit is copied unchanged from data/feature_availability.csv.",
        ),
    ),
    Check(
        "availability: available values",
        values_check(
            AVAILABILITY, ("available",),
            "available is hours_after_visit <= 0: True when the feature is known by the end of the visit.",
        ),
    ),
    Check(
        "availability: decision values",
        values_check(
            AVAILABILITY, ("decision",),
            'decision is "Keep" where available is True and "Exclude (leakage)" where it is False, as '
            "np.where(available, ...) writes it; any label starting with keep or exclude counts.",
        ),
    ),
    # Task 2.2: output/split_summary.csv
    Check(
        "split summary: columns",
        columns_check(
            SPLIT,
            "Build one row per partition with its len() and the min() and max() of its followup_time.",
        ),
    ),
    Check(
        "split summary: one row per partition",
        rows_check(SPLIT, "name the three partitions train, validation, and test."),
    ),
    Check(
        "split summary: row_count values",
        values_check(
            SPLIT, ("row_count",),
            "split on followup_time, the target time: train is before VALIDATION_START, validation runs from "
            "VALIDATION_START up to TEST_START, and test is from TEST_START on.",
        ),
    ),
    Check(
        "split summary: first_target_time values",
        values_check(
            SPLIT, ("first_target_time",),
            "first_target_time is the partition's followup_time.min(), the full timestamp.",
        ),
    ),
    Check(
        "split summary: last_target_time values",
        values_check(
            SPLIT, ("last_target_time",),
            "last_target_time is the partition's followup_time.max(), the full timestamp.",
        ),
    ),
    # Task 3.1: output/validation_metrics.csv
    Check(
        "validation metrics: columns",
        columns_check(VALIDATION, "Save one row per approach with its MAE, RMSE, and R² on the validation rows."),
    ),
    Check(
        "validation metrics: one row per approach",
        rows_check(VALIDATION, "name the two approaches mean_baseline and linear_pipeline."),
    ),
    Check(
        "validation metrics: mae values",
        values_check(VALIDATION, ("mae",), f"mae is mean_absolute_error(actual, predicted); {PIPELINE_HINT}"),
    ),
    Check(
        "validation metrics: rmse values",
        values_check(VALIDATION, ("rmse",), f"rmse is np.sqrt(mean_squared_error(actual, predicted)); {PIPELINE_HINT}"),
    ),
    Check(
        "validation metrics: r2 values",
        values_check(VALIDATION, ("r2",), f"r2 is r2_score(actual, predicted); {PIPELINE_HINT}"),
    ),
    # Task 3.2: output/test_metrics.csv and output/test_predictions.csv
    Check(
        "test metrics: columns",
        columns_check(TEST_METRICS, "Save one row with the frozen approach's MAE, RMSE, and R² on the test rows."),
    ),
    Check(
        "test metrics: one row for the frozen approach",
        rows_check(
            TEST_METRICS,
            "validation chose linear_pipeline, so the test rows are scored once, for that approach only.",
        ),
    ),
    Check("test metrics: mae value", values_check(TEST_METRICS, ("mae",), TEST_HINT)),
    Check("test metrics: rmse value", values_check(TEST_METRICS, ("rmse",), TEST_HINT)),
    Check("test metrics: r2 value", values_check(TEST_METRICS, ("r2",), TEST_HINT)),
    Check(
        "test predictions: columns",
        columns_check(
            TEST_PREDICTIONS,
            "Start from the test rows' visit_id, followup_time, and sbp_followup and add predicted_sbp.",
        ),
    ),
    Check(
        "test predictions: one row per test visit",
        rows_check(TEST_PREDICTIONS, "keep one row per test visit, V39 to V48, and no other visits."),
    ),
    Check(
        "test predictions: followup_time and sbp_followup values",
        values_check(
            TEST_PREDICTIONS, ("followup_time", "sbp_followup"),
            "copy each test visit's followup_time and sbp_followup unchanged from data/followup_visits.csv.",
        ),
    ),
    Check(
        "test predictions: predicted_sbp values",
        values_check(TEST_PREDICTIONS, ("predicted_sbp",), TEST_HINT),
    ),
    # Task 3.3: output/readmission_metrics.csv
    Check(
        "readmission metrics: columns",
        columns_check(READMISSION, "Save one row per flag column with its accuracy, precision, and recall."),
    ),
    Check(
        "readmission metrics: one row per flag",
        rows_check(READMISSION, "name each row after its column in data/readmission_flags.csv: model_flag and never_flag."),
    ),
    Check(
        "readmission metrics: accuracy values",
        values_check(READMISSION, ("accuracy",), "accuracy is accuracy_score(readmitted_30d, flag)."),
    ),
    Check(
        "readmission metrics: precision values",
        values_check(
            READMISSION, ("precision",),
            "precision is precision_score(readmitted_30d, flag, zero_division=0), which is 0 for a flag that "
            "never fires.",
        ),
    ),
    Check(
        "readmission metrics: recall values",
        values_check(READMISSION, ("recall",), "recall is recall_score(readmitted_30d, flag)."),
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
