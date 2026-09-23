"""Shape checks for Assignment 03.

These are the checks that run in your own repository. They answer one question:
is each artifact well formed? Every required file exists and is readable UTF-8,
every required label is present, and every value parses and sits inside a range
a clinician would accept. Nothing here recomputes an answer from
`data/bp_readings.csv`, so nothing here can tell you whether a value is right;
your values are checked when you push.

The course checks that run on GitHub carry the same check names, order, and
point values, and add the comparison against the supplied readings.

Nothing here imports, runs, or inspects student source code.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


SUMMARY_FILE = "output/vitals_summary.txt"
RECORD_COUNT_FILE = "output/record_count.txt"
ENVIRONMENT_FILE = "output/environment.txt"
MONITOR_COUNTS_PATTERN = re.compile(r"^monitor_counts_\d{8}_\d{6}\.txt$")
MONITOR_COUNTS_NAME = "output/monitor_counts_<timestamp>.txt"


# What `check_assignment.py` tells the reader this run verified.
CHECKS_SCOPE = "shape"
SCOPE_NOTE = "These checks confirm the shape of your artifacts; your values are checked when you push."
SCORE_LABEL = "Shape"
COMPLETE_NOTE = "Every artifact is well formed. Your values are checked when you push."

_NUMPY_SCALAR = re.compile(r"(?:np|numpy)\.[A-Za-z_]+\d*\(\s*([^()]*?)\s*\)")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_VERSION = re.compile(r"\d+(?:\.\d+)+")
_PINNED_NUMPY = re.compile(r"^numpy\s*==\s*([0-9][^\s;#]*)", re.IGNORECASE)
# A path to an interpreter inside a .venv directory, on either platform.
_VENV_INTERPRETER = re.compile(r"(?:^|[/\\])\.venv[/\\].*python", re.IGNORECASE)
_COUNT_SPLIT = re.compile(r"[\s,;:=|]+")


@dataclass(frozen=True)
class PublicCheck:
    name: str
    action: Callable[[Path], None]


@dataclass(frozen=True)
class AnswerShape:
    """What a value must look like: a label, or a number inside a wide band."""

    band: str
    low: float | None = None
    high: float | None = None
    whole: bool = False
    label: bool = False


_COUNT = AnswerShape("a whole number, zero or more", low=0, whole=True)
_MMHG = AnswerShape("a systolic pressure in mmHg, between 60 and 250", low=60, high=250)
_WHOLE_MMHG = AnswerShape("a whole number of mmHg, between 60 and 250", low=60, high=250, whole=True)

# Report order: the order README.md lists the answers in.
ANSWER_SHAPES = {
    "patients": AnswerShape("a whole number, one or more", low=1, whole=True),
    "readings": AnswerShape("a whole number, one or more", low=1, whole=True),
    "mean_sbp": _MMHG,
    "sd_sbp": AnswerShape("a spread in mmHg, between 0 and 100", low=0, high=100),
    "min_sbp": _WHOLE_MMHG,
    "max_sbp": _WHOLE_MMHG,
    "stage2_patients": _COUNT,
    "highest_patient": AnswerShape("a patient_id as written in the file", label=True),
    "highest_patient_mean": _MMHG,
    "peak_hour_column": AnswerShape("an hour column name as written in the header", label=True),
    "peak_hour_mean": _MMHG,
    "high_monitor": AnswerShape("a monitor id as written in the file", label=True),
    "monitor_offset": AnswerShape("a gap in mmHg, between -100 and 100", low=-100, high=100),
    "stage2_other_monitors": _COUNT,
}
ANSWER_KEYS = tuple(ANSWER_SHAPES)
LABEL_KEYS = tuple(key for key, shape in ANSWER_SHAPES.items() if shape.label)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _text(root: Path, name: str) -> str:
    path = root / name
    _assert(path.is_file() and not path.is_symlink(), f"{name} is missing; commit it as a regular file.")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError(f"{name} must be UTF-8 text.") from error
    return text.lstrip("﻿")


def _unwrap(raw: str) -> str:
    """Rewrite a NumPy scalar repr as the value inside it: np.int64(96) -> 96."""
    return _NUMPY_SCALAR.sub(lambda match: match.group(1), raw)


def _as_number(raw: str) -> float | None:
    match = _NUMBER.search(_unwrap(raw))
    if match is None:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def _as_label(raw: str) -> str:
    return _unwrap(raw).strip().strip("`'\"").strip()


def read_summary(root: Path) -> dict[str, str]:
    """Return the `key: value` pairs in the summary artifact, however it is spaced."""
    pairs: dict[str, str] = {}
    for line in _text(root, SUMMARY_FILE).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().strip("`*-").strip().casefold().replace(" ", "_").replace("-", "_")
        if key and key not in pairs:
            pairs[key] = value.strip()
    return pairs


def read_environment(root: Path) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in _text(root, ENVIRONMENT_FILE).splitlines():
        key, separator, value = line.strip().partition(":")
        if not separator:
            continue
        key = key.strip().strip("`*-").strip().casefold()
        if key and key not in pairs:
            pairs[key] = value.strip()
    return pairs


def read_count_pairs(text: str) -> dict[str, int]:
    """Return {label: count} for every line that pairs a count with a label.

    `uniq -c` order, the reverse order, and separators such as `M01: 58` or
    `M01,58` all read the same, because README.md promises spacing and line
    order do not matter.
    """
    pairs: dict[str, int] = {}
    for line in text.splitlines():
        fields = [field for field in _COUNT_SPLIT.split(line.strip()) if field]
        if len(fields) < 2:
            continue
        if fields[0].isdigit() and not fields[-1].isdigit():
            count, label = fields[0], " ".join(fields[1:])
        elif fields[-1].isdigit() and not fields[0].isdigit():
            count, label = fields[-1], " ".join(fields[:-1])
        else:
            continue
        label = label.strip("`'\"").strip()
        if label:
            pairs[label.casefold()] = int(count)
    return pairs


def _report(problems: list[str]) -> None:
    _assert(not problems, " ".join(problems))



def check_environment_probe(root: Path) -> None:
    """`output/environment.txt` reports the environment the analysis ran in."""
    probe = read_environment(root)
    problems: list[str] = []

    for key in ("numpy", "interpreter"):  # the python line is never graded
        if key not in probe:
            problems.append(f"{ENVIRONMENT_FILE} has no `{key}` line.")
    if problems:
        _report(problems)

    pinned = None
    for line in _text(root, "requirements.txt").splitlines():
        match = _PINNED_NUMPY.match(line.strip())
        if match:
            pinned = match.group(1)
            break
    reported = _VERSION.search(probe["numpy"])
    if pinned is None:
        problems.append("requirements.txt pins no numpy version to compare the probe against.")
    elif reported is None or reported.group() != pinned:
        problems.append(
            f"{ENVIRONMENT_FILE} reports numpy `{probe['numpy']}`, which is not the version "
            "requirements.txt pins; install the requirements into the environment you ran."
        )

    if _VENV_INTERPRETER.search(probe["interpreter"]) is None:
        problems.append(
            f"{ENVIRONMENT_FILE} reports interpreter `{probe['interpreter']}`, which is not a path to "
            "an interpreter inside the project's .venv; activate the environment before saving the probe."
        )

    _report(problems)


def check_record_count(root: Path) -> None:
    """`output/record_count.txt` holds a count."""
    given = _as_number(_text(root, RECORD_COUNT_FILE))
    _assert(given is not None, f"{RECORD_COUNT_FILE} contains no number.")
    _assert(
        given >= 0 and float(given).is_integer(),
        f"{RECORD_COUNT_FILE} records {given:g}, which is not a whole number of patient records.",
    )


def _timestamped_counts_files(root: Path) -> list[Path]:
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Create a regular output/ directory.")
    candidates = sorted(path for path in output.glob("monitor_counts*.txt") if path.is_file())
    timestamped = [path for path in candidates if MONITOR_COUNTS_PATTERN.match(path.name)]
    _assert(
        timestamped,
        f"No {MONITOR_COUNTS_NAME} found"
        + (
            f" (output/ holds {', '.join(path.name for path in candidates)}); name the file with the "
            "run timestamp, as YYYYMMDD_HHMMSS."
            if candidates
            else "; save the per-monitor counts under a name that carries the run timestamp, "
            "as YYYYMMDD_HHMMSS."
        ),
    )
    return timestamped


def check_monitor_counts(root: Path) -> None:
    """A timestamped counts file pairs counts with monitor ids."""
    problems: list[str] = []
    for path in _timestamped_counts_files(root):
        try:
            text = path.read_text(encoding="utf-8").lstrip("﻿")
        except UnicodeDecodeError:
            problems.append(f"{path.name} must be UTF-8 text.")
            continue
        if read_count_pairs(text):
            return
        problems.append(f"{path.name} holds no `count monitor` pairs.")
    _report(problems)


def check_summary_format(root: Path) -> None:
    """`output/vitals_summary.txt` carries one readable `key: value` line per answer."""
    summary = read_summary(root)
    missing = [key for key in ANSWER_KEYS if key not in summary]
    problems: list[str] = []
    if missing:
        problems.append(f"{SUMMARY_FILE} has no line for: " + ", ".join(f"`{key}`" for key in missing) + ".")
    empty = [key for key in ANSWER_KEYS if key in summary and not summary[key]]
    if empty:
        problems.append("These keys have no value: " + ", ".join(f"`{key}`" for key in empty) + ".")
    unreadable = [
        key
        for key in ANSWER_KEYS
        if key not in LABEL_KEYS and summary.get(key) and _as_number(summary[key]) is None
    ]
    if unreadable:
        problems.append("These keys need a number: " + ", ".join(f"`{key}`" for key in unreadable) + ".")
    _report(problems)


def _check_answer_shape(root: Path, key: str) -> None:
    """Confirm one value is present and plausible, never that it is correct."""
    summary = read_summary(root)
    _assert(key in summary, f"{SUMMARY_FILE} has no `{key}` line.")
    raw = summary[key]
    shape = ANSWER_SHAPES[key]

    if shape.label:
        label = _as_label(raw)
        _assert(label != "", f"`{key}` has no value; it has to be {shape.band}.")
        _assert(len(label.split()) == 1, f"`{key}` is `{raw}`; it has to be {shape.band}.")
        return

    given = _as_number(raw)
    _assert(given is not None, f"`{key}` is `{raw}`, which is not a number; it has to be {shape.band}.")
    if shape.whole and not float(given).is_integer():
        raise AssertionError(f"`{key}` is {raw}; it has to be {shape.band}.")
    if (shape.low is not None and given < shape.low) or (shape.high is not None and given > shape.high):
        raise AssertionError(f"`{key}` is {raw}; it has to be {shape.band}.")


def _answer_check(key: str) -> PublicCheck:
    return PublicCheck(f"answer: {key}", lambda root, key=key: _check_answer_shape(root, key))


PUBLIC_CHECKS = (
    PublicCheck("environment probe", check_environment_probe),
    PublicCheck("record count artifact", check_record_count),
    PublicCheck("monitor counts artifact", check_monitor_counts),
    PublicCheck("summary artifact format", check_summary_format),
) + tuple(_answer_check(key) for key in ANSWER_KEYS)


def run_public_checks(root: Path) -> list[tuple[str, str | None]]:
    """Return one (check name, problem or None) pair per check."""
    results = []
    for check in PUBLIC_CHECKS:
        try:
            check.action(Path(root))
        except (AssertionError, OSError, ValueError, UnicodeDecodeError) as error:
            results.append((check.name, str(error)))
        else:
            results.append((check.name, None))
    return results
