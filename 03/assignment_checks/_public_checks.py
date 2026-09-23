"""Value checks for Assignment 03.

Course-side material. This file ships in the checks repository the assignment
workflow downloads at run time, never in a student fork. It recomputes every
answer from the supplied ``data/bp_readings.csv`` and compares it with the
committed artifacts, so there is no answer key to copy and any submission that
answers the questions in README.md scores, whatever code produced it. Numeric
answers are compared with a tolerance, so rounding and spacing never decide a
score.

The fork carries a shape-only twin of this module with the same check names,
order, and point values. Fix a parsing, environment, or naming rule in both.

Nothing here imports, runs, or inspects student source code.
"""

from __future__ import annotations

import codecs
import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


DATA_FILE = "data/bp_readings.csv"
DATA_SHA256 = "b2300f98336816323b86a910f1dc1cc8ce89f83e9b564e3e10b82bf8e2aadadc"
SUMMARY_FILE = "output/vitals_summary.txt"
RECORD_COUNT_FILE = "output/record_count.txt"
ENVIRONMENT_FILE = "output/environment.txt"
MONITOR_COUNTS_PATTERN = re.compile(r"^monitor_counts_\d{8}_\d{6}\.txt$")
MONITOR_COUNTS_NAME = "output/monitor_counts_<timestamp>.txt"

STAGE_2_MMHG = 140
MMHG_TOLERANCE = 0.6

# What `check_assignment.py` tells the reader this run verified.
CHECKS_SCOPE = "values"
SCOPE_NOTE = "These checks recompute every answer from data/bp_readings.csv and compare it with your artifacts."
SCORE_LABEL = "Score"
COMPLETE_NOTE = "All checks passed."

_NUMPY_SCALAR = re.compile(r"(?:np|numpy)\.[A-Za-z_]+\d*\(\s*([^()]*?)\s*\)")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_VERSION = re.compile(r"\d+(?:\.\d+)+")
_PINNED_NUMPY = re.compile(r"^numpy\s*==\s*([0-9][^\s;#]*)", re.IGNORECASE)
# A path to an interpreter inside a .venv directory, on either platform.
_VENV_INTERPRETER = re.compile(r"(?:^|[/\\])\.venv[/\\].*python", re.IGNORECASE)
_COUNT_SPLIT = re.compile(r"[\s,;:=|]+")

# key -> what the value must answer, phrased for a failure message.
COUNT_KEYS = {
    "patients": "the number of patients",
    "readings": "the number of individual readings",
    "min_sbp": "the lowest single reading",
    "max_sbp": "the highest single reading",
    "stage2_patients": f"the number of patients whose 12-hour mean is {STAGE_2_MMHG} mmHg or higher",
    "stage2_other_monitors": (
        f"the number of patients whose 12-hour mean is {STAGE_2_MMHG} mmHg or higher "
        "once the patients on the high-reading monitor are left out"
    ),
}
MMHG_KEYS = {
    "mean_sbp": "the mean of every reading",
    "sd_sbp": "the standard deviation of every reading",
    "highest_patient_mean": "the 12-hour mean of the patient with the highest 12-hour mean",
    "peak_hour_mean": "the mean of the hour column with the highest mean",
    "monitor_offset": (
        "the gap between the high-reading monitor's average and the average of the "
        "patients on the other monitors"
    ),
}
LABEL_KEYS = {
    "highest_patient": "the patient_id of the patient with the highest 12-hour mean",
    "peak_hour_column": "the header name of the hour column with the highest mean",
    "high_monitor": "the monitor whose patients' 12-hour means average highest",
}
# Report order: the order README.md lists the answers in.
ANSWER_KEYS = (
    "patients",
    "readings",
    "mean_sbp",
    "sd_sbp",
    "min_sbp",
    "max_sbp",
    "stage2_patients",
    "highest_patient",
    "highest_patient_mean",
    "peak_hour_column",
    "peak_hour_mean",
    "high_monitor",
    "monitor_offset",
    "stage2_other_monitors",
)


@dataclass(frozen=True)
class PublicCheck:
    name: str
    action: Callable[[Path], None]


@dataclass(frozen=True)
class Dataset:
    patients: tuple[str, ...]
    monitors: tuple[str, ...]
    hour_columns: tuple[str, ...]
    readings: tuple[tuple[int, ...], ...]


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


def load_dataset(root: Path) -> Dataset:
    """Read the supplied dataset, refusing a file that is not the one handed out."""
    path = root / DATA_FILE
    _assert(path.is_file() and not path.is_symlink(), f"{DATA_FILE} is missing; restore the supplied dataset.")
    raw = path.read_bytes()
    if raw.startswith(codecs.BOM_UTF8):
        raw = raw[len(codecs.BOM_UTF8) :]
    raw = raw.replace(b"\r\n", b"\n")
    _assert(
        hashlib.sha256(raw).hexdigest() == DATA_SHA256,
        f"{DATA_FILE} is not the supplied dataset. Restore it with "
        f"`git checkout {DATA_FILE}` and rerun your analysis on it.",
    )

    lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    header = lines[0].split(",")
    rows = [line.split(",") for line in lines[1:]]
    return Dataset(
        patients=tuple(row[0] for row in rows),
        monitors=tuple(row[1] for row in rows),
        hour_columns=tuple(header[2:]),
        readings=tuple(tuple(int(value) for value in row[2:]) for row in rows),
    )


def _mean(values) -> float:
    values = tuple(values)
    return sum(values) / len(values)


def expected_answers(data: Dataset) -> dict[str, float | str]:
    """Recompute every answer the summary file is asked for.

    A monitor's average is the mean of its patients' 12-hour means, which is the
    definition README.md states. Every patient has the same number of readings,
    so it equals the mean of that monitor's readings.
    """
    flat = [value for row in data.readings for value in row]
    overall_mean = _mean(flat)
    patient_means = [_mean(row) for row in data.readings]
    hour_means = [_mean(row[hour] for row in data.readings) for hour in range(len(data.hour_columns))]

    best_patient = max(range(len(patient_means)), key=patient_means.__getitem__)
    peak_hour = max(range(len(hour_means)), key=hour_means.__getitem__)

    monitor_means = {
        monitor: _mean(mean for mean, owner in zip(patient_means, data.monitors) if owner == monitor)
        for monitor in set(data.monitors)
    }
    high_monitor = max(monitor_means, key=monitor_means.__getitem__)
    other_means = [mean for mean, owner in zip(patient_means, data.monitors) if owner != high_monitor]

    return {
        "patients": len(data.patients),
        "readings": len(flat),
        "mean_sbp": overall_mean,
        "sd_sbp": math.sqrt(sum((value - overall_mean) ** 2 for value in flat) / len(flat)),
        "min_sbp": min(flat),
        "max_sbp": max(flat),
        "stage2_patients": sum(1 for mean in patient_means if mean >= STAGE_2_MMHG),
        "highest_patient": data.patients[best_patient],
        "highest_patient_mean": patient_means[best_patient],
        "peak_hour_column": data.hour_columns[peak_hour],
        "peak_hour_mean": hour_means[peak_hour],
        "high_monitor": high_monitor,
        "monitor_offset": monitor_means[high_monitor] - _mean(other_means),
        "stage2_other_monitors": sum(
            1
            for mean, owner in zip(patient_means, data.monitors)
            if owner != high_monitor and mean >= STAGE_2_MMHG
        ),
    }


def monitor_counts(data: Dataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for monitor in data.monitors:
        counts[monitor] = counts.get(monitor, 0) + 1
    return counts


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
    return _unwrap(raw).strip().strip("`'\"").strip().casefold()


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
    """`output/record_count.txt` counts the patient records in the dataset."""
    data = load_dataset(root)
    given = _as_number(_text(root, RECORD_COUNT_FILE))
    _assert(given is not None, f"{RECORD_COUNT_FILE} contains no number.")
    if abs(given - len(data.patients)) < 1e-9:
        return
    if abs(given - (len(data.patients) + 1)) < 1e-9:
        raise AssertionError(
            f"{RECORD_COUNT_FILE} records {given:g}, which counts the header line as a patient record; "
            "drop the header with `tail -n +2` before counting."
        )
    raise AssertionError(
        f"{RECORD_COUNT_FILE} records {given:g}, which is not the number of patient rows in {DATA_FILE}."
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
    """A timestamped counts file lists how many patients each monitor recorded."""
    timestamped = _timestamped_counts_files(root)
    expected = {monitor.casefold(): count for monitor, count in monitor_counts(load_dataset(root)).items()}
    problems: list[str] = []
    for path in timestamped:
        found = read_count_pairs(path.read_text(encoding="utf-8").lstrip("\ufeff"))
        # Every monitor has to be there with the right count. A label that is not a
        # monitor is an extra line, which the README says is ignored: a run total or
        # a title must not cost the student this check.
        if all(found.get(monitor) == count for monitor, count in expected.items()):
            return
        if not found:
            problems.append(f"{path.name} holds no `count monitor` pairs.")
            continue
        missing = sorted(set(expected) - set(found))
        wrong = sorted(monitor for monitor in set(found) & set(expected) if found[monitor] != expected[monitor])
        details = []
        if missing:
            details.append("no count for " + ", ".join(monitor.upper() for monitor in missing))
        if wrong:
            details.append("the wrong count for " + ", ".join(monitor.upper() for monitor in wrong))
        problems.append(f"{path.name} has " + "; ".join(details) + ".")

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


def _check_answer(root: Path, key: str) -> None:
    """Compare one answer with the value recomputed from the supplied readings."""
    summary = read_summary(root)
    _assert(key in summary, f"{SUMMARY_FILE} has no `{key}` line.")
    raw = summary[key]
    expected = expected_answers(load_dataset(root))[key]

    if key in LABEL_KEYS:
        _assert(
            _as_label(raw) == str(expected).casefold(),
            f"`{key}` is `{raw}`, which is not {LABEL_KEYS[key]} in {DATA_FILE}.",
        )
        return

    given = _as_number(raw)
    _assert(given is not None, f"`{key}` is `{raw}`, which is not a number.")
    if key in COUNT_KEYS:
        _assert(
            abs(given - float(expected)) <= 1e-9,
            f"`{key}` is {raw}, which is not {COUNT_KEYS[key]} in {DATA_FILE}.",
        )
    else:
        _assert(
            abs(given - float(expected)) <= MMHG_TOLERANCE,
            f"`{key}` is {raw}, which is not {MMHG_KEYS[key]} in {DATA_FILE} "
            f"(allowed difference {MMHG_TOLERANCE} mmHg).",
        )


def _answer_check(key: str) -> PublicCheck:
    return PublicCheck(f"answer: {key}", lambda root, key=key: _check_answer(root, key))


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
