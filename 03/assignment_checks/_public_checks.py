"""Checks for Assignment 03.

The course keeps these checks in 03/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They recompute every answer
from the copy of the supplied ``data/bp_readings.csv`` kept at the end of this
module and compare it with the committed artifacts, so there is no answer key
to copy and any submission that answers the questions in README.md scores,
whatever code produced it. Numeric answers are compared with a tolerance, so
rounding and spacing never decide a score.

The checks read only the files the student writes, in ``output/``. Supplied
files, the data among them, are never read, so changing or deleting one never
changes a score.

Assignment 03 is out with students, so every reading rule here only adds ways
to pass: where a more lenient rule was added, the rule the released checks
applied is tried first and still passes whatever it passed.

Nothing here imports, runs, or inspects student source code.
"""

from __future__ import annotations

import codecs
import difflib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


DATA_FILE = "data/bp_readings.csv"
SUMMARY_FILE = "output/vitals_summary.txt"
RECORD_COUNT_FILE = "output/record_count.txt"
ENVIRONMENT_FILE = "output/environment.txt"
MONITOR_COUNTS_PATTERN = re.compile(r"^monitor_counts_\d{8}_\d{6}\.txt$")
MONITOR_COUNTS_NAME = "output/monitor_counts_<timestamp>.txt"

STAGE_2_MMHG = 140
MMHG_TOLERANCE = 0.6
# A saved value is shown in feedback up to this many characters.
SHOWN_LENGTH = 80
# How closely a key must resemble a missing one to be shown as the likely attempt at it.
SIMILAR_KEY = 0.6

# The step that saves each artifact, as a sentence that a missing-file message ends with.
SAVE_STEPS = {
    ENVIRONMENT_FILE: "Save its three labelled lines with the Task 1.2 commands, then commit it.",
    RECORD_COUNT_FILE: "Save the patient count there with the Task 2.1 pipeline, then commit it.",
    SUMMARY_FILE: "Run analysis.py so it writes one `key: value` line per answer there (Task 3), then commit it.",
}
SUMMARY_FIX = f"Fix that answer in analysis.py (Task 3), rerun it, and commit {SUMMARY_FILE}."

# What `check_assignment.py` prints before and after the report.
SCOPE_NOTE = (
    "These checks read only your files in output/ and compare them with the answers the supplied "
    "data/bp_readings.csv gives."
)
SCORE_LABEL = "Score"
COMPLETE_NOTE = "All checks passed."

_NUMPY_SCALAR = re.compile(r"(?:np|numpy)\.[A-Za-z_]+\d*\(\s*([^()]*?)\s*\)")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
# The same number read through Python's `_` digit grouping, which `f"{n:_}"` prints: 3_600.
_GROUPED_NUMBER = re.compile(r"(?<![\w.])[-+]?\d(?:[\d,]|_(?=\d))*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_VERSION = re.compile(r"\d+(?:\.\d+)+")
_COUNT_SPLIT = re.compile(r"[\s,;:=|]+")
# A whole number standing on its own: not the digits of an id such as M01, nor either side of a decimal point.
_WHOLE_NUMBER = re.compile(r"(?<![\w.])\d+(?!\w|\.\d)")
# A Markdown list marker at the start of a line: `- `, `* `, `+ `, `1. ` or `1) `.
_LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
# Characters that decorate a key in Markdown or JSON: bullets, bold, italics, code spans, and quotes.
_KEY_DECORATION = "`*_-+'\""
# Characters that decorate a label answer: code spans, quotes, bold, and italics.
_LABEL_DECORATION = "`'\"*_"
# A line with no usable colon, split at its first `=`, `==`, comma, tab, or space: `numpy==2.3.3`, `patients 300`.
_UNCOLONED = re.compile(r"^([^\s=,]+)\s*(?:==?|,)?\s*(.*)$")
# A word that turns an id into a non-answer, as in `not M04`.
_NEGATION = re.compile(r"(?<!\w)(?:not|no|never|except|excluding|without)(?!\w)")

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
    # The file the check reads and the check's name within it, for the report's `Left to fix` line.
    artifact: str = ""
    label: str = ""


@dataclass(frozen=True)
class Dataset:
    patients: tuple[str, ...]
    monitors: tuple[str, ...]
    hour_columns: tuple[str, ...]
    readings: tuple[tuple[int, ...], ...]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _shown(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SHOWN_LENGTH else text[: SHOWN_LENGTH - 3] + "..."


def _file_state(path: Path) -> str:
    """What stands where a file belongs, in plain words."""
    if path.is_symlink():
        return "is a link, not a file"
    if path.is_dir():
        return "is a folder, not a file"
    return "is not a regular file" if path.exists() else "is missing"


def _decode(raw: bytes) -> str:
    """An artifact's text, without a byte-order mark.

    UTF-8 is expected. A UTF-16 file, which Windows PowerShell 5.1's `>` writes,
    is read through its byte-order mark, and a byte that is not UTF-8, such as a
    `±` saved in Windows' cp1252, reads as U+FFFD: every key and value graded is
    plain ASCII, so a stray character in a note costs nothing.
    """
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace").lstrip("\ufeff")


def _artifact(root: Path, name: str) -> Path | None:
    """The artifact's path, matching its file name in any letter case, or None.

    macOS and Windows disks ignore letter case, so a local run there finds
    `Vitals_Summary.txt` where GitHub's Linux runner would not; matching the
    name in any case makes the two runs agree.
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


def _missing(root: Path, name: str) -> str:
    """Say an artifact is missing, and name any file that looks like a misnamed or misplaced copy."""
    wanted = root / name
    if wanted.exists() or wanted.is_symlink():
        return f"{name} {_file_state(wanted)}. Delete it, then {SAVE_STEPS[name][0].lower()}{SAVE_STEPS[name][1:]}"
    look_alikes = sorted(
        path.relative_to(root).as_posix()
        for folder in {root, wanted.parent}
        if folder.is_dir()
        for path in folder.iterdir()
        if path.is_file()
        and path != wanted
        and difflib.SequenceMatcher(None, wanted.name.casefold(), path.name.casefold()).ratio() >= 0.8
    )
    if look_alikes:
        return f"{name} is missing. Found {', '.join(look_alikes)}; rename or move it to {name}, then commit it."
    return f"{name} is missing. {SAVE_STEPS[name]}"


def _read(path: Path, name: str) -> str:
    try:
        return _decode(path.read_bytes())
    except OSError as error:
        raise AssertionError(
            f"{name} cannot be opened ({error.strerror or 'unreadable'}); save it again, then commit it."
        ) from None


def _text(root: Path, name: str) -> str:
    path = _artifact(root, name)
    if path is None:
        raise AssertionError(_missing(root, name))
    return _read(path, name)


def load_dataset() -> Dataset:
    """The supplied dataset, read from the copy at the end of this module, never from a submission."""
    lines = [line.rstrip() for line in SUPPLIED_READINGS.splitlines() if line.strip()]
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


def _as_numbers(raw: str) -> tuple[float, ...]:
    """Each reading of the first number in raw, the released checks' reading first.

    `3_600` reads as 3, as the released checks read it, and as 3600, which is
    what Python's `_` digit grouping means; an answer passes when either reading
    is right.
    """
    text = _unwrap(raw)
    readings: list[float] = []
    for pattern in (_NUMBER, _GROUPED_NUMBER):
        match = pattern.search(text)
        if match is None:
            continue
        try:
            value = float(match.group().replace(",", "").replace("_", ""))
        except ValueError:
            continue
        if value not in readings:
            readings.append(value)
    return tuple(readings)


def _as_number(raw: str) -> float | None:
    readings = _as_numbers(raw)
    return readings[0] if readings else None


def _bare(text: str, decoration: str) -> str:
    """text with whitespace and decoration characters removed from both ends."""
    previous = None
    while text != previous:
        previous = text
        text = text.strip().strip(decoration)
    return text


def _as_label(raw: str) -> str:
    return _bare(_unwrap(raw), _LABEL_DECORATION).casefold()


def _label_matches(raw: str, expected: str) -> bool:
    """Whether a label answer names the expected id.

    Beyond the bare id, this accepts the id inside brackets, quotes, or Markdown
    emphasis, as a one-item list prints (`['M02']`), and the id with a note
    after it, as in `M02 (128.4 mmHg)`, or before it, as in `monitor M02`, as
    long as no other id of the same kind appears and no word such as `not`
    comes before it. An id of the same kind is the expected id's letters
    followed by any digits, so a hedge such as `M02, M05` names two monitors
    and is not accepted.
    """
    target = expected.casefold()
    if _as_label(raw) == target:
        return True
    text = _unwrap(raw).casefold()
    same_kind = re.compile(r"(?<!\w)" + re.sub(r"\d+", r"\\d+", re.escape(target)) + r"(?!\w)")
    if set(same_kind.findall(text)) != {target}:
        return False
    if text.lstrip("[({`'\"*_ \t").startswith(target):
        return True
    return _NEGATION.search(text[: same_kind.search(text).start()]) is None


def _key_name(raw: str) -> str:
    """A key without Markdown or JSON decoration: `- **Mean SBP**`, `"mean_sbp"`, and `1. mean-sbp` all read mean_sbp."""
    key = _bare(_LIST_MARKER.sub("", raw.strip()), _KEY_DECORATION)
    return re.sub(r"[\s_-]+", "_", key.casefold())


def _released_pair(line: str) -> tuple[str, str]:
    """The key and value the released checks read from a stripped line with a colon."""
    key, _, value = line.partition(":")
    return key.strip().strip("`*-").strip().casefold().replace(" ", "_").replace("-", "_"), value.strip()


def _lenient_pair(line: str, known: Callable[[str], str | None]) -> tuple[str | None, str]:
    """The known key a stripped line names and its value, however the key is decorated or separated."""
    body = _LIST_MARKER.sub("", line)
    key, colon, value = body.partition(":")
    if colon and known(_key_name(key)):
        return known(_key_name(key)), value.strip()
    uncoloned = _UNCOLONED.match(body)
    if uncoloned and known(_key_name(uncoloned.group(1))):
        return known(_key_name(uncoloned.group(1))), uncoloned.group(2).strip()
    return None, ""


def _read_pairs(text: str, known: Callable[[str], str | None]) -> tuple[dict[str, str], dict[str, int]]:
    """The value graded for each known key, and how many lines name each key.

    The first line the released checks read for a key always supplies its
    value, so no submission reads worse than it did. A key they could not read
    comes from the first line that names it once Markdown or JSON decoration
    is removed, or, on a line without a usable colon, from `key = value`,
    `key,value`, or `key value`.
    """
    released: dict[str, str] = {}
    lenient: dict[str, str] = {}
    lines_naming: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keys = set()
        if ":" in line:
            key, value = _released_pair(line)
            if key and known(key) == key:
                released.setdefault(key, value)
                keys.add(key)
        key, value = _lenient_pair(line, known)
        if key is not None:
            lenient.setdefault(key, value)
            keys.add(key)
        for key in keys:
            lines_naming[key] = lines_naming.get(key, 0) + 1
    return {**lenient, **released}, lines_naming


def _answer_key(key: str) -> str | None:
    return key if key in ANSWER_KEYS else None


def _environment_key(key: str) -> str | None:
    """`numpy version` reads as numpy, and `interpreter path` or `sys.executable` as interpreter."""
    if key.startswith("numpy"):
        return "numpy"
    if key.startswith("interpreter") or "executable" in key:
        return "interpreter"
    return None


def read_summary(root: Path) -> dict[str, str]:
    """Return the answer each `key: value` line in the summary artifact gives, however it is spaced."""
    return _read_pairs(_text(root, SUMMARY_FILE), _answer_key)[0]


def read_environment(root: Path) -> dict[str, str]:
    return _read_pairs(_text(root, ENVIRONMENT_FILE), _environment_key)[0]


def _whole(text: str) -> int | None:
    """text as an int, or None for a digit int() cannot read, such as `²`, or a number too long to convert."""
    try:
        return int(text)
    except ValueError:
        return None


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
        if label and _whole(count) is not None:
            pairs[label.casefold()] = int(count)
    return pairs


def read_monitor_lines(text: str, monitors) -> dict[str, list[int]]:
    """{monitor: counts} from every line that names exactly one of the monitors.

    This reads the pairs that read_count_pairs misses: a bullet or a note around
    the pair, as in `- M01: 58`, `M01 (58)`, `M01 -> 58`, or `58 patients on
    M01`. A count is kept only when the line holds exactly one whole number, so
    a monitor named without a readable count maps to an empty list.
    """
    naming = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(monitor) for monitor in monitors) + r")(?!\w)",
                        re.IGNORECASE)
    found: dict[str, list[int]] = {}
    for line in text.splitlines():
        line = _LIST_MARKER.sub("", line.strip())
        named = {monitor.casefold() for monitor in naming.findall(line)}
        if len(named) != 1:
            continue
        counts = found.setdefault(named.pop(), [])
        numbers = _WHOLE_NUMBER.findall(line)
        if len(numbers) == 1 and _whole(numbers[0]) is not None:
            counts.append(int(numbers[0]))
    return found


def _report(problems: list[str]) -> None:
    _assert(not problems, " ".join(problems))


def check_environment_probe(root: Path) -> None:
    """`output/environment.txt` records a numpy version and an interpreter path.

    Only that the probe was taken is graded. Which numpy version, which
    interpreter, and the `python` line never cost points, so a probe saved
    outside the project environment still passes.
    """
    probe = read_environment(root)
    problems: list[str] = []
    if "numpy" not in probe:
        problems.append(f"{ENVIRONMENT_FILE} has no `numpy` line; Task 1.2 saves `numpy: <version>` there.")
    elif _VERSION.search(probe["numpy"]) is None:
        problems.append(
            f"The `numpy` line in {ENVIRONMENT_FILE} reads `{_shown(probe['numpy'])}`, which holds no version "
            "number such as `2.3.3`; save the version numpy reports."
        )
    if "interpreter" not in probe:
        problems.append(
            f"{ENVIRONMENT_FILE} has no `interpreter` line; Task 1.2 saves `interpreter: <path>` there."
        )
    elif not probe["interpreter"]:
        problems.append(
            f"The `interpreter` line in {ENVIRONMENT_FILE} is empty; save the path to the active interpreter."
        )
    if problems:
        problems.append(
            f"Rerun the Task 1.2 commands with the environment active, using Lecture 03's one-line Python "
            f"commands for the numpy version and the interpreter path, then commit {ENVIRONMENT_FILE}."
        )
    _report(problems)


def check_record_count(root: Path) -> None:
    """`output/record_count.txt` counts the patient records in the dataset."""
    data = load_dataset()
    text = _text(root, RECORD_COUNT_FILE)
    readings = _as_numbers(text)
    _assert(
        bool(readings),
        f"{RECORD_COUNT_FILE} "
        + (f"reads `{_shown(text)}`, which holds no number" if text.strip() else "is empty")
        + f"; Task 2.1 saves the number of patient records in {DATA_FILE} there. Save the count your pipeline "
        "prints, then commit the file.",
    )
    if any(abs(value - len(data.patients)) < 1e-9 for value in readings):
        return
    given = readings[0]
    if abs(given - (len(data.patients) + 1)) < 1e-9:
        raise AssertionError(
            f"{RECORD_COUNT_FILE} records {given:g}, which counts the header line as a patient record; "
            f"the supplied {DATA_FILE} has {len(data.patients)} patient rows. Drop the header with `tail -n +2` "
            "before counting "
            "(Task 2.1), then save the new count and commit it."
        )
    raise AssertionError(
        f"{RECORD_COUNT_FILE} records {given:g}, but the supplied {DATA_FILE} has {len(data.patients)} patient rows. "
        "Count the "
        "lines after the header with the Task 2.1 pipeline, then save the new count and commit it."
    )


def _timestamped_counts_files(root: Path) -> list[Path]:
    output = root / "output"
    _assert(
        output.is_dir() and not output.is_symlink(),
        "output/ " + ("is missing" if not (output.exists() or output.is_symlink()) else "is not a folder")
        + ", so no Task 2.2 counts file is there. Make it a folder, save the per-monitor counts in it as "
        "monitor_counts_YYYYMMDD_HHMMSS.txt, and commit them.",
    )
    candidates = sorted(path for path in output.glob("monitor_counts*") if path.is_file())
    timestamped = [path for path in candidates if MONITOR_COUNTS_PATTERN.match(path.name)]
    if timestamped:
        return timestamped
    if not candidates:
        hint = (
            ". Task 2.2 saves the per-monitor counts under a name that carries the run timestamp, as "
            "monitor_counts_YYYYMMDD_HHMMSS.txt."
        )
    elif any(path.suffix != ".txt" for path in candidates):
        hint = (
            f"; it holds {', '.join(path.name for path in candidates)}. Name the file "
            "monitor_counts_YYYYMMDD_HHMMSS.txt, with the run timestamp and the `.txt` extension (Task 2.2)."
        )
    else:
        hint = (
            f"; it holds {', '.join(path.name for path in candidates)}. Name the file with the run timestamp, "
            "as monitor_counts_YYYYMMDD_HHMMSS.txt (Task 2.2)."
        )
    raise AssertionError(f"output/ has no {MONITOR_COUNTS_NAME.removeprefix('output/')} file{hint} Then commit it.")


def _counts_problem(name: str, text: str, expected: dict[str, int]) -> str:
    """What keeps one counts file from listing every monitor with its count."""
    name = f"output/{name}"
    found = read_monitor_lines(text, expected)
    if not found:
        if not read_count_pairs(text):
            return (
                f"{name} holds no `count monitor` pairs; Task 2.2 saves one line per monitor with its patient "
                "count and its id, as `uniq -c` prints them."
            )
        return (
            f"{name} names none of the monitors {', '.join(monitor.upper() for monitor in sorted(expected))}; "
            "count field 2 of each row, the monitor, with `cut -d',' -f2` (Task 2.2)."
        )
    counted = {monitor for monitor, counts in found.items() if counts and monitor in expected}
    missing = sorted(set(expected) - counted)
    # A monitor on several lines usually means `uniq -c` ran on unsorted input.
    repeated = sorted(
        monitor for monitor in counted if len(found[monitor]) > 1 and set(found[monitor]) != {expected[monitor]}
    )
    wrong = sorted(
        monitor for monitor in counted if monitor not in repeated and found[monitor][0] != expected[monitor]
    )
    details = []
    if missing:
        details.append(
            f"counts for {len(counted)} of the {len(expected)} monitors, with none for "
            + ", ".join(monitor.upper() for monitor in missing)
        )
    if repeated:
        details.append("several lines for " + ", ".join(monitor.upper() for monitor in repeated))
    if wrong:
        details.append(
            "the wrong count for "
            + ", ".join(f"{monitor.upper()} ({found[monitor][0]}, where the supplied {DATA_FILE} has "
                        f"{expected[monitor]})"
                        for monitor in wrong)
        )
    problem = f"{name} has " + "; ".join(details) + "."
    uncounted = [monitor for monitor in missing if monitor in found]
    if uncounted:
        problem += (
            " A line names " + ", ".join(monitor.upper() for monitor in uncounted)
            + " without exactly one whole-number count on it; write the count and the id, as `uniq -c` prints them."
        )
    elif missing and counted:
        # Demo 1 ends its pipeline with `head -n 5`, which drops a sixth monitor.
        problem += " Keep every line `uniq -c` prints, with no `head` stage after it."
    if repeated:
        problem += " `uniq -c` merges only lines that sit next to each other, so `sort` before it."
    return problem


def check_monitor_counts(root: Path) -> None:
    """A timestamped counts file lists how many patients each monitor recorded."""
    timestamped = _timestamped_counts_files(root)
    expected = {monitor.casefold(): count for monitor, count in monitor_counts(load_dataset()).items()}
    problems: list[str] = []
    for path in timestamped:
        text = _read(path, f"output/{path.name}")
        # Every monitor has to be there with the right count. A label that is not a
        # monitor is an extra line, which the README says is ignored: a run total or
        # a title must not cost the student this check.
        pairs = read_count_pairs(text)
        if all(pairs.get(monitor) == count for monitor, count in expected.items()):
            return
        found = read_monitor_lines(text, expected)
        if all(set(found.get(monitor, [])) == {count} for monitor, count in expected.items()):
            return
        problems.append(_counts_problem(path.name, text, expected))

    if problems:
        problems.append("Rerun the Task 2.2 pipeline to save a new timestamped file, then commit it.")
    _report(problems)


def _readable(key: str, value: str) -> bool:
    """Whether an answer check can read this value: any text for a label, a number otherwise."""
    return bool(value) and (key in LABEL_KEYS or _as_number(value) is not None)


def _format_problem(text: str, summary: dict[str, str]) -> str | None:
    """Why the summary has no readable answer line at all, or None when it has one."""
    if any(_readable(key, summary.get(key, "")) for key in ANSWER_KEYS):
        return None
    unknown = next(
        (
            key.strip()
            for key, colon, _ in (line.strip().partition(":") for line in text.splitlines())
            if colon and key.strip() and not key.startswith("#") and _answer_key(_key_name(key)) is None
        ),
        None,
    )
    found = ": it is empty" if not text.strip() else f"; its first key, `{_shown(unknown)}`, is not one of them" if unknown else ""
    return (
        f"{SUMMARY_FILE} has no readable line for any key in README.md's table{found}. Task 3 writes one "
        "`key: value` line per answer, such as `patients: <whole number>`. Fix analysis.py, rerun it, and commit "
        f"{SUMMARY_FILE}."
    )


def check_summary_format(root: Path) -> None:
    """`output/vitals_summary.txt` is readable text with at least one readable answer line.

    Each `answer: <key>` check reports its own missing or unreadable line, so
    this check does not take points a second time for the same gap.
    """
    text = _text(root, SUMMARY_FILE)
    problem = _format_problem(text, _read_pairs(text, _answer_key)[0])
    if problem is not None:
        raise AssertionError(problem)


def _mmhg_matches(given: float, expected: float) -> bool:
    """Within the tolerance, or the value with its decimals dropped, as `int()` or `astype(int)` writes it."""
    return abs(given - expected) <= MMHG_TOLERANCE or (given.is_integer() and given == math.trunc(expected))


def _asked(key: str) -> str:
    """What an answer holds, as the README's table asks for it."""
    return (LABEL_KEYS | COUNT_KEYS | MMHG_KEYS)[key]


def _closest_key(text: str, key: str) -> str:
    """`; the closest line reads ...` naming the line whose key most resembles `key`, or nothing."""
    best, best_ratio = None, SIMILAR_KEY
    for line in text.splitlines():
        head = _LIST_MARKER.sub("", line.strip()).partition(":")[0]
        candidate = _key_name(head)
        if not candidate or _answer_key(candidate) is not None:
            continue
        ratio = difflib.SequenceMatcher(None, key, candidate).ratio()
        if ratio >= best_ratio:
            best, best_ratio = line, ratio
    return f"; the closest line reads `{_shown(best)}`" if best is not None else ""


def _check_answer(root: Path, key: str) -> None:
    """Compare one answer with the value recomputed from the supplied readings."""
    _assert(_artifact(root, SUMMARY_FILE) is not None, _missing(root, SUMMARY_FILE))
    text = _text(root, SUMMARY_FILE)
    summary, lines_naming = _read_pairs(text, _answer_key)
    if key not in summary:
        # With no readable answer at all, every missing key has the format check's advice.
        raise AssertionError(
            _format_problem(text, summary)
            or f"{SUMMARY_FILE} has no `{key}` line{_closest_key(text, key)}. Add `{key}: <value>` with "
            f"{_asked(key)}. {SUMMARY_FIX}"
        )
    raw = summary[key]
    _assert(
        raw,
        f"The `{key}` line in {SUMMARY_FILE} has no value after the colon; write {_asked(key)} there. {SUMMARY_FIX}",
    )
    expected = expected_answers(load_dataset())[key]
    repeated = (
        f" `{key}` appears on {lines_naming[key]} lines and the checks read the first; "
        'write the file with "w" so each run replaces it.'
        if lines_naming.get(key, 0) > 1
        else ""
    )

    given = f"{SUMMARY_FILE} gives `{key}: {_shown(raw)}`, which is not"
    if key in LABEL_KEYS:
        _assert(
            _label_matches(raw, str(expected)),
            f"{given} {LABEL_KEYS[key]}: the supplied {DATA_FILE} gives `{expected}`.{repeated} {SUMMARY_FIX}",
        )
        return

    readings = _as_numbers(raw)
    _assert(bool(readings), f"{given} a number; write {_asked(key)} there as a number.{repeated} {SUMMARY_FIX}")
    if key in COUNT_KEYS:
        _assert(
            any(abs(given_value - float(expected)) <= 1e-9 for given_value in readings),
            f"{given} {COUNT_KEYS[key]}: the supplied {DATA_FILE} gives {expected}.{repeated} {SUMMARY_FIX}",
        )
    else:
        _assert(
            any(_mmhg_matches(given_value, float(expected)) for given_value in readings),
            f"{given} {MMHG_KEYS[key]}: the supplied {DATA_FILE} gives {float(expected):.2f} mmHg "
            f"(allowed difference {MMHG_TOLERANCE} mmHg).{repeated} {SUMMARY_FIX}",
        )


def _answer_check(key: str) -> PublicCheck:
    return PublicCheck(f"answer: {key}", lambda root, key=key: _check_answer(root, key), SUMMARY_FILE, key)


PUBLIC_CHECKS = (
    PublicCheck("environment probe", check_environment_probe, ENVIRONMENT_FILE, "environment probe"),
    PublicCheck("record count artifact", check_record_count, RECORD_COUNT_FILE, "record count"),
    PublicCheck("monitor counts artifact", check_monitor_counts, MONITOR_COUNTS_NAME, "monitor counts"),
    PublicCheck("summary artifact format", check_summary_format, SUMMARY_FILE, "format"),
) + tuple(_answer_check(key) for key in ANSWER_KEYS)


def run_public_checks(root: Path) -> list[tuple[str, str | None]]:
    """Return one (check name, problem or None) pair per check."""
    results = []
    for check in PUBLIC_CHECKS:
        try:
            check.action(Path(root))
        except AssertionError as error:
            results.append((check.name, str(error)))
        except OSError as error:
            results.append((check.name, _unreadable(Path(root), error)))
        except (ValueError, UnicodeDecodeError):
            results.append((check.name, f"{check.artifact} could not be read; save it again as plain text, then commit it."))
        else:
            results.append((check.name, None))
    return results


def _unreadable(root: Path, error: OSError) -> str:
    """An error opening a file, in plain words."""
    name = error.filename
    try:
        name = Path(name).relative_to(root).as_posix()
    except (TypeError, ValueError):
        pass
    return f"{name or 'A file'} cannot be opened ({error.strerror or 'unreadable'}); save it again, then commit it."


# The supplied data/bp_readings.csv, byte for byte; the self-test confirms it matches the handout's.
# Every expected value comes from this copy, never from the file in a submission.
SUPPLIED_READINGS = """\
patient_id,monitor,sbp_h01,sbp_h02,sbp_h03,sbp_h04,sbp_h05,sbp_h06,sbp_h07,sbp_h08,sbp_h09,sbp_h10,sbp_h11,sbp_h12
P0001,M06,111,122,128,124,125,119,119,113,109,105,107,108
P0002,M05,130,128,141,134,131,132,128,126,130,130,126,122
P0003,M01,123,138,137,142,149,135,121,124,125,122,118,129
P0004,M04,130,134,134,135,129,138,125,124,129,130,125,127
P0005,M03,139,138,151,146,148,143,137,133,135,133,125,129
P0006,M01,124,128,123,133,124,127,115,127,131,117,123,117
P0007,M02,97,106,116,117,100,112,102,100,96,95,102,105
P0008,M02,124,113,112,117,121,114,108,107,103,111,100,110
P0009,M04,134,130,135,130,133,128,125,127,128,121,125,124
P0010,M01,126,135,133,130,130,130,126,117,114,113,117,118
P0011,M02,116,114,130,126,119,125,114,116,113,108,108,107
P0012,M04,120,117,121,122,123,126,114,113,112,106,114,111
P0013,M05,157,153,153,164,165,161,150,157,152,151,152,155
P0014,M02,128,135,129,135,124,132,133,127,120,112,116,118
P0015,M05,147,159,157,153,147,157,133,154,144,143,147,138
P0016,M01,154,158,169,161,166,167,156,156,156,157,157,147
P0017,M05,139,154,151,156,153,159,136,141,147,137,148,134
P0018,M01,129,130,130,131,135,128,121,134,124,131,123,113
P0019,M06,140,128,139,141,142,142,136,133,134,134,125,131
P0020,M05,138,140,136,142,146,130,127,129,132,128,132,131
P0021,M01,131,142,149,134,136,135,132,129,125,136,132,129
P0022,M06,97,99,99,104,99,103,95,96,92,89,99,100
P0023,M01,131,139,144,132,134,130,122,131,127,133,123,117
P0024,M03,131,133,132,145,137,131,134,124,127,137,136,125
P0025,M02,114,114,122,124,119,119,111,112,109,110,113,113
P0026,M05,118,125,137,125,130,125,126,131,125,111,124,121
P0027,M06,117,126,132,128,120,125,121,119,107,118,110,116
P0028,M04,119,123,122,123,120,112,110,112,116,115,107,102
P0029,M04,150,148,148,149,154,148,148,142,143,136,144,136
P0030,M06,134,122,133,141,129,130,123,121,125,119,116,122
P0031,M03,114,114,135,132,127,124,125,114,127,121,107,110
P0032,M05,155,155,151,161,153,155,149,144,147,149,141,148
P0033,M06,132,124,129,130,121,129,117,119,123,117,125,132
P0034,M05,109,124,128,130,126,125,111,116,118,114,112,112
P0035,M05,116,117,120,130,120,117,116,112,112,118,107,116
P0036,M05,94,98,97,105,100,100,101,97,96,96,96,97
P0037,M05,155,151,158,158,149,144,149,151,136,131,136,134
P0038,M05,139,143,144,145,152,146,135,144,150,130,137,139
P0039,M05,115,116,116,119,117,108,109,108,108,113,109,104
P0040,M03,135,152,141,145,134,145,130,130,139,130,140,130
P0041,M02,130,118,129,125,140,127,127,116,124,112,114,116
P0042,M02,129,141,136,126,136,131,142,125,130,122,127,129
P0043,M01,148,151,151,150,148,153,143,143,135,139,137,140
P0044,M04,143,146,148,149,144,135,138,141,142,141,135,131
P0045,M02,121,133,128,129,126,122,133,120,130,121,113,111
P0046,M01,135,133,140,137,137,124,118,129,124,133,125,127
P0047,M03,111,120,123,119,122,116,114,117,116,111,108,107
P0048,M01,128,127,133,133,130,127,133,124,109,112,110,125
P0049,M03,153,148,153,146,156,152,149,148,148,136,138,127
P0050,M03,131,125,137,135,132,122,125,124,128,128,115,125
P0051,M04,140,134,134,141,142,146,137,142,134,129,130,130
P0052,M01,127,133,135,131,137,126,116,124,121,115,116,116
P0053,M02,137,125,125,127,129,127,132,118,116,110,112,132
P0054,M05,117,110,105,113,118,95,117,103,97,106,99,104
P0055,M03,113,118,121,132,123,123,116,120,115,115,113,104
P0056,M06,130,128,132,143,135,130,129,129,134,121,135,129
P0057,M03,139,143,147,144,144,146,137,145,142,130,138,132
P0058,M04,142,159,155,155,155,142,153,142,144,156,127,139
P0059,M06,137,131,141,146,131,128,134,129,130,131,124,118
P0060,M04,131,139,142,138,134,135,129,123,130,124,125,126
P0061,M06,119,134,127,130,123,128,119,124,119,122,110,108
P0062,M02,139,153,154,155,147,149,140,140,136,141,140,139
P0063,M03,149,158,155,166,146,152,144,144,145,143,147,148
P0064,M03,124,122,142,138,135,135,123,126,116,133,125,128
P0065,M02,116,121,122,124,120,129,117,111,118,108,121,124
P0066,M05,133,144,138,140,140,137,125,125,122,127,127,125
P0067,M01,125,116,127,126,132,122,120,110,116,107,106,113
P0068,M05,131,126,129,132,128,122,123,122,125,123,116,114
P0069,M03,109,114,121,119,115,116,121,109,107,117,108,107
P0070,M05,128,117,125,131,133,130,129,123,124,118,124,120
P0071,M06,109,108,119,115,114,105,104,97,99,104,106,101
P0072,M03,141,135,135,146,130,138,135,124,130,129,123,125
P0073,M03,115,109,127,113,121,118,117,118,110,111,105,105
P0074,M01,125,137,132,142,127,131,127,130,128,127,114,131
P0075,M03,130,126,130,117,126,119,123,122,129,110,121,114
P0076,M06,153,147,142,152,150,150,142,140,140,143,134,132
P0077,M01,123,125,121,126,118,112,119,114,119,110,116,112
P0078,M05,150,156,160,151,158,155,151,147,144,141,142,141
P0079,M01,135,143,147,137,141,146,132,136,130,131,146,124
P0080,M03,114,111,120,120,116,119,116,108,110,107,110,119
P0081,M03,124,129,124,128,127,120,116,122,113,119,114,111
P0082,M03,143,141,147,139,140,150,141,140,130,132,129,136
P0083,M03,121,135,136,142,133,136,138,133,123,132,127,125
P0084,M06,118,122,128,129,120,123,120,118,114,114,122,108
P0085,M01,133,145,143,142,146,134,131,131,129,130,125,125
P0086,M06,125,129,132,142,125,129,120,118,117,124,120,119
P0087,M01,146,153,149,158,165,148,153,142,148,150,142,142
P0088,M06,132,135,141,145,139,147,144,138,133,129,128,127
P0089,M03,123,131,126,130,131,130,133,117,128,120,124,116
P0090,M02,131,120,129,128,123,124,121,124,118,117,117,128
P0091,M06,128,130,142,132,145,135,137,130,128,125,133,120
P0092,M04,134,139,140,150,146,143,137,140,138,138,125,129
P0093,M03,156,161,173,167,169,168,161,162,155,157,153,143
P0094,M06,106,105,106,104,104,96,92,90,90,100,94,96
P0095,M05,134,128,141,140,137,145,132,123,133,121,131,122
P0096,M05,154,144,138,148,149,139,142,141,138,138,140,132
P0097,M06,137,134,132,135,129,136,126,130,123,124,117,119
P0098,M03,129,135,132,134,130,129,124,118,121,129,131,115
P0099,M05,132,137,137,146,144,143,133,122,134,130,131,128
P0100,M01,127,118,131,127,126,126,127,122,121,122,113,119
P0101,M01,139,144,141,138,145,134,137,127,135,130,127,136
P0102,M03,110,111,126,115,115,114,114,110,113,105,111,113
P0103,M01,106,108,121,121,119,117,111,115,114,99,104,112
P0104,M01,141,144,146,147,149,140,138,133,137,142,135,130
P0105,M05,123,127,124,120,115,115,113,112,113,118,109,107
P0106,M04,145,148,159,168,151,143,148,144,140,132,137,135
P0107,M01,133,139,145,143,152,152,139,144,136,143,133,134
P0108,M04,129,135,139,123,133,129,127,125,125,124,124,116
P0109,M01,123,117,123,133,128,130,119,122,116,117,111,121
P0110,M05,145,164,162,163,152,158,148,159,145,150,140,152
P0111,M04,110,110,112,119,118,113,108,100,109,98,101,103
P0112,M05,140,136,130,145,140,142,126,139,131,137,120,123
P0113,M03,130,134,128,141,130,138,123,126,129,128,120,118
P0114,M03,125,126,137,145,132,127,123,127,126,117,115,121
P0115,M03,119,124,130,117,126,125,121,119,112,118,105,116
P0116,M04,158,142,154,160,156,146,139,146,145,145,138,147
P0117,M06,130,135,137,129,139,146,141,125,121,134,126,132
P0118,M04,131,145,148,150,141,138,149,136,140,143,139,133
P0119,M02,112,117,111,117,109,114,116,111,103,101,109,99
P0120,M03,137,141,134,148,141,135,135,140,135,131,140,131
P0121,M06,115,113,128,124,125,124,118,111,116,102,112,116
P0122,M06,129,131,140,143,140,136,127,139,138,126,128,129
P0123,M01,124,127,136,138,133,122,125,111,118,121,119,117
P0124,M05,151,143,151,152,156,153,152,145,142,147,146,147
P0125,M01,133,131,134,134,129,126,128,122,122,117,128,115
P0126,M02,126,132,139,129,136,128,126,124,136,122,126,128
P0127,M06,120,127,128,120,123,124,127,128,126,118,123,112
P0128,M05,144,154,147,156,153,141,150,142,143,142,142,144
P0129,M02,154,150,156,163,154,148,161,151,149,147,151,133
P0130,M06,117,115,120,125,120,118,118,113,118,114,119,105
P0131,M06,172,168,176,179,173,160,171,164,172,162,163,160
P0132,M03,120,127,124,129,130,127,126,117,118,108,114,116
P0133,M03,132,130,136,130,132,126,130,127,123,124,128,120
P0134,M01,122,131,130,137,129,128,128,121,122,129,126,113
P0135,M02,108,110,108,108,114,102,102,90,93,98,105,100
P0136,M01,145,141,163,158,154,157,143,139,143,138,145,139
P0137,M02,99,119,118,120,112,106,120,113,107,105,105,111
P0138,M05,131,138,146,142,136,133,138,123,132,134,129,124
P0139,M03,152,139,148,138,144,150,140,140,135,129,127,128
P0140,M01,115,118,129,121,116,125,115,120,105,115,110,113
P0141,M03,121,130,131,135,137,125,135,131,134,124,127,123
P0142,M02,128,135,135,137,123,128,128,121,127,119,120,125
P0143,M03,110,118,132,129,128,121,122,118,111,127,120,117
P0144,M05,136,143,152,162,139,142,141,144,133,138,141,138
P0145,M02,110,103,112,106,106,110,100,107,99,105,94,97
P0146,M04,161,151,152,163,152,150,148,150,146,142,149,147
P0147,M06,128,124,124,132,127,116,118,116,120,110,126,120
P0148,M03,129,127,139,137,133,128,126,129,126,118,121,121
P0149,M02,127,138,137,138,137,141,123,130,133,117,124,130
P0150,M04,136,143,137,153,149,142,143,140,146,137,142,133
P0151,M05,151,152,154,160,156,158,145,143,149,141,148,135
P0152,M02,143,146,148,150,151,141,135,137,137,136,130,140
P0153,M02,122,137,128,134,134,128,127,128,135,128,128,124
P0154,M02,133,139,134,136,140,133,127,128,132,124,124,130
P0155,M06,139,146,142,145,139,146,139,143,133,134,138,126
P0156,M04,153,157,165,168,166,158,161,154,161,156,153,141
P0157,M03,154,159,151,157,153,156,149,149,129,148,136,139
P0158,M02,132,130,141,135,145,123,136,129,125,124,107,126
P0159,M03,144,141,148,139,139,137,144,136,139,129,131,132
P0160,M01,114,118,119,129,122,116,116,107,108,99,102,112
P0161,M06,159,162,156,172,158,157,162,157,155,160,154,153
P0162,M05,135,133,134,143,141,128,136,138,129,132,121,129
P0163,M03,117,128,135,132,126,132,122,121,124,121,117,117
P0164,M01,137,130,142,139,138,128,133,129,136,126,127,140
P0165,M04,133,135,130,134,137,130,123,129,129,129,111,117
P0166,M05,137,145,140,157,143,141,135,137,122,134,131,137
P0167,M01,141,146,147,148,148,144,130,132,138,124,130,128
P0168,M06,95,102,110,115,107,102,102,95,98,96,96,101
P0169,M03,153,141,146,162,153,142,144,148,139,148,138,137
P0170,M06,130,133,129,132,140,135,129,129,134,122,128,121
P0171,M01,122,114,119,108,115,121,115,114,115,109,99,109
P0172,M01,150,147,147,157,155,149,146,150,139,138,138,149
P0173,M05,128,133,129,134,140,128,125,125,117,127,119,127
P0174,M06,141,139,140,146,134,131,147,128,129,130,137,125
P0175,M03,125,123,138,134,123,129,122,128,126,131,120,124
P0176,M05,136,145,147,143,140,132,141,125,137,132,131,130
P0177,M02,138,144,147,158,150,146,141,141,139,141,144,139
P0178,M04,127,128,131,126,128,120,126,123,113,127,121,112
P0179,M01,133,135,141,138,128,126,137,124,128,128,125,112
P0180,M06,126,136,128,131,136,142,131,137,125,123,123,122
P0181,M06,114,122,126,129,131,118,122,114,120,109,116,115
P0182,M03,112,117,111,115,114,106,99,100,114,109,109,103
P0183,M05,121,120,137,138,128,128,126,131,109,114,111,111
P0184,M02,145,142,146,158,146,135,138,142,149,152,135,135
P0185,M03,112,124,125,120,118,120,119,119,115,109,110,113
P0186,M06,138,155,155,152,151,139,152,141,136,136,144,133
P0187,M03,127,136,138,144,134,135,135,129,117,129,117,129
P0188,M01,116,118,117,123,111,117,119,107,106,107,106,111
P0189,M05,125,137,137,138,128,128,122,122,125,120,129,132
P0190,M01,130,130,134,137,140,137,139,130,131,128,116,127
P0191,M03,148,151,162,153,153,162,151,154,149,145,154,152
P0192,M02,122,131,129,134,136,126,131,123,119,127,111,121
P0193,M03,133,137,135,146,141,131,125,128,123,124,136,124
P0194,M01,154,159,158,157,161,152,156,156,145,151,156,146
P0195,M04,142,134,149,144,136,138,131,142,130,132,144,131
P0196,M05,135,140,139,144,139,129,133,138,137,128,118,127
P0197,M04,145,144,140,154,150,150,137,146,144,137,140,141
P0198,M01,141,139,146,146,143,130,135,129,131,136,136,131
P0199,M04,157,151,156,160,151,151,148,156,146,150,157,140
P0200,M04,140,147,151,147,151,141,141,144,139,142,133,135
P0201,M03,116,138,123,122,123,115,123,115,116,119,110,124
P0202,M01,109,119,117,117,110,110,103,109,115,110,116,102
P0203,M06,118,118,127,115,132,120,118,122,113,117,105,106
P0204,M02,142,145,155,156,140,152,136,139,136,144,137,135
P0205,M01,126,146,141,136,132,141,138,139,127,130,124,126
P0206,M04,122,123,128,126,130,119,123,116,116,115,112,123
P0207,M03,124,135,119,139,120,126,119,114,117,118,121,119
P0208,M03,130,135,138,136,138,131,143,131,132,126,119,128
P0209,M05,135,146,146,148,156,139,146,145,128,127,140,134
P0210,M04,136,146,147,150,149,147,143,135,137,136,125,144
P0211,M02,110,123,134,133,126,129,119,122,122,125,122,125
P0212,M04,166,158,162,168,164,161,153,153,151,152,154,154
P0213,M05,126,126,125,134,122,135,131,123,114,114,120,115
P0214,M05,120,117,121,127,125,121,119,114,124,117,113,117
P0215,M01,123,126,137,128,123,130,125,127,125,107,112,109
P0216,M04,135,130,132,146,136,130,126,129,120,126,122,112
P0217,M02,126,122,126,120,122,123,124,119,119,112,116,119
P0218,M01,146,140,149,146,150,138,142,141,131,139,135,145
P0219,M01,154,156,151,158,153,151,149,154,142,142,140,144
P0220,M05,138,134,141,144,140,139,140,139,126,122,127,131
P0221,M04,124,136,139,137,141,120,126,132,130,129,119,120
P0222,M05,126,133,138,134,129,125,125,128,120,129,128,121
P0223,M02,149,150,159,158,157,145,142,154,139,143,138,141
P0224,M06,130,135,132,137,139,134,121,132,132,113,128,117
P0225,M03,131,148,131,143,136,130,134,135,129,134,124,132
P0226,M06,107,109,108,112,128,109,103,103,107,107,105,95
P0227,M03,111,118,138,127,119,130,113,115,124,120,113,118
P0228,M01,114,130,130,121,124,122,115,118,118,112,117,115
P0229,M04,151,165,152,163,159,158,144,149,146,140,145,147
P0230,M04,154,155,148,151,150,156,147,143,139,143,145,136
P0231,M03,154,158,163,163,159,163,163,153,158,148,154,151
P0232,M03,106,109,110,121,114,112,117,94,98,97,103,103
P0233,M02,130,132,128,137,133,130,125,124,115,122,134,118
P0234,M01,124,127,127,134,132,114,117,119,127,116,120,123
P0235,M05,126,130,125,129,122,119,115,110,109,122,117,109
P0236,M06,139,154,158,140,148,141,137,137,134,142,130,132
P0237,M02,111,108,114,121,116,111,108,110,103,101,106,100
P0238,M03,117,121,122,137,124,119,109,113,113,103,107,121
P0239,M01,146,160,156,155,154,158,148,151,156,144,144,139
P0240,M06,145,142,147,142,139,141,143,136,138,135,128,134
P0241,M02,135,143,145,143,143,134,132,134,134,134,132,138
P0242,M05,149,155,157,155,155,160,157,149,148,146,146,154
P0243,M02,126,108,127,126,116,116,111,105,111,108,107,104
P0244,M03,155,159,162,174,163,160,154,152,153,152,151,146
P0245,M02,141,161,167,142,148,141,145,143,145,148,145,136
P0246,M03,117,128,123,133,124,122,120,125,118,112,109,112
P0247,M05,137,142,139,150,149,144,136,134,137,140,139,130
P0248,M05,152,152,163,161,154,164,154,159,150,151,144,131
P0249,M05,108,104,106,115,111,102,98,100,97,92,91,103
P0250,M03,124,118,114,125,123,100,110,108,114,110,108,116
P0251,M06,123,137,134,143,135,133,132,116,125,131,121,122
P0252,M06,145,147,142,147,150,154,135,135,140,131,132,127
P0253,M02,151,143,154,154,143,141,131,129,139,132,132,129
P0254,M02,136,138,135,126,125,131,127,128,128,130,124,116
P0255,M01,118,123,117,134,124,122,126,115,119,106,104,113
P0256,M03,134,136,143,140,133,130,127,124,130,128,125,124
P0257,M01,115,128,134,128,124,131,122,113,118,115,116,113
P0258,M05,127,127,139,132,126,128,124,123,124,124,123,116
P0259,M04,139,130,139,138,140,141,139,129,129,129,137,130
P0260,M04,141,134,136,142,130,125,140,130,124,118,129,131
P0261,M04,134,137,136,133,139,138,124,132,125,130,132,124
P0262,M02,127,130,132,135,121,118,124,119,106,118,115,117
P0263,M01,93,111,101,116,106,101,97,98,102,107,88,94
P0264,M03,141,146,145,150,141,141,144,139,141,132,141,131
P0265,M01,118,129,124,129,125,124,123,125,110,111,120,107
P0266,M02,147,144,152,148,142,142,141,137,133,143,141,139
P0267,M01,151,142,149,145,147,138,144,145,144,139,135,135
P0268,M04,144,150,156,151,148,152,151,147,141,139,148,145
P0269,M01,127,134,130,136,135,135,131,121,116,122,121,123
P0270,M06,129,129,138,131,133,114,121,120,122,124,115,121
P0271,M02,126,120,126,124,120,113,111,109,118,114,107,111
P0272,M05,131,137,143,132,138,136,146,126,134,119,132,129
P0273,M01,137,136,137,140,139,128,126,129,125,127,129,132
P0274,M02,127,136,137,132,137,136,131,126,129,117,125,128
P0275,M01,115,122,124,120,119,117,109,111,119,123,110,113
P0276,M05,132,134,136,139,129,124,132,118,117,118,122,126
P0277,M06,139,139,146,139,145,142,141,135,139,141,138,131
P0278,M04,133,136,135,128,137,126,127,120,128,125,128,114
P0279,M05,143,147,145,149,145,153,136,136,142,135,139,133
P0280,M05,109,118,133,124,123,127,120,108,118,117,117,117
P0281,M03,100,115,106,103,99,106,104,93,100,100,99,94
P0282,M06,116,125,129,117,118,113,109,114,108,106,116,104
P0283,M03,121,131,126,134,126,119,124,118,116,115,120,125
P0284,M06,142,154,145,151,151,148,151,142,139,143,140,138
P0285,M03,149,144,156,151,145,145,139,132,130,139,138,133
P0286,M01,134,139,145,138,127,137,126,126,125,127,117,134
P0287,M04,148,138,151,146,147,139,147,138,139,137,141,134
P0288,M02,161,168,165,180,158,165,158,166,162,153,163,169
P0289,M01,148,159,157,160,160,152,149,146,150,147,137,153
P0290,M03,148,155,140,148,137,145,147,138,141,137,131,141
P0291,M05,123,128,125,135,126,126,128,120,114,114,124,108
P0292,M06,120,137,136,146,143,132,124,130,136,130,134,122
P0293,M02,127,139,135,142,133,128,132,120,127,136,119,123
P0294,M06,143,150,151,154,156,145,147,144,139,147,141,140
P0295,M03,131,131,138,137,125,133,127,125,127,130,123,118
P0296,M04,122,130,122,133,133,135,118,115,130,117,117,127
P0297,M04,151,155,164,157,167,155,165,149,159,155,145,147
P0298,M01,121,120,128,128,130,111,127,120,116,117,120,113
P0299,M01,116,120,126,120,118,116,113,103,100,107,104,99
P0300,M06,125,143,137,144,147,131,125,130,124,128,132,132
"""
