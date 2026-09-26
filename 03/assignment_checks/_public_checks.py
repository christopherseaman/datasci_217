"""Checks for Assignment 03.

The course keeps these checks in 03/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They recompute every answer
from the supplied ``data/bp_readings.csv`` and compare it with the committed
artifacts, so there is no answer key to copy and any submission that answers
the questions in README.md scores, whatever code produced it. Numeric answers
are compared with a tolerance, so rounding and spacing never decide a score.

Assignment 03 is out with students, so every reading rule here only adds ways
to pass: where a more lenient rule was added, the rule the released checks
applied is tried first and still passes whatever it passed.

Nothing here imports, runs, or inspects student source code.
"""

from __future__ import annotations

import codecs
import difflib
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
SCOPE_NOTE = "These checks recompute every answer from data/bp_readings.csv and compare it with your artifacts."
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


def load_dataset(root: Path) -> Dataset:
    """Read the supplied dataset, refusing a file that is not the one handed out.

    A BOM, line endings, trailing spaces, and blank lines change no value, so
    the fingerprint is taken with them removed; the supplied file has none.
    """
    path = root / DATA_FILE
    restore = (
        f"Restore it with `git checkout {DATA_FILE}`, rerun your Task 2 pipelines and analysis.py on it, "
        "and commit the new outputs."
    )
    _assert(
        path.is_file() and not path.is_symlink(),
        f"{DATA_FILE} {_file_state(path)}, and every answer is recomputed from it. {restore}",
    )
    lines = [line.rstrip() for line in _read(path, DATA_FILE).splitlines() if line.strip()]
    _assert(
        hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest() == DATA_SHA256,
        f"{DATA_FILE} is not the supplied dataset, and every answer is recomputed from it. {restore}",
    )

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
    data = load_dataset(root)
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
            f"{DATA_FILE} has {len(data.patients)} patient rows. Drop the header with `tail -n +2` before counting "
            "(Task 2.1), then save the new count and commit it."
        )
    raise AssertionError(
        f"{RECORD_COUNT_FILE} records {given:g}, but {DATA_FILE} has {len(data.patients)} patient rows. Count the "
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
            + ", ".join(f"{monitor.upper()} ({found[monitor][0]}, where {DATA_FILE} has {expected[monitor]})"
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
    expected = {monitor.casefold(): count for monitor, count in monitor_counts(load_dataset(root)).items()}
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
    expected = expected_answers(load_dataset(root))[key]
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
            f"{given} {LABEL_KEYS[key]}: {DATA_FILE} gives `{expected}`.{repeated} {SUMMARY_FIX}",
        )
        return

    readings = _as_numbers(raw)
    _assert(bool(readings), f"{given} a number; write {_asked(key)} there as a number.{repeated} {SUMMARY_FIX}")
    if key in COUNT_KEYS:
        _assert(
            any(abs(given_value - float(expected)) <= 1e-9 for given_value in readings),
            f"{given} {COUNT_KEYS[key]}: {DATA_FILE} gives {expected}.{repeated} {SUMMARY_FIX}",
        )
    else:
        _assert(
            any(_mmhg_matches(given_value, float(expected)) for given_value in readings),
            f"{given} {MMHG_KEYS[key]}: {DATA_FILE} gives {float(expected):.2f} mmHg "
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
