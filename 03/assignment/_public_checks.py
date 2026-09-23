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


@dataclass(frozen=True)
class Dataset:
    patients: tuple[str, ...]
    monitors: tuple[str, ...]
    hour_columns: tuple[str, ...]
    readings: tuple[tuple[int, ...], ...]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
    message = f"{name} is missing; commit it as a regular file."
    wanted = root / name
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
        message += f" Found {', '.join(look_alikes)}; rename or move it to {name}."
    return message


def _text(root: Path, name: str) -> str:
    path = _artifact(root, name)
    if path is None:
        raise AssertionError(_missing(root, name))
    return _decode(path.read_bytes())


def load_dataset(root: Path) -> Dataset:
    """Read the supplied dataset, refusing a file that is not the one handed out.

    A BOM, line endings, trailing spaces, and blank lines change no value, so
    the fingerprint is taken with them removed; the supplied file has none.
    """
    path = root / DATA_FILE
    _assert(path.is_file() and not path.is_symlink(), f"{DATA_FILE} is missing; restore the supplied dataset.")
    lines = [line.rstrip() for line in _decode(path.read_bytes()).splitlines() if line.strip()]
    _assert(
        hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest() == DATA_SHA256,
        f"{DATA_FILE} is not the supplied dataset. Restore it with "
        f"`git checkout {DATA_FILE}` and rerun your analysis on it.",
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
        problems.append(f"{ENVIRONMENT_FILE} has no `numpy` line.")
    elif _VERSION.search(probe["numpy"]) is None:
        problems.append(
            f"The `numpy` line in {ENVIRONMENT_FILE} holds no version number such as `2.3.3`; "
            "save the version numpy reports."
        )
    if "interpreter" not in probe:
        problems.append(f"{ENVIRONMENT_FILE} has no `interpreter` line.")
    elif not probe["interpreter"]:
        problems.append(
            f"The `interpreter` line in {ENVIRONMENT_FILE} is empty; save the path to the interpreter."
        )
    _report(problems)


def check_record_count(root: Path) -> None:
    """`output/record_count.txt` counts the patient records in the dataset."""
    data = load_dataset(root)
    readings = _as_numbers(_text(root, RECORD_COUNT_FILE))
    _assert(bool(readings), f"{RECORD_COUNT_FILE} contains no number.")
    if any(abs(value - len(data.patients)) < 1e-9 for value in readings):
        return
    given = readings[0]
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
    candidates = sorted(path for path in output.glob("monitor_counts*") if path.is_file())
    timestamped = [path for path in candidates if MONITOR_COUNTS_PATTERN.match(path.name)]
    if timestamped:
        return timestamped
    if not candidates:
        hint = "; save the per-monitor counts under a name that carries the run timestamp, as YYYYMMDD_HHMMSS."
    elif any(path.suffix != ".txt" for path in candidates):
        hint = (
            f" (output/ holds {', '.join(path.name for path in candidates)}); name the file "
            "monitor_counts_YYYYMMDD_HHMMSS.txt, with the run timestamp and the `.txt` extension."
        )
    else:
        hint = (
            f" (output/ holds {', '.join(path.name for path in candidates)}); name the file with the "
            "run timestamp, as YYYYMMDD_HHMMSS."
        )
    raise AssertionError(f"No {MONITOR_COUNTS_NAME} found{hint}")


def _counts_problem(name: str, text: str, expected: dict[str, int]) -> str:
    """What keeps one counts file from listing every monitor with its count."""
    found = read_monitor_lines(text, expected)
    if not found:
        if not read_count_pairs(text):
            return f"{name} holds no `count monitor` pairs."
        return (
            f"{name} names none of the monitors {', '.join(monitor.upper() for monitor in sorted(expected))}; "
            "count field 2 of each row, the monitor, with `cut -d',' -f2`."
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
        details.append("the wrong count for " + ", ".join(monitor.upper() for monitor in wrong))
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
        text = _decode(path.read_bytes())
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

    _report(problems)


def _readable(key: str, value: str) -> bool:
    """Whether an answer check can read this value: any text for a label, a number otherwise."""
    return bool(value) and (key in LABEL_KEYS or _as_number(value) is not None)


def check_summary_format(root: Path) -> None:
    """`output/vitals_summary.txt` is readable text with at least one readable answer line.

    Each `answer: <key>` check reports its own missing or unreadable line, so
    this check does not take points a second time for the same gap.
    """
    text = _text(root, SUMMARY_FILE)
    summary = _read_pairs(text, _answer_key)[0]
    if any(_readable(key, summary.get(key, "")) for key in ANSWER_KEYS):
        return
    message = (
        f"{SUMMARY_FILE} has no readable line for any key in README.md's table; "
        "write one `key: value` line per answer, such as `patients: <whole number>`."
    )
    unknown = next(
        (
            key.strip()
            for key, colon, _ in (line.strip().partition(":") for line in text.splitlines())
            if colon and key.strip() and not key.startswith("#") and _answer_key(_key_name(key)) is None
        ),
        None,
    )
    if unknown:
        message += f" Its first key, `{unknown}`, is not one of them."
    raise AssertionError(message)


def _mmhg_matches(given: float, expected: float) -> bool:
    """Within the tolerance, or the value with its decimals dropped, as `int()` or `astype(int)` writes it."""
    return abs(given - expected) <= MMHG_TOLERANCE or (given.is_integer() and given == math.trunc(expected))


def _check_answer(root: Path, key: str) -> None:
    """Compare one answer with the value recomputed from the supplied readings."""
    _assert(
        _artifact(root, SUMMARY_FILE) is not None,
        f"There is no {SUMMARY_FILE} to read; the summary artifact format check says why.",
    )
    summary, lines_naming = _read_pairs(_text(root, SUMMARY_FILE), _answer_key)
    _assert(key in summary, f"{SUMMARY_FILE} has no `{key}` line.")
    raw = summary[key]
    _assert(raw, f"The `{key}` line in {SUMMARY_FILE} has no value after the colon.")
    expected = expected_answers(load_dataset(root))[key]
    repeated = (
        f" `{key}` appears on {lines_naming[key]} lines and the checks read the first; "
        'write the file with "w" so each run replaces it.'
        if lines_naming.get(key, 0) > 1
        else ""
    )

    if key in LABEL_KEYS:
        _assert(
            _label_matches(raw, str(expected)),
            f"`{key}` is `{raw}`, which is not {LABEL_KEYS[key]} in {DATA_FILE}." + repeated,
        )
        return

    readings = _as_numbers(raw)
    _assert(bool(readings), f"`{key}` is `{raw}`, which is not a number." + repeated)
    if key in COUNT_KEYS:
        _assert(
            any(abs(given - float(expected)) <= 1e-9 for given in readings),
            f"`{key}` is {raw}, which is not {COUNT_KEYS[key]} in {DATA_FILE}." + repeated,
        )
    else:
        _assert(
            any(_mmhg_matches(given, float(expected)) for given in readings),
            f"`{key}` is {raw}, which is not {MMHG_KEYS[key]} in {DATA_FILE} "
            f"(allowed difference {MMHG_TOLERANCE} mmHg)." + repeated,
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
