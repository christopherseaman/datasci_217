"""Value checks for Assignment 02. Course-side only; never shipped in a fork.

Every expected value is recomputed from the supplied encounter file, so there is
no answer key here either -- but a student who could read this file could read
the recomputation, so it lives in the course checks repository and reaches a
submission only through the GitHub Actions run.

The checks read committed artifacts only: `README.md`, `.gitignore`,
`data/clinic_encounters.csv`, and the two files in `output/`. Student source
code is never read, imported, or executed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re


DATA_FILE = Path("data") / "clinic_encounters.csv"
# The supplied encounter file, ignoring line-ending and trailing-whitespace
# differences. A changed file cannot be summarized into a known answer.
DATA_FINGERPRINT = "2ff169aa160fb1d0e4157aae93164ff59bd6c7cac0f0a8115db5b24f5c0e21df"

REPORT_FILE = Path("output") / "vitals_report.txt"
FOLLOWUP_FILE = Path("output") / "followup_list.txt"
REPORT_LABELS = (
    "usable encounters",
    "skipped rows",
    "patients seen",
    "mean systolic",
    "highest systolic",
    "lowest systolic",
)

FIELD_COUNT = 3
SYSTOLIC_MIN = 60
SYSTOLIC_MAX = 250
CUTOFF_MIN = 120
CUTOFF_MAX = 180

MEAN_TOLERANCE = 0.1
COUNT_TOLERANCE = 0.5
REASON_MIN_LENGTH = 20
REASON_MAX_LENGTH = 300
DESCRIPTION_MIN_LENGTH = 30
DESCRIPTION_MAX_LENGTH = 300

# A Python 3.13 command that runs a script, wherever it sits in the `## Run`
# section: alone on a line, inside a sentence, in a bullet, or in a code fence.
RUN_COMMAND = re.compile(
    r"(?<![\w.-])(?:python(?:3(?:\.13)?)?|py\s+-3(?:\.13)?)\s+[\w./\\-]*\.py(?![\w.])",
    re.IGNORECASE,
)
# NumPy prints a scalar as `np.float64(129.44)`, which is a correct answer
# written by a correct program.
NUMPY_SCALAR = re.compile(r"^(?:np|numpy)\.\w+\((.*)\)$")
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
BARE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# The bytecode cache, written the standard ways: the directory Python creates,
# or a glob for the files it puts there, including the `*.py[cod]` line in
# GitHub's own Python template.
CACHE_DIRECTORY_PATTERN = re.compile(r"^/?(?:\*\*/)?__pycache__/?(?:\*\*?/?)?$")
CACHE_FILE_PATTERN = re.compile(
    r"^/?(?:\*\*/)?(?:__pycache__/)?\*(?:\.py(?:[cod]|\[[cod]+\])|\$py\.class)$",
    re.IGNORECASE,
)
LABEL_LINES = ("cutoff", "reason")


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


@dataclass(frozen=True)
class SuppliedEncounters:
    """The supplied encounters, summarized the way the assignment defines them."""

    usable: tuple[tuple[str, int], ...]
    data_rows: int
    all_ids: frozenset[str]

    @property
    def skipped_rows(self) -> int:
        return self.data_rows - len(self.usable)

    @property
    def readings(self) -> tuple[int, ...]:
        return tuple(systolic for _, systolic in self.usable)

    @property
    def patients(self) -> set[str]:
        return {patient_id.casefold() for patient_id, _ in self.usable}

    def patients_at_or_above(self, cutoff: float) -> set[str]:
        return {patient_id.casefold() for patient_id, systolic in self.usable if systolic >= cutoff}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_artifact(root: Path, relative: Path, missing_message: str) -> str:
    """Read a committed artifact as text, tolerating the BOM Notepad and Excel add."""
    path = root / relative
    _assert(path.is_file() and not path.is_symlink(), missing_message)
    try:
        # utf-8-sig drops a leading byte-order mark and is otherwise plain UTF-8.
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise AssertionError(f"{relative.as_posix()} must be UTF-8 text.") from error
    except OSError as error:
        raise AssertionError(f"{relative.as_posix()} could not be read: {error}") from error


def _fingerprint(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _data_rows(text: str) -> list[str]:
    """Every line after the header, minus the blank line a final newline leaves."""
    rows = text.splitlines()[1:]
    while rows and not rows[-1].strip():
        rows.pop()
    return rows


def load_supplied_encounters(root: Path) -> SuppliedEncounters:
    """Apply the assignment's usable-row rule to the supplied encounter file."""
    text = _read_artifact(
        root,
        DATA_FILE,
        f"{DATA_FILE.as_posix()} is missing; restore the supplied encounter file.",
    )
    _assert(
        _fingerprint(text) == DATA_FINGERPRINT,
        f"{DATA_FILE.as_posix()} is not the supplied encounter file; restore it and summarize it as it ships.",
    )

    usable: list[tuple[str, int]] = []
    all_ids: set[str] = set()
    rows = _data_rows(text)
    for row in rows:
        fields = row.strip().split(",")
        if fields[0].strip():
            all_ids.add(fields[0].strip().casefold())
        if len(fields) != FIELD_COUNT:
            continue
        try:
            systolic = int(fields[2])
        except ValueError:
            continue
        if SYSTOLIC_MIN <= systolic <= SYSTOLIC_MAX:
            usable.append((fields[0].strip(), systolic))
    return SuppliedEncounters(usable=tuple(usable), data_rows=len(rows), all_ids=frozenset(all_ids))


def _labelled_values(text: str) -> dict[str, str]:
    """Map each `Label: value` line to its value, keeping the first of a repeat."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        values.setdefault(_label(label), value.strip())
    return values


def _label(text: str) -> str:
    """Normalize a label: case, inner spacing, and any bullet or emphasis marks."""
    return " ".join(text.split()).strip("-*#>\u2022 ").casefold()


def _parse_number(raw: str) -> tuple[float, int] | None:
    """Find the number on a labelled line and how many decimals it was written to.

    Words around the number are ignored, so `129.4 mm Hg`, `129.4 mmHg (mean)`
    and `np.float64(129.44)` all read as the number the student computed.
    """
    text = raw.strip()
    wrapped = NUMPY_SCALAR.match(text)
    if wrapped is not None:
        text = wrapped.group(1).strip()
    found = NUMBER.search(text)
    if found is None:
        return None
    written = found.group(0)
    _, point, fraction = written.partition(".")
    decimals = len(fraction) if point and "e" not in fraction.lower() else 0
    return float(written), decimals


def _report_values(root: Path) -> dict[str, str]:
    text = _read_artifact(
        root,
        REPORT_FILE,
        f"{REPORT_FILE.as_posix()} is missing; save your summary there.",
    )
    return _labelled_values(text)


def _report_number(root: Path, label: str) -> tuple[float, int]:
    values = _report_values(root)
    _assert(
        label in values,
        f"{REPORT_FILE.as_posix()} has no `{label.capitalize()}:` line.",
    )
    parsed = _parse_number(values[label])
    _assert(
        parsed is not None,
        f"{REPORT_FILE.as_posix()} gives no number after `{label.capitalize()}:`.",
    )
    return parsed


def _wrong(label: str, hint: str) -> str:
    return (
        f"{REPORT_FILE.as_posix()} gives a `{label.capitalize()}:` value that does not match "
        f"{DATA_FILE.as_posix()}. {hint}"
    )


def _check_count(root: Path, label: str, expected: int, hint: str) -> None:
    reported, _ = _report_number(root, label)
    _assert(abs(reported - expected) < COUNT_TOLERANCE, _wrong(label, hint))


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _readme_section(root: Path, heading: str) -> str | None:
    readme = _read_artifact(root, Path("README.md"), "README.md is missing.")
    found = re.search(
        rf"^## {heading}\s*$\n(.*?)(?=\n## |\Z)",
        readme,
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    return None if found is None else found.group(1)


def check_readme_description(root: Path) -> None:
    section = _readme_section(root, "Project description")
    _assert(
        section is not None,
        "README.md: keep the `## Project description` heading and write your description under it.",
    )
    written = " ".join(section.split())
    _assert(
        "TODO" not in written,
        "README.md: replace the TODO line under `## Project description` with your own description.",
    )
    _assert(
        DESCRIPTION_MIN_LENGTH <= len(written) <= DESCRIPTION_MAX_LENGTH,
        f"README.md: write {DESCRIPTION_MIN_LENGTH}-{DESCRIPTION_MAX_LENGTH} characters under "
        f"`## Project description` (yours has {len(written)}).",
    )


def check_readme_run_command(root: Path) -> None:
    section = _readme_section(root, "Run")
    _assert(section is not None, "README.md: keep the `## Run` heading and write the command under it.")
    _assert(
        RUN_COMMAND.search(section) is not None,
        "README.md: put a Python 3.13 command that runs your report script under `## Run`, such as "
        "`python3 clinic_report.py` or `py -3.13 clinic_report.py`. A sentence, a bullet, a code "
        "fence, or the bare command all count; the script name has to end in `.py`.",
    )


def check_gitignore_cache(root: Path) -> None:
    text = _read_artifact(root, Path(".gitignore"), ".gitignore is missing.")
    patterns = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    _assert(
        any(CACHE_DIRECTORY_PATTERN.match(pattern) for pattern in patterns)
        or any(CACHE_FILE_PATTERN.match(pattern) for pattern in patterns),
        ".gitignore lists no pattern for Python's bytecode cache. The standard patterns are "
        "`__pycache__/` for the directory and `*.pyc` (or `*.py[cod]`) for the compiled files.",
    )


def check_report_format(root: Path) -> None:
    values = _report_values(root)
    missing = [label for label in REPORT_LABELS if label not in values]
    _assert(
        not missing,
        f"{REPORT_FILE.as_posix()} has no line for: "
        + ", ".join(f"`{label.capitalize()}:`" for label in missing)
        + ".",
    )


def check_usable_encounters(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    _check_count(
        root,
        "usable encounters",
        len(encounters.usable),
        "Recheck the rule: three comma-separated fields, a systolic value `int()` can read, and a "
        f"reading from {SYSTOLIC_MIN} to {SYSTOLIC_MAX} mmHg.",
    )


def check_skipped_rows(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    _check_count(
        root,
        "skipped rows",
        encounters.skipped_rows,
        "Count every data row you could not use. The header is not a data row, but the blank line "
        "inside the export is one: it is a row you skipped, the way Demo 3 reports it.",
    )


def check_distinct_patients(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    _check_count(
        root,
        "patients seen",
        len(encounters.patients),
        "Count each patient ID once across the usable encounters only; some patients visited twice, "
        "and a patient whose only row was skipped was not seen.",
    )


def check_mean_systolic(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    reported, decimals = _report_number(root, "mean systolic")
    expected = sum(encounters.readings) / len(encounters.readings)
    # Compared at the precision the student wrote, so rounding never decides a grade.
    tolerance = max(MEAN_TOLERANCE, 0.5 * 10**-decimals)
    _assert(
        abs(reported - expected) <= tolerance,
        _wrong(
            "mean systolic",
            "Average every usable reading, including a patient's second visit. Any value within "
            f"{MEAN_TOLERANCE} mmHg passes, and so does the mean rounded to the decimals you wrote.",
        ),
    )


def check_highest_systolic(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    _check_count(
        root,
        "highest systolic",
        max(encounters.readings),
        "Take the largest usable reading, not the largest number in the file.",
    )


def check_lowest_systolic(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    _check_count(
        root,
        "lowest systolic",
        min(encounters.readings),
        f"Take the smallest usable reading, so nothing below {SYSTOLIC_MIN} mmHg counts.",
    )


def _followup_text(root: Path) -> str:
    return _read_artifact(
        root,
        FOLLOWUP_FILE,
        f"{FOLLOWUP_FILE.as_posix()} is missing; save your follow-up list there.",
    )


def _listed_patients(root: Path, known: frozenset[str]) -> set[str]:
    """The patient IDs named in the follow-up list, ignoring anything else on the page.

    A line is read for IDs unless it is the `Cutoff:` or `Reason:` line, so a
    heading, a blank line, a separator, or a bullet marker is simply ignored.
    """
    listed: set[str] = set()
    for line in _followup_text(root).splitlines():
        label, separator, _ = line.partition(":")
        if separator and _label(label) in LABEL_LINES:
            continue
        for token in BARE_TOKEN.findall(line):
            if token.casefold() in known:
                listed.add(token.casefold())
    return listed


def _declared_cutoff(root: Path) -> float:
    values = _labelled_values(_followup_text(root))
    _assert(
        "cutoff" in values,
        f"{FOLLOWUP_FILE.as_posix()} has no `Cutoff:` line, so there is no cutoff to check your list against.",
    )
    parsed = _parse_number(values["cutoff"])
    _assert(
        parsed is not None,
        f"{FOLLOWUP_FILE.as_posix()} gives no number after `Cutoff:` (the unit is optional).",
    )
    cutoff = parsed[0]
    _assert(
        CUTOFF_MIN <= cutoff <= CUTOFF_MAX,
        f"{FOLLOWUP_FILE.as_posix()} declares a cutoff of {cutoff:g} mmHg; choose one from "
        f"{CUTOFF_MIN} to {CUTOFF_MAX} mmHg.",
    )
    return cutoff


def check_followup_cutoff(root: Path) -> None:
    _declared_cutoff(root)


def check_followup_reason(root: Path) -> None:
    values = _labelled_values(_followup_text(root))
    _assert(
        "reason" in values,
        f"{FOLLOWUP_FILE.as_posix()} has no `Reason:` line saying why you chose your cutoff.",
    )
    reason = " ".join(values["reason"].split())
    _assert(
        REASON_MIN_LENGTH <= len(reason) <= REASON_MAX_LENGTH,
        f"{FOLLOWUP_FILE.as_posix()} needs a `Reason:` of {REASON_MIN_LENGTH}-{REASON_MAX_LENGTH} "
        f"characters on one line (yours has {len(reason)}).",
    )


def check_followup_patients(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    try:
        cutoff = _declared_cutoff(root)
    except AssertionError as error:
        # The list is read against the cutoff, so that line has to be right first.
        raise AssertionError(f"{error} Fix the `Cutoff:` line, and this list is checked against it.") from error
    listed = _listed_patients(root, encounters.all_ids)
    expected = encounters.patients_at_or_above(cutoff)

    missing = len(expected - listed)
    extra = len(listed - expected)
    problems = []
    if missing:
        problems.append(
            f"{missing} patient(s) with a usable reading at or above {cutoff:g} mmHg are not listed"
        )
    if extra:
        problems.append(f"{extra} listed patient(s) have no usable reading that high")
    _assert(
        not problems,
        f"{FOLLOWUP_FILE.as_posix()}: "
        + "; ".join(problems)
        + f". List one patient ID per line, each patient once, for the cutoff of {cutoff:g} mmHg "
        "you declared.",
    )


CHECKS = (
    Check("README project description", check_readme_description),
    Check("README run command", check_readme_run_command),
    Check(".gitignore bytecode cache", check_gitignore_cache),
    Check("vitals report format", check_report_format),
    Check("usable encounters", check_usable_encounters),
    Check("skipped rows", check_skipped_rows),
    Check("patients seen", check_distinct_patients),
    Check("mean systolic", check_mean_systolic),
    Check("highest systolic", check_highest_systolic),
    Check("lowest systolic", check_lowest_systolic),
    Check("follow-up cutoff", check_followup_cutoff),
    Check("follow-up reason", check_followup_reason),
    Check("follow-up patient list", check_followup_patients),
)


def run_checks(root: Path) -> list[tuple[str, str | None]]:
    results = []
    for check in CHECKS:
        try:
            check.action(root)
        except (AssertionError, OSError) as error:
            results.append((check.name, str(error)))
        else:
            results.append((check.name, None))
    return results
