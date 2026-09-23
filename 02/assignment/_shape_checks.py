"""Shape checks for Assignment 02: are the artifacts well formed?

These run in your own repository, so they deliberately know nothing about the
answers. They confirm that each required file exists, is readable UTF-8, carries
the labels the assignment asks for, and gives a number where a number belongs --
one a clinician could read without blinking. They never open the encounter file,
never recompute anything from it, and never say whether a value is right.

Your values are checked when you push, by the checks the GitHub Actions run
downloads from the course.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re


REPORT_FILE = Path("output") / "vitals_report.txt"
FOLLOWUP_FILE = Path("output") / "followup_list.txt"

# Wide bands a clinician would accept, not answers: any real systolic reading
# and any real count lands inside them.
SYSTOLIC_MIN = 60
SYSTOLIC_MAX = 250
CUTOFF_MIN = 120
CUTOFF_MAX = 180

REASON_MIN_LENGTH = 20
REASON_MAX_LENGTH = 300
DESCRIPTION_MIN_LENGTH = 30
DESCRIPTION_MAX_LENGTH = 300

RUN_COMMAND = re.compile(
    r"(?<![\w.-])(?:python(?:3(?:\.13)?)?|py\s+-3(?:\.13)?)\s+[\w./\\-]*\.py(?![\w.])",
    re.IGNORECASE,
)
NUMPY_SCALAR = re.compile(r"^(?:np|numpy)\.\w+\((.*)\)$")
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
PATIENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CACHE_DIRECTORY_PATTERN = re.compile(r"^/?(?:\*\*/)?__pycache__/?(?:\*\*?/?)?$")
CACHE_FILE_PATTERN = re.compile(
    r"^/?(?:\*\*/)?(?:__pycache__/)?\*(?:\.py(?:[cod]|\[[cod]+\])|\$py\.class)$",
    re.IGNORECASE,
)
LABEL_LINES = ("cutoff", "reason")

COUNT_LABELS = ("usable encounters", "skipped rows", "patients seen")
SYSTOLIC_LABELS = ("mean systolic", "highest systolic", "lowest systolic")
REPORT_LABELS = COUNT_LABELS + SYSTOLIC_LABELS


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_artifact(root: Path, relative: Path, missing_message: str) -> str:
    """Read a saved artifact as text, tolerating the BOM Notepad and Excel add."""
    path = root / relative
    _assert(path.is_file() and not path.is_symlink(), missing_message)
    try:
        # utf-8-sig drops a leading byte-order mark and is otherwise plain UTF-8.
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise AssertionError(f"{relative.as_posix()} must be UTF-8 text.") from error
    except OSError as error:
        raise AssertionError(f"{relative.as_posix()} could not be read: {error}") from error


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


def _parse_number(raw: str) -> float | None:
    """Find the number on a labelled line, ignoring the words around it."""
    text = raw.strip()
    wrapped = NUMPY_SCALAR.match(text)
    if wrapped is not None:
        text = wrapped.group(1).strip()
    found = NUMBER.search(text)
    return None if found is None else float(found.group(0))


def _report_values(root: Path) -> dict[str, str]:
    text = _read_artifact(
        root,
        REPORT_FILE,
        f"{REPORT_FILE.as_posix()} is missing; save your summary there.",
    )
    return _labelled_values(text)


def _followup_text(root: Path) -> str:
    return _read_artifact(
        root,
        FOLLOWUP_FILE,
        f"{FOLLOWUP_FILE.as_posix()} is missing; save your follow-up list there.",
    )


def _number_on_line(relative: Path, values: dict[str, str], label: str) -> float:
    _assert(
        label in values,
        f"{relative.as_posix()} has no `{label.capitalize()}:` line.",
    )
    number = _parse_number(values[label])
    _assert(
        number is not None,
        f"{relative.as_posix()} gives no number after `{label.capitalize()}:`.",
    )
    return number


def _check_count_shape(root: Path, label: str) -> None:
    """A count is a whole number of things, so it cannot be negative or fractional."""
    number = _number_on_line(REPORT_FILE, _report_values(root), label)
    _assert(
        number >= 0 and abs(number - round(number)) < 1e-9,
        f"{REPORT_FILE.as_posix()} gives `{number:g}` after `{label.capitalize()}:`; a count is a "
        "whole number of rows or patients, so it is never negative and never fractional.",
    )


def _check_systolic_shape(root: Path, label: str) -> None:
    """A systolic value has to be a blood pressure, whatever the right answer is."""
    number = _number_on_line(REPORT_FILE, _report_values(root), label)
    _assert(
        SYSTOLIC_MIN <= number <= SYSTOLIC_MAX,
        f"{REPORT_FILE.as_posix()} gives `{number:g}` after `{label.capitalize()}:`, which is not a "
        f"systolic blood pressure; the usable readings run from {SYSTOLIC_MIN} to {SYSTOLIC_MAX} mmHg.",
    )


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
    _check_count_shape(root, "usable encounters")


def check_skipped_rows(root: Path) -> None:
    _check_count_shape(root, "skipped rows")


def check_distinct_patients(root: Path) -> None:
    _check_count_shape(root, "patients seen")


def check_mean_systolic(root: Path) -> None:
    _check_systolic_shape(root, "mean systolic")


def check_highest_systolic(root: Path) -> None:
    _check_systolic_shape(root, "highest systolic")


def check_lowest_systolic(root: Path) -> None:
    _check_systolic_shape(root, "lowest systolic")


def check_followup_cutoff(root: Path) -> None:
    values = _labelled_values(_followup_text(root))
    cutoff = _number_on_line(FOLLOWUP_FILE, values, "cutoff")
    _assert(
        CUTOFF_MIN <= cutoff <= CUTOFF_MAX,
        f"{FOLLOWUP_FILE.as_posix()} declares a cutoff of {cutoff:g} mmHg; choose one from "
        f"{CUTOFF_MIN} to {CUTOFF_MAX} mmHg.",
    )


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


def _listed_ids(text: str) -> list[str]:
    """Read one patient ID per line, ignoring bullets, punctuation and decoration."""
    listed = []
    for line in text.splitlines():
        entry = line.strip()
        label, separator, _ = entry.partition(":")
        if separator:
            if _label(label) in LABEL_LINES:
                continue
            entry = label.strip()
        entry = entry.lstrip("-*• \t").strip().rstrip(",;")
        if PATIENT_ID.match(entry) and any(character.isdigit() for character in entry):
            listed.append(entry.casefold())
    return listed


def check_followup_patients(root: Path) -> None:
    """The list has patient IDs on it, one per line, and none listed twice."""
    listed = _listed_ids(_followup_text(root))
    _assert(
        listed,
        f"{FOLLOWUP_FILE.as_posix()} lists no patient IDs. After the `Cutoff:` and `Reason:` lines, "
        "write one patient ID per line.",
    )
    repeated = sorted({entry for entry in listed if listed.count(entry) > 1})
    _assert(
        not repeated,
        f"{FOLLOWUP_FILE.as_posix()} lists {', '.join(repeated)} more than once; each patient is "
        "called back once, however many visits they made.",
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
