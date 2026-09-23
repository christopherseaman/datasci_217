"""Shape checks for Assignment 01: are the artifacts well formed?

These run in your own repository, so they deliberately know nothing about the
answers. They confirm that each required file exists as a regular file, that
the readiness report is UTF-8 text with the right number of lines and a
`Python family:` first line, and that the identity file holds one hash. They
never say whether a report line is right or whether a hash is on the roster.

Your values are checked when you push, by the checks the GitHub Actions run
downloads from the course.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re


PRACTICE_DIR = Path("terminal-practice")
PRACTICE_FILES = ("source.txt", "path-check.txt")
OUTPUT_DIR = Path("output")
READINESS_FILE = OUTPUT_DIR / "readiness.txt"
IDENTITY_FILE = OUTPUT_DIR / "student_identity.txt"

# The report's first line records whichever Python ran readiness.py; any version counts.
PYTHON_FAMILY = re.compile(r"Python family: \d+\.\d+")
# A SHA-256 hash as capture_identity.py saves it: 64 hexadecimal digits.
IDENTITY_HASH = re.compile(r"[0-9a-f]{64}")
# One line each from readiness.py (3), measurement_summary.py (8), and debug_report.py (3).
READINESS_LINES = 14


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_text(path: Path, missing_message: str, encoding_message: str) -> str:
    """Read a committed artifact as UTF-8 text; a folder or a symlink is not one."""
    _assert(path.is_file() and not path.is_symlink(), missing_message)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError(encoding_message) from error


def _after_python_family(report: str) -> str | None:
    """The report without its first line, which records whichever Python ran readiness.py."""
    first, _, rest = report.partition("\n")
    return rest if PYTHON_FAMILY.fullmatch(first) else None


def _readiness_report(root: Path) -> str:
    output = root / OUTPUT_DIR
    _assert(output.is_dir() and not output.is_symlink(), "Create a regular output/ directory.")
    return _read_text(
        root / READINESS_FILE,
        "Commit output/readiness.txt as a regular file.",
        "output/readiness.txt must be UTF-8 text.",
    )


def _identity_hash(root: Path) -> str:
    """The saved hash with surrounding whitespace and letter case ignored."""
    text = _read_text(
        root / IDENTITY_FILE,
        "Run capture_identity.py and commit output/student_identity.txt.",
        "student_identity.txt must be UTF-8 text.",
    )
    identity_hash = text.strip().lower()
    _assert(
        IDENTITY_HASH.fullmatch(identity_hash) is not None,
        "student_identity.txt must contain one SHA-256 hash; surrounding whitespace and letter case are ignored.",
    )
    return identity_hash


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_terminal_practice(root: Path) -> None:
    practice = root / PRACTICE_DIR
    _assert(practice.is_dir() and not practice.is_symlink(), "Create a regular terminal-practice directory.")
    for name in PRACTICE_FILES:
        _assert(
            (practice / name).is_file() and not (practice / name).is_symlink(),
            f"terminal-practice/{name} must be a regular file.",
        )


def check_output_artifact(root: Path) -> None:
    report = _readiness_report(root)
    _assert(
        _after_python_family(report) is not None,
        "output/readiness.txt must start with a line such as `Python family: 3.13`; "
        "run make_output.py again after readiness.py prints it.",
    )
    _assert(
        report.endswith("\n"),
        "output/readiness.txt must end with a newline; save it with make_output.py instead of editing it.",
    )
    lines = len(report.splitlines())
    _assert(
        lines == READINESS_LINES,
        f"output/readiness.txt must hold {READINESS_LINES} lines, one per printed line (yours has {lines}); "
        "run make_output.py again after all three scripts print what the README shows.",
    )
    _identity_hash(root)


CHECKS = (
    Check("terminal practice evidence", check_terminal_practice),
    Check("committed readiness and identity artifacts", check_output_artifact),
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
