"""Public artifact checks for Assignment 01."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


EXPECTED_READINESS = """Python family: 3.12
Project: DataSci 217 Assignment 01
Script: readiness.py
First measurement: 18
Measurement 18: within range
Measurement 21: review
Measurement 24: review
Measurement 19: within range
Count: 4
Total: 82
Mean: 20.5
Review count: 2
Readiness: complete
Participant count: 4
Next checkpoint: 5
"""


@dataclass(frozen=True)
class PublicCheck:
    name: str
    action: Callable[[Path], None]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_terminal_practice(root: Path) -> None:
    practice = root / "terminal-practice"
    _assert(practice.is_dir() and not practice.is_symlink(), "Create a regular terminal-practice directory.")
    names = {path.name for path in practice.iterdir()}
    _assert(names == {"source.txt", "path-check.txt"}, "terminal-practice must contain only source.txt and path-check.txt.")
    for name in names:
        _assert((practice / name).is_file() and not (practice / name).is_symlink(), f"terminal-practice/{name} must be a regular file.")


def check_output_artifact(root: Path) -> None:
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Create a regular output/ directory.")
    names = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert(names == {".gitkeep", "readiness.txt"}, "Commit exactly output/.gitkeep and output/readiness.txt.")
    report = output / "readiness.txt"
    _assert(report.is_file() and not report.is_symlink(), "Commit output/readiness.txt as a regular file.")
    try:
        stored = report.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError("output/readiness.txt must be UTF-8 text.") from error
    _assert(stored == EXPECTED_READINESS, "output/readiness.txt must contain the documented 15-line readiness report.")


PUBLIC_CHECKS = (
    PublicCheck("terminal practice evidence", check_terminal_practice),
    PublicCheck("committed readiness artifact", check_output_artifact),
)


def run_public_checks(root: Path) -> list[tuple[str, str | None]]:
    results = []
    for check in PUBLIC_CHECKS:
        try:
            check.action(root)
        except (AssertionError, OSError) as error:
            results.append((check.name, str(error)))
        else:
            results.append((check.name, None))
    return results
