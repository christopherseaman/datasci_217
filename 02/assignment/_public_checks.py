"""Public artifact checks for Assignment 02."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re


EXPECTED_REPORT = """Morning mean: 21.0
Evening mean: 22.7
Overnight mean: no measurements
"""
RUN_COMMAND = re.compile(r"(?<![\w.-])(?:python(?:3(?:\.13)?)?|py\s+-3\.13)\s+(?:\./)?main\.py(?=$|[\s`])")


@dataclass(frozen=True)
class PublicCheck:
    name: str
    action: Callable[[Path], None]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_project_documents(root: Path) -> None:
    readme = (root / "README.md").read_text(encoding="utf-8")
    description = re.search(r"## Project description\n\n([^\n]+)", readme)
    run = re.search(r"## Run\n\n(.*?)(?=\n## |\Z)", readme, re.DOTALL)
    _assert(
        description is not None
        and 30 <= len(description.group(1).strip()) <= 300
        and "measurement" in description.group(1).lower(),
        "Write a 30–300 character project description containing the word `measurement`.",
    )
    _assert(
        run is not None and RUN_COMMAND.search(run.group(1)),
        "Put a Python 3.13 command that runs main.py in the Run section.",
    )


def check_git_state_answers(root: Path) -> None:
    answers = (root / "GIT_STATE_CHECK.md").read_text(encoding="utf-8")
    answer_block = re.search(r"<!-- ANSWERS START -->(.*?)<!-- ANSWERS END -->", answers, re.DOTALL)
    _assert(answer_block is not None, "Complete the four documented Git state answers.")
    lines = [line.strip() for line in answer_block.group(1).strip().splitlines()]
    expected = (
        ("1.", ("working tree", "diff")),
        ("2.", ("staging area", "commit")),
        ("3.", ("local branch", "remote", "synchronize")),
        ("4.", ("merge", "conflict")),
    )
    _assert(len(lines) == len(expected), "Complete the four documented Git state answers.")
    for line, (number, terms) in zip(lines, expected, strict=True):
        actual = tuple(part.strip().casefold() for part in line.removeprefix(number).split(";"))
        _assert(line.startswith(number) and actual == terms, "Complete the four documented Git state answers.")


def check_report_artifact(root: Path) -> None:
    report = root / "report.txt"
    _assert(report.is_file() and not report.is_symlink(), "Commit report.txt as a regular file.")
    try:
        stored = report.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError("report.txt must be UTF-8 text.") from error
    _assert(stored == EXPECTED_REPORT, "report.txt must contain the documented three-line measurement report.")


PUBLIC_CHECKS = (
    PublicCheck("project description and run command", check_project_documents),
    PublicCheck("Git state snapshots", check_git_state_answers),
    PublicCheck("committed report artifact", check_report_artifact),
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
