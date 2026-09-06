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
    run = re.search(r"## Run\n\n([^\n]+)", readme)
    _assert(description is not None and "TODO" not in description.group(1), "Replace the project-description TODO line.")
    _assert(run is not None and run.group(1) == "python main.py", "Put exactly `python main.py` in the Run section.")
    _assert((root / ".gitignore").read_text(encoding="utf-8") == "__pycache__/\n*.pyc\n", "Set .gitignore to the two documented cache patterns.")


def check_git_state_answers(root: Path) -> None:
    answers = (root / "GIT_STATE_CHECK.md").read_text(encoding="utf-8")
    answer_block = re.search(r"<!-- ANSWERS START -->(.*?)<!-- ANSWERS END -->", answers, re.DOTALL)
    _assert(answer_block is not None and "TODO" not in answer_block.group(1), "Replace every TODO in the GIT_STATE_CHECK.md answer block.")


def check_report_artifact(root: Path) -> None:
    report = root / "report.txt"
    _assert(report.is_file() and not report.is_symlink(), "Commit report.txt as a regular file.")
    try:
        stored = report.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError("report.txt must be UTF-8 text.") from error
    _assert(stored == EXPECTED_REPORT, "report.txt must contain the documented three-line measurement report.")


PUBLIC_CHECKS = (
    PublicCheck("project description, run command, and gitignore", check_project_documents),
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
