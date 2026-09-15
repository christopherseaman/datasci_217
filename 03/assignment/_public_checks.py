"""Public artifact checks for Assignment 03."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


EXPECTED_PYTHON_RECORD = "3.14\n"
EXPECTED_REQUIREMENTS = "numpy==2.3.3\n"
EXPECTED_ENVIRONMENT_OUTPUT = ("Python: 3.14", "NumPy: 2.3.3")
EXPECTED_HEAD = ("site,baseline,follow_up", "north,10,20", "south,20,30")
EXPECTED_TAIL = ("south,20,30", "north,30,40")
EXPECTED_ANALYSIS = (
    "Measurements shape: (6, 2)",
    "Measurements dtype: float64",
    "Overall mean: 25.0",
    "Column means: [20. 30.]",
    "Row means: [15. 25. 35. 15. 25. 35.]",
    "Values at or above 30: 6",
    "First value: 10.0",
    "Second row: [20. 30.]",
    "Second column: [20. 30. 20. 20. 30. 40.]",
    "Top-left block: [[10. 20.]",
    " [20. 30.]]",
    "View before change: [[20. 30.]",
    " [30. 40.]]",
    "Copy before change: [[20. 30.]",
    " [30. 40.]]",
    "View after source change: [[-99.  30.]",
    " [ 30.  40.]]",
    "Copy after source change: [[20. 30.]",
    " [30. 40.]]",
    "Mask at or above 30: [False  True False False  True  True]",
    "Selected values: [30. 30. 40.]",
    "Difference from baseline: [10. 10. 10. 10. 10. 10.]",
    "Adjusted values: [25. 35. 45. 25. 35. 45.]",
    "Grid: [[10. 20. 20. 30.]",
    " [30. 40. 10. 20.]",
    " [20. 30. 30. 40.]]",
    "Transpose: [[10. 30. 20.]",
    " [20. 40. 30.]",
    " [20. 10. 30.]",
    " [30. 20. 40.]]",
)


@dataclass(frozen=True)
class PublicCheck:
    name: str
    action: Callable[[Path], None]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _lines(root: Path, filename: str) -> tuple[str, ...]:
    path = root / filename
    _assert(path.is_file() and not path.is_symlink(), f"Commit {filename} as a regular artifact.")
    try:
        return tuple(path.read_text(encoding="utf-8").splitlines())
    except UnicodeDecodeError as error:
        raise AssertionError(f"{filename} must be UTF-8 text.") from error


def check_environment_artifacts(root: Path) -> None:
    _assert((root / ".python-version").read_text(encoding="utf-8") == EXPECTED_PYTHON_RECORD, "Replace .python-version with exactly `3.14` and one final newline.")
    _assert((root / "requirements.txt").read_text(encoding="utf-8") == EXPECTED_REQUIREMENTS, "Replace requirements.txt with exactly `numpy==2.3.3` and one final newline.")
    _assert(_lines(root, "output/environment_check.txt") == EXPECTED_ENVIRONMENT_OUTPUT, "output/environment_check.txt must record the documented Python and NumPy versions.")


def check_pipeline_artifacts(root: Path) -> None:
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Create a regular output/ directory.")
    _assert(_lines(root, "output/head_preview.txt") == EXPECTED_HEAD, "head_preview.txt must show the first three fixture lines.")
    _assert(_lines(root, "output/tail_preview.txt") == EXPECTED_TAIL, "tail_preview.txt must show the final two fixture lines.")
    counts = [line.split() for line in _lines(root, "output/site_counts.txt")]
    _assert(counts == [["3", "north"], ["2", "south"], ["1", "west"]], "site_counts.txt must record the three documented count/name pairs.")
    _assert((root / "output/site_count_lines.txt").read_text(encoding="utf-8").split() == ["3", "output/site_counts.txt"], "site_count_lines.txt must record the documented wc result.")


def check_analysis_artifact(root: Path) -> None:
    _assert(_lines(root, "output/analysis.txt") == EXPECTED_ANALYSIS, "analysis.txt must contain the documented NumPy learning transcript.")


PUBLIC_CHECKS = (
    PublicCheck("committed environment records and probe", check_environment_artifacts),
    PublicCheck("committed pipeline artifacts", check_pipeline_artifacts),
    PublicCheck("committed analysis transcript", check_analysis_artifact),
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
