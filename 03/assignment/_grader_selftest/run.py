"""Regression checks for Assignment 03 saved artifacts only."""

from pathlib import Path
import sys
import tempfile


ASSIGNMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT))
from _public_checks import EXPECTED_ANALYSIS, EXPECTED_ENVIRONMENT_OUTPUT, EXPECTED_HEAD, EXPECTED_TAIL
from grading import grade_submission


def write(path: Path, lines: tuple[str, ...] | str) -> None:
    path.write_text(("\n".join(lines) if isinstance(lines, tuple) else lines) + "\n", encoding="utf-8")


def run() -> None:
    with tempfile.TemporaryDirectory(dir=ASSIGNMENT.parents[1] / "scratch", prefix="a03-artifacts-") as temporary:
        root = Path(temporary) / "submission"
        root.mkdir()
        assert grade_submission(root)["score"] == 0
        write(root / ".python-version", "3.14")
        write(root / "requirements.txt", "numpy==2.3.3")
        output = root / "output"
        output.mkdir()
        write(output / "environment_check.txt", EXPECTED_ENVIRONMENT_OUTPUT)
        write(output / "head_preview.txt", EXPECTED_HEAD)
        write(output / "tail_preview.txt", EXPECTED_TAIL)
        write(output / "site_counts.txt", "3 north\n2 south\n1 west")
        write(output / "site_count_lines.txt", "3 output/site_counts.txt")
        write(output / "analysis.txt", EXPECTED_ANALYSIS)
        assert grade_submission(root)["score"] == 100
        (root / "notes.md").write_text("allowed\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 100
        write(output / "analysis.txt", "wrong")
        assert grade_submission(root)["score"] == 60
        (output / "head_preview.txt").unlink()
        assert grade_submission(root)["score"] == 20
    print("Assignment 03: starter, artifact-only, extra-file, missing, and wrong-artifact regressions passed.")


if __name__ == "__main__":
    run()
