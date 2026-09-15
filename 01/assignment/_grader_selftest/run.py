"""Regression checks for Assignment 01 saved artifacts only."""

from pathlib import Path
import sys
import tempfile


ASSIGNMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT))
from _assignment_checks import EXPECTED_READINESS
from grading import grade_submission


def run() -> None:
    with tempfile.TemporaryDirectory(dir=ASSIGNMENT.parents[1] / "scratch", prefix="a01-artifacts-") as temporary:
        root = Path(temporary) / "submission"
        root.mkdir()
        assert grade_submission(root)["score"] == 0
        practice = root / "terminal-practice"
        practice.mkdir()
        for name in ("source.txt", "path-check.txt"):
            (practice / name).write_text("", encoding="utf-8")
        output = root / "output"
        output.mkdir()
        (output / "readiness.txt").write_text(EXPECTED_READINESS, encoding="utf-8")
        assert grade_submission(root)["score"] == 100
        (root / "notes.txt").write_text("extra files are allowed\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 100
        (output / "readiness.txt").write_text("wrong\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 20
        (output / "readiness.txt").unlink()
        assert grade_submission(root)["score"] == 20
        (output / "readiness.txt").write_text(EXPECTED_READINESS, encoding="utf-8")
        (practice / "source.txt").unlink()
        assert grade_submission(root)["score"] == 80
    print("Assignment 01: starter, artifact-only, extra-file, missing, and wrong-artifact regressions passed.")


if __name__ == "__main__":
    run()
