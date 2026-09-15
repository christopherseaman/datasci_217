"""Regression checks for Assignment 02 saved materials only."""

from pathlib import Path
import sys
import tempfile


ASSIGNMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT))
from _public_checks import EXPECTED_REPORT
from grading import grade_submission

README = "# Assignment\n\n## Project description\n\nA measurement summary project for saved readings.\n\n## Run\n\npython main.py\n"
GIT = "<!-- ANSWERS START -->\n1. working tree; diff\n2. staging area; commit\n3. local branch; remote; synchronize\n4. merge; conflict\n<!-- ANSWERS END -->\n"


def run() -> None:
    with tempfile.TemporaryDirectory(dir=ASSIGNMENT.parents[1] / "scratch", prefix="a02-artifacts-") as temporary:
        root = Path(temporary) / "submission"
        root.mkdir()
        assert grade_submission(root)["score"] == 0
        (root / "README.md").write_text(README, encoding="utf-8")
        (root / "GIT_STATE_CHECK.md").write_text(GIT, encoding="utf-8")
        (root / "report.txt").write_text(EXPECTED_REPORT, encoding="utf-8")
        assert grade_submission(root)["score"] == 100
        (root / "extra.bin").write_bytes(b"allowed")
        assert grade_submission(root)["score"] == 100
        (root / "report.txt").write_text("wrong\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 60
        (root / "GIT_STATE_CHECK.md").unlink()
        assert grade_submission(root)["score"] == 30
    print("Assignment 02: starter, artifact-only, extra-file, missing, and wrong-artifact regressions passed.")


if __name__ == "__main__":
    run()
