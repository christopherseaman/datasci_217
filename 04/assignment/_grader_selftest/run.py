"""Regression checks for committed artifacts; no student code is executed."""

import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile


ASSIGNMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT))
from grading import grade_submission

ARTIFACTS = {
    "labeled_block.csv": "record_id,baseline_c,follow_up_c\nsite-102,15,23\nsite-103,10,17\n",
    "selected_purchases.csv": "purchase_id,item,quantity,unit_price,line_total\nP008,Laptop Stand,2,20.0,40.0\nP003,Water Bottle,3,10.0,30.0\nP004,Desk Lamp,2,15.0,30.0\nP006,Headphones,4,7.5,30.0\nP001,USB Cable,2,8.0,16.0\nP011,USB Hub,2,8.0,16.0\nP007,Webcam Cover,5,3.0,15.0\nP009,Cable Tie,3,5.0,15.0\nP012,Notebook,3,5.0,15.0\n",
}


def public(root: Path, expected: bool) -> None:
    result = subprocess.run([sys.executable, "-B", str(ASSIGNMENT / "check_assignment.py"), str(root)], text=True, capture_output=True)
    assert (result.returncode == 0) == expected, result.stdout + result.stderr


def run() -> None:
    with tempfile.TemporaryDirectory(dir=ASSIGNMENT.parents[1] / "scratch", prefix="a04-artifacts-") as temporary:
        root = Path(temporary) / "submission"
        root.mkdir()
        public(root, False)
        assert grade_submission(root)["score"] == 0
        output = root / "output"
        output.mkdir()
        for name, content in ARTIFACTS.items():
            (output / name).write_text(content, encoding="utf-8")
        public(root, True)
        assert grade_submission(root)["score"] == 100
        (root / "notes.txt").write_text("extra files are allowed\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 100
        path = output / "selected_purchases.csv"
        rows = list(csv.reader(io.StringIO(path.read_text())))
        rows[1][-1] = "41"
        stream = io.StringIO()
        csv.writer(stream).writerows(rows)
        path.write_text(stream.getvalue(), encoding="utf-8")
        assert grade_submission(root)["score"] == 40
        (output / "labeled_block.csv").unlink()
        assert grade_submission(root)["score"] == 0
    print("Assignment 04: starter, artifact-only, extra-file, missing, and wrong-artifact regressions passed.")


if __name__ == "__main__":
    run()
