"""Regression checks for committed artifacts; no student code is executed."""
from __future__ import annotations

import csv
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ASSIGNMENT = Path(__file__).resolve().parents[1]
REPO = ASSIGNMENT.parents[1]
ARTIFACTS = {'labeled_block.csv': 'record_id,baseline_c,follow_up_c\nsite-102,15,23\nsite-103,10,17\n', 'selected_purchases.csv': 'purchase_id,item,quantity,unit_price,line_total\nP008,Laptop Stand,2,20.0,40.0\nP003,Water Bottle,3,10.0,30.0\nP004,Desk Lamp,2,15.0,30.0\nP006,Headphones,4,7.5,30.0\nP001,USB Cable,2,8.0,16.0\nP011,USB Hub,2,8.0,16.0\nP007,Webcam Cover,5,3.0,15.0\nP009,Cable Tie,3,5.0,15.0\nP012,Notebook,3,5.0,15.0\n'}
NUMBER = "04"
POINTS = [20, 30, 50]
MUTATION = ('selected_purchases.csv', 'line_total')


def run() -> None:
    scratch = REPO / "scratch"
    scratch.mkdir(exist_ok=True)
    for key in ("ASSIGNMENT", "SUBMISSION_TAG", "COMMIT_URL", "RELEASE_URL"):
        os.environ[key] = "artifact-regression"
    path = Path(__file__).with_name("grader_core.py" if NUMBER == "04" else "grader.py")
    spec = importlib.util.spec_from_file_location(f"a{NUMBER}_regression_grader", path)
    grader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = grader
    spec.loader.exec_module(grader)

    def scores(root):
        result = grader.grade_submission(root)
        if NUMBER == "04":
            return [test.score for test in result]
        return [test["score"] for test in result["tests"]]

    def public(root, expected):
        result = subprocess.run(
            [sys.executable, "-B", "check_assignment.py"], cwd=root,
            text=True, capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert (result.returncode == 0) == expected, result.stdout + result.stderr

    with tempfile.TemporaryDirectory(dir=scratch, prefix=f"a{NUMBER}-artifact-regression-") as temporary:
        root = Path(temporary) / "submission"
        shutil.copytree(ASSIGNMENT, root, ignore=shutil.ignore_patterns("_grader_selftest", "__pycache__", ".pytest_cache", ".venv"))
        public(root, False)
        for name, content in ARTIFACTS.items():
            (root / "output" / name).write_text(content, encoding="utf-8")
        public(root, True)
        baseline = scores(root)
        assert baseline == POINTS, baseline
        for name in ARTIFACTS:
            path = root / "output" / name
            rows = list(csv.reader(io.StringIO(path.read_text())))
            stream = io.StringIO()
            csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\r\n").writerows(rows)
            path.write_bytes(stream.getvalue().encode())
        public(root, True)
        assert scores(root) == POINTS

        first = next(iter(ARTIFACTS))
        (root / "output" / first).unlink()
        public(root, False)
        partial = scores(root)
        assert partial[1] == 0 and partial[2:] == POINTS[2:], partial
        (root / "output" / first).write_text(ARTIFACTS[first], encoding="utf-8")

        name, column = MUTATION
        path = root / "output" / name
        rows = list(csv.reader(io.StringIO(path.read_text())))
        position = rows[0].index(column)
        rows[1][position] = str(float(rows[1][position]) + 1)
        stream = io.StringIO()
        csv.writer(stream).writerows(rows)
        path.write_text(stream.getvalue(), encoding="utf-8")
        public(root, False)
        partial = scores(root)
        assert partial[1] == POINTS[1] and partial[2] == 0, partial
    print(f"Assignment {NUMBER}: starter, complete, equivalent CSV, missing milestone, and wrong-value regressions passed.")


if __name__ == "__main__":
    run()
