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
ARTIFACTS = {'specimen_merge_audit.csv': 'specimen_id,collector_id,collection_number,station_code,material,mass_g,station_name,region,_merge\nSP101,C01,1,R,soil,12.5,River Station,north,both\nSP102,C01,2,R,water,8.0,River Station,north,both\nSP103,C02,1,S,soil,10.5,Shore Station,south,both\nSP104,C03,1,T,water,9.0,Trail Station,west,both\nSP105,C04,1,R,soil,11.5,River Station,north,both\nSP106,C05,1,X,air,4.0,,,left_only\nSP107,C06,1,S,water,7.5,Shore Station,south,both\n', 'combined_specimens.csv': 'specimen_id,collector_id,collection_number,station_code,material,mass_g,source_partition\nSP101,C01,1,R,soil,12.5,batch_a\nSP102,C01,2,R,water,8.0,batch_a\nSP103,C02,1,S,soil,10.5,batch_a\nSP104,C03,1,T,water,9.0,batch_a\nSP105,C04,1,R,soil,11.5,batch_b\nSP106,C05,1,X,air,4.0,batch_b\nSP107,C06,1,S,water,7.5,batch_b\n', 'aligned_features.csv': 'specimen_id,mass_g,review_score\nSP101,12.5,\nSP102,8.0,7.0\nSP103,10.5,9.0\nSP108,,6.0\n', 'sensor_scores_long.csv': 'sensor_id,station_code,measurement_label,value\nSN01,R,baseline_value,10.0\nSN02,S,baseline_value,8.5\nSN03,T,baseline_value,11.0\nSN04,R,baseline_value,9.5\nSN01,R,followup_value,12.5\nSN02,S,followup_value,9.0\nSN03,T,followup_value,13.5\nSN04,R,followup_value,10.5\n', 'sensor_scores_round_trip.csv': 'sensor_id,station_code,baseline_value,followup_value\nSN01,R,10.0,12.5\nSN02,S,8.5,9.0\nSN03,T,11.0,13.5\nSN04,R,9.5,10.5\n'}
NUMBER = "06"
POINTS = [0, 45, 30, 25]
MUTATION = ('combined_specimens.csv', 'mass_g')


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
