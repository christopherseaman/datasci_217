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
ARTIFACTS = {'issue_audit.csv': 'issue,count\nschema mismatch,0\nempty full-name tokens,1\nempty date tokens,1\nage sentinel tokens,3\nstatus sentinel tokens,1\nage parse failures,1\nnumeric but noninteger age values,1\nage values outside 0 through 120,1\ndate parse failures,3\nrows in exact duplicate sets,2\nrows with repeated candidate IDs,2\nsite values needing format normalization,4\nstatus values needing format normalization,3\nunexpected site values,0\nunexpected non-sentinel status values,0\n', 'cleaned_people.csv': 'record_id,full_name,site,status,age,visit_date,needs_review\nR001,Alice Smith,north,active,34,2026-01-15,False\nR002,Bob Jones,north,active,,,True\nR003,Carla Ruiz,south,pending,,2026-03-01,True\nR004,,south,,45,,True\nR005,Evan Li,west,complete,52,2026-02-14,False\nR006,Fatima Noor,north,active,,2026-04-01,True\nR007,Grace Chen,south,active,,2026-05-01,True\nR008,Hugo Diaz,west,pending,,2026-06-01,True\nR009,Inez Park,north,complete,39,,True\nR010,Jamie Okafor,west,complete,28,2026-07-15,False\nR011,Kai Patel,south,pending,0,2026-08-01,False\n', 'decision_log.csv': 'field,issue,action,reason,source,source_sha256,rows_before,rows_after\nfull_name,empty optional name,retain as missing,Names are optional and empty source tokens should remain reviewable.,data/people_raw.csv,d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b,12,11\n"full_name, site, status",surrounding whitespace and case variants,strip surrounding whitespace and normalize bounded field case,Canonical text formatting makes valid categories comparable.,data/people_raw.csv,d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b,12,11\nstatus,NA sentinel,convert the documented sentinel to missing,The documented status sentinel carries no usable status.,data/people_raw.csv,d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b,12,11\nage_text,unknown and -9 sentinels,convert the documented sentinels to missing,These documented tokens represent unavailable age.,data/people_raw.csv,d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b,12,11\nage_text,"nonnumeric, fractional, or out-of-range values",coerce invalid values to missing without rounding,The age contract accepts finite integers from 0 through 120.,data/people_raw.csv,d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b,12,11\nvisit_date,"empty, lexically invalid, or calendar-invalid values",coerce invalid values to missing after an exact-format check,Only real ASCII YYYY-MM-DD calendar dates satisfy the date contract.,data/people_raw.csv,d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b,12,11\nall raw columns,exact duplicate submissions,keep the first exact raw row only,Repeated identical submissions add no new record information.,data/people_raw.csv,d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b,12,11\nall fields,adjacent-row filling,do not forward-fill or backward-fill,"Rows are independent people, so neighboring values cannot be borrowed.",data/people_raw.csv,d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b,12,11\n'}
ARTIFACTS.update({
    "raw_preview.txt": (
        "$ head -n 4 data/people_raw.csv\n"
        "record_id,full_name,site,status,age_text,visit_date\n"
        "R001, Alice Smith , North ,Active,34,2026-01-15\n"
        "R002,BOB JONES,north,active,unknown,2026-02-30\n"
        "R002,BOB JONES,north,active,unknown,2026-02-30\n"
        "$ tail -n 2 data/people_raw.csv\n"
        "R010,Jamie Okafor,West,Complete,28,2026-07-15\n"
        "R011,Kai Patel,south, pending ,0,2026-08-01\n"
    ),
    "numpy_age_summary.csv": "metric,value\ncount,6\nmin,0\nmax,52\nsum,198\nmean,33.0\n",
    "pandas_selection.csv": "record_id,site,status\nR001, North ,Active\nR003,SOUTH,pending\nR010,West,Complete\n",
    "pipeline_summary.txt": (
        "raw_rows=12\n"
        "raw_columns=6\n"
        "exact_duplicate_rows=1\n"
        "candidate_id_duplicate_rows=1\n"
        "clean_rows=11\n"
    ),
})
NUMBER = "05"
POINTS = [0, 25, 35, 25]
MUTATION = ('cleaned_people.csv', 'age')


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
            if not name.endswith(".csv"):
                continue
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
