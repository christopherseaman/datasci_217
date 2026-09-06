"""Smoke-test Assignment 02's saved-artifact contract."""

from pathlib import Path
import shutil
import sys
from uuid import uuid4


SOURCE = Path(__file__).resolve().parents[1]
SCRATCH = SOURCE.parents[1] / "scratch" / f"a02-artifacts-{uuid4().hex}"
SCRATCH.parent.mkdir(exist_ok=True)
shutil.copytree(SOURCE, SCRATCH, ignore=shutil.ignore_patterns("_grader_selftest", "__pycache__"))
sys.path.insert(0, str(SCRATCH))
from _public_checks import EXPECTED_REPORT, check_report_artifact


(SCRATCH / "report.txt").write_text(EXPECTED_REPORT, encoding="utf-8")
check_report_artifact(SCRATCH)
(SCRATCH / "report.txt").write_text("wrong\n", encoding="utf-8")
try:
    check_report_artifact(SCRATCH)
except AssertionError:
    print("[PASS] accepted the saved measurement report and rejected a mutation")
else:
    raise AssertionError("Mutated measurement report was accepted")
