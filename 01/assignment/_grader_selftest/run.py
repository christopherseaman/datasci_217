"""Smoke-test Assignment 01's saved-artifact contract."""

from pathlib import Path
import shutil
from uuid import uuid4


SOURCE = Path(__file__).resolve().parents[1]
SCRATCH = SOURCE.parents[1] / "scratch" / f"a01-artifacts-{uuid4().hex}"
SCRATCH.parent.mkdir(exist_ok=True)
shutil.copytree(SOURCE, SCRATCH, ignore=shutil.ignore_patterns("_grader_selftest", "__pycache__"))

import sys
sys.path.insert(0, str(SCRATCH))
from _assignment_checks import EXPECTED_READINESS, check_output_artifact


(SCRATCH / "output" / "readiness.txt").write_text(EXPECTED_READINESS, encoding="utf-8")
check_output_artifact(SCRATCH)
(SCRATCH / "output" / "readiness.txt").write_text("wrong\n", encoding="utf-8")
try:
    check_output_artifact(SCRATCH)
except AssertionError:
    print("[PASS] accepted the saved readiness report and rejected a mutation")
else:
    raise AssertionError("Mutated readiness report was accepted")
