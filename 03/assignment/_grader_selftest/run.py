"""Smoke-test Assignment 03's saved-artifact contract."""

from pathlib import Path
import shutil
import sys
from uuid import uuid4


SOURCE = Path(__file__).resolve().parents[1]
SCRATCH = SOURCE.parents[1] / "scratch" / f"a03-artifacts-{uuid4().hex}"
SCRATCH.parent.mkdir(exist_ok=True)
shutil.copytree(SOURCE, SCRATCH, ignore=shutil.ignore_patterns("_grader_selftest", "__pycache__"))
sys.path.insert(0, str(SCRATCH))
from _public_checks import (
    EXPECTED_ANALYSIS,
    EXPECTED_ENVIRONMENT_OUTPUT,
    EXPECTED_HEAD,
    EXPECTED_TAIL,
    check_pipeline_artifacts,
)


(SCRATCH / "output" / "environment_check.txt").write_text("\n".join(EXPECTED_ENVIRONMENT_OUTPUT) + "\n", encoding="utf-8")
(SCRATCH / "output" / "head_preview.txt").write_text("\n".join(EXPECTED_HEAD) + "\n", encoding="utf-8")
(SCRATCH / "output" / "tail_preview.txt").write_text("\n".join(EXPECTED_TAIL) + "\n", encoding="utf-8")
(SCRATCH / "output" / "site_counts.txt").write_text("3 north\n2 south\n1 west\n", encoding="utf-8")
(SCRATCH / "output" / "site_count_lines.txt").write_text("3 output/site_counts.txt\n", encoding="utf-8")
(SCRATCH / "output" / "analysis.txt").write_text("\n".join(EXPECTED_ANALYSIS) + "\n", encoding="utf-8")
check_pipeline_artifacts(SCRATCH)
(SCRATCH / "output" / "analysis.txt").write_text("wrong\n", encoding="utf-8")
try:
    check_pipeline_artifacts(SCRATCH)
except AssertionError:
    print("[PASS] accepted saved pipeline results and transcript, then rejected a mutation")
else:
    raise AssertionError("Mutated analysis transcript was accepted")
