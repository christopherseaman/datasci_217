"""Small artifact regression test; it never executes a student notebook."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from grader import grade_submission


ROOT = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\n"


def _context():
    original = os.environ.copy()
    os.environ.update({"ASSIGNMENT": "assignment-07", "SUBMISSION_TAG": "selftest", "COMMIT_URL": "https://example.test/commit", "RELEASE_URL": "https://example.test/release"})
    return original


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _numeric_rows(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    numeric = {"activities_completed", "reflection_score", "checkpoint_number", "completion_percent"}
    return [{key: float(value) if key in numeric else value for key, value in row.items()} for row in rows]


def _sample(root: Path) -> None:
    output = root / "output"
    output.mkdir(exist_ok=True)
    sessions = _numeric_rows(_rows(root / "data" / "session_observations.csv"))
    spec = {"data": {"values": list(reversed(sessions))}, "mark": {"type": "point"}, "encoding": {"shape": {"type": "nominal", "field": "pathway"}, "color": {"field": "pathway", "type": "N"}, "y": {"field": "reflection_score", "type": "quantitative"}, "x": {"type": "Q", "field": "activities_completed"}}}
    (output / "exploratory_spec.json").write_text(json.dumps(spec, indent=4) + "\n", encoding="utf-8")
    for name in ("critique_redesign.png", "pathway_explanatory.png"):
        (output / name).write_bytes(PNG + b"different-size-is-fine")
    pathways = _numeric_rows(_rows(root / "data" / "pathway_checkpoints.csv"))
    with (output / "explanatory_supporting_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["completion_percent", "pathway", "checkpoint_number"])
        writer.writeheader()
        writer.writerows(({key: f"{value:.1f}" if isinstance(value, float) else value for key, value in row.items()} for row in reversed(pathways)))
    evidence = {"text_alternative": "A descriptive text alternative.", "variable_roles": {"completion_percent": "quantitative", "pathway": "categorical", "checkpoint_number": "ordered"}, "grain": "one row per pathway and checkpoint", "displayed_unit": "percent", "intended_claim": "A bounded comparison.", "audience": "Coordinator", "question": "What changes?", "critique": [{"category": category, "problem": "A visible problem.", "repair": "A concrete repair."} for category in ("color-only encoding", "missing unit", "distracting decoration", "unsupported claim", "truncated baseline")]}
    (output / "visualization_evidence.json").write_text(json.dumps(evidence, indent=4) + "\n", encoding="utf-8")
    (output / "explanatory_text_alternative.txt").write_text("A descriptive text alternative.\n", encoding="utf-8")


def _score(root: Path) -> list[int]:
    return [test["score"] for test in grade_submission(root)["tests"]]


def main() -> int:
    original = _context()
    scratch = ROOT / "_grader_selftest" / ".scratch"
    scratch.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            base = Path(temporary) / "submission"
            shutil.copytree(ROOT, base, ignore=shutil.ignore_patterns("_grader_selftest", ".venv", "__pycache__"))
            _sample(base)
            public = subprocess.run([sys.executable, str(base / "check_assignment.py")], cwd=base, text=True, capture_output=True)
            assert public.returncode == 0, public.stdout + public.stderr
            assert _score(base) == [10, 15, 25, 25, 5]
            altered = Path(temporary) / "altered"
            shutil.copytree(base, altered)
            data = json.loads((altered / "output" / "exploratory_spec.json").read_text())
            data["data"]["values"][0]["activities_completed"] = 999
            (altered / "output" / "exploratory_spec.json").write_text(json.dumps(data), encoding="utf-8")
            assert _score(altered) == [10, 0, 25, 25, 5]
            missing = Path(temporary) / "missing"
            shutil.copytree(base, missing)
            (missing / "output" / "critique_redesign.png").unlink()
            assert _score(missing) == [10, 15, 0, 25, 0]
            for label, mutate in (
                ("empty-text", lambda evidence: evidence.__setitem__("question", "  ")),
                ("wrong-role", lambda evidence: evidence.__setitem__("variable_roles", {})),
            ):
                invalid = Path(temporary) / label
                shutil.copytree(base, invalid)
                evidence_path = invalid / "output" / "visualization_evidence.json"
                evidence = json.loads(evidence_path.read_text())
                mutate(evidence)
                evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
                assert _score(invalid) == [10, 15, 25, 0, 5]
            mismatch = Path(temporary) / "mismatched-sidecar"
            shutil.copytree(base, mismatch)
            (mismatch / "output" / "explanatory_text_alternative.txt").write_text("Different text.\n", encoding="utf-8")
            assert _score(mismatch) == [10, 15, 25, 0, 5]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        os.environ.clear()
        os.environ.update(original)
    print("Artifact regression selftest passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
