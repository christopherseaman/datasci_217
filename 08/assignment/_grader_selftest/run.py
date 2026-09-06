"""Artifact-only regression checks for Assignment 08."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import grader
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENV = {"ASSIGNMENT": "a08", "SUBMISSION_TAG": "selftest", "COMMIT_URL": "https://example.invalid/c", "RELEASE_URL": "https://example.invalid/r"}

def copy_case(work: Path, name: str, *, completed: bool = True) -> Path:
    target = work / name
    shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns("_grader_selftest", ".venv", "__pycache__", "result.json"))
    if completed:
        source = grader._fixture(target)
        output = target / "output"
        source.groupby("center", sort=True).agg(request_count=("request_id", "size"), satisfaction_count=("satisfaction_score", "count"), unique_agent_count=("agent_id", "nunique")).reset_index().to_csv(output / "center_count_summary.csv", index=False)
        source.groupby("center", sort=True).agg(request_count=("request_id", "size"), satisfaction_count=("satisfaction_score", "count"), unique_agent_count=("agent_id", "nunique"), total_resolution_minutes=("resolution_minutes", "sum"), mean_resolution_minutes=("resolution_minutes", "mean")).reset_index().to_csv(output / "center_summary.csv", index=False)
        means = source.groupby("center")["resolution_minutes"].transform("mean")
        context = source.copy(); context["center_mean_resolution_minutes"] = means; context["difference_from_center_mean"] = source["resolution_minutes"] - means
        context.to_csv(output / "requests_with_context.csv", index=False)
        source.groupby(["center", "channel"], sort=True).agg(request_count=("request_id", "size"), mean_resolution_minutes=("resolution_minutes", "mean")).reset_index().to_csv(output / "center_channel_summary.csv", index=False)
        pd.pivot_table(source, index="center", columns="channel", values="resolution_minutes", aggfunc="mean", sort=True).reindex(columns=["Email", "Phone", "Chat"]).reset_index().to_csv(output / "mean_resolution_pivot.csv", index=False)
    return target

def score(root: Path) -> int:
    saved = os.environ.copy(); os.environ.update(ENV)
    try:
        score = grader.grade_submission(root)["score"]
        result = subprocess.run([sys.executable, "-B", "check_assignment.py"], cwd=root, text=True, capture_output=True)
        assert result.returncode == (0 if score == 90 else 1), result.stdout + result.stderr
        return score
    finally: os.environ.clear(); os.environ.update(saved)

def main() -> int:
    (ROOT.parents[1] / "scratch").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT.parents[1] / "scratch", prefix="a08-artifacts-") as temporary:
        work = Path(temporary); starter = copy_case(work, "starter", completed=False); assert score(starter) < 90
        accepted = copy_case(work, "accepted"); assert score(accepted) == 90
        portable = copy_case(work, "portable"); path = portable / "output/center_count_summary.csv"
        path.write_bytes(path.read_bytes().replace(b"Central", b'"Central"').replace(b"\n", b"\r\n")); assert score(portable) == 90
        shuffled = copy_case(work, "shuffled"); path = shuffled / "output/center_summary.csv"; rows = path.read_text().splitlines()
        path.write_text("\n".join([rows[0], *reversed(rows[1:])]) + "\n"); assert score(shuffled) == 90
        broken = copy_case(work, "broken"); path = broken / "output/center_count_summary.csv"
        path.write_text(path.read_text().replace("Central,5,4,3", "Central,999,4,3")); assert score(broken) == 70
    print("artifact regressions passed"); return 0

if __name__ == "__main__": raise SystemExit(main())
