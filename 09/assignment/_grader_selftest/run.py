"""Artifact-only regression checks for Assignment 09."""
from __future__ import annotations
from pathlib import Path
import tempfile
import grader
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def copy_case(work: Path, name: str, *, completed: bool = True) -> Path:
    target = work / name
    target.mkdir(); output = target / "output"; output.mkdir()
    if completed:
        source = grader._fixture()
        source.to_csv(output / "prepared_panel.csv", index=False)
        hourly = source.set_index("recorded_at").groupby("zone")[["co2_ppm", "source_row"]].resample("h").asfreq().reset_index()
        hourly["grid_created_row"] = hourly["source_row"].isna(); hourly["source_value_missing"] = hourly["source_row"].eq(1) & hourly["co2_ppm"].isna()
        hourly.to_csv(output / "hourly_grid.csv", index=False)
        source.set_index("recorded_at").groupby("zone").resample("2h", closed="left", label="left").agg(mean_co2_ppm=("co2_ppm", "mean"), reading_count=("source_row", "sum")).reset_index().to_csv(output / "two_hour_summary.csv", index=False)
        features = source[["zone", "recorded_at", "co2_ppm"]].copy(); group = features.groupby("zone")["co2_ppm"]
        features["co2_lag_1"] = group.shift(); features["co2_difference"] = group.diff(); features["mean_previous_2_observations"] = group.transform(lambda values: values.shift().rolling(2, min_periods=1).mean())
        elapsed = features.set_index("recorded_at").groupby("zone")["co2_ppm"].rolling("2h", closed="left", min_periods=1).mean().rename("mean_previous_2h").reset_index()
        features.merge(elapsed, on=["zone", "recorded_at"], validate="one_to_one", sort=False).to_csv(output / "temporal_features.csv", index=False)
        pd.DataFrame({"candidate": ["calendar hour", "previous recorded CO2", "centered three-observation mean", "next recorded CO2"], "latest_required_timestamp": pd.to_datetime(["2026-01-20 18:00Z", "2026-01-20 17:00Z", "2026-01-20 19:00Z", "2026-01-20 19:00Z"], utc=True), "available_by_prediction_time": [True, True, False, False], "decision": ["keep", "keep", "reject", "reject"]}).to_csv(output / "availability_decisions.csv", index=False)
        blocks = source.copy(); blocks["block"] = np.where(blocks["recorded_at"] < pd.Timestamp("2026-01-20 18:00", tz="UTC"), "earlier", "later_holdout"); blocks.to_csv(output / "chronological_blocks.csv", index=False)
    return target

def score(root: Path) -> int:
    return grader.grade_submission(root)["score"]

def main() -> int:
    (ROOT.parents[1] / "scratch").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT.parents[1] / "scratch", prefix="a09-artifacts-") as temporary:
        work = Path(temporary); starter = copy_case(work, "starter", completed=False); assert score(starter) == 0
        accepted = copy_case(work, "accepted"); assert score(accepted) == 100
        (accepted / "unrelated.txt").write_text("allowed\n"); assert score(accepted) == 100
        portable = copy_case(work, "portable"); path = portable / "output/availability_decisions.csv"
        path.write_bytes(path.read_bytes().replace(b"calendar hour", b'"calendar hour"').replace(b"\n", b"\r\n")); assert score(portable) == 100
        broken = copy_case(work, "broken"); path = broken / "output/chronological_blocks.csv"; rows = path.read_text().splitlines()
        path.write_text("\n".join([rows[0], *reversed(rows[1:])]) + "\n"); assert score(broken) == 65
    print("artifact regressions passed"); return 0

if __name__ == "__main__": raise SystemExit(main())
