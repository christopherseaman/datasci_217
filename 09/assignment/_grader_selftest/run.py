# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "ipykernel==6.29.5",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.0.2",
#   "pandas==3.0.5",
# ]
# ///

"""Author-side release harness for Assignment 09.

The harness materializes a disposable correct notebook from the released
starter, exercises real fresh-kernel entry points, refutes named mutations, and
tests the official Classroom50 success/failure/infrastructure contract.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

import classroom50_grader as grader


ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
RUNNER_ENV = {
    "CLASSROOM": "datasci-217-test",
    "ASSIGNMENT": "assignment-09",
    "SUBMISSION_TAG": "submission-test-001",
    "COMMIT_URL": "https://example.invalid/commit/a09",
    "RELEASE_URL": "https://example.invalid/release/a09",
    "REVIEW_URL": "https://example.invalid/review/a09",
}


CORRECT_SOURCES = {
    "a09-load": '''raw_readings = pd.read_csv(
    FIXTURE_PATH,
    dtype={"zone": "string", "recorded_at": "string", "co2_ppm": "float64"},
)
assert raw_readings.shape == (12, 3)
assert raw_readings.columns.tolist() == EXPECTED_COLUMNS
assert raw_readings["zone"].tolist() and set(raw_readings["zone"]) == {"atrium", "studio"}
assert raw_readings[["zone", "recorded_at"]].notna().all().all()
assert not raw_readings.duplicated(["zone", "recorded_at"]).any()
assert int(raw_readings["co2_ppm"].isna().sum()) == 1
raw_snapshot = raw_readings.copy(deep=True)''',
    "a09-task1-values": '''temporal_representation = "timestamp"
input_row_grain = "one recorded CO2 reading for one zone and local timestamp"
entity_key = ["zone"]
row_key = ["zone", "recorded_at"]
sort_keys = ["zone", "recorded_at"]
series_structure = "panel"
predicted_entities = ["atrium", "studio"]
source_timezone = "America/New_York"
output_timezone = "UTC"
predicted_source_rows = 12
predicted_gap_hours = {
    "atrium": [1.0, 2.0, 1.0, 2.0, 1.0],
    "studio": [2.0, 1.0, 2.0, 1.0, 1.0],
}
predicted_regularity = {"atrium": "irregular", "studio": "irregular"}''',
    "a09-prepare-function": '''def prepare_temporal_panel(reading_table, source_timezone):
    """Return a copied, UTC, entity-time-sorted prepared panel."""
    prepared = reading_table.copy(deep=True)
    parsed = pd.to_datetime(prepared["recorded_at"], format="%Y-%m-%d %H:%M")
    if parsed.dt.tz is not None:
        raise ValueError("recorded_at text must parse as naive local clock values")
    prepared["recorded_at"] = parsed.dt.tz_localize(source_timezone).dt.tz_convert("UTC")
    prepared["source_row"] = np.ones(len(prepared), dtype=np.int64)
    prepared = prepared.sort_values(["zone", "recorded_at"], kind="stable").reset_index(drop=True)
    if prepared[["zone", "recorded_at"]].isna().any().any():
        raise ValueError("entity/time keys must be nonmissing")
    if prepared.duplicated(["zone", "recorded_at"]).any():
        raise ValueError("entity/time keys must be unique")
    if not all(group["recorded_at"].is_monotonic_increasing for _, group in prepared.groupby("zone", observed=True, sort=True, dropna=True)):
        raise ValueError("time must increase within each zone")
    return prepared[["zone", "recorded_at", "co2_ppm", "source_row"]]''',
    "a09-task1-run": '''prepared_panel = prepare_temporal_panel(raw_readings, source_timezone)
assert prepared_panel.shape == (predicted_source_rows, 4)
assert prepared_panel.columns.tolist() == ["zone", "recorded_at", "co2_ppm", "source_row"]
assert prepared_panel["zone"].dtype == pd.StringDtype()
assert str(prepared_panel["recorded_at"].dtype) == "datetime64[us, UTC]"
assert prepared_panel["co2_ppm"].dtype == np.dtype("float64")
assert prepared_panel["source_row"].dtype == np.dtype("int64")
assert prepared_panel["zone"].drop_duplicates().tolist() == predicted_entities
assert not prepared_panel.duplicated(row_key).any()
gaps = {name: group["recorded_at"].diff().dropna().dt.total_seconds().div(3600).tolist() for name, group in prepared_panel.groupby("zone", observed=True, sort=True, dropna=True)}
assert gaps == predicted_gap_hours
assert {name: "irregular" if len(set(values)) > 1 else "regular" for name, values in gaps.items()} == predicted_regularity
assert prepared_panel["recorded_at"].duplicated().any()
pd.testing.assert_frame_equal(raw_readings, raw_snapshot)
indexed_panel = prepared_panel.set_index("recorded_at")
assert isinstance(indexed_panel.index, pd.DatetimeIndex) and str(indexed_panel.index.tz) == output_timezone
studio_series = indexed_panel.loc[indexed_panel["zone"].eq("studio")].copy()
assert studio_series["zone"].nunique() == 1 and len(indexed_panel) == 12''',
    "a09-task1-save": '''prepared_panel.to_csv(PREPARED_PATH, index=False, encoding="utf-8", lineterminator="\\n", na_rep="")
prepared_readback = pd.read_csv(PREPARED_PATH, dtype={"zone": "string", "co2_ppm": "float64", "source_row": "int64"}, parse_dates=["recorded_at"])
pd.testing.assert_frame_equal(prepared_readback, prepared_panel)
assert PREPARED_PATH.read_bytes().endswith(b"\\n") and b"\\r" not in PREPARED_PATH.read_bytes()''',
    "a09-hourly-function": '''def build_hourly_grid(prepared_table):
    """Return an entity-scoped exact-label hourly grid."""
    working = prepared_table.copy(deep=True)
    if not working["recorded_at"].eq(working["recorded_at"].dt.floor("h")).all():
        raise ValueError("Every source timestamp must fall on a whole UTC hour before asfreq.")
    hourly = (
        working.set_index("recorded_at")
        .groupby("zone", observed=True, sort=True, dropna=True)[["co2_ppm", "source_row"]]
        .resample("h")
        .asfreq()
        .reset_index()
    )
    hourly["grid_created_row"] = hourly["source_row"].isna()
    hourly["source_value_missing"] = hourly["source_row"].eq(1) & hourly["co2_ppm"].isna()
    return hourly[["zone", "recorded_at", "co2_ppm", "source_row", "grid_created_row", "source_value_missing"]]''',
    "a09-summary-function": '''def build_two_hour_summary(prepared_table):
    """Return left-closed, left-labeled zone–two-hour summaries."""
    working = prepared_table.copy(deep=True)
    return (
        working.set_index("recorded_at")
        .groupby("zone", observed=True, sort=True, dropna=True)
        .resample("2h", closed="left", label="left")
        .agg(mean_co2_ppm=("co2_ppm", "mean"), reading_count=("source_row", "sum"))
        .reset_index()
    )''',
    "a09-task2-run": '''hourly_grid = build_hourly_grid(prepared_panel)
two_hour_summary = build_two_hour_summary(prepared_panel)
assert hourly_grid.shape == (16, 6) and hourly_grid["zone"].drop_duplicates().tolist() == predicted_entities
assert str(hourly_grid["recorded_at"].dtype) == "datetime64[us, UTC]"
assert hourly_grid["source_row"].dtype == np.dtype("float64")
assert hourly_grid["grid_created_row"].dtype == np.dtype("bool") and hourly_grid["source_value_missing"].dtype == np.dtype("bool")
assert int(hourly_grid["grid_created_row"].sum()) == 4 and int(hourly_grid["source_value_missing"].sum()) == 1
assert not (hourly_grid["grid_created_row"] & hourly_grid["source_value_missing"]).any()
assert int(hourly_grid["source_row"].notna().sum()) == 12
assert two_hour_summary.shape == (8, 4) and two_hour_summary["reading_count"].dtype == np.dtype("int64")
assert two_hour_summary["reading_count"].tolist() == [2, 1, 1, 2, 1, 2, 1, 2]
assert int(two_hour_summary["reading_count"].sum()) == 12
assert pd.isna(two_hour_summary.loc[1, "mean_co2_ppm"]) and two_hour_summary.loc[3, "mean_co2_ppm"] == 490.0''',
    "a09-task2-save": '''hourly_grid.to_csv(HOURLY_PATH, index=False, encoding="utf-8", lineterminator="\\n", na_rep="")
two_hour_summary.to_csv(TWO_HOUR_PATH, index=False, encoding="utf-8", lineterminator="\\n", na_rep="")
hourly_readback = pd.read_csv(HOURLY_PATH, dtype={"zone": "string", "co2_ppm": "float64", "source_row": "float64", "grid_created_row": "bool", "source_value_missing": "bool"}, parse_dates=["recorded_at"])
summary_readback = pd.read_csv(TWO_HOUR_PATH, dtype={"zone": "string", "mean_co2_ppm": "float64", "reading_count": "int64"}, parse_dates=["recorded_at"])
pd.testing.assert_frame_equal(hourly_readback, hourly_grid)
pd.testing.assert_frame_equal(summary_readback, two_hour_summary)''',
    "a09-features-function": '''def build_past_features(prepared_table):
    """Return entity-scoped lag, difference, and two past-only means."""
    features = prepared_table[["zone", "recorded_at", "co2_ppm"]].copy(deep=True)
    by_zone = features.groupby("zone", observed=True, sort=True, dropna=True)["co2_ppm"]
    features["co2_lag_1"] = by_zone.shift(1)
    features["co2_difference"] = by_zone.diff()
    features["mean_previous_2_observations"] = features.groupby("zone", observed=True, sort=True, dropna=True)["co2_ppm"].transform(lambda values: values.shift(1).rolling(window=2, min_periods=1).mean())
    elapsed = (
        features.set_index("recorded_at")
        .groupby("zone", observed=True, sort=True, dropna=True)["co2_ppm"]
        .rolling("2h", closed="left", min_periods=1)
        .mean()
        .rename("mean_previous_2h")
        .reset_index()
    )
    return features.merge(elapsed, on=["zone", "recorded_at"], how="left", validate="one_to_one", sort=False)[["zone", "recorded_at", "co2_ppm", "co2_lag_1", "co2_difference", "mean_previous_2_observations", "mean_previous_2h"]]''',
    "a09-task3-features-run": '''temporal_features = build_past_features(prepared_panel)
assert temporal_features.shape == (12, 7)
assert temporal_features[["zone", "recorded_at", "co2_ppm"]].equals(prepared_panel[["zone", "recorded_at", "co2_ppm"]])
assert all(temporal_features[column].dtype == np.dtype("float64") for column in temporal_features.columns[2:])
first = temporal_features.groupby("zone", observed=True, sort=True, dropna=True).head(1)
assert first["co2_lag_1"].isna().all() and first["co2_difference"].isna().all()
studio_17 = temporal_features.loc[temporal_features["zone"].eq("studio") & temporal_features["recorded_at"].eq(pd.Timestamp("2026-01-20 17:00", tz="UTC"))].iloc[0]
studio_18 = temporal_features.loc[temporal_features["zone"].eq("studio") & temporal_features["recorded_at"].eq(pd.Timestamp("2026-01-20 18:00", tz="UTC"))].iloc[0]
assert [studio_17["co2_lag_1"], studio_17["co2_difference"], studio_17["mean_previous_2_observations"], studio_17["mean_previous_2h"]] == [540.0, 20.0, 530.0, 540.0]
assert [studio_18["co2_lag_1"], studio_18["co2_difference"], studio_18["mean_previous_2_observations"], studio_18["mean_previous_2h"]] == [560.0, 20.0, 550.0, 560.0]
pd.testing.assert_frame_equal(prepared_panel, prepare_temporal_panel(raw_readings, source_timezone))''',
    "a09-availability-values": '''prediction_zone = "studio"
prediction_timestamp = pd.Timestamp("2026-01-20 18:00", tz="UTC")
availability_decisions = pd.DataFrame({
    "candidate": pd.Series(["calendar hour", "previous recorded CO2", "centered three-observation mean", "next recorded CO2"], dtype="string"),
    "latest_required_timestamp": pd.to_datetime(["2026-01-20 18:00Z", "2026-01-20 17:00Z", "2026-01-20 19:00Z", "2026-01-20 19:00Z"], utc=True),
    "available_by_prediction_time": np.array([True, True, False, False], dtype=np.bool_),
    "decision": pd.Series(["keep", "keep", "reject", "reject"], dtype="string"),
})''',
    "a09-blocks-function": '''def build_chronological_blocks(prepared_table, holdout_start):
    """Label copied source rows as earlier or later_holdout."""
    blocks = prepared_table.copy(deep=True)
    blocks["block"] = pd.Series(np.where(blocks["recorded_at"].lt(holdout_start), "earlier", "later_holdout"), index=blocks.index, dtype="string")
    return blocks[["zone", "recorded_at", "co2_ppm", "source_row", "block"]]''',
    "a09-task3-run": '''chronological_blocks = build_chronological_blocks(prepared_panel, prediction_timestamp)
assert prediction_zone == "studio" and availability_decisions["decision"].tolist() == ["keep", "keep", "reject", "reject"]
assert chronological_blocks["block"].value_counts().to_dict() == {"earlier": 8, "later_holdout": 4}
earlier = chronological_blocks.loc[chronological_blocks["block"].eq("earlier")]
later = chronological_blocks.loc[chronological_blocks["block"].eq("later_holdout")]
assert earlier["recorded_at"].max() == pd.Timestamp("2026-01-20 17:00", tz="UTC")
assert later["recorded_at"].min() == pd.Timestamp("2026-01-20 18:00", tz="UTC")
assert earlier["recorded_at"].max() < later["recorded_at"].min()
assert set(earlier["zone"]) == {"atrium", "studio"} and set(later["zone"]) == {"atrium", "studio"}
pd.testing.assert_frame_equal(prepared_panel, prepare_temporal_panel(raw_readings, source_timezone))''',
    "a09-task3-save": '''temporal_features.to_csv(FEATURES_PATH, index=False, encoding="utf-8", lineterminator="\\n", na_rep="")
availability_decisions.to_csv(AVAILABILITY_PATH, index=False, encoding="utf-8", lineterminator="\\n", na_rep="")
chronological_blocks.to_csv(BLOCKS_PATH, index=False, encoding="utf-8", lineterminator="\\n", na_rep="")
features_readback = pd.read_csv(FEATURES_PATH, dtype={"zone": "string", "co2_ppm": "float64", "co2_lag_1": "float64", "co2_difference": "float64", "mean_previous_2_observations": "float64", "mean_previous_2h": "float64"}, parse_dates=["recorded_at"])
availability_readback = pd.read_csv(AVAILABILITY_PATH, dtype={"candidate": "string", "available_by_prediction_time": "bool", "decision": "string"}, parse_dates=["latest_required_timestamp"])
blocks_readback = pd.read_csv(BLOCKS_PATH, dtype={"zone": "string", "co2_ppm": "float64", "source_row": "int64", "block": "string"}, parse_dates=["recorded_at"])
pd.testing.assert_frame_equal(features_readback, temporal_features)
pd.testing.assert_frame_equal(availability_readback, availability_decisions)
pd.testing.assert_frame_equal(blocks_readback, chronological_blocks)''',
}

CORRECT_MARKDOWN = {
    "a09-task1-contract": "## Task 1 contract\nA timestamp is an instant and a period is a span. Each input row is one zone reading at one local timestamp; `zone` is the entity key and `zone,recorded_at` is the row key. This is an irregular panel. Localize New York clock text, convert to UTC, and sort by zone then time.",
    "a09-task1-explain": "### Task 1 explanation\nLocalization attaches New York to naive clock text; conversion expresses the same instant in UTC. Repeated UTC times are valid across separate zones. Timestamp-only sorting can interleave histories, while zone/time sorting preserves the entity boundary. Varying one- and two-hour gaps make both histories irregular.",
    "a09-task2-explain": "### Task 2 explanation\n`asfreq` conforms exact labels without combining; resample combines values inside explicit bins. A missing source value retains its source marker, while a grid-created row has none. CO2 is a state so its recorded values are averaged; source markers are additive. Left boundaries name `[left, right)` bins. Atrium's 14:00 bin has one recorded row whose CO2 is missing.",
    "a09-task3-explain": "### Task 3 explanation\nAt Studio 17:00 the previous two rows average 530, while only the 15:00 reading falls in `[15:00,17:00)`, giving 540. First rows have no same-entity past. Historical table presence does not prove prediction-time availability, so centered and next values requiring 19:00 are rejected without computation. The later block is only a temporal handoff; Lecture 10 defines evaluation roles.",
    "a09-synthesis": "## Synthesis\nExplicit grain, ordering, frequency meaning, and provenance preserve zone boundaries. Past-only calculations and latest-required timestamps prevent future information from entering candidate evidence. The small synthetic one-day fixture cannot establish behavior in other buildings, dates, or operating conditions.",
}


def _copy_starter(destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {"_grader_selftest", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", "result.json"}.intersection(names)
    shutil.copytree(ASSIGNMENT_DIR, destination, ignore=ignore)


def _notebook(root: Path) -> dict:
    return json.loads((root / "assignment.ipynb").read_text(encoding="utf-8"))


def _write_notebook(root: Path, notebook: dict) -> None:
    (root / "assignment.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _materialize_correct(root: Path) -> None:
    notebook = _notebook(root)
    for cell in notebook["cells"]:
        if cell["id"] in CORRECT_SOURCES:
            cell["source"] = CORRECT_SOURCES[cell["id"]]
        elif cell["id"] in CORRECT_MARKDOWN:
            cell["source"] = CORRECT_MARKDOWN[cell["id"]]
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    _write_notebook(root, notebook)


def _replace(root: Path, cell_id: str, old: str, new: str) -> None:
    notebook = _notebook(root)
    cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
    source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if old not in source:
        raise AssertionError(f"mutation target absent: {cell_id}: {old}")
    cell["source"] = source.replace(old, new, 1)
    _write_notebook(root, notebook)


@contextmanager
def _runner_context(include_review: bool = True):
    previous = {key: os.environ.get(key) for key in RUNNER_ENV}
    try:
        for key, value in RUNNER_ENV.items():
            if key != "REVIEW_URL" or include_review:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _execute_solution(root: Path) -> None:
    grader._execute(root, root)
    grader._check_artifacts(root)


def _execute_setup_only(root: Path) -> None:
    notebook = nbformat.read(root / "assignment.ipynb", as_version=4)
    setup = next(cell for cell in notebook.cells if cell.get("id") == "a09-setup")
    setup_notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(setup.source, id="a09-setup-only")],
        metadata=notebook.metadata,
    )
    NotebookClient(
        setup_notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
    ).execute()


def _verify_all_static(root: Path) -> None:
    _by_id, tree = grader._load_and_validate_template(root)
    grader._check_task1_source(tree)
    grader._check_task2_source(tree)
    grader._check_task3_source(tree)
    grader._check_artifacts(root)


def _mutate_notebook(root: Path, callback) -> None:
    notebook = _notebook(root)
    callback(notebook)
    _write_notebook(root, notebook)


def _run_mutants(correct: Path, temporary: Path) -> int:
    cases = []
    def add(label, mutation, verifier=_verify_all_static):
        cases.append((label, mutation, verifier))

    add("missing fixture", lambda root: (root / "data/zone_co2_readings.csv").unlink())
    add("renamed fixture", lambda root: (root / "data/zone_co2_readings.csv").rename(root / "data/readings.csv"))
    add("extra fixture", lambda root: (root / "data/extra.csv").write_text("x\n"))
    add("corrupt fixture", lambda root: (root / "data/zone_co2_readings.csv").write_bytes(b"broken\n"))
    add("CRLF fixture", lambda root: (root / "data/zone_co2_readings.csv").write_bytes((root / "data/zone_co2_readings.csv").read_bytes().replace(b"\n", b"\r\n")))
    add("edited manifest", lambda root: (root / "data/fixture.json").write_text((root / "data/fixture.json").read_text().replace("America/New_York", "UTC")))
    add("edited checker", lambda root: (root / "check_assignment.py").write_text("raise SystemExit(0)\n"))
    add("edited README", lambda root: (root / "README.md").write_text("changed\n"))
    add("edited platform guide", lambda root: (root / "PLATFORM_CHECK.md").write_text("changed\n"))
    add("edited requirements", lambda root: (root / "requirements.txt").write_text("pandas\n"))
    add("edited python version", lambda root: (root / ".python-version").write_text("3.11\n"))
    add("hidden outputs", lambda root: (root / ".gitignore").write_text((root / ".gitignore").read_text() + "output/\n"))
    add("malformed notebook", lambda root: (root / "assignment.ipynb").write_text("{bad"))
    add("missing cell", lambda root: _mutate_notebook(root, lambda nb: nb["cells"].pop(4)))
    add("duplicated cell ID", lambda root: _mutate_notebook(root, lambda nb: nb["cells"].__setitem__(4, {**nb["cells"][4], "id": nb["cells"][3]["id"]})))
    add("reordered cells", lambda root: _mutate_notebook(root, lambda nb: nb["cells"].__setitem__(slice(3, 5), list(reversed(nb["cells"][3:5])))))
    add("type-changed cell", lambda root: _mutate_notebook(root, lambda nb: nb["cells"][4].__setitem__("cell_type", "code")))
    add("edited protected cell", lambda root: _replace(root, "a09-task2-prompt", "Upsampling", "Grid expansion"))
    add("missing output", lambda root: (root / "output/prepared_panel.csv").unlink())
    add("stale output", lambda root: (root / "output/hourly_grid.csv").write_text("stale\n"))
    add("binary output", lambda root: (root / "output/two_hour_summary.csv").write_bytes(b"\x00\xff"))
    add("truncated output", lambda root: (root / "output/temporal_features.csv").write_bytes((root / "output/temporal_features.csv").read_bytes()[:30]))
    add("legacy output", lambda root: (root / "output/q1_report.csv").write_text("legacy\n"))
    add("foreign final output", lambda root: (root / "output/sentinel.txt").write_text("keep during setup\n"))
    add("parse without format", lambda root: _replace(root, "a09-prepare-function", ', format="%Y-%m-%d %H:%M"', ""))
    add("wrong sort key", lambda root: _replace(root, "a09-prepare-function", '["zone", "recorded_at"]', '["recorded_at"]'))
    add("unstable sort", lambda root: _replace(root, "a09-prepare-function", 'kind="stable"', 'kind="quicksort"'))
    add("missing input copy", lambda root: _replace(root, "a09-prepare-function", 'reading_table.copy(deep=True)', "reading_table"))
    add("canonical hard-code", lambda root: _replace(root, "a09-prepare-function", 'prepared = reading_table.copy(deep=True)', 'prepared = reading_table.loc[reading_table["zone"].isin(["atrium", "studio"])].copy(deep=True)'))
    add("global hourly grid", lambda root: _replace(root, "a09-hourly-function", '.groupby("zone", observed=True, sort=True, dropna=True)[["co2_ppm", "source_row"]]\n        ', ""))
    add("uppercase H", lambda root: _replace(root, "a09-hourly-function", '.resample("h")', '.resample("H")'))
    add("missing off-grid rejection", lambda root: _replace(root, "a09-hourly-function", '    if not working["recorded_at"].eq(working["recorded_at"].dt.floor("h")).all():\n        raise ValueError("Every source timestamp must fall on a whole UTC hour before asfreq.")\n', ""))
    add("filled hourly grid", lambda root: _replace(root, "a09-hourly-function", '.asfreq()', '.asfreq().ffill()'))
    add("wrong provenance", lambda root: _replace(root, "a09-hourly-function", 'hourly["source_row"].isna()', 'hourly["co2_ppm"].isna()'))
    add("wrong bin boundary", lambda root: _replace(root, "a09-summary-function", 'closed="left", label="left"', 'closed="right", label="right"'))
    add("wrong count operation", lambda root: _replace(root, "a09-summary-function", '("source_row", "sum")', '("source_row", "mean")'))
    add("pooled lag", lambda root: _replace(root, "a09-features-function", 'features.groupby("zone", observed=True, sort=True, dropna=True)["co2_ppm"]', 'features["co2_ppm"]', ))
    add("negative lag", lambda root: _replace(root, "a09-features-function", 'by_zone.shift(1)', 'by_zone.shift(-1)'))
    add("current observation window", lambda root: _replace(root, "a09-features-function", 'values.shift(1).rolling', 'values.rolling'))
    add("row window for elapsed", lambda root: _replace(root, "a09-features-function", '.rolling("2h", closed="left", min_periods=1)', '.rolling(2, min_periods=1)'))
    add("wrong elapsed boundary", lambda root: _replace(root, "a09-features-function", 'closed="left"', 'closed="right"'))
    add("unvalidated merge", lambda root: _replace(root, "a09-features-function", ', validate="one_to_one"', ""))
    add("centered window", lambda root: _replace(root, "a09-features-function", 'window=2, min_periods=1', 'window=2, min_periods=1, center=True'))
    add("EWM scope", lambda root: _replace(root, "a09-features-function", '.mean()\n        .rename("mean_previous_2h")', '.mean().ewm(span=2).mean()\n        .rename("mean_previous_2h")'))
    add("plotting scope", lambda root: _replace(root, "a09-task3-run", 'chronological_blocks =', 'import matplotlib.pyplot as plt\nchronological_blocks ='))
    add("modeling scope", lambda root: _replace(root, "a09-task3-run", 'chronological_blocks =', 'from sklearn.linear_model import LinearRegression\nchronological_blocks ='))
    add("network scope", lambda root: _replace(root, "a09-task3-run", 'chronological_blocks =', 'import requests\nchronological_blocks ='))
    add("random scope", lambda root: _replace(root, "a09-task3-run", 'chronological_blocks =', 'value = np.random.random()\nchronological_blocks ='))
    add("absolute path", lambda root: _replace(root, "a09-task3-run", 'chronological_blocks =', 'path = "/content/output.csv"\nchronological_blocks ='))
    add("mutable date", lambda root: _replace(root, "a09-task3-run", 'chronological_blocks =', 'now = pd.Timestamp.now()\nchronological_blocks ='))
    add("function file I/O", lambda root: _replace(root, "a09-blocks-function", '    blocks = prepared_table.copy(deep=True)', '    blocks = pd.read_csv("data.csv")'))

    rejected = 0
    for number, (label, mutation, verifier) in enumerate(cases, start=1):
        target = temporary / f"mutant-{number:02d}"
        shutil.copytree(correct, target)
        mutation(target)
        try:
            verifier(target)
        except Exception:
            rejected += 1
        else:
            raise AssertionError(f"Mutant was not rejected: {label}")
    print(f"Rejected adversarial mutants: {rejected}/{len(cases)}")
    return rejected


def _run_cli(grader_path: Path, target: Path, cwd: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(grader_path), str(target)], cwd=cwd, env=environment, text=True, capture_output=True, check=False)


def _assert_delivery_inventory(correct: Path, temporary: Path) -> None:
    accepted = temporary / "accepted-delivery"
    shutil.copytree(correct, accepted)
    (accepted / ".classroom50.yaml").write_text("version: 1\n", encoding="utf-8")
    workflow = accepted / ".github/workflows/autograde.yaml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text("name: autograde\n", encoding="utf-8")
    git_config = accepted / ".git/config"
    git_config.parent.mkdir(parents=True)
    git_config.write_text("[core]\n", encoding="utf-8")
    public = subprocess.run(
        [sys.executable, str(accepted / "check_assignment.py")],
        cwd=accepted,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy() | {"PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert public.returncode == 0, public.stdout + public.stderr
    with _runner_context():
        assert grader.grade_submission(accepted)["score"] == 90
        production_env = os.environ.copy() | {"PYTHONDONTWRITEBYTECODE": "1"}
    production = subprocess.run(
        [sys.executable, str(ASSIGNMENT_DIR / "_grader_selftest/autograder.py")],
        cwd=accepted,
        env=production_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert production.returncode == 0, production.stdout + production.stderr
    assert json.loads((accepted / "result.json").read_text())["score"] == 90
    (accepted / "result.json").unlink()

    for label, relative in (
        ("extra-root", "notes.txt"),
        ("extra-workflow", ".github/workflows/extra.yaml"),
        ("grader-tree", "_grader_selftest/copied.py"),
        ("nested-git", "ordinary/.git/nested.txt"),
    ):
        rejected = temporary / f"inventory-{label}"
        shutil.copytree(accepted, rejected)
        path = rejected / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unexpected\n", encoding="utf-8")
        public = subprocess.run(
            [sys.executable, str(rejected / "check_assignment.py")],
            cwd=rejected,
            text=True,
            capture_output=True,
            check=False,
            env=os.environ.copy() | {"PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert public.returncode == 1
        with _runner_context():
            assert grader.grade_submission(rejected)["score"] < 90


def _check_pep723() -> None:
    expected = ["ipykernel==6.29.5", "nbclient==0.10.2", "nbformat==5.10.4", "numpy==2.0.2", "pandas==3.0.5"]
    requirements = (ASSIGNMENT_DIR / "_grader_selftest/requirements.txt").read_text().splitlines()
    assert requirements == expected
    for path in (Path(__file__), ASSIGNMENT_DIR / "_grader_selftest/classroom50_grader.py"):
        module = ast.parse(path.read_text())
        assert module.body
        source = path.read_text()
        assert '# requires-python = "==3.12.13"' in source
        assert all(f'"{requirement}"' in source for requirement in expected)
    bootstrap = (ASSIGNMENT_DIR / "_grader_selftest/autograder.py").read_text()
    assert '# requires-python = "==3.12.13"' in bootstrap
    assert "# dependencies = []" in bootstrap
    public_source = (ASSIGNMENT_DIR / "check_assignment.py").read_text()
    assert '# requires-python = "==3.12.13"' in public_source
    assert '"numpy==2.0.2"' in public_source and '"pandas==3.0.5"' in public_source


def main() -> int:
    _check_pep723()
    with tempfile.TemporaryDirectory(prefix="a09-author-") as temporary_name:
        temporary = Path(temporary_name)
        starter = temporary / "starter"
        correct = temporary / "correct submission"
        _copy_starter(starter)
        _copy_starter(correct)
        _materialize_correct(correct)
        _execute_solution(correct)
        expected_artifacts = grader._artifact_bytes(correct)
        (correct / "output/hourly_grid.csv").write_bytes(b"\x00\xffcorrupt")
        (correct / "output/temporal_features.csv").unlink()
        _execute_solution(correct)
        assert grader._artifact_bytes(correct) == expected_artifacts
        sentinel = correct / "output/foreign-sentinel.txt"
        sentinel.write_text("preserve during setup\n", encoding="utf-8")
        _execute_setup_only(correct)
        assert sentinel.read_text(encoding="utf-8") == "preserve during setup\n"
        sentinel.unlink()
        _execute_solution(correct)
        _verify_all_static(correct)
        with _runner_context():
            correct_result = grader.grade_submission(correct)
            starter_result = grader.grade_submission(starter)
        assert correct_result["score"] == correct_result["max-score"] == 90
        assert starter_result["score"] < 90
        assert sum(test["max-score"] for test in correct_result["tests"]) == 90
        assert all(set(test) == {"test-name", "passed", "score", "max-score"} for test in correct_result["tests"])
        assert set(correct_result) == {"schema", "classroom", "assignment", "submission", "commit", "release", "review", "datetime", "score", "max-score", "tests"}
        assert correct_result["review"] == RUNNER_ENV["REVIEW_URL"]
        _assert_delivery_inventory(correct, temporary)

        stored_fake = temporary / "stored-output fake"
        shutil.copytree(correct, stored_fake)
        _replace(
            stored_fake,
            "a09-prepare-function",
            'parsed = pd.to_datetime(prepared["recorded_at"], format="%Y-%m-%d %H:%M")',
            'parsed = prepared["recorded_at"]',
        )
        with _runner_context():
            stored_fake_result = grader.grade_submission(stored_fake)
        assert stored_fake_result["score"] < 90

        correct_checker = subprocess.run(
            [sys.executable, "check_assignment.py"],
            cwd=correct,
            text=True,
            capture_output=True,
            check=False,
            env=os.environ.copy() | {"PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert correct_checker.returncode == 0
        assert correct_checker.stdout.splitlines() == [
            "All public checks passed. The notebook and six artifacts are ready for fresh central grading.",
            "The public checker does not award points or assess Markdown reasoning.",
        ]
        assert not correct_checker.stderr

        starter_checker = subprocess.run(
            [sys.executable, "check_assignment.py"],
            cwd=starter,
            text=True,
            capture_output=True,
            check=False,
            env=os.environ.copy() | {"PYTHONDONTWRITEBYTECODE": "1"},
        )
        expected_starter_lines = [
            "[FIX] notebook source and five functions: Complete every TODO in the notebook.",
            "[FIX] six generated artifacts: Keep exactly six required CSVs plus output/.gitkeep.",
        ]
        assert starter_checker.returncode == 1
        assert starter_checker.stdout.splitlines() == expected_starter_lines
        assert not starter_checker.stderr

        rejected = _run_mutants(correct, temporary)
        assert rejected >= 50

        corrected = temporary / "corrected resubmission"
        shutil.copytree(correct, corrected)
        _replace(corrected, "a09-summary-function", '("source_row", "sum")', '("source_row", "mean")')
        try:
            _verify_all_static(corrected)
        except Exception:
            pass
        else:
            raise AssertionError("broken resubmission was not rejected")
        _materialize_correct(corrected)
        _execute_solution(corrected)
        with _runner_context():
            corrected_result = grader.grade_submission(corrected)
        assert corrected_result["score"] == 90

        grader_path = ASSIGNMENT_DIR / "_grader_selftest/autograder.py"
        base_env = os.environ.copy() | RUNNER_ENV | {"PYTHONDONTWRITEBYTECODE": "1"}
        success_cwd = temporary / "cli-success"; success_cwd.mkdir()
        success = _run_cli(grader_path, correct, success_cwd, base_env)
        assert success.returncode == 0
        success_json = json.loads((success_cwd / "result.json").read_text())
        assert success_json["score"] == 90 and success_json["review"] == RUNNER_ENV["REVIEW_URL"]

        failure_cwd = temporary / "cli-student-failure"; failure_cwd.mkdir()
        failure = _run_cli(grader_path, starter, failure_cwd, base_env)
        assert failure.returncode == 0
        failure_json = json.loads((failure_cwd / "result.json").read_text())
        assert failure_json["score"] < 90 and any(not test["passed"] for test in failure_json["tests"])

        fallback_cwd = temporary / "cli-review-fallback"; fallback_cwd.mkdir()
        fallback_env = dict(base_env); fallback_env.pop("REVIEW_URL", None)
        fallback = _run_cli(grader_path, correct, fallback_cwd, fallback_env)
        assert fallback.returncode == 0
        assert json.loads((fallback_cwd / "result.json").read_text())["review"] == RUNNER_ENV["COMMIT_URL"]

        missing_cwd = temporary / "cli-missing-context"; missing_cwd.mkdir()
        missing_env = dict(base_env); missing_env.pop("CLASSROOM", None)
        missing = _run_cli(grader_path, correct, missing_cwd, missing_env)
        assert missing.returncode != 0 and not (missing_cwd / "result.json").exists()

        write_cwd = temporary / "cli-write-failure"; write_cwd.mkdir(); (write_cwd / "result.json").mkdir()
        write_failure = _run_cli(grader_path, correct, write_cwd, base_env)
        assert write_failure.returncode != 0 and not (write_cwd / "result.json").is_file()

        print("Correct score: 90/90")
        print(f"Starter score: {starter_result['score']}/90")
        print("Corrected resubmission score: 90/90")
        print("Correct public checker: exit 0 with exact readiness summary")
        print("Starter public checker: exit 1 with 2 exact actionable diagnostics")
        print("Canonical artifacts: 6/6 exact bytes and hashes")
        print("Deleted/binary/stale outputs regenerate; setup preserves a foreign sentinel")
        print("Correct-looking stored artifacts with broken source score below full")
        print("Alternate functions: 5/5 with off-grid rejection")
        print("Path layouts: flattened/direct, spaces, nested, course-root, relocated")
        print("Official CLI: success=0, student-failure=0, fallback=0, missing-context/nonwrite nonzero")
        print("Assignment 09 adversarial release harness passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
