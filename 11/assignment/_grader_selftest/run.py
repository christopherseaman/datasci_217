"""Author-only end-to-end self-test for the Assignment 11 graders."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np
import pandas as pd

import grader


ASSIGNMENT = Path(__file__).resolve().parents[1]
ENVIRONMENT = {
    "ASSIGNMENT": "assignment-11",
    "SUBMISSION_TAG": "submission-test-001", "COMMIT_URL": "https://example.invalid/commit/a11",
    "RELEASE_URL": "https://example.invalid/release/a11", "REVIEW_URL": "https://example.invalid/review/a11",
}
DOCUMENTED_ARTIFACT_SCHEMAS = {
    "q1_release_audit.csv": ["check_name", "expected", "observed", "passed"],
    "q1_station_coverage.csv": ["station_name", "expected_hours", "observed_hours", "missing_hours", "coverage_pct", "first_timestamp", "last_timestamp"],
    "q2_cleaned_observations.csv": [*grader.RAW_COLUMNS, "measurement_timestamp_utc"],
    "q2_cleaning_audit.csv": ["rule", "affected_values", "result"],
    "q2_missingness.csv": ["station_name", "column_name", "missing_count", "missing_pct"],
    "q3_hourly_panel.csv": ["station_name", "measurement_timestamp_utc", *grader.SENSOR_COLUMNS, "source_observed", "hour", "day_of_week", "month"],
    "q3_panel_summary.csv": ["station_name", "expected_hours", "observed_hours", "missing_hours", "gap_runs", "longest_gap_hours"],
    "q4_features.csv": ["row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc", grader.TARGET, "model_eligible", *grader.NUMERIC_FEATURES],
    "q4_feature_manifest.csv": ["feature_name", "source", "earliest_offset_hours", "latest_offset_hours", "role"],
    "q5_monthly_station_summary.csv": ["station_name", "year", "month", "n_observed", "mean_air_temperature_c", "std_air_temperature_c", "min_air_temperature_c", "max_air_temperature_c"],
    "q5_correlations.csv": ["feature", *grader.CORRELATION_FEATURES],
    **{f"q6_X_{split}.csv": ["row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc", *grader.NUMERIC_FEATURES] for split in ["train", "validation", "test"]},
    **{f"q6_y_{split}.csv": ["row_id", grader.TARGET] for split in ["train", "validation", "test"]},
    "q6_split_summary.csv": ["split", "n_rows", "target_start", "target_end", "n_features"],
    "q7_model_spec.csv": ["estimator_module", "estimator_class", "parameters_json", "feature_columns", "random_state"],
    "q7_validation_predictions.csv": ["row_id", "station_name", "target_timestamp_utc", "actual", "persistence_prediction", "model_prediction"],
    "q7_validation_metrics.csv": ["model", "mae", "rmse", "r2", "n"],
    "q7_permutation_importance.csv": ["feature", "mean_mae_increase", "std_mae_increase"],
    "q8_test_predictions.csv": ["row_id", "station_name", "target_timestamp_utc", "actual", "persistence_prediction", "model_prediction", "model_error", "model_absolute_error"],
    "q8_test_metrics.csv": ["model", "mae", "rmse", "r2", "n"],
    "q8_station_metrics.csv": ["model", "station_name", "n", "mae", "rmse", "r2"],
}


@contextmanager
def _context(include_review: bool = True):
    previous = {name: os.environ.get(name) for name in ENVIRONMENT}
    try:
        for name, value in ENVIRONMENT.items():
            if name == "REVIEW_URL" and not include_review:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _new_submission(path: Path) -> None:
    (path / "data").mkdir(parents=True)
    (path / "output").mkdir()
    for name in [grader.RELEASE_NAME, grader.MANIFEST_NAME]:
        shutil.copy2(ASSIGNMENT / "data" / name, path / "data" / name)
    shutil.copy2(ASSIGNMENT / "check_assignment.py", path / "check_assignment.py")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _png() -> bytes:
    header = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    pixels = b"\x00\x20\x60\xa0\x20\x60\xa0\x00\xa0\x60\x20\xa0\x60\x20"
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(pixels)) + _chunk(b"IEND", b"")


def _report(validation: pd.DataFrame, test: pd.DataFrame) -> str:
    rows = []
    for label, metrics in [("Validation", validation), ("Test", test)]:
        rows.extend(f"| {label} | {row.model} | {row.mae:.6f} | {row.rmse:.6f} | {row.r2:.6f} | {int(row.n)} |" for row in metrics.itertuples())
    metric_rows = "\n".join(rows)
    return f"""# Chicago Beach Weather Next-Hour Forecast

## Executive Summary
The artifacts evaluate deterministic next-hour air-temperature forecasts for both released Chicago beach stations.

## Data and Cleaning
The immutable release was checked against its manifest. Six ambiguous fall-back local timestamps were rejected, invalid measurements became missing, and no values were filled or interpolated.

![Release overview](output/q1_visualizations.png)

## Patterns
Monthly summaries and correlations were calculated from training target times only.

![Training patterns](output/q5_patterns.png)

## Forecast Design
Each row predicts the next elapsed UTC hour from information available at the current hour. The artifact checks cannot prove that fitting and pattern exploration used only training rows; that training provenance requires source or execution review.

## Model Results
The persistence baseline uses current air temperature. The student-model values below are evaluated on the held-out test rows.

| Evaluation set | Model | MAE | RMSE | R2 | n |
|---|---|---:|---:|---:|---:|
{metric_rows}

![Final results](output/q8_final_visualizations.png)

## Limitations
Artifact-only grading cannot establish model-fitting provenance or prove that exploratory decisions excluded later labels. Sensor outages and retained measurement missingness also limit the eligible forecast population.
"""


def _references(root: Path) -> dict[str, pd.DataFrame]:
    return grader._references(
        str((root / "data" / grader.RELEASE_NAME).resolve()),
        str((root / "data" / grader.MANIFEST_NAME).resolve()),
    )


def _materialize(root: Path) -> None:
    refs = _references(root)
    artifacts = {
        "q1_release_audit.csv": refs["audit"], "q1_station_coverage.csv": refs["coverage"],
        "q2_cleaned_observations.csv": refs["clean"], "q2_cleaning_audit.csv": refs["cleaning_audit"],
        "q2_missingness.csv": refs["missingness"], "q3_hourly_panel.csv": refs["panel"],
        "q3_panel_summary.csv": refs["panel_summary"], "q4_features.csv": refs["model"],
        "q4_feature_manifest.csv": refs["feature_manifest"], "q5_monthly_station_summary.csv": refs["monthly"],
        "q5_correlations.csv": refs["correlations"], "q6_split_summary.csv": refs["split_summary"],
    }
    for split in ["train", "validation", "test"]:
        artifacts[f"q6_X_{split}.csv"] = refs[f"x_{split}"]
        artifacts[f"q6_y_{split}.csv"] = refs[f"y_{split}"]
    for name, frame in artifacts.items():
        _write_csv(frame, root / "output" / name)

    parameters = {"constant": None, "quantile": None, "strategy": "mean"}
    spec = pd.DataFrame([{
        "estimator_module": "sklearn.dummy", "estimator_class": "DummyRegressor",
        "parameters_json": json.dumps(parameters, sort_keys=True),
        "feature_columns": "|".join(grader.FEATURES), "random_state": 217,
    }])
    _write_csv(spec, root / "output/q7_model_spec.csv")
    validation_prediction, importance_mean, importance_std = grader._declared_model_results(root, refs, "validation")
    importance = pd.DataFrame({
        "feature": grader.FEATURES, "mean_mae_increase": importance_mean,
        "std_mae_increase": importance_std,
    })
    _write_csv(importance, root / "output/q7_permutation_importance.csv")

    result_metrics = {}
    for split, prefix, test in [("validation", "q7_validation", False), ("test", "q8_test", True)]:
        predictions = refs[f"x_{split}"][["row_id", "station_name", "target_timestamp_utc", "air_temperature_c_t"]].merge(refs[f"y_{split}"], on="row_id", validate="one_to_one").rename(columns={grader.TARGET: "actual", "air_temperature_c_t": "persistence_prediction"})
        model_prediction = validation_prediction if split == "validation" else grader._declared_model_results(root, refs, "test")[0]
        predictions["model_prediction"] = model_prediction
        predictions = predictions[["row_id", "station_name", "target_timestamp_utc", "actual", "persistence_prediction", "model_prediction"]]
        if test:
            predictions["model_error"] = predictions["model_prediction"] - predictions["actual"]
            predictions["model_absolute_error"] = predictions["model_error"].abs()
        _write_csv(predictions, root / "output" / f"{prefix}_predictions.csv")
        metrics = grader._metrics(predictions)
        _write_csv(metrics, root / "output" / f"{prefix}_metrics.csv")
        result_metrics[split] = metrics
        if test:
            station_rows = []
            for model, column in [("persistence_baseline", "persistence_prediction"), ("student_model", "model_prediction")]:
                for station, group in predictions.groupby("station_name", sort=True, observed=True):
                    residual = group[column] - group["actual"]; denominator = float(((group["actual"] - group["actual"].mean()) ** 2).sum())
                    station_rows.append({"model": model, "station_name": station, "n": len(group), "mae": residual.abs().mean(), "rmse": np.sqrt((residual ** 2).mean()), "r2": 1 - float((residual ** 2).sum()) / denominator})
            _write_csv(pd.DataFrame(station_rows), root / "output/q8_station_metrics.csv")
    for name in ["q1_visualizations.png", "q5_patterns.png", "q8_final_visualizations.png"]:
        (root / "output" / name).write_bytes(_png())
    (root / "report.md").write_text(_report(result_metrics["validation"], result_metrics["test"]), encoding="utf-8")


def _score(root: Path) -> int:
    with _context():
        return grader.grade_submission(root)["score"]


def _public_score(root: Path) -> int:
    result = subprocess.run([sys.executable, str(root / "check_assignment.py")], cwd=root, text=True, capture_output=True, check=False)
    final_line = result.stdout.splitlines()[-1]
    assert final_line.startswith("Readiness score: "), result.stdout + result.stderr
    return int(final_line.removeprefix("Readiness score: ").removesuffix("/100"))


def _replace_temporarily(path: Path, replacement: bytes | None, operation) -> None:
    original = path.read_bytes()
    if replacement is None:
        path.unlink()
    else:
        path.write_bytes(replacement)
    try:
        operation()
    finally:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(original)


def _isolated_mutations(correct: Path) -> None:
    nonfinite = pd.read_csv(correct / "output/q7_validation_predictions.csv")
    nonfinite.loc[0, "model_prediction"] = np.inf
    cases = [
        ("Q1", "output/q1_release_audit.csv", b"wrong\n", 92),
        ("Q2", "output/q2_cleaned_observations.csv", None, 14),
        ("Q3", "output/q3_hourly_panel.csv", b"\x00\xff", 24),
        ("Q4", "output/q4_feature_manifest.csv", b"wrong\n", 36),
        ("Q5", "output/q5_monthly_station_summary.csv", b"wrong\n", 92),
        ("Q6", "output/q6_split_summary.csv", b"wrong\n", 60),
        ("Q7", "output/q7_validation_predictions.csv", nonfinite.to_csv(index=False, lineterminator="\n").encode(), 72),
        ("Q8", "output/q8_station_metrics.csv", b"wrong\n", 86),
        ("Q9", "report.md", b"# incomplete\n", 94),
    ]
    for label, relative, replacement, expected in cases:
        def verify(label=label, expected=expected):
            actual = _score(correct)
            assert actual == expected, f"{label}: expected {expected}, got {actual}"
            public_actual = _public_score(correct)
            assert public_actual == expected, f"public {label}: expected {expected}, got {public_actual}"
        _replace_temporarily(correct / relative, replacement, verify)
    print("Isolated rubric mutations rejected: 9/9")


def _extra_failures(correct: Path) -> None:
    validation_path = correct / "output/q7_validation_predictions.csv"
    validation = pd.read_csv(validation_path)
    negative = validation.copy(); negative.loc[0, "model_prediction"] = -1.0
    negative_metrics = grader._metrics(negative)
    metric_path = correct / "output/q7_validation_metrics.csv"
    def verify_prediction_mismatch() -> None:
        updated_report = _report(negative_metrics, pd.read_csv(correct / "output/q8_test_metrics.csv"))
        def verify_metrics_and_report() -> None:
            _replace_temporarily(correct / "report.md", updated_report.encode(), lambda: _assert_score(correct, 72))
        _replace_temporarily(metric_path, negative_metrics.to_csv(index=False, lineterminator="\n").encode(), verify_metrics_and_report)
    _replace_temporarily(validation_path, negative.to_csv(index=False, lineterminator="\n").encode(), verify_prediction_mismatch)
    nonfinite = validation.copy(); nonfinite.loc[0, "model_prediction"] = -np.inf
    _replace_temporarily(validation_path, nonfinite.to_csv(index=False, lineterminator="\n").encode(), lambda: (_assert_score(correct, 72)))
    metrics = pd.read_csv(metric_path); metrics.loc[0, "mae"] += 1
    _replace_temporarily(metric_path, metrics.to_csv(index=False, lineterminator="\n").encode(), lambda: _assert_score(correct, 66))
    png_path = correct / "output/q5_patterns.png"
    malformed = bytearray(png_path.read_bytes()); malformed[-1] ^= 1
    _replace_temporarily(png_path, bytes(malformed), lambda: _assert_score(correct, 86))

    audit_path = correct / "output/q2_cleaning_audit.csv"
    audit = pd.read_csv(audit_path); audit["rule"] = [f"alternate decision {index}" for index in range(len(audit))]
    manifest_path = correct / "output/q4_feature_manifest.csv"
    manifest = pd.read_csv(manifest_path); manifest["source"] = [f"alternate source {index}" for index in range(len(manifest))]
    def verify_alternate_wording() -> None:
        _replace_temporarily(manifest_path, manifest.to_csv(index=False, lineterminator="\n").encode(), lambda: (_assert_score(correct, 100), _assert_public_score(correct, 100)))
    _replace_temporarily(audit_path, audit.to_csv(index=False, lineterminator="\n").encode(), verify_alternate_wording)

    report_path = correct / "report.md"
    report_lines = report_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(report_lines):
        if line.startswith("| Validation | persistence_baseline |"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            cells[2] = "999"
            report_lines[index] = "| " + " | ".join(cells) + " |"
            break
    fabricated = "\n".join(report_lines) + "\n"
    _replace_temporarily(report_path, fabricated.encode(), lambda: (_assert_score(correct, 94), _assert_public_score(correct, 94)))

    spec_path = correct / "output/q7_model_spec.csv"
    spec = pd.read_csv(spec_path); spec.loc[0, "estimator_class"] = "StandardScaler"
    spec.loc[0, "estimator_module"] = "sklearn.preprocessing"
    _replace_temporarily(spec_path, spec.to_csv(index=False, lineterminator="\n").encode(), lambda: (_assert_score(correct, 72), _assert_public_score(correct, 72)))

    mismatch = pd.read_csv(spec_path)
    parameters = json.loads(mismatch.loc[0, "parameters_json"]); parameters["strategy"] = "median"
    mismatch.loc[0, "parameters_json"] = json.dumps(parameters, sort_keys=True)
    _replace_temporarily(spec_path, mismatch.to_csv(index=False, lineterminator="\n").encode(), lambda: _assert_score(correct, 72))


def _assert_score(root: Path, expected: int) -> None:
    actual = _score(root)
    assert actual == expected, f"expected score {expected}, got {actual}"


def _assert_public_score(root: Path, expected: int) -> None:
    actual = _public_score(root)
    assert actual == expected, f"expected public score {expected}, got {actual}"


def _run_cli(script: Path, target: Path, cwd: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(script), str(target)], cwd=cwd, env=environment, text=True, capture_output=True, check=False)


def main() -> int:
    assert sys.version_info[:3] == (3, 12, 13)
    assert (np.__version__, pd.__version__) == ("2.0.2", "3.0.5")
    assert (ASSIGNMENT / "_grader_selftest/requirements.txt").read_text(encoding="utf-8").splitlines() == ["numpy==2.0.2", "pandas==3.0.5", "scikit-learn==1.9.0"]
    public_source = (ASSIGNMENT / "check_assignment.py").read_text(encoding="utf-8")
    central_source = (ASSIGNMENT / "_grader_selftest/grader.py").read_text(encoding="utf-8")
    harness_source = Path(__file__).read_text(encoding="utf-8")
    contract_source = (ASSIGNMENT / "assignment.md").read_text(encoding="utf-8")
    assert "_grader_selftest" not in public_source and "grader.py" not in public_source
    assert all(name not in public_source for name in ["_references", "_reference", "_gap_summary"])
    stale_names = ["q4_" + "forecast_features.csv", "q5_" + "monthly_summary.csv", "q5_" + "correlation_matrix.csv"]
    assert all(name not in source for name in stale_names for source in [public_source, central_source, harness_source])
    for filename in grader.DOCUMENTED_FILENAMES:
        assert filename in contract_source and filename in public_source and filename in central_source
    with tempfile.TemporaryDirectory(prefix="a11-author-") as temporary_name:
        temporary = Path(temporary_name)
        correct, empty = temporary / "correct submission", temporary / "empty submission"
        _new_submission(correct); _new_submission(empty); _materialize(correct)
        assert not (correct / "_grader_selftest").exists() and not (empty / "_grader_selftest").exists()
        assert not any((correct / "output" / name).exists() for name in stale_names)
        for filename, columns in DOCUMENTED_ARTIFACT_SCHEMAS.items():
            assert pd.read_csv(correct / "output" / filename).columns.tolist() == columns, filename
        with _context():
            result = grader.grade_submission(correct)
            empty_result = grader.grade_submission(empty)
        assert result["score"] == result["max-score"] == 100 and empty_result["score"] == 0
        assert [test["max-score"] for test in result["tests"]] == grader.POINTS
        assert len(result["tests"]) == 9 and all(set(test) == {"test-name", "passed", "score", "max-score"} for test in result["tests"])
        assert set(result) == {"schema", "assignment", "submission", "commit", "release", "review", "datetime", "score", "max-score", "tests"}
        assert result["schema"] == "datasci217/grading-result/v1" and result["score"] == sum(test["score"] for test in result["tests"])
        assert result["review"] == ENVIRONMENT["REVIEW_URL"]
        dt_value = pd.Timestamp(result["datetime"]); assert dt_value.tzinfo is not None and str(dt_value.tzinfo) == "UTC"
        with _context(include_review=False):
            assert grader.grade_submission(correct)["review"] == ENVIRONMENT["COMMIT_URL"]

        public = subprocess.run([sys.executable, str(correct / "check_assignment.py")], cwd=correct, text=True, capture_output=True, check=False)
        assert public.returncode == 0 and public.stdout.splitlines()[-1] == "Readiness score: 100/100", public.stdout + public.stderr
        assert not (correct / "result.json").exists()
        empty_public = subprocess.run([sys.executable, str(empty / "check_assignment.py")], cwd=empty, text=True, capture_output=True, check=False)
        assert empty_public.returncode == 1 and empty_public.stdout.splitlines()[-1] == "Readiness score: 0/100"
        assert not (empty / "result.json").exists()

        _isolated_mutations(correct)
        _extra_failures(correct)
        (correct / "output/q4_features.csv").write_text("broken\n", encoding="utf-8")
        _assert_score(correct, 36)
        _materialize(correct)
        _assert_score(correct, 100)

        script = ASSIGNMENT / "_grader_selftest/autograder.py"
        environment = os.environ.copy() | ENVIRONMENT | {"PYTHONDONTWRITEBYTECODE": "1"}
        success_cwd = temporary / "bootstrap-success"; success_cwd.mkdir()
        success = _run_cli(script, correct, success_cwd, environment)
        assert success.returncode == 0, success.stdout + success.stderr
        assert json.loads((success_cwd / "result.json").read_text(encoding="utf-8"))["score"] == 100
        learner_cwd = temporary / "learner-failure"; learner_cwd.mkdir()
        learner = _run_cli(script, empty, learner_cwd, environment)
        assert learner.returncode == 0 and json.loads((learner_cwd / "result.json").read_text(encoding="utf-8"))["score"] == 0
        context_cwd = temporary / "context-failure"; context_cwd.mkdir()
        missing_environment = dict(environment); missing_environment.pop("ASSIGNMENT", None)
        missing = _run_cli(script, correct, context_cwd, missing_environment)
        assert missing.returncode == 2 and not (context_cwd / "result.json").exists()
        bad_bundle = temporary / "bad bundle"; shutil.copytree(ASSIGNMENT / "_grader_selftest", bad_bundle)
        (bad_bundle / "requirements.txt").write_text("not-a-real-package-a11==0\n", encoding="utf-8")
        install_cwd = temporary / "install-failure"; install_cwd.mkdir()
        install = _run_cli(bad_bundle / "autograder.py", correct, install_cwd, environment)
        assert install.returncode == 2 and not (install_cwd / "result.json").exists()

        print("Correct fixture: public 100/100; trusted grader 100/100")
        print("Empty fixture: public 0/100; trusted grader 0/100")
        print("Blocking, malformed CSV/PNG/report, prediction, metric, and resubmission checks passed")
        print("Result schema, context/review fallback, and bootstrap install/failure checks passed")
        print("Assignment 11 Chicago Beach Weather artifact grader self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
