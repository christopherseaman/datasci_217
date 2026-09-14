# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#   "ipykernel==6.29.5",
#   "matplotlib==3.11.1",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.3.3",
#   "pandas==3.0.5",
#   "Pillow==12.3.0",
#   "scikit-learn==1.9.0",
#   "statsmodels==0.14.6",
# ]
# ///

"""Instructor-controlled artifact-first grader for Assignment 10.

Production grading reads committed CSV/PNG artifacts directly; notebook
execution and alternate-input checks remain optional release QA.
"""

from __future__ import annotations

import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys


FIXTURES = {
    "data/fixture.json": "aa50eeffc2b07c5d98cb56a0e3d18115909958f777899d5d403cf6323dd1de41",
    "data/mixing_runs.csv": "00b8a1ce84110f4a7fa85620742283c82a4b9d600dbe0ebea0d4721956938957",
    "data/batch_strength.csv": "f14faf7da64347dfc255aa84b14e79eef7f2d0de94b394c747323319d937baa3",
    "data/feature_availability.csv": "a47b8df048607045640b9a6785b038fe1c70036f58d5b61ed20ec98860b556da",
    "data/supplied_binary_predictions.csv": "7a8809010fa94345cd04787c826ef86ee5fd13cbf0bd95953e2220c3294a239a",
}
PROTECTED_FILE_SHA256 = {
    "README.md": "708992f0fe369b56897ed578e3667d32d11dcba884f7adbc7c8e6e31406b6123",
    "PLATFORM_CHECK.md": "384047ae73eeced17d91de75e883173998be1b882f334336a65aeb27e072d1c3",
    ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95",
    ".python-version": "a876e0b10411037a012498b9fe18d9bc1df32ed8b722a13564dc944ddcfd9135",
    "requirements.txt": "740a377ce40a7c62f5c544b0873b224a071d50effb7484a2b9bef6b36f5e0fe3",
}
BASE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "grading.py", "requirements.txt", *FIXTURES,
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
}
POINTS = [10, 25, 30, 30, 5]
TEST_NAMES = [
    "Submission package and fixture integrity",
    "Task 1 bounded inference and intervals",
    "Task 2 contract, availability, leakage, chronological split",
    "Task 3 train-only comparison, freeze, final test, binary metrics",
    "Residual figure PNG",
]


class InfrastructureError(RuntimeError):
    pass


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _resolve_root(start: Path) -> Path:
    start = start.resolve()
    for base in (start, *start.parents):
        for candidate in (base, base / "10" / "assignment"):
            if (candidate / "assignment.ipynb").is_file() and (candidate / "data/fixture.json").is_file():
                return candidate.resolve()
    raise InfrastructureError("cannot locate the complete Assignment 10 learner package")


def _inventory(root: Path) -> None:
    git_entry = root / ".git"
    if git_entry.exists() or git_entry.is_symlink():
        _assert(git_entry.is_dir() and not git_entry.is_symlink(), "top-level .git must be a genuine directory")
    output_entry = root / "output"
    _assert(output_entry.is_dir() and not output_entry.is_symlink(), "output must be a genuine directory")
    _assert(not (root / "_grader_selftest").exists() and not (root / "_grader_selftest").is_symlink(), "instructor bundle entered learner package")
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] in {".git", "output"}:
            continue
        if path.is_file() or path.is_symlink():
            actual.add(relative.as_posix())
    expected = BASE_FILES
    _assert(expected <= actual, f"required learner package files are missing: {sorted(expected - actual)}")
    _assert(not any((root / relative).is_symlink() for relative in expected), "required learner package contains a symlink")
    for relative, digest in FIXTURES.items():
        _assert(sha256((root / relative).read_bytes()).hexdigest() == digest,
                f"restore immutable fixture {relative}")
    for relative, digest in PROTECTED_FILE_SHA256.items():
        path = root / relative
        _assert(path.is_file() and not path.is_symlink() and sha256(path.read_bytes()).hexdigest() == digest,
                f"restore immutable course file {relative}")


ARTIFACT_COLUMNS = {
    "inference_summary.csv": ["term", "estimate", "standard_error", "confidence_low_95", "confidence_high_95"],
    "inference_case_intervals.csv": ["mix_minutes", "initial_temp_c", "predicted_mean", "mean_ci_low_95", "mean_ci_high_95", "prediction_ci_low_95", "prediction_ci_high_95"],
    "inference_residuals.csv": ["run_id", "actual", "fitted", "residual"],
    "availability_decisions.csv": ["candidate_feature", "latest_required_offset_hours", "available_by_prediction_time", "decision"],
    "split_manifest.csv": ["partition", "row_count", "first_target_timestamp", "last_target_timestamp"],
    "validation_metrics.csv": ["approach", "mae", "rmse", "r2"],
    "final_test_metrics.csv": ["approach", "mae", "rmse", "r2"],
    "final_predictions.csv": ["batch_id", "target_timestamp", "actual_strength_mpa", "predicted_strength_mpa"],
    "binary_metrics.csv": ["approach", "accuracy", "precision", "recall"],
}


FLOAT_TOLERANCE = 1e-4


def issue(surface: str, message: str, errors: list[str]) -> None:
    errors.append(f"[FIX] {surface}: {message}")


def _csv_rows(path: Path, columns: list[str], errors: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != columns:
                issue("output", f"{path.name} columns must be {columns}", errors)
                return []
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        issue("output", f"cannot read {path.name}: {error}", errors)
        return []
    if any(set(row) != set(columns) or any(value is None or value == "" for value in row.values()) for row in rows):
        issue("output", f"{path.name} has missing values", errors)
        return []
    return rows


def _number(row: dict[str, str], column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"{column} must be finite")
    return value


def _table_values(name: str, rows: list[dict[str, str]], key: str, expected: dict[str, dict[str, object]], errors: list[str]) -> None:
    values = [row.get(key) for row in rows]
    if len(values) != len(set(values)) or set(values) != set(expected):
        issue("output", f"{name} has the wrong {key} values", errors)
        return
    try:
        for row in rows:
            for column, want in expected[row[key]].items():
                got = _number(row, column) if isinstance(want, float) else row[column]
                if isinstance(want, float):
                    if abs(got - want) > FLOAT_TOLERANCE:
                        raise ValueError(f"{row[key]} {column}")
                elif got != want:
                    raise ValueError(f"{row[key]} {column}")
    except (KeyError, ValueError) as error:
        issue("output", f"{name} has an invalid value ({error})", errors)


def output_checks(root: Path, errors: list[str]) -> None:
    output = root / "output"
    if not output.is_dir() or output.is_symlink():
        for name in (*ARTIFACT_COLUMNS, "inference_residuals.png"):
            issue("output", f"{name} needs a regular output directory", errors)
        return
    actual = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    expected = {".gitkeep", *ARTIFACT_COLUMNS, "inference_residuals.png"}
    if not expected <= actual:
        issue("output", f"required output artifacts are missing: {sorted(expected - actual)}", errors)
    tables = {}
    for name, columns in ARTIFACT_COLUMNS.items():
        path = output / name
        if not path.is_file() or path.is_symlink():
            continue
        tables[name] = _csv_rows(path, columns, errors)
    _table_values("inference_summary.csv", tables.get("inference_summary.csv", []), "term", {
        "Intercept": {"estimate": 51.959310, "standard_error": 1.715679, "confidence_low_95": 48.302426, "confidence_high_95": 55.616194},
        "mix_minutes": {"estimate": 0.651471, "standard_error": 0.021679, "confidence_low_95": 0.605262, "confidence_high_95": 0.697679},
        "initial_temp_c": {"estimate": 0.720189, "standard_error": 0.070929, "confidence_low_95": 0.569008, "confidence_high_95": 0.871370},
    }, errors)
    case_rows = tables.get("inference_case_intervals.csv", [])
    try:
        expected_case = [26.0, 22.0, 84.741704, 84.376661, 85.106747, 83.154332, 86.329076]
        if len(case_rows) != 1 or any(abs(_number(case_rows[0], column) - want) > FLOAT_TOLERANCE for column, want in zip(ARTIFACT_COLUMNS["inference_case_intervals.csv"], expected_case)):
            raise ValueError("expected one supplied case")
    except (KeyError, ValueError):
        issue("output", "inference_case_intervals.csv has invalid case values", errors)
    _table_values("inference_residuals.csv", tables.get("inference_residuals.csv", []), "run_id", {
        "M01": {"actual": 74.15, "fitted": 73.820642, "residual": .329358}, "M02": {"actual": 78., "fitted": 78.809750, "residual": -.809750}, "M03": {"actual": 83.45, "fitted": 82.649855, "residual": .800145},
        "M04": {"actual": 87.4, "fitted": 88.359152, "residual": -.959152}, "M05": {"actual": 79.1, "fitted": 78.364434, "residual": .735566}, "M06": {"actual": 82.2, "fitted": 82.856010, "residual": -.656010},
        "M07": {"actual": 83.95, "fitted": 83.523984, "residual": .426016}, "M08": {"actual": 92.75, "fitted": 92.765507, "residual": -.015507}, "M09": {"actual": 77.05, "fitted": 77.146714, "residual": -.096714},
        "M10": {"actual": 84.45, "fitted": 85.239235, "residual": -.789235}, "M11": {"actual": 87.55, "fitted": 86.987492, "residual": .562508}, "M12": {"actual": 90.7, "fitted": 90.398785, "residual": .301215},
        "M13": {"actual": 80.4, "fitted": 80.832880, "residual": -.432880}, "M14": {"actual": 89.95, "fitted": 88.273930, "residual": 1.676070}, "M15": {"actual": 89.9, "fitted": 90.382282, "residual": -.482282},
        "M16": {"actual": 92.3, "fitted": 92.421915, "residual": -.121915}, "M17": {"actual": 82.4, "fitted": 82.702071, "residual": -.302071}, "M18": {"actual": 83.65, "fitted": 83.815360, "residual": -.165360},
    }, errors)
    _table_values("availability_decisions.csv", tables.get("availability_decisions.csv", []), "candidate_feature", {
        "batch_sequence": {"latest_required_offset_hours": "0", "available_by_prediction_time": "True", "decision": "keep"},
        "ambient_temp_c": {"latest_required_offset_hours": "0", "available_by_prediction_time": "True", "decision": "keep"},
        "pre_mix_moisture_pct": {"latest_required_offset_hours": "0", "available_by_prediction_time": "True", "decision": "keep"},
        "early_24h_strength_mpa": {"latest_required_offset_hours": "24", "available_by_prediction_time": "False", "decision": "exclude"},
        "next_day_strength_mpa": {"latest_required_offset_hours": "24", "available_by_prediction_time": "False", "decision": "exclude"},
    }, errors)
    _table_values("split_manifest.csv", tables.get("split_manifest.csv", []), "partition", {
        "train": {"row_count": "29", "first_target_timestamp": "2026-04-02T00:00:00Z", "last_target_timestamp": "2026-04-30T00:00:00Z"},
        "validation": {"row_count": "8", "first_target_timestamp": "2026-05-01T00:00:00Z", "last_target_timestamp": "2026-05-08T00:00:00Z"},
        "test": {"row_count": "11", "first_target_timestamp": "2026-05-09T00:00:00Z", "last_target_timestamp": "2026-05-19T00:00:00Z"},
    }, errors)
    _table_values("validation_metrics.csv", tables.get("validation_metrics.csv", []), "approach", {
        "mean_baseline": {"mae": 4.259573, "rmse": 4.504803, "r2": -8.441848}, "linear_pipeline": {"mae": .255929, "rmse": .312760, "r2": .954488},
    }, errors)
    _table_values("final_test_metrics.csv", tables.get("final_test_metrics.csv", []), "approach", {"linear_pipeline": {"mae": .265686, "rmse": .332477, "r2": .830552}}, errors)
    _table_values("binary_metrics.csv", tables.get("binary_metrics.csv", []), "approach", {
        "supplied_model": {"accuracy": .833333, "precision": .666667, "recall": .666667}, "dummy_baseline": {"accuracy": .75, "precision": 0., "recall": 0.},
    }, errors)
    predictions = tables.get("final_predictions.csv", [])
    expected_ids = {f"B{number:03d}" for number in range(38, 49)}
    if len(predictions) != 11 or {row.get("batch_id") for row in predictions} != expected_ids:
        issue("output", "final_predictions.csv must align one row to each chronological test batch", errors)
    else:
        try:
            source_rows = {row["batch_id"]: row for row in csv.DictReader((root / "data" / "batch_strength.csv").open(encoding="utf-8"))}
            for row in predictions:
                source_row = source_rows[row["batch_id"]]
                if row["target_timestamp"] != source_row["target_timestamp"] or abs(_number(row, "actual_strength_mpa") - float(source_row["next_day_strength_mpa"])) > FLOAT_TOLERANCE or not math.isfinite(_number(row, "predicted_strength_mpa")):
                    raise ValueError(row["batch_id"])
        except (KeyError, ValueError, OSError, UnicodeError, csv.Error):
            issue("output", "final_predictions.csv has misaligned IDs, timestamps, or values", errors)
    _table_values("final_predictions.csv", predictions, "batch_id", {
        f"B{number:03d}": {"predicted_strength_mpa": value}
        for number, value in enumerate([
            36.619379, 36.938898, 37.153664, 37.249329, 37.213505,
            37.045450, 36.749396, 36.348188, 35.868423, 35.350631,
            34.842601,
        ], start=38)
    }, errors)
    png = output / "inference_residuals.png"
    if png.is_symlink():
        issue("output", "inference_residuals.png must be a regular file", errors)
    if png.is_file() and not png.is_symlink():
        if not png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            issue("output", "inference_residuals.png must be a PNG file", errors)


def _artifact_errors(root: Path) -> list[str]:
    errors: list[str] = []
    output_checks(root, errors)
    return errors


def grade_root(root: Path) -> tuple[list[dict], dict]:
    diagnostics = {"instructor_environment": True, "artifact_only": True, "fresh_runs": 0}
    try:
        _inventory(root)
        template_test = {"name": TEST_NAMES[0], "score": POINTS[0], "max-score": POINTS[0], "output": "candidate package inventory passes"}
    except Exception as error:
        template_test = {"name": TEST_NAMES[0], "score": 0, "max-score": POINTS[0], "output": str(error)}
    errors = _artifact_errors(root)
    groups = [
        (POINTS[1], {"inference_summary.csv", "inference_case_intervals.csv", "inference_residuals.csv"}),
        (POINTS[2], {"availability_decisions.csv", "split_manifest.csv"}),
        (POINTS[3], {"validation_metrics.csv", "final_test_metrics.csv", "final_predictions.csv", "binary_metrics.csv"}),
        (POINTS[4], {"inference_residuals.png"}),
    ]
    tests = [template_test]
    for name, (points, artifacts) in zip(TEST_NAMES[1:], groups):
        relevant = [error for error in errors if any(artifact in error for artifact in artifacts)]
        tests.append({"name": name, "score": 0 if relevant else points, "max-score": points, "output": "; ".join(relevant) if relevant else "committed artifact values pass"})
    return tests, diagnostics



def grade_submission(submission_root: str | Path) -> dict:
    """Grade committed artifacts only; submitted code is never imported or run."""
    root = Path(submission_root).resolve()
    if not root.is_dir():
        raise InfrastructureError(f"submission root is not a directory: {root}")
    rows, _ = grade_root(root)
    tests = [{"test-name": row["name"], "passed": row["score"] == row["max-score"],
              "score": row["score"], "max-score": row["max-score"],
              "detail": row["output"]} for row in rows]
    return {"schema": "datasci217/grading-result/v1",
            "score": sum(test["score"] for test in tests), "max-score": 100,
            "tests": tests}


def main(argv: list[str] | None = None) -> int:
    """Run the one public artifact grader."""
    import argparse
    from contextlib import redirect_stdout

    parser = argparse.ArgumentParser(description="Grade saved assignment artifacts.")
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr if args.as_json else sys.stdout):
            result = grade_submission(args.submission_dir)
    except Exception as error:
        if args.as_json:
            print(json.dumps({"schema": "datasci217/grading-result/v1",
                              "error": f"{type(error).__name__}: {error}"}))
        else:
            print(f"[INFRASTRUCTURE] {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    else:
        for test in result["tests"]:
            print(f"[{'PASS' if test['passed'] else 'FIX'}] {test['test-name']} "
                  f"({test['score']}/{test['max-score']})"
                  + (f": {test['detail']}" if test.get("detail") else ""))
        print(f"Automated score: {result['score']}/{result['max-score']}")
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
