# /// script
# requires-python = ">=3.13,<3.14"
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

"""Artifact regression checks for Assignment 10's public and central graders."""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path
import tempfile

import grader


ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS = {'availability_decisions.csv': 'candidate_feature,latest_required_offset_hours,available_by_prediction_time,decision\n'
                               'batch_sequence,0,True,keep\n'
                               'ambient_temp_c,0,True,keep\n'
                               'pre_mix_moisture_pct,0,True,keep\n'
                               'early_24h_strength_mpa,24,False,exclude\n'
                               'next_day_strength_mpa,24,False,exclude\n',
 'binary_metrics.csv': 'approach,accuracy,precision,recall\n'
                       'supplied_model,0.833333,0.666667,0.666667\n'
                       'dummy_baseline,0.750000,0.000000,0.000000\n',
 'final_predictions.csv': 'batch_id,target_timestamp,actual_strength_mpa,predicted_strength_mpa\n'
                          'B038,2026-05-09T00:00:00Z,35.985000,36.619379\n'
                          'B039,2026-05-10T00:00:00Z,36.810000,36.938898\n'
                          'B040,2026-05-11T00:00:00Z,37.331000,37.153664\n'
                          'B041,2026-05-12T00:00:00Z,36.924000,37.249329\n'
                          'B042,2026-05-13T00:00:00Z,37.497000,37.213505\n'
                          'B043,2026-05-14T00:00:00Z,36.581000,37.045450\n'
                          'B044,2026-05-15T00:00:00Z,36.709000,36.749396\n'
                          'B045,2026-05-16T00:00:00Z,36.403000,36.348188\n'
                          'B046,2026-05-17T00:00:00Z,35.299000,35.868423\n'
                          'B047,2026-05-18T00:00:00Z,35.278000,35.350631\n'
                          'B048,2026-05-19T00:00:00Z,35.014000,34.842601\n',
 'final_test_metrics.csv': 'approach,mae,rmse,r2\nlinear_pipeline,0.265686,0.332477,0.830552\n',
 'inference_case_intervals.csv': 'mix_minutes,initial_temp_c,predicted_mean,mean_ci_low_95,mean_ci_high_95,prediction_ci_low_95,prediction_ci_high_95\n'
                                 '26.000000,22.000000,84.741704,84.376661,85.106747,83.154332,86.329076\n',
 'inference_summary.csv': 'term,estimate,standard_error,confidence_low_95,confidence_high_95\n'
                          'Intercept,51.959310,1.715679,48.302426,55.616194\n'
                          'mix_minutes,0.651471,0.021679,0.605262,0.697679\n'
                          'initial_temp_c,0.720189,0.070929,0.569008,0.871370\n',
 'split_manifest.csv': 'partition,row_count,first_target_timestamp,last_target_timestamp\n'
                       'train,29,2026-04-02T00:00:00Z,2026-04-30T00:00:00Z\n'
                       'validation,8,2026-05-01T00:00:00Z,2026-05-08T00:00:00Z\n'
                       'test,11,2026-05-09T00:00:00Z,2026-05-19T00:00:00Z\n',
 'validation_metrics.csv': 'approach,mae,rmse,r2\n'
                           'mean_baseline,4.259573,4.504803,-8.441848\n'
                           'linear_pipeline,0.255929,0.312760,0.954488\n'}
RESIDUALS = """run_id,actual,fitted,residual
M01,74.150000,73.820642,0.329358
M02,78.000000,78.809750,-0.809750
M03,83.450000,82.649855,0.800145
M04,87.400000,88.359152,-0.959152
M05,79.100000,78.364434,0.735566
M06,82.200000,82.856010,-0.656010
M07,83.950000,83.523984,0.426016
M08,92.750000,92.765507,-0.015507
M09,77.050000,77.146714,-0.096714
M10,84.450000,85.239235,-0.789235
M11,87.550000,86.987492,0.562508
M12,90.700000,90.398785,0.301215
M13,80.400000,80.832880,-0.432880
M14,89.950000,88.273930,1.676070
M15,89.900000,90.382282,-0.482282
M16,92.300000,92.421915,-0.121915
M17,82.400000,82.702071,-0.302071
M18,83.650000,83.815360,-0.165360
"""
TINY_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mNk+M/wHwAF/gL+z8lA9QAAAABJRU5ErkJggg==")


def fresh_export(destination: Path) -> Path:
    destination.mkdir(parents=True)
    output = destination / "output"
    output.mkdir()
    for name, content in ARTIFACTS.items():
        (output / name).write_text(content, encoding="utf-8")
    (output / "inference_residuals.png").write_bytes(TINY_PNG)
    (destination / "output" / "inference_residuals.csv").write_text(RESIDUALS, encoding="utf-8", newline="\n")
    return destination


def central(root: Path) -> dict:
    return grader.grade_submission(root)


def quote_and_reverse(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\r\n").writerows([rows[0], *reversed(rows[1:])])


def replace_value(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def main() -> int:
    scratch = ASSIGNMENT_DIR.parents[1] / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        root = fresh_export(Path(temporary) / "fresh export")
        result = central(root)
        assert [test["score"] for test in result["tests"]] == [30, 35, 30, 5], result
        (root / "unrelated.txt").write_text("allowed\n", encoding="utf-8")
        assert central(root)["score"] == 100
        quote_and_reverse(root / "output" / "inference_summary.csv")
        quote_and_reverse(root / "output" / "final_predictions.csv")
        (root / "output" / "inference_residuals.png").write_bytes(TINY_PNG)
        result = central(root)
        assert [test["score"] for test in result["tests"]] == [30, 35, 30, 5], result
        replace_value(root / "output" / "inference_summary.csv", "51.959310", "99.000000")
        result = central(root)
        assert [test["score"] for test in result["tests"]] == [0, 35, 30, 5], result
        root = fresh_export(Path(temporary) / "missing artifact")
        replace_value(root / "output" / "validation_metrics.csv", "4.259573", "99.000000")
        result = central(root)
        assert [test["score"] for test in result["tests"]] == [30, 35, 0, 5], result
        root = fresh_export(Path(temporary) / "missing artifact second")
        (root / "output" / "validation_metrics.csv").unlink()
        result = central(root)
        assert [test["score"] for test in result["tests"]] == [30, 35, 0, 5], result
        root = fresh_export(Path(temporary) / "wrong predictions")
        replace_value(root / "output" / "final_predictions.csv", "36.619379", "99.000000")
        result = central(root)
        assert [test["score"] for test in result["tests"]] == [30, 35, 0, 5], result
    print(json.dumps({"artifact_regression": "pass", "cases": 7}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
