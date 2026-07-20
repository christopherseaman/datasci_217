# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "ipykernel==6.29.5",
#   "matplotlib==3.11.1",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.0.2",
#   "pandas==3.0.3",
#   "Pillow==12.3.0",
#   "scikit-learn==1.9.0",
#   "statsmodels==0.14.6",
# ]
# ///

"""Author-side candidate harness for Assignment 10.

The correct solution exists only in disposable directories. This harness uses
the direct candidate environment; it cannot certify the absent release lock.
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

sys.dont_write_bytecode = True

import classroom50_grader as grader


ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
RUNNER_ENV = {
    "CLASSROOM": "datasci-217-test",
    "ASSIGNMENT": "assignment-10",
    "SUBMISSION_TAG": "submission-test-001",
    "COMMIT_URL": "https://example.invalid/commit/a10",
    "RELEASE_URL": "https://example.invalid/release/a10",
    "REVIEW_URL": "https://example.invalid/review/a10",
}


CORRECT_SOURCES = {
"a10-load": '''mixing_runs = pd.read_csv(
    DATA_DIR / "mixing_runs.csv",
    dtype={"run_id": "string", "mix_minutes": "int64", "initial_temp_c": "float64", "finish_quality_score": "float64"},
)
batch_strength = pd.read_csv(
    DATA_DIR / "batch_strength.csv",
    dtype={"batch_id": "string", "prediction_timestamp": "string", "target_timestamp": "string", "batch_sequence": "int64", "ambient_temp_c": "float64", "pre_mix_moisture_pct": "float64", "next_day_strength_mpa": "float64"},
)
for timestamp_column in ["prediction_timestamp", "target_timestamp"]:
    batch_strength[timestamp_column] = pd.to_datetime(batch_strength[timestamp_column], utc=True)
feature_availability = pd.read_csv(DATA_DIR / "feature_availability.csv", dtype={"candidate_feature": "string", "latest_required_offset_hours": "int64"})
supplied_binary = pd.read_csv(DATA_DIR / "supplied_binary_predictions.csv", dtype={"case_id": "string", "actual_label": "int64", "supplied_model_prediction": "int64", "dummy_prediction": "int64"})
assert mixing_runs.columns.tolist() == ["run_id", "mix_minutes", "initial_temp_c", "finish_quality_score"] and len(mixing_runs) == 18
assert batch_strength.columns.tolist() == ["batch_id", "prediction_timestamp", "target_timestamp", "batch_sequence", "ambient_temp_c", "pre_mix_moisture_pct", "next_day_strength_mpa"] and len(batch_strength) == 48
assert feature_availability.columns.tolist() == ["candidate_feature", "latest_required_offset_hours"] and len(feature_availability) == 5
assert supplied_binary.columns.tolist() == ["case_id", "actual_label", "supplied_model_prediction", "dummy_prediction"] and len(supplied_binary) == 12
assert mixing_runs["run_id"].tolist() == [f"M{row:02d}" for row in range(1, 19)]
assert batch_strength["batch_id"].tolist() == [f"B{row:03d}" for row in range(1, 49)]
assert batch_strength["batch_sequence"].tolist() == list(range(1, 49))
assert (batch_strength["prediction_timestamp"] < batch_strength["target_timestamp"]).all()
assert supplied_binary[["actual_label", "supplied_model_prediction", "dummy_prediction"]].isin([0, 1]).all().all()
for table in [mixing_runs, batch_strength, feature_availability, supplied_binary]:
    assert isinstance(table.index, pd.RangeIndex) and table.index.start == 0 and table.index.step == 1
mixing_snapshot = mixing_runs.copy(deep=True)
batch_snapshot = batch_strength.copy(deep=True)
availability_snapshot = feature_availability.copy(deep=True)
binary_snapshot = supplied_binary.copy(deep=True)
display(mixing_runs.head(), batch_strength.head(), feature_availability, supplied_binary)''',
"a10-ols-function": '''def fit_bounded_ols(inference_table, predictor_columns, outcome_column):
    if not isinstance(inference_table, pd.DataFrame):
        raise TypeError("inference_table must be a pandas DataFrame")
    if not isinstance(predictor_columns, (list, tuple)) or any(not isinstance(name, str) for name in predictor_columns):
        raise TypeError("predictor_columns must be an ordered list or tuple of names")
    if not isinstance(outcome_column, str):
        raise TypeError("outcome_column must be a column name")
    if len(predictor_columns) != 2 or len(set(predictor_columns)) != 2 or outcome_column in predictor_columns:
        raise ValueError("supply exactly two distinct predictors and a distinct outcome")
    names = [outcome_column, *predictor_columns]
    if any(not name.isidentifier() for name in names):
        raise ValueError("model column names must be valid Python identifiers")
    missing = [name for name in names if name not in inference_table]
    if missing:
        raise KeyError(missing)
    if len(inference_table) < 4:
        raise ValueError("at least four rows are required")
    model_table = inference_table[names].copy(deep=True)
    try:
        model_table = model_table.astype("float64")
    except (TypeError, ValueError) as error:
        raise ValueError("model values must be numeric") from error
    if not np.isfinite(model_table.to_numpy()).all():
        raise ValueError("model values must be finite and nonmissing")
    design = np.column_stack([np.ones(len(model_table)), model_table[list(predictor_columns)].to_numpy()])
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("intercept-plus-predictor design must have full rank")
    formula = outcome_column + " ~ " + " + ".join(predictor_columns)
    result = smf.ols(formula=formula, data=model_table).fit()
    if result.params.index.tolist() != ["Intercept", *predictor_columns]:
        raise ValueError("formula term order differs")
    return result''',
"a10-task1-run": '''ols_model = fit_bounded_ols(mixing_runs, ["mix_minutes", "initial_temp_c"], "finish_quality_score")
confidence = ols_model.conf_int(alpha=0.05)
inference_summary = pd.DataFrame({
    "term": pd.Series(ols_model.params.index, dtype="string"),
    "estimate": ols_model.params.to_numpy(dtype="float64"),
    "standard_error": ols_model.bse.to_numpy(dtype="float64"),
    "confidence_low_95": confidence.iloc[:, 0].to_numpy(dtype="float64"),
    "confidence_high_95": confidence.iloc[:, 1].to_numpy(dtype="float64"),
})
new_case = pd.DataFrame([MANIFEST["inference_new_case"]], dtype="float64")
prediction_frame = ols_model.get_prediction(new_case).summary_frame(alpha=0.05)
inference_case_intervals = pd.DataFrame({
    "mix_minutes": [new_case.loc[0, "mix_minutes"]],
    "initial_temp_c": [new_case.loc[0, "initial_temp_c"]],
    "predicted_mean": [prediction_frame.loc[0, "mean"]],
    "mean_ci_low_95": [prediction_frame.loc[0, "mean_ci_lower"]],
    "mean_ci_high_95": [prediction_frame.loc[0, "mean_ci_upper"]],
    "prediction_ci_low_95": [prediction_frame.loc[0, "obs_ci_lower"]],
    "prediction_ci_high_95": [prediction_frame.loc[0, "obs_ci_upper"]],
})
assert inference_summary["term"].tolist() == ["Intercept", "mix_minutes", "initial_temp_c"]
assert inference_summary.shape == (3, 5) and np.isfinite(inference_summary.iloc[:, 1:].to_numpy()).all()
assert inference_case_intervals.shape == (1, 7) and np.isfinite(inference_case_intervals.to_numpy()).all()
assert inference_case_intervals.loc[0, "prediction_ci_low_95"] < inference_case_intervals.loc[0, "mean_ci_low_95"] < inference_case_intervals.loc[0, "predicted_mean"] < inference_case_intervals.loc[0, "mean_ci_high_95"] < inference_case_intervals.loc[0, "prediction_ci_high_95"]
pd.testing.assert_frame_equal(mixing_runs, mixing_snapshot)
display(inference_summary, inference_case_intervals)''',
"a10-residual-figure": '''fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=100)
ax.scatter(ols_model.fittedvalues, ols_model.resid)
ax.axhline(0.0, color="black", linewidth=1.0)
ax.set(title="Residuals versus fitted values", xlabel="Fitted finish quality score", ylabel="Residual")
fig.tight_layout()
fig.savefig(RESIDUAL_FIGURE_PATH, dpi=100, metadata={"Software": "datasci_217"})
display(fig)
plt.close(fig)''',
"a10-task1-save": '''inference_summary.to_csv(INFERENCE_SUMMARY_PATH, index=False, encoding="utf-8", lineterminator="\\n", float_format="%.6f")
inference_case_intervals.to_csv(INFERENCE_CASE_PATH, index=False, encoding="utf-8", lineterminator="\\n", float_format="%.6f")
inference_summary_readback = pd.read_csv(INFERENCE_SUMMARY_PATH, dtype={"term": "string", "estimate": "float64", "standard_error": "float64", "confidence_low_95": "float64", "confidence_high_95": "float64"})
inference_case_readback = pd.read_csv(INFERENCE_CASE_PATH, dtype={column: "float64" for column in inference_case_intervals.columns})
assert inference_summary_readback.shape == (3, 5) and inference_case_readback.shape == (1, 7)
assert INFERENCE_SUMMARY_PATH.read_bytes().endswith(b"\\n") and b"\\r" not in INFERENCE_SUMMARY_PATH.read_bytes()
assert INFERENCE_CASE_PATH.read_bytes().endswith(b"\\n") and b"\\r" not in INFERENCE_CASE_PATH.read_bytes()
display(INFERENCE_SUMMARY_PATH, INFERENCE_CASE_PATH, RESIDUAL_FIGURE_PATH)''',
"a10-task1-explain": '''## Task 1 explanation

Holding initial temperature fixed, a one-minute difference in mixing time is associated with about a 0.651-point difference in expected finish quality. This is a conditional association, not a causal effect. Its 95% confidence interval describes uncertainty in that population coefficient under the model assumptions; it does not say that 95% of individual outcomes lie inside it. At the supplied predictors, the individual prediction interval is wider than the mean-response interval because it includes both uncertainty in the mean and individual outcome variation. A residuals-versus-fitted plot can reveal curvature or changing spread, but it cannot establish causation. These synthetic observational associations do not show that changing mixing time causes finish quality to change.''',
"a10-task2-contract": '''## Prediction contract

The question is: for one synthetic batch, using inputs available at its `prediction_timestamp`, what will its `next_day_strength_mpa` be at `target_timestamp`? The prediction unit is one synthetic batch. Prediction time is when the batch inputs must be available. The target is next-day strength, and target time is when that later strength is observed.''',
"a10-task2-values": '''prediction_contract = {
    "unit": MANIFEST["prediction_contract"]["unit"],
    "prediction_time": MANIFEST["prediction_contract"]["prediction_time"],
    "target": MANIFEST["prediction_contract"]["target"],
    "target_time": MANIFEST["prediction_contract"]["target_time"],
}
feature_columns = list(MANIFEST["prediction_contract"]["feature_columns"])
validation_start = pd.Timestamp(MANIFEST["split_boundaries"]["validation_start"])
test_start = pd.Timestamp(MANIFEST["split_boundaries"]["test_start"])
assert prediction_contract == {"unit": "one synthetic batch", "prediction_time": "prediction_timestamp", "target": "next_day_strength_mpa", "target_time": "target_timestamp"}
assert feature_columns == ["batch_sequence", "ambient_temp_c", "pre_mix_moisture_pct"]
display(prediction_contract, feature_columns, validation_start, test_start)''',
"a10-availability-function": '''def audit_feature_availability(candidate_table):
    if not isinstance(candidate_table, pd.DataFrame):
        raise TypeError("candidate_table must be a pandas DataFrame")
    required = ["candidate_feature", "latest_required_offset_hours"]
    missing = [name for name in required if name not in candidate_table]
    if missing:
        raise KeyError(missing)
    result = candidate_table[required].copy(deep=True)
    names = result["candidate_feature"]
    if names.isna().any() or names.astype("string").str.strip().eq("").any() or names.duplicated().any():
        raise ValueError("candidate features must be unique, nonmissing, and nonblank")
    offsets = pd.to_numeric(result["latest_required_offset_hours"], errors="raise")
    if offsets.isna().any() or not np.isfinite(offsets.to_numpy(dtype="float64")).all() or not np.equal(offsets.to_numpy(dtype="float64"), offsets.to_numpy(dtype="int64")).all():
        raise ValueError("availability offsets must be exact int64 values")
    result["candidate_feature"] = names.astype("string")
    result["latest_required_offset_hours"] = offsets.astype("int64")
    result["available_by_prediction_time"] = result["latest_required_offset_hours"].le(0).to_numpy(dtype="bool")
    result["decision"] = pd.Series(np.where(result["available_by_prediction_time"], "keep", "exclude"), index=result.index, dtype="string")
    return result''',
"a10-split-function": '''def build_chronological_splits(prediction_table, validation_start, test_start):
    if not isinstance(prediction_table, pd.DataFrame):
        raise TypeError("prediction_table must be a pandas DataFrame")
    required = ["batch_id", "prediction_timestamp", "target_timestamp"]
    missing = [name for name in required if name not in prediction_table]
    if missing:
        raise KeyError(missing)
    try:
        validation_cutoff = pd.Timestamp(validation_start)
        test_cutoff = pd.Timestamp(test_start)
    except Exception as error:
        raise TypeError("cutoffs must be timestamp-compatible") from error
    if validation_cutoff.tzinfo is None or test_cutoff.tzinfo is None or str(validation_cutoff.tz) != "UTC" or str(test_cutoff.tz) != "UTC":
        raise ValueError("cutoffs must be UTC-aware")
    if validation_cutoff >= test_cutoff:
        raise ValueError("validation_start must precede test_start")
    working = prediction_table.copy(deep=True)
    identifiers = working["batch_id"]
    if identifiers.isna().any() or identifiers.astype("string").str.strip().eq("").any() or identifiers.duplicated().any():
        raise ValueError("batch_id must be unique, nonmissing, and nonblank")
    for column in ["prediction_timestamp", "target_timestamp"]:
        if working[column].isna().any() or not isinstance(working[column].dtype, pd.DatetimeTZDtype) or str(working[column].dt.tz) != "UTC":
            raise ValueError("prediction and target timestamps must be nonmissing UTC values")
    if (working["prediction_timestamp"] >= working["target_timestamp"]).any():
        raise ValueError("prediction_timestamp must precede target_timestamp")
    working = working.sort_values("target_timestamp", kind="stable").reset_index(drop=True)
    parts = {
        "train": working.loc[working["target_timestamp"] < validation_cutoff].copy().reset_index(drop=True),
        "validation": working.loc[(working["target_timestamp"] >= validation_cutoff) & (working["target_timestamp"] < test_cutoff)].copy().reset_index(drop=True),
        "test": working.loc[working["target_timestamp"] >= test_cutoff].copy().reset_index(drop=True),
    }
    if any(part.empty for part in parts.values()) or sum(len(part) for part in parts.values()) != len(working):
        raise ValueError("split must produce three nonempty, exhaustive partitions")
    all_ids = [identifier for part in parts.values() for identifier in part["batch_id"].tolist()]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(working["batch_id"]):
        raise ValueError("split partitions must be exclusive and exhaustive")
    manifest = pd.DataFrame({
        "partition": pd.Series(list(parts), dtype="string"),
        "row_count": np.array([len(part) for part in parts.values()], dtype="int64"),
        "first_target_timestamp": pd.Series([part["target_timestamp"].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ") for part in parts.values()], dtype="string"),
        "last_target_timestamp": pd.Series([part["target_timestamp"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ") for part in parts.values()], dtype="string"),
    })
    return parts, manifest''',
"a10-task2-run": '''availability_decisions = audit_feature_availability(feature_availability)
split_parts, split_manifest = build_chronological_splits(batch_strength, validation_start, test_start)
train_table, validation_table, test_table = split_parts["train"], split_parts["validation"], split_parts["test"]
assert availability_decisions["decision"].tolist() == ["keep", "keep", "keep", "exclude", "exclude"]
assert [len(train_table), len(validation_table), len(test_table)] == [29, 8, 11]
all_ids = train_table["batch_id"].tolist() + validation_table["batch_id"].tolist() + test_table["batch_id"].tolist()
assert len(all_ids) == len(set(all_ids)) == 48 and set(all_ids) == set(batch_strength["batch_id"])
assert train_table["target_timestamp"].max() < validation_start <= validation_table["target_timestamp"].min()
assert validation_table["target_timestamp"].max() < test_start <= test_table["target_timestamp"].min()
EXPECTED_PARTITION_IDS["validation"] = validation_table["batch_id"].tolist()
EXPECTED_PARTITION_IDS["test"] = test_table["batch_id"].tolist()
pd.testing.assert_frame_equal(feature_availability, availability_snapshot)
pd.testing.assert_frame_equal(batch_strength, batch_snapshot)
display(availability_decisions, split_manifest)''',
"a10-task2-save": '''availability_decisions.to_csv(AVAILABILITY_PATH, index=False, encoding="utf-8", lineterminator="\\n")
split_manifest.to_csv(SPLIT_PATH, index=False, encoding="utf-8", lineterminator="\\n")
availability_readback = pd.read_csv(AVAILABILITY_PATH, dtype={"candidate_feature": "string", "latest_required_offset_hours": "int64", "available_by_prediction_time": "bool", "decision": "string"})
split_readback = pd.read_csv(SPLIT_PATH, dtype={"partition": "string", "row_count": "int64", "first_target_timestamp": "string", "last_target_timestamp": "string"})
pd.testing.assert_frame_equal(availability_readback, availability_decisions)
pd.testing.assert_frame_equal(split_readback, split_manifest)
display(AVAILABILITY_PATH, SPLIT_PATH)''',
"a10-task2-explain": '''## Task 2 explanation

Both `+24` candidates require information from 24 hours after prediction time, so they leak future information. A chronological split mirrors using earlier cases to predict later cases; shuffling would mix future conditions into training and can make future use look easier than it is. Validation is used to compare the two supplied approaches. Test stays untouched until that choice is frozen and is then used once for a final estimate.''',
"a10-regression-metrics-function": '''def regression_metrics(actual, predicted):
    try:
        actual_values = np.asarray(actual, dtype="float64")
        predicted_values = np.asarray(predicted, dtype="float64")
    except (TypeError, ValueError) as error:
        raise TypeError("actual and predicted must be numeric array-like values") from error
    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("metric inputs must be one-dimensional")
    if len(actual_values) != len(predicted_values) or len(actual_values) < 2:
        raise ValueError("metric inputs must have equal length of at least two")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("metric inputs must be finite")
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "r2": float(r2_score(actual_values, predicted_values)),
    }''',
"a10-candidates-function": '''def fit_prediction_candidates(train_table, feature_columns, target_column):
    if not isinstance(train_table, pd.DataFrame):
        raise TypeError("train_table must be a pandas DataFrame")
    if not isinstance(feature_columns, (list, tuple)) or any(not isinstance(name, str) for name in feature_columns) or not isinstance(target_column, str):
        raise TypeError("feature and target names must use the documented containers")
    if not feature_columns or len(set(feature_columns)) != len(feature_columns) or target_column in feature_columns:
        raise ValueError("features must be nonempty/distinct and exclude the target")
    missing = [name for name in [*feature_columns, target_column] if name not in train_table]
    if missing:
        raise KeyError(missing)
    if train_table.empty:
        raise ValueError("training data must be nonempty")
    try:
        features = train_table[list(feature_columns)].copy(deep=True).astype("float64")
        target = train_table[target_column].copy(deep=True).astype("float64")
    except (TypeError, ValueError) as error:
        raise ValueError("model values must be numeric") from error
    if not np.isfinite(features.to_numpy()).all() or not np.isfinite(target.to_numpy()).all():
        raise ValueError("model values must be finite")
    mean_baseline = DummyRegressor(strategy="mean")
    mean_baseline.fit(features, target)
    _record_fit("mean_baseline", mean_baseline, train_table)
    linear_pipeline = Pipeline([("scale", StandardScaler()), ("linear", LinearRegression())])
    linear_pipeline.fit(features, target)
    _record_fit("linear_pipeline", linear_pipeline, train_table)
    return {"mean_baseline": mean_baseline, "linear_pipeline": linear_pipeline}''',
"a10-validation-run": '''def choose_validation_winner(metrics_table, metric_column):
    if not isinstance(metrics_table, pd.DataFrame) or not isinstance(metric_column, str):
        raise TypeError("metrics_table must be a DataFrame and metric_column a string")
    if "approach" not in metrics_table:
        raise KeyError("approach")
    if metric_column not in metrics_table:
        raise KeyError(metric_column)
    working = metrics_table[["approach", metric_column]].copy(deep=True)
    if working.empty or working["approach"].isna().any() or not working["approach"].map(lambda value: isinstance(value, str)).all() or working["approach"].str.strip().eq("").any():
        raise ValueError("approach values must be nonempty strings")
    try:
        numeric = pd.to_numeric(working[metric_column], errors="raise").astype("float64")
    except (TypeError, ValueError) as error:
        raise ValueError("metric values must be numeric") from error
    finite = np.isfinite(numeric.to_numpy())
    if not finite.any():
        raise ValueError("at least one finite metric row is required")
    minimum = numeric.loc[finite].min()
    return str(sorted(working.loc[finite & numeric.eq(minimum), "approach"].tolist())[0])

prediction_candidates = fit_prediction_candidates(train_table, feature_columns, prediction_contract["target"])
validation_features = validation_table.set_index("batch_id")[feature_columns]
validation_actual = validation_table[prediction_contract["target"]].to_numpy(dtype="float64")
validation_rows = []
for approach_name, fitted_candidate in prediction_candidates.items():
    candidate_predictions = record_predictions("validation", approach_name, fitted_candidate, validation_features)
    validation_rows.append({"approach": approach_name, **regression_metrics(validation_actual, candidate_predictions)})
validation_metrics = pd.DataFrame(validation_rows).astype({"approach": "string", "mae": "float64", "rmse": "float64", "r2": "float64"})
selected_approach = choose_validation_winner(validation_metrics, "mae")
assert selected_approach == "linear_pipeline"
display(validation_metrics, selected_approach)''',
"a10-validation-save": '''validation_metrics.to_csv(VALIDATION_PATH, index=False, encoding="utf-8", lineterminator="\\n", float_format="%.6f")
validation_readback = pd.read_csv(VALIDATION_PATH, dtype={"approach": "string", "mae": "float64", "rmse": "float64", "r2": "float64"})
assert validation_readback.shape == (2, 4)
display(VALIDATION_PATH)''',
"a10-final-test-run": '''test_features = test_table.set_index("batch_id")[feature_columns]
test_actual = test_table[prediction_contract["target"]].to_numpy(dtype="float64")
test_predictions = record_predictions("test", FROZEN_SELECTED_APPROACH, prediction_candidates[FROZEN_SELECTED_APPROACH], test_features)
final_test_metrics = pd.DataFrame([{"approach": FROZEN_SELECTED_APPROACH, **regression_metrics(test_actual, test_predictions)}]).astype({"approach": "string", "mae": "float64", "rmse": "float64", "r2": "float64"})
final_predictions = pd.DataFrame({
    "batch_id": test_table["batch_id"].reset_index(drop=True).astype("string"),
    "target_timestamp": test_table["target_timestamp"].reset_index(drop=True),
    "actual_strength_mpa": test_actual,
    "predicted_strength_mpa": test_predictions,
})
assert final_test_metrics.shape == (1, 4) and final_predictions.shape == (11, 4)
display(final_test_metrics, final_predictions)''',
"a10-final-test-save": '''final_test_metrics.to_csv(FINAL_METRICS_PATH, index=False, encoding="utf-8", lineterminator="\\n", float_format="%.6f")
final_predictions.to_csv(FINAL_PREDICTIONS_PATH, index=False, encoding="utf-8", lineterminator="\\n", date_format="%Y-%m-%dT%H:%M:%SZ", float_format="%.6f")
final_metrics_readback = pd.read_csv(FINAL_METRICS_PATH, dtype={"approach": "string", "mae": "float64", "rmse": "float64", "r2": "float64"})
final_predictions_readback = pd.read_csv(FINAL_PREDICTIONS_PATH, dtype={"batch_id": "string", "target_timestamp": "string", "actual_strength_mpa": "float64", "predicted_strength_mpa": "float64"})
assert final_metrics_readback.shape == (1, 4) and final_predictions_readback.shape == (11, 4)
display(FINAL_METRICS_PATH, FINAL_PREDICTIONS_PATH)''',
"a10-binary-function": '''def compute_binary_metrics(prediction_table, actual_column, prediction_columns):
    if not isinstance(prediction_table, pd.DataFrame) or not isinstance(actual_column, str) or not isinstance(prediction_columns, dict):
        raise TypeError("use a DataFrame, actual-column string, and insertion-ordered dictionary")
    if prediction_table.empty or not prediction_columns:
        raise ValueError("prediction table and mapping must be nonempty")
    if any(not isinstance(name, str) or not name.strip() for name in prediction_columns) or any(not isinstance(name, str) or not name.strip() for name in prediction_columns.values()):
        raise ValueError("approach and prediction column names must be nonblank strings")
    if len(set(prediction_columns)) != len(prediction_columns) or len(set(prediction_columns.values())) != len(prediction_columns):
        raise ValueError("approach and prediction column names must be unique")
    required = [actual_column, *prediction_columns.values()]
    missing = [name for name in required if name not in prediction_table]
    if missing:
        raise KeyError(missing)
    working = prediction_table[required].copy(deep=True)
    for column in required:
        if working[column].isna().any() or not working[column].isin([0, 1]).all() or not pd.api.types.is_integer_dtype(working[column].dtype):
            raise ValueError("actual and prediction values must be integer 0 or 1")
    actual = working[actual_column].to_numpy(dtype="int64")
    rows = []
    for approach, column in prediction_columns.items():
        predicted = working[column].to_numpy(dtype="int64")
        rows.append({"approach": approach, "accuracy": float(accuracy_score(actual, predicted)), "precision": float(precision_score(actual, predicted, zero_division=0)), "recall": float(recall_score(actual, predicted, zero_division=0))})
    return pd.DataFrame(rows).astype({"approach": "string", "accuracy": "float64", "precision": "float64", "recall": "float64"})''',
"a10-binary-run-save": '''binary_metrics = compute_binary_metrics(supplied_binary, "actual_label", {"supplied_model": "supplied_model_prediction", "dummy_baseline": "dummy_prediction"})
assert binary_metrics["approach"].tolist() == ["supplied_model", "dummy_baseline"]
assert binary_metrics.loc[0, "recall"] > binary_metrics.loc[1, "recall"] and binary_metrics.loc[0, "accuracy"] > binary_metrics.loc[1, "accuracy"]
binary_metrics.to_csv(BINARY_PATH, index=False, encoding="utf-8", lineterminator="\\n", float_format="%.6f")
binary_readback = pd.read_csv(BINARY_PATH, dtype={"approach": "string", "accuracy": "float64", "precision": "float64", "recall": "float64"})
assert binary_readback.shape == (2, 4)
display(binary_metrics, BINARY_PATH)''',
"a10-task3-explain": '''## Task 3 explanation

Validation MAE may guide the choice because validation was reserved for comparing the two already-specified approaches. Test cannot guide that choice: it is used once only after the selection is frozen. The mean baseline's negative validation R-squared means it predicts those validation outcomes worse than the metric's mean-reference comparison. One final test estimate is uncertain and may not represent other periods. Here accuracy summarizes all correct labels, while recall specifically shows that the supplied model finds some actual positives and the all-zero dummy finds none. These were supplied predictions; I did not fit a classifier.''',
}


def _source(cell) -> str:
    return "".join(cell.source) if isinstance(cell.source, list) else cell.source


def copy_starter(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for relative in sorted(grader.BASE_FILES):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ASSIGNMENT_DIR / relative, target)
    shutil.copytree(ASSIGNMENT_DIR / "output", destination / "output")
    return destination


def materialize_solution(destination: Path) -> Path:
    root = copy_starter(destination)
    notebook = nbformat.read(root / "assignment.ipynb", as_version=4)
    by_id = {cell.id: cell for cell in notebook.cells}
    for cell_id, source in CORRECT_SOURCES.items():
        by_id[cell_id].source = source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}})
    client.execute()
    nbformat.write(notebook, root / "assignment.ipynb")
    return root


def run_checker(root: Path, expected: int) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        [sys.executable, str(root / "check_assignment.py")],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == expected, completed.stdout + completed.stderr
    return completed


def run_uv_checker(root: Path, expected: int, cache_dir: Path) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        ["uv", "run", "--python", "3.12.13", str(root / "check_assignment.py")],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "UV_CACHE_DIR": str(cache_dir), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == expected, completed.stdout + completed.stderr
    assert "[FIX] runtime:" not in completed.stdout
    return completed


@contextmanager
def mutated_copy(source: Path, temporary_root: Path, name: str):
    target = temporary_root / name
    shutil.copytree(source, target, symlinks=True)
    yield target


def expect_rejected(root: Path, label: str, action, rejected: list[str]) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "submission"
        shutil.copytree(root, candidate, symlinks=True)
        action(candidate)
        try:
            grader._validate_template(candidate)
        except Exception:
            rejected.append(label)
            return
        raise AssertionError(f"mutant unexpectedly accepted: {label}")


def assert_integrity_result(result: dict, label: str) -> None:
    assert result["score"] == 0 and result["max-score"] == 90, (label, result)
    assert len(result["tests"]) == 5, (label, result)
    assert [test["score"] for test in result["tests"]] == [0, 0, 0, 0, 0], (label, result)
    assert [test["max-score"] for test in result["tests"]] == [10, 20, 25, 30, 5], (label, result)
    assert all("blocked" in test["output"] for test in result["tests"][1:]), (label, result)


def mutate_file_byte(root: Path, relative: str) -> None:
    path = root / relative
    raw = path.read_bytes()
    path.write_bytes(raw + (b"x" if not raw.endswith(b"\n") else b"#"))


def public_integrity_maps() -> tuple[dict[str, str], dict[str, str]]:
    tree = ast.parse((ASSIGNMENT_DIR / "check_assignment.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"CANDIDATE_PROTECTED_FILE_SHA256", "CANDIDATE_PROTECTED_CELL_SHA256"}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    return values["CANDIDATE_PROTECTED_FILE_SHA256"], values["CANDIDATE_PROTECTED_CELL_SHA256"]


def replace_first(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> int:
    assert not (ASSIGNMENT_DIR / "_grader_selftest/constraints.txt").exists(), "candidate must not invent constraints.txt"
    assert grader._resolve_root(ASSIGNMENT_DIR) == ASSIGNMENT_DIR
    starter = run_checker(ASSIGNMENT_DIR, 1)
    assert "scaffold" in starter.stdout.lower() and "output" in starter.stdout.lower()
    public_files, public_cells = public_integrity_maps()
    assert public_files == {key: value for key, value in grader.CANDIDATE_PROTECTED_FILE_SHA256.items() if key != "check_assignment.py"}
    assert public_cells == grader.CANDIDATE_PROTECTED_CELL_SHA256
    grader._validate_integrity_profile()

    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        solution = materialize_solution(temporary_root / "course root" / "10" / "assignment")
        checker = run_checker(solution, 0)
        assert "structure is complete" in checker.stdout
        empty_cache = temporary_root / "empty uv cache"
        direct_solution = run_uv_checker(solution, 0, empty_cache)
        assert "structure is complete" in direct_solution.stdout
        direct_starter = run_uv_checker(ASSIGNMENT_DIR, 1, empty_cache)
        assert "scaffold" in direct_starter.stdout.lower()
        tests, diagnostics = grader.grade_root(solution)
        assert [test["score"] for test in tests] == [10, 20, 25, 30, 5], tests
        assert sum(test["score"] for test in tests) == 90 and diagnostics["fresh_runs"] == 2 and diagnostics["alternate_checks"] == 7
        assert {path.name for path in (solution / "output").iterdir()} == {".gitkeep", *grader.CSV_HASHES, "inference_residuals.png"}

        central_script = ASSIGNMENT_DIR / "_grader_selftest/classroom50_grader.py"
        integrity_cases = []
        for relative in grader.CANDIDATE_PROTECTED_FILE_SHA256:
            candidate = temporary_root / "integrity files" / relative.replace("/", "_").replace(".", "dot")
            shutil.copytree(solution, candidate)
            mutate_file_byte(candidate, relative)
            result = _run_official(central_script, candidate, {**RUNNER_ENV, "SUBMISSION_TAG": f"integrity-file-{len(integrity_cases):02d}"})
            assert_integrity_result(result, relative)
            if relative != "check_assignment.py":
                public = run_checker(candidate, 1)
                assert "[FIX] integrity:" in public.stdout, (relative, public.stdout)
            integrity_cases.append(f"file:{relative}")

        for cell_id in sorted(grader.PROTECTED_CELL_IDS):
            candidate = temporary_root / "integrity cells" / cell_id
            shutil.copytree(solution, candidate)
            notebook = nbformat.read(candidate / "assignment.ipynb", as_version=4)
            cell = next(cell for cell in notebook.cells if cell.id == cell_id)
            suffix = "\n# harmless candidate-integrity mutation" if cell.cell_type == "code" else "\n<!-- harmless candidate-integrity mutation -->"
            cell.source = _source(cell) + suffix
            nbformat.write(notebook, candidate / "assignment.ipynb")
            result = _run_official(central_script, candidate, {**RUNNER_ENV, "SUBMISSION_TAG": f"integrity-cell-{len(integrity_cases):02d}"})
            assert_integrity_result(result, cell_id)
            public = run_checker(candidate, 1)
            assert "[FIX] integrity:" in public.stdout, (cell_id, public.stdout)
            integrity_cases.append(f"cell:{cell_id}")
        assert len(integrity_cases) == 20 and "file:README.md" in integrity_cases and "file:check_assignment.py" in integrity_cases and "file:requirements.txt" in integrity_cases and "cell:a10-setup" in integrity_cases

        original_files = dict(grader.CANDIDATE_PROTECTED_FILE_SHA256)
        map_logic_cases = []
        try:
            grader.CANDIDATE_PROTECTED_FILE_SHA256.pop("README.md")
            try: grader._validate_integrity_profile()
            except AssertionError: map_logic_cases.append("missing-key")
            else: raise AssertionError("missing integrity-map key accepted")
        finally:
            grader.CANDIDATE_PROTECTED_FILE_SHA256.clear(); grader.CANDIDATE_PROTECTED_FILE_SHA256.update(original_files)
        try:
            grader.CANDIDATE_PROTECTED_FILE_SHA256["unexpected.txt"] = "0" * 64
            try: grader._validate_integrity_profile()
            except AssertionError: map_logic_cases.append("extra-key")
            else: raise AssertionError("extra integrity-map key accepted")
        finally:
            grader.CANDIDATE_PROTECTED_FILE_SHA256.clear(); grader.CANDIDATE_PROTECTED_FILE_SHA256.update(original_files)
        try:
            grader.CANDIDATE_PROTECTED_FILE_SHA256["README.md"] = "0" * 64
            try: grader._validate_template(solution)
            except AssertionError: map_logic_cases.append("wrong-digest")
            else: raise AssertionError("wrong integrity-map digest accepted")
        finally:
            grader.CANDIDATE_PROTECTED_FILE_SHA256.clear(); grader.CANDIDATE_PROTECTED_FILE_SHA256.update(original_files)
        assert map_logic_cases == ["missing-key", "extra-key", "wrong-digest"]

        pep_cases = {
            "wrong-requires-python": lambda root: replace_first(root / "check_assignment.py", '# requires-python = "==3.12.13"', '# requires-python = "==3.13.0"'),
            "missing-dependency": lambda root: replace_first(root / "check_assignment.py", '#   "numpy==2.0.2",\n', ""),
            "extra-dependency": lambda root: replace_first(root / "check_assignment.py", "# dependencies = [\n", '# dependencies = [\n#   "scipy==1.17.0",\n'),
            "reordered-dependency": lambda root: replace_first(root / "check_assignment.py", '#   "matplotlib==3.11.1",\n#   "numpy==2.0.2",', '#   "numpy==2.0.2",\n#   "matplotlib==3.11.1",'),
            "wrong-version-pin": lambda root: replace_first(root / "check_assignment.py", '#   "numpy==2.0.2",', '#   "numpy==2.0.1",'),
            "requirements-disagreement": lambda root: replace_first(root / "requirements.txt", "numpy==2.0.2", "numpy==2.0.1"),
        }
        pep_static_cases = []
        for label, mutate in pep_cases.items():
            candidate = temporary_root / "pep static" / label
            shutil.copytree(solution, candidate)
            mutate(candidate)
            public = run_checker(candidate, 1)
            assert "[FIX] integrity:" in public.stdout, (label, public.stdout)
            try: grader._validate_checker_static(candidate)
            except AssertionError: pass
            else: raise AssertionError(f"PEP/static mutation accepted: {label}")
            result = _run_official(central_script, candidate, {**RUNNER_ENV, "SUBMISSION_TAG": f"pep-{label}"})
            assert_integrity_result(result, label)
            pep_static_cases.append(label)
        assert len(pep_static_cases) == 6

        # A second arbitrary layout executes the real notebook entrypoint and reproduces exact bytes.
        relocated = materialize_solution(temporary_root / "relocated arbitrary" / "deep" / "A10 package")
        run_checker(relocated, 0)
        for name, digest in grader.CSV_HASHES.items():
            assert grader.sha256((relocated / "output" / name).read_bytes()).hexdigest() == digest

        # Stored output is not execution evidence.
        fake = temporary_root / "fake stored success"
        shutil.copytree(solution, fake)
        notebook = nbformat.read(fake / "assignment.ipynb", as_version=4)
        cell = next(cell for cell in notebook.cells if cell.id == "a10-ols-function")
        cell.source = 'def fit_bounded_ols(inference_table, predictor_columns, outcome_column):\n    raise RuntimeError("broken")'
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="Everything passed\n")]
        nbformat.write(notebook, fake / "assignment.ipynb")
        fake_tests, _ = grader.grade_root(fake)
        assert sum(test["score"] for test in fake_tests) < 90

        rejected = []
        actions = {
            "extra file": lambda root: (root / "notes.txt").write_text("extra"),
            "alternate workflow": lambda root: ((root / ".github/workflows").mkdir(parents=True), (root / ".github/workflows/grade.yaml").write_text("x")),
            "nested .git": lambda root: ((root / "ordinary/.git").mkdir(parents=True), (root / "ordinary/.git/nested.txt").write_text("x")),
            "nested .classroom50": lambda root: ((root / "ordinary").mkdir(exist_ok=True), (root / "ordinary/.classroom50.yaml").write_text("x")),
            "injected instructor bundle": lambda root: ((root / "_grader_selftest").mkdir(), (root / "_grader_selftest/copied.py").write_text("x")),
            "fixture corrupt": lambda root: (root / "data/mixing_runs.csv").write_bytes(b"corrupt\n"),
            "fixture CRLF": lambda root: (root / "data/feature_availability.csv").write_bytes((root / "data/feature_availability.csv").read_bytes().replace(b"\n", b"\r\n")),
            "fixture missing": lambda root: (root / "data/supplied_binary_predictions.csv").unlink(),
            "manifest corrupt": lambda root: (root / "data/fixture.json").write_text("{}\n"),
            "notebook malformed": lambda root: (root / "assignment.ipynb").write_text("not json"),
            "notebook missing cell": lambda root: _mutate_notebook(root, lambda notebook: notebook.cells.pop()),
            "notebook duplicate cell": lambda root: _mutate_notebook(root, lambda notebook: notebook.cells.append(notebook.cells[-1])),
            "notebook reorder": lambda root: _mutate_notebook(root, lambda notebook: notebook.cells.reverse()),
            "notebook retag": lambda root: _mutate_notebook(root, lambda notebook: setattr(notebook.cells[0], "id", "changed")),
            "starter scaffold": lambda root: _mutate_notebook(root, lambda notebook: setattr(next(cell for cell in notebook.cells if cell.id == "a10-ols-function"), "source", "# TODO\nraise NotImplementedError")),
            "wrong signature": lambda root: _replace_source(root, "a10-ols-function", "def fit_bounded_ols(data):\n    return data"),
            "student import": lambda root: _replace_source(root, "a10-ols-function", "import os\ndef fit_bounded_ols(inference_table, predictor_columns, outcome_column):\n    return None"),
            "network source": lambda root: _append_source(root, "a10-load", "\nrequests.get('https://example.invalid')"),
            "absolute Colab path": lambda root: _append_source(root, "a10-load", "\npath = '/content/data.csv'"),
            "advanced model": lambda root: _append_source(root, "a10-candidates-function", "\nmodel = RandomForestRegressor()"),
            "matrix API": lambda root: _append_source(root, "a10-ols-function", "\nadd_constant = True"),
            "p-value": lambda root: _append_source(root, "a10-task1-run", "\nthing = ols_model.pvalues"),
            "direct predict": lambda root: _append_source(root, "a10-validation-run", "\nthing = prediction_candidates['mean_baseline'].predict(validation_features)"),
            "missing stable sort": lambda root: _replace_text(root, "a10-split-function", 'kind="stable"', 'kind="quicksort"'),
            "missing zero division": lambda root: _replace_text(root, "a10-binary-function", "zero_division=0", "zero_division=1"),
            "missing formula API": lambda root: _replace_text(root, "a10-ols-function", "smf.ols", "smf.glm"),
            "wrong baseline": lambda root: _replace_text(root, "a10-candidates-function", "DummyRegressor", "LinearRegression"),
            "wrong pipeline": lambda root: _replace_text(root, "a10-candidates-function", "StandardScaler", "LinearRegression"),
        }
        for label, action in actions.items():
            expect_rejected(solution, label, action, rejected)

        # Exact delivery-owned regular files and genuine top-level .git descendants are accepted.
        metadata = temporary_root / "metadata accepted"
        shutil.copytree(solution, metadata)
        (metadata / ".git/objects").mkdir(parents=True)
        (metadata / ".git/objects/sentinel").write_text("ignored")
        (metadata / ".classroom50.yaml").write_text("assignment: a10\n")
        (metadata / ".github/workflows").mkdir(parents=True)
        (metadata / ".github/workflows/autograde.yaml").write_text("name: grade\n")
        grader._validate_template(metadata)
        run_checker(metadata, 0)

        # Symlink boundaries reject both ordinary and delivery paths.
        for relative in ["README.md", ".classroom50.yaml", ".github/workflows/autograde.yaml"]:
            candidate = temporary_root / ("symlink " + relative.replace("/", "_"))
            shutil.copytree(solution, candidate)
            target = candidate / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): target.unlink()
            target.symlink_to(candidate / "README.md")
            try: grader._validate_template(candidate)
            except Exception: rejected.append("symlink " + relative)
            else: raise AssertionError(f"symlink accepted: {relative}")

        # Foreign output sentinel survives setup cleanup, then exact final inventory rejects it.
        sentinel = temporary_root / "sentinel run"
        shutil.copytree(solution, sentinel)
        (sentinel / "output/foreign.txt").write_text("preserve")
        notebook = nbformat.read(sentinel / "assignment.ipynb", as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code": cell.outputs = []; cell.execution_count = None
        try:
            NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(sentinel)}}).execute()
        except Exception:
            assert (sentinel / "output/foreign.txt").read_text() == "preserve"
            rejected.append("foreign output sentinel")
        else:
            raise AssertionError("foreign output sentinel was not rejected")

        # Official direct central result: correct 90, starter below full, corrected resubmission 90.
        solved_result = _run_official(central_script, solution, RUNNER_ENV)
        assert solved_result["score"] == 90 and solved_result["max-score"] == 90 and len(solved_result["tests"]) == 5
        starter_root = copy_starter(temporary_root / "official starter")
        starter_result = _run_official(central_script, starter_root, RUNNER_ENV)
        assert starter_result["score"] < 90 and starter_result["schema"] == "classroom50/result/v1"
        corrected_result = _run_official(central_script, solution, {**RUNNER_ENV, "SUBMISSION_TAG": "submission-corrected-002"})
        assert corrected_result["score"] == 90

        # Missing runner context is infrastructure failure with no result.
        missing_context = subprocess.run([sys.executable, str(central_script)], cwd=solution, text=True, capture_output=True, check=False, env={key: value for key, value in os.environ.items() if key not in RUNNER_ENV})
        assert missing_context.returncode == 2 and "[INFRASTRUCTURE]" in missing_context.stderr
        if (solution / "result.json").exists(): (solution / "result.json").unlink()

        unwritable_result = temporary_root / "unwritable result path"
        shutil.copytree(solution, unwritable_result)
        (unwritable_result / "result.json").mkdir()
        blocked_write = subprocess.run([sys.executable, str(central_script)], cwd=unwritable_result, text=True, capture_output=True, check=False, env={**os.environ, **RUNNER_ENV})
        assert blocked_write.returncode == 2 and "[INFRASTRUCTURE]" in blocked_write.stderr and (unwritable_result / "result.json").is_dir()

        # Production bootstrap must refuse this candidate before installation/grading.
        bootstrap = subprocess.run([sys.executable, str(ASSIGNMENT_DIR / "_grader_selftest/autograder.py")], cwd=solution, text=True, capture_output=True, check=False, env={**os.environ, **RUNNER_ENV})
        assert bootstrap.returncode == 2 and "constraints.txt is absent" in bootstrap.stderr and not (solution / "result.json").exists()

        assert len(rejected) == 32, rejected
        print(json.dumps({
            "candidate": "pass",
            "release_certified": False,
            "starter_checker_exit": 1,
            "solution_checker_exit": 0,
            "central_score": 90,
            "corrected_resubmission_score": 90,
            "alternate_functions": 7,
            "rejected_mutants": len(rejected),
            "integrity_mutants": len(integrity_cases),
            "integrity_map_logic_cases": len(map_logic_cases),
            "pep_static_cases": len(pep_static_cases),
            "candidate_integrity_profile": "candidate-nonrelease",
            "candidate_protected_file_sha256": grader.CANDIDATE_PROTECTED_FILE_SHA256,
            "candidate_protected_cell_sha256": grader.CANDIDATE_PROTECTED_CELL_SHA256,
            "fresh_notebook_runs_minimum": 8,
            "csv_artifacts": 8,
            "png_artifacts": 1,
            "production_bootstrap_exit_without_lock": 2,
        }, sort_keys=True))
    return 0


def _mutate_notebook(root: Path, operation) -> None:
    notebook = nbformat.read(root / "assignment.ipynb", as_version=4)
    operation(notebook)
    nbformat.write(notebook, root / "assignment.ipynb")


def _replace_source(root: Path, cell_id: str, source: str) -> None:
    _mutate_notebook(root, lambda notebook: setattr(next(cell for cell in notebook.cells if cell.id == cell_id), "source", source))


def _append_source(root: Path, cell_id: str, suffix: str) -> None:
    _mutate_notebook(root, lambda notebook: setattr(next(cell for cell in notebook.cells if cell.id == cell_id), "source", _source(next(cell for cell in notebook.cells if cell.id == cell_id)) + suffix))


def _replace_text(root: Path, cell_id: str, old: str, new: str) -> None:
    def operation(notebook):
        cell = next(cell for cell in notebook.cells if cell.id == cell_id)
        assert old in _source(cell)
        cell.source = _source(cell).replace(old, new)
    _mutate_notebook(root, operation)


def _run_official(script: Path, root: Path, environment: dict[str, str]) -> dict:
    result = root / "result.json"
    if result.exists() or result.is_symlink(): result.unlink()
    completed = subprocess.run([sys.executable, str(script)], cwd=root, text=True, capture_output=True, check=False, env={**os.environ, **environment})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(result.read_text())
    result.unlink()
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
