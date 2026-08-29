# /// script
# requires-python = "==3.12.13"
# dependencies = ["numpy==2.0.2", "pandas==3.0.5", "scikit-learn==1.9.0"]
# ///

"""Structural and cross-artifact readiness checker for Assignment 11."""

from __future__ import annotations

from hashlib import sha256
import importlib
import json
from pathlib import Path
import re
import struct
import zlib

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
LOCAL_TZ = "America/Chicago"
RELEASE_SHA256 = "7209cddd9b80e9475f9af17169b935e1ac2ef4a7a32fb72963ad0566b3474139"
MANIFEST_SHA256 = "0dfafa6d0981dc00bf8e68f45ba16f371ab5ae75d20d2835fbc76e9748b96192"
RAW_COLUMNS = [
    "station_name", "measurement_timestamp", "air_temperature_c", "wet_bulb_temperature_c",
    "relative_humidity_pct", "rain_intensity_mm_per_hour", "interval_rain_mm", "total_rain_mm",
    "precipitation_type_code", "wind_direction_deg", "wind_speed_mps", "maximum_wind_speed_mps",
    "barometric_pressure_hpa", "solar_radiation_w_m2", "battery_voltage_v",
]
SENSOR_COLUMNS = RAW_COLUMNS[2:]
NUMERIC_FEATURES = [
    "air_temperature_c_t", "relative_humidity_pct_t", "interval_rain_mm_t", "wind_speed_mps_t",
    "maximum_wind_speed_mps_t", "barometric_pressure_hpa_t", "solar_radiation_w_m2_t",
    "wind_direction_sin_t", "wind_direction_cos_t", "air_temperature_lag_1h_c",
    "air_temperature_lag_24h_c", "air_temperature_lag_168h_c", "air_temperature_mean_past_24h_c",
    "air_temperature_change_1h_c", "target_hour_sin", "target_hour_cos",
    "target_day_of_year_sin", "target_day_of_year_cos",
]
FEATURES = ["station_name", *NUMERIC_FEATURES]
CORRELATION_FEATURES = NUMERIC_FEATURES[:7]
STATIONS = ["Foster Weather Station", "Oak Street Weather Station"]
POINTS = [8, 10, 12, 16, 8, 12, 14, 14, 6]
LABELS = [
    "Q1 release audit and coverage", "Q2 deterministic cleaned observations",
    "Q3 complete elapsed-UTC station panel", "Q4 past-only next-hour features",
    "Q5 training-only patterns", "Q6 fixed chronological model files",
    "Q7 validation evaluation", "Q8 test evaluation", "Q9 report contract",
]
DEPENDENCIES = {1: [], 2: [], 3: [2], 4: [3], 5: [4], 6: [4], 7: [6], 8: [6, 7], 9: []}
HEADINGS = ["Executive Summary", "Data and Cleaning", "Patterns", "Forecast Design", "Model Results", "Limitations"]
PANEL_COLUMNS = ["station_name", "measurement_timestamp_utc", *SENSOR_COLUMNS, "source_observed", "hour", "day_of_week", "month"]
Q4_COLUMNS = ["row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc", "target_air_temperature_c", "model_eligible", *NUMERIC_FEATURES]
X_COLUMNS = ["row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc", *NUMERIC_FEATURES]
RANGES = {
    "air_temperature_c": (-50, 50), "wet_bulb_temperature_c": (-50, 50),
    "relative_humidity_pct": (0, 100), "rain_intensity_mm_per_hour": (0, 300),
    "interval_rain_mm": (0, 100), "total_rain_mm": (0, 2000),
    "wind_direction_deg": (0, 359), "wind_speed_mps": (0, 75),
    "maximum_wind_speed_mps": (0, 100), "barometric_pressure_hpa": (850, 1100),
    "solar_radiation_w_m2": (0, 1500), "battery_voltage_v": (0, 20),
}
DOCUMENTED_FILENAMES = [
    "q1_release_audit.csv", "q1_station_coverage.csv", "q1_visualizations.png",
    "q2_cleaned_observations.csv", "q2_cleaning_audit.csv", "q2_missingness.csv",
    "q3_hourly_panel.csv", "q3_panel_summary.csv", "q4_features.csv", "q4_feature_manifest.csv",
    "q5_monthly_station_summary.csv", "q5_correlations.csv", "q5_patterns.png",
    "q6_X_train.csv", "q6_X_validation.csv", "q6_X_test.csv", "q6_y_train.csv", "q6_y_validation.csv", "q6_y_test.csv",
    "q6_split_summary.csv", "q7_model_spec.csv", "q7_validation_predictions.csv", "q7_validation_metrics.csv",
    "q7_permutation_importance.csv", "q8_test_predictions.csv", "q8_test_metrics.csv", "q8_station_metrics.csv", "q8_final_visualizations.png",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read(name: str, columns: list[str]) -> pd.DataFrame:
    path = OUTPUT / name
    _assert(path.is_file() and not path.is_symlink(), f"missing regular artifact: output/{name}")
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise AssertionError(f"cannot parse output/{name}: {error}") from error
    _assert(frame.columns.tolist() == columns, f"output/{name} columns/order differ; expected {columns}")
    return frame


def _utc(series: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, format="mixed", errors="coerce")
    _assert(parsed.notna().all() and series.astype("string").str.contains(r"(?:Z|[+-]00:00)$", regex=True).all(), f"{label} must contain explicit UTC timestamps")
    return parsed


def _numeric(series: pd.Series, label: str, finite: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if finite:
        _assert(np.isfinite(values).all(), f"{label} must be finite")
    return values


def _png(path: Path) -> None:
    _assert(path.is_file() and not path.is_symlink(), f"missing regular PNG: {path.name}")
    data = path.read_bytes(); _assert(data.startswith(b"\x89PNG\r\n\x1a\n"), f"{path.name} is not a PNG")
    position, header, ending = 8, False, False
    while position + 12 <= len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]; kind = data[position + 4:position + 8]; end = position + length + 12
        _assert(end <= len(data), f"{path.name} has a truncated PNG chunk"); payload = data[position + 8:position + 8 + length]
        _assert(struct.unpack(">I", data[position + 8 + length:end])[0] == zlib.crc32(kind + payload) & 0xFFFFFFFF, f"{path.name} has an invalid PNG checksum")
        if not header:
            width, height = struct.unpack(">II", payload[:8]); _assert(kind == b"IHDR" and length == 13 and width > 0 and height > 0, f"{path.name} has no valid IHDR"); header = True
        if kind == b"IEND":
            _assert(length == 0 and end == len(data), f"{path.name} has invalid trailing data"); ending = True; break
        position = end
    _assert(header and ending, f"{path.name} is incomplete")


def _audit_semantics(frame: pd.DataFrame) -> None:
    rules = frame["rule"].astype("string").str.strip(); affected = _numeric(frame["affected_values"], "affected_values")
    _assert(len(frame) > 0 and rules.notna().all() and rules.ne("").all() and rules.is_unique, "audit rules must be nonblank and unique")
    _assert(frame["result"].isin(["rows_rejected", "set_missing", "set_to_zero"]).all(), "audit result is invalid")
    _assert(affected.notna().all() and np.isfinite(affected).all() and affected.ge(0).all() and affected.mod(1).eq(0).all(), "affected_values must be nonnegative integers")
    _assert(int(affected[frame["result"].eq("rows_rejected")].sum()) == 6, "rejected-value total differs")
    _assert(int(affected[frame["result"].eq("set_to_zero")].sum()) == 5044, "solar correction total differs")
    positive = affected[frame["result"].eq("set_missing") & affected.gt(0)]
    _assert(int(positive.eq(1).sum()) >= 3, "three one-value range corrections are required")


def _q1(_state: dict) -> None:
    release, manifest_path = ROOT / "data/chicago_beach_sensors_2022_2024.csv", ROOT / "data/release_manifest.json"
    _assert(release.is_file() and manifest_path.is_file() and not release.is_symlink() and not manifest_path.is_symlink(), "release or manifest missing")
    release_bytes, manifest_bytes = release.read_bytes(), manifest_path.read_bytes()
    _assert(sha256(release_bytes).hexdigest() == RELEASE_SHA256 and sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256, "frozen release integrity differs")
    manifest = json.loads(manifest_bytes)
    raw = pd.read_csv(release)
    facts = [release.name, sha256(release_bytes).hexdigest(), len(release_bytes), len(raw), len(raw.columns), "|".join(raw.columns), LOCAL_TZ]
    expected = [manifest["release_filename"], manifest["release_sha256"], manifest["release_byte_size"], manifest["row_count"], manifest["column_count"], "|".join(manifest["columns"]), manifest["source_timezone"]]
    audit = _read("q1_release_audit.csv", ["check_name", "expected", "observed", "passed"])
    names = ["release_filename", "release_sha256", "release_byte_size", "row_count", "column_count", "column_names", "source_timezone"]
    _assert(audit["check_name"].tolist() == names and audit["expected"].astype(str).tolist() == [str(value) for value in expected] and audit["observed"].astype(str).tolist() == [str(value) for value in facts] and audit["passed"].astype("string").str.lower().eq("true").all(), "release audit values differ")
    coverage = _read("q1_station_coverage.csv", ["station_name", "expected_hours", "observed_hours", "missing_hours", "coverage_pct", "first_timestamp", "last_timestamp"])
    _assert(coverage["station_name"].tolist() == STATIONS and (_numeric(coverage["expected_hours"], "expected_hours") == 26304).all(), "station coverage grain differs")
    observed, missing = _numeric(coverage["observed_hours"], "observed_hours"), _numeric(coverage["missing_hours"], "missing_hours")
    _assert((observed + missing).eq(26304).all() and np.allclose(_numeric(coverage["coverage_pct"], "coverage_pct"), observed / 26304 * 100), "station coverage arithmetic differs")
    _utc(coverage["first_timestamp"], "first_timestamp"); _utc(coverage["last_timestamp"], "last_timestamp"); _png(OUTPUT / "q1_visualizations.png")


def _q2(state: dict) -> None:
    clean = _read("q2_cleaned_observations.csv", [*RAW_COLUMNS, "measurement_timestamp_utc"]); _assert(len(clean) == 50889, "cleaned row count differs")
    times = _utc(clean["measurement_timestamp_utc"], "measurement_timestamp_utc")
    _assert(clean["station_name"].isin(STATIONS).all() and not pd.DataFrame({"station": clean["station_name"], "time": times}).duplicated().any(), "cleaned keys are invalid or duplicated")
    _assert(pd.DataFrame({"time": times, "station": clean["station_name"]}).equals(pd.DataFrame({"time": times, "station": clean["station_name"]}).sort_values(["time", "station"], kind="stable").reset_index(drop=True)), "cleaned rows are not sorted")
    for column, (low, high) in RANGES.items():
        values = _numeric(clean[column], column); _assert(values.dropna().between(low, high, inclusive="both").all(), f"{column} range differs")
    precipitation = _numeric(clean["precipitation_type_code"], "precipitation_type_code"); _assert(precipitation.dropna().isin([0, 40, 60, 70]).all(), "precipitation codes differ")
    audit = _read("q2_cleaning_audit.csv", ["rule", "affected_values", "result"]); _audit_semantics(audit)
    missingness = _read("q2_missingness.csv", ["station_name", "column_name", "missing_count", "missing_pct"])
    expected_keys = [(station, column) for station in STATIONS for column in SENSOR_COLUMNS]
    _assert(list(missingness[["station_name", "column_name"]].itertuples(index=False, name=None)) == expected_keys, "missingness keys/order differ")
    for row in missingness.itertuples(index=False):
        selected = clean.loc[clean["station_name"].eq(row.station_name), row.column_name]; count = int(selected.isna().sum())
        _assert(int(row.missing_count) == count and np.isclose(float(row.missing_pct), count / len(selected) * 100), "missingness values differ")
    state["clean"], state["clean_time"] = clean, times


def _q3(state: dict) -> None:
    panel = _read("q3_hourly_panel.csv", PANEL_COLUMNS); _assert(len(panel) == 52608, "panel row count differs")
    times = _utc(panel["measurement_timestamp_utc"], "panel timestamp")
    expected_hours = pd.date_range("2022-01-01 06:00:00+00:00", "2025-01-01 06:00:00+00:00", freq="h", inclusive="left")
    for station in STATIONS:
        selected = times.loc[panel["station_name"].eq(station)].reset_index(drop=True); _assert(selected.equals(pd.Series(expected_hours)), f"{station} panel hours differ")
    observed = panel["source_observed"].astype("string").str.lower().map({"true": True, "false": False}); _assert(observed.notna().all(), "source_observed must be boolean")
    local = times.dt.tz_convert(LOCAL_TZ); _assert(np.array_equal(_numeric(panel["hour"], "hour"), local.dt.hour) and np.array_equal(_numeric(panel["day_of_week"], "day_of_week"), local.dt.dayofweek) and np.array_equal(_numeric(panel["month"], "month"), local.dt.month), "panel calendar values differ")
    keys = pd.MultiIndex.from_arrays([panel["station_name"], times]); clean_keys = pd.MultiIndex.from_arrays([state["clean"]["station_name"], state["clean_time"]])
    _assert(pd.Series(keys.isin(clean_keys)).equals(observed.reset_index(drop=True)), "source_observed does not match cleaned keys")
    summary = _read("q3_panel_summary.csv", ["station_name", "expected_hours", "observed_hours", "missing_hours", "gap_runs", "longest_gap_hours"])
    _assert(summary["station_name"].tolist() == STATIONS and (_numeric(summary["expected_hours"], "expected_hours") == 26304).all(), "panel summary keys differ")
    for row in summary.itertuples(index=False):
        flags = observed.loc[panel["station_name"].eq(row.station_name)].to_numpy(); missing = ~flags; starts = missing & np.r_[True, ~missing[:-1]]; groups = np.cumsum(starts); longest = int(pd.Series(groups[missing]).value_counts().max()) if missing.any() else 0
        _assert((int(row.observed_hours), int(row.missing_hours), int(row.gap_runs), int(row.longest_gap_hours)) == (int(flags.sum()), int(missing.sum()), int(starts.sum()), longest), "panel summary values differ")
    state["panel"], state["panel_time"] = panel, times


def _q4(state: dict) -> None:
    features = _read("q4_features.csv", Q4_COLUMNS); _assert(len(features) == 52608 and features["row_id"].is_unique, "Q4 row count/IDs differ")
    cutoff, target = _utc(features["cutoff_timestamp_utc"], "cutoff"), _utc(features["target_timestamp_utc"], "target")
    _assert((target - cutoff).eq(pd.Timedelta(hours=1)).all(), "target must be cutoff plus one hour")
    slugs = features["station_name"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"); _assert(features["row_id"].eq(slugs + "_" + cutoff.dt.strftime("%Y%m%d%H")).all(), "row_id values differ")
    panel_global = state["panel"]
    panel = panel_global.sort_values(["station_name", "measurement_timestamp_utc"], kind="stable").copy(); grouped = panel.groupby("station_name", sort=False, observed=True)["air_temperature_c"]
    expected_target = grouped.shift(-1); actual_target = _numeric(features["target_air_temperature_c"], "target")
    _assert(np.allclose(actual_target, expected_target.sort_index(), equal_nan=True), "next-hour targets differ")
    eligible = features["model_eligible"].astype("string").str.lower().map({"true": True, "false": False}); current = _numeric(features["air_temperature_c_t"], "current temperature")
    _assert(eligible.equals((current.notna() & actual_target.notna())), "model_eligible differs")
    for feature, source in zip(NUMERIC_FEATURES[:7], ["air_temperature_c", "relative_humidity_pct", "interval_rain_mm", "wind_speed_mps", "maximum_wind_speed_mps", "barometric_pressure_hpa", "solar_radiation_w_m2"]):
        _assert(np.allclose(_numeric(features[feature], feature), panel_global[source], equal_nan=True), f"{feature} differs from panel")
    for feature, offset in [("air_temperature_lag_1h_c", 1), ("air_temperature_lag_24h_c", 24), ("air_temperature_lag_168h_c", 168)]:
        _assert(np.allclose(_numeric(features[feature], feature), grouped.shift(offset).sort_index(), equal_nan=True), f"{feature} differs")
    rolling = grouped.transform(lambda values: values.rolling(24, min_periods=1).mean()).sort_index(); _assert(np.allclose(_numeric(features["air_temperature_mean_past_24h_c"], "rolling"), rolling, equal_nan=True), "past-24-hour mean differs")
    manifest = _read("q4_feature_manifest.csv", ["feature_name", "source", "earliest_offset_hours", "latest_offset_hours", "role"])
    expected_offsets = [(0, 0)] * 10 + [(-1, -1), (-24, -24), (-168, -168), (-23, 0), (-1, 0)] + [(0, 0)] * 4
    _assert(manifest["feature_name"].tolist() == FEATURES and list(zip(_numeric(manifest["earliest_offset_hours"], "earliest"), _numeric(manifest["latest_offset_hours"], "latest"))) == expected_offsets and manifest["role"].tolist() == ["categorical"] + ["numeric"] * 18, "feature manifest contract differs")
    _assert(manifest["source"].astype("string").str.strip().ne("").all(), "feature manifest source must be nonblank")
    state["features"], state["eligible"] = features, eligible


def _q5(state: dict) -> None:
    monthly = _read("q5_monthly_station_summary.csv", ["station_name", "year", "month", "n_observed", "mean_air_temperature_c", "std_air_temperature_c", "min_air_temperature_c", "max_air_temperature_c"])
    train = state["features"].loc[state["eligible"] & (_utc(state["features"]["target_timestamp_utc"], "target").dt.tz_convert(LOCAL_TZ) < pd.Timestamp("2024-01-01", tz=LOCAL_TZ))]
    _assert(int(_numeric(monthly["n_observed"], "n_observed").sum()) == len(train), "monthly counts do not cover training rows")
    correlations = _read("q5_correlations.csv", ["feature", *CORRELATION_FEATURES]); _assert(correlations["feature"].tolist() == CORRELATION_FEATURES and correlations.shape == (7, 8), "correlation matrix labels differ")
    values = correlations[CORRELATION_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(); _assert(np.isfinite(values).all() and np.allclose(np.diag(values), 1), "correlation matrix values differ")
    _png(OUTPUT / "q5_patterns.png")


def _split_for(target: pd.Series) -> np.ndarray:
    local = target.dt.tz_convert(LOCAL_TZ)
    return np.select([local < pd.Timestamp("2024-01-01", tz=LOCAL_TZ), local < pd.Timestamp("2024-07-01", tz=LOCAL_TZ)], ["train", "validation"], default="test")


def _q6(state: dict) -> None:
    eligible = state["features"].loc[state["eligible"]].copy(); target = _utc(eligible["target_timestamp_utc"], "target"); assignments = _split_for(target)
    for split in ["train", "validation", "test"]:
        x = _read(f"q6_X_{split}.csv", X_COLUMNS); y = _read(f"q6_y_{split}.csv", ["row_id", "target_air_temperature_c"]); expected = eligible.loc[assignments == split].sort_values(["target_timestamp_utc", "station_name"], kind="stable")
        _assert(x["row_id"].tolist() == expected["row_id"].tolist() == y["row_id"].tolist(), f"{split} IDs/order differ")
        _assert(np.allclose(_numeric(y["target_air_temperature_c"], "y"), expected["target_air_temperature_c"]), f"{split} targets differ")
        state[f"x_{split}"], state[f"y_{split}"] = x, y
    summary = _read("q6_split_summary.csv", ["split", "n_rows", "target_start", "target_end", "n_features"]); _assert(summary["split"].tolist() == ["train", "validation", "test"] and (_numeric(summary["n_features"], "n_features") == 19).all(), "split summary labels/features differ")
    for row in summary.itertuples(index=False):
        x = state[f"x_{row.split}"]; times = _utc(x["target_timestamp_utc"], "target"); _assert((int(row.n_rows), pd.Timestamp(row.target_start), pd.Timestamp(row.target_end)) == (len(x), times.min(), times.max()), "split summary values differ")


def _model_spec() -> None:
    spec = _read("q7_model_spec.csv", ["estimator_module", "estimator_class", "parameters_json", "feature_columns", "random_state"]); _assert(len(spec) == 1, "model spec must contain one row"); row = spec.iloc[0]
    _assert(isinstance(row.estimator_module, str) and row.estimator_module.startswith("sklearn.") and re.fullmatch(r"[A-Za-z_]\w*", str(row.estimator_class)), "model class path is invalid")
    try:
        parameters = json.loads(row.parameters_json)
    except Exception as error:
        raise AssertionError(f"parameters_json is invalid: {error}") from error
    _assert(isinstance(parameters, dict) and row.parameters_json == json.dumps(parameters, sort_keys=True) and row.feature_columns == "|".join(FEATURES) and int(row.random_state) == 217, "model spec values differ")
    estimator = getattr(importlib.import_module(row.estimator_module), row.estimator_class, None); _assert(isinstance(estimator, type) and issubclass(estimator, RegressorMixin), "model spec must name an sklearn regressor class")
    _assert("random_state" not in parameters or parameters["random_state"] == 217, "random_state parameter differs"); _assert("n_jobs" not in parameters or parameters["n_jobs"] == 1, "n_jobs parameter differs")


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    actual, rows = predictions["actual"], []; denominator = float(((actual - actual.mean()) ** 2).sum())
    for model, column in [("persistence_baseline", "persistence_prediction"), ("student_model", "model_prediction")]:
        residual = predictions[column] - actual; rows.append({"model": model, "mae": residual.abs().mean(), "rmse": np.sqrt((residual ** 2).mean()), "r2": 1 - float((residual ** 2).sum()) / denominator, "n": len(actual)})
    return pd.DataFrame(rows)


def _prediction_check(state: dict, split: str, name: str, test: bool) -> pd.DataFrame:
    columns = ["row_id", "station_name", "target_timestamp_utc", "actual", "persistence_prediction", "model_prediction"] + (["model_error", "model_absolute_error"] if test else []); frame = _read(name, columns); x, y = state[f"x_{split}"], state[f"y_{split}"]
    _assert(frame["row_id"].tolist() == x["row_id"].tolist() and frame["station_name"].tolist() == x["station_name"].tolist(), f"{name} IDs differ")
    for column in ["actual", "persistence_prediction", "model_prediction"]:
        frame[column] = _numeric(frame[column], column, finite=True)
    _assert(np.allclose(frame["actual"], y["target_air_temperature_c"]) and np.allclose(frame["persistence_prediction"], x["air_temperature_c_t"]), f"{name} actual/persistence values differ")
    _assert(_utc(frame["target_timestamp_utc"], "prediction target").equals(_utc(x["target_timestamp_utc"], "X target")), f"{name} timestamps differ")
    if test:
        error = frame["model_prediction"] - frame["actual"]; _assert(np.allclose(_numeric(frame["model_error"], "error"), error) and np.allclose(_numeric(frame["model_absolute_error"], "absolute error"), error.abs()), "test errors differ")
    return frame


def _q7(state: dict) -> None:
    _model_spec(); predictions = _prediction_check(state, "validation", "q7_validation_predictions.csv", False); metrics = _read("q7_validation_metrics.csv", ["model", "mae", "rmse", "r2", "n"])
    expected = _metric_rows(predictions); _assert(metrics["model"].tolist() == expected["model"].tolist() and np.allclose(metrics[["mae", "rmse", "r2", "n"]].apply(pd.to_numeric), expected[["mae", "rmse", "r2", "n"]]), "validation metrics differ")
    importance = _read("q7_permutation_importance.csv", ["feature", "mean_mae_increase", "std_mae_increase"]); _assert(importance["feature"].tolist() == FEATURES and np.isfinite(importance[["mean_mae_increase", "std_mae_increase"]].apply(pd.to_numeric, errors="coerce")).all().all(), "permutation importance differs")


def _q8(state: dict) -> None:
    predictions = _prediction_check(state, "test", "q8_test_predictions.csv", True); metrics = _read("q8_test_metrics.csv", ["model", "mae", "rmse", "r2", "n"]); expected = _metric_rows(predictions)
    _assert(metrics["model"].tolist() == expected["model"].tolist() and np.allclose(metrics[["mae", "rmse", "r2", "n"]].apply(pd.to_numeric), expected[["mae", "rmse", "r2", "n"]]), "test metrics differ")
    station = _read("q8_station_metrics.csv", ["model", "station_name", "n", "mae", "rmse", "r2"]); _assert(list(station[["model", "station_name"]].itertuples(index=False, name=None)) == [(model, name) for model in ["persistence_baseline", "student_model"] for name in STATIONS], "station metric order differs")
    _png(OUTPUT / "q8_final_visualizations.png")


def _section(text: str, heading: str, following: str | None) -> str:
    start = text.index(f"## {heading}") + len(f"## {heading}"); return text[start:text.index(f"## {following}", start) if following else len(text)]


def _q9(_state: dict) -> None:
    path = ROOT / "report.md"; _assert(path.is_file() and not path.is_symlink(), "missing report.md"); text = path.read_text(encoding="utf-8"); _assert(re.findall(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE) == HEADINGS, "report headings differ")
    lowered = text.lower()
    for placeholder in ["todo", "[value]", "[replace", "[summarize", "[describe", "[report", "[explain", "your text here"]:
        _assert(placeholder not in lowered, f"report contains placeholder: {placeholder}")
    section = _section(text, "Model Results", "Limitations"); lines = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]; _assert(len(lines) == 6, "report must contain exactly four metric rows")
    _assert([cell.strip().lower() for cell in lines[0].strip("|").split("|")] == ["evaluation set", "model", "mae", "rmse", "r2", "n"], "report metric header differs")
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]; keys = [("validation", "persistence_baseline"), ("validation", "student_model"), ("test", "persistence_baseline"), ("test", "student_model")]
    _assert(all(len(row) == 6 for row in rows) and [(row[0].lower(), row[1].lower()) for row in rows] == keys, "report metric rows differ")
    saved = {"validation": _read("q7_validation_metrics.csv", ["model", "mae", "rmse", "r2", "n"]), "test": _read("q8_test_metrics.csv", ["model", "mae", "rmse", "r2", "n"])}
    for row in rows:
        expected = saved[row[0].lower()].loc[saved[row[0].lower()]["model"].eq(row[1])].iloc[0]; values = [float(row[2]), float(row[3]), float(row[4]), int(row[5])]
        _assert(np.allclose(values[:3], expected[["mae", "rmse", "r2"]].astype(float), rtol=1e-5, atol=1e-5) and values[3] == int(expected["n"]), "report metrics differ from artifacts")
    required = {"output/q1_visualizations.png", "output/q5_patterns.png", "output/q8_final_visualizations.png"}; links = re.findall(r"!\[[^]]*\]\(([^)]+\.png)\)", text); _assert(required.issubset(links), "required report images are missing")
    for link in links:
        relative = Path(link); _assert(not relative.is_absolute() and ".." not in relative.parts and relative.parts[:1] == ("output",), f"unsafe image path: {link}"); _png(ROOT / relative)


CHECKS = [_q1, _q2, _q3, _q4, _q5, _q6, _q7, _q8, _q9]


def main() -> int:
    state, passed, score = {}, {}, 0
    for number, (label, points, check) in enumerate(zip(LABELS, POINTS, CHECKS), start=1):
        blockers = [dependency for dependency in DEPENDENCIES[number] if not passed.get(dependency, False)]
        if blockers:
            passed[number] = False; print(f"[BLOCKED] {label} ({points} points): blocked by failed " + ", ".join(f"Q{item}" for item in blockers)); continue
        try:
            check(state)
        except Exception as error:
            passed[number] = False; print(f"[FIX] {label} ({points} points): {type(error).__name__}: {error}")
        else:
            passed[number] = True; score += points; print(f"[PASS] {label} ({points} points)")
    print(f"Readiness score: {score}/100"); return 0 if score == 100 else 1


if __name__ == "__main__":
    raise SystemExit(main())
