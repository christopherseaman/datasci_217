"""Instructor-controlled, artifact-only Assignment 11 grader.

Exact saved training summaries are enforced, but artifacts cannot prove that model
fitting or exploratory decisions used training rows only. That provenance remains
a human source/execution-review question.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import zlib

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RELEASE_NAME = "chicago_beach_sensors_2022_2024.csv"
MANIFEST_NAME = "release_manifest.json"
RELEASE_SHA256 = "7209cddd9b80e9475f9af17169b935e1ac2ef4a7a32fb72963ad0566b3474139"
MANIFEST_SHA256 = "0dfafa6d0981dc00bf8e68f45ba16f371ab5ae75d20d2835fbc76e9748b96192"
RELEASE_ROWS = 50_895
LOCAL_TZ = "America/Chicago"
RAW_COLUMNS = [
    "station_name", "measurement_timestamp", "air_temperature_c",
    "wet_bulb_temperature_c", "relative_humidity_pct",
    "rain_intensity_mm_per_hour", "interval_rain_mm", "total_rain_mm",
    "precipitation_type_code", "wind_direction_deg", "wind_speed_mps",
    "maximum_wind_speed_mps", "barometric_pressure_hpa",
    "solar_radiation_w_m2", "battery_voltage_v",
]
SENSOR_COLUMNS = RAW_COLUMNS[2:]
NUMERIC_FEATURES = [
    "air_temperature_c_t", "relative_humidity_pct_t", "interval_rain_mm_t",
    "wind_speed_mps_t", "maximum_wind_speed_mps_t", "barometric_pressure_hpa_t",
    "solar_radiation_w_m2_t", "wind_direction_sin_t", "wind_direction_cos_t",
    "air_temperature_lag_1h_c", "air_temperature_lag_24h_c",
    "air_temperature_lag_168h_c", "air_temperature_mean_past_24h_c",
    "air_temperature_change_1h_c", "target_hour_sin", "target_hour_cos",
    "target_day_of_year_sin", "target_day_of_year_cos",
]
FEATURES = ["station_name", *NUMERIC_FEATURES]
ID_COLUMNS = ["row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc"]
TARGET = "target_air_temperature_c"
CORRELATION_FEATURES = NUMERIC_FEATURES[:7]
DOCUMENTED_FILENAMES = [
    "q1_release_audit.csv", "q1_station_coverage.csv", "q1_visualizations.png",
    "q2_cleaned_observations.csv", "q2_cleaning_audit.csv", "q2_missingness.csv",
    "q3_hourly_panel.csv", "q3_panel_summary.csv", "q4_features.csv", "q4_feature_manifest.csv",
    "q5_monthly_station_summary.csv", "q5_correlations.csv", "q5_patterns.png",
    "q6_X_train.csv", "q6_X_validation.csv", "q6_X_test.csv", "q6_y_train.csv",
    "q6_y_validation.csv", "q6_y_test.csv", "q6_split_summary.csv", "q7_model_spec.csv",
    "q7_validation_predictions.csv", "q7_validation_metrics.csv", "q7_permutation_importance.csv",
    "q8_test_predictions.csv", "q8_test_metrics.csv", "q8_station_metrics.csv", "q8_final_visualizations.png",
]
POINTS = [8, 10, 12, 16, 8, 12, 14, 14, 6]
TEST_NAMES = [
    "Q1 release audit and coverage", "Q2 deterministic cleaned observations",
    "Q3 complete elapsed-UTC station panel", "Q4 past-only next-hour features",
    "Q5 training-only patterns", "Q6 fixed chronological model files",
    "Q7 validation evaluation", "Q8 test evaluation", "Q9 report contract",
]
DEPENDENCIES = {1: [], 2: [], 3: [2], 4: [3], 5: [4], 6: [4], 7: [6], 8: [6, 7], 9: []}
REPORT_HEADINGS = [
    "Executive Summary", "Data and Cleaning", "Patterns", "Forecast Design",
    "Model Results", "Limitations",
]
RANGES = [
    ("air_temperature_c", -50, 50), ("wet_bulb_temperature_c", -50, 50),
    ("relative_humidity_pct", 0, 100), ("rain_intensity_mm_per_hour", 0, 300),
    ("interval_rain_mm", 0, 100), ("total_rain_mm", 0, 2000),
    ("wind_direction_deg", 0, 359), ("wind_speed_mps", 0, 75),
    ("maximum_wind_speed_mps", 0, 100), ("barometric_pressure_hpa", 850, 1100),
    ("battery_voltage_v", 0, 20),
]


class InfrastructureError(RuntimeError):
    pass


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _context() -> dict[str, str]:
    names = {
        "assignment": "ASSIGNMENT",
        "submission": "SUBMISSION_TAG", "commit": "COMMIT_URL", "release": "RELEASE_URL",
    }
    result = {}
    for field, variable in names.items():
        value = os.environ.get(variable, "").strip()
        if not value:
            raise InfrastructureError(f"missing required grader runner context: {variable}")
        result[field] = value
    result["review"] = os.environ.get("REVIEW_URL", "").strip() or result["commit"]
    result["datetime"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


def _release_path(root: Path) -> Path:
    return root / "data" / RELEASE_NAME


def _validate_environment_and_release(root: Path) -> None:
    if sys.version_info[:3] != (3, 12, 13):
        raise InfrastructureError(f"grader requires Python 3.12.13; found {sys.version.split()[0]}")
    if (np.__version__, pd.__version__, sklearn.__version__) != ("2.0.2", "3.0.5", "1.9.0"):
        raise InfrastructureError(f"dependency versions differ: numpy={np.__version__}, pandas={pd.__version__}, sklearn={sklearn.__version__}")
    release, manifest = _release_path(root), root / "data" / MANIFEST_NAME
    if any(not path.is_file() or path.is_symlink() for path in (release, manifest)):
        raise InfrastructureError("frozen release or manifest is missing or linked")
    if sha256(release.read_bytes()).hexdigest() != RELEASE_SHA256:
        raise InfrastructureError("frozen release SHA-256 differs")
    if sha256(manifest.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise InfrastructureError("frozen manifest SHA-256 differs")


def _gap_summary(values: pd.Series) -> tuple[int, int]:
    missing = ~values.to_numpy(dtype=bool)
    if not missing.any():
        return 0, 0
    starts = missing & np.r_[True, ~missing[:-1]]
    groups = np.cumsum(starts)
    lengths = pd.Series(groups[missing]).value_counts()
    return int(starts.sum()), int(lengths.max())


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    actual = predictions["actual"].astype(float)
    denominator = float(((actual - actual.mean()) ** 2).sum())
    for model, column in [("persistence_baseline", "persistence_prediction"), ("student_model", "model_prediction")]:
        residual = predictions[column].astype(float) - actual
        squared = float((residual ** 2).sum())
        rows.append({
            "model": model, "mae": float(residual.abs().mean()),
            "rmse": float(np.sqrt((residual ** 2).mean())),
            "r2": 1.0 - squared / denominator if denominator else np.nan, "n": len(actual),
        })
    return pd.DataFrame(rows)


@lru_cache(maxsize=4)
def _references(release_name: str, manifest_name: str) -> dict[str, pd.DataFrame]:
    release, manifest_path = Path(release_name), Path(manifest_name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_bytes = release.read_bytes()
    raw = pd.read_csv(release, dtype={"station_name": "string", "measurement_timestamp": "string"})
    _assert(raw.columns.tolist() == RAW_COLUMNS and len(raw) == RELEASE_ROWS, "release schema/count differs")

    checks = [
        ("release_filename", manifest["release_filename"], release.name),
        ("release_sha256", manifest["release_sha256"], sha256(raw_bytes).hexdigest()),
        ("release_byte_size", manifest["release_byte_size"], len(raw_bytes)),
        ("row_count", manifest["row_count"], len(raw)),
        ("column_count", manifest["column_count"], len(raw.columns)),
        ("column_names", "|".join(manifest["columns"]), "|".join(raw.columns)),
        ("source_timezone", manifest["source_timezone"], LOCAL_TZ),
    ]
    audit = pd.DataFrame([
        {"check_name": name, "expected": str(expected), "observed": str(observed), "passed": expected == observed}
        for name, expected, observed in checks
    ])
    parsed = pd.to_datetime(raw["measurement_timestamp"], errors="coerce")
    local = parsed.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
    valid_station = raw["station_name"].isin(manifest["stations"])
    valid_time = local.notna() & local.ge(pd.Timestamp("2022-01-01", tz=LOCAL_TZ)) & local.lt(pd.Timestamp("2025-01-01", tz=LOCAL_TZ))
    duplicate_key = pd.DataFrame({"station": raw["station_name"], "time": local}).duplicated(keep=False)
    keep = valid_station & valid_time & ~duplicate_key
    clean = raw.loc[keep].copy()
    clean["measurement_timestamp"] = local.loc[keep]
    for column in SENSOR_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    cleaning_rows = [
        {"rule": "invalid_station_name", "affected_values": int((~valid_station).sum()), "result": "rows_rejected"},
        {"rule": "ambiguous_or_nonexistent_local_time", "affected_values": int((parsed.notna() & local.isna()).sum()), "result": "rows_rejected"},
        {"rule": "missing_or_out_of_window_timestamp", "affected_values": int((~valid_time & ~(parsed.notna() & local.isna())).sum()), "result": "rows_rejected"},
        {"rule": "duplicate_station_timestamp_key", "affected_values": int((duplicate_key & valid_station & valid_time).sum()), "result": "rows_rejected"},
    ]
    for column, low, high in RANGES:
        invalid = clean[column].notna() & ~clean[column].between(low, high, inclusive="both")
        cleaning_rows.append({"rule": f"{column}_outside_[{low},{high}]", "affected_values": int(invalid.sum()), "result": "set_missing"})
        clean.loc[invalid, column] = np.nan
    precipitation_invalid = clean["precipitation_type_code"].notna() & ~clean["precipitation_type_code"].isin([0, 40, 60, 70])
    cleaning_rows.append({"rule": "precipitation_type_code_not_in_{0,40,60,70}", "affected_values": int(precipitation_invalid.sum()), "result": "set_missing"})
    clean.loc[precipitation_invalid, "precipitation_type_code"] = np.nan
    solar_invalid = clean["solar_radiation_w_m2"].notna() & ~clean["solar_radiation_w_m2"].between(-20, 1500, inclusive="both")
    cleaning_rows.append({"rule": "solar_radiation_w_m2_outside_[-20,1500]", "affected_values": int(solar_invalid.sum()), "result": "set_missing"})
    clean.loc[solar_invalid, "solar_radiation_w_m2"] = np.nan
    solar_negative = clean["solar_radiation_w_m2"].ge(-20) & clean["solar_radiation_w_m2"].lt(0)
    cleaning_rows.append({"rule": "solar_radiation_w_m2_in_[-20,0)", "affected_values": int(solar_negative.sum()), "result": "set_to_zero"})
    clean.loc[solar_negative, "solar_radiation_w_m2"] = 0.0
    clean["measurement_timestamp_utc"] = clean["measurement_timestamp"].dt.tz_convert("UTC")
    clean = clean[[*RAW_COLUMNS, "measurement_timestamp_utc"]].sort_values(
        ["measurement_timestamp_utc", "station_name"], kind="stable"
    ).reset_index(drop=True)
    cleaning_audit = pd.DataFrame(cleaning_rows)
    missingness = []
    for station in manifest["stations"]:
        selected = clean.loc[clean["station_name"].eq(station)]
        for column in SENSOR_COLUMNS:
            count = int(selected[column].isna().sum())
            missingness.append({
                "station_name": station, "column_name": column, "missing_count": count,
                "missing_pct": count / len(selected) * 100,
            })
    missingness = pd.DataFrame(missingness)

    hours = pd.date_range(
        pd.Timestamp("2022-01-01", tz=LOCAL_TZ).tz_convert("UTC"),
        pd.Timestamp("2025-01-01", tz=LOCAL_TZ).tz_convert("UTC"), freq="h", inclusive="left",
    )
    grid = pd.MultiIndex.from_product([hours, manifest["stations"]], names=["measurement_timestamp_utc", "station_name"]).to_frame(index=False)
    observed = clean[["station_name", "measurement_timestamp_utc", *SENSOR_COLUMNS]].copy()
    observed["source_observed"] = True
    panel = grid.merge(observed, on=["measurement_timestamp_utc", "station_name"], how="left", validate="one_to_one", sort=False)
    panel["source_observed"] = panel["source_observed"].fillna(False).astype(bool)
    panel_local = panel["measurement_timestamp_utc"].dt.tz_convert(LOCAL_TZ)
    panel["hour"] = panel_local.dt.hour.astype("int64")
    panel["day_of_week"] = panel_local.dt.dayofweek.astype("int64")
    panel["month"] = panel_local.dt.month.astype("int64")
    panel = panel[["station_name", "measurement_timestamp_utc", *SENSOR_COLUMNS, "source_observed", "hour", "day_of_week", "month"]]
    panel = panel.sort_values(["measurement_timestamp_utc", "station_name"], kind="stable").reset_index(drop=True)
    panel_summary = []
    for station in manifest["stations"]:
        selected = panel.loc[panel["station_name"].eq(station), "source_observed"]
        runs, longest = _gap_summary(selected)
        panel_summary.append({
            "station_name": station, "expected_hours": len(selected), "observed_hours": int(selected.sum()),
            "missing_hours": int((~selected).sum()), "gap_runs": runs, "longest_gap_hours": longest,
        })
    panel_summary = pd.DataFrame(panel_summary)

    coverage = panel_summary[["station_name", "expected_hours", "observed_hours", "missing_hours"]].copy()
    coverage["coverage_pct"] = coverage["observed_hours"] / coverage["expected_hours"] * 100
    coverage_rows = []
    for station in sorted(manifest["stations"]):
        keys = clean.loc[clean["station_name"].eq(station), "measurement_timestamp_utc"]
        row = coverage.loc[coverage["station_name"].eq(station)].iloc[0]
        coverage_rows.append({**row.to_dict(), "coverage_pct": row["coverage_pct"], "first_timestamp": keys.min(), "last_timestamp": keys.max()})
    coverage = pd.DataFrame(coverage_rows)[["station_name", "expected_hours", "observed_hours", "missing_hours", "coverage_pct", "first_timestamp", "last_timestamp"]]

    work = panel.sort_values(["station_name", "measurement_timestamp_utc"], kind="stable").copy()
    grouped = work.groupby("station_name", sort=False, observed=True)["air_temperature_c"]
    rename_current = {
        "air_temperature_c": "air_temperature_c_t", "relative_humidity_pct": "relative_humidity_pct_t",
        "interval_rain_mm": "interval_rain_mm_t", "wind_speed_mps": "wind_speed_mps_t",
        "maximum_wind_speed_mps": "maximum_wind_speed_mps_t", "barometric_pressure_hpa": "barometric_pressure_hpa_t",
        "solar_radiation_w_m2": "solar_radiation_w_m2_t",
    }
    for source, feature in rename_current.items():
        work[feature] = work[source]
    radians = np.deg2rad(work["wind_direction_deg"])
    work["wind_direction_sin_t"], work["wind_direction_cos_t"] = np.sin(radians), np.cos(radians)
    work["air_temperature_lag_1h_c"] = grouped.shift(1)
    work["air_temperature_lag_24h_c"] = grouped.shift(24)
    work["air_temperature_lag_168h_c"] = grouped.shift(168)
    work["air_temperature_mean_past_24h_c"] = grouped.transform(lambda values: values.rolling(24, min_periods=1).mean())
    work["air_temperature_change_1h_c"] = work["air_temperature_c"] - work["air_temperature_lag_1h_c"]
    work["target_timestamp_utc"] = work["measurement_timestamp_utc"] + pd.Timedelta(hours=1)
    work[TARGET] = grouped.shift(-1)
    target_local = work["target_timestamp_utc"].dt.tz_convert(LOCAL_TZ)
    target_hour = target_local.dt.hour
    day_of_year = target_local.dt.dayofyear - 1
    work["target_hour_sin"], work["target_hour_cos"] = np.sin(2 * np.pi * target_hour / 24), np.cos(2 * np.pi * target_hour / 24)
    work["target_day_of_year_sin"], work["target_day_of_year_cos"] = np.sin(2 * np.pi * day_of_year / 366), np.cos(2 * np.pi * day_of_year / 366)
    work["model_eligible"] = work["air_temperature_c"].notna() & work[TARGET].notna()
    slugs = work["station_name"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    work["row_id"] = slugs + "_" + work["measurement_timestamp_utc"].dt.strftime("%Y%m%d%H")
    model = work.rename(columns={"measurement_timestamp_utc": "cutoff_timestamp_utc"})[["row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc", TARGET, "model_eligible", *NUMERIC_FEATURES]]
    model = model.sort_values(["cutoff_timestamp_utc", "station_name"], kind="stable").reset_index(drop=True)
    manifest_rows = [
        ("station_name", "station identity at cutoff", 0, 0, "categorical"),
        *[(name, source, low, high, "numeric") for name, source, low, high in [
            ("air_temperature_c_t", "air_temperature_c at cutoff", 0, 0), ("relative_humidity_pct_t", "relative_humidity_pct at cutoff", 0, 0),
            ("interval_rain_mm_t", "interval_rain_mm at cutoff", 0, 0), ("wind_speed_mps_t", "wind_speed_mps at cutoff", 0, 0),
            ("maximum_wind_speed_mps_t", "maximum_wind_speed_mps at cutoff", 0, 0), ("barometric_pressure_hpa_t", "barometric_pressure_hpa at cutoff", 0, 0),
            ("solar_radiation_w_m2_t", "solar_radiation_w_m2 at cutoff", 0, 0), ("wind_direction_sin_t", "sine of wind_direction_deg at cutoff", 0, 0),
            ("wind_direction_cos_t", "cosine of wind_direction_deg at cutoff", 0, 0), ("air_temperature_lag_1h_c", "air_temperature_c", -1, -1),
            ("air_temperature_lag_24h_c", "air_temperature_c", -24, -24), ("air_temperature_lag_168h_c", "air_temperature_c", -168, -168),
            ("air_temperature_mean_past_24h_c", "air_temperature_c rolling cutoff and prior 23 hours", -23, 0),
            ("air_temperature_change_1h_c", "air_temperature_c cutoff minus prior hour", -1, 0),
            ("target_hour_sin", "known target local hour", 0, 0), ("target_hour_cos", "known target local hour", 0, 0),
            ("target_day_of_year_sin", "known target local day of year", 0, 0), ("target_day_of_year_cos", "known target local day of year", 0, 0),
        ]],
    ]
    feature_manifest = pd.DataFrame(manifest_rows, columns=["feature_name", "source", "earliest_offset_hours", "latest_offset_hours", "role"])

    eligible_model = model.loc[model["model_eligible"]].copy()
    target_local = eligible_model["target_timestamp_utc"].dt.tz_convert(LOCAL_TZ)
    splits = np.select(
        [target_local.lt(pd.Timestamp("2024-01-01", tz=LOCAL_TZ)), target_local.lt(pd.Timestamp("2024-07-01", tz=LOCAL_TZ))],
        ["train", "validation"], default="test",
    )
    monthly_source = eligible_model.loc[splits == "train"].copy()
    monthly_source["year"] = monthly_source["target_timestamp_utc"].dt.tz_convert(LOCAL_TZ).dt.year
    monthly_source["month"] = monthly_source["target_timestamp_utc"].dt.tz_convert(LOCAL_TZ).dt.month
    monthly = monthly_source.groupby(["station_name", "year", "month"], sort=True, observed=True).agg(
        n_observed=(TARGET, "size"), mean_air_temperature_c=(TARGET, "mean"), std_air_temperature_c=(TARGET, "std"),
        min_air_temperature_c=(TARGET, "min"), max_air_temperature_c=(TARGET, "max"),
    ).reset_index()
    correlations = monthly_source[CORRELATION_FEATURES].corr(); correlations.index.name = "feature"; correlations = correlations.reset_index()

    split_summary = []
    x_frames, y_frames = {}, {}
    for split in ["train", "validation", "test"]:
        selected = eligible_model.loc[splits == split].sort_values(["target_timestamp_utc", "station_name"], kind="stable").reset_index(drop=True)
        x_frames[split] = selected[["row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc", *NUMERIC_FEATURES]]
        y_frames[split] = selected[["row_id", TARGET]]
        split_summary.append({
            "split": split, "n_rows": len(selected), "target_start": selected["target_timestamp_utc"].min(),
            "target_end": selected["target_timestamp_utc"].max(), "n_features": len(FEATURES),
        })
    return {
        "audit": audit, "coverage": coverage, "clean": clean, "cleaning_audit": cleaning_audit,
        "missingness": missingness, "panel": panel, "panel_summary": panel_summary,
        "model": model, "feature_manifest": feature_manifest, "monthly": monthly,
        "correlations": correlations,
        "split_summary": pd.DataFrame(split_summary), **{f"x_{key}": value for key, value in x_frames.items()},
        **{f"y_{key}": value for key, value in y_frames.items()},
    }


def _artifact(root: Path, name: str) -> Path:
    return root / "output" / name


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    _assert(path.is_file() and not path.is_symlink(), f"missing regular artifact: output/{path.name}")
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise AssertionError(f"cannot parse output/{path.name}: {error}") from error
    _assert(frame.columns.tolist() == columns, f"output/{path.name} columns/order differ; expected {columns}")
    return frame


def _assert_frame(path: Path, expected: pd.DataFrame, tolerance: float = 1e-9) -> pd.DataFrame:
    observed = _read_csv(path, expected.columns.tolist())
    _assert(observed.shape == expected.shape, f"output/{path.name} shape differs: expected {expected.shape}, found {observed.shape}")
    for column in expected:
        wanted, got = expected[column].reset_index(drop=True), observed[column]
        if isinstance(wanted.dtype, pd.DatetimeTZDtype):
            parsed = pd.to_datetime(got, utc=True, format="mixed", errors="coerce")
            _assert(parsed.equals(wanted.dt.tz_convert("UTC")), f"output/{path.name} differs in {column}")
            if str(wanted.dt.tz) != "UTC":
                offsets = got.astype("string").str.extract(r"([+-]\d\d:\d\d)$", expand=False)
                wanted_offsets = wanted.map(lambda value: value.strftime("%z")[:3] + ":" + value.strftime("%z")[3:]).astype("string")
                _assert(offsets.equals(wanted_offsets), f"output/{path.name} timezone offsets differ in {column}")
        elif pd.api.types.is_datetime64_any_dtype(wanted.dtype):
            _assert(pd.to_datetime(got, errors="coerce").equals(wanted), f"output/{path.name} differs in {column}")
        elif pd.api.types.is_bool_dtype(wanted.dtype):
            normalized = got.astype("string").str.lower().map({"true": True, "false": False})
            _assert(normalized.equals(wanted.astype(bool)), f"output/{path.name} differs in {column}")
        elif pd.api.types.is_numeric_dtype(wanted.dtype):
            numeric = pd.to_numeric(got, errors="coerce")
            _assert(np.allclose(numeric, wanted, rtol=tolerance, atol=tolerance, equal_nan=True), f"output/{path.name} differs in {column}")
        else:
            _assert(got.astype("string").equals(wanted.astype("string")), f"output/{path.name} differs in {column}")
    return observed


def _valid_png(path: Path) -> None:
    _assert(path.is_file() and not path.is_symlink(), f"missing regular PNG: {path.name}")
    data = path.read_bytes()
    _assert(data.startswith(b"\x89PNG\r\n\x1a\n"), f"{path.name} is not a PNG")
    position, seen_header, seen_end = 8, False, False
    while position + 12 <= len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        end = position + 12 + length
        _assert(end <= len(data), f"{path.name} has a truncated PNG chunk")
        payload = data[position + 8:position + 8 + length]
        checksum = struct.unpack(">I", data[position + 8 + length:end])[0]
        _assert(checksum == zlib.crc32(kind + payload) & 0xFFFFFFFF, f"{path.name} has an invalid PNG checksum")
        if not seen_header:
            _assert(kind == b"IHDR" and length == 13, f"{path.name} has no valid IHDR")
            width, height = struct.unpack(">II", payload[:8]); _assert(width > 0 and height > 0, f"{path.name} has invalid dimensions")
            seen_header = True
        if kind == b"IEND":
            _assert(length == 0 and end == len(data), f"{path.name} has invalid trailing data")
            seen_end = True
            break
        position = end
    _assert(seen_header and seen_end, f"{path.name} is incomplete")


def _check_q1(root: Path, refs: dict[str, pd.DataFrame]) -> None:
    audit = _assert_frame(_artifact(root, "q1_release_audit.csv"), refs["audit"])
    _assert(audit["passed"].astype("string").str.lower().eq("true").all(), "release audit has a failed row")
    _assert_frame(_artifact(root, "q1_station_coverage.csv"), refs["coverage"])
    _valid_png(_artifact(root, "q1_visualizations.png"))


def _check_q2(root: Path, refs: dict[str, pd.DataFrame]) -> None:
    _assert_frame(_artifact(root, "q2_cleaned_observations.csv"), refs["clean"])
    audit = _read_csv(_artifact(root, "q2_cleaning_audit.csv"), ["rule", "affected_values", "result"])
    rules = audit["rule"].astype("string").str.strip()
    _assert(len(audit) > 0 and rules.notna().all() and rules.ne("").all() and rules.is_unique, "cleaning audit rules must be nonblank and unique")
    _assert(audit["result"].isin(["rows_rejected", "set_missing", "set_to_zero"]).all(), "cleaning audit result is invalid")
    affected = pd.to_numeric(audit["affected_values"], errors="coerce")
    _assert(affected.notna().all() and np.isfinite(affected).all() and affected.ge(0).all() and affected.mod(1).eq(0).all(), "affected_values must be nonnegative integers")
    _assert(int(affected.loc[audit["result"].eq("rows_rejected")].sum()) == 6, "cleaning audit rejected-value total differs")
    _assert(int(affected.loc[audit["result"].eq("set_to_zero")].sum()) == 5044, "cleaning audit solar correction total differs")
    positive_missing = affected.loc[audit["result"].eq("set_missing") & affected.gt(0)]
    _assert(int(positive_missing.eq(1).sum()) >= 3, "cleaning audit must include the three one-value range corrections")
    _assert_frame(_artifact(root, "q2_missingness.csv"), refs["missingness"])


def _check_q3(root: Path, refs: dict[str, pd.DataFrame]) -> None:
    _assert_frame(_artifact(root, "q3_hourly_panel.csv"), refs["panel"])
    _assert_frame(_artifact(root, "q3_panel_summary.csv"), refs["panel_summary"])


def _check_q4(root: Path, refs: dict[str, pd.DataFrame]) -> None:
    _assert_frame(_artifact(root, "q4_features.csv"), refs["model"])
    manifest = _read_csv(_artifact(root, "q4_feature_manifest.csv"), refs["feature_manifest"].columns.tolist())
    _assert(len(manifest) == len(refs["feature_manifest"]), "feature manifest row count differs")
    for column in ["feature_name", "earliest_offset_hours", "latest_offset_hours", "role"]:
        expected = refs["feature_manifest"][column]
        observed = manifest[column]
        if pd.api.types.is_numeric_dtype(expected):
            _assert(np.allclose(pd.to_numeric(observed, errors="coerce"), expected, rtol=0, atol=0), f"feature manifest differs in {column}")
        else:
            _assert(observed.astype("string").equals(expected.astype("string")), f"feature manifest differs in {column}")
    _assert(manifest["source"].astype("string").str.strip().ne("").all(), "feature manifest source text must be nonblank")


def _check_q5(root: Path, refs: dict[str, pd.DataFrame]) -> None:
    _assert_frame(_artifact(root, "q5_monthly_station_summary.csv"), refs["monthly"])
    _assert_frame(_artifact(root, "q5_correlations.csv"), refs["correlations"])
    _valid_png(_artifact(root, "q5_patterns.png"))


def _check_q6(root: Path, refs: dict[str, pd.DataFrame]) -> None:
    for split in ["train", "validation", "test"]:
        _assert_frame(_artifact(root, f"q6_X_{split}.csv"), refs[f"x_{split}"])
        _assert_frame(_artifact(root, f"q6_y_{split}.csv"), refs[f"y_{split}"])
    _assert_frame(_artifact(root, "q6_split_summary.csv"), refs["split_summary"])


def _pipeline_from_model_spec(root: Path) -> Pipeline:
    columns = ["estimator_module", "estimator_class", "parameters_json", "feature_columns", "random_state"]
    spec = _read_csv(_artifact(root, "q7_model_spec.csv"), columns)
    _assert(len(spec) == 1, "q7_model_spec.csv must contain one row")
    row = spec.iloc[0]
    _assert(isinstance(row["estimator_module"], str) and row["estimator_module"].startswith("sklearn."), "estimator_module must be a sklearn module")
    _assert(isinstance(row["estimator_class"], str) and re.fullmatch(r"[A-Za-z_]\w*", row["estimator_class"]) is not None, "estimator_class is invalid")
    try:
        parameters = json.loads(row["parameters_json"])
    except Exception as error:
        raise AssertionError(f"parameters_json is invalid: {error}") from error
    _assert(isinstance(parameters, dict) and row["parameters_json"] == json.dumps(parameters, sort_keys=True), "parameters_json must be a sorted-key object")
    _assert(row["feature_columns"] == "|".join(FEATURES), "model feature order differs")
    module = importlib.import_module(row["estimator_module"])
    estimator_class = getattr(module, row["estimator_class"], None)
    _assert(isinstance(estimator_class, type) and issubclass(estimator_class, RegressorMixin), "model spec must name an sklearn regressor class")
    try:
        estimator = estimator_class(**parameters)
    except Exception as error:
        raise AssertionError(f"cannot recreate estimator from model spec: {error}") from error
    random_state = pd.to_numeric(pd.Series([row["random_state"]]), errors="coerce").iloc[0]
    _assert(random_state == 217, "random_state must be 217")
    if "random_state" in parameters:
        _assert(parameters["random_state"] == 217, "supported random_state parameter must be 217")
    if "n_jobs" in parameters:
        _assert(parameters["n_jobs"] == 1, "supported n_jobs parameter must be 1")
    supported = estimator.get_params(deep=False)
    if "random_state" in supported:
        _assert(supported["random_state"] == 217, "supported random_state parameter must be 217")
    if "n_jobs" in supported:
        _assert(supported["n_jobs"] == 1, "supported n_jobs parameter must be 1")
    preprocessing = ColumnTransformer([
        ("station", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["station_name"]),
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
    ])
    return Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])


def _declared_model_results(root: Path, refs: dict[str, pd.DataFrame], split: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    pipeline = _pipeline_from_model_spec(root)
    fit_splits = ["train"] if split == "validation" else ["train", "validation"]
    x_fit = pd.concat([refs[f"x_{name}"] for name in fit_splits], ignore_index=True)
    y_fit = pd.concat([refs[f"y_{name}"] for name in fit_splits], ignore_index=True)
    pipeline.fit(x_fit[FEATURES], y_fit[TARGET])
    predictions = np.asarray(pipeline.predict(refs[f"x_{split}"][FEATURES]), dtype=float)
    _assert(np.isfinite(predictions).all(), "declared model produced nonfinite predictions")
    if split != "validation":
        return predictions, None, None
    result = permutation_importance(
        pipeline, refs["x_validation"][FEATURES], refs["y_validation"][TARGET],
        scoring="neg_mean_absolute_error", n_repeats=10, random_state=217,
    )
    return predictions, result.importances_mean, result.importances_std


def _prediction_frame(root: Path, refs: dict[str, pd.DataFrame], split: str, name: str, test: bool) -> pd.DataFrame:
    columns = ["row_id", "station_name", "target_timestamp_utc", "actual", "persistence_prediction", "model_prediction"]
    if test:
        columns += ["model_error", "model_absolute_error"]
    got = _read_csv(_artifact(root, name), columns)
    expected = refs[f"x_{split}"][["row_id", "station_name", "target_timestamp_utc", "air_temperature_c_t"]].merge(refs[f"y_{split}"], on="row_id", validate="one_to_one").rename(columns={TARGET: "actual", "air_temperature_c_t": "persistence_prediction"})
    expected = expected[["row_id", "station_name", "target_timestamp_utc", "actual", "persistence_prediction"]]
    _assert_frame_values(got[["row_id", "station_name", "target_timestamp_utc", "actual", "persistence_prediction"]], expected, name)
    got["actual"] = pd.to_numeric(got["actual"], errors="coerce")
    got["persistence_prediction"] = pd.to_numeric(got["persistence_prediction"], errors="coerce")
    got["model_prediction"] = pd.to_numeric(got["model_prediction"], errors="coerce")
    _assert(np.isfinite(got[["actual", "persistence_prediction", "model_prediction"]].to_numpy()).all(), f"output/{name} contains nonfinite values")
    if test:
        error = got["model_prediction"] - got["actual"]
        _assert(np.allclose(pd.to_numeric(got["model_error"], errors="coerce"), error, rtol=1e-9, atol=1e-9), "Q8 model_error differs")
        _assert(np.allclose(pd.to_numeric(got["model_absolute_error"], errors="coerce"), error.abs(), rtol=1e-9, atol=1e-9), "Q8 model_absolute_error differs")
    return got


def _assert_frame_values(observed: pd.DataFrame, expected: pd.DataFrame, name: str) -> None:
    _assert(observed.shape == expected.shape, f"output/{name} row count differs")
    for column in expected:
        if column.endswith("_utc"):
            got = pd.to_datetime(observed[column], utc=True, format="mixed", errors="coerce")
            _assert(got.equals(expected[column].reset_index(drop=True)), f"output/{name} differs in {column}")
        elif pd.api.types.is_numeric_dtype(expected[column]):
            _assert(np.allclose(pd.to_numeric(observed[column], errors="coerce"), expected[column], rtol=0, atol=0), f"output/{name} differs in {column}")
        else:
            _assert(observed[column].astype("string").equals(expected[column].astype("string")), f"output/{name} differs in {column}")


def _check_metrics(root: Path, predictions: pd.DataFrame, name: str) -> None:
    _assert_frame(_artifact(root, name), _metrics(predictions))


def _check_q7(root: Path, refs: dict[str, pd.DataFrame]) -> None:
    expected_predictions, expected_mean, expected_std = _declared_model_results(root, refs, "validation")
    predictions = _prediction_frame(root, refs, "validation", "q7_validation_predictions.csv", False)
    _assert(np.allclose(predictions["model_prediction"], expected_predictions, rtol=1e-9, atol=1e-9), "validation predictions do not match the declared train-fitted pipeline")
    _check_metrics(root, predictions, "q7_validation_metrics.csv")
    importance = _read_csv(_artifact(root, "q7_permutation_importance.csv"), ["feature", "mean_mae_increase", "std_mae_increase"])
    _assert(importance["feature"].tolist() == FEATURES, "permutation importance feature set/order differs")
    values = importance[["mean_mae_increase", "std_mae_increase"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    _assert(np.isfinite(values).all(), "permutation importance values must be finite")
    _assert(np.allclose(values[:, 0], expected_mean, rtol=1e-9, atol=1e-9) and np.allclose(values[:, 1], expected_std, rtol=1e-9, atol=1e-9), "permutation importance does not match the declared train-fitted pipeline")


def _check_q8(root: Path, refs: dict[str, pd.DataFrame]) -> None:
    expected_predictions, _, _ = _declared_model_results(root, refs, "test")
    predictions = _prediction_frame(root, refs, "test", "q8_test_predictions.csv", True)
    _assert(np.allclose(predictions["model_prediction"], expected_predictions, rtol=1e-9, atol=1e-9), "test predictions do not match the declared train-plus-validation-fitted pipeline")
    _check_metrics(root, predictions, "q8_test_metrics.csv")
    rows = []
    for model, column in [("persistence_baseline", "persistence_prediction"), ("student_model", "model_prediction")]:
        for station, group in predictions.groupby("station_name", sort=True, observed=True):
            residual = group[column] - group["actual"]; denominator = float(((group["actual"] - group["actual"].mean()) ** 2).sum())
            rows.append({"model": model, "station_name": station, "n": len(group), "mae": residual.abs().mean(), "rmse": np.sqrt((residual ** 2).mean()), "r2": 1 - float((residual ** 2).sum()) / denominator if denominator else np.nan})
    _assert_frame(_artifact(root, "q8_station_metrics.csv"), pd.DataFrame(rows))
    _valid_png(_artifact(root, "q8_final_visualizations.png"))


def _section(text: str, heading: str, next_heading: str | None) -> str:
    start = text.index(f"## {heading}") + len(f"## {heading}")
    end = text.index(f"## {next_heading}", start) if next_heading else len(text)
    return text[start:end].strip()


def _check_q9(root: Path, _refs: dict[str, pd.DataFrame]) -> None:
    report = root / "report.md"
    _assert(report.is_file() and not report.is_symlink(), "missing regular root report.md")
    text = report.read_text(encoding="utf-8")
    headings = re.findall(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE)
    _assert(headings == REPORT_HEADINGS, f"report H2 headings/order differ; expected {REPORT_HEADINGS}")
    lowered = text.lower()
    for placeholder in ["todo", "[value]", "[replace", "[summarize", "[describe", "[report", "[explain", "your text here"]:
        _assert(placeholder not in lowered, f"report contains starter placeholder: {placeholder}")
    for index, heading in enumerate(REPORT_HEADINGS):
        body = _section(text, heading, REPORT_HEADINGS[index + 1] if index + 1 < len(REPORT_HEADINGS) else None)
        _assert(re.search(r"\w", re.sub(r"!\[[^]]*\]\([^)]+\)", "", body)) is not None, f"report section '{heading}' lacks content")
    section = _section(text, "Model Results", "Limitations")
    table_lines = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
    _assert(len(table_lines) == 6, "Model Results must contain one header, separator, and exactly four metric rows")
    header = [cell.strip().lower() for cell in table_lines[0].strip("|").split("|")]
    _assert(header == ["evaluation set", "model", "mae", "rmse", "r2", "n"], "report metrics header differs")
    candidate_rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in table_lines[2:]]
    expected_keys = [("validation", "persistence_baseline"), ("validation", "student_model"), ("test", "persistence_baseline"), ("test", "student_model")]
    _assert([(row[0].lower(), row[1].lower()) for row in candidate_rows] == expected_keys and all(len(row) == 6 for row in candidate_rows), "report metric rows/order differ")
    saved = {
        "validation": _read_csv(_artifact(root, "q7_validation_metrics.csv"), ["model", "mae", "rmse", "r2", "n"]),
        "test": _read_csv(_artifact(root, "q8_test_metrics.csv"), ["model", "mae", "rmse", "r2", "n"]),
    }
    try:
        for row in candidate_rows:
            values = [float(row[2]), float(row[3]), float(row[4]), int(row[5])]
            expected = saved[row[0].lower()].loc[saved[row[0].lower()]["model"].eq(row[1].lower())].iloc[0]
            _assert(np.allclose(values[:3], [expected["mae"], expected["rmse"], expected["r2"]], rtol=1e-5, atol=1e-5) and values[3] == int(expected["n"]), "report metrics differ from saved artifacts")
    except (ValueError, TypeError) as error:
        raise AssertionError("report metric values must be numeric") from error
    required = {"output/q1_visualizations.png", "output/q5_patterns.png", "output/q8_final_visualizations.png"}
    links = re.findall(r"!\[[^]]*\]\(([^)]+\.png)\)", text, flags=re.IGNORECASE)
    _assert(required.issubset(set(links)), f"report must embed required PNGs: {sorted(required)}")
    for link in links:
        relative = Path(link)
        _assert(not relative.is_absolute() and ".." not in relative.parts and relative.parts[:1] == ("output",), f"unsafe report image path: {link}")
        _valid_png(root / relative)


CHECKS = [_check_q1, _check_q2, _check_q3, _check_q4, _check_q5, _check_q6, _check_q7, _check_q8, _check_q9]


def evaluate_submission(root: Path) -> list[dict]:
    refs = _references(str(_release_path(root).resolve()), str((root / "data" / MANIFEST_NAME).resolve()))
    rows, passed = [], {}
    for number, (name, points, check) in enumerate(zip(TEST_NAMES, POINTS, CHECKS), start=1):
        blockers = [dependency for dependency in DEPENDENCIES[number] if not passed.get(dependency, False)]
        if blockers:
            ok, status, detail = False, "BLOCKED", "blocked by failed " + ", ".join(f"Q{item}" for item in blockers)
        else:
            try:
                check(root, refs)
            except Exception as error:
                ok, status, detail = False, "FIX", f"{type(error).__name__}: {error}"
            else:
                ok, status, detail = True, "PASS", "artifact contract passed"
        passed[number] = ok
        rows.append({"number": number, "name": name, "points": points, "passed": ok, "status": status, "detail": detail})
    return rows


def grade_submission(submission_root: str | Path) -> dict:
    context = _context()
    root = Path(submission_root).resolve()
    if not root.is_dir():
        raise InfrastructureError(f"submission root is not a directory: {root}")
    _validate_environment_and_release(root)
    rows = evaluate_submission(root)
    tests = []
    for row in rows:
        score = row["points"] if row["passed"] else 0
        print(f"[{row['status']}] {row['name']} ({row['points']} points): {row['detail']}")
        tests.append({"test-name": row["name"], "passed": row["passed"], "score": score, "max-score": row["points"]})
    return {"schema": "datasci217/grading-result/v1", **context, "score": sum(test["score"] for test in tests), "max-score": 100, "tests": tests}


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    result_path = Path.cwd() / "result.json"
    try:
        if result_path.exists() or result_path.is_symlink():
            if not result_path.is_file() and not result_path.is_symlink():
                raise InfrastructureError("result.json path is not a regular file")
            result_path.unlink()
        result = grade_submission(target)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    except Exception as error:
        if result_path.is_file() or result_path.is_symlink():
            result_path.unlink()
        print(f"Grader infrastructure failure: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
