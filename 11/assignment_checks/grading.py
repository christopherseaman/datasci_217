"""Course-owned grading rules for Assignment 11, the final exam.

The exam handout ships no checks. After the deadline the course grades each
fork's committed files with this module, through `check_assignment.py` and
`scripts/grade_submissions.py`: the CSV and PNG files in `output/`, plus a
zero-point note on the structure of `report.md`. Student code is never
imported, executed, or read; the notebooks and the report's reasoning are left
to human review.

Every expected value is recomputed from the supplied release in
`11/assignment/data/`, whose SHA-256 is checked before grading starts.

Each check reads its own artifact and compares parsed values, not text: line
endings, a byte-order mark, spaces around cells and header names, a missing
final newline, column order, extra columns, a leading unnamed index column,
number format (`2`, `2.0`, `2.00`, and values rounded to two decimals), the
common spellings of booleans and missing values, and the letter case of labels
never cost points. Rows are matched by their key, not their position. A missing
or extra row costs only the rows check; a value check judges the rows that are
present. A downstream value also passes when it follows from the student's own
upstream file (Q3 from Q2, Q4 from Q3, Q5 and Q6 from Q4, Q7 and Q8 from Q6,
and metrics from the predictions they summarize), so one mistake is charged
once, where it was made. Without the predictions, the student model's metrics
cannot be recomputed, so they need only be numbers with the right row count.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
from typing import Callable

import numpy as np
import pandas as pd


SCHEMA = "datasci217/grading-result/v1"
DATA_DIR = Path(__file__).resolve().parents[1] / "assignment" / "data"
RELEASE_NAME = "chicago_beach_sensors_2022_2024.csv"
MANIFEST_NAME = "release_manifest.json"
RELEASE_SHA256 = "7209cddd9b80e9475f9af17169b935e1ac2ef4a7a32fb72963ad0566b3474139"
MANIFEST_SHA256 = "13aa33f04d6011c77446c34dfaa664011d0875959028e0d0be4a3beca8fee2bc"
RELEASE_ROWS = 50_895
LOCAL_TZ = "America/Chicago"
WINDOW = (pd.Timestamp("2022-01-01", tz=LOCAL_TZ), pd.Timestamp("2025-01-01", tz=LOCAL_TZ))
VALIDATION_START = pd.Timestamp("2024-01-01", tz=LOCAL_TZ)
TEST_START = pd.Timestamp("2024-07-01", tz=LOCAL_TZ)
SPLITS = ("train", "validation", "test")
HOUR_NS = 3_600_000_000_000

RAW_COLUMNS = [
    "station_name", "measurement_timestamp", "air_temperature_c",
    "wet_bulb_temperature_c", "relative_humidity_pct",
    "rain_intensity_mm_per_hour", "interval_rain_mm", "total_rain_mm",
    "precipitation_type_code", "wind_direction_deg", "wind_speed_mps",
    "maximum_wind_speed_mps", "barometric_pressure_hpa",
    "solar_radiation_w_m2", "battery_voltage_v",
]
SENSOR_COLUMNS = RAW_COLUMNS[2:]
RANGES = {
    "air_temperature_c": (-50, 50), "wet_bulb_temperature_c": (-50, 50),
    "relative_humidity_pct": (0, 100), "rain_intensity_mm_per_hour": (0, 300),
    "interval_rain_mm": (0, 100), "total_rain_mm": (0, 2000),
    "wind_direction_deg": (0, 359), "wind_speed_mps": (0, 75),
    "maximum_wind_speed_mps": (0, 100), "barometric_pressure_hpa": (850, 1100),
    "solar_radiation_w_m2": (-20, 1500), "battery_voltage_v": (0, 20),
}
PRECIPITATION_CODES = (0, 40, 60, 70)
# The sensor columns whose values a Q2 range rule changes in this release.
RULE_COLUMNS = ["interval_rain_mm", "wind_speed_mps", "maximum_wind_speed_mps"]
UNCHANGED_COLUMNS = [c for c in SENSOR_COLUMNS if c not in [*RULE_COLUMNS, "solar_radiation_w_m2"]]
CURRENT_SOURCES = {
    "air_temperature_c_t": "air_temperature_c", "relative_humidity_pct_t": "relative_humidity_pct",
    "interval_rain_mm_t": "interval_rain_mm", "wind_speed_mps_t": "wind_speed_mps",
    "maximum_wind_speed_mps_t": "maximum_wind_speed_mps",
    "barometric_pressure_hpa_t": "barometric_pressure_hpa", "solar_radiation_w_m2_t": "solar_radiation_w_m2",
}
CURRENT_FEATURES = list(CURRENT_SOURCES)
WIND_FEATURES = ["wind_direction_sin_t", "wind_direction_cos_t"]
LAG_FEATURES = ["air_temperature_lag_1h_c", "air_temperature_lag_24h_c", "air_temperature_lag_168h_c"]
MEAN_FEATURE = "air_temperature_mean_past_24h_c"
CHANGE_FEATURE = "air_temperature_change_1h_c"
HOUR_FEATURES = ["target_hour_sin", "target_hour_cos"]
DAY_FEATURES = ["target_day_of_year_sin", "target_day_of_year_cos"]
NUMERIC_FEATURES = [*CURRENT_FEATURES, *WIND_FEATURES, *LAG_FEATURES, MEAN_FEATURE, CHANGE_FEATURE,
                    *HOUR_FEATURES, *DAY_FEATURES]
FEATURES = ["station_name", *NUMERIC_FEATURES]
TARGET = "target_air_temperature_c"
OFFSETS = {
    "station_name": (0, 0, "categorical"),
    **{name: (0, 0, "numeric") for name in [*CURRENT_FEATURES, *WIND_FEATURES]},
    "air_temperature_lag_1h_c": (-1, -1, "numeric"), "air_temperature_lag_24h_c": (-24, -24, "numeric"),
    "air_temperature_lag_168h_c": (-168, -168, "numeric"), MEAN_FEATURE: (-23, 0, "numeric"),
    CHANGE_FEATURE: (-1, 0, "numeric"),
    **{name: (0, 0, "numeric") for name in [*HOUR_FEATURES, *DAY_FEATURES]},
}
MODELS = ("persistence_baseline", "student_model")
METRICS = ("mae", "rmse", "r2")
REPORT_HEADINGS = ["Executive Summary", "Data and Cleaning", "Patterns", "Forecast Design",
                   "Model Results", "Limitations"]
REPORT_IMAGES = ["output/q1_visualizations.png", "output/q5_patterns.png", "output/q8_final_visualizations.png"]

# Every graded artifact: its required columns (None for a PNG) and the task that saves it.
ARTIFACTS = {
    "q1_release_audit.csv": (["check_name", "expected", "observed", "passed"], "Q1 section 1.2"),
    "q1_station_coverage.csv": (["station_name", "expected_hours", "observed_hours", "missing_hours",
                                 "coverage_pct", "first_timestamp", "last_timestamp"], "Q1 section 1.3"),
    "q1_visualizations.png": (None, "Q1 section 1.4"),
    "q2_cleaned_observations.csv": ([*RAW_COLUMNS, "measurement_timestamp_utc"], "Q2 sections 2.2 and 2.3"),
    "q2_cleaning_audit.csv": (["rule", "affected_values", "result"], "Q2 section 2.4"),
    "q2_missingness.csv": (["station_name", "column_name", "missing_count", "missing_pct"], "Q2 section 2.4"),
    "q3_hourly_panel.csv": (["station_name", "measurement_timestamp_utc", *SENSOR_COLUMNS, "source_observed",
                             "hour", "day_of_week", "month"], "Q3 section 3.2"),
    "q3_panel_summary.csv": (["station_name", "expected_hours", "observed_hours", "missing_hours", "gap_runs",
                              "longest_gap_hours"], "Q3 section 3.3"),
    "q4_features.csv": (["row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc", TARGET,
                         "model_eligible", *NUMERIC_FEATURES], "Q4 section 4.2"),
    "q4_feature_manifest.csv": (["feature_name", "source", "earliest_offset_hours", "latest_offset_hours",
                                 "role"], "Q4 section 4.3"),
    "q5_monthly_station_summary.csv": (["station_name", "year", "month", "n_observed", "mean_air_temperature_c",
                                        "std_air_temperature_c", "min_air_temperature_c",
                                        "max_air_temperature_c"], "Q5 section 5.2"),
    "q5_correlations.csv": (["feature", *CURRENT_FEATURES], "Q5 section 5.3"),
    "q5_patterns.png": (None, "Q5 section 5.4"),
    **{f"q6_X_{split}.csv": (["row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc",
                              *NUMERIC_FEATURES], "Q6 section 6.3") for split in SPLITS},
    **{f"q6_y_{split}.csv": (["row_id", TARGET], "Q6 section 6.3") for split in SPLITS},
    "q6_split_summary.csv": (["split", "n_rows", "target_start", "target_end", "n_features"], "Q6 section 6.4"),
    "q7_model_spec.csv": (["estimator_module", "estimator_class", "parameters_json", "feature_columns",
                           "random_state"], "Q7 section 7.4"),
    "q7_validation_predictions.csv": (["row_id", "station_name", "target_timestamp_utc", "actual",
                                       "persistence_prediction", "model_prediction"], "Q7 section 7.3"),
    "q7_validation_metrics.csv": (["model", "mae", "rmse", "r2", "n"], "Q7 section 7.3"),
    "q7_permutation_importance.csv": (["feature", "mean_mae_increase", "std_mae_increase"], "Q7 section 7.4"),
    "q8_test_predictions.csv": (["row_id", "station_name", "target_timestamp_utc", "actual",
                                 "persistence_prediction", "model_prediction", "model_error",
                                 "model_absolute_error"], "Q8 section 8.2"),
    "q8_test_metrics.csv": (["model", "mae", "rmse", "r2", "n"], "Q8 section 8.2"),
    "q8_station_metrics.csv": (["model", "station_name", "n", "mae", "rmse", "r2"], "Q8 section 8.3"),
    "q8_final_visualizations.png": (None, "Q8 section 8.4"),
}

TOLERANCE = 0.006          # measured and derived values: two decimals or more pass
PERCENT_TOLERANCE = 0.051  # percentages: one decimal or more pass
MIN_PRESENT = 0.5          # a value check needs at least half of the expected rows present
MISSING_TOKENS = {"", "nan", "na", "n/a", "<na>", "none", "null", "nat"}
TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0"}
FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0"}
OFFSET_SUFFIX = r"(?:[+-]\d{2}:?\d{2}|[zZ]|\s*UTC)$"
DELIMITERS = (",", ";", "\t")  # a CSV may separate its cells with commas, semicolons, or tabs
DECIMAL_COMMA = r"^([+-]?\d*),(\d+)$"  # 12,5 for 12.5, as a semicolon-separated spreadsheet file writes it
NAT = np.iinfo(np.int64).min


class InfrastructureError(RuntimeError):
    """The trusted release is missing or changed, so nothing can be graded."""


# ---------------------------------------------------------------------------
# Parsing text cells


def text_series(values) -> pd.Series:
    return pd.Series(values, dtype="str")


def numbers(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Parse text cells as floats; return (values with NaN for missing, invalid-cell mask)."""
    values = values.astype(object)
    lowered = values.astype(str).str.strip().str.lower()
    blank = (lowered.isin(MISSING_TOKENS) | values.isna()).to_numpy(dtype=bool)
    parsed = pd.to_numeric(values.where(~blank, np.nan), errors="coerce").to_numpy(dtype=float)
    return parsed, np.isnan(parsed) & ~blank


def flags(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Parse text cells as booleans; return (1.0, 0.0, or NaN, invalid-cell mask)."""
    lowered = values.astype(str).str.strip().str.lower()
    parsed = np.full(len(values), np.nan)
    parsed[lowered.isin(TRUE_TOKENS).to_numpy(dtype=bool)] = 1.0
    parsed[lowered.isin(FALSE_TOKENS).to_numpy(dtype=bool)] = 0.0
    blank = lowered.isin(MISSING_TOKENS).to_numpy(dtype=bool)
    return parsed, np.isnan(parsed) & ~blank


def _ns(parsed: pd.Series) -> np.ndarray:
    """Naive datetimes as int64 nanoseconds, with NAT for missing."""
    return pd.to_datetime(parsed).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def as_utc(nanoseconds) -> pd.Series:
    """int64 nanoseconds (NAT for missing) as a UTC datetime Series, without float rounding."""
    return pd.Series(np.asarray(nanoseconds, dtype=np.int64).view("datetime64[ns]")).dt.tz_localize("UTC")


def _parse_times(values: pd.Series, utc: bool) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, utc=utc, format="ISO8601", errors="coerce")
    except (ValueError, TypeError):
        parsed = None
    if parsed is None or (parsed.isna() & ~values.str.lower().isin(MISSING_TOKENS)).any():
        cleaned = values.str.replace(r"\s*UTC$", "+00:00", regex=True)
        parsed = pd.to_datetime(cleaned, utc=utc, format="mixed", errors="coerce")
    return parsed


def _offsets(values: pd.Series) -> np.ndarray:
    return values.str.contains(OFFSET_SUFFIX, regex=True).to_numpy(dtype=bool)


def instants(values: pd.Series, naive: str = "utc") -> np.ndarray:
    """UTC nanoseconds per cell; a time without an offset is read as UTC or as Chicago local time."""
    values = values.astype("str").str.strip()
    out = np.full(len(values), NAT, dtype=np.int64)
    aware = _offsets(values)
    if aware.any():
        out[aware] = _ns(_parse_times(values[aware], utc=True).dt.tz_localize(None))
    if (~aware).any():
        parsed = _parse_times(values[~aware], utc=False)
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        elif naive == "local":
            parsed = parsed.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
            parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        out[~aware] = _ns(parsed)
    return out


def wall_times(values: pd.Series) -> np.ndarray:
    """Chicago wall-clock nanoseconds: a time with an offset is converted to Chicago time, a naive one is kept."""
    values = values.astype("str").str.strip()
    out = np.full(len(values), NAT, dtype=np.int64)
    aware = _offsets(values)
    if aware.any():
        parsed = _parse_times(values[aware], utc=True)
        out[aware] = _ns(parsed.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None))
    if (~aware).any():
        parsed = _parse_times(values[~aware], utc=False)
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            parsed = parsed.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        out[~aware] = _ns(parsed)
    return out


def utc_to_wall(nanoseconds: np.ndarray) -> np.ndarray:
    return _ns(as_utc(nanoseconds).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None))


def station_key(values: pd.Series) -> pd.Series:
    return values.astype("str").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def label_key(values: pd.Series) -> pd.Series:
    return (values.astype("str").str.strip().str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))


def model_key(values: pd.Series) -> pd.Series:
    """Map a model label to persistence_baseline or student_model, whatever its case and spacing.

    When every other row carries one single label, such as the regressor's name, that label is the student model.
    """
    labels = label_key(values).fillna("")
    persistence = (labels.str.contains("persist") | labels.str.contains("baseline")).to_numpy(dtype=bool)
    student = ~persistence & (labels.str.contains("student") | labels.str.contains("model")).to_numpy(dtype=bool)
    others = set(labels[~persistence & ~student])
    if persistence.any() and not student.any() and len(others) == 1:
        student = ~persistence
    return pd.Series(np.select([persistence, student], list(MODELS), default=labels.to_numpy(dtype=object)),
                     index=values.index, dtype="str")


def slug(station: pd.Series) -> pd.Series:
    return station.str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")


def time_label(nanoseconds) -> str:
    if nanoseconds is None or (isinstance(nanoseconds, float) and math.isnan(nanoseconds)) or int(nanoseconds) == NAT:
        return "no readable time"
    stamp = pd.Timestamp(int(nanoseconds), unit="ns", tz="UTC")
    return f"{stamp:%Y-%m-%d %H:%M} UTC ({stamp.tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M %Z})"


def number_label(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "missing"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def flag_label(value) -> str:
    return "missing" if value is None or (isinstance(value, float) and math.isnan(value)) else str(bool(value))


def describe_key(key) -> str:
    key = str(key)
    if "|" in key:
        left, right = key.split("|", 1)
        if right.startswith("W") and right[1:].lstrip("-").isdigit():
            return f"{left.title()} at {pd.Timestamp(int(right[1:]), unit='ns'):%Y-%m-%d %H:%M} Chicago time"
        if right.lstrip("-").isdigit() and len(right) > 12:
            return f"{left.title()} at {time_label(int(right))}"
        return f"{left} {right}"
    return key


# ---------------------------------------------------------------------------
# The trusted release and the reference pipeline. The same builders rebuild a
# downstream table from a student's own upstream file.


def _verify_release() -> tuple[pd.DataFrame, dict]:
    release, manifest_path = DATA_DIR / RELEASE_NAME, DATA_DIR / MANIFEST_NAME
    if not release.is_file() or not manifest_path.is_file():
        raise InfrastructureError(f"the supplied release is missing from {DATA_DIR}")
    release_bytes = release.read_bytes()
    if sha256(release_bytes).hexdigest() != RELEASE_SHA256:
        raise InfrastructureError(f"{release} differs from the frozen release (SHA-256 mismatch)")
    if sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise InfrastructureError(f"{manifest_path} differs from the frozen manifest (SHA-256 mismatch)")
    raw = pd.read_csv(io.BytesIO(release_bytes), dtype={"station_name": "str", "measurement_timestamp": "str"})
    if raw.columns.tolist() != RAW_COLUMNS or len(raw) != RELEASE_ROWS:
        raise InfrastructureError("the frozen release has an unexpected shape")
    return raw, json.loads(manifest_path.read_text(encoding="utf-8"))


def build_clean(raw: pd.DataFrame, stations: list[str]) -> tuple[pd.DataFrame, dict]:
    """The Q2 cleaned table (station, wall, utc, sensors) and the audit totals by result."""
    naive = pd.to_datetime(raw["measurement_timestamp"], errors="coerce")
    local = naive.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
    valid = raw["station_name"].isin(stations) & local.notna() & local.ge(WINDOW[0]) & local.lt(WINDOW[1])
    valid &= ~pd.DataFrame({"s": raw["station_name"], "t": local}).duplicated(keep=False)
    clean = pd.DataFrame({
        "station": station_key(raw.loc[valid, "station_name"]).to_numpy(),
        "wall": _ns(naive[valid]),
        "utc": _ns(local[valid].dt.tz_convert("UTC").dt.tz_localize(None)),
    })
    totals = {"rows_rejected": int((~valid).sum()), "set_missing": 0, "set_to_zero": 0}
    for column in SENSOR_COLUMNS:
        values = pd.to_numeric(raw.loc[valid, column], errors="coerce").to_numpy(dtype=float, copy=True)
        with np.errstate(invalid="ignore"):
            if column == "precipitation_type_code":
                invalid = ~np.isnan(values) & ~np.isin(values, PRECIPITATION_CODES)
            else:
                low, high = RANGES[column]
                invalid = ~np.isnan(values) & ((values < low) | (values > high))
            totals["set_missing"] += int(invalid.sum())
            values[invalid] = np.nan
            if column == "solar_radiation_w_m2":
                near_zero = (values >= -20) & (values < 0)
                totals["set_to_zero"] += int(near_zero.sum())
                values[near_zero] = 0.0
        clean[column] = values
    return clean, totals


def build_missingness(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, group in clean.groupby("station", sort=True):
        for column in SENSOR_COLUMNS:
            count = int(np.isnan(group[column].to_numpy(dtype=float)).sum())
            rows.append({"key": f"{station}|{column}", "missing_count": count,
                         "missing_pct": count / len(group) * 100})
    return pd.DataFrame(rows).set_index("key")


def hour_grid(stations: list[str]) -> pd.DataFrame:
    hours = pd.date_range(WINDOW[0].tz_convert("UTC"), WINDOW[1].tz_convert("UTC"), freq="h", inclusive="left")
    ns = _ns(pd.Series(hours.tz_localize(None)))
    keys = sorted(station_key(text_series(stations)).tolist())
    return pd.DataFrame({"station": np.repeat(keys, len(ns)), "utc": np.tile(ns, len(keys))})


def build_panel(clean: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    """The Q3 station-hour panel from a cleaned table with station, utc, and sensor columns."""
    observed = clean[["station", "utc", *SENSOR_COLUMNS]].drop_duplicates(["station", "utc"]).copy()
    observed["source_observed"] = 1.0
    panel = hour_grid(stations).merge(observed, on=["station", "utc"], how="left")
    panel["source_observed"] = panel["source_observed"].fillna(0.0)
    local = as_utc(panel["utc"]).dt.tz_convert(LOCAL_TZ)
    panel["hour"] = local.dt.hour.to_numpy(dtype=float)
    panel["day_of_week"] = local.dt.dayofweek.to_numpy(dtype=float)
    panel["month"] = local.dt.month.to_numpy(dtype=float)
    return panel


def gap_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, group in panel.sort_values(["station", "utc"], kind="stable").groupby("station", sort=True):
        observed = group["source_observed"].to_numpy() == 1.0
        missing = ~observed
        starts = missing & np.r_[True, ~missing[:-1]]
        runs = int(starts.sum())
        longest = int(pd.Series(np.cumsum(starts)[missing]).value_counts().max()) if runs else 0
        rows.append({"station": station, "expected_hours": len(group), "observed_hours": int(observed.sum()),
                     "missing_hours": int(missing.sum()), "gap_runs": runs, "longest_gap_hours": longest})
    return pd.DataFrame(rows).set_index("station")


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """The Q4 feature table from a panel with station, utc, and sensor columns."""
    work = (panel.loc[panel["utc"] != NAT].drop_duplicates(["station", "utc"])
            .sort_values(["station", "utc"], kind="stable").reset_index(drop=True))
    air = work.groupby("station", sort=False)["air_temperature_c"]
    cutoff = as_utc(work["utc"])
    features = pd.DataFrame({"station": work["station"], "cutoff": work["utc"]})
    features["row_id"] = slug(work["station"]) + "_" + cutoff.dt.strftime("%Y%m%d%H")
    features["target_ts"] = work["utc"] + HOUR_NS
    features[TARGET] = air.shift(-1).to_numpy()
    features["model_eligible"] = (work["air_temperature_c"].notna() & air.shift(-1).notna()).to_numpy(dtype=float)
    for feature, source in CURRENT_SOURCES.items():
        features[feature] = work[source].to_numpy(dtype=float)
    radians = np.deg2rad(work["wind_direction_deg"].to_numpy(dtype=float))
    features["wind_direction_sin_t"], features["wind_direction_cos_t"] = np.sin(radians), np.cos(radians)
    for feature, hours in zip(LAG_FEATURES, (1, 24, 168)):
        features[feature] = air.shift(hours).to_numpy()
    features[MEAN_FEATURE] = air.transform(lambda values: values.rolling(24, min_periods=1).mean()).to_numpy()
    features[CHANGE_FEATURE] = features["air_temperature_c_t"] - features["air_temperature_lag_1h_c"]
    target_local = as_utc(features["target_ts"]).dt.tz_convert(LOCAL_TZ)
    hour = target_local.dt.hour.to_numpy(dtype=float)
    day = target_local.dt.dayofyear.to_numpy(dtype=float) - 1
    features["target_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["target_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["target_day_of_year_sin"] = np.sin(2 * np.pi * day / 366)
    features["target_day_of_year_cos"] = np.cos(2 * np.pi * day / 366)
    return features


def split_of(target_ns) -> np.ndarray:
    local = as_utc(target_ns).dt.tz_convert(LOCAL_TZ)
    return np.select([local.lt(VALIDATION_START).to_numpy(dtype=bool), local.lt(TEST_START).to_numpy(dtype=bool),
                      local.notna().to_numpy(dtype=bool)], ["train", "validation", "test"], default="")


def build_training(features: pd.DataFrame) -> pd.DataFrame:
    eligible = features.loc[features["model_eligible"].to_numpy() == 1.0]
    return eligible.loc[split_of(eligible["target_ts"]) == "train"]


def build_monthly(features: pd.DataFrame) -> pd.DataFrame:
    training = build_training(features).copy()
    local = as_utc(training["target_ts"]).dt.tz_convert(LOCAL_TZ)
    training["year"], training["month"] = local.dt.year.to_numpy(), local.dt.month.to_numpy()
    grouped = training.groupby(["station", "year", "month"], sort=True)[TARGET]
    monthly = grouped.agg(n_observed="count", mean_air_temperature_c="mean", min_air_temperature_c="min",
                          max_air_temperature_c="max")
    monthly["std_air_temperature_c"] = grouped.std(ddof=1)
    monthly["std_population"] = grouped.std(ddof=0)
    monthly.index = [f"{station}|{int(year)}-{int(month)}" for station, year, month in monthly.index]
    return monthly


def build_correlations(features: pd.DataFrame) -> pd.DataFrame:
    return build_training(features)[CURRENT_FEATURES].corr(method="pearson")


def build_splits(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    eligible = features.loc[features["model_eligible"].to_numpy() == 1.0]
    labels = split_of(eligible["target_ts"])
    return {split: eligible.loc[labels == split].drop_duplicates("row_id").set_index("row_id") for split in SPLITS}


def summarize_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {split: {"n_rows": len(frame), "target_start": int(frame["target_ts"].min()) if len(frame) else NAT,
                    "target_end": int(frame["target_ts"].max()) if len(frame) else NAT, "n_features": len(FEATURES)}
            for split, frame in splits.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def metric_rows(actual: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    total = float(((actual - actual.mean()) ** 2).sum()) if len(actual) else 0.0
    rows = {}
    for model, predicted in predictions.items():
        residual = predicted - actual
        rows[model] = {
            "mae": float(np.abs(residual).mean()) if len(actual) else math.nan,
            "rmse": float(np.sqrt((residual ** 2).mean())) if len(actual) else math.nan,
            "r2": 1.0 - float((residual ** 2).sum()) / total if total else math.nan,
            "n": float(len(actual)),
        }
    return rows


@dataclass
class Reference:
    manifest: dict
    stations: list[str]
    clean: pd.DataFrame
    totals: dict
    missingness: pd.DataFrame
    panel: pd.DataFrame
    summary: pd.DataFrame
    features: pd.DataFrame
    monthly: pd.DataFrame
    correlations: pd.DataFrame
    splits: dict
    split_summary: pd.DataFrame


_REFERENCE: Reference | None = None


def reference() -> Reference:
    global _REFERENCE
    if _REFERENCE is None:
        raw, manifest = _verify_release()
        stations = list(manifest["stations"])
        clean, totals = build_clean(raw, stations)
        panel = build_panel(clean, stations)
        features = build_features(panel)
        splits = build_splits(features)
        _REFERENCE = Reference(manifest, stations, clean, totals, build_missingness(clean), panel,
                               gap_summary(panel), features, build_monthly(features),
                               build_correlations(features), splits, summarize_splits(splits))
    return _REFERENCE


# ---------------------------------------------------------------------------
# Submitted files


@dataclass
class Table:
    """One submitted CSV as stripped text cells under lower-case header names."""

    name: str
    frame: pd.DataFrame

    def has(self, column: str) -> bool:
        return column in self.frame.columns

    def missing(self, columns: list[str]) -> list[str]:
        return [column for column in columns if not self.has(column)]

    def text(self, column: str) -> pd.Series:
        return self.frame[column]


def read_table(root: Path, name: str, keep_first: bool = False) -> tuple[Table | None, str]:
    """Read output/<name> as text cells; return (table, problem).

    Cells may be separated by commas, semicolons, or tabs: whichever splits the
    header line into the most cells is used, and a tie keeps commas. A file
    separated by semicolons may write decimal commas.
    """
    path = root / "output" / name
    task = ARTIFACTS[name][1]
    if path.is_symlink() or not path.is_file():
        return None, f"output/{name} is missing; make it in {task}."
    text = path.read_bytes().decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return None, f"output/{name} is empty; make it, with its header line and rows, in {task}."
    try:
        header = next((line for line in text.split("\n") if line.strip()), "")
        delimiter = max(DELIMITERS, key=lambda mark: len(next(csv.reader([header], delimiter=mark), [])))
        frame = pd.read_csv(io.StringIO(text), sep=delimiter, dtype="str", keep_default_na=False,
                            skip_blank_lines=True)
    except Exception as error:  # noqa: BLE001 - any parse failure is reported to the student
        return None, f"output/{name} cannot be read as a CSV ({error}); save it with to_csv() in {task}."
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame = frame.apply(lambda column: column.astype("str").str.strip())
    if delimiter == ";":
        frame = frame.apply(lambda column: column.str.replace(DECIMAL_COMMA, r"\1.\2", regex=True))
    if not keep_first and len(frame.columns) > 1:
        first = frame.columns[0]
        if first == "" or first.startswith("unnamed:"):
            frame = frame.iloc[:, 1:]
    return Table(name, frame.reset_index(drop=True)), ""


@dataclass
class Keyed:
    """A submitted table with one text key per row; `unique` keeps the first row of each readable key."""

    keys: pd.Series
    frame: pd.DataFrame
    unreadable: int = 0
    duplicates: int = 0
    unique: pd.DataFrame = field(default_factory=pd.DataFrame)
    _parsed: dict = field(default_factory=dict)

    def parsed(self, column: str, parse) -> tuple[np.ndarray, np.ndarray]:
        key = (column, parse)
        if key not in self._parsed:
            self._parsed[key] = parse(self.unique[column])
        return self._parsed[key]


def keyed(frame: pd.DataFrame, keys: pd.Series) -> Keyed:
    keys = pd.Series(np.asarray(keys, dtype=object), index=frame.index).astype("str")
    readable = keys.notna().to_numpy(dtype=bool)
    first = readable & ~keys.duplicated().to_numpy(dtype=bool)
    unique = frame.loc[first].set_axis(pd.Index(keys[first].to_numpy(), dtype="str"))
    return Keyed(keys, frame, int((~readable).sum()), int(readable.sum() - first.sum()), unique)


def station_time_key(station: pd.Series, nanoseconds, wall: bool = False) -> pd.Series:
    """One text key per row: station and UTC nanoseconds, or Chicago wall-clock nanoseconds marked with W."""
    times = pd.Series(np.asarray(nanoseconds, dtype=np.int64), index=station.index)
    keys = station.astype("str") + ("|W" if wall else "|") + times.astype(str)
    return keys.where((times != NAT) & station.notna())


class Submission:
    """The files of one submission, parsed once and shared by the checks."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, object] = {}

    def cached(self, key: str, build: Callable[[], object]) -> object:
        if key not in self._cache:
            try:
                self._cache[key] = build()
            except Exception:  # noqa: BLE001 - an unusable upstream file only removes an alternative
                self._cache[key] = None
        return self._cache[key]

    def table(self, name: str, keep_first: bool = False) -> tuple[Table | None, str]:
        result = self.cached(f"table:{name}:{keep_first}", lambda: read_table(self.root, name, keep_first))
        return result if result is not None else (None, f"output/{name} cannot be read; make it again in {ARTIFACTS[name][1]}.")

    def clean(self) -> pd.DataFrame | None:
        """The student's Q2 table as station, wall, utc, and sensor values."""
        def build():
            table, _ = self.table("q2_cleaned_observations.csv")
            if table is None or not table.has("station_name"):
                return None
            frame = pd.DataFrame({"station": station_key(table.text("station_name"))})
            utc = instants(table.text("measurement_timestamp_utc")) if table.has("measurement_timestamp_utc") else None
            options = []
            if table.has("measurement_timestamp"):
                options.append(wall_times(table.text("measurement_timestamp")))
            if utc is not None:
                options.append(utc_to_wall(utc))
            if not options:
                return None
            expected = set(station_time_key(reference().clean["station"], reference().clean["wall"], wall=True))
            frame["wall"] = max(options, key=lambda wall: len(set(station_time_key(frame["station"], wall, wall=True)
                                                                  .dropna()) & expected))
            frame["utc"] = utc if utc is not None else NAT
            for column in SENSOR_COLUMNS:
                frame[column] = numbers(table.text(column))[0] if table.has(column) else np.nan
            return frame
        return self.cached("clean", build)

    def panel(self) -> pd.DataFrame | None:
        def build():
            table, _ = self.table("q3_hourly_panel.csv")
            if table is None or table.missing(["station_name", "measurement_timestamp_utc"]):
                return None
            frame = pd.DataFrame({"station": station_key(table.text("station_name")),
                                  "utc": instants(table.text("measurement_timestamp_utc"))})
            for column in SENSOR_COLUMNS:
                frame[column] = numbers(table.text(column))[0] if table.has(column) else np.nan
            frame["source_observed"] = flags(table.text("source_observed"))[0] if table.has("source_observed") else np.nan
            return frame
        return self.cached("panel", build)

    def features(self) -> pd.DataFrame | None:
        def build():
            table, _ = self.table("q4_features.csv")
            if table is None or table.missing(["row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc"]):
                return None
            frame = pd.DataFrame({
                "station": station_key(table.text("station_name")),
                "cutoff": instants(table.text("cutoff_timestamp_utc")),
                "row_id": table.text("row_id").str.lower(),
                "target_ts": instants(table.text("target_timestamp_utc")),
            })
            frame[TARGET] = numbers(table.text(TARGET))[0] if table.has(TARGET) else np.nan
            frame["model_eligible"] = flags(table.text("model_eligible"))[0] if table.has("model_eligible") else np.nan
            for column in NUMERIC_FEATURES:
                frame[column] = numbers(table.text(column))[0] if table.has(column) else np.nan
            return frame.loc[frame["target_ts"] != NAT]
        return self.cached("features", build)

    def split_files(self, split: str) -> pd.DataFrame | None:
        """The student's own Q6 X and y for one split, keyed by row_id."""
        def build():
            x_table, _ = self.table(f"q6_X_{split}.csv")
            if x_table is None or not x_table.has("row_id"):
                return None
            frame = pd.DataFrame({"row_id": x_table.text("row_id").str.lower()})
            frame["station"] = station_key(x_table.text("station_name")) if x_table.has("station_name") else ""
            frame["target_ts"] = (instants(x_table.text("target_timestamp_utc"))
                                  if x_table.has("target_timestamp_utc") else NAT)
            frame["air_temperature_c_t"] = (numbers(x_table.text("air_temperature_c_t"))[0]
                                            if x_table.has("air_temperature_c_t") else np.nan)
            frame = frame.drop_duplicates("row_id").set_index("row_id")
            y_table, _ = self.table(f"q6_y_{split}.csv")
            frame[TARGET] = np.nan
            if y_table is not None and not y_table.missing(["row_id", TARGET]):
                y = pd.Series(numbers(y_table.text(TARGET))[0], index=y_table.text("row_id").str.lower())
                frame[TARGET] = y[~y.index.duplicated()].reindex(frame.index).to_numpy()
            return frame
        return self.cached(f"split:{split}", build)

    def predictions(self, name: str) -> pd.DataFrame | None:
        def build():
            table, _ = self.table(name)
            columns = ["actual", "persistence_prediction", "model_prediction"]
            if table is None or table.missing(columns):
                return None
            frame = pd.DataFrame({"station": station_key(table.text("station_name")) if table.has("station_name") else ""})
            for column in columns:
                frame[column] = numbers(table.text(column))[0]
            return frame
        return self.cached(f"predictions:{name}", build)


# ---------------------------------------------------------------------------
# Comparison helpers


def same(got, want, kind: str = "number", tolerance: float = TOLERANCE) -> np.ndarray:
    if kind == "text":
        got = pd.Series(np.asarray(got, dtype=object)).fillna("<missing>").astype(str).to_numpy()
        want = pd.Series(np.asarray(want, dtype=object)).fillna("<missing>").astype(str).to_numpy()
        return got == want
    if kind == "time":
        return np.asarray(got, dtype=np.int64) == np.asarray(want, dtype=np.int64)
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    both_missing = np.isnan(got) & np.isnan(want)
    if kind == "exact":
        return both_missing | (got == want)
    finite = np.isfinite(got) & np.isfinite(want)
    with np.errstate(invalid="ignore"):
        close = finite & (np.abs(got - want) <= tolerance + 1e-9 * np.abs(np.where(finite, want, 0.0)))
    return both_missing | close | (got == want)


@dataclass
class ColumnResult:
    right: int
    total: int
    detail: str

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.right == self.total


def parse_text(cells: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    return cells.str.lower().to_numpy(dtype=object), np.zeros(len(cells), dtype=bool)


def parse_station(cells: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    return station_key(cells).to_numpy(dtype=object), np.zeros(len(cells), dtype=bool)


def parse_time(cells: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = instants(cells)
    return values, values == NAT


LABELS = {"number": number_label, "exact": number_label, "time": time_label, "text": str}


def compare_column(found: Keyed, column: str, candidates: list[pd.DataFrame | None], kind: str,
                   parse, name: str, tolerance: float = TOLERANCE, label=None) -> ColumnResult:
    """Compare one submitted column with each candidate expectation over the rows both have; keep the best."""
    if column not in found.unique.columns:
        return ColumnResult(0, 1, f"column {column} is missing from output/{name}")
    got_all, invalid_all = found.parsed(column, parse)
    label = label or LABELS[kind]
    best: ColumnResult | None = None
    for candidate in candidates:
        if candidate is None or column not in candidate.columns or not len(candidate):
            continue
        shared = candidate.index.intersection(found.unique.index)
        if len(shared) < MIN_PRESENT * len(candidate):
            result = ColumnResult(0, len(candidate), f"only {len(shared):,} of the {len(candidate):,} expected rows "
                                                     f"are in output/{name}, too few to judge {column}")
        else:
            positions = found.unique.index.get_indexer(shared)
            got, invalid = got_all[positions], invalid_all[positions]
            want = candidate[column].reindex(shared).to_numpy()
            right = same(got, want, kind, tolerance) & ~invalid
            detail = ""
            if not right.all():
                first = int(np.flatnonzero(~right)[0])
                shown = found.unique[column].iloc[positions[first]] or "an empty cell"
                detail = (f"{column} differs in {int((~right).sum()):,} of {len(shared):,} rows; first at "
                          f"{describe_key(shared[first])}: expected {label(want[first])}, found {shown}")
            result = ColumnResult(int(right.sum()), len(shared), detail)
        if best is None or result.ok or (not best.ok and result.right / result.total > best.right / best.total):
            best = result
        if best.ok:
            break
    return best or ColumnResult(0, 1, f"no expected values are available for {column}")


def rows_check(points: int, found: Keyed, expected: pd.Index, name: str, what: str, task: str,
               alternatives: list[pd.Index] = ()) -> tuple[int, str]:
    """Half the points for every expected row present, half for no extra, repeated, or unreadable rows.

    A one-point rows check needs both. An alternative row set (from the student's own upstream file)
    is accepted in place of the reference when it fits better.
    """
    present = pd.Index(found.unique.index)
    best = None
    for option in [expected, *alternatives]:
        missing, extra = option.difference(present), present.difference(option)
        score = int(not len(missing)) + int(not len(extra) and not found.duplicates and not found.unreadable)
        if best is None or score > best[0]:
            best = (score, option, missing, extra)
    score, option, missing, extra = best
    if score == 2:
        return points, ""
    problems = []
    if len(missing):
        problems.append(f"{len(missing):,} of the {len(option):,} expected rows are missing "
                        f"(first: {describe_key(missing[0])})")
    if len(extra):
        problems.append(f"{len(extra):,} rows are not expected (first: {describe_key(extra[0])})")
    if found.duplicates:
        problems.append(f"{found.duplicates:,} rows repeat a key already listed")
    if found.unreadable:
        problems.append(f"{found.unreadable:,} rows have a key that cannot be read")
    earned = 0 if points == 1 else points * score // 2
    return earned, f"output/{name}: {'; '.join(problems)}. Expected one row for each of the {what}; fix {task}."


def proportional(points: int, right: int, total: int) -> int:
    if total <= 0:
        return 0
    return points if right >= total else points * right // total


def cell_matches(cell: str, want, kind: str, candidate: pd.DataFrame | None = None, key=None,
                 column: str = "") -> bool:
    series = text_series([cell])
    if kind == "time":
        return int(want) in (int(instants(series, "utc")[0]), int(instants(series, "local")[0]))
    value = numbers(series)[0]
    if kind == "percent":
        return bool(same(value, [want], "number", PERCENT_TOLERANCE)[0]
                    or same(value, [want / 100], "number", PERCENT_TOLERANCE / 100)[0])
    if same(value, [want], kind)[0]:
        return True
    if column == "std_air_temperature_c" and candidate is not None and "std_population" in candidate.columns:
        return bool(same(value, [candidate.loc[key, "std_population"]], "number")[0])
    return False


def cells_check(points: int, table: Table, keys: pd.Series, candidates: list[pd.DataFrame | None],
                columns: dict[str, str], expected_keys, name: str, task: str, advice: str,
                unverifiable: frozenset = frozenset()) -> tuple[int, str]:
    """Score every (row, column) cell of a small table against any candidate; extra rows count against it.

    A (key, column) cell in `unverifiable`, which no file can check, needs only hold a finite number.
    """
    rows = keyed(table.frame, keys).unique
    right, total, problems = 0, 0, []
    for key in expected_keys:
        for column, kind in columns.items():
            total += 1
            if key not in rows.index or column not in rows.columns:
                problems.append(f"{describe_key(key)} {column} is missing")
                continue
            cell = rows.loc[key, column]
            if (key, column) in unverifiable:
                if np.isfinite(numbers(text_series([cell]))[0][0]):
                    right += 1
                else:
                    problems.append(f"{describe_key(key)} {column}: expected a number, found {cell or 'an empty cell'}")
                continue
            if any(candidate is not None and key in candidate.index and column in candidate.columns
                   and cell_matches(cell, candidate.loc[key, column], kind, candidate, key, column)
                   for candidate in candidates):
                right += 1
            else:
                reference_value = candidates[0].loc[key, column] if key in candidates[0].index else None
                shown = time_label(reference_value) if kind == "time" else number_label(reference_value)
                problems.append(f"{describe_key(key)} {column}: expected {shown}, found {cell or 'an empty cell'}")
    extra = len(set(rows.index) - {str(key) for key in expected_keys})
    total += extra * len(columns)
    if extra:
        problems.append(f"{extra} rows are not expected")
    if not problems:
        return points, ""
    return proportional(points, right, total), (
        f"output/{name}: {right} of {total} values right; {'; '.join(problems[:3])}"
        f"{' ...' if len(problems) > 3 else ''}. {advice}; fix {task}.")


def valid_png(root: Path, name: str) -> tuple[bool, str]:
    path = root / "output" / name
    task = ARTIFACTS[name][1]
    if path.is_symlink() or not path.is_file():
        return False, f"output/{name} is missing; save it with plt.savefig() in {task}."
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
        return False, f"output/{name} is not a PNG image; save the figure with plt.savefig('output/{name}') in {task}."
    width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if width < 50 or height < 50:
        return False, f"output/{name} is only {width}x{height} pixels; save the whole figure in {task}."
    return True, ""


def join(*details: str, limit: int = 3) -> str:
    details = [detail.rstrip(".") for detail in details if detail]
    return "; ".join(details[:limit]) + (" ..." if len(details) > limit else "")


def is_number(value, wanted: float) -> bool:
    parsed = numbers(text_series([value]))[0][0]
    return not math.isnan(parsed) and parsed == float(wanted)


def name_list(value: str) -> list[str]:
    """Column names joined with | or printed as a list, such as ['station_name', 'measurement_timestamp']."""
    text = value.strip().strip("[]()")
    return [part.strip().strip("'\"").strip().lower() for part in re.split(r"[|,]", text) if part.strip()]


# ---------------------------------------------------------------------------
# Q1


def check_release_audit(sub: Submission, points: int) -> tuple[int, str]:
    name = "q1_release_audit.csv"
    task = ARTIFACTS[name][1]
    table, problem = sub.table(name)
    if table is None:
        return 0, problem
    if not table.has("check_name"):
        return 0, f"output/{name} has no check_name column; its header is check_name,expected,observed,passed ({task})."
    manifest = reference().manifest
    tests = {
        "release_filename": lambda v: Path(v.replace("\\", "/")).name.lower() == manifest["release_filename"].lower(),
        "release_sha256": lambda v: v.lower() == manifest["release_sha256"],
        "release_byte_size": lambda v: is_number(v, manifest["release_byte_size"]),
        "row_count": lambda v: is_number(v, manifest["row_count"]),
        "column_count": lambda v: is_number(v, manifest["column_count"]),
        "column_names": lambda v: name_list(v) == [c.lower() for c in manifest["columns"]],
        "source_timezone": lambda v: v.lower() == manifest["source_timezone"].lower(),
    }
    rows = keyed(table.frame, label_key(table.text("check_name"))).unique
    right, problems = 0, []
    for check_name, test in tests.items():
        if check_name not in rows.index:
            problems.append(f"no {check_name} row")
            continue
        row = rows.loc[check_name]
        bad = [column for column in ("expected", "observed") if column not in row.index or not test(row[column])]
        if "passed" not in row.index or flags(text_series([row["passed"]]))[0][0] != 1.0:
            bad.append("passed")
        if bad:
            problems.append(f"{check_name} has a wrong {' and '.join(bad)}")
        else:
            right += 1
    extra = len(set(rows.index) - set(tests))
    if extra:
        problems.append(f"{extra} rows name no required check")
    if not problems:
        return points, ""
    return proportional(points, right, len(tests) + extra), (
        f"output/{name}: {right} of 7 checks right; {join(*problems)}. Each row holds the manifest value in "
        f"expected, the value you measured from the CSV file in observed, and passed True; fix {task}.")


def check_coverage(sub: Submission, points: int) -> tuple[int, str]:
    name = "q1_station_coverage.csv"
    task = ARTIFACTS[name][1]
    table, problem = sub.table(name)
    if table is None:
        return 0, problem
    if not table.has("station_name"):
        return 0, f"output/{name} has no station_name column ({task})."
    ref = reference()
    expected = ref.summary[["expected_hours"]].copy()
    counts = ref.clean.groupby("station").size()
    expected["observed_hours"] = counts
    expected["missing_hours"] = expected["expected_hours"] - counts
    expected["coverage_pct"] = counts / expected["expected_hours"] * 100
    expected["first_timestamp"] = ref.clean.groupby("station")["utc"].min()
    expected["last_timestamp"] = ref.clean.groupby("station")["utc"].max()
    columns = {"expected_hours": "exact", "observed_hours": "exact", "missing_hours": "exact",
               "coverage_pct": "percent", "first_timestamp": "time", "last_timestamp": "time"}
    return cells_check(points, table, station_key(table.text("station_name")), [expected], columns,
                       expected.index, name, task,
                       "Count hours from the valid localized station-hour keys over the full release window")


def png_check(name: str):
    def check(sub: Submission, points: int) -> tuple[int, str]:
        ok, detail = valid_png(sub.root, name)
        return (points if ok else 0), detail
    return check


# Q2


def _reference_clean() -> pd.DataFrame:
    ref = reference()
    frame = ref.clean.copy()
    frame.index = pd.Index(station_time_key(frame["station"], frame["wall"], wall=True), dtype="str")
    frame["measurement_timestamp_utc"] = frame["utc"]
    return frame


def _clean_found(sub: Submission) -> tuple[Keyed | None, str]:
    def build():
        table, problem = sub.table("q2_cleaned_observations.csv")
        if table is None:
            return None, problem
        clean = sub.clean()
        if clean is None:
            return None, ("output/q2_cleaned_observations.csv needs its station_name and measurement_timestamp "
                          "columns to match rows; fix Q2 section 2.3.")
        return keyed(table.frame, station_time_key(clean["station"], clean["wall"], wall=True)), ""
    return sub.cached("found:clean", build) or (None, "output/q2_cleaned_observations.csv cannot be read.")


def check_clean_rows(sub: Submission, points: int) -> tuple[int, str]:
    found, problem = _clean_found(sub)
    if found is None:
        return 0, problem
    return rows_check(points, found, sub.cached("ref:clean", _reference_clean).index, "q2_cleaned_observations.csv",
                      "release rows with a valid station and a valid local time (only the six ambiguous "
                      "fall-back rows are rejected)", "Q2 section 2.2")


def check_clean_utc(sub: Submission, points: int) -> tuple[int, str]:
    found, problem = _clean_found(sub)
    if found is None:
        return 0, problem
    result = compare_column(found, "measurement_timestamp_utc", [sub.cached("ref:clean", _reference_clean)],
                            "time", parse_time, "q2_cleaned_observations.csv")
    if result.ok:
        return points, ""
    return proportional(points, result.right, result.total), (
        f"output/q2_cleaned_observations.csv: {result.detail}. Localize measurement_timestamp to America/Chicago "
        f"with ambiguous='NaT' and nonexistent='NaT', then convert to UTC; fix Q2 section 2.2.")


def clean_columns_check(columns: list[str], rule: str):
    def check(sub: Submission, points: int) -> tuple[int, str]:
        found, problem = _clean_found(sub)
        if found is None:
            return 0, problem
        expected = sub.cached("ref:clean", _reference_clean)
        results = [compare_column(found, column, [expected], "number", numbers, "q2_cleaned_observations.csv")
                   for column in columns]
        right = sum(result.ok for result in results)
        if right == len(results):
            return points, ""
        return proportional(points, right, len(results)), (
            f"output/q2_cleaned_observations.csv: {right} of {len(results)} columns right; "
            f"{join(*(r.detail for r in results))}. {rule} Fix Q2 section 2.3.")
    return check


def check_cleaning_audit(sub: Submission, points: int) -> tuple[int, str]:
    name = "q2_cleaning_audit.csv"
    task = ARTIFACTS[name][1]
    table, problem = sub.table(name)
    if table is None:
        return 0, problem
    missing = table.missing(["affected_values", "result"])
    if missing:
        return 0, f"output/{name} is missing column(s) {', '.join(missing)}; its header is rule,affected_values,result ({task})."
    labels = label_key(table.text("result")).fillna("")
    category = np.select(
        [(labels.str.contains("reject") | labels.str.contains("drop") | labels.str.contains("remov")).to_numpy(dtype=bool),
         labels.str.contains("zero").to_numpy(dtype=bool),
         (labels.str.contains("missing") | labels.str.contains("nan")).to_numpy(dtype=bool)],
        ["rows_rejected", "set_to_zero", "set_missing"], default="")
    counts, invalid = numbers(table.text("affected_values"))
    right, problems = 0, []
    for result, wanted in reference().totals.items():
        chosen = (category == result) & ~invalid & ~np.isnan(counts)
        if chosen.any() and counts[chosen].sum() == wanted:
            right += 1
        else:
            found = f"{counts[chosen].sum():g}" if chosen.any() else "no such rows"
            problems.append(f"the {result} rows add up to {found}, expected {wanted:,}")
    unknown = int((category == "").sum())
    if unknown:
        problems.append(f"{unknown} rows have a result other than rows_rejected, set_missing, or set_to_zero")
    if int(invalid.sum()):
        problems.append(f"{int(invalid.sum())} affected_values cells are not numbers")
    if not problems:
        return points, ""
    return proportional(points, right, 3), (
        f"output/{name}: {'; '.join(problems)}. Record every rule with the number of rows or values it changed; "
        f"fix {task}.")


def check_missingness(sub: Submission, points: int) -> tuple[int, str]:
    name = "q2_missingness.csv"
    task = ARTIFACTS[name][1]
    table, problem = sub.table(name)
    if table is None:
        return 0, problem
    missing = table.missing(["station_name", "column_name"])
    if missing:
        return 0, f"output/{name} is missing column(s) {', '.join(missing)} ({task})."
    own = sub.cached("own:missingness", lambda: build_missingness(sub.clean()) if sub.clean() is not None else None)
    keys = station_key(table.text("station_name")) + "|" + table.text("column_name").str.lower()
    return cells_check(points, table, keys, [reference().missingness, own],
                       {"missing_count": "exact", "missing_pct": "percent"}, reference().missingness.index,
                       name, task, "Count missing values in your cleaned table for each station and sensor column, "
                                   "and give them as a percent of that station's cleaned rows")


# Q3


def _indexed(frame: pd.DataFrame | None, time_column: str) -> pd.DataFrame | None:
    if frame is None:
        return None
    frame = frame.copy()
    frame.index = pd.Index(station_time_key(frame["station"], frame[time_column]), dtype="str")
    return frame.loc[frame.index.notna() & ~frame.index.duplicated()]


def _panel_found(sub: Submission) -> tuple[Keyed | None, str]:
    def build():
        name = "q3_hourly_panel.csv"
        table, problem = sub.table(name)
        if table is None:
            return None, problem
        missing = table.missing(["station_name", "measurement_timestamp_utc"])
        if missing:
            return None, f"output/{name} is missing column(s) {', '.join(missing)}; fix Q3 section 3.2."
        keys = station_time_key(station_key(table.text("station_name")),
                                instants(table.text("measurement_timestamp_utc")))
        return keyed(table.frame, keys), ""
    return sub.cached("found:panel", build) or (None, "output/q3_hourly_panel.csv cannot be read.")


def _panel_candidates(sub: Submission) -> list[pd.DataFrame | None]:
    ref = sub.cached("ref:panel", lambda: _indexed(reference().panel, "utc"))
    own = sub.cached("own:panel", lambda: _indexed(build_panel(sub.clean(), reference().stations), "utc")
                     if sub.clean() is not None else None)
    return [ref, own]


def check_panel_rows(sub: Submission, points: int) -> tuple[int, str]:
    found, problem = _panel_found(sub)
    if found is None:
        return 0, problem
    return rows_check(points, found, _panel_candidates(sub)[0].index, "q3_hourly_panel.csv",
                      "station and UTC hour pairs (both stations at every elapsed hour of the window)", "Q3 section 3.2")


def check_panel_sensors(sub: Submission, points: int) -> tuple[int, str]:
    found, problem = _panel_found(sub)
    if found is None:
        return 0, problem
    candidates = _panel_candidates(sub)
    results = [compare_column(found, column, candidates, "number", numbers, "q3_hourly_panel.csv")
               for column in SENSOR_COLUMNS]
    right = sum(result.ok for result in results)
    if right == len(results):
        return points, ""
    return proportional(points, right, len(results)), (
        f"output/q3_hourly_panel.csv: {right} of 13 sensor columns right; {join(*(r.detail for r in results))}. "
        f"Left-join the cleaned rows onto the station-hour grid and leave hours without a source row missing; "
        f"fix Q3 section 3.2.")


def check_source_observed(sub: Submission, points: int) -> tuple[int, str]:
    found, problem = _panel_found(sub)
    if found is None:
        return 0, problem
    result = compare_column(found, "source_observed", _panel_candidates(sub), "exact", flags,
                            "q3_hourly_panel.csv", label=flag_label)
    if result.ok:
        return points, ""
    return proportional(points, result.right, result.total), (
        f"output/q3_hourly_panel.csv: {result.detail}. source_observed is True exactly where a cleaned source row "
        f"exists for that station and hour; fix Q3 section 3.2.")


def calendar_check(column: str, meaning: str):
    def check(sub: Submission, points: int) -> tuple[int, str]:
        found, problem = _panel_found(sub)
        if found is None:
            return 0, problem
        result = compare_column(found, column, _panel_candidates(sub)[:1], "exact", numbers, "q3_hourly_panel.csv")
        if result.ok:
            return points, ""
        return proportional(points, result.right, result.total), (
            f"output/q3_hourly_panel.csv: {result.detail}. {column} is {meaning} of the row's America/Chicago "
            f"local time; fix Q3 section 3.2.")
    return check


def check_panel_summary(sub: Submission, points: int) -> tuple[int, str]:
    name = "q3_panel_summary.csv"
    task = ARTIFACTS[name][1]
    table, problem = sub.table(name)
    if table is None:
        return 0, problem
    if not table.has("station_name"):
        return 0, f"output/{name} has no station_name column ({task})."
    own = sub.cached("own:summary", lambda: gap_summary(sub.panel().dropna(subset=["source_observed"]))
                     if sub.panel() is not None else None)
    columns = {column: "exact" for column in ARTIFACTS[name][0][1:]}
    return cells_check(points, table, station_key(table.text("station_name")), [reference().summary, own], columns,
                       reference().summary.index, name, task,
                       "Count each station's panel hours, observed hours, gap runs, and longest run from source_observed")


# Q4


def _features_found(sub: Submission) -> tuple[Keyed | None, str]:
    def build():
        name = "q4_features.csv"
        table, problem = sub.table(name)
        if table is None:
            return None, problem
        missing = table.missing(["station_name", "cutoff_timestamp_utc"])
        if missing:
            return None, f"output/{name} is missing column(s) {', '.join(missing)}; fix Q4 section 4.2."
        keys = station_time_key(station_key(table.text("station_name")), instants(table.text("cutoff_timestamp_utc")))
        return keyed(table.frame, keys), ""
    return sub.cached("found:features", build) or (None, "output/q4_features.csv cannot be read.")


def _feature_candidates(sub: Submission) -> list[pd.DataFrame | None]:
    def index(features):
        frame = _indexed(features, "cutoff")
        if frame is not None:
            frame["target_timestamp_utc"] = frame["target_ts"]
        return frame
    ref = sub.cached("ref:features", lambda: index(reference().features))
    own = sub.cached("own:features", lambda: index(build_features(sub.panel())) if sub.panel() is not None else None)
    return [ref, own]


def check_feature_rows(sub: Submission, points: int) -> tuple[int, str]:
    found, problem = _features_found(sub)
    if found is None:
        return 0, problem
    return rows_check(points, found, _feature_candidates(sub)[0].index, "q4_features.csv",
                      "station and cutoff-hour pairs (every panel row, eligible or not)", "Q4 section 4.2")


def feature_check(columns: list[str], rule: str, own: bool = True, kind: str = "number", parse=numbers, label=None):
    def check(sub: Submission, points: int) -> tuple[int, str]:
        found, problem = _features_found(sub)
        if found is None:
            return 0, problem
        candidates = _feature_candidates(sub) if own else _feature_candidates(sub)[:1]
        results = [compare_column(found, column, candidates, kind, parse, "q4_features.csv", label=label)
                   for column in columns]
        right = sum(result.ok for result in results)
        if right == len(results):
            return points, ""
        return proportional(points, right, len(results)), (
            f"output/q4_features.csv: {join(*(r.detail for r in results))}. {rule} Fix Q4 section 4.2.")
    return check


def check_manifest(sub: Submission, points: int) -> tuple[int, str]:
    name = "q4_feature_manifest.csv"
    task = ARTIFACTS[name][1]
    table, problem = sub.table(name)
    if table is None:
        return 0, problem
    if not table.has("feature_name"):
        return 0, f"output/{name} has no feature_name column ({task})."
    rows = keyed(table.frame, table.text("feature_name").str.lower()).unique
    right, problems = 0, []
    for feature, (earliest, latest, role) in OFFSETS.items():
        if feature not in rows.index:
            problems.append(f"no row for {feature}")
            continue
        row, bad = rows.loc[feature], []
        for column, want in (("earliest_offset_hours", earliest), ("latest_offset_hours", latest)):
            if column not in row.index or not is_number(row[column], want):
                bad.append(f"{column} {row.get(column, 'missing') or 'blank'}")
        if str(row.get("role", "")).lower() != role:
            bad.append(f"role {row.get('role', 'missing') or 'blank'}")
        if not str(row.get("source", "")).strip():
            bad.append("a blank source")
        if bad:
            problems.append(f"{feature} has {', '.join(bad)}")
        else:
            right += 1
    extra = len(set(rows.index) - set(OFFSETS))
    if extra:
        problems.append(f"{extra} rows name no fixed predictor")
    if not problems:
        return points, ""
    return proportional(points, right, len(OFFSETS) + extra), (
        f"output/{name}: {right} of 19 predictors right; {join(*problems)}. The offsets are the earliest and latest "
        f"hour a feature reads, relative to the cutoff (0 is the cutoff hour; target calendar features read no "
        f"measurement and use 0 and 0), and role is categorical or numeric; fix {task}.")


# Q5


def check_monthly(sub: Submission, points: int) -> tuple[int, str]:
    name = "q5_monthly_station_summary.csv"
    task = ARTIFACTS[name][1]
    table, problem = sub.table(name)
    if table is None:
        return 0, problem
    missing = table.missing(["station_name", "year", "month"])
    if missing:
        return 0, f"output/{name} is missing column(s) {', '.join(missing)} ({task})."
    own = sub.cached("own:monthly", lambda: build_monthly(sub.features()) if sub.features() is not None else None)
    year = pd.Series(numbers(table.text("year"))[0]).fillna(-1).astype(int).astype(str).to_numpy()
    month = pd.Series(numbers(table.text("month"))[0]).fillna(-1).astype(int).astype(str).to_numpy()
    keys = station_key(table.text("station_name")) + "|" + year + "-" + month
    columns = {"n_observed": "exact", **{column: "number" for column in ARTIFACTS[name][0][4:]}}
    return cells_check(points, table, keys, [reference().monthly, own], columns, reference().monthly.index, name,
                       task, "Summarize the target temperatures of the eligible training rows by station and by the "
                             "target's local year and month")


def check_correlations(sub: Submission, points: int) -> tuple[int, str]:
    name = "q5_correlations.csv"
    task = ARTIFACTS[name][1]
    table, problem = sub.table(name, keep_first=True)
    if table is None:
        return 0, problem
    first = table.frame.columns[0]
    if table.has("feature"):
        keys = table.text("feature").str.lower()
    elif first in CURRENT_FEATURES and len(table.frame) == len(CURRENT_FEATURES):
        keys = pd.Series(CURRENT_FEATURES, index=table.frame.index)  # saved without row labels, in fixed order
    else:
        keys = table.text(first).str.lower()
    own = sub.cached("own:corr", lambda: build_correlations(sub.features()) if sub.features() is not None else None)
    return cells_check(points, table, keys, [reference().correlations, own],
                       {column: "number" for column in CURRENT_FEATURES}, CURRENT_FEATURES, name, task,
                       "Save the training rows' Pearson correlation matrix of the seven current predictors, with "
                       "its row labels in a first column named feature")


# Q6


def _split_candidates(sub: Submission, split: str) -> list[pd.DataFrame]:
    def build():
        own_splits = sub.cached("own:splits", lambda: build_splits(sub.features()) if sub.features() is not None else None)
        frames = [reference().splits[split]] + ([own_splits[split]] if own_splits else [])
        candidates = []
        for frame in frames:
            frame = frame.copy()
            frame["station_name"] = frame["station"]
            frame["cutoff_timestamp_utc"] = frame["cutoff"]
            frame["target_timestamp_utc"] = frame["target_ts"]
            candidates.append(frame)
        return candidates
    return sub.cached(f"candidates:split:{split}", build)


def _split_found(sub: Submission, name: str) -> tuple[Keyed | None, str]:
    def build():
        table, problem = sub.table(name)
        if table is None:
            return None, problem
        if not table.has("row_id"):
            return None, f"output/{name} has no row_id column; fix {ARTIFACTS[name][1]}."
        return keyed(table.frame, table.text("row_id").str.lower()), ""
    return sub.cached(f"found:{name}", build) or (None, f"output/{name} cannot be read.")


def check_x_rows(sub: Submission, points: int) -> tuple[int, str]:
    earned, details = 0, []
    for split in SPLITS:
        name = f"q6_X_{split}.csv"
        found, problem = _split_found(sub, name)
        if found is None:
            details.append(problem)
            continue
        candidates = _split_candidates(sub, split)
        score, detail = rows_check(1, found, candidates[0].index, name,
                                   f"eligible Q4 rows whose target time falls in the {split} period",
                                   "Q6 section 6.2", [candidate.index for candidate in candidates[1:]])
        earned += score
        details.append(detail)
    return earned * points // len(SPLITS), join(*details)


X_COLUMNS = [("station_name", "text", parse_station), ("cutoff_timestamp_utc", "time", parse_time),
             ("target_timestamp_utc", "time", parse_time), *[(c, "number", numbers) for c in NUMERIC_FEATURES]]


def check_x_values(sub: Submission, points: int) -> tuple[int, str]:
    right, total, details = 0, 0, []
    for split in SPLITS:
        name = f"q6_X_{split}.csv"
        found, problem = _split_found(sub, name)
        total += len(X_COLUMNS)
        if found is None:
            details.append(problem)
            continue
        candidates = _split_candidates(sub, split)
        for column, kind, parse in X_COLUMNS:
            result = compare_column(found, column, candidates, kind, parse, name)
            right += result.ok
            if not result.ok:
                details.append(f"{name}: {result.detail}")
    if right == total:
        return points, ""
    return proportional(points, right, total), (
        f"{right} of {total} columns right across the three X files; {join(*details)}. Copy each eligible row's "
        f"identifiers and predictors from q4_features.csv unchanged; fix Q6 section 6.3.")


def check_y(sub: Submission, points: int) -> tuple[int, str]:
    right, details = 0, []
    for split in SPLITS:
        name = f"q6_y_{split}.csv"
        found, problem = _split_found(sub, name)
        if found is None:
            details.append(problem)
            continue
        candidates = _split_candidates(sub, split)
        score, detail = rows_check(1, found, candidates[0].index, name,
                                   f"eligible Q4 rows whose target time falls in the {split} period",
                                   "Q6 section 6.3", [candidate.index for candidate in candidates[1:]])
        right += score
        details.append(detail)
        result = compare_column(found, TARGET, candidates, "number", numbers, name)
        right += result.ok
        if not result.ok:
            details.append(f"output/{name}: {result.detail}; copy {TARGET} from q4_features.csv (Q6 section 6.3)")
    if right == 2 * len(SPLITS):
        return points, ""
    return proportional(points, right, 2 * len(SPLITS)), join(*details)


def check_split_summary(sub: Submission, points: int) -> tuple[int, str]:
    name = "q6_split_summary.csv"
    task = ARTIFACTS[name][1]
    table, problem = sub.table(name)
    if table is None:
        return 0, problem
    if not table.has("split"):
        return 0, f"output/{name} has no split column ({task})."
    own = {}
    for split in SPLITS:
        found, _ = _split_found(sub, f"q6_X_{split}.csv")
        if found is not None and "target_timestamp_utc" in found.frame.columns:
            times = instants(found.frame["target_timestamp_utc"])
            times = times[times != NAT]
            if len(times):
                own[split] = {"n_rows": len(found.frame), "target_start": int(times.min()),
                              "target_end": int(times.max()), "n_features": len(FEATURES)}
    columns = {"n_rows": "exact", "target_start": "time", "target_end": "time", "n_features": "exact"}
    return cells_check(points, table, label_key(table.text("split")),
                       [reference().split_summary, pd.DataFrame.from_dict(own, orient="index") if own else None],
                       columns, list(SPLITS), name, task,
                       "Give each split's row count, first and last target time, and the count of all 19 predictors")


# Q7 and Q8


def check_model_spec(sub: Submission, points: int) -> tuple[int, str]:
    name = "q7_model_spec.csv"
    task = ARTIFACTS[name][1]
    table, problem = sub.table(name)
    if table is None:
        return 0, problem
    frame = table.frame
    if not len(frame):
        return 0, f"output/{name} has a header but no row; save its one row in {task}."
    row = frame.iloc[0]
    right, problems = 0, []
    module, estimator = str(row.get("estimator_module", "")), str(row.get("estimator_class", ""))
    if len(frame) == 1 and re.fullmatch(r"sklearn(\.\w+)+", module) and re.fullmatch(r"[A-Za-z_]\w*", estimator):
        right += 1
    else:
        problems.append(f"expected one row naming a scikit-learn module (such as sklearn.linear_model) and a class, "
                        f"found {len(frame)} row(s) with module {module or 'blank'!r} and class {estimator or 'blank'!r}")
    text = str(row.get("parameters_json", ""))
    try:
        parameters = json.loads(text)
    except ValueError:
        try:
            parameters = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parameters = None
    if isinstance(parameters, dict) and parameters.get("random_state", 217) in (217, None) \
            and parameters.get("n_jobs", 1) in (1, None):
        right += 1
    else:
        problems.append("parameters_json must be a JSON object of get_params(deep=False), with random_state 217 and "
                        "n_jobs 1 when the estimator has them")
    names = [part.strip().lower() for part in re.split(r"[|,]", str(row.get("feature_columns", ""))) if part.strip()]
    if sorted(names) == sorted(FEATURES):
        right += 1
    else:
        problems.append("feature_columns must list the 19 fixed predictors joined by |")
    if is_number(row.get("random_state", ""), 217):
        right += 1
    else:
        problems.append(f"random_state must be 217, found {row.get('random_state', 'nothing') or 'a blank'}")
    if not problems:
        return points, ""
    return proportional(points, right, 4), f"output/{name}: {'; '.join(problems)}; fix {task}."


def _prediction_candidates(sub: Submission, split: str) -> list[pd.DataFrame]:
    def build():
        frames = [reference().splits[split]]
        own = sub.split_files(split)
        if own is not None:
            frames.append(own)
        candidates = []
        for frame in frames:
            candidate = pd.DataFrame(index=frame.index)
            candidate["station_name"] = frame["station"]
            candidate["target_timestamp_utc"] = frame["target_ts"]
            candidate["actual"] = frame[TARGET]
            candidate["persistence_prediction"] = frame["air_temperature_c_t"]
            candidates.append(candidate)
        return candidates
    return sub.cached(f"candidates:predictions:{split}", build)


def prediction_check(name: str, split: str, part: str):
    task = ARTIFACTS[name][1]

    def check(sub: Submission, points: int) -> tuple[int, str]:
        found, problem = _split_found(sub, name)
        if found is None:
            return 0, problem
        candidates = _prediction_candidates(sub, split)
        if part == "rows":
            return rows_check(points, found, candidates[0].index, name, f"rows of q6_X_{split}.csv", task,
                              [candidate.index for candidate in candidates[1:]])
        if part == "ids":
            results = [compare_column(found, "station_name", candidates, "text", parse_station, name),
                       compare_column(found, "target_timestamp_utc", candidates, "time", parse_time, name)]
            if all(result.ok for result in results):
                return points, ""
            return 0, (f"output/{name}: {join(*(r.detail for r in results))}. Copy station_name and "
                       f"target_timestamp_utc from q6_X_{split}.csv for each row_id; fix {task}.")
        if part in ("actual", "persistence_prediction"):
            source = f"{TARGET} from q6_y_{split}.csv" if part == "actual" else f"air_temperature_c_t from q6_X_{split}.csv"
            result = compare_column(found, part, candidates, "number", numbers, name)
            if result.ok:
                return points, ""
            return 0, f"output/{name}: {result.detail}. {part} is {source} for the same row_id; fix {task}."
        if part == "model_prediction":
            if "model_prediction" not in found.unique.columns:
                return 0, f"output/{name} is missing column model_prediction; fix {task}."
            coverage = [(candidate.index.intersection(found.unique.index), len(candidate)) for candidate in candidates]
            present, expected = max(coverage, key=lambda item: len(item[0]) / max(item[1], 1))
            if len(present) < MIN_PRESENT * expected:
                return 0, (f"output/{name}: only {len(present):,} of the {expected:,} expected rows are present, "
                           f"too few to judge model_prediction; fix {task}.")
            values, invalid = numbers(found.unique.loc[present, "model_prediction"])
            bad = int((~np.isfinite(values) | invalid).sum())
            if bad:
                return 0, (f"output/{name}: {bad:,} rows have no finite model_prediction; predict every row with "
                           f"your fitted pipeline; fix {task}.")
            return points, ""
        # model_error and model_absolute_error follow from the file's own columns.
        table, _ = sub.table(name)
        missing = table.missing(["actual", "model_prediction", part])
        if missing:
            return 0, f"output/{name} is missing column(s) {', '.join(missing)}; fix {task}."
        if not len(table.frame):
            return 0, f"output/{name} has no rows; fix {task}."
        error = numbers(table.text("model_prediction"))[0] - numbers(table.text("actual"))[0]
        value, invalid = numbers(table.text(part))
        options = [error] if part == "model_error" else [np.abs(error)]
        if part == "model_absolute_error" and table.has("model_error"):
            options.append(np.abs(numbers(table.text("model_error"))[0]))
        ok = np.zeros(len(value), dtype=bool)
        for option in options:
            ok |= same(value, option) & ~invalid & ~np.isnan(option)
        if ok.all():
            return points, ""
        meaning = "model_prediction minus actual" if part == "model_error" else "the absolute value of model_error"
        return 0, (f"output/{name}: {part} differs from {meaning} in {int((~ok).sum()):,} of {len(value):,} rows; "
                   f"fix {task}.")
    return check


def _metric_candidates(sub: Submission, predictions_name: str, split: str, by_station: bool) -> list[pd.DataFrame]:
    """Metrics recomputed from the student's own predictions, and the reference persistence metrics."""
    ref = reference().splits[split]
    sources = []
    own = sub.predictions(predictions_name)
    if own is not None:
        sources.append((own, True))
    sources.append((pd.DataFrame({"station": ref["station"], "actual": ref[TARGET],
                                  "persistence_prediction": ref["air_temperature_c_t"]}), False))
    candidates = []
    for frame, has_model in sources:
        groups = frame.groupby("station", sort=True) if by_station else [("", frame)]
        values = {}
        for station, group in groups:
            columns = {"persistence_baseline": group["persistence_prediction"].to_numpy(dtype=float)}
            if has_model:
                columns["student_model"] = group["model_prediction"].to_numpy(dtype=float)
            for model, row in metric_rows(group["actual"].to_numpy(dtype=float), columns).items():
                values[f"{model}|{station}" if by_station else model] = row
        candidates.append(pd.DataFrame.from_dict(values, orient="index"))
    return candidates


def metrics_check(name: str, predictions_name: str, split: str, by_station: bool):
    task = ARTIFACTS[name][1]

    def check(sub: Submission, points: int) -> tuple[int, str]:
        table, problem = sub.table(name)
        if table is None:
            return 0, problem
        missing = table.missing(["model", "station_name"] if by_station else ["model"])
        if missing:
            return 0, f"output/{name} is missing column(s) {', '.join(missing)}; fix {task}."
        keys = model_key(table.text("model"))
        if by_station:
            keys = keys + "|" + station_key(table.text("station_name"))
        stations = sorted(set(reference().splits[split]["station"]))
        expected = [f"{model}|{station}" for model in MODELS for station in stations] if by_station else list(MODELS)
        candidates = _metric_candidates(sub, predictions_name, split, by_station)
        unverifiable = frozenset()
        if sub.predictions(predictions_name) is None:
            # Without its predictions the student model's errors cannot be recomputed. That file's own checks charge
            # its absence, so here the model's rows need only hold numbers, with the same n as the baseline's.
            reference_rows = candidates[-1]
            for key in list(reference_rows.index):
                reference_rows.loc[key.replace(MODELS[0], MODELS[1], 1), "n"] = reference_rows.loc[key, "n"]
            unverifiable = frozenset((key, metric) for key in expected if key.startswith(MODELS[1])
                                     for metric in METRICS)
        return cells_check(points, table, keys, candidates,
                           {"mae": "number", "rmse": "number", "r2": "number", "n": "exact"}, expected, name, task,
                           f"Compute MAE, RMSE, R2, and n from the rows of {predictions_name}", unverifiable)
    return check


def check_importance(sub: Submission, points: int) -> tuple[int, str]:
    name = "q7_permutation_importance.csv"
    task = ARTIFACTS[name][1]
    table, problem = sub.table(name)
    if table is None:
        return 0, problem
    if not table.has("feature"):
        return 0, f"output/{name} has no feature column ({task})."
    earned, problems = 0, []
    features = table.text("feature").str.lower()
    if sorted(features) == sorted(FEATURES):
        earned += 1
    else:
        missing = sorted(set(FEATURES) - set(features))
        extra = sorted(set(features) - set(FEATURES)) + sorted(set(features[features.duplicated()]))
        problems.append(f"expected one row for each of the 19 fixed predictors; missing {missing or 'none'}, "
                        f"unexpected or repeated {extra or 'none'}")
    columns = ["mean_mae_increase", "std_mae_increase"]
    if table.missing(columns):
        problems.append(f"missing column(s) {', '.join(table.missing(columns))}")
    else:
        mean, std = numbers(table.text(columns[0]))[0], numbers(table.text(columns[1]))[0]
        if len(mean) and np.isfinite(mean).all() and np.isfinite(std).all() and (std >= 0).all():
            earned += 1
        else:
            problems.append("every row needs a finite mean_mae_increase (importances_mean) and a finite, nonnegative "
                            "std_mae_increase (importances_std)")
    if not problems:
        return points, ""
    return earned * points // 2, f"output/{name}: {'; '.join(problems)}; fix {task}."


# Q9


def check_report(sub: Submission, points: int) -> tuple[int, str]:
    """Notes for the human reviewer on report.md; worth no points."""
    path = sub.root / "report.md"
    if path.is_symlink() or not path.is_file():
        return 0, "report.md is missing; Q9 completes the root report.md."
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    problems = []
    headings = [h.strip().lower() for h in re.findall(r"^##\s+(.+?)\s*#*\s*$", text, flags=re.MULTILINE)]
    position = 0
    for heading in REPORT_HEADINGS:
        if heading.lower() in headings[position:]:
            position = headings.index(heading.lower(), position) + 1
        else:
            problems.append(f"the heading '## {heading}' is missing or out of order")
    if re.search(r"\[(?:value|state|summarize|report|describe|replace|compare|give)\b[^\]]*\]", text, flags=re.I):
        problems.append("a bracketed scaffold placeholder remains")
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in text.splitlines()
            if line.strip().startswith("|") and not re.fullmatch(r"[\s:|-]+", line.strip())]
    header = next((i for i, row in enumerate(rows) if [c.lower() for c in row[:2]] == ["evaluation set", "model"]), None)
    if header is None:
        problems.append("no metrics table headed Evaluation set | Model | MAE | RMSE | R2 | n")
    else:
        saved = {}
        for split, file in (("validation", "q7_validation_metrics.csv"), ("test", "q8_test_metrics.csv")):
            table, _ = sub.table(file)
            if table is not None and not table.missing(["model", *METRICS, "n"]):
                for key, (_, row) in zip(model_key(table.text("model")), table.frame.iterrows()):
                    saved[(split, key)] = numbers(text_series([row[c] for c in (*METRICS, "n")]))[0]
        found = {(row[0].lower(), model_key(text_series([row[1]])).iloc[0]): row
                 for row in rows[header + 1:header + 5] if len(row) >= 6}
        for key in [(split, model) for split in ("validation", "test") for model in MODELS]:
            if key not in found:
                problems.append(f"no metrics row for {key[0]} {key[1]}")
            elif key in saved:
                values = numbers(text_series(found[key][2:6]))[0]
                if not (same(values[:3], saved[key][:3], tolerance=0.0051).all() and values[3] == saved[key][3]):
                    problems.append(f"the {key[0]} {key[1]} row differs from the saved metrics file")
    for image in REPORT_IMAGES:
        if not re.search(r"!\[[^\]]*\]\(\s*(?:\./)?" + re.escape(image) + r"\s*\)", text):
            problems.append(f"no image embed for {image}")
        elif not valid_png(sub.root, image.split("/", 1)[1])[0]:
            problems.append(f"{image} is not a readable PNG")
    if problems:
        return 0, f"report.md (for the human reviewer): {'; '.join(problems)}."
    return 0, ""


# ---------------------------------------------------------------------------
# The checks in handout order: (name, points, check).

CHECKS = [
    ("Q1 q1_release_audit.csv", 2, check_release_audit),
    ("Q1 q1_station_coverage.csv", 3, check_coverage),
    ("Q1 q1_visualizations.png", 1, png_check("q1_visualizations.png")),
    ("Q2 q2_cleaned_observations.csv: rows", 2, check_clean_rows),
    ("Q2 q2_cleaned_observations.csv: measurement_timestamp_utc", 2, check_clean_utc),
    ("Q2 q2_cleaned_observations.csv: solar_radiation_w_m2", 1, clean_columns_check(
        ["solar_radiation_w_m2"], "Values from -20 up to but not including 0 become 0, and values outside -20 to 1500 "
                                  "become missing.")),
    ("Q2 q2_cleaned_observations.csv: interval_rain_mm, wind_speed_mps, maximum_wind_speed_mps", 2, clean_columns_check(
        RULE_COLUMNS, "Values outside each column's inclusive valid range become missing.")),
    ("Q2 q2_cleaned_observations.csv: the other nine sensor columns", 1, clean_columns_check(
        UNCHANGED_COLUMNS, "Keep valid values unchanged and set only invalid values to missing; do not fill, "
                           "interpolate, or clip.")),
    ("Q2 q2_cleaning_audit.csv", 2, check_cleaning_audit),
    ("Q2 q2_missingness.csv", 1, check_missingness),
    ("Q3 q3_hourly_panel.csv: rows", 2, check_panel_rows),
    ("Q3 q3_hourly_panel.csv: sensor columns", 2, check_panel_sensors),
    ("Q3 q3_hourly_panel.csv: source_observed", 2, check_source_observed),
    ("Q3 q3_hourly_panel.csv: hour", 1, calendar_check("hour", "the hour (0 to 23)")),
    ("Q3 q3_hourly_panel.csv: day_of_week", 1, calendar_check("day_of_week", "the weekday, Monday 0 to Sunday 6,")),
    ("Q3 q3_hourly_panel.csv: month", 1, calendar_check("month", "the month (1 to 12)")),
    ("Q3 q3_panel_summary.csv", 2, check_panel_summary),
    ("Q4 q4_features.csv: rows", 1, check_feature_rows),
    ("Q4 q4_features.csv: row_id", 1, feature_check(
        ["row_id"], "row_id is the lowercase station name with underscores, then _ and the cutoff UTC hour as "
                    "YYYYMMDDHH.", own=False, kind="text", parse=parse_text)),
    ("Q4 q4_features.csv: target_timestamp_utc", 1, feature_check(
        ["target_timestamp_utc"], "The target time is the cutoff plus one elapsed hour.", own=False, kind="time",
        parse=parse_time)),
    ("Q4 q4_features.csv: target_air_temperature_c", 1, feature_check(
        [TARGET], "The target is the same station's panel air temperature one row (one hour) later.")),
    ("Q4 q4_features.csv: model_eligible", 1, feature_check(
        ["model_eligible"], "A row is eligible when both the cutoff and the next-hour air temperatures are observed.",
        kind="exact", parse=flags, label=flag_label)),
    ("Q4 q4_features.csv: current-hour predictors", 1, feature_check(
        CURRENT_FEATURES, "Each _t column copies the panel value at the cutoff hour.")),
    ("Q4 q4_features.csv: wind direction sine and cosine", 1, feature_check(
        WIND_FEATURES, "Take np.sin and np.cos of 2 * pi * wind_direction_deg / 360.")),
    ("Q4 q4_features.csv: lags", 1, feature_check(
        LAG_FEATURES, "Shift air temperature 1, 24, and 168 rows within each station on the complete panel.")),
    ("Q4 q4_features.csv: air_temperature_mean_past_24h_c", 1, feature_check(
        [MEAN_FEATURE], "Roll 24 rows within each station, including the cutoff row, with min_periods=1.")),
    ("Q4 q4_features.csv: air_temperature_change_1h_c", 1, feature_check(
        [CHANGE_FEATURE], "Subtract the 1-hour lag from the cutoff temperature.")),
    ("Q4 q4_features.csv: target hour sine and cosine", 1, feature_check(
        HOUR_FEATURES, "Use the target's America/Chicago hour with the angle 2 * pi * hour / 24.", own=False)),
    ("Q4 q4_features.csv: target day-of-year sine and cosine", 1, feature_check(
        DAY_FEATURES, "Use the target's America/Chicago day of year with the angle 2 * pi * (dayofyear - 1) / 366.",
        own=False)),
    ("Q4 q4_feature_manifest.csv", 2, check_manifest),
    ("Q5 q5_monthly_station_summary.csv", 3, check_monthly),
    ("Q5 q5_correlations.csv", 2, check_correlations),
    ("Q5 q5_patterns.png", 1, png_check("q5_patterns.png")),
    ("Q6 q6_X_train/validation/test.csv: rows", 3, check_x_rows),
    ("Q6 q6_X_train/validation/test.csv: values", 3, check_x_values),
    ("Q6 q6_y_train/validation/test.csv", 3, check_y),
    ("Q6 q6_split_summary.csv", 2, check_split_summary),
    ("Q7 q7_model_spec.csv", 4, check_model_spec),
    ("Q7 q7_validation_predictions.csv: rows", 1, prediction_check("q7_validation_predictions.csv", "validation", "rows")),
    ("Q7 q7_validation_predictions.csv: station_name and target_timestamp_utc", 1,
     prediction_check("q7_validation_predictions.csv", "validation", "ids")),
    ("Q7 q7_validation_predictions.csv: actual", 1,
     prediction_check("q7_validation_predictions.csv", "validation", "actual")),
    ("Q7 q7_validation_predictions.csv: persistence_prediction", 1,
     prediction_check("q7_validation_predictions.csv", "validation", "persistence_prediction")),
    ("Q7 q7_validation_predictions.csv: model_prediction", 1,
     prediction_check("q7_validation_predictions.csv", "validation", "model_prediction")),
    ("Q7 q7_validation_metrics.csv", 2,
     metrics_check("q7_validation_metrics.csv", "q7_validation_predictions.csv", "validation", False)),
    ("Q7 q7_permutation_importance.csv", 2, check_importance),
    ("Q8 q8_test_predictions.csv: rows", 1, prediction_check("q8_test_predictions.csv", "test", "rows")),
    ("Q8 q8_test_predictions.csv: station_name and target_timestamp_utc", 1,
     prediction_check("q8_test_predictions.csv", "test", "ids")),
    ("Q8 q8_test_predictions.csv: actual", 1, prediction_check("q8_test_predictions.csv", "test", "actual")),
    ("Q8 q8_test_predictions.csv: persistence_prediction", 1,
     prediction_check("q8_test_predictions.csv", "test", "persistence_prediction")),
    ("Q8 q8_test_predictions.csv: model_prediction", 1,
     prediction_check("q8_test_predictions.csv", "test", "model_prediction")),
    ("Q8 q8_test_predictions.csv: model_error", 1, prediction_check("q8_test_predictions.csv", "test", "model_error")),
    ("Q8 q8_test_predictions.csv: model_absolute_error", 1,
     prediction_check("q8_test_predictions.csv", "test", "model_absolute_error")),
    ("Q8 q8_test_metrics.csv", 2, metrics_check("q8_test_metrics.csv", "q8_test_predictions.csv", "test", False)),
    ("Q8 q8_station_metrics.csv", 3, metrics_check("q8_station_metrics.csv", "q8_test_predictions.csv", "test", True)),
    ("Q8 q8_final_visualizations.png", 1, png_check("q8_final_visualizations.png")),
    ("Q9 report.md structure (0 points; human review grades the report)", 0, check_report),
]


def grade_submission(submission_dir: str | Path) -> dict:
    """Grade one submission directory from its committed files."""
    reference()  # raises InfrastructureError when the trusted release is unusable
    submission = Submission(Path(submission_dir).resolve())
    tests = []
    for name, points, check in CHECKS:
        try:
            score, detail = check(submission, points)
        except Exception as error:  # noqa: BLE001 - one unreadable artifact must not stop the other checks
            score, detail = 0, f"this artifact could not be checked ({type(error).__name__}: {error})"
        score = max(0, min(points, int(score)))
        tests.append({"test-name": name, "passed": not detail and score == points, "score": score,
                      "max-score": points, "detail": detail})
    return {"schema": SCHEMA, "score": sum(test["score"] for test in tests),
            "max-score": sum(test["max-score"] for test in tests), "tests": tests}
