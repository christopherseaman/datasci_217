"""Development self-test for the Assignment 11 (final exam) checks.

Course-side QA, not a second grading mode. It answers the exam with pandas,
NumPy, and scikit-learn from `11/assignment/data/` the way the handout asks,
writes submissions in ignored `scratch/`, and confirms what each one scores:

    uv run --python 3.13 --with-requirements 11/assignment/requirements.txt \
        python 11/assignment_checks/_grader_selftest/run.py
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


CHECKS = Path(__file__).resolve().parents[1]
HANDOUT = CHECKS.parent / "assignment"
SCRATCH = CHECKS.parents[1] / "scratch"
sys.path.insert(0, str(CHECKS))
import grading  # noqa: E402

DATA = HANDOUT / "data" / "chicago_beach_sensors_2022_2024.csv"
MANIFEST = HANDOUT / "data" / "release_manifest.json"
TZ = "America/Chicago"
FEATURES = grading.FEATURES
NUMERIC = grading.NUMERIC_FEATURES


def png(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(2, 2))
    axis.plot([0, 1], [0, 1])
    figure.savefig(path)
    plt.close(figure)


# ---------------------------------------------------------------------------
# An independent answer, written the way a student would write it.


def solve(root: Path) -> None:
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw_bytes = DATA.read_bytes()
    raw = pd.read_csv(DATA)
    import hashlib
    observed = {
        "release_filename": DATA.name, "release_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "release_byte_size": len(raw_bytes), "row_count": len(raw), "column_count": raw.shape[1],
        "column_names": "|".join(raw.columns), "source_timezone": TZ,
    }
    expected = {key: ("|".join(manifest["columns"]) if key == "column_names" else manifest[key]) for key in observed}
    pd.DataFrame([{"check_name": key, "expected": expected[key], "observed": observed[key],
                   "passed": str(expected[key]) == str(observed[key])} for key in observed]).to_csv(
        out / "q1_release_audit.csv", index=False)

    local = pd.to_datetime(raw["measurement_timestamp"]).dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    start, end = pd.Timestamp("2022-01-01", tz=TZ), pd.Timestamp("2025-01-01", tz=TZ)
    hours = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")
    keep = local.notna() & raw["station_name"].isin(manifest["stations"])
    clean = raw.loc[keep].copy()
    clean["measurement_timestamp_utc"] = local[keep].dt.tz_convert("UTC")
    coverage = []
    for station, group in clean.groupby("station_name"):
        coverage.append({"station_name": station, "expected_hours": len(hours), "observed_hours": len(group),
                         "missing_hours": len(hours) - len(group), "coverage_pct": len(group) / len(hours) * 100,
                         "first_timestamp": group["measurement_timestamp_utc"].min(),
                         "last_timestamp": group["measurement_timestamp_utc"].max()})
    pd.DataFrame(coverage).to_csv(out / "q1_station_coverage.csv", index=False)

    audit = [{"rule": "ambiguous or nonexistent local time", "affected_values": int((~keep).sum()),
              "result": "rows_rejected"}]
    ranges = {"air_temperature_c": (-50, 50), "wet_bulb_temperature_c": (-50, 50), "relative_humidity_pct": (0, 100),
              "rain_intensity_mm_per_hour": (0, 300), "interval_rain_mm": (0, 100), "total_rain_mm": (0, 2000),
              "wind_direction_deg": (0, 359), "wind_speed_mps": (0, 75), "maximum_wind_speed_mps": (0, 100),
              "barometric_pressure_hpa": (850, 1100), "solar_radiation_w_m2": (-20, 1500),
              "battery_voltage_v": (0, 20)}
    sensors = manifest["columns"][2:]
    for column in sensors:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        if column == "precipitation_type_code":
            bad = clean[column].notna() & ~clean[column].isin([0, 40, 60, 70])
        else:
            low, high = ranges[column]
            bad = clean[column].notna() & ~clean[column].between(low, high)
        audit.append({"rule": f"{column} outside valid values", "affected_values": int(bad.sum()),
                      "result": "set_missing"})
        clean.loc[bad, column] = np.nan
    near_zero = clean["solar_radiation_w_m2"].between(-20, 0, inclusive="left")
    audit.append({"rule": "solar near zero", "affected_values": int(near_zero.sum()), "result": "set_to_zero"})
    clean.loc[near_zero, "solar_radiation_w_m2"] = 0
    clean["measurement_timestamp"] = local[keep]
    clean = clean.sort_values(["measurement_timestamp_utc", "station_name"])
    clean.to_csv(out / "q2_cleaned_observations.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "q2_cleaning_audit.csv", index=False)
    missing = []
    for station, group in clean.groupby("station_name"):
        for column in sensors:
            count = int(group[column].isna().sum())
            missing.append({"station_name": station, "column_name": column, "missing_count": count,
                            "missing_pct": count / len(group) * 100})
    pd.DataFrame(missing).to_csv(out / "q2_missingness.csv", index=False)

    grid = pd.merge(pd.DataFrame({"station_name": manifest["stations"]}),
                    pd.DataFrame({"measurement_timestamp_utc": hours}), how="cross")
    panel = grid.merge(clean[["station_name", "measurement_timestamp_utc", *sensors]],
                       on=["station_name", "measurement_timestamp_utc"], how="left", validate="one_to_one",
                       indicator=True)
    panel["source_observed"] = panel["_merge"].eq("both")
    panel = panel.drop(columns="_merge")
    panel_local = panel["measurement_timestamp_utc"].dt.tz_convert(TZ)
    panel["hour"], panel["day_of_week"], panel["month"] = (panel_local.dt.hour, panel_local.dt.dayofweek,
                                                           panel_local.dt.month)
    panel = panel.sort_values(["measurement_timestamp_utc", "station_name"])
    panel.to_csv(out / "q3_hourly_panel.csv", index=False)
    summary = []
    for station, group in panel.sort_values("measurement_timestamp_utc").groupby("station_name"):
        gap = ~group["source_observed"]
        run_id = (gap != gap.shift()).cumsum()
        runs = gap.groupby(run_id).agg(["first", "size"])
        runs = runs[runs["first"]]
        summary.append({"station_name": station, "expected_hours": len(group),
                        "observed_hours": int(group["source_observed"].sum()), "missing_hours": int(gap.sum()),
                        "gap_runs": len(runs), "longest_gap_hours": int(runs["size"].max())})
    pd.DataFrame(summary).to_csv(out / "q3_panel_summary.csv", index=False)

    panel = panel.sort_values(["station_name", "measurement_timestamp_utc"])
    air = panel.groupby("station_name")["air_temperature_c"]
    f = pd.DataFrame({"station_name": panel["station_name"], "cutoff_timestamp_utc": panel["measurement_timestamp_utc"]})
    f["row_id"] = (f["station_name"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_") + "_"
                   + f["cutoff_timestamp_utc"].dt.strftime("%Y%m%d%H"))
    f["target_timestamp_utc"] = f["cutoff_timestamp_utc"] + pd.Timedelta(hours=1)
    f["target_air_temperature_c"] = air.shift(-1)
    f["model_eligible"] = panel["air_temperature_c"].notna() & f["target_air_temperature_c"].notna()
    for name, source in grading.CURRENT_SOURCES.items():
        f[name] = panel[source]
    radians = np.deg2rad(panel["wind_direction_deg"])
    f["wind_direction_sin_t"], f["wind_direction_cos_t"] = np.sin(radians), np.cos(radians)
    f["air_temperature_lag_1h_c"] = air.shift(1)
    f["air_temperature_lag_24h_c"] = air.shift(24)
    f["air_temperature_lag_168h_c"] = air.shift(168)
    f["air_temperature_mean_past_24h_c"] = air.transform(lambda s: s.rolling(24, min_periods=1).mean())
    f["air_temperature_change_1h_c"] = f["air_temperature_c_t"] - f["air_temperature_lag_1h_c"]
    target_local = f["target_timestamp_utc"].dt.tz_convert(TZ)
    hour_angle = 2 * np.pi * target_local.dt.hour / 24
    day_angle = 2 * np.pi * (target_local.dt.dayofyear - 1) / 366
    f["target_hour_sin"], f["target_hour_cos"] = np.sin(hour_angle), np.cos(hour_angle)
    f["target_day_of_year_sin"], f["target_day_of_year_cos"] = np.sin(day_angle), np.cos(day_angle)
    f = f[grading.ARTIFACTS["q4_features.csv"][0]].sort_values(["cutoff_timestamp_utc", "station_name"])
    f.to_csv(out / "q4_features.csv", index=False)
    pd.DataFrame([{"feature_name": name, "source": f"from {name}", "earliest_offset_hours": early,
                   "latest_offset_hours": late, "role": role}
                  for name, (early, late, role) in grading.OFFSETS.items()]).to_csv(
        out / "q4_feature_manifest.csv", index=False)

    eligible = f[f["model_eligible"]].copy()
    target_local = eligible["target_timestamp_utc"].dt.tz_convert(TZ)
    split = np.where(target_local < pd.Timestamp("2024-01-01", tz=TZ), "train",
                     np.where(target_local < pd.Timestamp("2024-07-01", tz=TZ), "validation", "test"))
    training = eligible[split == "train"].copy()
    training_local = training["target_timestamp_utc"].dt.tz_convert(TZ)
    training["year"], training["month"] = training_local.dt.year, training_local.dt.month
    training.groupby(["station_name", "year", "month"])["target_air_temperature_c"].agg(
        n_observed="count", mean_air_temperature_c="mean", std_air_temperature_c="std",
        min_air_temperature_c="min", max_air_temperature_c="max").reset_index().to_csv(
        out / "q5_monthly_station_summary.csv", index=False)
    training[grading.CURRENT_FEATURES].corr().to_csv(out / "q5_correlations.csv", index=True, index_label="feature")

    frames = {}
    rows = []
    for name in grading.SPLITS:
        part = eligible[split == name].sort_values(["target_timestamp_utc", "station_name"])
        frames[name] = part
        part[["row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc", *NUMERIC]].to_csv(
            out / f"q6_X_{name}.csv", index=False)
        part[["row_id", "target_air_temperature_c"]].to_csv(out / f"q6_y_{name}.csv", index=False)
        rows.append({"split": name, "n_rows": len(part), "target_start": part["target_timestamp_utc"].min(),
                     "target_end": part["target_timestamp_utc"].max(), "n_features": len(FEATURES)})
    pd.DataFrame(rows).to_csv(out / "q6_split_summary.csv", index=False)

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    def pipeline() -> Pipeline:
        prep = ColumnTransformer([
            ("station", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["station_name"]),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC)])
        return Pipeline([("prep", prep), ("model", Ridge(alpha=1.0, random_state=217))])

    def predictions(part: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
        return pd.DataFrame({"row_id": part["row_id"], "station_name": part["station_name"],
                             "target_timestamp_utc": part["target_timestamp_utc"],
                             "actual": part["target_air_temperature_c"],
                             "persistence_prediction": part["air_temperature_c_t"],
                             "model_prediction": model.predict(part[FEATURES])})

    def metrics(frame: pd.DataFrame) -> pd.DataFrame:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        return pd.DataFrame([{"model": model, "mae": mean_absolute_error(frame["actual"], frame[column]),
                              "rmse": np.sqrt(mean_squared_error(frame["actual"], frame[column])),
                              "r2": r2_score(frame["actual"], frame[column]), "n": len(frame)}
                             for model, column in [("persistence_baseline", "persistence_prediction"),
                                                   ("student_model", "model_prediction")]])

    model = pipeline().fit(frames["train"][FEATURES], frames["train"]["target_air_temperature_c"])
    validation = predictions(frames["validation"], model)
    validation.to_csv(out / "q7_validation_predictions.csv", index=False)
    metrics(validation).to_csv(out / "q7_validation_metrics.csv", index=False)
    estimator = model.named_steps["model"]
    pd.DataFrame([{"estimator_module": type(estimator).__module__, "estimator_class": type(estimator).__name__,
                   "parameters_json": json.dumps(estimator.get_params(deep=False)),
                   "feature_columns": "|".join(FEATURES), "random_state": 217}]).to_csv(
        out / "q7_model_spec.csv", index=False)
    importance = permutation_importance(model, frames["validation"][FEATURES],
                                        frames["validation"]["target_air_temperature_c"],
                                        scoring="neg_mean_absolute_error", n_repeats=2, random_state=217)
    pd.DataFrame({"feature": FEATURES, "mean_mae_increase": importance.importances_mean,
                  "std_mae_increase": importance.importances_std}).to_csv(
        out / "q7_permutation_importance.csv", index=False)

    both = pd.concat([frames["train"], frames["validation"]])
    final = pipeline().fit(both[FEATURES], both["target_air_temperature_c"])
    test = predictions(frames["test"], final)
    test["model_error"] = test["model_prediction"] - test["actual"]
    test["model_absolute_error"] = test["model_error"].abs()
    test.to_csv(out / "q8_test_predictions.csv", index=False)
    metrics(test).to_csv(out / "q8_test_metrics.csv", index=False)
    station_rows = []
    for station, group in test.groupby("station_name"):
        for _, row in metrics(group).iterrows():
            station_rows.append({"model": row["model"], "station_name": station, "n": row["n"], "mae": row["mae"],
                                 "rmse": row["rmse"], "r2": row["r2"]})
    pd.DataFrame(station_rows).to_csv(out / "q8_station_metrics.csv", index=False)
    for name in ("q1_visualizations.png", "q5_patterns.png", "q8_final_visualizations.png"):
        png(out / name)
    table = ["| Evaluation set | Model | MAE | RMSE | R2 | n |", "|---|---|---:|---:|---:|---:|"]
    for label, frame in (("Validation", metrics(validation)), ("Test", metrics(test))):
        table += [f"| {label} | {r.model} | {r.mae:.3f} | {r.rmse:.3f} | {r.r2:.3f} | {r.n} |" for r in frame.itertuples()]
    report = HANDOUT.joinpath("report.md").read_text(encoding="utf-8")
    report = re.sub(r"\| Evaluation set.*?\n\n", "\n".join(table) + "\n\n", report, flags=re.S)
    report = re.sub(r"\[[^\]]*\](?!\()", "Written from the saved artifacts.", report)
    (root / "report.md").write_text(report, encoding="utf-8")


# ---------------------------------------------------------------------------
# Running the checker


def checker(root: Path) -> tuple[dict, int]:
    result = subprocess.run([sys.executable, "-B", str(CHECKS / "check_assignment.py"), str(root), "--json"],
                            cwd=root, capture_output=True, text=True, check=False)
    return json.loads(result.stdout), result.returncode


def losses(result: dict) -> dict[str, int]:
    return {test["test-name"]: test["max-score"] - test["score"] for test in result["tests"]
            if test["score"] != test["max-score"]}


def details(result: dict) -> str:
    return "\n".join(f"  {test['test-name']}: {test['detail']}" for test in result["tests"] if test["detail"])


def copy(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def rewrite(path: Path, change) -> None:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = change(frame)
    frame.to_csv(path, index=False)


def loosen(root: Path) -> None:
    """Rewrite every CSV the way a careless but correct student might: nothing here should cost a point."""
    rng = np.random.default_rng(217)
    for path in sorted((root / "output").glob("*.csv")):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame = frame[list(reversed(frame.columns))] if path.name != "q5_correlations.csv" else frame
        for column in frame.columns:
            if column in ("station_name", "model", "split", "check_name", "column_name", "role", "result"):
                frame[column] = frame[column].str.upper()
        frame = frame.sample(frac=1, random_state=1) if len(frame) > 1 and path.name != "q5_correlations.csv" else frame
        text = frame.to_csv(index=path.name.startswith(("q2_missingness", "q6_split")), lineterminator="\r\n")
        lines = text.split("\r\n")
        lines[0] = ", ".join(f" {name} " for name in lines[0].split(","))
        text = "\r\n".join(line + ("  " if i % 7 == 0 and line else "") for i, line in enumerate(lines)).rstrip("\r\n")
        path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    for name, column in (("q4_features.csv", "model_eligible"), ("q3_hourly_panel.csv", "source_observed")):
        path = root / "output" / name
        text = path.read_text(encoding="utf-8-sig")
        path.write_text(text.replace("TRUE", "yes").replace("True", "yes").replace("FALSE", "no").replace("False", "no"),
                        encoding="utf-8")
    del rng


def alternative(root: Path) -> None:
    """Valid alternatives the handout allows."""
    out = root / "output"
    # Q2: measurement_timestamp kept as the naive local text; UTC written with a trailing Z.
    clean = pd.read_csv(out / "q2_cleaned_observations.csv", dtype=str, keep_default_na=False)
    clean["measurement_timestamp"] = clean["measurement_timestamp"].str.slice(0, 19)
    clean["measurement_timestamp_utc"] = clean["measurement_timestamp_utc"].str.replace("+00:00", "Z")
    clean.to_csv(out / "q2_cleaned_observations.csv", index=False)
    # Q1: coverage times as Chicago local text; percentages rounded to one decimal.
    coverage = pd.read_csv(out / "q1_station_coverage.csv")
    for column in ("first_timestamp", "last_timestamp"):
        coverage[column] = pd.to_datetime(coverage[column], utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    coverage["coverage_pct"] = coverage["coverage_pct"].round(1)
    coverage.to_csv(out / "q1_station_coverage.csv", index=False)
    # Q1 audit: both column_names cells written as the printed list of names.
    audit = pd.read_csv(out / "q1_release_audit.csv", dtype=str, keep_default_na=False)
    names = audit["check_name"] == "column_names"
    audit.loc[names, ["expected", "observed"]] = str(audit.loc[names, "expected"].iloc[0].split("|"))
    # The release file name as a path, in another letter case.
    release = audit["check_name"] == "release_filename"
    audit.loc[release, "observed"] = "data/" + audit.loc[release, "observed"].str.upper()
    audit.to_csv(out / "q1_release_audit.csv", index=False)
    # Q2 missingness as a fraction; audit written with plain-English results.
    missing = pd.read_csv(out / "q2_missingness.csv")
    missing["missing_pct"] = missing["missing_pct"] / 100
    missing.to_csv(out / "q2_missingness.csv", index=False)
    audit = pd.read_csv(out / "q2_cleaning_audit.csv")
    audit["result"] = audit["result"].map({"rows_rejected": "Rows rejected", "set_missing": "Set to missing",
                                           "set_to_zero": "Set to zero"})
    audit.to_csv(out / "q2_cleaning_audit.csv", index=False)
    # Q5: population standard deviation; everything rounded to two decimals.
    monthly = pd.read_csv(out / "q5_monthly_station_summary.csv")
    monthly = monthly.round(2)
    monthly.to_csv(out / "q5_monthly_station_summary.csv", index=False)
    # Q4, Q6, Q7, Q8: floats rounded to 4 decimals, timestamps with a T separator.
    for name in ["q4_features.csv", *[f"q6_X_{s}.csv" for s in grading.SPLITS], "q7_validation_predictions.csv",
                 "q8_test_predictions.csv"]:
        frame = pd.read_csv(out / name, dtype=str, keep_default_na=False)
        for column in frame.columns:
            if column.endswith("_utc"):
                frame[column] = frame[column].str.replace(" ", "T", n=1)
        frame.to_csv(out / name, index=False)
    for name in ["q4_features.csv", "q7_validation_metrics.csv", "q8_test_metrics.csv", "q8_station_metrics.csv"]:
        frame = pd.read_csv(out / name)
        frame.to_csv(out / name, index=False, float_format="%.4f")
    # Q5 correlations saved with index=True but no label.
    corr = pd.read_csv(out / "q5_correlations.csv", index_col=0)
    corr.index.name = None
    corr.to_csv(out / "q5_correlations.csv")
    # Q8 station metrics labeled with the regressor's name instead of student_model.
    station = pd.read_csv(out / "q8_station_metrics.csv")
    station["model"] = station["model"].replace({"student_model": "Ridge"})
    station.to_csv(out / "q8_station_metrics.csv", index=False)
    # Q7 spec with a Python dict for parameters and comma-separated features.
    spec = pd.read_csv(out / "q7_model_spec.csv")
    spec["parameters_json"] = spec["parameters_json"].map(lambda text: repr(json.loads(text)))
    spec["feature_columns"] = spec["feature_columns"].str.replace("|", ", ")
    spec.to_csv(out / "q7_model_spec.csv", index=False)
    # Cells separated by semicolons, with a European spreadsheet's decimal commas, or by tabs.
    for name, options in (("q1_station_coverage.csv", {"sep": ";", "decimal": ","}),
                          ("q3_panel_summary.csv", {"sep": ";", "decimal": ","}),
                          ("q2_cleaned_observations.csv", {"sep": "\t"}),
                          ("q8_test_metrics.csv", {"sep": "\t"})):
        pd.read_csv(out / name, dtype={"station_name": str}).to_csv(out / name, index=False, **options)
    assert re.search(r";\d+,\d", (out / "q1_station_coverage.csv").read_text(encoding="utf-8"))


def mutations() -> list[tuple[str, str, object, dict[str, int]]]:
    """(label, file, change, expected losses) for single mistakes."""
    def column(name, func):
        return lambda frame: frame.assign(**{name: func(frame)})
    return [
        ("audit passed False", "q1_release_audit.csv",
         lambda f: f.assign(passed=np.where(f["check_name"] == "row_count", "False", f["passed"])),
         {"Q1 q1_release_audit.csv": 1}),
        ("audit names another release file", "q1_release_audit.csv",
         lambda f: f.assign(observed=np.where(f["check_name"] == "release_filename",
                                              "chicago_beach_sensors_2022_2023.csv", f["observed"])),
         {"Q1 q1_release_audit.csv": 1}),
        ("kept one ambiguous row", "q2_cleaned_observations.csv",
         lambda f: pd.concat([f, f.iloc[[0]].assign(measurement_timestamp="2022-11-06 01:00:00-05:00",
                                                      measurement_timestamp_utc="2022-11-06 06:00:00+00:00")]),
         {"Q2 q2_cleaned_observations.csv: rows": 1}),
        ("forgot the solar near-zero rule", "q2_cleaned_observations.csv",
         column("solar_radiation_w_m2", lambda f: f["solar_radiation_w_m2"].replace("0.0", "-1.0")),
         {"Q2 q2_cleaned_observations.csv: solar_radiation_w_m2": 1}),
        ("did not convert to UTC", "q2_cleaned_observations.csv",
         column("measurement_timestamp_utc", lambda f: f["measurement_timestamp"].str.slice(0, 19)),
         {"Q2 q2_cleaned_observations.csv: measurement_timestamp_utc": 2}),
        ("forgot the wind speed rule", "q2_cleaned_observations.csv",
         column("wind_speed_mps", lambda f: f["wind_speed_mps"].replace("", "999.9")),
         {"Q2 q2_cleaned_observations.csv: interval_rain_mm, wind_speed_mps, maximum_wind_speed_mps": 1,
          "Q2 q2_cleaned_observations.csv: the other nine sensor columns": 0}),
        ("filled Foster's wet bulb", "q2_cleaned_observations.csv",
         column("wet_bulb_temperature_c", lambda f: f["wet_bulb_temperature_c"].replace("", "0.0")),
         {"Q2 q2_cleaned_observations.csv: the other nine sensor columns": 1}),
        ("audit missing the solar row", "q2_cleaning_audit.csv", lambda f: f[f["result"] != "set_to_zero"],
         {"Q2 q2_cleaning_audit.csv": 1}),
        ("filled panel gaps", "q3_hourly_panel.csv",
         column("air_temperature_c", lambda f: f["air_temperature_c"].replace("", "0.0")),
         {"Q3 q3_hourly_panel.csv: sensor columns": 1}),
        ("weekday Monday=1", "q3_hourly_panel.csv",
         column("day_of_week", lambda f: (f["day_of_week"].astype(int) + 1).astype(str)),
         {"Q3 q3_hourly_panel.csv: day_of_week": 1}),
        ("dropped ineligible rows", "q4_features.csv", lambda f: f[f["model_eligible"] == "True"],
         {"Q4 q4_features.csv: rows": 1}),
        ("lag of 2 hours", "q4_features.csv",
         column("air_temperature_lag_1h_c", lambda f: f["air_temperature_lag_24h_c"]),
         {"Q4 q4_features.csv: lags": 1, "Q4 q4_features.csv: air_temperature_change_1h_c": 0}),
        ("rolling mean shifted", "q4_features.csv",
         column("air_temperature_mean_past_24h_c", lambda f: f["air_temperature_lag_1h_c"]),
         {"Q4 q4_features.csv: air_temperature_mean_past_24h_c": 1}),
        ("manifest offset", "q4_feature_manifest.csv",
         lambda f: f.assign(earliest_offset_hours=np.where(f["feature_name"] == grading.MEAN_FEATURE, "-24",
                                                           f["earliest_offset_hours"])),
         {"Q4 q4_feature_manifest.csv": 1}),
        ("monthly one wrong mean", "q5_monthly_station_summary.csv",
         lambda f: f.assign(mean_air_temperature_c=["99"] + f["mean_air_temperature_c"].tolist()[1:]),
         {}),
        ("split summary wrong count", "q6_split_summary.csv",
         lambda f: f.assign(n_rows=["1"] + f["n_rows"].tolist()[1:]), {}),
        ("spec random_state 218", "q7_model_spec.csv", lambda f: f.assign(random_state="218"),
         {"Q7 q7_model_spec.csv": 1}),
        ("nonfinite model prediction", "q7_validation_predictions.csv",
         lambda f: f.assign(model_prediction=["inf"] + f["model_prediction"].tolist()[1:]),
         {"Q7 q7_validation_predictions.csv: model_prediction": 1, "Q7 q7_validation_metrics.csv": 1}),
        ("error sign flipped", "q8_test_predictions.csv",
         column("model_error", lambda f: (-pd.to_numeric(f["model_error"])).astype(str)),
         {"Q8 q8_test_predictions.csv: model_error": 1}),
        ("station metrics one wrong", "q8_station_metrics.csv",
         lambda f: f.assign(mae=["9"] + f["mae"].tolist()[1:]), {"Q8 q8_station_metrics.csv": 1}),
    ]


def run() -> None:
    SCRATCH.mkdir(exist_ok=True)
    handout_files = {path.relative_to(HANDOUT).as_posix() for path in HANDOUT.rglob("*") if path.is_file()}
    shipped = sorted(name for name in handout_files if name.split("/")[-1] in {"check_assignment.py", "grading.py"}
                     or name.startswith((".github/", "_grader_selftest/")) or name.endswith(".badmath.toml"))
    assert not shipped, f"the exam handout ships checks: {shipped}"
    grep = [path for path in [*CHECKS.rglob("*.py"), *HANDOUT.rglob("*.py")]
            # The brackets keep this guard itself out of a plain-text search for version checks.
            if re.search(r"python_versio[n]|sys\.versio[n]|version_inf[o]", path.read_text(encoding="utf-8"))
            and path.name != "run.py"]
    assert not grep, f"a check reads the Python version: {grep}"

    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a11-selftest-") as directory:
        base = Path(directory)
        empty = base / "empty"
        empty.mkdir()
        result, code = checker(empty)
        assert result["score"] == 0 and result["max-score"] == 85 and code == 1, result
        result, code = checker(HANDOUT)
        assert result["score"] == 0 and code == 1, losses(result)
        print("empty directory and untouched handout: 0/85")

        correct = base / "correct"
        solve(correct)
        (correct / "grading.py").write_text("raise SystemExit('submitted code must not run')\n")
        (correct / "output" / "pandas.py").write_text("raise SystemExit('submitted code must not run')\n")
        result, code = checker(correct)
        assert result["score"] == 85 and code == 0, details(result)
        print("independent correct answer: 85/85, and no submitted file ran")

        loose = copy(correct, base / "loose")
        loosen(loose)
        result, _ = checker(loose)
        assert result["score"] == 85, details(result)
        print("reordered columns and rows, upper-case labels, CRLF, BOM, header and trailing spaces, "
              "index columns, yes/no flags: 85/85")

        alt = copy(correct, base / "alternative")
        alternative(alt)
        result, _ = checker(alt)
        assert result["score"] == 85, details(result)
        print("naive local times, Z suffixes, rounding, fractions for percents, population std, unlabeled "
              "correlation index, Python-dict parameters, semicolon and tab separators, an upper-case release "
              "file name: 85/85")

        for label, name, change, expected in mutations():
            case = copy(correct, base / re.sub(r"\W+", "-", label))
            rewrite(case / "output" / name, change)
            result, _ = checker(case)
            lost = losses(result)
            for test, amount in expected.items():
                assert lost.get(test, 0) == amount, (label, test, lost, details(result))
            unexpected = {test: amount for test, amount in lost.items()
                          if test not in expected and not test.startswith("Q9")}
            owned = [test for test in lost if name.split(".")[0] in test or test.startswith("Q9")]
            assert not unexpected or set(unexpected) <= set(owned), (label, unexpected, details(result))
            assert sum(lost.values()) >= 1 and all(result["tests"][i]["detail"] for i, t in enumerate(result["tests"])
                                                   if t["score"] < t["max-score"]), (label, lost)
            print(f"single mistake '{label}': -{sum(lost.values())} ({', '.join(f'{k} -{v}' for k, v in lost.items())})")

        # A wrong upstream file costs only its own points: downstream files built from it still pass.
        cascade = copy(correct, base / "cascade")
        rewrite(cascade / "output" / "q2_cleaned_observations.csv",
                lambda f: f.assign(wind_speed_mps=f["wind_speed_mps"].replace("", "999.9")))
        panel = pd.read_csv(cascade / "output" / "q3_hourly_panel.csv", dtype=str, keep_default_na=False)
        clean = pd.read_csv(cascade / "output" / "q2_cleaned_observations.csv", dtype=str, keep_default_na=False)
        wind = dict(zip(clean["station_name"] + clean["measurement_timestamp_utc"], clean["wind_speed_mps"]))
        panel["wind_speed_mps"] = (panel["station_name"] + panel["measurement_timestamp_utc"]).map(wind).fillna("")
        panel.to_csv(cascade / "output" / "q3_hourly_panel.csv", index=False)
        result, _ = checker(cascade)
        lost = losses(result)
        assert "Q3 q3_hourly_panel.csv: sensor columns" not in lost, details(result)
        print(f"an upstream Q2 mistake carried into Q3: charged once ({lost})")

        # A wrong value in a semicolon-separated file costs only its own check.
        semicolons = copy(correct, base / "semicolon-wrong-value")
        station = pd.read_csv(semicolons / "output" / "q8_station_metrics.csv")
        station.loc[0, "mae"] = 9.0
        station.to_csv(semicolons / "output" / "q8_station_metrics.csv", sep=";", decimal=",", index=False)
        result, _ = checker(semicolons)
        assert losses(result) == {"Q8 q8_station_metrics.csv": 1}, details(result)
        print("a wrong mae in a semicolon-separated q8_station_metrics.csv: -1, that check only")

        # Without its predictions file a metrics file is judged on what can still be checked: the missing file
        # costs its own seven points, while a model n that disagrees with the baseline's still costs the metrics.
        unpredicted = copy(correct, base / "no-test-predictions")
        (unpredicted / "output" / "q8_test_predictions.csv").unlink()
        result, _ = checker(unpredicted)
        lost = losses(result)
        assert all(name.startswith("Q8 q8_test_predictions.csv") for name in lost) and sum(lost.values()) == 7, \
            details(result)
        metrics = pd.read_csv(unpredicted / "output" / "q8_test_metrics.csv", dtype=str, keep_default_na=False)
        metrics.loc[metrics["model"] == "student_model", "n"] = "1"
        metrics.to_csv(unpredicted / "output" / "q8_test_metrics.csv", index=False)
        result, _ = checker(unpredicted)
        assert losses(result).get("Q8 q8_test_metrics.csv") == 1, details(result)
        print("q8_test_predictions.csv missing: -7, its own checks only; with a wrong model n as well, the metrics -1")

        # Documented handout contract: tree and checklist name every artifact with its header.
        readme = (HANDOUT / "README.md").read_text(encoding="utf-8")
        contract = (HANDOUT / "assignment.md").read_text(encoding="utf-8")
        for name, (columns, _) in grading.ARTIFACTS.items():
            assert f"output/{name}" in readme and name in contract, f"README or assignment.md omits {name}"
            if columns:
                assert f"`{','.join(columns)}`" in readme, f"README omits the first line of {name}"
                assert f"`{','.join(columns)}`" in contract, f"assignment.md omits the first line of {name}"
        for forbidden in ("check_assignment", "GitHub Actions", "including final newlines", "automated feedback",
                          "checker", "autograd"):
            for path in [*HANDOUT.glob("*.md"), *HANDOUT.glob("example_report/*.md")]:
                assert forbidden.lower() not in path.read_text(encoding="utf-8").lower(), (path.name, forbidden)
        ref = grading.reference()
        answers = [*ref.summary.to_numpy().ravel(), *ref.split_summary["n_rows"], ref.totals["set_to_zero"],
                   len(ref.panel), len(ref.panel) + 1, len(ref.clean), len(ref.clean) + 1]
        answers += [f"{value:.2f}" for value in ref.clean.groupby("station").size() / ref.summary["expected_hours"] * 100]
        for path in [*HANDOUT.glob("*.md"), *HANDOUT.glob("example_report/*.md")]:
            text = path.read_text(encoding="utf-8")
            for value in answers:
                shown = [str(value)] if isinstance(value, str) else [str(int(value)), f"{int(value):,}"]
                assert not any(re.search(rf"(?<![\d.,]){re.escape(item)}(?![\d.,]\d)", text) for item in shown), \
                    (path.name, value)
        for path in HANDOUT.glob("q*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            assert not any(cell.get("outputs") or cell.get("execution_count") for cell in notebook["cells"]
                           if cell["cell_type"] == "code"), f"{path.name} has outputs"
        assert [p.name for p in (HANDOUT / "output").iterdir()] == [".gitkeep"], "the handout output/ is not empty"
        print("handout README and assignment.md list every artifact and first line, print no expected value, "
              "and mention no checker; notebooks are cleared and output/ is empty")
    print("Assignment 11 self-test passed")


if __name__ == "__main__":
    run()
