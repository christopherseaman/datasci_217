"""Development regression checks for the Assignment 10 checks.

Builds submissions in ignored `scratch/`, answers the assignment with
statsmodels, scikit-learn, and pandas the way the lecture does, and confirms
what each kind of submission scores: a correct one earns 100 with the frozen
pipeline fitted on training rows, and so does one refitted on training plus
validation rows, one written differently (column and row order, CRLF, a BOM,
UTF-16, quoting, padding, letter case, rounding, row-number columns, other
timestamp spellings, 1/0 booleans, the array interface's `const`, and Demo 1's
`valid` label), and one saved with every row index. An empty directory and the
untouched handout earn 0, one wrong value or one misnamed column costs exactly
its own check, and a missing file costs only its own artifact's checks. It also
recomputes every expected value in `_value_checks.py` from
`10/assignment/data/`, confirms that the README's checkpoints, contract, and
task numbering agree with the checks and the notebook, that the handout ships
every file in `CHECKS_FILES` byte for byte and no other Python file, and that
the handout notebook ships with its outputs cleared.

    uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with statsmodels==0.14.6 \\
        --with scikit-learn==1.9.0 --with 'pytest>=8,<9' python 10/assignment_checks/_grader_selftest/run.py
"""

from __future__ import annotations

from pathlib import Path
import codecs
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Self-tests grade with the working copy, never the copy downloaded from GitHub.
os.environ["DS217_LOCAL_CHECKS"] = "1"

CHECKS = Path(__file__).resolve().parents[1]
HANDOUT = CHECKS.parent / "assignment"
DATA = HANDOUT / "data"
SCRATCH = CHECKS.parents[1] / "scratch"
sys.path.insert(0, str(CHECKS))
import _value_checks as value_checks  # noqa: E402
from grading import POINTS, grade_submission  # noqa: E402

NAMES = [check.name for check in value_checks.CHECKS]
POINTS_BY_NAME = dict(zip(NAMES, POINTS, strict=True))
PREFIX = {
    "output/ols_coefficients.csv": "coefficients:",
    "output/new_patient_intervals.csv": "new-patient intervals:",
    "output/ols_residuals.csv": "residuals:",
    "output/residuals_vs_fitted.png": "residual plot:",
    "output/availability_decisions.csv": "availability:",
    "output/split_summary.csv": "split summary:",
    "output/validation_metrics.csv": "validation metrics:",
    "output/test_metrics.csv": "test metrics:",
    "output/test_predictions.csv": "test predictions:",
    "output/readmission_metrics.csv": "readmission metrics:",
}
FEATURES = ["age", "bmi", "sbp_today"]
TARGET = "sbp_followup"
VALIDATION_START = pd.Timestamp("2026-05-01", tz="UTC")
TEST_START = pd.Timestamp("2026-05-09", tz="UTC")
NEW_PATIENT = pd.DataFrame({"age": [60], "bmi": [31.0]})


def minimal_png() -> bytes:
    """A valid 1x1 grey PNG, built without matplotlib."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return (value_checks.PNG_SIGNATURE + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(b"\x00\x80")) + chunk(b"IEND", b""))


def load() -> dict[str, pd.DataFrame]:
    visits = pd.read_csv(DATA / "followup_visits.csv")
    visits["visit_time"] = pd.to_datetime(visits["visit_time"], utc=True)
    visits["followup_time"] = pd.to_datetime(visits["followup_time"], utc=True)
    return {
        "patients": pd.read_csv(DATA / "clinic_bp.csv"),
        "candidates": pd.read_csv(DATA / "feature_availability.csv"),
        "visits": visits,
        "flags": pd.read_csv(DATA / "readmission_flags.csv"),
    }


def metrics(actual, predicted) -> dict[str, float]:
    return {"mae": mean_absolute_error(actual, predicted),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": r2_score(actual, predicted)}


def solve(refit: bool = False) -> dict[str, pd.DataFrame]:
    """Answer the assignment with the lecture's calls; returns {artifact path: frame to save}."""
    data = load()
    patients, candidates, visits, flags = data["patients"], data["candidates"], data["visits"], data["flags"]

    results = smf.ols("sbp ~ age + bmi", data=patients).fit()
    coefficients = pd.DataFrame({"coef": results.params, "std_err": results.bse,
                                 "ci_lower": results.conf_int()[0], "ci_upper": results.conf_int()[1]})
    coefficients.index.name = "term"
    intervals = results.get_prediction(NEW_PATIENT).summary_frame(alpha=0.05)
    new_patient_intervals = pd.concat([NEW_PATIENT, intervals], axis=1)
    residual_table = pd.DataFrame({"patient_id": patients["patient_id"], "observed": patients["sbp"],
                                   "fitted": results.fittedvalues, "residual": results.resid})

    decisions = candidates.copy()
    decisions["available"] = decisions["hours_after_visit"] <= 0
    decisions["decision"] = np.where(decisions["available"], "Keep", "Exclude (leakage)")
    assert decisions.loc[decisions["available"], "candidate_feature"].tolist() == FEATURES

    train = visits[visits["followup_time"] < VALIDATION_START]
    valid = visits[(visits["followup_time"] >= VALIDATION_START) & (visits["followup_time"] < TEST_START)]
    test = visits[visits["followup_time"] >= TEST_START]
    parts = {"train": train, "validation": valid, "test": test}
    split_summary = pd.DataFrame({
        "partition": list(parts),
        "row_count": [len(part) for part in parts.values()],
        "first_target_time": [part["followup_time"].min() for part in parts.values()],
        "last_target_time": [part["followup_time"].max() for part in parts.values()],
    })

    baseline = DummyRegressor(strategy="mean").fit(train[FEATURES], train[TARGET])
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(train[FEATURES], train[TARGET])
    validation_metrics = pd.DataFrame([
        {"approach": name, **metrics(valid[TARGET], fitted.predict(valid[FEATURES]))}
        for name, fitted in [("mean_baseline", baseline), ("linear_pipeline", pipeline)]
    ])
    assert validation_metrics.loc[validation_metrics["mae"].idxmin(), "approach"] == "linear_pipeline"

    rows = pd.concat([train, valid]) if refit else train
    final = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(rows[FEATURES], rows[TARGET])
    test_predicted = final.predict(test[FEATURES])
    test_metrics = pd.DataFrame([{"approach": "linear_pipeline", **metrics(test[TARGET], test_predicted)}])
    test_predictions = test[["visit_id", "followup_time", TARGET]].copy()
    test_predictions["predicted_sbp"] = test_predicted

    readmission_metrics = pd.DataFrame([
        {"approach": column,
         "accuracy": accuracy_score(flags["readmitted_30d"], flags[column]),
         "precision": precision_score(flags["readmitted_30d"], flags[column], zero_division=0),
         "recall": recall_score(flags["readmitted_30d"], flags[column])}
        for column in ["model_flag", "never_flag"]
    ])
    return {
        "output/ols_coefficients.csv": coefficients,
        "output/new_patient_intervals.csv": new_patient_intervals,
        "output/ols_residuals.csv": residual_table,
        "output/availability_decisions.csv": decisions,
        "output/split_summary.csv": split_summary,
        "output/validation_metrics.csv": validation_metrics,
        "output/test_metrics.csv": test_metrics,
        "output/test_predictions.csv": test_predictions,
        "output/readmission_metrics.csv": readmission_metrics,
    }


def as_files(frames: dict[str, pd.DataFrame]) -> dict[str, str | bytes]:
    """The way the README saves each frame: index=False, except the coefficients, which keep their term index."""
    files: dict[str, str | bytes] = {
        path: frame.to_csv() if path == "output/ols_coefficients.csv" else frame.to_csv(index=False)
        for path, frame in frames.items()
    }
    files[value_checks.RESIDUAL_PLOT] = minimal_png()
    return files



def failing(result: dict) -> set[str]:
    return {test["test-name"] for test in result["tests"] if not test["passed"]}


def detail(result: dict, name: str) -> str:
    return next(test["detail"] for test in result["tests"] if test["test-name"] == name)


def cli_report(checks: Path, root: Path) -> dict:
    """Grade through the `check_assignment.py` in `checks`, as a student or CI runs it."""
    finished = subprocess.run(
        [sys.executable, "-B", str(checks / "check_assignment.py"), str(root), "--json"],
        capture_output=True, text=True, check=False,
    )
    assert finished.stdout, f"{checks / 'check_assignment.py'} printed nothing: {finished.stderr}"
    report = json.loads(finished.stdout)
    assert (finished.returncode == 0) == (report["score"] == report["max-score"]), (checks, root, finished.returncode)
    return report


def graded(root: Path) -> dict:
    """Grade in-process and through both CLIs, and insist all three agree."""
    result = grade_submission(root)
    assert result["schema"] == "datasci217/grading-result/v1"
    assert result["max-score"] == 100 == sum(POINTS)
    assert cli_report(CHECKS, root) == result, root
    assert cli_report(HANDOUT, root) == result, root
    return result


def cost(*names: str) -> int:
    return sum(POINTS_BY_NAME[name] for name in names)


def artifact_checks(path: str) -> list[str]:
    return [name for name in NAMES if name.startswith(PREFIX[path])]


def near(got: float, want: float) -> bool:
    return abs(got - want) <= 1e-5


def check_expected_values() -> None:
    """Every expected value in _value_checks.py matches a fresh computation from the handout's data."""
    assert sorted(path.name for path in DATA.iterdir()) == [
        "clinic_bp.csv", "feature_availability.csv", "followup_visits.csv", "readmission_flags.csv"]
    patients = load()["patients"]
    assert value_checks.CLINIC_PREDICTORS == {
        row.patient_id: (row.age, row.bmi) for row in patients.itertuples()
    }
    frames = solve()
    coefficients = frames["output/ols_coefficients.csv"]
    for term, expected in value_checks.COEFFICIENT_VALUES.items():
        for column, want in expected.items():
            assert near(coefficients.loc[term, column], want), (term, column)
    intervals = frames["output/new_patient_intervals.csv"].iloc[0]
    assert (intervals["age"], intervals["bmi"]) == value_checks.NEW_PATIENT
    for column, want in value_checks.NEW_PATIENT_VALUES.items():
        assert near(intervals[column], want), column
    residuals = frames["output/ols_residuals.csv"].set_index("patient_id")
    assert list(residuals.index) == list(value_checks.RESIDUAL_VALUES)
    for patient, expected in value_checks.RESIDUAL_VALUES.items():
        for column, want in expected.items():
            assert near(residuals.loc[patient, column], want), (patient, column)
    hours = load()["candidates"].set_index("candidate_feature")["hours_after_visit"].to_dict()
    assert hours == value_checks.HOURS_AFTER_VISIT, hours
    split = frames["output/split_summary.csv"].set_index("partition")
    for partition, expected in value_checks.SPLIT_VALUES.items():
        assert split.loc[partition, "row_count"] == expected["row_count"]
        for column in ("first_target_time", "last_target_time"):
            assert split.loc[partition, column] == pd.Timestamp(expected[column]), (partition, column)
    validation = frames["output/validation_metrics.csv"].set_index("approach")
    for approach, expected in value_checks.VALIDATION_VALUES.items():
        for column, want in expected.items():
            assert near(validation.loc[approach, column], want), (approach, column)
    for refit, name in ((False, "train_only"), (True, "train_plus_validation")):
        frames = solve(refit=refit)
        test_metrics = frames["output/test_metrics.csv"].iloc[0]
        for column, want in value_checks.TEST_METRIC_VALUES[name].items():
            assert near(test_metrics[column], want), (name, column)
        predictions = frames["output/test_predictions.csv"].set_index("visit_id")
        assert list(predictions.index) == list(value_checks.TEST_PREDICTION_VALUES)
        for visit, expected in value_checks.TEST_PREDICTION_VALUES.items():
            assert predictions.loc[visit, "followup_time"] == pd.Timestamp(expected["followup_time"])
            assert predictions.loc[visit, TARGET] == expected["sbp_followup"]
            assert near(predictions.loc[visit, "predicted_sbp"], expected[name]), (visit, name)
    readmission = frames["output/readmission_metrics.csv"].set_index("approach")
    for approach, expected in value_checks.READMISSION_VALUES.items():
        for column, want in expected.items():
            assert near(readmission.loc[approach, column], want), (approach, column)
    check_slip_values()
    # The two refits must be told apart from wrong answers, not from each other: both are accepted.
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for line in ('new_patient = pd.DataFrame({"age": [60], "bmi": [31.0]})',
                 'VALIDATION_START = pd.Timestamp("2026-05-01", tz="UTC")',
                 'TEST_START = pd.Timestamp("2026-05-09", tz="UTC")',
                 'TARGET = "sbp_followup"'):
        assert line in source, line


def check_slip_values() -> None:
    """Every common wrong answer in _value_checks.py matches a fresh computation of that slip."""
    data = load()
    visits, flags = data["visits"], data["flags"]
    train = visits[visits["followup_time"] < VALIDATION_START]
    valid = visits[(visits["followup_time"] >= VALIDATION_START) & (visits["followup_time"] < TEST_START)]
    test = visits[visits["followup_time"] >= TEST_START]
    by_visit = {"train": visits[visits["visit_time"] < VALIDATION_START],
                "validation": visits[(visits["visit_time"] >= VALIDATION_START) & (visits["visit_time"] < TEST_START)],
                "test": visits[visits["visit_time"] >= TEST_START]}
    for (partition,), expected in value_checks.SPLIT_ON_VISIT_TIME.rows.items():
        part = by_visit[partition]
        assert len(part) == expected["row_count"], partition
        for column, stat in (("first_target_time", "min"), ("last_target_time", "max")):
            if column in expected:
                got = getattr(part["followup_time"], stat)()
                want = expected[column]
                assert (pd.isna(got) if want is None else got == pd.Timestamp(want.text)), (partition, column)
    parts = {"train": train, "validation": valid, "test": test}
    for (partition,), expected in value_checks.RANGES_FROM_VISIT_TIME.rows.items():
        assert parts[partition]["visit_time"].min() == pd.Timestamp(expected["first_target_time"].text), partition
        assert parts[partition]["visit_time"].max() == pd.Timestamp(expected["last_target_time"].text), partition
    for (patient,), expected in value_checks.RESIDUAL_COLUMNS_SWAPPED.rows.items():
        values = value_checks.RESIDUAL_VALUES[patient]
        assert (expected["fitted"], expected["residual"]) == (values["residual"], values["fitted"]), patient
    leaky_sets = [FEATURES + ["callback_sbp"], FEATURES + ["a1c_result"], FEATURES + ["a1c_result", "callback_sbp"]]
    valid_scores, test_scores = [], []
    test_predicted = {visit: [] for visit in test["visit_id"]}
    for features in leaky_sets:
        fitted = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(train[features], train[TARGET])
        valid_scores.append(metrics(valid[TARGET], fitted.predict(valid[features])))
        for rows in (train, pd.concat([train, valid])):
            final = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(rows[features], rows[TARGET])
            test_scores.append(metrics(test[TARGET], final.predict(test[features])))
            for visit, predicted in zip(test["visit_id"], final.predict(test[features]), strict=True):
                test_predicted[visit].append(predicted)
    for slip, scores in ((value_checks.LEAKY_VALIDATION, valid_scores), (value_checks.LEAKY_TEST, test_scores)):
        for column, either in slip.rows[("linear_pipeline",)].items():
            assert len(either.values) == len(scores), column
            for want, got in zip(either.values, [score[column] for score in scores]):
                assert near(got, want), (column, got, want)
    assert list(value_checks.LEAKY_TEST_PREDICTIONS.rows) == [(visit,) for visit in test_predicted]
    for (visit,), expected in value_checks.LEAKY_TEST_PREDICTIONS.rows.items():
        wanted = expected["predicted_sbp"].values
        assert len(wanted) == len(test_predicted[visit]) == 6, visit
        for want, got in zip(wanted, test_predicted[visit], strict=True):
            assert near(got, want), (visit, got, want)
    # RMSE saved without its square root: the mean squared error.
    baseline = DummyRegressor(strategy="mean").fit(train[FEATURES], train[TARGET])
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(train[FEATURES], train[TARGET])
    for (approach,), expected in value_checks.RMSE_NOT_ROOTED.rows.items():
        fitted = baseline if approach == "mean_baseline" else pipeline
        assert near(mean_squared_error(valid[TARGET], fitted.predict(valid[FEATURES])), expected["rmse"]), approach
    squared = [
        mean_squared_error(test[TARGET], Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
                           .fit(rows[FEATURES], rows[TARGET]).predict(test[FEATURES]))
        for rows in (train, pd.concat([train, valid]))
    ]
    wanted = value_checks.TEST_RMSE_NOT_ROOTED.rows[("linear_pipeline",)]["rmse"].values
    assert all(near(got, want) for got, want in zip(squared, wanted, strict=True)), (squared, wanted)
    baseline = DummyRegressor(strategy="mean").fit(valid[FEATURES], valid[TARGET])
    for column, want in value_checks.BASELINE_FROM_VALIDATION.rows[("mean_baseline",)].items():
        assert near(metrics(valid[TARGET], baseline.predict(valid[FEATURES]))[column], want), column
    swapped = value_checks.FLAG_PASSED_FIRST.rows[("model_flag",)]
    assert near(precision_score(flags["model_flag"], flags["readmitted_30d"]), swapped["precision"])
    assert near(recall_score(flags["model_flag"], flags["readmitted_30d"]), swapped["recall"])


def check_readme() -> None:
    """Each README checkpoint names its artifact's header, and the contract table matches the checks."""
    readme = (HANDOUT / "README.md").read_text(encoding="utf-8")
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    checks_md = (HANDOUT / "CHECKS.md").read_text(encoding="utf-8")
    assert "(CHECKS.md)" in readme, "README.md no longer links CHECKS.md"
    cells = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    wanted = {
        artifact.path: set(artifact.columns) | set(artifact.optional_columns) for artifact in value_checks.ARTIFACTS
    }
    checkpoints = re.findall(r"> \*\*Checkpoint: `([^`]+)`\*\*\n> The header line `([^`]+)`", readme)
    assert {path: set(header.split(",")) for path, header in checkpoints} == wanted, checkpoints
    in_notebook = re.findall(r"> \*\*Checkpoint: `([^`]+)`\*\*\n> .*under the header `([^`]+)`", cells)
    assert dict(in_notebook) == dict(checkpoints), (in_notebook, checkpoints)
    for text in (readme, cells):
        assert f"> **Checkpoint: `{value_checks.RESIDUAL_PLOT}`**" in text
    contract = re.findall(r"^\| `(output/[^`]+)` \| .* \| ([^|]+) \| (\d+) \|$", checks_md, re.M)
    assert [(name.strip(), int(points)) for _, name, points in contract] == list(zip(NAMES, POINTS)), contract
    for path, name, _ in contract:
        assert name.strip().startswith(PREFIX[path]), (path, name)
    first = value_checks.CHECKS[0].name
    missing = value_checks._missing(HANDOUT, value_checks.COEFFICIENTS.path, value_checks.COEFFICIENTS.task)
    assert f"[FIX ]  0/{POINTS[0]}  {first}\n         {missing}" in checks_md
    last = value_checks.CHECKS[-1].name
    assert f"[PASS]  {POINTS[-1]}/{POINTS[-1]}  {last}\n\nScore: 100/100" in checks_md
    # The README and the notebook number their tasks the same way.
    notebook_headings = [line for line in cells.splitlines() if re.match(r"#{2,3} (Task \d|\d\.\d)", line)]
    readme_headings = [line for line in readme.splitlines() if re.match(r"#{2,3} (Task \d|\d\.\d)", line)]
    assert notebook_headings == readme_headings, (notebook_headings, readme_headings)
    # No em dashes in anything a student reads.
    for text in (readme, json.dumps(notebook), (CHECKS / "_value_checks.py").read_text(encoding="utf-8")):
        assert "\u2014" not in text


def check_handout_files() -> None:
    """The handout ships every CHECKS_FILES file byte for byte, and no other Python or QA files."""
    workflow = (HANDOUT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    listed = re.search(r"^  CHECKS_FILES: \|\n((?:    \S.*\n)+)", workflow, re.M).group(1).split()
    assert 'CHECKS_PATH: "10/assignment_checks"' in workflow
    shipped = sorted(
        path.relative_to(CHECKS).as_posix()
        for path in CHECKS.rglob("*")
        if path.is_file() and not {"__pycache__", ".pytest_cache", "_grader_selftest"} & set(path.relative_to(CHECKS).parts)
        and path.name != "README.md"
    )
    assert sorted(listed) == shipped, (listed, shipped)
    for name in listed:
        assert (HANDOUT / name).read_bytes() == (CHECKS / name).read_bytes(), f"{name} differs from the handout copy"
    handout_python = sorted(
        path.relative_to(HANDOUT).as_posix() for path in HANDOUT.rglob("*.py")
        if "__pycache__" not in path.parts and ".venv" not in path.parts
    )
    assert handout_python == sorted(name for name in listed if name.endswith(".py")), handout_python
    assert not (HANDOUT / "_grader_selftest").exists()
    assert sorted(path.name for path in (HANDOUT / "output").iterdir()) == [".gitkeep"]
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None, cell.get("id")


def alternative() -> dict[str, str | bytes]:
    """The same answers written the way other careful students might: {artifact path: file contents}.

    Built with the refit on training plus validation rows and the array interface.
    """
    data = load()
    patients = data["patients"]
    frames = solve(refit=True)

    # The array interface names the intercept const; saved with an unnamed index, two decimals, CRLF, UTF-16.
    array_fit = sm.OLS(patients["sbp"], sm.add_constant(patients[["age", "bmi"]])).fit()
    coefficients = pd.DataFrame({"coef": array_fit.params, "std_err": array_fit.bse,
                                 "ci_lower": array_fit.conf_int()[0], "ci_upper": array_fit.conf_int()[1]}).round(2)
    coefficients = coefficients[["ci_upper", "coef", "ci_lower", "std_err"]]

    # summary_frame() alone, age and bmi added at the end, mean_se dropped, one decimal, saved with its index.
    intervals = frames["output/new_patient_intervals.csv"].drop(columns=["mean_se", "age", "bmi"]).round(1)
    intervals["BMI"] = 31
    intervals["Age"] = 60.0

    residuals = frames["output/ols_residuals.csv"][["residual", "fitted", "observed", "patient_id"]].round(1)
    residuals["patient_id"] = residuals["patient_id"].str.lower()

    decisions = frames["output/availability_decisions.csv"].copy()
    decisions["available"] = decisions["available"].astype(int)
    decisions["decision"] = np.where(decisions["available"] == 1, "keep", "exclude")
    decisions["hours_after_visit"] = decisions["hours_after_visit"].astype(float)

    # Demo 1's layout: partitions as an unnamed index, with Demo 1's valid label, and other timestamp spellings.
    split = frames["output/split_summary.csv"].set_index("partition").rename(index={"validation": "valid"})
    split["first_target_time"] = split["first_target_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    split["last_target_time"] = split["last_target_time"].dt.tz_convert("America/Los_Angeles")
    split.index.name = None

    validation = frames["output/validation_metrics.csv"].iloc[::-1].round(2)
    validation["approach"] = validation["approach"].str.replace("_", " ").str.title()
    validation.columns = [column.upper() for column in validation.columns]

    test_metrics = frames["output/test_metrics.csv"].round(3)

    predictions = frames["output/test_predictions.csv"].iloc[::-1].round({"predicted_sbp": 1})
    predictions["followup_time"] = predictions["followup_time"].dt.tz_localize(None)

    readmission = frames["output/readmission_metrics.csv"][["recall", "approach", "precision", "accuracy"]].round(3)
    readmission["approach"] = readmission["approach"].str.upper()

    return {
        "output/ols_coefficients.csv": codecs.BOM_UTF16_LE + coefficients.to_csv(lineterminator="\r\n").encode("utf-16-le"),
        "output/new_patient_intervals.csv": intervals.to_csv(),
        # A trailing space on each line and no final newline.
        "output/ols_residuals.csv": "\n".join(line + " " for line in residuals.to_csv(index=False).splitlines()),
        # A BOM and every cell quoted.
        "output/availability_decisions.csv": "﻿" + decisions.to_csv(index=False, quoting=1),
        "output/split_summary.csv": split.to_csv(),
        "output/validation_metrics.csv": validation.to_csv(index=False),
        # A space after every comma.
        "output/test_metrics.csv": test_metrics.to_csv(index=False).replace(",", ", "),
        # Saved with its row index, which here numbers the test rows 38 to 47.
        "output/test_predictions.csv": predictions.to_csv(),
        "output/readmission_metrics.csv": readmission.to_csv(index=False, lineterminator="\r\n"),
        value_checks.RESIDUAL_PLOT: minimal_png(),
    }


def edit(text: str, row_key: str, column: str, value: str) -> str:
    """The CSV text with one cell replaced; `row_key` names the row's first cell."""
    lines = text.splitlines()
    header = lines[0].split(",")
    position = header.index(column)
    for number, line in enumerate(lines[1:], start=1):
        cells = line.split(",")
        if cells[0] == row_key:
            cells[position] = value
            lines[number] = ",".join(cells)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"no row {row_key}")


def rename(text: str, column: str, new: str) -> str:
    lines = text.splitlines()
    lines[0] = ",".join(new if cell == column else cell for cell in lines[0].split(","))
    return "\n".join(lines) + "\n"


def drop_last_row(text: str) -> str:
    return "\n".join(text.splitlines()[:-1]) + "\n"


def run() -> None:
    check_expected_values()
    check_readme()
    check_handout_files()
    frames = solve()
    correct = as_files(frames)

    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a10-selftest-") as temporary:
        workspace = Path(temporary)

        def submission(name: str, files: dict[str, str | bytes]) -> Path:
            root = workspace / name
            (root / "output").mkdir(parents=True)
            for path, contents in files.items():
                if isinstance(contents, bytes):
                    (root / path).write_bytes(contents)
                else:
                    (root / path).write_text(contents, encoding="utf-8", newline="")
            return root

        empty = workspace / "empty"
        empty.mkdir()
        assert graded(empty)["score"] == 0
        scaffold = workspace / "scaffold"
        shutil.copytree(HANDOUT, scaffold, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv"))
        result = graded(scaffold)
        assert result["score"] == 0 and failing(result) == set(NAMES)
        assert detail(result, NAMES[0]).startswith("output/ols_coefficients.csv is missing"), detail(result, NAMES[0])

        root = submission("correct", correct)
        result = graded(root)
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}

        # Schema owns an omitted column; recognizable present siblings still need correct values.
        partial = frames["output/ols_coefficients.csv"].reset_index().drop(columns="ci_lower").copy()
        for label in ("missing", "missing-and-wrong", "header-only", "unrecognizable"):
            changed = partial.copy()
            if label == "missing-and-wrong":
                changed.loc[changed.index[0], "ci_upper"] += 17
            elif label == "header-only":
                changed = changed.iloc[:0]
            elif label == "unrecognizable":
                changed = changed.iloc[:1].astype(object)
                changed.iloc[0, :] = "WRONG"
            files = dict(correct)
            files["output/ols_coefficients.csv"] = changed.to_csv(index=False)
            result = graded(submission("present-siblings-" + label, files))
            if label == "missing":
                assert failing(result) == {"coefficients: columns"}, failing(result)
            elif label == "missing-and-wrong":
                assert failing(result) == {"coefficients: columns", "coefficients: confidence interval values"}, failing(result)
                message = detail(result, "coefficients: confidence interval values")
                assert "expected" in message and "ci_upper" in message, message
            else:
                assert "coefficients: confidence interval values" in failing(result), failing(result)

        # A one-row artifact cannot earn missing values from an entirely unrecognizable key.
        files = dict(correct)
        files["output/new_patient_intervals.csv"] = "age,bmi\n999,999\n"
        result = graded(submission("unrecognizable-patient-missing-values", files))
        assert failing(result) == set(artifact_checks("output/new_patient_intervals.csv")), failing(result)

        # The checks read only output/: poisoned data and code in the submission change nothing.
        (root / "data").mkdir()
        (root / "data" / "followup_visits.csv").write_text("poison\n", encoding="utf-8")
        for name in ("grading.py", "_value_checks.py", "check_assignment.py"):
            (root / name).write_text("raise RuntimeError('submission code must not run')\n", encoding="utf-8")
        assert graded(root)["score"] == 100

        # The lecture's refit on training plus validation rows scores the same as the train-only fit.
        result = graded(submission("refit", as_files(solve(refit=True))))
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}

        result = graded(submission("alternative", alternative()))
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}

        # Saved with the row index everywhere, as to_csv() does without index=False.
        root = workspace / "with-index"
        (root / "output").mkdir(parents=True)
        for path, frame in frames.items():
            frame.to_csv(root / path)
        (root / value_checks.RESIDUAL_PLOT).write_bytes(minimal_png())
        assert graded(root)["score"] == 100

        # Cells separated by semicolons, with a European spreadsheet's decimal commas, or by tabs: full marks.
        def separated(sep: str, decimal: str = ".") -> dict[str, str | bytes]:
            files: dict[str, str | bytes] = {
                path: frame.to_csv(index=path == "output/ols_coefficients.csv", sep=sep, decimal=decimal)
                for path, frame in frames.items()
            }
            files[value_checks.RESIDUAL_PLOT] = minimal_png()
            return files

        semicolons = separated(";", ",")
        assert any(re.search(r";-?\d+,\d", text) for text in semicolons.values() if isinstance(text, str))
        for label, files in (("semicolons", semicolons), ("tabs", separated("\t"))):
            result = graded(submission(label, files))
            assert result["score"] == 100, (label, {name: detail(result, name) for name in failing(result)})
        # One wrong value in a semicolon-separated file still costs only its own check.
        wrong = frames["output/validation_metrics.csv"].copy()
        wrong.loc[wrong["approach"] == "linear_pipeline", "mae"] = 3.9
        files = separated(";", ",")
        files["output/validation_metrics.csv"] = wrong.to_csv(index=False, sep=";", decimal=",")
        result = graded(submission("semicolons-one-wrong-value", files))
        assert failing(result) == {"validation metrics: mae values"}, {name: detail(result, name) for name in failing(result)}
        message = detail(result, "validation metrics: mae values")
        assert "linear_pipeline has mae 3.9, expected 3.34" in message, message

        # Every number rounded to one decimal, as DataFrame.round(1) writes it: full marks.
        rounded = as_files({path: frame.round(dict.fromkeys(frame.select_dtypes("float").columns, 1))
                            for path, frame in frames.items()})
        assert "model_flag,0.8,0.6,0.8" in rounded["output/readmission_metrics.csv"]
        result = graded(submission("one-decimal", rounded))
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}
        # A one-decimal value that rounds something else still costs only its own check.
        files = dict(rounded)
        files["output/validation_metrics.csv"] = edit(rounded["output/validation_metrics.csv"], "linear_pipeline", "mae", "3.5")
        result = graded(submission("one-decimal-wrong-mae", files))
        assert failing(result) == {"validation metrics: mae values"}, failing(result)

        # Accuracy, precision, recall, and R² written as percents: full marks.
        percents = dict(correct)
        readmission_frame = frames["output/readmission_metrics.csv"].copy()
        for column in ("accuracy", "precision", "recall"):
            readmission_frame[column] = (readmission_frame[column] * 100).map("{:g}%".format)
        percents["output/readmission_metrics.csv"] = readmission_frame.to_csv(index=False)
        assert "model_flag,85%,60%,75%" in percents["output/readmission_metrics.csv"]
        for path in ("output/validation_metrics.csv", "output/test_metrics.csv"):
            frame = frames[path].copy()
            frame["r2"] = (frame["r2"] * 100).round(1).map("{:g}%".format)
            percents[path] = frame.to_csv(index=False)
        result = graded(submission("percents", percents))
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}
        # A wrong percent costs only its own check, and a percent means nothing in a column that is not a proportion.
        files = dict(percents)
        files["output/readmission_metrics.csv"] = edit(percents["output/readmission_metrics.csv"], "model_flag",
                                                       "accuracy", "80%")
        files["output/test_metrics.csv"] = edit(percents["output/test_metrics.csv"], "linear_pipeline", "mae", "264%")
        result = graded(submission("percents-wrong", files))
        assert failing(result) == {"readmission metrics: accuracy values", "test metrics: mae value"}, failing(result)

        # The coefficient table saved with index=False loses its terms: only the columns check fails.
        files = dict(correct)
        files["output/ols_coefficients.csv"] = frames["output/ols_coefficients.csv"].to_csv(index=False)
        result = graded(submission("lost-terms", files))
        assert failing(result) == {"coefficients: columns"}, {name: detail(result, name) for name in failing(result)}
        assert "is missing term" in detail(result, "coefficients: columns")

        # Compact ISO timestamps must work even with the student's Python 3.9.
        files = dict(correct)
        split_frame = frames["output/split_summary.csv"].copy()
        for column in ("first_target_time", "last_target_time"):
            split_frame[column] = pd.to_datetime(split_frame[column], utc=True).dt.strftime("%Y%m%dT%H%M%SZ")
        files["output/split_summary.csv"] = split_frame.to_csv(index=False)
        assert graded(submission("compact-timestamps", files))["score"] == 100
        # A month name and a 12-hour clock still identify the same UTC instants.
        files = dict(correct)
        for path, columns in (("output/split_summary.csv", ("first_target_time", "last_target_time")),
                              ("output/test_predictions.csv", ("followup_time",))):
            frame = frames[path].copy()
            for column in columns:
                frame[column] = pd.to_datetime(frame[column], utc=True).dt.strftime("%d-%B-%Y %I:%M:%S %p UTC")
            files[path] = frame.to_csv(index=False)
        assert graded(submission("month-name-12-hour-timestamps", files))["score"] == 100

        # A duplicated header costs once, regardless of which copy comes last.
        for first in (False, True):
            files = dict(correct)
            frame = frames["output/validation_metrics.csv"]
            wrong = frame[["mae"]].copy().assign(mae=999)
            files["output/validation_metrics.csv"] = pd.concat([wrong, frame] if first else [frame, wrong], axis=1).to_csv(index=False)
            assert failing(graded(submission(f"duplicate-mae-{first}", files))) == {"validation metrics: columns"}
        files = dict(correct)
        frame = frames["output/availability_decisions.csv"]
        wrong = frame[["available"]].copy().assign(available=True)
        files["output/availability_decisions.csv"] = pd.concat([wrong, frame], axis=1).to_csv(index=False)
        assert failing(graded(submission("duplicate-boolean", files))) == {"availability: columns"}
        # One early mistake does not remove credit for its correctly calculated follow-ons.
        files = dict(correct)
        frame = frames["output/availability_decisions.csv"].copy()
        frame.loc[0, ["available", "decision"]] = [False, "Exclude (leakage)"]
        files["output/availability_decisions.csv"] = frame.to_csv(index=False)
        assert failing(graded(submission("own-availability-decision", files))) == {"availability: available values"}
        files = dict(correct)
        frame = frames["output/ols_residuals.csv"].copy()
        frame.loc[0, "fitted"] += 3
        frame["residual"] = frame["observed"] - frame["fitted"]
        files["output/ols_residuals.csv"] = frame.to_csv(index=False)
        assert failing(graded(submission("own-fitted-residual", files))) == {"residuals: fitted values"}
        files = dict(correct)
        frame = frames["output/test_predictions.csv"].copy()
        frame["predicted_sbp"] += 3
        files["output/test_predictions.csv"] = frame.to_csv(index=False)
        files["output/test_metrics.csv"] = pd.DataFrame([{
            "approach": "linear_pipeline", **metrics(frame["sbp_followup"], frame["predicted_sbp"]),
        }]).to_csv(index=False)
        assert failing(graded(submission("own-prediction-metrics", files))) == {"test predictions: predicted_sbp values"}
        files = dict(correct)
        frame = frames["output/ols_coefficients.csv"].copy()
        frame.loc["Intercept", "coef"] += 5
        files["output/ols_coefficients.csv"] = frame.to_csv()
        frame = frames["output/ols_residuals.csv"].copy()
        frame["fitted"] += 5
        frame["residual"] -= 5
        files["output/ols_residuals.csv"] = frame.to_csv(index=False)
        frame = frames["output/new_patient_intervals.csv"].copy()
        for column in ("mean", "mean_ci_lower", "mean_ci_upper", "obs_ci_lower", "obs_ci_upper"):
            frame[column] += 5
        files["output/new_patient_intervals.csv"] = frame.to_csv(index=False)
        assert failing(graded(submission("own-coefficient-follow-ons", files))) == {"coefficients: coef values"}

        misplaced = submission("coefficient-outside-output", correct)
        (misplaced / "output/ols_coefficients.csv").rename(misplaced / "ols_coefficients.csv")
        assert failing(graded(misplaced)) == {"coefficients: columns"}

        # PNG magic bytes alone do not show a saved image.
        image = minimal_png()
        for label, broken in (("signature-only", image[:8]), ("header-only", image[:24]),
                              ("missing-end", image[:-12]), ("bad-checksum", image[:-1] + b"x")):
            files = dict(correct)
            files[value_checks.RESIDUAL_PLOT] = broken
            result = graded(submission("png-" + label, files))
            assert failing(result) == {"residual plot: PNG image"}, failing(result)
            assert "complete PNG" in detail(result, "residual plot: PNG image")

        # CSV readers differ by Python version on NUL; a broken file is unreadable on both.
        files = dict(correct)
        files["output/ols_coefficients.csv"] = b"\xff\xfe\x00"
        result = graded(submission("nul-byte-file", files))
        assert failing(result) == {name for name in NAMES if name.startswith(PREFIX["output/ols_coefficients.csv"])}, failing(result)

        # One mistake costs exactly its own check.
        c = correct
        coefficients, intervals = c["output/ols_coefficients.csv"], c["output/new_patient_intervals.csv"]
        residuals, decisions = c["output/ols_residuals.csv"], c["output/availability_decisions.csv"]
        split, validation = c["output/split_summary.csv"], c["output/validation_metrics.csv"]
        test_metrics, predictions = c["output/test_metrics.csv"], c["output/test_predictions.csv"]
        readmission = c["output/readmission_metrics.csv"]
        mistakes = {
            "coefficients: columns": ("output/ols_coefficients.csv", rename(coefficients, "std_err", "se")),
            "coefficients: one row per term": ("output/ols_coefficients.csv", drop_last_row(coefficients)),
            "coefficients: coef values": ("output/ols_coefficients.csv", edit(coefficients, "age", "coef", "0.55")),
            "coefficients: std_err values": (
                "output/ols_coefficients.csv", edit(coefficients, "bmi", "std_err", "0.35")),
            "coefficients: confidence interval values": (
                "output/ols_coefficients.csv", edit(coefficients, "age", "ci_upper", "0.80")),
            "new-patient intervals: columns": (
                "output/new_patient_intervals.csv", "\n".join(line + ",x" for line in intervals.splitlines()) + "\n"),
            "new-patient intervals: one row for the new patient": (
                "output/new_patient_intervals.csv", edit(intervals, "60", "age", "61")),
            "new-patient intervals: mean": ("output/new_patient_intervals.csv", edit(intervals, "60", "mean", "152.0")),
            "new-patient intervals: mean-response interval": (
                "output/new_patient_intervals.csv", edit(intervals, "60", "mean_ci_lower", "136.33")),
            "new-patient intervals: prediction interval": (
                "output/new_patient_intervals.csv", edit(intervals, "60", "obs_ci_upper", "154.17")),
            "residuals: columns": ("output/ols_residuals.csv", rename(residuals, "residual", "resid")),
            "residuals: one row per patient": ("output/ols_residuals.csv", drop_last_row(residuals)),
            "residuals: observed values": ("output/ols_residuals.csv", edit(residuals, "P05", "observed", "145")),
            "residuals: fitted values": ("output/ols_residuals.csv", edit(residuals, "P02", "fitted", "160.0")),
            "residuals: residual values": ("output/ols_residuals.csv", edit(residuals, "P03", "residual", "6.5368")),
            "residual plot: PNG image": (value_checks.RESIDUAL_PLOT, b"\xff\xd8\xff\xe0 a JPEG saved as .png"),
            "availability: columns": ("output/availability_decisions.csv", rename(decisions, "decision", "choice")),
            "availability: one row per candidate": ("output/availability_decisions.csv", drop_last_row(decisions)),
            "availability: hours_after_visit values": (
                "output/availability_decisions.csv", edit(decisions, "a1c_result", "hours_after_visit", "12")),
            "availability: available values": (
                "output/availability_decisions.csv", edit(decisions, "callback_sbp", "available", "True")),
            "availability: decision values": (
                "output/availability_decisions.csv", edit(decisions, "bmi", "decision", "Exclude (leakage)")),
            "split summary: columns": ("output/split_summary.csv", rename(split, "row_count", "rows")),
            "split summary: one row per partition": ("output/split_summary.csv", drop_last_row(split)),
            "split summary: row_count values": ("output/split_summary.csv", edit(split, "train", "row_count", "31")),
            "split summary: first_target_time values": (
                "output/split_summary.csv", edit(split, "validation", "first_target_time", "2026-04-30 17:07:00+00:00")),
            "split summary: last_target_time values": (
                "output/split_summary.csv", edit(split, "test", "last_target_time", "2026-05-18")),
            "validation metrics: columns": ("output/validation_metrics.csv", rename(validation, "r2", "r_squared")),
            "validation metrics: one row per approach": ("output/validation_metrics.csv", drop_last_row(validation)),
            "validation metrics: mae values": (
                "output/validation_metrics.csv", edit(validation, "mean_baseline", "mae", "7.5")),
            "validation metrics: rmse values": (
                "output/validation_metrics.csv", edit(validation, "linear_pipeline", "rmse", "3.9")),
            "validation metrics: r2 values": (
                "output/validation_metrics.csv", edit(validation, "mean_baseline", "r2", "0.233125")),
            "test metrics: columns": (
                "output/test_metrics.csv", "\n".join(line + ",x" for line in test_metrics.splitlines()) + "\n"),
            "test metrics: one row for the frozen approach": (
                "output/test_metrics.csv", test_metrics.replace("linear_pipeline", "ridge_pipeline")),
            "test metrics: mae value": ("output/test_metrics.csv", edit(test_metrics, "linear_pipeline", "mae", "3.0")),
            "test metrics: rmse value": ("output/test_metrics.csv", edit(test_metrics, "linear_pipeline", "rmse", "4.0")),
            "test metrics: r2 value": ("output/test_metrics.csv", edit(test_metrics, "linear_pipeline", "r2", "0.5")),
            "test predictions: columns": (
                "output/test_predictions.csv", rename(predictions, "predicted_sbp", "prediction")),
            "test predictions: one row per test visit": ("output/test_predictions.csv", drop_last_row(predictions)),
            "test predictions: followup_time and sbp_followup values": (
                "output/test_predictions.csv", edit(predictions, "V40", "sbp_followup", "143")),
            "test predictions: predicted_sbp values": (
                "output/test_predictions.csv", edit(predictions, "V41", "predicted_sbp", "149")),
            "readmission metrics: columns": (
                "output/readmission_metrics.csv", rename(readmission, "recall", "sensitivity")),
            "readmission metrics: one row per flag": ("output/readmission_metrics.csv", drop_last_row(readmission)),
            "readmission metrics: accuracy values": (
                "output/readmission_metrics.csv", edit(readmission, "model_flag", "accuracy", "0.75")),
            "readmission metrics: precision values": (
                "output/readmission_metrics.csv", edit(readmission, "model_flag", "precision", "0.75")),
            "readmission metrics: recall values": (
                "output/readmission_metrics.csv", edit(readmission, "never_flag", "recall", "1.0")),
        }
        assert sorted(mistakes) == sorted(NAMES)
        for number, (name, (path, changed)) in enumerate(mistakes.items()):
            files = dict(correct)
            files[path] = changed
            result = graded(submission(f"mistake-{number:02}", files))
            assert failing(result) == {name}, (name, {other: detail(result, other) for other in failing(result)})
            assert result["score"] == 100 - cost(name), (name, result["score"])

        # A missing file costs only its own artifact's checks.
        for number, path in enumerate(correct):
            files = {other: text for other, text in correct.items() if other != path}
            result = graded(submission(f"missing-{number}", files))
            assert failing(result) == set(artifact_checks(path)), (path, failing(result))
            # The other required files are named correctly, so none of them is offered as a look-alike.
            for name in failing(result):
                assert "Found " not in detail(result, name), (path, detail(result, name))

        # Labels the checks do not recognize cost the rows check alone; the values beneath them are still read.
        files = dict(correct)
        files["output/validation_metrics.csv"] = (
            validation.replace("mean_baseline", "baseline").replace("linear_pipeline", "pipeline"))
        result = graded(submission("relabeled", files))
        assert failing(result) == {"validation metrics: one row per approach"}, failing(result)
        files["output/validation_metrics.csv"] = edit(files["output/validation_metrics.csv"], "pipeline", "mae", "3.9")
        result = graded(submission("relabeled-wrong-mae", files))
        assert failing(result) == {"validation metrics: one row per approach", "validation metrics: mae values"}, (
            failing(result))
        assert "linear_pipeline has mae 3.9, expected 3.34" in detail(result, "validation metrics: mae values")

        # A timestamp ending in UTC, as strftime("%Z") writes it, is the same instant.
        files = dict(correct)
        files["output/split_summary.csv"] = split.replace("+00:00", " UTC")
        assert graded(submission("utc-suffix", files))["score"] == 100

        # Realistic wrong answers land on the checks that name them.
        data = load()
        visits = data["visits"]
        leaky = FEATURES + ["callback_sbp"]
        train = visits[visits["followup_time"] < VALIDATION_START]
        valid = visits[(visits["followup_time"] >= VALIDATION_START) & (visits["followup_time"] < TEST_START)]
        test = visits[visits["followup_time"] >= TEST_START]
        pipeline = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(train[leaky], train[TARGET])
        baseline = DummyRegressor(strategy="mean").fit(train[leaky], train[TARGET])
        files = dict(correct)
        files["output/validation_metrics.csv"] = pd.DataFrame([
            {"approach": name, **metrics(valid[TARGET], fitted.predict(valid[leaky]))}
            for name, fitted in [("mean_baseline", baseline), ("linear_pipeline", pipeline)]
        ]).to_csv(index=False)
        result = graded(submission("leaky", files))
        assert failing(result) == {"validation metrics: mae values"}, failing(result)
        message = detail(result, "validation metrics: mae values")
        assert "linear_pipeline has mae" in message and "expected 3.34" in message and "age, bmi, and sbp_today" in message
        assert "also uses a1c_result or callback_sbp" in message, message

        # Splitting on the visit time instead of the target time.
        by_visit = {"train": visits[visits["visit_time"] < VALIDATION_START],
                    "validation": visits[(visits["visit_time"] >= VALIDATION_START) & (visits["visit_time"] < TEST_START)],
                    "test": visits[visits["visit_time"] >= TEST_START]}
        files = dict(correct)
        files["output/split_summary.csv"] = pd.DataFrame({
            "partition": list(by_visit),
            "row_count": [len(part) for part in by_visit.values()],
            "first_target_time": [part["followup_time"].min() for part in by_visit.values()],
            "last_target_time": [part["followup_time"].max() for part in by_visit.values()],
        }).to_csv(index=False)
        result = graded(submission("split-on-visit", files))
        assert "split summary: row_count values" in failing(result), failing(result)
        message = detail(result, "split summary: row_count values")
        assert "train has row_count 44, expected 30" in message and "split on followup_time" in message, message
        assert "a split on visit_time gives" in message, message
        assert failing(result) == {"split summary: row_count values"}, failing(result)

        # Other common slips cost only the checks they touch, and the feedback names their cause.
        def slip(label: str, path: str, contents: str, names: set[str], cause: str) -> None:
            files = dict(correct)
            files[path] = contents
            result = graded(submission(label, files))
            assert failing(result) == names, (label, failing(result))
            for name in names:
                assert cause in detail(result, name), (label, detail(result, name))

        frame = frames["output/ols_residuals.csv"].copy()
        frame["residual"] = frame["fitted"] - frame["observed"]
        slip("residual-sign", "output/ols_residuals.csv", frame.to_csv(index=False),
             {"residuals: residual values"}, "opposite sign")
        # Only three flipped: the cause describes the rows listed, never every residual.
        frame = frames["output/ols_residuals.csv"].copy()
        frame.loc[:2, "residual"] = -frame.loc[:2, "residual"]
        slip("residual-sign-partial", "output/ols_residuals.csv", frame.to_csv(index=False),
             {"residuals: residual values"}, "Those residuals have the opposite sign")
        files = dict(correct)
        files["output/ols_residuals.csv"] = frame.to_csv(index=False)
        message = detail(graded(submission("residual-sign-partial-wording", files)), "residuals: residual values")
        assert "P04" not in message and "Every" not in message, message
        frame = frames["output/ols_residuals.csv"].copy()
        frame[["fitted", "residual"]] = frame[["residual", "fitted"]].to_numpy()
        slip("residual-columns-swapped", "output/ols_residuals.csv", frame.to_csv(index=False),
             {"residuals: fitted values"}, "traded places")
        # The right split, with ranges taken from visit_time.
        frame = frames["output/split_summary.csv"].copy()
        frame["first_target_time"] = [part["visit_time"].min() for part in (train, valid, test)]
        frame["last_target_time"] = [part["visit_time"].max() for part in (train, valid, test)]
        slip("ranges-from-visit-time", "output/split_summary.csv", frame.to_csv(index=False),
             {"split summary: first_target_time values"},
             "Those are visit_time values")
        # RMSE saved without np.sqrt(), on validation and on test rows.
        frame = frames["output/validation_metrics.csv"].copy()
        frame["rmse"] = frame["rmse"] ** 2
        slip("rmse-not-rooted", "output/validation_metrics.csv", frame.to_csv(index=False),
             {"validation metrics: rmse values"}, "mean squared errors")
        frame = frames["output/test_metrics.csv"].copy()
        frame["rmse"] = frame["rmse"] ** 2
        slip("test-rmse-not-rooted", "output/test_metrics.csv", frame.to_csv(index=False),
             {"test metrics: rmse value"}, "mean squared error")
        # Test predictions from a pipeline that also uses a leaky feature.
        leaky_final = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(
            train[FEATURES + ["a1c_result"]], train[TARGET])
        frame = frames["output/test_predictions.csv"].copy()
        frame["predicted_sbp"] = leaky_final.predict(test[FEATURES + ["a1c_result"]])
        slip("leaky-test-predictions", "output/test_predictions.csv", frame.to_csv(index=False),
             {"test predictions: predicted_sbp values"}, "also uses a1c_result or callback_sbp")
        # Test metrics that only happen to land near a leaky score get the neutral hint, and each wrong
        # metric is reported; leaky test metrics are named when the validation metrics show the leak.
        files = dict(correct)
        files["output/test_metrics.csv"] = "approach,mae,rmse,r2\nlinear_pipeline,3.0,4.0,1.0\n"
        result = graded(submission("near-leaky-test-metrics", files))
        names = {"test metrics: mae value", "test metrics: rmse value", "test metrics: r2 value"}
        assert failing(result) == names, failing(result)
        assert not any("also uses" in detail(result, name) for name in names), result
        files["output/validation_metrics.csv"] = pd.DataFrame([
            {"approach": name, **metrics(valid[TARGET], fitted.predict(valid[leaky]))}
            for name, fitted in [("mean_baseline", baseline), ("linear_pipeline", pipeline)]
        ]).to_csv(index=False)
        files["output/test_metrics.csv"] = pd.DataFrame([
            {"approach": "linear_pipeline", **metrics(test[TARGET], pipeline.predict(test[leaky]))}
        ]).to_csv(index=False)
        result = graded(submission("leaky-validation-and-test-metrics", files))
        assert failing(result) == {"validation metrics: mae values", "test metrics: mae value"}, failing(result)
        assert "test scores of a pipeline that also uses" in detail(result, "test metrics: mae value"), result
        # The Task 3.3 save line copied from Task 3.1 writes the readmission table over validation_metrics.csv.
        files = dict(correct)
        files["output/validation_metrics.csv"] = readmission
        del files["output/readmission_metrics.csv"]
        message = detail(graded(submission("readmission-over-validation", files)), "validation metrics: columns")
        assert "the one Task 3.3 writes to output/readmission_metrics.csv" in message, message
        # Test metrics share validation's header, so a mix-up there names no other file.
        files = dict(correct)
        files["output/validation_metrics.csv"] = rename(validation, "r2", "r_squared")
        message = detail(graded(submission("renamed-no-owner", files)), "validation metrics: columns")
        assert "The saved header is" not in message, message
        frame = data["candidates"].copy()
        frame["available"] = frame["hours_after_visit"] < 0
        frame["decision"] = np.where(frame["available"], "Keep", "Exclude (leakage)")
        slip("strict-availability", "output/availability_decisions.csv", frame.to_csv(index=False),
             {"availability: available values"}, "hours_after_visit < 0 gives")
        frame = frames["output/validation_metrics.csv"].copy()
        baseline = DummyRegressor(strategy="mean").fit(valid[FEATURES], valid[TARGET])
        frame.loc[frame["approach"] == "mean_baseline", ["mae", "rmse", "r2"]] = list(
            metrics(valid[TARGET], baseline.predict(valid[FEATURES])).values())
        slip("baseline-from-validation", "output/validation_metrics.csv", frame.to_csv(index=False),
             {"validation metrics: mae values"},
             "a mean taken from the validation rows")
        frame = frames["output/validation_metrics.csv"].query("approach == 'linear_pipeline'")
        slip("test-scored-on-validation", "output/test_metrics.csv", frame.to_csv(index=False),
             {"test metrics: mae value"}, "validation scores")
        flags = data["flags"]
        frame = frames["output/readmission_metrics.csv"].copy()
        frame["precision"] = [precision_score(flags[column], flags["readmitted_30d"], zero_division=0)
                              for column in frame["approach"]]
        frame["recall"] = [recall_score(flags[column], flags["readmitted_30d"], zero_division=0)
                           for column in frame["approach"]]
        slip("flag-passed-first", "output/readmission_metrics.csv", frame.to_csv(index=False),
             {"readmission metrics: precision values"}, "traded places")
        # A wrong value that no slip explains names no cause.
        files = dict(correct)
        files["output/validation_metrics.csv"] = mistakes["validation metrics: mae values"][1]
        message = detail(graded(submission("no-cause", files)), "validation metrics: mae values")
        assert "Those are" not in message and "That is" not in message, message

        # Feedback names what was expected and what was found.
        files = dict(correct)
        files["output/readmission_metrics.csv"] = mistakes["readmission metrics: precision values"][1]
        message = detail(graded(submission("feedback-precision", files)), "readmission metrics: precision values")
        assert "model_flag has precision 0.75, expected 0.6" in message and "Task 3.3" in message, message
        files = dict(correct)
        files["output/availability_decisions.csv"] = mistakes["availability: available values"][1]
        message = detail(graded(submission("feedback-available", files)), "availability: available values")
        assert "callback_sbp has available True, expected False" in message and "<= 0" in message, message
        files = dict(correct)
        files["output/test_predictions.csv"] = mistakes["test predictions: predicted_sbp values"][1]
        message = detail(graded(submission("feedback-prediction", files)), "test predictions: predicted_sbp values")
        assert "V41 has predicted_sbp 149, expected 139.91 or 140.13" in message and "Task 3.2" in message, message
        files = dict(correct)
        files["output/split_summary.csv"] = mistakes["split summary: last_target_time values"][1]
        message = detail(graded(submission("feedback-time", files)), "split summary: last_target_time values")
        assert "test has last_target_time 2026-05-18, expected 2026-05-18T21:51:00Z" in message, message

        # pytest, which GitHub runs, reports one test per check.
        for name, expected_failures in (("correct", 0), ("scaffold", len(NAMES))):
            target = workspace / f"pytest-{name}"
            shutil.copytree(HANDOUT, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv"))
            if name == "correct":
                for path, contents in correct.items():
                    if isinstance(contents, bytes):
                        (target / path).write_bytes(contents)
                    else:
                        (target / path).write_text(contents, encoding="utf-8")
            finished = subprocess.run(
                [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", ".github/test/test_assignment.py"],
                cwd=target, capture_output=True, text=True, check=False,
            )
            summary_line = finished.stdout.strip().splitlines()[-1]
            if expected_failures:
                assert f"{expected_failures} failed" in summary_line, finished.stdout
            else:
                assert finished.returncode == 0 and f"{len(NAMES)} passed" in summary_line, finished.stdout

    print(
        "Assignment 10 checks: the expected values match the data, and the README checkpoints, contract, and "
        "numbering and the handout's byte-identical checks agree; empty and scaffold earn 0; correct, refit, "
        "alternative, with-index, semicolon- and tab-separated, one-decimal, and percent submissions earn 100; a coefficient table without its terms costs only the "
        f"columns check; each of the {len(NAMES)} single mistakes and each missing file costs only its own checks; "
        "leaky features, a split on the visit time, and the other common slips land on the checks that name "
        "their cause; feedback and pytest pass."
    )


if __name__ == "__main__":
    run()
