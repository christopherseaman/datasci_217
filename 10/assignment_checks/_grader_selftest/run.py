"""Development regression checks for the Assignment 10 checks.

Recomputes every frozen value in `_value_checks.py` from `10/assignment/data/`
with `scripts/build_assignment10_data.py`, answers the assignment the way the
README does, builds submissions in ignored `scratch/`, and confirms what each
kind scores: a correct answer and the same answers written differently earn
100; an empty folder and the untouched handout earn 0; one mistake per check
costs exactly that check; each common slip is charged once, at the check that
names its cause; and the handout's README, CHECKS.md, notebook, and checker
files agree with the checks.

    uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 \\
        --with statsmodels==0.14.6 --with scikit-learn==1.9.0 --with 'pytest>=8,<9' \\
        python 10/assignment_checks/_grader_selftest/run.py
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib

import numpy as np
import pandas as pd

# Self-tests grade with the working copy, never the copy downloaded from GitHub.
os.environ["DS217_LOCAL_CHECKS"] = "1"

CHECKS = Path(__file__).resolve().parents[1]
HANDOUT = CHECKS.parent / "assignment"
ROOT = CHECKS.parents[1]
SCRATCH = ROOT / "scratch" / "a10_selftest"
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(ROOT / "scripts"))
sys.dont_write_bytecode = True
import _value_checks as value_checks  # noqa: E402
import build_assignment10_data as build  # noqa: E402
from grading import POINTS, grade_submission  # noqa: E402

NAMES = [check.name for check in value_checks.CHECKS]
POINTS_BY_NAME = dict(zip(NAMES, POINTS, strict=True))
ARTIFACT_OF = {  # the short names the frozen variants use
    "coefficients": value_checks.COEFFICIENTS, "intervals": value_checks.NEW_PATIENT_INTERVALS,
    "bands": value_checks.RESIDUAL_BANDS, "availability": value_checks.AVAILABILITY, "split": value_checks.SPLIT,
    "validation": value_checks.VALIDATION, "test": value_checks.TEST_METRICS, "clinics": value_checks.CLINIC_ERRORS,
    "readmission": value_checks.READMISSION,
}


def minimal_png() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\x00\x00")) + chunk(b"IEND", b"")


def load() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(HANDOUT / "data" / f"{name}.parquet")
            for name in ("bp_patients", "followup_visits", "feature_candidates", "discharges")}


def frame(artifact, rows: dict) -> pd.DataFrame:
    """A DataFrame of the artifact's expected rows, key columns first, timestamps as UTC text."""
    records = []
    for key, values in rows.items():
        record = dict(zip(artifact.key, key))
        for column, value in values.items():
            if isinstance(value, value_checks.Instant):
                value = value.text
            elif isinstance(value, value_checks.Flag):
                value = value.value
            elif isinstance(value, value_checks.Starts):
                value = "Keep" if value.word == "keep" else "Exclude (leakage)"
            record[column] = value
        records.append(record)
    return pd.DataFrame(records, columns=[*artifact.columns])


def correct() -> dict[str, pd.DataFrame]:
    return {artifact.path: frame(artifact, artifact.rows) for artifact in value_checks.ARTIFACTS}


def write(root: Path, frames: dict[str, pd.DataFrame], png: bytes | None = None, raw: dict | None = None) -> Path:
    if root.exists():
        shutil.rmtree(root)
    (root / "output").mkdir(parents=True)
    for path, table in frames.items():
        table.to_csv(root / path, index=False)
    for path, text in (raw or {}).items():
        (root / path).write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    (root / value_checks.RESIDUAL_PLOT).write_bytes(minimal_png() if png is None else png)
    return root


def cli_report(checks: Path, root: Path) -> dict:
    finished = subprocess.run([sys.executable, "-B", str(checks / "check_assignment.py"), str(root), "--json"],
                              capture_output=True, text=True, check=False)
    assert finished.stdout, f"{checks / 'check_assignment.py'} printed nothing: {finished.stderr}"
    report = json.loads(finished.stdout)
    assert (finished.returncode == 0) == (report["score"] == report["max-score"]), (checks, root)
    return report


def graded(root: Path, cli: bool = False) -> dict:
    result = grade_submission(root)
    assert result["schema"] == "datasci217/grading-result/v1"
    assert result["max-score"] == 100 == sum(POINTS)
    if cli:
        assert cli_report(CHECKS, root) == result, root
        assert cli_report(HANDOUT, root) == result, root
    return result


def failing(result: dict) -> list[str]:
    return [test["test-name"] for test in result["tests"] if not test["passed"]]


def detail(result: dict, name: str) -> str:
    return next(test["detail"] for test in result["tests"] if test["test-name"] == name)


STEMS = {
    "output/ols_coefficients.csv": "coefficients", "output/new_patient_intervals.csv": "new-patient intervals",
    "output/residual_bands.csv": "residual bands", "output/availability_decisions.csv": "availability",
    "output/split_summary.csv": "split summary", "output/validation_metrics.csv": "validation metrics",
    "output/test_metrics.csv": "test metrics", "output/clinic_test_errors.csv": "clinic errors",
    "output/readmission_metrics.csv": "readmission metrics",
}


def artifact_checks(artifact) -> list[str]:
    return [name for name in NAMES if name.startswith(STEMS[artifact.path] + ":")]


def check_expected_values() -> None:
    """Every frozen value matches a fresh computation from the handout's data."""
    fresh = build.expected(load())
    for name, value in fresh.items():
        frozen = getattr(value_checks, name)
        assert frozen == value, f"{name} in _value_checks.py differs from a fresh computation; rebuild the data"
    print(f"frozen values: {len(fresh)} names match a fresh computation")


def check_scores() -> None:
    base = correct()
    result = graded(write(SCRATCH / "correct", base), cli=True)
    assert result["score"] == 100, failing(result)

    # The same answers written differently: one decimal, CRLF, BOM, semicolons with decimal commas, tabs,
    # headers in capitals, columns reversed, a row-number column, percents, 1/0 booleans, other labels.
    styled = {}
    for path, table in base.items():
        out = table.iloc[::-1, ::-1].copy()
        for column in out.columns:
            if column in value_checks.PERCENT_COLUMNS:
                out[column] = [f"{value * 100:.1f}%" for value in out[column]]
            elif out[column].dtype == float:
                out[column] = out[column].round(2)
        styled[path] = out
    styled[value_checks.AVAILABILITY.path]["available"] = styled[value_checks.AVAILABILITY.path]["available"].astype(int)
    styled[value_checks.SPLIT.path]["partition"] = ["Testing", "valid", "fit_rows", "Training"]
    styled[value_checks.RESIDUAL_BANDS.path]["age_band"] = [
        f"({low}, {high}]" for low, high in reversed(list(zip(value_checks.AGE_BINS[:-1], value_checks.AGE_BINS[1:])))]
    styled[value_checks.COEFFICIENTS.path]["term"] = ["bmi", "age", "const"]
    raw = {}
    for number, (path, table) in enumerate(styled.items()):
        table = table.rename(columns=str.upper)
        if number % 3 == 0:
            raw[path] = "﻿" + table.to_csv(lineterminator="\r\n")
        elif number % 3 == 1:
            raw[path] = table.to_csv(sep=";", decimal=",", index=False)
        else:
            raw[path] = table.to_csv(sep="\t", index=False)
    result = graded(write(SCRATCH / "styled", {}, raw=raw), cli=True)
    assert result["score"] == 100, [(name, detail(result, name)) for name in failing(result)]

    # Real student data and code in the folder change nothing: the checks read output/ only.
    poisoned = write(SCRATCH / "poisoned", base)
    (poisoned / "data").mkdir()
    (poisoned / "data" / "bp_patients.parquet").write_bytes(b"not parquet")
    (poisoned / "grading_helper.py").write_text("raise SystemExit('student code ran')\n")
    assert graded(poisoned)["score"] == 100

    empty = SCRATCH / "empty"
    shutil.rmtree(empty, ignore_errors=True)
    empty.mkdir(parents=True)
    assert graded(empty, cli=True)["score"] == 0
    assert graded(HANDOUT)["score"] == 0
    print("scores: correct 100, styled 100, poisoned 100, empty 0, handout 0")


def wrong(value):
    """A clearly wrong value of the same kind."""
    if isinstance(value, (bool, np.bool_)):
        return not value
    if isinstance(value, (int, np.integer)):
        return value + 7
    if isinstance(value, (float, np.floating)):
        return value + 7.5
    if isinstance(value, str) and value.endswith("Z"):
        return "2020-01-01T00:00:00Z"
    return "Exclude (leakage)" if str(value).startswith("Keep") else "Keep"


# Values checks whose name does not spell out the one column a single mistake changes.
COLUMN_FOR = {
    "coefficients: confidence interval values": "ci_lower",
    "new-patient intervals: mean": "mean",
    "new-patient intervals: mean-response interval": "mean_ci_lower",
    "new-patient intervals: prediction interval": "obs_ci_upper",
}


def check_single_mistakes() -> None:
    """One mistake per check costs exactly that check."""
    base = correct()
    for name in NAMES:
        frames = {path: table.copy() for path, table in base.items()}
        png = None
        if name == "residual plot: PNG image":
            png = b"not a png"
        else:
            artifact = next(a for a in value_checks.ARTIFACTS if name in artifact_checks(a))
            table = frames[artifact.path]
            value_columns = [c for c in artifact.columns if c not in artifact.key]
            if name.endswith(": columns"):
                # A value column saved under another name with the right values costs only the columns check.
                table = table.rename(columns={value_columns[0]: "my_" + value_columns[0]})
            elif ": one row" in name and len(artifact.rows) == 1:
                table = table.astype(object)
                table.loc[0, artifact.key[-1]] = "lasso" if artifact.key[-1] == "approach" else 61
            elif ": one row" in name:
                table = table.iloc[:-1]
            else:
                column = COLUMN_FOR.get(name) or re.match(r".*: (\w+) values?$", name).group(1)
                table.loc[0, column] = wrong(table.loc[0, column])
            frames[artifact.path] = table
        result = graded(write(SCRATCH / "single", frames, png=png))
        assert failing(result) == [name], (name, failing(result), [detail(result, n) for n in failing(result)])
        message = detail(result, name)
        assert any(word in message for word in ("expected", "missing", "PNG", "should hold")), message
    print(f"single mistakes: each of {len(NAMES)} checks costs only itself")


# The first check each slip should reach, and the only one it should cost.
SLIP_CHARGED = {
    "formula_without_age": "coefficients: one row per term",
    "formula_without_bmi": "coefficients: one row per term",
    "intervals_at_90": "new-patient intervals: prediction interval",  # the 90% mean interval rounds to the 95% one
    "bands_size_with_missing": "residual bands: patients values",
    "bands_sign_flipped": "residual bands: mean_residual values",
    "features_strict_zero": "availability: available values",
    "features_equals_zero": "availability: available values",
    "features_no_audit": "availability: available values",
    "split_on_visit_time": "split summary: row_count values",
    "ranges_from_visit_time": "split summary: first_target_time values",
    "fit_not_trimmed": "split summary: row_count values",
    "baseline_from_validation": "validation metrics: mae values",
    "rmse_not_rooted": "validation metrics: rmse values",
    "r2_args_swapped": "validation metrics: r2 values",
    "validation_on_test_rows": "validation metrics: mae values",
    "refit_on_validation": "test metrics: mae value",
    "test_scored_on_validation": "test metrics: mae value",
    "mean_error_sign_flipped": "clinic errors: mean_error values",
    "lace_strictly_greater": "readmission metrics: flagged values",
    "flag_passed_first": "readmission metrics: precision values",
}


def check_slips() -> None:
    """Each common wrong answer costs its first check once, and the message names its cause."""
    assert set(SLIP_CHARGED) == set(value_checks.VARIANTS) == set(value_checks.CAUSES), "update SLIP_CHARGED"
    hours = value_checks.CANDIDATE_HOURS
    for variant, artifacts in value_checks.VARIANTS.items():
        frames = correct()
        for short, rows in artifacts.items():
            artifact = ARTIFACT_OF[short]
            table = frames[artifact.path].set_index(list(artifact.key)).astype(object)
            for key, values in rows.items():
                for column, value in values.items():
                    table.loc[key if len(key) > 1 else key[0], column] = value
            frames[artifact.path] = table.reset_index()[list(artifact.columns)]
        if variant.startswith("formula_without_"):
            # The left-out term has no row at all.
            term = variant.removeprefix("formula_without_")
            table = frames[value_checks.COEFFICIENTS.path]
            frames[value_checks.COEFFICIENTS.path] = table[table["term"] != term]
        if variant in value_checks.FEATURE_RULES:
            rule = value_checks.FEATURE_RULES[variant]
            table = frames[value_checks.AVAILABILITY.path]
            table["available"] = [rule(hours[feature]) for feature in table["candidate_feature"]]
            table["decision"] = np.where(table["available"], "Keep", "Exclude (leakage)")
        result = graded(write(SCRATCH / "slips", frames))
        expected = SLIP_CHARGED[variant]
        assert failing(result) == [expected], (variant, failing(result),
                                               [detail(result, name) for name in failing(result)])
        if not variant.startswith("formula_without_"):
            assert value_checks.CAUSES[variant] in detail(result, expected), (variant, detail(result, expected))
    print(f"slips: each of {len(value_checks.VARIANTS)} common mistakes costs one check and names its cause")


def check_missing_file() -> None:
    for artifact in value_checks.ARTIFACTS:
        frames = {path: table for path, table in correct().items() if path != artifact.path}
        result = graded(write(SCRATCH / "missing", frames))
        assert failing(result) == artifact_checks(artifact), (artifact.path, failing(result))
    print("missing files: each costs only its own artifact's checks")


def check_readme() -> None:
    readme = (HANDOUT / "README.md").read_text(encoding="utf-8")
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    checks_md = (HANDOUT / "CHECKS.md").read_text(encoding="utf-8")
    assert "(CHECKS.md)" in readme
    cells = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    wanted = {artifact.path: set(artifact.columns) | set(artifact.optional_columns)
              for artifact in value_checks.ARTIFACTS}
    checkpoints = re.findall(r"> \*\*Checkpoint: `([^`]+)`\*\*\n> The header line `([^`]+)`", readme)
    assert {path: set(header.split(",")) for path, header in checkpoints} == wanted, checkpoints
    in_notebook = re.findall(r"> \*\*Checkpoint: `([^`]+)`\*\*\n> .*under the header `([^`]+)`", cells)
    assert dict(in_notebook) == dict(checkpoints), (in_notebook, checkpoints)
    contract = re.findall(r"^\| `(output/[^`]+)` \| .* \| ([^|]+) \| (\d+) \|$", checks_md, re.M)
    assert [(name.strip(), int(points)) for _, name, points in contract] == list(zip(NAMES, POINTS)), contract
    missing = value_checks._missing(HANDOUT, value_checks.COEFFICIENTS.path, value_checks.COEFFICIENTS.task)
    assert f"[FIX ]  0/{POINTS[0]}  {NAMES[0]}\n         {missing}" in checks_md
    assert f"[PASS]  {POINTS[-1]}/{POINTS[-1]}  {NAMES[-1]}\n\nScore: 100/100" in checks_md
    headings = lambda text: [line for line in text.splitlines() if re.match(r"#{2,3} (Task \d|\d\.\d)", line)]
    assert headings(cells) == headings(readme), (headings(cells), headings(readme))
    for text in (readme, checks_md, json.dumps(notebook, ensure_ascii=False),
                 (CHECKS / "_value_checks.py").read_text(encoding="utf-8")):
        assert "\u2014" not in text and "\u2013" not in text
    # Every count the README promises for the data matches the files.
    assert f"{value_checks.ROWS_USED} of {value_checks.PATIENT_ROWS}" in readme
    assert f"visits: ({value_checks.VISIT_ROWS}, 15)" in readme
    assert f"features: {list(value_checks.FEATURES)}" in readme
    # No full answer expressions: the README and notebook name methods and arguments but never print a finished call.
    for leaked in ('"sbp ~ age + bmi"', "get_prediction(new_patient", ".summary_frame(alpha", "conf_int()[",
                   'train["followup_time"] <=', 'valid["visit_time"].min()', "Pipeline([",
                   'SimpleImputer(strategy="median")', "np.sqrt(mean_squared_error", '>= 13).astype(int)',
                   '["lace_score"] >=', "groupby(\"age_band\"", "groupby(\"clinic\""):
        for text in (readme, cells):
            assert leaked not in text, leaked
    print("README, CHECKS.md, and notebook agree with the checks")


def check_handout_files() -> None:
    workflow = (HANDOUT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    listed = re.search(r"^  CHECKS_FILES: \|\n((?:    \S.*\n)+)", workflow, re.M).group(1).split()
    assert 'CHECKS_PATH: "10/assignment_checks"' in workflow
    shipped = sorted(
        path.relative_to(CHECKS).as_posix() for path in CHECKS.rglob("*")
        if path.is_file() and not {"__pycache__", ".pytest_cache", "_grader_selftest"} & set(path.relative_to(CHECKS).parts)
        and path.name != "README.md"
    )
    assert sorted(listed) == shipped, (listed, shipped)
    for name in listed:
        assert (HANDOUT / name).read_bytes() == (CHECKS / name).read_bytes(), f"{name} differs from the handout copy"
    for name in ("_value_checks.py", "grading.py"):
        cached = HANDOUT / ".checks" / name
        assert not cached.exists() or cached.read_bytes() == (CHECKS / name).read_bytes(), cached
    handout_python = sorted(
        path.relative_to(HANDOUT).as_posix() for path in HANDOUT.rglob("*.py")
        if not {"__pycache__", ".venv", ".checks"} & set(path.parts)
    )
    assert handout_python == sorted(name for name in listed if name.endswith(".py")), handout_python
    assert sorted(path.name for path in (HANDOUT / "output").iterdir()) == [".gitkeep"]
    assert sorted(path.name for path in (HANDOUT / "data").iterdir()) == sorted(
        f"{name}.parquet" for name in ("bp_patients", "followup_visits", "feature_candidates", "discharges"))
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None, cell.get("id")
    pytest_run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                 str(CHECKS / "test_assignment.py")], capture_output=True, text=True,
                                cwd=CHECKS, check=False)
    assert pytest_run.returncode == 1 and "failed" in pytest_run.stdout, pytest_run.stdout[-500:]
    print("handout files match the checks byte for byte; notebook outputs cleared")


def run() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    check_expected_values()
    check_scores()
    check_single_mistakes()
    check_slips()
    check_missing_file()
    check_readme()
    check_handout_files()
    print("All Assignment 10 self-tests passed.")


if __name__ == "__main__":
    run()
