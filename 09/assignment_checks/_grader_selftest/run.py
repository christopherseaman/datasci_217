"""Development regression checks for the Assignment 09 checks.

Builds submissions in ignored `scratch/`, answers the assignment with pandas
the way the lecture does, and confirms what each kind of submission scores: a
correct one earns 100, and so does one written differently (column and row
order, CRLF, a BOM, UTF-16, quoting, padding, letter case, number format,
timestamps in other spellings and zones, other spellings of True and False,
row-number columns, `NaN` for an empty cell, and `source_row` kept or left out
where it is optional). An empty directory and the untouched handout earn 0, one
wrong value or one misnamed column costs exactly its own check, a table saved
without its key columns costs only the columns check, unconverted clock times
cost only Task 1.1's UTC check, and a missing file costs only its own
artifact's checks. It also confirms that the copies of the data in
`_value_checks.py` match `09/assignment/data/`, that the README's checkpoints,
completion contract, and numbering agree with the checks and the notebook, that
the handout ships every file in `CHECKS_FILES` byte for byte and no other Python
file, and that the handout notebook ships with its outputs cleared.

    uv run --python 3.13 --with pandas==3.0.5 --with 'pytest>=8,<9' \\
        python 09/assignment_checks/_grader_selftest/run.py
"""

from __future__ import annotations

from pathlib import Path
import codecs
import json
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
import _value_checks as value_checks  # noqa: E402
from grading import POINTS, grade_submission  # noqa: E402

NAMES = [check.name for check in value_checks.CHECKS]
POINTS_BY_NAME = dict(zip(NAMES, POINTS, strict=True))
ARTIFACTS = {artifact.path: artifact for artifact in value_checks.ARTIFACTS}
PREFIX = {
    "output/prepared_vitals.csv": "prepared vitals:",
    "output/hourly_grid.csv": "hourly grid:",
    "output/two_hour_summary.csv": "two-hour summary:",
    "output/past_features.csv": "past features:",
    "output/lab_availability.csv": "lab availability:",
    "output/chronological_blocks.csv": "chronological blocks:",
}
CLOCK = "%Y-%m-%d %H:%M"
PREDICTION_TIME = pd.Timestamp("2026-01-20 18:00", tz="UTC")


def to_utc(text: pd.Series) -> pd.Series:
    """Task 1.1's conversion: New York clock text to UTC instants."""
    return pd.to_datetime(text, format=CLOCK).dt.tz_localize("America/New_York").dt.tz_convert("UTC")


def prepare(convert: bool = True) -> pd.DataFrame:
    """Task 1.1; with convert=False, the clock text is parsed but left on the New York clock with no zone."""
    vitals = pd.read_csv(HANDOUT / "data" / "vitals.csv")
    if convert:
        vitals["recorded_at"] = to_utc(vitals["recorded_at"])
    else:
        vitals["recorded_at"] = pd.to_datetime(vitals["recorded_at"], format=CLOCK)
    vitals = vitals.sort_values(["patient_id", "recorded_at"]).reset_index(drop=True)
    vitals["source_row"] = 1
    return vitals


def grid_from(vitals: pd.DataFrame) -> pd.DataFrame:
    assert vitals["recorded_at"].eq(vitals["recorded_at"].dt.floor("h")).all()
    grid = (vitals.set_index("recorded_at").groupby("patient_id")[["heart_rate", "source_row"]]
            .resample("h").asfreq().reset_index())
    grid["grid_created"] = grid["source_row"].isna()
    grid["value_missing"] = grid["source_row"].notna() & grid["heart_rate"].isna()
    return grid


def summary_from(vitals: pd.DataFrame) -> pd.DataFrame:
    return (vitals.set_index("recorded_at").groupby("patient_id").resample("2h")
            .agg(mean_hr=("heart_rate", "mean"), n_rows=("source_row", "count")))


def features_from(vitals: pd.DataFrame) -> pd.DataFrame:
    features = vitals[["patient_id", "recorded_at", "heart_rate"]].copy()
    by_patient = features.groupby("patient_id")["heart_rate"]
    features["previous_hr"] = by_patient.shift(1)
    features["hr_change"] = by_patient.diff()
    features["mean_prev_2"] = by_patient.transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    prev_2h = (features.set_index("recorded_at").groupby("patient_id")["heart_rate"]
               .rolling("2h", closed="left").mean().rename("mean_prev_2h").reset_index())
    return features.merge(prev_2h, on=["patient_id", "recorded_at"], validate="one_to_one")


def solve() -> dict[str, pd.DataFrame]:
    """Answer the assignment with the lecture's pandas calls; returns {artifact path: frame to save}."""
    vitals = prepare()
    assert str(vitals["recorded_at"].dt.tz) == "UTC" and len(vitals) == 12

    grid = grid_from(vitals)
    assert len(grid) == 16 and grid["grid_created"].sum() == 4 and grid["value_missing"].sum() == 1

    summary = summary_from(vitals).reset_index()
    assert len(summary) == 9 and summary["n_rows"].sum() == 12

    features = features_from(vitals)
    assert len(features) == 12

    labs = pd.read_csv(HANDOUT / "data" / "labs.csv")
    for column in ("collected_at", "resulted_at"):
        labs[column] = to_utc(labs[column])
    labs["available"] = labs["resulted_at"] <= PREDICTION_TIME
    assert labs["available"].sum() == 3

    blocks = vitals.copy()
    blocks["block"] = np.where(blocks["recorded_at"] < PREDICTION_TIME, "earlier", "later_holdout")
    assert pd.crosstab(blocks["patient_id"], blocks["block"]).to_numpy().tolist() == [[4, 2], [4, 2]]

    return {
        "output/prepared_vitals.csv": vitals,
        "output/hourly_grid.csv": grid,
        "output/two_hour_summary.csv": summary,
        "output/past_features.csv": features,
        "output/lab_availability.csv": labs,
        "output/chronological_blocks.csv": blocks,
    }


def as_files(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """The way the README saves each frame: index=False."""
    return {path: frame.to_csv(index=False) for path, frame in frames.items()}



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


def notebook_cells() -> list[dict]:
    return json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))["cells"]


def check_supplied_data() -> None:
    """The copies of the data in _value_checks.py match the handout's data files, README, and notebook."""
    vitals = pd.read_csv(HANDOUT / "data" / "vitals.csv", dtype={"recorded_at": str})
    assert tuple(vitals.columns) == value_checks.VITALS_COLUMNS
    rows = tuple(
        (row.patient_id, row.recorded_at, None if pd.isna(row.heart_rate) else int(row.heart_rate))
        for row in vitals.itertuples()
    )
    assert rows == value_checks.VITALS, rows
    labs = pd.read_csv(HANDOUT / "data" / "labs.csv", dtype=str)
    assert tuple(labs.columns) == value_checks.LABS_COLUMNS
    assert tuple(map(tuple, labs.to_numpy().tolist())) == value_checks.LABS
    assert sorted(path.name for path in (HANDOUT / "data").iterdir()) == ["labs.csv", "vitals.csv"]
    # Every clock time falls in January, where the fixed UTC-5 offset in _value_checks.py is exact.
    for clock in [*vitals["recorded_at"], *labs["collected_at"], *labs["resulted_at"]]:
        assert value_checks.to_utc(clock) == to_utc(pd.Series([clock]))[0].to_pydatetime(), clock
    assert pd.Timestamp(value_checks.PREDICTION_TIME) == PREDICTION_TIME
    source = "\n".join("".join(cell["source"]) for cell in notebook_cells())
    assert 'pd.Timestamp("2026-01-20 18:00", tz="UTC")' in source
    assert "`2026-01-20 18:00:00+00:00`" in (HANDOUT / "README.md").read_text(encoding="utf-8")


def check_timestamp_spellings() -> None:
    """Every unambiguous spelling of the prediction time reads as that instant; other cells read as none."""
    spellings = [
        "2026-01-20 18:00:00+00:00", "2026-01-20T18:00:00Z", "2026-01-20 18:00 UTC", "2026-01-20 18:00:00+00:00 UTC",
        "2026-01-20 18:00:00 +0000", "2026-01-20 13:00:00-05:00", "2026-01-20 13:00 EST", "2026-01-20 18:00:00",
        "2026/01/20 18:00", "01/20/2026 06:00 PM", "20/01/2026 18:00", "Jan 20, 2026 6:00 PM UTC",
        "20 January 2026 13:00 -05:00",
    ]
    for spelling in spellings:
        assert value_checks.instant(spelling) == PREDICTION_TIME, spelling
    for other in ["", "84.0", "P01", "True", "later_holdout", "nan", "est"]:
        assert value_checks.instant(other) is None, other


def check_readme() -> None:
    """Each README checkpoint names its artifact's header, and the contract table matches the checks."""
    readme = (HANDOUT / "README.md").read_text(encoding="utf-8")
    checkpoints = re.findall(r"> \*\*Checkpoint: `([^`]+)`\*\*\n> The header line `([^`]+)`", readme)
    expected = {path: ",".join(artifact.columns) for path, artifact in ARTIFACTS.items()}
    assert dict(checkpoints) == expected, checkpoints
    # The notebook's checkpoints name the same header lines.
    notebook_checkpoints = re.findall(
        r"> \*\*Checkpoint: `([^`]+)`\*\*\n> .* under the header `([^`]+)`",
        "\n".join("".join(cell["source"]) for cell in notebook_cells() if cell["cell_type"] == "markdown"),
    )
    assert dict(notebook_checkpoints) == expected, notebook_checkpoints
    contract = re.findall(r"^\| `(output/[^`]+)` \| .* \| ([^|]+) \| (\d+) \|$", readme, re.M)
    assert [(name.strip(), int(points)) for _, name, points in contract] == list(zip(NAMES, POINTS)), contract
    for path, name, _ in contract:
        assert name.strip().startswith(PREFIX[path]), (path, name)
    first = value_checks.CHECKS[0].name
    assert f"[FIX ]  0/{POINTS[0]}  {first}\n         {value_checks._missing(HANDOUT, value_checks.PREPARED)}" in readme
    last = value_checks.CHECKS[-1].name
    assert f"[PASS]  {POINTS[-1]}/{POINTS[-1]}  {last}\n\nScore: 100/100" in readme
    # The README and the notebook number their tasks the same way.
    notebook_headings = [
        line for cell in notebook_cells() if cell["cell_type"] == "markdown"
        for line in "".join(cell["source"]).splitlines() if re.match(r"#{2,3} (Task \d|\d\.\d)", line)
    ]
    readme_headings = [line for line in readme.splitlines() if re.match(r"#{2,3} (Task \d|\d\.\d)", line)]
    assert notebook_headings == readme_headings, (notebook_headings, readme_headings)
    assert chr(0x2014) not in readme and " - " not in readme.replace("\n- ", "\n")


def check_handout_files() -> None:
    """The handout ships every CHECKS_FILES file byte for byte, and no other Python or QA files."""
    workflow = (HANDOUT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    listed = re.search(r"^  CHECKS_FILES: \|\n((?:    \S.*\n)+)", workflow, re.M).group(1).split()
    assert 'CHECKS_PATH: "09/assignment_checks"' in workflow
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
    for cell in notebook_cells():
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None, cell.get("id")


def alternative() -> dict[str, str | bytes]:
    """The same answers written the way another careful student might: {artifact path: file contents}."""
    frames = solve()

    prepared = frames["output/prepared_vitals.csv"].copy()
    prepared["recorded_at"] = prepared["recorded_at"].dt.tz_convert("America/New_York")
    prepared["patient_id"] = prepared["patient_id"].str.lower()
    prepared = prepared[list(reversed(prepared.columns))]

    grid = frames["output/hourly_grid.csv"].drop(columns="source_row")
    grid["recorded_at"] = grid["recorded_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    grid["grid_created"] = grid["grid_created"].map({True: "true", False: "false"})
    grid["value_missing"] = grid["value_missing"].astype(int)

    summary = frames["output/two_hour_summary.csv"].iloc[::-1].copy()
    summary["recorded_at"] = summary["recorded_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
    summary["mean_hr"] = summary["mean_hr"].round(1)

    features = frames["output/past_features.csv"].copy()
    features["source_row"] = 1
    means = ["mean_prev_2", "mean_prev_2h"]
    features[means] = features[means].round(1)
    features = features[["mean_prev_2h", "patient_id", "hr_change", "recorded_at", "source_row", "heart_rate",
                         "mean_prev_2", "previous_hr"]]
    features.columns = [column.title() for column in features.columns]

    labs = frames["output/lab_availability.csv"].copy()
    for column in ("collected_at", "resulted_at"):
        labs[column] = labs[column].dt.tz_localize(None)
    labs["test"] = labs["test"].str.upper()
    labs["available"] = labs["available"].astype(int)

    blocks = frames["output/chronological_blocks.csv"].drop(columns="source_row")
    blocks["block"] = blocks["block"].map({"earlier": "Earlier", "later_holdout": "Later Holdout"})

    return {
        # New York offsets, lower-case IDs, columns reversed, CRLF, and UTF-16, as Windows PowerShell 5.1 writes.
        "output/prepared_vitals.csv": codecs.BOM_UTF16_LE
        + prepared.to_csv(index=False, lineterminator="\r\n").encode("utf-16-le"),
        # Z timestamps, lower-case and 1/0 flags, no source_row, a leading row-number column, NaN for empty cells.
        "output/hourly_grid.csv": grid.to_csv(na_rep="NaN"),
        # Rows reversed, "UTC" timestamps, one decimal, a space after every comma, trailing spaces, no final newline.
        "output/two_hour_summary.csv": "\n".join(
            line.replace(",", ", ") + " " for line in summary.to_csv(index=False).splitlines()),
        # Title-case headers, columns shuffled, source_row kept, one decimal, a BOM, and every cell quoted.
        "output/past_features.csv": "﻿" + features.to_csv(index=False, quoting=1),
        # Naive UTC timestamps, upper-case test names, and 1/0 for True/False.
        "output/lab_availability.csv": labs.to_csv(index=False, lineterminator="\r\n"),
        # "Later Holdout" with a space and capitals, and no source_row.
        "output/chronological_blocks.csv": blocks.to_csv(index=False),
    }


def edit(text: str, row_key: str, column: str, value: str) -> str:
    """The CSV text with one cell replaced; `row_key` names the first cell, or `first|second` cells."""
    lines = text.splitlines()
    header = lines[0].split(",")
    position = header.index(column)
    wanted = row_key.split("|")
    for number, line in enumerate(lines[1:], start=1):
        cells = line.split(",")
        if cells[: len(wanted)] == wanted:
            cells[position] = value
            lines[number] = ",".join(cells)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"no row {row_key}")


def drop_last_row(text: str) -> str:
    return "\n".join(text.splitlines()[:-1]) + "\n"


def run() -> None:
    check_supplied_data()
    check_timestamp_spellings()
    check_readme()
    check_handout_files()
    frames = solve()
    correct = as_files(frames)

    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a09-selftest-") as temporary:
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
        assert detail(result, NAMES[0]).startswith("output/prepared_vitals.csv is missing"), detail(result, NAMES[0])

        root = submission("correct", correct)
        result = graded(root)
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}

        # The checks read only output/: poisoned data and code in the submission change nothing.
        (root / "data").mkdir()
        (root / "data" / "vitals.csv").write_text("poison\n", encoding="utf-8")
        for name in ("grading.py", "_value_checks.py", "check_assignment.py"):
            (root / name).write_text("raise RuntimeError('submission code must not run')\n", encoding="utf-8")
        assert graded(root)["score"] == 100

        result = graded(submission("alternative", alternative()))
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}

        # Saved with the row index everywhere, as to_csv() does without index=False.
        root = workspace / "with-index"
        (root / "output").mkdir(parents=True)
        for path, frame in frames.items():
            frame.to_csv(root / path)
        assert graded(root)["score"] == 100

        # Cells separated by semicolons, with a European spreadsheet's decimal commas, or by tabs: full marks.
        def separated(sep: str, decimal: str = ".") -> dict[str, str]:
            return {
                path: frame.to_csv(index=False, sep=sep, decimal=decimal) for path, frame in frames.items()
            }

        semicolons = separated(";", ",")
        assert any(re.search(r";-?\d+,\d", text) for text in semicolons.values() if isinstance(text, str))
        for label, files in (("semicolons", semicolons), ("tabs", separated("\t"))):
            result = graded(submission(label, files))
            assert result["score"] == 100, (label, {name: detail(result, name) for name in failing(result)})
        # One wrong value in a semicolon-separated file still costs only its own check.
        wrong = frames["output/two_hour_summary.csv"].copy()
        wrong.loc[wrong["mean_hr"].first_valid_index(), "mean_hr"] += 10.5
        files = separated(";", ",")
        files["output/two_hour_summary.csv"] = wrong.to_csv(index=False, sep=";", decimal=",")
        result = graded(submission("semicolons-one-wrong-value", files))
        assert failing(result) == {"two-hour summary: mean_hr values"}, {name: detail(result, name) for name in failing(result)}
        message = detail(result, "two-hour summary: mean_hr values")
        assert ".5, expected" in message, message

        # The two-hour summary saved with its (patient_id, recorded_at) index instead of reset_index(): full marks.
        vitals = prepare()
        files = dict(correct)
        files["output/two_hour_summary.csv"] = summary_from(vitals).to_csv()
        assert graded(submission("index-keys", files))["score"] == 100

        # The same summary saved with index=False loses its key columns: only the columns check fails.
        files["output/two_hour_summary.csv"] = summary_from(vitals).to_csv(index=False)
        result = graded(submission("lost-keys", files))
        assert failing(result) == {"two-hour summary: columns"}, {name: detail(result, name) for name in failing(result)}
        assert "is missing patient_id and recorded_at" in detail(result, "two-hour summary: columns")

        # Task 1.1 left on the New York clock, and the grid and features built from it: only the UTC check fails.
        naive = prepare(convert=False)
        files = dict(correct)
        files["output/prepared_vitals.csv"] = naive.to_csv(index=False)
        files["output/hourly_grid.csv"] = grid_from(naive).to_csv(index=False)
        files["output/past_features.csv"] = features_from(naive).to_csv(index=False)
        result = graded(submission("new-york-clock", files))
        assert failing(result) == {"prepared vitals: recorded_at in UTC"}, {
            name: detail(result, name) for name in failing(result)}
        message = detail(result, "prepared vitals: recorded_at in UTC")
        assert "5 hours behind" in message and "reads 2026-01-20 07:00:00, expected 2026-01-20 12:00:00+00:00" in message
        assert 'tz_localize("America/New_York")' in message, message

        # One mistake costs exactly its own check.
        prepared, grid = correct["output/prepared_vitals.csv"], correct["output/hourly_grid.csv"]
        summary, features = correct["output/two_hour_summary.csv"], correct["output/past_features.csv"]
        labs, blocks = correct["output/lab_availability.csv"], correct["output/chronological_blocks.csv"]
        as_utc_clock = prepare(convert=False)
        as_utc_clock["recorded_at"] = as_utc_clock["recorded_at"].dt.tz_localize("UTC")
        stacked_shift = frames["output/past_features.csv"].copy()
        stacked_shift["previous_hr"] = stacked_shift["heart_rate"].shift(1)
        mistakes = {
            "prepared vitals: columns": ("output/prepared_vitals.csv", prepared.replace("source_row", "source", 1)),
            "prepared vitals: one row per reading": ("output/prepared_vitals.csv", drop_last_row(prepared)),
            "prepared vitals: recorded_at in UTC": ("output/prepared_vitals.csv", as_utc_clock.to_csv(index=False)),
            "prepared vitals: heart_rate values": (
                "output/prepared_vitals.csv", edit(prepared, "P01|2026-01-20 15:00:00+00:00", "heart_rate", "0.0")),
            "prepared vitals: source_row values": (
                "output/prepared_vitals.csv", edit(prepared, "P02|2026-01-20 13:00:00+00:00", "source_row", "0")),
            "hourly grid: columns": ("output/hourly_grid.csv", grid.replace("grid_created", "created", 1)),
            "hourly grid: one row per patient-hour": ("output/hourly_grid.csv", drop_last_row(grid)),
            "hourly grid: heart_rate values": (
                "output/hourly_grid.csv", edit(grid, "P01|2026-01-20 14:00:00+00:00", "heart_rate", "0.0")),
            "hourly grid: grid_created flags": (
                "output/hourly_grid.csv", edit(grid, "P01|2026-01-20 15:00:00+00:00", "grid_created", "True")),
            "hourly grid: value_missing flags": (
                "output/hourly_grid.csv", edit(grid, "P01|2026-01-20 14:00:00+00:00", "value_missing", "True")),
            "two-hour summary: columns": ("output/two_hour_summary.csv", summary.replace("n_rows", "count", 1)),
            "two-hour summary: one row per patient and bin": ("output/two_hour_summary.csv", drop_last_row(summary)),
            "two-hour summary: mean_hr values": (
                "output/two_hour_summary.csv", edit(summary, "P01|2026-01-20 18:00:00+00:00", "mean_hr", "104.0")),
            "two-hour summary: n_rows values": (
                "output/two_hour_summary.csv", edit(summary, "P01|2026-01-20 14:00:00+00:00", "n_rows", "0")),
            "past features: columns": (
                "output/past_features.csv", "\n".join(line + ",x" for line in features.splitlines()) + "\n"),
            "past features: one row per reading": ("output/past_features.csv", drop_last_row(features)),
            "past features: previous_hr values": ("output/past_features.csv", stacked_shift.to_csv(index=False)),
            "past features: hr_change values": (
                "output/past_features.csv", edit(features, "P02|2026-01-20 16:00:00+00:00", "hr_change", "4.0")),
            "past features: mean_prev_2 values": (
                "output/past_features.csv", edit(features, "P02|2026-01-20 20:00:00+00:00", "mean_prev_2", "73.0")),
            "past features: mean_prev_2h values": (
                "output/past_features.csv", edit(features, "P01|2026-01-20 12:00:00+00:00", "mean_prev_2h", "84.0")),
            "lab availability: columns": ("output/lab_availability.csv", labs.replace("available", "known", 1)),
            "lab availability: one row per lab": ("output/lab_availability.csv", drop_last_row(labs)),
            "lab availability: collected_at and resulted_at in UTC": (
                "output/lab_availability.csv", edit(labs, "P01|creatinine", "collected_at", "2026-01-20 12:20:00")),
            "lab availability: available flags": (
                "output/lab_availability.csv", edit(labs, "P02|troponin", "available", "False")),
            "chronological blocks: columns": ("output/chronological_blocks.csv", blocks.replace("block", "split", 1)),
            "chronological blocks: one row per reading": ("output/chronological_blocks.csv", drop_last_row(blocks)),
            "chronological blocks: block labels": (
                "output/chronological_blocks.csv", edit(blocks, "P02|2026-01-20 18:00:00+00:00", "block", "earlier")),
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

        # Feedback names what was expected and what was found, and the task to fix.
        files = dict(correct)
        files["output/past_features.csv"] = mistakes["past features: previous_hr values"][1]
        message = detail(graded(submission("feedback-shift", files)), "past features: previous_hr values")
        assert "P02 at 13:00 UTC has previous_hr 110.0, expected blank" in message and "Task 3.1" in message, message
        files = dict(correct)
        files["output/two_hour_summary.csv"] = mistakes["two-hour summary: n_rows values"][1]
        message = detail(graded(submission("feedback-count", files)), "two-hour summary: n_rows values")
        assert "P01 at 14:00 UTC has n_rows 0, expected 1" in message and "source_row" in message, message
        files = dict(correct)
        files["output/lab_availability.csv"] = mistakes["lab availability: available flags"][1]
        message = detail(graded(submission("feedback-labs", files)), "lab availability: available flags")
        assert "P02 troponin has available False, expected True" in message and "<=" in message, message
        files = dict(correct)
        files["output/hourly_grid.csv"] = mistakes["hourly grid: one row per patient-hour"][1]
        message = detail(graded(submission("feedback-rows", files)), "hourly grid: one row per patient-hour")
        assert "should hold 16 rows" in message and "is missing P02 at 20:00 UTC" in message, message

        # pytest, which GitHub runs, reports one test per check.
        for name, expected_failures in (("correct", 0), ("scaffold", len(NAMES))):
            target = workspace / f"pytest-{name}"
            shutil.copytree(HANDOUT, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv"))
            if name == "correct":
                for path, text in correct.items():
                    (target / path).write_text(text, encoding="utf-8")
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
        "Assignment 09 checks: the data copies, README checkpoints, contract, and numbering, and the handout's "
        "byte-identical checks agree; empty and scaffold earn 0; correct, alternative, with-index, semicolon- and "
        "tab-separated, and index-key "
        "submissions earn 100; lost key columns cost only the columns check; New York clock times cost only the "
        f"UTC check; each of the {len(NAMES)} single mistakes and each missing file costs only its own checks; "
        "feedback and pytest pass."
    )


if __name__ == "__main__":
    run()
