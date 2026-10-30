"""Development regression checks for the Assignment 09 checks.

Builds submissions in ignored `scratch/`, answers the assignment with pandas
the way the README's steps do, and confirms what each kind of submission
scores: a correct one earns 100, and so do another valid route (features from
time masks instead of shift and rolling) and one written differently (column
and row order, CRLF, a BOM, UTF-16, quoting, padding, letter case, number
format, rounding, plain dates, row-number columns, `NaN` for an empty cell,
semicolons or tabs). An empty directory and the untouched handout earn 0, one
wrong value or one misnamed column costs exactly its own check, an omitted
value column and renamed columns with the right values cost only the columns
check, and a missing file costs only its own artifact's checks. The plausible
mistakes (UTC days, count() for size, hourly means, a leaking window, lab
clocks localized as UTC, episodes with no gap dropped or left blank, lactates
picked by collection time) each cost only their own check, with messages
naming the cause. It also confirms that the data and the frozen values in
`_value_checks.py` match `scripts/build_assignment09_data.py`, that the
README's checkpoints, contract, and numbering agree with the checks and the
notebook, that the handout ships every file in `CHECKS_FILES` byte for byte and
no other Python file, and that the handout notebook ships with its outputs
cleared.

    uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 \\
        --with 'pytest>=8,<9' python 09/assignment_checks/_grader_selftest/run.py
"""

from __future__ import annotations

from pathlib import Path
import codecs
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

# Self-tests grade with the working copy, never the copy downloaded from GitHub.
os.environ["DS217_LOCAL_CHECKS"] = "1"


CHECKS = Path(__file__).resolve().parents[1]
HANDOUT = CHECKS.parent / "assignment"
REPO = CHECKS.parents[1]
SCRATCH = REPO / "scratch"
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(REPO / "scripts"))
sys.dont_write_bytecode = True
import _value_checks as value_checks  # noqa: E402
import build_assignment09_data as generator  # noqa: E402
from grading import POINTS, grade_submission  # noqa: E402

NAMES = [check.name for check in value_checks.CHECKS]
POINTS_BY_NAME = dict(zip(NAMES, POINTS, strict=True))
ARTIFACTS = {artifact.path: artifact for artifact in value_checks.ARTIFACTS}
DAILY, COVERAGE, RULES = "output/daily_feed.csv", "output/episode_coverage.csv", "output/alert_rules.csv"
TURNAROUND, FEATURES, HOLDOUT = "output/lab_turnaround.csv", "output/score_features.csv", "output/holdout_summary.csv"
PREFIX = {DAILY: "daily feed:", COVERAGE: "episode coverage:", RULES: "alert rules:", TURNAROUND: "lab turnaround:",
          FEATURES: "score features:", HOLDOUT: "holdout summary:"}
TABLES = {name: pd.read_parquet(HANDOUT / "data" / f"{name}.parquet") for name in ("episodes", "vitals", "labs")}
FEATURE_COLUMNS = ["episode_id", "prev_hour_hr", "mean_prev_6h", "last_lactate"]


def solve(hourly_stat: str = "median", day_zone: str = "America/New_York", size: str = "size",
          lab_zone: str = "America/New_York", leak: bool = False, lactate_time: str = "resulted_at",
          masks: bool = False, hour_count: str = "source_row") -> dict[str, pd.DataFrame]:
    """Answer the assignment with the README's pandas calls; returns {artifact path: frame to save}.

    The options reproduce one slip each; `masks=True` takes another valid route to the Task 3.2 features, and
    `hour_count="heart_rate"` counts Task 2.1's hours from heart rates, which the checks also accept.
    """
    episodes = TABLES["episodes"]
    vitals = TABLES["vitals"].copy()
    vitals["recorded_at"] = pd.to_datetime(vitals["recorded_at"], utc=True)
    vitals = vitals.sort_values(["episode_id", "recorded_at"]).reset_index(drop=True)

    local = vitals.copy()
    local["recorded_at"] = local["recorded_at"].dt.tz_convert(day_zone)
    daily = local.set_index("recorded_at").resample("D").agg(readings=("heart_rate", size),
                                                             episodes=("episode_id", "nunique"))
    daily_feed = daily.reset_index().rename(columns={"recorded_at": "date"})

    vitals["source_row"] = 1
    counts = (vitals.set_index("recorded_at").groupby("episode_id")[hour_count].resample("h").count()
              .rename("readings").reset_index())
    counts["source_observed"] = counts["readings"] > 0
    counts["gap_hour"] = ~counts["source_observed"]
    counts["run_id"] = counts.groupby("episode_id")["source_observed"].cumsum()
    runs = counts[counts["gap_hour"]].groupby(["episode_id", "run_id"]).size().rename("gap_hours").reset_index()
    run_summary = runs.groupby("episode_id")["gap_hours"].agg(gap_runs="count", longest_gap_h="max").reset_index()
    coverage = counts.groupby("episode_id", as_index=False).agg(grid_hours=("readings", "size"),
                                                                gap_hours=("gap_hour", "sum"))
    coverage = coverage.merge(run_summary, on="episode_id", how="left", validate="one_to_one")
    coverage = coverage.fillna({"gap_runs": 0, "longest_gap_h": 0})
    assert len(counts) == 108293
    assert (len(runs), (coverage["gap_runs"] == 0).sum()) == ((3102, 201) if hour_count == "source_row" else (3120, 199))

    grouped = vitals.set_index("recorded_at").groupby("episode_id")["heart_rate"].resample("h")
    hourly = getattr(grouped, hourly_stat)().reset_index()
    hourly["rolling_3h"] = hourly.groupby("episode_id")["heart_rate"].transform(lambda s: s.rolling(3).mean())
    peaks = pd.DataFrame({
        "any_reading": vitals.groupby("episode_id")["heart_rate"].max(),
        "hourly_median": hourly.groupby("episode_id")["heart_rate"].max(),
        "rolling_3h": hourly.groupby("episode_id")["rolling_3h"].max(),
    })
    flags = peaks >= value_checks.ALERT_BPM
    escalated = flags.index.isin(episodes.loc[episodes["outcome"] == "escalated", "episode_id"])
    alert_rules = pd.DataFrame({"rule": flags.columns, "episodes_flagged": flags.sum().to_numpy(),
                                "escalations_flagged": flags[escalated].sum().to_numpy()})

    labs = TABLES["labs"].copy()
    for column in ["collected_at", "resulted_at"]:
        labs[column] = (pd.to_datetime(labs[column], format="%Y-%m-%d %H:%M")
                        .dt.tz_localize(lab_zone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC"))
    labs["turnaround_h"] = (labs["resulted_at"] - labs["collected_at"]).dt.total_seconds() / 3600
    lab_turnaround = labs.groupby("test", as_index=False).agg(
        orders=("episode_id", "size"), timed_results=("resulted_at", "count"),
        median_turnaround_h=("turnaround_h", "median"))

    scored = episodes[["episode_id", "admitted_at", "outcome"]].copy()
    scored["prediction_time"] = scored["admitted_at"].dt.floor("h") + pd.Timedelta(hours=24)
    if masks:
        # Another valid route: the hourly rows before each prediction time, picked by time instead of by shift.
        timed = hourly.merge(scored, on="episode_id", validate="many_to_one")
        before = timed[timed["recorded_at"] < timed["prediction_time"]]
        last_hour = before[before["recorded_at"] == before["prediction_time"] - pd.Timedelta(hours=1)]
        six = before[before["recorded_at"] >= before["prediction_time"] - pd.Timedelta(hours=6)]
        on_grid = timed.loc[timed["recorded_at"] == timed["prediction_time"], ["episode_id", "prediction_time", "outcome"]]
        score_features = (on_grid
                          .merge(last_hour[["episode_id", "heart_rate"]].rename(columns={"heart_rate": "prev_hour_hr"}),
                                 on="episode_id", how="left", validate="one_to_one")
                          .merge(six.groupby("episode_id")["heart_rate"].mean().rename("mean_prev_6h").reset_index(),
                                 on="episode_id", how="left", validate="one_to_one"))
    else:
        features = hourly[["episode_id", "recorded_at", "heart_rate"]].copy()
        by_episode = features.groupby("episode_id")["heart_rate"]
        features["prev_hour_hr"] = by_episode.shift(1)
        start = 0 if leak else 1
        features["mean_prev_6h"] = by_episode.transform(lambda s: s.shift(start).rolling(6, min_periods=1).mean())
        features = features.merge(scored, on="episode_id", validate="many_to_one")
        score_features = features[features["recorded_at"] == features["prediction_time"]]
    lactate = labs[labs["test"] == "Lactate"].merge(scored[["episode_id", "prediction_time"]], on="episode_id",
                                                     validate="many_to_one")
    known = lactate[lactate[lactate_time] <= lactate["prediction_time"]]
    last_lactate = (known.sort_values(lactate_time).groupby("episode_id")["value"].last().rename("last_lactate")
                    .reset_index())
    score_features = score_features.merge(last_lactate, on="episode_id", how="left", validate="one_to_one")
    assert len(score_features) == len(value_checks.FEATURES)

    blocks = score_features.copy()
    blocks["block"] = np.where(blocks["prediction_time"] < pd.Timestamp("2026-03-01", tz="UTC"), "earlier",
                               "later_holdout")
    blocks["escalated"] = blocks["outcome"] == "escalated"
    blocks["flagged"] = blocks["mean_prev_6h"] >= value_checks.FLAG_BPM
    blocks["flagged_escalation"] = blocks["flagged"] & blocks["escalated"]
    holdout = blocks.groupby("block", as_index=False).agg(
        episodes=("episode_id", "size"), escalations=("escalated", "sum"), flagged=("flagged", "sum"),
        flagged_escalations=("flagged_escalation", "sum"))

    return {DAILY: daily_feed, COVERAGE: coverage, RULES: alert_rules, TURNAROUND: lab_turnaround,
            FEATURES: score_features[FEATURE_COLUMNS], HOLDOUT: holdout}


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


def graded(root: Path, both_clis: bool = True) -> dict:
    """Grade in-process and through both CLIs, and insist all three agree."""
    result = grade_submission(root)
    assert result["schema"] == "datasci217/grading-result/v1"
    assert result["max-score"] == 100 == sum(POINTS)
    if both_clis:
        assert cli_report(CHECKS, root) == result, root
        assert cli_report(HANDOUT, root) == result, root
    return result


def cost(*names: str) -> int:
    return sum(POINTS_BY_NAME[name] for name in names)


def artifact_checks(path: str) -> list[str]:
    return [name for name in NAMES if name.startswith(PREFIX[path])]


def check_supplied_data() -> None:
    """The handout's Parquet files are the generator's output, and the frozen values match them."""
    assert sorted(path.name for path in (HANDOUT / "data").iterdir()) == ["episodes.parquet", "labs.parquet",
                                                                          "vitals.parquet"]
    for name, table in generator.build().items():
        assert TABLES[name].equals(table), f"data/{name}.parquet differs from build_assignment09_data.build()"
    for name, value in generator.expected(TABLES).items():
        assert getattr(value_checks, name) == value, f"{name} in _value_checks.py differs from the data; rerun the build"
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/09/assignment/data" in source
    for line in (f"ALERT_BPM = {generator.ALERT_BPM}", f"FLAG_BPM = {generator.FLAG_BPM}",
                 f"SCORE_AFTER = pd.Timedelta(hours={generator.SCORE_AFTER_H})",
                 f'HOLDOUT_FROM = pd.Timestamp("{generator.HOLDOUT_FROM.strftime("%Y-%m-%d")}", tz="UTC")'):
        assert line in source, line
    gitignore = (HANDOUT / ".gitignore").read_text(encoding="utf-8")
    assert "data" not in gitignore and "parquet" not in gitignore, "the handout must commit data/*.parquet"


def check_readme() -> None:
    """Each README checkpoint names its artifact's header, and the contract table matches the checks."""
    readme = (HANDOUT / "README.md").read_text(encoding="utf-8")
    checks_md = (HANDOUT / "CHECKS.md").read_text(encoding="utf-8")
    assert "(CHECKS.md)" in readme, "README.md no longer links CHECKS.md"
    checkpoints = re.findall(r"> \*\*Checkpoint: `([^`]+)`\*\*\n> The header line `([^`]+)`", readme)
    assert {path: header for path, header in checkpoints} == {
        path: ",".join(artifact.columns) for path, artifact in ARTIFACTS.items()
    }, checkpoints
    contract = re.findall(r"^\| `(output/[^`]+)` \| .* \| ([^|]+) \| (\d+) \|$", checks_md, re.M)
    assert [(name.strip(), int(points)) for _, name, points in contract] == list(zip(NAMES, POINTS)), contract
    for path, name, _ in contract:
        assert name.strip().startswith(PREFIX[path]), (path, name)
    first = value_checks.CHECKS[0].name
    assert f"[FIX ]  0/{POINTS[0]}  {first}\n         {value_checks._missing(HANDOUT, value_checks.DAILY_TABLE)}" in checks_md
    last = value_checks.CHECKS[-1].name
    assert f"[PASS]  {POINTS[-1]}/{POINTS[-1]}  {last}\n\nScore: 100/100" in checks_md
    for count in (f"then {len(value_checks.DAILY_FEED)} rows", f"then {len(value_checks.COVERAGE):,} rows",
                  f"then {len(value_checks.ALERT_RULES)} rows", f"then {len(value_checks.TURNAROUND)} rows",
                  f"then {len(value_checks.FEATURES):,} rows", "then 2 rows"):
        assert count in readme, count
    assert f"{value_checks.VITALS_ROWS:,} patch readings" in readme
    # No full answer expressions: the README names methods and arguments but never prints a finished call chain.
    for leaked in ("groupby(\"episode_id\")[\"heart_rate\"].resample", "s.shift(1).rolling(6", ".dt.tz_localize(\"America"):
        assert leaked not in readme, leaked
    # The README and the notebook number their tasks the same way.
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    notebook_headings = [
        line for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        for line in "".join(cell["source"]).splitlines() if re.match(r"#{2,3} (Task \d|\d\.\d)", line)
    ]
    readme_headings = [line for line in readme.splitlines() if re.match(r"#{2,3} (Task \d|\d\.\d)", line)]
    assert notebook_headings == readme_headings, (notebook_headings, readme_headings)
    for text in (readme, checks_md):
        assert "\u2014" not in text and "\u2013" not in text, "no em or en dashes"


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
    # A student's gitignored .venv/ and the downloaded .checks/ cache are not part of the handout.
    handout_python = sorted(
        path.relative_to(HANDOUT).as_posix() for path in HANDOUT.rglob("*.py")
        if "__pycache__" not in path.parts and ".venv" not in path.parts and ".checks" not in path.parts
    )
    assert handout_python == sorted(name for name in listed if name.endswith(".py")), handout_python
    assert not (HANDOUT / "_grader_selftest").exists()
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None, cell.get("id")


def alternative() -> dict[str, str | bytes]:
    """The same answers by another route and written the way another careful student might."""
    frames = solve(masks=True)

    daily = frames[DAILY].copy()
    daily["date"] = daily["date"].dt.strftime("%m/%d/%Y")
    daily = daily[["episodes", "readings", "date"]]

    coverage = frames[COVERAGE].astype({"gap_runs": int, "longest_gap_h": int}).iloc[::-1]
    coverage["episode_id"] = coverage["episode_id"].str.lower()

    rules = frames[RULES].copy()
    rules["rule"] = rules["rule"].str.replace("_", " ").str.title()

    turnaround = frames[TURNAROUND].round(2)
    turnaround.columns = [column.upper() for column in turnaround.columns]

    features = frames[FEATURES].round(1)

    holdout = frames[HOLDOUT].copy()
    holdout["block"] = holdout["block"].str.replace("_", " ").str.capitalize()

    return {
        # Plain US dates, columns reversed, CRLF, and UTF-16, as Windows PowerShell 5.1 writes.
        DAILY: codecs.BOM_UTF16_LE + daily.to_csv(index=False, lineterminator="\r\n").encode("utf-16-le"),
        # Whole numbers, lower-case IDs, rows reversed, a leading row-number column.
        COVERAGE: coverage.to_csv(),
        # Title-case rule names with spaces, every cell quoted, a BOM.
        RULES: "﻿" + rules.to_csv(index=False, quoting=1),
        # Upper-case headers, two decimals, a trailing space on each line, no final newline.
        TURNAROUND: "\n".join(line + " " for line in turnaround.to_csv(index=False).splitlines()),
        # Features from time masks, one decimal, NaN for the blanks, a space after every comma.
        FEATURES: features.to_csv(index=False, na_rep="NaN").replace(",", ", "),
        HOLDOUT: holdout.to_csv(index=False, sep="\t"),
    }


def edit(text: str, row_key: str, column: str, value: str) -> str:
    """The CSV text with one cell replaced; `row_key` names the first cell."""
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


def drop_last_row(text: str) -> str:
    return "\n".join(text.splitlines()[:-1]) + "\n"


def run() -> None:
    check_supplied_data()
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

        def only(name: str, files: dict[str, str | bytes], expected: set[str], *parts: str, check: str = "") -> dict:
            """Grade `files` and insist exactly `expected` fail; each of `parts` must appear in `check`'s message."""
            result = graded(submission(name, files), both_clis=False)
            assert failing(result) == expected, (name, {other: detail(result, other) for other in failing(result)})
            assert result["score"] == 100 - cost(*expected), (name, result["score"])
            message = detail(result, check or next(iter(expected))) if parts else ""
            for part in parts:
                assert part in message, (name, part, message)
            return result

        def changed(path: str, text: str | bytes) -> dict[str, str | bytes]:
            files = dict(correct)
            files[path] = text
            return files

        empty = workspace / "empty"
        empty.mkdir()
        assert graded(empty)["score"] == 0
        scaffold = workspace / "scaffold"
        shutil.copytree(HANDOUT, scaffold, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv",
                                                                         ".checks", "data"))
        result = graded(scaffold)
        assert result["score"] == 0 and failing(result) == set(NAMES)
        assert detail(result, NAMES[0]).startswith("output/daily_feed.csv is missing"), detail(result, NAMES[0])

        root = submission("correct", correct)
        result = graded(root)
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}
        size = sum((root / path).stat().st_size for path in correct)
        assert size < 100_000, f"a complete output/ is {size} bytes"

        # The checks read only output/: poisoned data and code in the submission change nothing.
        (root / "data").mkdir()
        (root / "data" / "vitals.parquet").write_text("poison\n", encoding="utf-8")
        for name in ("grading.py", "_value_checks.py", "check_assignment.py"):
            (root / name).write_text("raise RuntimeError('submission code must not run')\n", encoding="utf-8")
        assert graded(root)["score"] == 100

        result = graded(submission("alternative", alternative()))
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}

        # Task 2.1's hours counted from heart_rate skip readings with no heart rate: accepted too.
        result = graded(submission("heart-rate-hours", changed(COVERAGE, as_files(solve(hour_count="heart_rate"))[COVERAGE])))
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}

        # Saved with the row index everywhere, as to_csv() does without index=False.
        root = workspace / "with-index"
        (root / "output").mkdir(parents=True)
        for path, frame in frames.items():
            frame.to_csv(root / path)
        assert graded(root)["score"] == 100

        # Cells separated by semicolons, with a European spreadsheet's decimal commas, or by tabs: full marks.
        def separated(sep: str, decimal: str = ".") -> dict[str, str]:
            return {path: frame.to_csv(index=False, sep=sep, decimal=decimal) for path, frame in frames.items()}

        semicolons = separated(";", ",")
        assert any(re.search(r";-?\d+,\d", text) for text in semicolons.values())
        for label, files in (("semicolons", semicolons), ("tabs", separated("\t"))):
            result = graded(submission(label, files))
            assert result["score"] == 100, (label, {name: detail(result, name) for name in failing(result)})

        # Summaries grouped without as_index=False keep their keys when saved with the index: full marks.
        index_files = dict(correct)
        index_files[TURNAROUND] = frames[TURNAROUND].set_index("test").to_csv()
        index_files[HOLDOUT] = frames[HOLDOUT].set_index("block").to_csv()
        assert graded(submission("index-keys", index_files))["score"] == 100
        # Saved with index=False they lose their key columns: only the columns checks fail.
        lost = dict(correct)
        lost[TURNAROUND] = frames[TURNAROUND].drop(columns="test").to_csv(index=False)
        lost[HOLDOUT] = frames[HOLDOUT].drop(columns="block").to_csv(index=False)
        only("lost-keys", lost, {"lab turnaround: columns", "holdout summary: columns"}, "is missing test",
             check="lab turnaround: columns")

        # An omitted value column costs only its columns check; two renamed with the right values the same.
        only("dropped-value-column", changed(COVERAGE, frames[COVERAGE].drop(columns="gap_runs").to_csv(index=False)),
             {"episode coverage: columns"}, "is missing gap_runs")
        renamed = frames[RULES].rename(columns={"episodes_flagged": "flagged", "escalations_flagged": "hits"})
        only("renamed-two-right", changed(RULES, renamed.to_csv(index=False)), {"alert rules: columns"},
             "is missing episodes_flagged and escalations_flagged and also has flagged and hits")
        wrong_renamed = renamed.copy()
        wrong_renamed[["flagged", "hits"]] = [[7, 6], [7, 6], [7, 6]]
        only("renamed-two-wrong", changed(RULES, wrong_renamed.to_csv(index=False)),
             {"alert rules: columns", "alert rules: episodes_flagged values", "alert rules: escalations_flagged values"},
             "no saved column holds episodes_flagged", "expected episodes_flagged 1211 for any_reading",
             check="alert rules: episodes_flagged values")
        only("key-only", changed(RULES, frames[RULES][["rule"]].to_csv(index=False)), {"alert rules: columns"},
             "is missing episodes_flagged and escalations_flagged")

        # Plausible mistakes, each charged once with a message that names the cause.
        only("utc-days", changed(DAILY, solve(day_zone="UTC")[DAILY].to_csv(index=False)),
             {"daily feed: readings values"}, "UTC offset", 'America/New_York before resample("D")')
        only("count-not-size", changed(DAILY, solve(size="count")[DAILY].to_csv(index=False)),
             {"daily feed: readings values"}, "count() skips", '"size"')
        means = as_files(solve(hourly_stat="mean"))
        files = dict(correct)
        files.update({RULES: means[RULES], FEATURES: means[FEATURES], HOLDOUT: means[HOLDOUT]})
        only("hourly-means", files, {"alert rules: episodes_flagged values"}, "hourly means")
        leaked = as_files(solve(leak=True))
        files = dict(correct)
        files.update({FEATURES: leaked[FEATURES], HOLDOUT: leaked[HOLDOUT]})
        assert leaked[HOLDOUT] != correct[HOLDOUT]
        only("leaking-window", files, {"score features: mean_prev_6h values"}, "never the prediction hour itself")
        utc_labs = as_files(solve(lab_zone="UTC"))
        files = dict(correct)
        files.update({TURNAROUND: utc_labs[TURNAROUND], FEATURES: utc_labs[FEATURES]})
        assert utc_labs[FEATURES] != correct[FEATURES]
        only("labs-localized-utc", files, {"lab turnaround: orders and timed_results values"}, 'nonexistent="NaT"')
        only("lactate-by-collection", changed(FEATURES, as_files(solve(lactate_time="collected_at"))[FEATURES]),
             {"score features: last_lactate values"}, "resulted_at (in UTC) is at or before")
        inner = frames[COVERAGE][frames[COVERAGE]["gap_runs"] > 0]
        only("no-gap-dropped", changed(COVERAGE, inner.to_csv(index=False)), {"episode coverage: one row per episode"},
             "including those with no gap")
        blank = frames[COVERAGE].copy()
        blank.loc[blank["gap_runs"] == 0, ["gap_runs", "longest_gap_h"]] = np.nan
        only("no-gap-blank", changed(COVERAGE, blank.to_csv(index=False)),
             {"episode coverage: gap_runs and longest_gap_h values"}, "H00004 has gap_runs blank, expected 0", "not a blank")

        # Repeated headers: an identical duplicate is fine, a conflicting one costs its values check.
        rules = correct[RULES]
        lines = rules.splitlines()
        files = changed(RULES, "\n".join(line + "," + line.split(",")[1] for line in lines) + "\n")
        assert graded(submission("duplicate-header", files), both_clis=False)["score"] == 100
        files[RULES] = files[RULES].replace("any_reading,1211,138,1211", "any_reading,1211,138,99")
        only("conflicting-duplicate", files, {"alert rules: episodes_flagged values"})
        only("nul-byte-file", changed(RULES, b"\xff\xfe\x00"), set(artifact_checks(RULES)))

        # One mistake costs exactly its own check.
        daily, coverage, turnaround = correct[DAILY], correct[COVERAGE], correct[TURNAROUND]
        features, holdout = correct[FEATURES], correct[HOLDOUT]
        feb17 = next(line.split(",")[0] for line in daily.splitlines() if line.startswith("2026-02-17"))
        mistakes = {
            "daily feed: columns": (DAILY, daily.replace("episodes", "patients", 1)),
            "daily feed: one row per New York day": (DAILY, drop_last_row(daily)),
            "daily feed: readings values": (DAILY, edit(daily, feb17, "readings", "7836")),
            "daily feed: episodes values": (DAILY, edit(daily, feb17, "episodes", "97")),
            "episode coverage: columns": (COVERAGE, coverage.replace("grid_hours", "hours", 1)),
            "episode coverage: one row per episode": (COVERAGE, drop_last_row(coverage)),
            "episode coverage: grid_hours and gap_hours values": (COVERAGE, edit(coverage, "H00002", "gap_hours", "9")),
            "episode coverage: gap_runs and longest_gap_h values": (
                COVERAGE, edit(coverage, "H00002", "longest_gap_h", "4.0")),
            "alert rules: columns": (RULES, rules.replace("rule", "alert", 1)),
            "alert rules: one row per rule": (RULES, drop_last_row(rules)),
            "alert rules: episodes_flagged values": (RULES, edit(rules, "hourly_median", "episodes_flagged", "60")),
            "alert rules: escalations_flagged values": (RULES, edit(rules, "hourly_median", "escalations_flagged", "41")),
            "lab turnaround: columns": (TURNAROUND, turnaround.replace("orders", "n", 1)),
            "lab turnaround: one row per test": (TURNAROUND, drop_last_row(turnaround)),
            "lab turnaround: orders and timed_results values": (
                TURNAROUND, edit(turnaround, "Lactate", "timed_results", "1253")),
            "lab turnaround: median_turnaround_h values": (
                TURNAROUND, edit(turnaround, "Lactate", "median_turnaround_h", "3.2")),
            "score features: columns": (FEATURES, features.replace("last_lactate", "lactate", 1)),
            "score features: one row per scored episode": (FEATURES, drop_last_row(features)),
            "score features: prev_hour_hr values": (FEATURES, edit(features, "H00001", "prev_hour_hr", "77.5")),
            "score features: mean_prev_6h values": (FEATURES, edit(features, "H00001", "mean_prev_6h", "79.0")),
            "score features: last_lactate values": (FEATURES, edit(features, "H00007", "last_lactate", "2.5")),
            "holdout summary: columns": (HOLDOUT, holdout.replace("flagged_escalations", "hits", 1)),
            "holdout summary: one row per block": (HOLDOUT, drop_last_row(holdout)),
            "holdout summary: episodes and escalations values": (HOLDOUT, edit(holdout, "earlier", "escalations", "100")),
            "holdout summary: flagged values": (HOLDOUT, edit(holdout, "earlier", "flagged", "30")),
        }
        assert sorted(mistakes) == sorted(NAMES)
        for number, (name, (path, text)) in enumerate(mistakes.items()):
            only(f"mistake-{number:02}", changed(path, text), {name})
        message = detail(graded(submission("feedback", changed(*mistakes["score features: mean_prev_6h values"])),
                                both_clis=False), "score features: mean_prev_6h values")
        assert "H00001 has mean_prev_6h 79.0, expected 78.75" in message and "shift(1)" in message, message

        # A missing file costs only its own artifact's checks, and is never told to take another required file's name.
        for number, path in enumerate(correct):
            files = {other: text for other, text in correct.items() if other != path}
            result = graded(submission(f"missing-{number}", files), both_clis=False)
            assert failing(result) == set(artifact_checks(path)), (path, failing(result))
            assert not any("Found" in detail(result, name) for name in failing(result)), path
        files = dict(correct)
        files["output/alert_rule.csv"] = files.pop(RULES)
        message = detail(graded(submission("typo-name", files), both_clis=False), "alert rules: columns")
        assert message.endswith("Found output/alert_rule.csv; save it as output/alert_rules.csv instead."), message

        # pytest, which GitHub runs, reports one test per check.
        for name, expected_failures in (("correct", 0), ("scaffold", len(NAMES))):
            target = workspace / f"pytest-{name}"
            shutil.copytree(HANDOUT, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv",
                                                                           ".checks", "data"))
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
        "Assignment 09 checks: the data, the frozen values, README checkpoints, contract, and numbering, and the "
        "handout's byte-identical checks agree; empty and scaffold earn 0; correct, alternative-route, with-index, "
        "index-key, semicolon- and tab-separated submissions earn 100; each of the "
        f"{len(NAMES)} single mistakes, each plausible mistake, and each missing file costs only its own checks; "
        "feedback and pytest pass."
    )


if __name__ == "__main__":
    run()
