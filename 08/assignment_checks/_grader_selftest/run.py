"""Development regression checks for the Assignment 08 checks.

Builds submissions in ignored `scratch/`, answers the assignment with pandas
the way the lecture does, and confirms what each kind of submission scores:
a correct one earns 100, and so do one built from plain text keys instead of
the ordered categorical and one written differently (column and row order,
CRLF, a BOM, UTF-16, quoting, padding, letter case, number format, means
rounded to one decimal, row-number columns, `NaN` for an empty cell, and the
zero-visit rows `observed=False` adds). An empty directory and the untouched
handout earn 0, one wrong value or one misnamed column costs exactly its own
check, a summary saved without its key column costs only the columns check,
and a missing file costs only its own artifact's checks, with no advice to
rename another required file. A pivot with its axes swapped costs only its
columns check, margins=True is named as the cause of a Total row and column,
and wait_vs_clinic worked out from a wrong clinic_mean_wait costs only the
clinic_mean_wait check. It also confirms that
the copy of the visit log in `_value_checks.py` matches
`08/assignment/data/clinic_visits.csv`, that the README's checkpoints and
completion contract agree with the checks, that the handout ships every file in
`CHECKS_FILES` byte for byte and no other Python file, and that the handout
notebook ships with its outputs cleared.

    uv run --python 3.13 --with pandas==3.0.5 --with 'pytest>=8,<9' \\
        python 08/assignment_checks/_grader_selftest/run.py
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

import pandas as pd

# Self-tests grade with the working copy, never the copy downloaded from GitHub.
os.environ["DS217_LOCAL_CHECKS"] = "1"


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
    "output/clinic_counts.csv": "clinic counts:",
    "output/clinic_summary.csv": "clinic summary:",
    "output/visits_with_context.csv": "visit context:",
    "output/clinic_visit_type_summary.csv": "clinic and visit type:",
    "output/mean_wait_pivot.csv": "mean wait pivot:",
}
COUNTS = {"visit_count": ("visit_id", "size"), "satisfaction_count": ("satisfaction", "count"),
          "patient_count": ("patient_id", "nunique")}
WAITS = {"total_wait_min": ("wait_min", "sum"), "mean_wait_min": ("wait_min", "mean")}


def read_visits() -> pd.DataFrame:
    return pd.read_csv(HANDOUT / "data" / "clinic_visits.csv")


def solve(categorical: bool = True, observed: bool = True) -> dict[str, pd.DataFrame]:
    """Answer the assignment with the lecture's pandas calls; returns {artifact path: frame to save}."""
    visits = read_visits()
    if categorical:
        visits["clinic"] = pd.Categorical(visits["clinic"], categories=list(value_checks.CLINIC_ORDER), ordered=True)
        per_clinic = visits.groupby("clinic", observed=False).size()
        assert per_clinic.to_dict() == {"Mission": 6, "Sunset": 5, "Bayview": 4, "Excelsior": 0}, per_clinic

    by_clinic = visits.groupby("clinic", as_index=False, observed=observed)
    clinic_counts = by_clinic.agg(**COUNTS)
    clinic_summary = by_clinic.agg(**COUNTS, **WAITS)

    context = visits.copy()
    context["clinic_mean_wait"] = visits.groupby("clinic", observed=True)["wait_min"].transform("mean")
    context["wait_vs_clinic"] = context["wait_min"] - context["clinic_mean_wait"]
    assert "clinic_mean_wait" not in visits.columns

    pairs = visits.groupby(["clinic", "visit_type"], as_index=False, observed=observed).agg(
        visit_count=("visit_id", "size"), mean_wait_min=("wait_min", "mean"))

    pivot = pd.pivot_table(visits, values="wait_min", index="clinic", columns="visit_type", aggfunc="mean",
                           observed=observed, dropna=observed)
    twin = visits.groupby(["clinic", "visit_type"], observed=True)["wait_min"].mean().unstack()
    if observed:
        assert pivot.equals(twin)
        assert int(pivot.isna().sum().sum()) == 1
    assert pd.crosstab(visits["clinic"], visits["visit_type"]).loc["Sunset", "Telehealth"] == 0

    return {
        "output/clinic_counts.csv": clinic_counts,
        "output/clinic_summary.csv": clinic_summary,
        "output/visits_with_context.csv": context,
        "output/clinic_visit_type_summary.csv": pairs,
        "output/mean_wait_pivot.csv": pivot,
    }


def as_files(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """The way the README saves each frame: index=False, except the pivot, which keeps its clinic index."""
    return {
        path: frame.to_csv() if path == "output/mean_wait_pivot.csv" else frame.to_csv(index=False)
        for path, frame in frames.items()
    }



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


def check_supplied_data() -> None:
    """The copy of the visit log in _value_checks.py matches the handout's data file and notebook."""
    visits = read_visits()
    assert tuple(visits.columns) == value_checks.VISIT_COLUMNS
    as_dicts = {
        row.pop("visit_id"): {name: (None if pd.isna(value) else value) for name, value in row.items()}
        for row in visits.to_dict("records")
    }
    assert as_dicts == value_checks.VISITS, as_dicts
    assert sorted(visits["visit_type"].unique()) == list(value_checks.VISIT_TYPES)
    assert set(visits["clinic"]) == set(value_checks.OBSERVED) and value_checks.UNUSED == ("Excelsior",)
    assert sorted(path.name for path in (HANDOUT / "data").iterdir()) == ["clinic_visits.csv"]
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert f"CLINIC_ORDER = {list(value_checks.CLINIC_ORDER)!r}".replace("'", '"') in source


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
    assert f"[FIX ]  0/{POINTS[0]}  {first}\n         {value_checks._missing(HANDOUT, value_checks.CLINIC_COUNTS)}" in checks_md
    last = value_checks.CHECKS[-1].name
    assert f"[PASS]  {POINTS[-1]}/{POINTS[-1]}  {last}\n\nScore: 100/100" in checks_md
    # The README and the notebook number their tasks the same way.
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    notebook_headings = [
        line for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        for line in "".join(cell["source"]).splitlines() if re.match(r"#{2,3} (Task \d|\d\.\d)", line)
    ]
    readme_headings = [line for line in readme.splitlines() if re.match(r"#{2,3} (Task \d|\d\.\d)", line)]
    assert notebook_headings == readme_headings, (notebook_headings, readme_headings)


def check_handout_files() -> None:
    """The handout ships every CHECKS_FILES file byte for byte, and no other Python or QA files."""
    workflow = (HANDOUT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    listed = re.search(r"^  CHECKS_FILES: \|\n((?:    \S.*\n)+)", workflow, re.M).group(1).split()
    assert 'CHECKS_PATH: "08/assignment_checks"' in workflow
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
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None, cell.get("id")


def alternative() -> dict[str, str | bytes]:
    """The same answers written the way another careful student might: {artifact path: file contents}.

    Built with observed=False, so every table also carries the zero-visit rows.
    """
    frames = solve(observed=False)

    counts = frames["output/clinic_counts.csv"]
    counts = counts[list(reversed(counts.columns))].copy()
    counts["clinic"] = counts["clinic"].astype(str).str.upper()

    summary = frames["output/clinic_summary.csv"].copy()
    summary["mean_wait_min"] = summary["mean_wait_min"].round(1)

    context = frames["output/visits_with_context.csv"].copy()
    context = context[["wait_vs_clinic", "clinic_mean_wait", *value_checks.VISIT_COLUMNS]].round(2)
    context.columns = [column.title() for column in context.columns]

    pairs = frames["output/clinic_visit_type_summary.csv"].iloc[::-1]

    pivot = frames["output/mean_wait_pivot.csv"][["Telehealth", "New", "Follow-up"]].round(1).reset_index()
    pivot = pivot.rename(columns={"clinic": "CLINIC"})

    return {
        # Columns reversed, upper-case clinic names, CRLF, and UTF-16, as Windows PowerShell 5.1 writes.
        "output/clinic_counts.csv": codecs.BOM_UTF16_LE + counts.to_csv(index=False, lineterminator="\r\n").encode("utf-16-le"),
        # A leading row-number column, a BOM, every cell quoted, NaN for the empty mean, means to one decimal.
        "output/clinic_summary.csv": "﻿" + summary.to_csv(quoting=1, na_rep="NaN"),
        # Title-case headers, columns shuffled, two decimals, a trailing space on each line, no final newline.
        "output/visits_with_context.csv": "\n".join(line + " " for line in context.to_csv(index=False).splitlines()),
        # Rows reversed, two-decimal numbers, and a space after every comma.
        "output/clinic_visit_type_summary.csv": pairs.to_csv(index=False, float_format="%.2f").replace(",", ", "),
        # reset_index() saved with its row numbers, visit types reversed, one decimal, and an all-empty Excelsior row.
        "output/mean_wait_pivot.csv": pivot.to_csv(),
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
    check_readme()
    check_handout_files()
    frames = solve()
    correct = as_files(frames)

    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a08-selftest-") as temporary:
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
        assert detail(result, NAMES[0]).startswith("output/clinic_counts.csv is missing"), detail(result, NAMES[0])

        root = submission("correct", correct)
        result = graded(root)
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}

        # Schema owns an omitted column; recognizable present siblings still need correct values.
        partial = frames["output/visits_with_context.csv"].drop(columns="satisfaction").copy()
        for label in ("missing", "missing-and-wrong", "header-only", "unrecognizable"):
            changed = partial.copy()
            if label == "missing-and-wrong":
                changed.loc[changed.index[0], "wait_min"] += 17
            elif label == "header-only":
                changed = changed.iloc[:0]
            elif label == "unrecognizable":
                changed = changed.iloc[:1].astype(object)
                changed.iloc[0, :] = "WRONG"
            files = dict(correct)
            files["output/visits_with_context.csv"] = changed.to_csv(index=False)
            result = graded(submission("present-siblings-" + label, files))
            if label == "missing":
                assert failing(result) == {"visit context: columns"}, failing(result)
            elif label == "missing-and-wrong":
                assert failing(result) == {"visit context: columns", "visit context: original visit values"}, failing(result)
                message = detail(result, "visit context: original visit values")
                assert "expected" in message and "wait_min" in message, message
            else:
                assert "visit context: original visit values" in failing(result), failing(result)

        # The checks read only output/: poisoned data and code in the submission change nothing.
        (root / "data").mkdir()
        (root / "data" / "clinic_visits.csv").write_text("poison\n", encoding="utf-8")
        for name in ("grading.py", "_value_checks.py", "check_assignment.py"):
            (root / name).write_text("raise RuntimeError('submission code must not run')\n", encoding="utf-8")
        assert graded(root)["score"] == 100

        # Plain text keys, with no ordered categorical: alphabetical rows and columns score the same.
        result = graded(submission("plain-strings", as_files(solve(categorical=False))))
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}

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
                path: frame.to_csv(index=path == "output/mean_wait_pivot.csv", sep=sep, decimal=decimal)
                for path, frame in frames.items()
            }

        semicolons = separated(";", ",")
        assert any(re.search(r";-?\d+,\d", text) for text in semicolons.values() if isinstance(text, str))
        for label, files in (("semicolons", semicolons), ("tabs", separated("\t"))):
            result = graded(submission(label, files))
            assert result["score"] == 100, (label, {name: detail(result, name) for name in failing(result)})
        # One wrong value in a semicolon-separated file still costs only its own check.
        wrong = frames["output/clinic_summary.csv"].copy()
        wrong.loc[wrong["clinic"] == "Mission", "mean_wait_min"] = 15.5
        files = separated(";", ",")
        files["output/clinic_summary.csv"] = wrong.to_csv(index=False, sep=";", decimal=",")
        result = graded(submission("semicolons-one-wrong-value", files))
        assert failing(result) == {"clinic summary: mean_wait_min values"}, {name: detail(result, name) for name in failing(result)}
        message = detail(result, "clinic summary: mean_wait_min values")
        assert "Mission has mean_wait_min 15.5" in message, message

        # Summaries grouped without as_index=False and saved with their index keep their keys: full marks.
        visits = read_visits()
        files = dict(correct)
        files["output/clinic_counts.csv"] = visits.groupby("clinic").agg(**COUNTS).to_csv()
        files["output/clinic_visit_type_summary.csv"] = visits.groupby(["clinic", "visit_type"]).agg(
            visit_count=("visit_id", "size"), mean_wait_min=("wait_min", "mean")).to_csv()
        assert graded(submission("index-keys", files))["score"] == 100

        # The same summaries saved with index=False lose their key columns: only the columns checks fail.
        files["output/clinic_counts.csv"] = visits.groupby("clinic").agg(**COUNTS).to_csv(index=False)
        files["output/clinic_visit_type_summary.csv"] = visits.groupby(["clinic", "visit_type"]).agg(
            visit_count=("visit_id", "size"), mean_wait_min=("wait_min", "mean")).to_csv(index=False)
        files["output/mean_wait_pivot.csv"] = frames["output/mean_wait_pivot.csv"].to_csv(index=False)
        result = graded(submission("lost-keys", files))
        lost = {"clinic counts: columns", "clinic and visit type: columns", "mean wait pivot: columns"}
        assert failing(result) == lost, {name: detail(result, name) for name in failing(result)}
        assert "is missing clinic" in detail(result, "clinic counts: columns"), detail(result, "clinic counts: columns")

        files = dict(correct)
        counts = files["output/clinic_counts.csv"]
        files["output/clinic_counts.csv"] = "\n".join(
            line + "," + line.split(",")[1] for line in counts.splitlines())
        assert graded(submission("duplicate-header", files))["score"] == 100
        files["output/clinic_counts.csv"] = files["output/clinic_counts.csv"].replace("Mission,6,", "Mission,99,")
        result = graded(submission("conflicting-duplicate", files))
        assert failing(result) == {"clinic counts: visit_count values"}, failing(result)

        # A conflicting duplicate cannot hide behind a correct occurrence in either order.
        counts_frame = frames["output/clinic_counts.csv"]
        for wrong_first in (False, True):
            original = counts_frame.to_csv(index=False).splitlines()
            changed = [original[0] + ",visit_count"]
            for line in original[1:]:
                cells = line.split(",")
                duplicate = cells[1] if wrong_first else "99"
                if wrong_first:
                    cells[1] = "99"
                changed.append(",".join(cells) + "," + duplicate)
            files = dict(correct)
            files["output/clinic_counts.csv"] = "\n".join(changed)
            result = graded(submission(f"conflicting-duplicate-{wrong_first}", files))
            assert failing(result) == {"clinic counts: visit_count values"}, failing(result)
            assert result["score"] == 96
        files = dict(correct)
        original = counts_frame.to_csv(index=False).splitlines()
        files["output/clinic_counts.csv"] = "\n".join(
            line + "," + ("visit_count" if i == 0 else str(float(line.split(",")[1])))
            for i, line in enumerate(original))
        assert graded(submission("numeric-equivalent-duplicate", files))["score"] == 100
        files["output/clinic_counts.csv"] = "\n".join(
            line + "," + ("visit_count" if i == 0 else "") for i, line in enumerate(original))
        assert graded(submission("blank-duplicate", files))["score"] == 100

        # CSV readers differ by Python version on NUL; a broken file is unreadable on both.
        files = dict(correct)
        files["output/clinic_counts.csv"] = b"\xff\xfe\x00"
        result = graded(submission("nul-byte-file", files))
        assert failing(result) == {name for name in NAMES if name.startswith(PREFIX["output/clinic_counts.csv"])}, failing(result)

        # One mistake costs exactly its own check.
        counts, summary = correct["output/clinic_counts.csv"], correct["output/clinic_summary.csv"]
        context, pairs = correct["output/visits_with_context.csv"], correct["output/clinic_visit_type_summary.csv"]
        pivot = correct["output/mean_wait_pivot.csv"]
        mistakes = {
            "clinic counts: columns": ("output/clinic_counts.csv", counts.replace("patient_count", "patients", 1)),
            "clinic counts: one row per clinic": ("output/clinic_counts.csv", drop_last_row(counts)),
            "clinic counts: visit_count values": ("output/clinic_counts.csv", edit(counts, "Mission", "visit_count", "7")),
            "clinic counts: satisfaction_count values": (
                "output/clinic_counts.csv", edit(counts, "Sunset", "satisfaction_count", "5")),
            "clinic counts: patient_count values": ("output/clinic_counts.csv", edit(counts, "Mission", "patient_count", "6")),
            "clinic summary: columns": (
                "output/clinic_summary.csv", "\n".join(line + ",x" for line in summary.splitlines()) + "\n"),
            "clinic summary: one row per clinic": ("output/clinic_summary.csv", summary + summary.splitlines()[2] + "\n"),
            "clinic summary: count values": (
                "output/clinic_summary.csv", edit(summary, "Bayview", "satisfaction_count", "3")),
            "clinic summary: total_wait_min values": (
                "output/clinic_summary.csv", edit(summary, "Mission", "total_wait_min", "96")),
            "clinic summary: mean_wait_min values": (
                "output/clinic_summary.csv", edit(summary, "Sunset", "mean_wait_min", "31.5")),
            "visit context: columns": ("output/visits_with_context.csv", context.replace("wait_vs_clinic", "wait_diff", 1)),
            "visit context: one row per visit": ("output/visits_with_context.csv", drop_last_row(context)),
            "visit context: original visit values": (
                "output/visits_with_context.csv", edit(context, "V003", "satisfaction", "0.0")),
            "visit context: clinic_mean_wait values": (
                "output/visits_with_context.csv", edit(context, "V010", "clinic_mean_wait", "14.333333333333334")),
            "visit context: wait_vs_clinic values": (
                "output/visits_with_context.csv", edit(context, "V001", "wait_vs_clinic", "-3.666666666666666")),
            "clinic and visit type: columns": (
                "output/clinic_visit_type_summary.csv", pairs.replace("visit_type", "type", 1)),
            "clinic and visit type: one row per pair": ("output/clinic_visit_type_summary.csv", drop_last_row(pairs)),
            "clinic and visit type: visit_count values": (
                "output/clinic_visit_type_summary.csv", edit(pairs, "Mission|Follow-up", "visit_count", "2")),
            "clinic and visit type: mean_wait_min values": (
                "output/clinic_visit_type_summary.csv", edit(pairs, "Sunset|New", "mean_wait_min", "18.25")),
            "mean wait pivot: columns": ("output/mean_wait_pivot.csv", pivot.replace("Telehealth", "Tele", 1)),
            "mean wait pivot: one row per clinic": ("output/mean_wait_pivot.csv", pivot + "All,14.6,27.0,6.33\n"),
            "mean wait pivot: mean waits": ("output/mean_wait_pivot.csv", edit(pivot, "Bayview", "New", "13.0")),
            "mean wait pivot: empty cell stays empty": ("output/mean_wait_pivot.csv", edit(pivot, "Sunset", "Telehealth", "0.0")),
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

        # A zero-visit row must show zero visits: Excelsior with blank counts costs the count checks.
        files = dict(correct)
        files["output/clinic_counts.csv"] = counts + "Excelsior,,,\n"
        result = graded(submission("blank-excelsior", files))
        assert failing(result) == {
            "clinic counts: visit_count values", "clinic counts: satisfaction_count values",
            "clinic counts: patient_count values"}, failing(result)
        assert "Excelsior has visit_count blank, expected 0" in detail(result, "clinic counts: visit_count values")

        # A pivot saved after .dropna() loses Sunset's row and with it the cell that should stay empty.
        files = dict(correct)
        files["output/mean_wait_pivot.csv"] = frames["output/mean_wait_pivot.csv"].dropna().to_csv()
        result = graded(submission("pivot-dropna", files))
        assert failing(result) == {
            "mean wait pivot: one row per clinic"}, failing(result)

        # Only the optional zero-visit row earns no value checks: there is nothing required to compare.
        files = dict(correct)
        files["output/clinic_counts.csv"] = counts.splitlines()[0] + "\nExcelsior,0,0,0\n"
        result = graded(submission("excelsior-only", files))
        assert failing(result) == set(artifact_checks("output/clinic_counts.csv")) - {"clinic counts: columns"}, (
            failing(result))

        # Feedback names what was expected and what was found.
        files = dict(correct)
        files["output/mean_wait_pivot.csv"] = pd.pivot_table(
            visits, values="wait_min", index="clinic", columns="visit_type", aggfunc="mean", fill_value=0).to_csv()
        result = graded(submission("feedback-fill", files))
        assert failing(result) == {"mean wait pivot: empty cell stays empty"}, failing(result)
        message = detail(result, "mean wait pivot: empty cell stays empty")
        assert "Sunset has Telehealth 0" in message and "expected blank" in message and "filling it with 0" in message, message
        files = dict(correct)
        files["output/visits_with_context.csv"] = mistakes["visit context: wait_vs_clinic values"][1]
        message = detail(graded(submission("feedback-sign", files)), "visit context: wait_vs_clinic values")
        assert "V001 has wait_vs_clinic -3.666666666666666, expected 3.67" in message and "minus" in message, message
        files = dict(correct)
        files["output/clinic_visit_type_summary.csv"] = mistakes["clinic and visit type: columns"][1]
        message = detail(graded(submission("feedback-columns", files)), "clinic and visit type: columns")
        assert "is missing visit_type and also has type" in message and "as_index=False" in message, message

        # A missing file is never told to take another required file's name, however alike the two names are.
        tasks_1_to_2_2 = ("output/clinic_counts.csv", "output/clinic_summary.csv", "output/visits_with_context.csv")
        result = graded(submission("midway", {path: correct[path] for path in tasks_1_to_2_2}))
        assert failing(result) == set(artifact_checks("output/clinic_visit_type_summary.csv")) | set(
            artifact_checks("output/mean_wait_pivot.csv")), failing(result)
        assert not any("Found" in detail(result, name) for name in failing(result)), [
            detail(result, name) for name in failing(result)]
        files = {path: text for path, text in correct.items() if path != "output/clinic_summary.csv"}
        message = detail(graded(submission("summary-missing", files)), "clinic summary: columns")
        assert "Found" not in message, message
        files = dict(correct)
        files["output/clinic_visit_types_summary.csv"] = files.pop("output/clinic_visit_type_summary.csv")
        message = detail(graded(submission("typo-name", files)), "clinic and visit type: columns")
        assert message.endswith("Found output/clinic_visit_types_summary.csv; save it as "
                                "output/clinic_visit_type_summary.csv instead."), message

        # A pivot with index= and columns= swapped is read the right way round: only its columns check fails.
        files = dict(correct)
        files["output/mean_wait_pivot.csv"] = pd.pivot_table(
            visits, values="wait_min", index="visit_type", columns="clinic", aggfunc="mean").to_csv()
        result = graded(submission("pivot-swapped", files))
        assert failing(result) == {"mean wait pivot: columns"}, {name: detail(result, name) for name in failing(result)}
        message = detail(result, "mean wait pivot: columns")
        assert "swapped" in message and 'index="clinic" and columns="visit_type"' in message, message

        # margins=True adds a Total row and column: the columns and rows checks say where it came from.
        files = dict(correct)
        files["output/mean_wait_pivot.csv"] = pd.pivot_table(
            visits, values="wait_min", index="clinic", columns="visit_type", aggfunc="mean", margins=True,
            margins_name="Total").to_csv()
        result = graded(submission("pivot-margins", files))
        assert failing(result) == {"mean wait pivot: columns", "mean wait pivot: one row per clinic"}, failing(result)
        for name in failing(result):
            assert "Total comes from margins=True" in detail(result, name), detail(result, name)
        # Headers are quoted as the file writes them, not lowercased.
        assert "also has Total;" in detail(result, "mean wait pivot: columns"), detail(result, "mean wait pivot: columns")

        # wait_vs_clinic worked out from a wrong clinic_mean_wait costs only the clinic_mean_wait check.
        context = visits.copy()
        context["clinic_mean_wait"] = visits.groupby("clinic")["wait_min"].transform("median")
        context["wait_vs_clinic"] = context["wait_min"] - context["clinic_mean_wait"]
        files = dict(correct)
        files["output/visits_with_context.csv"] = context.to_csv(index=False)
        result = graded(submission("context-median", files))
        assert failing(result) == {"visit context: clinic_mean_wait values"}, failing(result)

        # A column assigned from .mean() is blank on every row, and the message names that likely cause.
        context["clinic_mean_wait"] = visits.groupby("clinic")["wait_min"].mean()
        context["wait_vs_clinic"] = context["wait_min"] - context["clinic_mean_wait"]
        files["output/visits_with_context.csv"] = context.to_csv(index=False)
        result = graded(submission("context-mean-not-transform", files))
        assert failing(result) == {"visit context: clinic_mean_wait values"}
        message = detail(result, "visit context: clinic_mean_wait values")
        assert "Every clinic_mean_wait is blank" in message and "transform" in message, message

        # A mistake copied from an earlier artifact is charged at its source.
        files = dict(correct)
        for path in ("output/clinic_counts.csv", "output/clinic_summary.csv"):
            files[path] = edit(files[path], "Mission", "patient_count", "6")
        result = graded(submission("copied-count-error", files))
        assert failing(result) == {"clinic counts: patient_count values"}, failing(result)
        files = dict(correct)
        files["output/clinic_visit_type_summary.csv"] = edit(pairs, "Sunset|New", "mean_wait_min", "99")
        files["output/mean_wait_pivot.csv"] = edit(pivot, "Sunset", "New", "99")
        result = graded(submission("copied-pair-error", files))
        assert failing(result) == {"clinic and visit type: mean_wait_min values"}, failing(result)
        files = dict(correct)
        files["output/clinic_summary.csv"] = edit(summary, "Sunset", "mean_wait_min", "99")
        changed = frames["output/visits_with_context.csv"].copy()
        changed.loc[changed["clinic"] == "Sunset", "clinic_mean_wait"] = 99
        changed["wait_vs_clinic"] = changed["wait_min"] - changed["clinic_mean_wait"]
        files["output/visits_with_context.csv"] = changed.to_csv(index=False)
        result = graded(submission("copied-clinic-mean-error", files))
        assert failing(result) == {"clinic summary: mean_wait_min values"}, failing(result)

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
        "Assignment 08 checks: the visit-log copy, README checkpoints, contract, and numbering, and the handout's "
        "byte-identical checks agree; empty and scaffold earn 0; correct, plain-string, alternative, with-index, "
        "semicolon- and tab-separated, "
        f"and index-key submissions earn 100; lost key columns cost only the columns checks; each of the "
        f"{len(NAMES)} single mistakes and each missing file costs only its own checks; swapped axes, margins, "
        "and a derived column cost only their own checks; feedback and pytest pass."
    )


if __name__ == "__main__":
    run()
