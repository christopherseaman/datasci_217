"""Development regression checks for the Assignment 06 checks.

Builds submissions in ignored `scratch/`, answers the assignment with pandas
the way the lecture does, and confirms what each kind of submission scores:
a correct one earns 100, a correct one written differently (column and row
order, CRLF, a BOM, quoting, padding, letter case, number format, row-number
and index columns, melt's default column names, `NaN` for an empty cell, and
the extra `record_status` column the lecture's merge keeps) earns 100 too, an
empty directory and the untouched handout earn 0, one wrong value or one
misnamed column costs exactly its own check, and a missing file costs only its
own artifact's checks. It also confirms that the copies of the supplied data in
`_value_checks.py` match `06/assignment/data/`, that the README's checkpoints
and completion contract agree with the checks, that the handout ships every
file in `CHECKS_FILES` byte for byte and no other Python file, and that the
handout notebook ships with its outputs cleared.

    uv run --python 3.13 --with pandas==3.0.5 --with 'pytest>=8,<9' \\
        python 06/assignment_checks/_grader_selftest/run.py
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd


CHECKS = Path(__file__).resolve().parents[1]
HANDOUT = CHECKS.parent / "assignment"
SCRATCH = CHECKS.parents[1] / "scratch"
sys.path.insert(0, str(CHECKS))
import _value_checks as value_checks  # noqa: E402
from grading import POINTS, grade_submission  # noqa: E402

NAMES = [check.name for check in value_checks.CHECKS]
POINTS_BY_NAME = dict(zip(NAMES, POINTS, strict=True))
ARTIFACTS = {
    artifact.path: artifact
    for artifact in (
        value_checks.MERGE_AUDIT,
        value_checks.COMBINED,
        value_checks.ALIGNED,
        value_checks.SBP_LONG,
        value_checks.SBP_ROUND_TRIP,
    )
}
PREFIX = {
    "output/specimen_merge_audit.csv": "merge audit:",
    "output/combined_specimens.csv": "combined specimens:",
    "output/aligned_features.csv": "aligned features:",
    "output/sbp_long.csv": "SBP long:",
    "output/sbp_round_trip.csv": "SBP round trip:",
}


def read_data(name: str) -> pd.DataFrame:
    return pd.read_csv(HANDOUT / "data" / name)


def solve() -> dict[str, pd.DataFrame]:
    """Answer the assignment with the lecture's pandas calls; returns {artifact path: frame to save}."""
    specimens = read_data("specimens.csv")
    clinics_history = read_data("clinics_history.csv")
    batch_a = read_data("specimens_batch_a.csv")
    batch_b = read_data("specimens_batch_b.csv")
    transit_times = read_data("transit_times.csv")
    sbp_wide = read_data("sbp_wide.csv")

    current = clinics_history.loc[clinics_history["record_status"] == "current", ["clinic_id", "clinic_name", "region"]]
    merge_audit = pd.merge(specimens, current, on="clinic_id", how="left", validate="many_to_one", indicator=True)

    batch_a["source_partition"] = "batch_a"
    batch_b["source_partition"] = "batch_b"
    combined = pd.concat([batch_a, batch_b], ignore_index=True)
    assert combined.drop(columns="source_partition").equals(specimens)

    volumes = batch_a[["specimen_id", "volume_ml"]].set_index("specimen_id")
    aligned = pd.concat([volumes, transit_times.set_index("specimen_id")], axis=1).reset_index()

    sbp_long = pd.melt(sbp_wide, id_vars=["patient_id"], value_vars=["baseline", "followup"],
                       var_name="visit", value_name="sbp")
    round_trip = sbp_long.pivot(index="patient_id", columns="visit", values="sbp").reset_index()
    round_trip.columns.name = None
    assert round_trip.equals(sbp_wide)

    return {
        "output/specimen_merge_audit.csv": merge_audit,
        "output/combined_specimens.csv": combined,
        "output/aligned_features.csv": aligned,
        "output/sbp_long.csv": sbp_long,
        "output/sbp_round_trip.csv": round_trip,
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
    """The copies of the supplied data in _value_checks.py match the handout's data files."""
    specimens = read_data("specimens.csv").astype({"collection_number": int})
    assert tuple(specimens.columns) == value_checks.SPECIMEN_COLUMNS
    as_dicts = {row.pop("specimen_id"): row for row in specimens.to_dict("records")}
    assert as_dicts == value_checks.SPECIMENS, as_dicts
    batch_a, batch_b = read_data("specimens_batch_a.csv"), read_data("specimens_batch_b.csv")
    assert tuple(batch_a["specimen_id"]) == value_checks.BATCH_A
    assert pd.concat([batch_a, batch_b], ignore_index=True).equals(read_data("specimens.csv"))
    clinics = read_data("clinics_history.csv")
    for status, expected in (("current", value_checks.CURRENT_CLINICS), ("retired", value_checks.RETIRED_CLINICS)):
        rows = clinics[clinics["record_status"] == status]
        assert rows["clinic_id"].is_unique
        assert {row["clinic_id"]: {"clinic_name": row["clinic_name"], "region": row["region"]}
                for row in rows.to_dict("records")} == expected, status
    transit = read_data("transit_times.csv")
    assert dict(zip(transit["specimen_id"], transit["transit_min"])) == value_checks.TRANSIT_MIN
    sbp = read_data("sbp_wide.csv")
    assert tuple(sbp.columns) == ("patient_id", *value_checks.VISITS)
    assert {row.pop("patient_id"): row for row in sbp.to_dict("records")} == value_checks.SBP_WIDE
    assert sorted(path.name for path in (HANDOUT / "data").iterdir()) == sorted(
        ["specimens.csv", "clinics_history.csv", "specimens_batch_a.csv", "specimens_batch_b.csv",
         "transit_times.csv", "sbp_wide.csv"]
    )


def check_readme() -> None:
    """Each README checkpoint names its artifact's header, and the contract table matches the checks."""
    readme = (HANDOUT / "README.md").read_text(encoding="utf-8")
    checkpoints = re.findall(r"> \*\*Checkpoint: `([^`]+)`\*\*\n> The header line `([^`]+)`", readme)
    assert {path: header for path, header in checkpoints} == {
        path: ",".join(artifact.columns) for path, artifact in ARTIFACTS.items()
    }, checkpoints
    contract = re.findall(r"^\| `(output/[^`]+)` \| .* \| ([^|]+) \| (\d+) \|$", readme, re.M)
    assert [(name.strip(), int(points)) for _, name, points in contract] == list(zip(NAMES, POINTS)), contract
    for path, name, _ in contract:
        assert name.strip().startswith(PREFIX[path]), (path, name)
    first = value_checks.CHECKS[0].name
    assert f"[FIX ]  0/{POINTS[0]}  {first}\n         {value_checks._missing(HANDOUT, value_checks.MERGE_AUDIT)}" in readme


def check_handout_files() -> None:
    """The handout ships every CHECKS_FILES file byte for byte, and no other Python or QA files."""
    workflow = (HANDOUT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    listed = re.search(r"^  CHECKS_FILES: \|\n((?:    \S.*\n)+)", workflow, re.M).group(1).split()
    assert 'CHECKS_PATH: "06/assignment_checks"' in workflow
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


def alternative(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """The same answers written the way another careful student might: {artifact path: file text}."""
    merge = frames["output/specimen_merge_audit.csv"].copy()
    merge["record_status"] = merge["_merge"].map({"both": "current"})
    merge["_merge"] = merge["_merge"].astype(str).str.upper()
    merge = merge[list(reversed(merge.columns))].iloc[::-1]

    combined = frames["output/combined_specimens.csv"].copy()
    combined["source_partition"] = combined["source_partition"].str.replace("batch", "Batch")
    combined["specimen_id"] = combined["specimen_id"] + "  "

    aligned = frames["output/aligned_features.csv"].set_index("specimen_id")
    aligned.index.name = None

    sbp_long = frames["output/sbp_long.csv"].rename(columns={"visit": "Visit", "sbp": "SBP"})
    sbp_long["Visit"] = sbp_long["Visit"].str.title()
    sbp_long["SBP"] = sbp_long["SBP"].astype(float)

    round_trip = frames["output/sbp_round_trip.csv"][["followup", "patient_id", "baseline"]]

    return {
        # A trailing record_status column, reversed column and row order, upper-case labels, CRLF.
        "output/specimen_merge_audit.csv": merge.to_csv(index=False, na_rep="NaN", lineterminator="\r\n"),
        # A leading row-number column, a BOM, padded IDs, quoted cells, and two-decimal numbers.
        "output/combined_specimens.csv": "﻿" + combined.to_csv(quoting=1, float_format="%.2f"),
        # The specimen IDs saved in an index column with no header, and a trailing space on each line.
        "output/aligned_features.csv": "\n".join(line + " " for line in aligned.to_csv().splitlines()),
        # Title-case labels and headers, SBP as 148.0, and no final newline.
        "output/sbp_long.csv": sbp_long.to_csv(index=False).rstrip("\n"),
        # Columns in another order and a space after every comma.
        "output/sbp_round_trip.csv": round_trip.to_csv(index=False).replace(",", ", "),
    }


def edit(text: str, row_key: str, column: str, value: str) -> str:
    """The CSV text with one cell replaced; rows are found by the first cell."""
    lines = text.splitlines()
    header = lines[0].split(",")
    position = header.index(column)
    for number, line in enumerate(lines[1:], start=1):
        cells = line.split(",")
        if cells[0] == row_key.split("|")[0] and (len(row_key.split("|")) == 1 or row_key.split("|")[1] in cells):
            cells[position] = value
            lines[number] = ",".join(cells)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"no row {row_key}")


def run() -> None:
    check_supplied_data()
    check_readme()
    check_handout_files()
    frames = solve()
    correct = {path: frame.to_csv(index=False) for path, frame in frames.items()}

    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a06-selftest-") as temporary:
        workspace = Path(temporary)

        def submission(name: str, files: dict[str, str]) -> Path:
            root = workspace / name
            (root / "output").mkdir(parents=True)
            for path, text in files.items():
                (root / path).write_text(text, encoding="utf-8", newline="")
            return root

        empty = workspace / "empty"
        empty.mkdir()
        assert graded(empty)["score"] == 0
        scaffold = workspace / "scaffold"
        shutil.copytree(HANDOUT, scaffold, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv"))
        result = graded(scaffold)
        assert result["score"] == 0 and failing(result) == set(NAMES)
        assert detail(result, NAMES[0]).startswith("output/specimen_merge_audit.csv is missing"), detail(result, NAMES[0])

        root = submission("correct", correct)
        result = graded(root)
        assert result["score"] == 100, failing(result)

        # The checks read only output/: poisoned data and code in the submission change nothing.
        for name in ("specimens.csv", "sbp_wide.csv"):
            (root / "data").mkdir(exist_ok=True)
            (root / "data" / name).write_text("poison\n", encoding="utf-8")
        for name in ("grading.py", "_value_checks.py", "check_assignment.py"):
            (root / name).write_text("raise RuntimeError('submission code must not run')\n", encoding="utf-8")
        assert graded(root)["score"] == 100

        result = graded(submission("alternative", alternative(frames)))
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
        wrong = frames["output/combined_specimens.csv"].copy()
        wrong.loc[wrong["specimen_id"] == "SP103", "volume_ml"] = 40.5
        files = separated(";", ",")
        files["output/combined_specimens.csv"] = wrong.to_csv(index=False, sep=";", decimal=",")
        result = graded(submission("semicolons-one-wrong-value", files))
        assert failing(result) == {"combined specimens: specimen values"}, {name: detail(result, name) for name in failing(result)}
        message = detail(result, "combined specimens: specimen values")
        assert "SP103 has volume_ml 40.5, expected 4" in message, message

        # The round trip saved straight from pivot(), index and all, and the aligned table with its index.
        pivoted = frames["output/sbp_long.csv"].pivot(index="patient_id", columns="visit", values="sbp")
        files = dict(correct)
        files["output/sbp_round_trip.csv"] = pivoted.to_csv()
        files["output/aligned_features.csv"] = frames["output/aligned_features.csv"].set_index("specimen_id").to_csv()
        assert graded(submission("pivot-with-index", files))["score"] == 100

        # Saved without the ID column (index=False on the aligned table, no reset_index() on the round
        # trip): the columns and rows checks fail and say how to keep it, and correct values still count.
        files = dict(correct)
        files["output/aligned_features.csv"] = frames["output/aligned_features.csv"].drop(columns="specimen_id").to_csv(index=False)
        files["output/sbp_round_trip.csv"] = pivoted.to_csv(index=False)
        result = graded(submission("no-id-column", files))
        assert failing(result) == {"aligned features: columns", "aligned features: one row per specimen",
                                   "SBP round trip: columns", "SBP round trip: one row per patient"}, failing(result)
        assert "leave out index=False" in detail(result, "aligned features: one row per specimen")
        assert "reset_index()" in detail(result, "SBP round trip: one row per patient")
        files["output/sbp_round_trip.csv"] = pivoted.to_csv(index=False).replace("162", "126")
        result = graded(submission("no-id-column-wrong-value", files))
        assert "SBP round trip: baseline values" in failing(result), failing(result)
        assert "SBP round trip: followup values" not in failing(result), failing(result)

        # One mistake costs exactly its own check.
        mistakes = {
            "merge audit: columns": ("output/specimen_merge_audit.csv", lambda text: text.replace(",region,", ",clinic_region,", 1)),
            "merge audit: one row per specimen": ("output/specimen_merge_audit.csv", lambda text: "\n".join(text.splitlines()[:-1]) + "\n"),
            "merge audit: specimen values": ("output/specimen_merge_audit.csv", lambda text: edit(text, "SP103", "volume_ml", "40.0")),
            "merge audit: current clinic names and regions": ("output/specimen_merge_audit.csv", lambda text: edit(text, "SP101", "clinic_name", "Bayview Annex")),
            "merge audit: _merge indicator": ("output/specimen_merge_audit.csv", lambda text: edit(text, "SP106", "_merge", "both")),
            "combined specimens: columns": ("output/combined_specimens.csv", lambda text: "\n".join(line + ",x" for line in text.splitlines()) + "\n"),
            "combined specimens: one row per specimen": ("output/combined_specimens.csv", lambda text: text + text.splitlines()[5] + "\n"),
            "combined specimens: specimen values": ("output/combined_specimens.csv", lambda text: edit(text, "SP106", "specimen_type", "blood")),
            "combined specimens: source_partition labels": ("output/combined_specimens.csv", lambda text: edit(text, "SP105", "source_partition", "batch_a")),
            "aligned features: columns": ("output/aligned_features.csv", lambda text: text.replace("transit_min", "transit_minutes", 1)),
            "aligned features: one row per specimen": ("output/aligned_features.csv", lambda text: "\n".join(text.splitlines()[:-1]) + "\n"),
            "aligned features: volume_ml values": ("output/aligned_features.csv", lambda text: edit(text, "SP108", "volume_ml", "0.0")),
            "aligned features: transit_min values": ("output/aligned_features.csv", lambda text: edit(text, "SP102", "transit_min", "54.0")),
            "SBP long: columns": ("output/sbp_long.csv", lambda text: text.replace("patient_id,visit,sbp", "patient_id,variable,value", 1)),
            "SBP long: one row per patient and visit": ("output/sbp_long.csv", lambda text: "\n".join(text.splitlines()[:-1]) + "\n"),
            "SBP long: sbp values": ("output/sbp_long.csv", lambda text: edit(text, "P203|followup", "sbp", "114")),
            "SBP round trip: columns": ("output/sbp_round_trip.csv", lambda text: text.replace("followup", "follow_up", 1)),
            "SBP round trip: one row per patient": ("output/sbp_round_trip.csv", lambda text: text + "P205,150,140\n"),
            "SBP round trip: baseline values": ("output/sbp_round_trip.csv", lambda text: edit(text, "P201", "baseline", "184")),
            "SBP round trip: followup values": ("output/sbp_round_trip.csv", lambda text: edit(text, "P204", "followup", "124")),
        }
        assert sorted(mistakes) == sorted(NAMES)
        for number, (name, (path, change)) in enumerate(mistakes.items()):
            files = dict(correct)
            files[path] = change(correct[path])
            result = graded(submission(f"mistake-{number:02}", files))
            assert failing(result) == {name}, (name, {other: detail(result, other) for other in failing(result)})
            assert result["score"] == 100 - cost(name), (name, result["score"])

        # A missing file costs only its own artifact's checks.
        for number, path in enumerate(correct):
            files = {other: text for other, text in correct.items() if other != path}
            result = graded(submission(f"missing-{number}", files))
            assert failing(result) == set(artifact_checks(path)), (path, failing(result))

        # Merging every clinic record repeats K01's specimens under both names: the rows and names
        # checks fail, and the rows check says to keep the current records.
        specimens = read_data("specimens.csv")
        unfiltered = pd.merge(specimens, read_data("clinics_history.csv")[["clinic_id", "clinic_name", "region"]],
                              on="clinic_id", how="left", indicator=True)
        files = dict(correct)
        files["output/specimen_merge_audit.csv"] = unfiltered.to_csv(index=False)
        result = graded(submission("unfiltered-merge", files))
        assert failing(result) == {"merge audit: one row per specimen", "merge audit: current clinic names and regions"}
        message = detail(result, "merge audit: one row per specimen")
        assert "SP101 2 times" in message and 'record_status is "current"' in message, message
        message = detail(result, "merge audit: current clinic names and regions")
        assert "SP101 has clinic_name Bayview Annex, expected Bayview Clinic" in message, message

        # An inner merge drops SP106, which costs the rows check alone.
        inner = pd.merge(specimens, read_data("clinics_history.csv").query("record_status == 'current'")
                         [["clinic_id", "clinic_name", "region"]], on="clinic_id", indicator=True)
        files = dict(correct)
        files["output/specimen_merge_audit.csv"] = inner.to_csv(index=False)
        result = graded(submission("inner-merge", files))
        assert failing(result) == {"merge audit: one row per specimen"}, failing(result)
        assert "is missing SP106" in detail(result, "merge audit: one row per specimen")

        # Feedback names what was expected and what was found.
        files = dict(correct)
        files["output/aligned_features.csv"] = edit(correct["output/aligned_features.csv"], "SP108", "volume_ml", "0.0")
        message = detail(graded(submission("feedback", files)), "aligned features: volume_ml values")
        assert "SP108 has volume_ml 0.0, expected blank" in message, message
        files = dict(correct)
        files["output/sbp_long.csv"] = correct["output/sbp_long.csv"].replace("patient_id,visit,sbp", "patient_id,variable,value")
        message = detail(graded(submission("feedback-columns", files)), "SBP long: columns")
        assert "is missing visit and sbp and also has variable and value" in message and 'var_name="visit"' in message, message

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
            summary = finished.stdout.strip().splitlines()[-1]
            if expected_failures:
                assert f"{expected_failures} failed" in summary, finished.stdout
            else:
                assert finished.returncode == 0 and f"{len(NAMES)} passed" in summary, finished.stdout

    print(
        "Assignment 06 checks: supplied-data copies, README checkpoints and contract, and the handout's "
        "byte-identical checks agree; empty and scaffold earn 0; correct, alternative, with-index, "
        "pivot-with-index, and semicolon- and tab-separated submissions earn 100; each of the 20 single mistakes and each missing file costs "
        "only its own checks; a file without its ID column keeps its value checks; feedback and pytest pass."
    )


if __name__ == "__main__":
    run()
