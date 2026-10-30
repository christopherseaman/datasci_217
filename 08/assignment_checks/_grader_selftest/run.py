"""Development regression checks for the Assignment 08 checks.

Builds submissions in ignored `scratch/`, answers the assignment with pandas
the way the lecture does, and confirms what each kind of submission scores:
a correct one earns 100, and so do one built from plain text keys instead of
the ordered categorical, one built with observed=False, and one written
differently (column and row order, CRLF, a BOM, UTF-16, quoting, padding,
letter case, number format, rounding, row-number columns, `NaN` for an empty
cell, semicolons or tabs). An empty directory and the untouched handout earn
0, one wrong value or one misnamed column costs exactly its own check, an
omitted value column and renamed columns with the right values cost only the
columns check, and a missing file costs only its own artifact's checks. The
plausible mistakes (a peer median over every priority, `>=` for `>`, a dose
filter over every modality, top reports over every priority, a row mask in
place of filter, `fill_value=0`, swapped pivot axes, margins, list arguments)
each cost only their own checks, with messages naming the cause. It also
confirms that the frozen values in `_value_checks.py` match
`08/assignment/data/imaging_exams.parquet` and the generator, that the README's
checkpoints and the completion contract agree with the checks and the README
prints no answer expressions, that the
handout ships every file in `CHECKS_FILES` byte for byte and no other Python
file, and that the handout notebook ships with its outputs cleared.

    uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 \\
        --with 'pytest>=8,<9' python 08/assignment_checks/_grader_selftest/run.py
"""

from __future__ import annotations

from pathlib import Path
import codecs
import io
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
REPO = CHECKS.parents[1]
SCRATCH = REPO / "scratch"
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(REPO / "scripts"))
sys.dont_write_bytecode = True
import _value_checks as value_checks  # noqa: E402
import build_assignment08_data as generator  # noqa: E402
from grading import POINTS, grade_submission  # noqa: E402

NAMES = [check.name for check in value_checks.CHECKS]
POINTS_BY_NAME = dict(zip(NAMES, POINTS, strict=True))
ARTIFACTS = {artifact.path: artifact for artifact in value_checks.ARTIFACTS}
SUMMARY, PAIRS, DELAYED = "output/site_summary.csv", "output/site_modality_summary.csv", "output/delayed_stat_exams.csv"
DOSE, SLOWEST, PIVOT = "output/ct_dose_review.csv", "output/slowest_stat_reports.csv", "output/median_report_pivot.csv"
PREFIX = {SUMMARY: "site summary:", PAIRS: "site and modality:", DELAYED: "delayed STAT exams:",
          DOSE: "CT dose review:", SLOWEST: "slowest STAT reports:", PIVOT: "median report pivot:"}
DELAYED_COLUMNS = ["exam_id", "site", "modality", "wait_hr", "peer_median_wait_hr", "wait_vs_peer_hr"]
EXAMS = pd.read_parquet(HANDOUT / "data" / "imaging_exams.parquet")


def solve(categorical: bool = True, observed: bool = True, use_filter: bool = False) -> dict[str, pd.DataFrame]:
    """Answer the assignment with the lecture's pandas calls; returns {artifact path: frame to save}."""
    exams = EXAMS.copy()
    if categorical:
        exams["site"] = pd.Categorical(exams["site"], categories=list(value_checks.SITE_ORDER), ordered=True)
        per_site = exams.groupby("site", observed=False).size()
        assert per_site["Eastport"] == 0 and per_site["Valley"] == 25384, per_site

    site_summary = exams.groupby("site", as_index=False, observed=observed).agg(
        exam_count=("exam_id", "size"), report_count=("report_hr", "count"), patient_count=("patient_id", "nunique"),
        total_scan_min=("scan_min", "sum"), mean_scan_min=("scan_min", "mean"))
    site_modality = exams.groupby(["site", "modality"], as_index=False, observed=observed).agg(
        exam_count=("exam_id", "size"), median_wait_hr=("wait_hr", "median"),
        median_report_hr=("report_hr", "median"))

    stat = exams[exams["priority"] == "STAT"].copy()
    stat["peer_median_wait_hr"] = stat.groupby(["site", "modality"], observed=observed)["wait_hr"].transform("median")
    stat["wait_vs_peer_hr"] = stat["wait_hr"] - stat["peer_median_wait_hr"]
    delayed = stat[stat["wait_hr"] > 8][DELAYED_COLUMNS]
    assert len(stat) == 152925 and len(delayed) == 190

    ct = exams[exams["modality"] == "CT"]
    dose = ct[ct.groupby("patient_id")["dose_msv"].transform("sum") >= 100]
    if use_filter:
        assert ct.groupby("patient_id").filter(lambda g: g["dose_msv"].sum() >= 100).equals(dose)
    assert len(dose) == 1253 and dose["patient_id"].nunique() == 159 and len(ct[ct["dose_msv"] >= 100]) == 0

    slowest = stat.groupby("site", observed=observed).apply(lambda g: g.nlargest(3, "report_hr"), include_groups=False)
    slowest = slowest.reset_index(level="site")[["exam_id", "site", "modality", "report_hr"]]

    pivot = pd.pivot_table(exams, values="report_hr", index="site", columns="modality", aggfunc="median",
                           observed=observed, dropna=observed)
    if observed:
        twin = exams.groupby(["site", "modality"], observed=True)["report_hr"].median().unstack()
        assert pivot.equals(twin) and int(pivot.isna().sum().sum()) == 1
    assert pd.crosstab(exams["site"], exams["modality"]).loc["Valley", "MRI"] == 0

    return {SUMMARY: site_summary, PAIRS: site_modality, DELAYED: delayed,
            DOSE: dose[["exam_id", "patient_id", "dose_msv"]], SLOWEST: slowest, PIVOT: pivot}


def as_files(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """The way the README saves each frame: index=False, except the pivot, which keeps its site index."""
    return {path: frame.to_csv() if path == PIVOT else frame.to_csv(index=False) for path, frame in frames.items()}


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
    """The handout's Parquet file is the generator's output, and the frozen values match it."""
    assert sorted(path.name for path in (HANDOUT / "data").iterdir()) == ["imaging_exams.parquet"]
    assert EXAMS.equals(generator.build()), "data/imaging_exams.parquet differs from build_assignment08_data.build()"
    for name, value in generator.expected(EXAMS).items():
        assert getattr(value_checks, name) == value, f"{name} in _value_checks.py differs from the data; rerun the build"
    assert value_checks.UNUSED == ("Eastport",)
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert f"SITE_ORDER = {list(value_checks.SITE_ORDER)!r}".replace("'", '"') in source
    assert "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/08/assignment/data" in source


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
    assert f"[FIX ]  0/{POINTS[0]}  {first}\n         {value_checks._missing(HANDOUT, value_checks.SITE_SUMMARY_TABLE)}" in checks_md
    last = value_checks.CHECKS[-1].name
    assert f"[PASS]  {POINTS[-1]}/{POINTS[-1]}  {last}\n\nScore: 100/100" in checks_md
    for count in (len(value_checks.SITE_MODALITY), len(value_checks.DELAYED_STAT)):
        assert f"then {count} rows" in readme, count
    assert f"then {len(value_checks.CT_DOSE_REVIEW):,} rows" in readme
    # No full answer expressions: the README names methods and snippets but never prints a finished call.
    for leaked in ('ct["dose_msv"] >=', '"size"', '"count"', '"nunique"', '"median"', 'transform("sum")',
                   "include_groups=", "fill_value=0", "index=False"):
        assert leaked not in readme, leaked
    for text in (readme, checks_md):
        assert "\u2014" not in text and "\u2013" not in text, "no em or en dashes"
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
    """The same answers written the way another careful student might: {artifact path: file contents}.

    Built with observed=False, so the summaries also carry the zero-exam rows.
    """
    frames = solve(observed=False)

    summary = frames[SUMMARY]
    summary = summary[list(reversed(summary.columns))].copy()
    summary["site"] = summary["site"].astype(str).str.upper()
    summary["mean_scan_min"] = summary["mean_scan_min"].round(1)

    pairs = frames[PAIRS].iloc[::-1].copy()
    pairs[["median_wait_hr", "median_report_hr"]] = pairs[["median_wait_hr", "median_report_hr"]].round(1)

    delayed = frames[DELAYED][list(reversed(DELAYED_COLUMNS))].round(2)
    delayed.columns = [column.title() for column in delayed.columns]

    # The flat route without apply keeps the same exams, with their original row labels as the index.
    stat = EXAMS[EXAMS["priority"] == "STAT"]
    slowest = stat.sort_values("report_hr", ascending=False).groupby("site").head(3)[["modality", "report_hr", "site",
                                                                                       "exam_id"]]

    pivot = frames[PIVOT][["X-ray", "Ultrasound", "MRI", "CT"]].round(1).reset_index().rename(columns={"site": "SITE"})

    return {
        # Columns reversed, upper-case site names, one decimal, CRLF, and UTF-16, as Windows PowerShell 5.1 writes.
        SUMMARY: codecs.BOM_UTF16_LE + summary.to_csv(index=False, lineterminator="\r\n").encode("utf-16-le"),
        # A leading row-number column, a BOM, every cell quoted, NaN for the empty medians, rows reversed.
        PAIRS: "﻿" + pairs.to_csv(quoting=1, na_rep="NaN"),
        # Title-case headers, columns reversed, two decimals, a trailing space on each line, no final newline.
        DELAYED: "\n".join(line + " " for line in delayed.to_csv(index=False).splitlines()),
        # Rows sorted by patient, a space after every comma.
        DOSE: frames[DOSE].sort_values(["patient_id", "exam_id"]).to_csv(index=False).replace(",", ", "),
        SLOWEST: slowest.to_csv(),
        # reset_index() saved with its row numbers, modalities reversed, one decimal, an all-empty Eastport row.
        PIVOT: pivot.to_csv(),
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
    frames = solve(use_filter=True)
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

        def only(name: str, files: dict[str, str | bytes], expected: set[str], *parts: str, check: str = "") -> dict:
            """Grade `files` and insist exactly `expected` fail; each of `parts` must appear in `check`'s message."""
            result = graded(submission(name, files), both_clis=False)
            assert failing(result) == expected, (name, {other: detail(result, other) for other in failing(result)})
            assert result["score"] == 100 - cost(*expected), (name, result["score"])
            assert check or len(expected) == 1 or not parts, name
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
        shutil.copytree(HANDOUT, scaffold, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv", ".checks"))
        result = graded(scaffold)
        assert result["score"] == 0 and failing(result) == set(NAMES)
        assert detail(result, NAMES[0]).startswith("output/site_summary.csv is missing"), detail(result, NAMES[0])

        root = submission("correct", correct)
        result = graded(root)
        assert result["score"] == 100, {name: detail(result, name) for name in failing(result)}
        size = sum((root / path).stat().st_size for path in correct)
        assert size < 100_000, f"a complete output/ is {size} bytes"

        # The checks read only output/: poisoned data and code in the submission change nothing.
        (root / "data").mkdir()
        (root / "data" / "imaging_exams.parquet").write_text("poison\n", encoding="utf-8")
        for name in ("grading.py", "_value_checks.py", "check_assignment.py"):
            (root / name).write_text("raise RuntimeError('submission code must not run')\n", encoding="utf-8")
        assert graded(root)["score"] == 100

        for label, files in (("plain-strings", as_files(solve(categorical=False))),
                             ("observed-false", as_files(solve(observed=False))),
                             ("alternative", alternative())):
            result = graded(submission(label, files))
            assert result["score"] == 100, (label, {name: detail(result, name) for name in failing(result)})

        # Saved with the row index everywhere, as to_csv() does without index=False, and with apply's MultiIndex.
        root = workspace / "with-index"
        (root / "output").mkdir(parents=True)
        for path, frame in frames.items():
            frame.to_csv(root / path)
        stat = EXAMS[EXAMS["priority"] == "STAT"]
        stat.groupby("site").apply(lambda g: g.nlargest(3, "report_hr"), include_groups=False)[
            ["exam_id", "modality", "report_hr"]].to_csv(root / SLOWEST)
        assert graded(root)["score"] == 100

        # Cells separated by semicolons, with a European spreadsheet's decimal commas, or by tabs: full marks.
        def separated(sep: str, decimal: str = ".") -> dict[str, str]:
            return {path: frame.to_csv(index=path == PIVOT, sep=sep, decimal=decimal) for path, frame in frames.items()}

        semicolons = separated(";", ",")
        assert any(re.search(r";-?\d+,\d", text) for text in semicolons.values())
        for label, files in (("semicolons", semicolons), ("tabs", separated("\t"))):
            result = graded(submission(label, files))
            assert result["score"] == 100, (label, {name: detail(result, name) for name in failing(result)})
        wrong = frames[SUMMARY].copy()
        wrong.loc[wrong["site"] == "Valley", "mean_scan_min"] = 15.5
        files = separated(";", ",")
        files[SUMMARY] = wrong.to_csv(index=False, sep=";", decimal=",")
        only("semicolons-one-wrong-value", files, {"site summary: mean_scan_min values"}, "Valley has mean_scan_min 15.5")

        # Summaries grouped without as_index=False keep their keys when saved with the index: full marks.
        exams = EXAMS
        index_files = dict(correct)
        index_files[SUMMARY] = exams.groupby("site").agg(
            exam_count=("exam_id", "size"), report_count=("report_hr", "count"), patient_count=("patient_id", "nunique"),
            total_scan_min=("scan_min", "sum"), mean_scan_min=("scan_min", "mean")).to_csv()
        index_files[PAIRS] = exams.groupby(["site", "modality"]).agg(
            exam_count=("exam_id", "size"), median_wait_hr=("wait_hr", "median"),
            median_report_hr=("report_hr", "median")).to_csv()
        assert graded(submission("index-keys", index_files))["score"] == 100
        # Saved with index=False they lose their key columns: only the columns checks fail.
        lost = dict(correct)
        lost[SUMMARY] = pd.read_csv(io.StringIO(index_files[SUMMARY]), index_col=0).to_csv(index=False)
        lost[PAIRS] = pd.read_csv(io.StringIO(index_files[PAIRS]), index_col=[0, 1]).to_csv(index=False)
        lost[PIVOT] = frames[PIVOT].to_csv(index=False)
        only("lost-keys", lost, {"site summary: columns", "site and modality: columns", "median report pivot: columns"},
             "is missing site", check="site summary: columns")

        # An omitted value column costs only its columns check; two renamed with the right values the same.
        only("dropped-value-column", changed(SUMMARY, frames[SUMMARY].drop(columns="patient_count").to_csv(index=False)),
             {"site summary: columns"}, "is missing patient_count")
        renamed = frames[SUMMARY].rename(columns={"exam_count": "exams", "patient_count": "patients"})
        only("renamed-two-right", changed(SUMMARY, renamed.to_csv(index=False)), {"site summary: columns"},
             "is missing exam_count and patient_count and also has exams and patients")
        wrong_renamed = renamed.copy()
        wrong_renamed.loc[wrong_renamed["site"] == "Valley", ["exams", "patients"]] = [7, 6]
        only("renamed-two-wrong", changed(SUMMARY, wrong_renamed.to_csv(index=False)),
             {"site summary: columns", "site summary: exam_count values", "site summary: patient_count values"},
             "no saved column holds exam_count", "expected exam_count 129876 for Downtown",
             check="site summary: exam_count values")
        only("key-only", changed(SUMMARY, frames[SUMMARY][["site"]].to_csv(index=False)), {"site summary: columns"},
             "is missing exam_count, report_count, patient_count, total_scan_min and mean_scan_min")
        only("pivot-dropna-columns", changed(PIVOT, frames[PIVOT].dropna(axis=1).to_csv()),
             {"median report pivot: columns"}, "is missing MRI")
        only("delayed-without-site", changed(DELAYED, frames[DELAYED].drop(columns="site").to_csv(index=False)),
             {"delayed STAT exams: columns"}, "is missing site")

        # Plausible mistakes, each charged once with a message that names the cause. A selection that shares no row
        # with the expected one leaves the values check nothing to compare, so it costs that check too.
        stat = EXAMS[EXAMS["priority"] == "STAT"].copy()
        all_priorities = EXAMS.groupby(["site", "modality"])["wait_hr"].transform("median")
        stat["peer_median_wait_hr"] = all_priorities[stat.index]
        stat["wait_vs_peer_hr"] = stat["wait_hr"] - stat["peer_median_wait_hr"]
        only("peer-over-every-priority", changed(DELAYED, stat[stat["wait_hr"] > 8][DELAYED_COLUMNS].to_csv(index=False)),
             {"delayed STAT exams: peer_median_wait_hr values"}, "over all the STAT exams")
        stat["peer_median_wait_hr"] = stat.groupby(["site", "modality"])["wait_hr"].transform("median")
        stat["wait_vs_peer_hr"] = stat["wait_hr"] - stat["peer_median_wait_hr"]
        at_least = stat[stat["wait_hr"] >= 8][DELAYED_COLUMNS]
        assert len(at_least) > 190
        only("delayed-at-least-8", changed(DELAYED, at_least.to_csv(index=False)),
             {"delayed STAT exams: one row per delayed exam"}, "also has a row for", '"> 8"')
        late = stat[stat["wait_hr"] > 8].copy()
        late["peer_median_wait_hr"] = late.groupby(["site", "modality"])["wait_hr"].transform("median")
        late["wait_vs_peer_hr"] = late["wait_hr"] - late["peer_median_wait_hr"]
        only("peer-after-selecting", changed(DELAYED, late[DELAYED_COLUMNS].to_csv(index=False)),
             {"delayed STAT exams: peer_median_wait_hr values"}, "before keeping the ones over 8 hours")
        assigned = stat.copy()
        assigned["peer_median_wait_hr"] = stat.groupby("site")["wait_hr"].median()
        assigned["wait_vs_peer_hr"] = assigned["wait_hr"] - assigned["peer_median_wait_hr"]
        only("peer-from-median-summary", changed(DELAYED, assigned[assigned["wait_hr"] > 8][DELAYED_COLUMNS].to_csv(index=False)),
             {"delayed STAT exams: peer_median_wait_hr values"}, "Every peer_median_wait_hr is blank", "transform")
        renamed_pair = late[DELAYED_COLUMNS].rename(columns={"peer_median_wait_hr": "typical", "wait_vs_peer_hr": "gap"})
        only("peer-wrong-and-renamed", changed(DELAYED, renamed_pair.to_csv(index=False)),
             {"delayed STAT exams: columns", "delayed STAT exams: peer_median_wait_hr values"})

        every_modality = EXAMS[EXAMS.groupby("patient_id")["dose_msv"].transform("sum") >= 100]
        only("dose-over-every-modality", changed(DOSE, every_modality[["exam_id", "patient_id", "dose_msv"]].to_csv(index=False)),
             {"CT dose review: whole patients kept"}, "also has a row for", "grouping every modality")
        ct = EXAMS[EXAMS["modality"] == "CT"]
        result = graded(submission("dose-row-mask", changed(DOSE, ct[ct["dose_msv"] >= 100][
            ["exam_id", "patient_id", "dose_msv"]].to_csv(index=False))), both_clis=False)
        assert failing(result) == {"CT dose review: whole patients kept", "CT dose review: original exam values"}
        assert "A row mask" in detail(result, "CT dose review: whole patients kept")
        only("dose-all-columns", changed(DOSE, EXAMS.loc[frames[DOSE].index].to_csv(index=False)),
             {"CT dose review: columns"}, "also has")

        every_priority = EXAMS.groupby("site").apply(lambda g: g.nlargest(3, "report_hr"), include_groups=False)
        only("slowest-every-priority", changed(SLOWEST, every_priority.reset_index(level="site")[
            ["exam_id", "site", "modality", "report_hr"]].to_csv(index=False)),
            {"slowest STAT reports: three per site", "slowest STAT reports: original exam values"}, "keep the STAT exams",
            check="slowest STAT reports: three per site")
        smallest = stat.groupby("site").apply(lambda g: g.nsmallest(3, "report_hr"), include_groups=False)
        only("slowest-nsmallest", changed(SLOWEST, smallest.reset_index(level="site")[
            ["exam_id", "site", "modality", "report_hr"]].to_csv(index=False)),
            {"slowest STAT reports: three per site", "slowest STAT reports: original exam values"}, "largest report_hr",
            check="slowest STAT reports: three per site")

        only("pivot-fill-0", changed(PIVOT, pd.pivot_table(EXAMS, values="report_hr", index="site", columns="modality",
                                                         aggfunc="median", fill_value=0).to_csv()),
             {"median report pivot: empty cell stays empty"}, "Valley has MRI 0", "expected blank", "filling it with 0")
        only("pivot-mean", changed(PIVOT, pd.pivot_table(EXAMS, values="report_hr", index="site", columns="modality",
                                                       aggfunc="mean").to_csv()),
             {"median report pivot: median reports"}, 'aggfunc="median"')
        only("pivot-swapped", changed(PIVOT, pd.pivot_table(EXAMS, values="report_hr", index="modality", columns="site",
                                                          aggfunc="median").to_csv()),
             {"median report pivot: columns"}, "swapped", 'index="site" and columns="modality"')
        margins = pd.pivot_table(EXAMS, values="report_hr", index="site", columns="modality", aggfunc="median",
                                 margins=True, margins_name="Total")
        only("pivot-margins", changed(PIVOT, margins.to_csv()), {"median report pivot: columns"},
             "Total comes from margins=True", "also has Total;")
        only("pivot-margins-row-only", changed(PIVOT, margins.drop(columns="Total").to_csv()),
             {"median report pivot: one row per site"}, "Total comes from margins=True")
        for label, arguments in (("values", {"values": ["report_hr"], "aggfunc": "median"}),
                                 ("aggfunc", {"values": "report_hr", "aggfunc": ["median"]})):
            only(f"pivot-{label}-list", changed(PIVOT, pd.pivot_table(EXAMS, index="site", columns="modality",
                                                                      **arguments).to_csv()),
                 {"median report pivot: columns"}, "list for values= or aggfunc= adds a header level",
                 'values="report_hr", aggfunc="median"')
        only("pivot-mean-list", changed(PIVOT, pd.pivot_table(EXAMS, values="report_hr", index="site",
                                                            columns="modality", aggfunc=["mean"]).to_csv()),
             {"median report pivot: columns", "median report pivot: median reports"})
        only("pivot-dropna", changed(PIVOT, frames[PIVOT].dropna().to_csv()), {"median report pivot: one row per site"},
             "is missing Valley")

        # A mistake copied from the long table into the pivot is charged at its source.
        pairs, pivot = correct[PAIRS], correct[PIVOT]
        files = changed(PAIRS, edit(pairs, "Valley|CT", "median_report_hr", "9.9"))
        files[PIVOT] = edit(pivot, "Valley", "CT", "9.9")
        only("copied-pair-error", files, {"site and modality: median_report_hr values"})

        # A zero-exam row must show zero exams; a file holding only that optional row earns no value checks.
        summary = correct[SUMMARY]
        only("blank-eastport", changed(SUMMARY, summary + "Eastport,,,,,\n"),
             {"site summary: exam_count values", "site summary: report_count values",
              "site summary: patient_count values", "site summary: total_scan_min values"},
             "Eastport has exam_count blank, expected 0", check="site summary: exam_count values")
        only("eastport-only", changed(SUMMARY, summary.splitlines()[0] + "\nEastport,0,0,0,0,\n"),
             set(artifact_checks(SUMMARY)) - {"site summary: columns"})

        # Repeated headers: an identical duplicate is fine, a conflicting one costs its values check.
        lines = summary.splitlines()
        files = changed(SUMMARY, "\n".join(line + "," + line.split(",")[1] for line in lines))
        assert graded(submission("duplicate-header", files), both_clis=False)["score"] == 100
        files[SUMMARY] = files[SUMMARY].replace("Valley,25384,", "Valley,99,")
        only("conflicting-duplicate", files, {"site summary: exam_count values"})
        only("nul-byte-file", changed(SUMMARY, b"\xff\xfe\x00"), set(artifact_checks(SUMMARY)))

        # One mistake costs exactly its own check.
        delayed, dose, slowest = correct[DELAYED], correct[DOSE], correct[SLOWEST]
        first_dose = dose.splitlines()[1].split(",")[0]
        mistakes = {
            "site summary: columns": (SUMMARY, summary.replace("patient_count", "patients", 1)),
            "site summary: one row per site": (SUMMARY, drop_last_row(summary)),
            "site summary: exam_count values": (SUMMARY, edit(summary, "Valley", "exam_count", "25383")),
            "site summary: report_count values": (SUMMARY, edit(summary, "Valley", "report_count", "25384")),
            "site summary: patient_count values": (SUMMARY, edit(summary, "Valley", "patient_count", "25384")),
            "site summary: total_scan_min values": (SUMMARY, edit(summary, "Valley", "total_scan_min", "359151")),
            "site summary: mean_scan_min values": (SUMMARY, edit(summary, "Valley", "mean_scan_min", "14.5")),
            "site and modality: columns": (PAIRS, pairs.replace("modality", "type", 1)),
            "site and modality: one row per pair": (PAIRS, drop_last_row(pairs)),
            "site and modality: exam_count values": (PAIRS, edit(pairs, "Valley|X-ray", "exam_count", "11984")),
            "site and modality: median_wait_hr values": (PAIRS, edit(pairs, "Valley|X-ray", "median_wait_hr", "13.0")),
            "site and modality: median_report_hr values": (PAIRS, edit(pairs, "Valley|X-ray", "median_report_hr", "5.5")),
            "delayed STAT exams: columns": (DELAYED, delayed.replace("wait_vs_peer_hr", "wait_diff", 1)),
            "delayed STAT exams: one row per delayed exam": (DELAYED, drop_last_row(delayed)),
            "delayed STAT exams: original exam values": (DELAYED, edit(delayed, "E0000544", "modality", "CT")),
            "delayed STAT exams: peer_median_wait_hr values": (
                DELAYED, edit(delayed, "E0000544", "peer_median_wait_hr", "3.0")),
            "delayed STAT exams: wait_vs_peer_hr values": (DELAYED, edit(delayed, "E0000544", "wait_vs_peer_hr", "-15.7")),
            "CT dose review: columns": (DOSE, dose.replace("dose_msv", "dose", 1)),
            "CT dose review: whole patients kept": (DOSE, drop_last_row(dose)),
            "CT dose review: original exam values": (DOSE, edit(dose, first_dose, "dose_msv", "50")),
            "slowest STAT reports: columns": (SLOWEST, slowest.replace("modality", "type", 1)),
            "slowest STAT reports: three per site": (SLOWEST, drop_last_row(slowest)),
            "slowest STAT reports: original exam values": (SLOWEST, edit(slowest, "E0313443", "report_hr", "1.0")),
            "median report pivot: columns": (PIVOT, pivot.replace("X-ray", "Xray", 1)),
            "median report pivot: one row per site": (PIVOT, pivot + "All,1.5,12.0,5.5,3.0\n"),
            "median report pivot: median reports": (PIVOT, edit(pivot, "Valley", "CT", "2.5")),
            "median report pivot: empty cell stays empty": (PIVOT, edit(pivot, "Valley", "MRI", "0.0")),
        }
        assert sorted(mistakes) == sorted(NAMES)
        for number, (name, (path, text)) in enumerate(mistakes.items()):
            only(f"mistake-{number:02}", changed(path, text), {name})
        message = detail(graded(submission("feedback-sign", changed(*mistakes["delayed STAT exams: wait_vs_peer_hr values"])),
                                both_clis=False), "delayed STAT exams: wait_vs_peer_hr values")
        assert "E0000544 has wait_vs_peer_hr -15.7, expected 15.7" in message and "minus" in message, message

        # A missing file costs only its own artifact's checks, and is never told to take another required file's name.
        for number, path in enumerate(correct):
            files = {other: text for other, text in correct.items() if other != path}
            result = graded(submission(f"missing-{number}", files), both_clis=False)
            assert failing(result) == set(artifact_checks(path)), (path, failing(result))
            assert not any("Found" in detail(result, name) for name in failing(result)), path
        files = dict(correct)
        files["output/site_modalities_summary.csv"] = files.pop(PAIRS)
        message = detail(graded(submission("typo-name", files), both_clis=False), "site and modality: columns")
        assert message.endswith("Found output/site_modalities_summary.csv; save it as "
                                "output/site_modality_summary.csv instead."), message

        # pytest, which GitHub runs, reports one test per check.
        for name, expected_failures in (("correct", 0), ("scaffold", len(NAMES))):
            target = workspace / f"pytest-{name}"
            shutil.copytree(HANDOUT, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv", ".checks"))
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
        "Assignment 08 checks: the data, the frozen values, README checkpoints, contract, and numbering, and the "
        "handout's byte-identical checks agree; empty and scaffold earn 0; correct, plain-string, observed=False, "
        "alternative, with-index, index-key, semicolon- and tab-separated submissions earn 100; each of the "
        f"{len(NAMES)} single mistakes, each plausible mistake, and each missing file costs only its own checks; "
        "feedback and pytest pass."
    )


if __name__ == "__main__":
    run()
