"""Development self-test for the Assignment 05 (midterm) checks.

Course-side QA, not a second grading mode. It answers the midterm with pandas
and NumPy from `05/assignment/data/people_raw.csv` the way the README asks,
writes submissions in ignored `scratch/`, and confirms what each one scores:

    uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 \
        python 05/assignment_checks/_grader_selftest/run.py
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


CHECKS = Path(__file__).resolve().parents[1]
HANDOUT = CHECKS.parent / "assignment"
SCRATCH = CHECKS.parents[1] / "scratch"
sys.path.insert(0, str(CHECKS))
import check_assignment  # noqa: E402
import grading  # noqa: E402

DATA = HANDOUT / "data" / "people_raw.csv"
FIXTURE = HANDOUT / "data" / "fixture.json"
README = HANDOUT / "README.md"
EXACT_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
AGE_SENTINELS = ["unknown", "-9"]
STATUS_SENTINEL = "NA"
SITES = ["north", "south", "west"]
STATUSES = ["active", "pending", "complete"]
RAW_COLUMNS = ["record_id", "full_name", "site", "status", "age_text", "visit_date"]
HEADERS = {
    "raw_preview.txt": "$ head -n 4 data/people_raw.csv",
    "pipeline_summary.txt": "raw_rows=<number>",
    "numpy_age_summary.csv": "metric,value",
    "pandas_selection.csv": "record_id,site,status",
    "issue_audit.csv": "issue,count",
    "cleaned_people.csv": ",".join(grading.CLEANED_COLUMNS),
    "decision_log.csv": ",".join(grading.DECISION_COLUMNS),
}


def readme_audit_labels() -> list[str]:
    """The issue labels, in order, from the README's issue_audit.csv block."""
    block = re.search(r"```text\nissue,count\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert block, "the README shows no issue_audit.csv block starting `issue,count`"
    return [line.rsplit(",", 1)[0] for line in block.group(1).splitlines()]


def readme_decisions() -> list[tuple[str, str, str]]:
    """The field, issue, and action of each decision in the README's table."""
    rows = re.findall(r"^\| `([^`]+)` \| `([^`]+)` \| `([^`]+)` \|$", README.read_text(encoding="utf-8"), re.M)
    return [row for row in rows if row[0] != "field"]


def readme_checklist() -> dict[str, tuple[str, int]]:
    """Each output file's first line and line count from the README's checklist table."""
    rows = re.findall(r"^\| `output/([^`]+)` \| [^|]+ \| `([^`]+)` \| (\d+) \|$",
                      README.read_text(encoding="utf-8"), re.M)
    return {name: (header, int(lines)) for name, header, lines in rows}


def readme_contract() -> dict[str, int]:
    """Each output file's points from the README's Completion contract table."""
    points = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and (name := re.fullmatch(r"`output/([^`]+)`", cells[0])) and cells[2].isdigit():
            points[name.group(1)] = int(cells[2])
    return points


def readme_review_points() -> int:
    """The human-review points in the README's rubric: each category's full credit times the categories."""
    table = re.search(r"^\| Category \| What it reads \| Full credit \((\d+)\) \|.*\n\|[- |]+\|\n((?:\|.*\n)+)",
                      README.read_text(encoding="utf-8"), re.M)
    assert table, "the README shows no human-review rubric table"
    return int(table.group(1)) * len(table.group(2).splitlines())


def solve() -> dict[str, str]:
    """Answer the midterm with lecture methods; return each output file's text."""
    raw = pd.read_csv(DATA, dtype="string", keep_default_na=False)
    source_lines = DATA.read_text(encoding="utf-8").splitlines()
    files = {"raw_preview.txt": "\n".join([
        "$ head -n 4 data/people_raw.csv", *source_lines[:4],
        "$ tail -n 2 data/people_raw.csv", *source_lines[-2:],
    ]) + "\n"}

    summary = {
        "raw_rows": len(raw),
        "raw_columns": raw.shape[1],
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "candidate_id_duplicate_rows": int(raw.duplicated(subset=["record_id"]).sum()),
        "clean_rows": len(raw.drop_duplicates()),
    }
    files["pipeline_summary.txt"] = "".join(f"{key}={value}\n" for key, value in summary.items())

    ages = pd.to_numeric(raw["age_text"], errors="coerce")
    whole = ages.mod(1).eq(0) & ages.between(0, 120)
    valid = np.array(ages[whole.fillna(False)].astype(int))
    files["numpy_age_summary.csv"] = pd.DataFrame({
        "metric": ["count", "min", "max", "sum", "mean"],
        "value": [valid.size, valid.min(), valid.max(), valid.sum(), valid.mean()],
    }).to_csv(index=False)

    files["pandas_selection.csv"] = (
        raw.set_index("record_id").loc[["R001", "R003", "R010"], ["site", "status"]].to_csv())

    age_text = raw["age_text"].str.strip()
    sentinel_age = age_text.isin(AGE_SENTINELS)
    numbers = pd.to_numeric(age_text.where(~sentinel_age), errors="coerce")
    dates = raw["visit_date"].str.strip()
    parsed_dates = pd.to_datetime(dates.where(dates.str.fullmatch(EXACT_DATE)), format="%Y-%m-%d",
                                  errors="coerce")
    status = raw["status"].str.strip()
    sites = raw["site"].str.strip().str.lower()
    statuses = status.str.lower().where(status.ne(STATUS_SENTINEL))
    counts = {
        "schema mismatch": len(set(RAW_COLUMNS) ^ set(raw.columns)),
        "empty full-name tokens": int(raw["full_name"].str.strip().eq("").sum()),
        "empty date tokens": int(dates.eq("").sum()),
        "age sentinel tokens": int(sentinel_age.sum()),
        "status sentinel tokens": int(status.eq(STATUS_SENTINEL).sum()),
        "age parse failures": int((numbers.isna() & ~sentinel_age & age_text.ne("")).sum()),
        "numeric but noninteger age values": int(numbers.mod(1).ne(0).fillna(False).sum()),
        "age values outside 0 through 120": int((numbers.mod(1).eq(0) & ~numbers.between(0, 120))
                                                .fillna(False).sum()),
        "date parse failures": int((dates.ne("") & parsed_dates.isna()).sum()),
        "rows in exact duplicate sets": int(raw.duplicated(keep=False).sum()),
        "rows with repeated candidate IDs": int(raw.duplicated(subset=["record_id"], keep=False).sum()),
        "site values needing format normalization": int(raw["site"].ne(sites).sum()),
        "status values needing format normalization": int(
            (raw["status"].ne(raw["status"].str.strip().str.lower()) & status.ne(STATUS_SENTINEL)).sum()),
        "unexpected site values": int((~sites.isin(SITES)).sum()),
        "unexpected non-sentinel status values": int((statuses.notna() & ~statuses.isin(STATUSES)).sum()),
    }
    labels = readme_audit_labels()
    assert labels == list(counts), ("README issue labels differ from the audit the self-test computes", labels)
    files["issue_audit.csv"] = pd.DataFrame({"issue": list(counts), "count": list(counts.values())}).to_csv(index=False)

    clean = raw.drop_duplicates(keep="first").copy(deep=True)
    name = clean["full_name"].str.strip().str.title()
    clean["full_name"] = name.where(name.ne(""))
    clean["site"] = clean["site"].str.strip().str.lower()
    clean_status = clean["status"].str.strip()
    clean["status"] = clean_status.str.lower().where(clean_status.ne(STATUS_SENTINEL))
    clean_age_text = clean["age_text"].str.strip()
    clean_numbers = pd.to_numeric(clean_age_text.where(~clean_age_text.isin(AGE_SENTINELS)), errors="coerce")
    keep = (clean_numbers.mod(1).eq(0) & clean_numbers.between(0, 120)).fillna(False)
    clean["age"] = clean_numbers.where(keep).astype("Int64")
    clean_dates = clean["visit_date"].str.strip()
    clean["visit_date"] = pd.to_datetime(clean_dates.where(clean_dates.str.fullmatch(EXACT_DATE)),
                                         format="%Y-%m-%d", errors="coerce")
    clean["needs_review"] = (clean["age"].isna() | clean["visit_date"].isna()).astype("boolean")
    clean = clean[list(grading.CLEANED_COLUMNS)].reset_index(drop=True)
    files["cleaned_people.csv"] = clean.to_csv(index=False)

    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    decisions = readme_decisions()
    files["decision_log.csv"] = pd.DataFrame({
        "field": [field for field, _, _ in decisions],
        "issue": [issue for _, issue, _ in decisions],
        "action": [action for _, _, action in decisions],
        "reason": [f"Documented rule {number} for this synthetic intake file." for number in range(1, 9)],
        "source": "data/people_raw.csv",
        "source_sha256": sha,
        "rows_before": len(raw),
        "rows_after": len(clean),
    }).to_csv(index=False)

    expected = {record_id: values for record_id, values in grading.CLEANED.items()}
    solved = {}
    for row in clean.itertuples(index=False):
        solved[row.record_id] = {
            "full_name": None if pd.isna(row.full_name) else row.full_name,
            "site": row.site,
            "status": None if pd.isna(row.status) else row.status,
            "age": None if pd.isna(row.age) else int(row.age),
            "visit_date": None if pd.isna(row.visit_date) else row.visit_date.date(),
            "needs_review": bool(row.needs_review),
        }
    assert solved == expected, "grading.CLEANED differs from the cleaned table the self-test computes"
    assert summary == grading.PIPELINE_SUMMARY, summary
    assert dict(zip(["count", "min", "max", "sum", "mean"],
                    [valid.size, valid.min(), valid.max(), valid.sum(), valid.mean()])) == grading.NUMPY_AGE_SUMMARY
    assert tuple(counts.items()) == grading.ISSUE_AUDIT, counts
    assert tuple(decisions) == grading.DECISIONS, decisions
    selection = raw.set_index("record_id").loc[list(grading.PANDAS_SELECTION), ["site", "status"]]
    assert {key: tuple(value) for key, value in selection.iterrows()} == grading.PANDAS_SELECTION
    assert tuple(source_lines[:4]) == grading.PREVIEW_LINES["head"]
    assert tuple(source_lines[-2:]) == grading.PREVIEW_LINES["tail"]
    assert sha == grading.SOURCE_SHA256 == json.loads(FIXTURE.read_text(encoding="utf-8"))["sha256"]
    assert (len(raw), len(clean)) == (grading.RAW_ROWS, grading.CLEAN_ROWS)
    return files


def write(root: Path, files: dict[str, str | bytes]) -> Path:
    (root / "output").mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = root / "output" / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8", newline="")
    return root


def checker(root: Path) -> tuple[dict, int]:
    """Grade through check_assignment.py, from inside the submission, as scripts/grade_submissions.py does."""
    finished = subprocess.run(
        [sys.executable, "-B", "-E", "-s", str(CHECKS / "check_assignment.py"), str(root), "--json"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert finished.stdout, finished.stderr
    return json.loads(finished.stdout), finished.returncode


def losses(result: dict) -> dict[str, int]:
    return {test["test-name"]: test["max-score"] - test["score"]
            for test in result["tests"] if test["score"] != test["max-score"]}


def detail(result: dict, name: str) -> str:
    return next(test["detail"] for test in result["tests"] if test["test-name"] == name)


def edit_csv(text: str, change) -> str:
    table = pd.read_csv(io.StringIO(text), dtype="string", keep_default_na=False)
    return change(table).to_csv(index=False)


def loose(files: dict[str, str]) -> dict[str, str | bytes]:
    """The same answers written every way the README allows or whitespace changes."""
    crlf = lambda text: text.replace("\n", "\r\n")  # noqa: E731
    variant: dict[str, str | bytes] = {}
    head, tail = files["raw_preview.txt"].split("$ tail")
    variant["raw_preview.txt"] = crlf(head.replace("$ head -n 4", "$  head -n 4").replace("\n", "   \n")
                                      + "\n$ tail" + tail).rstrip("\r\n")
    variant["pipeline_summary.txt"] = crlf("\n".join(
        f"{key.upper()} : {value}.0" if index % 2 else f"{key} = {value}"
        for index, (key, value) in enumerate(reversed(list(grading.PIPELINE_SUMMARY.items())))))
    numpy_rows = pd.read_csv(io.StringIO(files["numpy_age_summary.csv"]))
    numpy_rows["metric"] = numpy_rows["metric"].str.upper()
    variant["numpy_age_summary.csv"] = "﻿" + numpy_rows.iloc[::-1][["value", "metric"]].to_csv()
    variant["pandas_selection.csv"] = edit_csv(files["pandas_selection.csv"], lambda table: table.assign(
        site=table["site"].str.strip().str.lower(), status=table["status"].str.upper())
        .iloc[::-1][["status", "record_id", "site"]])
    variant["issue_audit.csv"] = crlf(edit_csv(files["issue_audit.csv"], lambda table: table.assign(
        issue=table["issue"].str.upper().str.replace("-", " "), count=table["count"] + ".0").iloc[::-1]))
    cleaned = pd.read_csv(io.StringIO(files["cleaned_people.csv"]), dtype="string",
                          keep_default_na=False)
    cleaned["record_id"] = cleaned["record_id"].str.lower()
    cleaned["age"] = cleaned["age"].replace("", "NaN").where(cleaned["age"].eq(""), cleaned["age"] + ".0")
    cleaned["visit_date"] = cleaned["visit_date"].where(cleaned["visit_date"].eq(""),
                                                        cleaned["visit_date"] + " 00:00:00")
    cleaned["needs_review"] = cleaned["needs_review"].map({"True": "yes", "False": "no"})
    cleaned["full_name"] = cleaned["full_name"].replace("", "<NA>")
    variant["cleaned_people.csv"] = crlf(cleaned.iloc[::-1, ::-1].to_csv()).replace("\r\n", "  \r\n")
    variant["decision_log.csv"] = edit_csv(files["decision_log.csv"], lambda table: table.assign(
        field=table["field"].str.upper().str.replace(", ", ","), action=table["action"] + ".",
        source="./data/people_raw.csv", source_sha256=table["source_sha256"].str.upper(),
        rows_before="12.0", notes="extra column").iloc[::-1])
    return variant


def run() -> None:
    files = solve()

    readme = README.read_text(encoding="utf-8")
    checklist = readme_checklist()
    assert set(checklist) == set(files), ("README checklist rows", sorted(checklist))
    for name, text in files.items():
        header, count = checklist[name]
        assert header == HEADERS[name], (name, header)
        assert len(text.splitlines()) == count, (name, count, len(text.splitlines()))
        tree = readme.split("## The data")[0]
        assert re.search(rf"── {re.escape(name)} +# you make in Task", tree), (name, "missing from the file tree")
    graded: dict[str, int] = {}
    for check in grading.CHECKS:
        file_name = check.name.split(":")[0]
        graded[file_name] = graded.get(file_name, 0) + check.points
    assert readme_contract() == graded, ("README Completion contract points", readme_contract(), graded)
    review = readme_review_points()
    assert grading.MAX_SCORE + review == 100 and check_assignment.HUMAN_REVIEW_POINTS == review, review
    assert (f"{grading.MAX_SCORE} points graded from your committed files after the deadline, "
            f"{review} by human review") in readme, "the README states a different point split"
    # Every line of every answer file except the labels and headers the README has to show.
    shown = {HEADERS[name] for name in files} | {"$ tail -n 2 data/people_raw.csv", ",".join(RAW_COLUMNS)}
    leaks = [line for name in files if name != "decision_log.csv"
             for line in files[name].splitlines() if line not in shown and line in readme]
    assert not leaks, ("the README publishes expected values", leaks)
    for phrase in ("check_assignment", "GitHub Actions", "final newline", "grading.py", "automated feedback"):
        assert phrase not in readme, ("the exam README mentions", phrase)

    shipped = [name for name in ("check_assignment.py", "grading.py", "test_assignment.py", "_grader_selftest",
                                 ".github", ".badmath.toml") if (HANDOUT / name).exists()]
    assert not shipped, ("the exam handout ships automated checks", shipped)
    assert "data/people_raw.csv -text" in (HANDOUT / ".gitattributes").read_text(encoding="utf-8")
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    assert all(not cell.get("outputs") and cell.get("execution_count") is None
               for cell in notebook["cells"] if cell["cell_type"] == "code"), "clear the handout notebook's outputs"
    assert sorted(path.name for path in (HANDOUT / "output").iterdir()) == [".gitkeep"], "stray handout outputs"

    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a05-selftest-") as temporary:
        work = Path(temporary)
        empty = work / "empty"
        empty.mkdir()
        for root in (empty, HANDOUT):
            result, code = checker(root)
            assert (result["score"], result["max-score"], code) == (0, 75, 1), (root, result["score"], code)
            assert all(test["detail"] for test in result["tests"]), "every failing check says why"

        complete = write(work / "complete", files)
        result, code = checker(complete)
        assert (result["score"], code) == (75, 0), losses(result)
        assert grading.grade_submission(complete) == result

        result, _ = checker(write(work / "loose", loose(files)))
        assert result["score"] == 75, [(name, detail(result, name)) for name in losses(result)]

        # No submitted file runs: every one of these would leave a marker if imported or executed.
        trapped = write(work / "trapped", files)
        marker = work / "ran"
        trap = f"open({str(marker)!r}, 'w').write(__file__)\n"
        for name in ("grading.py", "check_assignment.py", "json.py", "csv.py", "re.py", "sitecustomize.py",
                     "output/grading.py"):
            (trapped / name).write_text(trap, encoding="utf-8")
        result, _ = checker(trapped)
        assert result["score"] == 75 and not marker.exists(), "a submitted file ran"

        def cleaned(change) -> str:
            return edit_csv(files["cleaned_people.csv"], change)

        def decisions(change) -> str:
            return edit_csv(files["decision_log.csv"], change)

        def at(table: pd.DataFrame, record_id: str, column: str, value: str) -> pd.DataFrame:
            table.loc[table["record_id"].eq(record_id), column] = value
            return table

        mutations = {
            "R011's age of 0 written blank": (
                {"cleaned_people.csv": cleaned(lambda t: at(t, "R011", "age", ""))},
                {"cleaned_people.csv: age": 1}, ("R011", "age")),
            "R007's 40.5 rounded to 40 and flagged from it": (
                {"cleaned_people.csv": cleaned(lambda t: at(at(t, "R007", "age", "40"), "R007", "needs_review",
                                                            "False"))},
                {"cleaned_people.csv: age": 1}, ("R007",)),
            "R009's 2026-7-01 accepted as a date": (
                {"cleaned_people.csv": cleaned(lambda t: at(at(t, "R009", "visit_date", "2026-07-01"), "R009",
                                                            "needs_review", "False"))},
                {"cleaned_people.csv: visit_date": 1}, ("R009", "visit_date")),
            "the NA status sentinel lowercased instead of converted": (
                {"cleaned_people.csv": cleaned(lambda t: at(t, "R004", "status", "na"))},
                {"cleaned_people.csv: status": 1}, ("R004", "'na'")),
            "sites stripped but not lowercased": (
                {"cleaned_people.csv": cleaned(lambda t: at(at(at(t, "R001", "site", "North"), "R003", "site",
                                                                "SOUTH"), "R010", "site", "West"))},
                {"cleaned_people.csv: site": 2}, ("R001", "'North'")),
            "the exact duplicate kept, and counted in rows_after": (
                {"cleaned_people.csv": cleaned(lambda t: pd.concat([t.iloc[:2], t.iloc[1:]])),
                 "decision_log.csv": decisions(lambda t: t.assign(rows_after="12"))},
                {"cleaned_people.csv: record_id": 1}, ("R002",)),
            "needs_review left out": (
                {"cleaned_people.csv": cleaned(lambda t: t.drop(columns="needs_review"))},
                {"cleaned_people.csv: needs_review": 4}, ("needs_review",)),
            "rows with any gap dropped": (
                {"cleaned_people.csv": cleaned(lambda t: t[t.ne("").all(axis=1)]),
                 "decision_log.csv": decisions(lambda t: t.assign(rows_after="4"))},
                {"cleaned_people.csv: record_id": 3, **{f"cleaned_people.csv: {column}": 3
                                                        for column in grading.CLEANED_COLUMNS[1:]}}, ("missing",)),
            "one audit count off by one": (
                {"issue_audit.csv": edit_csv(files["issue_audit.csv"], lambda t: t.assign(
                    count=t["count"].mask(t["issue"].eq("age parse failures"), "2")))},
                {"issue_audit.csv": 1}, ("age parse failures",)),
            "raw_preview.txt not committed": (
                {"raw_preview.txt": None}, {"raw_preview.txt": 4}, ("missing",)),
            "raw_preview.txt without its head label": (
                {"raw_preview.txt": files["raw_preview.txt"].replace("$ head -n 4 data/people_raw.csv\n", "")},
                {"raw_preview.txt": 2}, ("head -n 4",)),
            "`raw_rows = 12` with spaces": (
                {"pipeline_summary.txt": files["pipeline_summary.txt"].replace("raw_rows=", "raw_rows = ")},
                {}, ()),
            "a wrong mean": (
                {"numpy_age_summary.csv": files["numpy_age_summary.csv"].replace("mean,33.0", "mean,39.6")},
                {"numpy_age_summary.csv": 1}, ("mean",)),
            "the whole raw table saved as the selection": (
                {"pandas_selection.csv": pd.read_csv(DATA, dtype="string", keep_default_na=False)
                 .to_csv(index=False)},
                {"pandas_selection.csv": 4}, ("only",)),
            "one decision's action changed": (
                {"decision_log.csv": decisions(lambda t: t.assign(action=t["action"].mask(
                    t["field"].eq("status"), "drop the row")))},
                {"decision_log.csv: decisions": 1}, ("NA sentinel",)),
            "one reason left blank": (
                {"decision_log.csv": decisions(lambda t: t.assign(reason=t["reason"].mask(t.index == 3, "")))},
                {"decision_log.csv: reason": 1}, ("row 4",)),
            "counts written as NumPy scalars, np.int64(12)": (
                {"pipeline_summary.txt": re.sub(r"=(\d+)", r"=np.int64(\1)", files["pipeline_summary.txt"]),
                 "numpy_age_summary.csv": files["numpy_age_summary.csv"].replace("mean,33.0", "mean,np.float64(33.0)")},
                {}, ()),
            "the age summary saved as a Series, its metric names in an unnamed index": (
                {"numpy_age_summary.csv": files["numpy_age_summary.csv"].replace("metric,value", ",value")},
                {}, ()),
            "dates written YYYY/MM/DD": (
                {"cleaned_people.csv": cleaned(lambda t: t.assign(visit_date=t["visit_date"].str.replace("-", "/")))},
                {}, ()),
            "numpy_age_summary.csv with CR-only line endings": (
                {"numpy_age_summary.csv": files["numpy_age_summary.csv"].replace("\n", "\r")},
                {}, ()),
            "a decision_log.csv that no CSV reader accepts": (
                {"decision_log.csv": "field,issue\n" + "x" * 200_000 + "\n"},
                {name: points for name, points in (
                    ("decision_log.csv: decisions", 8), ("decision_log.csv: reason", 2),
                    ("decision_log.csv: source", 1), ("decision_log.csv: source_sha256", 1),
                    ("decision_log.csv: rows_before", 1), ("decision_log.csv: rows_after", 1))},
                ("cannot be read",)),
            "the record_id column left out": (
                {"cleaned_people.csv": cleaned(lambda t: t.drop(columns="record_id"))},
                {"cleaned_people.csv: record_id": 4}, ("record_id",)),
            "cleaned_people.csv written with a space after every comma": (
                {"cleaned_people.csv": files["cleaned_people.csv"].replace(",", ", ")}, {}, ()),
            "cleaned_people.csv with its columns padded to line up": (
                {"cleaned_people.csv": files["cleaned_people.csv"].replace(",", " , ")}, {}, ()),
            "a space after every comma, with R001's name left unstripped": (
                {"cleaned_people.csv": files["cleaned_people.csv"].replace("Alice Smith", " Alice Smith ")
                 .replace(",", ", ")},
                {"cleaned_people.csv: full_name": 1}, ("R001", "found 'Alice Smith '")),
            "the age summary separated by semicolons with a decimal comma, the audit by tabs": (
                {"numpy_age_summary.csv": files["numpy_age_summary.csv"].replace(",", ";").replace("33.0", "33,0"),
                 "issue_audit.csv": files["issue_audit.csv"].replace(",", "\t")},
                {}, ()),
            "a wrong mean in a semicolon-separated age summary": (
                {"numpy_age_summary.csv": files["numpy_age_summary.csv"].replace(",", ";").replace("33.0", "39,6")},
                {"numpy_age_summary.csv": 1}, ("mean", "39.6")),
            "a checksum computed from the wrong file": (
                {"decision_log.csv": decisions(lambda t: t.assign(source_sha256="0" * 64))},
                {"decision_log.csv: source_sha256": 1}, ("sha256",)),
        }
        for description, (changes, expected_losses, mentions) in mutations.items():
            root = write(work / re.sub(r"\W+", "-", description), files)
            for name, content in changes.items():
                if content is None:
                    (root / "output" / name).unlink()
                else:
                    (root / "output" / name).write_text(content, encoding="utf-8")
            result, code = checker(root)
            assert losses(result) == expected_losses, (description, losses(result))
            assert result["score"] == 75 - sum(expected_losses.values()), description
            assert code == (0 if not expected_losses else 1), (description, code)
            for name in expected_losses:
                text = detail(result, name)
                assert "Fix: Task" in text and all(word in text for word in mentions), (description, text)

    print("Assignment 05 checks: constants match an independent pandas solution; the README lists every "
          "artifact with its header, line count, and points, states the 75/25 split, and publishes no "
          "expected value; the handout ships no "
          "checks; empty and handout score 0; complete and loosely formatted submissions score 75; no "
          f"submitted file runs; {len(mutations)} single variations each cost exactly their own points.")


if __name__ == "__main__":
    run()
