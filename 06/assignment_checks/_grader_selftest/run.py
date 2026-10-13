"""Development self-test for the Assignment 06 (midterm) checks.

Course-side QA, not a second grading mode. It answers the midterm with pandas
from `06/assignment/data/` the way the README asks, writes submissions in
ignored `scratch/`, and confirms what each one scores:

    uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 \
        python 06/assignment_checks/_grader_selftest/run.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
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
import check_assignment  # noqa: E402
import grading  # noqa: E402

DATA = HANDOUT / "data"
README = HANDOUT / "README.md"
CHECKS_MD = HANDOUT / "CHECKS.md"
EXACT_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
TABLES = ("patients", "sites", "visits_north", "visits_south", "a1c_wide")
VISIT_COLUMNS = ["patient_id", "visit", "visit_date", "sbp_mmhg", "weight_kg", "source_file"]
ANALYSIS_COLUMNS = ["patient_id", "visit", "visit_date", "sbp_mmhg", "weight_kg", "a1c_pct", "sex", "birth_year",
                    "site_code", "site_name", "source_file"]
HEADERS = {
    "file_inventory.txt": "<count> data/a1c_wide.csv",
    "table_profile.csv": "table,rows,columns,exact_duplicate_rows",
    "issue_audit.csv": "issue,count",
    "patients_clean.csv": "patient_id,sex,birth_year,site_code",
    "visits_clean.csv": ",".join(VISIT_COLUMNS),
    "a1c_long.csv": "patient_id,visit,a1c_pct",
    "merge_audit.csv": "check,count",
    "analysis_visits.csv": ",".join(ANALYSIS_COLUMNS),
    "sbp_wide.csv": "patient_id,baseline,week12,sbp_change",
    "decision_log.csv": "table,decision,reason",
}


def readme_block_labels(header: str) -> list[str]:
    """The labels, in order, from the README's text block that starts with `header`."""
    block = re.search(rf"```text\n{re.escape(header)}\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert block, f"the README shows no block starting `{header}`"
    return [line.rsplit(",", 1)[0] for line in block.group(1).splitlines()]


def readme_decisions() -> list[tuple[str, str]]:
    rows = re.findall(r"^\| `([a-z]+)` \| `([^`]+)` \|$", README.read_text(encoding="utf-8"), re.M)
    return [row for row in rows if row[0] in ("patients", "visits", "sites")]


def readme_checklist() -> dict[str, tuple[str, int]]:
    rows = re.findall(r"^\| `output/([^`]+)` \| [^|]+ \| `([^`]+)` \| (\d+) \|$",
                      README.read_text(encoding="utf-8"), re.M)
    return {name: (header, int(lines)) for name, header, lines in rows}


def checks_contract() -> dict[str, int]:
    points = {}
    for line in CHECKS_MD.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and (name := re.fullmatch(r"`output/([^`]+)`", cells[0])) and cells[2].isdigit():
            points[name.group(1)] = int(cells[2])
    return points


def checks_review_points() -> int:
    table = re.search(r"^\| Category \| What it reads \| Full credit \((\d+)\) \|.*\n\|[- |]+\|\n((?:\|.*\n)+)",
                      CHECKS_MD.read_text(encoding="utf-8"), re.M)
    assert table, "CHECKS.md shows no human-review rubric table"
    return int(table.group(1)) * len(table.group(2).splitlines())


def solve() -> dict[str, str]:
    """Answer the midterm with lecture methods; return each output file's text."""
    raw = {name: pd.read_csv(DATA / f"{name}.csv", dtype="string", keep_default_na=False) for name in TABLES}
    files = {"file_inventory.txt": subprocess.run("wc -l data/*.csv", shell=True, cwd=HANDOUT, check=True,
                                                  capture_output=True, text=True).stdout}

    profile = pd.DataFrame([{"table": f"{name}.csv", "rows": len(table), "columns": table.shape[1],
                             "exact_duplicate_rows": int(table.duplicated().sum())} for name, table in raw.items()])
    files["table_profile.csv"] = profile.to_csv(index=False)

    north = raw["visits_north"].copy()
    south = raw["visits_south"].rename(columns={"wt_kg": "weight_kg"})
    north["source_file"] = "visits_north.csv"
    south["source_file"] = "visits_south.csv"
    visits_raw = pd.concat([north, south], ignore_index=True)
    assert visits_raw.shape == (48, 7) and not visits_raw.eq("").all().any(), visits_raw.shape

    patients, sites = raw["patients"], raw["sites"]
    sex = patients["sex"].str.strip()
    labels = visits_raw["visit"].str.strip().str.lower()
    sbp_raw = pd.to_numeric(visits_raw["sbp_mmhg"], errors="coerce")
    dates = visits_raw["visit_date"].str.strip()
    parsed = pd.to_datetime(dates.where(dates.str.fullmatch(EXACT_DATE)), format="%Y-%m-%d", errors="coerce")
    audit = {
        "patients: sex values needing normalization": (sex.ne("") & patients["sex"].ne(sex.str.upper())).sum(),
        "patients: empty sex": sex.eq("").sum(),
        "patients: birth_year sentinel 9999": patients["birth_year"].str.strip().eq("9999").sum(),
        "patients: site codes needing normalization":
            patients["site_code"].ne(patients["site_code"].str.strip().str.upper()).sum(),
        "sites: rows with a repeated site_code": sites.duplicated(subset=["site_code"], keep=False).sum(),
        "visits: visit labels needing normalization": visits_raw["visit"].ne(labels).sum(),
        "visits: sbp sentinel 999": visits_raw["sbp_mmhg"].str.strip().eq("999").sum(),
        "visits: sbp outside 70 through 250": (sbp_raw.ne(999) & ~sbp_raw.between(70, 250)).sum(),
        "visits: date parse failures": (dates.ne("") & parsed.isna()).sum(),
        "visits: rows in repeated patient_id and visit pairs":
            visits_raw.assign(visit=labels).duplicated(subset=["patient_id", "visit"], keep=False).sum(),
    }
    audit = {issue: int(count) for issue, count in audit.items()}
    assert readme_block_labels("issue,count") == list(audit), "README issue labels differ from the self-test's"
    files["issue_audit.csv"] = pd.DataFrame({"issue": list(audit), "count": list(audit.values())}).to_csv(index=False)

    pc = patients.copy(deep=True).drop_duplicates(keep="first")
    pc["sex"] = pc["sex"].str.strip().str.upper().replace("", pd.NA)
    pc["birth_year"] = pd.to_numeric(pc["birth_year"].replace("9999", pd.NA)).astype("Int64")
    pc["site_code"] = pc["site_code"].str.strip().str.upper()
    pc = pc.reset_index(drop=True)
    files["patients_clean.csv"] = pc.to_csv(index=False)

    v = visits_raw.copy(deep=True).drop_duplicates(keep="first")
    v["visit"] = v["visit"].str.strip().str.lower()
    sbp = pd.to_numeric(v["sbp_mmhg"], errors="coerce")
    v["sbp_mmhg"] = sbp.where(sbp.between(70, 250)).astype("Int64")
    v["weight_kg"] = pd.to_numeric(v["weight_kg"], errors="coerce").astype("float64")
    day = v["visit_date"].str.strip()
    v["visit_date"] = pd.to_datetime(day.where(day.str.fullmatch(EXACT_DATE)), format="%Y-%m-%d", errors="coerce")
    repeated = v[v.duplicated(subset=["patient_id", "visit"], keep=False)]
    assert len(repeated) == 2 and repeated["patient_id"].nunique() == 1, repeated
    v = v.sort_values("visit_time").drop_duplicates(subset=["patient_id", "visit"], keep="last")
    v = v.sort_values(["patient_id", "visit"]).reset_index(drop=True)[VISIT_COLUMNS]
    files["visits_clean.csv"] = v.to_csv(index=False)

    a1c = raw["a1c_wide"].melt(id_vars=["patient_id"], value_vars=["a1c_baseline", "a1c_week12"],
                               var_name="visit", value_name="a1c_pct")
    a1c["visit"] = a1c["visit"].map({"a1c_baseline": "baseline", "a1c_week12": "week12"})
    a1c["a1c_pct"] = pd.to_numeric(a1c["a1c_pct"], errors="coerce").astype("float64")
    a1c = a1c.dropna(subset=["a1c_pct"]).reset_index(drop=True)
    files["a1c_long.csv"] = a1c.to_csv(index=False)

    try:
        pd.merge(pc, sites, on="site_code", how="left", validate="many_to_one")
        raise AssertionError("the site list should break a many-to-one merge")
    except pd.errors.MergeError:
        pass
    current = sites[sites["record_status"] == "current"][["site_code", "site_name"]]
    matched = pd.merge(v, a1c, on=["patient_id", "visit"], how="outer", indicator=True)["_merge"].value_counts()
    merge_audit = {
        "visits with an A1c result": matched["both"],
        "visits without an A1c result": matched["left_only"],
        "A1c results without a visit": matched["right_only"],
        "visits whose patient is not in the registry": (~v["patient_id"].isin(pc["patient_id"])).sum(),
        "registry patients with no visit": (~pc["patient_id"].isin(v["patient_id"])).sum(),
        "registry patients whose site is not in the site list": (~pc["site_code"].isin(sites["site_code"])).sum(),
        "site list rows removed by the current-record rule": len(sites) - len(current),
    }
    merge_audit = {name: int(count) for name, count in merge_audit.items()}
    assert readme_block_labels("check,count") == list(merge_audit), "README merge labels differ from the self-test's"
    files["merge_audit.csv"] = pd.DataFrame({"check": list(merge_audit),
                                             "count": list(merge_audit.values())}).to_csv(index=False)

    analysis = (v.merge(pc, on="patient_id", how="left", validate="many_to_one")
                .merge(current, on="site_code", how="left", validate="many_to_one")
                .merge(a1c, on=["patient_id", "visit"], how="left", validate="one_to_one"))[ANALYSIS_COLUMNS]
    assert len(analysis) == len(v)
    files["analysis_visits.csv"] = analysis.to_csv(index=False)

    wide = v.pivot(index="patient_id", columns="visit", values="sbp_mmhg").reset_index()
    wide.columns.name = None
    wide["sbp_change"] = wide["week12"] - wide["baseline"]
    files["sbp_wide.csv"] = wide.to_csv(index=False)

    decisions = readme_decisions()
    files["decision_log.csv"] = pd.DataFrame({
        "table": [table for table, _ in decisions], "decision": [decision for _, decision in decisions],
        "reason": [f"Documented rule {number} for these synthetic clinic files." for number in range(1, 9)],
    }).to_csv(index=False)

    # The grader's constants are the independent answer.
    for name, table in raw.items():
        assert grading.TABLE_PROFILE[f"{name}.csv"] == (len(table), table.shape[1], int(table.duplicated().sum()))
        assert grading.INVENTORY[f"{name}.csv"] == len(table) + 1
    assert tuple(audit.items()) == grading.ISSUE_AUDIT, audit
    assert tuple(merge_audit.items()) == grading.MERGE_AUDIT, merge_audit
    assert tuple(decisions) == grading.DECISIONS, decisions

    def value(cell: object) -> object:
        return None if pd.isna(cell) else cell

    assert {row.patient_id: {"sex": value(row.sex), "birth_year": value(row.birth_year), "site_code": row.site_code}
            for row in pc.itertuples()} == grading.PATIENTS
    assert {(row.patient_id, row.visit): {
        "visit_date": None if pd.isna(row.visit_date) else row.visit_date.date(), "sbp_mmhg": value(row.sbp_mmhg),
        "weight_kg": value(row.weight_kg), "source_file": row.source_file.split("_")[1].split(".")[0]}
        for row in v.itertuples()} == grading.VISITS
    assert {(row.patient_id, row.visit): row.a1c_pct for row in a1c.itertuples()} == grading.A1C
    assert dict(zip(current["site_code"], current["site_name"])) == grading.CURRENT_SITES
    assert {(row.patient_id, row.visit): {"a1c_pct": value(row.a1c_pct), "sex": value(row.sex),
                                          "birth_year": value(row.birth_year), "site_code": value(row.site_code),
                                          "site_name": value(row.site_name)}
            for row in analysis.itertuples()} == grading.ANALYSIS
    assert {row.patient_id: {"baseline": value(row.baseline), "week12": value(row.week12)}
            for row in wide.itertuples()} == grading.SBP_WIDE
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert all(hashlib.sha256((DATA / name).read_bytes()).hexdigest() == entry["sha256"]
               for name, entry in manifest["files"].items()), "data/manifest.json does not match the data files"
    return files


def write(root: Path, files: dict[str, str | bytes]) -> Path:
    (root / "output").mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = root / "output" / name
        if content is None:
            path.unlink(missing_ok=True)
        elif isinstance(content, bytes):
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


def table(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), dtype="string", keep_default_na=False)


def edit(text: str, change) -> str:
    return change(table(text)).to_csv(index=False)


def at(frame: pd.DataFrame, where: dict[str, str], column: str, new: str) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for key, wanted in where.items():
        mask &= frame[key].eq(wanted)
    assert mask.sum() == 1, (where, mask.sum())
    frame.loc[mask, column] = new
    return frame


def loose(files: dict[str, str]) -> dict[str, str | bytes]:
    """The same answers written every way the README allows or whitespace changes."""
    crlf = lambda text: text.replace("\n", "\r\n")  # noqa: E731
    variant: dict[str, str | bytes] = {
        "file_inventory.txt": "\n".join(line for line in files["file_inventory.txt"].splitlines()
                                        if "total" not in line).replace("data/", "./data/"),
        "table_profile.csv": "﻿" + edit(files["table_profile.csv"], lambda t: t.assign(
            table=t["table"].str.replace(".csv", "", regex=False).str.upper(), rows=t["rows"] + ".0")
            .iloc[::-1, ::-1]),
        "issue_audit.csv": crlf(files["issue_audit.csv"].replace(",", ";")),
        "merge_audit.csv": table(files["merge_audit.csv"]).set_index("check")["count"].to_csv(),
        "decision_log.csv": edit(files["decision_log.csv"], lambda t: t.assign(
            decision=t["decision"].str.upper() + ".", notes="extra column").iloc[::-1]),
    }
    patients = table(files["patients_clean.csv"])
    patients["patient_id"] = patients["patient_id"].str.lower()
    patients["sex"] = patients["sex"].replace("", "<NA>")
    patients["birth_year"] = patients["birth_year"].where(patients["birth_year"].eq(""), patients["birth_year"] + ".0")
    variant["patients_clean.csv"] = crlf(patients.iloc[::-1, ::-1].to_csv()).replace("\r\n", "  \r\n")
    visits = table(files["visits_clean.csv"])
    visits["visit_date"] = visits["visit_date"].where(visits["visit_date"].eq(""), visits["visit_date"] + " 00:00:00")
    visits["sbp_mmhg"] = visits["sbp_mmhg"].replace("", "NaN")
    visits["source_file"] = "data/" + visits["source_file"]
    variant["visits_clean.csv"] = visits.iloc[::-1].to_csv(index=False).replace(",", ", ")
    # melt() without dropna keeps rows for the blank results; they hold nothing to check.
    long = table(files["a1c_long.csv"])
    variant["a1c_long.csv"] = pd.concat([long, pd.DataFrame({"patient_id": ["P018", "P021"],
                                                             "visit": ["week12", "week12"],
                                                             "a1c_pct": ["", ""]})]).to_csv(index=False)
    analysis = table(files["analysis_visits.csv"])
    analysis["visit_date"] = analysis["visit_date"].str.replace("-", "/")
    variant["analysis_visits.csv"] = analysis[list(reversed(ANALYSIS_COLUMNS))].to_csv(sep="\t")
    variant["sbp_wide.csv"] = files["sbp_wide.csv"].replace("\n", "\r").rstrip("\r")
    return variant


def run() -> None:
    files = solve()

    readme = README.read_text(encoding="utf-8")
    checks_md = CHECKS_MD.read_text(encoding="utf-8")
    checklist = readme_checklist()
    assert set(checklist) == set(files), ("README checklist rows", sorted(checklist))
    tree = readme.split("## Task 1")[0]
    for name, text in files.items():
        header, count = checklist[name]
        assert header == HEADERS[name], (name, header)
        assert len(text.splitlines()) == count, (name, count, len(text.splitlines()))
        assert re.search(rf"── {re.escape(name)} +# you make in Task", tree), (name, "missing from the file tree")
    assert re.fullmatch(r"\s*\d+ data/a1c_wide\.csv", files["file_inventory.txt"].splitlines()[0])
    graded: dict[str, int] = {}
    for check in grading.CHECKS:
        file_name = check.name.split(":")[0]
        graded[file_name] = graded.get(file_name, 0) + check.points
    assert checks_contract() == graded, ("CHECKS.md Completion contract points", checks_contract(), graded)
    review = checks_review_points()
    assert grading.MAX_SCORE + review == 100 and check_assignment.HUMAN_REVIEW_POINTS == review, review
    assert (f"{grading.MAX_SCORE} points graded from your committed files after the deadline, "
            f"{review} by human review") in checks_md, "CHECKS.md states a different point split"
    shown = set(HEADERS.values()) | {f"{table},{decision}" for table, decision in grading.DECISIONS}
    leaks = [line for name in files if name not in ("decision_log.csv", "file_inventory.txt")
             for line in files[name].splitlines() if line not in shown and line in readme + checks_md]
    assert not leaks, ("the README publishes expected values", leaks)
    for phrase in ("check_assignment", "GitHub Actions", "final newline", "grading.py", "automated feedback"):
        assert phrase not in readme + checks_md, ("the exam README mentions", phrase)

    shipped = [name for name in ("check_assignment.py", "grading.py", "test_assignment.py", "_grader_selftest",
                                 ".github", ".badmath.toml") if (HANDOUT / name).exists()]
    assert not shipped, ("the exam handout ships automated checks", shipped)
    assert "data/*.csv -text" in (HANDOUT / ".gitattributes").read_text(encoding="utf-8")
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    assert all(not cell.get("outputs") and cell.get("execution_count") is None
               for cell in notebook["cells"] if cell["cell_type"] == "code"), "clear the handout notebook's outputs"
    assert sorted(path.name for path in (HANDOUT / "output").iterdir()) == [".gitkeep"], "stray handout outputs"
    for path in [CHECKS / "grading.py", CHECKS / "check_assignment.py", *HANDOUT.rglob("*.py")]:
        assert "version_info" not in path.read_text(encoding="utf-8"), (path, "reads the Python version")

    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a06-selftest-") as temporary:
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

        partial = write(work / "partial", {name: files[name] for name in ("file_inventory.txt", "table_profile.csv")})
        result, _ = checker(partial)
        assert result["score"] == 10, losses(result)

        # No submitted file runs: every one of these would leave a marker if imported or executed.
        trapped = write(work / "trapped", files)
        marker = work / "ran"
        trap = f"open({str(marker)!r}, 'w').write(__file__)\n"
        for name in ("grading.py", "check_assignment.py", "json.py", "csv.py", "re.py", "sitecustomize.py",
                     "output/grading.py"):
            (trapped / name).write_text(trap, encoding="utf-8")
        result, _ = checker(trapped)
        assert result["score"] == 75 and not marker.exists(), "a submitted file ran"

        visits_csv = files["visits_clean.csv"]
        wide_csv = files["sbp_wide.csv"]
        analysis_csv = files["analysis_visits.csv"]
        p015 = {"patient_id": "P015", "visit": "baseline"}
        long = table(files["a1c_long.csv"])
        unmapped = long.assign(visit="a1c_" + long["visit"]).to_csv(index=False)
        no_lab = edit(analysis_csv, lambda t: t.assign(a1c_pct=""))
        site_rows = table(analysis_csv)
        retired = site_rows[site_rows["site_code"].eq("S02")].assign(site_name="Southgate Clinic")

        p002 = {"patient_id": "P002", "visit": "baseline"}
        p002_wide = {"patient_id": "P002"}

        mutations = {
            "sex left in lower case": (
                {"patients_clean.csv": edit(files["patients_clean.csv"], lambda t: t.assign(sex=t["sex"].str.lower()))},
                {"patients_clean.csv: sex": 2}, ("strip and uppercase",)),
            "P002's Baseline label left unnormalized, carried into the merges and the pivot": (
                {"visits_clean.csv": edit(visits_csv, lambda t: at(t, p002, "visit", "Baseline")),
                 "analysis_visits.csv": edit(analysis_csv, lambda t: at(at(t, p002, "a1c_pct", ""), p002, "visit",
                                                                         "Baseline")),
                 "merge_audit.csv": edit(files["merge_audit.csv"], lambda t: t.assign(
                     count=t["count"].mask(t.index == 0, "42").mask(t.index == 1, "4").mask(t.index == 2, "4"))),
                 "sbp_wide.csv": edit(wide_csv, lambda t: at(at(at(t.assign(Baseline=""), p002_wide, "Baseline",
                                                                   "161"), p002_wide, "baseline", ""),
                                                             p002_wide, "sbp_change", ""))},
                {"visits_clean.csv: rows": 1}, ("P002", "'Baseline'", ".str.strip().str.lower()")),
            "P009's exact repeat kept": (
                {"patients_clean.csv": edit(files["patients_clean.csv"], lambda t: pd.concat([t, t.iloc[[8]]]))},
                {"patients_clean.csv: rows": 1}, ("P009", "drop_duplicates")),
            "birth_year 9999 kept": (
                {"patients_clean.csv": edit(files["patients_clean.csv"],
                                            lambda t: at(t, {"patient_id": "P007"}, "birth_year", "9999"))},
                {"patients_clean.csv: birth_year": 1}, ("P007", "9999")),
            "P015's earlier reading kept, carried into sbp_wide": (
                {"visits_clean.csv": edit(visits_csv, lambda t: at(t, p015, "sbp_mmhg", "168")),
                 "sbp_wide.csv": edit(wide_csv, lambda t: at(at(t, {"patient_id": "P015"}, "baseline", "168"),
                                                             {"patient_id": "P015"}, "sbp_change", "-17"))},
                {"visits_clean.csv: sbp_mmhg": 1}, ("P015", "visit_time")),
            "both P015 baseline readings kept": (
                {"visits_clean.csv": edit(visits_csv, lambda t: pd.concat(
                    [t, at(t[t["patient_id"].eq("P015") & t["visit"].eq("baseline")].copy(), p015, "sbp_mmhg",
                           "168")]))},
                {"visits_clean.csv: rows": 1}, ("P015 baseline",)),
            "wt_kg never renamed, so south weights sit in their own column": (
                {"visits_clean.csv": edit(visits_csv, lambda t: t.assign(
                    wt_kg=t["weight_kg"].where(t["source_file"].str.contains("south"), ""),
                    weight_kg=t["weight_kg"].where(t["source_file"].str.contains("north"), "")))},
                {"visits_clean.csv: weight_kg": 2}, ("weight_kg", "wt_kg")),
            "wt_kg kept in place of weight_kg": (
                {"visits_clean.csv": edit(visits_csv, lambda t: t.rename(columns={"weight_kg": "wt_kg"}))},
                {"visits_clean.csv: weight_kg": 2}, ("rename wt_kg",)),
            "2026-5-10 accepted as a date": (
                {"visits_clean.csv": edit(visits_csv, lambda t: at(t, {"patient_id": "P010", "visit": "week12"},
                                                                   "visit_date", "2026-05-10"))},
                {"visits_clean.csv: visit_date": 1}, ("P010", "exact")),
            "sbp 999 kept": (
                {"visits_clean.csv": edit(visits_csv, lambda t: at(
                    at(t, {"patient_id": "P006", "visit": "baseline"}, "sbp_mmhg", "999"),
                    {"patient_id": "P017", "visit": "week12"}, "sbp_mmhg", "999"))},
                {"visits_clean.csv: sbp_mmhg": 1}, ("P006", "999")),
            "source_file left out": (
                {"visits_clean.csv": edit(visits_csv, lambda t: t.drop(columns="source_file"))},
                {"visits_clean.csv: source_file": 2}, ("source_file",)),
            "visits_clean.csv not committed": (
                {"visits_clean.csv": None},
                {name: points for name, points in (
                    ("visits_clean.csv: rows", 4), ("visits_clean.csv: visit_date", 3),
                    ("visits_clean.csv: sbp_mmhg", 3), ("visits_clean.csv: weight_kg", 2),
                    ("visits_clean.csv: source_file", 2))},
                ("missing", "Task 3.2")),
            "A1c labels never mapped, carried into the join and the audit": (
                {"a1c_long.csv": unmapped, "analysis_visits.csv": no_lab,
                 "merge_audit.csv": edit(files["merge_audit.csv"], lambda t: t.assign(
                     count=t["count"].mask(t.index == 0, "0").mask(t.index == 1, "46").mask(t.index == 2, "46")))},
                {"a1c_long.csv: rows": 2}, ("map the labels",)),
            "the retired S02 name joined too": (
                {"analysis_visits.csv": pd.concat([site_rows, retired]).to_csv(index=False)},
                {"analysis_visits.csv: rows": 2}, ("S02",)),
            "the retired S02 name joined in place of the current one": (
                {"analysis_visits.csv": edit(analysis_csv, lambda t: t.assign(site_name=t["site_name"].mask(
                    t["site_code"].eq("S02"), "Southgate Clinic")))},
                {"analysis_visits.csv: site_name": 2}, ("current",)),
            "A1c merged onto the wrong visit": (
                {"analysis_visits.csv": edit(analysis_csv, lambda t: at(
                    t, {"patient_id": "P001", "visit": "week12"}, "a1c_pct", "7.7"))},
                {"analysis_visits.csv: a1c_pct": 1}, ("P001 week12",)),
            "n01 left unnormalized, so P004 finds no site": (
                {"patients_clean.csv": edit(files["patients_clean.csv"],
                                            lambda t: at(t, {"patient_id": "P004"}, "site_code", "n01")),
                 "analysis_visits.csv": edit(analysis_csv, lambda t: at(
                     at(at(at(t, {"patient_id": "P004", "visit": "baseline"}, "site_name", ""),
                           {"patient_id": "P004", "visit": "week12"}, "site_name", ""),
                        {"patient_id": "P004", "visit": "baseline"}, "site_code", "n01"),
                     {"patient_id": "P004", "visit": "week12"}, "site_code", "n01"))},
                {"patients_clean.csv: site_code": 1}, ("P004", "'n01'")),
            "sbp_change written baseline minus week12": (
                {"sbp_wide.csv": edit(wide_csv, lambda t: t.assign(sbp_change=t["sbp_change"].str.replace(
                    r"^(-?)(\d+)$", lambda m: ("" if m.group(1) else "-") + m.group(2), regex=True)))},
                {"sbp_wide.csv: sbp_change": 2}, ("week12 minus baseline",)),
            "wc counts written without the header lines": (
                {"file_inventory.txt": re.sub(r"(\d+) data", lambda m: f"{int(m.group(1)) - 1} data",
                                              files["file_inventory.txt"])},
                {"file_inventory.txt": 5}, ("without the header",)),
            "one audit count off by one": (
                {"issue_audit.csv": edit(files["issue_audit.csv"], lambda t: t.assign(count=t["count"].mask(
                    t["issue"].str.contains("date parse"), "1")))},
                {"issue_audit.csv": 1}, ("date parse failures", "fullmatch")),
            "one merge count off": (
                {"merge_audit.csv": edit(files["merge_audit.csv"], lambda t: t.assign(count=t["count"].mask(
                    t["check"].str.contains("no visit"), "0")))},
                {"merge_audit.csv": 1}, ("registry patients with no visit",)),
            "one reason left blank": (
                {"decision_log.csv": edit(files["decision_log.csv"],
                                          lambda t: t.assign(reason=t["reason"].mask(t.index == 3, "")))},
                {"decision_log.csv: reason": 1}, ("row 4",)),
            "one decision rewritten": (
                {"decision_log.csv": edit(files["decision_log.csv"], lambda t: t.assign(
                    decision=t["decision"].mask(t.index == 7, "drop every site")))},
                {"decision_log.csv: decisions": 1}, ("current site records",)),
            "a profile that counts the header as a row": (
                {"table_profile.csv": edit(files["table_profile.csv"],
                                           lambda t: t.assign(rows=(t["rows"].astype(int) + 1).astype(str)))},
                {"table_profile.csv": 5}, ("header is not a row",)),
            "an audit file with embedded NUL": (
                {"issue_audit.csv": files["issue_audit.csv"] + "\x00"},
                {"issue_audit.csv": 10}, ("embedded NUL",)),
        }
        for description, (changes, expected_losses, mentions) in mutations.items():
            root = write(work / re.sub(r"\W+", "-", description), {**files, **changes})
            result, code = checker(root)
            assert losses(result) == expected_losses, (description, losses(result))
            assert result["score"] == 75 - sum(expected_losses.values()), description
            assert code == (0 if not expected_losses else 1), (description, code)
            for name in expected_losses:
                text = detail(result, name)
                assert "Fix: Task" in text and all(word in text for word in mentions), (description, text)

    print("Assignment 06 checks: constants match an independent pandas solution; the README lists every "
          "artifact with its header, line count, and points, states the 75/25 split, and publishes no expected "
          "value; the handout ships no checks; empty and handout score 0, a partial submission 10; complete and "
          f"loosely formatted submissions score 75; no submitted file runs; {len(mutations)} single variations "
          "each cost exactly their own points.")


if __name__ == "__main__":
    run()
