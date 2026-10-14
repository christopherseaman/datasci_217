"""Development self-test for the Assignment 06 (midterm) checks.

Course-side QA, not a second grading mode. It generates the data with the
handout's `generate_data.py`, answers the midterm with pandas the way the
README asks (and again with single mistakes carried through every later file,
as a student's own pipeline would carry them), writes submissions in ignored
`scratch/`, and confirms what each one scores:

    uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 \
        python 06/mid-term_checks/_grader_selftest/run.py
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
import time

import pandas as pd


# Self-tests grade with the working copy, never the copy downloaded from GitHub.
os.environ["DS217_LOCAL_CHECKS"] = "1"
sys.dont_write_bytecode = True

CHECKS = Path(__file__).resolve().parents[1]
HANDOUT = CHECKS.parent / "mid-term"
REPO = CHECKS.parents[1]
SCRATCH = REPO / "scratch"
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(HANDOUT))
import check_assignment  # noqa: E402
import generate_data  # noqa: E402
import grading  # noqa: E402

README = HANDOUT / "README.md"
CHECKS_MD = HANDOUT / "CHECKS.md"
EXACT_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
# The spellings in the files, stripped and lowercased, mapped to the data dictionary's codes.
GENDER_MAP = {"f": "F", "female": "F", "woman": "F", "m": "M", "male": "M", "man": "M",
              "x": "X", "nb": "X", "nonbinary": "X", "non-binary": "X"}
VISIT_MAP = {"baseline": "baseline", "bl": "baseline", "week12": "week12", "week 12": "week12", "wk12": "week12"}
FILES = {"patients": "patients.parquet", "sites": "sites.csv", "visits_north": "visits_north.csv",
         "visits_south": "visits_south.csv", "a1c_wide": "a1c_wide.parquet"}
VISIT_COLUMNS = ["patient_id", "visit", "visit_date", "sbp_mmhg", "weight_kg", "source_file"]
ANALYSIS_COLUMNS = ["patient_id", "visit", "visit_date", "sbp_mmhg", "weight_kg", "a1c_pct", "gender", "birth_year",
                    "site_code", "site_name", "source_file"]
PATIENT_METRICS = ["patients", "birth_year missing", "birth_year min", "birth_year max", "site_code values",
                   "patients at N01", "patients at S02", "patients at E04"]
VISIT_METRICS = ["visits", "baseline visits", "week12 visits", "visit_date missing", "visit_date min",
                 "visit_date max", "sbp_mmhg missing", "sbp_mmhg min", "sbp_mmhg max", "weight_kg missing",
                 "weight_kg mean", "visits from visits_north.csv", "visits from visits_south.csv"]
HEADERS = {
    "file_inventory.txt": "<count> data/sites.csv",
    "table_profile.csv": "table,rows,columns,exact_duplicate_rows",
    "issue_audit.csv": "issue,count",
    "patients_summary.csv": "metric,value",
    "gender_categories.csv": "gender,count,code",
    "visits_summary.csv": "metric,value",
    "decision_log.csv": "table,decision,reason",
    "a1c_summary.csv": "visit,results,a1c_pct_mean",
    "merge_audit.csv": "check,count",
    "sbp_wide_sample.csv": "patient_id,baseline,week12,sbp_change",
    "analysis_sample.csv": ",".join(ANALYSIS_COLUMNS),
}
WORK = Path(tempfile.mkdtemp(dir=SCRATCH if SCRATCH.is_dir() else None, prefix="a06-selftest-"))
DATA = WORK / "data"


def generate() -> None:
    """Write the generated data into the self-test folder, as the notebook's first cell does, and time it."""
    started = time.perf_counter()
    built = generate_data.files()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    generate_data.write(DATA)
    seconds = time.perf_counter() - started
    assert seconds < 3, ("generate_data.py takes", seconds)
    assert built == generate_data.files(), "generate_data.py builds different tables on a rerun"
    assert sorted(path.name for path in DATA.iterdir()) == sorted(built) == sorted(FILES.values())
    for name, data in built.items():
        assert b"\r" not in data, (name, "carriage return in the data")
        if name.endswith(".csv"):
            assert (DATA / name).read_bytes() == data, (name, "the CSV file differs from its table")
        else:
            # The notebook's first cell hashes this text, never the Parquet bytes.
            frame = pd.read_parquet(DATA / name)
            assert frame.to_csv(index=False, lineterminator="\n").encode("utf-8") == data, name
            assert {str(dtype) for column, dtype in frame.dtypes.items() if not column.startswith("a1c_")} == \
                {"string"}, (name, frame.dtypes)
    manifest = json.loads((HANDOUT / "data_manifest.json").read_text(encoding="utf-8"))
    assert {name: entry["sha256"] for name, entry in manifest["files"].items()} == grading.DATA_SHA256, \
        "grading.DATA_SHA256 differs from data_manifest.json"
    assert all(hashlib.sha256(built[name]).hexdigest() == entry["sha256"] for name, entry in manifest["files"].items())
    sizes = [len(data) for data in built.values()]
    assert 2e6 <= sum(sizes) <= 8e6, ("the data should total 2 to 8 MB", sizes)


def readme_block_labels(header: str) -> list[str]:
    """The labels, in order, from the README's text block that starts with `header` and holds `<count>` rows."""
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(rf"```text\n{re.escape(header)}\n(.*?)```", text, re.S)
    block = next((found for found in blocks if "<" in found), None)
    assert block, f"the README shows no block starting `{header}` with placeholders"
    return [line.rsplit(",", 1)[0] for line in block.splitlines()]


def readme_decisions() -> list[tuple[str, str]]:
    section = README.read_text(encoding="utf-8").split("### 3.3")[1].split("## Task 4")[0]
    rows = re.findall(r"^\| `([a-z]+)` \| `([^`]+)` \|$", section, re.M)
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


def metrics(names: list[str], values: list) -> str:
    return pd.DataFrame({"metric": names, "value": values}).to_csv(index=False)


def solve(gender_map=GENDER_MAP, keep_patient_repeats=False, keep_9999=False, site_rule="full",
          rename_wt=True, visit_map=VISIT_MAP, recheck_keep="last", exact_dates=True, sbp_range=True,
          melt_map=None, melt_dropna=True, a1c_how="left", site_table="current", reverse_change=False,
          references: dict | None = None) -> dict[str, str]:
    """Answer the midterm with lecture methods, with any one mistake carried through; return each file's text."""
    raw = {name: pd.read_parquet(DATA / file) if file.endswith(".parquet")
           else pd.read_csv(DATA / file, dtype="string", keep_default_na=False) for name, file in FILES.items()}
    files = {"file_inventory.txt": subprocess.run("wc -l data/*.csv", shell=True, cwd=WORK, check=True,
                                                  capture_output=True, text=True).stdout}
    profile = pd.DataFrame([{"table": FILES[name], "rows": len(table), "columns": table.shape[1],
                             "exact_duplicate_rows": int(table.duplicated().sum())} for name, table in raw.items()])
    files["table_profile.csv"] = profile.to_csv(index=False)

    north = raw["visits_north"].copy()
    south = raw["visits_south"].rename(columns={"wt_kg": "weight_kg"}) if rename_wt else raw["visits_south"].copy()
    north["source_file"] = "visits_north.csv"
    south["source_file"] = "visits_south.csv"
    visits_raw = pd.concat([north, south], ignore_index=True)

    patients, sites = raw["patients"], raw["sites"]
    gender = patients["gender"].str.strip()
    site_codes = patients["site_code"].str.upper().str.replace("-", "").str.replace(" ", "")
    sbp_raw = pd.to_numeric(visits_raw["sbp_mmhg"], errors="coerce")
    dates = visits_raw["visit_date"].str.strip()
    parsed = pd.to_datetime(dates.where(dates.str.fullmatch(EXACT_DATE)), format="%Y-%m-%d", errors="coerce")
    labels = visits_raw["visit"].str.strip().str.lower().map(VISIT_MAP)
    audit = {
        "patients: gender values that are not a code":
            (gender.ne("") & ~patients["gender"].isin(["F", "M", "X"])).sum(),
        "patients: empty gender": gender.eq("").sum(),
        "patients: birth_year sentinel 9999": patients["birth_year"].str.strip().eq("9999").sum(),
        "patients: site codes needing normalization": patients["site_code"].ne(site_codes).sum(),
        "sites: rows with a repeated site_code": sites.duplicated(subset=["site_code"], keep=False).sum(),
        "visits: visit labels that are not baseline or week12":
            (~visits_raw["visit"].isin(["baseline", "week12"])).sum(),
        "visits: sbp sentinel 999": visits_raw["sbp_mmhg"].str.strip().eq("999").sum(),
        "visits: sbp outside 70 through 250": (sbp_raw.ne(999) & ~sbp_raw.between(70, 250)).sum(),
        "visits: date parse failures": (dates.ne("") & parsed.isna()).sum(),
        "visits: rows in repeated patient_id and visit pairs":
            visits_raw.assign(visit=labels).duplicated(subset=["patient_id", "visit"], keep=False).sum(),
    }
    audit = {issue: int(count) for issue, count in audit.items()}
    files["issue_audit.csv"] = pd.DataFrame({"issue": list(audit), "count": list(audit.values())}).to_csv(index=False)

    pc = patients.copy(deep=True)
    if not keep_patient_repeats:
        pc = pc.drop_duplicates(keep="first")
    stripped = pc["gender"].str.strip()
    pc["gender"] = (stripped.str.lower().map(gender_map) if gender_map is not None
                    else stripped.replace("", pd.NA)).astype("category")
    pc["birth_year"] = pd.to_numeric(pc["birth_year"] if keep_9999 else pc["birth_year"].replace("9999", pd.NA)
                                     ).astype("Int64")
    pc["site_code"] = (pc["site_code"].str.upper().str.replace("-", "").str.replace(" ", "") if site_rule == "full"
                       else pc["site_code"].str.upper().str.strip())
    pc = pc.reset_index(drop=True)
    year = pc["birth_year"]
    files["patients_summary.csv"] = metrics(PATIENT_METRICS, [
        len(pc), year.isna().sum(), year.min(), year.max(), pc["site_code"].nunique(),
        pc["site_code"].eq("N01").sum(), pc["site_code"].eq("S02").sum(), pc["site_code"].eq("E04").sum()])
    gender_counts = pc["gender"].value_counts(dropna=False).reset_index()
    gender_counts["code"] = gender_counts["gender"].cat.codes
    files["gender_categories.csv"] = gender_counts.to_csv(index=False)

    v = visits_raw.copy(deep=True).drop_duplicates(keep="first")
    v["visit"] = v["visit"].str.strip().str.lower().map(visit_map)
    sbp = pd.to_numeric(v["sbp_mmhg"], errors="coerce")
    v["sbp_mmhg"] = (sbp.where(sbp.between(70, 250)) if sbp_range else sbp).astype("Int64")
    v["weight_kg"] = pd.to_numeric(v["weight_kg"], errors="coerce").astype("float64")
    day = v["visit_date"].str.strip()
    v["visit_date"] = pd.to_datetime(day.where(day.str.fullmatch(EXACT_DATE)) if exact_dates else day,
                                     format="%Y-%m-%d", errors="coerce")
    repeated = v[v.duplicated(subset=["patient_id", "visit"], keep=False)]
    v = v.sort_values("visit_time").drop_duplicates(subset=["patient_id", "visit"], keep=recheck_keep)
    v = v.sort_values(["patient_id", "visit"]).reset_index(drop=True)[VISIT_COLUMNS]
    files["visits_summary.csv"] = metrics(VISIT_METRICS, [
        len(v), v["visit"].eq("baseline").sum(), v["visit"].eq("week12").sum(), v["visit_date"].isna().sum(),
        v["visit_date"].min().date(), v["visit_date"].max().date(), v["sbp_mmhg"].isna().sum(),
        v["sbp_mmhg"].min(), v["sbp_mmhg"].max(), v["weight_kg"].isna().sum(), round(v["weight_kg"].mean(), 2),
        v["source_file"].eq("visits_north.csv").sum(), v["source_file"].eq("visits_south.csv").sum()])

    decisions = readme_decisions()
    files["decision_log.csv"] = pd.DataFrame({
        "table": [table for table, _ in decisions], "decision": [decision for _, decision in decisions],
        "reason": [f"Documented rule {number} for these synthetic clinic files." for number in
                   range(1, len(decisions) + 1)],
    }).to_csv(index=False)

    a1c = raw["a1c_wide"].melt(id_vars=["patient_id"], value_vars=["a1c_baseline", "a1c_week12"],
                               var_name="visit", value_name="a1c_pct")
    if melt_map != "none":
        a1c["visit"] = a1c["visit"].map(melt_map or {"a1c_baseline": "baseline", "a1c_week12": "week12"})
    a1c["a1c_pct"] = pd.to_numeric(a1c["a1c_pct"], errors="coerce").astype("float64")
    if melt_dropna:
        a1c = a1c.dropna(subset=["a1c_pct"])
    a1c = a1c.reset_index(drop=True)
    visit_labels = list(dict.fromkeys(a1c["visit"]))
    files["a1c_summary.csv"] = pd.DataFrame({
        "visit": visit_labels, "results": [int(a1c["visit"].eq(label).sum()) for label in visit_labels],
        "a1c_pct_mean": [round(a1c.loc[a1c["visit"].eq(label), "a1c_pct"].mean(), 2) for label in visit_labels],
    }).to_csv(index=False)

    try:
        pd.merge(pc, sites, on="site_code", how="left", validate="many_to_one")
        raise AssertionError("the site list should break a many-to-one merge")
    except pd.errors.MergeError:
        pass
    current = sites[sites["record_status"] == site_table][["site_code", "site_name"]] if site_table != "all" \
        else sites[["site_code", "site_name"]]
    matched = pd.merge(v, a1c, on=["patient_id", "visit"], how="outer", indicator=True)["_merge"].value_counts()
    merge_audit = {
        "visits with an A1c result": matched["both"],
        "visits without an A1c result": matched["left_only"],
        "A1c results without a visit": matched["right_only"],
        "visits whose patient is not in the registry": (~v["patient_id"].isin(pc["patient_id"])).sum(),
        "registry patients with no visit": (~pc["patient_id"].isin(v["patient_id"])).sum(),
        "registry patients whose site is not in the site list": (~pc["site_code"].isin(sites["site_code"])).sum(),
        "site list rows removed by the current-record rule": len(sites) - len(sites[sites["record_status"] == "current"]),
    }
    merge_audit = {name: int(count) for name, count in merge_audit.items()}
    files["merge_audit.csv"] = pd.DataFrame({"check": list(merge_audit),
                                             "count": list(merge_audit.values())}).to_csv(index=False)

    # A mistake upstream makes validate= raise; the carried-through pipeline joins without it, as a student
    # who removed validate= would, so the later files show the mistake.
    strict = not keep_patient_repeats and site_table == "current"
    analysis = (v.merge(pc, on="patient_id", how="left", validate="many_to_one" if strict else None)
                .merge(current, on="site_code", how="left", validate="many_to_one" if strict else None)
                .merge(a1c, on=["patient_id", "visit"], how=a1c_how,
                       validate="one_to_one" if strict and melt_dropna else None))[ANALYSIS_COLUMNS]
    sample = analysis[analysis["patient_id"].isin(grading.SAMPLE_IDS)].reset_index(drop=True)
    files["analysis_sample.csv"] = sample.to_csv(index=False)

    wide = v.pivot(index="patient_id", columns="visit", values="sbp_mmhg").reset_index()
    wide.columns.name = None
    for column in ("baseline", "week12"):
        if column not in wide:
            wide[column] = pd.NA
    wide["sbp_change"] = (wide["baseline"] - wide["week12"]) if reverse_change else wide["week12"] - wide["baseline"]
    wide = wide[["patient_id", "baseline", "week12", "sbp_change"]]
    files["sbp_wide_sample.csv"] = wide[wide["patient_id"].isin(grading.SAMPLE_IDS)].to_csv(index=False)
    if references is not None:
        references.update(raw=raw, visits_raw=visits_raw, patients=pc, visits=v, a1c=a1c, current=current,
                          analysis=analysis, wide=wide, repeated=repeated, audit=audit, merge_audit=merge_audit,
                          decisions=decisions)
    return files


def confirm_reference(files: dict[str, str], found: dict) -> None:
    """The grader's constants are the independent answer, and the sample covers every planted pattern."""
    raw, pc, v, a1c, analysis, wide = (found[name] for name in ("raw", "patients", "visits", "a1c", "analysis",
                                                                  "wide"))
    for name, frame in raw.items():
        assert grading.TABLE_PROFILE[FILES[name]] == (len(frame), frame.shape[1], int(frame.duplicated().sum()))
        if FILES[name].endswith(".csv"):
            assert grading.INVENTORY[FILES[name]] == len(frame) + 1
    assert tuple(found["audit"].items()) == grading.ISSUE_AUDIT, found["audit"]
    assert tuple(found["merge_audit"].items()) == grading.MERGE_AUDIT, found["merge_audit"]
    assert tuple(found["decisions"]) == grading.DECISIONS, found["decisions"]
    assert readme_block_labels("issue,count") == list(found["audit"]), "README issue labels differ"
    assert readme_block_labels("check,count") == list(found["merge_audit"]), "README merge labels differ"
    assert readme_block_labels("metric,value") in (PATIENT_METRICS, VISIT_METRICS), "README metric labels differ"
    blocks = re.findall(r"```text\nmetric,value\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert [[line.rsplit(",", 1)[0] for line in block.splitlines()] for block in blocks if "<" in block] == [
        PATIENT_METRICS, VISIT_METRICS], "README metric blocks differ from the graded labels"

    def value(cell: object) -> object:
        return None if pd.isna(cell) else cell

    summary = dict(zip(*(table(files["patients_summary.csv"])[c] for c in ("metric", "value"))))
    assert {name: float(cell) for name, cell in summary.items()} == grading.PATIENT_SUMMARY, summary
    counts = pc["gender"].value_counts(dropna=False)
    assert {value(code): int(count) for code, count in counts.items()} == grading.GENDER_COUNTS, counts
    visits = dict(zip(*(table(files["visits_summary.csv"])[c] for c in ("metric", "value"))))
    for name, want in grading.VISIT_SUMMARY.items():
        got = visits[name]
        assert (got == want.isoformat() if hasattr(want, "isoformat") else abs(float(got) - want) < 0.006), name
    for visit, (results, mean) in grading.A1C_SUMMARY.items():
        part = a1c.loc[a1c["visit"].eq(visit), "a1c_pct"]
        assert len(part) == results and abs(part.mean() - mean) < 1e-9, visit
    assert {(row.patient_id, row.visit): row.a1c_pct for row in a1c.itertuples()} == grading.A1C
    assert dict(zip(found["current"]["site_code"], found["current"]["site_name"])) == grading.CURRENT_SITES
    assert {(row.patient_id, row.visit): {"a1c_pct": value(row.a1c_pct), "gender": value(row.gender),
                                          "birth_year": value(row.birth_year), "site_code": value(row.site_code),
                                          "site_name": value(row.site_name)}
            for row in analysis.itertuples()} == grading.ANALYSIS
    assert {row.patient_id: {"baseline": value(row.baseline), "week12": value(row.week12)}
            for row in wide.itertuples()} == grading.SBP_WIDE
    assert len(analysis) == len(v) and not analysis.duplicated(subset=["patient_id", "visit"]).any()

    # The data's planted structures: clinic days only, a diabetes cohort, SBP rows on both limits, rechecks.
    assert v["visit_date"].dropna().dt.weekday.lt(5).all(), "a visit on a weekend"
    assert a1c.loc[a1c["visit"].eq("baseline"), "a1c_pct"].ge(6.5).mean() > 0.85
    assert v["sbp_mmhg"].eq(70).any() and v["sbp_mmhg"].eq(250).any(), "no SBP exactly on a limit"
    pairs = found["repeated"].groupby(["patient_id", "visit"]).size()
    assert len(found["repeated"]) == 2 * len(pairs) and pairs.eq(2).all(), pairs.value_counts()
    extra = raw["patients"][raw["patients"].duplicated()]
    assert (extra["gender"].str.strip().ne("") & ~extra["gender"].isin(["F", "M", "X"])).any(), \
        "no exact patient repeat changes an audit count"

    # The extracts' patients cover every pattern, so the row checks see each rule.
    ids = set(grading.SAMPLE_IDS)
    patients, visits_raw = raw["patients"], found["visits_raw"]
    mine = patients[patients["patient_id"].isin(ids)]
    seen = visits_raw[visits_raw["patient_id"].isin(ids)]
    sbp = pd.to_numeric(seen["sbp_mmhg"], errors="coerce")
    sample_pairs = found["repeated"][found["repeated"]["patient_id"].isin(ids)]
    lab = raw["a1c_wide"][raw["a1c_wide"]["patient_id"].isin(ids)]
    covered = {
        "a repeated registry row": mine.duplicated().any(),
        "birth_year 9999": mine["birth_year"].eq("9999").any(),
        "a hyphenated site code": mine["site_code"].str.contains("-").any(),
        "site E04": mine["site_code"].eq("E04").any(),
        "a gender spelling to map": (~mine["gender"].isin(["F", "M", "X", ""])).any(),
        "a visit with no registry row": (~seen["patient_id"].isin(patients["patient_id"])).any(),
        "a registry patient with no visit": (~mine["patient_id"].isin(visits_raw["patient_id"])).any(),
        "SBP 999": seen["sbp_mmhg"].eq("999").any(),
        "SBP out of range": (sbp.ne(999) & ~sbp.between(70, 250)).any(),
        "SBP 70": sbp.eq(70).any(), "SBP 250": sbp.eq(250).any(),
        "an impossible date": (seen["visit_date"].str.fullmatch(EXACT_DATE)
                               & pd.to_datetime(seen["visit_date"], format="%Y-%m-%d", errors="coerce").isna()).any(),
        "a date not written YYYY-MM-DD": (~seen["visit_date"].str.fullmatch(EXACT_DATE)).any(),
        "a blank weight": seen["weight_kg"].eq("").any(),
        "an unmapped visit label": (~seen["visit"].isin(["baseline", "week12"])).any(),
        "three rechecks with different readings":
            sample_pairs.groupby(["patient_id", "visit"])["sbp_mmhg"].nunique().eq(2).sum() >= 3,
        "a missing week12 visit": wide[wide["patient_id"].isin(ids)]["week12"].isna().any(),
        "a blank A1c result": lab[["a1c_baseline", "a1c_week12"]].isna().any().any(),
    }
    assert all(covered.values()), [name for name, ok in covered.items() if not ok]


def write(root: Path, files: dict[str, str | bytes | None]) -> Path:
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


def loose(files: dict[str, str]) -> dict[str, str | bytes]:
    """The same answers written every way the README allows or whitespace changes."""
    crlf = lambda text: text.replace("\n", "\r\n")  # noqa: E731
    patients = table(files["patients_summary.csv"])
    visits = table(files["visits_summary.csv"])
    visits["value"] = visits["value"].where(~visits["metric"].isin(["visit_date min", "visit_date max"]),
                                            visits["value"].str.replace("-", "/") + " 00:00:00")
    analysis = table(files["analysis_sample.csv"])
    analysis["visit_date"] = analysis["visit_date"].str.replace("-", "/")
    analysis["gender"] = analysis["gender"].str.lower()
    analysis["patient_id"] = analysis["patient_id"].str.lower()
    categories = table(files["gender_categories.csv"]).iloc[::-1]
    return {
        "file_inventory.txt": "\n".join(line for line in files["file_inventory.txt"].splitlines()
                                        if "total" not in line).replace("data/", "./data/"),
        "table_profile.csv": "﻿" + edit(files["table_profile.csv"], lambda t: t.assign(
            table=t["table"].str.replace(".csv", "", regex=False).str.upper(), rows=t["rows"] + ".0")
            .iloc[::-1, ::-1]),
        "issue_audit.csv": crlf(files["issue_audit.csv"].replace(",", ";")),
        "patients_summary.csv": crlf(patients.assign(metric=patients["metric"].str.upper(),
                                                     value=patients["value"] + ".0").iloc[::-1].to_csv()),
        "gender_categories.csv": categories.assign(gender=categories["gender"].str.lower().replace(
            "", "NaN"))[["code", "count", "gender"]].to_csv(index=False),
        "visits_summary.csv": visits.set_index("metric")["value"].to_csv().replace(",", ", "),
        "decision_log.csv": edit(files["decision_log.csv"], lambda t: t.assign(
            decision=t["decision"].str.upper() + ".", notes="extra column").iloc[::-1]),
        "a1c_summary.csv": edit(files["a1c_summary.csv"], lambda t: t.assign(visit=t["visit"].str.title())
                                .iloc[::-1, ::-1]).replace("\n", "  \n"),
        "merge_audit.csv": table(files["merge_audit.csv"]).set_index("check")["count"].to_csv(),
        "sbp_wide_sample.csv": files["sbp_wide_sample.csv"].replace("\n", "\r").rstrip("\r"),
        "analysis_sample.csv": analysis[list(reversed(ANALYSIS_COLUMNS))].iloc[::-1].to_csv(sep="\t"),
    }


def check_handout(files: dict[str, str]) -> None:
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
    assert re.fullmatch(r"\s*\d+ data/sites\.csv", files["file_inventory.txt"].splitlines()[0])
    # Analyst-sized outputs: a few KB each, well under 200 KB in all.
    sizes = {name: len(text.encode("utf-8")) for name, text in files.items()}
    assert max(sizes.values()) < 8_000 and sum(sizes.values()) < 200_000, sizes
    graded: dict[str, int] = {}
    for check in grading.CHECKS:
        file_name = check.name.split(":")[0]
        graded[file_name] = graded.get(file_name, 0) + check.points
    assert checks_contract() == graded, ("CHECKS.md Completion contract points", checks_contract(), graded)
    review = checks_review_points()
    assert grading.MAX_SCORE + review == 100 and check_assignment.HUMAN_REVIEW_POINTS == review, review
    assert (f"{grading.MAX_SCORE} points graded from your committed files after the deadline, "
            f"{review} by human review") in checks_md, "CHECKS.md states a different point split"
    # The exam publishes no graded value: no output line, and no graded count written anywhere in the text.
    shown = set(HEADERS.values()) | {f"{table},{decision}" for table, decision in grading.DECISIONS}
    leaks = [line for name in files if name not in ("decision_log.csv", "file_inventory.txt")
             for line in files[name].splitlines() if line not in shown and line in readme + checks_md]
    assert not leaks, ("the README publishes expected values", leaks)
    graded_numbers = {str(count) for _, count in (*grading.ISSUE_AUDIT, *grading.MERGE_AUDIT) if count > 9}
    graded_numbers |= {str(value) for value in (*grading.PATIENT_SUMMARY.values(),
                                                *grading.VISIT_SUMMARY.values()) if isinstance(value, int) and value > 9
                       and value not in (70, 250)}
    published = set(re.findall(r"(?<![\w.])\d+(?![\w.])", readme + checks_md))
    assert not graded_numbers & published - {"1940", "1996"}, ("the README states graded counts",
                                                                graded_numbers & published)
    for phrase in ("check_assignment", "GitHub Actions", "final newline", "grading.py", "automated feedback"):
        assert phrase not in readme + checks_md, ("the exam README mentions", phrase)
    listed = re.findall(r"`(P\d{5})`", readme)
    assert tuple(listed) == grading.SAMPLE_IDS, ("README SAMPLE_IDS", listed)

    shipped = [name for name in ("check_assignment.py", "grading.py", "test_assignment.py", "_grader_selftest",
                                 ".github", ".badmath.toml") if (HANDOUT / name).exists()]
    assert not shipped, ("the exam handout ships automated checks", shipped)
    assert "data/" in (HANDOUT / ".gitignore").read_text(encoding="utf-8").split(), "data/ is not ignored"
    tracked = subprocess.run(["git", "ls-files", "--", "data"], cwd=HANDOUT, capture_output=True, text=True).stdout
    assert not [line for line in tracked.split() if (HANDOUT / line).exists()], ("data files still tracked", tracked)
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(not cell.get("outputs") and cell.get("execution_count") is None
               for cell in notebook["cells"] if cell["cell_type"] == "code"), "clear the handout notebook's outputs"
    assert f"SAMPLE_IDS = {list(grading.SAMPLE_IDS)}".replace("'", '"').replace(", ", ",") == \
        re.search(r"SAMPLE_IDS = \[.*?\]", code[0], re.S).group().replace("\n", "").replace(" ", "").replace(
            "SAMPLE_IDS=", "SAMPLE_IDS = "), "the notebook's SAMPLE_IDS differ from grading.SAMPLE_IDS"
    assert "generate_data.py" in code[0] and "data_manifest.json" in code[0]
    # Only the gated file is saved after the validation assert; every other file is saved before it.
    gate = next(index for index, source in enumerate(code) if "assert checks.all()" in source)
    join = next(index for index, source in enumerate(code) if "three left merges" in source)
    for path in re.findall(r"^(\w+_PATH) = OUTPUT_DIR", code[0], re.M):
        first = next(index for index, source in enumerate(code[1:], start=1) if path in source)
        assert (first > gate) == (path == "ANALYSIS_SAMPLE_PATH"), (path, "saved on the wrong side of the gate")
        # A MergeError in the join stops Run All, so only the gated extract may come after it.
        assert (first > join) == (path == "ANALYSIS_SAMPLE_PATH"), (path, "saved after the Task 4.4 join")
    assert sorted(path.name for path in (HANDOUT / "output").iterdir()) == [".gitkeep"], "stray handout outputs"
    for path in [CHECKS / "grading.py", CHECKS / "check_assignment.py", *HANDOUT.rglob("*.py")]:
        assert "version_info" not in path.read_text(encoding="utf-8"), (path, "reads the Python version")


def run() -> None:
    generate()
    found: dict = {}
    files = solve(references=found)
    confirm_reference(files, found)
    check_handout(files)

    empty = WORK / "empty"
    empty.mkdir()
    for root in (empty, HANDOUT):
        result, code = checker(root)
        assert (result["score"], result["max-score"], code) == (0, 75, 1), (root, result["score"], code)
        assert all(test["detail"] for test in result["tests"]), "every failing check says why"

    complete = write(WORK / "complete", files)
    result, code = checker(complete)
    assert (result["score"], code) == (75, 0), losses(result)
    assert grading.grade_submission(complete) == result

    result, _ = checker(write(WORK / "loose", loose(files)))
    assert result["score"] == 75, [(name, detail(result, name)) for name in losses(result)]

    result, _ = checker(write(WORK / "partial", {name: files[name] for name in ("file_inventory.txt",
                                                                                 "table_profile.csv")}))
    assert result["score"] == 10, losses(result)

    # No submitted file runs: every one of these would leave a marker if imported or executed.
    trapped = write(WORK / "trapped", files)
    marker = WORK / "ran"
    trap = f"open({str(marker)!r}, 'w').write(__file__)\n"
    for name in ("grading.py", "check_assignment.py", "generate_data.py", "json.py", "csv.py", "re.py",
                 "sitecustomize.py", "output/grading.py"):
        (trapped / name).write_text(trap, encoding="utf-8")
    result, _ = checker(trapped)
    assert result["score"] == 75 and not marker.exists(), "a submitted file ran"

    def bump(text: str, column: str, changes: dict[str, str]) -> str:
        frame = table(text)
        for name, change in changes.items():
            old = int(frame.loc[frame[column].eq(name), "count"].iloc[0])
            frame.loc[frame[column].eq(name), "count"] = str(old + int(change))
        return frame.to_csv(index=False)

    no_woman = {key: code for key, code in GENDER_MAP.items() if key != "woman"}
    no_bl = {key: label for key, label in VISIT_MAP.items() if key != "bl"}
    analysis_csv = files["analysis_sample.csv"]
    # Each mistake is carried through every later file, as the student's own pipeline would carry it, and is
    # charged once: (the changed files, the points each check loses, words the feedback must use).
    mutations = {
        "woman left out of the gender dictionary": (solve(gender_map=no_woman),
                                                    {"gender_categories.csv: counts": 1}, ("dictionary", "value_counts()")),
        "gender stripped but never mapped": (solve(gender_map=None), {"gender_categories.csv: counts": 2},
                                             ("Series.map",)),
        "gender left as text, so the counts have no code column": (
            {"gender_categories.csv": edit(files["gender_categories.csv"], lambda t: t.drop(columns="code"))},
            {"gender_categories.csv: codes": 2}, ("code", "astype")),
        "category codes typed by hand, all 0": (
            {"gender_categories.csv": edit(files["gender_categories.csv"], lambda t: t.assign(code="0"))},
            {"gender_categories.csv: codes": 2}, (".cat.codes",)),
        "gender counts saved without dropna=False": (
            {"gender_categories.csv": edit(files["gender_categories.csv"], lambda t: t[t["gender"].ne("")])},
            {"gender_categories.csv: counts": 1}, ("missing", "dropna=False")),
        "the exact patient repeats kept": (solve(keep_patient_repeats=True),
                                           {"patients_summary.csv: patients": 2}, ("drop_duplicates",)),
        "birth_year 9999 kept": (solve(keep_9999=True), {"patients_summary.csv: birth_year": 2}, ("9999",)),
        "site codes uppercased but hyphens and inner spaces kept": (
            solve(site_rule="upper"), {"patients_summary.csv: site_code": 2}, ("hyphen",)),
        "wt_kg never renamed": (solve(rename_wt=False), {"visits_summary.csv: weight_kg": 2}, ("wt_kg",)),
        "BL left out of the visit dictionary": (solve(visit_map=no_bl), {"visits_summary.csv: rows": 3},
                                                ("dictionary",)),
        "the earlier reading of each recheck kept": (
            solve(recheck_keep="first"), {"sbp_wide_sample.csv: baseline": 1, "sbp_wide_sample.csv: week12": 1},
            ("keep='last'",)),
        "dates accepted without the exact-text check": (
            solve(exact_dates=False), {"visits_summary.csv: visit_date": 1}, ("fullmatch",)),
        "sbp range rule skipped, 999 kept": (solve(sbp_range=False), {"visits_summary.csv: sbp_mmhg": 3},
                                             ("999",)),
        "melt labels swapped": (solve(melt_map={"a1c_baseline": "week12", "a1c_week12": "baseline"}),
                                {"a1c_summary.csv: results": 2}, ("map each label",)),
        "melt labels never mapped": (solve(melt_map="none"), {"a1c_summary.csv: results": 2}, ("map the labels",)),
        "melt rows with no result kept": (solve(melt_dropna=False), {"a1c_summary.csv: results": 2},
                                          ("drop the rows",)),
        "the A1c merged with how='inner'": (solve(a1c_how="inner"), {"analysis_sample.csv: rows": 1},
                                            ("how='left'",)),
        "every site record joined, retired names included": (solve(site_table="all"),
                                                             {"analysis_sample.csv: rows": 3}, ("validate=",)),
        "the retired site names joined in place of the current ones": (
            solve(site_table="retired"), {"analysis_sample.csv: site_name": 2}, ("current",)),
        "sbp_change written baseline minus week12": (solve(reverse_change=True),
                                                     {"sbp_wide_sample.csv: sbp_change": 2},
                                                     ("week12 minus baseline",)),
        "the whole analysis table saved in place of the extract": (
            {"analysis_sample.csv": table(analysis_csv).pipe(lambda t: pd.concat(
                [t, t.assign(patient_id="P99999")])).to_csv(index=False)},
            {"analysis_sample.csv: rows": 3}, ("SAMPLE_IDS",)),
        "analysis_sample.csv not committed": (
            {"analysis_sample.csv": None},
            {"analysis_sample.csv: rows": 3, "analysis_sample.csv: site_name": 2, "analysis_sample.csv: a1c_pct": 2,
             "analysis_sample.csv: gender and birth_year": 2}, ("missing", "Task 4.4")),
        "wc counts written without the header lines": (
            {"file_inventory.txt": re.sub(r"(\d+) data", lambda m: f"{int(m.group(1)) - 1} data",
                                          files["file_inventory.txt"])},
            {"file_inventory.txt": 5}, ("without the header",)),
        "one audit count off by one": (
            {"issue_audit.csv": bump(files["issue_audit.csv"], "issue", {"visits: date parse failures": "-1"})},
            {"issue_audit.csv": 1}, ("date parse failures", "fullmatch")),
        "one merge count off": (
            {"merge_audit.csv": bump(files["merge_audit.csv"], "check", {"registry patients with no visit": "+1"})},
            {"merge_audit.csv": 1}, ("registry patients with no visit",)),
        "one reason left blank": (
            {"decision_log.csv": edit(files["decision_log.csv"],
                                      lambda t: t.assign(reason=t["reason"].mask(t.index == 3, "")))},
            {"decision_log.csv: reason": 1}, ("row 4",)),
        "one decision rewritten": (
            {"decision_log.csv": edit(files["decision_log.csv"], lambda t: t.assign(
                decision=t["decision"].mask(t.index == 8, "drop every site")))},
            {"decision_log.csv: decisions": 1}, ("current site records",)),
        "a profile that counts the header as a row": (
            {"table_profile.csv": edit(files["table_profile.csv"],
                                       lambda t: t.assign(rows=(t["rows"].astype(int) + 1).astype(str)))},
            {"table_profile.csv": 5}, ("header is not a row",)),
        "an audit file with embedded NUL": (
            {"issue_audit.csv": files["issue_audit.csv"] + "\x00"}, {"issue_audit.csv": 10}, ("embedded NUL",)),
    }
    failures = []
    for description, (changes, expected_losses, mentions) in mutations.items():
        changed = {name: text for name, text in changes.items() if text != files.get(name)}
        root = write(WORK / re.sub(r"\W+", "-", description), {**files, **changed})
        result, code = checker(root)
        if losses(result) != expected_losses or code != (0 if not expected_losses else 1):
            failures.append((description, losses(result), [detail(result, name) for name in losses(result)]))
            continue
        for name in expected_losses:
            text = detail(result, name)
            if "Fix: Task" not in text or not all(word in text for word in mentions):
                failures.append((description, "feedback", text))
    assert not failures, "\n".join(map(str, failures))

    print("Assignment 06 checks: generate_data.py writes the data_manifest.json tables, as Parquet and CSV, in "
          "under 3 seconds and the values grading.py derives match an independent pandas solution; the extracts' "
          "patients cover every planted pattern; the README lists every artifact with its header and line count and states no "
          "graded count; CHECKS.md states the 75/25 split and each file's points; the handout ships no checks and "
          "no data, and saves every ungated file before the validation assert; empty and handout score 0, a "
          "partial submission 10; complete and loosely formatted submissions score 75; no submitted file runs; "
          f"{len(mutations)} single mistakes, carried through every later file, each cost exactly their own points.")


if __name__ == "__main__":
    try:
        run()
    finally:
        import shutil
        shutil.rmtree(WORK, ignore_errors=True)
