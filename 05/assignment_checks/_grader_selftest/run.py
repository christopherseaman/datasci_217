"""Development regression checks for the Assignment 05 checks.

Answers the assignment with pandas the way a student would, builds
submissions in ignored `scratch/`, and confirms what each kind of submission
scores: a correct one scores 100 however it is formatted, and each mistake
costs only its own checks. It also confirms that the data the checks hold
match the handout's data, and that the handout in `05/assignment/` ships the
course-owned checks byte for byte. Nothing here reads or runs student code.

    uv run --python 3.13 --with pandas==3.0.5 --with pyarrow==25.0.0 python 05/assignment_checks/_grader_selftest/run.py
"""

import contextlib
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# Self-tests grade with the working copy, never the copy downloaded from GitHub.
os.environ["DS217_LOCAL_CHECKS"] = "1"

CHECKS = Path(__file__).resolve().parents[1]
HANDOUT = CHECKS.parent / "assignment"
SCRATCH = CHECKS.parents[1] / "scratch"
DATA = HANDOUT / "data" / "lab_visits.csv"
sys.path.insert(0, str(CHECKS))
from _value_checks import (  # noqa: E402
    CHECKS as VALUE_CHECKS,
    CLEAN_FILE,
    CODED_FILE,
    MISSING_FILE,
    RAW_COLUMNS,
    RAW_ROWS,
    TYPED_FILE,
    VALIDATION_FILE,
)
from grading import POINTS, grade_submission  # noqa: E402

CHECK_POINTS = {check.name: points for check, points in zip(VALUE_CHECKS, POINTS, strict=True)}
FILES = (MISSING_FILE, TYPED_FILE, CODED_FILE, VALIDATION_FILE, CLEAN_FILE)


def group(prefix: str) -> set[str]:
    return {name for name in CHECK_POINTS if name.startswith(prefix)}


MISSING, TYPED, CODED = group("missing report"), group("typed table"), group("coded table")
VALIDATION, CLEAN = group("validation report"), group("clean table")
SEX_CODES = {"f": "F", "female": "F", "m": "M", "male": "M"}


def solve(root: Path, wrong: frozenset = frozenset()) -> None:
    """Write every artifact with Lecture 05's pandas methods, as the README asks; `wrong` names mistakes to make."""
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    labs = pd.read_csv(DATA)

    # Task 1
    gaps = labs.dropna(subset=["hba1c_pct", "glucose_mg_dl"], how="all") if "count-after-dropna" in wrong else labs
    missing_report = pd.DataFrame({
        "missing_count": gaps.notna().sum() if "count-present" in wrong else gaps.isna().sum(),
        "missing_pct": (gaps.isna().mean() * (1 if "pct-as-fraction" in wrong else 100)).round(1),
    })
    missing_report.index.name = "column"
    missing_report.to_csv(output / "missing_report.csv", index="report-index-false" not in wrong)
    if "dropna-any" in wrong:
        labs = labs.dropna(subset=["hba1c_pct", "glucose_mg_dl"], how="any")
    elif "no-dropna" not in wrong:
        labs = labs.dropna(subset=["hba1c_pct", "glucose_mg_dl"], how="all")

    # Task 2
    if "dedupe-by-patient" in wrong:
        labs = labs.drop_duplicates(subset=["patient_id"])
    elif "dedupe-keep-false" in wrong:
        labs = labs.drop_duplicates(keep=False)
    elif "no-dedupe" not in wrong:
        labs = labs.drop_duplicates()
    if "sentinel-kept" not in wrong:
        labs = labs.replace({"weight_kg": {-999: np.nan}, "glucose_mg_dl": {-999: np.nan}})
    if "age-text" not in wrong:
        labs["age"] = pd.to_numeric(labs["age"], errors="coerce").astype("Int64")
    if "dates-text" not in wrong:
        labs["visit_date"] = pd.to_datetime(labs["visit_date"], format="%Y-%m-%d", errors="coerce")
    labs.to_csv(output / "labs_typed.csv", index=False)

    # Task 3
    ids = labs["patient_id"] if "ids-unstripped" in wrong else labs["patient_id"].str.strip()
    labs["patient_id"] = ids if "ids-not-upper" in wrong else ids.str.upper()
    clinic = labs["clinic"] if "clinic-unstripped" in wrong else labs["clinic"].str.strip()
    labs["clinic"] = clinic if "clinic-not-title" in wrong else clinic.str.title()
    if "clinic-not-filled" not in wrong:
        labs["clinic"] = labs["clinic"].fillna("Unknown")
    if "sex-upper-only" in wrong:
        labs["sex"] = labs["sex"].str.strip().str.upper()
    elif "sex-replace" in wrong:
        labs["sex"] = labs["sex"].str.strip().str.lower().replace(SEX_CODES)
    else:
        labs["sex"] = labs["sex"].str.strip().str.lower().map(SEX_CODES)
    bins = [0, 5.7, 6.5, 20] if "cut-wrong-edges" in wrong else [0, 5.6, 6.4, 20]
    labs["a1c_category"] = pd.cut(labs["hba1c_pct"], bins=bins, labels=["normal", "prediabetes", "diabetes"])
    if "no-dummies" not in wrong:
        labs = labs.join(pd.get_dummies(labs["clinic"], prefix="clinic", dtype="int64"))
    labs.to_csv(output / "labs_coded.csv", index=False)

    # Task 4
    hba1c_ok = labs["hba1c_pct"].between(4, 20)
    weight_ok = labs["weight_kg"].between(30, 250)
    if "rules-blank-fails" not in wrong:
        hba1c_ok = hba1c_ok | labs["hba1c_pct"].isna()
        weight_ok = weight_ok | labs["weight_kg"].isna()
    rules = pd.DataFrame({
        "patient_id_format": labs["patient_id"].str.fullmatch(r"D[0-9]{4}", na=False),
        "visit_date_present": labs["visit_date"].notna(),
        "sex_valid": labs["sex"].isin(["F", "M"]),
        "hba1c_in_range": hba1c_ok,
        "weight_in_range": weight_ok,
    })
    validation_report = rules.sum() if "rules-pass-count" in wrong else (~rules).sum()
    validation_report.index.name = "rule"
    validation_report.name = "failing_rows"
    validation_report.to_csv(output / "validation_report.csv")
    glucose = labs["glucose_mg_dl"]
    q1, q3 = glucose.quantile(0.25), glucose.quantile(0.75)
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    if "outlier-blank-flagged" in wrong:
        labs["glucose_outlier"] = ~glucose.between(lower, upper)
    elif "no-outlier" not in wrong:
        labs["glucose_outlier"] = (glucose < lower) | (glucose > upper)

    # Task 5
    clean = labs.copy()
    if "drop-every-failure" in wrong:
        clean = clean[rules.all(axis="columns")]
    elif "keep-bad-ids" not in wrong:
        clean = clean[rules["patient_id_format"] & rules["visit_date_present"]]
    if "range-kept" not in wrong:
        clean["hba1c_pct"] = clean["hba1c_pct"].where(clean["hba1c_pct"].between(4, 20))
        clean["weight_kg"] = clean["weight_kg"].where(clean["weight_kg"].between(30, 250))
    if "fill-median" in wrong:
        clean = clean.fillna(clean.median(numeric_only=True))
    clean = clean.reset_index(drop=True)
    if "clean-twice" in wrong:
        clean = pd.concat([clean, clean.iloc[:1]])
    clean.to_csv(output / "labs_clean.csv", index=False)


def scores(root: Path) -> dict[str, int]:
    result = grade_submission(root)
    assert result["schema"] == "datasci217/grading-result/v1", result["schema"]
    assert result["max-score"] == sum(POINTS) == 100, result["max-score"]
    assert result["score"] == sum(test["score"] for test in result["tests"]), result
    return {test["test-name"]: test["score"] for test in result["tests"]}


def lost(root: Path) -> set[str]:
    return {name for name, score in scores(root).items() if score < CHECK_POINTS[name]}


def details(root: Path) -> dict[str, str]:
    return {test["test-name"]: test["detail"] for test in grade_submission(root)["tests"] if test["detail"]}


def checker_report(checks: Path, root: Path) -> dict:
    """Grade through the `check_assignment.py` in `checks`, as a student or CI runs it."""
    finished = subprocess.run(
        [sys.executable, "-B", str(checks / "check_assignment.py"), str(root), "--json"],
        capture_output=True, text=True, check=False,
    )
    assert finished.stdout, finished.stderr
    report = json.loads(finished.stdout)
    assert (finished.returncode == 0) == (report["score"] == report["max-score"]), (finished.returncode, report)
    return report


def printed(root: Path) -> str:
    return subprocess.run(
        [sys.executable, "-B", str(CHECKS / "check_assignment.py"), str(root)],
        capture_output=True, text=True, check=False,
    ).stdout


def edit(path: Path, change) -> None:
    path.write_text(change(path.read_text(encoding="utf-8")), encoding="utf-8")


def run() -> None:
    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a05-selftest-") as temporary:
        workspace = Path(temporary)

        empty = workspace / "empty"
        empty.mkdir()
        assert sum(scores(empty).values()) == 0
        assert checker_report(CHECKS, empty)["score"] == 0

        fresh = workspace / "handout"
        shutil.copytree(HANDOUT, fresh, ignore=shutil.ignore_patterns("__pycache__", ".venv", ".pytest_cache", ".checks"))
        assert sum(scores(fresh).values()) == 0

        correct = workspace / "correct"
        solve(correct)
        assert lost(correct) == set(), details(correct)
        for checks in (CHECKS, HANDOUT):
            assert checker_report(checks, correct)["score"] == 100, checks
            assert checker_report(checks, empty) == checker_report(CHECKS, empty), checks

        def variant(label: str) -> Path:
            root = workspace / label
            shutil.copytree(correct, root)
            return root

        def full_marks(label: str, change) -> None:
            root = variant(label)
            change(root)
            assert lost(root) == set(), (label, details(root))

        # Formatting that changes no value never costs points.
        def loose(root: Path) -> None:
            for name in FILES:
                edit(root / name, lambda text: "\r\n".join(
                    ",".join(cell.upper() for cell in line.split(",")) for line in text.splitlines()))

        full_marks("upper-case-crlf", loose)

        def respaced(root: Path) -> None:
            for name in (MISSING_FILE, VALIDATION_FILE):
                edit(root / name, lambda text: "\n".join(
                    ", ".join(f" {cell} " for cell in line.split(",")) for line in text.splitlines()))

        full_marks("spaced-reports", respaced)

        def reformatted(root: Path) -> None:
            for name in (TYPED_FILE, CODED_FILE, CLEAN_FILE):
                table = pd.read_csv(root / name, dtype=str, keep_default_na=False)
                table = table[table.columns[::-1]]
                table["visit_date"] = table["visit_date"].map(lambda d: f"{d} 00:00:00" if d else "NaT")
                table.to_csv(root / name, sep=";", quoting=csv.QUOTE_ALL)
            report = pd.read_csv(root / MISSING_FILE)
            report.sample(frac=1, random_state=1).to_csv(root / MISSING_FILE, sep="\t", index=False)

        full_marks("quoted-reordered-row-numbers", reformatted)

        def redated(root: Path) -> None:
            for name, form in ((TYPED_FILE, "%Y/%m/%d"), (CODED_FILE, "%Y%m%d"), (CLEAN_FILE, "%Y/%m/%d")):
                table = pd.read_csv(root / name, parse_dates=["visit_date"])
                table.to_csv(root / name, index=False, date_format=form)

        full_marks("slash-and-compact-dates", redated)

        def encoded(root: Path) -> None:
            text = (root / TYPED_FILE).read_text(encoding="utf-8")
            (root / TYPED_FILE).write_bytes(text.encode("utf-16"))
            text = (root / CLEAN_FILE).read_text(encoding="utf-8")
            (root / CLEAN_FILE).unlink()
            (root / "output" / "Labs_Clean.csv").write_bytes(("﻿" + text).encode("utf-8"))

        full_marks("utf16-bom-renamed", encoded)

        def alternatives(root: Path) -> None:
            # Unrounded percents, True/False indicators, 0/1 flags, and rules saved with readable names.
            labs = pd.read_csv(DATA)
            report = pd.DataFrame({"missing_count": labs.isna().sum(), "missing_pct": labs.isna().mean() * 100})
            report.index.name = "column"
            report.to_csv(root / MISSING_FILE)
            coded = pd.read_csv(root / CODED_FILE, keep_default_na=False, dtype=str)
            for column in [c for c in coded.columns if c.startswith("clinic_")]:
                coded[column] = coded[column].map({"1": "True", "0": "False"})
            coded.to_csv(root / CODED_FILE, index=False)
            clean = pd.read_csv(root / CLEAN_FILE, keep_default_na=False, dtype=str)
            clean["glucose_outlier"] = clean["glucose_outlier"].map({"True": "1", "False": "0"})
            clean.to_csv(root / CLEAN_FILE, index=False)
            edit(root / VALIDATION_FILE, lambda text: text.replace("patient_id_format", "Patient ID format")
                 .replace("hba1c_in_range", "HbA1c range"))

        full_marks("alternative-forms", alternatives)

        # Each mistake costs only the checks it gets wrong.
        mistakes = []

        def mistake(label: str, expected: set[str], hint: str, change=None) -> Path:
            root = workspace / label
            mistakes.append(label)
            if change is None:
                solve(root, frozenset({label}))
            else:
                shutil.copytree(correct, root)
                change(root)
            found = lost(root)
            assert found == expected, (label, found, details(root))
            assert any(hint in details(root)[name] for name in expected), (label, details(root))
            assert checker_report(HANDOUT, root) == checker_report(CHECKS, root), label
            return root

        mistake("count-after-dropna", {"missing report: missing_count"}, "before Task 1.2's dropna()")
        mistake("count-present", {"missing report: missing_count"}, "isna().sum() counts the gaps")
        mistake("pct-as-fraction", {"missing report: missing_pct"}, "multiply it by 100")
        mistake("report-index-false", MISSING, "index=False leaves them out")
        mistake("no-dropna", {"typed table: no-lab visit dropped"}, 'how="all"')
        mistake("dropna-any", {"typed table: no-lab visit dropped"}, "has one of the two lab results")
        mistake("no-dedupe", {"typed table: exact repeats removed"}, "entered twice by mistake")
        mistake("dedupe-keep-false", {"typed table: exact repeats removed"}, "keep=False drops every copy")
        mistake("dedupe-by-patient", {"typed table: repeat visit kept"}, 'subset=["patient_id"]')
        mistake("sentinel-kept", {"typed table: -999 read as missing"}, "-999 is the export's code")
        mistake("age-text", {"typed table: age as numbers"}, "errors=\"coerce\"")
        mistake("dates-text", {"typed table: visit dates parsed"}, "a date that does not exist into NaT")
        mistake("ids-unstripped", {"coded table: patient IDs cleaned"}, "spaces around them")
        mistake("ids-not-upper", {"coded table: patient IDs cleaned"}, "mixes letter cases")
        mistake("clinic-unstripped", {"coded table: clinic names cleaned"}, "spaces around them")
        mistake("clinic-not-title", {"coded table: clinic names cleaned"}, "spells mission as")
        mistake("clinic-not-filled", {"coded table: missing clinic filled"}, 'fillna("Unknown")')
        mistake("sex-upper-only", {"coded table: sex codes mapped"}, "expected F")
        mistake("sex-replace", {"coded table: sex codes mapped"}, "replace() leaves it")
        mistake("cut-wrong-edges", {"coded table: a1c_category"}, "includes its right edge")
        mistake("no-dummies", {"coded table: clinic indicators"}, "pd.get_dummies")
        mistake("rules-pass-count", {"validation report: failing rows"}, "(~rules).sum()")
        mistake("rules-blank-fails", {"validation report: failing rows"}, "between() is False for NaN")
        mistake("keep-bad-ids", {"clean table: failing visits dropped"}, "fail the patient_id_format")
        mistake("drop-every-failure", {"clean table: failing visits dropped"}, "stays, with the bad value blanked")
        mistake("range-kept", {"clean table: impossible values blanked"}, ".where(")
        mistake("no-outlier", {"clean table: glucose_outlier flag"}, "has no glucose_outlier column")
        mistake("outlier-blank-flagged", {"clean table: glucose_outlier flag"}, "a blank glucose is not an outlier")
        mistake("fill-median", {"clean table: other values kept"}, "keeps blanks as blanks")
        mistake("clean-twice", {"clean table: one row per visit"}, "each visit should appear once")
        mistake("typed-missing", TYPED, "is missing; run the Task 2.3 cell",
                lambda root: (root / TYPED_FILE).unlink())
        mistake("coded-in-root", CODED, "the assignment folder itself has labs_coded.csv",
                lambda root: (root / CODED_FILE).rename(root / "labs_coded.csv"))
        mistake("validation-sideways", VALIDATION, "rules across the top",
                lambda root: pd.read_csv(root / VALIDATION_FILE, index_col="rule").T.to_csv(root / VALIDATION_FILE))

        def written_twice(root: Path) -> None:
            text = (root / VALIDATION_FILE).read_text(encoding="utf-8")
            (root / VALIDATION_FILE).write_text(text + text, encoding="utf-8")

        mistake("validation-written-twice", {"validation report: one row per rule"}, "holds the same table 2 times",
                written_twice)

        # A mistake upstream is charged where it was made, and later files are judged against the student's own.
        cascade = {
            "no-dedupe": "typed table: exact repeats removed",
            "sentinel-kept": "typed table: -999 read as missing",
            "dates-text": "typed table: visit dates parsed",
            "ids-unstripped": "coded table: patient IDs cleaned",
            "clinic-unstripped": "coded table: clinic names cleaned",
            "clinic-not-filled": "coded table: missing clinic filled",
        }
        for label, name in cascade.items():
            assert lost(workspace / label) == {name}, (label, details(workspace / label))

        # The printed report says each shared fix once and ends by naming the checks left to fix.
        report = printed(empty)
        assert report.count("is missing; run the Task") == 5, report
        assert report.rstrip().endswith(
            "Left to fix (100 points): missing report (all 3 checks); typed table (all 7 checks); coded table "
            "(all 6 checks); validation report (all 2 checks); clean table (all 5 checks)."), report
        report = printed(workspace / "sentinel-kept")
        assert report.rstrip().endswith("Score: 95/100\nLeft to fix (5 points): typed table: -999 read as missing."), report
        assert printed(correct).rstrip().endswith("Score: 100/100\nAll checks passed."), printed(correct)

        # The notebook's last cell calls run_checks(): same report dict and same text as the command line.
        from check_assignment import run_checks
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = run_checks(correct)
        assert result["score"] == result["max-score"] == 100, result
        assert buffer.getvalue() == printed(correct), (buffer.getvalue(), printed(correct))

    # A fresh handout prints exactly the "Before Task 1" example CHECKS.md shows.
    readme = (HANDOUT / "README.md").read_text(encoding="utf-8")
    checks_md = (HANDOUT / "CHECKS.md").read_text(encoding="utf-8")
    assert "(CHECKS.md)" in readme, "README.md no longer links CHECKS.md"
    shown = re.search(r"Before Task 1, for example, the missing-report checks report:\n\n```text\n(.*?)```", checks_md,
                      re.DOTALL)
    assert shown is not None, "CHECKS.md no longer shows the Before Task 1 example"
    handout_report = subprocess.run(
        [sys.executable, "-B", "check_assignment.py"], cwd=HANDOUT, capture_output=True, text=True, check=False
    ).stdout
    assert shown.group(1) in handout_report, (shown.group(1), handout_report)

    # CHECKS.md's completion contract agrees with the checks and their points.
    contract = {
        name: int(points)
        for name, points in re.findall(r"^\| `output/[^|]*\| [^|]+\| ([^|]+) \| (\d+) \|$", checks_md, re.MULTILINE)
    }
    assert contract == CHECK_POINTS, (contract, CHECK_POINTS)

    print(f"Assignment 05 checks: empty, handout, correct, and 5 reformatted submissions score as intended, "
          f"and so do {len(mistakes)} submissions with mistakes.")


def run_handout() -> None:
    """The handout's data match the checks, and it ships the course's checks unchanged."""
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == RAW_COLUMNS, "data/lab_visits.csv's header differs from RAW_COLUMNS"
    assert tuple(map(tuple, rows[1:])) == RAW_ROWS, "data/lab_visits.csv differs from RAW_ROWS in _value_checks.py"
    sys.path.insert(0, str(CHECKS.parents[1] / "scripts"))
    import make_assignment05_data
    assert make_assignment05_data.COLUMNS == RAW_COLUMNS and make_assignment05_data.ROWS == RAW_ROWS, (
        "scripts/make_assignment05_data.py and RAW_ROWS differ")

    workflow = (HANDOUT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert re.search(r'^  CHECKS_PATH: "05/assignment_checks"$', workflow, re.MULTILINE), (
        "tests.yml does not download from 05/assignment_checks"
    )
    listed = re.search(r"^  CHECKS_FILES: \|\n((?:    \S.*\n)+)", workflow, re.MULTILINE)
    assert listed is not None, "tests.yml lists no CHECKS_FILES"
    files = listed.group(1).split()

    # The list names every checker file here, so a download never misses a module.
    course_owned = [path.name for path in CHECKS.glob("*.py")]
    course_owned += [f".github/test/{path.name}" for path in (CHECKS / ".github" / "test").iterdir() if path.is_file()]
    assert sorted(files) == sorted(course_owned), (sorted(files), sorted(course_owned))
    for name in files:
        assert (HANDOUT / name).read_bytes() == (CHECKS / name).read_bytes(), (
            f"05/assignment/{name} differs from 05/assignment_checks/{name}; copy the course-owned file over it"
        )

    # The handout's only Python files are the checks; the self-test and its answers stay here.
    handout_python = {path.relative_to(HANDOUT).as_posix() for path in HANDOUT.rglob("*.py")
                      if ".venv" not in path.parts and ".checks" not in path.parts}
    assert handout_python == set(files) - {".github/test/requirements.txt"}, sorted(handout_python)
    assert not (HANDOUT / "_grader_selftest").exists()

    # Setup follows Lecture 03: uv sync builds the notebook's environment from pyproject.toml and uv.lock.
    assert not (HANDOUT / "requirements.txt").exists(), "the handout ships pyproject.toml and uv.lock, not requirements.txt"
    project = (HANDOUT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (HANDOUT / "uv.lock").read_text(encoding="utf-8")
    for name, version in (("numpy", "2.3.3"), ("pandas", "3.0.5"), ("ipykernel", "6.29.5"), ("pyarrow", "25.0.0")):
        assert f'"{name}=={version}"' in project, f"pyproject.toml does not pin {name}=={version}"
        assert f'name = "{name}"\nversion = "{version}"' in lock, f"uv.lock does not lock {name} {version}; run uv lock"
    assert (HANDOUT / ".python-version").read_text(encoding="utf-8").strip() == "3.13", ".python-version should name 3.13"

    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, "clear the handout notebook's outputs"
    assert "run_checks()" in "".join(notebook["cells"][-1]["source"]), "the notebook's last cell should run the checks"

    print(f"Assignment 05 handout: data match the checks, pyproject.toml and uv.lock pin the course packages, "
          f"and all {len(files)} check files match 05/assignment_checks byte for byte.")


if __name__ == "__main__":
    run()
    run_handout()
