"""Development regression checks for the Assignment 05 checks.

Answers the assignment with pandas the way a student would, builds submissions in ignored `scratch/`, and
confirms what each kind of submission scores: a correct one scores 100 however it is formatted, and each
mistake costs only its own checks. It also confirms that the handout's generator writes the data its
manifest names, and that the handout in `05/assignment/` ships the course-owned checks byte for byte.
Nothing here reads or runs student code.

    uv run --python 3.13 --with pandas==3.0.5 --with pyarrow==25.0.0 python 05/assignment_checks/_grader_selftest/run.py
"""

import contextlib
import csv
import hashlib
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
REPO = CHECKS.parents[1]
SCRATCH = REPO / "scratch"
sys.path.insert(0, str(CHECKS))
import generate_data  # noqa: E402
from _value_checks import (  # noqa: E402
    CATEGORY_FILE,
    CHECKS as VALUE_CHECKS,
    DECISION_FILE,
    MISSING_FILE,
    PROFILE_FILE,
    SUMMARY_FILE,
    VALIDATION_FILE,
)
from grading import POINTS, grade_submission  # noqa: E402

CHECK_POINTS = {check.name: points for check, points in zip(VALUE_CHECKS, POINTS, strict=True)}
FILES = (MISSING_FILE, PROFILE_FILE, CATEGORY_FILE, VALIDATION_FILE, DECISION_FILE, SUMMARY_FILE)
REPORTS = (MISSING_FILE, CATEGORY_FILE, VALIDATION_FILE, DECISION_FILE)
GENDER_CODES = {"f": "F", "female": "F", "m": "M", "male": "M", "x": "X", "nb": "X", "nonbinary": "X"}


def group(prefix: str) -> set[str]:
    return {name for name in CHECK_POINTS if name.startswith(prefix)}


class GateStopped(Exception):
    """Task 5.2's assert stopped the cell before clean_summary.csv was saved."""


def solve(root: Path, data: Path, wrong: frozenset = frozenset()) -> None:
    """Write every artifact with Lecture 05's pandas methods, as the README asks; `wrong` names mistakes to make."""
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    # A valid route: read -999 as missing at once, which Task 2's replace() then leaves unchanged.
    labs = pd.read_csv(data, na_values=[-999]) if "na-values-read" in wrong else pd.read_csv(data)

    # Task 1
    gaps = labs.dropna(subset=["hba1c_pct", "glucose_mg_dl"], how="all") if "count-after-dropna" in wrong else labs
    missing_report = pd.DataFrame({
        "missing_count": gaps.notna().sum() if "count-present" in wrong else gaps.isna().sum(),
        "missing_pct": (gaps.isna().mean() * (1 if "pct-as-fraction" in wrong else 100)).round(1),
    })
    missing_report.index.name = "column"
    missing_report.to_csv(output / "missing_report.csv")
    rows_read = len(labs)
    if "sentinel-before-dropna" in wrong:
        labs = labs.replace({"weight_kg": {-999: np.nan}, "glucose_mg_dl": {-999: np.nan}})
    if "dropna-any" in wrong:
        labs = labs.dropna(subset=["hba1c_pct", "glucose_mg_dl"], how="any")
    elif "bare-dropna" in wrong:
        labs = labs.dropna()
    elif "no-dropna" not in wrong:
        labs = labs.dropna(subset=["hba1c_pct", "glucose_mg_dl"], how="all")
    rows_with_labs = len(labs)
    if "dedupe-by-patient" in wrong:
        labs = labs.drop_duplicates(subset=["patient_id"])
    elif "dedupe-by-visit" in wrong:
        labs = labs.drop_duplicates(subset=["patient_id", "visit_date"])
    elif "dedupe-keep-false" in wrong:
        labs = labs.drop_duplicates(keep=False)
    elif "no-dedupe" not in wrong:
        labs = labs.drop_duplicates()
    if "extra-dropna-weight" in wrong:
        labs = labs.dropna(subset=["weight_kg"])
    rows_unique = len(labs)

    # Task 2
    if "sentinel-kept" not in wrong:
        labs = labs.replace({"weight_kg": {-999: np.nan}, "glucose_mg_dl": {-999: np.nan}})
    if "age-text" not in wrong:
        labs["age"] = pd.to_numeric(labs["age"], errors="coerce").astype("Int64")
    if "age-fill-zero" in wrong:
        labs["age"] = labs["age"].fillna(0)
    if "dates-text" not in wrong:
        labs["visit_date"] = pd.to_datetime(labs["visit_date"], format="%Y-%m-%d", errors="coerce")
    if "drop-undated" in wrong:
        labs = labs.dropna(subset=["visit_date"])
    typed = labs[["visit_date", "age", "weight_kg", "hba1c_pct", "glucose_mg_dl"]]
    typed_profile = pd.DataFrame({
        "dtype": typed.dtypes.astype(str),
        "missing_count": typed.isna().sum(),
        "min": typed.min(),
        "max": typed.max(),
        "mean": typed.mean(numeric_only=True).round(2),
    })
    typed_profile.index.name = "column"
    typed_profile.to_csv(output / "typed_profile.csv")

    # Task 3
    ids = labs["patient_id"] if "ids-unstripped" in wrong else labs["patient_id"].str.strip()
    labs["patient_id"] = ids if "ids-not-upper" in wrong else ids.str.upper()
    clinic = labs["clinic"] if "clinic-unstripped" in wrong else labs["clinic"].str.strip()
    labs["clinic"] = clinic if "clinic-not-title" in wrong else clinic.str.title()
    if "clinic-not-filled" not in wrong:
        labs["clinic"] = labs["clinic"].fillna("Missing" if "clinic-missing-label" in wrong else "Unknown")
    if "gender-upper-only" in wrong:
        labs["gender"] = labs["gender"].str.strip().str.upper()
    elif "gender-no-x" in wrong:
        binary = {key: code for key, code in GENDER_CODES.items() if code != "X"}
        labs["gender"] = labs["gender"].str.strip().str.lower().map(binary)
    elif "gender-no-nb" in wrong:
        spellings = {key: code for key, code in GENDER_CODES.items() if key not in ("nb", "nonbinary")}
        labs["gender"] = labs["gender"].str.strip().str.lower().map(spellings)
    elif "gender-replace" in wrong:
        labs["gender"] = labs["gender"].str.strip().str.lower().replace(GENDER_CODES)
    else:
        labs["gender"] = labs["gender"].str.strip().str.lower().map(GENDER_CODES)
    if "cut-left-closed-right-edges" in wrong:
        bins = [0, 5.7, 6.5, 20.1]
    elif "cut-wrong-edges" in wrong or "cut-left-closed-20" in wrong:
        bins = [0, 5.7, 6.5, 20]
    else:
        bins = [0, 5.6, 6.4, 20]
    labs["a1c_category"] = pd.cut(labs["hba1c_pct"], bins=bins, labels=["normal", "prediabetes", "diabetes"],
                                  right=not any(label.startswith("cut-left-closed") for label in wrong))
    if "no-dummies" in wrong:
        category_counts = pd.concat([labs[column].value_counts() for column in ("clinic", "gender", "a1c_category")])
    else:
        coded = pd.get_dummies(labs[["clinic", "gender", "a1c_category"]], dtype="int64",
                               drop_first="dummies-drop-first" in wrong, dummy_na="dummies-na" in wrong)
        category_counts = coded.sum()
    category_counts.index.name = "indicator"
    category_counts.name = "visits"
    category_counts.to_csv(output / "category_counts.csv")

    # Task 4
    glucose = labs["glucose_mg_dl"]
    q1, q3 = glucose.quantile(0.25), glucose.quantile(0.75)
    multiplier = 1 if "iqr-multiplier-1" in wrong else 1.5
    lower, upper = q1 - multiplier * (q3 - q1), q3 + multiplier * (q3 - q1)
    if "outlier-blank-flagged" in wrong:
        labs["glucose_outlier"] = ~glucose.between(lower, upper)
    else:
        labs["glucose_outlier"] = (glucose < lower) | (glucose > upper)
    hba1c_ok = labs["hba1c_pct"].between(4, 20, inclusive="neither" if "rules-edges-excluded" in wrong else "both")
    weight_ok = labs["weight_kg"].between(30, 250)
    if "rules-blank-fails" not in wrong:
        hba1c_ok = hba1c_ok | labs["hba1c_pct"].isna()
        weight_ok = weight_ok | labs["weight_kg"].isna()
    rules = pd.DataFrame({
        "patient_id_format": (labs["patient_id"].str.contains if "ids-contains" in wrong
                              else labs["patient_id"].str.fullmatch)(r"D[0-9]{5}", na=False),
        "visit_date_present": labs["visit_date"].notna(),
        "gender_valid": labs["gender"].isin(["F", "M", "X"]),
        "hba1c_in_range": hba1c_ok,
        ("weight_ok" if "rule-renamed" in wrong else "weight_in_range"): weight_ok,
        "glucose_within_fences": labs["glucose_outlier"] if "outlier-not-negated" in wrong else ~labs["glucose_outlier"],
    })
    validation_report = rules.sum() if "rules-pass-count" in wrong else (~rules).sum(
        axis=1 if "rules-sum-per-visit" in wrong else 0)
    validation_report.index.name = "rule"
    validation_report.name = "failing_rows"
    validation_report.to_csv(output / "validation_report.csv")

    # Task 5
    clean = labs.copy()
    if "drop-every-failure" in wrong:
        clean = clean[rules.all(axis="columns")]
    elif "drop-bad-gender" in wrong:
        clean = clean[rules["patient_id_format"] & rules["visit_date_present"] & rules["gender_valid"]]
    elif "keep-bad-ids" in wrong:
        clean = clean[rules["visit_date_present"]]
    elif "either-filter" in wrong:
        clean = clean[rules["patient_id_format"] | rules["visit_date_present"]]
    else:
        clean = clean[rules["patient_id_format"] & rules["visit_date_present"]]
    hba1c_outside = ((clean["hba1c_pct"] < 4) | (clean["hba1c_pct"] > 20)).sum()
    weight_outside = ((clean["weight_kg"] < 30) | (clean["weight_kg"] > 250)).sum()
    if "blank-edges" in wrong:
        clean["hba1c_pct"] = clean["hba1c_pct"].where((clean["hba1c_pct"] > 4) & (clean["hba1c_pct"] < 20))
        clean["weight_kg"] = clean["weight_kg"].where(clean["weight_kg"].between(30, 250))
    elif "range-kept" not in wrong:
        if "hba1c-kept" not in wrong:
            clean["hba1c_pct"] = clean["hba1c_pct"].where(clean["hba1c_pct"].between(4, 20))
        if "weight-kept" not in wrong:
            clean["weight_kg"] = clean["weight_kg"].where(clean["weight_kg"].between(30, 250))
    if "count-after-blanking" in wrong:
        hba1c_outside = ((clean["hba1c_pct"] < 4) | (clean["hba1c_pct"] > 20)).sum()
        weight_outside = ((clean["weight_kg"] < 30) | (clean["weight_kg"] > 250)).sum()
    if "fill-median" in wrong:
        clean = clean.fillna(clean.median(numeric_only=True))
    if "fill-weight-mean" in wrong:
        clean["weight_kg"] = clean["weight_kg"].fillna(clean["weight_kg"].mean())
    clean = clean.reset_index(drop=True)
    decision_log = pd.Series({"rows_read": rows_read, "rows_with_labs": rows_with_labs, "rows_unique": rows_unique,
                              "visits_dropped": len(labs) - len(clean), "hba1c_blanked": hba1c_outside,
                              "weight_blanked": weight_outside}, name="rows")
    decision_log.index.name = "decision"
    decision_log.to_csv(output / "decision_log.csv")
    checks = pd.Series({
        "one row per visit": not clean.duplicated(subset=["patient_id", "visit_date"]).any(),
        "patient IDs valid": clean["patient_id"].str.fullmatch(r"D[0-9]{5}").all(),
        "visit dates present": clean["visit_date"].notna().all(),
        "hba1c in range": clean["hba1c_pct"].dropna().between(4, 20).all(),
        "weight in range": clean["weight_kg"].dropna().between(30, 250).all(),
    })
    if not checks.all() and "no-gate" not in wrong:
        raise GateStopped(checks[~checks].index.tolist())
    columns = [c for c in ("age", "weight_kg", "hba1c_pct", "glucose_mg_dl") if c in clean.columns]
    if "summary-of-labs" in wrong:
        clean_summary = labs[columns].describe().round(2).T
    elif "summary-fill-zero" in wrong:
        clean_summary = clean[columns].fillna(0).describe().round(2).T
    else:
        clean_summary = clean[columns].describe().round(2).T
    clean_summary.to_csv(output / "clean_summary.csv")


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
        data = workspace / "lab_visits.csv"
        with data.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(generate_data.COLUMNS)
            writer.writerows(generate_data.generate_rows())

        empty = workspace / "empty"
        empty.mkdir()
        assert sum(scores(empty).values()) == 0
        assert checker_report(CHECKS, empty)["score"] == 0

        fresh = workspace / "handout"
        shutil.copytree(HANDOUT, fresh, ignore=shutil.ignore_patterns("__pycache__", ".venv", ".pytest_cache",
                                                                      ".checks", "data"))
        assert sum(scores(fresh).values()) == 0

        correct = workspace / "correct"
        solve(correct, data)
        assert lost(correct) == set(), details(correct)
        for checks in (CHECKS, HANDOUT):
            assert checker_report(checks, correct)["score"] == 100, checks
            assert checker_report(checks, empty) == checker_report(CHECKS, empty), checks
        total = sum((correct / name).stat().st_size for name in FILES)
        assert total < 200_000 and all((correct / name).stat().st_size < 10_000 for name in FILES), total

        def variant(label: str) -> Path:
            root = workspace / label
            shutil.copytree(correct, root)
            return root

        reformatted_labels = []

        def full_marks(label: str, change) -> None:
            reformatted_labels.append(label)
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
            for name in REPORTS + (PROFILE_FILE, SUMMARY_FILE):
                edit(root / name, lambda text: "\n".join(
                    ", ".join(f" {cell} " for cell in line.split(",")) for line in text.splitlines()))

        full_marks("spaced-reports", respaced)

        def reformatted(root: Path) -> None:
            for name in (MISSING_FILE, PROFILE_FILE, CATEGORY_FILE, SUMMARY_FILE):
                report = pd.read_csv(root / name, dtype=str, keep_default_na=False)
                report = report[report.columns[::-1]] if name == CATEGORY_FILE else report
                report.sample(frac=1, random_state=1).to_csv(root / name, sep="\t")
            for name in (VALIDATION_FILE, DECISION_FILE):
                report = pd.read_csv(root / name, dtype=str, keep_default_na=False)
                report[report.columns[::-1]].to_csv(root / name, index=False, sep=";", quoting=csv.QUOTE_ALL)

        full_marks("quoted-reordered-row-numbers", reformatted)

        def redated(root: Path) -> None:
            profile = pd.read_csv(root / PROFILE_FILE, index_col=0, dtype=str)
            for stat, form in (("min", "%d/%m/%Y"), ("max", "%B %d, %Y")):
                day = pd.to_datetime(profile.loc["visit_date", stat])
                profile.loc["visit_date", stat] = day.strftime(form)
            profile.to_csv(root / PROFILE_FILE)

        full_marks("other-date-formats", redated)

        def encoded(root: Path) -> None:
            text = (root / PROFILE_FILE).read_text(encoding="utf-8")
            (root / PROFILE_FILE).write_bytes(text.encode("utf-16"))
            text = (root / SUMMARY_FILE).read_text(encoding="utf-8")
            (root / SUMMARY_FILE).unlink()
            (root / "output" / "Clean_Summary.csv").write_bytes(("\ufeff" + text).encode("utf-8"))

        full_marks("utf16-bom-renamed", encoded)

        def alternatives(root: Path) -> None:
            # Unrounded percents, a sideways describe(), indicators named with another separator, and other labels.
            labs = pd.read_csv(data)
            report = pd.DataFrame({"missing_count": labs.isna().sum(), "missing_pct": labs.isna().mean() * 100})
            report.index.name = "column"
            report.to_csv(root / MISSING_FILE)
            pd.read_csv(root / SUMMARY_FILE, index_col=0).T.to_csv(root / SUMMARY_FILE)
            edit(root / CATEGORY_FILE, lambda text: text.replace("a1c_category_", "a1c: ").replace("clinic_", "clinic-")
                 .replace("visits", "0"))
            edit(root / VALIDATION_FILE, lambda text: text.replace("patient_id_format", "Patient ID format")
                 .replace("hba1c_in_range", "HbA1c in range").replace("glucose_within_fences", "Glucose within fences"))
            edit(root / DECISION_FILE, lambda text: text.replace("rows_with_labs", "after dropna")
                 .replace("rows_unique", "after drop_duplicates").replace("rows_read", "rows read"))

        full_marks("alternative-forms", alternatives)

        def whole_percents(root: Path) -> None:
            labs = pd.read_csv(data)
            report = pd.DataFrame({"missing_count": labs.isna().sum(), "missing_pct": (labs.isna().mean() * 100).round()})
            report.index.name = "column"
            report.to_csv(root / MISSING_FILE)

        full_marks("whole-percents", whole_percents)

        def missing_word(root: Path) -> None:
            # to_csv(na_rep="missing") writes the word missing for every blank.
            table = pd.read_csv(root / PROFILE_FILE, dtype=str, keep_default_na=False)
            table.replace("", "missing").to_csv(root / PROFILE_FILE, index=False)

        full_marks("na-rep-missing", missing_word)

        def sideways_counts(root: Path) -> None:
            pd.read_csv(root / CATEGORY_FILE, index_col=0).T.to_csv(root / CATEGORY_FILE, index=False)

        full_marks("sideways-category-counts", sideways_counts)

        def without_index(root: Path) -> None:
            # index=False drops the labels; rows in the README's order are read in that order.
            for name in FILES:
                frame = pd.read_csv(root / name, dtype=str, keep_default_na=False)
                frame.iloc[:, 1:].to_csv(root / name, index=False)

        full_marks("index-false", without_index)

        def describe_untransposed_index_false(root: Path) -> None:
            pd.read_csv(root / SUMMARY_FILE, index_col=0).T.to_csv(root / SUMMARY_FILE, index=False)

        full_marks("describe-untransposed-index-false", describe_untransposed_index_false)

        def one_decimal_means(root: Path) -> None:
            for name in (PROFILE_FILE, SUMMARY_FILE):
                frame = pd.read_csv(root / name, index_col=0)
                frame["mean"] = frame["mean"].round(1)
                frame.to_csv(root / name)

        full_marks("one-decimal-means", one_decimal_means)

        def right_false_edges(root: Path) -> None:
            shutil.rmtree(root / "output")
            solve(root, data, frozenset({"cut-left-closed-right-edges"}))

        full_marks("cut-left-closed-right-edges", right_false_edges)

        # -999 made missing before Task 1.2's dropna(), at read or just after Task 1.1, is a valid route: later
        # files are judged against the rows it leaves.
        for route in ("na-values-read", "sentinel-before-dropna"):
            def early_sentinel(root: Path, route: str = route) -> None:
                shutil.rmtree(root / "output")
                solve(root, data, frozenset({route}))

            full_marks(route, early_sentinel)

        # Each mistake costs only the checks it gets wrong.
        mistakes = []

        def mistake(label: str, expected: set[str], hint: str, change=None, gate: bool = True) -> Path:
            """`gate=False` skips Task 5.2's assert, which stops this mistake, to show the later files still count."""
            root = workspace / label
            mistakes.append(label)
            if change is None:
                solve(root, data, frozenset({label} if gate else {label, "no-gate"}))
            else:
                shutil.copytree(correct, root)
                change(root)
            found = lost(root)
            assert found == expected, (label, found, details(root))
            assert any(hint in details(root)[name] for name in expected), (label, hint, details(root))
            return root

        mistake("count-after-dropna", {"missing report: missing_count"}, "before Task 1.2's dropna()")
        mistake("count-present", {"missing report: missing_count"}, "isna().sum() counts the gaps")
        mistake("pct-as-fraction", {"missing report: missing_pct"}, "multiply it by 100")
        mistake("no-dropna", {"decision log: no-lab visits dropped"}, "nothing was dropped")
        mistake("dropna-any", {"decision log: no-lab visits dropped"}, 'how="any"')
        mistake("no-dedupe", {"decision log: exact repeats removed"}, "nothing was removed", gate=False)
        mistake("dedupe-keep-false", {"decision log: exact repeats removed"}, "keep=False drops every copy")
        mistake("dedupe-by-patient", {"decision log: repeat visits kept"}, "one row per patient")
        mistake("dedupe-by-visit", {"decision log: exact repeats removed"}, "treats two blanks as equal")
        mistake("sentinel-kept", {"typed profile: -999 read as missing"}, "-999, the export's code")
        mistake("age-text", {"typed profile: age as numbers"}, 'errors="coerce"')
        mistake("age-fill-zero", {"typed profile: unreadable ages left blank"}, "fillna() with 0")
        mistake("dates-text", {"typed profile: visit dates parsed"}, "expected datetime64")
        mistake("drop-undated", {"typed profile: undated visits kept"}, 'dropna(subset=["visit_date"])')
        mistake("ids-unstripped", {"validation report: patient IDs cleaned"}, "add .str.strip()")
        mistake("ids-not-upper", {"validation report: patient IDs cleaned"}, "add .str.upper()")
        mistake("clinic-unstripped", {"category counts: clinic names cleaned"}, "mission as")
        mistake("clinic-not-title", {"category counts: clinic names cleaned"}, "mission as")
        mistake("clinic-not-filled", {"category counts: missing clinic filled"}, 'fillna("Unknown")')
        mistake("gender-upper-only", {"category counts: gender codes mapped"}, "which are spellings")
        mistake("gender-replace", {"category counts: gender codes mapped"}, "replace() leaves it")
        mistake("gender-no-x", {"category counts: gender codes mapped"}, "there is no gender_X")
        mistake("gender-no-nb", {"category counts: gender codes mapped"}, "missing from gender_codes (nb, nonbinary)")
        mistake("cut-wrong-edges", {"category counts: a1c_category"}, "includes its right edge")
        mistake("cut-left-closed", {"category counts: a1c_category"}, "lands in the next band")
        mistake("cut-left-closed-20", {"category counts: a1c_category"}, "leaves out an HbA1c of exactly 20")
        mistake("no-dummies", {"category counts: indicator columns"}, "pd.get_dummies")
        mistake("rules-pass-count", {"validation report: failing rows"}, "(~rules).sum()")
        mistake("rules-sum-per-visit", {"validation report: one row per rule"}, "without axis=1")
        mistake("rules-blank-fails", {"validation report: failing rows"}, "between() is False for NaN")
        mistake("rule-renamed", {"validation report: rule names"}, "it has weight_ok for weight_in_range")
        mistake("iqr-multiplier-1", {"validation report: glucose outliers flagged"}, "1 * IQR")
        mistake("outlier-blank-flagged", {"validation report: glucose outliers flagged"}, "~between() is True for NaN")
        mistake("outlier-not-negated", {"validation report: glucose outliers flagged"}, '~labs["glucose_outlier"]')
        mistake("keep-bad-ids", {"decision log: failing visits dropped"}, "drops only the undated visits", gate=False)
        mistake("drop-bad-gender", {"decision log: failing visits dropped"}, "visits without a valid ID")
        mistake("drop-every-failure", {"decision log: failing visits dropped"}, "fails any rule")
        mistake("count-after-blanking", {"decision log: out-of-range values counted"}, "Count them before blanking")
        mistake("fill-median", {"clean summary: blanks kept"}, "keeps blanks as blanks")
        # One count mistake in one column costs the blanks-kept check only, not the statistics too.
        mistake("fill-weight-mean", {"clean summary: blanks kept"}, "count is the values present")
        mistake("blank-edges", {"clean summary: blanks kept"}, "keeps the edges 4 and 20")

        def mean_off(root: Path) -> None:
            summary = pd.read_csv(root / SUMMARY_FILE, index_col=0)
            summary.loc["hba1c_pct", "mean"] += 1
            summary.to_csv(root / SUMMARY_FILE)

        mistake("summary-mean-off", {"clean summary: statistics"}, "changes only the out-of-range", mean_off)

        def count_dropped_means_off(root: Path) -> None:
            summary = pd.read_csv(root / SUMMARY_FILE, index_col=0).drop(columns="count")
            summary["mean"] = 999
            summary.to_csv(root / SUMMARY_FILE)

        # Without counts, the statistics are still compared.
        mistake("summary-count-dropped", {"clean summary: blanks kept", "clean summary: statistics"}, "mean is 999",
                count_dropped_means_off)
        # One Task 5.2 mistake costs one check.
        mistake("summary-fill-zero", {"clean summary: blanks kept"}, "the blanks were filled in")
        mistake("summary-of-labs", {"clean summary: impossible values blanked"}, "the Task 5.2 cell summarized labs")
        mistake("counts-header-only", group("category counts"), "header but no indicator rows",
                lambda root: (root / CATEGORY_FILE).write_text("indicator,visits\n", encoding="utf-8"))
        mistake("a1c-other-labels", {"category counts: a1c_category"}, "labels should be normal, prediabetes",
                lambda root: edit(root / CATEGORY_FILE, lambda text: text.replace("_normal", "_low")
                                  .replace("_prediabetes", "_mid").replace("_diabetes", "_high")))

        def profile_mean_off(root: Path) -> None:
            profile = pd.read_csv(root / PROFILE_FILE, index_col=0)
            profile.loc["hba1c_pct", "mean"] += 0.021
            profile.to_csv(root / PROFILE_FILE)

        mistake("profile-mean-off", {"typed profile: other values unchanged"}, "hba1c_pct's mean is",
                profile_mean_off)
        mistake("range-kept", {"clean summary: impossible values blanked"}, ".where(", gate=False)
        # One column left unblanked costs the impossible-values check only, not its count or statistics too.
        mistake("weight-kept", {"clean summary: impossible values blanked"}, ".where(", gate=False)
        mistake("hba1c-kept", {"clean summary: impossible values blanked"}, ".where(", gate=False)

        def gate_stops(root: Path) -> None:
            (root / SUMMARY_FILE).unlink()
            try:
                solve(root, data, frozenset({"range-kept"}))
            except GateStopped:
                return
            raise AssertionError("Task 5.2's checks should stop a submission that keeps impossible values")

        mistake("gate-stopped", {"clean summary: impossible values blanked"}, "assert probably stopped", gate_stops)

        def repeats_stop(root: Path) -> None:
            shutil.rmtree(root / "output")
            try:
                solve(root, data, frozenset({"no-dedupe"}))
            except GateStopped:
                return
            raise AssertionError("Task 5.2's checks should stop a submission that keeps exact repeats")

        mistake("no-dedupe-stopped", {"decision log: exact repeats removed"}, "nothing was removed", repeats_stop)

        def stopped(wrong: str):
            """Solve with `wrong`, which Task 5.2's assert stops before clean_summary.csv is saved."""
            def change(root: Path) -> None:
                shutil.rmtree(root / "output")
                try:
                    solve(root, data, frozenset({wrong}))
                except GateStopped:
                    return
                raise AssertionError(f"Task 5.2's checks should stop {wrong}")
            return change

        # A stop that a mistake an earlier check charged explains costs nothing more.
        mistake("keep-bad-ids-stopped", {"decision log: failing visits dropped"}, "drops only the undated visits",
                stopped("keep-bad-ids"))
        mistake("either-filter", {"decision log: failing visits dropped"}, "| means or", stopped("either-filter"))
        mistake("ids-contains", {"validation report: patient IDs cleaned"}, "str.contains()", stopped("ids-contains"))
        mistake("bare-dropna", {"decision log: no-lab visits dropped"}, "a plain dropna()")
        mistake("extra-dropna-weight", {"decision log: exact repeats removed"}, "a blank weight_kg")
        mistake("clinic-missing-label", {"category counts: missing clinic filled"}, "names them Unknown")
        mistake("dummies-drop-first", {"category counts: indicator columns"}, "drop_first=True")
        mistake("dummies-na", {"category counts: indicator columns"}, "dummy_na=True")
        mistake("rules-edges-excluded", {"validation report: failing rows"}, "includes 4 and 20")

        def summary_twice(root: Path) -> None:
            text = (root / SUMMARY_FILE).read_text(encoding="utf-8")
            (root / SUMMARY_FILE).write_text(text + text, encoding="utf-8")

        mistake("summary-written-twice", {"clean summary: statistics"}, "holds the same table 2 times", summary_twice)
        mistake("profile-missing", group("typed profile"), "is missing; run the Task 2 cell to write it with "
                "typed_profile.to_csv(PROFILE_PATH), then commit it. If an earlier cell stopped with an error",
                lambda root: (root / PROFILE_FILE).unlink())
        mistake("counts-in-root", group("category counts"), "the assignment folder itself has category_counts.csv",
                lambda root: (root / CATEGORY_FILE).rename(root / "category_counts.csv"))
        mistake("validation-sideways", {"validation report: one row per rule"}, "rules across the top",
                lambda root: pd.read_csv(root / VALIDATION_FILE, index_col="rule").T.to_csv(root / VALIDATION_FILE))

        def written_twice(root: Path) -> None:
            text = (root / VALIDATION_FILE).read_text(encoding="utf-8")
            (root / VALIDATION_FILE).write_text(text + text, encoding="utf-8")

        mistake("validation-written-twice", {"validation report: one row per rule"}, "holds the same table 2 times",
                written_twice)

        for label in ("sentinel-kept", "ids-unstripped", "gate-stopped"):
            assert checker_report(HANDOUT, workspace / label) == checker_report(CHECKS, workspace / label), label

        # A mistake upstream is charged where it was made, and later files are judged against the student's own.
        for label in ("no-dropna", "no-dedupe", "sentinel-kept", "age-fill-zero", "dates-text", "drop-undated",
                      "ids-unstripped", "clinic-not-filled", "gender-replace", "gender-no-x", "keep-bad-ids"):
            assert len(lost(workspace / label)) == 1, (label, details(workspace / label))

        # The printed report says each shared fix once and ends by naming the checks left to fix.
        report = printed(empty)
        assert report.count("is missing; run the Task") == len(FILES), report
        assert report.rstrip().endswith(
            "Left to fix (100 points): missing report (all 3 checks); typed profile (all 6 checks); category "
            "counts (all 5 checks); validation report (all 5 checks); decision log (all 5 checks); clean summary "
            "(all 3 checks)."), report
        report = printed(workspace / "sentinel-kept")
        assert report.rstrip().endswith("Score: 96/100\nLeft to fix (4 points): typed profile: -999 read as missing."), report
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
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a05-handout-") as temporary:
        copy = Path(temporary) / "handout"
        shutil.copytree(HANDOUT, copy, ignore=shutil.ignore_patterns("__pycache__", ".venv", ".checks", "output"))
        handout_report = subprocess.run(
            [sys.executable, "-B", "check_assignment.py"], cwd=copy, capture_output=True, text=True, check=False
        ).stdout
    assert shown.group(1) in handout_report, (shown.group(1), handout_report)

    # CHECKS.md's completion contract agrees with the checks and their points.
    contract = {
        name: int(points)
        for name, points in re.findall(r"^\| `output/[^|]*\| [^|]+\| ([^|]+) \| (\d+) \|$", checks_md, re.MULTILINE)
    }
    assert contract == CHECK_POINTS, (contract, CHECK_POINTS)

    print(f"Assignment 05 checks: empty, handout, correct, and {len(reformatted_labels)} reformatted submissions "
          f"score as intended, and so do {len(mistakes)} submissions with mistakes.")


def run_handout() -> None:
    """The handout's generator writes the data its manifest names, and it ships the course's checks unchanged."""
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
    checker = (CHECKS / "check_assignment.py").read_text(encoding="utf-8")
    latest = re.search(r"^LATEST_FILES = \((.*)\)$", checker, re.MULTILINE)
    assert latest is not None and {"_value_checks.py", "grading.py", "generate_data.py"} == set(
        re.findall(r'"([^"]+)"', latest.group(1))), "check_assignment.py must download every module the checks import"

    # The handout's only Python files are the checks and the generator; the self-test and its answers stay here.
    handout_python = {path.relative_to(HANDOUT).as_posix() for path in HANDOUT.rglob("*.py")
                      if ".venv" not in path.parts and ".checks" not in path.parts}
    assert handout_python == set(files) - {".github/test/requirements.txt"}, sorted(handout_python)
    assert not (HANDOUT / "_grader_selftest").exists()

    # The data are generated, never committed, and the generator writes exactly what the manifest names.
    committed = subprocess.run(["git", "ls-files", "05/assignment/data"], cwd=REPO, capture_output=True, text=True,
                               check=True).stdout.split()
    # A deletion not yet committed still lists the file, so only files on disk count.
    committed = [name for name in committed if (REPO / name).exists()]
    assert committed == [], f"the handout commits generated data: {committed}"
    assert "data/" in (HANDOUT / ".gitignore").read_text(encoding="utf-8").split(), ".gitignore should list data/"
    manifest = json.loads((HANDOUT / "data_manifest.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a05-generate-") as temporary:
        folder = Path(temporary)
        shutil.copy(HANDOUT / "generate_data.py", folder)
        for _ in range(2):
            subprocess.run([sys.executable, "-B", "generate_data.py"], cwd=folder, check=True, capture_output=True)
            for name, expected in manifest["files"].items():
                body = (folder / name).read_bytes()
                assert hashlib.sha256(body).hexdigest() == expected["sha256"], f"{name} differs from data_manifest.json"
                assert body.count(b"\n") - 1 == expected["rows"] and b"\r" not in body, name
                assert 2_000_000 <= len(body) <= 8_000_000, len(body)

    # Setup follows Lecture 03: uv sync builds the notebook's environment from pyproject.toml and uv.lock.
    assert not (HANDOUT / "requirements.txt").exists(), "the handout ships pyproject.toml and uv.lock, not requirements.txt"
    project = (HANDOUT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (HANDOUT / "uv.lock").read_text(encoding="utf-8")
    for name, version in (("numpy", "2.3.3"), ("pandas", "3.0.5"), ("ipykernel", "6.29.5"), ("pyarrow", "25.0.0")):
        assert f'"{name}=={version}"' in project, f"pyproject.toml does not pin {name}=={version}"
        assert f'name = "{name}"\nversion = "{version}"' in lock, f"uv.lock does not lock {name} {version}; run uv lock"
    assert (HANDOUT / ".python-version").read_text(encoding="utf-8").strip() == "3.13", ".python-version should name 3.13"

    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, "clear the handout notebook's outputs"
    assert "run_checks()" in "".join(notebook["cells"][-1]["source"]), "the notebook's last cell should run the checks"
    assert "generate_data.py" in code[0] and "data_manifest.json" in code[0], "the first cell should build the data"
    # Only the gated summary is saved after Task 5.2's assert, so one failing check stops no other file.
    source = "\n".join(code)
    gate = source.index("# TODO: assert that every check passes")
    before, after = source[:gate], source[gate:].split("run_checks")[0]
    assert set(re.findall(r"\b[A-Z0-9]+_PATH\b", after)) == {"SUMMARY_PATH"}, "save only clean_summary.csv after the assert"
    saved = set(re.findall(r"save \w+ to ([A-Z0-9]+_PATH)|and \w+ to ([A-Z0-9]+_PATH)", before))
    assert {name for pair in saved for name in pair if name} == {
        name for name in re.findall(r"^([A-Z0-9]+_PATH) = ", source, re.MULTILINE) if name not in ("SUMMARY_PATH", "DATA_PATH")
    }, "save every file but clean_summary.csv before the assert"

    print(f"Assignment 05 handout: generate_data.py writes the bytes data_manifest.json names, pyproject.toml and "
          f"uv.lock pin the course packages, and all {len(files)} check files match 05/assignment_checks byte for byte.")


if __name__ == "__main__":
    run()
    run_handout()
