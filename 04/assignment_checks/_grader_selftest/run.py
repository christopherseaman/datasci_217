"""Development regression checks for the Assignment 04 checks.

Answers the assignment with pandas the way a student would, builds
submissions in ignored `scratch/`, and confirms what each kind of submission
scores: a correct one scores 100 however it is formatted, and each mistake
costs only its own checks. It also confirms that the values the checks hold
match the handout's data, and that the handout in `04/assignment/` ships the
course-owned checks byte for byte. Nothing here reads or runs student code.

    uv run --python 3.13 --with pandas==3.0.5 --with pyarrow==25.0.0 python 04/assignment_checks/_grader_selftest/run.py
"""

from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable

import pandas as pd


CHECKS = Path(__file__).resolve().parents[1]
HANDOUT = CHECKS.parent / "assignment"
SCRATCH = CHECKS.parents[1] / "scratch"
DATA = HANDOUT / "data" / "bp_followup.csv"
HOME = HANDOUT / "data" / "home_bp.csv"
sys.path.insert(0, str(CHECKS))
from _value_checks import (  # noqa: E402
    BP_EXPORT,
    CHECKS as VALUE_CHECKS,
    COUNTS_FILE,
    FOLLOWUP_FILE,
    GAP_FILE,
    HOME_SBP,
    LOADED_FILE,
    PARQUET_FILE,
    SUMMARY_FILE,
    UNITS_ROW,
    program_order,
)
from grading import POINTS, grade_submission  # noqa: E402

CHECK_POINTS = {check.name: points for check, points in zip(VALUE_CHECKS, POINTS, strict=True)}


def group(prefix: str) -> set[str]:
    return {name for name in CHECK_POINTS if name.startswith(prefix)}


LOADED, SUMMARY, COUNTS = group("bp loaded"), group("visit summary"), group("clinic counts")
FOLLOWUP, PARQUET, GAP = group("follow-up list"), group("follow-up Parquet"), group("white-coat gap")
READINGS = ["sbp_baseline", "sbp_week4", "sbp_week8"]
USECOLS = ["patient_id", "clinic", "age", *READINGS]


def load(**changes) -> pd.DataFrame:
    options = dict(sep=";", skiprows=[1], usecols=USECOLS, na_values=["-999"], index_col="patient_id")
    options.update(changes)
    return pd.read_csv(DATA, **{key: value for key, value in options.items() if value is not None})


def summarize(bp: pd.DataFrame) -> pd.DataFrame:
    readings = bp[READINGS]
    summary = pd.DataFrame({"mean": readings.mean(), "median": readings.median(), "count": readings.count()})
    summary.index.name = "visit"
    return summary


def select(bp: pd.DataFrame, mask=None) -> pd.DataFrame:
    if mask is None:
        mask = bp["clinic"].isin(["North", "East"]) & (bp["sbp_baseline"] >= 140)
    followup = bp.loc[mask].drop(columns=["age"])
    followup["sbp_mean"] = followup[READINGS].mean(axis="columns")
    followup["change_week8"] = followup["sbp_week8"] - followup["sbp_baseline"]
    return followup


def order(followup: pd.DataFrame, **rank) -> pd.DataFrame:
    followup = followup.sort_values(by=["change_week8", "patient_id"])
    followup["improvement_rank"] = followup["change_week8"].rank(**({"method": "min"} | rank))
    return followup


def gap(bp: pd.DataFrame, home: pd.DataFrame) -> pd.Series:
    result = bp["sbp_week8"] - home["home_sbp_week8"]
    result.name = "white_coat_gap_mmhg"
    return result


def solve(root: Path, bp: pd.DataFrame | None = None) -> None:
    """Write every artifact with the lecture's pandas methods, as the README asks."""
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    bp = load() if bp is None else bp
    bp.to_csv(output / "bp_loaded.csv")
    summarize(bp).to_csv(output / "visit_summary.csv")
    bp["clinic"].value_counts().to_csv(output / "clinic_counts.csv")
    followup = order(select(bp))
    followup.to_csv(output / "followup_priority.csv")
    followup.to_parquet(output / "followup_priority.parquet")
    assert pd.read_parquet(output / "followup_priority.parquet").equals(followup)
    gap(bp, pd.read_csv(HOME, index_col="patient_id")).to_csv(output / "white_coat_gap.csv")


def scores(root: Path) -> dict[str, int]:
    result = grade_submission(root)
    assert result["schema"] == "datasci217/grading-result/v1", result["schema"]
    assert result["max-score"] == sum(POINTS) == 100, result["max-score"]
    assert result["score"] == sum(test["score"] for test in result["tests"]), result
    return {test["test-name"]: test["score"] for test in result["tests"]}


def lost(root: Path) -> set[str]:
    return {name for name, score in scores(root).items() if score < CHECK_POINTS[name]}


def detail(root: Path, name: str) -> str:
    return next(test["detail"] for test in grade_submission(root)["tests"] if test["test-name"] == name)


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


def edit(root: Path, name: str, change) -> None:
    path = root / name
    path.write_text(change(path.read_text(encoding="utf-8")), encoding="utf-8")


def run() -> None:
    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a04-selftest-") as temporary:
        workspace = Path(temporary)

        def variant(label: str, base: Path | None = None) -> Path:
            root = workspace / label
            shutil.copytree(base or correct, root)
            return root

        empty = workspace / "empty"
        empty.mkdir()
        assert sum(scores(empty).values()) == 0
        assert checker_report(CHECKS, empty)["score"] == 0

        fresh = workspace / "handout"
        shutil.copytree(HANDOUT, fresh, ignore=shutil.ignore_patterns("__pycache__", ".venv", ".pytest_cache"))
        assert sum(scores(fresh).values()) == 0

        correct = workspace / "correct"
        solve(correct)
        assert lost(correct) == set(), {name: detail(correct, name) for name in lost(correct)}
        for checks in (CHECKS, HANDOUT):
            assert checker_report(checks, correct)["score"] == 100, checks
            assert checker_report(checks, empty) == checker_report(CHECKS, empty), checks

        bp = load()
        home = pd.read_csv(HOME, index_col="patient_id")
        followup = order(select(bp))

        def full_marks(label: str, change) -> None:
            root = variant(label)
            change(root)
            assert lost(root) == set(), (label, {name: detail(root, name) for name in lost(root)})

        # Formatting that changes no value never costs points.
        def loose(root: Path) -> None:
            for name in (LOADED_FILE, SUMMARY_FILE, COUNTS_FILE, FOLLOWUP_FILE, GAP_FILE):
                edit(root, name, lambda text: "\r\n".join(
                    "  " + ", ".join(cell.upper() for cell in line.split(",")) + "  " for line in text.splitlines()))

        full_marks("loose", loose)
        full_marks("quoted", lambda root: followup.to_csv(root / FOLLOWUP_FILE, quoting=1, float_format="%.2f"))

        def separated(root: Path) -> None:
            bp.to_csv(root / LOADED_FILE, sep=";", decimal=",")
            summarize(bp).to_csv(root / SUMMARY_FILE, sep="\t")
            followup[followup.columns[::-1]].to_csv(root / FOLLOWUP_FILE, sep=";", decimal=",")

        full_marks("semicolons-tabs-column-order", separated)

        def encoded(root: Path) -> None:
            text = (root / LOADED_FILE).read_text(encoding="utf-8")
            (root / LOADED_FILE).write_bytes(text.encode("utf-16"))
            text = (root / GAP_FILE).read_text(encoding="utf-8")
            (root / GAP_FILE).unlink()
            (root / "output" / "White_Coat_Gap.csv").write_bytes(("﻿" + text).encode("utf-8"))

        full_marks("utf16-bom-renamed", encoded)

        # IDs moved into a column, with or without the row numbers to_csv() then writes.
        def reset(root: Path) -> None:
            bp.reset_index().to_csv(root / LOADED_FILE)
            followup.reset_index().to_csv(root / FOLLOWUP_FILE, index=False)
            followup.reset_index().to_parquet(root / PARQUET_FILE, index=False)

        full_marks("reset-index", reset)

        # Means rounded to one decimal, a summary saved sideways or with describe(), and a rank over every patient.
        def alternatives(root: Path) -> None:
            followup.round(1).to_csv(root / FOLLOWUP_FILE)
            summarize(bp).T.to_csv(root / SUMMARY_FILE)

        full_marks("rounded-and-transposed", alternatives)
        full_marks("describe", lambda root: bp[READINGS].describe().to_csv(root / SUMMARY_FILE))

        def rank_everyone(root: Path) -> None:
            everyone = select(bp, mask=pd.Series(True, index=bp.index))
            ranks = everyone["change_week8"].rank(method="min")
            frame = order(select(bp))
            frame["improvement_rank"] = ranks
            frame.to_csv(root / FOLLOWUP_FILE)

        full_marks("rank-over-every-patient", rank_everyone)

        # Each mistake costs only the checks it gets wrong.
        mistakes = []

        def mistake(label: str, change, expected: set[str], hint: str | None = None) -> Path:
            root = variant(label)
            mistakes.append(label)
            change(root)
            assert lost(root) == expected, (label, lost(root), {name: detail(root, name) for name in lost(root)})
            if hint is not None:
                assert any(hint in detail(root, name) for name in expected), (
                    label, {name: detail(root, name) for name in expected})
            assert checker_report(HANDOUT, root) == checker_report(CHECKS, root), label
            return root

        # -999 kept is one mistake: every later file is judged against the student's own table.
        kept = load(na_values=None)
        sentinel = mistake("sentinel-kept", lambda root: solve(root, kept),
                           {"bp loaded: -999 read as missing"}, 'add na_values=["-999"]')
        assert "P104's sbp_week4 is -999" in detail(sentinel, "bp loaded: -999 read as missing")

        mistake("loaded-ids-lost", lambda root: bp.to_csv(root / LOADED_FILE, index=False),
                {"bp loaded: patient_id column"}, "the patient IDs were lost")
        mistake("loaded-units-row-kept", lambda root: load(skiprows=None).to_csv(root / LOADED_FILE),
                {"bp loaded: units row skipped"}, "skiprows=[1]")
        mistake("loaded-units-row-as-header",
                lambda root: pd.read_csv(DATA, sep=";", skiprows=[0]).to_csv(root / LOADED_FILE, index=False),
                {"bp loaded: patient_id column", "bp loaded: units row skipped", "bp loaded: clinic, age, and readings"},
                "skiprows=[1] skips the units row")
        assert "(same fix as above)" in printed(workspace / "loaded-units-row-as-header")
        mistake("loaded-note-kept", lambda root: load(usecols=None).to_csv(root / LOADED_FILE),
                {"bp loaded: coordinator_note left out"}, "usecols")
        mistake("loaded-nrows", lambda root: load(nrows=10).to_csv(root / LOADED_FILE),
                {"bp loaded: all 14 patients once"}, "missing P111, P112, P113 and P114")
        mistake("loaded-no-age", lambda root: bp.drop(columns=["age"]).to_csv(root / LOADED_FILE),
                {"bp loaded: clinic, age, and readings"}, "no age column")
        mistake("loaded-wrong-value",
                lambda root: edit(root, LOADED_FILE, lambda text: text.replace("P108,South,68,171.0", "P108,South,68,117.0")),
                {"bp loaded: clinic, age, and readings"}, "P108's sbp_baseline is 117.0, expected 171")

        def written_twice(frame, path: Path, header: bool = True, **options) -> None:
            frame.to_csv(path, **options)
            frame.to_csv(path, mode="a", header=header, **options)

        mistake("loaded-written-twice", lambda root: written_twice(bp, root / LOADED_FILE),
                {"bp loaded: all 14 patients once"}, "holds the same table 2 times")
        mistake("loaded-missing", lambda root: (root / LOADED_FILE).unlink(), LOADED, "is missing")
        mistake("loaded-saved-at-root", lambda root: (root / LOADED_FILE).rename(root / "bp_loaded.csv"), LOADED,
                "the assignment folder itself has bp_loaded.csv")

        mistake("summary-across-rows",
                lambda root: pd.DataFrame({"mean": bp[READINGS].mean(axis="columns")}).to_csv(root / SUMMARY_FILE),
                {"visit summary: one row per visit"}, 'axis="columns"')
        mistake("summary-count-with-len",
                lambda root: summarize(bp).assign(count=len(bp)).to_csv(root / SUMMARY_FILE),
                {"visit summary: count"}, "sbp_baseline has 14, expected 13")
        mistake("summary-no-median",
                lambda root: summarize(bp).drop(columns=["median"]).to_csv(root / SUMMARY_FILE),
                {"visit summary: median"}, "no median column")
        mistake("summary-mean-is-max",
                lambda root: summarize(bp).assign(mean=bp[READINGS].max()).to_csv(root / SUMMARY_FILE),
                {"visit summary: mean"}, "sbp_baseline has 171.0, expected 152.15")

        mistake("counts-normalized",
                lambda root: bp["clinic"].value_counts(normalize=True).to_csv(root / COUNTS_FILE),
                {"clinic counts: patients per clinic"}, "normalize=True")
        mistake("counts-sorted-by-name",
                lambda root: bp["clinic"].value_counts().sort_index().to_csv(root / COUNTS_FILE),
                {"clinic counts: most common first"}, "East (4 patients) before North (5)")

        mistake("followup-greater-than",
                lambda root: order(select(bp, bp["clinic"].isin(["North", "East"]) & (bp["sbp_baseline"] > 140)))
                .to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: program patients"}, "> 140 drops it")
        mistake("followup-or-mask",
                lambda root: order(select(bp, bp["clinic"].isin(["North", "East"]) | (bp["sbp_baseline"] >= 140)))
                .to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: program patients"}, "are not at North or East")
        mistake("followup-no-isin",
                lambda root: order(select(bp, bp["sbp_baseline"] >= 140)).to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: program patients"}, "are not at North or East")
        mistake("followup-no-baseline-mask",
                lambda root: order(select(bp, bp["clinic"].isin(["North", "East"]))).to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: program patients"}, "a baseline below 140 mmHg or none at all")
        mistake("followup-ids-lost", lambda root: followup.to_csv(root / FOLLOWUP_FILE, index=False),
                {"follow-up list: patient_id column"}, "the patient IDs were lost")
        mistake("followup-age-kept",
                lambda root: followup.join(bp[["age"]]).to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: age dropped"}, '.drop(columns=["age"])')
        mistake("followup-mean-renamed",
                lambda root: followup.rename(columns={"sbp_mean": "mean_sbp"}).to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: derived column names"}, "mean_sbp where Task 4.2 names it sbp_mean")
        mistake("followup-no-mean",
                lambda root: followup.drop(columns=["sbp_mean"]).to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: sbp_mean values"}, "no sbp_mean column")
        mistake("followup-mean-is-sum",
                lambda root: followup.assign(sbp_mean=followup[READINGS].sum(axis="columns"))
                .to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: sbp_mean values"}, "P110 has 457.0, expected 152.33")

        def flipped(root: Path) -> None:
            frame = select(bp)
            frame["change_week8"] = frame["sbp_baseline"] - frame["sbp_week8"]
            frame = frame.sort_values(by=["change_week8", "patient_id"])
            frame["improvement_rank"] = frame["change_week8"].rank(method="min")
            frame.to_csv(root / FOLLOWUP_FILE)

        mistake("followup-change-flipped", flipped, {"follow-up list: change_week8 values"}, "P107 has 7.0")

        def not_reassigned(root: Path) -> None:
            frame = select(bp)
            frame.sort_values(by=["change_week8", "patient_id"])
            frame["improvement_rank"] = frame["change_week8"].rank(method="min")
            frame.to_csv(root / FOLLOWUP_FILE)

        mistake("followup-not-reassigned", not_reassigned, {"follow-up list: largest drop first"},
                "so assign it back")
        mistake("followup-descending",
                lambda root: followup.sort_values(by=["change_week8", "patient_id"], ascending=False)
                .to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: largest drop first"}, "P107 (change -7) comes before P114")
        mistake("followup-ties-reversed",
                lambda root: followup.sort_values(by=["change_week8", "patient_id"], ascending=[True, False])
                .to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: ties in patient_id order"}, "tie goes to the smaller patient_id, P101")
        mistake("followup-rank-average", lambda root: order(select(bp), method="average").to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: improvement_rank values"}, 'add method="min"')
        mistake("followup-rank-descending",
                lambda root: order(select(bp), ascending=False).to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: improvement_rank values"}, "leave out ascending=False")
        mistake("followup-no-rank",
                lambda root: followup.drop(columns=["improvement_rank"]).to_csv(root / FOLLOWUP_FILE),
                {"follow-up list: improvement_rank values"}, "no improvement_rank")
        mistake("followup-written-twice", lambda root: written_twice(followup, root / FOLLOWUP_FILE),
                {"follow-up list: program patients"}, "holds the same table 2 times")

        mistake("parquet-missing", lambda root: (root / PARQUET_FILE).unlink(), PARQUET, "is missing")
        mistake("parquet-is-csv", lambda root: followup.to_csv(root / PARQUET_FILE), PARQUET, "is CSV text")
        mistake("parquet-index-dropped", lambda root: followup.to_parquet(root / PARQUET_FILE, index=False),
                {"follow-up Parquet: same columns"}, "index=False leaves it out")
        mistake("parquet-before-rank", lambda root: select(bp).to_parquet(root / PARQUET_FILE),
                {"follow-up Parquet: same columns"}, "it is missing improvement_rank")

        mistake("gap-home-not-indexed", lambda root: gap(bp, pd.read_csv(HOME)).to_csv(root / GAP_FILE),
                {"white-coat gap: one row per patient in either table"}, 'index_col="patient_id"')
        mistake("gap-sign-flipped",
                lambda root: (-gap(bp, home)).rename("white_coat_gap_mmhg").to_csv(root / GAP_FILE),
                {"white-coat gap: gaps matched by patient"}, "wrong sign")

        def fill_value(root: Path) -> None:
            bp["sbp_week8"].sub(home["home_sbp_week8"], fill_value=0).rename("white_coat_gap_mmhg").to_csv(
                root / GAP_FILE)

        mistake("gap-fill-value", fill_value, {"white-coat gap: gaps matched by patient"}, "fill_value=0")
        # A lost index is charged once, by the check that names it; the values are still graded by row order.
        mistake("gap-ids-lost", lambda root: gap(bp, home).to_csv(root / GAP_FILE, index=False),
                {"white-coat gap: one row per patient in either table"}, "has no patient IDs")
        mistake("gap-ids-lost-sign-flipped", lambda root: (-gap(bp, home)).to_csv(root / GAP_FILE, index=False), GAP,
                "wrong sign")
        mistake("summary-ids-lost", lambda root: summarize(bp).to_csv(root / SUMMARY_FILE, index=False),
                {"visit summary: one row per visit"}, "index=False leaves them out")
        mistake("summary-ids-lost-mean-is-max",
                lambda root: summarize(bp).assign(mean=bp[READINGS].max()).to_csv(root / SUMMARY_FILE, index=False),
                {"visit summary: one row per visit", "visit summary: mean"}, "sbp_baseline has 171.0")
        mistake("counts-ids-lost", lambda root: bp["clinic"].value_counts().to_csv(root / COUNTS_FILE, index=False),
                {"clinic counts: patients per clinic"}, "index=False leaves them out")
        mistake("counts-ids-lost-sorted-by-name",
                lambda root: bp["clinic"].value_counts().sort_index().to_csv(root / COUNTS_FILE, index=False), COUNTS,
                "lists a count of 4 before 5")

        # One mistake in bp_loaded.csv carries into every later file, so only the loaded-table check is lost.
        mistake("cascade-nrows", lambda root: solve(root, load(nrows=10)), {"bp loaded: all 14 patients once"},
                "missing P111, P112, P113 and P114")
        mistake("cascade-no-index-col", lambda root: solve(root, load(index_col=None)),
                {"white-coat gap: one row per patient in either table"}, "bp or home was read")
        mistake("parquet-id-misnamed",
                lambda root: followup.reset_index().rename(columns={"patient_id": "pid"})
                .to_parquet(root / PARQUET_FILE, index=False),
                {"follow-up Parquet: same columns"}, "rename it to patient_id")
        mistake("gap-missing-text",
                lambda root: gap(bp, home).to_csv(root / GAP_FILE, na_rep="missing"),
                {"white-coat gap: gaps matched by patient"}, "remove the replacement text such as missing")

        # Placeholders for a missing value, headers with spaces or CamelCase, and decimal commas cost nothing.
        for marker in ("-", ".", "N/A"):
            def placeholder(root: Path, marker: str = marker) -> None:
                bp.to_csv(root / LOADED_FILE, na_rep=marker)
                gap(bp, home).to_csv(root / GAP_FILE, na_rep=marker)
                followup.to_csv(root / FOLLOWUP_FILE, na_rep=marker)

            full_marks(f"missing-as-{marker}", placeholder)

        def reheader(style) -> Callable:
            def change(root: Path) -> None:
                for name in (LOADED_FILE, SUMMARY_FILE, COUNTS_FILE, FOLLOWUP_FILE, GAP_FILE):
                    edit(root, name, lambda text: ",".join(style(c) for c in text.split("\n", 1)[0].split(","))
                         + "\n" + text.split("\n", 1)[1])
            return change

        full_marks("headers-title-case", reheader(lambda c: c.replace("_", " ").title()))
        full_marks("headers-camel-case", reheader(lambda c: c.title().replace("_", "")))
        full_marks("headers-hyphens", reheader(lambda c: c.replace("_", "-")))
        full_marks("tab-decimal-comma", lambda root: followup.to_csv(root / FOLLOWUP_FILE, sep="\t", decimal=","))
        full_marks("quoted-decimal-comma", lambda root: followup.to_csv(root / FOLLOWUP_FILE, decimal=",", quoting=1))
        mistake("gap-written-twice", lambda root: written_twice(gap(bp, home), root / GAP_FILE),
                {"white-coat gap: one row per patient in either table"}, "holds the same table 2 times")

        # The printed report says each shared fix once and ends by naming the checks left to fix.
        report = printed(empty)
        assert report.count("is missing; run the Task") == 6, report
        assert report.rstrip().endswith(
            "Left to fix (100 points): bp loaded (all 6 checks); visit summary (all 4 checks); clinic counts "
            "(all 2 checks); follow-up list (all 9 checks); follow-up Parquet (all 2 checks); white-coat gap "
            "(all 2 checks)."), report
        report = printed(workspace / "sentinel-kept")
        assert report.rstrip().endswith("Score: 96/100\nLeft to fix (4 points): bp loaded: -999 read as missing."), report
        assert printed(correct).rstrip().endswith("Score: 100/100\nAll checks passed."), printed(correct)

        # The notebook's last cell calls run_checks(): same report dict and same text as the command line.
        import contextlib
        import io
        from check_assignment import run_checks
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            report = run_checks(correct)
        assert report["score"] == report["max-score"] == 100, report
        assert buffer.getvalue() == printed(correct), (buffer.getvalue(), printed(correct))

        single_mistakes = len(mistakes)

    # A fresh handout prints exactly the "Before Task 2" example README.md shows.
    readme = (HANDOUT / "README.md").read_text(encoding="utf-8")
    checks_md = (HANDOUT / "CHECKS.md").read_text(encoding="utf-8")
    assert "(CHECKS.md)" in readme, "README.md no longer links CHECKS.md"
    shown = re.search(r"Before Task 2, for example, the loaded-table checks report:\n\n( *)```text\n(.*?)```", checks_md,
                      re.DOTALL)
    assert shown is not None, "CHECKS.md no longer shows the Before Task 2 example"
    example = re.sub(f"(?m)^{shown.group(1)}", "", shown.group(2))
    handout_report = subprocess.run(
        [sys.executable, "-B", "check_assignment.py"], cwd=HANDOUT, capture_output=True, text=True, check=False
    ).stdout
    assert example in handout_report, (example, handout_report)

    # README's checkpoint order and completion contract agree with the checks.
    listed = re.search(r"seven patients in this order of `patient_id`: ([A-Z0-9, ]+)\.", readme).group(1).split(", ")
    assert listed == program_order(), (listed, program_order())
    contract = {
        name: int(points)
        for name, points in re.findall(r"^\| `output/[^|]+\| [^|]+\| ([^|]+) \| (\d+) \|$", checks_md, re.MULTILINE)
    }
    assert contract == CHECK_POINTS, (contract, CHECK_POINTS)

    print(
        "Assignment 04 checks: empty, handout, correct, loosely formatted, separated, UTF-16 and renamed, "
        f"reset-index, rounded, transposed, describe(), and globally ranked submissions score as intended, "
        f"and so do {single_mistakes} submissions with mistakes."
    )


def run_handout() -> None:
    """The handout's data match the checks, and it ships the course's checks unchanged."""
    lines = DATA.read_text(encoding="utf-8").splitlines()
    assert tuple(lines[1].split(";")) == UNITS_ROW, "data/bp_followup.csv's units row differs from UNITS_ROW"
    export = pd.read_csv(DATA, sep=";", skiprows=[1], usecols=USECOLS, index_col="patient_id")
    held = {
        patient: (row.clinic, row.age, *(None if value == -999 else value for value in row[READINGS]))
        for patient, row in export.iterrows()
    }
    assert held == BP_EXPORT, "data/bp_followup.csv differs from BP_EXPORT in _value_checks.py"
    homes = pd.read_csv(HOME, index_col="patient_id")["home_sbp_week8"].to_dict()
    assert homes == HOME_SBP and list(homes) == list(HOME_SBP), "data/home_bp.csv differs from HOME_SBP"

    workflow = (HANDOUT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert re.search(r'^  CHECKS_PATH: "04/assignment_checks"$', workflow, re.MULTILINE), (
        "tests.yml does not download from 04/assignment_checks"
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
            f"04/assignment/{name} differs from 04/assignment_checks/{name}; copy the course-owned file over it"
        )

    # The handout's only Python files are the checks; the self-test and its answers stay here.
    handout_python = {path.relative_to(HANDOUT).as_posix() for path in HANDOUT.rglob("*.py")
                      if ".venv" not in path.parts}
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

    print(f"Assignment 04 handout: data match the checks, pyproject.toml and uv.lock pin the course packages, "
          f"and all {len(files)} check files match 04/assignment_checks byte for byte.")


if __name__ == "__main__":
    run()
    run_handout()
