"""Development regression checks for the Assignment 04 checks.

Answers the assignment with pandas the way a student would, builds
submissions in ignored `scratch/`, and confirms what each kind of submission
scores: a correct one scores 100 however it is formatted, and each mistake
costs only its own check. It also confirms that the values the checks hold
match the handout's data and notebook, and that the handout in
`04/assignment/` ships the course-owned checks byte for byte. Nothing here
reads or runs student code.

    uv run --python 3.13 --with pandas==3.0.5 python 04/assignment_checks/_grader_selftest/run.py
"""

from pathlib import Path
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


CHECKS = Path(__file__).resolve().parents[1]
HANDOUT = CHECKS.parent / "assignment"
SCRATCH = CHECKS.parents[1] / "scratch"
sys.path.insert(0, str(CHECKS))
from _value_checks import (  # noqa: E402
    CHECKS as VALUE_CHECKS,
    FRIDGE_FILE,
    FRIDGE_READINGS,
    SUPPLIES_FILE,
    SUPPLY_ORDER,
    expected_selection,
)
from grading import POINTS, grade_submission  # noqa: E402

CHECK_POINTS = {check.name: points for check, points in zip(VALUE_CHECKS, POINTS, strict=True)}
FRIDGE_CHECKS = {name for name in CHECK_POINTS if name.startswith("fridge block")}
SUPPLY_CHECKS = {name for name in CHECK_POINTS if name.startswith("selected supplies")}


def supplied_fridge_log() -> pd.DataFrame:
    """The Task 2.1 DataFrame, built from the array the handout notebook supplies."""
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "fridge_readings = np.array(" in "".join(cell["source"])
    )
    literal = re.search(r"fridge_readings = np\.array\((.*?)\n\)", source, re.DOTALL).group(1)
    fridge_log = pd.DataFrame(
        np.array(ast.literal_eval(literal.strip())),
        index=["FRG-101", "FRG-102", "FRG-103", "FRG-104"],
        columns=["am_temp_c", "pm_temp_c"],
    )
    fridge_log.index.name = "fridge_id"
    return fridge_log


def solve(root: Path) -> None:
    """Write both artifacts with the lecture's pandas methods, as the README asks."""
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    fridge_log = supplied_fridge_log()
    label_block = fridge_log.loc["FRG-102":"FRG-103", ["am_temp_c", "pm_temp_c"]]
    assert label_block.equals(fridge_log.iloc[1:3, 0:2])
    label_block.to_csv(output / "fridge_block.csv")

    supplies = pd.read_csv(HANDOUT / "data" / "supply_order.csv")
    quantity_at_least_two = supplies["quantity"] >= 2
    selected = supplies.loc[quantity_at_least_two, ["item_id", "item", "quantity", "unit_price_usd"]].copy()
    selected["line_total_usd"] = selected["quantity"] * selected["unit_price_usd"]
    selected = selected.sort_values(by=["line_total_usd", "item_id"], ascending=[False, True])
    selected.to_csv(output / "selected_supplies.csv", index=False)


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


def edit(root: Path, name: str, change) -> None:
    path = root / name
    path.write_text(change(path.read_text(encoding="utf-8")), encoding="utf-8")


def variant(workspace: Path, label: str, base: Path) -> Path:
    root = workspace / label
    shutil.copytree(base, root)
    return root


def run() -> None:
    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a04-selftest-") as temporary:
        workspace = Path(temporary)

        empty = workspace / "empty"
        empty.mkdir()
        assert sum(scores(empty).values()) == 0
        assert checker_report(CHECKS, empty)["score"] == 0

        fresh = workspace / "handout"
        shutil.copytree(HANDOUT, fresh, ignore=shutil.ignore_patterns("__pycache__", ".venv", ".pytest_cache"))
        assert sum(scores(fresh).values()) == 0

        correct = workspace / "correct"
        correct.mkdir()
        solve(correct)
        assert lost(correct) == set(), {name: detail(correct, name) for name in lost(correct)}
        for checks in (CHECKS, HANDOUT):
            assert checker_report(checks, correct)["score"] == 100, checks
            assert checker_report(checks, empty) == checker_report(CHECKS, empty), checks

        # Formatting that changes no value never costs points.
        loose = variant(workspace, "loose", correct)
        edit(loose, FRIDGE_FILE, lambda text: "\r\n".join(
            "  " + ", ".join(cell.lower() for cell in line.split(",")) + "  " for line in text.splitlines()))
        supplies = pd.read_csv(correct / SUPPLIES_FILE)
        supplies = supplies[["line_total_usd", "item", "unit_price_usd", "item_id", "quantity"]]
        supplies["item"] = supplies["item"].str.upper()
        supplies["item_id"] = supplies["item_id"].str.lower()
        supplies["quantity"] = supplies["quantity"].astype(float)
        supplies.to_csv(loose / SUPPLIES_FILE, float_format="%.2f", quoting=1)  # row numbers kept, every cell quoted
        edit(loose, SUPPLIES_FILE, lambda text: text.replace("\n", " \r\n").rstrip())
        assert lost(loose) == set(), {name: detail(loose, name) for name in lost(loose)}

        encoded = variant(workspace, "utf16-bom-renamed", correct)
        text = (encoded / FRIDGE_FILE).read_text(encoding="utf-8")
        (encoded / FRIDGE_FILE).write_bytes(text.encode("utf-16"))
        text = (encoded / SUPPLIES_FILE).read_text(encoding="utf-8")
        (encoded / SUPPLIES_FILE).unlink()
        (encoded / "output" / "Selected_Supplies.csv").write_bytes(("\ufeff" + text).encode("utf-8"))
        assert lost(encoded) == set(), {name: detail(encoded, name) for name in lost(encoded)}

        reset = variant(workspace, "reset-index", correct)
        supplied_fridge_log().loc["FRG-102":"FRG-103"].reset_index().to_csv(reset / FRIDGE_FILE)
        # reset_index() and then to_csv() with the index kept: two row-number columns.
        pd.read_csv(correct / SUPPLIES_FILE).reset_index().to_csv(reset / SUPPLIES_FILE)
        assert lost(reset) == set(), {name: detail(reset, name) for name in lost(reset)}

        # Cells separated by semicolons, with a European spreadsheet's decimal commas, or by tabs.
        separated = variant(workspace, "semicolons-and-tabs", correct)
        pd.read_csv(correct / FRIDGE_FILE).to_csv(separated / FRIDGE_FILE, sep=";", decimal=",", index=False)
        pd.read_csv(correct / SUPPLIES_FILE).to_csv(separated / SUPPLIES_FILE, sep="\t", index=False)
        assert "FRG-102;3,8;6,2" in (separated / FRIDGE_FILE).read_text(encoding="utf-8")
        assert lost(separated) == set(), {name: detail(separated, name) for name in lost(separated)}

        # Each mistake costs only the check it gets wrong.
        mistakes = []

        def mistake(label: str, change, expected: set[str], hint: str | None = None) -> None:
            root = variant(workspace, label, correct)
            mistakes.append(label)
            change(root)
            assert lost(root) == expected, (label, lost(root), {name: detail(root, name) for name in lost(root)})
            if hint is not None:
                named = next(iter(expected))
                assert hint in detail(root, named), (label, detail(root, named))
            assert checker_report(HANDOUT, root) == checker_report(CHECKS, root), label

        fridge = supplied_fridge_log()
        mistake("fridge-index-dropped",
                lambda root: fridge.loc["FRG-102":"FRG-103"].to_csv(root / FRIDGE_FILE, index=False),
                {"fridge block: fridge_id index column"}, "leave out index=False")

        def unnamed_index(root: Path) -> None:
            block = fridge.loc["FRG-102":"FRG-103"].copy()
            block.index.name = None
            block.to_csv(root / FRIDGE_FILE)

        mistake("fridge-index-unnamed", unnamed_index, {"fridge block: fridge_id index column"},
                'fridge_log.index.name = "fridge_id"')
        mistake("fridge-slice-short", lambda root: fridge.iloc[1:2].to_csv(root / FRIDGE_FILE),
                {"fridge block: rows FRG-102 and FRG-103"}, "missing FRG-103")
        mistake("fridge-slice-long", lambda root: fridge.loc["FRG-101":"FRG-103"].to_csv(root / FRIDGE_FILE),
                {"fridge block: rows FRG-102 and FRG-103"}, "also holds FRG-101")
        mistake("fridge-am-only", lambda root: fridge.loc["FRG-102":"FRG-103", ["am_temp_c"]].to_csv(root / FRIDGE_FILE),
                {"fridge block: pm_temp_c values"}, "no pm_temp_c column")
        mistake("fridge-one-wrong-value",
                lambda root: edit(root, FRIDGE_FILE, lambda text: text.replace("FRG-103,5.0,", "FRG-103,5.5,")),
                {"fridge block: am_temp_c values"}, "FRG-103 has 5.5")
        mistake("fridge-missing", lambda root: (root / FRIDGE_FILE).unlink(), FRIDGE_CHECKS)
        mistake("fridge-semicolons-one-wrong-value",
                lambda root: pd.read_csv(correct / FRIDGE_FILE).replace({5.0: 5.5}).to_csv(
                    root / FRIDGE_FILE, sep=";", decimal=",", index=False),
                {"fridge block: am_temp_c values"}, "FRG-103 has 5.5")

        # Two mistakes at once still cost only their own checks: without the index, a row
        # with one wrong reading is still named by its other reading.
        def unlabeled_wrong_value(root: Path) -> None:
            block = fridge.loc["FRG-102":"FRG-103"].copy()
            block.loc["FRG-103", "am_temp_c"] = 5.5
            block.to_csv(root / FRIDGE_FILE, index=False)

        mistake("fridge-index-dropped-and-wrong-value", unlabeled_wrong_value,
                {"fridge block: fridge_id index column", "fridge block: am_temp_c values"})
        assert "FRG-103 has 5.5" in detail(workspace / "fridge-index-dropped-and-wrong-value",
                                           "fridge block: am_temp_c values")

        source = pd.read_csv(HANDOUT / "data" / "supply_order.csv")

        def rewrite(root: Path, frame: pd.DataFrame) -> None:
            frame.to_csv(root / SUPPLIES_FILE, index=False)

        def with_totals(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            frame["line_total_usd"] = frame["quantity"] * frame["unit_price_usd"]
            return frame.sort_values(by=["line_total_usd", "item_id"], ascending=[False, True])

        mistake("supplies-no-mask", lambda root: rewrite(root, with_totals(source)),
                {"selected supplies: rows with quantity 2 or more"}, "C1022 (quantity 1)")
        mistake("supplies-mask-greater-than",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] > 2])),
                {"selected supplies: rows with quantity 2 or more"}, "missing C1833")
        mistake("supplies-no-tiebreak",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] >= 2]).sort_index()
                                     .sort_values("line_total_usd", ascending=False)),
                {"selected supplies: sort order"}, "C3150 comes before C1833, but both total 57.00")
        mistake("supplies-ascending",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] >= 2])
                                     .sort_values(by=["line_total_usd", "item_id"])),
                {"selected supplies: sort order"})
        mistake("supplies-one-wrong-total",
                lambda root: edit(root, SUPPLIES_FILE, lambda text: text.replace("12.75", "12.5")),
                {"selected supplies: values and line totals"}, "C2318's line_total_usd is 12.5")
        mistake("supplies-sum-not-product",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] >= 2]).assign(
                    line_total_usd=lambda frame: frame["quantity"] + frame["unit_price_usd"])),
                {"selected supplies: values and line totals"})
        mistake("supplies-extra-column",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] >= 2]).assign(flag="bulk")),
                {"selected supplies: columns"}, "also has flag")
        mistake("supplies-total-misnamed",
                lambda root: edit(root, SUPPLIES_FILE, lambda text: text.replace("line_total_usd", "line_total", 1)),
                {"selected supplies: columns"}, "rather than line_total")
        mistake("supplies-no-total",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] >= 2]).drop(
                    columns=["line_total_usd"])),
                {"selected supplies: columns", "selected supplies: values and line totals"}, None)
        mistake("supplies-missing", lambda root: (root / SUPPLIES_FILE).unlink(), SUPPLY_CHECKS)
        mistake("supplies-tabs-one-wrong-total",
                lambda root: pd.read_csv(correct / SUPPLIES_FILE).replace({12.75: 12.5}).to_csv(
                    root / SUPPLIES_FILE, sep="\t", index=False),
                {"selected supplies: values and line totals"}, "C2318's line_total_usd is 12.5")

        single_mistakes = len(mistakes)

    # A fresh handout prints exactly the "Before Task 2" example README.md shows.
    readme = (HANDOUT / "README.md").read_text(encoding="utf-8")
    shown = re.search(r"Before Task 2, for example, the first check reports:\n\n```text\n(.*?)```", readme, re.DOTALL)
    assert shown is not None, "README.md no longer shows the Before Task 2 example"
    printed = subprocess.run(
        [sys.executable, "-B", "check_assignment.py"], cwd=HANDOUT, capture_output=True, text=True, check=False
    ).stdout
    assert shown.group(1) in printed, (shown.group(1), printed)

    # README's checkpoint order and completion contract agree with the checks.
    order = re.search(r"in this order of `item_id`: ([A-Z0-9, ]+)\.", readme).group(1).split(", ")
    assert order == expected_selection(), (order, expected_selection())
    contract = {
        name: int(points)
        for name, points in re.findall(r"^\| `output/[^|]+\| [^|]+\| ([^|]+) \| (\d+) \|$", readme, re.MULTILINE)
    }
    assert contract == CHECK_POINTS, (contract, CHECK_POINTS)

    print(
        "Assignment 04 checks: empty, handout, correct, loosely formatted, UTF-16 and renamed, reset-index, "
        f"semicolon- and tab-separated, and {single_mistakes} submissions with mistakes all score as intended."
    )


def run_handout() -> None:
    """The handout's data and notebook match the checks, and it ships the course's checks unchanged."""
    supplies = pd.read_csv(HANDOUT / "data" / "supply_order.csv")
    held = {row.item_id: (row.item, row.quantity, row.unit_price_usd) for row in supplies.itertuples()}
    assert held == SUPPLY_ORDER, "data/supply_order.csv differs from SUPPLY_ORDER in _value_checks.py"
    readings = supplied_fridge_log().to_dict(orient="index")
    assert readings == FRIDGE_READINGS, "the notebook's fridge_readings differ from FRIDGE_READINGS"

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
    handout_python = {path.relative_to(HANDOUT).as_posix() for path in HANDOUT.rglob("*.py")}
    assert handout_python == set(files) - {".github/test/requirements.txt"}, sorted(handout_python)
    assert not (HANDOUT / "_grader_selftest").exists()

    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, "clear the handout notebook's outputs"

    print(f"Assignment 04 handout: data and notebook match the checks, and all {len(files)} check files "
          "match 04/assignment_checks byte for byte.")


if __name__ == "__main__":
    run()
    run_handout()
