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

        duplicate = variant(workspace, "duplicate-header", correct)
        edit(duplicate, FRIDGE_FILE, lambda text: "\n".join(
            line + "," + line.split(",")[1] for line in text.splitlines()))
        assert lost(duplicate) == set(), lost(duplicate)
        edit(duplicate, FRIDGE_FILE, lambda text: text.replace("FRG-102,3.8,", "FRG-102,99,"))
        assert lost(duplicate) == {"fridge block: am_temp_c values"}, lost(duplicate)

        duplicate_last = variant(workspace, "duplicate-header-wrong-later", correct)
        edit(duplicate_last, FRIDGE_FILE, lambda text: "\n".join(
            line + "," + (line.split(",")[1] if number == 0 else "99")
            for number, line in enumerate(text.splitlines())))
        assert lost(duplicate_last) == {"fridge block: am_temp_c values"}, lost(duplicate_last)
        for label, extra in (("unknown-fridge-row", "WRONG,99,99"),
                             ("repeated-fridge-row", "FRG-102,3.8,6.2")):
            root = variant(workspace, label, correct)
            edit(root, FRIDGE_FILE, lambda text: text.rstrip() + "\n" + extra + "\n")
            assert lost(root) == {"fridge block: rows FRG-102 and FRG-103"}, lost(root)

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
                {"fridge block: pm_temp_c column"}, "no pm_temp_c column")

        # A reading column saved under another name costs only its name check; its values are still graded.
        def renamed_columns(*names: str):
            def change(root: Path) -> None:
                block = fridge.loc["FRG-102":"FRG-103"].copy()
                block.columns = list(names)
                block.to_csv(root / FRIDGE_FILE)
            return change

        mistake("fridge-am-renamed", renamed_columns("am_tmp_c", "pm_temp_c"), {"fridge block: am_temp_c column"},
                'a column named am_tmp_c where Task 2.1 names it am_temp_c; build fridge_log with '
                'columns=["am_temp_c", "pm_temp_c"]')
        mistake("fridge-both-renamed", renamed_columns("am_temp", "pm_temp"),
                {"fridge block: am_temp_c column", "fridge block: pm_temp_c column"}, "Task 2.1 names it")
        mistake("fridge-one-wrong-value",
                lambda root: edit(root, FRIDGE_FILE, lambda text: text.replace("FRG-103,5.0,", "FRG-103,5.5,")),
                {"fridge block: am_temp_c values"}, "FRG-103 has 5.5")
        mistake("fridge-missing", lambda root: (root / FRIDGE_FILE).unlink(), FRIDGE_CHECKS)
        mistake("fridge-nul-bytes", lambda root: (root / FRIDGE_FILE).write_bytes(b"\xff\xfe\x00"),
                FRIDGE_CHECKS, "embedded NUL")
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

        # Without the index, two rows are the block by position when a row's readings name no fridge,
        # so wrong readings cost the value checks, not the rows check.
        def unlabeled_rows(*readings: tuple[float, float]):
            def change(root: Path) -> None:
                pd.DataFrame(list(readings), columns=["am_temp_c", "pm_temp_c"]).to_csv(root / FRIDGE_FILE, index=False)
            return change

        unlabeled_values = {"fridge block: fridge_id index column", "fridge block: am_temp_c values",
                            "fridge block: pm_temp_c values"}
        mistake("fridge-index-dropped-row-wrong", unlabeled_rows((3.8, 6.2), (5.5, 7.9)), unlabeled_values)
        assert "FRG-103 has 7.9" in detail(workspace / "fridge-index-dropped-row-wrong", "fridge block: pm_temp_c values")
        mistake("fridge-index-dropped-columns-swapped", unlabeled_rows((6.2, 3.8), (7.4, 5.0)), unlabeled_values)
        mistake("fridge-index-dropped-wrong-slice", unlabeled_rows((4.1, 5.6), (3.8, 6.2)),
                {"fridge block: fridge_id index column", "fridge block: rows FRG-102 and FRG-103"})
        assert "missing FRG-103; it also holds FRG-101" in detail(workspace / "fridge-index-dropped-wrong-slice",
                                                                  "fridge block: rows FRG-102 and FRG-103")

        # A file saved in the assignment folder instead of output/ is named in the fix.
        def saved_at_root(root: Path) -> None:
            (root / FRIDGE_FILE).rename(root / "fridge_block.csv")

        mistake("fridge-saved-at-root", saved_at_root, FRIDGE_CHECKS,
                "the assignment folder itself has fridge_block.csv; in Task 2.2, save with "
                "label_block.to_csv(FRIDGE_OUTPUT_PATH)")

        # Task 2.2's save line copied into Task 3.2 writes the supply lines over the fridge block.
        def supplies_over_fridge(root: Path) -> None:
            (root / SUPPLIES_FILE).replace(root / FRIDGE_FILE)

        mistake("supplies-saved-over-fridge", supplies_over_fridge, FRIDGE_CHECKS | SUPPLY_CHECKS,
                "holds the supply order lines")
        for name in FRIDGE_CHECKS | SUPPLY_CHECKS:
            assert "correct the path in the call that differs" in detail(workspace / "supplies-saved-over-fridge", name)

        # Another labeled object saved in Task 2.2: its labels are not fridge IDs, and it has no readings.
        def other_series(root: Path) -> None:
            latest_by_area = pd.Series([4.5, 5.0, 3.5, 6.0], name="temp_c",
                                       index=["pharmacy", "pediatrics", "family_medicine", "urgent_care"])
            latest_by_area.to_csv(root / FRIDGE_FILE)

        mistake("fridge-another-object", other_series, FRIDGE_CHECKS, "holds another table")
        for name in FRIDGE_CHECKS:
            assert "save the fridge block with label_block.to_csv(FRIDGE_OUTPUT_PATH)" in detail(
                workspace / "fridge-another-object", name), name

        # Row labels that are not fridge IDs cost only the fridge_id check: the readings still name the rows.
        def wrong_labels(root: Path) -> None:
            block = fridge.loc["FRG-102":"FRG-103"].copy()
            block.index = ["F2", "F3"]
            block.to_csv(root / FRIDGE_FILE)

        mistake("fridge-index-wrong-labels", wrong_labels, {"fridge block: fridge_id index column"},
                "its row labels (F2 and F3) are not fridge IDs")

        # A column saved with no header is named in the list of columns, never left blank.
        def unnamed_am_only(root: Path) -> None:
            block = fridge.loc["FRG-102":"FRG-103", ["am_temp_c"]].copy()
            block.index.name = None
            block.to_csv(root / FRIDGE_FILE)

        mistake("fridge-index-unnamed-am-only", unnamed_am_only,
                {"fridge block: fridge_id index column", "fridge block: pm_temp_c column"})
        assert "(its columns are a column with no header and am_temp_c)" in detail(
            workspace / "fridge-index-unnamed-am-only", "fridge block: pm_temp_c column")

        # A header or unknown labels alone do not establish omitted reading values.
        for label, text in (("header-only", "fridge_id\n"), ("unknown-id", "fridge_id\nwrong\n")):
            mistake("fridge-missing-readings-" + label,
                    lambda root, text=text: (root / FRIDGE_FILE).write_text(text),
                    {"fridge block: rows FRG-102 and FRG-103", "fridge block: am_temp_c column",
                     "fridge block: pm_temp_c column", "fridge block: am_temp_c values",
                     "fridge block: pm_temp_c values"})

        source = pd.read_csv(HANDOUT / "data" / "supply_order.csv")

        def rewrite(root: Path, frame: pd.DataFrame) -> None:
            frame.to_csv(root / SUPPLIES_FILE, index=False)

        def with_totals(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            frame["line_total_usd"] = frame["quantity"] * frame["unit_price_usd"]
            return frame.sort_values(by=["line_total_usd", "item_id"], ascending=[False, True])

        line_checks = {f"selected supplies: line {item_id}" for item_id in expected_selection()}
        mistake("supplies-no-mask", lambda root: rewrite(root, with_totals(source)),
                {"selected supplies: no other lines"}, "C1022, C1407 and C1560, which have quantity 1")
        mistake("supplies-mask-greater-than",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] > 2])),
                {"selected supplies: line C1833", "selected supplies: line C4105", "selected supplies: line C2655"},
                '>= 2 keeps it, while > 2 drops it')
        mistake("supplies-no-tiebreak",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] >= 2]).sort_index()
                                     .sort_values("line_total_usd", ascending=False)),
                {"selected supplies: ties in item_id order"}, "C3150 before C1833, but both have the line total 57.00")
        mistake("supplies-ascending",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] >= 2])
                                     .sort_values(by=["line_total_usd", "item_id"])),
                {"selected supplies: highest line total first"}, "C2318 (line total 12.75) comes before")
        mistake("supplies-both-descending",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] >= 2])
                                     .sort_values(by=["line_total_usd", "item_id"], ascending=False)),
                {"selected supplies: ties in item_id order"}, "the tie goes to the smaller item_id")
        mistake("supplies-one-wrong-total",
                lambda root: edit(root, SUPPLIES_FILE, lambda text: text.replace("12.75", "12.5")),
                {"selected supplies: line totals"}, "C2318's line_total_usd is 12.5, expected 3 * 4.25 = 12.75")
        mistake("supplies-one-wrong-quantity",
                lambda root: edit(root, SUPPLIES_FILE, lambda text: text.replace(",6,9.5,", ",7,9.5,")),
                {"selected supplies: line C3150"}, "C3150's quantity is 7, expected 6")
        mistake("supplies-sum-not-product",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] >= 2]).assign(
                    line_total_usd=lambda frame: frame["quantity"] + frame["unit_price_usd"])),
                {"selected supplies: line totals"}, "C1833's line_total_usd is 30.5, expected 2 * 28.50 = 57.00")
        mistake("supplies-extra-column",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] >= 2]).assign(flag="bulk")),
                {"selected supplies: no extra columns"}, "also has flag")
        mistake("supplies-total-misnamed",
                lambda root: edit(root, SUPPLIES_FILE, lambda text: text.replace("line_total_usd", "line_total", 1)),
                {"selected supplies: line_total_usd column"}, "a column named line_total where")
        mistake("supplies-total-renamed-cost",
                lambda root: edit(root, SUPPLIES_FILE, lambda text: text.replace("line_total_usd", "line_cost_usd", 1)),
                {"selected supplies: line_total_usd column"}, "a column named line_cost_usd where")
        mistake("supplies-quantity-misnamed",
                lambda root: edit(root, SUPPLIES_FILE, lambda text: text.replace("quantity", "qty", 1)),
                {"selected supplies: quantity column"}, "a column named qty where")
        mistake("supplies-no-total",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] >= 2]).drop(
                    columns=["line_total_usd"])),
                {"selected supplies: line_total_usd column"}, None)
        mistake("supplies-header-only",
                lambda root: rewrite(root, with_totals(source.loc[source["quantity"] > 10])),
                line_checks | {"selected supplies: line totals", "selected supplies: highest line total first",
                               "selected supplies: ties in item_id order"})
        for name in line_checks | {"selected supplies: line totals"}:
            assert "has a header but no order lines" in detail(workspace / "supplies-header-only", name), name

        # sort_values() without assigning the result back leaves the lines in data order.
        def not_reassigned(root: Path) -> None:
            selected = source.loc[source["quantity"] >= 2].copy()
            selected["line_total_usd"] = selected["quantity"] * selected["unit_price_usd"]
            selected.sort_values(by=["line_total_usd", "item_id"], ascending=[False, True])
            rewrite(root, selected)

        mistake("supplies-not-reassigned", not_reassigned,
                {"selected supplies: highest line total first", "selected supplies: ties in item_id order"},
                "so assign the result back: selected_supplies = selected_supplies.sort_values(")
        mistake("supplies-missing", lambda root: (root / SUPPLIES_FILE).unlink(), SUPPLY_CHECKS)
        mistake("supplies-tabs-one-wrong-total",
                lambda root: pd.read_csv(correct / SUPPLIES_FILE).replace({12.75: 12.5}).to_csv(
                    root / SUPPLIES_FILE, sep="\t", index=False),
                {"selected supplies: line totals"}, "C2318's line_total_usd is 12.5")

        # The printed report says each shared fix once and ends by naming the checks left to fix.
        def printed(root: Path) -> str:
            return subprocess.run(
                [sys.executable, "-B", str(CHECKS / "check_assignment.py"), str(root)],
                capture_output=True, text=True, check=False,
            ).stdout

        report = printed(empty)
        assert report.count("is missing; run the Task") == 2, report
        assert "[FIX ]  0/10 fridge block: rows FRG-102 and FRG-103  (same fix as above)" in report, report
        assert report.rstrip().endswith(
            "Left to fix (100 points): fridge block (all 6 checks); selected supplies (all 19 checks)."), report
        report = printed(workspace / "supplies-one-wrong-total")
        assert report.rstrip().endswith("Score: 91/100\nLeft to fix (9 points): selected supplies: line totals."), report
        report = printed(workspace / "supplies-mask-greater-than")
        assert report.rstrip().endswith(
            "Left to fix (9 points): selected supplies: line C1833, line C4105 and line C2655."), report
        report = printed(workspace / "supplies-quantity-misnamed")
        assert report.rstrip().endswith("Left to fix (2 points): selected supplies: quantity column."), report
        report = printed(workspace / "supplies-no-total")
        assert report.rstrip().endswith(
            "Left to fix (2 points): selected supplies: line_total_usd column."), report
        report = printed(workspace / "fridge-am-renamed")
        assert report.rstrip().endswith("Left to fix (2 points): fridge block: am_temp_c column."), report
        report = printed(workspace / "supplies-header-only")
        assert report.count("has a header but no order lines") == 1, report
        report = printed(workspace / "supplies-not-reassigned")
        assert report.count("so assign the result back") == 1, report
        report = printed(workspace / "supplies-saved-over-fridge")
        assert report.count("correct the path in the call that differs") == 2, report
        assert "saves the fridge IDs" not in report, report
        assert printed(correct).rstrip().endswith("Score: 100/100\nAll checks passed."), printed(correct)

        single_mistakes = len(mistakes)

    # A fresh handout prints exactly the "Before Task 2" example README.md shows.
    readme = (HANDOUT / "README.md").read_text(encoding="utf-8")
    shown = re.search(r"Before Task 2, for example, the fridge block checks report:\n\n```text\n(.*?)```", readme, re.DOTALL)
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
    # The unsorted-file message recognizes the data file's order, so SUPPLY_ORDER keeps it.
    assert list(held) == list(SUPPLY_ORDER), "SUPPLY_ORDER lists the lines in another order than data/supply_order.csv"
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

    # Setup follows Lecture 03: uv sync builds the notebook's environment from pyproject.toml and uv.lock.
    assert not (HANDOUT / "requirements.txt").exists(), "the handout ships pyproject.toml and uv.lock, not requirements.txt"
    project = (HANDOUT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (HANDOUT / "uv.lock").read_text(encoding="utf-8")
    for name, version in (("numpy", "2.3.3"), ("pandas", "3.0.5"), ("ipykernel", "6.29.5")):
        assert f'"{name}=={version}"' in project, f"pyproject.toml does not pin {name}=={version}"
        assert f'name = "{name}"\nversion = "{version}"' in lock, f"uv.lock does not lock {name} {version}; run uv lock"
    assert (HANDOUT / ".python-version").read_text(encoding="utf-8").strip() == "3.13", ".python-version should name 3.13"

    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, "clear the handout notebook's outputs"

    print(f"Assignment 04 handout: data and notebook match the checks, pyproject.toml and uv.lock pin the course "
          f"packages, and all {len(files)} check files match 04/assignment_checks byte for byte.")


if __name__ == "__main__":
    run()
    run_handout()
