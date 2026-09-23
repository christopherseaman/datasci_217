"""Regression checks for the Assignment 02 value checks.

Builds submissions in ignored `scratch/` and confirms each check scores what it
should. Nothing here reads or runs student code.
"""

from pathlib import Path
import shutil
import sys
import tempfile


CHECKS = Path(__file__).resolve().parents[1]
REPO = CHECKS.parents[1]
ASSIGNMENT = REPO / "02" / "assignment"
sys.path.insert(0, str(CHECKS))
from _value_checks import (  # noqa: E402
    CHECKS as VALUE_CHECKS,
    DATA_FILE,
    DATA_FINGERPRINT,
    _fingerprint,
    load_supplied_encounters,
)
from grading import POINTS, grade_submission  # noqa: E402

CHECK_POINTS = {check.name: points for check, points in zip(VALUE_CHECKS, POINTS, strict=True)}

README = (
    "# Assignment\n\n## Project description\n\n"
    "Summarizes a week of clinic encounters and saves the follow-up call list.\n\n"
    "## Run\n\npython3 clinic_report.py\n"
)
GITIGNORE = "# Python cache\n__pycache__/\n*.pyc\n"
REASON = "Reason: stage 2 hypertension gets the first calls from this week's clinic list.\n"


def scores(root: Path) -> dict[str, int]:
    result = grade_submission(root)
    assert result["schema"] == "datasci217/grading-result/v1", result["schema"]
    assert result["max-score"] == sum(POINTS) == 100, result["max-score"]
    return {test["test-name"]: test["score"] for test in result["tests"]}


def write_submission(root: Path, *, report: str, followup: str) -> None:
    (root / "README.md").write_text(README, encoding="utf-8")
    (root / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (root / "output").mkdir(exist_ok=True)
    (root / "output" / "vitals_report.txt").write_text(report, encoding="utf-8")
    (root / "output" / "followup_list.txt").write_text(followup, encoding="utf-8")


def run() -> None:
    supplied = ASSIGNMENT / DATA_FILE
    assert _fingerprint(supplied.read_text(encoding="utf-8")) == DATA_FINGERPRINT, (
        f"{DATA_FILE.as_posix()} changed; update DATA_FINGERPRINT in _value_checks.py"
    )

    with tempfile.TemporaryDirectory(dir=REPO / "scratch", prefix="a02-selftest-") as temporary:
        root = Path(temporary) / "submission"
        root.mkdir()

        # An untouched handout scores nothing and still reports every check.
        assert grade_submission(root)["score"] == 0
        assert len(grade_submission(root)["tests"]) == len(POINTS)

        (root / "data").mkdir()
        shutil.copy(supplied, root / DATA_FILE)
        encounters = load_supplied_encounters(root)
        readings = encounters.readings
        mean = sum(readings) / len(readings)
        cutoff = 140
        listed = sorted(encounters.patients_at_or_above(cutoff))
        report = (
            f"Usable encounters: {len(encounters.usable)}\n"
            f"Skipped rows: {encounters.skipped_rows}\n"
            f"Patients seen: {len(encounters.patients)}\n"
            f"Mean systolic: {mean:.1f} mmHg\n"
            f"Highest systolic: {max(readings)} mmHg\n"
            f"Lowest systolic: {min(readings)} mmHg\n"
        )
        followup = f"Cutoff: {cutoff} mmHg\n{REASON}" + "".join(f"{patient}\n" for patient in listed)

        write_submission(root, report=report, followup=followup)
        assert grade_submission(root)["score"] == 100, grade_submission(root)

        # Formatting differences never decide a grade.
        for variant in (
            report.replace("Usable encounters", "usable encounters").replace(": ", ":   "),
            report.replace(" mmHg", ""),
            report.replace("mmHg", "mm Hg"),
            report.replace("mmHg", "mmHg (systolic)"),
            report.replace("\n", "\r\n"),
            "﻿" + report,
            "\n".join(reversed(report.strip().splitlines())) + "\nextra notes, ignored\n",
            report.replace(f"{mean:.1f}", f"{mean:.3f}"),
            report.replace(f"{mean:.1f}", f"{mean:.0f}"),
            report.replace(f"{mean:.1f} mmHg", f"np.float64({mean})"),
            report.replace(f"{max(readings)} mmHg", f"np.int64({max(readings)})"),
        ):
            write_submission(root, report=variant, followup=followup)
            assert grade_submission(root)["score"] == 100, variant

        for variant in (
            followup.replace("Cutoff:", "cutoff:").replace(f"{cutoff} mmHg", f"{cutoff}.0"),
            followup.upper(),
            followup.replace("\n", "\r\n"),
            "﻿" + followup,
            followup.replace(f"{cutoff} mmHg", f"np.int64({cutoff})"),
            # A heading, a separator and a bullet are not patient IDs.
            followup.replace(REASON, REASON + "\nPatients to call\n----------------\n").replace(
                "\np0", "\n- p0"
            ),
        ):
            write_submission(root, report=report, followup=variant)
            assert grade_submission(root)["score"] == 100, variant

        # Every cutoff in range is right when the list matches it.
        for alternative in range(120, 181):
            listed_alternative = sorted(encounters.patients_at_or_above(alternative))
            write_submission(
                root,
                report=report,
                followup=f"Cutoff: {alternative}\n{REASON}" + "".join(f"{p}\n" for p in listed_alternative),
            )
            assert grade_submission(root)["score"] == 100, alternative

        # A cutoff outside the range, or a list from another cutoff, loses only its own check.
        write_submission(root, report=report, followup=f"Cutoff: 95\n{REASON}" + "".join(f"{p}\n" for p in listed))
        assert scores(root)["follow-up cutoff"] == 0
        assert scores(root)["follow-up reason"] == 5
        assert scores(root)["follow-up patient list"] == 0
        assert scores(root)["mean systolic"] == 15

        write_submission(root, report=report, followup=f"Cutoff: 150 mmHg\n{REASON}" + "".join(f"{p}\n" for p in listed))
        assert scores(root)["follow-up cutoff"] == 5
        assert scores(root)["follow-up patient list"] == 0

        write_submission(root, report=report, followup=f"Cutoff: {cutoff} mmHg\nReason: short.\n")
        assert scores(root)["follow-up cutoff"] == 5
        assert scores(root)["follow-up reason"] == 0

        # Each wrong value costs its own check and nothing else.
        for label, wrong, check in (
            ("Usable encounters", len(encounters.usable) + 2, "usable encounters"),
            ("Skipped rows", encounters.skipped_rows + 2, "skipped rows"),
            ("Patients seen", len(encounters.patients) + 1, "patients seen"),
            ("Mean systolic", round(mean) + 3, "mean systolic"),
            ("Highest systolic", max(readings) + 10, "highest systolic"),
            ("Lowest systolic", min(readings) - 10, "lowest systolic"),
        ):
            damaged = []
            for line in report.splitlines():
                damaged.append(f"{label}: {wrong}" if line.startswith(f"{label}:") else line)
            write_submission(root, report="\n".join(damaged) + "\n", followup=followup)
            result = scores(root)
            assert result[check] == 0, (label, result)
            assert result["vitals report format"] == 10, (label, result)
            assert sum(result.values()) == 100 - CHECK_POINTS[check], (label, result)

        # A missing label costs the format check and that value's check only.
        write_submission(
            root,
            report="\n".join(line for line in report.splitlines() if not line.startswith("Patients seen")) + "\n",
            followup=followup,
        )
        assert scores(root) == {
            "README project description": 5,
            "README run command": 5,
            ".gitignore bytecode cache": 5,
            "vitals report format": 0,
            "usable encounters": 8,
            "skipped rows": 7,
            "patients seen": 0,
            "mean systolic": 15,
            "highest systolic": 5,
            "lowest systolic": 5,
            "follow-up cutoff": 5,
            "follow-up reason": 5,
            "follow-up patient list": 15,
        }

        # Documentation artifacts are checked for shape, not for wording.
        write_submission(root, report=report, followup=followup)
        for run_line, expected in (
            ("`python clinic_report.py`", 5),
            ("```bash\npy -3.13 ./clinic_report.py\n```", 5),
            ("$ python3.13 clinic_report.py", 5),
            ("Run `python3 clinic_report.py` from this folder.", 5),
            ("- `python3 clinic_report.py`", 5),
            ("python3 clinic_report", 0),
            ("Run it from this folder.", 0),
        ):
            (root / "README.md").write_text(README.replace("python3 clinic_report.py", run_line), encoding="utf-8")
            assert scores(root)["README run command"] == expected, run_line
            assert scores(root)["README project description"] == 5, run_line
        (root / "README.md").write_text(README.replace("Summarizes a week", "TODO: Summarizes a week"), encoding="utf-8")
        assert scores(root)["README project description"] == 0
        assert scores(root)["README run command"] == 5
        (root / "README.md").write_text(README, encoding="utf-8")

        for patterns, expected in (
            ("__pycache__/\n*.pyc\n", 5),
            ("# cache\n**/__pycache__/\n*.pyc\n", 5),
            ("__pycache__\n*.pyc\n", 5),
            # GitHub's own Python template.
            ("__pycache__/\n*.py[cod]\n*$py.class\n", 5),
            ("*.py[cod]\n", 5),
            ("*.pyc\n", 5),
            ("__pycache__/\n", 5),
            ("# TODO: the cache directory\n# TODO: the compiled files\n", 0),
            ("output/\n", 0),
        ):
            (root / ".gitignore").write_text(patterns, encoding="utf-8")
            assert scores(root)[".gitignore bytecode cache"] == expected, patterns
        (root / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

        # A changed encounter file cannot be summarized into a passing answer.
        (root / DATA_FILE).write_text("patient_id,visit_date,systolic\nP001,2026-03-02,140\n", encoding="utf-8")
        result = scores(root)
        assert result["usable encounters"] == 0
        assert result["skipped rows"] == 0
        assert result["mean systolic"] == 0
        assert result["follow-up patient list"] == 0
        assert result["README project description"] == 5
        assert result["vitals report format"] == 10

    print(
        f"Assignment 02 value checks: {len(POINTS)} checks worth {sum(POINTS)} points, recomputed "
        "answers, tolerant formats, and per-check partial credit all pass."
    )


def run_shape() -> None:
    """The checks in the student fork judge shape, and cannot judge an answer."""
    import ast
    import importlib.util

    source = (ASSIGNMENT / "_shape_checks.py").read_text(encoding="utf-8")
    encounters = load_supplied_encounters(ASSIGNMENT)
    readings = encounters.readings
    answers = {
        len(encounters.usable),
        encounters.skipped_rows,
        len(encounters.patients),
        max(readings),
        min(readings),
        round(sum(readings) / len(readings), 2),
    }
    constants = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
    }
    assert not constants & answers, sorted(constants & answers)
    assert "clinic_encounters" not in source and "data" not in ast_strings(source), "shape checks read the data"

    specification = importlib.util.spec_from_file_location("_shape_checks", ASSIGNMENT / "_shape_checks.py")
    shape = importlib.util.module_from_spec(specification)
    sys.modules["_shape_checks"] = shape  # dataclasses look their own module up by name
    specification.loader.exec_module(shape)

    with tempfile.TemporaryDirectory(dir=REPO / "scratch", prefix="a02-shape-") as temporary:
        root = Path(temporary) / "submission"
        root.mkdir()

        # An untouched handout fails every shape check; no data file is needed.
        assert all(detail for _, detail in shape.run_checks(root))

        right = (
            f"Usable encounters: {len(encounters.usable)}\nSkipped rows: {encounters.skipped_rows}\n"
            f"Patients seen: {len(encounters.patients)}\n"
            f"Mean systolic: {sum(readings) / len(readings):.1f} mmHg\n"
            f"Highest systolic: {max(readings)} mmHg\nLowest systolic: {min(readings)} mmHg\n"
        )
        listed = sorted(encounters.patients_at_or_above(140))
        right_followup = f"Cutoff: 140 mmHg\n{REASON}" + "".join(f"{p}\n" for p in listed)
        # Plausible and wrong: every number is in range, every value is incorrect.
        wrong = (
            "Usable encounters: 30\nSkipped rows: 1\nPatients seen: 29\n"
            "Mean systolic: 142.7 mmHg\nHighest systolic: 200 mmHg\nLowest systolic: 61 mmHg\n"
        )
        wrong_followup = f"Cutoff: 121 mmHg\n{REASON}P001\nP007\n"

        for report, followup in ((right, right_followup), (wrong, wrong_followup)):
            write_submission(root, report=report, followup=followup)
            failures = [detail for _, detail in shape.run_checks(root) if detail]
            assert not failures, failures
            # No data file at all: the shape checks still pass, because they never open it.
            shutil.rmtree(root / "data", ignore_errors=True)
            assert not [detail for _, detail in shape.run_checks(root) if detail]

        # Implausible artifacts are still caught.
        for report, failing in (
            (right.replace("Mean systolic: ", "Mean systolic: 1"), "mean systolic"),
            (right.replace("Skipped rows: ", "Skipped rows: -"), "skipped rows"),
            (right.replace("Patients seen:", "Patients:"), "vitals report format"),
            (right.replace(f"Lowest systolic: {min(readings)}", "Lowest systolic: not recorded"), "lowest systolic"),
        ):
            write_submission(root, report=report, followup=right_followup)
            broken = {name for name, detail in shape.run_checks(root) if detail}
            assert broken and failing in broken, (failing, broken)

        for followup, failing in (
            (f"Cutoff: 95\n{REASON}P001\n", "follow-up cutoff"),
            (f"Cutoff: 140\nReason: too short.\nP001\n", "follow-up reason"),
            (f"Cutoff: 140\n{REASON}", "follow-up patient list"),
            (f"Cutoff: 140\n{REASON}P001\nP001\n", "follow-up patient list"),
        ):
            write_submission(root, report=right, followup=followup)
            broken = {name for name, detail in shape.run_checks(root) if detail}
            assert failing in broken, (failing, broken)

    print(
        "Assignment 02 shape checks: correct and plausibly wrong submissions are indistinguishable, "
        "no answer appears in the module, and the supplied data is never read."
    )


def ast_strings(source: str) -> set[str]:
    import ast

    return {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


if __name__ == "__main__":
    run()
    run_shape()
