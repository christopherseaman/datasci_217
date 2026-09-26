"""Development regression checks for the Assignment 07 checks.

Answers the assignment with pandas, matplotlib, and Altair the way the lecture
does, builds submissions in ignored `scratch/`, and confirms what each kind of
submission scores: a correct one scores 100 however it is formatted, and each
mistake costs only its own checks. It also confirms that the values the checks
hold match the handout's data, that the README agrees with the checks, and that
the handout in `07/assignment/` ships the course-owned checks byte for byte.
Nothing here reads or runs student code.

    uv run --python 3.13 --with-requirements 07/assignment/requirements.txt \\
        python 07/assignment_checks/_grader_selftest/run.py
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import tempfile

import altair as alt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


CHECKS = Path(__file__).resolve().parents[1]
HANDOUT = CHECKS.parent / "assignment"
SCRATCH = CHECKS.parents[1] / "scratch"
sys.path.insert(0, str(CHECKS))
import _value_checks as value_checks  # noqa: E402
from grading import POINTS, grade_submission  # noqa: E402

NAMES = [check.name for check in value_checks.CHECKS]
CHECK_POINTS = dict(zip(NAMES, POINTS, strict=True))
SPEC_CHECKS = {name for name in NAMES if name.startswith("exploratory spec")}
CRITIQUE_CHECKS = {name for name in NAMES if name.startswith("critique:")}
SUPPORTING_CHECKS = {name for name in NAMES if name.startswith("supporting data")}
EVIDENCE_CHECKS = {name for name in NAMES if name.startswith(("evidence:", "data type:"))}
TASK_POINTS = {
    "Task 1": sum(CHECK_POINTS[name] for name in SPEC_CHECKS),
    "Task 2": CHECK_POINTS["critique redesign: PNG image"] + sum(CHECK_POINTS[name] for name in CRITIQUE_CHECKS),
}

SPEC = value_checks.SPEC_FILE
EVIDENCE = value_checks.EVIDENCE_FILE
REDESIGN = value_checks.REDESIGN_FILE
SUPPORTING = value_checks.SUPPORTING_FILE
EXPLANATORY = value_checks.EXPLANATORY_FILE
TEXT_ALTERNATIVE = value_checks.TEXT_ALTERNATIVE_FILE

CRITIQUE = [
    {"category": "unsupported claim", "problem": "The title claims a cause the data cannot show.",
     "repair": "Describe what the bars show."},
    {"category": "truncated baseline", "problem": "The y-axis starts at 76%, exaggerating small gaps.",
     "repair": "Start the bar axis at zero."},
    {"category": "missing unit", "problem": "The y-axis has no label or unit.",
     "repair": "Label it Scheduled sessions attended (%)."},
    {"category": "color-only encoding", "problem": "Only red and green separate the programs.",
     "repair": "Add a hatch per program and colorblind-safe colors."},
    {"category": "distracting decoration", "problem": "Thick edges, a heavy grid, and one hatch on every bar.",
     "repair": "Remove them and hide the top and right spines."},
]
TEXT_ALTERNATIVE_TEXT = (
    "Line chart of patients meeting the weekly exercise goal (%) at four follow-up visits for two programs. "
    "Home-based rises from 58% to 70% and Center-based from 57% to 79%, 9 points higher at visit 4. "
    "Patients chose their program, so the gap is descriptive."
)


def scatter(patients: pd.DataFrame) -> alt.Chart:
    """Task 1.1's chart, as Lecture 07's "Encode the study table" snippet builds it."""
    return alt.Chart(patients).mark_point(filled=True, size=90).encode(
        x=alt.X("sessions_attended:Q", title="Sessions attended (of 36)"),
        y=alt.Y("walk_distance_m:Q", title="Six-minute walk distance (m)", scale=alt.Scale(zero=False)),
        color=alt.Color("program:N", title="Program"),
        shape=alt.Shape("program:N", title="Program"),
        tooltip=["patient_id:N", "program:N", "sessions_attended:Q", "walk_distance_m:Q"],
    ).properties(title="Walk distance and sessions attended")


def redesign(path: Path) -> None:
    """Task 2.3's grouped bars, as Lecture 07's "Grouped Bars That Work in Grayscale" snippet draws them."""
    attendance = pd.read_csv(HANDOUT / "data" / "session_attendance.csv")
    x = np.arange(2)
    width = 0.38
    home = attendance.loc[attendance["program"] == "Home-based", "attended_pct"]
    center = attendance.loc[attendance["program"] == "Center-based", "attended_pct"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    home_bars = ax.bar(x - width / 2, home, width, label="Home-based", color="#E69F00", hatch="..")
    center_bars = ax.bar(x + width / 2, center, width, label="Center-based", color="#0072B2", hatch="//")
    ax.bar_label(home_bars, fmt="%d%%")
    ax.bar_label(center_bars, fmt="%d%%")
    ax.set_xticks(x, ["Q1", "Q2"])
    ax.set(ylim=(0, 100), xlabel="Quarter", ylabel="Scheduled sessions attended (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Program", loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def explanatory(supporting: pd.DataFrame, path: Path) -> None:
    """Task 3.2's line chart, as Lecture 07's "Redundant Cues on a Line Chart" snippet draws it."""
    home = supporting.loc[supporting["program"] == "Home-based"]
    center = supporting.loc[supporting["program"] == "Center-based"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(home["visit_number"], home["goal_met_pct"], color="#E69F00", marker="s", linestyle="--")
    ax.plot(center["visit_number"], center["goal_met_pct"], color="#0072B2", marker="o", linestyle="-")
    ax.text(4.08, 70, "Home-based", va="center")
    ax.text(4.08, 79, "Center-based", va="center")
    ax.annotate("9 points higher at visit 4", xy=(4, 79), xytext=(2, 83), arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="Follow-up visit", ylabel="Patients meeting the exercise goal (%)")
    ax.set_xticks([1, 2, 3, 4])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evidence() -> dict:
    return {
        "critique": [dict(entry) for entry in CRITIQUE],
        "question": "How often did patients in each program meet the exercise goal at each visit?",
        "audience": "The rehab program coordinator, deciding where to add follow-up support",
        "intended_claim": "By visit 4, Center-based patients met the goal 9 points more often.",
        "displayed_unit": "Patients meeting the weekly exercise goal (%)",
        "grain": "One program at one follow-up visit",
        "data_types": {"program": "categorical", "visit_number": "ordinal", "goal_met_pct": "quantitative"},
        "text_alternative": TEXT_ALTERNATIVE_TEXT,
    }


def write_json(root: Path, name: str, value) -> None:
    with open(root / name, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def solve(root: Path) -> None:
    """Write all six artifacts with the lecture's methods, as the README asks."""
    (root / "output").mkdir(parents=True, exist_ok=True)
    patients = pd.read_csv(HANDOUT / "data" / "rehab_patients.csv")
    scatter(patients).save(root / SPEC)
    redesign(root / REDESIGN)
    followup = pd.read_csv(HANDOUT / "data" / "followup_goals.csv")
    supporting = followup[["program", "visit_number", "goal_met_pct"]]
    supporting.to_csv(root / SUPPORTING, index=False)
    assert pd.read_csv(root / SUPPORTING).equals(supporting)
    explanatory(supporting, root / EXPLANATORY)
    write_json(root, EVIDENCE, evidence())
    with open(root / TEXT_ALTERNATIVE, "w", encoding="utf-8") as file:
        file.write(TEXT_ALTERNATIVE_TEXT)


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


def details(root: Path) -> dict[str, str]:
    return {name: detail(root, name) for name in lost(root)}


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


def edit_text(root: Path, name: str, change) -> None:
    path = root / name
    path.write_text(change(path.read_text(encoding="utf-8")), encoding="utf-8")


def edit_json(root: Path, name: str, change) -> None:
    path = root / name
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def variant(workspace: Path, label: str, base: Path) -> Path:
    root = workspace / label
    shutil.copytree(base, root)
    return root


def run() -> None:
    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a07-selftest-") as temporary:
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
        assert lost(correct) == set(), details(correct)
        for checks in (CHECKS, HANDOUT):
            assert checker_report(checks, correct)["score"] == 100, checks
            assert checker_report(checks, empty) == checker_report(CHECKS, empty), checks

        # Formatting and reasonable alternatives that change no value never cost points.
        loose = variant(workspace, "loose", correct)
        supporting = pd.read_csv(correct / SUPPORTING)
        supporting = supporting[["goal_met_pct", "program", "visit_number"]].astype(
            {"goal_met_pct": float, "visit_number": float})
        supporting["program"] = supporting["program"].str.upper()
        supporting.to_csv(loose / SUPPORTING, float_format="%.2f", quoting=1)  # row numbers kept, all quoted
        edit_text(loose, SUPPORTING, lambda text: text.replace("\n", "  \r\n").rstrip())
        loose_evidence = evidence()
        loose_evidence["critique"] = [
            {key.title(): value.upper() if key == "category" else value for key, value in entry.items()}
            for entry in reversed(CRITIQUE)
        ]
        loose_evidence["critique"][1]["Category"] = "COLOUR ONLY ENCODING"
        loose_evidence["Intended Claim"] = loose_evidence.pop("intended_claim")
        loose_evidence["displayed-unit"] = loose_evidence.pop("displayed_unit")
        loose_evidence["variables"] = {
            "Program": "Nominal: the group, by color and marker",
            "visit_number": "ordered categorical (the visits in sequence)",
            "GOAL_MET_PCT": "Q",
        }
        del loose_evidence["data_types"]
        (loose / EVIDENCE).write_bytes(("\ufeff" + json.dumps(loose_evidence, indent=1).replace("\n", "\r\n")).encode())
        (loose / TEXT_ALTERNATIVE).write_bytes(("  " + TEXT_ALTERNATIVE_TEXT.replace(". ", ".\r\n") + "\r\n\r\n").encode())
        patients = pd.read_csv(HANDOUT / "data" / "rehab_patients.csv")
        plotted = patients[["walk_distance_m", "program", "sessions_attended"]].iloc[::-1]
        chart = alt.Chart(plotted).mark_point().encode(
            x="sessions_attended", y="walk_distance_m", color="program", shape="program",
        ).interactive()
        (loose / SPEC).write_text(json.dumps(chart.to_dict()), encoding="utf-16")
        (loose / EXPLANATORY).rename(loose / "output" / "Explanatory_Chart.PNG")
        assert lost(loose) == set(), details(loose)

        reread = variant(workspace, "row-numbers-twice", correct)
        pd.read_csv(correct / SUPPORTING).to_csv(reread / SUPPORTING)
        pd.read_csv(reread / SUPPORTING).to_csv(reread / SUPPORTING)  # a second row-number column
        assert lost(reread) == set(), details(reread)

        inline = variant(workspace, "inline-values-and-dict-critique", correct)

        def inline_values(spec: dict) -> None:
            name = spec["data"]["name"]
            spec["data"] = {"values": spec.pop("datasets")[name]}
            for row in spec["data"]["values"]:
                row["sessions_attended"] = f"{row['sessions_attended']}.00"

        edit_json(inline, SPEC, inline_values)
        edit_json(inline, EVIDENCE, lambda value: value.update(
            critique={entry["category"]: {"problem": entry["problem"], "repair": entry["repair"]} for entry in CRITIQUE},
            data_types={"program": "categorical (x groups)", "visit_number": ":O", "goal_met_pct": "quantitative"},
        ))
        assert lost(inline) == set(), details(inline)

        layered = variant(workspace, "layered", correct)
        base = scatter(patients)
        labels = alt.Chart(patients).mark_text(dx=12).encode(
            x="sessions_attended:Q", y="walk_distance_m:Q", text="patient_id:N")
        alt.layer(base, labels).save(layered / SPEC)
        assert lost(layered) == set(), details(layered)

        # Other reasonable ways to name each data type.
        phrased = variant(workspace, "data-type-phrasings", correct)
        edit_json(phrased, EVIDENCE, lambda value: value.update(data_types={
            "program": "qualitative (categorical, not ordered)",
            "visit_number": "categorical (ordered)",
            "goal_met_pct": "numeric, continuous",
        }))
        assert lost(phrased) == set(), details(phrased)

        # The supporting data separated by semicolons or tabs, and the text alternative in another letter case.
        for label, separator in (("semicolons", ";"), ("tabs", "\t")):
            separated = variant(workspace, f"supporting-{label}-text-alternative-case", correct)
            pd.read_csv(correct / SUPPORTING).to_csv(separated / SUPPORTING, sep=separator, index=False)
            (separated / TEXT_ALTERNATIVE).write_text(TEXT_ALTERNATIVE_TEXT.upper(), encoding="utf-8")
            assert lost(separated) == set(), details(separated)

        # Each mistake costs only the checks it gets wrong.
        mistakes = []

        def mistake(label: str, change, expected: set[str], hint: str | None = None) -> None:
            root = variant(workspace, label, correct)
            mistakes.append(label)
            change(root)
            assert lost(root) == expected, (label, lost(root), details(root))
            if hint is not None:
                named = sorted(expected)[0]
                assert hint in detail(root, named), (label, detail(root, named))
            assert checker_report(HANDOUT, root) == checker_report(CHECKS, root), label

        def save_chart(chart: alt.Chart):
            return lambda root: chart.save(root / SPEC)

        mistake("spec-circle-mark", save_chart(scatter(patients).mark_circle()),
                {"exploratory spec: point mark"}, "draws circle marks")
        no_shape = alt.Chart(patients).mark_point(filled=True, size=90).encode(
            x="sessions_attended:Q", y="walk_distance_m:Q", color="program:N")
        mistake("spec-no-shape", save_chart(no_shape),
                {"exploratory spec: shape encoding"}, "has no shape encoding")
        mistake("spec-color-ordinal", save_chart(scatter(patients).encode(color="program:O")),
                {"exploratory spec: color encoding"}, "type ordinal, not nominal")
        mistake("spec-axes-swapped",
                save_chart(scatter(patients).encode(x="walk_distance_m:Q", y="sessions_attended:Q")),
                {"exploratory spec: x encoding", "exploratory spec: y encoding"}, "encodes walk_distance_m as x")
        mistake("spec-one-patient-changed", save_chart(scatter(patients.replace({"walk_distance_m": {431: 413}}))),
                {"exploratory spec: embedded patient rows"}, "R03 has walk_distance_m 413, expected 431")
        mistake("spec-one-program-only",
                save_chart(scatter(patients.loc[patients["program"] == "Home-based"])),
                {"exploratory spec: embedded patient rows"}, "is missing 6 of the 12 patients: R02, R04")
        from_url = alt.Chart("data/rehab_patients.csv").mark_point().encode(
            x="sessions_attended:Q", y="walk_distance_m:Q", color="program:N", shape="program:N")
        mistake("spec-data-url", save_chart(from_url),
                {"exploratory spec: embedded patient rows"}, "instead of embedding the rows")
        mistake("spec-missing", lambda root: (root / SPEC).unlink(), SPEC_CHECKS, "run the Task 1.1 cell")
        mistake("spec-truncated", lambda root: edit_text(root, SPEC, lambda text: text[: len(text) // 2]),
                SPEC_CHECKS, "is not valid JSON")

        mistake("redesign-missing", lambda root: (root / REDESIGN).unlink(),
                {"critique redesign: PNG image"}, "run the Task 2.3 cell")
        mistake("redesign-as-svg", lambda root: (root / REDESIGN).write_text("<svg></svg>"),
                {"critique redesign: PNG image"}, "is not a PNG image")
        mistake("redesign-wrong-name",
                lambda root: (root / REDESIGN).rename(root / "output" / "critique_redesign.jpg"),
                {"critique redesign: PNG image"}, "Found output/critique_redesign.jpg")
        mistake("critique-blank-repair",
                lambda root: edit_json(root, EVIDENCE, lambda value: value["critique"][1].update(repair=" ")),
                {"critique: truncated baseline"}, "has no repair text")
        mistake("critique-category-dropped",
                lambda root: edit_json(root, EVIDENCE, lambda value: value["critique"].pop(3)),
                {"critique: color-only encoding"}, "has no color-only encoding entry")
        mistake("critique-key-missing",
                lambda root: edit_json(root, EVIDENCE, lambda value: value.pop("critique")),
                CRITIQUE_CHECKS, "has no critique key")

        mistake("explanatory-missing", lambda root: (root / EXPLANATORY).unlink(),
                {"explanatory chart: PNG image"}, "run the Task 3.2 cell")
        followup = pd.read_csv(HANDOUT / "data" / "followup_goals.csv")
        mistake("supporting-all-columns", lambda root: followup.to_csv(root / SUPPORTING, index=False),
                {"supporting data: columns"}, "also has patients_seen")
        mistake("supporting-misnamed-column",
                lambda root: edit_text(root, SUPPORTING, lambda text: text.replace("goal_met_pct", "goal_pct")),
                {"supporting data: columns"}, "is missing goal_met_pct and also has goal_pct")
        mistake("supporting-one-wrong-value",
                lambda root: edit_text(root, SUPPORTING, lambda text: text.replace("Center-based,4,79", "Center-based,4,97")),
                {"supporting data: rows and values"}, "Center-based, visit 4 has goal_met_pct 97, expected 79")
        mistake("supporting-row-dropped",
                lambda root: followup[["program", "visit_number", "goal_met_pct"]].iloc[:7].to_csv(
                    root / SUPPORTING, index=False),
                {"supporting data: rows and values"}, "has no row for Center-based, visit 4")
        mistake("supporting-row-repeated",
                lambda root: pd.concat([followup[["program", "visit_number", "goal_met_pct"]]] * 2).iloc[:9].to_csv(
                    root / SUPPORTING, index=False),
                {"supporting data: rows and values"}, "lists Home-based, visit 1 twice")
        mistake("critique-null", lambda root: edit_json(root, EVIDENCE, lambda value: value.update(critique=None)),
                CRITIQUE_CHECKS, "critique is null, not a list of entries")
        mistake("data-types-list",
                lambda root: edit_json(root, EVIDENCE, lambda value: value.update(data_types=["categorical"])),
                {name for name in NAMES if name.startswith("data type:")}, "data_types is a list, not an object")
        mistake("spec-json-null", lambda root: (root / SPEC).write_text("null", encoding="utf-8"),
                SPEC_CHECKS, "holds null, not a chart specification")
        mistake("supporting-missing", lambda root: (root / SUPPORTING).unlink(), SUPPORTING_CHECKS,
                "run the Task 3.1 cell")

        mistake("evidence-blank-grain", lambda root: edit_json(root, EVIDENCE, lambda value: value.update(grain="")),
                {"evidence: grain"}, "write grain in Task 3.1")
        mistake("evidence-no-question", lambda root: edit_json(root, EVIDENCE, lambda value: value.pop("question")),
                {"evidence: question"}, "has no question key")
        mistake("data-type-temporal",
                lambda root: edit_json(root, EVIDENCE, lambda value: value["data_types"].update(visit_number="temporal")),
                {"data type: visit_number"}, "not ordinal")
        mistake("data-type-numeric-program",
                lambda root: edit_json(root, EVIDENCE, lambda value: value["data_types"].update(program="quantitative")),
                {"data type: program"}, "not categorical")
        mistake("data-type-ordered-program",
                lambda root: edit_json(root, EVIDENCE, lambda value: value["data_types"].update(
                    program="categorical (ordered)")),
                {"data type: program"}, "not categorical")
        mistake("data-type-not-ordinal",
                lambda root: edit_json(root, EVIDENCE, lambda value: value["data_types"].update(
                    visit_number="categorical, not ordinal")),
                {"data type: visit_number"}, "not ordinal")
        mistake("supporting-semicolons-one-wrong-value",
                lambda root: pd.read_csv(correct / SUPPORTING).replace({79: 97}).to_csv(
                    root / SUPPORTING, sep=";", index=False),
                {"supporting data: rows and values"}, "Center-based, visit 4 has goal_met_pct 97, expected 79")
        mistake("data-types-missing", lambda root: edit_json(root, EVIDENCE, lambda value: value.pop("data_types")),
                {name for name in NAMES if name.startswith("data type:")}, "has no data_types key")
        mistake("text-alternative-differs",
                lambda root: (root / TEXT_ALTERNATIVE).write_text("A different paragraph.", encoding="utf-8"),
                {"text alternative file"}, "differs from text_alternative")
        mistake("text-alternative-missing", lambda root: (root / TEXT_ALTERNATIVE).unlink(),
                {"text alternative file"}, "run the Task 3.3 cell")
        mistake("evidence-truncated", lambda root: edit_text(root, EVIDENCE, lambda text: text[:200]),
                CRITIQUE_CHECKS | EVIDENCE_CHECKS, "is not valid JSON")
        mistake("evidence-critique-only", lambda root: write_json(root, EVIDENCE, {"critique": CRITIQUE}),
                EVIDENCE_CHECKS, "in Task 3.3, add")

        # A submission that stops after Task 2 scores Tasks 1 and 2 in full.
        tasks_1_2 = variant(workspace, "tasks-1-and-2-only", correct)
        for name in (SUPPORTING, EXPLANATORY, TEXT_ALTERNATIVE):
            (tasks_1_2 / name).unlink()
        write_json(tasks_1_2, EVIDENCE, {"critique": CRITIQUE})
        assert sum(scores(tasks_1_2).values()) == TASK_POINTS["Task 1"] + TASK_POINTS["Task 2"] == 62

    # A fresh handout prints exactly the "Before Task 1" example README.md shows, and a clean run ends as shown.
    readme = (HANDOUT / "README.md").read_text(encoding="utf-8")
    shown = re.search(r"Before Task 1, for example, the first check reports:\n\n```text\n(.*?)```", readme, re.DOTALL)
    assert shown is not None, "README.md no longer shows the Before Task 1 example"
    printed = subprocess.run(
        [sys.executable, "-B", "check_assignment.py"], cwd=HANDOUT, capture_output=True, text=True, check=False
    ).stdout
    assert shown.group(1) in printed, (shown.group(1), printed)
    last = re.search(r"A clean run ends with:\n\n```text\n(\[PASS\][^\n]*)\n", readme)
    assert last is not None and last.group(1).endswith(NAMES[-1]), "README's clean-run example ends on the last check"

    # README's completion contract agrees with the checks, name for name and point for point.
    contract = {
        name: int(points)
        for name, points in re.findall(r"^\| `output/[^|]+\| [^|]+\| ([^|]+) \| (\d+) \|$", readme, re.MULTILINE)
    }
    assert contract == CHECK_POINTS, (contract, CHECK_POINTS)

    print(
        "Assignment 07 checks: empty, handout, correct, loosely formatted, row-numbers-twice, inline-values, "
        "layered, data-type-phrasings, semicolon- and tab-separated, Tasks-1-and-2-only, and "
        f"{len(mistakes)} single-mistake submissions all score as intended."
    )


def run_handout() -> None:
    """The handout's data match the checks, and it ships the course's checks unchanged."""
    patients = pd.read_csv(HANDOUT / "data" / "rehab_patients.csv")
    held = {row.patient_id: (row.program, row.sessions_attended, row.walk_distance_m) for row in patients.itertuples()}
    assert held == value_checks.REHAB_PATIENTS, "data/rehab_patients.csv differs from REHAB_PATIENTS"
    followup = pd.read_csv(HANDOUT / "data" / "followup_goals.csv")
    goals = tuple(followup[["program", "visit_number", "goal_met_pct"]].itertuples(index=False, name=None))
    assert goals == value_checks.FOLLOWUP_GOALS, "data/followup_goals.csv differs from FOLLOWUP_GOALS"

    workflow = (HANDOUT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert re.search(r'^  CHECKS_PATH: "07/assignment_checks"$', workflow, re.MULTILINE), (
        "tests.yml does not download from 07/assignment_checks"
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
            f"07/assignment/{name} differs from 07/assignment_checks/{name}; copy the course-owned file over it"
        )

    # The handout's only Python files are the checks; the self-test stays here.
    handout_python = {path.relative_to(HANDOUT).as_posix() for path in HANDOUT.rglob("*.py")}
    assert handout_python == set(files) - {".github/test/requirements.txt"}, sorted(handout_python)
    assert not (HANDOUT / "_grader_selftest").exists()

    # The notebook ships with its outputs cleared, and its task headings match the README's.
    notebook = json.loads((HANDOUT / "assignment.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, "clear the handout notebook's outputs"
    headings = re.compile(r"^#{2,3} (?:Task \d+: .+|\d\.\d .+)$", re.MULTILINE)
    in_notebook = headings.findall("\n".join("".join(cell["source"]) for cell in notebook["cells"]))
    in_readme = headings.findall((HANDOUT / "README.md").read_text(encoding="utf-8"))
    assert in_notebook == in_readme, (in_notebook, in_readme)

    print(f"Assignment 07 handout: data match the checks, the notebook's headings match the README, and all "
          f"{len(files)} check files match 07/assignment_checks byte for byte.")


if __name__ == "__main__":
    run()
    run_handout()
