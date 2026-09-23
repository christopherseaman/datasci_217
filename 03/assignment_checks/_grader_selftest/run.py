"""Development regression checks for the Assignment 03 checks.

Builds submissions in ignored `scratch/`, computes the correct answers with
NumPy (independently of the pure-Python recomputation the checks use), and
confirms what each kind of submission scores. It also confirms that the
handout in `03/assignment/` ships the course-owned checks byte for byte, so a
student's local run reports exactly what the GitHub run reports.

    python 03/assignment_checks/_grader_selftest/run.py
"""

from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np


CHECKS = Path(__file__).resolve().parents[1]
FORK = CHECKS.parent / "assignment"
SCRATCH = CHECKS.parents[2] / "scratch"
sys.path.insert(0, str(CHECKS))
from _public_checks import DATA_FILE  # noqa: E402
from grading import POINTS, grade_submission  # noqa: E402

COUNTS_NAME = "monitor_counts_20260922_101500.txt"
ARTIFACT_POINTS = sum(POINTS[:4])
ANSWER_POINTS = dict(
    zip(
        (
            "patients", "readings", "mean_sbp", "sd_sbp", "min_sbp", "max_sbp",
            "stage2_patients", "highest_patient", "highest_patient_mean",
            "peak_hour_column", "peak_hour_mean", "high_monitor", "monitor_offset",
            "stage2_other_monitors",
        ),
        POINTS[4:],
        strict=True,
    )
)


def solve(data_path: Path) -> tuple[list[str], dict[str, int]]:
    """Answer the assignment with NumPy, the way a student would.

    A monitor's average is the mean of its patients' 12-hour means, which is
    the definition the assignment README states and the one the checks
    recompute.
    """
    lines = data_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    rows = [line.split(",") for line in lines[1:] if line.strip()]
    patient_ids = np.array([row[0] for row in rows])
    monitors = np.array([row[1] for row in rows])
    hour_columns = np.array(header[2:])
    readings = np.array([row[2:] for row in rows]).astype(int)

    patient_means = readings.mean(axis=1)
    hour_means = readings.mean(axis=0)
    highest = patient_means.argmax()
    peak = hour_means.argmax()
    names = sorted(set(monitors.tolist()))
    monitor_means = np.array([patient_means[monitors == name].mean() for name in names])
    high = names[int(monitor_means.argmax())]
    others = patient_means[monitors != high]

    summary = [
        f"patients: {readings.shape[0]}",
        f"readings: {readings.size}",
        f"mean_sbp: {readings.mean():.1f}",
        f"sd_sbp: {readings.std():.1f}",
        f"min_sbp: {readings.min()}",
        f"max_sbp: {readings.max()}",
        f"stage2_patients: {(patient_means >= 140).sum()}",
        f"highest_patient: {patient_ids[highest]}",
        f"highest_patient_mean: {patient_means[highest]:.1f}",
        f"peak_hour_column: {hour_columns[peak]}",
        f"peak_hour_mean: {hour_means[peak]:.1f}",
        f"high_monitor: {high}",
        f"monitor_offset: {patient_means[monitors == high].mean() - others.mean():.1f}",
        f"stage2_other_monitors: {(others >= 140).sum()}",
    ]
    counts = {name: int((monitors == name).sum()) for name in names}
    return summary, counts


def failing(result: dict) -> set[str]:
    return {test["test-name"] for test in result["tests"] if not test["passed"]}


def detail(result: dict, name: str) -> str:
    return next(test["detail"] for test in result["tests"] if test["test-name"] == name)


def answers_cost(*keys: str) -> int:
    return sum(ANSWER_POINTS[key] for key in keys)


def handout_grade(root: Path) -> dict:
    """Grade through the handout's own entry point, the command students run."""
    finished = subprocess.run(
        [sys.executable, "-B", str(FORK / "check_assignment.py"), str(root), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.stdout, f"the handout checker printed nothing: {finished.stderr}"
    return json.loads(finished.stdout)


def build(root: Path, summary: list[str], counts: dict[str, int], records: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    shutil.copy(FORK / DATA_FILE, root / DATA_FILE)
    requirements = (FORK / "requirements.txt").read_text(encoding="utf-8")
    (root / "requirements.txt").write_text(requirements, encoding="utf-8")
    pinned = re.search(r"numpy==([^\s]+)", requirements).group(1)
    (root / ".python-version").write_text("3.13\n", encoding="utf-8")
    output = root / "output"
    output.mkdir(exist_ok=True)
    (output / "environment.txt").write_text(
        f"python: Python 3.13.14\nnumpy: {pinned}\ninterpreter: /home/alice/a03/.venv/bin/python\n",
        encoding="utf-8",
    )
    (output / "record_count.txt").write_text(f"{records}\n", encoding="utf-8")
    (output / COUNTS_NAME).write_text(
        "".join(f"{count:>7} {name}\n" for name, count in sorted(counts.items())), encoding="utf-8"
    )
    (output / "vitals_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def replaced(summary: list[str], **values: str) -> list[str]:
    """The same summary with some values replaced."""
    lines = []
    for line in summary:
        key, _, value = line.partition(": ")
        lines.append(f"{key}: {values.get(key, value)}")
    return lines


def run() -> None:
    summary, counts = solve(FORK / DATA_FILE)
    records = sum(counts.values())
    answers = dict(line.partition(": ")[::2] for line in summary)

    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a03-selftest-") as temporary:
        workspace = Path(temporary)

        empty = workspace / "empty"
        empty.mkdir()
        assert grade_submission(empty)["score"] == 0, "an empty submission must score nothing"
        assert handout_grade(empty) == grade_submission(empty), "the handout must report what the course reports"

        complete = workspace / "complete"
        build(complete, summary, counts, records)
        result = grade_submission(complete)
        assert result["score"] == 100, f"a correct submission scored {result['score']}: {failing(result)}"
        assert handout_grade(complete) == result, "the handout must report what the course reports"

        (complete / "notes.md").write_text("scratch notes\n", encoding="utf-8")
        assert grade_submission(complete)["score"] == 100, "an extra file must not cost points"

        # Same answers, rounded and spaced differently, with the keys reordered,
        # a BOM in front, CRLF line endings, `M01: 58` counts, and a record
        # count line that carries a word after the number.
        loose = workspace / "loose"
        rounded = []
        for line in reversed(summary):
            key, _, value = line.partition(": ")
            rounded.append(f"  {key.upper()}  :  {round(float(value)) if _is_number(value) else value}  ")
        build(loose, rounded, counts, records)
        (loose / "output" / "vitals_summary.txt").write_text(
            "﻿" + "\r\n".join(rounded) + "\r\n", encoding="utf-8"
        )
        (loose / "output" / "record_count.txt").write_text(f"   {records} patients\n", encoding="utf-8")
        (loose / "output" / COUNTS_NAME).write_text(
            "﻿" + "\r\n".join(f"{name}: {count}" for name, count in counts.items()) + "\r\n",
            encoding="utf-8",
        )
        result = grade_submission(loose)
        assert result["score"] == 100, f"loose formatting scored {result['score']}: {failing(result)}"

        # NumPy scalar reprs, which a beginner prints constantly.
        repred = workspace / "repred"
        build(
            repred,
            replaced(
                summary,
                patients=f"np.int64({answers['patients']})",
                mean_sbp=f"np.float64({answers['mean_sbp']})",
                min_sbp=f"np.int64({answers['min_sbp']})",
                max_sbp=f"numpy.int64({answers['max_sbp']})",
                stage2_patients=f"np.int64({answers['stage2_patients']})",
                high_monitor=f"np.str_('{answers['high_monitor']}')",
                peak_hour_column=f"np.str_('{answers['peak_hour_column']}')",
            ),
            counts,
            records,
        )
        result = grade_submission(repred)
        assert result["score"] == 100, f"NumPy scalar reprs scored {result['score']}: {failing(result)}"

        # The sample standard deviation is accepted beside the population one.
        sampled = workspace / "sampled"
        readings = np.array(
            [row.split(",")[2:] for row in (FORK / DATA_FILE).read_text(encoding="utf-8").splitlines()[1:]]
        ).astype(int)
        build(sampled, replaced(summary, sd_sbp=f"{readings.std(ddof=1):.4f}"), counts, records)
        assert grade_submission(sampled)["score"] == 100, "the sample SD must be accepted"

        # One wrong answer costs that answer only: the score is diagnostic.
        drifted = workspace / "drifted"
        build(drifted, replaced(summary, mean_sbp=f"{float(answers['mean_sbp']) + 3.0:.1f}"), counts, records)
        result = grade_submission(drifted)
        assert failing(result) == {"answer: mean_sbp"}, failing(result)
        assert result["score"] == 100 - answers_cost("mean_sbp"), result["score"]
        assert "allowed difference" in detail(result, "answer: mean_sbp")

        # Two wrong answers in two different groups, the rest still earned.
        mixed = workspace / "mixed"
        build(mixed, replaced(summary, sd_sbp="99.9", high_monitor="M99"), counts, records)
        result = grade_submission(mixed)
        assert failing(result) == {"answer: sd_sbp", "answer: high_monitor"}, failing(result)
        assert result["score"] == 100 - answers_cost("sd_sbp", "high_monitor"), result["score"]
        assert "which is not the monitor whose patients" in detail(result, "answer: high_monitor")
        assert handout_grade(mixed) == result, "the handout must report what the course reports"

        # Right shape, wrong numbers: the artifact checks still pass.
        wrong = workspace / "wrong"
        build(wrong, [f"{line.partition(':')[0]}: 1" for line in summary], counts, records)
        result = grade_submission(wrong)
        assert failing(result) == {f"answer: {key}" for key in ANSWER_POINTS}, failing(result)
        assert result["score"] == ARTIFACT_POINTS, result["score"]

        # Half the analysis, and an untimestamped counts file.
        partial = workspace / "partial"
        build(partial, summary[:6], counts, records)
        (partial / "output" / COUNTS_NAME).rename(partial / "output" / "monitor_counts.txt")
        result = grade_submission(partial)
        assert failing(result) == {"monitor counts artifact", "summary artifact format"} | {
            f"answer: {key}" for key in list(ANSWER_POINTS)[6:]
        }, failing(result)
        assert 0 < result["score"] < 100

        # Counting the header line is named as such.
        miscounted = workspace / "miscounted"
        build(miscounted, summary, counts, records + 1)
        result = grade_submission(miscounted)
        assert failing(result) == {"record count artifact"}, failing(result)
        assert "counts the header line" in detail(result, "record count artifact")

        # A probe from outside the project environment.
        unactivated = workspace / "unactivated"
        build(unactivated, summary, counts, records)
        (unactivated / "output" / "environment.txt").write_text(
            "python: Python 3.13.14\nnumpy: 2.3.3\ninterpreter: /usr/bin/python3\n", encoding="utf-8"
        )
        assert failing(grade_submission(unactivated)) == {"environment probe"}

        # The Python version is never graded: any .python-version and any python line, even none, pass.
        for series, reported, expected in (("3.14", "Python 3.14.4", set()), ("3.12.7", "Python 3.12.7", set()),
                                           ("latest", "Python 3.13.14", set()),
                                           ("3.13", "Python", set())):
            other = workspace / f"python-{series}-{len(reported)}"
            build(other, summary, counts, records)
            (other / ".python-version").write_text(series + "\n", encoding="utf-8")
            probe = (other / "output" / "environment.txt").read_text(encoding="utf-8")
            (other / "output" / "environment.txt").write_text(
                probe.replace("Python 3.13.14", reported), encoding="utf-8")
            assert failing(grade_submission(other)) == expected, (series, reported)
        bare = workspace / "no-python-records"  # neither file has to mention Python at all
        build(bare, summary, counts, records)
        (bare / ".python-version").unlink()
        probe = (bare / "output" / "environment.txt").read_text(encoding="utf-8")
        (bare / "output" / "environment.txt").write_text(
            "".join(line for line in probe.splitlines(keepends=True) if not line.startswith("python:")),
            encoding="utf-8")
        assert failing(grade_submission(bare)) == set(), failing(grade_submission(bare))

        # The counting artifacts must not be gradeable against an edited dataset.
        tampered = workspace / "tampered"
        build(tampered, summary, counts, records)
        rows = (tampered / DATA_FILE).read_text(encoding="utf-8").splitlines()[:4]
        (tampered / DATA_FILE).write_text("\n".join(rows) + "\n", encoding="utf-8")
        result = grade_submission(tampered)
        assert "record count artifact" in failing(result)
        assert all(
            "not the supplied dataset" in test["detail"]
            for test in result["tests"]
            if not test["passed"] and test["test-name"] not in {"summary artifact format"}
        ), failing(result)

    print(
        "Assignment 03 checks: empty, correct, extra-file, loosely formatted, NumPy-repr, "
        "sample-SD, one-wrong, two-wrong, all-wrong, partial, miscounted, unactivated and "
        "tampered-data submissions all score as intended."
    )


def run_handout_copy() -> None:
    """Students run locally exactly the checks GitHub runs: the handout's copy is the course's."""
    workflow = (FORK / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert re.search(r'^  CHECKS_PATH: "03/assignment_checks"$', workflow, re.MULTILINE), (
        "tests.yml does not download from 03/assignment_checks"
    )
    listed = re.search(r"^  CHECKS_FILES: \|\n((?:    \S.*\n)+)", workflow, re.MULTILINE)
    assert listed is not None, "tests.yml lists no CHECKS_FILES"
    files = listed.group(1).split()

    # The list names every checker file here, so a download never misses a module.
    course_owned = [path.name for path in CHECKS.glob("*.py")]
    course_owned += [f".github/test/{path.name}" for path in (CHECKS / ".github" / "test").iterdir() if path.is_file()]
    assert sorted(files) == sorted(course_owned), (sorted(files), sorted(course_owned))

    for name in files:
        assert (FORK / name).read_bytes() == (CHECKS / name).read_bytes(), (
            f"03/assignment/{name} differs from 03/assignment_checks/{name}; copy the course-owned file over it"
        )

    # Besides the checks, the handout's only Python file is the scaffold the student completes.
    handout_python = {path.name for path in FORK.glob("*.py")}
    assert handout_python == {name for name in files if "/" not in name and name.endswith(".py")} | {
        "analysis.py"
    }, sorted(handout_python)

    print(f"Assignment 03 handout: all {len(files)} check files match 03/assignment_checks byte for byte.")


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    run()
    run_handout_copy()
