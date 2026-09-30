"""Development regression checks for the Assignment 03 checks.

Builds submissions in ignored `scratch/`, computes the correct answers with
NumPy (independently of the pure-Python recomputation the checks use), and
confirms what each kind of submission scores. Every submission is also graded
with the checks as committed at HEAD, and no check HEAD passed may fail now,
except a check a submission is built to show tightened: Assignment 03 opens to
students on 2026-09-30, and from then on a change to the checks may only ever
raise a score. It also confirms that the handout in `03/assignment/` ships the
course-owned checks byte for byte, so a student's local run reports exactly
what the GitHub run reports.

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
from _public_checks import ANSWER_KEYS, DATA_FILE, MONITORS, PUBLIC_CHECKS, SUPPLIED_READINGS  # noqa: E402
from grading import POINTS, grade_submission  # noqa: E402

COUNTS_NAME = "monitor_counts_20260922_101500.txt"
# The released checks each change is compared with. Once a change is committed
# it becomes the baseline for the next, so scores can only ratchet upward.
BASELINE_REVISION = "HEAD"
CHECK_POINTS = dict(zip((check.name for check in PUBLIC_CHECKS), POINTS, strict=True))
ANSWER_POINTS = {key: CHECK_POINTS[f"answer: {key}"] for key in ANSWER_KEYS}
ARTIFACT_POINTS = 100 - sum(ANSWER_POINTS.values())
FORMAT_POINTS = CHECK_POINTS["summary artifact format"]
NUMPY_CHECK, INTERPRETER_CHECK = "environment probe: numpy", "environment probe: interpreter"
NAME_CHECK = "monitor counts: timestamped name"
MONITOR_CHECKS = {f"monitor counts: {monitor}" for monitor in MONITORS}
# Checks the baseline ran as one and the current checks run in parts: a submission that passed the
# whole check passes every part. Once the split is committed, the baseline has the parts' names.
SPLIT_CHECKS = {
    "environment probe": {NUMPY_CHECK, INTERPRETER_CHECK},
    "monitor counts artifact": {NAME_CHECK} | MONITOR_CHECKS,
}


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


def checker_report(checks: Path, root: Path) -> dict:
    """Grade through the `check_assignment.py` in `checks`, as a student or CI runs it."""
    finished = subprocess.run(
        [sys.executable, "-B", str(checks / "check_assignment.py"), str(root), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.stdout, f"{checks / 'check_assignment.py'} printed nothing: {finished.stderr}"
    return json.loads(finished.stdout)


def handout_grade(root: Path) -> dict:
    """Grade through the handout's own entry point, the command students run."""
    return checker_report(FORK, root)


def baseline_checks(directory: Path) -> Path:
    """Write the checks as committed at BASELINE_REVISION into `directory`."""
    listed = subprocess.run(
        ["git", "ls-tree", "--name-only", BASELINE_REVISION, "./"],
        cwd=CHECKS, capture_output=True, text=True, check=False,
    )
    assert listed.returncode == 0, f"git cannot list the checks at {BASELINE_REVISION}: {listed.stderr}"
    directory.mkdir()
    for name in listed.stdout.split():
        if name.endswith(".py"):
            shown = subprocess.run(
                ["git", "show", f"{BASELINE_REVISION}:./{name}"], cwd=CHECKS, capture_output=True, check=False
            )
            assert shown.returncode == 0, f"git cannot show {name} at {BASELINE_REVISION}: {shown.stderr}"
            (directory / name).write_bytes(shown.stdout)
    return directory


def no_check_lost(root: Path, result: dict, baseline: Path, tightened: frozenset[str] = frozenset()) -> bool:
    """Assert the baseline passed nothing the current checks fail; return whether the score rose.

    `tightened` names the checks this submission shows failing on purpose: a
    stricter rule made before the assignment opened. Once that rule is committed,
    the baseline fails them too and the allowance goes unused.
    """
    before = checker_report(baseline, root)
    passed_before = {
        part for test in before["tests"] if test["passed"] and test["test-name"] not in tightened
        for part in SPLIT_CHECKS.get(test["test-name"], {test["test-name"]})
    }
    passed_now = {test["test-name"] for test in result["tests"] if test["passed"]}
    assert passed_before <= passed_now, (
        f"{root.name}: {sorted(passed_before - passed_now)} passed under the checks at "
        f"{BASELINE_REVISION} and fail now; a released assignment's checks may only get more lenient"
    )
    lost = sum(test["score"] for test in before["tests"] if test["test-name"] in tightened)
    assert result["score"] >= before["score"] - lost, (root.name, before["score"], result["score"])
    return result["score"] > before["score"]


def build(root: Path, summary: list[str], counts: dict[str, int], records: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    shutil.copy(FORK / DATA_FILE, root / DATA_FILE)
    project = (FORK / "pyproject.toml").read_text(encoding="utf-8")
    (root / "pyproject.toml").write_text(project, encoding="utf-8")
    shutil.copy(FORK / "uv.lock", root / "uv.lock")
    pinned = re.search(r'"numpy==([^"\s]+)"', project).group(1)
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
    assert (FORK / DATA_FILE).read_bytes().decode("utf-8") == SUPPLIED_READINGS, (
        f"03/assignment/{DATA_FILE} changed; copy it into SUPPLIED_READINGS at the end of _public_checks.py"
    )
    summary, counts = solve(FORK / DATA_FILE)
    assert tuple(counts) == MONITORS, f"MONITORS in _public_checks.py must list {sorted(counts)}"
    records = sum(counts.values())
    answers = dict(line.partition(": ")[::2] for line in summary)

    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="a03-selftest-") as temporary:
        workspace = Path(temporary)
        baseline = baseline_checks(workspace / "baseline-checks")
        compared: set[str] = set()
        raised: set[str] = set()

        def graded(root: Path, tightened: frozenset[str] = frozenset()) -> dict:
            """Grade with the current checks, and prove the baseline scored this submission no higher."""
            result = grade_submission(root)
            compared.add(root.name)
            if no_check_lost(root, result, baseline, tightened):
                raised.add(root.name)
            return result

        empty = workspace / "empty"
        empty.mkdir()
        assert graded(empty)["score"] == 0, "an empty submission must score nothing"
        assert handout_grade(empty) == graded(empty), "the handout must report what the course reports"

        complete = workspace / "complete"
        build(complete, summary, counts, records)
        result = graded(complete)
        assert result["score"] == 100, f"a correct submission scored {result['score']}: {failing(result)}"
        assert handout_grade(complete) == result, "the handout must report what the course reports"

        (complete / "notes.md").write_text("scratch notes\n", encoding="utf-8")
        assert graded(complete)["score"] == 100, "an extra file must not cost points"

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
        result = graded(loose)
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
        result = graded(repred)
        assert result["score"] == 100, f"NumPy scalar reprs scored {result['score']}: {failing(result)}"

        # The sample standard deviation is accepted beside the population one.
        sampled = workspace / "sampled"
        readings = np.array(
            [row.split(",")[2:] for row in (FORK / DATA_FILE).read_text(encoding="utf-8").splitlines()[1:]]
        ).astype(int)
        build(sampled, replaced(summary, sd_sbp=f"{readings.std(ddof=1):.4f}"), counts, records)
        assert graded(sampled)["score"] == 100, "the sample SD must be accepted"

        # One wrong answer costs that answer only: the score is diagnostic.
        drifted = workspace / "drifted"
        build(drifted, replaced(summary, mean_sbp=f"{float(answers['mean_sbp']) + 3.0:.1f}"), counts, records)
        result = graded(drifted)
        assert failing(result) == {"answer: mean_sbp"}, failing(result)
        assert result["score"] == 100 - answers_cost("mean_sbp"), result["score"]
        assert "allowed difference" in detail(result, "answer: mean_sbp")

        # Two wrong answers in two different groups, the rest still earned.
        mixed = workspace / "mixed"
        build(mixed, replaced(summary, sd_sbp="99.9", high_monitor="M99"), counts, records)
        result = graded(mixed)
        assert failing(result) == {"answer: sd_sbp", "answer: high_monitor"}, failing(result)
        assert result["score"] == 100 - answers_cost("sd_sbp", "high_monitor"), result["score"]
        assert "which is not the monitor whose patients" in detail(result, "answer: high_monitor")
        assert handout_grade(mixed) == result, "the handout must report what the course reports"

        # Right shape, wrong numbers: the artifact checks still pass.
        wrong = workspace / "wrong"
        build(wrong, [f"{line.partition(':')[0]}: 1" for line in summary], counts, records)
        result = graded(wrong)
        assert failing(result) == {f"answer: {key}" for key in ANSWER_POINTS}, failing(result)
        assert result["score"] == ARTIFACT_POINTS, result["score"]

        # Half the analysis, and an untimestamped counts file. Each missing answer
        # costs its own check only; the format check passes on the readable half.
        partial = workspace / "partial"
        build(partial, summary[:6], counts, records)
        (partial / "output" / COUNTS_NAME).rename(partial / "output" / "monitor_counts.txt")
        result = graded(partial)
        assert failing(result) == {NAME_CHECK} | {
            f"answer: {key}" for key in list(ANSWER_POINTS)[6:]
        }, failing(result)
        assert "has no `highest_patient` line" in detail(result, "answer: highest_patient")
        assert 0 < result["score"] < 100

        # One readable answer is enough for the format check.
        single = workspace / "single-answer"
        build(single, [line for line in summary if line.startswith("high_monitor:")], counts, records)
        result = graded(single)
        assert failing(result) == {f"answer: {key}" for key in ANSWER_POINTS if key != "high_monitor"}, (
            failing(result)
        )

        # A summary with no readable answer fails the format check, and each
        # answer check names its own problem.
        unreadable = workspace / "unreadable-summary"
        build(unreadable, ["notes: pending", "mean_sbp: high", "high_monitor:"], counts, records)
        result = graded(unreadable)
        assert failing(result) == {"summary artifact format"} | {f"answer: {key}" for key in ANSWER_POINTS}, (
            failing(result)
        )
        assert result["score"] == ARTIFACT_POINTS - FORMAT_POINTS, result["score"]
        assert "no readable line" in detail(result, "summary artifact format")
        assert "not a number" in detail(result, "answer: mean_sbp")
        assert "no value" in detail(result, "answer: high_monitor")
        (unreadable / "output" / "vitals_summary.txt").write_text("", encoding="utf-8")
        assert "summary artifact format" in failing(graded(unreadable))

        # Demo 1 ends its pipeline with `head -n 5`, which drops the sixth monitor.
        cut = workspace / "cut-counts"
        build(cut, summary, counts, records)
        kept = sorted(counts)[:5]
        (cut / "output" / COUNTS_NAME).write_text(
            "".join(f"{counts[name]:>7} {name}\n" for name in kept), encoding="utf-8"
        )
        result = graded(cut)
        dropped = sorted(set(counts) - set(kept))
        # Each monitor is its own check, so the five counted monitors keep their points.
        assert failing(result) == {f"monitor counts: {name}" for name in dropped}, failing(result)
        assert result["score"] == 100 - sum(CHECK_POINTS[f"monitor counts: {name}"] for name in dropped)
        assert f"with none for {', '.join(dropped)}" in detail(result, f"monitor counts: {dropped[0]}"), detail(
            result, f"monitor counts: {dropped[0]}"
        )
        assert "no `head` stage" in detail(result, f"monitor counts: {dropped[0]}")

        # A label may sit in brackets or quotes, or be followed by a note, when it
        # names no other id of the same kind.
        data_lines = (FORK / DATA_FILE).read_text(encoding="utf-8").splitlines()
        patient_ids = [line.split(",")[0] for line in data_lines[1:]]
        hour_columns = data_lines[0].split(",")[2:]
        other_patient = next(name for name in patient_ids if name != answers["highest_patient"])
        other_hour = next(name for name in hour_columns if name != answers["peak_hour_column"])
        other_monitor = next(name for name in sorted(counts) if name != answers["high_monitor"])
        for name, values in (
            ("bracketed-labels", {
                "highest_patient": f'["{answers["highest_patient"]}"]',
                "peak_hour_column": f"['{answers['peak_hour_column']}']",
                "high_monitor": f"['{answers['high_monitor']}']",
            }),
            ("annotated-labels", {
                "highest_patient": f"{answers['highest_patient']} ({answers['highest_patient_mean']} mmHg)",
                "peak_hour_column": f"{answers['peak_hour_column']} (mean {answers['peak_hour_mean']})",
                "high_monitor": f"[np.str_('{answers['high_monitor']}')] is {answers['monitor_offset']} higher",
            }),
        ):
            labelled = workspace / name
            build(labelled, replaced(summary, **values), counts, records)
            result = graded(labelled)
            assert result["score"] == 100, f"{name} scored {result['score']}: {failing(result)}"
        hedged = workspace / "hedged-labels"
        build(
            hedged,
            replaced(
                summary,
                highest_patient=f"['{answers['highest_patient']}', '{other_patient}']",
                peak_hour_column=f"{answers['peak_hour_column']} / {other_hour}",
                high_monitor=f"{answers['high_monitor']} or {other_monitor}",
            ),
            counts,
            records,
        )
        result = graded(hedged)
        assert failing(result) == {f"answer: {key}" for key in ("highest_patient", "peak_hour_column",
                                                                 "high_monitor")}, failing(result)
        # The id has to lead the value and stand as a whole word.
        build(
            hedged,
            replaced(
                summary,
                peak_hour_column=f"{answers['peak_hour_column']}0",
                high_monitor=f"not {answers['high_monitor']}",
            ),
            counts,
            records,
        )
        assert failing(graded(hedged)) == {"answer: peak_hour_column", "answer: high_monitor"}

        # Counting the header line is named as such.
        miscounted = workspace / "miscounted"
        build(miscounted, summary, counts, records + 1)
        result = graded(miscounted)
        assert failing(result) == {"record count artifact"}, failing(result)
        assert "counts the header line" in detail(result, "record count artifact")

        # Which numpy and which interpreter are never graded: a probe from outside
        # the project environment, another numpy, or a Windows path all pass, and
        # so does a submission whose pyproject.toml and uv.lock are gone.
        for name, probe in (
            ("unactivated", "python: Python 3.13.14\nnumpy: 2.3.3\ninterpreter: /usr/bin/python3\n"),
            ("other-numpy", "numpy: 2.2.6\ninterpreter: /home/alice/a03/.venv/bin/python\n"),
            ("windows", "NumPy: numpy 2.3.3\ninterpreter: C:\\Users\\alice\\a03\\.venv\\Scripts\\python.exe\n"),
        ):
            probed = workspace / name
            build(probed, summary, counts, records)
            (probed / "output" / "environment.txt").write_text(probe, encoding="utf-8")
            assert failing(graded(probed)) == set(), (name, failing(graded(probed)))
        unpinned = workspace / "no-project-files"
        build(unpinned, summary, counts, records)
        (unpinned / "pyproject.toml").unlink()
        (unpinned / "uv.lock").unlink()
        assert failing(graded(unpinned)) == set(), failing(graded(unpinned))

        # The probe still has to record a numpy version and an interpreter, each its own check. Writing
        # every probe line with `>` leaves only the last, and the feedback says `>` replaced the file.
        for name, probe, check, problem in (
            ("numpy-unversioned", "numpy: installed\ninterpreter: /usr/bin/python3\n", NUMPY_CHECK,
             "no version number"),
            ("interpreter-empty", "numpy: 2.3.3\ninterpreter:\n", INTERPRETER_CHECK, "is empty"),
            ("numpy-missing", "python: Python 3.13.14\ninterpreter: /usr/bin/python3\n", NUMPY_CHECK,
             "no `numpy` line"),
            ("probe-overwritten", "interpreter: /home/alice/a03/.venv/bin/python\n", NUMPY_CHECK,
             "`>` replaces the file each time"),
        ):
            probed = workspace / name
            build(probed, summary, counts, records)
            (probed / "output" / "environment.txt").write_text(probe, encoding="utf-8")
            result = graded(probed)
            assert failing(result) == {check}, (name, failing(result))
            assert result["score"] == 100 - CHECK_POINTS[check], (name, result["score"])
            assert problem in detail(result, check), (name, detail(result, check))
        assert "`>`" not in detail(graded(workspace / "numpy-missing"), NUMPY_CHECK)

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
            assert failing(graded(other)) == expected, (series, reported)
        bare = workspace / "no-python-records"  # neither file has to mention Python at all
        build(bare, summary, counts, records)
        (bare / ".python-version").unlink()
        probe = (bare / "output" / "environment.txt").read_text(encoding="utf-8")
        (bare / "output" / "environment.txt").write_text(
            "".join(line for line in probe.splitlines(keepends=True) if not line.startswith("python:")),
            encoding="utf-8")
        assert failing(graded(bare)) == set(), failing(graded(bare))

        # The checks read only output/. Editing or deleting a supplied file, the data, the scaffold, or the
        # requirement list, changes nothing: not a score, not a word of feedback.
        tampered = workspace / "tampered"
        build(tampered, summary, counts, records)
        untouched = graded(tampered)
        assert untouched["score"] == 100, failing(untouched)
        rows = (tampered / DATA_FILE).read_text(encoding="utf-8").splitlines()[:4]
        (tampered / DATA_FILE).write_text("\n".join(rows) + "\n", encoding="utf-8")
        (tampered / "analysis.py").write_text("raise SystemExit('never run')\n", encoding="utf-8")
        (tampered / "pyproject.toml").write_text(
            (tampered / "pyproject.toml").read_text(encoding="utf-8").replace("numpy==", "numpy==1.0 # was "),
            encoding="utf-8")
        assert graded(tampered) == untouched, failing(graded(tampered))
        (tampered / DATA_FILE).unlink()
        (tampered / "data").rmdir()
        (tampered / "pyproject.toml").unlink()
        (tampered / "uv.lock").unlink()
        assert graded(tampered) == untouched, failing(graded(tampered))
        wrong_mean = workspace / "wrong-mean-no-data"
        build(wrong_mean, replaced(summary, mean_sbp=f"{float(answers['mean_sbp']) + 3.0:.1f}"), counts, records)
        with_data = graded(wrong_mean)
        (wrong_mean / DATA_FILE).write_text("\n".join(rows) + "\n", encoding="utf-8")
        assert graded(wrong_mean) == with_data
        (wrong_mean / DATA_FILE).unlink()
        assert graded(wrong_mean) == with_data
        # A wrong value names the supplied dataset, in case the student's copy was edited.
        assert f"the supplied {DATA_FILE} gives" in detail(with_data, "answer: mean_sbp")

        # Validation found correct work the released checks marked wrong. Each of these
        # now scores 100, and the baseline comparison proves none of them lost a check.
        mmhg_keys = ("mean_sbp", "sd_sbp", "highest_patient_mean", "peak_hour_mean", "monitor_offset")
        exact = {
            "mean_sbp": readings.mean(),
            "sd_sbp": readings.std(),
            "highest_patient_mean": readings.mean(axis=1).max(),
            "peak_hour_mean": readings.mean(axis=0).max(),
        }
        monitor_list = np.array(
            [line.split(",")[1] for line in (FORK / DATA_FILE).read_text(encoding="utf-8").splitlines()[1:]]
        )
        patient_means = readings.mean(axis=1)
        exact["monitor_offset"] = (
            patient_means[monitor_list == answers["high_monitor"]].mean()
            - patient_means[monitor_list != answers["high_monitor"]].mean()
        )
        assert any(abs(int(exact[key]) - exact[key]) > 0.6 for key in mmhg_keys), "no key tests truncation"
        json_answers = "{\n" + ",\n".join(
            f'  "{key}": ' + (f'"{value}"' if not _is_number(value) else value) for key, value in answers.items()
        ) + "\n}\n"
        counts_lines = {
            "dash": "{name} - {count}", "arrow": "{name} -> {count}", "parens": "{name} ({count})",
            "unit": "{name}: {count} patients", "sentence": "{count} patients on {name}",
            "bullet": "- {name}: {count}", "note": "{count} {name} ({share:.1f}%)",
        }
        for name, change in (
            ("truncated-mmhg", ("summary", replaced(summary, **{key: str(int(exact[key])) for key in mmhg_keys}))),
            ("markdown-keys", ("summary", [f"- `{line.replace(': ', '`: ', 1)}" for line in summary])),
            ("bold-bullets", ("summary", [f"* **{line.replace(': ', '**: ', 1)}" for line in summary])),
            ("bold-colon", ("summary", [f"**{line.replace(': ', ':** ', 1)}" for line in summary])),
            ("numbered-keys", ("summary", [f"{n}. {line}" for n, line in enumerate(summary, 1)])),
            ("json-summary", ("summary-text", json_answers)),
            ("equals-summary", ("summary", [line.replace(": ", " = ", 1) for line in summary])),
            ("printed-summary", ("summary", [line.replace(": ", " ", 1) for line in summary])),
            ("grouped-digits", ("summary", replaced(summary, readings=f"{int(answers['readings']):_}"))),
            ("leading-note-labels", ("summary", replaced(
                summary,
                highest_patient=f"patient {answers['highest_patient']}",
                peak_hour_column=f"hour {int(answers['peak_hour_column'][-2:])} ({answers['peak_hour_column']})",
                high_monitor=f"monitor {answers['high_monitor']}",
            ))),
            ("markdown-probe", ("environment", "- `python`: Python 3.13.14\n- `numpy`: 2.3.3\n"
                                                "- `interpreter`: /home/alice/a03/.venv/bin/python\n")),
            ("described-probe", ("environment", "numpy version: 2.3.3\ninterpreter path: /usr/bin/python3\n")),
            ("pinned-probe", ("environment", "numpy==2.3.3\nsys.executable: /usr/bin/python3\n")),
        ) + tuple(
            (f"counts-{label}", ("counts", "".join(
                pattern.format(name=monitor, count=count, share=100 * count / records) + "\n"
                for monitor, count in counts.items()
            )))
            for label, pattern in counts_lines.items()
        ):
            lenient = workspace / name
            build(lenient, summary, counts, records)
            kind, content = change
            if kind == "summary":
                (lenient / "output" / "vitals_summary.txt").write_text("\n".join(content) + "\n", encoding="utf-8")
            elif kind == "summary-text":
                (lenient / "output" / "vitals_summary.txt").write_text(content, encoding="utf-8")
            elif kind == "environment":
                (lenient / "output" / "environment.txt").write_text(content, encoding="utf-8")
            else:
                (lenient / "output" / COUNTS_NAME).write_text(content, encoding="utf-8")
            result = graded(lenient)
            assert result["score"] == 100, f"{name} scored {result['score']}: {failing(result)}"

        # A file saved as UTF-16 by Windows PowerShell 5.1, or with one cp1252 byte in a
        # note, reads as its text; a stray digit int() cannot read is skipped, not raised.
        encoded = workspace / "utf16-artifacts"
        build(encoded, summary, counts, records)
        for artifact in (encoded / "output").iterdir():
            artifact.write_text(artifact.read_text(encoding="utf-8"), encoding="utf-16")
        assert graded(encoded)["score"] == 100, failing(graded(encoded))
        noted = workspace / "cp1252-note"
        build(noted, summary, counts, records)
        with open(noted / "output" / "vitals_summary.txt", "a", encoding="cp1252") as file:
            file.write("sd note: \u00b1 one SD\n")
        with open(noted / "output" / COUNTS_NAME, "a", encoding="utf-8") as file:
            file.write("\u00b2 M01\n" + "9" * 5000 + " total\n")
        assert graded(noted)["score"] == 100, failing(graded(noted))

        # Whitespace at the end of the dataset changes no value; an edited value still does.
        for name, rewrite in (
            ("data-extra-newline", lambda raw: raw + b"\n"),
            ("data-no-final-newline", lambda raw: raw.rstrip(b"\n")),
            ("data-trailing-spaces", lambda raw: raw.replace(b"\n", b"  \n")),
        ):
            spaced = workspace / name
            build(spaced, summary, counts, records)
            (spaced / DATA_FILE).write_bytes(rewrite((FORK / DATA_FILE).read_bytes()))
            assert graded(spaced)["score"] == 100, (name, failing(graded(spaced)))

        # A file named in another letter case is found, as a macOS or Windows disk
        # would find it locally; a misnamed or misplaced one is named in the feedback.
        cased = workspace / "cased-summary"
        build(cased, summary, counts, records)
        (cased / "output" / "vitals_summary.txt").rename(cased / "output" / "Vitals_Summary.txt")
        assert graded(cased)["score"] == 100, failing(graded(cased))
        for name, moved in (("misnamed-summary", "output/vital_summary.txt"), ("root-summary", "vitals_summary.txt")):
            misplaced = workspace / name
            build(misplaced, summary, counts, records)
            (misplaced / "output" / "vitals_summary.txt").rename(misplaced / moved)
            result = graded(misplaced)
            assert result["score"] == ARTIFACT_POINTS - FORMAT_POINTS, result["score"]
            assert f"Found {moved}; rename or move it" in detail(result, "summary artifact format"), name
            assert detail(result, "answer: patients") == detail(result, "summary artifact format"), name

        # A key written twice reads its first line, and the feedback says so.
        appended = workspace / "appended-summary"
        build(appended, summary, counts, records)
        (appended / "output" / "vitals_summary.txt").write_text(
            f"high_monitor: {other_monitor}\n" + "\n".join(summary) + "\n", encoding="utf-8"
        )
        result = graded(appended)
        assert failing(result) == {"answer: high_monitor"}, failing(result)
        assert "appears on 2 lines and the checks read the first" in detail(result, "answer: high_monitor")

        # Counts files that are wrong say what to change.
        data_rows = (FORK / DATA_FILE).read_text(encoding="utf-8").splitlines()[1:]
        for name, content, extension, problem in (
            ("unsorted-counts", "".join(f"{count - count // 3:>7} {monitor}\n" for monitor, count in counts.items())
             + "".join(f"{count // 3:>7} {monitor}\n" for monitor, count in counts.items()), ".txt", "`sort` before it"),
            ("patient-counts", "      1 P0001\n      1 P0002\n", ".txt", "field 2"),
            # `cut -f3` counts readings: no monitor anywhere, and the same field-2 hint.
            ("reading-counts", "      2 111\n      3 130\n", ".txt", "count field 2 of each row, the monitor"),
            # `cut -f2` without `-d','` splits at tabs and passes each CSV row through whole.
            ("undelimited-counts", "".join(f"      1 {line}\n" for line in data_rows), ".txt", "`cut -d',' -f2`"),
        ):
            miscounted_monitors = workspace / name
            build(miscounted_monitors, summary, counts, records)
            (miscounted_monitors / "output" / COUNTS_NAME).unlink()
            (miscounted_monitors / "output" / COUNTS_NAME.replace(".txt", extension)).write_text(
                content, encoding="utf-8"
            )
            result = graded(miscounted_monitors)
            assert failing(result) == MONITOR_CHECKS, (name, failing(result))
            assert problem in detail(result, "monitor counts: M01"), (name, detail(result, "monitor counts: M01"))
            assert len({detail(result, check) for check in MONITOR_CHECKS}) == 1, name

        # A counts file's name only has to carry a run timestamp: another timestamp layout, another
        # extension, or another letter case passes.
        for name in ("monitor_counts_2026-09-26_15-30-00.txt", "monitor_counts_20260926-153000.txt",
                     "monitor_counts_20260926_1015.txt", "monitor_counts_20260926_153000.csv",
                     "Monitor_Counts_20260926_101010.txt"):
            dated = workspace / f"dated-{name}"
            build(dated, summary, counts, records)
            (dated / "output" / COUNTS_NAME).rename(dated / "output" / name)
            assert graded(dated)["score"] == 100, (name, failing(graded(dated)))
        # A name with no timestamp costs only the name check, and says the counts inside are right; an
        # empty `$timestamp` says so. A file outside output/ fails every counts check and says to move it.
        for name, moved, failed, problem in (
            ("undated-counts", "output/monitor_counts.txt", {NAME_CHECK}, "only the name is off"),
            ("empty-timestamp", "output/monitor_counts_.txt", {NAME_CHECK}, "`$timestamp` was empty"),
            ("root-counts", "monitor_counts_20260926_150530.txt", {NAME_CHECK} | MONITOR_CHECKS,
             "at the top level of the repository, with every monitor's count right; move it into output/."),
        ):
            misplaced = workspace / name
            build(misplaced, summary, counts, records)
            (misplaced / "output" / COUNTS_NAME).rename(misplaced / moved)
            result = graded(misplaced)
            assert failing(result) == failed, (name, failing(result))
            assert problem in detail(result, NAME_CHECK), (name, detail(result, NAME_CHECK))
        assert "`$timestamp`" not in detail(graded(workspace / "undated-counts"), NAME_CHECK)
        assert "Move the file there, then commit it." in detail(graded(workspace / "root-counts"), NAME_CHECK)
        # A misplaced file whose counts are also wrong, as `head -n 5` leaves them, is told to rerun the
        # pipeline, not only to move the file.
        root_short = workspace / "root-counts-short"
        build(root_short, summary, counts, records)
        kept = (root_short / "output" / COUNTS_NAME).read_text(encoding="utf-8").splitlines(keepends=True)[:5]
        (root_short / "output" / COUNTS_NAME).unlink()
        (root_short / "monitor_counts_20260926_150530.txt").write_text("".join(kept), encoding="utf-8")
        result = graded(root_short)
        assert failing(result) == {NAME_CHECK} | MONITOR_CHECKS, failing(result)
        message = detail(result, NAME_CHECK)
        assert "with no `head` stage after it" in message and "Move the file there" not in message, message
        assert message.endswith("Rerun the pipeline to save the file under that name, then commit it."), message
        # A script at the top level that makes the counts is not a misplaced counts file.
        scripted = workspace / "root-counts-script"
        build(scripted, summary, counts, records)
        (scripted / "output" / COUNTS_NAME).unlink()
        (scripted / "monitor_counts.sh").write_text(
            "ts=$(date +%Y%m%d_%H%M%S)\ntail -n +2 data/bp_readings.csv | cut -d',' -f2 | sort | uniq -c "
            "> output/monitor_counts_$ts.txt\n", encoding="utf-8"
        )
        result = graded(scripted)
        assert failing(result) == {NAME_CHECK} | MONITOR_CHECKS, failing(result)
        assert all("monitor_counts.sh" not in detail(result, check) for check in {NAME_CHECK} | MONITOR_CHECKS), (
            detail(result, NAME_CHECK)
        )

        # monitor_offset and stage2_other_monitors are also accepted for the monitor high_monitor
        # names, so a wrong pick, as averaging with .sum() makes, costs only the high_monitor check.
        monitor_names = np.array([row.split(",")[1] for row in data_rows])
        row_means = readings.mean(axis=1)
        sums = {name: row_means[monitor_names == name].sum() for name in sorted(counts)}
        summed_pick = max(sums, key=sums.get)
        assert summed_pick != answers["high_monitor"], "the .sum() slip no longer picks another monitor"
        picked_rest = row_means[monitor_names != summed_pick]
        picked = {
            "monitor_offset": f"{row_means[monitor_names == summed_pick].mean() - picked_rest.mean():.2f}",
            "stage2_other_monitors": str(int((picked_rest >= 140).sum())),
        }
        follow_on = workspace / "follow-on-pick"
        build(follow_on, replaced(summary, high_monitor=summed_pick, **picked), counts, records)
        result = graded(follow_on)
        assert failing(result) == {"answer: high_monitor"}, failing(result)
        assert result["score"] == 100 - answers_cost("high_monitor"), result["score"]
        assert f"are right for {summed_pick}, so only this answer is wrong" in detail(result, "answer: high_monitor")
        follow_off = workspace / "follow-on-wrong"
        build(follow_off, replaced(summary, high_monitor=summed_pick, monitor_offset="99.0",
                                   stage2_other_monitors="1"), counts, records)
        result = graded(follow_off)
        assert failing(result) == {f"answer: {key}" for key in ("high_monitor", "monitor_offset",
                                                                 "stage2_other_monitors")}, failing(result)
        assert f"or {picked['stage2_other_monitors']} with {summed_pick}, the monitor your `high_monitor` line " \
               "names" in detail(result, "answer: stage2_other_monitors"), detail(result, "answer: stage2_other_monitors")
        assert "are right for" not in detail(result, "answer: high_monitor")

        # A line whose first number is not the answer says which number the checks read, and a label
        # line that names two ids says so.
        noted_first = workspace / "noted-first"
        build(noted_first, replaced(
            summary,
            readings=f"300 x 12 = {answers['readings']}",
            peak_hour_mean=f"hour {int(answers['peak_hour_column'][-2:])} averages {answers['peak_hour_mean']} mmHg",
            high_monitor=f"{answers['high_monitor']} ({answers['monitor_offset']} mmHg above {other_monitor})",
        ), counts, records)
        result = graded(noted_first)
        assert failing(result) == {"answer: readings", "answer: peak_hour_mean", "answer: high_monitor"}, failing(result)
        assert (f"the checks read its first number, 300, which is not" in detail(result, "answer: readings")
                and f"The line also holds {answers['readings']}; put the answer first" in detail(result, "answer: readings"))
        assert f"The line also holds {answers['peak_hour_mean']}" in detail(result, "answer: peak_hour_mean")
        assert f"which names {answers['high_monitor']} and {other_monitor}" in detail(result, "answer: high_monitor")

        # Whole-file layouts: a Markdown table, a row number before each key, and a dict printed on
        # one line all read like `key: value` lines; so does a record count below a title line.
        for name, content in (
            ("table-summary", "| key | value |\n| --- | --- |\n"
             + "".join(f"| {line.replace(': ', ' | ', 1)} |\n" for line in summary)),
            ("indexed-summary", "".join(f"{n} {line}\n" for n, line in enumerate(summary))),
            ("dict-summary", "{" + ", ".join(
                f"'{key}': " + (value if _is_number(value) else f"'{value}'") for key, value in answers.items()
            ) + "}\n"),
        ):
            laid_out = workspace / name
            build(laid_out, summary, counts, records)
            (laid_out / "output" / "vitals_summary.txt").write_text(content, encoding="utf-8")
            assert graded(laid_out)["score"] == 100, (name, failing(graded(laid_out)))
        titled = workspace / "titled-record-count"
        build(titled, summary, counts, records)
        (titled / "output" / "record_count.txt").write_text(f"Assignment 03 record count\n{records}\n",
                                                           encoding="utf-8")
        assert graded(titled)["score"] == 100, failing(graded(titled))
        # A label with a number in it puts the right count second: it still fails, and the fix says to put the
        # count first rather than to recount.
        for name, content in (("numbered-label-count", f"Task 2.1: {records}\n"),
                              ("bracketed-label-count", f"Records (Task 2.1): {records}\n")):
            labelled = workspace / name
            build(labelled, summary, counts, records)
            (labelled / "output" / "record_count.txt").write_text(content, encoding="utf-8")
            result = graded(labelled)
            assert failing(result) == {"record count artifact"}, (name, failing(result))
            message = detail(result, "record count artifact")
            assert f"The patient count, {records}, comes later on the line, so put it first" in message, message
            assert "Count the lines after the header" not in message, message
        # A note that shows its arithmetic has already dropped the header, so it too is told to put the count
        # first, not that it counts the header line.
        for name, content in (("worked-count", f"{records + 1} lines - 1 header = {records} patients\n"),
                              ("labelled-worked-count", f"records: {records + 1} lines, {records} patients\n")):
            worked = workspace / name
            build(worked, summary, counts, records)
            (worked / "output" / "record_count.txt").write_text(content, encoding="utf-8")
            result = graded(worked)
            assert failing(result) == {"record count artifact"}, (name, failing(result))
            message = detail(result, "record count artifact")
            assert f"The patient count, {records}, comes later on the line, so put it first" in message, message
            assert "counts the header line" not in message, message
        wrong_label = workspace / "numbered-label-wrong-count"
        build(wrong_label, summary, counts, records)
        (wrong_label / "output" / "record_count.txt").write_text(f"Task 2.1: {records + 5}\n", encoding="utf-8")
        assert "Count the lines after the header" in detail(graded(wrong_label), "record count artifact")

        # An empty numpy line is what the probe saves when numpy is not installed: the fix says to install it.
        uninstalled = workspace / "numpy-uninstalled"
        build(uninstalled, summary, counts, records)
        (uninstalled / "output" / "environment.txt").write_text(
            "python: Python 3.13.14\nnumpy: \ninterpreter: /home/alice/a03/.venv/bin/python\n", encoding="utf-8")
        result = graded(uninstalled)
        assert failing(result) == {NUMPY_CHECK}, failing(result)
        assert "Task 1.1's `uv sync`" in detail(result, NUMPY_CHECK)

        # Writes that leave out "\n" run the answers onto one line, `patients: 300readings: 3600...`. Each
        # answer is read up to the next answer's key, so right values score in full however many lines
        # are joined, including a label run straight into the next key (`high_monitor: M02monitor_offset:`).
        lines = [line.rstrip("\n") for line in summary]
        for name, content in (
            ("joined-summary", "".join(lines)),
            ("joined-last-three", "\n".join(lines[:11]) + "\n" + "".join(lines[11:]) + "\n"),
            ("joined-two-runs", "\n".join(lines[:1]) + "\n" + "".join(lines[1:3]) + "\n"
             + "\n".join(lines[3:11]) + "\n" + "".join(lines[11:13]) + "\n" + lines[13] + "\n"),
        ):
            joined = workspace / name
            build(joined, summary, counts, records)
            (joined / "output" / "vitals_summary.txt").write_text(content, encoding="utf-8")
            result = graded(joined)
            assert result["score"] == 100, (name, failing(result))
        assert handout_grade(workspace / "joined-summary") == graded(workspace / "joined-summary")
        # A wrong answer on a joined line quotes only its own value and still says to end each write with "\n".
        wrong_joined = workspace / "joined-wrong"
        build(wrong_joined, summary, counts, records)
        wrong_lines = replaced(summary, sd_sbp="99.9", high_monitor=other_monitor)
        (wrong_joined / "output" / "vitals_summary.txt").write_text("".join(wrong_lines), encoding="utf-8")
        result = graded(wrong_joined)
        assert failing(result) == {"answer: sd_sbp", "answer: high_monitor"}, failing(result)
        for key, value in (("sd_sbp", "99.9"), ("high_monitor", other_monitor)):
            message = detail(result, f"answer: {key}")
            assert f"gives `{key}: {value}`" in message, message
            assert 'end each write with "\\n"' in message, message
        # An empty value run into the next answer says it has no value; the answer after it still scores.
        empty_joined = workspace / "joined-empty-value"
        build(empty_joined, summary, counts, records)
        (empty_joined / "output" / "vitals_summary.txt").write_text(
            "patients: " + "".join(lines[1:3]) + "\n" + "\n".join(lines[3:]) + "\n", encoding="utf-8")
        result = graded(empty_joined)
        assert failing(result) == {"answer: patients"}, failing(result)
        assert "has no value after the colon" in detail(result, "answer: patients")
        assert 'end each write with "\\n"' in detail(result, "answer: patients")
        # A key mentioned in a note on another answer's line never replaces the key's own line.
        noted = workspace / "joined-note"
        build(noted, replaced(summary, mean_sbp=f"{answers['mean_sbp']} (over all readings: 12)"), counts,
              records)
        result = graded(noted)
        assert result["score"] == 100, failing(result)

        # Patient answers computed with axis=0, one mean per hour column, say that a 12-hour mean is
        # axis=1. A count of 0 alone is too common a slip to name its cause.
        hour_means = readings.mean(axis=0)
        peak = int(hour_means.argmax())
        per_column = workspace / "patients-axis0"
        build(per_column, replaced(
            summary,
            stage2_patients=str(int((hour_means >= 140).sum())),
            highest_patient=patient_ids[peak],
            highest_patient_mean=f"{hour_means[peak]:.2f}",
        ), counts, records)
        result = graded(per_column)
        slips = {"answer: stage2_patients", "answer: highest_patient", "answer: highest_patient_mean"}
        assert failing(result) == slips, failing(result)
        assert all("`readings.mean(axis=1)`, one per patient" in detail(result, check) for check in slips)
        zero_only = workspace / "stage2-zero"
        build(zero_only, replaced(summary, stage2_patients="0"), counts, records)
        result = graded(zero_only)
        assert failing(result) == {"answer: stage2_patients"}, failing(result)
        assert "axis=" not in detail(result, "answer: stage2_patients")

        # Labels written as positions, as Demo 3.3 prints them (`Patient  63`, `#1`), counting from 0 or
        # from 1, say to index the ids with the position; a number that is not the position says nothing.
        best = patient_ids.index(answers["highest_patient"])
        peak_hour = hour_columns.index(answers["peak_hour_column"])
        high_position = sorted(counts).index(answers["high_monitor"])
        positions = workspace / "label-positions"
        build(positions, replaced(
            summary,
            highest_patient=f"Patient {best:3d}",
            peak_hour_column=f"#{peak_hour + 1}",
            high_monitor=str(high_position),
        ), counts, records)
        result = graded(positions)
        labels = {"answer: highest_patient", "answer: peak_hour_column", "answer: high_monitor"}
        assert failing(result) == labels, failing(result)
        assert result["score"] == 100 - answers_cost("highest_patient", "peak_hour_column", "high_monitor")
        for check, array in (("answer: highest_patient", "`patient_ids`"), ("answer: peak_hour_column",
                             "`hour_columns`"), ("answer: high_monitor", "`sorted(set(monitors))`")):
            assert "is a position, not a " in detail(result, check), detail(result, check)
            assert array in detail(result, check), detail(result, check)
        not_position = workspace / "label-not-position"
        build(not_position, replaced(summary, highest_patient=str(best + 2)), counts, records)
        result = graded(not_position)
        assert failing(result) == {"answer: highest_patient"}, failing(result)
        assert "is a position" not in detail(result, "answer: highest_patient")

        # Stage 2 counts of readings rather than patients say that the question counts patients.
        monitors = np.array([line.split(",")[1] for line in data_lines[1:]])
        on_others = monitors != answers["high_monitor"]
        counted_readings = workspace / "stage2-readings"
        build(counted_readings, replaced(
            summary,
            stage2_patients=str(int((readings >= 140).sum())),
            stage2_other_monitors=str(int((readings[on_others] >= 140).sum())),
        ), counts, records)
        result = graded(counted_readings)
        stage2 = {"answer: stage2_patients", "answer: stage2_other_monitors"}
        assert failing(result) == stage2, failing(result)
        assert all("counting every hour of every patient" in detail(result, check) for check in stage2)
        assert "every patient on the other monitors" in detail(result, "answer: stage2_other_monitors")
        assert "& others" in detail(result, "answer: stage2_other_monitors")
        assert "other monitors" not in detail(result, "answer: stage2_patients")
        off_by_one = workspace / "stage2-off-by-one"
        build(off_by_one, replaced(summary, stage2_patients=str(int(answers["stage2_patients"]) + 1)), counts,
              records)
        result = graded(off_by_one)
        assert failing(result) == {"answer: stage2_patients"}, failing(result)
        assert "counting every hour" not in detail(result, "answer: stage2_patients")

        # mean_sbp written as the per-hour array `readings.mean(axis=0)` prints: its first element sits
        # within the tolerance of the overall mean, so reading only the first number passed it. An array
        # of several numbers now fails, printed or repr'd, wrapped or not, and says why; a one-item list
        # still reads as its number. The baseline checks predate this rule, so it is a declared tightening.
        mean_sbp_check = frozenset({"answer: mean_sbp"})
        hour_means = readings.mean(axis=0)
        assert abs(hour_means[0] - readings.mean()) <= 0.6, "the per-hour case no longer tests a first-number pass"
        for name, value in (
            ("mean-per-hour", str(hour_means)),
            ("mean-per-hour-repr", repr(hour_means)),
            ("mean-per-hour-list", str(hour_means.tolist())),
        ):
            per_hour = workspace / name
            build(per_hour, replaced(summary, mean_sbp=value), counts, records)
            result = graded(per_hour, mean_sbp_check)
            assert failing(result) == {"answer: mean_sbp"}, (name, failing(result))
            assert result["score"] == 100 - answers_cost("mean_sbp"), (name, result["score"])
            assert "holds 12 numbers" in detail(result, "answer: mean_sbp"), detail(result, "answer: mean_sbp")
            assert "`readings.mean()`, with no `axis=`" in detail(result, "answer: mean_sbp")
        per_hour_sd = workspace / "sd-per-hour"
        build(per_hour_sd, replaced(summary, sd_sbp=str(readings.std(axis=0))), counts, records)
        result = graded(per_hour_sd, frozenset({"answer: sd_sbp"}))
        assert failing(result) == {"answer: sd_sbp"}, failing(result)
        assert "`readings.std()`, with no `axis=`" in detail(result, "answer: sd_sbp")
        one_item = workspace / "mean-one-item-list"
        build(one_item, replaced(summary, mean_sbp=f"[{answers['mean_sbp']}]"), counts, records)
        assert graded(one_item)["score"] == 100, failing(graded(one_item))

        # A fresh handout prints exactly the "Before Task 1" example README.md shows.
        fresh = workspace / "fresh-handout"
        shutil.copytree(FORK, fresh, ignore=shutil.ignore_patterns("__pycache__", ".venv"))
        for artifact in (fresh / "output").iterdir():
            if artifact.name != ".gitkeep":
                artifact.unlink()
        shown = re.search(
            r"Before Task 1, for example, the first two checks report:\n\n```text\n(.*?)```",
            (FORK / "README.md").read_text(encoding="utf-8"),
            re.DOTALL,
        )
        assert shown is not None, "README.md no longer shows the Before Task 1 example"
        printed = subprocess.run(
            [sys.executable, "-B", "check_assignment.py"], cwd=fresh, capture_output=True, text=True, check=False
        ).stdout
        assert shown.group(1) in printed, (shown.group(1), printed)
        assert "[FIX ]   0/4   answer: patients  (same fix as above)\n" in printed, printed
        assert printed.endswith(
            "Score: 0/100\nLeft to fix (100 points): output/environment.txt (both checks); "
            "output/record_count.txt: record count; output/monitor_counts_<timestamp>.txt (all 7 checks); "
            "output/vitals_summary.txt (all 15 checks).\n"), printed

        # A wrong answer says what the data gives, and the report ends by naming what is left to fix.
        printed = subprocess.run(
            [sys.executable, "-B", str(CHECKS / "check_assignment.py"), str(workspace / "mixed")],
            capture_output=True, text=True, check=False,
        ).stdout
        assert f"which is not the monitor whose patients' 12-hour means average highest: the supplied {DATA_FILE} " \
               f"gives `{answers['high_monitor']}`." in printed, printed
        assert printed.endswith("Score: 93/100\nLeft to fix (7 points): output/vitals_summary.txt: sd_sbp and "
                                "high_monitor.\n"), printed
        assert subprocess.run(
            [sys.executable, "-B", str(CHECKS / "check_assignment.py"), str(complete)],
            capture_output=True, text=True, check=False,
        ).stdout.endswith("Score: 100/100\nAll checks passed.\n")

        # Every submission this self-test graded before the lenient-only rule went
        # through the baseline comparison too.
        earlier = {"empty", "complete", "loose", "repred", "sampled", "drifted", "mixed", "wrong", "partial",
                   "miscounted", "unactivated", "no-python-records", "tampered"}
        assert earlier <= compared, sorted(earlier - compared)

    print(
        "Assignment 03 checks: empty, correct, extra-file, loosely formatted, NumPy-repr, "
        "sample-SD, one-wrong, two-wrong, all-wrong, partial, single-answer, unreadable, cut-counts, "
        "bracketed and annotated label, hedged label, miscounted, unactivated, other-numpy, "
        "unversioned-probe, supplied-files-edited, truncated, Markdown and JSON, re-separated, leading-note, "
        "UTF-16, whitespace-edited, renamed, appended, miscounted-monitor, undelimited-cut, dated, undated and "
        "misplaced counts, wrong misplaced counts, empty-timestamp, follow-on pick, first-number, table, numbered, "
        "dict, titled-count, numbered-label count, "
        "uninstalled-numpy, overwritten-probe, joined-answer, per-column patient, position label, stage 2 readings, "
        "per-hour-array and one-item-list submissions all score as intended."
    )
    print(
        f"Assignment 03 checks against {BASELINE_REVISION}: none of {len(compared)} submissions lost a check; "
        + (f"{len(raised)} scored higher ({', '.join(sorted(raised))})." if raised else "none scored higher.")
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
