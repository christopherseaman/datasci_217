"""Materialize fixture submissions and validate the Assignment 01 public grader."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT_DIR))

from _assignment_checks import run_public_checks  # noqa: E402


CORRECT_READINESS = '''# SUPPLIED BLOCK: do not edit this block.
import sys
from pathlib import Path

PROJECT_LABEL = "DataSci 217 Assignment 01"
python_family = f"{sys.version_info.major}.{sys.version_info.minor}"
script_filename = Path(__file__).name
# END SUPPLIED BLOCK

print(f"Python family: {python_family}")
print(f"Project: {PROJECT_LABEL}")
print(f"Script: {script_filename}")
'''

CORRECT_MEASUREMENT = '''measurements = [18, 21, 24, 19]
review_threshold_text = "20"
review_threshold = int(review_threshold_text)
first_measurement = measurements[0]
total = 0
review_count = 0

print(f"First measurement: {first_measurement}")

for measurement in measurements:
    total = total + measurement
    if measurement >= review_threshold:
        status = "review"
        review_count = review_count + 1
    else:
        status = "within range"
    print(f"Measurement {measurement}: {status}")

mean = total / len(measurements)
print(f"Count: {len(measurements)}")
print(f"Total: {total}")
print(f"Mean: {mean:.1f}")
print(f"Review count: {review_count}")
'''

CORRECT_DEBUG = '''participant_count_text = "4"
participant_count = int(participant_count_text)

if participant_count >= 4:
    print("Readiness: complete")

print(f"Participant count: {participant_count}")

next_checkpoint = participant_count + 1
print(f"Next checkpoint: {next_checkpoint}")
'''


def write_fixture(root: Path, overrides: dict[str, str] | None = None) -> None:
    overrides = overrides or {}
    sources = {
        "readiness.py": CORRECT_READINESS,
        "measurement_summary.py": CORRECT_MEASUREMENT,
        "debug_report.py": CORRECT_DEBUG,
    }
    sources.update(overrides)
    for filename, source in sources.items():
        (root / filename).write_text(source, encoding="utf-8")

    for filename in ("make_output.py",):
        shutil.copy2(ASSIGNMENT_DIR / filename, root / filename)

    practice = root / "terminal-practice"
    practice.mkdir()
    (practice / "source.txt").touch()
    (practice / "path-check.txt").touch()
    (root / "output").mkdir()


def generate_output(root: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(root / "make_output.py")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)


def failures_for(root: Path) -> dict[str, str]:
    return {name: error for name, error in run_public_checks(root) if error is not None}


def assert_rejected(case: str, overrides: dict[str, str], expected_check: str, stale: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix=f"ds217-a01-{case}-") as temporary:
        root = Path(temporary)
        write_fixture(root)
        generate_output(root)
        for filename, source in overrides.items():
            (root / filename).write_text(source, encoding="utf-8")
        if stale:
            (root / "output" / "readiness.txt").write_text("old output\n", encoding="utf-8")
        failures = failures_for(root)
        if expected_check not in failures:
            raise AssertionError(
                f"{case}: expected `{expected_check}` to fail, got {sorted(failures)}"
            )
        print(f"[PASS] rejected {case} fixture through {expected_check}")


with tempfile.TemporaryDirectory(prefix="ds217-a01-correct-") as temporary:
    correct_root = Path(temporary)
    write_fixture(correct_root)
    generate_output(correct_root)
    correct_failures = failures_for(correct_root)
    if correct_failures:
        raise AssertionError(f"correct fixture failed: {correct_failures}")
    print("[PASS] accepted correct fixture")
    if "--pytest" in sys.argv:
        shutil.copy2(ASSIGNMENT_DIR / "_assignment_checks.py", correct_root)
        shutil.copy2(ASSIGNMENT_DIR / "test_assignment.py", correct_root)
        pytest_result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_assignment.py", "-q"],
            cwd=correct_root,
            text=True,
        )
        if pytest_result.returncode != 0:
            raise AssertionError("pytest rejected the correct fixture")
        print("[PASS] pytest accepted all public tests for the correct fixture")

assert_rejected(
    "partial",
    {"debug_report.py": CORRECT_DEBUG.replace("participant_count}", "participant_cout}")},
    "debug corrections",
)
assert_rejected(
    "hardcoded",
    {
        "measurement_summary.py": '''measurements = [18, 21, 24, 19]
review_threshold_text = "20"
review_threshold = int(review_threshold_text)
first_measurement = measurements[0]
print("First measurement: 18")
print("Measurement 18: within range")
print("Measurement 21: review")
print("Measurement 24: review")
print("Measurement 19: within range")
print("Count: 4")
print("Total: 82")
print("Mean: 20.5")
print("Review count: 2")
'''
    },
    "measurement structure",
)
assert_rejected(
    "hardcoded_readiness",
    {
        "readiness.py": CORRECT_READINESS.replace(
            'print(f"Python family: {python_family}")\n'
            'print(f"Project: {PROJECT_LABEL}")\n'
            'print(f"Script: {script_filename}")',
            'print("Python family: 3.12")\n'
            'print("Project: DataSci 217 Assignment 01")\n'
            'print("Script: readiness.py")',
        )
    },
    "readiness structure",
)
assert_rejected(
    "partial_slice_loop",
    {
        "measurement_summary.py": '''measurements = [18, 21, 24, 19]
review_threshold_text = "20"
review_threshold = int(review_threshold_text)
first_measurement = measurements[0]
total = first_measurement
review_count = 0

print(f"First measurement: {first_measurement}")
if first_measurement >= review_threshold:
    first_status = "review"
    review_count = review_count + 1
else:
    first_status = "within range"
print(f"Measurement {first_measurement}: {first_status}")

for measurement in measurements[1:]:
    total = total + measurement
    if measurement >= review_threshold:
        status = "review"
        review_count = review_count + 1
    else:
        status = "within range"
    print(f"Measurement {measurement}: {status}")

mean = total / len(measurements)
print(f"Count: {len(measurements)}")
print(f"Total: {total}")
print(f"Mean: {mean:.1f}")
print(f"Review count: {review_count}")
'''
    },
    "measurement structure",
)
assert_rejected(
    "hardcoded_debug_dead_code",
    {
        "debug_report.py": '''participant_count_text = "4"
participant_count = int(participant_count_text)
next_checkpoint = participant_count + 1

if participant_count >= 4:
    pass

if False:
    print(f"Participant count: {participant_count}")
    print(f"Next checkpoint: {next_checkpoint}")

print("Readiness: complete")
print("Participant count: 4")
print("Next checkpoint: 5")
'''
    },
    "debug corrections",
)
assert_rejected(
    "dynamic_file_io",
    {
        "readiness.py": CORRECT_READINESS
        + 'getattr(Path("forbidden.txt"), "write_text")("created")\n'
    },
    "readiness structure",
)
assert_rejected(
    "boundary",
    {"measurement_summary.py": CORRECT_MEASUREMENT.replace(">= review_threshold", "> review_threshold")},
    "measurement alternate values",
)
assert_rejected(
    "divisor",
    {"measurement_summary.py": CORRECT_MEASUREMENT.replace("total / len(measurements)", "total / 4")},
    "measurement alternate values",
)
assert_rejected(
    "missing_loop",
    {
        "measurement_summary.py": CORRECT_MEASUREMENT.replace(
            "for measurement in measurements:",
            "for measurement in []:",
        ).replace("total = 0", "total = sum(measurements)")
    },
    "measurement structure",
)
assert_rejected(
    "missing_else",
    {"measurement_summary.py": CORRECT_MEASUREMENT.replace("    else:\n        status = \"within range\"\n", "")},
    "measurement structure",
)
assert_rejected(
    "forbidden_construct",
    {"measurement_summary.py": "import math\n" + CORRECT_MEASUREMENT},
    "measurement structure",
)
assert_rejected(
    "label_rounding",
    {
        "measurement_summary.py": CORRECT_MEASUREMENT.replace(
            'print(f"Mean: {mean:.1f}")',
            'print(f"Average: {mean:.2f}")',
        )
    },
    "measurement default output",
)
assert_rejected(
    "stale_output",
    {},
    "fresh output artifact",
    stale=True,
)

BROKEN_CWD_WRAPPER = '''from pathlib import Path
Path("output").mkdir(exist_ok=True)
Path("output/readiness.txt").write_text("wrong location\\n", encoding="utf-8")
print("Wrote output/readiness.txt from three fresh script runs.")
'''
with tempfile.TemporaryDirectory(prefix="ds217-a01-cwd-") as temporary:
    cwd_root = Path(temporary)
    write_fixture(cwd_root)
    generate_output(cwd_root)
    (cwd_root / "make_output.py").write_text(BROKEN_CWD_WRAPPER, encoding="utf-8")
    cwd_failures = failures_for(cwd_root)
    if "output wrapper working directory" not in cwd_failures:
        raise AssertionError(f"cwd: expected wrapper check to fail, got {sorted(cwd_failures)}")
    print("[PASS] rejected cwd fixture through output wrapper working directory")

print("All Assignment 01 grader fixture checks passed.")
