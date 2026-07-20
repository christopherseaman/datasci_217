"""Materialize fixtures and validate the Assignment 02 public checker."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT_DIR))

from _public_checks import EXPECTED_REPORT, EXPECTED_STDOUT, run_public_checks  # noqa: E402


CORRECT_README = '''# Assignment 02: Reusable Measurement Summary

## Project description

This measurement project calculates group means and saves a reusable text summary.

## Run

Run the program with `python main.py` from the assignment directory.
'''

CORRECT_GITIGNORE = '''__pycache__/
*.pyc
'''

CORRECT_STATE = '''# Git state snapshots

<!-- ANSWERS START -->
1. working tree; diff
2. staging area; commit
3. local branch; remote; synchronize
4. merge; conflict
<!-- ANSWERS END -->
'''

CORRECT_ANALYSIS = '''def mean(values):
    """Return the arithmetic mean, or None for empty input."""
    if not values:
        return None

    total = 0
    for value in values:
        total = total + value

    return total / len(values)


def format_summary(record):
    """Return a one-line summary for a measurement record."""
    average = mean(record["values"])

    if average is None:
        return f'{record["label"]} mean: no measurements'

    return f'{record["label"]} mean: {average:.1f}'
'''

CORRECT_MAIN = '''from analysis_utils import format_summary


def main():
    """Write, read back, and print the measurement report."""
    records = [
        {"label": "Morning", "values": [18, 21, 24]},
        {"label": "Evening", "values": [20, 22, 26]},
        {"label": "Overnight", "values": []},
    ]

    report_text = ""
    for record in records:
        summary_line = format_summary(record)
        report_text = report_text + summary_line + "\\n"

    with open("report.txt", "w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    with open("report.txt", "r", encoding="utf-8") as report_file:
        saved_report = report_file.read()

    print(saved_report, end="")
    print(f"Saved report matches: {saved_report == report_text}")


if __name__ == "__main__":
    main()
'''


def write_fixture(root: Path, overrides: dict[str, str] | None = None) -> None:
    sources = {
        "README.md": CORRECT_README,
        ".gitignore": CORRECT_GITIGNORE,
        "GIT_STATE_CHECK.md": CORRECT_STATE,
        "analysis_utils.py": CORRECT_ANALYSIS,
        "main.py": CORRECT_MAIN,
    }
    if overrides:
        sources.update(overrides)
    for filename, source in sources.items():
        (root / filename).write_text(source, encoding="utf-8")


def failures_for(root: Path) -> dict[str, str]:
    return {
        name: error
        for name, error in run_public_checks(root)
        if error is not None
    }


def assert_accepted(case: str, overrides: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix=f"ds217-a02-{case}-") as temporary:
        root = Path(temporary)
        write_fixture(root, overrides)
        failures = failures_for(root)
        if failures:
            raise AssertionError(f"{case}: compliant fixture failed: {failures}")
        print(f"[PASS] accepted {case} fixture through all public checks")


def assert_rejected(
    case: str,
    overrides: dict[str, str],
    expected_check: str,
    *,
    prepare=None,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"ds217-a02-{case}-") as temporary:
        root = Path(temporary)
        write_fixture(root, overrides)
        if prepare is not None:
            prepare(root)
        failures = failures_for(root)
        if expected_check not in failures:
            raise AssertionError(
                f"{case}: expected `{expected_check}` to fail, got {sorted(failures)}"
            )
        print(f"[PASS] rejected {case} fixture through {expected_check}")


with tempfile.TemporaryDirectory(prefix="ds217-a02-correct-") as temporary:
    correct_root = Path(temporary)
    write_fixture(correct_root)
    correct_failures = failures_for(correct_root)
    if correct_failures:
        raise AssertionError(f"correct fixture failed: {correct_failures}")
    print("[PASS] accepted correct fixture through all public checks")

    checker_files = ("_public_checks.py", "check_assignment.py")
    for filename in checker_files:
        shutil.copy2(ASSIGNMENT_DIR / filename, correct_root / filename)
    checker = subprocess.run(
        [sys.executable, "check_assignment.py"],
        cwd=correct_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if checker.returncode != 0 or "All public checks passed." not in checker.stdout:
        raise AssertionError(f"dependency-free checker rejected correct fixture: {checker.stdout}{checker.stderr}")
    print("[PASS] dependency-free checker accepted correct fixture")

    if len(EXPECTED_STDOUT.encode("utf-8")) != 97 or len(EXPECTED_REPORT.encode("utf-8")) != 70:
        raise AssertionError("expected stdout/report byte contract changed")
    with tempfile.TemporaryDirectory(prefix="ds217-a02-other-cwd-") as other_temporary:
        other_root = Path(other_temporary)
        different_cwd = subprocess.run(
            [sys.executable, "-B", str(correct_root / "main.py")],
            cwd=other_root,
            capture_output=True,
            timeout=30,
        )
        if (
            different_cwd.returncode != 0
            or different_cwd.stderr != b""
            or different_cwd.stdout != EXPECTED_STDOUT.encode("utf-8")
            or (other_root / "report.txt").read_bytes() != EXPECTED_REPORT.encode("utf-8")
        ):
            raise AssertionError("correct fixture failed exact output from a different working directory")
    print("[PASS] correct fixture produced exact 97-byte stdout and 70-byte report from another cwd")

    if "--pytest" in sys.argv:
        shutil.copy2(ASSIGNMENT_DIR / "test_assignment.py", correct_root / "test_assignment.py")
        pytest_result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_assignment.py", "-q"],
            cwd=correct_root,
            text=True,
            timeout=60,
        )
        if pytest_result.returncode != 0:
            raise AssertionError("pytest rejected the correct fixture")
        print("[PASS] pytest accepted all public tests for the correct fixture")


NAMED_COMPARISON_MAIN = CORRECT_MAIN.replace(
    '    print(f"Saved report matches: {saved_report == report_text}")',
    '    matches = saved_report == report_text\n'
    '    print(f"Saved report matches: {matches}")',
)
assert_accepted(
    "named_readback_comparison",
    {"main.py": NAMED_COMPARISON_MAIN},
)


PARTIAL_ANALYSIS = '''def mean(values):
    """Return the arithmetic mean, or None for empty input."""
    return None
'''
assert_rejected(
    "partial",
    {"analysis_utils.py": PARTIAL_ANALYSIS},
    "analysis_utils structure",
)

HARDCODED_MEAN = CORRECT_ANALYSIS.replace(
    "return total / len(values)",
    "return 21.0",
)
assert_rejected(
    "hardcoded_mean",
    {"analysis_utils.py": HARDCODED_MEAN},
    "mean behavior",
)

DEAD_LOOP_LOOKUP_MEAN = '''def mean(values):
    """Return the arithmetic mean, or None for empty input."""
    if not values:
        return None
    total = 0
    for value in []:
        total = total + value
    known = {
        "[18, 21, 24]": 21.0,
        "[0, 0]": 0.0,
        "[1.5, 2.5]": 2.0,
        "[2, 6]": 4.0,
        "[10, 20]": 15.0,
        "[-3, 8, 2.5]": 2.5,
        "[0.25, 1.75, 5.0]": 7.0 / 3.0,
    }
    return known[str(values)]


def format_summary(record):
    """Return a one-line summary for a measurement record."""
    average = mean(record["values"])
    if average is None:
        return f'{record["label"]} mean: no measurements'
    return f'{record["label"]} mean: {average:.1f}'
'''
assert_rejected(
    "dead_loop_lookup_mean",
    {"analysis_utils.py": DEAD_LOOP_LOOKUP_MEAN},
    "analysis_utils structure",
)

ZERO_FOR_EMPTY = CORRECT_ANALYSIS.replace(
    "if not values:\n        return None",
    "if not values:\n        if values == [\"unreachable\"]:\n            return None\n        return 0.0",
)
assert_rejected(
    "zero_instead_of_none",
    {"analysis_utils.py": ZERO_FOR_EMPTY},
    "mean behavior",
)

FALSEY_ZERO = CORRECT_ANALYSIS.replace(
    "if average is None:",
    "if not average:",
)
assert_rejected(
    "falsey_zero",
    {"analysis_utils.py": FALSEY_ZERO},
    "format_summary behavior and mean use",
)

PRINT_IN_MEAN = CORRECT_ANALYSIS.replace(
    "return total / len(values)",
    "print(total / len(values))\n    return total / len(values)",
)
assert_rejected(
    "print_instead_of_quiet_return",
    {"analysis_utils.py": PRINT_IN_MEAN},
    "analysis_utils structure",
)

GLOBAL_STATE = '''accumulated_total = 0

def mean(values):
    """Return the arithmetic mean, or None for empty input."""
    global accumulated_total
    if not values:
        return None
    total = 0
    for value in values:
        total = total + value
        accumulated_total = accumulated_total + value
    return accumulated_total / len(values)


def format_summary(record):
    """Return a one-line summary for a measurement record."""
    average = mean(record["values"])
    if average is None:
        return f'{record["label"]} mean: no measurements'
    return f'{record["label"]} mean: {average:.1f}'
'''
assert_rejected(
    "global_accumulated_state",
    {"analysis_utils.py": GLOBAL_STATE},
    "analysis_utils structure",
)

MUTATING_MEAN = CORRECT_ANALYSIS.replace(
    "total = 0\n    for value in values:",
    "removed_value = values.pop()\n    total = removed_value\n    for value in values:",
).replace(
    "return total / len(values)",
    "return total / (len(values) + 1)",
)
assert_rejected(
    "input_mutation",
    {"analysis_utils.py": MUTATING_MEAN},
    "mean behavior",
)

DUPLICATED_FORMATTING = CORRECT_ANALYSIS.replace(
    '    average = mean(record["values"])\n\n',
    '    total = 0\n    for value in record["values"]:\n        total = total + value\n    if record["values"]:\n        average = total / len(record["values"])\n    else:\n        average = None\n\n',
)
assert_rejected(
    "duplicated_formatting_without_mean_call",
    {"analysis_utils.py": DUPLICATED_FORMATTING},
    "analysis_utils structure",
)

CALL_THEN_RECOMPUTE_FORMATTING = CORRECT_ANALYSIS.replace(
    '    average = mean(record["values"])\n\n',
    '    ignored_average = mean(record["values"])\n'
    '    total = 0\n'
    '    for value in record["values"]:\n'
    '        total = total + value\n'
    '    if record["values"]:\n'
    '        average = total / len(record["values"])\n'
    '    else:\n'
    '        average = None\n\n',
)
assert_rejected(
    "format_calls_mean_then_recomputes",
    {"analysis_utils.py": CALL_THEN_RECOMPUTE_FORMATTING},
    "analysis_utils structure",
)

CALL_THEN_SUM_FORMATTING = CORRECT_ANALYSIS.replace(
    'return f\'{record["label"]} mean: {average:.1f}\'',
    'return f\'{record["label"]} mean: {sum(record["values"]) / len(record["values"]):.1f}\'',
)
assert_rejected(
    "format_calls_mean_then_sum",
    {"analysis_utils.py": CALL_THEN_SUM_FORMATTING},
    "analysis_utils structure",
)

CALL_THEN_DIRECT_ARITHMETIC = CORRECT_ANALYSIS.replace(
    'return f\'{record["label"]} mean: {average:.1f}\'',
    'return f\'{record["label"]} mean: '
    '{(record["values"][0] + record["values"][-1]) / 2:.1f}\'',
)
assert_rejected(
    "format_calls_mean_then_direct_arithmetic",
    {"analysis_utils.py": CALL_THEN_DIRECT_ARITHMETIC},
    "analysis_utils structure",
)

COMPREHENSION_MEAN = CORRECT_ANALYSIS.replace(
    "total = 0\n    for value in values:\n        total = total + value",
    "total = sum([value for value in values])\n    for value in []:\n        total = total + value",
)
assert_rejected(
    "comprehension",
    {"analysis_utils.py": COMPREHENSION_MEAN},
    "analysis_utils structure",
)

MISSING_DOCSTRING = CORRECT_ANALYSIS.replace(
    '    """Return the arithmetic mean, or None for empty input."""\n',
    "",
    1,
)
assert_rejected(
    "missing_docstring",
    {"analysis_utils.py": MISSING_DOCSTRING},
    "analysis_utils structure",
)

for dangerous_name in ("exec", "eval", "compile", "__import__"):
    dangerous_source = CORRECT_ANALYSIS.replace(
        "    total = 0\n",
        f'    {dangerous_name}("not permitted")\n    total = 0\n',
        1,
    )
    assert_rejected(
        f"dangerous_{dangerous_name.strip('_')}_call",
        {"analysis_utils.py": dangerous_source},
        "analysis_utils structure",
    )

INDIRECT_MEAN_CALL = CORRECT_ANALYSIS.replace(
    '    average = mean(record["values"])',
    '    selected_calculation = mean\n'
    '    average = selected_calculation(record["values"])',
)
assert_rejected(
    "indirect_dynamic_mean_call",
    {"analysis_utils.py": INDIRECT_MEAN_CALL},
    "analysis_utils structure",
)

BUILTINS_SUBSCRIPT_CALL = CORRECT_ANALYSIS.replace(
    '    average = mean(record["values"])',
    '    average = __builtins__["mean"](record["values"])',
)
assert_rejected(
    "builtins_subscript_call",
    {"analysis_utils.py": BUILTINS_SUBSCRIPT_CALL},
    "analysis_utils structure",
)

UNAPPROVED_IMPORT = "import math\n\n" + CORRECT_ANALYSIS
assert_rejected(
    "unapproved_import",
    {"analysis_utils.py": UNAPPROVED_IMPORT},
    "analysis_utils structure",
)

UNAPPROVED_CALL = CORRECT_ANALYSIS.replace(
    "return total / len(values)",
    "return total / abs(len(values))",
)
assert_rejected(
    "unapproved_direct_call",
    {"analysis_utils.py": UNAPPROVED_CALL},
    "analysis_utils structure",
)

TOP_LEVEL_CALL = CORRECT_MAIN.replace(
    'if __name__ == "__main__":\n    main()',
    'main()\n\nif __name__ == "__main__":\n    main()',
)
assert_rejected(
    "top_level_call",
    {"main.py": TOP_LEVEL_CALL},
    "main structure and read-back",
)

MALFORMED_GUARD = CORRECT_MAIN.replace(
    'if __name__ == "__main__":',
    'if __name__ == "main":',
)
assert_rejected(
    "malformed_guard",
    {"main.py": MALFORMED_GUARD},
    "main structure and read-back",
)

APPEND_REPORT = CORRECT_MAIN.replace(
    'open("report.txt", "w", encoding="utf-8")',
    'open("report.txt", "a", encoding="utf-8")',
)
assert_rejected(
    "append_mode",
    {"main.py": APPEND_REPORT},
    "main structure and read-back",
)

DUMMY_WRITE_THEN_APPEND = CORRECT_MAIN.replace(
    '''    with open("report.txt", "w", encoding="utf-8") as report_file:
        report_file.write(report_text)
''',
    '''    with open("report.txt", "w", encoding="utf-8") as report_file:
        report_file.write("")

    with open("report.txt", "a", encoding="utf-8") as report_file:
        report_file.write(report_text)
''',
)
assert_rejected(
    "dummy_write_then_append",
    {"main.py": DUMMY_WRITE_THEN_APPEND},
    "main structure and read-back",
)

INDIRECT_FILE_WRITE = CORRECT_MAIN.replace(
    "        report_file.write(report_text)",
    "        selected_write = report_file.write\n"
    "        selected_write(report_text)",
)
assert_rejected(
    "indirect_file_write",
    {"main.py": INDIRECT_FILE_WRITE},
    "main structure and read-back",
)

INDIRECT_FILE_READ = CORRECT_MAIN.replace(
    "        saved_report = report_file.read()",
    "        selected_read = report_file.read\n"
    "        saved_report = selected_read()",
)
assert_rejected(
    "indirect_file_read",
    {"main.py": INDIRECT_FILE_READ},
    "main structure and read-back",
)

NO_READBACK = CORRECT_MAIN.replace(
    '''    with open("report.txt", "r", encoding="utf-8") as report_file:
        saved_report = report_file.read()

''',
    '    saved_report = report_text\n\n',
)
assert_rejected(
    "no_readback",
    {"main.py": NO_READBACK},
    "main structure and read-back",
)

def prepare_stale_report(root: Path) -> None:
    (root / "report.txt").write_text(EXPECTED_REPORT, encoding="utf-8")


assert_rejected(
    "stale_correct_report_broken_code",
    {"main.py": CORRECT_MAIN.replace("def main():", "def main(:")},
    "main structure and read-back",
    prepare=prepare_stale_report,
)

EXTERNAL_DEPENDENCY = CORRECT_MAIN.replace(
    '    report_text = ""\n',
    '    with open("external_label.txt", "r", encoding="utf-8") as external_file:\n'
    '        external_label = external_file.read()\n'
    '    report_text = external_label\n',
)
assert_rejected(
    "external_path_dependency",
    {"main.py": EXTERNAL_DEPENDENCY},
    "fresh main output and overwritten report",
)

HARDCODED_MAIN = CORRECT_MAIN.replace(
    '''    report_text = ""
    for record in records:
        summary_line = format_summary(record)
        report_text = report_text + summary_line + "\\n"
''',
    '''    for record in records:
        unused_summary = format_summary(record)
    report_text = "Morning mean: 21.0\\nEvening mean: 22.7\\nOvernight mean: no measurements\\n"
''',
)
assert_rejected(
    "hardcoded_main_ignores_formatter_results",
    {"main.py": HARDCODED_MAIN},
    "main formatter call order",
)

SPY_SPECIAL_CASE_MAIN = CORRECT_MAIN.replace(
    '''    report_text = ""
    for record in records:
        summary_line = format_summary(record)
        report_text = report_text + summary_line + "\\n"
''',
    '''    observed_report = ""
    for record in records:
        summary_line = format_summary(record)
        observed_report = observed_report + summary_line + "\\n"

    published_spy = "SPY Morning\\nSPY Evening\\nSPY Overnight\\n"
    if observed_report == published_spy:
        report_text = observed_report
    else:
        report_text = "Morning mean: 21.0\\nEvening mean: 22.7\\nOvernight mean: no measurements\\n"
''',
)
assert_rejected(
    "published_spy_special_case",
    {"main.py": SPY_SPECIAL_CASE_MAIN},
    "main formatter call order",
)

WRONG_CALL_ORDER = CORRECT_MAIN.replace(
    "for record in records:",
    "for record in [records[1], records[0], records[2]]:",
)
assert_rejected(
    "wrong_formatter_call_order",
    {"main.py": WRONG_CALL_ORDER},
    "fresh main output and overwritten report",
)

print("All Assignment 02 grader fixture checks passed.")
