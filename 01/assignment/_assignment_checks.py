"""Shared public checks for Assignment 01.

This is supplied grader machinery. Lecture 01 students run it but do not need to
understand its imports, functions, classes, subprocesses, or file operations.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


EXPECTED_PROJECT_LABEL = "DataSci 217 Assignment 01"
EXPECTED_DEFAULT_MEASUREMENTS = [18, 21, 24, 19]
EXPECTED_DEFAULT_THRESHOLD_TEXT = "20"
@dataclass(frozen=True)
class PublicCheck:
    name: str
    action: Callable[[Path], None]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read(root: Path, filename: str) -> str:
    path = root / filename
    _assert(path.is_file(), f"Missing {filename}.")
    return path.read_text(encoding="utf-8")


def _parse(root: Path, filename: str) -> ast.Module:
    source = _read(root, filename)
    try:
        return ast.parse(source, filename=filename)
    except SyntaxError as error:
        location = f"line {error.lineno}" if error.lineno else "an unknown line"
        raise AssertionError(
            f"{filename} still has a SyntaxError at {location}: {error.msg}."
        ) from error


def _top_level_value(tree: ast.Module, name: str) -> ast.expr | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return node.value
    return None


def _used_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _updates_name_from(node: ast.AST, target_name: str, required_names: set[str]) -> bool:
    if isinstance(node, ast.Assign):
        assigns_target = any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        )
        return assigns_target and required_names <= _used_names(node.value)
    if isinstance(node, ast.AugAssign):
        return (
            isinstance(node.target, ast.Name)
            and node.target.id == target_name
            and isinstance(node.op, ast.Add)
            and (required_names - {target_name}) <= _used_names(node.value)
        )
    return False


def _run_script(path: Path, cwd: Path) -> str:
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(f"{path.name} did not finish within 10 seconds.") from error

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no error text"
        raise AssertionError(
            f"{path.name} exited with an error. Run it directly and read the final "
            f"traceback line. Last output: {detail.splitlines()[-1]}"
        )
    return completed.stdout


def _expected_readiness(
    filename: str = "readiness.py",
    family: str | None = None,
    project_label: str = EXPECTED_PROJECT_LABEL,
) -> str:
    if family is None:
        family = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"Python family: {family}\n"
        f"Project: {project_label}\n"
        f"Script: {filename}\n"
    )


def _expected_measurement_output(measurements: list[int], threshold: int) -> str:
    lines = [f"First measurement: {measurements[0]}"]
    total = 0
    review_count = 0
    for measurement in measurements:
        total += measurement
        if measurement >= threshold:
            status = "review"
            review_count += 1
        else:
            status = "within range"
        lines.append(f"Measurement {measurement}: {status}")

    mean = total / len(measurements)
    lines.extend(
        (
            f"Count: {len(measurements)}",
            f"Total: {total}",
            f"Mean: {mean:.1f}",
            f"Review count: {review_count}",
        )
    )
    return "\n".join(lines) + "\n"


def _expected_debug_output(participant_count: int) -> str:
    return (
        "Readiness: complete\n"
        f"Participant count: {participant_count}\n"
        f"Next checkpoint: {participant_count + 1}\n"
    )


def _student_restriction_errors(
    tree: ast.Module,
    *,
    allow_supplied_imports: bool,
) -> list[str]:
    errors = []
    forbidden_nodes = {
        ast.FunctionDef: "student-defined function",
        ast.AsyncFunctionDef: "student-defined async function",
        ast.ClassDef: "class",
        ast.Dict: "dictionary",
        ast.DictComp: "dictionary comprehension",
        ast.Set: "set",
        ast.ListComp: "list comprehension",
        ast.SetComp: "set comprehension",
        ast.GeneratorExp: "generator expression",
        ast.Lambda: "lambda",
        ast.Try: "try/except",
        ast.TryStar: "try/except-star",
        ast.With: "with block or file I/O",
        ast.AsyncWith: "async with block",
        ast.While: "while loop",
        ast.Break: "break",
        ast.Continue: "continue",
        ast.Raise: "raise",
        ast.Match: "match statement",
        ast.NamedExpr: "assignment expression",
    }
    allowed_call_names = {"print", "type", "int", "float", "len", "range"}
    if allow_supplied_imports:
        allowed_call_names.add("Path")
    file_method_names = {
        "open",
        "read",
        "read_text",
        "read_bytes",
        "write",
        "write_text",
        "write_bytes",
    }

    for node in ast.walk(tree):
        for node_type, description in forbidden_nodes.items():
            if isinstance(node, node_type):
                errors.append(description)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not allow_supplied_imports:
            errors.append("student-added import")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_call_names:
                    errors.append(f"call to out-of-scope {node.func.id}()")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in file_method_names:
                    errors.append(f"possible file I/O through .{node.func.attr}()")
                else:
                    errors.append(f"out-of-scope method call .{node.func.attr}()")
            else:
                errors.append("indirect or dynamically selected function call")

    return sorted(set(errors))


def check_terminal_practice(root: Path) -> None:
    practice = root / "terminal-practice"
    _assert(practice.is_dir(), "Create terminal-practice from the assignment directory.")
    _assert(
        (practice / "source.txt").is_file(),
        "terminal-practice/source.txt is missing; repeat the named touch command.",
    )
    _assert(
        (practice / "path-check.txt").is_file(),
        "terminal-practice/path-check.txt is missing; repeat the copy and move commands.",
    )
    _assert(
        not (practice / "remove-me.txt").exists(),
        "terminal-practice/remove-me.txt still exists; verify pwd and remove that named file.",
    )


def check_readiness_structure(root: Path) -> None:
    tree = _parse(root, "readiness.py")
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    _assert(len(imports) == 2, "Do not add, remove, or change imports in readiness.py.")
    first_import, second_import = imports
    _assert(
        isinstance(first_import, ast.Import)
        and len(first_import.names) == 1
        and first_import.names[0].name == "sys"
        and first_import.names[0].asname is None,
        "Keep the supplied `import sys` line unchanged.",
    )
    _assert(
        isinstance(second_import, ast.ImportFrom)
        and second_import.module == "pathlib"
        and len(second_import.names) == 1
        and second_import.names[0].name == "Path"
        and second_import.names[0].asname is None,
        "Keep the supplied `from pathlib import Path` line unchanged.",
    )
    errors = _student_restriction_errors(tree, allow_supplied_imports=True)
    _assert(not errors, f"readiness.py uses out-of-scope code: {', '.join(errors)}.")
    project_value = _top_level_value(tree, "PROJECT_LABEL")
    _assert(
        isinstance(project_value, ast.Constant)
        and project_value.value == EXPECTED_PROJECT_LABEL,
        "Keep the supplied PROJECT_LABEL assignment unchanged.",
    )
    family_value = _top_level_value(tree, "python_family")
    _assert(
        family_value is not None and "sys" in _used_names(family_value),
        "Keep the supplied dynamic python_family assignment unchanged.",
    )
    filename_value = _top_level_value(tree, "script_filename")
    _assert(
        filename_value is not None
        and "Path" in _used_names(filename_value)
        and "__file__" in _used_names(filename_value),
        "Keep the supplied dynamic script_filename assignment unchanged.",
    )
    print_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "print"
    ]
    _assert(
        len(print_calls) == 3,
        "Keep exactly the three requested readiness print lines.",
    )
    for print_node, required_name in zip(
        print_calls,
        ("python_family", "PROJECT_LABEL", "script_filename"),
    ):
        _assert(
            required_name in _used_names(print_node),
            f"Use the supplied {required_name} variable in its print line.",
        )


def check_readiness_behavior(root: Path) -> None:
    script = root / "readiness.py"
    actual = _run_script(script, root)
    _assert(
        actual == _expected_readiness(),
        "readiness.py output does not match the three exact labels and supplied values.",
    )

    with tempfile.TemporaryDirectory(prefix="ds217-a01-readiness-") as temporary:
        temporary_path = Path(temporary)
        renamed = temporary_path / "renamed_readiness.py"
        shutil.copy2(script, renamed)
        renamed_output = _run_script(renamed, temporary_path)
        _assert(
            renamed_output == _expected_readiness("renamed_readiness.py"),
            "Use script_filename in readiness.py instead of hard-coding readiness.py.",
        )
        alternate_tree = ast.parse(_read(root, "readiness.py"), filename="readiness.py")
        for node in alternate_tree.body:
            if not isinstance(node, ast.Assign):
                continue
            target_names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if "PROJECT_LABEL" in target_names:
                node.value = ast.Constant(value="Alternate readiness project")
            if "python_family" in target_names:
                node.value = ast.Constant(value="9.9")
        ast.fix_missing_locations(alternate_tree)
        renamed.write_text(ast.unparse(alternate_tree) + "\n", encoding="utf-8")
        alternate_output = _run_script(renamed, temporary_path)
        _assert(
            alternate_output
            == _expected_readiness(
                "renamed_readiness.py",
                family="9.9",
                project_label="Alternate readiness project",
            ),
            "Print the supplied python_family, PROJECT_LABEL, and script_filename "
            "variables instead of hard-coded values.",
        )


def check_measurement_structure(root: Path) -> None:
    tree = _parse(root, "measurement_summary.py")
    errors = _student_restriction_errors(tree, allow_supplied_imports=False)
    _assert(not errors, f"measurement_summary.py uses out-of-scope code: {', '.join(errors)}.")
    total_value = _top_level_value(tree, "total")
    review_count_value = _top_level_value(tree, "review_count")
    _assert(
        isinstance(total_value, ast.Constant)
        and type(total_value.value) is int
        and total_value.value == 0,
        "Start total at zero before the loop.",
    )
    _assert(
        isinstance(review_count_value, ast.Constant)
        and type(review_count_value.value) is int
        and review_count_value.value == 0,
        "Start review_count at zero before the loop.",
    )

    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    _assert(
        len(loops) == 1,
        "Use one direct for loop to visit every supplied measurement.",
    )
    measurement_loop = loops[0]
    _assert(
        isinstance(measurement_loop.target, ast.Name)
        and isinstance(measurement_loop.iter, ast.Name)
        and measurement_loop.iter.id == "measurements",
        "Loop directly over the full list: `for ... in measurements:`.",
    )
    loop_name = measurement_loop.target.id
    loop_nodes = list(ast.walk(measurement_loop))
    _assert(
        any(
            _updates_name_from(node, "total", {"total", loop_name})
            for node in loop_nodes
        ),
        "Add the current loop value to total inside the loop.",
    )
    loop_decisions = [
        node
        for node in loop_nodes
        if isinstance(node, ast.If)
        and node.orelse
        and {loop_name, "review_threshold"} <= _used_names(node.test)
    ]
    _assert(
        loop_decisions,
        "Put the threshold if/else decision inside the measurement loop.",
    )
    _assert(
        any(
            _updates_name_from(node, "review_count", {"review_count"})
            for decision in loop_decisions
            for node in ast.walk(decision)
        ),
        "Increment review_count in the loop's threshold decision.",
    )
    _assert(
        any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            and loop_name in _used_names(node)
            for node in loop_nodes
        ),
        "Print the current measurement's labeled line from inside the loop.",
    )
    threshold_text = _top_level_value(tree, "review_threshold_text")
    _assert(
        isinstance(threshold_text, ast.Constant)
        and threshold_text.value == EXPECTED_DEFAULT_THRESHOLD_TEXT,
        "Keep the supplied top-level review_threshold_text value as the string \"20\".",
    )
    threshold = _top_level_value(tree, "review_threshold")
    _assert(
        isinstance(threshold, ast.Call)
        and isinstance(threshold.func, ast.Name)
        and threshold.func.id == "int"
        and "review_threshold_text" in _used_names(threshold),
        "Convert review_threshold_text with int() and assign it to review_threshold.",
    )
    _assert(
        any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "measurements"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == 0
            for node in ast.walk(tree)
        ),
        "Select the first value with measurements[0].",
    )


def check_measurement_default(root: Path) -> None:
    actual = _run_script(root / "measurement_summary.py", root)
    expected = _expected_measurement_output(
        EXPECTED_DEFAULT_MEASUREMENTS,
        int(EXPECTED_DEFAULT_THRESHOLD_TEXT),
    )
    _assert(
        actual == expected,
        "measurement_summary.py does not match the exact default labels, values, order, and one-decimal mean.",
    )


def _alternate_measurement_source(
    source: str,
    measurements: list[int],
    threshold: int,
) -> str:
    tree = ast.parse(source, filename="measurement_summary.py")
    found_measurements = False
    found_threshold_text = False

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "measurements":
                node.value = ast.List(
                    elts=[ast.Constant(value=value) for value in measurements],
                    ctx=ast.Load(),
                )
                found_measurements = True
            if isinstance(target, ast.Name) and target.id == "review_threshold_text":
                node.value = ast.Constant(value=str(threshold))
                found_threshold_text = True

    _assert(found_measurements, "Keep a top-level assignment named measurements.")
    _assert(
        found_threshold_text,
        "Keep a top-level assignment named review_threshold_text.",
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def check_measurement_alternates(root: Path) -> None:
    source = _read(root, "measurement_summary.py")
    cases = (
        ([5, 10, 20], 10),
        ([20, 19], 20),
        ([1, 2, 2], 99),
    )

    with tempfile.TemporaryDirectory(prefix="ds217-a01-alternate-") as temporary:
        temporary_path = Path(temporary)
        script = temporary_path / "measurement_summary.py"
        for measurements, threshold in cases:
            script.write_text(
                _alternate_measurement_source(source, measurements, threshold),
                encoding="utf-8",
            )
            actual = _run_script(script, temporary_path)
            expected = _expected_measurement_output(measurements, threshold)
            _assert(
                actual == expected,
                "The measurement summary failed alternate supplied values. Check the >= boundary, "
                "actual list-length divisor, else branch, labels, and one-decimal mean.",
            )


def check_debug_report(root: Path) -> None:
    tree = _parse(root, "debug_report.py")
    errors = _student_restriction_errors(tree, allow_supplied_imports=False)
    _assert(not errors, f"debug_report.py uses out-of-scope code: {', '.join(errors)}.")
    participant_decisions = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and "participant_count" in _used_names(node.test)
    ]
    _assert(
        len(participant_decisions) == 1,
        "Keep the supplied top-level participant_count decision.",
    )
    participant_text = _top_level_value(tree, "participant_count_text")
    _assert(
        isinstance(participant_text, ast.Constant) and participant_text.value == "4",
        "Keep the supplied participant_count_text value.",
    )
    participant_count = _top_level_value(tree, "participant_count")
    _assert(
        isinstance(participant_count, ast.Call)
        and isinstance(participant_count.func, ast.Name)
        and participant_count.func.id == "int"
        and "participant_count_text" in _used_names(participant_count),
        "Keep the supplied conversion from participant_count_text to participant_count.",
    )
    next_checkpoint = _top_level_value(tree, "next_checkpoint")
    _assert(
        isinstance(next_checkpoint, ast.BinOp)
        and isinstance(next_checkpoint.op, ast.Add)
        and "participant_count" in _used_names(next_checkpoint),
        "Fix next_checkpoint by adding 1 to the converted participant_count.",
    )
    print_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    _assert(
        len(print_calls) == 3,
        "Keep exactly the three requested debug-report print lines.",
    )
    _assert(
        any("participant_count" in _used_names(node) for node in print_calls),
        "Print participant_count instead of a prepared participant answer.",
    )
    _assert(
        any("next_checkpoint" in _used_names(node) for node in print_calls),
        "Print next_checkpoint instead of a prepared checkpoint answer.",
    )
    _assert(
        any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            for node in ast.walk(participant_decisions[0])
        ),
        "Keep the readiness print inside the participant_count decision.",
    )
    actual = _run_script(root / "debug_report.py", root)
    _assert(
        actual == _expected_debug_output(4),
        "debug_report.py must produce the exact three clean lines after the three small corrections.",
    )

    alternate_tree = ast.parse(_read(root, "debug_report.py"), filename="debug_report.py")
    found_participant_text = False
    for node in alternate_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "participant_count_text"
            for target in node.targets
        ):
            node.value = ast.Constant(value="7")
            found_participant_text = True
    _assert(found_participant_text, "Keep the supplied participant_count_text assignment.")
    ast.fix_missing_locations(alternate_tree)
    with tempfile.TemporaryDirectory(prefix="ds217-a01-debug-") as temporary:
        temporary_path = Path(temporary)
        alternate_script = temporary_path / "debug_report.py"
        alternate_script.write_text(ast.unparse(alternate_tree) + "\n", encoding="utf-8")
        alternate_output = _run_script(alternate_script, temporary_path)
    _assert(
        alternate_output == _expected_debug_output(7),
        "Use participant_count and next_checkpoint in the printed output instead of prepared answers.",
    )


def _fresh_combined_output(root: Path) -> str:
    outputs = []
    for filename in ("readiness.py", "measurement_summary.py", "debug_report.py"):
        outputs.append(_run_script(root / filename, root).rstrip())
    return "\n".join(outputs) + "\n"


def check_output_artifact(root: Path) -> None:
    output_path = root / "output" / "readiness.txt"
    _assert(
        output_path.is_file(),
        "output/readiness.txt is missing; run python make_output.py after fixing all scripts.",
    )
    stored = output_path.read_text(encoding="utf-8")
    fresh = _fresh_combined_output(root)
    _assert(
        stored == fresh,
        "output/readiness.txt is stale or was edited by hand; rerun python make_output.py.",
    )


def check_wrapper_from_other_cwd(root: Path) -> None:
    wrapper = root / "make_output.py"
    _assert(wrapper.is_file(), "Missing supplied make_output.py.")
    with tempfile.TemporaryDirectory(prefix="ds217-a01-wrapper-") as temporary:
        outside = Path(temporary)
        copied_root = outside / "copied-assignment"
        copied_root.mkdir()
        for filename in (
            "readiness.py",
            "measurement_summary.py",
            "debug_report.py",
            "make_output.py",
        ):
            source = root / filename
            _assert(source.is_file(), f"Missing supplied {filename}.")
            shutil.copy2(source, copied_root / filename)
        copied_output = copied_root / "output"
        copied_output.mkdir()
        (copied_output / "readiness.txt").write_text(
            "grader stale-output sentinel\n",
            encoding="utf-8",
        )

        actual = _run_script(copied_root / "make_output.py", outside)
        _assert(
            actual == "Wrote output/readiness.txt from three fresh script runs.\n",
            "Do not edit make_output.py; its success message or working-directory behavior changed.",
        )
        check_output_artifact(copied_root)


PUBLIC_CHECKS = (
    PublicCheck("terminal practice evidence", check_terminal_practice),
    PublicCheck("readiness structure", check_readiness_structure),
    PublicCheck("readiness output and dynamic filename", check_readiness_behavior),
    PublicCheck("measurement structure", check_measurement_structure),
    PublicCheck("measurement default output", check_measurement_default),
    PublicCheck("measurement alternate values", check_measurement_alternates),
    PublicCheck("debug corrections", check_debug_report),
    PublicCheck("fresh output artifact", check_output_artifact),
    PublicCheck("output wrapper working directory", check_wrapper_from_other_cwd),
)


def run_public_checks(root: Path) -> list[tuple[str, str | None]]:
    results = []
    for check in PUBLIC_CHECKS:
        try:
            check.action(root)
        except (AssertionError, OSError) as error:
            results.append((check.name, str(error)))
        else:
            results.append((check.name, None))
    return results
