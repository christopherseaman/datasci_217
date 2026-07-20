"""Shared dependency-free public checks for Assignment 02.

This is supplied grader machinery. Students run it but are not expected to use
its imports, classes, subprocesses, temporary directories, or AST inspection in
their own Lecture 02 work.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


EXPECTED_RECORDS = [
    {"label": "Morning", "values": [18, 21, 24]},
    {"label": "Evening", "values": [20, 22, 26]},
    {"label": "Overnight", "values": []},
]
EXPECTED_REPORT = (
    "Morning mean: 21.0\n"
    "Evening mean: 22.7\n"
    "Overnight mean: no measurements\n"
)
EXPECTED_STDOUT = EXPECTED_REPORT + "Saved report matches: True\n"
EXPECTED_STATE_ANSWERS = (
    "1. working tree; diff",
    "2. staging area; commit",
    "3. local branch; remote; synchronize",
    "4. merge; conflict",
)


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
            f"{filename} has a SyntaxError at {location}: {error.msg}."
        ) from error


def _section(markdown: str, heading: str) -> str:
    pattern = rf"(?ms)^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, markdown)
    _assert(match is not None, f"Keep the `## {heading}` section in README.md.")
    return match.group(1).strip()


def _normalized_answer(line: str) -> str:
    return " ".join(line.replace("`", "").strip().lower().split())


def _function_map(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _validate_signature(function: ast.FunctionDef, expected_name: str, arguments: list[str]) -> None:
    _assert(function.name == expected_name, f"Keep the function name {expected_name}.")
    actual_arguments = [argument.arg for argument in function.args.args]
    _assert(
        not function.args.posonlyargs and actual_arguments == arguments,
        f"Use the exact signature {expected_name}({', '.join(arguments)}).",
    )
    _assert(
        function.args.vararg is None
        and function.args.kwarg is None
        and not function.args.kwonlyargs
        and not function.args.defaults
        and not function.args.kw_defaults,
        f"Do not add defaults, keyword-only parameters, *args, or **kwargs to {expected_name}().",
    )
    _assert(
        function.returns is None
        and all(argument.annotation is None for argument in function.args.args),
        f"Do not add type annotations to {expected_name}().",
    )
    _assert(not function.decorator_list, f"Do not add decorators to {expected_name}().")
    docstring = ast.get_docstring(function, clean=False)
    _assert(docstring is not None and docstring.strip(), f"Add a one-line docstring to {expected_name}().")
    _assert("\n" not in docstring, f"Keep the {expected_name}() docstring on one line.")
    _assert("todo" not in docstring.lower(), f"Replace the TODO docstring in {expected_name}().")


def _forbidden_constructs(tree: ast.Module) -> list[str]:
    forbidden_types = {
        ast.ListComp: "list comprehension",
        ast.SetComp: "set comprehension",
        ast.DictComp: "dictionary comprehension",
        ast.GeneratorExp: "generator expression",
        ast.Lambda: "lambda",
        ast.ClassDef: "class",
        ast.AsyncFunctionDef: "async function",
        ast.Try: "try/except",
        ast.TryStar: "try/except-star",
        ast.Global: "global statement",
        ast.Nonlocal: "nonlocal statement",
        ast.AnnAssign: "type annotation",
    }
    found = []
    for node in ast.walk(tree):
        for node_type, description in forbidden_types.items():
            if isinstance(node, node_type):
                found.append(description)
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            found.append("decorator")
        if isinstance(node, ast.arg) and node.annotation is not None:
            found.append("type annotation")
    return sorted(set(found))


def _used_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _assert_direct_calls(
    node: ast.AST,
    allowed_names: set[str],
    context: str,
) -> None:
    invalid = []
    for call in (child for child in ast.walk(node) if isinstance(child, ast.Call)):
        if isinstance(call.func, ast.Name) and call.func.id in allowed_names:
            continue
        if isinstance(call.func, ast.Name):
            invalid.append(f"{call.func.id}()")
        elif isinstance(call.func, ast.Attribute):
            invalid.append(f".{call.func.attr}()")
        else:
            invalid.append("an indirect or dynamically selected call")
    _assert(
        not invalid,
        f"{context} may call only {', '.join(f'{name}()' for name in sorted(allowed_names))}; "
        f"found {', '.join(sorted(set(invalid)))}.",
    )


def _is_record_key(node: ast.AST, key: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "record"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == key
    )


def _run_python(
    arguments: list[str],
    cwd: Path,
    *,
    timeout: int = 5,
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        return subprocess.run(
            [sys.executable, "-B", *arguments],
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            f"Python did not finish within {timeout} seconds; check for an endless loop."
        ) from error


def _copy_files(root: Path, destination: Path, filenames: tuple[str, ...]) -> None:
    for filename in filenames:
        source = root / filename
        _assert(source.is_file(), f"Missing {filename}.")
        shutil.copy2(source, destination / filename)


def _run_harness(root: Path, harness_source: str, error_message: str) -> None:
    with tempfile.TemporaryDirectory(prefix="ds217-a02-harness-") as temporary:
        project = Path(temporary)
        _copy_files(root, project, ("analysis_utils.py",))
        (project / "_contract_harness.py").write_text(
            harness_source,
            encoding="utf-8",
        )
        completed = _run_python(["_contract_harness.py"], project)
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            last_line = detail.splitlines()[-1] if detail else "no error text"
            raise AssertionError(f"{error_message} Last result: {last_line}")


def check_project_documents(root: Path) -> None:
    readme = _read(root, "README.md")
    description = _section(readme, "Project description")
    run_section = _section(readme, "Run")
    _assert("todo" not in description.lower(), "Replace the Project description TODO.")
    description_length = len(re.sub(r"\s+", "", description))
    _assert(
        30 <= description_length <= 300,
        "Keep the project description between 30 and 300 non-whitespace characters.",
    )
    _assert(
        "measurement" in description.lower(),
        "Mention `measurement` in the project description.",
    )
    _assert("todo" not in run_section.lower(), "Replace the Run TODO.")
    _assert("python main.py" in run_section, "Include the exact command `python main.py` in Run.")

    ignore_lines = _read(root, ".gitignore").splitlines()
    _assert(
        ignore_lines == ["__pycache__/", "*.pyc"],
        "Replace .gitignore with exactly `__pycache__/` and `*.pyc`, one per line.",
    )


def check_git_state_answers(root: Path) -> None:
    source = _read(root, "GIT_STATE_CHECK.md")
    match = re.search(
        r"(?s)<!-- ANSWERS START -->\s*(.*?)\s*<!-- ANSWERS END -->",
        source,
    )
    _assert(match is not None, "Keep the answer markers in GIT_STATE_CHECK.md.")
    answers = [
        _normalized_answer(line)
        for line in match.group(1).splitlines()
        if line.strip()
    ]
    _assert(
        answers == list(EXPECTED_STATE_ANSWERS),
        "Correct the four Git snapshots using the requested terms and semicolon format.",
    )


def check_analysis_structure(root: Path) -> None:
    tree = _parse(root, "analysis_utils.py")
    forbidden = _forbidden_constructs(tree)
    _assert(not forbidden, f"analysis_utils.py uses out-of-scope code: {', '.join(forbidden)}.")
    _assert(
        not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
        "Do not add imports to analysis_utils.py.",
    )
    top_level_functions = _function_map(tree)
    _assert(
        list(top_level_functions) == ["mean", "format_summary"],
        "Keep exactly the two top-level functions mean() and format_summary(), in that order.",
    )
    _assert(
        all(
            isinstance(node, ast.FunctionDef)
            or (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
            for node in tree.body
        ),
        "Keep calculations inside the two functions; do not add top-level driver state.",
    )
    all_functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    _assert(len(all_functions) == 2, "Do not add nested or extra functions to analysis_utils.py.")

    mean_function = top_level_functions["mean"]
    format_function = top_level_functions["format_summary"]
    _validate_signature(mean_function, "mean", ["values"])
    _validate_signature(format_function, "format_summary", ["record"])

    _assert_direct_calls(mean_function, {"len"}, "mean()")
    _assert_direct_calls(format_function, {"mean"}, "format_summary()")

    _assert(
        len(mean_function.body) == 5
        and isinstance(mean_function.body[0], ast.Expr)
        and isinstance(mean_function.body[1], ast.If)
        and isinstance(mean_function.body[2], ast.Assign)
        and isinstance(mean_function.body[3], ast.For)
        and isinstance(mean_function.body[4], ast.Return),
        "Use the documented mean() sequence: docstring, empty-input return, total = 0, "
        "one loop over values, then return total / len(values).",
    )

    empty_branch = mean_function.body[1]
    _assert(
        isinstance(empty_branch.test, ast.UnaryOp)
        and isinstance(empty_branch.test.op, ast.Not)
        and isinstance(empty_branch.test.operand, ast.Name)
        and empty_branch.test.operand.id == "values"
        and len(empty_branch.body) == 1
        and isinstance(empty_branch.body[0], ast.Return)
        and isinstance(empty_branch.body[0].value, ast.Constant)
        and empty_branch.body[0].value.value is None
        and not empty_branch.orelse,
        "Begin mean() with `if not values:` and `return None`.",
    )

    total_initialization = mean_function.body[2]
    _assert(
        len(total_initialization.targets) == 1
        and isinstance(total_initialization.targets[0], ast.Name)
        and total_initialization.targets[0].id == "total"
        and isinstance(total_initialization.value, ast.Constant)
        and not isinstance(total_initialization.value.value, bool)
        and total_initialization.value.value == 0,
        "Initialize the local accumulator with `total = 0` before the loop.",
    )

    values_loop = mean_function.body[3]
    _assert(
        isinstance(values_loop.target, ast.Name)
        and values_loop.target.id != "total"
        and isinstance(values_loop.iter, ast.Name)
        and values_loop.iter.id == "values"
        and not values_loop.orelse
        and len(values_loop.body) == 1,
        "Use one direct `for` loop over values with one accumulator update in its body.",
    )

    loop_name = values_loop.target.id
    update = values_loop.body[0]
    assignment_update = (
        isinstance(update, ast.Assign)
        and len(update.targets) == 1
        and isinstance(update.targets[0], ast.Name)
        and update.targets[0].id == "total"
        and isinstance(update.value, ast.BinOp)
        and isinstance(update.value.op, ast.Add)
        and _used_names(update.value) == {"total", loop_name}
    )
    augmented_update = (
        isinstance(update, ast.AugAssign)
        and isinstance(update.target, ast.Name)
        and update.target.id == "total"
        and isinstance(update.op, ast.Add)
        and isinstance(update.value, ast.Name)
        and update.value.id == loop_name
    )
    _assert(
        assignment_update or augmented_update,
        "Update total directly with the current loop value on every iteration.",
    )

    result_return = mean_function.body[4]
    _assert(
        isinstance(result_return.value, ast.BinOp)
        and isinstance(result_return.value.op, ast.Div)
        and isinstance(result_return.value.left, ast.Name)
        and result_return.value.left.id == "total"
        and isinstance(result_return.value.right, ast.Call)
        and isinstance(result_return.value.right.func, ast.Name)
        and result_return.value.right.func.id == "len"
        and len(result_return.value.right.args) == 1
        and isinstance(result_return.value.right.args[0], ast.Name)
        and result_return.value.right.args[0].id == "values"
        and not result_return.value.right.keywords,
        "Return the live accumulator with `return total / len(values)`.",
    )

    _assert(
        len(format_function.body) == 4
        and isinstance(format_function.body[0], ast.Expr)
        and isinstance(format_function.body[1], ast.Assign)
        and isinstance(format_function.body[2], ast.If)
        and isinstance(format_function.body[3], ast.Return),
        "Use the documented format_summary() sequence: docstring, one assigned mean() "
        "result, the None branch, then the numeric return.",
    )
    mean_assignment = format_function.body[1]
    _assert(
        len(mean_assignment.targets) == 1
        and isinstance(mean_assignment.targets[0], ast.Name)
        and isinstance(mean_assignment.value, ast.Call)
        and isinstance(mean_assignment.value.func, ast.Name)
        and mean_assignment.value.func.id == "mean"
        and len(mean_assignment.value.args) == 1
        and _is_record_key(mean_assignment.value.args[0], "values")
        and not mean_assignment.value.keywords,
        'Assign exactly one `mean(record["values"])` result to a local name.',
    )
    mean_result_name = mean_assignment.targets[0].id
    none_branch = format_function.body[2]
    _assert(
        isinstance(none_branch.test, ast.Compare)
        and isinstance(none_branch.test.left, ast.Name)
        and none_branch.test.left.id == mean_result_name
        and len(none_branch.test.ops) == 1
        and isinstance(none_branch.test.ops[0], ast.Is)
        and len(none_branch.test.comparators) == 1
        and isinstance(none_branch.test.comparators[0], ast.Constant)
        and none_branch.test.comparators[0].value is None
        and len(none_branch.body) == 1
        and isinstance(none_branch.body[0], ast.Return)
        and not none_branch.orelse,
        "Test the assigned mean result with `is None` before returning the empty summary.",
    )
    _assert(
        mean_result_name in _used_names(format_function.body[3].value),
        "Build the numeric summary from the local result returned by mean().",
    )
    _assert(
        not any(isinstance(node, ast.For) for node in ast.walk(format_function))
        and not any(
            isinstance(node, ast.BinOp)
            and any(_is_record_key(child, "values") for child in ast.walk(node))
            for node in ast.walk(format_function)
        ),
        "Do not loop, sum, or recompute arithmetic inside format_summary(); use the mean() result.",
    )
    string_keys = {
        node.slice.value
        for node in ast.walk(format_function)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    _assert(
        string_keys <= {"label", "values"},
        "Use only the `label` and `values` dictionary keys in format_summary().",
    )

    _assert(
        not any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id in {"len", "mean"}
            for node in ast.walk(tree)
        ),
        "Do not replace or dynamically select the permitted len() and mean() calls.",
    )


def check_mean_behavior(root: Path) -> None:
    harness = r'''
from contextlib import redirect_stdout
from io import StringIO
from analysis_utils import mean

cases = (
    ([18, 21, 24], 21.0),
    ([], None),
    ([0, 0], 0.0),
    ([1.5, 2.5], 2.0),
    ([2, 6], 4.0),
    ([10, 20], 15.0),
    ([-3, 8, 2.5], 2.5),
    ([0.25, 1.75, 5.0], 7.0 / 3.0),
    ([2, 6], 4.0),
)

for values, expected in cases:
    before = list(values)
    captured = StringIO()
    with redirect_stdout(captured):
        actual = mean(values)
    assert captured.getvalue() == "", "mean() printed instead of only returning"
    assert actual == expected, (values, actual, expected)
    assert values == before, "mean() mutated its input"
'''
    _run_harness(
        root,
        harness,
        "mean() failed ordinary, empty, zero, decimal, repeated-call, quiet-return, or nonmutation behavior.",
    )


def check_format_behavior(root: Path) -> None:
    token = f"{root.name}-{os.getpid()}"
    harness = r'''
from contextlib import redirect_stdout
from io import StringIO
import analysis_utils

TOKEN = __DYNAMIC_TOKEN__

cases = (
    ({"label": "West", "values": [2, 4]}, "West mean: 3.0"),
    ({"label": "Empty", "values": []}, "Empty mean: no measurements"),
    ({"label": "Zero", "values": [0, 0]}, "Zero mean: 0.0"),
    ({"label": "Decimal", "values": [2, 3]}, "Decimal mean: 2.5"),
)

for record, expected in cases:
    before = {"label": record["label"], "values": list(record["values"])}
    captured = StringIO()
    with redirect_stdout(captured):
        actual = analysis_utils.format_summary(record)
    assert captured.getvalue() == "", "format_summary() printed instead of returning"
    assert actual == expected, (record, actual, expected)
    assert record == before, "format_summary() mutated its input"

spy_cases = (
    ({"label": TOKEN + " west", "values": [1000, -4, 0.25]}, -7.84, TOKEN + " west mean: -7.8"),
    ({"label": TOKEN + " decimal", "values": [2, 6]}, 123.04, TOKEN + " decimal mean: 123.0"),
    ({"label": TOKEN + " empty", "values": [999]}, None, TOKEN + " empty mean: no measurements"),
)

for record, supplied_result, expected in spy_cases:
    calls = []
    next_result = supplied_result

    def fake_mean(values):
        calls.append(values)
        return next_result

    before = {"label": record["label"], "values": list(record["values"])}
    analysis_utils.mean = fake_mean
    assert analysis_utils.format_summary(record) == expected
    assert len(calls) == 1 and calls[0] is record["values"], "format_summary() did not pass record values to mean() exactly once"
    assert record == before, "format_summary() mutated a spy record"
'''.replace("__DYNAMIC_TOKEN__", repr(token))
    _run_harness(
        root,
        harness,
        "format_summary() failed label, empty, zero, decimal, mean-call, quiet-return, or nonmutation behavior.",
    )


def _is_exact_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or node.orelse or len(node.body) != 1:
        return False
    test = node.test
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    ):
        return False
    expression = node.body[0]
    return (
        isinstance(expression, ast.Expr)
        and isinstance(expression.value, ast.Call)
        and isinstance(expression.value.func, ast.Name)
        and expression.value.func.id == "main"
        and not expression.value.args
        and not expression.value.keywords
    )


def _open_details(call: ast.Call) -> tuple[str | None, str | None, str | None]:
    if not isinstance(call.func, ast.Name) or call.func.id != "open":
        return None, None, None
    path = None
    mode = None
    encoding = None
    if call.args and isinstance(call.args[0], ast.Constant):
        path = call.args[0].value
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        mode = call.args[1].value
    for keyword in call.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode = keyword.value.value
        if keyword.arg == "encoding" and isinstance(keyword.value, ast.Constant):
            encoding = keyword.value.value
    return path, mode, encoding


def check_main_structure(root: Path) -> None:
    tree = _parse(root, "main.py")
    forbidden = _forbidden_constructs(tree)
    _assert(not forbidden, f"main.py uses out-of-scope code: {', '.join(forbidden)}.")

    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    _assert(len(imports) == 1, "Keep exactly the supplied local import in main.py.")
    supplied_import = imports[0]
    _assert(
        isinstance(supplied_import, ast.ImportFrom)
        and supplied_import.module == "analysis_utils"
        and supplied_import.level == 0
        and len(supplied_import.names) == 1
        and supplied_import.names[0].name == "format_summary"
        and supplied_import.names[0].asname is None,
        "Keep the exact `from analysis_utils import format_summary` line.",
    )

    functions = _function_map(tree)
    _assert(list(functions) == ["main"], "Keep exactly one top-level function named main() in main.py.")
    all_functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    _assert(len(all_functions) == 1, "Do not add nested or extra functions to main.py.")
    main_function = functions["main"]
    _validate_signature(main_function, "main", [])

    allowed_top_level = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            allowed_top_level.append(True)
        elif isinstance(node, ast.FunctionDef):
            allowed_top_level.append(True)
        elif _is_exact_main_guard(node):
            allowed_top_level.append(True)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            allowed_top_level.append(True)
        else:
            allowed_top_level.append(False)
    _assert(
        all(allowed_top_level) and sum(_is_exact_main_guard(node) for node in tree.body) == 1,
        "Keep all driver work inside main() and retain the exact main guard.",
    )

    record_assignments = [
        node
        for node in main_function.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "records" for target in node.targets)
    ]
    _assert(len(record_assignments) == 1, "Keep the one supplied records assignment inside main().")
    try:
        actual_records = ast.literal_eval(record_assignments[0].value)
    except (ValueError, TypeError) as error:
        raise AssertionError("Keep the supplied records as the shown literal list of dictionaries.") from error
    _assert(actual_records == EXPECTED_RECORDS, "Do not edit the three supplied records or their order.")

    direct_open_calls = sorted(
        (
            node
            for node in ast.walk(main_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    open_details = [_open_details(call) for call in direct_open_calls]
    _assert(
        open_details
        == [
            ("report.txt", "w", "utf-8"),
            ("report.txt", "r", "utf-8"),
        ],
        'Use exactly two open() calls in order: report.txt mode "w", then report.txt mode "r", '
        'both with encoding="utf-8".',
    )
    _assert(
        all(
            len(call.args) == 2
            and len(call.keywords) == 1
            and call.keywords[0].arg == "encoding"
            for call in direct_open_calls
        ),
        'Use the documented open("report.txt", mode, encoding="utf-8") form without extra arguments.',
    )

    with_blocks = sorted(
        (node for node in ast.walk(main_function) if isinstance(node, ast.With)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    _assert(
        len(with_blocks) == 2
        and all(len(node.items) == 1 for node in with_blocks)
        and all(isinstance(node.items[0].optional_vars, ast.Name) for node in with_blocks)
        and [node.items[0].context_expr for node in with_blocks] == direct_open_calls,
        "Put each of the two report.txt open() calls in its own with block.",
    )
    write_with, read_with = with_blocks
    write_handle = write_with.items[0].optional_vars.id
    read_handle = read_with.items[0].optional_vars.id
    _assert(
        write_handle == "report_file" and read_handle == "report_file",
        "Use `as report_file` for both documented report.txt with blocks.",
    )

    all_calls = [node for node in ast.walk(main_function) if isinstance(node, ast.Call)]
    invalid_calls = []
    for call in all_calls:
        if isinstance(call.func, ast.Name) and call.func.id in {
            "format_summary", "open", "print"
        }:
            continue
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and (
                (call.func.value.id == write_handle and call.func.attr == "write")
                or (call.func.value.id == read_handle and call.func.attr == "read")
            )
        ):
            continue
        if isinstance(call.func, ast.Name):
            invalid_calls.append(f"{call.func.id}()")
        elif isinstance(call.func, ast.Attribute):
            invalid_calls.append(f".{call.func.attr}()")
        else:
            invalid_calls.append("an indirect or dynamically selected call")
    _assert(
        not invalid_calls,
        "main() may directly call only format_summary(), open(), print(), and write()/read() "
        f"on their matching report handles; found {', '.join(sorted(set(invalid_calls)))}.",
    )
    _assert(
        not any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id in {"format_summary", "open", "print"}
            for node in ast.walk(main_function)
        ),
        "Do not replace or dynamically select format_summary(), open(), or print().",
    )

    write_calls = [
        node
        for node in ast.walk(write_with)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == write_handle
        and node.func.attr == "write"
    ]
    _assert(
        len(write_calls) == 1
        and len(write_calls[0].args) == 1
        and isinstance(write_calls[0].args[0], ast.Name)
        and write_calls[0].args[0].id == "report_text"
        and not write_calls[0].keywords,
        "Write the separately built report with `report_file.write(report_text)`.",
    )
    all_write_calls = [
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
    ]
    _assert(
        all_write_calls == write_calls,
        "Use exactly one write() call, inside the mode-w with block.",
    )
    _assert(
        any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == "report_text"
            and node.lineno < write_calls[0].lineno
            for node in ast.walk(main_function)
        ),
        "Build report_text before writing it.",
    )

    saved_names = set()
    read_calls = []
    for node in ast.walk(read_with):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == read_handle
            and call.func.attr == "read"
            and not call.args
            and not call.keywords
        ):
            continue
        read_calls.append(call)
        for target in node.targets:
            if isinstance(target, ast.Name):
                saved_names.add(target.id)
    _assert(
        len(read_calls) == 1
        and len(saved_names) == 1
        and "report_text" not in saved_names,
        "Assign one no-argument report_file.read() result to a local name in the read with block.",
    )
    all_read_calls = [
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read"
    ]
    _assert(
        all_read_calls == read_calls,
        "Use exactly one read() call, inside the mode-r with block.",
    )

    print_calls = [
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    _assert(
        any(
            saved_names & _used_names(call)
            and not any(isinstance(child, ast.Compare) for child in ast.walk(call))
            for call in print_calls
        ),
        "Print the text read back from report.txt.",
    )
    comparison_prints = [
        call
        for call in print_calls
        if any(
            isinstance(child, ast.Compare)
            and len(child.ops) == 1
            and isinstance(child.ops[0], ast.Eq)
            and _used_names(child) == saved_names | {"report_text"}
            for child in ast.walk(call)
        )
    ]
    comparison_result_names = set()
    for node in ast.walk(main_function):
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Compare)
            and len(node.value.ops) == 1
            and isinstance(node.value.ops[0], ast.Eq)
            and _used_names(node.value) == saved_names | {"report_text"}
        ):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                comparison_result_names.add(target.id)
    named_comparison_prints = [
        call
        for call in print_calls
        if comparison_result_names & _used_names(call)
    ]
    _assert(
        comparison_prints or named_comparison_prints,
        "Print either the saved-report == report_text comparison inline or a local name "
        "assigned from that comparison.",
    )


def check_import_safety(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ds217-a02-import-") as temporary:
        project = Path(temporary)
        _copy_files(root, project, ("analysis_utils.py", "main.py"))
        baseline = {path.name: path.read_bytes() for path in project.iterdir()}
        for module_name in ("analysis_utils", "main"):
            completed = _run_python(["-c", f"import {module_name}"], project)
            _assert(
                completed.returncode == 0,
                f"Importing {module_name} failed; keep both modules self-contained in one directory.",
            )
            _assert(
                completed.stdout == b"" and completed.stderr == b"",
                f"Importing {module_name} must be silent.",
            )
            _assert(
                {path.name: path.read_bytes() for path in project.iterdir()} == baseline,
                f"Importing {module_name} created or changed an artifact.",
            )


def check_main_execution(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ds217-a02-main-") as temporary:
        project = Path(temporary)
        _copy_files(root, project, ("analysis_utils.py", "main.py"))
        completed = _run_python(["main.py"], project)
        _assert(
            completed.returncode == 0,
            "main.py failed in a fresh copy containing only the two student modules.",
        )
        _assert(
            completed.stderr == b"" and completed.stdout == EXPECTED_STDOUT.encode("utf-8"),
            "main.py stdout must match the four exact documented lines.",
        )
        report = project / "report.txt"
        _assert(report.is_file(), "main.py did not create report.txt.")
        _assert(
            report.read_bytes() == EXPECTED_REPORT.encode("utf-8"),
            "report.txt must contain the exact three summary lines and final newline.",
        )

        report.write_text("stale correct-looking report\nextra line\n", encoding="utf-8")
        repeated = _run_python(["main.py"], project)
        _assert(
            repeated.returncode == 0
            and repeated.stdout == EXPECTED_STDOUT.encode("utf-8")
            and repeated.stderr == b"",
            "Rerunning main.py after a stale report must reproduce exact stdout.",
        )
        _assert(
            report.read_bytes() == EXPECTED_REPORT.encode("utf-8"),
            "Open report.txt in mode w so rerunning replaces stale content.",
        )


def check_main_uses_formatter(root: Path) -> None:
    for case_number in range(2):
        with tempfile.TemporaryDirectory(prefix="ds217-a02-spy-") as temporary:
            project = Path(temporary)
            token = f"{project.name}-{case_number}"
            returned_lines = [
                f"{token} :: {index} :: {record['label'][::-1]}"
                for index, record in enumerate(EXPECTED_RECORDS)
            ]
            fake_module = f'''EXPECTED_RECORDS = {EXPECTED_RECORDS!r}
RETURNED_LINES = {returned_lines!r}
call_index = 0

def format_summary(record):
    global call_index
    if call_index >= len(EXPECTED_RECORDS):
        raise AssertionError("format_summary called too many times")
    if record != EXPECTED_RECORDS[call_index]:
        raise AssertionError("wrong record or call order")
    with open("_format_calls.txt", "a", encoding="utf-8") as trace_file:
        trace_file.write(record["label"] + "\\n")
    result = RETURNED_LINES[call_index]
    call_index = call_index + 1
    return result
'''
            expected_report = "".join(line + "\n" for line in returned_lines)
            expected_stdout = expected_report + "Saved report matches: True\n"
            shutil.copy2(root / "main.py", project / "main.py")
            (project / "analysis_utils.py").write_text(fake_module, encoding="utf-8")
            completed = _run_python(["main.py"], project)
            _assert(
                completed.returncode == 0,
                "main.py did not work with the required imported format_summary() interface.",
            )
            _assert(
                completed.stderr == b"" and completed.stdout == expected_stdout.encode("utf-8"),
                "Build terminal output from each format_summary() result, including varied results.",
            )
            _assert(
                (project / "report.txt").read_bytes() == expected_report.encode("utf-8"),
                "Build report.txt from each imported format_summary() result.",
            )
            _assert(
                (project / "_format_calls.txt").read_text(encoding="utf-8")
                == "Morning\nEvening\nOvernight\n",
                "Call format_summary() exactly once per supplied record in the supplied order.",
            )


PUBLIC_CHECKS = (
    PublicCheck("project description, run command, and gitignore", check_project_documents),
    PublicCheck("Git state snapshots", check_git_state_answers),
    PublicCheck("analysis_utils structure", check_analysis_structure),
    PublicCheck("mean behavior", check_mean_behavior),
    PublicCheck("format_summary behavior and mean use", check_format_behavior),
    PublicCheck("main structure and read-back", check_main_structure),
    PublicCheck("quiet imports and no import artifact", check_import_safety),
    PublicCheck("fresh main output and overwritten report", check_main_execution),
    PublicCheck("main formatter call order", check_main_uses_formatter),
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
