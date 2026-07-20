"""Public Assignment 03 checks shared by the local and managed facades.

This supplied module may use standard-library facilities and NumPy that are
outside the student-code boundary. Pipeline text is parsed before execution
and is never passed to a shell.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import numpy as np


EXPECTED_PYTHON_RECORD = "3.12.13\n"
EXPECTED_REQUIREMENTS = "numpy==2.0.2\n"
EXPECTED_ENVIRONMENT_OUTPUT = "Python: 3.12.13\nNumPy: 2.0.2\n"
EXPECTED_GITIGNORE = (
    ".venv/\n"
    "recreation-check/\n"
    "__pycache__/\n"
    "*.pyc\n"
    ".pytest_cache/\n"
)
EXPECTED_OBSERVATIONS = (
    "site,baseline,follow_up\n"
    "north,10,20\n"
    "south,20,30\n"
    "north,30,40\n"
    "west,10,20\n"
    "south,20,30\n"
    "north,30,40\n"
)
EXPECTED_ENVIRONMENT_CHECK = '''"""Report the candidate Python and NumPy versions."""

import sys

import numpy as np


def main():
    """Print the candidate interpreter and direct dependency versions."""
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    print(f"NumPy: {np.__version__}")


if __name__ == "__main__":
    main()
'''
EXPECTED_DATA_LOADER = '''"""Supply the CSV-to-ndarray boundary for Assignment 03."""

import csv
from pathlib import Path

import numpy as np


def load_measurements(filename):
    """Return the numeric fixture fields as a homogeneous 2D ndarray."""
    data_path = Path(filename)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path

    rows = []
    with data_path.open("r", encoding="utf-8", newline="") as data_file:
        reader = csv.DictReader(data_file)
        for row in reader:
            rows.append([float(row["baseline"]), float(row["follow_up"])])

    return np.array(rows, dtype=np.float64)
'''
EXPECTED_PIPELINE_LINES = (
    "head -n 3 observations.csv > output/head_preview.txt",
    "tail -n 2 observations.csv > output/tail_preview.txt",
    "tail -n +2 observations.csv | cut -d',' -f1 | sort | uniq -c > output/site_counts.txt",
    "wc -l output/site_counts.txt > output/site_count_lines.txt",
)
EXPECTED_STDOUT = (
    "Measurements shape: (6, 2)\n"
    "Measurements dtype: float64\n"
    "Overall mean: 25.0\n"
    "Column means: [20. 30.]\n"
    "Column means shape: (2,)\n"
    "Row means: [15. 25. 35. 15. 25. 35.]\n"
    "Row means shape: (6,)\n"
    "Values at or above 30: 6\n"
)
EXPECTED_ARRAY_MODULE_DOCSTRING = "Complete the required ordinary-ndarray operations."
EXPECTED_ANALYSIS_MODULE_DOCSTRING = (
    "Run one import-safe NumPy analysis over the supplied fixture."
)

FUNCTION_SIGNATURES = (
    ("create_and_describe", ["values"]),
    ("select_parts", ["values"]),
    ("view_and_copy", ["values"]),
    ("vector_operations", ["values", "baseline", "threshold", "offset"]),
    ("reduction_summary", ["values"]),
    ("reshape_and_transpose", ["values", "rows", "columns"]),
    ("count_at_or_above", ["values", "threshold"]),
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


def _run_python(
    arguments: list[str],
    cwd: Path,
    *,
    python_path: Path | None = None,
    timeout: int = 10,
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if python_path is not None:
        environment["PYTHONPATH"] = str(python_path)
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
            f"Python did not finish within {timeout} seconds; check for an endless operation."
        ) from error


def _copy_files(root: Path, destination: Path, filenames: tuple[str, ...]) -> None:
    for filename in filenames:
        source = root / filename
        _assert(source.is_file(), f"Missing {filename}.")
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _last_error(completed: subprocess.CompletedProcess[bytes]) -> str:
    detail = completed.stderr.decode("utf-8", errors="replace").strip()
    return detail.splitlines()[-1] if detail else "no error text"


def _run_harness(root: Path, source: str, check_name: str) -> None:
    with tempfile.TemporaryDirectory(prefix="ds217-a03-harness-") as temporary:
        project = Path(temporary)
        _copy_files(root, project, ("array_analysis.py",))
        (project / "_contract_harness.py").write_text(source, encoding="utf-8")
        completed = _run_python(["_contract_harness.py"], project)
        if completed.returncode != 0:
            raise AssertionError(
                f"{check_name} failed on a fresh alternate input. Last result: {_last_error(completed)}"
            )


def _tracked_environment_paths(root: Path) -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--", ".venv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def check_runtime_records_and_probe(root: Path) -> None:
    _assert(
        _read(root, ".python-version") == EXPECTED_PYTHON_RECORD,
        "Replace .python-version with exactly `3.12.13` and one final newline.",
    )
    _assert(
        _read(root, "requirements.txt") == EXPECTED_REQUIREMENTS,
        "Replace requirements.txt with exactly `numpy==2.0.2` and one final newline.",
    )
    _assert(
        _read(root, ".gitignore") == EXPECTED_GITIGNORE,
        "Restore the supplied .gitignore environment, recreation, and cache rules.",
    )
    _assert(
        _read(root, "observations.csv") == EXPECTED_OBSERVATIONS,
        "Restore the supplied seven-line observations.csv fixture.",
    )
    _assert(
        _read(root, "environment_check.py") == EXPECTED_ENVIRONMENT_CHECK,
        "Restore the supplied environment_check.py without editing it.",
    )
    _assert(
        _read(root, "data_loader.py") == EXPECTED_DATA_LOADER,
        "Restore the supplied data_loader.py without editing it.",
    )
    _assert(
        not _tracked_environment_paths(root),
        "Remove the tracked .venv files from the submission; the local environment is recreated, not submitted.",
    )
    _assert(
        sys.version_info[:3] == (3, 12, 13),
        "Run the checker with the candidate Python 3.12.13 interpreter.",
    )
    _assert(
        np.__version__ == "2.0.2",
        "Run the checker after installing the exact direct requirement NumPy 2.0.2.",
    )

    saved_probe = _read(root, "output/environment_check.txt")
    _assert(
        saved_probe == EXPECTED_ENVIRONMENT_OUTPUT,
        "Regenerate output/environment_check.txt with the documented overwrite command.",
    )
    with tempfile.TemporaryDirectory(prefix="ds217-a03-probe-") as temporary:
        project = Path(temporary)
        _copy_files(root, project, ("environment_check.py",))
        completed = _run_python(["environment_check.py"], project)
        _assert(
            completed.returncode == 0 and completed.stderr == b"",
            f"The supplied environment probe did not run cleanly: {_last_error(completed)}.",
        )
        _assert(
            completed.stdout == saved_probe.encode("utf-8") == EXPECTED_ENVIRONMENT_OUTPUT.encode("utf-8"),
            "The saved environment probe is stale or does not match a fresh candidate-environment run.",
        )


def _pipeline_lines(root: Path) -> tuple[str, ...]:
    source = _read(root, "PIPELINE.md")
    blocks = re.findall(r"(?ms)^```bash\s*\n(.*?)^```\s*$", source)
    _assert(len(blocks) == 1, "Keep exactly one fenced `bash` command block in PIPELINE.md.")
    lines = tuple(line for line in blocks[0].splitlines() if line.strip())
    _assert(len(lines) == 4, "Keep exactly four nonblank commands in the PIPELINE.md block.")
    return lines


def _tokenize_pipeline(line: str) -> list[str]:
    _assert("\n" not in line and "\r" not in line, "Keep each pipeline command on one line.")
    forbidden_text = ("$", "`", ";", "&&", "||", "<<", "\\")
    _assert(
        not any(fragment in line for fragment in forbidden_text),
        "Pipeline commands may not use substitution, variables, semicolons, compounds, here-documents, or escapes.",
    )
    lexer = shlex.shlex(line, posix=True, punctuation_chars="|>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError as error:
        raise AssertionError(f"PIPELINE.md has invalid quoting: {error}.") from error


def _parse_pipeline(line: str) -> tuple[list[list[str]], str]:
    tokens = _tokenize_pipeline(line)
    _assert(tokens.count(">") == 1, "Each command must contain exactly one overwrite redirect `>`.")
    _assert(">>" not in tokens, "Append redirection `>>` is not allowed.")
    redirect_index = tokens.index(">")
    _assert(redirect_index == len(tokens) - 2, "Put one output filename immediately after the final `>`.")
    output_name = tokens[-1]
    _assert(
        re.fullmatch(r"output/[a-z_]+\.txt", output_name) is not None,
        "Redirect only to the documented output/*.txt target.",
    )
    command_tokens = tokens[:redirect_index]
    _assert(command_tokens and command_tokens[0] != "|" and command_tokens[-1] != "|", "Do not leave an empty pipeline stage.")
    stages: list[list[str]] = []
    current: list[str] = []
    for token in command_tokens:
        if token == "|":
            _assert(current, "Do not leave an empty pipeline stage.")
            stages.append(current)
            current = []
        else:
            _assert(token not in {">", ">>"}, "Use only the one final overwrite redirect.")
            current.append(token)
    _assert(current, "Do not leave an empty pipeline stage.")
    stages.append(current)

    allowed_programs = {"head", "tail", "cut", "sort", "uniq", "wc"}
    for stage in stages:
        _assert(stage[0] in allowed_programs, f"Pipeline tool `{stage[0]}` is outside the allowed bounded set.")
        _assert(all("/" not in token for token in stage[0:1]), "Do not select a command by path.")
    return stages, output_name


def _execute_pipeline(lines: tuple[str, ...], csv_text: str) -> dict[str, bytes]:
    parsed = [_parse_pipeline(line) for line in lines]
    with tempfile.TemporaryDirectory(prefix="ds217-a03-pipeline-") as temporary:
        project = Path(temporary)
        (project / "output").mkdir()
        (project / "observations.csv").write_text(csv_text, encoding="utf-8")
        for _, output_name in parsed:
            (project / output_name).write_bytes(b"STALE APPEND SENTINEL\n")

        for stages, output_name in parsed:
            stage_input: bytes | None = None
            for stage in stages:
                try:
                    completed = subprocess.run(
                        stage,
                        cwd=project,
                        input=stage_input,
                        capture_output=True,
                        timeout=5,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired) as error:
                    raise AssertionError(f"Could not execute bounded pipeline stage `{stage[0]}`: {error}.") from error
                _assert(
                    completed.returncode == 0,
                    f"Pipeline stage `{stage[0]}` failed: {completed.stderr.decode(errors='replace').strip()}.",
                )
                stage_input = completed.stdout
            (project / output_name).write_bytes(stage_input or b"")

        return {
            output_name: (project / output_name).read_bytes()
            for _, output_name in parsed
        }


def _normalized_counts(content: bytes) -> list[tuple[int, str]]:
    pairs = []
    for line in content.decode("utf-8").splitlines():
        fields = line.split()
        _assert(len(fields) == 2 and fields[0].isdigit(), "site_counts.txt must contain one integer and site name per line.")
        pairs.append((int(fields[0]), fields[1]))
    return pairs


def _assert_pipeline_outputs(
    outputs: dict[str, bytes],
    *,
    head: bytes,
    tail: bytes,
    counts: list[tuple[int, str]],
) -> None:
    _assert(outputs["output/head_preview.txt"] == head, "The head preview is not the first three fixture lines.")
    _assert(outputs["output/tail_preview.txt"] == tail, "The tail preview is not the final two fixture lines.")
    _assert(_normalized_counts(outputs["output/site_counts.txt"]) == counts, "The normalized sorted site counts are incorrect.")
    count_fields = outputs["output/site_count_lines.txt"].decode("utf-8").split()
    _assert(
        count_fields == [str(len(counts)), "output/site_counts.txt"],
        "site_count_lines.txt must be the direct `wc -l output/site_counts.txt` result.",
    )
    _assert(
        all(b"STALE" not in content for content in outputs.values()),
        "Each pipeline command must overwrite stale output rather than append to it.",
    )


def check_pipeline_structure_and_execution(root: Path) -> None:
    lines = _pipeline_lines(root)
    for line in lines:
        _parse_pipeline(line)
    _assert(
        lines == EXPECTED_PIPELINE_LINES,
        "Replace the four TODO lines with the four exact documented commands, in order.",
    )

    base_outputs = _execute_pipeline(lines, EXPECTED_OBSERVATIONS)
    _assert_pipeline_outputs(
        base_outputs,
        head=b"site,baseline,follow_up\nnorth,10,20\nsouth,20,30\n",
        tail=b"south,20,30\nnorth,30,40\n",
        counts=[(3, "north"), (2, "south"), (1, "west")],
    )

    alternate_csv = (
        "site,baseline,follow_up\n"
        "east,-5.5,1.5\n"
        "south,2.0,4.0\n"
        "north,7.5,8.5\n"
        "east,3.0,9.0\n"
        "south,1.0,2.0\n"
        "south,10.0,12.0\n"
    )
    alternate_outputs = _execute_pipeline(lines, alternate_csv)
    _assert_pipeline_outputs(
        alternate_outputs,
        head=b"site,baseline,follow_up\neast,-5.5,1.5\nsouth,2.0,4.0\n",
        tail=b"south,1.0,2.0\nsouth,10.0,12.0\n",
        counts=[(2, "east"), (1, "north"), (3, "south")],
    )


def _function_map(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _validate_signature(function: ast.FunctionDef, arguments: list[str]) -> None:
    actual = [argument.arg for argument in function.args.args]
    _assert(
        not function.args.posonlyargs and actual == arguments,
        f"Use the exact signature {function.name}({', '.join(arguments)}).",
    )
    _assert(
        function.args.vararg is None
        and function.args.kwarg is None
        and not function.args.kwonlyargs
        and not function.args.defaults
        and not function.args.kw_defaults,
        f"Do not add defaults, keyword-only parameters, *args, or **kwargs to {function.name}().",
    )
    _assert(
        function.returns is None
        and all(argument.annotation is None for argument in function.args.args),
        f"Do not add type annotations to {function.name}().",
    )
    _assert(not function.decorator_list, f"Do not add a decorator to {function.name}().")
    docstring = ast.get_docstring(function, clean=False)
    _assert(docstring is not None and docstring.strip(), f"Add a one-line docstring to {function.name}().")
    _assert("\n" not in docstring and "todo" not in docstring.lower(), f"Replace the one-line TODO docstring in {function.name}().")
    _assert(
        not any(isinstance(node, ast.Pass) for node in function.body),
        f"Replace the pass placeholder in {function.name}().",
    )


def _forbidden_constructs(tree: ast.Module) -> list[str]:
    forbidden_types = {
        ast.For: "for loop",
        ast.While: "while loop",
        ast.ListComp: "list comprehension",
        ast.SetComp: "set comprehension",
        ast.DictComp: "dictionary comprehension",
        ast.GeneratorExp: "generator expression",
        ast.Yield: "generator function",
        ast.YieldFrom: "generator function",
        ast.Try: "exception handling",
        ast.TryStar: "exception handling",
        ast.Raise: "raise statement",
        ast.With: "file/context-manager operation",
        ast.AsyncWith: "async context manager",
        ast.ClassDef: "class",
        ast.Lambda: "lambda",
        ast.AsyncFunctionDef: "async function",
        ast.Await: "await",
        ast.Global: "global statement",
        ast.Nonlocal: "nonlocal statement",
        ast.AnnAssign: "type annotation",
        ast.Delete: "delete statement",
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
        if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, ast.Store):
            found.append("input/result mutation")
        if isinstance(node, ast.AugAssign):
            found.append("augmented assignment")
    return sorted(set(found))


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return f"{call.func.value.id}.{call.func.attr}"
    if isinstance(call.func, ast.Attribute):
        return f".{call.func.attr}"
    return "<indirect>"


def _calls(function: ast.FunctionDef) -> list[str]:
    return [_call_name(node) for node in ast.walk(function) if isinstance(node, ast.Call)]


def _assert_exact_calls(function: ast.FunctionDef, expected: list[str]) -> None:
    actual = _calls(function)
    _assert(
        actual == expected,
        f"{function.name}() must make only the documented direct calls {expected}; found {actual}.",
    )


def _is_np_attribute(node: ast.AST, attribute: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "np"
        and node.attr == attribute
    )


def _direct_assignments(function: ast.FunctionDef, name: str) -> list[ast.AST]:
    values = []
    for statement in function.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
        ):
            values.append(statement.value)
    return values


def _return_value(function: ast.FunctionDef) -> ast.AST | None:
    returns = [node for node in function.body if isinstance(node, ast.Return)]
    return returns[-1].value if len(returns) == 1 else None


def check_student_code_structure(root: Path) -> None:
    array_tree = _parse(root, "array_analysis.py")
    analysis_tree = _parse(root, "analysis.py")
    forbidden = _forbidden_constructs(array_tree)
    _assert(not forbidden, f"array_analysis.py uses out-of-scope code: {', '.join(forbidden)}.")
    analysis_forbidden = _forbidden_constructs(analysis_tree)
    _assert(not analysis_forbidden, f"analysis.py uses out-of-scope code: {', '.join(analysis_forbidden)}.")
    _assert(
        ast.get_docstring(array_tree, clean=False) == EXPECTED_ARRAY_MODULE_DOCSTRING,
        "Keep the exact supplied array_analysis.py module docstring.",
    )
    _assert(
        ast.get_docstring(analysis_tree, clean=False) == EXPECTED_ANALYSIS_MODULE_DOCSTRING,
        "Keep the exact supplied analysis.py module docstring.",
    )

    _assert(
        len(array_tree.body) == 9
        and isinstance(array_tree.body[0], ast.Expr)
        and isinstance(array_tree.body[1], ast.Import)
        and len(array_tree.body[1].names) == 1
        and array_tree.body[1].names[0].name == "numpy"
        and array_tree.body[1].names[0].asname == "np",
        "Keep only the module docstring, exact `import numpy as np`, and seven required functions in array_analysis.py.",
    )
    functions = _function_map(array_tree)
    _assert(
        list(functions) == [name for name, _ in FUNCTION_SIGNATURES],
        "Keep exactly the seven required array_analysis.py functions in the documented order.",
    )
    _assert(
        len([node for node in ast.walk(array_tree) if isinstance(node, ast.FunctionDef)]) == 7,
        "Do not add nested or extra functions to array_analysis.py.",
    )
    for name, arguments in FUNCTION_SIGNATURES:
        _validate_signature(functions[name], arguments)

    for node in ast.walk(array_tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "np":
                _assert(
                    node.attr in {"array", "float64", "mean", "reshape", "sum"},
                    f"NumPy attribute np.{node.attr} is outside the required operation set.",
                )
            else:
                _assert(
                    node.attr in {"shape", "ndim", "size", "dtype", "copy", "T"},
                    f"Attribute .{node.attr} is outside the required metadata/view/transpose set.",
                )
        if isinstance(node, ast.Subscript) and isinstance(node.slice, (ast.List, ast.Set, ast.Dict)):
            raise AssertionError("Fancy integer/container indexing is outside this assignment.")

    _assert_exact_calls(functions["create_and_describe"], ["np.array"])
    _assert_exact_calls(functions["select_parts"], [])
    _assert_exact_calls(functions["view_and_copy"], [".copy"])
    _assert_exact_calls(functions["vector_operations"], [])
    _assert_exact_calls(functions["reduction_summary"], ["np.mean", "np.mean", "np.mean"])
    _assert_exact_calls(functions["reshape_and_transpose"], ["np.reshape"])
    _assert_exact_calls(functions["count_at_or_above"], ["np.reshape", "np.sum"])

    create_values = _direct_assignments(functions["create_and_describe"], "array")
    _assert(len(create_values) == 1 and isinstance(create_values[0], ast.Call), "Assign one direct np.array(...) result to `array`.")
    create_call = create_values[0]
    _assert(
        _is_np_attribute(create_call.func, "array")
        and len(create_call.args) == 1
        and isinstance(create_call.args[0], ast.Name)
        and create_call.args[0].id == "values"
        and len(create_call.keywords) == 1
        and create_call.keywords[0].arg == "dtype"
        and _is_np_attribute(create_call.keywords[0].value, "float64"),
        "Create `array` with exactly `np.array(values, dtype=np.float64)`.",
    )

    select_return = _return_value(functions["select_parts"])
    expected_select_return = ast.parse(
        """{
            "first_value": values[0, 0],
            "second_row": values[1],
            "second_column": values[:, 1],
            "top_left_block": values[:2, :2],
        }""",
        mode="eval",
    ).body
    _assert(
        ast.dump(select_return, include_attributes=False)
        == ast.dump(expected_select_return, include_attributes=False),
        "Return the four exact documented select_parts() keys mapped directly to "
        "`values[0, 0]`, `values[1]`, `values[:, 1]`, and `values[:2, :2]`.",
    )

    view_function = functions["view_and_copy"]
    view_values = _direct_assignments(view_function, "middle_view")
    copy_values = _direct_assignments(view_function, "middle_copy")
    _assert(len(view_values) == 1 and len(copy_values) == 1, "Assign both `middle_view` and `middle_copy` directly.")
    _assert(
        isinstance(copy_values[0], ast.Call)
        and isinstance(copy_values[0].func, ast.Attribute)
        and copy_values[0].func.attr == "copy",
        "Create `middle_copy` by calling `.copy()` on the same basic slice.",
    )

    reduction_calls = [node for node in ast.walk(functions["reduction_summary"]) if isinstance(node, ast.Call)]
    axes = []
    for call in reduction_calls:
        axis_keywords = [keyword for keyword in call.keywords if keyword.arg == "axis"]
        axes.append(None if not axis_keywords else getattr(axis_keywords[0].value, "value", object()))
    _assert(axes == [None, 0, 1], "Call np.mean for the whole array, axis=0, then axis=1.")

    reshape_values = _direct_assignments(functions["reshape_and_transpose"], "grid")
    _assert(
        len(reshape_values) == 1
        and isinstance(reshape_values[0], ast.Call)
        and _is_np_attribute(reshape_values[0].func, "reshape")
        and len(reshape_values[0].args) == 2
        and isinstance(reshape_values[0].args[1], ast.Tuple)
        and [getattr(item, "id", None) for item in reshape_values[0].args[1].elts] == ["rows", "columns"],
        "Assign `grid = np.reshape(values, (rows, columns))`.",
    )
    transposed_values = _direct_assignments(functions["reshape_and_transpose"], "transposed")
    _assert(
        len(transposed_values) == 1
        and isinstance(transposed_values[0], ast.Attribute)
        and isinstance(transposed_values[0].value, ast.Name)
        and transposed_values[0].value.id == "grid"
        and transposed_values[0].attr == "T",
        "Assign `transposed = grid.T`.",
    )

    count_values = _direct_assignments(functions["count_at_or_above"], "flattened")
    _assert(
        len(count_values) == 1
        and isinstance(count_values[0], ast.Call)
        and _is_np_attribute(count_values[0].func, "reshape")
        and len(count_values[0].args) == 2
        and isinstance(count_values[0].args[0], ast.Name)
        and count_values[0].args[0].id == "values"
        and isinstance(count_values[0].args[1], ast.Attribute)
        and isinstance(count_values[0].args[1].value, ast.Name)
        and count_values[0].args[1].value.id == "values"
        and count_values[0].args[1].attr == "size",
        "First assign `flattened = np.reshape(values, values.size)` in count_at_or_above().",
    )
    count_return = _return_value(functions["count_at_or_above"])
    _assert(
        isinstance(count_return, ast.Call)
        and _is_np_attribute(count_return.func, "sum")
        and len(count_return.args) == 1
        and not count_return.keywords,
        "Return the whole boolean-mask count directly with one-argument np.sum(...).",
    )

    expected_analysis_imports = (
        ast.ImportFrom(
            module="array_analysis",
            names=[
                ast.alias(name="count_at_or_above"),
                ast.alias(name="create_and_describe"),
                ast.alias(name="reduction_summary"),
            ],
            level=0,
        ),
        ast.ImportFrom(
            module="data_loader",
            names=[ast.alias(name="load_measurements")],
            level=0,
        ),
    )
    _assert(
        len(analysis_tree.body) == 5
        and isinstance(analysis_tree.body[0], ast.Expr)
        and ast.dump(analysis_tree.body[1], include_attributes=False) == ast.dump(expected_analysis_imports[0], include_attributes=False)
        and ast.dump(analysis_tree.body[2], include_attributes=False) == ast.dump(expected_analysis_imports[1], include_attributes=False)
        and isinstance(analysis_tree.body[3], ast.FunctionDef)
        and analysis_tree.body[3].name == "main"
        and isinstance(analysis_tree.body[4], ast.If),
        "Keep only the module docstring, exact supplied imports, main(), and exact main guard in analysis.py.",
    )
    main_function = analysis_tree.body[3]
    _validate_signature(main_function, [])
    _assert(
        _calls(main_function)
        == [
            "load_measurements",
            "create_and_describe",
            "reduction_summary",
            "count_at_or_above",
            "print",
            "print",
            "print",
            "print",
            "print",
            "print",
            "print",
            "print",
        ],
        "main() must call the loader and three helpers once in order, then make exactly eight direct print() calls.",
    )
    guard = analysis_tree.body[4]
    _assert(
        isinstance(guard.test, ast.Compare)
        and isinstance(guard.test.left, ast.Name)
        and guard.test.left.id == "__name__"
        and len(guard.test.ops) == 1
        and isinstance(guard.test.ops[0], ast.Eq)
        and len(guard.test.comparators) == 1
        and isinstance(guard.test.comparators[0], ast.Constant)
        and guard.test.comparators[0].value == "__main__"
        and len(guard.body) == 1
        and isinstance(guard.body[0], ast.Expr)
        and isinstance(guard.body[0].value, ast.Call)
        and isinstance(guard.body[0].value.func, ast.Name)
        and guard.body[0].value.func.id == "main"
        and not guard.body[0].value.args
        and not guard.orelse,
        "Keep the exact `if __name__ == \"__main__\": main()` guard.",
    )


def check_metadata_and_selection(root: Path) -> None:
    harness = r'''import numpy as np
from array_analysis import create_and_describe, select_parts

first = create_and_describe([[1, -2.5, 7], [4.25, 0, 9]])
assert set(first) == {"array", "shape", "ndim", "size", "dtype"}
np.testing.assert_array_equal(first["array"], np.array([[1, -2.5, 7], [4.25, 0, 9]], dtype=np.float64))
assert first["array"].dtype == np.dtype(np.float64)
assert first["shape"] == (2, 3)
assert first["ndim"] == 2
assert first["size"] == 6
assert first["dtype"] == np.dtype(np.float64)

second = create_and_describe([3, -1, 8, 2])
assert second["shape"] == (4,) and second["ndim"] == 1 and second["size"] == 4
assert second["dtype"] == np.dtype(np.float64)

table = np.array([[-4.5, 2.0, 8.5, 9.0], [1.25, -3.0, 7.0, 6.0], [10.0, 11.0, 12.0, 13.0]])
before = table.copy()
parts = select_parts(table)
assert set(parts) == {"first_value", "second_row", "second_column", "top_left_block"}
assert parts["first_value"] == -4.5
np.testing.assert_array_equal(parts["second_row"], table[1])
np.testing.assert_array_equal(parts["second_column"], table[:, 1])
np.testing.assert_array_equal(parts["top_left_block"], table[:2, :2])
np.testing.assert_array_equal(table, before)
'''
    _run_harness(root, harness, "Metadata and basic selection")


def check_view_copy_relationship(root: Path) -> None:
    harness = r'''import numpy as np
from array_analysis import view_and_copy

for source in (np.array([10, -2, 3.5, 40, 9]), np.array([1, 2, 3, 4], dtype=np.int16)):
    before = source.copy()
    result = view_and_copy(source)
    assert set(result) == {"view", "copy"}
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(result["view"], before[1:3])
    np.testing.assert_array_equal(result["copy"], before[1:3])
    assert np.shares_memory(result["view"], source)
    assert not np.shares_memory(result["copy"], source)
    assert not np.shares_memory(result["copy"], result["view"])
    original_first = source[1].copy()
    result["view"][0] = original_first + 7
    assert source[1] == original_first + 7
    source_after_view = source.copy()
    result["copy"][0] = original_first - 11
    np.testing.assert_array_equal(source, source_after_view)
'''
    _run_harness(root, harness, "View/copy relationship and nonmutation")


def check_vector_operations(root: Path) -> None:
    harness = r'''import numpy as np
from array_analysis import vector_operations

cases = [
    (np.array([-2.5, 0.0, 3.5, 3.6]), np.array([-3.0, 1.0, 2.0, 5.0]), 3.5, -1.25),
    (np.array([10, -4, 10, 2]), np.array([7, -6, 8, 3]), 10, 2),
]
for values, baseline, threshold, offset in cases:
    values_before = values.copy()
    baseline_before = baseline.copy()
    result = vector_operations(values, baseline, threshold, offset)
    assert set(result) == {"mask", "selected", "difference", "adjusted"}
    expected_mask = values_before >= threshold
    np.testing.assert_array_equal(result["mask"], expected_mask)
    assert result["mask"].dtype == np.dtype(bool) and result["mask"].ndim == 1
    np.testing.assert_array_equal(result["selected"], values_before[expected_mask])
    np.testing.assert_array_equal(result["difference"], values_before - baseline_before)
    np.testing.assert_array_equal(result["adjusted"], values_before + offset)
    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(baseline, baseline_before)
'''
    _run_harness(root, harness, "Vector mask, selection, arithmetic, and scalar broadcast")


def check_reductions_reshape_transpose_and_count(root: Path) -> None:
    harness = r'''import numpy as np
from array_analysis import count_at_or_above, reduction_summary, reshape_and_transpose

for values in (
    np.array([[-3.5, 1.0, 8.5], [2.5, -4.0, 7.0]]),
    np.array([[1, 2], [3, 5], [-7, 11]], dtype=np.int16),
):
    before = values.copy()
    result = reduction_summary(values)
    assert set(result) == {"overall_mean", "column_means", "column_means_shape", "row_means", "row_means_shape"}
    np.testing.assert_allclose(result["overall_mean"], np.mean(values))
    np.testing.assert_allclose(result["column_means"], np.mean(values, axis=0))
    np.testing.assert_allclose(result["row_means"], np.mean(values, axis=1))
    assert result["column_means_shape"] == np.mean(values, axis=0).shape
    assert result["row_means_shape"] == np.mean(values, axis=1).shape
    np.testing.assert_array_equal(values, before)

for values, rows, columns in (
    (np.array([-3, 1, 8, 2, -4, 7, 9, 0, 5, 6, 11, 12]), 3, 4),
    (np.array([0.5, -1.5, 2.5, 3.5, 4.5, 5.5]), 2, 3),
):
    before = values.copy()
    result = reshape_and_transpose(values, rows, columns)
    expected_grid = np.reshape(values, (rows, columns))
    assert set(result) == {"grid", "grid_shape", "transposed", "transposed_shape"}
    np.testing.assert_array_equal(result["grid"], expected_grid)
    assert result["grid_shape"] == (rows, columns)
    np.testing.assert_array_equal(result["transposed"], expected_grid.T)
    assert result["transposed_shape"] == (columns, rows)
    np.testing.assert_array_equal(values, before)

count_cases = [
    (np.array([[-2.5, 3.0, 3.1], [8.0, -1.0, 3.0]]), 3.0),
    (np.array([[[1, 5], [5, -2]], [[8, 0], [5, 4]]]), 5),
]
for values, threshold in count_cases:
    before = values.copy()
    result = count_at_or_above(values, threshold)
    assert np.isscalar(result)
    assert result == np.sum(np.reshape(values, values.size) >= threshold)
    np.testing.assert_array_equal(values, before)
'''
    _run_harness(root, harness, "Reductions, shapes, reshape, transpose, and whole count")


def _snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def check_import_safety(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ds217-a03-import-project-") as project_temporary:
        project = Path(project_temporary)
        _copy_files(
            root,
            project,
            ("array_analysis.py", "analysis.py", "data_loader.py", "observations.csv"),
        )
        before = _snapshot_files(project)
        with tempfile.TemporaryDirectory(prefix="ds217-a03-import-cwd-") as cwd_temporary:
            cwd_root = Path(cwd_temporary)
            cwd_before = _snapshot_files(cwd_root)
            completed = _run_python(
                ["-c", "import array_analysis; import data_loader; import analysis"],
                cwd_root,
                python_path=project,
            )
            cwd_after = _snapshot_files(cwd_root)
        after = _snapshot_files(project)
        _assert(
            completed.returncode == 0,
            f"Fresh imports failed: {_last_error(completed)}.",
        )
        _assert(completed.stdout == b"" and completed.stderr == b"", "Imports must be completely quiet.")
        _assert(
            after == before and cwd_after == cwd_before,
            "Imports must not create, remove, or change an artifact in either the project or working directory.",
        )


def _check_driver_spy(
    root: Path,
    *,
    label: str,
    data_loader_source: str,
    array_analysis_source: str,
    expected_stdout: bytes,
) -> None:
    with tempfile.TemporaryDirectory(prefix="ds217-a03-spy-project-") as project_temporary:
        project = Path(project_temporary)
        _copy_files(root, project, ("analysis.py",))
        (project / "data_loader.py").write_text(data_loader_source, encoding="utf-8")
        (project / "array_analysis.py").write_text(
            array_analysis_source,
            encoding="utf-8",
        )
        with tempfile.TemporaryDirectory(prefix="ds217-a03-spy-cwd-") as cwd_temporary:
            completed = _run_python([str(project / "analysis.py")], Path(cwd_temporary))
        _assert(
            completed.returncode == 0,
            f"analysis.py failed with the {label} spies from another working directory: {_last_error(completed)}.",
        )
        _assert(
            completed.stderr == b"load\ndescribe\nreduce\ncount\n",
            f"Call the loader and three helpers exactly once and in order for the {label} spy.",
        )
        _assert(
            completed.stdout == expected_stdout,
            f"Build all eight driver lines from the {label} helper results; do not select memorized output by shape.",
        )


def check_driver_dataflow_and_cwd(root: Path) -> None:
    _check_driver_spy(
        root,
        label="different-shape",
        data_loader_source='''import sys
import numpy as np

SENTINEL = np.array([[-9.0, 4.0], [2.0, -1.0], [7.5, 0.5]], dtype=np.float32)

def load_measurements(filename):
    assert filename == "observations.csv"
    print("load", file=sys.stderr)
    return SENTINEL
''',
        array_analysis_source='''import sys
import numpy as np
from data_loader import SENTINEL

def create_and_describe(values):
    assert values is SENTINEL
    print("describe", file=sys.stderr)
    return {"array": values, "shape": (3, 2), "ndim": 2, "size": 6, "dtype": np.dtype(np.float32)}

def reduction_summary(values):
    assert values is SENTINEL
    print("reduce", file=sys.stderr)
    return {
        "overall_mean": -1.5,
        "column_means": np.array([-2.5, -0.5]),
        "column_means_shape": (2,),
        "row_means": np.array([-3.0, -1.0, 0.5]),
        "row_means_shape": (3,),
    }

def count_at_or_above(values, threshold):
    assert values is SENTINEL and threshold == 30
    print("count", file=sys.stderr)
    return np.int64(4)
''',
        expected_stdout=(
            "Measurements shape: (3, 2)\n"
            "Measurements dtype: float32\n"
            "Overall mean: -1.5\n"
            "Column means: [-2.5 -0.5]\n"
            "Column means shape: (2,)\n"
            "Row means: [-3.  -1.   0.5]\n"
            "Row means shape: (3,)\n"
            "Values at or above 30: 4\n"
        ).encode("utf-8"),
    )

    _check_driver_spy(
        root,
        label="same-shape alternate",
        data_loader_source='''import sys
import numpy as np

SENTINEL = np.array(
    [
        [-12, 3],
        [4, -5],
        [6, 7],
        [-8, 9],
        [10, -11],
        [12, 13],
    ],
    dtype=np.int16,
)

def load_measurements(filename):
    assert filename == "observations.csv"
    print("load", file=sys.stderr)
    return SENTINEL
''',
        array_analysis_source='''import sys
import numpy as np
from data_loader import SENTINEL

def create_and_describe(values):
    assert values is SENTINEL
    print("describe", file=sys.stderr)
    return {
        "array": values,
        "shape": (6, 2),
        "ndim": 2,
        "size": 12,
        "dtype": np.dtype(np.int16),
    }

def reduction_summary(values):
    assert values is SENTINEL
    print("reduce", file=sys.stderr)
    return {
        "overall_mean": 99.5,
        "column_means": np.array([101, -202]),
        "column_means_shape": (2,),
        "row_means": np.array([1, -2, 3, -4, 5, -6]),
        "row_means_shape": (6,),
    }

def count_at_or_above(values, threshold):
    assert values is SENTINEL and threshold == 30
    print("count", file=sys.stderr)
    return np.int64(11)
''',
        expected_stdout=(
            "Measurements shape: (6, 2)\n"
            "Measurements dtype: int16\n"
            "Overall mean: 99.5\n"
            "Column means: [ 101 -202]\n"
            "Column means shape: (2,)\n"
            "Row means: [ 1 -2  3 -4  5 -6]\n"
            "Row means shape: (6,)\n"
            "Values at or above 30: 11\n"
        ).encode("utf-8"),
    )


def check_exact_driver_output(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ds217-a03-driver-project-") as project_temporary:
        project = Path(project_temporary)
        _copy_files(
            root,
            project,
            ("array_analysis.py", "analysis.py", "data_loader.py", "observations.csv"),
        )
        before = _snapshot_files(project)
        with tempfile.TemporaryDirectory(prefix="ds217-a03-driver-cwd-") as cwd_temporary:
            completed = _run_python([str(project / "analysis.py")], Path(cwd_temporary))
        after = _snapshot_files(project)
        _assert(
            completed.returncode == 0,
            f"analysis.py failed from another working directory: {_last_error(completed)}.",
        )
        _assert(completed.stderr == b"", "analysis.py must not write diagnostic text to stderr.")
        _assert(completed.stdout == EXPECTED_STDOUT.encode("utf-8"), "analysis.py stdout does not match the exact eight-line contract.")
        _assert(after == before, "Running analysis.py must not create, remove, or change a report or other artifact.")

        loader_check = _run_python(
            [
                "-c",
                "from data_loader import load_measurements; "
                "data = load_measurements('observations.csv'); "
                "print(data.shape); print(data.dtype)",
            ],
            project,
        )
        _assert(
            loader_check.returncode == 0
            and loader_check.stderr == b""
            and loader_check.stdout == b"(6, 2)\nfloat64\n",
            "The supplied loader boundary must return the exact float64 (6, 2) array.",
        )


PUBLIC_CHECKS = (
    PublicCheck("runtime records, supplied files, and fresh probe", check_runtime_records_and_probe),
    PublicCheck("safe pipeline structure and base/alternate execution", check_pipeline_structure_and_execution),
    PublicCheck("student code structure and direct-operation boundaries", check_student_code_structure),
    PublicCheck("array metadata and basic selection", check_metadata_and_selection),
    PublicCheck("view/copy relationship and nonmutation", check_view_copy_relationship),
    PublicCheck("vector mask, selection, arithmetic, and scalar broadcast", check_vector_operations),
    PublicCheck("reductions, shapes, reshape, transpose, and whole count", check_reductions_reshape_transpose_and_count),
    PublicCheck("quiet artifact-free imports", check_import_safety),
    PublicCheck("driver helper dataflow, call order, and different CWD", check_driver_dataflow_and_cwd),
    PublicCheck("exact fresh driver and loader output", check_exact_driver_output),
)


def run_public_checks(root: Path) -> list[tuple[str, str | None]]:
    results = []
    for check in PUBLIC_CHECKS:
        try:
            check.action(root)
        except Exception as error:  # keep local feedback moving through all checks
            results.append((check.name, str(error) or error.__class__.__name__))
        else:
            results.append((check.name, None))
    return results
