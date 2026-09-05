"""Dependency-free public structural checks for Assignment 07.

This checker reads files and notebook source. It does not execute notebook code,
award points, or certify the visual quality of rendered charts. Instructor
review may execute a disposable submission copy separately.
"""

from __future__ import annotations

import ast
import csv
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import struct
import sys


ASSIGNMENT_DIR = Path(__file__).resolve().parent
EXPECTED_PYTHON = "3.12.13\n"
EXPECTED_REQUIREMENTS = (
    "numpy==2.0.2\n"
    "pandas==3.0.5\n"
    "matplotlib==3.11.1\n"
    "seaborn==0.13.2\n"
    "altair==5.5.0\n"
)
EXPECTED_GITIGNORE = (
    ".ipynb_checkpoints/\n"
    "__pycache__/\n"
    "*.py[cod]\n"
    ".pytest_cache/\n"
    ".venv/\n"
    "venv/\n"
)
PROTECTED_FILE_SHA256 = {
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "requirements.txt": "841d58adacd0d8eeeb3e08d6691ed606e4f2f5b273bc87c9f2ed78054e585e2a",
    ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95",
    "README.md": "b49db5503c23046e863e708879466bfbdb44bc97968c6df83339481e5924e28c",
    "PLATFORM_CHECK.md": "97ae5dc7479181957c824972c6d5d8a8cdb442ab8bcad726b2062f0a72087b67",
    "data/fixture.json": "1c3397cb2d98ae239f6a7cd254bb3aa9980d94cd23af4546c834a9262de0a28c",
}
EXPECTED_CELL_IDS = [
    "a07-header",
    "a07-setup",
    "a07-terms-data",
    "a07-load",
    "a07-task1-contract",
    "a07-task1-evidence",
    "a07-explore-function",
    "a07-explore-run",
    "a07-task1-reflection",
    "a07-task2-context",
    "a07-supplied-flawed",
    "a07-task2-critique",
    "a07-critique-evidence",
    "a07-redesign-function",
    "a07-redesign-run",
    "a07-task3-contract",
    "a07-final-contract-values",
    "a07-supporting-data",
    "a07-explanatory-function",
    "a07-explanatory-run",
    "a07-evidence-export",
    "a07-visual-review",
    "a07-final-verify",
]
MARKDOWN_IDS = {
    "a07-header",
    "a07-terms-data",
    "a07-task1-contract",
    "a07-task1-reflection",
    "a07-task2-context",
    "a07-task2-critique",
    "a07-task3-contract",
    "a07-visual-review",
}
PROTECTED_CELL_SHA256 = {
    "a07-header": "d3ec4b0e9c59dafcc8b5eec41c17d06059c6810f81455099b4c325c3e29f785f",
    "a07-setup": "f7508db32412cddc016b2635c05c823aac4a4e2f94ad5af61a72764f2572030a",
    "a07-terms-data": "b14274cea7b43e45b41b229a020787836e05c38db6b8f0c299e5a456d04f4676",
    "a07-task2-context": "7ddee594b77784ea7b2684f82f1fb1215bbdf79224ffd153a2269bfec3278fa2",
    "a07-supplied-flawed": "96bfd9dd6114ad84f305dc8567e757ac8ce33cfed55c546447f763bb73bf867b",
    "a07-final-verify": "88b7c3f14677b07bfce5ec7ac96171e6dad5b35131ca7648bbb39d9e0829577f",
}
STUDENT_MARKDOWN_IDS = MARKDOWN_IDS - set(PROTECTED_CELL_SHA256)
STUDENT_CODE_IDS = set(EXPECTED_CELL_IDS) - MARKDOWN_IDS - set(PROTECTED_CELL_SHA256)
REQUIRED_FUNCTIONS = {
    "build_exploratory_chart",
    "build_critique_redesign",
    "build_explanatory_chart",
}
FIXTURE_MANIFEST = {
    "fixture_set_id": "a07-visualization-v1",
    "provenance": "Course-authored synthetic learning-format, session, and pathway records; no real or identifying data.",
    "files": [
        {
            "path": "format_completion.csv",
            "row_grain": "one row per delivery format and stage",
            "row_count": 4,
            "columns": ["format", "stage", "completion_percent"],
            "sha256": "20ad900633154f5f3a2c09cfbc2f890f8423da0897d6345841745332110be66a",
        },
        {
            "path": "pathway_checkpoints.csv",
            "row_grain": "one row per learning pathway and checkpoint",
            "row_count": 8,
            "columns": ["pathway", "checkpoint_number", "completion_percent"],
            "sha256": "ec9a336b7fb97418a6f058704f2509c8cee6b13d744efb7a6e3e99224ef8c258",
        },
        {
            "path": "session_observations.csv",
            "row_grain": "one row per synthetic learning session",
            "row_count": 12,
            "columns": ["session_id", "pathway", "activities_completed", "reflection_score"],
            "sha256": "fc4d69ab836288a2fe9c505c65c08413e137e51ab1914cd0e350f6e6636da096",
        },
    ],
}
OUTPUT_NAMES = {
    "critique_redesign.png",
    "pathway_explanatory.png",
    "explanatory_supporting_data.csv",
    "visualization_evidence.json",
    "explanatory_text_alternative.txt",
}
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json",
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
    *(f"data/{record['path']}" for record in FIXTURE_MANIFEST["files"]),
}
SUPPORTING_BYTES = (
    b"pathway,checkpoint_number,completion_percent\n"
    b"Independent,1,58\nIndependent,2,63\nIndependent,3,67\nIndependent,4,70\n"
    b"Facilitated,1,57\nFacilitated,2,65\nFacilitated,3,72\nFacilitated,4,79\n"
)
CRITIQUE_CATEGORIES = [
    "unsupported claim",
    "truncated baseline",
    "missing unit",
    "color-only encoding",
    "distracting decoration",
]
VARIABLE_ROLES = {
    "pathway": "categorical",
    "checkpoint_number": "ordered",
    "completion_percent": "quantitative",
}
EXPLORATION_ROLES = {
    "session_id": "identifier",
    "pathway": "categorical",
    "activities_completed": "quantitative",
    "reflection_score": "quantitative",
}
BANNED_CALLS = {
    "agg", "aggregate", "groupby", "pivot_table", "transform",
    "corr", "corrwith", "cov", "describe", "mean", "median", "sum",
    "std", "var", "sem", "quantile", "nunique", "value_counts",
    "merge", "join", "concat", "melt", "pivot", "crosstab",
    "bfill", "drop", "drop_duplicates", "dropna", "duplicated",
    "ffill", "fillna", "interpolate", "replace", "to_datetime", "astype",
    "resample", "rolling", "expanding", "ewm", "shift", "lag",
    "pie", "heatmap", "kdeplot", "regplot", "lmplot", "violinplot",
    "pairplot", "jointplot", "load_dataset", "show",
    "eval", "exec", "__import__",
}
BANNED_IMPORT_ROOTS = {
    "bokeh", "holoviews", "panel", "plotly", "scipy",
    "sklearn", "statsmodels", "streamlit", "requests", "urllib", "http",
    "random", "datetime",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _bytes(relative: str) -> bytes:
    path = ASSIGNMENT_DIR / relative
    _assert(path.is_file() and not path.is_symlink(), f"Missing protected file: {relative}.")
    return path.read_bytes()


def _json(path: Path, label: str):
    _assert(path.is_file() and not path.is_symlink(), f"Missing {label}.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"{label} must be valid UTF-8 JSON: {error}") from error


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source if isinstance(source, str) else ""


def _check_submission_inventory() -> None:
    actual = {
        path.relative_to(ASSIGNMENT_DIR).as_posix()
        for path in ASSIGNMENT_DIR.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.relative_to(ASSIGNMENT_DIR).parts[0] != ".git"
        and path.relative_to(ASSIGNMENT_DIR).parts[0] != "output"
    }
    _assert(actual == STUDENT_PACKAGE_FILES, "Remove unexpected submission files.")


def check_environment_and_protected_files() -> None:
    _check_submission_inventory()
    _assert(
        sys.version_info[:3] == (3, 12, 13),
        "Run this checker with the Assignment 07 CPython 3.12.13 interpreter.",
    )
    expected_packages = (
        ("numpy", "2.0.2"),
        ("pandas", "3.0.5"),
        ("matplotlib", "3.11.1"),
        ("seaborn", "0.13.2"),
    )
    for package, expected in expected_packages:
        try:
            observed = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise AssertionError(f"Install {package}=={expected} in this environment.") from error
        _assert(observed == expected, f"Expected {package}=={expected}; found {observed}.")
    _assert(_bytes(".python-version").decode() == EXPECTED_PYTHON, "Restore .python-version.")
    _assert(_bytes("requirements.txt").decode() == EXPECTED_REQUIREMENTS, "Restore requirements.txt.")
    gitignore = _bytes(".gitignore").decode()
    _assert(gitignore == EXPECTED_GITIGNORE, "Restore the supplied .gitignore.")
    hidden = ("output/", "*.png", "*.csv", "*.json", "*.txt")
    _assert(not any(item in gitignore for item in hidden), "Required outputs must remain visible to Git.")
    for relative, expected in PROTECTED_FILE_SHA256.items():
        _assert(sha256(_bytes(relative)).hexdigest() == expected, f"Restore protected {relative}.")


def check_fixtures() -> None:
    data_dir = ASSIGNMENT_DIR / "data"
    manifest = _json(data_dir / "fixture.json", "data/fixture.json")
    _assert(manifest == FIXTURE_MANIFEST, "Restore the exact fixture manifest.")
    actual = {
        path.relative_to(data_dir).as_posix()
        for path in data_dir.rglob("*")
        if path.is_file()
    }
    expected = {"fixture.json", *(record["path"] for record in FIXTURE_MANIFEST["files"])}
    _assert(actual == expected, "Restore the exact four-file fixture inventory.")
    for record in FIXTURE_MANIFEST["files"]:
        relative = Path(record["path"])
        _assert(not relative.is_absolute() and relative.parts == (record["path"],), "Unsafe fixture path.")
        path = data_dir / relative
        raw = path.read_bytes()
        _assert(raw.endswith(b"\n") and b"\r" not in raw, f"Restore LF/final-newline bytes in data/{path.name}.")
        _assert(sha256(raw).hexdigest() == record["sha256"], f"Restore immutable data/{path.name}.")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        _assert(rows and rows[0] == record["columns"], f"Wrong columns in data/{path.name}.")
        _assert(len(rows) - 1 == record["row_count"], f"Wrong row count in data/{path.name}.")


def _load_notebook() -> tuple[dict, dict[str, dict]]:
    notebook = _json(ASSIGNMENT_DIR / "assignment.ipynb", "assignment.ipynb")
    cells = notebook.get("cells")
    _assert(
        notebook.get("nbformat") == 4 and notebook.get("nbformat_minor") == 5,
        "Keep notebook format 4.5.",
    )
    _assert(isinstance(cells, list) and len(cells) == 23, "Restore the exact 23-cell notebook.")
    _assert(all(isinstance(cell, dict) for cell in cells), "Every notebook cell must be an object.")
    ids = [cell.get("id") for cell in cells]
    _assert(ids == EXPECTED_CELL_IDS and len(ids) == len(set(ids)), "Restore the supplied globally unique cell IDs and order.")
    for cell in cells:
        expected_type = "markdown" if cell["id"] in MARKDOWN_IDS else "code"
        _assert(cell.get("cell_type") == expected_type, f"Restore the type of {cell['id']}.")
    kernelspec = notebook.get("metadata", {}).get("kernelspec")
    _assert(
        kernelspec == {"display_name": "Python 3", "language": "python", "name": "python3"},
        "Restore the portable Python 3 kernelspec.",
    )
    return notebook, {cell["id"]: cell for cell in cells}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def check_notebook() -> None:
    _, by_id = _load_notebook()
    for cell_id, expected in PROTECTED_CELL_SHA256.items():
        observed = sha256(_source(by_id[cell_id]).encode()).hexdigest()
        _assert(observed == expected, f"Restore protected notebook cell {cell_id}.")

    student_markdown = "\n".join(_source(by_id[cell_id]) for cell_id in STUDENT_MARKDOWN_IDS)
    student_code = "\n".join(_source(by_id[cell_id]) for cell_id in STUDENT_CODE_IDS)
    _assert("TODO" not in student_markdown + student_code, "Complete every TODO in student-editable cells.")
    _assert("NotImplementedError" not in student_code, "Replace all code scaffolds.")
    try:
        tree = ast.parse(student_code)
    except SyntaxError as error:
        raise AssertionError(f"Student code has a syntax error: {error}") from error
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    _assert(REQUIRED_FUNCTIONS.issubset(functions), "Define all three required chart functions.")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            _assert(not roots or not set(roots) & BANNED_IMPORT_ROOTS, "Remove out-of-scope imports from student cells.")
            raise AssertionError("Do not add imports to student cells; use the supplied setup imports.")
        if isinstance(node, ast.Call):
            name = _call_name(node)
            _assert(name not in BANNED_CALLS, f"Out-of-scope API used: {name}().")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            absolute = value not in {"/", "//", "\\", "\\\\"} and (
                value.startswith(("/", "~")) or (
                len(value) > 2 and value[1] == ":" and value[2] in {"/", "\\"}
                )
            )
            _assert(not absolute, f"Remove absolute path literal: {value!r}.")
    lowered = student_code.lower()
    forbidden_fragments = (
        "/content", "drive.mount", "files.upload", "http://", "https://",
        "urlopen", "requests.", "np.random", "random.", "datetime.now",
        "plotly", "bokeh", "holoviews", "streamlit",
        "animation", "basemap", "geopandas", "credential",
    )
    for fragment in forbidden_fragments:
        _assert(fragment not in lowered, f"Remove nonportable or out-of-scope code: {fragment}.")
    for line in student_code.splitlines():
        _assert(not line.lstrip().startswith(("!", "%")), "Remove notebook magic or shell commands.")
    legacy_fragments = ("q1_", "q2_", "q3_", "exploratory.png", "sales", "customer", "product")
    for fragment in legacy_fragments:
        _assert(fragment not in lowered, f"Remove legacy or unrequested output logic: {fragment}.")

    load_tree = ast.parse(_source(by_id["a07-load"]))
    reads = [
        node for node in ast.walk(load_tree)
        if isinstance(node, ast.Call) and _call_name(node) == "read_csv"
    ]
    _assert(len(reads) == 3, "Load each of the three protected fixtures exactly once with pd.read_csv.")
    load_source = _source(by_id["a07-load"])
    for record in FIXTURE_MANIFEST["files"]:
        _assert(record["path"] in load_source, f"Load protected {record['path']} from DATA_DIR.")
    _assert("dtype" in load_source, "Use the supplied explicit dtype maps for all fixture reads.")
    _assert(not any(
        isinstance(node, ast.Call) and _call_name(node) in {"DataFrame", "read_json", "read_table"}
        for node in ast.walk(load_tree)
    ), "Do not embed or load replacement fixtures.")

    explore_tree = ast.parse(_source(by_id["a07-explore-function"]))
    chart_calls = [node for node in ast.walk(explore_tree) if isinstance(node, ast.Call) and _call_name(node) == "Chart"]
    _assert(len(chart_calls) == 1, "build_exploratory_chart must call alt.Chart exactly once.")
    _assert("mark_point" in _source(by_id["a07-explore-function"]), "The exploratory chart must use filled point marks.")
    _assert(all(channel in _source(by_id["a07-explore-function"]) for channel in ("alt.X", "alt.Y", "alt.Color", "alt.Shape", "tooltip")), "The exploratory chart needs typed position, color, shape, and tooltip encodings.")
    _assert(not any(isinstance(node, ast.Call) and _call_name(node) == "savefig" for node in ast.walk(explore_tree)), "The exploratory function must not save a PNG.")


def _png_contract(path: Path) -> None:
    _assert(path.is_file() and not path.is_symlink(), f"Missing output/{path.name}.")
    size = path.stat().st_size
    _assert(10_000 <= size <= 2_000_000, f"output/{path.name} has an implausible PNG byte size.")
    header = path.read_bytes()[:24]
    _assert(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", f"output/{path.name} is not a valid PNG header.")
    width, height = struct.unpack(">II", header[16:24])
    _assert(800 <= width <= 2000 and 450 <= height <= 1400, f"output/{path.name} has implausible dimensions {width}x{height}.")


def _nonempty_string(value, label: str) -> None:
    _assert(isinstance(value, str) and value.strip(), f"{label} must be nonempty authored text.")


def check_artifacts() -> None:
    output = ASSIGNMENT_DIR / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Missing output directory.")
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _assert(actual == {".gitkeep", *OUTPUT_NAMES}, "Create exactly .gitkeep and the five required output artifacts; remove legacy or extra files.")
    for name in OUTPUT_NAMES:
        _assert(not (output / name).is_symlink(), f"output/{name} must be a regular file.")
    _png_contract(output / "critique_redesign.png")
    _png_contract(output / "pathway_explanatory.png")

    supporting = (output / "explanatory_supporting_data.csv").read_bytes()
    _assert(supporting == SUPPORTING_BYTES, "Regenerate the exact explanatory supporting CSV with LF and a final newline.")
    _assert(sha256(supporting).hexdigest() == FIXTURE_MANIFEST["files"][1]["sha256"], "Supporting CSV checksum mismatch.")

    evidence_path = output / "visualization_evidence.json"
    raw_json = evidence_path.read_bytes()
    _assert(raw_json.endswith(b"\n") and b"\r" not in raw_json, "Write evidence JSON with LF and one final newline.")
    evidence = _json(evidence_path, "output/visualization_evidence.json")
    expected_keys = [
        "schema", "question", "audience", "intended_claim", "displayed_unit",
        "grain", "variable_roles", "exploration", "critique", "text_alternative",
    ]
    _assert(list(evidence) == expected_keys, "Use the exact evidence JSON key topology and insertion order.")
    _assert(evidence["schema"] == "datasci217/a07-visualization-evidence/v1", "Wrong evidence schema.")
    _assert(evidence["displayed_unit"] == "prepared completion percent", "Wrong displayed unit.")
    _assert(evidence["grain"] == "one row per pathway and checkpoint", "Wrong explanatory grain.")
    _assert(evidence["variable_roles"] == VARIABLE_ROLES, "Wrong explanatory variable roles.")
    for key in ("question", "audience", "intended_claim", "text_alternative"):
        _nonempty_string(evidence[key], f"evidence.{key}")
    exploration = evidence.get("exploration")
    _assert(isinstance(exploration, dict) and list(exploration) == ["question", "grain", "variable_roles", "observation", "limitation"], "Wrong exploration evidence topology.")
    _assert(exploration["grain"] == "one row per synthetic learning session", "Wrong exploration grain.")
    _assert(exploration["variable_roles"] == EXPLORATION_ROLES, "Wrong exploration variable roles.")
    for key in ("question", "observation", "limitation"):
        _nonempty_string(exploration[key], f"evidence.exploration.{key}")
    critique = evidence.get("critique")
    _assert(isinstance(critique, list) and len(critique) == 5, "Evidence critique must contain exactly five entries.")
    _assert([entry.get("category") for entry in critique if isinstance(entry, dict)] == CRITIQUE_CATEGORIES, "Use the exact ordered critique categories.")
    for index, entry in enumerate(critique):
        _assert(isinstance(entry, dict) and list(entry) == ["category", "problem", "repair"], f"Wrong keys in critique entry {index + 1}.")
        _nonempty_string(entry["problem"], f"critique entry {index + 1} problem")
        _nonempty_string(entry["repair"], f"critique entry {index + 1} repair")
    deterministic = (json.dumps(evidence, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _assert(raw_json == deterministic, "Serialize evidence JSON deterministically with two-space indentation and one final LF.")

    text_raw = (output / "explanatory_text_alternative.txt").read_bytes()
    _assert(text_raw == (evidence["text_alternative"] + "\n").encode("utf-8"), "Text alternative file must exactly match the JSON value plus one LF.")
    lowered = evidence["text_alternative"].lower()
    required_components = (
        ("line chart",), ("checkpoint",), ("prepared completion", "percent"),
        ("independent",), ("facilitated",), ("58", "70"), ("57", "79"),
        ("nine", "9"), ("cause", "causal"),
    )
    for alternatives in required_components:
        _assert(any(value in lowered for value in alternatives), f"Text alternative is missing a required component: {'/'.join(alternatives)}.")


def main() -> int:
    checks = (
        ("environment and protected files", check_environment_and_protected_files),
        ("fixture integrity", check_fixtures),
        ("notebook contract", check_notebook),
        ("generated artifacts", check_artifacts),
    )
    failures = []
    for label, check in checks:
        try:
            check()
        except Exception as error:
            failures.append(f"[FIX] {label}: {error}")
        else:
            print(f"[OK] {label}")
    if failures:
        print("\n".join(failures))
        print("Assignment 07 is not ready. Fix the messages, restart and run all 23 cells, then check again.")
        return 1
    print("Public machine-readable checks passed. Human chart review is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
