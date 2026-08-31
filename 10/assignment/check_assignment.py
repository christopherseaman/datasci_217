#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "matplotlib==3.11.1",
#   "numpy==2.0.2",
#   "pandas==3.0.5",
#   "scikit-learn==1.9.0",
#   "statsmodels==0.14.6",
# ]
# ///

"""Standard-library-only readiness checker for Assignment 10.

This checker never imports or executes notebook code. Central grading is
independent and clears stored output before fresh execution.
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


ROOT = Path(__file__).resolve().parent
EXPECTED_PYTHON = (3, 12, 13)
EXPECTED_DISTRIBUTIONS = {
    "matplotlib": "3.11.1",
    "numpy": "2.0.2",
    "pandas": "3.0.5",
    "scikit-learn": "1.9.0",
    "statsmodels": "0.14.6",
}
PEP_REQUIRES_PYTHON = "==3.12.13"
PEP_DEPENDENCIES = [
    "matplotlib==3.11.1",
    "numpy==2.0.2",
    "pandas==3.0.5",
    "scikit-learn==1.9.0",
    "statsmodels==0.14.6",
]
PEP_BLOCK = [
    "# /// script",
    f'# requires-python = "{PEP_REQUIRES_PYTHON}"',
    "# dependencies = [",
    *[f'#   "{dependency}",' for dependency in PEP_DEPENDENCIES],
    "# ]",
    "# ///",
]
FIXTURES = {
    "data/fixture.json": (2170, "aa50eeffc2b07c5d98cb56a0e3d18115909958f777899d5d403cf6323dd1de41"),
    "data/mixing_runs.csv": (370, "00b8a1ce84110f4a7fa85620742283c82a4b9d600dbe0ebea0d4721956938957"),
    "data/batch_strength.csv": (3449, "f14faf7da64347dfc255aa84b14e79eef7f2d0de94b394c747323319d937baa3"),
    "data/feature_availability.csv": (155, "a47b8df048607045640b9a6785b038fe1c70036f58d5b61ed20ec98860b556da"),
    "data/supplied_binary_predictions.csv": (184, "7a8809010fa94345cd04787c826ef86ee5fd13cbf0bd95953e2220c3294a239a"),
}
CELL_IDS = [
    "a10-header", "a10-setup", "a10-terms-inference", "a10-load",
    "a10-task1-prompt", "a10-ols-function", "a10-task1-run",
    "a10-residual-figure", "a10-task1-save", "a10-task1-explain",
    "a10-terms-prediction", "a10-task2-contract", "a10-task2-values",
    "a10-availability-function", "a10-split-function", "a10-task2-run",
    "a10-task2-save", "a10-task2-explain", "a10-terms-evaluation",
    "a10-regression-metrics-function", "a10-candidates-function",
    "a10-validation-run", "a10-validation-save", "a10-freeze",
    "a10-final-test-run", "a10-final-test-save", "a10-binary-function",
    "a10-binary-run-save", "a10-task3-explain", "a10-final-verify",
]
PROTECTED_CELL_IDS = {
    "a10-header", "a10-setup", "a10-terms-inference", "a10-task1-prompt",
    "a10-terms-prediction", "a10-terms-evaluation", "a10-freeze",
    "a10-final-verify",
}
PROTECTED_FILE_SHA256 = {
    ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95",
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "PLATFORM_CHECK.md": "26fa8f87d119d95cc556197c9c9304ffa4d9a19c74fbf6d6bee6d00a73c93cf1",
    "README.md": "8f2a523f5fc000601c3421950be36974735b045ac819148353a21256856a893c",
    "requirements.txt": "4c6d9eaa5d730c7dfb71124d1576070dfabefe9162124c74162d4bb172c77984",
    "data/fixture.json": "aa50eeffc2b07c5d98cb56a0e3d18115909958f777899d5d403cf6323dd1de41",
    "data/mixing_runs.csv": "00b8a1ce84110f4a7fa85620742283c82a4b9d600dbe0ebea0d4721956938957",
    "data/batch_strength.csv": "f14faf7da64347dfc255aa84b14e79eef7f2d0de94b394c747323319d937baa3",
    "data/feature_availability.csv": "a47b8df048607045640b9a6785b038fe1c70036f58d5b61ed20ec98860b556da",
    "data/supplied_binary_predictions.csv": "7a8809010fa94345cd04787c826ef86ee5fd13cbf0bd95953e2220c3294a239a",
    "output/.gitkeep": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
PROTECTED_CELL_SHA256 = {
    "a10-header": "d0a854bb0eebce383b25c11f2d17151432d27928110bd461b2e038de3ef5343a",
    "a10-setup": "a5caad874818b2edf045770960e00c610ea6d29a2e50d3dd89c3d6328caf20e8",
    "a10-terms-inference": "4c167e07f0ae870026a9746b40138fe5c3d6a4e74dddd417580e0e39e096696e",
    "a10-task1-prompt": "855729ff10c4244d077be4152576d05e17c99d104bff7e896495bdde402c193b",
    "a10-terms-prediction": "86136b5c6caedbd549a7148563f05280a83a589a32dcfae7ce7e3ba8cafc0a95",
    "a10-terms-evaluation": "84adc778e76702fb952469be3601793888ac302a66fb0360898cd8f52b5752b7",
    "a10-freeze": "2f1c03daef8bdec283d35e485918e3a4ab55bf55eb1cbd10cb509f74301ddfc1",
    "a10-final-verify": "e401fcb4ea4be0881f2b381f662d7c4a45d2b9b494d986fe29baa551ef4d88a8",
}
MARKDOWN_IDS = {
    "a10-header", "a10-terms-inference", "a10-task1-prompt",
    "a10-task1-explain", "a10-terms-prediction", "a10-task2-contract",
    "a10-task2-explain", "a10-terms-evaluation", "a10-task3-explain",
}
STUDENT_CODE_IDS = {
    "a10-load", "a10-ols-function", "a10-task1-run", "a10-residual-figure",
    "a10-task1-save", "a10-task2-values", "a10-availability-function",
    "a10-split-function", "a10-task2-run", "a10-task2-save",
    "a10-regression-metrics-function", "a10-candidates-function",
    "a10-validation-run", "a10-validation-save", "a10-final-test-run",
    "a10-final-test-save", "a10-binary-function", "a10-binary-run-save",
}
STUDENT_MARKDOWN_IDS = {"a10-task1-explain", "a10-task2-contract", "a10-task2-explain", "a10-task3-explain"}
SIGNATURES = {
    "fit_bounded_ols": ["inference_table", "predictor_columns", "outcome_column"],
    "audit_feature_availability": ["candidate_table"],
    "build_chronological_splits": ["prediction_table", "validation_start", "test_start"],
    "regression_metrics": ["actual", "predicted"],
    "fit_prediction_candidates": ["train_table", "feature_columns", "target_column"],
    "choose_validation_winner": ["metrics_table", "metric_column"],
    "compute_binary_metrics": ["prediction_table", "actual_column", "prediction_columns"],
}
BASE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
    *FIXTURES,
}
ARTIFACTS = {
    "inference_summary.csv": (214, "36965b53df5133e3e05f86502d230ec9241b58e9ffd93163eba588385c9f3f48"),
    "inference_case_intervals.csv": (186, "345e0d3aefc422606fa9a9ee1b35a06bd7a9f9007873fc7b05162cb9ef3e0951"),
    "availability_decisions.csv": (251, "36042dc19dd45f75603f2fb2d5783b0a7750dad274a54bd39e8d21d5f5c2ac81"),
    "split_manifest.csv": (221, "2b0f3f57e323fa7bfe7a0703c671755ed7b009854236e62dd0c3459b1aa67b21"),
    "validation_metrics.csv": (106, "65b105be797b109c2031ccde552972320c1d08cb59174cde628a23c1879832dc"),
    "final_test_metrics.csv": (64, "ca1bd6d4320ed84cd2ca5befe97c3c0f238746452b648e64103522517b9a77ce"),
    "final_predictions.csv": (575, "60b7457821655c387b07694e18cad262a873c50bc69093a9638bd8ea99239a1d"),
    "binary_metrics.csv": (119, "25d7b50cdb8160f8e275812010a9a90b295d700b03591b3ce7bfd712483616fa"),
}


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def normalized_source(cell: dict) -> str:
    return source(cell).replace("\r\n", "\n").replace("\r", "\n")


def issue(surface: str, message: str, errors: list[str]) -> None:
    errors.append(f"[FIX] {surface}: {message}")


def static_contract(errors: list[str]) -> None:
    try:
        checker_text = Path(__file__).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        issue("integrity", f"cannot read checker source: {error}", errors)
        return
    lines = checker_text.splitlines()
    start = 1 if lines and lines[0] == "#!/usr/bin/env python3" else 0
    if lines[start:start + len(PEP_BLOCK)] != PEP_BLOCK:
        issue("integrity", "restore the exact PEP 723 Python and ordered dependency block", errors)
    try:
        requirement_lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        issue("integrity", f"cannot read requirements.txt: {error}", errors)
    else:
        if requirement_lines != PEP_DEPENDENCIES:
            issue("integrity", "PEP 723 dependencies must exactly match requirements.txt in order", errors)
    try:
        checker_tree = ast.parse(checker_text)
    except SyntaxError as error:
        issue("integrity", f"checker source has invalid Python syntax: {error}", errors)
        return
    imported_roots = set()
    for node in ast.walk(checker_tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    implementation_dependencies = {dependency.split("==", 1)[0].replace("-", "_") for dependency in PEP_DEPENDENCIES}
    if imported_roots & implementation_dependencies:
        issue("integrity", "checker implementation must remain standard-library-only", errors)


def integrity(errors: list[str]) -> None:
    expected_file_keys = {
        ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
        "requirements.txt", "data/fixture.json", "data/mixing_runs.csv",
        "data/batch_strength.csv", "data/feature_availability.csv",
        "data/supplied_binary_predictions.csv", "output/.gitkeep",
    }
    if set(PROTECTED_FILE_SHA256) != expected_file_keys:
        issue("integrity", "protected-file map keys differ", errors)
    if set(PROTECTED_CELL_SHA256) != PROTECTED_CELL_IDS:
        issue("integrity", "protected-cell map keys differ", errors)
    for label, mapping in (("file", PROTECTED_FILE_SHA256), ("cell", PROTECTED_CELL_SHA256)):
        if any(len(digest) != 64 or digest.lower() != digest or any(character not in "0123456789abcdef" for character in digest) for digest in mapping.values()):
            issue("integrity", f"protected {label} digests must be lowercase SHA-256", errors)
    for relative, expected in PROTECTED_FILE_SHA256.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or sha256(path.read_bytes()).hexdigest() != expected:
            issue("integrity", f"restore immutable learner file {relative}", errors)
    try:
        notebook = json.loads((ROOT / "assignment.ipynb").read_text(encoding="utf-8"))
        by_id = {cell.get("id"): cell for cell in notebook.get("cells", [])}
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        issue("integrity", f"cannot inspect protected notebook cells: {error}", errors)
        return
    for cell_id, expected in PROTECTED_CELL_SHA256.items():
        cell = by_id.get(cell_id)
        actual = sha256(normalized_source(cell).encode("utf-8")).hexdigest() if cell else None
        if actual != expected:
            issue("integrity", f"restore protected cell {cell_id}", errors)


def inventory(errors: list[str]) -> None:
    git_entry = ROOT / ".git"
    if git_entry.exists() or git_entry.is_symlink():
        if not git_entry.is_dir() or git_entry.is_symlink():
            issue("package", "top-level .git must be a genuine directory, not a file or symlink", errors)
    actual = set()
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if relative.parts[0] == ".git" or relative.parts[0] == "output":
            continue
        if relative.parts[0] == "_grader_selftest":
            continue
        if path.is_file() or path.is_symlink():
            actual.add(relative.as_posix())
    expected = BASE_FILES
    if actual != expected:
        issue("package", f"unexpected or missing files: {sorted(actual ^ expected)}", errors)
    for relative in actual:
        path = ROOT / relative
        if path.is_symlink():
            issue("package", f"symlinks are not accepted: {relative}", errors)


def fixtures(errors: list[str]) -> None:
    for relative, (size, digest) in FIXTURES.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            issue("fixtures", f"missing regular fixture {relative}", errors)
            continue
        raw = path.read_bytes()
        if len(raw) != size or sha256(raw).hexdigest() != digest:
            issue("fixtures", f"restore exact course fixture {relative}", errors)
        if not raw.endswith(b"\n") or b"\r" in raw:
            issue("fixtures", f"{relative} must use LF and a final newline", errors)
    try:
        manifest = json.loads((ROOT / "data/fixture.json").read_text(encoding="utf-8"))
        if manifest.get("fixture_id") != "a10-bounded-modeling-v1" or len(manifest.get("files", [])) != 4:
            issue("manifest", "fixture ID or file inventory differs", errors)
        for record in manifest.get("files", []):
            name = record.get("path", "")
            if not name or Path(name).name != name or "/" in name or "\\" in name:
                issue("manifest", f"unsafe fixture path {name!r}", errors)
    except Exception as error:
        issue("manifest", f"cannot parse fixture.json: {error}", errors)


def notebook_checks(errors: list[str]) -> None:
    try:
        notebook = json.loads((ROOT / "assignment.ipynb").read_text(encoding="utf-8"))
    except Exception as error:
        issue("notebook", f"invalid UTF-8 JSON: {error}", errors)
        return
    cells = notebook.get("cells", [])
    ids = [cell.get("id") for cell in cells]
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5:
        issue("notebook", "use notebook format 4.5", errors)
    if ids != CELL_IDS or len(ids) != len(set(ids)):
        issue("notebook", "restore the exact 30 cell IDs and order", errors)
        return
    if notebook.get("metadata", {}).get("kernelspec") != {"display_name": "Python 3", "language": "python", "name": "python3"}:
        issue("notebook", "restore the portable Python 3 kernelspec", errors)
    by_id = {cell["id"]: cell for cell in cells}
    for cell in cells:
        expected = "markdown" if cell["id"] in MARKDOWN_IDS else "code"
        if cell.get("cell_type") != expected:
            issue("notebook", f"wrong cell type for {cell['id']}", errors)
    student_code = "\n".join(source(by_id[cell_id]) for cell_id in CELL_IDS if cell_id in STUDENT_CODE_IDS)
    student_markdown = "\n".join(source(by_id[cell_id]) for cell_id in STUDENT_MARKDOWN_IDS)
    if "TODO" in student_code + student_markdown or "NotImplementedError" in student_code or "raise NotImplementedError" in student_code:
        issue("notebook", "complete every student scaffold and explanation", errors)
    try:
        tree = ast.parse(student_code)
    except SyntaxError as error:
        issue("notebook", f"student source has a syntax error: {error}", errors)
        return
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for name, arguments in SIGNATURES.items():
        node = functions.get(name)
        if node is None or [arg.arg for arg in node.args.args] != arguments:
            issue("functions", f"restore exact signature for {name}", errors)
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        issue("source", "student cells must use protected imports", errors)
    lowered = student_code.lower().replace(" ", "")
    forbidden = [
        "requests", "urlopen", "http://", "https://", "/content", "drive.mount",
        "files.upload", "!pip", "random.", "xgboost", "randomforest", "cross_val",
        "gridsearch", "ridge(", "lasso(", ".score(", "pvalues", ".aic", "add_constant",
    ]
    for fragment in forbidden:
        if fragment in lowered:
            issue("scope", f"remove out-of-scope source {fragment!r}", errors)
    for name, node in functions.items():
        fn_source = ast.unparse(node)
        if any(token in fn_source for token in ("read_csv", "to_csv", "savefig", "Path(")):
            issue("functions", f"{name} must not perform file or path I/O", errors)
    if "smf.ols" not in student_code or '" ~ "' not in student_code:
        issue("Task 1", "derive an argument-based formula and call smf.ols", errors)
    if "DummyRegressor" not in student_code or "StandardScaler" not in student_code or "LinearRegression" not in student_code or "Pipeline" not in student_code:
        issue("Task 3", "use exactly the supplied baseline and scale→linear Pipeline", errors)
    if 'kind="stable"' not in student_code and "kind='stable'" not in student_code:
        issue("Task 2", "use a stable chronological sort", errors)
    if "zero_division=0" not in student_code:
        issue("binary metrics", "use zero_division=0", errors)
    direct_predict_cells = [cell_id for cell_id in STUDENT_CODE_IDS if ".predict(" in source(by_id[cell_id])]
    if direct_predict_cells:
        issue("prediction gate", f"use record_predictions instead of direct predict in {direct_predict_cells}", errors)
    for cell_id in ("a10-task1-run", "a10-task2-run", "a10-validation-run", "a10-final-test-run", "a10-binary-run-save", "a10-final-verify"):
        cell = by_id[cell_id]
        if not cell.get("outputs") or cell.get("execution_count") is None:
            issue("visible output", f"run and retain output for {cell_id}", errors)
        for output in cell.get("outputs", []):
            text = json.dumps(output).lower()
            if output.get("output_type") == "error" or "traceback" in text:
                issue("visible output", f"remove error output from {cell_id}", errors)


def output_checks(errors: list[str]) -> None:
    output = ROOT / "output"
    if not output.is_dir() or output.is_symlink():
        issue("output", "restore the regular output directory", errors)
        return
    actual = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    expected = {".gitkeep", *ARTIFACTS, "inference_residuals.png"}
    if actual != expected:
        issue("output", f"required output inventory differs: {sorted(actual ^ expected)}", errors)
    for name, (size, digest) in ARTIFACTS.items():
        path = output / name
        if not path.is_file() or path.is_symlink():
            continue
        raw = path.read_bytes()
        if len(raw) != size or sha256(raw).hexdigest() != digest:
            issue("output", f"rerun to reproduce exact {name}", errors)
        if not raw.endswith(b"\n") or b"\r" in raw:
            issue("output", f"{name} must use LF and a final newline", errors)
    png = output / "inference_residuals.png"
    if png.is_file() and not png.is_symlink():
        raw = png.read_bytes()
        if not raw.startswith(b"\x89PNG\r\n\x1a\n") or len(raw) <= 8192:
            issue("output", "residual PNG is missing or trivial", errors)
        elif len(raw) < 24 or struct.unpack(">II", raw[16:24]) != (720, 480):
            issue("output", "residual PNG must be 720×480", errors)


def runtime(errors: list[str]) -> None:
    if sys.version_info[:3] != EXPECTED_PYTHON:
        issue("runtime", f"use CPython {'.'.join(map(str, EXPECTED_PYTHON))}", errors)
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            issue("runtime", f"install {distribution}=={expected}", errors)
        else:
            if actual != expected:
                issue("runtime", f"expected {distribution}=={expected}, found {actual}", errors)


def main() -> int:
    errors: list[str] = []
    static_contract(errors)
    inventory(errors)
    integrity(errors)
    runtime(errors)
    fixtures(errors)
    notebook_checks(errors)
    output_checks(errors)
    if errors:
        print("\n".join(errors))
        print(f"Readiness check found {len(errors)} action item(s).")
        return 1
    print("Assignment 10 readiness structure is complete.")
    print("Central grading assigns points; explanation quality is reviewed separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
