"""Materialize fixtures and validate the Assignment 03 public checker."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile


ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT_DIR))

from _public_checks import EXPECTED_ENVIRONMENT_OUTPUT, run_public_checks  # noqa: E402


RUNTIME_CHECK = "runtime records, supplied files, and fresh probe"
PIPELINE_CHECK = "safe pipeline structure and base/alternate execution"
STRUCTURE_CHECK = "student code structure and direct-operation boundaries"
METADATA_CHECK = "array metadata and basic selection"
VIEW_CHECK = "view/copy relationship and nonmutation"
VECTOR_CHECK = "vector mask, selection, arithmetic, and scalar broadcast"
REDUCTION_CHECK = "reductions, shapes, reshape, transpose, and whole count"
IMPORT_CHECK = "quiet artifact-free imports"
DATAFLOW_CHECK = "driver helper dataflow, call order, and different CWD"
OUTPUT_CHECK = "exact fresh driver and loader output"

CORRECT_PIPELINE = '''# Bounded terminal pipeline

```bash
head -n 3 observations.csv > output/head_preview.txt
tail -n 2 observations.csv > output/tail_preview.txt
tail -n +2 observations.csv | cut -d',' -f1 | sort | uniq -c > output/site_counts.txt
wc -l output/site_counts.txt > output/site_count_lines.txt
```
'''

CORRECT_ARRAY = '''"""Complete the required ordinary-ndarray operations."""

import numpy as np


def create_and_describe(values):
    """Create one float64 array and return its metadata."""
    array = np.array(values, dtype=np.float64)
    return {
        "array": array,
        "shape": array.shape,
        "ndim": array.ndim,
        "size": array.size,
        "dtype": array.dtype,
    }


def select_parts(values):
    """Return the required basic 2D selections."""
    return {
        "first_value": values[0, 0],
        "second_row": values[1],
        "second_column": values[:, 1],
        "top_left_block": values[:2, :2],
    }


def view_and_copy(values):
    """Return the same basic slice as a view and an explicit copy."""
    middle_view = values[1:3]
    middle_copy = values[1:3].copy()
    return {"view": middle_view, "copy": middle_copy}


def vector_operations(values, baseline, threshold, offset):
    """Return one mask, selection, difference, and scalar adjustment."""
    mask = values >= threshold
    selected = values[mask]
    difference = values - baseline
    adjusted = values + offset
    return {
        "mask": mask,
        "selected": selected,
        "difference": difference,
        "adjusted": adjusted,
    }


def reduction_summary(values):
    """Return whole-array and axis means with result shapes."""
    overall_mean = np.mean(values)
    column_means = np.mean(values, axis=0)
    row_means = np.mean(values, axis=1)
    return {
        "overall_mean": overall_mean,
        "column_means": column_means,
        "column_means_shape": column_means.shape,
        "row_means": row_means,
        "row_means_shape": row_means.shape,
    }


def reshape_and_transpose(values, rows, columns):
    """Return one compatible reshape and its transpose."""
    grid = np.reshape(values, (rows, columns))
    transposed = grid.T
    return {
        "grid": grid,
        "grid_shape": grid.shape,
        "transposed": transposed,
        "transposed_shape": transposed.shape,
    }


def count_at_or_above(values, threshold):
    """Reshape to 1D and return the whole mask count."""
    flattened = np.reshape(values, values.size)
    return np.sum(flattened >= threshold)
'''

CORRECT_ANALYSIS = '''"""Run one import-safe NumPy analysis over the supplied fixture."""

from array_analysis import count_at_or_above, create_and_describe, reduction_summary
from data_loader import load_measurements


def main():
    """Load the fixture and print its deterministic summary."""
    measurements = load_measurements("observations.csv")
    description = create_and_describe(measurements)
    summary = reduction_summary(measurements)
    review_count = count_at_or_above(measurements, 30)

    print(f'Measurements shape: {description["shape"]}')
    print(f'Measurements dtype: {description["dtype"]}')
    print(f'Overall mean: {summary["overall_mean"]:.1f}')
    print(f'Column means: {summary["column_means"]}')
    print(f'Column means shape: {summary["column_means_shape"]}')
    print(f'Row means: {summary["row_means"]}')
    print(f'Row means shape: {summary["row_means_shape"]}')
    print(f"Values at or above 30: {review_count}")


if __name__ == "__main__":
    main()
'''


def write_fixture(root: Path, overrides: dict[str, str] | None = None) -> None:
    for filename in (
        "README.md",
        "PLATFORM_CHECK.md",
        ".gitignore",
        "observations.csv",
        "environment_check.py",
        "data_loader.py",
    ):
        shutil.copy2(ASSIGNMENT_DIR / filename, root / filename)
    sources = {
        ".python-version": "3.12.13\n",
        "requirements.txt": "numpy==2.0.2\n",
        "PIPELINE.md": CORRECT_PIPELINE,
        "array_analysis.py": CORRECT_ARRAY,
        "analysis.py": CORRECT_ANALYSIS,
    }
    if overrides:
        sources.update(overrides)
    for filename, source in sources.items():
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    (root / "output").mkdir(exist_ok=True)
    (root / "output/environment_check.txt").write_text(
        EXPECTED_ENVIRONMENT_OUTPUT,
        encoding="utf-8",
    )


def failures_for(root: Path) -> dict[str, str]:
    return {
        name: error
        for name, error in run_public_checks(root)
        if error is not None
    }


def assert_rejected(
    case: str,
    overrides: dict[str, str],
    expected_check: str,
    *,
    prepare=None,
    verify=None,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"ds217-a03-{case}-") as temporary:
        root = Path(temporary)
        write_fixture(root, overrides)
        if prepare is not None:
            prepare(root)
        failures = failures_for(root)
        if expected_check not in failures:
            raise AssertionError(
                f"{case}: expected `{expected_check}` to fail, got {sorted(failures)}"
            )
        if verify is not None:
            verify(root)
        print(f"[PASS] rejected {case} fixture through {expected_check}")


with tempfile.TemporaryDirectory(prefix="ds217-a03-correct-") as temporary:
    correct_root = Path(temporary)
    write_fixture(correct_root)
    correct_failures = failures_for(correct_root)
    if correct_failures:
        raise AssertionError(f"correct fixture failed: {correct_failures}")
    print("[PASS] accepted correct fixture through all ten public checks")

    for filename in ("_public_checks.py", "check_assignment.py"):
        shutil.copy2(ASSIGNMENT_DIR / filename, correct_root / filename)
    checker = subprocess.run(
        [sys.executable, "-B", "check_assignment.py"],
        cwd=correct_root,
        capture_output=True,
        text=True,
        timeout=60,
        env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if checker.returncode != 0 or "All public checks passed." not in checker.stdout:
        raise AssertionError(
            f"standard-library+NumPy checker rejected correct fixture: {checker.stdout}{checker.stderr}"
        )
    print("[PASS] local checker accepted correct fixture")

    if "--pytest" in sys.argv:
        shutil.copy2(ASSIGNMENT_DIR / "test_assignment.py", correct_root / "test_assignment.py")
        pytest_result = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "test_assignment.py", "-q"],
            cwd=correct_root,
            text=True,
            timeout=120,
            env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if pytest_result.returncode != 0:
            raise AssertionError("managed pytest rejected the correct fixture")
        print("[PASS] managed pytest accepted all ten public facade tests")


starter_failures = failures_for(ASSIGNMENT_DIR)
if set(starter_failures) != {
    RUNTIME_CHECK,
    PIPELINE_CHECK,
    STRUCTURE_CHECK,
    METADATA_CHECK,
    VIEW_CHECK,
    VECTOR_CHECK,
    REDUCTION_CHECK,
    DATAFLOW_CHECK,
    OUTPUT_CHECK,
}:
    raise AssertionError(f"starter failure surface changed: {starter_failures}")
print("[PASS] starter remains incomplete with nine actionable failures and a safe import")


assert_rejected("wrong_version", {".python-version": "3.12.12\n"}, RUNTIME_CHECK)
assert_rejected("wrong_dependency", {"requirements.txt": "numpy>=2.0\n"}, RUNTIME_CHECK)
assert_rejected(
    "edited_probe",
    {"environment_check.py": (ASSIGNMENT_DIR / "environment_check.py").read_text().replace("NumPy:", "numpy:")},
    RUNTIME_CHECK,
)


def stale_probe(root: Path) -> None:
    (root / "output/environment_check.txt").write_text("Python: 3.12.13\nNumPy: 2.0.1\n", encoding="utf-8")


assert_rejected("stale_probe", {}, RUNTIME_CHECK, prepare=stale_probe)


def track_environment(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".venv").mkdir()
    (root / ".venv/tracked.txt").write_text("must not be submitted\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", ".venv/tracked.txt"], cwd=root, check=True)


assert_rejected("committed_venv", {}, RUNTIME_CHECK, prepare=track_environment)

assert_rejected("blank_pipeline", {"PIPELINE.md": "# Pipeline\n\n```bash\n\n```\n"}, PIPELINE_CHECK)
assert_rejected(
    "hardcoded_pipeline",
    {"PIPELINE.md": CORRECT_PIPELINE.replace("tail -n +2 observations.csv | cut -d',' -f1 | sort | uniq -c", "printf '3 north\\n2 south\\n1 west\\n'")},
    PIPELINE_CHECK,
)
assert_rejected(
    "forbidden_tool_pipeline",
    {"PIPELINE.md": CORRECT_PIPELINE.replace("cut -d',' -f1", "awk -F',' '{print $1}'")},
    PIPELINE_CHECK,
)
assert_rejected(
    "append_pipeline",
    {"PIPELINE.md": CORRECT_PIPELINE.replace("> output/head_preview.txt", ">> output/head_preview.txt")},
    PIPELINE_CHECK,
)
assert_rejected(
    "missing_sort_pipeline",
    {"PIPELINE.md": CORRECT_PIPELINE.replace(" | sort | uniq -c", " | uniq -c")},
    PIPELINE_CHECK,
)
assert_rejected(
    "header_in_counts_pipeline",
    {"PIPELINE.md": CORRECT_PIPELINE.replace("tail -n +2 observations.csv | ", "")},
    PIPELINE_CHECK,
)


def no_malicious_effect(root: Path) -> None:
    if (root / "pwned.txt").exists():
        raise AssertionError("malicious pipeline text was executed")


assert_rejected(
    "malicious_pipeline",
    {"PIPELINE.md": CORRECT_PIPELINE.replace("head -n 3 observations.csv", "head -n 3 observations.csv; touch pwned.txt")},
    PIPELINE_CHECK,
    verify=no_malicious_effect,
)
assert_rejected(
    "base_only_pipeline",
    {"PIPELINE.md": CORRECT_PIPELINE.replace("sort | uniq -c", "grep -E 'north|south|west'")},
    PIPELINE_CHECK,
)

PARTIAL_ARRAY = '''import numpy as np

def create_and_describe(values):
    """Return nothing yet."""
    return None
'''
assert_rejected("partial_array", {"array_analysis.py": PARTIAL_ARRAY}, STRUCTURE_CHECK)

REPLACED_ARRAY_MODULE_DOCSTRING = CORRECT_ARRAY.replace(
    '"""Complete the required ordinary-ndarray operations."""',
    '"""A replacement module description."""',
    1,
)
assert_rejected(
    "replaced_array_module_docstring",
    {"array_analysis.py": REPLACED_ARRAY_MODULE_DOCSTRING},
    STRUCTURE_CHECK,
)

REPLACED_ANALYSIS_MODULE_DOCSTRING = CORRECT_ANALYSIS.replace(
    '"""Run one import-safe NumPy analysis over the supplied fixture."""',
    '"""A replacement driver description."""',
    1,
)
assert_rejected(
    "replaced_analysis_module_docstring",
    {"analysis.py": REPLACED_ANALYSIS_MODULE_DOCSTRING},
    STRUCTURE_CHECK,
)

HARDCODED_ARRAY = CORRECT_ARRAY.replace(
    "array = np.array(values, dtype=np.float64)",
    "array = np.array([[10, 20], [30, 40]], dtype=np.float64)",
)
assert_rejected("hardcoded_arrays", {"array_analysis.py": HARDCODED_ARRAY}, STRUCTURE_CHECK)

WRONG_DTYPE = CORRECT_ARRAY.replace("dtype=np.float64", "dtype=np.float32", 1)
assert_rejected("wrong_dtype", {"array_analysis.py": WRONG_DTYPE}, STRUCTURE_CHECK)

WRONG_METADATA = CORRECT_ARRAY.replace('"shape": array.shape', '"shape": (6, 2)', 1)
assert_rejected("hardcoded_shape_metadata", {"array_analysis.py": WRONG_METADATA}, METADATA_CHECK)

CONSTANT_FIRST_SELECTION = CORRECT_ARRAY.replace(
    '"first_value": values[0, 0]',
    '"first_value": -4.5',
)
assert_rejected(
    "constant_first_selection",
    {"array_analysis.py": CONSTANT_FIRST_SELECTION},
    STRUCTURE_CHECK,
)

LOOP_ARRAY = CORRECT_ARRAY.replace(
    "    array = np.array(values, dtype=np.float64)",
    "    for value in values:\n        pass\n    array = np.array(values, dtype=np.float64)",
    1,
)
assert_rejected("student_loop", {"array_analysis.py": LOOP_ARRAY}, STRUCTURE_CHECK)

COMPREHENSION_ARRAY = CORRECT_ARRAY.replace(
    "    array = np.array(values, dtype=np.float64)",
    "    copied = [value for value in values]\n    array = np.array(copied, dtype=np.float64)",
    1,
)
assert_rejected("student_comprehension", {"array_analysis.py": COMPREHENSION_ARRAY}, STRUCTURE_CHECK)

VIEW_AS_COPY = CORRECT_ARRAY.replace("middle_view = values[1:3]", "middle_view = values[1:3].copy()")
assert_rejected("view_as_copy", {"array_analysis.py": VIEW_AS_COPY}, STRUCTURE_CHECK)

SHARING_COPY = CORRECT_ARRAY.replace("middle_copy = values[1:3].copy()", "middle_copy = values[1:3]")
assert_rejected("sharing_copy", {"array_analysis.py": SHARING_COPY}, STRUCTURE_CHECK)

MUTATING_VIEW = CORRECT_ARRAY.replace(
    "    middle_copy = values[1:3].copy()",
    "    middle_copy = values[1:3].copy()\n    middle_view[0] = 999",
)
assert_rejected("mutation_during_call", {"array_analysis.py": MUTATING_VIEW}, STRUCTURE_CHECK)

WRONG_MASK = CORRECT_ARRAY.replace("mask = values >= threshold", "mask = values > threshold")
assert_rejected("wrong_mask_boundary", {"array_analysis.py": WRONG_MASK}, VECTOR_CHECK)

REVERSED_DIFFERENCE = CORRECT_ARRAY.replace("difference = values - baseline", "difference = baseline - values")
assert_rejected("reversed_difference", {"array_analysis.py": REVERSED_DIFFERENCE}, VECTOR_CHECK)

TWO_D_BROADCAST = CORRECT_ARRAY.replace("adjusted = values + offset", "adjusted = values + np.reshape(offset, (1, 1))")
assert_rejected("multidimensional_broadcast", {"array_analysis.py": TWO_D_BROADCAST}, STRUCTURE_CHECK)

SWAPPED_AXES = CORRECT_ARRAY.replace(
    "column_means = np.mean(values, axis=0)\n    row_means = np.mean(values, axis=1)",
    "column_means = np.mean(values, axis=1)\n    row_means = np.mean(values, axis=0)",
)
assert_rejected("swapped_axes", {"array_analysis.py": SWAPPED_AXES}, STRUCTURE_CHECK)

WHOLE_AS_AXIS = CORRECT_ARRAY.replace("overall_mean = np.mean(values)", "overall_mean = np.mean(values, axis=0)")
assert_rejected("whole_mean_as_axis", {"array_analysis.py": WHOLE_AS_AXIS}, STRUCTURE_CHECK)

HARDCODED_REDUCTION_SHAPES = CORRECT_ARRAY.replace(
    '"column_means_shape": column_means.shape',
    '"column_means_shape": (2,)',
).replace(
    '"row_means_shape": row_means.shape',
    '"row_means_shape": (6,)',
)
assert_rejected("hardcoded_reduction_shapes", {"array_analysis.py": HARDCODED_REDUCTION_SHAPES}, REDUCTION_CHECK)

BAD_RESHAPE = CORRECT_ARRAY.replace("np.reshape(values, (rows, columns))", "np.reshape(values, (columns, rows))", 1)
assert_rejected("bad_reshape", {"array_analysis.py": BAD_RESHAPE}, STRUCTURE_CHECK)

COUNT_WITHOUT_RESHAPE = CORRECT_ARRAY.replace(
    "    flattened = np.reshape(values, values.size)\n    return np.sum(flattened >= threshold)",
    "    return np.sum(values >= threshold)",
)
assert_rejected("count_without_1d_reshape", {"array_analysis.py": COUNT_WITHOUT_RESHAPE}, STRUCTURE_CHECK)

COUNT_BY_AXIS = CORRECT_ARRAY.replace(
    "return np.sum(flattened >= threshold)",
    "return np.sum(flattened >= threshold, axis=0)",
)
assert_rejected("count_by_axis", {"array_analysis.py": COUNT_BY_AXIS}, STRUCTURE_CHECK)

HARDCODED_DRIVER = '''"""Run one import-safe NumPy analysis over the supplied fixture."""

from array_analysis import count_at_or_above, create_and_describe, reduction_summary
from data_loader import load_measurements

def main():
    """Print memorized fixture output."""
    print("Measurements shape: (6, 2)")
    print("Measurements dtype: float64")
    print("Overall mean: 25.0")
    print("Column means: [20. 30.]")
    print("Column means shape: (2,)")
    print("Row means: [15. 25. 35. 15. 25. 35.]")
    print("Row means shape: (6,)")
    print("Values at or above 30: 6")

if __name__ == "__main__":
    main()
'''
assert_rejected("hardcoded_driver", {"analysis.py": HARDCODED_DRIVER}, STRUCTURE_CHECK)

IGNORED_RESULTS_DRIVER = CORRECT_ANALYSIS.replace(
    '''    print(f'Measurements shape: {description["shape"]}')
    print(f'Measurements dtype: {description["dtype"]}')
    print(f'Overall mean: {summary["overall_mean"]:.1f}')
    print(f'Column means: {summary["column_means"]}')
    print(f'Column means shape: {summary["column_means_shape"]}')
    print(f'Row means: {summary["row_means"]}')
    print(f'Row means shape: {summary["row_means_shape"]}')
    print(f"Values at or above 30: {review_count}")''',
    '''    print("Measurements shape: (6, 2)")
    print("Measurements dtype: float64")
    print("Overall mean: 25.0")
    print("Column means: [20. 30.]")
    print("Column means shape: (2,)")
    print("Row means: [15. 25. 35. 15. 25. 35.]")
    print("Row means shape: (6,)")
    print("Values at or above 30: 6")''',
)
assert_rejected("ignored_helper_results", {"analysis.py": IGNORED_RESULTS_DRIVER}, DATAFLOW_CHECK)

SHAPE_GATED_DRIVER = CORRECT_ANALYSIS.replace(
    '''    print(f'Measurements shape: {description["shape"]}')
    print(f'Measurements dtype: {description["dtype"]}')
    print(f'Overall mean: {summary["overall_mean"]:.1f}')
    print(f'Column means: {summary["column_means"]}')
    print(f'Column means shape: {summary["column_means_shape"]}')
    print(f'Row means: {summary["row_means"]}')
    print(f'Row means shape: {summary["row_means_shape"]}')
    print(f"Values at or above 30: {review_count}")''',
    '''    known_shape = description["shape"] == (6, 2)
    print("Measurements shape: (6, 2)" if known_shape else f'Measurements shape: {description["shape"]}')
    print("Measurements dtype: float64" if known_shape else f'Measurements dtype: {description["dtype"]}')
    print("Overall mean: 25.0" if known_shape else f'Overall mean: {summary["overall_mean"]:.1f}')
    print("Column means: [20. 30.]" if known_shape else f'Column means: {summary["column_means"]}')
    print("Column means shape: (2,)" if known_shape else f'Column means shape: {summary["column_means_shape"]}')
    print("Row means: [15. 25. 35. 15. 25. 35.]" if known_shape else f'Row means: {summary["row_means"]}')
    print("Row means shape: (6,)" if known_shape else f'Row means shape: {summary["row_means_shape"]}')
    print("Values at or above 30: 6" if known_shape else f"Values at or above 30: {review_count}")''',
)
assert_rejected(
    "shape_gated_driver",
    {"analysis.py": SHAPE_GATED_DRIVER},
    DATAFLOW_CHECK,
)

WRONG_CALL_ORDER = CORRECT_ANALYSIS.replace(
    "    description = create_and_describe(measurements)\n    summary = reduction_summary(measurements)",
    "    summary = reduction_summary(measurements)\n    description = create_and_describe(measurements)",
)
assert_rejected("wrong_driver_call_order", {"analysis.py": WRONG_CALL_ORDER}, STRUCTURE_CHECK)

TOP_LEVEL_DRIVER = CORRECT_ANALYSIS.replace(
    '\n\nif __name__ == "__main__":',
    '\n\nmain()\n\nif __name__ == "__main__":',
)
assert_rejected("top_level_driver", {"analysis.py": TOP_LEVEL_DRIVER}, STRUCTURE_CHECK)

IMPORT_ARTIFACT = CORRECT_ANALYSIS.replace(
    "from data_loader import load_measurements\n",
    "from data_loader import load_measurements\nfrom pathlib import Path\n\nPath(__file__).with_name('imported.txt').write_text('artifact')\n",
)
assert_rejected("import_artifact", {"analysis.py": IMPORT_ARTIFACT}, IMPORT_CHECK)

BROKEN_CWD_DRIVER = CORRECT_ANALYSIS.replace(
    'load_measurements("observations.csv")',
    'load_measurements("03/assignment/observations.csv")',
)
assert_rejected("broken_working_directory", {"analysis.py": BROKEN_CWD_DRIVER}, DATAFLOW_CHECK)

WRONG_LABEL_DRIVER = CORRECT_ANALYSIS.replace("Overall mean:", "Overall average:")
assert_rejected("wrong_exact_output", {"analysis.py": WRONG_LABEL_DRIVER}, DATAFLOW_CHECK)

print("All Assignment 03 grader self-tests passed.")
