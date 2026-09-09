# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#   "ipykernel==6.29.5",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.3.3",
#   "pandas==3.0.5",
# ]
# ///

"""Independent central-grader reference for Assignment 08.

Production grading reads the five committed CSV artifacts directly. Notebook
execution and alternate-input checks remain optional release QA.
"""

from __future__ import annotations

import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROTECTED_FILE_SHA256 = {
    ".python-version": "a876e0b10411037a012498b9fe18d9bc1df32ed8b722a13564dc944ddcfd9135",
    "requirements.txt": "dba7ebcc237068a6bfd7c7035b2c8d67ed138deae244d4ec6ac2d4d1d3476e47",
    ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95",
    "README.md": "2219b14e8cb7ce296d78fe2803a4317b1a00959a4ba2b59112806aab6f78b19c",
    "PLATFORM_CHECK.md": "a21980799fba433311bbeb867e1fc4f3d64780545c5025f0163ea5a78f014e4c",
    "check_assignment.py": "d603a7b780d837f38a2ab2cd4fde5d08852c67586f0d5611c5cb58ec2428dd12",
    "data/fixture.json": "b2fee1c48fb678b81318d2f085c42e2f9b480bd6c4eed6f07ef118b9bfd70860",
    "data/support_requests.csv": "a9136161332c5da9f8f1251d869bbd014ed762751675fb757f81a79cff5352d6",
}
ARTIFACT_NAMES = {"center_count_summary.csv", "center_summary.csv", "requests_with_context.csv", "center_channel_summary.csv", "mean_resolution_pivot.csv"}
REQUIRED_CONTEXT_ENV = {
    "assignment": "ASSIGNMENT",
    "submission": "SUBMISSION_TAG",
    "commit": "COMMIT_URL",
    "release": "RELEASE_URL",
}


class InfrastructureError(RuntimeError):
    """Raised when the runner contract is unavailable or grading cannot finish."""


def _context() -> dict[str, str]:
    context: dict[str, str] = {}
    missing: list[str] = []
    for field, environment_name in REQUIRED_CONTEXT_ENV.items():
        value = os.environ.get(environment_name, "").strip()
        if not value:
            missing.append(environment_name)
        context[field] = value
    if missing:
        raise InfrastructureError(
            "missing required grading context: " + ", ".join(missing)
        )
    context["review"] = os.environ.get("REVIEW_URL", "").strip() or context["commit"]
    context["datetime"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return context


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _result_test(name: str, maximum: int, error: Exception | None) -> dict:
    passed = error is None
    detail = "all automated checks passed" if passed else str(error)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return {
        "test-name": name,
        "passed": passed,
        "score": maximum if passed else 0,
        "max-score": maximum,
    }


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    try:
        result = grade_submission(target)
        Path("result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    except Exception as error:
        print(f"Grader infrastructure failure: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


def _csv(root: Path, name: str) -> pd.DataFrame:
    path = root / "output" / name
    _assert(path.is_file() and not path.is_symlink(), f"missing regular artifact: {name}")
    try:
        return pd.read_csv(path, keep_default_na=True)
    except Exception as error:
        raise AssertionError(f"invalid CSV {name}: {error}") from error


def _same(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str], label: str) -> None:
    _assert(actual.columns.tolist() == expected.columns.tolist(), f"{label} columns differ")
    _assert(not actual.duplicated(keys).any(), f"{label} has duplicate row identities")
    _assert(set(map(tuple, actual[keys].astype(str).to_numpy())) == set(map(tuple, expected[keys].astype(str).to_numpy())), f"{label} row identities differ")
    actual = actual.sort_values(keys, kind="stable").reset_index(drop=True)
    expected = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    for column in actual.columns:
        if column in keys:
            continue
        left, right = actual[column], expected[column]
        if pd.api.types.is_numeric_dtype(right):
            _assert(np.allclose(pd.to_numeric(left), right, rtol=1e-7, atol=1e-8, equal_nan=True), f"{label} values differ: {column}")
        else:
            _assert(left.fillna("<missing>").astype(str).equals(right.fillna("<missing>").astype(str)), f"{label} values differ: {column}")


def _fixture(root: Path) -> pd.DataFrame:
    path = root / "data" / "support_requests.csv"
    _assert(path.is_file() and sha256(path.read_bytes()).hexdigest() == PROTECTED_FILE_SHA256["data/support_requests.csv"], "fixture changed")
    return pd.read_csv(path, dtype={"request_id": "string", "center": "string", "agent_id": "string", "channel": "string", "resolution_minutes": "int64", "satisfaction_score": "Int64"})


def _artifact_inventory(root: Path) -> None:
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "missing regular output directory")
    actual = {p.name for p in output.iterdir() if p.is_file() or p.is_symlink()}
    _assert(ARTIFACT_NAMES | {".gitkeep"} <= actual, "required output artifacts are missing")


def _task1_artifacts(root: Path, source: pd.DataFrame) -> None:
    expected = source.groupby("center", sort=True, dropna=True).agg(request_count=("request_id", "size"), satisfaction_count=("satisfaction_score", "count"), unique_agent_count=("agent_id", "nunique")).reset_index()
    _same(_csv(root, "center_count_summary.csv"), expected, ["center"], "Task 1")


def _task2_artifacts(root: Path, source: pd.DataFrame) -> None:
    summary = source.groupby("center", sort=True, dropna=True).agg(request_count=("request_id", "size"), satisfaction_count=("satisfaction_score", "count"), unique_agent_count=("agent_id", "nunique"), total_resolution_minutes=("resolution_minutes", "sum"), mean_resolution_minutes=("resolution_minutes", "mean")).reset_index()
    _same(_csv(root, "center_summary.csv"), summary, ["center"], "center summary")
    means = source.groupby("center")["resolution_minutes"].transform("mean")
    context = source.copy(); context["center_mean_resolution_minutes"] = means; context["difference_from_center_mean"] = source["resolution_minutes"] - means
    _same(_csv(root, "requests_with_context.csv"), context, ["request_id"], "context")
    two = source.groupby(["center", "channel"], sort=True, dropna=True).agg(request_count=("request_id", "size"), mean_resolution_minutes=("resolution_minutes", "mean")).reset_index()
    _same(_csv(root, "center_channel_summary.csv"), two, ["center", "channel"], "two-key summary")


def _task3_artifacts(root: Path, source: pd.DataFrame) -> None:
    expected = pd.pivot_table(source, index="center", columns="channel", values="resolution_minutes", aggfunc="mean", sort=True, dropna=True).reindex(columns=["Email", "Phone", "Chat"]).reset_index()
    _same(_csv(root, "mean_resolution_pivot.csv"), expected, ["center"], "pivot")


def grade_submission(submission_root: str | Path) -> dict:
    """Grade only committed, trusted CSV artifacts; never execute student code."""
    context = _context(); root = Path(submission_root).resolve()
    specs = (("Fixture integrity", 10, lambda s: _fixture(root)), ("Task 1 count semantics", 25, lambda s: _task1_artifacts(root, s)), ("Task 2 grouped and aligned results", 40, lambda s: _task2_artifacts(root, s)), ("Task 3 pivot equivalence", 20, lambda s: _task3_artifacts(root, s)), ("Visible artifact inventory", 5, lambda s: _artifact_inventory(root)))
    source: pd.DataFrame | None = None; tests = []
    for name, maximum, check in specs:
        try:
            if source is None: source = _fixture(root)
            check(source)
        except Exception as error:
            tests.append(_result_test(name, maximum, error))
        else:
            tests.append(_result_test(name, maximum, None))
    return {"schema": "datasci217/grading-result/v1", **context, "score": sum(t["score"] for t in tests), "max-score": 100, "tests": tests}


if __name__ == "__main__":
    raise SystemExit(main())
