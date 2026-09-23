"""Independent central-grader reference for Assignment 08.

Production grading reads the five committed CSV artifacts directly. Notebook
execution and alternate-input checks remain optional release QA.
"""

from __future__ import annotations

import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


REFERENCE_ROOT = Path(__file__).resolve().parent


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _result_test(name: str, maximum: int, error: Exception | None) -> dict:
    passed = error is None
    detail = "all automated checks passed" if passed else str(error)
    return {
        "test-name": name,
        "passed": passed,
        "score": maximum if passed else 0,
        "max-score": maximum,
        "detail": detail,
    }


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
    path = REFERENCE_ROOT / "data" / "support_requests.csv"
    _assert(path.is_file() and not path.is_symlink(), "trusted fixture is missing")
    return pd.read_csv(path, dtype={"request_id": "string", "center": "string", "agent_id": "string", "channel": "string", "resolution_minutes": "int64", "satisfaction_score": "Int64"})


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
    root = Path(submission_root).resolve()
    specs = (("Task 1 count semantics", 29, lambda s: _task1_artifacts(root, s)), ("Task 2 grouped and aligned results", 47, lambda s: _task2_artifacts(root, s)), ("Task 3 pivot equivalence", 24, lambda s: _task3_artifacts(root, s)))
    source: pd.DataFrame | None = None; tests = []
    for name, maximum, check in specs:
        try:
            if source is None: source = _fixture(root)
            check(source)
        except Exception as error:
            tests.append(_result_test(name, maximum, error))
        else:
            tests.append(_result_test(name, maximum, None))
    return {"schema": "datasci217/grading-result/v1", "score": sum(t["score"] for t in tests), "max-score": 100, "tests": tests}

def _format_result(result: dict) -> str:
    return "\n".join(
        f"[{'PASS' if test['passed'] else 'FAIL'}] {test['test-name']}: "
        f"{test['score']}/{test['max-score']}"
        + (f" ({test['detail']})" if test.get("detail") else "")
        for test in result["tests"]
    ) + f"\nScore: {result['score']}/{result['max-score']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade committed assignment artifacts without executing submission code.")
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    result = grade_submission(args.submission_dir)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(_format_result(result))
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
