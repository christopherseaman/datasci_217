"""Public, artifact-only checks for Assignment 08."""
from __future__ import annotations
import csv
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROTECTED = {".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b", "requirements.txt": "90933f178a0a459399ff6696e8fe9407463cc65bbffd567f3e7b44cc9230ee21", ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95", "README.md": "b6be8a99c77cb4b0651552d5192c270bcc4e023c95ca642ca26620e85c0f8a00", "PLATFORM_CHECK.md": "c33b3fad9a28183df02ea4a911a890ea6ae18a1089c6fae4b12c39dc3540ba02", "data/fixture.json": "b2fee1c48fb678b81318d2f085c42e2f9b480bd6c4eed6f07ef118b9bfd70860", "data/support_requests.csv": "a9136161332c5da9f8f1251d869bbd014ed762751675fb757f81a79cff5352d6"}
ARTIFACTS = {"center_count_summary.csv": (3, ["center", "request_count", "satisfaction_count", "unique_agent_count"]), "center_summary.csv": (3, ["center", "request_count", "satisfaction_count", "unique_agent_count", "total_resolution_minutes", "mean_resolution_minutes"]), "requests_with_context.csv": (15, ["request_id", "center", "agent_id", "channel", "resolution_minutes", "satisfaction_score", "center_mean_resolution_minutes", "difference_from_center_mean"]), "center_channel_summary.csv": (8, ["center", "channel", "request_count", "mean_resolution_minutes"]), "mean_resolution_pivot.csv": (3, ["center", "Email", "Phone", "Chat"])}
def check_environment_and_protected_files() -> None:
    for name, digest in PROTECTED.items():
        path=ROOT/name
        assert path.is_file() and sha256(path.read_bytes()).hexdigest()==digest, f"Restore protected {name}."
def check_artifacts() -> None:
    output=ROOT/"output"; assert output.is_dir() and not output.is_symlink(), "Missing regular output/ directory."
    assert set(ARTIFACTS)|{".gitkeep"} <= {p.name for p in output.iterdir() if p.is_file() or p.is_symlink()}, "Required CSV artifacts or output/.gitkeep are missing."
    for name, (rows, columns) in ARTIFACTS.items():
        path=output/name; assert path.is_file() and not path.is_symlink(), f"output/{name} must be regular."
        with path.open(newline="", encoding="utf-8") as handle: parsed=list(csv.reader(handle))
        assert parsed and parsed[0]==columns and len(parsed)-1==rows, f"Wrong schema or row count in output/{name}."
    source = _fixture(ROOT)
    _task1_artifacts(ROOT, source)
    _task2_artifacts(ROOT, source)
    _task3_artifacts(ROOT, source)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
    _assert(path.is_file() and sha256(path.read_bytes()).hexdigest() == PROTECTED["data/support_requests.csv"], "fixture changed")
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


def main() -> int:
    failures=[]
    for label, check in (("environment and protected files",check_environment_and_protected_files),("five generated artifacts",check_artifacts)):
        try: check()
        except Exception as error: failures.append(f"[FIX] {label}: {error}")
    if failures: print("\n".join(failures)); return 1
    print("All public checks passed. The five committed artifacts are ready for grading."); return 0
if __name__=="__main__": raise SystemExit(main())
