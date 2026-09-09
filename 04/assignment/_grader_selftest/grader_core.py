"""Instructor-owned grader core for Assignment 04.

Production grading reads the two committed CSV artifacts directly. Each artifact milestone is graded independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


BUNDLE_DIR = Path(__file__).resolve().parent
PROTECTED_HASHES_PATH = BUNDLE_DIR / "protected_files.json"
REQUIRED_COPY_PATHS = (
    ".gitignore",
    ".python-version",
    "PLATFORM_CHECK.md",
    "README.md",
    "assignment.ipynb",
    "check_assignment.py",
    "requirements.txt",
    "data/fixture.json",
    "data/purchases.csv",
    ".github/test/requirements.txt",
    ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
)


@dataclass(frozen=True)
class GradeTest:
    name: str
    max_score: int
    passed: bool
    detail: str

    @property
    def score(self) -> int:
        return self.max_score if self.passed else 0


def _check_protected_files(submission: Path) -> str | None:
    try:
        protected = json.loads(PROTECTED_HASHES_PATH.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Could not load instructor protected-file manifest: {error}") from error

    errors = []
    for relative, expected_hash in protected.items():
        path = submission / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != expected_hash:
            errors.append(f"edited {relative}")
    return "; ".join(errors) or None


def _check_submission_inventory(submission: Path) -> str | None:
    actual = {
        path.relative_to(submission).as_posix()
        for path in submission.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.relative_to(submission).parts[0] != ".git"
        and path.relative_to(submission).parts[0] != "output"
    }
    if not set(REQUIRED_COPY_PATHS) <= actual:
        return "required student package files are missing"
    return None


def grade_submission(submission: Path) -> list[GradeTest]:
    submission = submission.resolve()
    protected_error = _check_protected_files(submission)
    inventory_error = _check_submission_inventory(submission)
    package_detail = "; ".join(detail for detail in (inventory_error, protected_error) if detail)
    tests = [GradeTest("protected package and required files", 20, not package_detail, package_detail)]
    for name, points, check in (
        ("committed labeled-block artifact", 30, _check_labeled),
        ("committed selected-purchases artifact", 50, _check_selected),
    ):
        try:
            check(submission)
        except Exception as error:
            tests.append(GradeTest(name, points, False, str(error)))
        else:
            tests.append(GradeTest(name, points, True, ""))
    return tests


def _read_artifact(root: Path, name: str) -> pd.DataFrame:
    output = root / "output"
    path = output / name
    if output.is_symlink() or not path.is_file() or path.is_symlink():
        raise AssertionError(f"Missing regular output/{name}")
    return pd.read_csv(path)


def _check_labeled(root: Path) -> None:
    expected = pd.DataFrame({
        "record_id": ["site-102", "site-103"],
        "baseline_c": [15, 10], "follow_up_c": [23, 17],
    })
    pd.testing.assert_frame_equal(_read_artifact(root, "labeled_block.csv"), expected, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9)


def _check_selected(root: Path) -> None:
    source = root / "data/purchases.csv"
    if sha256(source.read_bytes()).hexdigest() != "0e86448f20a071552f8456075b8decef7541669b21345949a505aa93c78a07c9":
        raise AssertionError("Restore the supplied purchases fixture.")
    purchases = pd.read_csv(source)
    expected = purchases.loc[purchases["quantity"] >= 2].copy()
    expected["line_total"] = expected["quantity"] * expected["unit_price"]
    expected = expected.sort_values(["line_total", "purchase_id"], ascending=[False, True]).reset_index(drop=True)
    actual = _read_artifact(root, "selected_purchases.csv")
    if actual["quantity"].tolist() != expected["quantity"].tolist():
        raise AssertionError("Purchase quantities differ.")
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9)
