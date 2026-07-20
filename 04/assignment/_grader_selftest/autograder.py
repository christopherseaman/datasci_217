# /// script
# requires-python = "==3.12.13"
# dependencies = []
# ///

"""Classroom50 per-assignment autograder prototype for Assignment 04.

This file and its siblings belong in the teacher-controlled Assignment 04
bundle, never in the student starter. Test failures are represented in the
required result.json; only grader infrastructure failure exits nonzero.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REQUIRED_CONTEXT_ENV = {
    "classroom": "CLASSROOM",
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
            "missing required Classroom50 context: " + ", ".join(missing)
        )
    context["review"] = os.environ.get("REVIEW_URL", "").strip() or context["commit"]
    context["datetime"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return context

def _prepare_result_path() -> Path:
    result_path = Path.cwd() / "result.json"
    if result_path.is_file() or result_path.is_symlink():
        result_path.unlink()
    return result_path


def _provision() -> None:
    if os.environ.get("A04_SKIP_INSTALL") == "1":
        return
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    if pip_check.returncode:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            "-r",
            str(HERE / "requirements.txt"),
        ],
        check=True,
    )


def main() -> int:
    result_path = _prepare_result_path()
    _provision()
    context = _context()
    from grader_core import grade_submission

    tests = grade_submission(Path.cwd())
    for test in tests:
        status = "PASS" if test.passed else "FAIL"
        detail = f": {test.detail}" if test.detail else ""
        print(f"[{status}] {test.name}{detail}")

    result_tests = [
        {
            "test-name": test.name,
            "passed": test.passed,
            "score": test.score,
            "max-score": test.max_score,
        }
        for test in tests
    ]
    result = {
        "schema": "classroom50/result/v1",
        **context,
        "score": sum(test["score"] for test in result_tests),
        "max-score": sum(test["max-score"] for test in result_tests),
        "tests": result_tests,
    }
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InfrastructureError as error:
        print(f"[INFRASTRUCTURE] {error}", file=sys.stderr)
        raise SystemExit(2) from error
    except Exception as error:
        print(
            f"[INFRASTRUCTURE] unexpected grader failure: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2) from error
