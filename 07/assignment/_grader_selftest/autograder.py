# /// script
# requires-python = "==3.12.13"
# dependencies = []
# ///

"""Plain-Python grading bootstrap for Assignment 07."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


BUNDLE_DIR = Path(__file__).resolve().parent


def main() -> int:
    result_path = Path.cwd() / "result.json"
    if result_path.is_file() or result_path.is_symlink():
        result_path.unlink()
    try:
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
                str(BUNDLE_DIR / "requirements.txt"),
            ],
            check=True,
        )
        from grader import main as grade

        return grade()
    except Exception as error:
        if result_path.is_file() or result_path.is_symlink():
            result_path.unlink()
        print(
            f"[INFRASTRUCTURE] provisioning or grader startup failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
