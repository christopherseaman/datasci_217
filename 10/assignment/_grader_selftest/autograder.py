# /// script
# requires-python = "==3.12.13"
# dependencies = []
# ///

"""Plain-Python Classroom50 bootstrap for Assignment 10.

The non-release candidate intentionally has no transitive constraints file.
Production grading must fail as infrastructure rather than silently resolving
an uncertified environment.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


sys.dont_write_bytecode = True


BUNDLE_DIR = Path(__file__).resolve().parent


def main() -> int:
    result_path = Path.cwd() / "result.json"
    if result_path.is_file() or result_path.is_symlink():
        result_path.unlink()
    constraints_path = BUNDLE_DIR / "constraints.txt"
    if not constraints_path.is_file() or constraints_path.is_symlink():
        print(
            "[INFRASTRUCTURE] Assignment 10 is not release-certified: "
            "the exact instructor constraints.txt is absent.",
            file=sys.stderr,
        )
        return 2
    try:
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            text=True,
            capture_output=True,
            check=False,
        )
        if pip_check.returncode:
            subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--quiet",
                "-c",
                str(constraints_path),
                "-r",
                str(BUNDLE_DIR / "requirements.txt"),
            ],
            check=True,
        )
        from classroom50_grader import main as grade

        return grade()
    except Exception as error:
        if result_path.is_file() or result_path.is_symlink():
            result_path.unlink()
        print(
            f"[INFRASTRUCTURE] provisioning or grader startup failed: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
