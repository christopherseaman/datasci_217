"""Compatibility launcher for the public Assignment 04 grader."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    checker = Path(__file__).resolve().parents[1] / "check_assignment.py"
    return subprocess.run([sys.executable, str(checker), str(Path.cwd()), "--json"], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
