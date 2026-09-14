"""Run the trusted canonical grader without provisioning dependencies."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading import main


if __name__ == "__main__":
    raise SystemExit(main())
