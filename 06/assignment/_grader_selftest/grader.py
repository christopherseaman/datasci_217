"""Development compatibility shim for the public canonical grader."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading import *  # noqa: F401,F403
