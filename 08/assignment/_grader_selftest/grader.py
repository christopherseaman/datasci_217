"""Development compatibility shim for the public canonical grader."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grading as _grading

globals().update(
    {name: getattr(_grading, name) for name in dir(_grading) if not name.startswith("__")}
)
