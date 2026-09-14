"""QA shim for the public canonical grader."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grading as _grading

globals().update(vars(_grading))
