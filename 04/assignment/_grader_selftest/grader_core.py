"""Compatibility import for Assignment 04's public grading rules."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading import grade_submission
