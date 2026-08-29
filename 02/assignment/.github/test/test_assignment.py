"""Portable pytest entrypoint for Assignment 02.

The assignment source keeps the public tests at repository root for local
backwards compatibility. This wrapper makes the same tests discoverable from
the standalone repository's conventional .github/test path.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[2]
PUBLIC_TESTS = ASSIGNMENT_DIR / "test_assignment.py"
sys.path.insert(0, str(ASSIGNMENT_DIR))

_spec = spec_from_file_location("_assignment_public_tests", PUBLIC_TESTS)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load public tests from {PUBLIC_TESTS}")
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

for _name, _value in vars(_module).items():
    if _name.startswith("test_"):
        globals()[_name] = _value
