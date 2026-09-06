"""Public pytest contract for Assignment 01 and the optional Actions workflow."""

from pathlib import Path

from _assignment_checks import check_output_artifact, check_terminal_practice


ASSIGNMENT_DIR = Path(__file__).resolve().parent


def test_terminal_practice_evidence():
    check_terminal_practice(ASSIGNMENT_DIR)


def test_committed_readiness_artifact():
    check_output_artifact(ASSIGNMENT_DIR)
