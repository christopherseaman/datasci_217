"""Public pytest contract for Assignment 01 and the optional Actions workflow."""

from pathlib import Path

from _assignment_checks import (
    check_debug_report,
    check_measurement_alternates,
    check_measurement_default,
    check_measurement_structure,
    check_output_artifact,
    check_readiness_behavior,
    check_readiness_structure,
    check_terminal_practice,
    check_wrapper_from_other_cwd,
)


ASSIGNMENT_DIR = Path(__file__).resolve().parent


def test_terminal_practice_evidence():
    check_terminal_practice(ASSIGNMENT_DIR)


def test_readiness_structure():
    check_readiness_structure(ASSIGNMENT_DIR)


def test_readiness_output_and_dynamic_filename():
    check_readiness_behavior(ASSIGNMENT_DIR)


def test_measurement_structure():
    check_measurement_structure(ASSIGNMENT_DIR)


def test_measurement_default_output():
    check_measurement_default(ASSIGNMENT_DIR)


def test_measurement_alternate_values():
    check_measurement_alternates(ASSIGNMENT_DIR)


def test_debug_corrections():
    check_debug_report(ASSIGNMENT_DIR)


def test_output_artifact_is_fresh():
    check_output_artifact(ASSIGNMENT_DIR)


def test_output_wrapper_from_other_working_directory():
    check_wrapper_from_other_cwd(ASSIGNMENT_DIR)
