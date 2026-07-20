"""Public managed-pytest contract for Assignment 02 and Classroom 50."""

from pathlib import Path

from _public_checks import (
    check_analysis_structure,
    check_format_behavior,
    check_git_state_answers,
    check_import_safety,
    check_main_execution,
    check_main_structure,
    check_main_uses_formatter,
    check_mean_behavior,
    check_project_documents,
)


ASSIGNMENT_DIR = Path(__file__).resolve().parent


def test_project_description_run_and_gitignore():
    check_project_documents(ASSIGNMENT_DIR)


def test_git_state_snapshots():
    check_git_state_answers(ASSIGNMENT_DIR)


def test_analysis_utils_structure():
    check_analysis_structure(ASSIGNMENT_DIR)


def test_mean_behavior():
    check_mean_behavior(ASSIGNMENT_DIR)


def test_format_summary_behavior_and_mean_use():
    check_format_behavior(ASSIGNMENT_DIR)


def test_main_structure_and_readback():
    check_main_structure(ASSIGNMENT_DIR)


def test_quiet_imports_and_no_import_artifact():
    check_import_safety(ASSIGNMENT_DIR)


def test_fresh_main_output_and_overwritten_report():
    check_main_execution(ASSIGNMENT_DIR)


def test_main_formatter_call_order():
    check_main_uses_formatter(ASSIGNMENT_DIR)
