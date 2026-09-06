"""Public managed-pytest contract for Assignment 02 and the optional Actions workflow."""

from pathlib import Path

from _public_checks import check_git_state_answers, check_project_documents, check_report_artifact


ASSIGNMENT_DIR = Path(__file__).resolve().parent


def test_project_description_run_and_gitignore():
    check_project_documents(ASSIGNMENT_DIR)


def test_git_state_snapshots():
    check_git_state_answers(ASSIGNMENT_DIR)


def test_committed_report_artifact():
    check_report_artifact(ASSIGNMENT_DIR)
