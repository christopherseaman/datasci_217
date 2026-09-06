"""Public managed-pytest facade for Assignment 03 and the optional Actions workflow."""

from pathlib import Path

from _public_checks import check_pipeline_artifacts, check_runtime_records_and_probe


ASSIGNMENT_DIR = Path(__file__).resolve().parent


def test_runtime_records_supplied_files_and_committed_probe():
    check_runtime_records_and_probe(ASSIGNMENT_DIR)


def test_committed_pipeline_and_analysis_artifacts():
    check_pipeline_artifacts(ASSIGNMENT_DIR)
