"""Public managed-pytest facade for Assignment 03 and the optional Actions workflow."""

from pathlib import Path

from _public_checks import (
    check_driver_dataflow_and_cwd,
    check_exact_driver_output,
    check_import_safety,
    check_metadata_and_selection,
    check_pipeline_structure_and_execution,
    check_reductions_reshape_transpose_and_count,
    check_runtime_records_and_probe,
    check_student_code_structure,
    check_vector_operations,
    check_view_copy_relationship,
)


ASSIGNMENT_DIR = Path(__file__).resolve().parent


def test_runtime_records_supplied_files_and_fresh_probe():
    check_runtime_records_and_probe(ASSIGNMENT_DIR)


def test_safe_pipeline_structure_and_base_alternate_execution():
    check_pipeline_structure_and_execution(ASSIGNMENT_DIR)


def test_student_code_structure_and_direct_operation_boundaries():
    check_student_code_structure(ASSIGNMENT_DIR)


def test_array_metadata_and_basic_selection():
    check_metadata_and_selection(ASSIGNMENT_DIR)


def test_view_copy_relationship_and_nonmutation():
    check_view_copy_relationship(ASSIGNMENT_DIR)


def test_vector_mask_selection_arithmetic_and_scalar_broadcast():
    check_vector_operations(ASSIGNMENT_DIR)


def test_reductions_shapes_reshape_transpose_and_whole_count():
    check_reductions_reshape_transpose_and_count(ASSIGNMENT_DIR)


def test_quiet_artifact_free_imports():
    check_import_safety(ASSIGNMENT_DIR)


def test_driver_helper_dataflow_call_order_and_different_cwd():
    check_driver_dataflow_and_cwd(ASSIGNMENT_DIR)


def test_exact_fresh_driver_and_loader_output():
    check_exact_driver_output(ASSIGNMENT_DIR)
