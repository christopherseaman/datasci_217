"""
Assignment 8 Tests - Data Aggregation and Group Operations

Tests verify that students correctly:
- Perform groupby operations with aggregation functions
- Use transform, filter, and apply operations
- Create pivot tables and cross-tabulations
- Optimize performance for large datasets
- Save required output files
"""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def output_dir():
    """Fixture that returns the output directory path."""
    return Path("output")


@pytest.fixture
def data_dir():
    """Fixture that returns the data directory path."""
    return Path("data")


def test_data_files_exist(data_dir):
    """Test that required data files were generated."""
    required_files = [
        "provider_data.csv",
        "facility_data.csv",
        "encounter_data.csv",
    ]

    for filename in required_files:
        filepath = data_dir / filename
        assert filepath.exists(), f"Data file not found: {filepath}"

        # Verify not empty
        df = pd.read_csv(filepath)
        assert len(df) > 0, f"Data file is empty: {filepath}"


def test_q1_groupby_analysis(output_dir):
    """Test Question 1: GroupBy analysis output."""
    output_file = output_dir / "q1_groupby_analysis.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate structure
    df = pd.read_csv(output_file)
    assert len(df) > 0, "GroupBy analysis data is empty"

    # Should have facility-related columns
    assert any("facility" in col.lower() for col in df.columns), (
        "Missing facility-related columns"
    )
    
    # Should have aggregation results (mean, sum, count, etc.)
    assert len(df.columns) >= 3, "GroupBy analysis should have multiple aggregated columns"


def test_q1_aggregation_report(output_dir):
    """Test Question 1: Aggregation report output."""
    output_file = output_dir / "q1_aggregation_report.txt"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Read and check content
    content = output_file.read_text()
    assert len(content) > 0, "Aggregation report is empty"

    # Should contain key analysis sections
    assert any(
        keyword in content.lower()
        for keyword in ["facility", "provider", "encounter", "analysis", "transform"]
    ), "Missing key analysis content"


def test_q2_filter_analysis(output_dir):
    """Test Question 2: Filter operations analysis output."""
    output_file = output_dir / "q2_filter_analysis.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate structure
    df = pd.read_csv(output_file)
    assert len(df) > 0, "Filter analysis data is empty"
    
    # Should have filtered results (fewer facilities than original)
    # This is a basic check - filtered data should exist
    assert len(df) > 0, "Filter operations should produce some results"


def test_q2_hierarchical_analysis(output_dir):
    """Test Question 2: Hierarchical analysis output."""
    output_file = output_dir / "q2_hierarchical_analysis.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate structure
    df = pd.read_csv(output_file)
    assert len(df) > 0, "Hierarchical analysis data is empty"
    
    # Hierarchical grouping should have facility and region columns
    assert any("facility" in col.lower() for col in df.columns), (
        "Missing facility column in hierarchical analysis"
    )
    assert any("region" in col.lower() for col in df.columns), (
        "Missing region column in hierarchical analysis"
    )


def test_q2_performance_report(output_dir):
    """Test Question 2: Performance report output."""
    output_file = output_dir / "q2_performance_report.txt"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Read and check content
    content = output_file.read_text()
    assert len(content) > 0, "Performance report is empty"
    
    # Should contain apply operation results
    assert any(
        keyword in content.lower()
        for keyword in ["provider", "statistics", "facility", "apply", "encounter"]
    ), "Missing key content in performance report (should contain apply operation results)"


def test_q3_pivot_analysis(output_dir):
    """Test Question 3: Pivot table analysis output."""
    output_file = output_dir / "q3_pivot_analysis.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate structure
    df = pd.read_csv(output_file)
    assert len(df) > 0, "Pivot analysis data is empty"

    # Should have multiple columns (pivot table format)
    assert len(df.columns) > 2, "Pivot table should have multiple columns"
    
    # Should have index column (procedure/diagnosis) and region columns
    assert any("procedure" in col.lower() or "diagnosis" in col.lower() or "encounter" in col.lower() for col in df.columns) or len(df.index) > 0, (
        "Pivot table should have procedure/diagnosis/encounter information"
    )


def test_q3_crosstab_analysis(output_dir):
    """Test Question 3: Cross-tabulation analysis output."""
    output_file = output_dir / "q3_crosstab_analysis.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate structure
    df = pd.read_csv(output_file)
    assert len(df) > 0, "Crosstab analysis data is empty"


def test_q3_pivot_visualization(output_dir):
    """Test Question 3: Pivot visualization output."""
    output_file = output_dir / "q3_pivot_visualization.png"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Verify file is not empty
    assert output_file.stat().st_size > 0, "Visualization file is empty"


def test_all_required_outputs(output_dir):
    """Test that all required output files exist."""
    required_outputs = [
        "q1_groupby_analysis.csv",
        "q1_aggregation_report.txt",
        "q2_filter_analysis.csv",
        "q2_hierarchical_analysis.csv",
        "q2_performance_report.txt",
        "q3_pivot_analysis.csv",
        "q3_crosstab_analysis.csv",
        "q3_pivot_visualization.png",
    ]

    missing_files = []
    for filename in required_outputs:
        filepath = output_dir / filename
        if not filepath.exists():
            missing_files.append(str(filepath))

    assert len(missing_files) == 0, (
        f"Missing required output files: {', '.join(missing_files)}"
    )


def test_csv_file_validation(output_dir):
    """Test that CSV files are properly formatted."""
    csv_files = [
        "q1_groupby_analysis.csv",
        "q2_filter_analysis.csv",
        "q2_hierarchical_analysis.csv",
        "q3_pivot_analysis.csv",
        "q3_crosstab_analysis.csv",
    ]

    for filename in csv_files:
        filepath = output_dir / filename
        if filepath.exists():
            # Try to read the CSV file
            try:
                df = pd.read_csv(filepath)
                assert len(df) > 0, f"CSV file {filename} is empty"
            except Exception as e:
                pytest.fail(f"Error reading CSV file {filename}: {e}")


def test_text_file_validation(output_dir):
    """Test that text files contain meaningful content."""
    text_files = ["q1_aggregation_report.txt", "q2_performance_report.txt"]

    for filename in text_files:
        filepath = output_dir / filename
        if filepath.exists():
            content = filepath.read_text()
            assert len(content.strip()) > 50, (
                f"Text file {filename} is too short or empty"
            )
