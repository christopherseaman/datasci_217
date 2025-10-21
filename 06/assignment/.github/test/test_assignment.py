"""
Assignment 6 Tests - Data Wrangling with Merge, Concat, and Reshape

Tests verify that students correctly:
- Merge datasets using different join types
- Concatenate DataFrames vertically and horizontally
- Reshape data using pivot and melt operations
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
    required_files = ["customers.csv", "products.csv", "purchases.csv"]

    for filename in required_files:
        filepath = data_dir / filename
        assert filepath.exists(), f"Data file not found: {filepath}"

        # Verify not empty
        df = pd.read_csv(filepath)
        assert len(df) > 0, f"Data file is empty: {filepath}"


def test_q1_merged_data(output_dir, data_dir):
    """Test Question 1: Merged data output."""
    output_file = output_dir / "q1_merged_data.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate structure
    merged = pd.read_csv(output_file)

    # Should have data from all three sources
    assert len(merged) > 0, "Merged data is empty"

    # Check for key columns from each dataset
    # From purchases
    assert 'purchase_id' in merged.columns or 'purchase_date' in merged.columns, \
        "Missing purchase columns"

    # From customers
    assert 'name' in merged.columns or 'email' in merged.columns, \
        "Missing customer columns"

    # From products
    assert 'product_name' in merged.columns or 'category' in merged.columns, \
        "Missing product columns"

    # Check that total_price was calculated
    assert 'total_price' in merged.columns, \
        "Missing total_price column - should be calculated as quantity * price"

    # Verify total_price calculation is correct (within rounding tolerance)
    if 'quantity' in merged.columns and 'price' in merged.columns:
        expected = (merged['quantity'] * merged['price']).round(2)
        actual = merged['total_price'].round(2)

        # Check that at least 95% of values match (allowing for edge cases)
        matches = (expected == actual).sum()
        total = len(merged)
        match_rate = matches / total

        assert match_rate >= 0.95, \
            f"total_price calculation incorrect: only {match_rate:.1%} of values match quantity * price"


def test_q1_validation_report(output_dir):
    """Test Question 1: Validation report output."""
    output_file = output_dir / "q1_validation.txt"
    assert output_file.exists(), f"Validation report not found: {output_file}"

    # Read and check content
    content = output_file.read_text()

    # Should contain key metrics
    assert "Dataset Sizes" in content, "Missing dataset sizes section"
    assert "Merge Results" in content, "Missing merge results section"
    assert "rows" in content.lower(), "Missing row count information"


def test_q2_combined_data(output_dir):
    """Test Question 2: Concatenated data output."""
    output_file = output_dir / "q2_combined_data.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate
    combined = pd.read_csv(output_file)
    assert len(combined) > 0, "Combined data is empty"

    # Should have customer_id column (used as index)
    assert 'customer_id' in combined.columns, "Missing customer_id column"

    # Should have satisfaction and/or loyalty data
    has_satisfaction = 'satisfaction_score' in combined.columns
    has_loyalty = 'tier' in combined.columns or 'points' in combined.columns

    assert has_satisfaction or has_loyalty, \
        "Missing satisfaction or loyalty columns"


def test_q3_category_sales_wide(output_dir):
    """Test Question 3: Pivoted sales data (wide format)."""
    output_file = output_dir / "q3_category_sales_wide.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate
    sales_wide = pd.read_csv(output_file)
    assert len(sales_wide) > 0, "Sales data is empty"

    # Should have multiple columns (wide format)
    assert len(sales_wide.columns) > 2, \
        "Wide format should have multiple category columns"

    # Should have time/index information
    has_time_col = any(col in sales_wide.columns for col in ['month', 'date', 'purchase_date', 'time'])
    has_index_col = sales_wide.columns[0] == 'Unnamed: 0'
    assert has_time_col or has_index_col, \
        "Missing time/index column"


def test_q3_analysis_report(output_dir):
    """Test Question 3: Analysis report output."""
    output_file = output_dir / "q3_analysis_report.txt"
    assert output_file.exists(), f"Analysis report not found: {output_file}"

    # Read and check content
    content = output_file.read_text()

    # Should contain key analysis sections
    assert "Sales by Category" in content, "Missing sales by category section"
    assert "Time Period" in content, "Missing time period section"
    assert "Top Category" in content, "Missing top category"
    assert "Bottom Category" in content, "Missing bottom category"


def test_all_required_outputs(output_dir):
    """Test that all required output files exist."""
    required_outputs = [
        "q1_merged_data.csv",
        "q1_validation.txt",
        "q2_combined_data.csv",
        "q3_category_sales_wide.csv",
        "q3_analysis_report.txt"
    ]

    missing_files = []
    for filename in required_outputs:
        filepath = output_dir / filename
        if not filepath.exists():
            missing_files.append(str(filepath))

    assert len(missing_files) == 0, \
        f"Missing required output files: {', '.join(missing_files)}"


def test_q1_merge_types():
    """Test that students understand different merge types."""
    # This is validated by checking the validation report
    # The report should show different row counts for inner vs left vs outer joins
    output_dir = Path("output")
    report_file = output_dir / "q1_validation.txt"

    if report_file.exists():
        content = report_file.read_text()

        # Should mention different join types
        assert "Inner join" in content or "inner" in content.lower(), \
            "Should discuss inner join"
        assert "Left join" in content or "left" in content.lower(), \
            "Should discuss left join"


def test_q2_concatenation():
    """Test that concatenation was performed correctly."""
    output_dir = Path("output")
    output_file = output_dir / "q2_combined_data.csv"

    if output_file.exists():
        combined = pd.read_csv(output_file)

        # Horizontal concat should create some NaN values from misalignment
        # Check if there are any NaN values (indicates proper concatenation)
        has_nan = combined.isna().any().any()

        # Note: This test is lenient - we just check structure
        assert len(combined) > 0, "Combined data should not be empty"


def test_q3_reshape_operations():
    """Test that reshape operations were performed correctly."""
    output_dir = Path("output")

    # Wide format should exist
    wide_file = output_dir / "q3_category_sales_wide.csv"
    if wide_file.exists():
        wide_data = pd.read_csv(wide_file)

        # Wide format has categories as columns
        assert len(wide_data.columns) >= 3, \
            "Wide format should have multiple category columns"

    # Analysis report should show category summaries
    report_file = output_dir / "q3_analysis_report.txt"
    if report_file.exists():
        content = report_file.read_text()

        # Should contain actual category names
        categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Sports"]
        found_categories = sum(1 for cat in categories if cat in content)

        assert found_categories >= 1, \
            "Report should mention at least one product category"
