import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Test fixtures
@pytest.fixture
def output_dir():
    """Expected output directory"""
    return Path("output")

def test_q1_exploration(output_dir):
    """Test Question 1: Data exploration and summary statistics"""
    output_file = output_dir / "exploration_summary.csv"

    # Check file exists
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load the summary statistics
    summary = pd.read_csv(output_file, index_col=0)

    # Check it's the output of .describe()
    assert 'count' in summary.index, "Summary should include 'count' row"
    assert 'mean' in summary.index, "Summary should include 'mean' row"
    assert 'std' in summary.index, "Summary should include 'std' row"
    assert 'min' in summary.index, "Summary should include 'min' row"
    assert 'max' in summary.index, "Summary should include 'max' row"

    # Should have numeric columns only
    expected_numeric_cols = ['quantity', 'price_per_item']
    for col in expected_numeric_cols:
        assert col in summary.columns, f"Expected numeric column '{col}' in summary"

    print("✓ Question 1 passed: Summary statistics generated correctly")

def test_q2_cleaning(output_dir):
    """Test Question 2: Data cleaning and filtering"""
    output_file = output_dir / "cleaned_data.csv"

    # Check file exists
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load cleaned data
    df_clean = pd.read_csv(output_file)

    # Check no missing values in quantity and shipping_method
    assert df_clean['quantity'].isnull().sum() == 0, "quantity column should have no missing values"
    assert df_clean['shipping_method'].isnull().sum() == 0, "shipping_method column should have no missing values"

    # Check quantity is integer type (will be int64 after read_csv)
    assert pd.api.types.is_integer_dtype(df_clean['quantity']), "quantity should be integer type"

    # Check purchase_date can be converted to datetime (string is OK in CSV)
    # The datetime conversion would have been done before saving, but CSV saves as string
    pd.to_datetime(df_clean['purchase_date'])  # This should not raise an error

    # Check only CA and NY states
    assert set(df_clean['customer_state'].unique()) <= {'CA', 'NY'}, "Should only contain CA and NY states"

    # Check quantity >= 2
    assert (df_clean['quantity'] >= 2).all(), "All quantity values should be >= 2"

    # Check we have reasonable amount of data left
    assert len(df_clean) > 0, "Cleaned data should not be empty"
    assert len(df_clean) < 15000, "Cleaned data should be filtered (less than original 15,000 rows)"

    print(f"✓ Question 2 passed: Data cleaned correctly ({len(df_clean)} rows)")

def test_q3_analysis(output_dir):
    """Test Question 3: Analysis and aggregation"""
    output_file = output_dir / "analysis_results.csv"

    # Check file exists
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load analysis results
    analysis = pd.read_csv(output_file)

    # Check it has product_category and total_revenue columns
    # (Students might structure this differently, so we're flexible)
    assert 'product_category' in analysis.columns or 'category' in str(analysis.columns).lower(), \
        "Analysis should include product category information"

    # Check there's aggregated revenue data
    revenue_cols = [col for col in analysis.columns if 'revenue' in col.lower() or 'total' in col.lower()]
    assert len(revenue_cols) > 0, "Analysis should include revenue/total information"

    # Check we have multiple categories (from the data generator, we have 5 categories)
    assert len(analysis) >= 3, "Analysis should include at least 3 product categories or products"

    # Check numeric values are reasonable (should be positive)
    numeric_col = revenue_cols[0] if revenue_cols else analysis.select_dtypes(include=['number']).columns[0]
    assert (analysis[numeric_col] > 0).all(), "Revenue values should be positive"

    print("✓ Question 3 passed: Analysis results generated correctly")

def test_all_outputs_exist(output_dir):
    """Meta-test: Check all three output files exist"""
    expected_files = [
        "exploration_summary.csv",
        "cleaned_data.csv",
        "analysis_results.csv"
    ]

    for filename in expected_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"Missing required output file: {filename}"

    print("✓ All output files present")

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
