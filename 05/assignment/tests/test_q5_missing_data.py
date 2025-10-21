#!/usr/bin/env python3
"""
Test suite for Q5 Missing Data Analysis
Validates cleaning operations and imputation strategies.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from q3_data_utils import load_data, detect_missing, fill_missing, clean_data

# Set working directory to assignment folder
ASSIGNMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ASSIGNMENT_DIR)


def test_clean_data_has_no_missing():
    """Test that cleaned data has no missing values."""
    df_clean = pd.read_csv('output/q5_cleaned_data.csv')
    missing_counts = detect_missing(df_clean)
    total_missing = missing_counts.sum()

    assert total_missing == 0, f"Expected 0 missing values, found {total_missing}"
    print("✓ Test passed: Cleaned data has no missing values")


def test_clean_data_preserves_rows():
    """Test that cleaning preserved most rows (acceptable retention rate)."""
    df_original = load_data('data/clinical_trial_raw.csv')
    df_clean = pd.read_csv('output/q5_cleaned_data.csv')

    retention_rate = len(df_clean) / len(df_original)
    assert retention_rate >= 0.95, \
        f"Expected >=95% retention, got {retention_rate*100:.1f}% ({len(df_clean)}/{len(df_original)})"
    print(f"✓ Test passed: {retention_rate*100:.1f}% row retention ({len(df_clean)}/{len(df_original)})")


def test_imputation_strategies():
    """Test that different imputation strategies work correctly."""
    # Create test data with missing values
    test_df = pd.DataFrame({
        'values': [10, 20, np.nan, 30, np.nan, 40]
    })

    # Test mean imputation
    df_mean = fill_missing(test_df, 'values', strategy='mean')
    assert df_mean['values'].isnull().sum() == 0, "Mean imputation failed"
    expected_mean = 25.0  # (10+20+30+40)/4
    assert np.isclose(df_mean['values'].mean(), expected_mean), \
        f"Mean after imputation should be {expected_mean}"
    print("✓ Test passed: Mean imputation works correctly")

    # Test median imputation
    df_median = fill_missing(test_df, 'values', strategy='median')
    assert df_median['values'].isnull().sum() == 0, "Median imputation failed"
    expected_median = 25.0  # median of [10,20,30,40]
    assert df_median['values'].median() == expected_median, \
        f"Median should be {expected_median}"
    print("✓ Test passed: Median imputation works correctly")

    # Test forward fill
    df_ffill = fill_missing(test_df, 'values', strategy='ffill')
    # First NaN gets filled with 20, second NaN gets filled with 30
    expected_ffill = [10, 20, 20, 30, 30, 40]
    assert list(df_ffill['values']) == expected_ffill, \
        "Forward fill produced incorrect results"
    print("✓ Test passed: Forward fill works correctly")


def test_imputation_preserves_mean():
    """Test that mean imputation preserves the original mean."""
    df = load_data('data/clinical_trial_raw.csv')

    # Get original mean (excluding missing)
    original_mean = df['bmi'].mean()

    # Impute with mean
    df_imputed = fill_missing(df, 'bmi', strategy='mean')
    imputed_mean = df_imputed['bmi'].mean()

    # Means should be very close (within floating point precision)
    assert np.isclose(original_mean, imputed_mean, rtol=1e-10), \
        f"Mean changed from {original_mean:.6f} to {imputed_mean:.6f}"
    print(f"✓ Test passed: Mean imputation preserves mean ({original_mean:.2f})")


def test_median_imputation_robustness():
    """Test that median imputation is robust to outliers."""
    # Create data with outliers
    test_df = pd.DataFrame({
        'values': [1, 2, 3, 4, 5, np.nan, 1000]  # 1000 is outlier
    })

    # Median should be 3, not affected by outlier
    df_median = fill_missing(test_df, 'values', strategy='median')
    filled_value = df_median.loc[5, 'values']  # The filled position

    # Median of [1,2,3,4,5,1000] is 3.5
    expected_median = 3.5
    assert filled_value == expected_median, \
        f"Expected {expected_median}, got {filled_value}"
    print(f"✓ Test passed: Median imputation is robust to outliers")


def test_critical_columns_complete():
    """Test that critical columns (patient_id, age, sex) are complete."""
    df_clean = pd.read_csv('output/q5_cleaned_data.csv')

    critical_cols = ['patient_id', 'age', 'sex']
    for col in critical_cols:
        missing = df_clean[col].isnull().sum()
        assert missing == 0, f"Critical column '{col}' has {missing} missing values"

    print(f"✓ Test passed: All critical columns are complete")


def test_output_files_exist():
    """Test that all expected output files were created."""
    expected_files = [
        'output/q5_cleaned_data.csv',
        'output/q5_missing_report.txt',
        'output/q5_missing_values_plot.png'
    ]

    for filepath in expected_files:
        assert os.path.exists(filepath), f"Missing output file: {filepath}"

    print("✓ Test passed: All output files created")


def test_cleaned_data_dtypes():
    """Test that cleaned data has appropriate data types."""
    df_clean = pd.read_csv('output/q5_cleaned_data.csv')

    # Numeric columns should be numeric
    numeric_cols = ['age', 'bmi', 'systolic_bp', 'diastolic_bp',
                   'cholesterol_total', 'glucose_fasting', 'adherence_pct']

    for col in numeric_cols:
        if col in df_clean.columns:
            assert pd.api.types.is_numeric_dtype(df_clean[col]), \
                f"Column '{col}' should be numeric type"

    print("✓ Test passed: Data types are appropriate")


def test_imputed_values_reasonable():
    """Test that imputed values are within reasonable ranges."""
    df_clean = pd.read_csv('output/q5_cleaned_data.csv')

    # Define reasonable ranges for clinical measurements
    ranges = {
        'bmi': (15, 50),
        'systolic_bp': (70, 200),
        'diastolic_bp': (40, 120),
        'cholesterol_total': (80, 400),  # Some people have cholesterol <100
        'glucose_fasting': (50, 200),
        'adherence_pct': (0, 100)
    }

    for col, (min_val, max_val) in ranges.items():
        if col in df_clean.columns:
            col_min = df_clean[col].min()
            col_max = df_clean[col].max()
            assert col_min >= min_val and col_max <= max_val, \
                f"{col} values outside reasonable range [{min_val}, {max_val}]: [{col_min}, {col_max}]"

    print("✓ Test passed: Imputed values are within reasonable ranges")


def run_all_tests():
    """Run all tests and report results."""
    print("="*70)
    print("Running Q5 Missing Data Analysis Tests")
    print("="*70)

    tests = [
        test_output_files_exist,
        test_clean_data_has_no_missing,
        test_clean_data_preserves_rows,
        test_imputation_strategies,
        test_imputation_preserves_mean,
        test_median_imputation_robustness,
        test_critical_columns_complete,
        test_cleaned_data_dtypes,
        test_imputed_values_reasonable
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
