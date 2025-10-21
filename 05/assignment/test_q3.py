#!/usr/bin/env python3
"""Test script for q3_data_utils.py functions"""

import sys
import pandas as pd
import numpy as np
from q3_data_utils import (
    load_data,
    clean_data,
    detect_missing,
    fill_missing,
    filter_data,
    transform_types,
    create_bins,
    summarize_by_group
)

def test_load_data():
    """Test data loading"""
    print("Testing load_data...")
    # Use clinical trial data which has proper headers
    df = load_data('data/clinical_trial_raw.csv')
    assert df is not None, "Should load data"
    assert len(df) > 0, "Should have data rows"
    print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")
    print("  ✓ PASSED\n")
    return df

def test_clean_data():
    """Test data cleaning"""
    print("Testing clean_data...")
    test_df = pd.DataFrame({
        'age': [25, 30, -999, 35, 40],
        'bmi': [22, 25, 28, pd.NA, 30]
    })
    df_clean = clean_data(test_df, sentinel_value=-999)
    assert pd.isna(df_clean['age'].iloc[2]), "Should replace -999 with NaN"
    print("  ✓ PASSED\n")

def test_detect_missing():
    """Test missing value detection"""
    print("Testing detect_missing...")
    test_df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [1, np.nan, np.nan, 4]
    })
    missing = detect_missing(test_df)
    assert missing['a'] == 1, "Column 'a' should have 1 missing value"
    assert missing['b'] == 2, "Column 'b' should have 2 missing values"
    print("  ✓ PASSED\n")

def test_fill_missing():
    """Test missing value imputation"""
    print("Testing fill_missing...")
    test_df = pd.DataFrame({'age': [20, 30, np.nan, 40, 50]})

    df_mean = fill_missing(test_df, 'age', strategy='mean')
    assert df_mean['age'].isna().sum() == 0, "Should fill with mean"

    df_median = fill_missing(test_df, 'age', strategy='median')
    assert df_median['age'].isna().sum() == 0, "Should fill with median"

    print("  ✓ PASSED\n")

def test_filter_data():
    """Test data filtering"""
    print("Testing filter_data...")
    test_df = pd.DataFrame({
        'age': [20, 30, 40, 50, 60],
        'site': ['A', 'B', 'A', 'C', 'B']
    })

    filters = [
        {'column': 'age', 'condition': 'greater_than', 'value': 25},
        {'column': 'site', 'condition': 'equals', 'value': 'A'}
    ]
    df_filtered = filter_data(test_df, filters)
    assert len(df_filtered) == 1, "Should filter to 1 row (age>25 and site='A')"
    print("  ✓ PASSED\n")

def test_transform_types():
    """Test type transformation"""
    print("Testing transform_types...")
    test_df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'age': ['20', '30'],
        'site': ['A', 'B']
    })

    type_map = {
        'date': 'datetime',
        'age': 'numeric',
        'site': 'category'
    }
    df_typed = transform_types(test_df, type_map)
    assert pd.api.types.is_datetime64_any_dtype(df_typed['date']), "Should be datetime"
    assert pd.api.types.is_numeric_dtype(df_typed['age']), "Should be numeric"
    assert pd.api.types.is_categorical_dtype(df_typed['site']), "Should be category"
    print("  ✓ PASSED\n")

def test_create_bins():
    """Test binning"""
    print("Testing create_bins...")
    test_df = pd.DataFrame({'age': [15, 25, 35, 45, 55, 70]})

    df_binned = create_bins(
        test_df,
        column='age',
        bins=[0, 18, 35, 50, 65, 100],
        labels=['<18', '18-34', '35-49', '50-64', '65+']
    )
    assert 'age_binned' in df_binned.columns, "Should create binned column"
    assert pd.api.types.is_categorical_dtype(df_binned['age_binned']), "Should be categorical"
    print("  ✓ PASSED\n")

def test_summarize_by_group():
    """Test grouping and summarization"""
    print("Testing summarize_by_group...")
    test_df = pd.DataFrame({
        'site': ['A', 'A', 'B', 'B'],
        'age': [20, 30, 25, 35]
    })

    summary = summarize_by_group(test_df, 'site', {'age': 'mean'})
    assert len(summary) == 2, "Should have 2 groups"
    print("  ✓ PASSED\n")

if __name__ == "__main__":
    try:
        test_load_data()
        test_clean_data()
        test_detect_missing()
        test_fill_missing()
        test_filter_data()
        test_transform_types()
        test_create_bins()
        test_summarize_by_group()

        print("=" * 50)
        print("ALL Q3 TESTS PASSED!")
        print("=" * 50)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
