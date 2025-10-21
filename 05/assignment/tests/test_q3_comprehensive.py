#!/usr/bin/env python3
"""Comprehensive test of q3_data_utils.py with clinical trial data"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
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

def main():
    print("=" * 60)
    print("Q3 DATA UTILITIES - COMPREHENSIVE TEST")
    print("=" * 60)
    print()

    # Test 1: Load data
    print("1. Loading clinical trial data...")
    df = load_data('../data/clinical_trial_raw.csv')
    print(f"   Loaded: {len(df)} rows × {len(df.columns)} columns")
    print(f"   Columns: {', '.join(df.columns[:5])}...")
    print()

    # Test 2: Detect missing values
    print("2. Detecting missing values...")
    missing = detect_missing(df)
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"   Found missing values in {len(missing_cols)} columns:")
        for col, count in missing_cols.head(5).items():
            print(f"     - {col}: {count} missing ({count/len(df)*100:.1f}%)")
    else:
        print("   No missing values detected")
    print()

    # Test 3: Clean data (remove duplicates, replace -999 and -1 with NaN)
    print("3. Cleaning data...")
    df_clean = clean_data(df, remove_duplicates=True, sentinel_value=-999)
    df_clean = clean_data(df_clean, remove_duplicates=False, sentinel_value=-1)
    print(f"   Original rows: {len(df)}")
    print(f"   After cleaning: {len(df_clean)}")
    missing_after = detect_missing(df_clean)
    missing_after_cols = missing_after[missing_after > 0]
    print(f"   Columns with missing values: {len(missing_after_cols)}")
    print()

    # Test 4: Fill missing values
    print("4. Filling missing values...")
    if 'age' in df_clean.columns and df_clean['age'].isna().any():
        df_filled = fill_missing(df_clean, 'age', strategy='median')
        before = df_clean['age'].isna().sum()
        after = df_filled['age'].isna().sum()
        print(f"   Age column - Missing before: {before}, after: {after}")
    else:
        print("   No missing values in 'age' to fill")
    print()

    # Test 5: Filter data
    print("5. Filtering data...")
    filters = [
        {'column': 'age', 'condition': 'in_range', 'value': [18, 65]}
    ]
    df_filtered = filter_data(df_clean, filters)
    print(f"   Rows after filtering (age 18-65): {len(df_filtered)}")
    print()

    # Test 6: Transform types
    print("6. Transforming data types...")
    type_map = {
        'enrollment_date': 'datetime',
        'age': 'numeric',
        'site': 'category',
        'intervention_group': 'category'
    }
    df_typed = transform_types(df_filtered, type_map)
    print("   Type transformations:")
    for col, dtype in type_map.items():
        if col in df_typed.columns:
            print(f"     - {col}: {df_typed[col].dtype}")
    print()

    # Test 7: Create bins
    print("7. Creating age bins...")
    df_binned = create_bins(
        df_typed,
        column='age',
        bins=[0, 30, 40, 50, 60, 100],
        labels=['<30', '30-39', '40-49', '50-59', '60+']
    )
    print("   Age distribution:")
    age_dist = df_binned['age_binned'].value_counts().sort_index()
    for category, count in age_dist.items():
        print(f"     - {category}: {count} ({count/len(df_binned)*100:.1f}%)")
    print()

    # Test 8: Summarize by group
    print("8. Summarizing by site...")
    if 'site' in df_binned.columns:
        summary = summarize_by_group(
            df_binned,
            'site',
            {'age': ['mean', 'count'], 'bmi': 'mean'}
        )
        print("   Summary by site:")
        print(summary.head())
    print()

    print("=" * 60)
    print("ALL COMPREHENSIVE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Summary of utility functions:")
    print("  ✓ load_data() - Loads CSV files")
    print("  ✓ clean_data() - Removes duplicates and sentinel values")
    print("  ✓ detect_missing() - Counts missing values per column")
    print("  ✓ fill_missing() - Imputes missing values (mean/median/ffill)")
    print("  ✓ filter_data() - Applies multiple filters sequentially")
    print("  ✓ transform_types() - Converts column data types")
    print("  ✓ create_bins() - Bins continuous data into categories")
    print("  ✓ summarize_by_group() - Groups and aggregates data")
    print()
    print("All functions follow DRY principles and are reusable!")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
