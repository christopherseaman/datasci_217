#!/usr/bin/env python3
"""
Assignment 5: Data Cleaning Exam
Complete the functions below to build a data cleaning pipeline.
"""

import pandas as pd
import numpy as np

def audit_data(df):
    """
    Audit data quality and return a report.

    Args:
        df (pd.DataFrame): Raw employee data

    Returns:
        dict: Data quality report with keys:
            - total_rows: int
            - total_columns: int
            - missing_count: dict mapping column names to missing count
            - duplicate_count: int
            - negative_salaries: int
            - invalid_dates: int (count of "UNKNOWN" in hire_date)
    """
    # TODO: Implement this function
    pass

def clean_data(df):
    """
    Clean the raw employee data.

    Args:
        df (pd.DataFrame): Raw employee data

    Returns:
        pd.DataFrame: Cleaned data with:
            - Sentinel values (-999) replaced with NaN
            - Missing numeric data filled appropriately
            - Text data standardized
            - Dates converted to datetime
            - Outliers removed (salary < 0 or > 500000)
            - Duplicates removed
    """
    # TODO: Implement this function
    pass

def transform_data(df_clean):
    """
    Add calculated and categorical columns to cleaned data.

    Args:
        df_clean (pd.DataFrame): Cleaned employee data

    Returns:
        pd.DataFrame: Transformed data with new columns:
            - experience_level: "Junior" (0-2), "Mid" (2-5), "Senior" (5+)
            - salary_tier: "Low", "Medium", "High" (using qcut)
            - high_performer: bool (performance_score >= 4)
            - tenure_years: years since hire_date (NaT filled with 0)
    """
    # TODO: Implement this function
    pass

def analyze_data(df_final):
    """
    Analyze cleaned and transformed data.

    Args:
        df_final (pd.DataFrame): Cleaned and transformed data

    Returns:
        dict: Analysis results with keys:
            - avg_salary_by_dept: Series with average salary per department
            - high_performer_count: int count of high performers
            - senior_avg_performance: float average performance for Seniors
            - top_department: str name of department with highest avg salary
    """
    # TODO: Implement this function
    pass

if __name__ == "__main__":
    # Test your functions
    df = pd.read_csv('employees_raw.csv')

    print("=== DATA AUDIT ===")
    audit_report = audit_data(df)
    print(audit_report)

    print("\n=== DATA CLEANING ===")
    df_clean = clean_data(df)
    print(f"Cleaned data shape: {df_clean.shape}")
    print(df_clean.head())

    print("\n=== DATA TRANSFORMATION ===")
    df_final = transform_data(df_clean)
    print(f"Transformed data shape: {df_final.shape}")
    print(df_final.head())

    print("\n=== DATA ANALYSIS ===")
    results = analyze_data(df_final)
    print(results)
