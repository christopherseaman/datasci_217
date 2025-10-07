#!/usr/bin/env python3
"""
Assignment 5: Data Cleaning Exam - Test Suite
Tests for all four functions with partial credit scoring.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from main import audit_data, clean_data, transform_data, analyze_data

@pytest.fixture
def raw_data():
    """Load raw employee data"""
    return pd.read_csv('employees_raw.csv')

# ============================================================================
# Part 1: Data Audit Tests (20 points)
# ============================================================================

def test_audit_returns_dict(raw_data):
    """Test audit_data returns a dictionary (5 points)"""
    result = audit_data(raw_data)
    assert isinstance(result, dict), "audit_data should return a dictionary"
    assert 'total_rows' in result, "Missing 'total_rows' key"
    assert 'total_columns' in result, "Missing 'total_columns' key"
    assert 'missing_count' in result, "Missing 'missing_count' key"
    assert 'duplicate_count' in result, "Missing 'duplicate_count' key"
    assert 'negative_salaries' in result, "Missing 'negative_salaries' key"
    assert 'invalid_dates' in result, "Missing 'invalid_dates' key"

def test_audit_row_col_count(raw_data):
    """Test audit_data counts rows and columns correctly (5 points)"""
    result = audit_data(raw_data)
    assert result['total_rows'] == len(raw_data), "Incorrect row count"
    assert result['total_columns'] == len(raw_data.columns), "Incorrect column count"

def test_audit_missing_values(raw_data):
    """Test audit_data identifies missing values (5 points)"""
    result = audit_data(raw_data)
    assert isinstance(result['missing_count'], dict), "missing_count should be a dict"
    # Check that missing counts are reasonable (some columns should have missing values)
    assert result['missing_count']['salary'] > 0, "Should detect missing salary values"

def test_audit_data_issues(raw_data):
    """Test audit_data identifies duplicates and data issues (5 points)"""
    result = audit_data(raw_data)
    assert result['duplicate_count'] > 0, "Should detect duplicate rows"
    assert result['negative_salaries'] > 0, "Should detect negative salaries"
    assert result['invalid_dates'] > 0, "Should detect UNKNOWN dates"

# ============================================================================
# Part 2: Data Cleaning Tests (40 points)
# ============================================================================

def test_clean_returns_dataframe(raw_data):
    """Test clean_data returns a DataFrame (5 points)"""
    result = clean_data(raw_data)
    assert isinstance(result, pd.DataFrame), "clean_data should return a DataFrame"
    assert len(result) > 0, "Cleaned DataFrame should not be empty"

def test_clean_sentinel_values(raw_data):
    """Test clean_data replaces sentinel values (5 points)"""
    result = clean_data(raw_data)
    # After cleaning, -999 should not exist in numeric columns
    if 'salary' in result.columns:
        assert -999 not in result['salary'].values, "Should replace -999 sentinel values"
    if 'years_experience' in result.columns:
        assert -999 not in result['years_experience'].values, "Should replace -999 sentinel values"

def test_clean_missing_numeric(raw_data):
    """Test clean_data fills missing numeric values (10 points)"""
    result = clean_data(raw_data)
    # Salary and experience should have no NaN after cleaning
    assert result['salary'].isnull().sum() == 0, "Should fill missing salary values"
    assert result['years_experience'].isnull().sum() == 0, "Should fill missing experience values"
    # Performance score should be filled with 3.0 for missing values
    assert result['performance_score'].isnull().sum() == 0, "Should fill missing performance scores"

def test_clean_text_standardization(raw_data):
    """Test clean_data standardizes text columns (10 points)"""
    result = clean_data(raw_data)
    # Name should be Title Case
    assert all(name == name.strip().title() for name in result['name']), "Names should be Title Case"
    # Department should be Title Case
    assert all(dept == dept.strip().title() for dept in result['department']), "Departments should be Title Case"
    # Status should be lowercase
    assert all(status == status.strip().lower() for status in result['status']), "Status should be lowercase"

def test_clean_dates(raw_data):
    """Test clean_data converts dates to datetime (5 points)"""
    result = clean_data(raw_data)
    assert pd.api.types.is_datetime64_any_dtype(result['hire_date']), "hire_date should be datetime type"

def test_clean_outliers_removed(raw_data):
    """Test clean_data removes salary outliers (5 points)"""
    result = clean_data(raw_data)
    # No negative salaries
    assert (result['salary'] < 0).sum() == 0, "Should remove negative salaries"
    # No salaries over 500k
    assert (result['salary'] > 500000).sum() == 0, "Should remove salaries > 500000"

def test_clean_duplicates_removed(raw_data):
    """Test clean_data removes duplicates (5 points)"""
    result = clean_data(raw_data)
    assert result.duplicated().sum() == 0, "Should remove duplicate rows"

# ============================================================================
# Part 3: Data Transformation Tests (25 points)
# ============================================================================

def test_transform_returns_dataframe(raw_data):
    """Test transform_data returns a DataFrame with new columns (5 points)"""
    cleaned = clean_data(raw_data)
    result = transform_data(cleaned)
    assert isinstance(result, pd.DataFrame), "transform_data should return a DataFrame"
    assert 'experience_level' in result.columns, "Should add experience_level column"
    assert 'salary_tier' in result.columns, "Should add salary_tier column"
    assert 'high_performer' in result.columns, "Should add high_performer column"
    assert 'tenure_years' in result.columns, "Should add tenure_years column"

def test_transform_experience_categories(raw_data):
    """Test transform_data creates experience categories correctly (7 points)"""
    cleaned = clean_data(raw_data)
    result = transform_data(cleaned)
    # Check categories exist
    valid_levels = {'Junior', 'Mid', 'Senior'}
    assert set(result['experience_level'].unique()).issubset(valid_levels), \
        "experience_level should only contain 'Junior', 'Mid', 'Senior'"
    # Check logic (0-2: Junior, 2-5: Mid, 5+: Senior)
    for idx, row in result.iterrows():
        if row['years_experience'] < 2:
            assert row['experience_level'] == 'Junior', "Years < 2 should be Junior"
        elif row['years_experience'] < 5:
            assert row['experience_level'] == 'Mid', "Years 2-5 should be Mid"
        else:
            assert row['experience_level'] == 'Senior', "Years 5+ should be Senior"

def test_transform_salary_tiers(raw_data):
    """Test transform_data creates salary tiers using qcut (7 points)"""
    cleaned = clean_data(raw_data)
    result = transform_data(cleaned)
    # Check categories exist
    valid_tiers = {'Low', 'Medium', 'High'}
    assert set(result['salary_tier'].unique()).issubset(valid_tiers), \
        "salary_tier should only contain 'Low', 'Medium', 'High'"
    # Check distribution is roughly equal (qcut creates equal-frequency bins)
    tier_counts = result['salary_tier'].value_counts()
    assert len(tier_counts) == 3, "Should have 3 salary tiers"

def test_transform_high_performer(raw_data):
    """Test transform_data creates high_performer boolean column (6 points)"""
    cleaned = clean_data(raw_data)
    result = transform_data(cleaned)
    # Check it's boolean
    assert result['high_performer'].dtype == bool, "high_performer should be boolean"
    # Check logic
    for idx, row in result.iterrows():
        expected = row['performance_score'] >= 4
        assert row['high_performer'] == expected, "high_performer should be True if performance_score >= 4"

def test_transform_tenure_years(raw_data):
    """Test transform_data calculates tenure correctly (5 points)"""
    cleaned = clean_data(raw_data)
    result = transform_data(cleaned)
    # Tenure should be numeric
    assert pd.api.types.is_numeric_dtype(result['tenure_years']), "tenure_years should be numeric"
    # No NaN values (should fill with 0)
    assert result['tenure_years'].isnull().sum() == 0, "tenure_years should have no NaN"
    # All values should be reasonable (0 to ~10 years)
    assert result['tenure_years'].min() >= 0, "tenure_years should be >= 0"

# ============================================================================
# Part 4: Data Analysis Tests (15 points)
# ============================================================================

def test_analyze_returns_dict(raw_data):
    """Test analyze_data returns a dictionary (3 points)"""
    cleaned = clean_data(raw_data)
    transformed = transform_data(cleaned)
    result = analyze_data(transformed)
    assert isinstance(result, dict), "analyze_data should return a dictionary"
    assert 'avg_salary_by_dept' in result, "Missing 'avg_salary_by_dept' key"
    assert 'high_performer_count' in result, "Missing 'high_performer_count' key"
    assert 'senior_avg_performance' in result, "Missing 'senior_avg_performance' key"
    assert 'top_department' in result, "Missing 'top_department' key"

def test_analyze_avg_salary_by_dept(raw_data):
    """Test analyze_data calculates average salary by department (5 points)"""
    cleaned = clean_data(raw_data)
    transformed = transform_data(cleaned)
    result = analyze_data(transformed)
    # Should be a Series
    assert isinstance(result['avg_salary_by_dept'], pd.Series), "avg_salary_by_dept should be a Series"
    # Should have multiple departments
    assert len(result['avg_salary_by_dept']) > 1, "Should have multiple departments"
    # Values should be positive numbers
    assert all(result['avg_salary_by_dept'] > 0), "Average salaries should be positive"

def test_analyze_high_performer_count(raw_data):
    """Test analyze_data counts high performers (3 points)"""
    cleaned = clean_data(raw_data)
    transformed = transform_data(cleaned)
    result = analyze_data(transformed)
    # Should be an int
    assert isinstance(result['high_performer_count'], (int, np.integer)), "high_performer_count should be int"
    # Should be positive
    assert result['high_performer_count'] > 0, "Should have some high performers"

def test_analyze_senior_avg_performance(raw_data):
    """Test analyze_data calculates senior average performance (4 points)"""
    cleaned = clean_data(raw_data)
    transformed = transform_data(cleaned)
    result = analyze_data(transformed)
    # Should be a float
    assert isinstance(result['senior_avg_performance'], (float, np.floating)), \
        "senior_avg_performance should be float"
    # Should be between 1 and 5 (valid performance score range)
    assert 1 <= result['senior_avg_performance'] <= 5, "Performance score should be between 1 and 5"

def test_analyze_top_department(raw_data):
    """Test analyze_data identifies top department by salary (3 points)"""
    cleaned = clean_data(raw_data)
    transformed = transform_data(cleaned)
    result = analyze_data(transformed)
    # Should be a string
    assert isinstance(result['top_department'], str), "top_department should be a string"
    # Should be one of the departments in the data
    assert result['top_department'] in transformed['department'].unique(), \
        "top_department should be a valid department name"

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
