#!/usr/bin/env python3
"""
Assignment 5: Midterm Exam - Test Suite
Tests for clinical trial data processing pipeline.
Total: 100 points across behavioral tests
"""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Import functions from Q2 and Q3 modules
from q2_process_metadata import parse_config, validate_config, generate_sample_data, calculate_statistics
from q3_data_utils import (load_data, clean_data, detect_missing, fill_missing, 
                          filter_data, transform_types, create_bins, summarize_by_group)


@pytest.fixture
def clinical_data():
    """Load clinical trial data"""
    return pd.read_csv('data/clinical_trial_raw.csv')

# ============================================================================
# Q1: Project Setup Tests (10 points)
# ============================================================================

def test_q1_script_executable():
    """Test Q1 script is executable (2 points)"""
    assert os.access('q1_setup_project.sh', os.X_OK), "Q1 script should be executable"

def test_q1_shebang():
    """Test Q1 script has shebang (1 point)"""
    with open('q1_setup_project.sh') as f:
        first_line = f.readline().strip()
        assert first_line.startswith('#!/bin/bash'), "Q1 script should have shebang"

def test_q1_directories_exist():
    """Test Q1 creates required directories (3 points)"""
    assert os.path.exists('data/'), "data/ directory should exist"
    assert os.path.exists('output/'), "output/ directory should exist"
    assert os.path.exists('reports/'), "reports/ directory should exist"

def test_q1_dataset_generated():
    """Test Q1 generates dataset (2 points)"""
    assert os.path.exists('data/clinical_trial_raw.csv'), "clinical_trial_raw.csv should exist"

def test_q1_directory_structure():
    """Test Q1 creates directory structure file (2 points)"""
    assert os.path.exists('reports/directory_structure.txt'), "directory_structure.txt should exist"

# ============================================================================
# Q2: Python Data Processing Tests (25 points)
# ============================================================================

def test_q2_script_executable():
    """Test Q2 script is executable (1 point)"""
    assert os.access('q2_process_metadata.py', os.X_OK), "Q2 script should be executable"

def test_q2_shebang():
    """Test Q2 script has shebang (1 point)"""
    with open('q2_process_metadata.py') as f:
        first_line = f.readline().strip()
        assert first_line.startswith('#!/usr/bin/env python3'), "Q2 script should have shebang"

def test_q2_parse_config():
    """Test parse_config function (3 points)"""
    config = parse_config('q2_config.txt')
    assert isinstance(config, dict), "parse_config should return dict"
    assert 'sample_data_rows' in config, "Missing sample_data_rows key"
    assert 'sample_data_min' in config, "Missing sample_data_min key"
    assert 'sample_data_max' in config, "Missing sample_data_max key"

def test_q2_validate_config():
    """Test validate_config function (3 points)"""
    test_config = {'sample_data_rows': '100', 'sample_data_min': '18', 'sample_data_max': '75'}
    validation = validate_config(test_config)
    assert isinstance(validation, dict), "validate_config should return dict"
    assert validation['sample_data_rows'] == True, "sample_data_rows should be valid"
    assert validation['sample_data_min'] == True, "sample_data_min should be valid"

def test_q2_generate_sample_data():
    """Test generate_sample_data function (3 points)"""
    config = {'sample_data_rows': '10', 'sample_data_min': '18', 'sample_data_max': '75'}
    generate_sample_data('test_sample.csv', config)
    assert os.path.exists('test_sample.csv'), "Sample data file should be created"
    with open('test_sample.csv') as f:
        lines = f.readlines()
        assert len(lines) == 10, "Should have 10 rows as specified"

def test_q2_calculate_statistics():
    """Test calculate_statistics function (3 points)"""
    stats = calculate_statistics([10, 20, 30, 40, 50])
    assert stats['mean'] == 30.0, "Mean should be 30.0"
    assert stats['median'] == 30.0, "Median should be 30.0"
    assert stats['sum'] == 150, "Sum should be 150"
    assert stats['count'] == 5, "Count should be 5"

def test_q2_outputs():
    """Test Q2 required outputs (9 points)"""
    assert os.path.exists('data/sample_data.csv'), "sample_data.csv should exist"
    assert os.path.exists('output/statistics.txt'), "statistics.txt should exist"

# ============================================================================
# Q3: Data Utilities Library Tests (20 points)
# ============================================================================

def test_q3_load_data(clinical_data):
    """Test load_data function (2 points)"""
    df = load_data('data/clinical_trial_raw.csv')
    assert isinstance(df, pd.DataFrame), "load_data should return DataFrame"
    assert len(df) > 0, "DataFrame should have data"

def test_q3_clean_data(clinical_data):
    """Test clean_data function (2 points)"""
    cleaned = clean_data(clinical_data, remove_duplicates=True, sentinel_value=-999)
    assert isinstance(cleaned, pd.DataFrame), "clean_data should return DataFrame"

def test_q3_detect_missing(clinical_data):
    """Test detect_missing function (2 points)"""
    missing = detect_missing(clinical_data)
    assert isinstance(missing, pd.Series), "detect_missing should return Series"
    assert len(missing) == len(clinical_data.columns), "Should have one count per column"

def test_q3_fill_missing(clinical_data):
    """Test fill_missing function (2 points)"""
    test_df = pd.DataFrame({'col': [1, np.nan, 3]})
    filled = fill_missing(test_df, 'col', 'mean')
    assert filled['col'].isnull().sum() == 0, "Should fill missing values"

def test_q3_filter_data(clinical_data):
    """Test filter_data function (5 points)"""
    filters = [{'column': 'age', 'condition': 'greater_than', 'value': 65}]
    filtered = filter_data(clinical_data, filters)
    assert all(filtered['age'] > 65), "Should filter correctly"
    
    # Test multiple filters
    filters = [
        {'column': 'age', 'condition': 'greater_than', 'value': 18},
        {'column': 'site', 'condition': 'equals', 'value': 'Site A'}
    ]
    filtered = filter_data(clinical_data, filters)
    assert all(filtered['age'] > 18), "Should apply multiple filters"
    assert all(filtered['site'] == 'Site A'), "Should apply multiple filters"

def test_q3_transform_types(clinical_data):
    """Test transform_types function (2 points)"""
    type_map = {'enrollment_date': 'datetime', 'site': 'category'}
    typed = transform_types(clinical_data, type_map)
    assert isinstance(typed, pd.DataFrame), "transform_types should return DataFrame"

def test_q3_create_bins(clinical_data):
    """Test create_bins function (2 points)"""
    binned = create_bins(clinical_data, 'age', [0, 40, 60, 100], ['<40', '40-59', '60+'])
    assert isinstance(binned, pd.DataFrame), "create_bins should return DataFrame"

def test_q3_summarize_by_group(clinical_data):
    """Test summarize_by_group function (3 points)"""
    summary = summarize_by_group(clinical_data, 'site', {'age': 'mean'})
    assert isinstance(summary, pd.DataFrame), "summarize_by_group should return DataFrame"

# ============================================================================
# Q4-Q7: Notebook Output Tests (55 points)
# ============================================================================

def test_q4_outputs():
    """Test Q4 required outputs (15 points)"""
    assert os.path.exists('output/q4_site_counts.csv'), "q4_site_counts.csv should exist"
    df = pd.read_csv('output/q4_site_counts.csv')
    assert len(df) >= 1, "Should have at least one site"
    assert len(df.columns) >= 2, "Should have site and count columns"

def test_q5_outputs():
    """Test Q5 required outputs (15 points)"""
    assert os.path.exists('output/q5_cleaned_data.csv'), "q5_cleaned_data.csv should exist"
    assert os.path.exists('output/q5_missing_report.txt'), "q5_missing_report.txt should exist"

def test_q6_outputs():
    """Test Q6 required outputs (20 points)"""
    assert os.path.exists('output/q6_transformed_data.csv'), "q6_transformed_data.csv should exist"
    original = pd.read_csv('data/clinical_trial_raw.csv')
    transformed = pd.read_csv('output/q6_transformed_data.csv')
    assert len(transformed.columns) > len(original.columns), "Should have more columns after transformation"

def test_q7_outputs():
    """Test Q7 required outputs (15 points)"""
    assert os.path.exists('output/q7_site_summary.csv'), "q7_site_summary.csv should exist"
    assert os.path.exists('output/q7_intervention_comparison.csv'), "q7_intervention_comparison.csv should exist"
    assert os.path.exists('output/q7_analysis_report.txt'), "q7_analysis_report.txt should exist"
    summary = pd.read_csv('output/q7_site_summary.csv')
    assert len(summary) >= 1, "Should have at least one site"

# ============================================================================
# Q8: Pipeline Automation Tests (5 points)
# ============================================================================

def test_q8_script_executable():
    """Test Q8 script is executable (1 point)"""
    assert os.access('q8_run_pipeline.sh', os.X_OK), "Q8 script should be executable"

def test_q8_shebang():
    """Test Q8 script has shebang (1 point)"""
    with open('q8_run_pipeline.sh') as f:
        first_line = f.readline().strip()
        assert first_line.startswith('#!/bin/bash'), "Q8 script should have shebang"

def test_q8_pipeline_log():
    """Test Q8 creates pipeline log (1 point)"""
    assert os.path.exists('reports/pipeline_log.txt'), "pipeline_log.txt should exist"

def test_q8_pipeline_execution():
    """Test Q8 runs notebooks in order (1 point)"""
    # This would test that the pipeline actually runs, but for now just check log exists
    assert os.path.exists('reports/pipeline_log.txt'), "Pipeline log should exist"

def test_q8_pipeline_success():
    """Test Q8 pipeline shows successful execution (1 point)"""
    if os.path.exists('reports/pipeline_log.txt'):
        with open('reports/pipeline_log.txt') as f:
            content = f.read()
            assert len(content) > 0, "Pipeline log should have content"

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
