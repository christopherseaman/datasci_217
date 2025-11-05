#!/usr/bin/env python3
"""
DataSci 217 - Assignment 8 Test Suite
Data Aggregation and Group Operations

Tests for validating assignment completion by checking output files.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Define base directory
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"

class TestQuestion1:
    """Test Question 1: Basic GroupBy Operations"""
    
    def test_q1_groupby_analysis_exists(self):
        """Test that q1_groupby_analysis.csv exists"""
        filepath = OUTPUT_DIR / "q1_groupby_analysis.csv"
        assert filepath.exists(), f"q1_groupby_analysis.csv not found at {filepath}"
    
    def test_q1_groupby_analysis_readable(self):
        """Test that q1_groupby_analysis.csv is readable and has data"""
        filepath = OUTPUT_DIR / "q1_groupby_analysis.csv"
        df = pd.read_csv(filepath)
        assert len(df) > 0, "q1_groupby_analysis.csv is empty"
        assert 'facility_name' in df.columns, "q1_groupby_analysis.csv missing 'facility_name' column"
    
    def test_q1_aggregation_report_exists(self):
        """Test that q1_aggregation_report.txt exists"""
        filepath = OUTPUT_DIR / "q1_aggregation_report.txt"
        assert filepath.exists(), f"q1_aggregation_report.txt not found at {filepath}"
    
    def test_q1_aggregation_report_has_content(self):
        """Test that q1_aggregation_report.txt has content"""
        filepath = OUTPUT_DIR / "q1_aggregation_report.txt"
        with open(filepath, 'r') as f:
            content = f.read()
        assert len(content) > 0, "q1_aggregation_report.txt is empty"
        assert "Assignment 8" in content or "Question 1" in content, "q1_aggregation_report.txt missing expected content"

class TestQuestion2:
    """Test Question 2: Advanced GroupBy Operations"""
    
    def test_q2_filter_analysis_exists(self):
        """Test that q2_filter_analysis.csv exists"""
        filepath = OUTPUT_DIR / "q2_filter_analysis.csv"
        assert filepath.exists(), f"q2_filter_analysis.csv not found at {filepath}"
    
    def test_q2_filter_analysis_readable(self):
        """Test that q2_filter_analysis.csv is readable and has data"""
        filepath = OUTPUT_DIR / "q2_filter_analysis.csv"
        df = pd.read_csv(filepath)
        assert len(df) > 0, "q2_filter_analysis.csv is empty"
    
    def test_q2_hierarchical_analysis_exists(self):
        """Test that q2_hierarchical_analysis.csv exists"""
        filepath = OUTPUT_DIR / "q2_hierarchical_analysis.csv"
        assert filepath.exists(), f"q2_hierarchical_analysis.csv not found at {filepath}"
    
    def test_q2_hierarchical_analysis_readable(self):
        """Test that q2_hierarchical_analysis.csv is readable and has data"""
        filepath = OUTPUT_DIR / "q2_hierarchical_analysis.csv"
        df = pd.read_csv(filepath)
        assert len(df) > 0, "q2_hierarchical_analysis.csv is empty"
        # Should have facility_type and region columns for hierarchical grouping
        assert 'facility_type' in df.columns or 'region' in df.columns, \
            "q2_hierarchical_analysis.csv missing expected hierarchical columns"
    
    def test_q2_performance_report_exists(self):
        """Test that q2_performance_report.txt exists"""
        filepath = OUTPUT_DIR / "q2_performance_report.txt"
        assert filepath.exists(), f"q2_performance_report.txt not found at {filepath}"
    
    def test_q2_performance_report_has_content(self):
        """Test that q2_performance_report.txt has content"""
        filepath = OUTPUT_DIR / "q2_performance_report.txt"
        with open(filepath, 'r') as f:
            content = f.read()
        assert len(content) > 0, "q2_performance_report.txt is empty"
        assert "Assignment 8" in content or "Question 2" in content, \
            "q2_performance_report.txt missing expected content"

class TestQuestion3:
    """Test Question 3: Pivot Tables and Cross-Tabulations"""
    
    def test_q3_pivot_analysis_exists(self):
        """Test that q3_pivot_analysis.csv exists"""
        filepath = OUTPUT_DIR / "q3_pivot_analysis.csv"
        assert filepath.exists(), f"q3_pivot_analysis.csv not found at {filepath}"
    
    def test_q3_pivot_analysis_readable(self):
        """Test that q3_pivot_analysis.csv is readable and has data"""
        filepath = OUTPUT_DIR / "q3_pivot_analysis.csv"
        df = pd.read_csv(filepath, index_col=0)  # Pivot tables often have index
        assert len(df) > 0, "q3_pivot_analysis.csv is empty"
        assert len(df.columns) > 0, "q3_pivot_analysis.csv has no columns"
    
    def test_q3_crosstab_analysis_exists(self):
        """Test that q3_crosstab_analysis.csv exists"""
        filepath = OUTPUT_DIR / "q3_crosstab_analysis.csv"
        assert filepath.exists(), f"q3_crosstab_analysis.csv not found at {filepath}"
    
    def test_q3_crosstab_analysis_readable(self):
        """Test that q3_crosstab_analysis.csv is readable and has data"""
        filepath = OUTPUT_DIR / "q3_crosstab_analysis.csv"
        df = pd.read_csv(filepath, index_col=0)  # Crosstab often has index
        assert len(df) > 0, "q3_crosstab_analysis.csv is empty"
        assert len(df.columns) > 0, "q3_crosstab_analysis.csv has no columns"
    
    def test_q3_pivot_visualization_exists(self):
        """Test that q3_pivot_visualization.png exists"""
        filepath = OUTPUT_DIR / "q3_pivot_visualization.png"
        assert filepath.exists(), f"q3_pivot_visualization.png not found at {filepath}"
    
    def test_q3_pivot_visualization_not_empty(self):
        """Test that q3_pivot_visualization.png is not empty"""
        filepath = OUTPUT_DIR / "q3_pivot_visualization.png"
        assert filepath.stat().st_size > 0, "q3_pivot_visualization.png is empty"

class TestOutputDirectory:
    """Test that output directory structure is correct"""
    
    def test_output_directory_exists(self):
        """Test that output directory exists"""
        assert OUTPUT_DIR.exists(), f"output directory not found at {OUTPUT_DIR}"
        assert OUTPUT_DIR.is_dir(), f"{OUTPUT_DIR} is not a directory"
    
    def test_all_required_files_exist(self):
        """Test that all required output files exist"""
        required_files = [
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
        for filename in required_files:
            filepath = OUTPUT_DIR / filename
            if not filepath.exists():
                missing_files.append(filename)
        
        assert len(missing_files) == 0, \
            f"Missing required files: {', '.join(missing_files)}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
