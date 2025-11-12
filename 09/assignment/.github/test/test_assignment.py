#!/usr/bin/env python3
"""
DataSci 217 - Assignment 9 Test Suite
Time Series Analysis

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
    """Test Question 1: datetime Fundamentals and Time Series Indexing"""

    def test_q1_datetime_analysis_exists(self):
        """Test that q1_datetime_analysis.csv exists"""
        filepath = OUTPUT_DIR / "q1_datetime_analysis.csv"
        assert filepath.exists(), (
            f"q1_datetime_analysis.csv not found at {filepath}"
        )

    def test_q1_datetime_analysis_readable(self):
        """Test that q1_datetime_analysis.csv is readable and has data"""
        filepath = OUTPUT_DIR / "q1_datetime_analysis.csv"
        df = pd.read_csv(filepath)
        assert len(df) > 0, "q1_datetime_analysis.csv is empty"
        # Should have datetime-related columns
        assert len(df.columns) > 0, "q1_datetime_analysis.csv has no columns"

    def test_q1_timezone_report_exists(self):
        """Test that q1_timezone_report.txt exists"""
        filepath = OUTPUT_DIR / "q1_timezone_report.txt"
        assert filepath.exists(), (
            f"q1_timezone_report.txt not found at {filepath}"
        )

    def test_q1_timezone_report_has_content(self):
        """Test that q1_timezone_report.txt has content"""
        filepath = OUTPUT_DIR / "q1_timezone_report.txt"
        with open(filepath, "r") as f:
            content = f.read()
        assert len(content) > 0, "q1_timezone_report.txt is empty"
        # Should contain some timezone-related content
        assert len(content.strip()) > 50, (
            "q1_timezone_report.txt seems too short"
        )


class TestQuestion2:
    """Test Question 2: Resampling and Frequency Conversion"""

    def test_q2_resampling_analysis_exists(self):
        """Test that q2_resampling_analysis.csv exists"""
        filepath = OUTPUT_DIR / "q2_resampling_analysis.csv"
        assert filepath.exists(), (
            f"q2_resampling_analysis.csv not found at {filepath}"
        )

    def test_q2_resampling_analysis_readable(self):
        """Test that q2_resampling_analysis.csv is readable and has data"""
        filepath = OUTPUT_DIR / "q2_resampling_analysis.csv"
        df = pd.read_csv(filepath)
        assert len(df) > 0, "q2_resampling_analysis.csv is empty"
        assert len(df.columns) > 0, "q2_resampling_analysis.csv has no columns"

    def test_q2_missing_data_report_exists(self):
        """Test that q2_missing_data_report.txt exists"""
        filepath = OUTPUT_DIR / "q2_missing_data_report.txt"
        assert filepath.exists(), (
            f"q2_missing_data_report.txt not found at {filepath}"
        )

    def test_q2_missing_data_report_has_content(self):
        """Test that q2_missing_data_report.txt has content"""
        filepath = OUTPUT_DIR / "q2_missing_data_report.txt"
        with open(filepath, "r") as f:
            content = f.read()
        assert len(content) > 0, "q2_missing_data_report.txt is empty"
        # Should contain some missing data-related content
        assert len(content.strip()) > 50, (
            "q2_missing_data_report.txt seems too short"
        )


class TestQuestion3:
    """Test Question 3: Rolling Window Operations and Visualization"""

    def test_q3_rolling_analysis_exists(self):
        """Test that q3_rolling_analysis.csv exists"""
        filepath = OUTPUT_DIR / "q3_rolling_analysis.csv"
        assert filepath.exists(), (
            f"q3_rolling_analysis.csv not found at {filepath}"
        )

    def test_q3_rolling_analysis_readable(self):
        """Test that q3_rolling_analysis.csv is readable and has data"""
        filepath = OUTPUT_DIR / "q3_rolling_analysis.csv"
        df = pd.read_csv(filepath)
        assert len(df) > 0, "q3_rolling_analysis.csv is empty"
        assert len(df.columns) > 0, "q3_rolling_analysis.csv has no columns"

    def test_q3_rolling_analysis_has_required_columns(self):
        """Test that q3_rolling_analysis.csv has expected columns from Parts 3.2 and 3.3"""
        filepath = OUTPUT_DIR / "q3_rolling_analysis.csv"
        df = pd.read_csv(filepath)
        # Should have basic rolling stats from Part 3.2
        expected_basic_cols = [
            "rolling_7d_mean",
            "rolling_7d_std",
            "rolling_30d_mean",
        ]
        # Should have advanced stats from Part 3.3
        expected_advanced_cols = ["ewm_span_7", "expanding_mean"]
        # Check that at least some expected columns exist
        basic_found = any(col in df.columns for col in expected_basic_cols)
        advanced_found = any(
            col in df.columns for col in expected_advanced_cols
        )
        assert basic_found, (
            f"q3_rolling_analysis.csv missing basic rolling columns. Found: {df.columns.tolist()}"
        )
        assert advanced_found, (
            f"q3_rolling_analysis.csv missing advanced rolling columns. Found: {df.columns.tolist()}"
        )

    def test_q3_trend_analysis_exists(self):
        """Test that q3_trend_analysis.png exists"""
        filepath = OUTPUT_DIR / "q3_trend_analysis.png"
        assert filepath.exists(), (
            f"q3_trend_analysis.png not found at {filepath}"
        )

    def test_q3_trend_analysis_not_empty(self):
        """Test that q3_trend_analysis.png is not empty"""
        filepath = OUTPUT_DIR / "q3_trend_analysis.png"
        assert filepath.stat().st_size > 0, "q3_trend_analysis.png is empty"

    def test_q3_visualization_exists(self):
        """Test that q3_visualization.png exists (bonus)"""
        filepath = OUTPUT_DIR / "q3_visualization.png"
        assert filepath.exists(), (
            f"q3_visualization.png not found at {filepath}"
        )

    def test_q3_visualization_not_empty(self):
        """Test that q3_visualization.png is not empty"""
        filepath = OUTPUT_DIR / "q3_visualization.png"
        assert filepath.stat().st_size > 0, "q3_visualization.png is empty"


class TestOutputDirectory:
    """Test that output directory structure is correct"""

    def test_output_directory_exists(self):
        """Test that output directory exists"""
        assert OUTPUT_DIR.exists(), (
            f"output directory not found at {OUTPUT_DIR}"
        )
        assert OUTPUT_DIR.is_dir(), f"{OUTPUT_DIR} is not a directory"

    def test_all_required_files_exist(self):
        """Test that all required output files exist"""
        required_files = [
            "q1_datetime_analysis.csv",
            "q1_timezone_report.txt",
            "q2_resampling_analysis.csv",
            "q2_missing_data_report.txt",
            "q3_rolling_analysis.csv",
            "q3_trend_analysis.png",
            "q3_visualization.png",  # Bonus but included in checklist
        ]

        missing_files = []
        for filename in required_files:
            filepath = OUTPUT_DIR / filename
            if not filepath.exists():
                missing_files.append(filename)

        assert len(missing_files) == 0, (
            f"Missing required files: {', '.join(missing_files)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
