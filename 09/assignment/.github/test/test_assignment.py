"""
Assignment 9 Tests - Time Series Analysis

Tests verify that students correctly:
- Handle datetime data types and parsing
- Perform time series indexing and selection
- Use resampling and frequency conversion
- Apply rolling window operations
- Handle time zones and temporal data
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
    required_files = ["stock_prices.csv", "weather_data.csv", "sales_data.csv"]

    for filename in required_files:
        filepath = data_dir / filename
        assert filepath.exists(), f"Data file not found: {filepath}"

        # Verify not empty
        df = pd.read_csv(filepath)
        assert len(df) > 0, f"Data file is empty: {filepath}"


def test_q1_datetime_analysis(output_dir):
    """Test Question 1: datetime analysis output."""
    output_file = output_dir / "q1_datetime_analysis.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate structure
    df = pd.read_csv(output_file)
    assert len(df) > 0, "Datetime analysis data is empty"

    # Should have datetime-related columns
    assert any('date' in col.lower() or 'time' in col.lower() for col in df.columns), \
        "Missing datetime-related columns"


def test_q1_timezone_report(output_dir):
    """Test Question 1: timezone report output."""
    output_file = output_dir / "q1_timezone_report.txt"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Read and check content
    content = output_file.read_text()
    assert len(content) > 0, "Timezone report is empty"

    # Should contain timezone-related content
    assert any(keyword in content.lower() for keyword in ['timezone', 'utc', 'time', 'zone']), \
        "Missing timezone analysis content"


def test_q2_resampling_analysis(output_dir):
    """Test Question 2: resampling analysis output."""
    output_file = output_dir / "q2_resampling_analysis.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate structure
    df = pd.read_csv(output_file)
    assert len(df) > 0, "Resampling analysis data is empty"


def test_q2_missing_data_report(output_dir):
    """Test Question 2: missing data report output."""
    output_file = output_dir / "q2_missing_data_report.txt"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Read and check content
    content = output_file.read_text()
    assert len(content) > 0, "Missing data report is empty"

    # Should contain missing data analysis
    assert any(keyword in content.lower() for keyword in ['missing', 'null', 'na', 'data']), \
        "Missing data analysis content"


def test_q3_rolling_analysis(output_dir):
    """Test Question 3: rolling analysis output."""
    output_file = output_dir / "q3_rolling_analysis.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate structure
    df = pd.read_csv(output_file)
    assert len(df) > 0, "Rolling analysis data is empty"

    # Should have rolling-related columns
    assert any('rolling' in col.lower() or 'moving' in col.lower() or 'ewm' in col.lower() for col in df.columns), \
        "Missing rolling analysis columns"


def test_q3_trend_analysis(output_dir):
    """Test Question 3: trend analysis visualization output."""
    output_file = output_dir / "q3_trend_analysis.png"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Verify file is not empty
    assert output_file.stat().st_size > 0, "Trend analysis visualization is empty"


def test_q4_visualization(output_dir):
    """Test Question 4: time series visualization output."""
    output_file = output_dir / "q4_visualization.png"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Verify file is not empty
    assert output_file.stat().st_size > 0, "Time series visualization is empty"


def test_q4_seasonal_analysis(output_dir):
    """Test Question 4: seasonal analysis output."""
    output_file = output_dir / "q4_seasonal_analysis.csv"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Load and validate structure
    df = pd.read_csv(output_file)
    assert len(df) > 0, "Seasonal analysis data is empty"


def test_q4_automation_report(output_dir):
    """Test Question 4: automation report output."""
    output_file = output_dir / "q4_automation_report.txt"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Read and check content
    content = output_file.read_text()
    assert len(content) > 0, "Automation report is empty"

    # Should contain automation-related content
    assert any(keyword in content.lower() for keyword in ['automation', 'cron', 'schedule', 'workflow']), \
        "Missing automation analysis content"


def test_all_required_outputs(output_dir):
    """Test that all required output files exist."""
    required_outputs = [
        "q1_datetime_analysis.csv",
        "q1_timezone_report.txt",
        "q2_resampling_analysis.csv",
        "q2_missing_data_report.txt",
        "q3_rolling_analysis.csv",
        "q3_trend_analysis.png",
        "q4_visualization.png",
        "q4_seasonal_analysis.csv",
        "q4_automation_report.txt"
    ]

    missing_files = []
    for filename in required_outputs:
        filepath = output_dir / filename
        if not filepath.exists():
            missing_files.append(str(filepath))

    assert len(missing_files) == 0, \
        f"Missing required output files: {', '.join(missing_files)}"


def test_csv_file_validation(output_dir):
    """Test that CSV files are properly formatted."""
    csv_files = [
        "q1_datetime_analysis.csv",
        "q2_resampling_analysis.csv",
        "q3_rolling_analysis.csv",
        "q4_seasonal_analysis.csv"
    ]

    for filename in csv_files:
        filepath = output_dir / filename
        if filepath.exists():
            # Try to read the CSV file
            try:
                df = pd.read_csv(filepath)
                assert len(df) > 0, f"CSV file {filename} is empty"
            except Exception as e:
                pytest.fail(f"Error reading CSV file {filename}: {e}")


def test_text_file_validation(output_dir):
    """Test that text files contain meaningful content."""
    text_files = [
        "q1_timezone_report.txt",
        "q2_missing_data_report.txt",
        "q4_automation_report.txt"
    ]

    for filename in text_files:
        filepath = output_dir / filename
        if filepath.exists():
            content = filepath.read_text()
            assert len(content.strip()) > 50, f"Text file {filename} is too short or empty"


def test_plot_file_sizes(output_dir):
    """Test that plot files are reasonable sizes."""
    plot_files = [
        "q3_trend_analysis.png",
        "q4_visualization.png"
    ]

    for filename in plot_files:
        filepath = output_dir / filename
        if filepath.exists():
            file_size = filepath.stat().st_size
            # Check that file is not too small (at least 1KB) and not too large (less than 10MB)
            assert file_size > 1024, f"Plot file {filename} is too small: {file_size} bytes"
            assert file_size < 10 * 1024 * 1024, f"Plot file {filename} is too large: {file_size} bytes"