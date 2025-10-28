"""
Assignment 7 Tests - Data Visualization

Tests verify that students correctly:
- Create matplotlib visualizations with proper formatting
- Generate seaborn statistical plots
- Use pandas plotting for data exploration
- Apply visualization best practices
- Save required output files
"""

import pytest
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


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
    required_files = ["sales_data.csv", "customer_data.csv", "product_data.csv"]

    for filename in required_files:
        filepath = data_dir / filename
        assert filepath.exists(), f"Data file not found: {filepath}"

        # Verify not empty
        df = pd.read_csv(filepath)
        assert len(df) > 0, f"Data file is empty: {filepath}"


def test_q1_matplotlib_plots(output_dir):
    """Test Question 1: matplotlib plots output."""
    output_file = output_dir / "q1_matplotlib_plots.png"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Verify file is not empty
    assert output_file.stat().st_size > 0, "Plot file is empty"


def test_q1_multi_panel(output_dir):
    """Test Question 1: Multi-panel visualization output."""
    output_file = output_dir / "q1_multi_panel.png"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Verify file is not empty
    assert output_file.stat().st_size > 0, "Plot file is empty"


def test_q2_seaborn_plots(output_dir):
    """Test Question 2: seaborn plots output."""
    output_file = output_dir / "q2_seaborn_plots.png"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Verify file is not empty
    assert output_file.stat().st_size > 0, "Plot file is empty"


def test_q2_correlation_heatmap(output_dir):
    """Test Question 2: Correlation heatmap output."""
    output_file = output_dir / "q2_correlation_heatmap.png"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Verify file is not empty
    assert output_file.stat().st_size > 0, "Plot file is empty"


def test_q3_pandas_plots(output_dir):
    """Test Question 3: pandas plots output."""
    output_file = output_dir / "q3_pandas_plots.png"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Verify file is not empty
    assert output_file.stat().st_size > 0, "Plot file is empty"


def test_q3_data_overview(output_dir):
    """Test Question 3: Data overview output."""
    output_file = output_dir / "q3_data_overview.png"
    assert output_file.exists(), f"Output file not found: {output_file}"

    # Verify file is not empty
    assert output_file.stat().st_size > 0, "Plot file is empty"




def test_all_required_outputs(output_dir):
    """Test that all required output files exist."""
    required_outputs = [
        "q1_matplotlib_plots.png",
        "q1_multi_panel.png",
        "q2_seaborn_plots.png",
        "q2_correlation_heatmap.png",
        "q3_pandas_plots.png",
        "q3_data_overview.png"
    ]

    missing_files = []
    for filename in required_outputs:
        filepath = output_dir / filename
        if not filepath.exists():
            missing_files.append(str(filepath))

    assert len(missing_files) == 0, \
        f"Missing required output files: {', '.join(missing_files)}"


def test_plot_file_sizes(output_dir):
    """Test that plot files are reasonable sizes (not too small or too large)."""
    plot_files = [
        "q1_matplotlib_plots.png",
        "q1_multi_panel.png",
        "q2_seaborn_plots.png",
        "q2_correlation_heatmap.png",
        "q3_pandas_plots.png",
        "q3_data_overview.png"
    ]

    for filename in plot_files:
        filepath = output_dir / filename
        if filepath.exists():
            file_size = filepath.stat().st_size
            # Check that file is not too small (at least 1KB) and not too large (less than 10MB)
            assert file_size > 1024, f"Plot file {filename} is too small: {file_size} bytes"
            assert file_size < 10 * 1024 * 1024, f"Plot file {filename} is too large: {file_size} bytes"