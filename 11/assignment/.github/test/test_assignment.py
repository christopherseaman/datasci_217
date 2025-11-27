"""
Auto-grading tests for Assignment 11: Chicago Beach Weather Sensors Analysis
Complete 9-phase data science workflow
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add assignment directory to path
assignment_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(assignment_dir))


@pytest.fixture(scope="module")
def setup_data():
    """Setup: Change to assignment directory and ensure output directory exists."""
    original_dir = os.getcwd()
    os.chdir(assignment_dir)
    os.makedirs("output", exist_ok=True)
    yield
    os.chdir(original_dir)


# ============================================================================
# Q1: Setup & Exploration
# ============================================================================


def test_q1_data_info_exists(setup_data):
    """Test that Q1 data info file exists."""
    assert os.path.exists("output/q1_data_info.txt"), (
        "q1_data_info.txt not found in output/ directory"
    )


def test_q1_data_info_content(setup_data):
    """Test that Q1 data info contains required information."""
    with open("output/q1_data_info.txt", "r") as f:
        content = f.read()

    # Check for key information
    assert (
        "shape" in content.lower()
        or "rows" in content.lower()
        or "columns" in content.lower()
    ), "Data info should contain shape/rows/columns information"
    assert (
        "date range" in content.lower()
        or "start" in content.lower()
        or "end" in content.lower()
    ), "Data info should contain date range information"


def test_q1_exploration_csv_exists(setup_data):
    """Test that Q1 exploration CSV file exists."""
    assert os.path.exists("output/q1_exploration.csv"), (
        "q1_exploration.csv not found in output/ directory"
    )


def test_q1_exploration_csv_format(setup_data):
    """Test that Q1 exploration CSV has correct format."""
    df = pd.read_csv("output/q1_exploration.csv")

    # Check required columns
    required_cols = [
        "column_name",
        "mean",
        "std",
        "min",
        "max",
        "missing_count",
    ]
    assert all(col in df.columns for col in required_cols), (
        f"CSV must contain columns: {required_cols}. Found: {list(df.columns)}"
    )

    # Check data types
    assert df["mean"].dtype in [np.float64, np.float32, float], (
        "mean must be numeric"
    )
    assert df["std"].dtype in [np.float64, np.float32, float], (
        "std must be numeric"
    )
    assert df["min"].dtype in [np.float64, np.float32, float], (
        "min must be numeric"
    )
    assert df["max"].dtype in [np.float64, np.float32, float], (
        "max must be numeric"
    )
    assert df["missing_count"].dtype in [np.int64, np.int32, int], (
        "missing_count must be integer"
    )

    # Check reasonable number of rows (at least 1 numeric column)
    assert len(df) >= 1, f"Expected at least 1 numeric column, got {len(df)}"


def test_q1_visualizations_exists(setup_data):
    """Test that Q1 visualizations file exists."""
    assert os.path.exists("output/q1_visualizations.png"), (
        "q1_visualizations.png not found in output/ directory"
    )


# ============================================================================
# Q2: Data Cleaning
# ============================================================================


def test_q2_cleaned_data_exists(setup_data):
    """Test that Q2 cleaned data CSV file exists."""
    assert os.path.exists("output/q2_cleaned_data.csv"), (
        "q2_cleaned_data.csv not found in output/ directory"
    )


def test_q2_cleaned_data_format(setup_data):
    """Test that Q2 cleaned data has reasonable format."""
    df = pd.read_csv("output/q2_cleaned_data.csv")

    # Check reasonable number of rows (should be similar to original, not empty)
    assert len(df) > 0, "Cleaned data should have at least some rows"
    assert len(df) >= 1000, f"Cleaned data seems too small: {len(df)} rows"

    # Check that numeric columns don't have extreme outliers (basic sanity check)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:3]:  # Check first 3 numeric columns
            if df[col].notna().sum() > 0:
                # Check for reasonable range (not all NaN, not all same value)
                unique_vals = df[col].nunique()
                assert unique_vals > 1 or df[col].isna().all(), (
                    f"Column {col} has only one unique value (may indicate cleaning issue)"
                )


def test_q2_cleaning_report_exists(setup_data):
    """Test that Q2 cleaning report file exists."""
    assert os.path.exists("output/q2_cleaning_report.txt"), (
        "q2_cleaning_report.txt not found in output/ directory"
    )


def test_q2_cleaning_report_content(setup_data):
    """Test that Q2 cleaning report contains required information."""
    with open("output/q2_cleaning_report.txt", "r") as f:
        content = f.read()

    # Check for key information
    assert (
        "missing" in content.lower()
        or "null" in content.lower()
        or "na" in content.lower()
    ), "Cleaning report should mention missing data handling"
    assert "row" in content.lower(), "Cleaning report should mention row counts"


def test_q2_rows_cleaned_exists(setup_data):
    """Test that Q2 rows cleaned file exists."""
    assert os.path.exists("output/q2_rows_cleaned.txt"), (
        "q2_rows_cleaned.txt not found in output/ directory"
    )


def test_q2_rows_cleaned_format(setup_data):
    """Test that Q2 rows cleaned contains a single number."""
    with open("output/q2_rows_cleaned.txt", "r") as f:
        content = f.read().strip()

    # Should be a single number
    try:
        row_count = int(content)
        assert row_count > 0, f"Row count should be positive, got {row_count}"
    except ValueError:
        pytest.fail(
            f"q2_rows_cleaned.txt should contain a single integer, got: {content}"
        )


# ============================================================================
# Q3: Data Wrangling
# ============================================================================


def test_q3_wrangled_data_exists(setup_data):
    """Test that Q3 wrangled data CSV file exists."""
    assert os.path.exists("output/q3_wrangled_data.csv"), (
        "q3_wrangled_data.csv not found in output/ directory"
    )


def test_q3_temporal_features_exists(setup_data):
    """Test that Q3 temporal features CSV file exists."""
    assert os.path.exists("output/q3_temporal_features.csv"), (
        "q3_temporal_features.csv not found in output/ directory"
    )


def test_q3_temporal_features_format(setup_data):
    """Test that Q3 temporal features has required columns."""
    df = pd.read_csv("output/q3_temporal_features.csv")

    # Check required columns
    required_cols = ["hour", "day_of_week", "month"]
    assert all(col in df.columns for col in required_cols), (
        f"Temporal features must contain columns: {required_cols}. Found: {list(df.columns)}"
    )

    # Check data types and ranges
    assert df["hour"].dtype in [np.int64, np.int32, int], "hour must be integer"
    assert df["hour"].min() >= 0 and df["hour"].max() <= 23, (
        f"hour must be between 0-23, got range [{df['hour'].min()}, {df['hour'].max()}]"
    )

    assert df["day_of_week"].dtype in [np.int64, np.int32, int], (
        "day_of_week must be integer"
    )
    assert df["day_of_week"].min() >= 0 and df["day_of_week"].max() <= 6, (
        f"day_of_week must be between 0-6, got range [{df['day_of_week'].min()}, {df['day_of_week'].max()}]"
    )

    assert df["month"].dtype in [np.int64, np.int32, int], (
        "month must be integer"
    )
    assert df["month"].min() >= 1 and df["month"].max() <= 12, (
        f"month must be between 1-12, got range [{df['month'].min()}, {df['month'].max()}]"
    )


def test_q3_datetime_info_exists(setup_data):
    """Test that Q3 datetime info file exists."""
    assert os.path.exists("output/q3_datetime_info.txt"), (
        "q3_datetime_info.txt not found in output/ directory"
    )


def test_q3_datetime_info_content(setup_data):
    """Test that Q3 datetime info contains date range."""
    with open("output/q3_datetime_info.txt", "r") as f:
        content = f.read()

    # Check for date range information
    assert (
        "start" in content.lower()
        or "end" in content.lower()
        or "date" in content.lower()
    ), "Datetime info should contain date range (start/end dates)"


# ============================================================================
# Q4: Feature Engineering
# ============================================================================


def test_q4_features_exists(setup_data):
    """Test that Q4 features CSV file exists."""
    assert os.path.exists("output/q4_features.csv"), (
        "q4_features.csv not found in output/ directory"
    )


def test_q4_rolling_features_exists(setup_data):
    """Test that Q4 rolling features CSV file exists."""
    assert os.path.exists("output/q4_rolling_features.csv"), (
        "q4_rolling_features.csv not found in output/ directory"
    )


def test_q4_rolling_features_format(setup_data):
    """Test that Q4 rolling features contains rolling window calculations."""
    df = pd.read_csv("output/q4_rolling_features.csv")

    # Check for rolling window columns (should have names like "*rolling*" or similar)
    rolling_cols = [
        col
        for col in df.columns
        if "rolling" in col.lower()
        or "moving" in col.lower()
        or "window" in col.lower()
    ]
    assert len(rolling_cols) >= 1, (
        f"Rolling features should contain at least one rolling window column. Found columns: {list(df.columns)}"
    )

    # Check that rolling values are numeric
    for col in rolling_cols:
        assert df[col].dtype in [np.float64, np.float32, float], (
            f"Rolling column {col} must be numeric"
        )


def test_q4_feature_list_exists(setup_data):
    """Test that Q4 feature list file exists."""
    assert os.path.exists("output/q4_feature_list.txt"), (
        "q4_feature_list.txt not found in output/ directory"
    )


def test_q4_feature_list_content(setup_data):
    """Test that Q4 feature list contains feature names."""
    with open("output/q4_feature_list.txt", "r") as f:
        content = f.read().strip()

    # Should have at least one feature name
    assert len(content) > 0, (
        "Feature list should contain at least one feature name"
    )

    # Should have multiple lines (features)
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    assert len(lines) >= 1, (
        f"Feature list should have at least 1 feature, got {len(lines)}"
    )


# ============================================================================
# Q5: Pattern Analysis
# ============================================================================


def test_q5_correlations_exists(setup_data):
    """Test that Q5 correlations CSV file exists."""
    assert os.path.exists("output/q5_correlations.csv"), (
        "q5_correlations.csv not found in output/ directory"
    )


def test_q5_correlations_format(setup_data):
    """Test that Q5 correlations is a valid correlation matrix."""
    df = pd.read_csv("output/q5_correlations.csv", index_col=0)

    # Should be square matrix
    assert df.shape[0] == df.shape[1], (
        f"Correlation matrix should be square, got shape {df.shape}"
    )

    # Values should be between -1 and 1
    for col in df.columns:
        assert df[col].min() >= -1.1 and df[col].max() <= 1.1, (
            f"Correlation values should be between -1 and 1, got range [{df[col].min()}, {df[col].max()}] for column {col}"
        )


def test_q5_patterns_exists(setup_data):
    """Test that Q5 patterns visualization file exists."""
    assert os.path.exists("output/q5_patterns.png"), (
        "q5_patterns.png not found in output/ directory"
    )


def test_q5_trend_summary_exists(setup_data):
    """Test that Q5 trend summary file exists."""
    assert os.path.exists("output/q5_trend_summary.txt"), (
        "q5_trend_summary.txt not found in output/ directory"
    )


def test_q5_trend_summary_content(setup_data):
    """Test that Q5 trend summary contains pattern information."""
    with open("output/q5_trend_summary.txt", "r") as f:
        content = f.read()

    # Check for key terms
    assert (
        "trend" in content.lower()
        or "pattern" in content.lower()
        or "seasonal" in content.lower()
        or "correlation" in content.lower()
    ), (
        "Trend summary should mention trends, patterns, seasonality, or correlations"
    )


# ============================================================================
# Q6: Modeling Preparation
# ============================================================================


def test_q6_X_train_exists(setup_data):
    """Test that Q6 X_train CSV file exists."""
    assert os.path.exists("output/q6_X_train.csv"), (
        "q6_X_train.csv not found in output/ directory"
    )


def test_q6_X_test_exists(setup_data):
    """Test that Q6 X_test CSV file exists."""
    assert os.path.exists("output/q6_X_test.csv"), (
        "q6_X_test.csv not found in output/ directory"
    )


def test_q6_y_train_exists(setup_data):
    """Test that Q6 y_train CSV file exists."""
    assert os.path.exists("output/q6_y_train.csv"), (
        "q6_y_train.csv not found in output/ directory"
    )


def test_q6_y_test_exists(setup_data):
    """Test that Q6 y_test CSV file exists."""
    assert os.path.exists("output/q6_y_test.csv"), (
        "q6_y_test.csv not found in output/ directory"
    )


def test_q6_train_test_split_format(setup_data):
    """Test that Q6 train/test split has correct format."""
    X_train = pd.read_csv("output/q6_X_train.csv")
    X_test = pd.read_csv("output/q6_X_test.csv")
    y_train = pd.read_csv("output/q6_y_train.csv")
    y_test = pd.read_csv("output/q6_y_test.csv")

    # Check shapes match
    assert len(X_train) == len(y_train), (
        f"X_train and y_train should have same length, got {len(X_train)} and {len(y_train)}"
    )
    assert len(X_test) == len(y_test), (
        f"X_test and y_test should have same length, got {len(X_test)} and {len(y_test)}"
    )

    # Check reasonable split (train should be larger than test)
    assert len(X_train) > len(X_test), (
        f"Training set should be larger than test set, got train={len(X_train)}, test={len(X_test)}"
    )

    # Check y has single column
    assert y_train.shape[1] == 1, (
        f"y_train should have single column, got {y_train.shape[1]} columns"
    )
    assert y_test.shape[1] == 1, (
        f"y_test should have single column, got {y_test.shape[1]} columns"
    )


def test_q6_train_test_info_exists(setup_data):
    """Test that Q6 train/test info file exists."""
    assert os.path.exists("output/q6_train_test_info.txt"), (
        "q6_train_test_info.txt not found in output/ directory"
    )


def test_q6_train_test_info_content(setup_data):
    """Test that Q6 train/test info contains split information."""
    with open("output/q6_train_test_info.txt", "r") as f:
        content = f.read()

    # Check for key information
    assert "train" in content.lower() and "test" in content.lower(), (
        "Train/test info should mention train and test sets"
    )
    assert "temporal" in content.lower() or "time" in content.lower(), (
        "Train/test info should mention temporal/time-based split"
    )


# ============================================================================
# Q7: Modeling
# ============================================================================


def test_q7_predictions_exists(setup_data):
    """Test that Q7 predictions CSV file exists."""
    assert os.path.exists("output/q7_predictions.csv"), (
        "q7_predictions.csv not found in output/ directory"
    )


def test_q7_predictions_format(setup_data):
    """Test that Q7 predictions has correct format."""
    df = pd.read_csv("output/q7_predictions.csv")

    # Check required columns
    assert "actual" in df.columns, (
        f"Predictions must contain 'actual' column. Found: {list(df.columns)}"
    )

    # Check for at least 2 prediction columns
    pred_cols = [
        col
        for col in df.columns
        if "predicted" in col.lower() or "model" in col.lower()
    ]
    assert len(pred_cols) >= 2, (
        f"Predictions should have at least 2 model prediction columns, got {len(pred_cols)}. Found: {pred_cols}"
    )

    # Check data types
    assert df["actual"].dtype in [np.float64, np.float32, float], (
        "actual must be numeric"
    )
    for col in pred_cols:
        assert df[col].dtype in [np.float64, np.float32, float], (
            f"{col} must be numeric"
        )


def test_q7_model_metrics_exists(setup_data):
    """Test that Q7 model metrics file exists."""
    assert os.path.exists("output/q7_model_metrics.txt"), (
        "q7_model_metrics.txt not found in output/ directory"
    )


def test_q7_model_metrics_content(setup_data):
    """Test that Q7 model metrics contains required metrics."""
    with open("output/q7_model_metrics.txt", "r") as f:
        content = f.read()

    # Check for R² metric (required minimum)
    assert (
        "r²" in content.lower()
        or "r-squared" in content.lower()
        or "r^2" in content.lower()
        or "r2" in content.lower()
    ), "Model metrics should contain R² or R-squared (required minimum metric)"


def test_q7_feature_importance_exists(setup_data):
    """Test that Q7 feature importance CSV file exists."""
    assert os.path.exists("output/q7_feature_importance.csv"), (
        "q7_feature_importance.csv not found in output/ directory"
    )


def test_q7_feature_importance_format(setup_data):
    """Test that Q7 feature importance has correct format."""
    df = pd.read_csv("output/q7_feature_importance.csv")

    # Check required columns
    required_cols = ["feature", "importance"]
    assert all(col in df.columns for col in required_cols), (
        f"Feature importance must contain columns: {required_cols}. Found: {list(df.columns)}"
    )

    # Check data types
    assert df["importance"].dtype in [np.float64, np.float32, float], (
        "importance must be numeric"
    )

    # Check importance values are numeric (allow any range for coefficient-based importance)
    assert not df["importance"].isna().all(), (
        "Importance values should not all be NaN"
    )


# ============================================================================
# Q8: Results
# ============================================================================


def test_q8_final_visualizations_exists(setup_data):
    """Test that Q8 final visualizations file exists."""
    assert os.path.exists("output/q8_final_visualizations.png"), (
        "q8_final_visualizations.png not found in output/ directory"
    )


def test_q8_summary_exists(setup_data):
    """Test that Q8 summary CSV file exists."""
    assert os.path.exists("output/q8_summary.csv"), (
        "q8_summary.csv not found in output/ directory"
    )


def test_q8_summary_format(setup_data):
    """Test that Q8 summary has correct format."""
    df = pd.read_csv("output/q8_summary.csv")

    # Should have Metric column
    assert "metric" in df.columns.str.lower(), (
        f"Summary should have 'Metric' column. Found: {list(df.columns)}"
    )

    # Should have at least 2 columns (Metric + at least 1 model)
    assert len(df.columns) >= 2, (
        f"Summary should have at least 2 columns (Metric + models), got {len(df.columns)}"
    )


def test_q8_key_findings_exists(setup_data):
    """Test that Q8 key findings file exists."""
    assert os.path.exists("output/q8_key_findings.txt"), (
        "q8_key_findings.txt not found in output/ directory"
    )


def test_q8_key_findings_content(setup_data):
    """Test that Q8 key findings contains summary information."""
    with open("output/q8_key_findings.txt", "r") as f:
        content = f.read()

    # Should have some content
    assert len(content.strip()) > 0, "Key findings should contain text content"


# ============================================================================
# Q9: Writeup (Manual grading - basic file check only)
# ============================================================================


def test_q9_report_exists(setup_data):
    """Test that Q9 report file exists."""
    assert os.path.exists("report.md"), (
        "report.md not found in assignment root directory"
    )


def test_q9_report_content(setup_data):
    """Test that Q9 report has reasonable content."""
    with open("report.md", "r") as f:
        content = f.read()

    # Should have substantial content (at least 1000 characters for 3-5 page report)
    assert len(content) >= 1000, (
        f"Report should be at least 3-5 pages, got {len(content)} characters"
    )

    # Check for required sections (basic check)
    content_lower = content.lower()
    assert "summary" in content_lower or "executive" in content_lower, (
        "Report should contain Executive Summary"
    )
    assert "model" in content_lower or "result" in content_lower, (
        "Report should contain Model Results section"
    )


# ============================================================================
# Overall Checks
# ============================================================================


def test_all_output_files_exist(setup_data):
    """Test that all required output files exist."""
    required_files = [
        "output/q1_data_info.txt",
        "output/q1_exploration.csv",
        "output/q1_visualizations.png",
        "output/q2_cleaned_data.csv",
        "output/q2_cleaning_report.txt",
        "output/q2_rows_cleaned.txt",
        "output/q3_wrangled_data.csv",
        "output/q3_temporal_features.csv",
        "output/q3_datetime_info.txt",
        "output/q4_features.csv",
        "output/q4_rolling_features.csv",
        "output/q4_feature_list.txt",
        "output/q5_correlations.csv",
        "output/q5_patterns.png",
        "output/q5_trend_summary.txt",
        "output/q6_X_train.csv",
        "output/q6_X_test.csv",
        "output/q6_y_train.csv",
        "output/q6_y_test.csv",
        "output/q6_train_test_info.txt",
        "output/q7_predictions.csv",
        "output/q7_model_metrics.txt",
        "output/q7_feature_importance.csv",
        "output/q8_final_visualizations.png",
        "output/q8_summary.csv",
        "output/q8_key_findings.txt",
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    assert len(missing_files) == 0, (
        f"Missing required output files: {missing_files}"
    )
