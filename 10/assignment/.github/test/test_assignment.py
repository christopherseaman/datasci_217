"""
Auto-grading tests for Assignment 10: Modeling Fundamentals
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import json

# Add assignment directory to path
assignment_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, assignment_dir)

@pytest.fixture(scope="module")
def setup_data():
    """Generate data and convert/run notebooks if needed."""
    # Change to assignment directory
    os.chdir(assignment_dir)
    
    # Generate data if data_generator.ipynb exists, otherwise try markdown
    if os.path.exists('data_generator.ipynb'):
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 
                       '--inplace', 'data_generator.ipynb'], check=True, capture_output=True)
    elif os.path.exists('data_generator.md'):
        # Convert markdown to notebook and execute
        subprocess.run(['jupytext', '--to', 'notebook', 'data_generator.md'], check=True)
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 
                       '--inplace', 'data_generator.ipynb'], check=True, capture_output=True)
    
    # Convert assignment markdown to notebook if needed
    if os.path.exists('assignment.md') and not os.path.exists('assignment.ipynb'):
        subprocess.run(['jupytext', '--to', 'notebook', 'assignment.md'], check=True)
    
    # Execute assignment notebook
    if os.path.exists('assignment.ipynb'):
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 
                       '--inplace', 'assignment.ipynb'], check=True, capture_output=True)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    yield
    
    # Cleanup (optional)
    pass

def test_q1_statistical_model_csv_exists(setup_data):
    """Test that Q1 statistical model CSV file exists."""
    assert os.path.exists('output/q1_statistical_model.csv'), \
        "q1_statistical_model.csv not found in output/ directory"

def test_q1_statistical_model_csv_format(setup_data):
    """Test that Q1 CSV has correct format."""
    df = pd.read_csv('output/q1_statistical_model.csv')
    
    # Check required columns
    required_cols = ['patient_id', 'actual_risk', 'predicted_risk']
    assert all(col in df.columns for col in required_cols), \
        f"CSV must contain columns: {required_cols}"
    
    # Check data types
    assert df['actual_risk'].dtype in [np.float64, np.float32, float], \
        "actual_risk must be numeric"
    assert df['predicted_risk'].dtype in [np.float64, np.float32, float], \
        "predicted_risk must be numeric"
    
    # Check reasonable number of rows (should be all patients)
    assert len(df) >= 1800, f"Expected at least 1800 rows, got {len(df)}"
    
    # Check predictions are reasonable (not all zeros, not all same value)
    assert df['predicted_risk'].nunique() > 1, \
        "All predictions are the same value"
    assert df['predicted_risk'].min() >= 0, \
        "Predictions should be non-negative"
    assert df['predicted_risk'].max() <= 200, \
        "Predictions seem unreasonably high"

def test_q1_model_summary_exists(setup_data):
    """Test that Q1 model summary text file exists."""
    assert os.path.exists('output/q1_model_summary.txt'), \
        "q1_model_summary.txt not found in output/ directory"

def test_q1_model_summary_content(setup_data):
    """Test that Q1 model summary contains required information."""
    with open('output/q1_model_summary.txt', 'r') as f:
        content = f.read()
    
    # Check for key statistics
    assert 'R-squared' in content or 'R²' in content or 'R^2' in content, \
        "Summary should contain R-squared"
    assert 'Observations' in content or 'observations' in content, \
        "Summary should contain number of observations"

def test_q2_ml_predictions_csv_exists(setup_data):
    """Test that Q2 ML predictions CSV file exists."""
    assert os.path.exists('output/q2_ml_predictions.csv'), \
        "q2_ml_predictions.csv not found in output/ directory"

def test_q2_ml_predictions_csv_format(setup_data):
    """Test that Q2 CSV has correct format."""
    df = pd.read_csv('output/q2_ml_predictions.csv')
    
    # Check required columns
    required_cols = ['patient_id', 'actual_risk', 'lr_predicted_risk', 'rf_predicted_risk']
    assert all(col in df.columns for col in required_cols), \
        f"CSV must contain columns: {required_cols}"
    
    # Check data types
    for col in ['actual_risk', 'lr_predicted_risk', 'rf_predicted_risk']:
        assert df[col].dtype in [np.float64, np.float32, float], \
            f"{col} must be numeric"
    
    # Check reasonable number of rows (should be test set, ~400 rows)
    assert 300 <= len(df) <= 500, \
        f"Expected ~400 rows (test set), got {len(df)}"
    
    # Check predictions are different between models
    assert not np.allclose(df['lr_predicted_risk'], df['rf_predicted_risk']), \
        "Linear regression and Random Forest predictions should differ"

def test_q2_model_comparison_exists(setup_data):
    """Test that Q2 model comparison text file exists."""
    assert os.path.exists('output/q2_model_comparison.txt'), \
        "q2_model_comparison.txt not found in output/ directory"

def test_q2_model_comparison_content(setup_data):
    """Test that Q2 model comparison contains required information."""
    with open('output/q2_model_comparison.txt', 'r') as f:
        content = f.read()
    
    # Check for both models
    assert 'Linear Regression' in content or 'linear' in content.lower(), \
        "Comparison should mention Linear Regression"
    assert 'Random Forest' in content or 'random forest' in content.lower(), \
        "Comparison should mention Random Forest"
    
    # Check for metrics
    assert 'R²' in content or 'R-squared' in content or 'R^2' in content, \
        "Comparison should contain R² or R-squared"

def test_q3_xgboost_model_csv_exists(setup_data):
    """Test that Q3 XGBoost model CSV file exists."""
    assert os.path.exists('output/q3_xgboost_model.csv'), \
        "q3_xgboost_model.csv not found in output/ directory"

def test_q3_xgboost_model_csv_format(setup_data):
    """Test that Q3 CSV has correct format."""
    df = pd.read_csv('output/q3_xgboost_model.csv')
    
    # Check required columns
    required_cols = ['patient_id', 'actual_risk', 'xgb_predicted_risk']
    assert all(col in df.columns for col in required_cols), \
        f"CSV must contain columns: {required_cols}"
    
    # Check data types
    assert df['actual_risk'].dtype in [np.float64, np.float32, float], \
        "actual_risk must be numeric"
    assert df['xgb_predicted_risk'].dtype in [np.float64, np.float32, float], \
        "xgb_predicted_risk must be numeric"
    
    # Check reasonable number of rows
    assert 300 <= len(df) <= 500, \
        f"Expected ~400 rows (test set), got {len(df)}"

def test_q3_feature_importance_exists(setup_data):
    """Test that Q3 feature importance text file exists."""
    assert os.path.exists('output/q3_feature_importance.txt'), \
        "q3_feature_importance.txt not found in output/ directory"

def test_q3_feature_importance_content(setup_data):
    """Test that Q3 feature importance contains required features."""
    with open('output/q3_feature_importance.txt', 'r') as f:
        content = f.read()
    
    # Check for expected features
    expected_features = ['age', 'bmi', 'chronic_conditions', 'medication_count', 'hospital_stay_days']
    found_features = sum(1 for feat in expected_features if feat in content.lower())
    
    assert found_features >= 3, \
        f"Feature importance should mention at least 3 of the expected features. Found: {found_features}"

def test_all_output_files_exist(setup_data):
    """Test that all required output files exist."""
    required_files = [
        'output/q1_statistical_model.csv',
        'output/q1_model_summary.txt',
        'output/q2_ml_predictions.csv',
        'output/q2_model_comparison.txt',
        'output/q3_xgboost_model.csv',
        'output/q3_feature_importance.txt'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    assert len(missing_files) == 0, \
        f"Missing required output files: {missing_files}"

