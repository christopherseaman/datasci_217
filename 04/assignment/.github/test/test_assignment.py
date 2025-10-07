import pytest
import pandas as pd
import numpy as np
from main import load_and_explore, clean_and_filter, analyze_orders

# Test data fixture
@pytest.fixture
def sample_df():
    """Create sample test data"""
    data = {
        'order_id': ['O001', 'O002', 'O003', 'O004', 'O005', 'O006'],
        'customer_id': ['C001', 'C002', 'C003', 'C001', 'C004', 'C002'],
        'product': ['Widget A', 'Widget B', 'Widget A', 'Widget C', 'Widget B', 'Widget A'],
        'quantity': [2, np.nan, 3, 1, 4, 2],
        'price': [29.99, 49.99, 29.99, 19.99, 49.99, 29.99],
        'order_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06'],
        # Region "Cancelled" demonstrates that only North/South rows remain once the cleaning filters are applied.
        'region': ['North', 'South', 'North', 'East', 'South', 'Cancelled'],
        'status': ['Complete', 'Complete', 'Complete', 'Complete', 'Cancelled', 'Complete']
    }
    return pd.DataFrame(data)

def test_load_and_explore(tmp_path, capsys):
    """Test Part 1: Data loading and exploration"""
    # Create test CSV
    test_csv = tmp_path / "orders.csv"
    test_data = """order_id,customer_id,product,quantity,price,order_date,region,status
O001,C001,Widget A,2,29.99,2024-01-01,North,Complete
O002,C002,Widget B,3,49.99,2024-01-02,South,Complete"""
    test_csv.write_text(test_data)

    # Change to temp directory
    import os
    original_dir = os.getcwd()
    os.chdir(tmp_path)

    try:
        df = load_and_explore()

        # Check return type
        assert isinstance(df, pd.DataFrame), "Function must return a DataFrame"

        # Check shape
        assert df.shape == (2, 8), f"Expected shape (2, 8), got {df.shape}"

        # Check columns
        expected_cols = ['order_id', 'customer_id', 'product', 'quantity', 'price', 'order_date', 'region', 'status']
        assert list(df.columns) == expected_cols, "Column names don't match"

    finally:
        os.chdir(original_dir)

def test_clean_and_filter(sample_df):
    """Test Part 2: Data cleaning and filtering"""
    df_clean = clean_and_filter(sample_df.copy())

    # Check return type
    assert isinstance(df_clean, pd.DataFrame), "Function must return a DataFrame"

    # Check no cancelled orders
    assert 'Cancelled' not in df_clean['status'].values, "Cancelled orders should be removed"

    # Check no missing quantity values
    assert df_clean['quantity'].isnull().sum() == 0, "Missing quantity values should be filled"

    # Check only North and South regions
    assert set(df_clean['region'].unique()) <= {'North', 'South'}, "Should only have North and South regions"

    # Check the filled value
    # Original had NaN at index 1, after filling should be 1
    original_nan_indices = sample_df[sample_df['quantity'].isnull()].index
    for idx in original_nan_indices:
        if idx in df_clean.index:
            assert df_clean.loc[idx, 'quantity'] == 1, "Missing quantity should be filled with 1"

def test_analyze_orders(sample_df):
    """Test Part 3: Analysis and insights"""
    # Clean the data first
    df_clean = sample_df[sample_df['status'] != 'Cancelled'].copy()
    df_clean['quantity'] = df_clean['quantity'].fillna(1)
    df_clean = df_clean[df_clean['region'].isin(['North', 'South'])]

    # Analyze
    results = analyze_orders(df_clean)

    # total_price should exist after analysis
    assert 'total_price' in df_clean.columns, "DataFrame should include total_price column after analyze_orders"

    # Check return type
    assert isinstance(results, dict), "Function must return a dictionary"
    assert 'revenue_by_region' in results, "Dictionary must have 'revenue_by_region' key"
    assert 'top_3_products' in results, "Dictionary must have 'top_3_products' key"

    # Check revenue_by_region
    revenue = results['revenue_by_region']
    assert isinstance(revenue, pd.Series), "revenue_by_region must be a Series"
    assert revenue.index.name == 'region' or 'region' in str(revenue.index), "Index should be regions"

    # Check top_3_products
    top_products = results['top_3_products']
    assert isinstance(top_products, pd.Series), "top_3_products must be a Series"
    assert len(top_products) <= 3, "Should return at most 3 products"
    assert top_products.index.name == 'product' or 'product' in str(top_products.index), "Index should be products"

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
