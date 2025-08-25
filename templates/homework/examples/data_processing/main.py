#!/usr/bin/env python3
"""
Example: Data Processing Assignment - Sales Data Analysis

This example demonstrates a complete data processing homework assignment
using pandas, with file I/O, data cleaning, and analysis.

Student: Example Student
Course: Data Science 217
Assignment: Data Processing and Analysis
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load sales data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        DataFrame containing the loaded data
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
        ValueError: If the file format is invalid
    
    Examples:
        >>> df = load_data('sales_data.csv')
        >>> len(df) > 0
        True
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise pd.errors.EmptyDataError("File contains no data")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the sales data by handling missing values and invalid data.
    
    Args:
        df: Raw DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    
    Raises:
        TypeError: If input is not a DataFrame
        ValueError: If DataFrame is empty or missing required columns
    
    Examples:
        >>> dirty_df = pd.DataFrame({'sales': [100, None, 200], 'product': ['A', 'B', 'C']})
        >>> clean_df = clean_data(dirty_df)
        >>> clean_df['sales'].isna().sum() == 0
        True
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Cannot clean empty DataFrame")
    
    # Work on a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Expected columns for sales data
    expected_columns = ['product', 'sales', 'quantity', 'price']
    missing_columns = [col for col in expected_columns if col not in cleaned_df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Handle missing values
    # For numeric columns, fill with median
    numeric_columns = cleaned_df.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        if col in cleaned_df.columns:
            median_value = cleaned_df[col].median()
            cleaned_df[col].fillna(median_value, inplace=True)
    
    # For categorical columns, fill with mode (most frequent)
    categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col in cleaned_df.columns and not cleaned_df[col].empty:
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col].fillna(mode_value[0], inplace=True)
            else:
                cleaned_df[col].fillna('Unknown', inplace=True)
    
    # Remove negative sales/quantities (data quality issue)
    if 'sales' in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df['sales'] >= 0]
    
    if 'quantity' in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df['quantity'] > 0]
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df


def analyze_data(df: pd.DataFrame) -> Dict[str, Union[float, int, Dict]]:
    """
    Perform analysis on the cleaned sales data.
    
    Args:
        df: Cleaned DataFrame to analyze
    
    Returns:
        Dictionary containing analysis results
    
    Raises:
        TypeError: If input is not a DataFrame
        ValueError: If DataFrame is empty or missing required columns
    
    Examples:
        >>> sample_df = pd.DataFrame({
        ...     'product': ['A', 'B', 'A'],
        ...     'sales': [100, 200, 150],
        ...     'quantity': [5, 10, 7]
        ... })
        >>> results = analyze_data(sample_df)
        >>> 'total_sales' in results
        True
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Cannot analyze empty DataFrame")
    
    # Check for required columns
    required_columns = ['sales', 'product']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns for analysis: {missing_columns}")
    
    results = {}
    
    # Basic statistics
    results['total_sales'] = df['sales'].sum()
    results['average_sales'] = df['sales'].mean()
    results['median_sales'] = df['sales'].median()
    results['max_sales'] = df['sales'].max()
    results['min_sales'] = df['sales'].min()
    results['total_transactions'] = len(df)
    
    # Product analysis
    product_sales = df.groupby('product')['sales'].agg(['sum', 'count', 'mean'])
    results['sales_by_product'] = product_sales.to_dict('index')
    
    # Top performing products
    top_products = product_sales['sum'].sort_values(ascending=False).head(5)
    results['top_products'] = top_products.to_dict()
    
    # Additional analysis if quantity column exists
    if 'quantity' in df.columns:
        results['total_quantity'] = df['quantity'].sum()
        results['average_quantity'] = df['quantity'].mean()
    
    # Performance metrics
    if 'quantity' in df.columns and 'price' in df.columns:
        # Calculate price per unit where possible
        df_temp = df.copy()
        df_temp['calculated_price'] = df_temp['sales'] / df_temp['quantity']
        results['average_price_per_unit'] = df_temp['calculated_price'].mean()
    
    return results


def save_results(results: Dict, output_file: str) -> None:
    """
    Save analysis results to a JSON file.
    
    Args:
        results: Dictionary containing analysis results
        output_file: Path where to save the results
    
    Raises:
        TypeError: If results is not a dictionary
        OSError: If file cannot be written
    
    Examples:
        >>> results = {'total_sales': 1000, 'products': 5}
        >>> save_results(results, 'output.json')
        >>> os.path.exists('output.json')
        True
    """
    if not isinstance(results, dict):
        raise TypeError("Results must be a dictionary")
    
    try:
        # Ensure directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            else:
                return obj
        
        serializable_results = convert_types(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
            
    except Exception as e:
        raise OSError(f"Error saving results to {output_file}: {e}")


def generate_sample_data(output_file: str = 'sample_sales_data.csv') -> None:
    """
    Generate sample sales data for testing.
    
    Args:
        output_file: Path where to save the sample data
    """
    import numpy as np
    np.random.seed(42)  # For reproducible results
    
    products = ['Widget A', 'Widget B', 'Widget C', 'Gadget X', 'Gadget Y']
    n_records = 100
    
    data = {
        'product': np.random.choice(products, n_records),
        'quantity': np.random.randint(1, 20, n_records),
        'price': np.round(np.random.uniform(10, 100, n_records), 2),
    }
    
    # Calculate sales (with some missing values)
    data['sales'] = data['quantity'] * data['price']
    
    # Add some missing values and anomalies for cleaning practice
    missing_indices = np.random.choice(n_records, size=10, replace=False)
    for idx in missing_indices[:5]:
        data['sales'][idx] = None
    
    for idx in missing_indices[5:]:
        data['product'][idx] = None
    
    # Add some negative values (data quality issues)
    for idx in np.random.choice(n_records, size=3, replace=False):
        data['sales'][idx] = -abs(data['sales'][idx])
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Generated sample data: {output_file}")


def main():
    """
    Main function demonstrating the complete data processing pipeline.
    """
    print("📊 Sales Data Analysis Assignment Demo")
    print("=" * 50)
    
    try:
        # Generate sample data if it doesn't exist
        data_file = 'sample_sales_data.csv'
        if not os.path.exists(data_file):
            print("📝 Generating sample data...")
            generate_sample_data(data_file)
        
        # Step 1: Load data
        print(f"\n📁 Loading data from {data_file}...")
        raw_data = load_data(data_file)
        print(f"Loaded {len(raw_data)} records with {len(raw_data.columns)} columns")
        print("Raw data preview:")
        print(raw_data.head())
        
        # Step 2: Clean data
        print(f"\n🧹 Cleaning data...")
        print(f"Missing values before cleaning:")
        print(raw_data.isnull().sum())
        
        cleaned_data = clean_data(raw_data)
        print(f"Cleaned data: {len(cleaned_data)} records")
        print(f"Missing values after cleaning:")
        print(cleaned_data.isnull().sum())
        
        # Step 3: Analyze data
        print(f"\n📈 Analyzing data...")
        analysis_results = analyze_data(cleaned_data)
        
        print("Analysis Results:")
        print(f"  Total Sales: ${analysis_results['total_sales']:,.2f}")
        print(f"  Average Sales: ${analysis_results['average_sales']:,.2f}")
        print(f"  Total Transactions: {analysis_results['total_transactions']:,}")
        
        print(f"\n🏆 Top 3 Products by Sales:")
        for i, (product, sales) in enumerate(list(analysis_results['top_products'].items())[:3], 1):
            print(f"  {i}. {product}: ${sales:,.2f}")
        
        # Step 4: Save results
        output_file = 'analysis_results.json'
        print(f"\n💾 Saving results to {output_file}...")
        save_results(analysis_results, output_file)
        print(f"Results saved successfully!")
        
        print("\n🎉 Data processing pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())