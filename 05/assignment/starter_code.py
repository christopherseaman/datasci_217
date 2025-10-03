#!/usr/bin/env python3
"""
Assignment 5: Data Cleaning and Preparation
==========================================

Starter code for the data cleaning assignment.
Complete the functions below according to the assignment requirements.

Run this in a Jupyter notebook for the best experience!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
%matplotlib inline

def assess_data_quality(df):
    """
    Task 1: Comprehensive data quality assessment
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Data quality metrics and issues
    """
    # TODO: Perform comprehensive data quality assessment
    # TODO: Identify missing values, duplicates, and data type issues
    # TODO: Create data quality report with statistics
    # TODO: Document all data quality issues found
    
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isna().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'issues': []
    }
    
    # Your assessment code here
    
    return quality_report

def handle_missing_data(df):
    """
    Task 2: Handle missing data appropriately
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with missing data handled
    """
    # TODO: Analyze patterns in missing data
    # TODO: Implement appropriate strategies for handling missing values
    # TODO: Justify approach for each column
    # TODO: Create visualizations showing before/after
    
    df_clean = df.copy()
    
    # Your missing data handling code here
    
    return df_clean

def clean_text_data(df):
    """
    Task 3: Clean and standardize text data
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with cleaned text data
    """
    # TODO: Clean and standardize text columns
    # TODO: Handle inconsistent formatting and encoding issues
    # TODO: Extract useful information from text fields
    # TODO: Validate text data quality
    
    df_clean = df.copy()
    
    # Your text cleaning code here
    
    return df_clean

def handle_outliers(df):
    """
    Task 4: Detect and handle outliers
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with outliers handled
    """
    # TODO: Detect outliers in numeric columns using multiple methods
    # TODO: Implement appropriate outlier handling strategies
    # TODO: Justify approach for each column
    # TODO: Create visualizations showing outlier detection results
    
    df_clean = df.copy()
    
    # Your outlier handling code here
    
    return df_clean

def validate_and_export(df, output_path):
    """
    Task 5: Validate data and export cleaned dataset
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        output_path (str): Path for output CSV file
        
    Returns:
        dict: Final data quality report
    """
    # TODO: Implement comprehensive data validation rules
    # TODO: Create data quality monitoring system
    # TODO: Export cleaned dataset
    # TODO: Generate final data quality report
    
    # Your validation and export code here
    
    final_report = {
        'final_shape': df.shape,
        'quality_score': 100,  # Calculate actual quality score
        'issues_resolved': [],
        'remaining_issues': []
    }
    
    return final_report

def create_data_quality_visualizations(df_before, df_after):
    """
    Create visualizations for data quality assessment
    
    Args:
        df_before (pd.DataFrame): Dataset before cleaning
        df_after (pd.DataFrame): Dataset after cleaning
        
    Returns:
        None: Creates and displays visualizations
    """
    # TODO: Create visualizations for data quality assessment
    # TODO: Show missing data patterns
    # TODO: Display outlier detection results
    # TODO: Compare before/after cleaning
    
    # Your visualization code here
    pass

def clean_data_pipeline(df):
    """
    Complete data cleaning pipeline
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Fully cleaned dataset
    """
    print("Starting data cleaning pipeline...")
    original_shape = df.shape
    
    # Step 1: Data quality assessment
    print("1. Assessing data quality...")
    quality_report = assess_data_quality(df)
    
    # Step 2: Handle missing data
    print("2. Handling missing data...")
    df_clean = handle_missing_data(df)
    
    # Step 3: Clean text data
    print("3. Cleaning text data...")
    df_clean = clean_text_data(df_clean)
    
    # Step 4: Handle outliers
    print("4. Handling outliers...")
    df_clean = handle_outliers(df_clean)
    
    # Step 5: Final validation and export
    print("5. Final validation and export...")
    final_report = validate_and_export(df_clean, 'cleaned_data.csv')
    
    # Create visualizations
    create_data_quality_visualizations(df, df_clean)
    
    print(f"Cleaning complete! Original shape: {original_shape}, Final shape: {df_clean.shape}")
    return df_clean

# Main execution
if __name__ == "__main__":
    # Load the dataset
    print("=== ASSIGNMENT 5: DATA CLEANING AND PREPARATION ===")
    df = pd.read_csv('messy_employee_data.csv')
    
    # Run the complete cleaning pipeline
    df_cleaned = clean_data_pipeline(df)
    
    print("\n=== ASSIGNMENT COMPLETE ===")
    print("Data cleaning pipeline completed successfully!")
    print(f"Final dataset shape: {df_cleaned.shape}")
