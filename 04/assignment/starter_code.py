#!/usr/bin/env python3
"""
Assignment 4: Pandas Basics and Data Exploration
===============================================

Starter code for the pandas basics assignment.
Complete the functions below according to the assignment requirements.

Run this in a Jupyter notebook for the best experience!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
%matplotlib inline

def load_and_explore_data(filepath):
    """
    Task 1: Load and explore the dataset
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # TODO: Load the dataset from CSV
    # TODO: Display first 5 rows
    # TODO: Show shape of dataset
    # TODO: Display data types
    # TODO: Check for missing values
    
    pass

def select_and_filter_data(df):
    """
    Task 2: Perform data selection and filtering
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Dictionary containing filtered datasets
    """
    # TODO: Select employees from 'Sales' department
    sales_employees = None
    
    # TODO: Find employees with salary > $60,000
    high_salary_employees = None
    
    # TODO: Get employees aged 25-35
    young_employees = None
    
    # TODO: Select Name, Age, Salary columns
    basic_info = None
    
    # TODO: Find top 5 highest-paid employees
    top_earners = None
    
    return {
        'sales_employees': sales_employees,
        'high_salary_employees': high_salary_employees,
        'young_employees': young_employees,
        'basic_info': basic_info,
        'top_earners': top_earners
    }

def analyze_data(df):
    """
    Task 3: Perform data analysis
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # TODO: Calculate average salary by department
    avg_salary_by_dept = None
    
    # TODO: Find department with highest average salary
    highest_paying_dept = None
    
    # TODO: Calculate age distribution
    age_stats = None
    
    # TODO: Count employees by city
    employees_by_city = None
    
    # TODO: Find correlation between age and salary
    age_salary_correlation = None
    
    return {
        'avg_salary_by_dept': avg_salary_by_dept,
        'highest_paying_dept': highest_paying_dept,
        'age_stats': age_stats,
        'employees_by_city': employees_by_city,
        'age_salary_correlation': age_salary_correlation
    }

def clean_data(df):
    """
    Task 4: Clean the dataset
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # TODO: Handle missing values
    # TODO: Remove duplicates
    # TODO: Clean Name column
    # TODO: Convert Salary to numeric
    # TODO: Create Salary_Category column
    
    cleaned_df = df.copy()
    
    # Your cleaning code here
    
    return cleaned_df

def export_and_summarize(df, output_path):
    """
    Task 5: Export data and create summary
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        output_path (str): Path for output CSV file
        
    Returns:
        dict: Summary statistics
    """
    # TODO: Export to CSV
    # TODO: Create summary report
    # TODO: Generate insights
    
    summary = {
        'total_employees': len(df),
        'departments': df['Department'].nunique(),
        'cities': df['City'].nunique(),
        'avg_salary': df['Salary'].mean(),
        'age_range': (df['Age'].min(), df['Age'].max())
    }
    
    return summary

# Main execution
if __name__ == "__main__":
    # Load and explore data
    print("=== TASK 1: DATA LOADING AND EXPLORATION ===")
    df = load_and_explore_data('employee_data.csv')
    
    # Select and filter data
    print("\n=== TASK 2: DATA SELECTION AND FILTERING ===")
    filtered_data = select_and_filter_data(df)
    
    # Analyze data
    print("\n=== TASK 3: DATA ANALYSIS ===")
    analysis_results = analyze_data(df)
    
    # Clean data
    print("\n=== TASK 4: DATA CLEANING ===")
    cleaned_df = clean_data(df)
    
    # Export and summarize
    print("\n=== TASK 5: EXPORT AND SUMMARY ===")
    summary = export_and_summarize(cleaned_df, 'cleaned_employee_data.csv')
    
    print("\n=== ASSIGNMENT COMPLETE ===")
    print("Summary:", summary)