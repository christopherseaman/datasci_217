#!/usr/bin/env python3
"""
Data Cleaning Demo
==================

This demo covers essential data cleaning techniques:
- Missing data handling
- Data type conversion
- String cleaning
- Outlier detection
- Data validation

Run this in a Jupyter notebook for the best experience!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

print("=== DATA CLEANING DEMO ===\n")

# 1. Create messy data for demonstration
print("1. CREATING MESSY DATA")
print("-" * 25)

# Create sample messy data
messy_data = {
    'Name': ['  Alice  ', 'Bob', '  Charlie  ', 'Diana', '', 'Eve', 'Frank'],
    'Age': [25, 30, None, 28, 35, 22, 150],  # None and unrealistic age
    'Email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'diana@email.com', '', 'eve@email.com', 'frank@email.com'],
    'Salary': [50000, 60000, 70000, 55000, 65000, 45000, -1000],  # Negative salary
    'Department': ['Sales', 'Marketing', 'IT', 'HR', 'Sales', 'IT', 'Sales'],
    'Phone': ['123-456-7890', '987-654-3210', '555-123-4567', '111-222-3333', '999-888-7777', '444-555-6666', '777-888-9999'],
    'Date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-25']
}

df = pd.DataFrame(messy_data)
print("Original messy data:")
print(df)
print(f"Shape: {df.shape}")
print()

# 2. Data Quality Assessment
print("2. DATA QUALITY ASSESSMENT")
print("-" * 35)

def assess_data_quality(df):
    """Comprehensive data quality assessment"""
    print("=== DATA QUALITY ASSESSMENT ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print("\n=== DUPLICATE ROWS ===")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    print("\n=== UNIQUE VALUES ===")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
    
    print("\n=== NUMERIC COLUMNS STATISTICS ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    
    print("\n=== CATEGORICAL COLUMNS ===")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head())

assess_data_quality(df)
print()

# 3. Handle Missing Data
print("3. HANDLING MISSING DATA")
print("-" * 30)

# Check missing data
print("Missing values before cleaning:")
print(df.isna().sum())
print()

# Fill missing values
df_clean = df.copy()

# Fill missing age with median
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
print("Filled missing age with median:")
print(f"Age median: {df_clean['Age'].median()}")
print()

# Fill missing email with placeholder
df_clean['Email'] = df_clean['Email'].fillna('No email provided')
print("Filled missing email with placeholder")
print()

# 4. Clean Text Data
print("4. CLEANING TEXT DATA")
print("-" * 25)

# Clean Name column
print("Cleaning Name column...")
df_clean['Name'] = df_clean['Name'].str.strip()  # Remove whitespace
df_clean['Name'] = df_clean['Name'].str.title()  # Title case
df_clean['Name'] = df_clean['Name'].replace('', 'Unknown')  # Replace empty strings

print("Name column after cleaning:")
print(df_clean['Name'])
print()

# Clean Email column
print("Cleaning Email column...")
df_clean['Email'] = df_clean['Email'].str.lower()  # Lowercase
df_clean['Email'] = df_clean['Email'].str.strip()  # Remove whitespace

print("Email column after cleaning:")
print(df_clean['Email'])
print()

# 5. Handle Outliers
print("5. HANDLING OUTLIERS")
print("-" * 25)

def detect_outliers_iqr(series):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# Check for outliers in Age
age_outliers = detect_outliers_iqr(df_clean['Age'])
print(f"Age outliers: {age_outliers.sum()}")
print(f"Outlier indices: {df_clean[age_outliers].index.tolist()}")
print(f"Outlier values: {df_clean[age_outliers]['Age'].tolist()}")
print()

# Check for outliers in Salary
salary_outliers = detect_outliers_iqr(df_clean['Salary'])
print(f"Salary outliers: {salary_outliers.sum()}")
print(f"Outlier indices: {df_clean[salary_outliers].index.tolist()}")
print(f"Outlier values: {df_clean[salary_outliers]['Salary'].tolist()}")
print()

# Handle outliers
print("Handling outliers...")
# Cap outliers instead of removing them
df_clean['Age'] = df_clean['Age'].clip(lower=18, upper=100)
df_clean['Salary'] = df_clean['Salary'].clip(lower=0, upper=200000)

print("After capping outliers:")
print(f"Age range: {df_clean['Age'].min()} - {df_clean['Age'].max()}")
print(f"Salary range: {df_clean['Salary'].min()} - {df_clean['Salary'].max()}")
print()

# 6. Data Type Conversion
print("6. DATA TYPE CONVERSION")
print("-" * 30)

# Convert data types
df_clean['Age'] = df_clean['Age'].astype('int64')
df_clean['Salary'] = df_clean['Salary'].astype('float64')
df_clean['Department'] = df_clean['Department'].astype('category')
df_clean['Date'] = pd.to_datetime(df_clean['Date'])

print("Data types after conversion:")
print(df_clean.dtypes)
print()

# 7. Remove Duplicates
print("7. REMOVING DUPLICATES")
print("-" * 30)

# Check for duplicates
duplicates = df_clean.duplicated()
print(f"Duplicate rows: {duplicates.sum()}")
print(f"Duplicate indices: {df_clean[duplicates].index.tolist()}")
print()

# Remove duplicates
df_clean = df_clean.drop_duplicates()
print(f"Shape after removing duplicates: {df_clean.shape}")
print()

# 8. Data Validation
print("8. DATA VALIDATION")
print("-" * 25)

def validate_data_rules(df):
    """Validate data against business rules"""
    violations = []
    
    # Rule 1: Age should be between 18 and 100
    invalid_ages = df[(df['Age'] < 18) | (df['Age'] > 100)]
    if len(invalid_ages) > 0:
        violations.append(f"Invalid ages found: {len(invalid_ages)} rows")
    
    # Rule 2: Salary should be positive
    negative_salaries = df[df['Salary'] < 0]
    if len(negative_salaries) > 0:
        violations.append(f"Negative salaries found: {len(negative_salaries)} rows")
    
    # Rule 3: Email should contain @
    invalid_emails = df[~df['Email'].str.contains('@', na=False)]
    if len(invalid_emails) > 0:
        violations.append(f"Invalid emails found: {len(invalid_emails)} rows")
    
    # Rule 4: Required fields should not be empty
    required_fields = ['Name', 'Email', 'Department']
    for field in required_fields:
        if field in df.columns:
            missing_required = df[df[field].isna()]
            if len(missing_required) > 0:
                violations.append(f"Missing required field '{field}': {len(missing_required)} rows")
    
    return violations

# Validate data
violations = validate_data_rules(df_clean)
if violations:
    print("Data validation violations found:")
    for violation in violations:
        print(f"- {violation}")
else:
    print("No data validation violations found!")
print()

# 9. Advanced String Operations
print("9. ADVANCED STRING OPERATIONS")
print("-" * 40)

# Extract domain from email
df_clean['Email_Domain'] = df_clean['Email'].str.extract(r'@(.+)')
print("Email domains:")
print(df_clean['Email_Domain'])
print()

# Clean phone numbers
df_clean['Phone_Clean'] = df_clean['Phone'].str.replace(r'[^\d]', '', regex=True)
print("Cleaned phone numbers:")
print(df_clean['Phone_Clean'])
print()

# 10. Create Derived Features
print("10. CREATING DERIVED FEATURES")
print("-" * 40)

# Create age groups
df_clean['Age_Group'] = pd.cut(df_clean['Age'], 
                              bins=[0, 25, 35, 50, 100], 
                              labels=['Young', 'Adult', 'Middle', 'Senior'])

# Create salary ranges
df_clean['Salary_Range'] = pd.cut(df_clean['Salary'], 
                                 bins=3, 
                                 labels=['Low', 'Medium', 'High'])

# Create dummy variables for Department
dept_dummies = pd.get_dummies(df_clean['Department'], prefix='Dept')
df_clean = pd.concat([df_clean, dept_dummies], axis=1)

print("Derived features created:")
print(f"Age groups: {df_clean['Age_Group'].value_counts().to_dict()}")
print(f"Salary ranges: {df_clean['Salary_Range'].value_counts().to_dict()}")
print(f"Department dummies: {dept_dummies.columns.tolist()}")
print()

# 11. Final Data Quality Check
print("11. FINAL DATA QUALITY CHECK")
print("-" * 40)

def automated_quality_check(df):
    """Automated data quality check"""
    quality_score = 100
    issues = []
    
    # Check completeness
    completeness = (1 - df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if completeness < 95:
        quality_score -= 20
        issues.append(f"Low completeness: {completeness:.1f}%")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        quality_score -= 10
        issues.append(f"Found {duplicates} duplicate rows")
    
    # Check for outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['Age', 'Salary']:  # Only check these columns
            outliers = detect_outliers_iqr(df[col])
            if outliers.sum() > len(df) * 0.05:  # More than 5% outliers
                quality_score -= 5
                issues.append(f"Too many outliers in {col}")
    
    print(f"Quality Score: {quality_score}/100")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"- {issue}")
    
    return quality_score, issues

# Run automated check
quality_score, issues = automated_quality_check(df_clean)
print()

# 12. Summary
print("12. CLEANING SUMMARY")
print("-" * 25)

print("=== CLEANING SUMMARY ===")
print(f"Original shape: {df.shape}")
print(f"Final shape: {df_clean.shape}")
print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")
print(f"Quality score: {quality_score}/100")
print()

print("=== FINAL DATASET ===")
print(df_clean)
print()

print("=== DATA TYPES ===")
print(df_clean.dtypes)
print()

print("=== MISSING VALUES ===")
print(df_clean.isna().sum())
print()

print("=== NUMERIC STATISTICS ===")
print(df_clean.describe())
print()

print("=== CATEGORICAL STATISTICS ===")
for col in df_clean.select_dtypes(include=['object', 'category']).columns:
    print(f"\n{col}:")
    print(df_clean[col].value_counts())
print()

print("=== CLEANING COMPLETE ===")
print("Data cleaning pipeline completed successfully!")
print("The dataset is now ready for analysis.")
