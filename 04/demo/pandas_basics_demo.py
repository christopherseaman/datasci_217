#!/usr/bin/env python3
"""
Pandas Basics Demo
==================

This demo covers the fundamental pandas operations:
- Creating Series and DataFrames
- Data selection and filtering
- Basic data manipulation
- File I/O operations

Run this in a Jupyter notebook for the best experience!
"""

import pandas as pd
import numpy as np

print("=== PANDAS BASICS DEMO ===\n")

# 1. Creating Series
print("1. CREATING SERIES")
print("-" * 20)

# From list
s1 = pd.Series([1, 2, 3, 4, 5])
print("Series from list:")
print(s1)
print(f"Type: {type(s1)}")
print(f"Index: {s1.index}")
print(f"Values: {s1.values}")
print()

# With custom index
s2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print("Series with custom index:")
print(s2)
print()

# From dictionary
s3 = pd.Series({'Alice': 85, 'Bob': 92, 'Charlie': 78})
print("Series from dictionary:")
print(s3)
print()

# 2. Creating DataFrames
print("2. CREATING DATAFRAMES")
print("-" * 25)

# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Paris'],
    'Salary': [50000, 60000, 70000, 55000]
}
df = pd.DataFrame(data)
print("DataFrame from dictionary:")
print(df)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Index: {df.index.tolist()}")
print()

# From list of lists
data_list = [
    ['Alice', 25, 'New York', 50000],
    ['Bob', 30, 'London', 60000],
    ['Charlie', 35, 'Tokyo', 70000],
    ['Diana', 28, 'Paris', 55000]
]
df2 = pd.DataFrame(data_list, columns=['Name', 'Age', 'City', 'Salary'])
print("DataFrame from list of lists:")
print(df2)
print()

# 3. Data Selection
print("3. DATA SELECTION")
print("-" * 20)

# Column selection
print("Single column (Series):")
print(df['Name'])
print()

print("Multiple columns (DataFrame):")
print(df[['Name', 'Age']])
print()

# Row selection
print("First 2 rows:")
print(df.iloc[0:2])
print()

print("Specific row and column:")
print(f"First person's name: {df.iloc[0, 0]}")
print()

# Boolean indexing
print("People older than 30:")
print(df[df['Age'] > 30])
print()

print("People from New York:")
print(df[df['City'] == 'New York'])
print()

# 4. Basic Operations
print("4. BASIC OPERATIONS")
print("-" * 25)

# Summary statistics
print("Summary statistics:")
print(df.describe())
print()

# Data types
print("Data types:")
print(df.dtypes)
print()

# Missing data check
print("Missing values:")
print(df.isna().sum())
print()

# Unique values
print("Unique cities:")
print(df['City'].unique())
print()

# Value counts
print("City value counts:")
print(df['City'].value_counts())
print()

# 5. Data Manipulation
print("5. DATA MANIPULATION")
print("-" * 25)

# Adding new column
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Adult')
print("Added Age_Group column:")
print(df[['Name', 'Age', 'Age_Group']])
print()

# Sorting
print("Sorted by Age:")
print(df.sort_values('Age'))
print()

# Grouping
print("Average salary by city:")
print(df.groupby('City')['Salary'].mean())
print()

# 6. File I/O Simulation
print("6. FILE I/O SIMULATION")
print("-" * 30)

# Save to CSV (simulated)
print("Saving DataFrame to CSV...")
df.to_csv('demo_data.csv', index=False)
print("DataFrame saved to 'demo_data.csv'")
print()

# Read from CSV (simulated)
print("Reading DataFrame from CSV...")
df_loaded = pd.read_csv('demo_data.csv')
print("Loaded DataFrame:")
print(df_loaded)
print()

# 7. Advanced Selection
print("7. ADVANCED SELECTION")
print("-" * 30)

# Using loc for label-based selection
print("Using loc for label-based selection:")
print("First row, Name and Age columns:")
print(df.loc[0, ['Name', 'Age']])
print()

# Using iloc for position-based selection
print("Using iloc for position-based selection:")
print("First 2 rows, first 3 columns:")
print(df.iloc[0:2, 0:3])
print()

# Conditional selection with multiple criteria
print("People aged 25-35 from New York or London:")
mask = (df['Age'] >= 25) & (df['Age'] <= 35) & (df['City'].isin(['New York', 'London']))
print(df[mask])
print()

# 8. String Operations
print("8. STRING OPERATIONS")
print("-" * 25)

# String methods
print("Names in uppercase:")
print(df['Name'].str.upper())
print()

print("Names starting with 'A':")
print(df[df['Name'].str.startswith('A')])
print()

print("Names containing 'a' (case insensitive):")
print(df[df['Name'].str.contains('a', case=False)])
print()

# 9. Data Cleaning Example
print("9. DATA CLEANING EXAMPLE")
print("-" * 30)

# Create some messy data
messy_data = {
    'Name': ['  Alice  ', 'Bob', '  Charlie  ', 'Diana'],
    'Age': [25, 30, None, 28],
    'City': ['New York', 'london', 'TOKYO', 'Paris'],
    'Salary': [50000, 60000, 70000, 55000]
}
messy_df = pd.DataFrame(messy_data)
print("Messy data:")
print(messy_df)
print()

# Clean the data
print("Cleaning data...")
messy_df['Name'] = messy_df['Name'].str.strip()  # Remove whitespace
messy_df['City'] = messy_df['City'].str.title()  # Title case
messy_df['Age'] = messy_df['Age'].fillna(messy_df['Age'].mean())  # Fill missing age
print("Cleaned data:")
print(messy_df)
print()

# 10. Summary
print("10. SUMMARY")
print("-" * 15)
print("Pandas provides powerful tools for:")
print("✓ Creating and manipulating data structures")
print("✓ Selecting and filtering data")
print("✓ Handling missing data")
print("✓ Performing data operations")
print("✓ Reading and writing files")
print()
print("This is just the beginning! Pandas has much more to offer.")
print("Next: Data cleaning, transformation, and analysis techniques.")
