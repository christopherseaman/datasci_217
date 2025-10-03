#!/usr/bin/env python3
"""
String Manipulation Demo
========================

This demo covers advanced string manipulation techniques:
- Basic string operations
- Regular expressions
- Text cleaning
- Pattern matching
- String extraction

Run this in a Jupyter notebook for the best experience!
"""

import pandas as pd
import numpy as np
import re

print("=== STRING MANIPULATION DEMO ===\n")

# 1. Create sample text data
print("1. CREATING SAMPLE TEXT DATA")
print("-" * 35)

# Create sample data with various text issues
text_data = {
    'Name': ['  Alice Smith  ', 'Bob Johnson', '  Charlie Brown  ', 'Diana Lee', 'Eve Wilson'],
    'Email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'diana@email.com', 'eve@email.com'],
    'Phone': ['(123) 456-7890', '987-654-3210', '555.123.4567', '111-222-3333', '999 888 7777'],
    'Address': ['123 Main St, New York, NY 10001', '456 Oak Ave, Los Angeles, CA 90210', '789 Pine Rd, Chicago, IL 60601', '321 Elm St, Houston, TX 77001', '654 Maple Dr, Phoenix, AZ 85001'],
    'Description': ['Software Engineer with 5 years experience', 'Data Scientist specializing in ML', 'Product Manager with MBA', 'UX Designer with 3 years experience', 'DevOps Engineer with AWS certification'],
    'Tags': ['python,data-science,machine-learning', 'java,spring-boot,backend', 'react,javascript,frontend', 'python,django,web-development', 'aws,devops,cloud'],
    'Price': ['$1,234.56', '$2,345.67', '$3,456.78', '$4,567.89', '$5,678.90'],
    'Date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12']
}

df = pd.DataFrame(text_data)
print("Original text data:")
print(df)
print()

# 2. Basic String Operations
print("2. BASIC STRING OPERATIONS")
print("-" * 35)

# Clean Name column
print("Cleaning Name column...")
df['Name_Clean'] = df['Name'].str.strip()  # Remove whitespace
df['Name_Clean'] = df['Name_Clean'].str.title()  # Title case
print("Name after cleaning:")
print(df[['Name', 'Name_Clean']])
print()

# String length
print("String lengths:")
df['Name_Length'] = df['Name_Clean'].str.len()
print(df[['Name_Clean', 'Name_Length']])
print()

# String slicing
print("String slicing:")
df['First_Name'] = df['Name_Clean'].str.split().str[0]
df['Last_Name'] = df['Name_Clean'].str.split().str[-1]
print(df[['Name_Clean', 'First_Name', 'Last_Name']])
print()

# 3. String Case Operations
print("3. STRING CASE OPERATIONS")
print("-" * 35)

# Different case operations
df['Name_Upper'] = df['Name_Clean'].str.upper()
df['Name_Lower'] = df['Name_Clean'].str.lower()
df['Name_Title'] = df['Name_Clean'].str.title()
df['Name_Capitalize'] = df['Name_Clean'].str.capitalize()

print("Case operations:")
print(df[['Name_Clean', 'Name_Upper', 'Name_Lower', 'Name_Title', 'Name_Capitalize']])
print()

# 4. String Pattern Matching
print("4. STRING PATTERN MATCHING")
print("-" * 35)

# Check if strings contain patterns
print("Pattern matching:")
df['Has_Smith'] = df['Name_Clean'].str.contains('Smith', case=False)
df['Starts_With_A'] = df['Name_Clean'].str.startswith('A')
df['Ends_With_Son'] = df['Name_Clean'].str.endswith('son')

print("Pattern matching results:")
print(df[['Name_Clean', 'Has_Smith', 'Starts_With_A', 'Ends_With_Son']])
print()

# 5. String Replacement
print("5. STRING REPLACEMENT")
print("-" * 30)

# Replace characters
print("Replacing characters:")
df['Phone_Clean'] = df['Phone'].str.replace(r'[^\d]', '', regex=True)  # Keep only digits
df['Phone_Formatted'] = df['Phone_Clean'].str.replace(r'(\d{3})(\d{3})(\d{4})', r'\1-\2-\3', regex=True)

print("Phone number cleaning:")
print(df[['Phone', 'Phone_Clean', 'Phone_Formatted']])
print()

# 6. Regular Expressions
print("6. REGULAR EXPRESSIONS")
print("-" * 30)

# Email validation
print("Email validation:")
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
df['Email_Valid'] = df['Email'].str.match(email_pattern, na=False)
print("Email validation results:")
print(df[['Email', 'Email_Valid']])
print()

# Extract domain from email
print("Extracting email domains:")
df['Email_Domain'] = df['Email'].str.extract(r'@(.+)')
print("Email domains:")
print(df[['Email', 'Email_Domain']])
print()

# 7. Address Parsing
print("7. ADDRESS PARSING")
print("-" * 25)

# Extract components from address
print("Parsing addresses:")
df['Street'] = df['Address'].str.extract(r'^([^,]+)')
df['City'] = df['Address'].str.extract(r',\s*([^,]+),')
df['State'] = df['Address'].str.extract(r',\s*([A-Z]{2})\s+\d+')
df['Zip'] = df['Address'].str.extract(r'(\d{5})')

print("Address components:")
print(df[['Address', 'Street', 'City', 'State', 'Zip']])
print()

# 8. Price Parsing
print("8. PRICE PARSING")
print("-" * 20)

# Extract numeric values from price strings
print("Parsing prices:")
df['Price_Numeric'] = df['Price'].str.replace(r'[^\d.]', '', regex=True).astype(float)
df['Price_Formatted'] = df['Price_Numeric'].apply(lambda x: f"${x:,.2f}")

print("Price parsing results:")
print(df[['Price', 'Price_Numeric', 'Price_Formatted']])
print()

# 9. Tag Processing
print("9. TAG PROCESSING")
print("-" * 25)

# Split tags and create dummy variables
print("Processing tags:")
df['Tags_List'] = df['Tags'].str.split(',')
df['Tag_Count'] = df['Tags_List'].str.len()

# Create dummy variables for tags
all_tags = set()
for tag_list in df['Tags_List']:
    all_tags.update(tag_list)

for tag in all_tags:
    df[f'Tag_{tag.strip()}'] = df['Tags'].str.contains(tag, case=False)

print("Tag processing results:")
print(f"Total unique tags: {len(all_tags)}")
print(f"Tags: {sorted(all_tags)}")
print(f"Tag counts: {df['Tag_Count'].tolist()}")
print()

# 10. Date Parsing
print("10. DATE PARSING")
print("-" * 20)

# Parse dates and extract components
print("Parsing dates:")
df['Date_Parsed'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date_Parsed'].dt.year
df['Month'] = df['Date_Parsed'].dt.month
df['Day'] = df['Date_Parsed'].dt.day
df['Weekday'] = df['Date_Parsed'].dt.day_name()

print("Date parsing results:")
print(df[['Date', 'Date_Parsed', 'Year', 'Month', 'Day', 'Weekday']])
print()

# 11. Text Analysis
print("11. TEXT ANALYSIS")
print("-" * 25)

# Analyze description text
print("Text analysis:")
df['Description_Length'] = df['Description'].str.len()
df['Word_Count'] = df['Description'].str.split().str.len()
df['Has_Experience'] = df['Description'].str.contains('experience', case=False)
df['Has_Years'] = df['Description'].str.contains(r'\d+\s+years?', case=False, regex=True)

print("Text analysis results:")
print(df[['Description', 'Description_Length', 'Word_Count', 'Has_Experience', 'Has_Years']])
print()

# 12. Advanced String Operations
print("12. ADVANCED STRING OPERATIONS")
print("-" * 40)

# String similarity (simple version)
def string_similarity(s1, s2):
    """Simple string similarity using Jaccard similarity"""
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# Find similar descriptions
print("Finding similar descriptions:")
similarities = []
for i in range(len(df)):
    for j in range(i+1, len(df)):
        sim = string_similarity(df.iloc[i]['Description'], df.iloc[j]['Description'])
        if sim > 0.3:  # Threshold for similarity
            similarities.append((i, j, sim))

print("Similar descriptions:")
for i, j, sim in similarities:
    print(f"Similarity {sim:.2f}: '{df.iloc[i]['Description']}' <-> '{df.iloc[j]['Description']}'")
print()

# 13. Text Cleaning Pipeline
print("13. TEXT CLEANING PIPELINE")
print("-" * 35)

def clean_text(text):
    """Comprehensive text cleaning pipeline"""
    if pd.isna(text):
        return text
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Convert to title case
    text = text.title()
    
    return text

# Apply cleaning pipeline
print("Applying text cleaning pipeline...")
df['Description_Clean'] = df['Description'].apply(clean_text)
df['Name_Clean_Final'] = df['Name'].apply(clean_text)

print("Text cleaning results:")
print(df[['Description', 'Description_Clean']])
print(df[['Name', 'Name_Clean_Final']])
print()

# 14. String Validation
print("14. STRING VALIDATION")
print("-" * 30)

def validate_strings(df):
    """Validate string data quality"""
    issues = []
    
    # Check for empty strings
    for col in df.select_dtypes(include=['object']).columns:
        empty_count = (df[col] == '').sum()
        if empty_count > 0:
            issues.append(f"Empty strings in {col}: {empty_count}")
    
    # Check for strings that are too long
    for col in df.select_dtypes(include=['object']).columns:
        long_strings = df[col].str.len() > 100
        if long_strings.any():
            issues.append(f"Very long strings in {col}: {long_strings.sum()}")
    
    # Check for strings with only whitespace
    for col in df.select_dtypes(include=['object']).columns:
        whitespace_only = df[col].str.strip() == ''
        if whitespace_only.any():
            issues.append(f"Whitespace-only strings in {col}: {whitespace_only.sum()}")
    
    return issues

# Validate strings
issues = validate_strings(df)
if issues:
    print("String validation issues found:")
    for issue in issues:
        print(f"- {issue}")
else:
    print("No string validation issues found!")
print()

# 15. Summary
print("15. STRING MANIPULATION SUMMARY")
print("-" * 40)

print("=== STRING MANIPULATION SUMMARY ===")
print(f"Original dataset shape: {df.shape}")
print(f"Columns processed: {len(df.select_dtypes(include=['object']).columns)}")
print(f"String operations performed:")
print("✓ Text cleaning and normalization")
print("✓ Pattern matching and extraction")
print("✓ Regular expression operations")
print("✓ String validation")
print("✓ Text analysis and similarity")
print()

print("=== FINAL DATASET ===")
print("Key columns after string manipulation:")
key_cols = ['Name_Clean_Final', 'Email', 'Phone_Formatted', 'City', 'State', 'Price_Numeric', 'Tag_Count']
available_cols = [col for col in key_cols if col in df.columns]
print(df[available_cols])
print()

print("=== STRING MANIPULATION COMPLETE ===")
print("String manipulation pipeline completed successfully!")
print("The text data is now clean and ready for analysis.")
