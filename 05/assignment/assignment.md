---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Assignment 5: Data Cleaning Exam

This is a comprehensive exam testing your data cleaning skills from Lectures 1-5. You'll build a complete data cleaning pipeline for a messy employee dataset.

## Setup

Import required libraries:

```python
import pandas as pd
import numpy as np
```

## Load the Data

First, let's load and inspect the raw data:

```python
# Load the messy employee data
df = pd.read_csv('employees_raw.csv')

print("Raw data shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))
print("\nData types:")
print(df.dtypes)
print("\nBasic info:")
print(df.info())
```

---

## Part 1: Data Audit (20 points)

Before cleaning, we need to understand what's wrong with the data.

### Task 1.1: Implement the audit function

```python
def audit_data(df):
    """
    Audit data quality and return a report.

    Args:
        df (pd.DataFrame): Raw employee data

    Returns:
        dict: Data quality report with keys:
            - total_rows: int
            - total_columns: int
            - missing_count: dict mapping column names to missing count
            - duplicate_count: int
            - negative_salaries: int
            - invalid_dates: int (count of "UNKNOWN" in hire_date)
    """
    # TODO: Implement this function

    # Hint: Use len(df) for row count
    # Hint: Use len(df.columns) for column count
    # Hint: Use df.isnull().sum() for missing values per column
    # Hint: Use df.duplicated().sum() for duplicate rows
    # Hint: Use (df['salary'] < 0).sum() for negative salaries
    # Hint: Use (df['hire_date'] == 'UNKNOWN').sum() for invalid dates

    pass

# Test your function
audit_report = audit_data(df)
print("=== DATA AUDIT REPORT ===")
for key, value in audit_report.items():
    print(f"{key}: {value}")
```

### Task 1.2: Analyze the audit results

```python
# TODO: Based on the audit report, answer these questions:
# - How many rows have missing data?
# - Which column has the most missing values?
# - Are there any duplicate rows?
# - How many data quality issues did you find?

```

---

## Part 2: Data Cleaning (40 points)

Now let's clean the data systematically.

### Task 2.1: Implement the cleaning function

```python
def clean_data(df):
    """
    Clean the raw employee data.

    Args:
        df (pd.DataFrame): Raw employee data

    Returns:
        pd.DataFrame: Cleaned data with:
            - Sentinel values (-999) replaced with NaN
            - Missing numeric data filled appropriately
            - Text data standardized
            - Dates converted to datetime
            - Outliers removed (salary < 0 or > 500000)
            - Duplicates removed
    """
    # TODO: Implement this function

    # Step 1: Replace sentinel values
    # Hint: Use .replace(-999, np.nan)

    # Step 2: Handle missing numeric data
    # Hint: Fill salary with median: df['salary'].fillna(df['salary'].median())
    # Hint: Fill years_experience with median
    # Hint: Fill performance_score with 3.0

    # Step 3: Clean text data
    # Hint: Use .str.strip().str.title() for name and department
    # Hint: Use .str.strip().str.lower() for status

    # Step 4: Fix dates
    # Hint: Use pd.to_datetime(df['hire_date'], errors='coerce')

    # Step 5: Remove outliers
    # Hint: Filter out rows where salary < 0 or salary > 500000

    # Step 6: Drop duplicates
    # Hint: Use .drop_duplicates()

    pass

# Test your function
df_clean = clean_data(df.copy())  # Use .copy() to preserve original
print(f"\nOriginal rows: {len(df)}, Cleaned rows: {len(df_clean)}")
print("\nCleaned data sample:")
print(df_clean.head())
print("\nCleaned data info:")
print(df_clean.info())
```

### Task 2.2: Verify the cleaning

```python
# TODO: Verify your cleaning worked:
# - Check for remaining missing values
# - Check for -999 sentinel values
# - Check data types (especially hire_date)
# - Check for negative or extreme salaries
# - Check for duplicates

print("\n=== VERIFICATION ===")
# Your verification code here
```

---

## Part 3: Data Transformation (25 points)

Add calculated and categorical columns to enrich the data.

### Task 3.1: Implement the transformation function

```python
def transform_data(df_clean):
    """
    Add calculated and categorical columns to cleaned data.

    Args:
        df_clean (pd.DataFrame): Cleaned employee data

    Returns:
        pd.DataFrame: Transformed data with new columns:
            - experience_level: "Junior" (0-2), "Mid" (2-5), "Senior" (5+)
            - salary_tier: "Low", "Medium", "High" (using qcut)
            - high_performer: bool (performance_score >= 4)
            - tenure_years: years since hire_date (NaT filled with 0)
    """
    # TODO: Implement this function

    # Create experience_level using pd.cut()
    # Hint: pd.cut(df['years_experience'], bins=[0, 2, 5, 100], labels=['Junior', 'Mid', 'Senior'])

    # Create salary_tier using pd.qcut()
    # Hint: pd.qcut(df['salary'], q=3, labels=['Low', 'Medium', 'High'])

    # Create high_performer boolean
    # Hint: df['performance_score'] >= 4

    # Create tenure_years
    # Hint: Calculate years from hire_date to 2024
    # Hint: Use (pd.Timestamp('2024-01-01') - df['hire_date']).dt.days / 365.25
    # Hint: Fill NaT with 0

    pass

# Test your function
df_final = transform_data(df_clean.copy())
print("\n=== TRANSFORMED DATA ===")
print(df_final.head())
print(f"\nNew columns added: {set(df_final.columns) - set(df_clean.columns)}")
print("\nExperience level distribution:")
print(df_final['experience_level'].value_counts())
print("\nSalary tier distribution:")
print(df_final['salary_tier'].value_counts())
```

### Task 3.2: Explore the new features

```python
# TODO: Analyze the new columns:
# - How many high performers are there?
# - What's the average salary by experience level?
# - What's the relationship between tenure and performance?

```

---

## Part 4: Data Analysis (15 points)

Extract insights from the cleaned and transformed data.

### Task 4.1: Implement the analysis function

```python
def analyze_data(df_final):
    """
    Analyze the cleaned and transformed data.

    Args:
        df_final (pd.DataFrame): Cleaned and transformed data

    Returns:
        dict: Analysis results with keys:
            - avg_salary_by_dept: Series with average salary per department
            - high_performer_count: int count of high performers
            - senior_avg_performance: float average performance for Seniors
            - top_department: str name of department with highest avg salary
    """
    # TODO: Implement this function

    # Calculate average salary by department
    # Hint: df.groupby('department')['salary'].mean()

    # Count high performers
    # Hint: df['high_performer'].sum()

    # Average performance for Seniors
    # Hint: df[df['experience_level'] == 'Senior']['performance_score'].mean()

    # Top department by salary
    # Hint: avg_salary_by_dept.idxmax()

    pass

# Test your function
results = analyze_data(df_final)
print("\n=== ANALYSIS RESULTS ===")
print("\nAverage salary by department:")
print(results['avg_salary_by_dept'])
print(f"\nHigh performers: {results['high_performer_count']}")
print(f"Senior avg performance: {results['senior_avg_performance']:.2f}")
print(f"Top paying department: {results['top_department']}")
```

### Task 4.2: Business insights

```python
# TODO: Answer these business questions:
# - Which department should we focus on for retention?
# - Is there a relationship between experience level and performance?
# - Are high performers fairly compensated?

```

---

## Final Validation

Run all functions together to ensure the complete pipeline works:

```python
print("=== COMPLETE PIPELINE TEST ===\n")

# Load data
df_raw = pd.read_csv('employees_raw.csv')
print(f"1. Loaded {len(df_raw)} raw records")

# Audit
audit_report = audit_data(df_raw)
print(f"2. Audit complete: found {audit_report['duplicate_count']} duplicates, "
      f"{audit_report['negative_salaries']} negative salaries")

# Clean
df_clean = clean_data(df_raw)
print(f"3. Cleaned data: {len(df_clean)} records remain")

# Transform
df_final = transform_data(df_clean)
print(f"4. Transformed data: added {len(df_final.columns) - len(df_clean.columns)} new columns")

# Analyze
results = analyze_data(df_final)
print(f"5. Analysis complete: {results['high_performer_count']} high performers identified")

print("\n✓ Pipeline complete!")
```

## Submission

Copy your four functions (`audit_data`, `clean_data`, `transform_data`, `analyze_data`) to `main.py` for autograding.

Your code will be tested on:
- Correct return types and structure
- Proper data cleaning (no missing values, correct types)
- Accurate transformations (correct categories, calculations)
- Valid analysis results (correct aggregations)

## Tips for Success

- **Start with Part 1** to understand the data issues
- **Use `.copy()`** to avoid modifying the original DataFrame
- **Test incrementally** - verify each step before moving on
- **Check data types** after each transformation
- **Print intermediate results** to debug issues
- **Remember**: `cut()` = equal-width bins, `qcut()` = equal-frequency bins
- **Handle edge cases**: NaN dates, division by zero, etc.

Good luck! This exam tests everything you've learned so far.
