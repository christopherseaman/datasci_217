Assignment 5: Data Cleaning Exam
================

**Type**: Exam (graded 0-100)
**Topics**: Missing data handling, data transformation, data validation, outlier detection

## Overview

You've been hired to clean a messy employee dataset from a company's HR system. The data has missing values, inconsistent formatting, sentinel values, and data quality issues. Your job is to build a complete data cleaning pipeline that transforms the raw data into analysis-ready format.

This assignment tests everything from Lectures 1-5: Python fundamentals, data structures, NumPy operations, pandas basics, and data cleaning techniques.

## Dataset

The file `employees_raw.csv` contains employee records with the following columns:
- `employee_id`: Unique identifier
- `name`: Employee full name
- `department`: Department name
- `salary`: Annual salary
- `years_experience`: Years of work experience
- `performance_score`: Performance rating (1-5 scale)
- `hire_date`: Date hired
- `status`: Employment status

**Known Issues**:
- Missing values across multiple columns
- Sentinel value `-999` used for missing numeric data
- Inconsistent text capitalization
- Invalid dates (some use "UNKNOWN")
- Salary outliers and data entry errors
- Inconsistent department names

## Tasks

Complete the functions in `main.py`. Each function is worth points as shown below.

### Part 1: Data Audit (20 points)

**Function**: `audit_data(df)`

Create a data quality report that returns a dictionary containing:
- `total_rows`: Total number of rows
- `total_columns`: Total number of columns
- `missing_count`: Dictionary mapping column names to count of missing/NaN values
- `duplicate_count`: Number of duplicate rows
- `negative_salaries`: Count of negative salary values
- `invalid_dates`: Count of rows with "UNKNOWN" in hire_date

**Requirements**:
- Count NaN values per column
- Identify duplicate rows (entire row duplicates)
- Find negative salaries (< 0)
- Find invalid date strings

### Part 2: Data Cleaning (40 points)

**Function**: `clean_data(df)`

Clean the raw data and return a cleaned DataFrame:
1. **Replace sentinel values**: Replace `-999` in numeric columns with NaN
2. **Handle missing numeric data**:
   - Fill missing `salary` with median salary
   - Fill missing `years_experience` with median experience
   - Fill missing `performance_score` with 3.0 (neutral rating)
3. **Clean text data**:
   - Standardize `name` to Title Case (strip whitespace)
   - Standardize `department` to Title Case (strip whitespace)
   - Standardize `status` to lowercase (strip whitespace)
4. **Fix dates**: Convert `hire_date` to datetime, replace "UNKNOWN" and invalid dates with NaT
5. **Remove outliers**: Drop rows where salary is negative OR greater than $500,000
6. **Drop duplicates**: Remove duplicate rows

**Return**: Cleaned DataFrame

### Part 3: Data Transformation (25 points)

**Function**: `transform_data(df_clean)`

Add calculated and categorical columns to the cleaned data:
1. **Experience categories**: Create `experience_level` column using `pd.cut()`:
   - 0-2 years: "Junior"
   - 2-5 years: "Mid"
   - 5+ years: "Senior"
2. **Salary categories**: Create `salary_tier` column using `pd.qcut()`:
   - Bottom 33%: "Low"
   - Middle 33%: "Medium"
   - Top 33%: "High"
3. **Performance indicator**: Create `high_performer` boolean column:
   - True if `performance_score >= 4`
   - False otherwise
4. **Tenure calculation**: Create `tenure_years` column:
   - Calculate years since hire_date (use 2024 as current year)
   - Fill NaT dates with 0

**Return**: Transformed DataFrame with new columns

### Part 4: Data Analysis (15 points)

**Function**: `analyze_data(df_final)`

Analyze the cleaned and transformed data. Return a dictionary with:
- `avg_salary_by_dept`: Series with average salary per department
- `high_performer_count`: Total count of high performers
- `senior_avg_performance`: Average performance score for Senior experience level
- `top_department`: Department name with highest average salary

**Requirements**:
- Use `groupby()` for aggregations
- Filter data appropriately
- Return correct data types (Series, int/float, string)

## Grading Breakdown

- **Part 1 (20 points)**: Data audit function
  - Correct dictionary structure: 5 points
  - Accurate missing value counts: 5 points
  - Correct duplicate and data issue detection: 10 points

- **Part 2 (40 points)**: Data cleaning function
  - Sentinel value replacement: 5 points
  - Missing data handling: 10 points
  - Text standardization: 10 points
  - Date handling: 5 points
  - Outlier removal: 5 points
  - Duplicate removal: 5 points

- **Part 3 (25 points)**: Data transformation function
  - Experience categories: 7 points
  - Salary tiers: 7 points
  - Performance indicator: 6 points
  - Tenure calculation: 5 points

- **Part 4 (15 points)**: Data analysis function
  - Average salary by department: 5 points
  - High performer count: 3 points
  - Senior average performance: 4 points
  - Top department: 3 points

## Files Provided

- `employees_raw.csv`: Raw employee data (DO NOT MODIFY)
- `main.py`: Template with function stubs (COMPLETE THIS)
- `test_assignment.py`: Local tests you can run

## Testing Your Code

Run local tests:
```bash
pytest test_assignment.py -v
```

Run a specific test:
```bash
pytest test_assignment.py::test_audit_data -v
```

## Submission

Push your completed `main.py` to GitHub. The autograder will run and report your score (0-100).

**Important**:
- Only modify `main.py`
- Do not modify `employees_raw.csv`
- Your code must run without errors
- Partial credit available for partially working functions

## Tips

- Start with Part 1 to understand the data
- Use `.copy()` to avoid modifying the original DataFrame
- Test each function independently
- Check data types after transformations
- Use `df.head()` and `df.info()` to inspect results
- Remember: `cut()` creates equal-width bins, `qcut()` creates equal-frequency bins

Good luck! This exam tests your complete data cleaning pipeline skills.