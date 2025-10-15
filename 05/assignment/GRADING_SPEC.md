# Assignment 5 Midterm: Grading Specification

**Total: 100 points**
**Philosophy: Test behavior, not implementation**

## Partial Credit Allocation

This document shows **example partial credit breakdowns** for each question. Instructors may adjust point allocations based on:

- Solutions demonstrating mastery of core data science workflows
- Approaches showing understanding of pandas operations and data cleaning techniques
- Novel approaches that amuse us (; creative problem-solving and clean code

**Question Breakdown:**

- **Q1:** 10 points - Project setup and shell scripting
- **Q2:** 25 points - Python fundamentals and file processing
- **Q3:** 20 points - Pandas utility library development
- **Q4:** 15 points - Data exploration and basic analysis
- **Q5:** 15 points - Missing data analysis and cleaning
- **Q6:** 20 points - Data transformation and feature engineering
- **Q7:** 15 points - Aggregation and statistical analysis
- **Q8:** 5 points - Pipeline automation and orchestration

## Testing Structure

1. **Git Artifacts:** Check for committed files and outputs first
2. **Pipeline Execution:** Run scripts/notebooks in order (data generator → Q1 → Q2 → Q3 → Q4 → Q5 → Q6 → Q7 → Q8)
3. **Re-Check Artifacts** Check for outputs missing from git commit
4. **Function Testing:** Import Q3 functions and test with sample data
5. **Output Validation:** Verify all required output files exist and contain expected data


## Q1: Project Setup Script (10 points)

**Deliverable:** `q1_setup_project.sh` (executable shell script)

**Tests:**

1. Script is executable (`chmod +x`) - 2 pts
2. Has shebang (`#!/bin/bash` or `#!/bin/sh`) - 1 pt
3. Directory `data/` exists - 1 pt
4. Directory `output/` exists - 1 pt
5. Directory `reports/` exists - 1 pt
6. File `data/clinical_trial_raw.csv` exists (dataset generated) - 2 pts
7. File `reports/directory_structure.txt` exists with content - 2 pts


## Q2: Python Data Processing (25 points)

**Deliverable:** `q2_process_metadata.py` (executable script)

**Structural (4 pts):**

1. Script is executable - 1 pt
2. Has shebang (`#!/usr/bin/env python3` or `#!/usr/bin/python3`) - 1 pt
3. Has `if __name__ == '__main__':` - 1 pt
4. Script executes without errors - 1 pt

**Function: `parse_config(filepath: str) -> dict` (3 pts):**

```python
result = parse_config('config.txt')
```

1. Parses key=value format correctly - 1 pt
2. Contains sample_data_rows, sample_data_min, sample_data_max keys - 1 pt
3. Values match config.txt content - 1 pt

**Function: `validate_config(config: dict) -> dict` (3 pts):**

```python
config = {'sample_data_rows': '100', 'sample_data_min': '18', 'sample_data_max': '75'}
result = validate_config(config)
```

1. Returns validation results dict - 1 pt
2. Correctly validates sample_data_rows > 0 - 1 pt
3. Correctly validates sample_data_min >= 1 and sample_data_max > sample_data_min - 1 pt

**Function: `generate_sample_data(filename: str, config: dict) -> None` (3 pts):**

1. Creates file with correct number of rows - 1 pt
2. Generates numbers within specified range - 1 pt
3. File format is correct (one number per row, no header) - 1 pt

**Function: `calculate_statistics(data: list) -> dict` (3 pts):**

1. Returns dict with mean, median, sum, count keys - 1 pt
2. Calculates mean correctly - 1 pt
3. Calculates median correctly - 1 pt

**Q2 Required Outputs (9 pts):**

1. `data/sample_data.csv` exists - 3 pts
2. `output/statistics.txt` exists - 3 pts
3. Output files contain expected content - 3 pts

## Q3: Data Utilities Library (20 points)

**Deliverable:** `q3_data_utils.py` - Core reusable pandas functions

**Function: `load_data(filepath: str) -> pd.DataFrame` (2 pts):**

1. Loads CSV file successfully - 1 pt
2. Returns DataFrame with data - 1 pt

**Function: `clean_data(df, remove_duplicates, sentinel_value) -> pd.DataFrame` (2 pts):**

1. Removes duplicates if requested - 1 pt
2. Replaces sentinel values with NaN - 1 pt

**Function: `detect_missing(df) -> pd.Series` (2 pts):**

1. Returns Series with missing value counts per column - 1 pt
2. Handles all column types correctly - 1 pt

**Function: `fill_missing(df, column, strategy) -> pd.DataFrame` (2 pts):**

1. Handles 'mean' and 'median' strategies - 1 pt
2. Handles 'ffill' strategy - 1 pt

**Function: `filter_data(df, filters) -> pd.DataFrame` (5 pts):**

1. Applies single filter correctly - 1 pt
2. Applies multiple filters in sequence - 1 pt
3. Handles 'equals' and 'greater_than' conditions - 1 pt
4. Handles 'less_than' and 'in_range' conditions - 1 pt
5. Handles 'in_list' condition - 1 pt

**Function: `transform_types(df, type_map) -> pd.DataFrame` (2 pts):**

1. Converts to datetime and numeric - 1 pt
2. Converts to category - 1 pt

**Function: `create_bins(df, column, bins, labels, new_column=None) -> pd.DataFrame` (2 pts):**

1. Creates binned column with correct labels - 1 pt
2. Handles new_column parameter correctly - 1 pt

**Function: `summarize_by_group(df, group_col, agg_dict=None) -> pd.DataFrame` (3 pts):**

1. Groups by specified column - 1 pt
2. Applies default aggregation (.describe()) - 1 pt
3. Handles custom agg_dict parameter - 1 pt


## Q4: Data Exploration Notebook (15 points)

**Deliverable:** `q4_exploration.ipynb`

**Required Output:**

- `output/q4_site_counts.csv` - Value counts for 'site' column (5 rows, 2 columns)

**Grading (points awarded for notebook execution + output validation):**

1. Notebook executes without errors - 3 pts
2. `output/q4_site_counts.csv` exists - 2 pts
3. CSV has correct structure (5 rows for 5 sites) - 3 pts
4. CSV contains expected columns (site, count) - 1 pt
5. Counts sum to 10000 - 1 pt


## Q5: Missing Data Analysis Notebook (15 points)

**Deliverable:** `q5_missing_data.ipynb`

**Required Outputs:**

- `output/q5_cleaned_data.csv` - Cleaned dataset with missing data handled
- `output/q5_missing_report.txt` - Text report on missing data

**Grading:**

1. Notebook executes without errors - 3 pts
2. `output/q5_cleaned_data.csv` exists - 2 pts
3. Cleaned data has fewer/no missing values than original - 3 pts
4. `output/q5_missing_report.txt` exists - 1 pt
5. Report contains missing value counts - 1 pt


## Q6: Data Transformation Notebook (20 points)

**Deliverable:** `q6_transformation.ipynb`

**Required Output:**

- `output/q6_transformed_data.csv` - Transformed dataset with new features

**Grading:**

1. Notebook executes without errors - 4 pts
2. `output/q6_transformed_data.csv` exists - 2 pts
3. Output has more columns than input (new features added) - 3 pts
4. Contains binned/categorical columns (e.g., age_group, bmi_category) - 3 pts
5. Contains calculated columns (e.g., cholesterol_ratio) - 3 pts


## Q7: Aggregation & Analysis Notebook (15 points)

**Deliverable:** `q7_aggregation.ipynb`

**Required Outputs:**

- `output/q7_site_summary.csv` - Summary statistics by site
- `output/q7_intervention_comparison.csv` - Comparison across intervention groups
- `output/q7_analysis_report.txt` - Analysis findings

**Grading:**

1. Notebook executes without errors - 3 pts
2. `output/q7_site_summary.csv` exists with 5 rows (one per site) - 2 pts
3. `output/q7_intervention_comparison.csv` exists - 2 pts
4. `output/q7_analysis_report.txt` exists with text content - 2 pts
5. Summary files contain aggregated/grouped data (not raw data) - 1 pt


## Q8: Pipeline Automation (5 points)

**Deliverable:** `q8_run_pipeline.sh` (executable)

**Tests:**

1. Script is executable - 1 pt
2. Has shebang (`#!/bin/bash` or `#!/bin/sh`) - 1 pt
3. `reports/pipeline_log.txt` exists - 1 pt
4. Script runs Q4-Q7 notebooks in order (assumes Q1 already run, Q2 and Q3 are not run directly) - 1 pt
5. Pipeline log shows successful execution - 1 pt


## Testing Implementation

### Test Q2 Functions

```python
from q2_process_metadata import parse_config, validate_config, generate_sample_data, calculate_statistics

# Test parse_config
config = parse_config('q2_config.txt')
assert isinstance(config, dict)
assert 'sample_data_rows' in config
assert config['sample_data_min'] == '18'

# Test validate_config
test_config = {'sample_data_rows': '100', 'sample_data_min': '18', 'sample_data_max': '75'}
validation = validate_config(test_config)
assert isinstance(validation, dict)
assert validation['sample_data_rows'] == True  # 100 > 0
assert validation['sample_data_min'] == True  # 18 >= 1

# Test generate_sample_data
config = {'sample_data_rows': '10', 'sample_data_min': '18', 'sample_data_max': '75'}
generate_sample_data('test_sample.csv', config)
# Verify file was created and has correct number of lines
with open('test_sample.csv') as f:
    lines = f.readlines()
    assert len(lines) == 10  # 10 rows as specified

# Test calculate_statistics
stats = calculate_statistics([10, 20, 30, 40, 50])
assert stats['mean'] == 30.0
assert stats['median'] == 30.0
```

### Test Q3 Functions

```python
from q3_data_utils import load_data, clean_data, detect_missing, fill_missing, filter_data, transform_types, create_bins, summarize_by_group

# Test load_data
df = load_data('data/clinical_trial_raw.csv')
assert isinstance(df, pd.DataFrame)
assert len(df) > 0  # Has data
assert len(df.columns) > 0  # Has columns

# Test clean_data
cleaned = clean_data(df, remove_duplicates=True, sentinel_value=-999)
assert isinstance(cleaned, pd.DataFrame)

# Test detect_missing
missing = detect_missing(df)
assert isinstance(missing, pd.Series)
assert len(missing) == len(df.columns)  # One count per column

# Test fill_missing with test data
test_df = pd.DataFrame({'col': [1, np.nan, 3]})
filled = fill_missing(test_df, 'col', 'mean')
assert filled['col'].isnull().sum() == 0

# Test filter_data
filters = [{'column': 'age', 'condition': 'greater_than', 'value': 65}]
filtered = filter_data(df, filters)
assert all(filtered['age'] > 65)

# Test multiple filters
filters = [
    {'column': 'age', 'condition': 'greater_than', 'value': 18},
    {'column': 'site', 'condition': 'equals', 'value': 'Site A'}
]
filtered = filter_data(df, filters)
assert all(filtered['age'] > 18)
assert all(filtered['site'] == 'Site A')

# Test in_list condition
filters = [{'column': 'site', 'condition': 'in_list', 'value': ['Site A', 'Site B']}]
filtered = filter_data(df, filters)
assert all(filtered['site'].isin(['Site A', 'Site B']))

# Test in_range condition
filters = [{'column': 'age', 'condition': 'in_range', 'value': [18, 65]}]
filtered = filter_data(df, filters)
assert all(filtered['age'] >= 18)
assert all(filtered['age'] <= 65)

# Test transform_types
type_map = {'enrollment_date': 'datetime', 'site': 'category'}
typed = transform_types(df, type_map)
assert isinstance(typed, pd.DataFrame)

# Test create_bins
binned = create_bins(df, 'age', [0, 40, 60, 100], ['<40', '40-59', '60+'])
assert isinstance(binned, pd.DataFrame)

# Test create_bins with custom column name
binned_custom = create_bins(df, 'age', [0, 40, 60, 100], ['<40', '40-59', '60+'], 'age_groups')
assert isinstance(binned_custom, pd.DataFrame)
assert 'age_groups' in binned_custom.columns

# Test summarize_by_group with custom aggregation
summary = summarize_by_group(df, 'site', {'age': 'mean'})
assert isinstance(summary, pd.DataFrame)

# Test summarize_by_group with default aggregation (agg_dict=None)
summary_default = summarize_by_group(df, 'site')
assert isinstance(summary_default, pd.DataFrame)
```

### Test Notebook Outputs

```python
import os
import pandas as pd

# Q4 outputs
assert os.path.exists('output/q4_site_counts.csv')
df = pd.read_csv('output/q4_site_counts.csv')
assert len(df) >= 1  # Has at least one site
assert len(df.columns) >= 2  # Has site and count columns
assert df.iloc[:, 1].sum() > 0  # Counts sum to positive number

# Q2 outputs
assert os.path.exists('data/sample_data.csv')
assert os.path.exists('output/statistics.txt')

# Q5 outputs
assert os.path.exists('output/q5_cleaned_data.csv')
assert os.path.exists('output/q5_missing_report.txt')

# Q6 outputs
assert os.path.exists('output/q6_transformed_data.csv')
original = pd.read_csv('data/clinical_trial_raw.csv')
transformed = pd.read_csv('output/q6_transformed_data.csv')
assert len(transformed.columns) > len(original.columns)

# Q7 outputs
assert os.path.exists('output/q7_site_summary.csv')
summary = pd.read_csv('output/q7_site_summary.csv')
assert len(summary) >= 1  # Has at least one site
```

### Test Shell Scripts

```python
import os

# Q1 executable check
assert os.access('q1_setup_project.sh', os.X_OK)
with open('q1_setup_project.sh') as f:
    assert f.readline().startswith('#!/bin/bash')

# Q1 outputs
assert os.path.exists('data/')
assert os.path.exists('output/')
assert os.path.exists('reports/')
assert os.path.exists('reports/directory_structure.txt')

# Q8 outputs
assert os.path.exists('reports/pipeline_log.txt')
```


## Grading Summary

**Total: 100 points**

- Q1: 10 pts (shell script, directories, files)
- Q2: 25 pts (Python fundamentals, 4 functions)
- Q3: 20 pts (pandas utility library, 8 functions)
- Q4: 15 pts (exploration notebook + outputs)
- Q5: 15 pts (missing data notebook + outputs)
- Q6: 20 pts (transformation notebook + outputs)
- Q7: 15 pts (aggregation notebook + outputs)
- Q8: 5 pts (pipeline automation)

**Grading Scale:**

- 90-100: A (90%+)
- 80-89: B (80%+)
- 70-79: C (70%+)
- 60-69: D (60%+)
- <60: F

**Testing Approach:**

- **Scripts (Q1-Q2, Q8):** Import functions and test directly with known inputs
- **Q3 Utility Library:** Import and test functions with sample data
- **Notebooks (Q4-Q7):** Execute notebooks + validate output artifacts

**Artifact-Based Grading:**
All notebook questions graded primarily on:

1. Notebook executes without errors (proves code works)
2. Required output files exist (fixed filenames)
3. Output files have expected structure/content (objective validation)

This approach is non-fragile - we test deliverables, not implementation details.

## Testing Robustness

**Key Principles:**

- Tests should be **data-agnostic** - work regardless of specific dataset values
- Tests should be **implementation-agnostic** - focus on behavior, not code structure
- Tests should handle **edge cases** gracefully
- Tests should provide **clear failure messages** for debugging

**Robust Test Examples:**

```python
# Good: Tests behavior, not specific values
assert len(df) > 0  # Has data
assert 'age' in df.columns  # Has expected column

# Avoid: Tests specific values that might change
assert df.shape == (10000, 18)  # Too rigid
assert df['age'].mean() == 45.2  # Too specific
```

**Edge Case Handling:**

- Empty datasets
- Missing columns
- Invalid data types
- File I/O errors
- Division by zero
- Missing configuration files
