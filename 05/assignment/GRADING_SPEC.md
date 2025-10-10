# Assignment 5 Midterm: Grading Specification

**Total: 125 points**
**Philosophy: Test behavior, not implementation**

## Structure
- **Scripts (Q1-Q2, Q8):** Test functions via imports and file outputs
- **Q3 Utility Library:** Test functions via imports
- **Notebooks (Q4-Q7):** Test via execution and output validation

---

## Q1: Project Setup Script (10 points)

**Deliverable:** `q1_setup_project.sh` (executable shell script)

**Tests:**
1. Script is executable (`chmod +x`) - 2 pts
2. Has shebang `#!/bin/bash` - 1 pt
3. Directory `data/` exists - 2 pts
4. Directory `output/` exists - 2 pts
5. Directory `reports/` exists - 2 pts
6. File `reports/directory_structure.txt` exists with content - 1 pt

---

## Q2: Python Data Processing (25 points)

**Deliverable:** `q2_process_metadata.py` (executable script)

**Structural (5 pts):**
1. Script is executable - 1 pt
2. Has shebang `#!/usr/bin/env python3` - 1 pt
3. Has `if __name__ == '__main__':` - 1 pt
4. Functions are importable - 2 pts

**Function: `parse_config(filepath: str) -> dict` (4 pts):**
```python
result = parse_config('config.txt')
```
1. Returns dict - 1 pt
2. Has expected keys - 2 pts
3. Has expected values - 1 pt

**Function: `validate_config(config: dict) -> dict` (4 pts):**
```python
config = {'min_age': '18', 'max_age': '85', 'target_enrollment': '10000', 'sites': '5'}
result = validate_config(config)
```
1. Returns dict - 1 pt
2. Validates min_age >= 18 - 1 pt
3. Validates max_age <= 100 - 1 pt
4. Validates target_enrollment > 0 and sites >= 1 - 1 pt

**Function: `process_files(file_list: list) -> list` (4 pts):**
```python
files = ['data1.csv', 'script.py', 'data2.csv', 'readme.txt']
result = process_files(files)
```
1. Returns list - 1 pt
2. Filters to only .csv files - 2 pts
3. Correct count - 1 pt

**Function: `calculate_statistics(data: list) -> dict` (4 pts):**
```python
data = [10, 20, 30, 40, 50]
result = calculate_statistics(data)
```
1. Returns dict - 1 pt
2. 'mean' = 30.0 - 1 pt
3. 'median' = 30.0 - 1 pt
4. Has 'sum' and 'count' - 1 pt

**Outputs (4 pts):**
1. `output/config_summary.txt` exists - 1 pt
2. `output/validation_report.txt` exists - 1 pt
3. `output/file_manifest.txt` exists - 1 pt
4. `output/statistics.txt` exists - 1 pt

---

## Q3: Data Utilities Library (20 points)

**Deliverable:** `q3_data_utils.py` - Core reusable pandas functions

**Function: `load_data(filepath: str) -> pd.DataFrame` (2 pts):**
1. Returns DataFrame - 1 pt
2. Correct shape - 1 pt

**Function: `clean_data(df, remove_duplicates, sentinel_value) -> pd.DataFrame` (3 pts):**
1. Returns DataFrame - 1 pt
2. Removes duplicates if requested - 1 pt
3. Replaces sentinel values with NaN - 1 pt

**Function: `detect_missing(df) -> pd.Series` (2 pts):**
1. Returns Series - 1 pt
2. Counts correct - 1 pt

**Function: `fill_missing(df, column, strategy) -> pd.DataFrame` (3 pts):**
1. Handles 'mean' strategy - 1 pt
2. Handles 'median' strategy - 1 pt
3. Handles 'ffill' strategy - 1 pt

**Function: `filter_data(df, column, **conditions) -> pd.DataFrame` (3 pts):**
1. Exact value filtering works - 1 pt
2. Range filtering works - 1 pt
3. List filtering (.isin) works - 1 pt

**Function: `transform_types(df, type_map) -> pd.DataFrame` (3 pts):**
1. Converts to datetime - 1 pt
2. Converts to numeric - 1 pt
3. Converts to category - 1 pt

**Function: `create_bins(df, column, bins, labels) -> pd.DataFrame` (2 pts):**
1. Creates binned column - 1 pt
2. Bins applied correctly - 1 pt

**Function: `summarize_by_group(df, group_col, agg_dict) -> pd.DataFrame` (2 pts):**
1. Groups correctly - 1 pt
2. Aggregations correct - 1 pt

---

## Q4: Data Exploration Notebook (15 points)

**Deliverable:** `q4_exploration.ipynb`

**Required Output:**
- `output/q4_site_counts.csv` - Value counts for 'site' column (5 rows, 2 columns)

**Grading (points awarded for notebook execution + output validation):**
1. Notebook executes without errors - 5 pts
2. `output/q4_site_counts.csv` exists - 3 pts
3. CSV has correct structure (5 rows for 5 sites) - 3 pts
4. CSV contains expected columns (site, count) - 2 pts
5. Counts sum to 10000 - 2 pts

---

## Q5: Missing Data Analysis Notebook (15 points)

**Deliverable:** `q5_missing_data.ipynb`

**Required Outputs:**
- `output/q5_cleaned_data.csv` - Cleaned dataset with missing data handled
- `output/q5_missing_report.txt` - Text report on missing data

**Grading:**
1. Notebook executes without errors - 5 pts
2. `output/q5_cleaned_data.csv` exists - 3 pts
3. Cleaned data has fewer/no missing values than original - 3 pts
4. `output/q5_missing_report.txt` exists - 2 pts
5. Report contains missing value counts - 2 pts

---

## Q6: Data Transformation Notebook (20 points)

**Deliverable:** `q6_transformation.ipynb`

**Required Output:**
- `output/q6_transformed_data.csv` - Transformed dataset with new features

**Grading:**
1. Notebook executes without errors - 5 pts
2. `output/q6_transformed_data.csv` exists - 3 pts
3. Output has more columns than input (new features added) - 4 pts
4. Contains binned/categorical columns (e.g., age_group, bmi_category) - 4 pts
5. Contains calculated columns (e.g., cholesterol_ratio) - 4 pts

---

## Q7: Aggregation & Analysis Notebook (15 points)

**Deliverable:** `q7_aggregation.ipynb`

**Required Outputs:**
- `output/q7_site_summary.csv` - Summary statistics by site
- `output/q7_intervention_comparison.csv` - Comparison across intervention groups
- `output/q7_analysis_report.txt` - Analysis findings

**Grading:**
1. Notebook executes without errors - 5 pts
2. `output/q7_site_summary.csv` exists with 5 rows (one per site) - 3 pts
3. `output/q7_intervention_comparison.csv` exists - 3 pts
4. `output/q7_analysis_report.txt` exists with text content - 2 pts
5. Summary files contain aggregated/grouped data (not raw data) - 2 pts

---

## Q8: Pipeline Automation (5 points)

**Deliverable:** `q8_run_pipeline.sh` (executable)

**Tests:**
1. Script is executable - 1 pt
2. Has shebang `#!/bin/bash` - 1 pt
3. `reports/pipeline_log.txt` exists - 1 pt
4. `reports/quality_report.txt` exists - 1 pt
5. `output/final_clean_data.csv` exists - 1 pt

---

## Testing Implementation

### Test Q2 Functions
```python
from q2_process_metadata import parse_config, validate_config, process_files, calculate_statistics

# Test parse_config
config = parse_config('config.txt')
assert isinstance(config, dict)
assert 'study_name' in config
assert config['min_age'] == '18'

# Test validate_config
test_config = {'min_age': '18', 'max_age': '85', 'target_enrollment': '10000', 'sites': '5'}
validation = validate_config(test_config)
assert isinstance(validation, dict)
assert validation['min_age'] == True  # 18 >= 18

# Test process_files
files = ['data1.csv', 'script.py', 'data2.csv']
csv_files = process_files(files)
assert len(csv_files) == 2
assert 'script.py' not in csv_files

# Test calculate_statistics
stats = calculate_statistics([10, 20, 30, 40, 50])
assert stats['mean'] == 30.0
assert stats['median'] == 30.0
```

### Test Q3 Functions
```python
from q3_data_utils import load_data, detect_missing, fill_missing, filter_data

# Test load_data
df = load_data('data/clinical_trial_raw.csv')
assert isinstance(df, pd.DataFrame)
assert df.shape == (10000, 18)

# Test detect_missing
missing = detect_missing(df)
assert isinstance(missing, pd.Series)
assert len(missing) == 18  # One count per column

# Test fill_missing with test data
test_df = pd.DataFrame({'col': [1, np.nan, 3]})
filled = fill_missing(test_df, 'col', 'mean')
assert filled['col'].isnull().sum() == 0

# Test filter_data
filtered = filter_data(df, 'age', min_value=65, max_value=100)
assert all(filtered['age'] >= 65)
assert all(filtered['age'] <= 100)
```

### Test Notebook Outputs
```python
import os
import pandas as pd

# Q4 outputs
assert os.path.exists('output/q4_site_counts.csv')
df = pd.read_csv('output/q4_site_counts.csv')
assert len(df) == 5  # 5 sites
assert df.iloc[:, 1].sum() == 10000  # Counts sum to total

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
assert len(summary) == 5  # One row per site
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
assert os.path.exists('output/final_clean_data.csv')
```

---

## Grading Summary

**Total: 125 points**
- Q1: 10 pts (shell script, directories, files)
- Q2: 25 pts (Python fundamentals, 4 functions)
- Q3: 20 pts (pandas utility library, 8 functions)
- Q4: 15 pts (exploration notebook + outputs)
- Q5: 15 pts (missing data notebook + outputs)
- Q6: 20 pts (transformation notebook + outputs)
- Q7: 15 pts (aggregation notebook + outputs)
- Q8: 5 pts (pipeline automation)

**Grading Scale:**
- 113-125: A (90%+)
- 100-112: B (80%+)
- 88-99: C (70%+)
- 75-87: D (60%+)
- <75: F

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
