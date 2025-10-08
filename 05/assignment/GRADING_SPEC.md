# Assignment 5 Midterm: Grading Specification

**Total: 100 points across ~85 behavioral tests**
**Max per test: 4 points (most are 1-2 pts)**
**Philosophy: Test behavior, not implementation**

---

## Q1: Project Setup Script (10 points)

**Deliverable:** `setup_project.sh` (executable shell script)

**Tests:**
1. Script is executable (`chmod +x`) - 2 pts
2. Has shebang `#!/bin/bash` - 1 pt
3. Directory `data/` exists - 2 pts
4. Directory `output/` exists - 2 pts
5. Directory `reports/` exists - 2 pts
6. File `reports/directory_structure.txt` exists with content - 1 pt

---

## Q2: Python Data Processing (25 points)

**Deliverable:** `process_metadata.py` (executable script)

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
config = {'threshold': '100', 'status': 'active'}
result = validate_config(config)
```
1. Returns dict - 1 pt
2. Validation logic correct for 'threshold' - 1 pt
3. Validation logic correct for 'status' - 1 pt
4. All validations work - 1 pt

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

## Q3: Data Loading & Exploration (10 points)

**Deliverable:** `exploration.py`

**Function: `load_data(filepath: str) -> pd.DataFrame` (2 pts):**
1. Returns DataFrame - 1 pt
2. Correct shape - 1 pt

**Function: `get_summary_stats(df: pd.DataFrame) -> pd.DataFrame` (4 pts):**
1. Returns DataFrame from .describe() - 1 pt
2. Has 8 rows (count, mean, std, min, 25%, 50%, 75%, max) - 2 pts
3. Values are correct - 1 pt

**Function: `get_value_counts(df: pd.DataFrame, column: str) -> pd.Series` (2 pts):**
1. Returns Series - 1 pt
2. Counts are correct - 1 pt

**Outputs (2 pts):**
1. `output/summary_stats.csv` exists - 1 pt
2. `output/value_counts.csv` exists - 1 pt

---

## Q4: Data Selection & Filtering (10 points)

**Deliverable:** `selection.py`

**Function: `select_numeric_columns(df) -> pd.DataFrame` (2 pts):**
1. Returns only numeric columns - 1 pt
2. All numeric columns present - 1 pt

**Function: `select_by_loc(df, rows: list, cols: list) -> pd.DataFrame` (2 pts):**
1. Correct shape - 1 pt
2. Correct rows and columns - 1 pt

**Function: `select_by_iloc(df, row_start: int, row_end: int, col_indices: list) -> pd.DataFrame` (1 pt):**
1. Correct subset - 1 pt

**Function: `filter_single_condition(df, column: str, threshold: float) -> pd.DataFrame` (2 pts):**
1. Filters correctly - 1 pt
2. Correct row count - 1 pt

**Function: `filter_multiple_conditions(df, col1: str, val1, col2: str, val2) -> pd.DataFrame` (2 pts):**
1. Both conditions applied - 1 pt
2. Correct result - 1 pt

**Function: `filter_by_category(df, column: str, categories: list) -> pd.DataFrame` (1 pt):**
1. Filters to categories - 1 pt

---

## Q5: Missing Data Handling (15 points)

**Deliverable:** `missing_data.py`

**Function: `detect_missing(df) -> pd.Series` (2 pts):**
1. Returns Series - 1 pt
2. Counts correct - 1 pt

**Function: `fill_with_mean(df, column: str) -> pd.DataFrame` (3 pts):**
1. Returns DataFrame - 1 pt
2. No missing values in column - 1 pt
3. Filled with correct mean - 1 pt

**Function: `fill_with_median(df, column: str) -> pd.DataFrame` (3 pts):**
1. Returns DataFrame - 1 pt
2. No missing values - 1 pt
3. Filled with correct median - 1 pt

**Function: `forward_fill(df, column: str) -> pd.DataFrame` (3 pts):**
1. Returns DataFrame - 1 pt
2. No gaps in data - 1 pt
3. Values propagated correctly - 1 pt

**Function: `drop_missing_rows(df, subset: list) -> pd.DataFrame` (4 pts):**
1. Returns DataFrame - 1 pt
2. Rows with missing in subset dropped - 2 pts
3. Correct row count - 1 pt

---

## Q6: Data Transformation (15 points)

**Deliverable:** `transform.py`

**Function: `convert_to_datetime(df, column: str) -> pd.DataFrame` (2 pts):**
1. Column dtype is datetime64 - 1 pt
2. Values are valid - 1 pt

**Function: `convert_to_numeric(df, column: str) -> pd.DataFrame` (2 pts):**
1. Column dtype is numeric - 1 pt
2. Invalid values became NaN - 1 pt

**Function: `convert_to_category(df, column: str) -> pd.DataFrame` (2 pts):**
1. Column dtype is category - 1 pt
2. Categories correct - 1 pt

**Function: `replace_sentinels(df, column: str, sentinel) -> pd.DataFrame` (2 pts):**
1. Sentinel values gone - 1 pt
2. NaN in their place - 1 pt

**Function: `apply_custom_function(df, column: str, func) -> pd.DataFrame` (2 pts):**
1. Function applied - 1 pt
2. Results correct - 1 pt

**Function: `map_values(df, column: str, mapping: dict) -> pd.DataFrame` (2 pts):**
1. Values mapped - 1 pt
2. Mapping correct - 1 pt

**Function: `clean_strings(df, column: str) -> pd.DataFrame` (1 pt):**
1. Strings lowercase and stripped - 1 pt

**Function: `add_calculated_column(df, new_col: str, col1: str, col2: str) -> pd.DataFrame` (2 pts):**
1. New column exists - 1 pt
2. Calculation correct - 1 pt

---

## Q7: Groupby & Aggregation (10 points)

**Deliverable:** `aggregation.py`

**Function: `group_and_sum(df, group_col: str, sum_col: str) -> pd.DataFrame` (3 pts):**
1. Returns DataFrame/Series - 1 pt
2. Groups correct - 1 pt
3. Sums correct - 1 pt

**Function: `group_and_aggregate_multiple(df, group_col: str, agg_dict: dict) -> pd.DataFrame` (3 pts):**
1. Returns DataFrame - 1 pt
2. Multiple aggregations present - 1 pt
3. Values correct - 1 pt

**Function: `get_top_n(df, column: str, n: int) -> pd.DataFrame` (2 pts):**
1. Correct number of rows - 1 pt
2. Top values correct - 1 pt

**Function: `group_and_sort(df, group_col: str, agg_col: str) -> pd.DataFrame` (2 pts):**
1. Sorted correctly - 1 pt
2. Aggregation correct - 1 pt

---

## Q8: Pipeline Automation (5 points)

**Deliverable:** `run_pipeline.sh` (executable)

**Tests:**
1. Script is executable - 1 pt
2. Has shebang `#!/bin/bash` - 1 pt
3. `reports/pipeline_log.txt` exists - 1 pt
4. `reports/quality_report.txt` exists - 1 pt
5. `output/final_clean_data.csv` exists - 1 pt

---

## Testing Implementation

### Import and Test Functions
```python
from process_metadata import parse_config, validate_config
from exploration import load_data, get_summary_stats
from selection import select_numeric_columns, filter_by_category
from missing_data import fill_with_mean, drop_missing_rows
from transform import convert_to_datetime, apply_custom_function
from aggregation import group_and_sum, get_top_n

# Test with known inputs
config = parse_config('test_config.txt')
assert config['project_name'] == 'DataAnalysis'

df = load_data('test_data.csv')
assert df.shape == (100, 10)
```

### Check File Outputs
```python
import os
import pandas as pd

# Files exist
assert os.path.exists('output/summary_stats.csv')

# Files have correct content
df = pd.read_csv('output/summary_stats.csv')
assert df.shape[0] == 8  # describe() has 8 rows
```

### Check Executables
```python
# Script is executable
assert os.access('setup_project.sh', os.X_OK)

# Has shebang
with open('setup_project.sh') as f:
    assert f.readline().startswith('#!/bin/bash')
```

---

## Grading Summary

**Total: 100 points**
- Q1: 10 pts (shell script, directories, files)
- Q2: 25 pts (Python fundamentals, functions, control flow)
- Q3: 10 pts (pandas loading, exploration)
- Q4: 10 pts (pandas selection, filtering)
- Q5: 15 pts (missing data handling)
- Q6: 15 pts (data transformation)
- Q7: 10 pts (groupby, aggregation)
- Q8: 5 pts (pipeline automation)

**Grading Scale:**
- 90-100: A
- 80-89: B
- 70-79: C
- 60-69: D
- <60: F

**Robustness:** Students can miss 10 points (10 tests) and still get a B!

**All tests are behavioral - we test WHAT the code does, not HOW it's written.**
