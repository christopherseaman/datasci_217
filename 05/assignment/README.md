# Assignment 5: Midterm Exam - Clinical Trial Data Analysis

**Total: 100 points**

## Scenario

You're analyzing data from a multi-site cardiovascular health clinical trial. The trial enrolled 10,000 patients across 5 hospital sites over 2 years, testing two different treatment interventions against a control group.

**Your task:** Build a complete data processing pipeline to clean, validate, and analyze the clinical trial data.

**Dataset:** `data/clinical_trial_raw.csv` (10,000 patients, 18 variables)

**Configuration:** `config.txt` (trial parameters)

**Variables:**
- **Demographics:** patient_id, age, sex, bmi, enrollment_date
- **Clinical measurements:** systolic_bp, diastolic_bp, cholesterol_total, cholesterol_hdl, cholesterol_ldl, glucose_fasting
- **Trial info:** site, intervention_group, follow_up_months, adverse_events
- **Outcomes:** outcome_cvd, adherence_pct, dropout

---

## Question 1: Project Setup Script (10 points)

**File:** `setup_project.sh`

Create an executable shell script that sets up the project structure.

### Requirements:

```bash
#!/bin/bash

# Create directory structure
mkdir -p data
mkdir -p output
mkdir -p reports

# Move/copy raw data file to data/ (if not already there)
# Save directory structure to reports/directory_structure.txt
```

**Your script must:**
1. Be executable (`chmod +x setup_project.sh`) - **2 pts**
2. Have shebang `#!/bin/bash` - **1 pt**
3. Create `data/` directory - **2 pts**
4. Create `output/` directory - **2 pts**
5. Create `reports/` directory - **2 pts**
6. Save directory listing to `reports/directory_structure.txt` - **1 pt**

**Test it:**
```bash
./setup_project.sh
ls -la
cat reports/directory_structure.txt
```

---

## Question 2: Python Data Processing (25 points)

**File:** `process_metadata.py`

Create an executable Python script that processes the trial configuration file (`config.txt`).

The config file format is:
```
study_name=CardioHealth Trial 2023
primary_investigator=Dr. Sarah Chen
min_age=18
max_age=85
target_enrollment=10000
sites=5
```

### Required Functions:

```python
#!/usr/bin/env python3
"""Process clinical trial metadata."""

def parse_config(filepath: str) -> dict:
    """
    Parse config file (key=value format) into dictionary.

    Args:
        filepath: Path to config.txt

    Returns:
        dict: Configuration as key-value pairs
    """
    # TODO: Read file, split on '=', create dict
    pass

def validate_config(config: dict) -> dict:
    """
    Validate configuration values using if/elif/else logic.

    Rules:
    - min_age must be >= 18
    - max_age must be <= 100
    - target_enrollment must be > 0
    - sites must be >= 1

    Args:
        config: Configuration dictionary

    Returns:
        dict: Validation results {key: True/False}
    """
    # TODO: Implement with if/elif/else
    pass

def process_files(file_list: list) -> list:
    """
    Filter file list to only .csv files.

    Args:
        file_list: List of filenames

    Returns:
        list: Filtered list of .csv files only
    """
    # TODO: Filter to .csv files (any valid approach)
    pass

def calculate_statistics(data: list) -> dict:
    """
    Calculate basic statistics.

    Args:
        data: List of numbers

    Returns:
        dict: {mean, median, sum, count}
    """
    # TODO: Calculate stats
    pass

if __name__ == '__main__':
    # TODO: Load and process config
    # TODO: Save outputs to output/
```

### Grading:

**Structure (5 pts):**
- Executable with shebang - 2 pts
- Has `if __name__ == '__main__':` - 1 pt
- Functions importable - 2 pts

**Functions (16 pts):**
- `parse_config()` works correctly - 4 pts
- `validate_config()` logic correct - 4 pts
- `process_files()` filters correctly - 4 pts
- `calculate_statistics()` returns correct stats - 4 pts

**Outputs (4 pts):**
- `output/config_summary.txt` - 1 pt
- `output/validation_report.txt` - 1 pt
- `output/file_manifest.txt` - 1 pt
- `output/statistics.txt` - 1 pt

---

## Question 3: Data Loading & Exploration (10 points)

**File:** `exploration.py`

Create a module to load and explore the clinical trial data.

### Required Functions:

```python
"""Data loading and exploration functions."""

import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    pass

def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return df.describe() for numeric columns."""
    pass

def get_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    """Return value_counts for specified column."""
    pass
```

### Save these outputs:
- `output/summary_stats.csv` (from .describe())
- `output/value_counts_site.csv` (value_counts for 'site' column)

### Grading (10 pts):
- `load_data()` returns DataFrame - 2 pts
- `get_summary_stats()` correct - 4 pts
- `get_value_counts()` correct - 2 pts
- Output files exist - 2 pts

---

## Question 4: Data Selection & Filtering (10 points)

**File:** `selection.py`

Create functions demonstrating different pandas selection methods.

### Required Functions:

```python
"""Data selection and filtering functions."""

import pandas as pd

def select_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric columns using .select_dtypes()."""
    pass

def select_by_loc(df: pd.DataFrame, rows: list, cols: list) -> pd.DataFrame:
    """Use .loc[] to select specific rows and columns."""
    pass

def select_by_iloc(df: pd.DataFrame, row_start: int, row_end: int, col_indices: list) -> pd.DataFrame:
    """Use .iloc[] for position-based selection."""
    pass

def filter_single_condition(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    """Filter using single boolean condition (e.g., age > threshold)."""
    pass

def filter_multiple_conditions(df: pd.DataFrame, col1: str, val1, col2: str, val2) -> pd.DataFrame:
    """Filter using multiple conditions with & operator."""
    pass

def filter_by_category(df: pd.DataFrame, column: str, categories: list) -> pd.DataFrame:
    """Filter using .isin() for categorical values."""
    pass
```

### Grading (10 pts):
- `select_numeric_columns()` - 2 pts
- `select_by_loc()` - 2 pts
- `select_by_iloc()` - 1 pt
- `filter_single_condition()` - 2 pts
- `filter_multiple_conditions()` - 2 pts
- `filter_by_category()` - 1 pt

---

## Question 5: Missing Data Handling (15 points)

**File:** `missing_data.py`

Implement multiple strategies for handling missing data.

### Required Functions:

```python
"""Missing data detection and handling."""

import pandas as pd
import numpy as np

def detect_missing(df: pd.DataFrame) -> pd.Series:
    """Return .isnull().sum() for all columns."""
    pass

def fill_with_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Fill missing values in column with mean using .fillna()."""
    pass

def fill_with_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Fill missing values in column with median."""
    pass

def forward_fill(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Forward fill missing values using .fillna(method='ffill')."""
    pass

def drop_missing_rows(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """Drop rows with missing values in specified subset of columns."""
    pass
```

### Grading (15 pts):
- `detect_missing()` - 2 pts
- `fill_with_mean()` - 3 pts
- `fill_with_median()` - 3 pts
- `forward_fill()` - 3 pts
- `drop_missing_rows()` - 4 pts

---

## Question 6: Data Transformation (23 points)

**File:** `transform.py`

Implement data cleaning and transformation functions.

### Required Functions:

```python
"""Data transformation functions."""

import pandas as pd
import numpy as np

def convert_to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert column to datetime using pd.to_datetime()."""
    pass

def convert_to_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert to numeric using pd.to_numeric(errors='coerce')."""
    pass

def convert_to_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert column to category type using .astype('category')."""
    pass

def replace_sentinels(df: pd.DataFrame, column: str, sentinel_value) -> pd.DataFrame:
    """Replace sentinel values (e.g., -999) with NaN using .replace()."""
    pass

def apply_custom_function(df: pd.DataFrame, column: str, func) -> pd.DataFrame:
    """Apply custom function to column using .apply()."""
    pass

def map_values(df: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
    """Map values using dictionary with .map()."""
    pass

def clean_strings(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Clean strings using .str.lower() and .str.strip()."""
    pass

def add_calculated_column(df: pd.DataFrame, new_col: str, col1: str, col2: str) -> pd.DataFrame:
    """Create new column from calculation (e.g., LDL/HDL ratio)."""
    pass

def remove_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """Remove duplicate rows using .drop_duplicates(subset=...)."""
    pass

def create_age_bins(df: pd.DataFrame, column: str, bins: list, labels: list) -> pd.DataFrame:
    """Create categorical bins from continuous data using pd.cut()."""
    pass

def encode_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Create dummy variables using pd.get_dummies(), drop original column."""
    pass

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Detect outliers using IQR method, return boolean Series."""
    pass
```

### Grading (23 pts):
- `convert_to_datetime()` - 2 pts
- `convert_to_numeric()` - 2 pts
- `convert_to_category()` - 2 pts
- `replace_sentinels()` - 2 pts
- `apply_custom_function()` - 2 pts
- `map_values()` - 2 pts
- `clean_strings()` - 1 pt
- `add_calculated_column()` - 2 pts
- `remove_duplicates()` - 2 pts
- `create_age_bins()` - 2 pts
- `encode_categorical()` - 2 pts
- `detect_outliers_iqr()` - 2 pts

---

## Question 7: Groupby & Aggregation (10 points)

**File:** `aggregation.py`

Implement groupby and aggregation functions.

### Required Functions:

```python
"""Groupby and aggregation functions."""

import pandas as pd

def group_and_sum(df: pd.DataFrame, group_col: str, sum_col: str) -> pd.DataFrame:
    """Group by column and sum another column."""
    pass

def group_and_aggregate_multiple(df: pd.DataFrame, group_col: str, agg_dict: dict) -> pd.DataFrame:
    """
    Group and apply multiple aggregations using .agg().

    Example agg_dict: {'age': 'mean', 'bmi': ['mean', 'std']}
    """
    pass

def get_top_n(df: pd.DataFrame, column: str, n: int) -> pd.DataFrame:
    """Get top N rows using .nlargest()."""
    pass

def group_and_sort(df: pd.DataFrame, group_col: str, agg_col: str) -> pd.DataFrame:
    """Group, aggregate, and sort results."""
    pass
```

### Grading (10 pts):
- `group_and_sum()` - 3 pts
- `group_and_aggregate_multiple()` - 3 pts
- `get_top_n()` - 2 pts
- `group_and_sort()` - 2 pts

---

## Question 8: Pipeline Automation Script (5 points)

**File:** `run_pipeline.sh`

Create an executable shell script that runs the entire analysis pipeline.

### Requirements:

```bash
#!/bin/bash

# Run pipeline steps in order
# Check exit codes after each step
# Generate logs and final report

echo "Starting clinical trial data pipeline..." > reports/pipeline_log.txt

# Example structure:
python process_metadata.py
if [ $? -ne 0 ]; then
    echo "ERROR: Metadata processing failed" >> reports/pipeline_log.txt
    exit 1
fi

# TODO: Run other scripts
# TODO: Generate reports/quality_report.txt
# TODO: Save output/final_clean_data.csv
```

### Grading (5 pts):
- Script is executable - 1 pt
- Has shebang - 1 pt
- `reports/pipeline_log.txt` exists - 1 pt
- `reports/quality_report.txt` exists - 1 pt
- `output/final_clean_data.csv` exists - 1 pt

---

## Submission Checklist

**Scripts (8 files):**
- [ ] `setup_project.sh` (executable)
- [ ] `process_metadata.py` (executable)
- [ ] `exploration.py`
- [ ] `selection.py`
- [ ] `missing_data.py`
- [ ] `transform.py`
- [ ] `aggregation.py`
- [ ] `run_pipeline.sh` (executable)

**Directories:**
- [ ] `data/` (contains clinical_trial_raw.csv)
- [ ] `output/` (contains CSV outputs)
- [ ] `reports/` (contains text reports)

---

## Grading Summary

| Question | Topic | Points |
|----------|-------|--------|
| Q1 | Project Setup (Shell) | 10 |
| Q2 | Python Fundamentals | 25 |
| Q3 | Data Loading | 10 |
| Q4 | Selection & Filtering | 10 |
| Q5 | Missing Data | 15 |
| Q6 | Transformation | 23 |
| Q7 | Aggregation | 10 |
| Q8 | Pipeline Automation | 5 |
| **TOTAL** | | **108** |

**Grading Scale:**
- 97-108: A (90%+)
- 86-96: B (80%+)
- 76-85: C (70%+)
- 65-75: D (60%+)
- <65: F

---

## Tips

See `TIPS.md` for:
- Code scaffolding and starter templates
- Common pandas patterns
- Debugging strategies

**Good luck!**
