# Assignment 5: Midterm Exam - Clinical Trial Data Analysis

**Total: 125 points**

## Scenario

You're analyzing data from a multi-site cardiovascular health clinical trial. The trial enrolled 10,000 patients across 5 hospital sites over 2 years, testing two different treatment interventions against a control group.

**Your task:** Build a complete data processing pipeline to clean, validate, and analyze the clinical trial data.

**Dataset:** `data/clinical_trial_raw.csv` (10,000 patients, 18 variables)

**Configuration:** `config/config.txt` (trial parameters)

---

## File Organization

**You will work with these files:**

**Scripts (complete the code):**
1. `q1_setup_project.sh` - Shell script to create directories (Q1)
2. `q2_process_metadata.py` - Python fundamentals and config processing (Q2)
3. `q3_data_utils.py` - **Core data utilities library** - 8 reusable functions (Q3)
4. `q8_run_pipeline.sh` - Pipeline automation script (Q8)

**Notebooks (complete the analysis):**
4. `q4_exploration.ipynb` - Data exploration using Q3 utilities (Q4)
5. `q5_missing_data.ipynb` - Missing data analysis (Q5)
6. `q6_transformation.ipynb` - Data transformation and feature engineering (Q6)
7. `q7_aggregation.ipynb` - Grouped analysis and reporting (Q7)

**Key Design:**
- **Q3 is your utility library** - Write 8 reusable pandas functions here
- **Q4-Q7 import from Q3** - Use your utilities in notebooks for real analysis
- **Q8 runs the pipeline** - Executes Q2 script and Q4-Q7 notebooks (Q3 is imported, not run directly)
- This mirrors professional data science workflows: utilities → analysis → automation

**Output structure:**
```
output/           # Data artifacts (CSV, TXT files with results)
reports/          # Logs and metadata (pipeline execution info)
data/             # Input data (provided)
config/           # Configuration files (trial params, filter examples)
```

See the Submission Checklist at the end for specific output files required.

---

**Variables:**

**Demographics:**
- `patient_id`: Unique patient identifier (P00001, P00002, ...)
- `age`: Patient age in years (18-85)
- `sex`: Patient sex (M/F)
- `bmi`: Body Mass Index (kg/m²)
- `enrollment_date`: Date patient enrolled in trial (YYYY-MM-DD)

**Clinical Measurements:**
- `systolic_bp`: Systolic blood pressure (mmHg)
- `diastolic_bp`: Diastolic blood pressure (mmHg)
- `cholesterol_total`: Total cholesterol (mg/dL)
- `cholesterol_hdl`: HDL ("good") cholesterol (mg/dL)
- `cholesterol_ldl`: LDL ("bad") cholesterol (mg/dL)
- `glucose_fasting`: Fasting blood glucose (mg/dL)

**Trial Information:**
- `site`: Hospital site (Site A, Site B, Site C, Site D, Site E)
- `intervention_group`: Treatment group (control, Treatment A, Treatment B)
- `follow_up_months`: Months of follow-up (0-24)
- `adverse_events`: Number of adverse events (0+)

**Outcomes:**
- `outcome_cvd`: Cardiovascular disease outcome (Yes/No)
- `adherence_pct`: Medication adherence percentage (0-100)
- `dropout`: Whether patient dropped out (Yes/No)

---

## Question 1: Project Setup Script (10 points)

**File:** `q1_setup_project.sh`

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
1. Be executable (`chmod +x q1_setup_project.sh`)
2. Have shebang `#!/bin/bash`
3. Create `data/` directory
4. Create `output/` directory
5. Create `reports/` directory
6. Save directory listing to `reports/directory_structure.txt`

**Test it:**
```bash
./q1_setup_project.sh
ls -la
cat reports/directory_structure.txt
```

---

## Question 2: Python Data Processing (25 points)

**File:** `q2_process_metadata.py`

Create an executable Python script that processes the trial configuration file (`config/config.txt`).

The config file format is:
```
study_name=CardioHealth Trial 2023
primary_investigator=Dr. Sarah Chen
enrollment_start=2022-01-01
enrollment_end=2023-12-31
min_age=18
max_age=85
target_enrollment=10000
sites=5
intervention_groups=3
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
    - intervention_groups must be >= 1
    - enrollment_start must be before enrollment_end (if both present)

    Args:
        config: Configuration dictionary

    Returns:
        dict: Validation results {key: True/False}
    """
    # TODO: Implement with if/elif/else
    # TODO: Add error checking for date format and logical consistency
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

**Structure:**
- Executable with shebang
- Has `if __name__ == '__main__':`
- Functions importable

**Functions:**
- `parse_config()` works correctly
- `validate_config()` logic correct
- `process_files()` filters correctly
- `calculate_statistics()` returns correct stats

**Outputs:**
- `output/config_summary.txt`
- `output/validation_report.txt`
- `output/file_manifest.txt`
- `output/statistics.txt`

---

## Question 3: Data Utilities Library (20 points)

**File:** `q3_data_utils.py`

Create a reusable library of pandas functions. These will be imported and used in Q4-Q7 notebooks.

### Required Functions:

```python
"""Core data utilities for clinical trial analysis."""

import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    pass

def clean_data(df: pd.DataFrame, remove_duplicates: bool = True, sentinel_value=-999) -> pd.DataFrame:
    """
    Clean data by removing duplicates and replacing sentinel values.

    Args:
        df: DataFrame to clean
        remove_duplicates: Whether to remove duplicate rows
        sentinel_value: Value to replace with NaN (default: -999)

    Returns:
        Cleaned DataFrame
    """
    pass

def detect_missing(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column."""
    pass

def fill_missing(df: pd.DataFrame, column: str, strategy: str = 'mean') -> pd.DataFrame:
    """
    Fill missing values using specified strategy.

    Args:
        df: DataFrame
        column: Column name to fill
        strategy: 'mean', 'median', or 'ffill'

    Returns:
        DataFrame with filled values
    """
    pass

def filter_data(df: pd.DataFrame, filters: list) -> pd.DataFrame:
    """
    Apply a list of filters to DataFrame in sequence.

    Args:
        df: Input DataFrame
        filters: List of filter dictionaries, each with keys:
                'column', 'condition', 'value'
                Conditions: 'equals', 'greater_than', 'less_than', 'in_range', 'in_list'

    Returns:
        pd.DataFrame: Filtered data

    Examples:
        >>> # Single filter
        >>> filters = [{'column': 'site', 'condition': 'equals', 'value': 'Site A'}]
        >>> df_filtered = filter_data(df, filters)
        >>>
        >>> # Multiple filters applied in order
        >>> filters = [
        ...     {'column': 'age', 'condition': 'greater_than', 'value': 18},
        ...     {'column': 'age', 'condition': 'less_than', 'value': 65},
        ...     {'column': 'site', 'condition': 'in_list', 'value': ['Site A', 'Site B']}
        ... ]
        >>> df_filtered = filter_data(df, filters)
        >>>
        >>> # Range filter example
        >>> filters = [{'column': 'age', 'condition': 'in_range', 'value': [18, 65]}]
        >>> df_filtered = filter_data(df, filters)
    """
    pass

def transform_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """
    Convert column types based on mapping.

    Args:
        df: DataFrame
        type_map: Dict mapping column names to types ('datetime', 'numeric', 'category')

    Returns:
        DataFrame with converted types
    """
    pass

def create_bins(df: pd.DataFrame, column: str, bins: list, labels: list, new_column: str = None) -> pd.DataFrame:
    """Create categorical bins from continuous column using pd.cut()."""
    pass

def summarize_by_group(df: pd.DataFrame, group_col: str, agg_dict: dict = None) -> pd.DataFrame:
    """
    Group and aggregate using multiple functions.

    Args:
        df: DataFrame
        group_col: Column to group by
        agg_dict: Dict of {column: function} or {column: [function1, function2]}

    Returns:
        Grouped and aggregated DataFrame
    """
    pass
```

### Grading:
- `load_data()`
- `clean_data()`
- `detect_missing()`
- `fill_missing()`
- `filter_data()`
- `transform_types()`
- `create_bins()`
- `summarize_by_group()`

---

## Question 4: Data Exploration (15 points)

**File:** `q4_exploration.ipynb`

Complete the notebook to explore the clinical trial data using your Q3 utilities.

### Tasks:

1. **Load and inspect data**
   - Import your `q3_data_utils` module
   - Load `data/clinical_trial_raw.csv`
   - Display basic info (shape, dtypes, first few rows)

2. **Generate site distribution**
   - Calculate value counts for the 'site' column
   - Save to `output/q4_site_counts.csv`

3. **Explore numeric variables**
   - Display summary statistics for numeric columns
   - Identify columns with outliers

4. **Categorical analysis**
   - Show distribution of intervention groups
   - Display sex distribution

### Required Output:
- `output/q4_site_counts.csv`

### Grading:
- Notebook executes without errors
- Site counts CSV exists and is correct
- Analysis cells present and meaningful

---

## Question 5: Missing Data Analysis (15 points)

**File:** `q5_missing_data.ipynb`

Complete the notebook to analyze and handle missing data.

### Tasks:

1. **Detect missing data**
   - Use your utility function to find missing values
   - Display counts and percentages

2. **Apply filling strategies**
   - Fill numeric columns with appropriate methods (mean/median)
   - Use forward fill for time-series columns
   - Document your choices

3. **Handle critical missing values**
   - Drop rows with missing values in critical columns (patient_id, outcome variables)

4. **Save cleaned data**
   - Save to `output/q5_cleaned_data.csv`
   - Generate missing data report to `output/q5_missing_report.txt`

### Required Outputs:
- `output/q5_cleaned_data.csv`
- `output/q5_missing_report.txt`

### Grading:
- Notebook executes without errors
- Cleaned data has fewer missing values
- Output files exist with correct content

---

## Question 6: Data Transformation (20 points)

**File:** `q6_transformation.ipynb`

Complete the notebook to transform and engineer features.

### Tasks:

1. **Type conversions**
   - Convert enrollment_date to datetime
   - Convert categorical columns to category dtype
   - Convert any string numbers to numeric

2. **Feature engineering**
   - Create cholesterol ratio (LDL/HDL)
   - Create age groups using bins: [0, 40, 55, 70, 100] → ['<40', '40-54', '55-69', '70+']
   - Create BMI categories: [0, 18.5, 25, 30, 100] → ['Underweight', 'Normal', 'Overweight', 'Obese']

3. **Clean and encode**
   - Remove any remaining duplicates
   - Create dummy variables for intervention_group

4. **Save transformed data**
   - Save to `output/q6_transformed_data.csv`

### Required Output:
- `output/q6_transformed_data.csv`

### Grading:
- Notebook executes without errors
- New feature columns exist
- Transformed data saved correctly

---

## Question 7: Aggregation & Analysis (15 points)

**File:** `q7_aggregation.ipynb`

Complete the notebook to perform grouped analysis.

### Tasks:

1. **Site-level summary**
   - Group by site
   - Calculate mean age, BMI, and patient count per site
   - Save to `output/q7_site_summary.csv`

2. **Intervention comparison**
   - Group by intervention_group
   - Compare outcome rates, adverse events, adherence
   - Save to `output/q7_intervention_comparison.csv`

3. **Advanced analysis**
   - Find top 10 patients by cholesterol_total
   - Calculate statistics by age_group (if created in Q6)

4. **Generate report**
   - Save key findings to `output/q7_analysis_report.txt`

### Required Outputs:
- `output/q7_site_summary.csv`
- `output/q7_intervention_comparison.csv`
- `output/q7_analysis_report.txt`

### Grading:
- Notebook executes without errors
- Site summary exists with correct structure
- Intervention comparison exists
- Analysis report has meaningful content

---

## Question 8: Pipeline Automation Script (5 points)

**File:** `q8_run_pipeline.sh`

Create an executable shell script that runs the entire analysis pipeline.

### Requirements:

```bash
#!/bin/bash

# Run pipeline steps in order
# Check exit codes after each step
# Generate logs and final report

echo "Starting clinical trial data pipeline..." > reports/pipeline_log.txt

# Example structure:
python q2_process_metadata.py
if [ $? -ne 0 ]; then
    echo "ERROR: Metadata processing failed" >> reports/pipeline_log.txt
    exit 1
fi

# TODO: Run other scripts
# TODO: Generate reports/quality_report.txt
# TODO: Save output/final_clean_data.csv
```

### Grading:
- Script is executable
- Has shebang
- `reports/pipeline_log.txt` exists
- `reports/quality_report.txt` exists
- `output/final_clean_data.csv` exists

---

## Submission Checklist

**Scripts (4 files):**
- [ ] `q1_setup_project.sh` (executable)
- [ ] `q2_process_metadata.py` (executable)
- [ ] `q3_data_utils.py` (importable module)
- [ ] `q8_run_pipeline.sh` (executable)

**Notebooks (4 files):**
- [ ] `q4_exploration.ipynb`
- [ ] `q5_missing_data.ipynb`
- [ ] `q6_transformation.ipynb`
- [ ] `q7_aggregation.ipynb`

**Directories and Outputs:**
- [ ] `data/` (contains clinical_trial_raw.csv)
- [ ] `config/` (contains config.txt)
- [ ] `output/` (data artifacts created by your scripts/notebooks)
  - Q2: `config_summary.txt`, `validation_report.txt`, `file_manifest.txt`, `statistics.txt`
  - Q4: `q4_site_counts.csv`
  - Q5: `q5_cleaned_data.csv`, `q5_missing_report.txt`
  - Q6: `q6_transformed_data.csv`
  - Q7: `q7_site_summary.csv`, `q7_intervention_comparison.csv`, `q7_analysis_report.txt`
  - Q8: `final_clean_data.csv`
- [ ] `reports/` (pipeline logs and process metadata)
  - Q1: `directory_structure.txt`
  - Q8: `pipeline_log.txt`, `quality_report.txt`

---

## Grading Summary

| Question | Topic | Points |
|----------|-------|--------|
| Q1 | Project Setup (Shell) | 10 |
| Q2 | Python Fundamentals | 25 |
| Q3 | Data Utilities Library | 20 |
| Q4 | Data Exploration (Notebook) | 15 |
| Q5 | Missing Data Analysis (Notebook) | 15 |
| Q6 | Data Transformation (Notebook) | 20 |
| Q7 | Aggregation & Analysis (Notebook) | 15 |
| Q8 | Pipeline Automation | 5 |
| **TOTAL** | | **125** |

**Grading Scale:**
- 113-125: A (90%+)
- 100-112: B (80%+)
- 88-99: C (70%+)
- 75-87: D (60%+)
- <75: F

---

## Tips

See `TIPS.md` for:

- Common pandas patterns
- Debugging strategies

**Good luck!**
