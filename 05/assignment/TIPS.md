# Assignment 5 Tips & Scaffolding

Quick reference for common patterns and helpful code snippets.

---

## General Tips

**Directory structure:**
```
05/assignment/
├── q1_setup_project.sh        # Q1: Your setup script
├── q2_process_metadata.py     # Q2: Metadata processing
├── q3_data_utils.py           # Q3: Reusable utility library
├── q4_exploration.ipynb       # Q4: Data exploration notebook
├── q5_missing_data.ipynb      # Q5: Missing data analysis notebook
├── q6_transformation.ipynb    # Q6: Data transformation notebook
├── q7_aggregation.ipynb       # Q7: Aggregation analysis notebook
├── q8_run_pipeline.sh         # Q8: Pipeline automation
├── config.txt                 # Input: Trial configuration
├── data/
│   └── clinical_trial_raw.csv # Input: Raw data
├── output/                    # Your CSV/text outputs go here
└── reports/                   # Your report files go here
```

**Working with the data:**
```python
import pandas as pd

# Load the data
df = pd.read_csv('data/clinical_trial_raw.csv')

# Quick exploration
print(df.shape)           # (10000, 18)
print(df.columns.tolist())
print(df.head())
print(df.dtypes)
```

---

## Q1: Shell Script Patterns

**Basic script structure:**
```bash
#!/bin/bash

# Create directories
mkdir -p directory_name

# Check if command succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create directory"
    exit 1
fi

# Save directory tree to file
ls -R > reports/directory_structure.txt
```

**Make script executable:**
```bash
chmod +x setup_project.sh
./setup_project.sh
```

---

## Q2: Python Fundamentals

**Reading key=value config files:**
```python
def parse_config(filepath: str) -> dict:
    config = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=')
                config[key] = value
    return config
```

**Validation with if/elif/else:**
```python
def validate_value(value, min_val, max_val):
    if value < min_val:
        return False
    elif value > max_val:
        return False
    else:
        return True
```

**Filtering lists:**
```python
# List comprehension
csv_files = [f for f in file_list if f.endswith('.csv')]

# Filter function
def is_csv(filename):
    return filename.endswith('.csv')

csv_files = list(filter(is_csv, file_list))
```

**Basic statistics:**
```python
import statistics

data = [1, 2, 3, 4, 5]
mean_val = statistics.mean(data)
median_val = statistics.median(data)
sum_val = sum(data)
count = len(data)
```

**Saving text outputs:**
```python
# Save simple text file
with open('output/summary.txt', 'w') as f:
    f.write(f"Study: {config['study_name']}\n")
    f.write(f"PI: {config['primary_investigator']}\n")
```

---

## Q3: Building the Data Utilities Library

**This is your reusable function library - Q4-Q7 notebooks will import from here.**

**Basic loading:**
```python
import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    return pd.read_csv(filepath)
```

**Cleaning with options:**
```python
def clean_data(df: pd.DataFrame, remove_duplicates: bool = True, sentinel_value=-999) -> pd.DataFrame:
    """Clean data by removing duplicates and replacing sentinel values."""
    df_clean = df.copy()

    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()

    # Replace sentinel values with NaN
    df_clean = df_clean.replace(sentinel_value, np.nan)

    return df_clean
```

**Flexible missing data handling:**
```python
def fill_missing(df: pd.DataFrame, column: str, strategy: str = 'mean') -> pd.DataFrame:
    """Fill missing values using specified strategy."""
    df_filled = df.copy()

    if strategy == 'mean':
        df_filled[column] = df_filled[column].fillna(df_filled[column].mean())
    elif strategy == 'median':
        df_filled[column] = df_filled[column].fillna(df_filled[column].median())
    elif strategy == 'ffill':
        df_filled[column] = df_filled[column].fillna(method='ffill')

    return df_filled
```

**Type conversion with mapping:**
```python
def transform_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """Convert column types based on mapping."""
    df_transformed = df.copy()

    for col, dtype in type_map.items():
        if dtype == 'datetime':
            df_transformed[col] = pd.to_datetime(df_transformed[col])
        elif dtype == 'numeric':
            df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce')
        elif dtype == 'category':
            df_transformed[col] = df_transformed[col].astype('category')

    return df_transformed
```

---

## Q4-Q7: Working in Notebooks

**Import your utilities:**
```python
# At the top of your notebook
import pandas as pd
import numpy as np
from q3_data_utils import load_data, clean_data, fill_missing, filter_data

# Load data using your utility
df = load_data('data/clinical_trial_raw.csv')

# Example filter_data usage
# Single filter
filters = [{'column': 'site', 'condition': 'equals', 'value': 'Site A'}]
site_a_patients = filter_data(df, filters)

# Multiple filters
filters = [
    {'column': 'age', 'condition': 'greater_than', 'value': 18},
    {'column': 'age', 'condition': 'less_than', 'value': 65},
    {'column': 'site', 'condition': 'in_list', 'value': ['Site A', 'Site B']}
]
filtered_patients = filter_data(df, filters)

# Range filter
filters = [{'column': 'age', 'condition': 'in_range', 'value': [18, 65]}]
age_range_patients = filter_data(df, filters)
```

**Q4: Selection & Filtering Patterns

**Selecting by data type:**
```python
numeric_df = df.select_dtypes(include=['number'])
text_df = df.select_dtypes(include=['object'])
```

**Label-based selection (.loc):**
```python
# Rows by index labels, columns by name
subset = df.loc[0:10, ['patient_id', 'age', 'bmi']]

# Boolean indexing with .loc
adults = df.loc[df['age'] >= 18]
```

**Position-based selection (.iloc):**
```python
# First 100 rows, first 5 columns
subset = df.iloc[0:100, 0:5]

# Specific column indices
subset = df.iloc[:, [0, 2, 4, 6]]
```

**Boolean filtering:**
```python
# Single condition
high_bp = df[df['systolic_bp'] > 140]

# Multiple conditions (use & and |, wrap each condition in parentheses)
at_risk = df[(df['systolic_bp'] > 140) & (df['cholesterol_total'] > 200)]

# .isin() for categorical filtering
sites_subset = df[df['site'].isin(['Site A', 'Site B'])]
```

---

## Q5: Missing Data Analysis (Notebook)

**Detecting missing data:**
```python
# Count missing per column
missing_counts = df.isnull().sum()

# Percentage missing
missing_pct = df.isnull().sum() / len(df) * 100
```

**Filling strategies:**
```python
import numpy as np

# Fill with mean
df['age_filled'] = df['age'].fillna(df['age'].mean())

# Fill with median
df['bmi_filled'] = df['bmi'].fillna(df['bmi'].median())

# Forward fill
df['date_filled'] = df['enrollment_date'].fillna(method='ffill')

# Backward fill
df['date_filled'] = df['enrollment_date'].fillna(method='bfill')
```

**Dropping missing data:**
```python
# Drop rows with any missing values
df_complete = df.dropna()

# Drop rows with missing values in specific columns
df_subset = df.dropna(subset=['age', 'bmi'])

# Drop rows with ALL values missing in subset
df_subset = df.dropna(subset=['blood_pressure', 'cholesterol'], how='all')
```

---

## Q6: Data Transformation (Notebook)

**Removing duplicates:**
```python
# Remove duplicate rows
df_unique = df.drop_duplicates()

# Remove duplicates based on specific columns
df_unique = df.drop_duplicates(subset=['patient_id'])

# Keep last occurrence instead of first
df_unique = df.drop_duplicates(subset=['patient_id'], keep='last')
```

**Type conversions:**
```python
# Convert to datetime
df['enrollment_date'] = pd.to_datetime(df['enrollment_date'])

# Convert to numeric (coerce errors to NaN)
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Convert to category
df['site'] = df['site'].astype('category')
```

**Replacing values:**
```python
# Replace sentinel values with NaN
df['income'] = df['income'].replace(-999, np.nan)

# Replace using dictionary
df['status'] = df['status'].replace({'Y': 'Yes', 'N': 'No'})
```

**Applying functions:**
```python
# .apply() with custom function
def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['bmi_category'] = df['bmi'].apply(bmi_category)

# .apply() with lambda
df['age_squared'] = df['age'].apply(lambda x: x ** 2)
```

**Mapping values:**
```python
# .map() with dictionary
education_map = {'HS': 'High School', 'BA': 'Bachelors', 'MA': 'Masters'}
df['education_full'] = df['education'].map(education_map)
```

**String operations:**
```python
# Clean strings
df['site_clean'] = df['site'].str.strip().str.lower()

# String methods chaining
df['name'] = df['name'].str.strip().str.title()
```

**Calculated columns:**
```python
# Create new column from calculation
df['cholesterol_ratio'] = df['cholesterol_ldl'] / df['cholesterol_hdl']
df['bmi_calc'] = df['weight_kg'] / (df['height_m'] ** 2)
```

**Creating categorical bins:**
```python
# Equal-width bins with pd.cut()
df['age_group'] = pd.cut(df['age'],
                         bins=[0, 30, 50, 100],
                         labels=['Young', 'Middle', 'Senior'])

# Or specify bin edges and labels
bins = [0, 18, 65, 120]
labels = ['Child', 'Adult', 'Senior']
df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels)

# Equal-frequency bins with pd.qcut()
df['bmi_quartile'] = pd.qcut(df['bmi'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

**Dummy variable encoding:**
```python
# Create dummy variables from categorical column
dummies = pd.get_dummies(df['intervention_group'], prefix='treatment')

# Join with original DataFrame and drop original column
df = pd.concat([df, dummies], axis=1)
df = df.drop('intervention_group', axis=1)

# Or in one step
df = pd.get_dummies(df, columns=['intervention_group'], drop_first=False)
```

**Outlier detection with IQR:**
```python
# Calculate IQR (Interquartile Range)
Q1 = df['cholesterol_total'].quantile(0.25)
Q3 = df['cholesterol_total'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers (boolean Series)
outliers = (df['cholesterol_total'] < lower_bound) | (df['cholesterol_total'] > upper_bound)

# Filter to only outliers
outlier_rows = df[outliers]

# Remove outliers
df_clean = df[~outliers]
```

---

## Q7: Groupby & Aggregation (Notebook)

**Basic groupby:**
```python
# Group and sum
site_totals = df.groupby('site')['adverse_events'].sum()

# Group and mean
site_avg_age = df.groupby('site')['age'].mean()
```

**Multiple aggregations:**
```python
# Multiple functions on one column
stats = df.groupby('site')['age'].agg(['mean', 'std', 'count'])

# Different functions on different columns
agg_dict = {
    'age': 'mean',
    'bmi': ['mean', 'std'],
    'patient_id': 'count'
}
summary = df.groupby('site').agg(agg_dict)
```

**Top N values:**
```python
# Get top 10 by column
top_10 = df.nlargest(10, 'cholesterol_total')

# Get bottom 5
bottom_5 = df.nsmallest(5, 'bmi')
```

**Group, aggregate, and sort:**
```python
# Group by site, count patients, sort by count
site_counts = df.groupby('site')['patient_id'].count().sort_values(ascending=False)

# Reset index to make groupby column a regular column
site_summary = df.groupby('site')['age'].mean().reset_index()
```

---

## Q8: Shell Pipeline Script

**Script structure with error checking:**
```bash
#!/bin/bash

echo "Starting pipeline..." > reports/pipeline_log.txt
date >> reports/pipeline_log.txt

# Run step 1: Metadata processing
echo "Step 1: Processing metadata..." >> reports/pipeline_log.txt
python q2_process_metadata.py
if [ $? -ne 0 ]; then
    echo "ERROR: Metadata processing failed" >> reports/pipeline_log.txt
    exit 1
fi

# Run step 2: Execute notebooks
echo "Step 2: Running analysis notebooks..." >> reports/pipeline_log.txt
jupyter nbconvert --to notebook --execute q4_exploration.ipynb
jupyter nbconvert --to notebook --execute q5_missing_data.ipynb
jupyter nbconvert --to notebook --execute q6_transformation.ipynb
jupyter nbconvert --to notebook --execute q7_aggregation.ipynb

# Continue for other steps...

echo "Pipeline complete!" >> reports/pipeline_log.txt
date >> reports/pipeline_log.txt
```

**Generating reports:**
```bash
# Count files in output/
echo "Output files generated: $(ls output/ | wc -l)" >> reports/quality_report.txt

# Check for missing values in final data
echo "Data quality checks:" >> reports/quality_report.txt
python -c "import pandas as pd; df = pd.read_csv('output/final_clean_data.csv'); print(f'Missing values: {df.isnull().sum().sum()}')" >> reports/quality_report.txt
```

---

## Common Pandas Patterns

**Chaining operations:**
```python
# Read, filter, select, save
(pd.read_csv('data/clinical_trial_raw.csv')
   .loc[lambda df: df['age'] > 50]
   [['patient_id', 'age', 'bmi']]
   .to_csv('output/seniors.csv', index=False))
```

**Handling dates:**
```python
df['enrollment_date'] = pd.to_datetime(df['enrollment_date'])
df['year'] = df['enrollment_date'].dt.year
df['month'] = df['enrollment_date'].dt.month
df['day_of_week'] = df['enrollment_date'].dt.day_name()
```

**Creating categories from continuous data:**
```python
# Equal-width bins
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])

# Equal-frequency bins (quantiles)
df['bmi_quartile'] = pd.qcut(df['bmi'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

**Saving outputs:**
```python
# DataFrame to CSV
df.to_csv('output/result.csv', index=False)

# Series to CSV
df['site'].value_counts().to_csv('output/site_counts.csv')

# Text file
with open('output/summary.txt', 'w') as f:
    f.write(f"Total patients: {len(df)}\n")
    f.write(f"Average age: {df['age'].mean():.1f}\n")
```

---

## Debugging Tips

**Check data types:**
```python
print(df.dtypes)
print(df.info())
```

**Inspect specific values:**
```python
print(df['column_name'].unique())
print(df['column_name'].value_counts())
print(df['column_name'].describe())
```

**Find problematic rows:**
```python
# Rows with missing age
print(df[df['age'].isnull()])

# Rows with negative values
print(df[df['systolic_bp'] < 0])

# Duplicates
print(df[df.duplicated()])
```

**Test functions interactively:**
```python
# In Python interpreter or Jupyter
from process_metadata import parse_config

config = parse_config('config.txt')
print(config)
```

**Common errors:**
- `KeyError`: Column name doesn't exist (check spelling, spaces)
- `ValueError`: Type conversion failed (check for invalid values)
- `FileNotFoundError`: Wrong file path (check relative paths)
- `SettingWithCopyWarning`: Use `.copy()` when creating subset DataFrames

---

## Testing Your Work

**Check file existence:**
```bash
ls output/
ls reports/
```

**Verify script executability:**
```bash
ls -la *.sh
# Should show -rwxr-xr-x (x = executable)
```

**Test functions by importing:**
```python
# Create test_my_work.py
from q2_process_metadata import parse_config, validate_config
from q3_data_utils import load_data, clean_data

config = parse_config('config.txt')
print("Config loaded:", config)

validation = validate_config(config)
print("Validation results:", validation)

# Test your utilities
df = load_data('data/clinical_trial_raw.csv')
print("Data shape:", df.shape)
```

**Run individual scripts:**
```bash
python q2_process_metadata.py

# Test notebooks
jupyter nbconvert --to notebook --execute q4_exploration.ipynb
# etc.
```

**Check outputs match expected format:**
```python
# Quick checks
import pandas as pd

# Q4 outputs
site_counts = pd.read_csv('output/q4_site_counts.csv')
print(site_counts)

# Q5 outputs
cleaned = pd.read_csv('output/q5_cleaned_data.csv')
print(f"Cleaned data shape: {cleaned.shape}")

# Q6 outputs
transformed = pd.read_csv('output/q6_transformed_data.csv')
print(f"New columns: {set(transformed.columns) - set(cleaned.columns)}")

# Q7 outputs
site_summary = pd.read_csv('output/q7_site_summary.csv')
print(site_summary)
```

---

## Good Luck!

Remember:
- Read error messages carefully
- Test functions individually before running full pipeline
- Check intermediate outputs to verify correctness
- Use `df.head()` and `df.info()` frequently while developing
