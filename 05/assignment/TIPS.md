# Assignment 5 Tips & Scaffolding

Quick reference for common patterns and helpful code snippets.

---

## General Tips

**Directory structure:**
```
05/assignment/
├── setup_project.sh          # Q1: Your setup script
├── process_metadata.py        # Q2: Metadata processing
├── exploration.py             # Q3: Data loading
├── selection.py               # Q4: Selection & filtering
├── missing_data.py            # Q5: Missing data handling
├── transform.py               # Q6: Transformations
├── aggregation.py             # Q7: Groupby operations
├── run_pipeline.sh            # Q8: Pipeline automation
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

## Q3: Pandas Loading & Exploration

**Loading data:**
```python
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)
```

**Summary statistics:**
```python
def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()

# Save to CSV
summary = df.describe()
summary.to_csv('output/summary_stats.csv')
```

**Value counts:**
```python
def get_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].value_counts()

# Save to CSV
counts = df['site'].value_counts()
counts.to_csv('output/value_counts_site.csv')
```

---

## Q4: Selection & Filtering

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

## Q5: Missing Data

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

## Q6: Data Transformation

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

---

## Q7: Groupby & Aggregation

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

# Run step 1
echo "Step 1: Processing metadata..." >> reports/pipeline_log.txt
python process_metadata.py
if [ $? -ne 0 ]; then
    echo "ERROR: Metadata processing failed" >> reports/pipeline_log.txt
    exit 1
fi

# Run step 2
echo "Step 2: Loading data..." >> reports/pipeline_log.txt
python exploration.py
if [ $? -ne 0 ]; then
    echo "ERROR: Data loading failed" >> reports/pipeline_log.txt
    exit 1
fi

# Continue for other scripts...

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
from process_metadata import parse_config, validate_config

config = parse_config('config.txt')
print("Config loaded:", config)

validation = validate_config(config)
print("Validation results:", validation)
```

**Run individual scripts:**
```bash
python process_metadata.py
python exploration.py
python selection.py
# etc.
```

**Check outputs match expected format:**
```python
# Quick checks
import pandas as pd

# Should be a DataFrame with summary stats
summary = pd.read_csv('output/summary_stats.csv')
print(summary.shape)

# Should have site names and counts
counts = pd.read_csv('output/value_counts_site.csv')
print(counts.head())
```

---

## Good Luck!

Remember:
- Read error messages carefully
- Test functions individually before running full pipeline
- Check intermediate outputs to verify correctness
- Use `df.head()` and `df.info()` frequently while developing
