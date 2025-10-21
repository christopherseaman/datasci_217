# Q3 Data Utilities - Quick Reference Card

## Import Statement
```python
from q3_data_utils import (
    load_data,
    clean_data,
    detect_missing,
    fill_missing,
    filter_data,
    transform_types,
    create_bins,
    summarize_by_group
)
```

---

## 1. load_data()
**Load CSV file into DataFrame**

```python
df = load_data('data/clinical_trial_raw.csv')
```

---

## 2. clean_data()
**Remove duplicates and replace sentinel values**

```python
# Replace -999 with NaN, remove duplicates
df = clean_data(df, remove_duplicates=True, sentinel_value=-999)

# Chain multiple sentinel values
df = clean_data(df, remove_duplicates=False, sentinel_value=-1)
```

---

## 3. detect_missing()
**Count missing values per column**

```python
missing = detect_missing(df)
print(missing[missing > 0])  # Only show columns with missing values
```

---

## 4. fill_missing()
**Impute missing values**

```python
# Fill with median
df = fill_missing(df, 'age', strategy='median')

# Fill with mean
df = fill_missing(df, 'bmi', strategy='mean')

# Forward fill (carry last value forward)
df = fill_missing(df, 'temperature', strategy='ffill')
```

---

## 5. filter_data()
**Apply sequential filters**

```python
# Single filter
filters = [
    {'column': 'age', 'condition': 'greater_than', 'value': 18}
]
df = filter_data(df, filters)

# Multiple filters
filters = [
    {'column': 'age', 'condition': 'in_range', 'value': [18, 65]},
    {'column': 'site', 'condition': 'in_list', 'value': ['Site A', 'Site B']},
    {'column': 'bmi', 'condition': 'less_than', 'value': 35}
]
df = filter_data(df, filters)

# Available conditions:
# - 'equals': exact match
# - 'greater_than': >
# - 'less_than': <
# - 'in_range': between [min, max] inclusive
# - 'in_list': match any value in list
```

---

## 6. transform_types()
**Convert column data types**

```python
type_map = {
    'enrollment_date': 'datetime',  # Convert to datetime
    'age': 'numeric',                # Convert to int/float
    'site': 'category',              # Convert to category (memory efficient)
    'patient_id': 'string'           # Convert to string
}
df = transform_types(df, type_map)

# Available types:
# - 'datetime': pd.to_datetime()
# - 'numeric': pd.to_numeric()
# - 'category': .astype('category')
# - 'string': .astype('string')
```

---

## 7. create_bins()
**Bin continuous data into categories**

```python
# Basic usage
df = create_bins(
    df,
    column='age',
    bins=[0, 18, 35, 50, 65, 100],
    labels=['<18', '18-34', '35-49', '50-64', '65+']
)
# Creates new column: 'age_binned'

# Custom column name
df = create_bins(
    df,
    column='bmi',
    bins=[0, 18.5, 25, 30, 100],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
    new_column='bmi_category'
)
```

---

## 8. summarize_by_group()
**Group and aggregate data**

```python
# Automatic summary (uses .describe())
summary = summarize_by_group(df, 'site')

# Single aggregation per column
summary = summarize_by_group(df, 'site', {'age': 'mean', 'bmi': 'median'})

# Multiple aggregations per column
summary = summarize_by_group(
    df,
    'site',
    {
        'age': ['mean', 'std', 'count'],
        'bmi': ['mean', 'median'],
        'systolic_bp': 'max'
    }
)

# Available aggregations:
# 'mean', 'median', 'std', 'min', 'max', 'count', 'sum', 'first', 'last'
```

---

## Complete Pipeline Example

```python
from q3_data_utils import *

# 1. Load
df = load_data('data/clinical_trial_raw.csv')

# 2. Clean
df = clean_data(df, sentinel_value=-999)
df = clean_data(df, remove_duplicates=False, sentinel_value=-1)

# 3. Check missing
missing = detect_missing(df)
print("Missing values:\n", missing[missing > 0])

# 4. Fill missing
df = fill_missing(df, 'age', strategy='median')
df = fill_missing(df, 'bmi', strategy='mean')

# 5. Transform types
type_map = {
    'enrollment_date': 'datetime',
    'age': 'numeric',
    'site': 'category',
    'intervention_group': 'category'
}
df = transform_types(df, type_map)

# 6. Filter
filters = [
    {'column': 'age', 'condition': 'in_range', 'value': [18, 65]},
    {'column': 'site', 'condition': 'in_list', 'value': ['Site A', 'Site B', 'Site C']}
]
df = filter_data(df, filters)

# 7. Bin continuous data
df = create_bins(
    df,
    column='age',
    bins=[0, 30, 40, 50, 60, 100],
    labels=['<30', '30-39', '40-49', '50-59', '60+']
)

# 8. Summarize
summary = summarize_by_group(
    df,
    'site',
    {
        'age': ['mean', 'std', 'count'],
        'bmi': 'mean',
        'systolic_bp': ['mean', 'std']
    }
)

print("\nSummary by site:")
print(summary)
```

---

## Common Patterns

### Pattern 1: Data Quality Check
```python
df = load_data('data/file.csv')
missing = detect_missing(df)
print(f"Columns with missing data: {(missing > 0).sum()}")
print(f"Total missing values: {missing.sum()}")
print(f"Missing percentage: {missing.sum() / df.size * 100:.2f}%")
```

### Pattern 2: Age-Based Analysis
```python
# Filter adults, bin by age group, summarize
filters = [{'column': 'age', 'condition': 'greater_than', 'value': 18}]
df = filter_data(df, filters)

df = create_bins(df, 'age', bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])

summary = summarize_by_group(df, 'age_binned', {'bmi': 'mean', 'systolic_bp': 'mean'})
```

### Pattern 3: Site Comparison
```python
# Clean, standardize, compare sites
df = clean_data(df, sentinel_value=-999)
df = transform_types(df, {'site': 'category', 'age': 'numeric'})

summary = summarize_by_group(
    df,
    'site',
    {
        'age': ['mean', 'count'],
        'bmi': ['mean', 'std'],
        'adherence_pct': 'mean'
    }
)
```

### Pattern 4: Temporal Analysis
```python
# Convert dates, filter by time period
df = transform_types(df, {'enrollment_date': 'datetime'})

# Filter by date (create date filters manually)
df_2022 = df[df['enrollment_date'].dt.year == 2022]
df_q1 = df[df['enrollment_date'].dt.quarter == 1]
```

---

## Tips & Best Practices

1. **Always use .copy()**: Functions already do this, but if you modify DataFrames manually, use `.copy()` to avoid changing original data

2. **Check missing data first**: Run `detect_missing()` before filling to understand data quality

3. **Fill strategically**:
   - Use `'median'` for skewed data
   - Use `'mean'` for normally distributed data
   - Use `'ffill'` for time-series data

4. **Filter in sequence**: Filters are applied in order, so put broad filters first:
   ```python
   # Good: broad to specific
   filters = [
       {'column': 'age', 'condition': 'greater_than', 'value': 18},
       {'column': 'site', 'condition': 'equals', 'value': 'Site A'}
   ]
   ```

5. **Use categories for repeated values**: Convert columns with limited unique values to category type to save memory

6. **Check bin distribution**: After binning, check value_counts() to ensure bins make sense:
   ```python
   df = create_bins(df, 'age', bins, labels)
   print(df['age_binned'].value_counts().sort_index())
   ```

---

## Testing

Run tests to verify everything works:

```bash
# Unit tests
python test_q3.py

# Comprehensive tests
python tests/test_q3_comprehensive.py

# Quick validation
python -c "from q3_data_utils import *; print('✓ All utilities imported successfully')"
```

---

## Documentation

For detailed documentation, see:
- `docs/Q3_UTILITIES_SUMMARY.md` - Complete function documentation
- `docs/Q3_COMPLETION_REPORT.md` - Implementation report

---

**Quick Import**: `from q3_data_utils import *`
