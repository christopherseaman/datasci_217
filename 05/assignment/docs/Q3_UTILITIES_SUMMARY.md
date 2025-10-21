# Q3 Data Utilities Library - Summary

## Overview

The `q3_data_utils.py` module provides 8 reusable utility functions for data loading, cleaning, transformation, and analysis. All functions follow DRY (Don't Repeat Yourself) principles and are designed for use in Q4-Q7 notebooks.

## Utility Functions

### 1. load_data(filepath: str) → pd.DataFrame
**Purpose**: Load CSV files into pandas DataFrame

**Key Features**:
- Simple wrapper around `pd.read_csv()`
- Provides consistent interface across all notebooks
- Handles standard CSV format with headers

**Example**:
```python
df = load_data('data/clinical_trial_raw.csv')
# Loaded: 10000 rows × 18 columns
```

---

### 2. clean_data(df, remove_duplicates=True, sentinel_value=-999) → pd.DataFrame
**Purpose**: Basic data cleaning operations

**Key Features**:
- Removes duplicate rows (optional)
- Replaces sentinel values (-999, -1, etc.) with NaN
- Returns a copy to preserve original data

**Example**:
```python
df_clean = clean_data(df, sentinel_value=-999)
df_clean = clean_data(df_clean, remove_duplicates=False, sentinel_value=-1)
```

**Methods Used**:
- `df.copy()` - Creates deep copy
- `df.drop_duplicates()` - Removes duplicate rows
- `df.replace()` - Replaces values
- `pd.NA` - Modern pandas missing value marker

---

### 3. detect_missing(df) → pd.Series
**Purpose**: Count missing values per column

**Key Features**:
- Returns Series with column names and missing counts
- Works with NaN, None, pd.NA
- Essential for data quality assessment

**Example**:
```python
missing = detect_missing(df)
# bmi: 438 missing (4.4%)
# systolic_bp: 414 missing (4.1%)
```

**Methods Used**:
- `df.isnull()` or `df.isna()` - Detects missing values
- `.sum()` - Aggregates by column

---

### 4. fill_missing(df, column, strategy='mean') → pd.DataFrame
**Purpose**: Impute missing values using various strategies

**Key Features**:
- Supports 'mean', 'median', 'ffill' strategies
- Works on single column at a time
- Preserves other columns unchanged

**Example**:
```python
df_filled = fill_missing(df, 'age', strategy='median')
# Missing before: 200, after: 0
```

**Methods Used**:
- `df[column].mean()` - Calculate mean
- `df[column].median()` - Calculate median
- `df[column].fillna()` - Fill missing values
- `df[column].ffill()` - Forward fill (carry last valid value forward)

---

### 5. filter_data(df, filters: list) → pd.DataFrame
**Purpose**: Apply sequential filters to DataFrame

**Key Features**:
- Supports multiple filter conditions:
  - `'equals'` - Exact match
  - `'greater_than'` - Numeric comparison
  - `'less_than'` - Numeric comparison
  - `'in_range'` - Between two values (inclusive)
  - `'in_list'` - Match any value in list
- Filters applied in order
- Flexible and composable

**Example**:
```python
filters = [
    {'column': 'age', 'condition': 'in_range', 'value': [18, 65]},
    {'column': 'site', 'condition': 'in_list', 'value': ['Site A', 'Site B']}
]
df_filtered = filter_data(df, filters)
# Rows after filtering: 1474
```

**Methods Used**:
- Boolean indexing: `df[df[column] == value]`
- Comparison operators: `>`, `<`, `>=`, `<=`
- `&` operator - Combine multiple conditions
- `df[column].isin()` - Check membership in list

---

### 6. transform_types(df, type_map: dict) → pd.DataFrame
**Purpose**: Convert column data types

**Key Features**:
- Supports 'datetime', 'numeric', 'category', 'string'
- Handles conversion errors gracefully with `errors='coerce'`
- Optimizes memory with categorical types

**Example**:
```python
type_map = {
    'enrollment_date': 'datetime',
    'age': 'numeric',
    'site': 'category',
    'intervention_group': 'category'
}
df_typed = transform_types(df, type_map)
```

**Methods Used**:
- `pd.to_datetime()` - Convert to datetime
- `pd.to_numeric()` - Convert to numeric (int/float)
- `.astype('category')` - Convert to categorical
- `.astype('string')` - Convert to string
- `errors='coerce'` parameter - Replace invalid values with NaN

---

### 7. create_bins(df, column, bins, labels, new_column=None) → pd.DataFrame
**Purpose**: Create categorical bins from continuous data

**Key Features**:
- Uses `pd.cut()` for binning
- Creates new column (preserves original)
- Flexible bin edges and labels

**Example**:
```python
df_binned = create_bins(
    df,
    column='age',
    bins=[0, 30, 40, 50, 60, 100],
    labels=['<30', '30-39', '40-49', '50-59', '60+']
)
# Age distribution:
#   - 40-49: 64 (4.3%)
#   - 50-59: 651 (44.2%)
#   - 60+: 759 (51.5%)
```

**Methods Used**:
- `pd.cut()` - Bin continuous data into discrete intervals
  - `bins` parameter - Define interval boundaries
  - `labels` parameter - Name each interval

---

### 8. summarize_by_group(df, group_col, agg_dict=None) → pd.DataFrame
**Purpose**: Group data and apply aggregations

**Key Features**:
- Groups by single column
- Supports custom aggregations or automatic `.describe()`
- Multiple aggregations per column

**Example**:
```python
# Custom aggregations
summary = summarize_by_group(
    df,
    'site',
    {'age': ['mean', 'count'], 'bmi': 'mean'}
)

# Automatic summary statistics
summary = summarize_by_group(df, 'site')  # Uses .describe()
```

**Methods Used**:
- `df.groupby()` - Group rows by column values
- `.agg()` - Apply multiple aggregation functions
- `.describe()` - Generate summary statistics (count, mean, std, min, quartiles, max)

---

## Methods That May Not Be Covered in Lectures

Based on typical introductory Python/pandas courses, these methods might be new:

### 1. **pd.NA vs NaN**
- `pd.NA` is the modern pandas missing value marker
- Works better with mixed-type columns than `np.nan`
- Introduced in pandas 1.0+

### 2. **errors='coerce' parameter**
Used in `pd.to_datetime()` and `pd.to_numeric()`:
- Converts invalid values to NaN instead of raising errors
- Essential for real-world messy data
- Alternative to try/except blocks

### 3. **pd.cut() for binning**
- Divides continuous data into discrete bins
- Different from manual if/elif/else logic
- More efficient and flexible for many intervals

### 4. **Forward fill (ffill)**
- Propagates last valid observation forward
- Useful for time-series data
- Alternative to mean/median imputation

### 5. **Category dtype**
- Memory-efficient for columns with repeated values
- Enables ordered comparisons
- Optimizes groupby operations

### 6. **Multiple aggregations in groupby**
```python
.agg({'age': ['mean', 'std'], 'bmi': 'mean'})
```
- Apply different functions to different columns
- Return multi-level column index
- More powerful than single aggregation

---

## Testing Results

### Unit Tests (test_q3.py)
✓ All 8 functions tested individually
✓ All tests passed
✓ Edge cases verified

### Comprehensive Tests (tests/test_q3_comprehensive.py)
✓ Tested with real clinical trial data (10,000 rows)
✓ All functions work together correctly
✓ Data pipeline validated end-to-end

**Sample Output**:
- Loaded: 10,000 rows × 18 columns
- Detected: 8 columns with missing values
- Filtered: 1,474 rows (age 18-65)
- Binned age into 5 categories
- Summarized by site (5 groups)

---

## Design Principles Applied

1. **DRY (Don't Repeat Yourself)**
   - Each function has single, well-defined purpose
   - Reusable across Q4-Q7 notebooks
   - No code duplication

2. **Composability**
   - Functions work together in pipelines
   - Each returns DataFrame (except detect_missing)
   - Can chain operations

3. **Immutability**
   - All functions use `.copy()` to preserve original data
   - Functional programming style
   - Safer for exploratory analysis

4. **Defensive Programming**
   - Input validation with clear error messages
   - Graceful handling of missing columns
   - Type conversion with error handling

5. **Documentation**
   - Docstrings with Args, Returns, Examples
   - Consistent format across all functions
   - Helps with IDE autocomplete

---

## Usage in Notebooks (Q4-Q7)

Import utilities at the top of each notebook:
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

Then use throughout the analysis:
```python
# Load and clean
df = load_data('data/clinical_trial_raw.csv')
df = clean_data(df, sentinel_value=-999)

# Analyze and transform
missing = detect_missing(df)
df = fill_missing(df, 'bmi', strategy='median')
df = transform_types(df, {'enrollment_date': 'datetime'})

# Filter and summarize
filters = [{'column': 'age', 'condition': 'in_range', 'value': [18, 65]}]
df_filtered = filter_data(df, filters)
summary = summarize_by_group(df_filtered, 'site', {'age': 'mean'})
```

---

## Conclusion

The Q3 utilities library provides a complete toolkit for data wrangling in Python/pandas:
- ✓ Clean, reusable code following DRY principles
- ✓ Well-tested with both unit and integration tests
- ✓ Comprehensive documentation
- ✓ Ready for use in Q4-Q7 analysis notebooks
- ✓ No numpy dependency (pandas only)

All functions demonstrated practical data science workflows with real clinical trial data.
