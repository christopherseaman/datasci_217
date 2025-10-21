# Assignment 5, Question 3: Data Utilities Library - Design Notes

## Overview
Core reusable pandas utilities for data loading, cleaning, and transformation. These functions will be imported by Q4-Q7 notebooks for clinical trial data analysis.

## Function Design Decisions

### 1. `load_data(filepath: str) -> pd.DataFrame`
**Purpose**: Load CSV files into pandas DataFrames
**Design**: Simple wrapper around `pd.read_csv()` for consistency
**Lecture Coverage**: Lecture 4 - pandas DataFrame basics

### 2. `clean_data(df, remove_duplicates=True, sentinel_value=-999) -> pd.DataFrame`
**Purpose**: Basic data cleaning operations
**Design Decisions**:
- Uses `.copy()` to avoid modifying original DataFrame
- `.drop_duplicates()` removes exact duplicate rows
- `.replace()` converts sentinel values (-999, -1) to `pd.NA`
- Returns new DataFrame (immutable pattern)
**Lecture Coverage**: Lecture 5 - Data cleaning strategies, sentinel values

### 3. `detect_missing(df) -> pd.Series`
**Purpose**: Count missing values per column
**Design**: Uses `.isnull().sum()` chain for efficient counting
**Returns**: Series with column names as index, counts as values
**Lecture Coverage**: Lecture 4 - Missing data detection with `.isnull()`

### 4. `fill_missing(df, column, strategy='mean') -> pd.DataFrame`
**Purpose**: Fill missing values with different strategies
**Design Decisions**:
- Three strategies: 'mean', 'median', 'ffill' (forward fill)
- Calculates fill value first, then applies with `.fillna()`
- Raises `ValueError` for unknown strategies
- Only modifies specified column
**Lecture Coverage**: Lecture 4 - `.fillna()` method and strategies

### 5. `filter_data(df, filters: list) -> pd.DataFrame`
**Purpose**: Apply sequential filters to DataFrame
**Design Decisions**:
- Accepts list of filter dictionaries with keys: 'column', 'condition', 'value'
- Supports 5 conditions:
  - 'equals': Exact match (`==`)
  - 'greater_than': Numeric comparison (`>`)
  - 'less_than': Numeric comparison (`<`)
  - 'in_range': Inclusive range (uses `&` for compound condition)
  - 'in_list': Value in list (`.isin()` method)
- Filters applied sequentially (order matters)
- Each filter reduces the DataFrame further
**Lecture Coverage**: Lecture 4 - Boolean indexing and filtering

**Example Use Cases**:
```python
# Age range filter
filters = [{'column': 'age', 'condition': 'in_range', 'value': [18, 65]}]

# Multiple sequential filters
filters = [
    {'column': 'age', 'condition': 'greater_than', 'value': 18},
    {'column': 'site', 'condition': 'in_list', 'value': ['Site A', 'Site B']}
]
```

### 6. `transform_types(df, type_map: dict) -> pd.DataFrame`
**Purpose**: Convert column data types
**Design Decisions**:
- Accepts mapping: `{column_name: target_type}`
- Four supported types:
  - 'datetime': Uses `pd.to_datetime()` with `errors='coerce'`
  - 'numeric': Uses `pd.to_numeric()` with `errors='coerce'`
  - 'category': Uses `.astype('category')` for categorical data
  - 'string': Uses `.astype('string')` for text data
- `errors='coerce'` converts unparseable values to NaN (robust handling)
- Skips columns not in DataFrame
**Lecture Coverage**: Lecture 4 - Data types and `.astype()` method

**Why 'category' type?**
- Memory efficient for repetitive text values
- Enables categorical operations
- Common for site names, treatment groups

### 7. `create_bins(df, column, bins, labels, new_column=None) -> pd.DataFrame`
**Purpose**: Create categorical bins from continuous data
**Design Decisions**:
- Uses `pd.cut()` for equal-width binning
- Bins: List of edges (n+1 edges for n bins)
- Labels: List of names for each bin (length = len(bins) - 1)
- Default new column name: `{column}_binned`
- Common use: Age groups, BMI categories
**Lecture Coverage**: Lecture 5 - `pd.cut()` for discretization

**Example**:
```python
bins = [0, 18, 35, 50, 65, 100]
labels = ['<18', '18-34', '35-49', '50-64', '65+']
df = create_bins(df, 'age', bins, labels)
# Creates 'age_binned' column with 5 categories
```

### 8. `summarize_by_group(df, group_col, agg_dict=None) -> pd.DataFrame`
**Purpose**: Group data and apply aggregations
**Design Decisions**:
- Uses `.groupby()` followed by `.agg()` or `.describe()`
- Two modes:
  1. Default (`agg_dict=None`): Uses `.describe()` on all numeric columns
  2. Custom: Applies specified aggregations per column
- Aggregation examples: 'mean', 'std', 'min', 'max', 'count'
- Can apply multiple aggregations to same column
**Lecture Coverage**: Lecture 4 - GroupBy operations

**Example Aggregations**:
```python
# Simple summary
summary = summarize_by_group(df, 'site')

# Custom aggregations
agg_dict = {
    'age': ['mean', 'std'],
    'bmi': 'mean',
    'adherence': ['mean', 'count']
}
summary = summarize_by_group(df, 'site', agg_dict)
```

## Key Design Patterns

### 1. Immutability
All functions use `.copy()` and return new DataFrames without modifying originals. This prevents unexpected side effects.

### 2. Error Handling
- `transform_types()`: `errors='coerce'` converts unparseable values to NaN
- `fill_missing()`: Raises `ValueError` for invalid strategies
- `filter_data()`: Raises `ValueError` for unknown conditions

### 3. Default Parameters
- Sensible defaults reduce boilerplate
- Optional parameters for customization
- Examples: `new_column=None`, `agg_dict=None`, `remove_duplicates=True`

### 4. Chainable Operations
Functions return DataFrames that can be chained:
```python
df = load_data('data.csv')
df = clean_data(df)
df = fill_missing(df, 'age', 'mean')
df = create_bins(df, 'age', [0, 18, 65, 100], ['Young', 'Adult', 'Senior'])
```

## Methods Not Covered in Lectures 1-5

All methods used are covered in the lectures:
- **Lecture 4**: DataFrame basics, Series, `.loc`/`.iloc`, missing data (`.isnull()`, `.fillna()`), data types (`.astype()`), GroupBy
- **Lecture 5**: Data cleaning (`.drop_duplicates()`, sentinel values), `pd.cut()` for binning

**Additional pandas methods used** (but covered in lectures):
- `.copy()` - Create independent copy (Lecture 4)
- `.replace()` - Replace values (Lecture 4)
- `.sum()` - Aggregation (Lecture 4)
- `.isin()` - Membership test (Lecture 4)
- `.mean()`, `.median()` - Statistical methods (Lecture 4)
- `.ffill()` - Forward fill strategy (Lecture 4)

## Testing Strategy

The `__main__` block includes a simple test:
1. Creates test DataFrame with missing values and sentinel value (-999)
2. Tests `detect_missing()` before cleaning
3. Tests `clean_data()` to replace sentinel values
4. Verifies missing value count increases (sentinel → NaN)

**To run full tests**:
```bash
uv run --with pandas --with numpy python 05/assignment/q3_data_utils.py
```

## Usage in Q4-Q7 Notebooks

These functions will be imported:
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

# Load and clean data
df = load_data('data/clinical_trial_raw.csv')
df = clean_data(df, sentinel_value=-999)

# Handle missing data
missing = detect_missing(df)
df = fill_missing(df, 'age', 'median')

# Filter for analysis
filters = [
    {'column': 'age', 'condition': 'in_range', 'value': [18, 65]},
    {'column': 'site', 'condition': 'in_list', 'value': ['Site A', 'Site B']}
]
df = filter_data(df, filters)

# Type conversions
df = transform_types(df, {
    'enrollment_date': 'datetime',
    'site': 'category'
})

# Create age groups
df = create_bins(df, 'age', [0, 18, 35, 50, 65, 100],
                 ['<18', '18-34', '35-49', '50-64', '65+'])

# Summarize by site
summary = summarize_by_group(df, 'site', {'age': ['mean', 'std']})
```

## Requirements Compliance

✅ **8 pandas functions implemented**:
1. `load_data()` - Load CSV ✓
2. `clean_data()` - Remove duplicates and sentinel values ✓
3. `detect_missing()` - Count missing per column ✓
4. `fill_missing()` - Fill with mean/median/ffill ✓
5. `filter_data()` - Apply multiple filters ✓
6. `transform_types()` - Convert types (datetime, numeric, category) ✓
7. `create_bins()` - Categorical bins with `pd.cut()` ✓
8. `summarize_by_group()` - Group and aggregate ✓

✅ **No numpy used** (assignment restriction)
✅ **Well-documented** with docstrings and examples
✅ **Reusable** for Q4-Q7 notebooks
✅ **Covered by lectures 1-5**

## Potential Enhancements (Not Required)

If needed in later questions:
- `merge_data()` - Join multiple DataFrames
- `pivot_data()` - Reshape with `.pivot_table()`
- `handle_outliers()` - IQR-based outlier detection
- `normalize_data()` - Min-max or z-score scaling
- `export_data()` - Save cleaned data to CSV

These can be added if Q4-Q7 require additional functionality.
