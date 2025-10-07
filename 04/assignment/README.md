# Assignment 4: Pandas Data Analysis

## Overview

This assignment tests your ability to work with pandas DataFrames for real-world data analysis. You'll load data, clean it, analyze it, and extract insights - the core workflow of a data scientist.

**Assignment Type:** Pass/Fail (auto-graded via pytest)

## Learning Objectives

- Load and inspect data from CSV files
- Select and filter data using pandas methods
- Handle missing values appropriately
- Perform groupby aggregations
- Create calculated columns and summaries

## Dataset

You'll work with a fictional dataset of customer orders from an e-commerce platform.

**File:** `orders.csv` (provided)

**Schema:**
```
order_id,customer_id,product,quantity,price,order_date,region,status
```

## Assignment Tasks

The assignment has **3 parts** that build on each other. Complete them in order.

### Part 1: Data Loading and Exploration (Foundation)

**File to edit:** `main.py`

Implement the `load_and_explore()` function that:

1. Loads `orders.csv` into a DataFrame
2. Prints the shape of the data
3. Prints the first 5 rows
4. Returns the DataFrame

**Expected behavior:**
```python
df = load_and_explore()
# Should print:
# Shape: (100, 8)
# First 5 rows: [displays DataFrame.head()]
# Returns: DataFrame with 100 rows, 8 columns
```

**Testing:** `pytest test_assignment.py::test_load_and_explore -v`

### Part 2: Data Cleaning and Filtering (Build on Part 1)

**File to edit:** `main.py`

Implement the `clean_and_filter()` function that:

1. Takes a DataFrame as input (from Part 1)
2. Removes rows where `status` is 'Cancelled'
3. Fills missing `quantity` values with 1
4. Keeps only orders from 'North' and 'South' regions
5. Returns the cleaned DataFrame

**Expected behavior:**
```python
df_clean = clean_and_filter(df)
# Should return DataFrame with:
# - No 'Cancelled' orders
# - No missing quantity values
# - Only 'North' and 'South' regions
```

**Testing:** `pytest test_assignment.py::test_clean_and_filter -v`

### Part 3: Analysis and Insights (Combine Part 1 & 2)

**File to edit:** `main.py`

Implement the `analyze_orders()` function that:

1. Takes a cleaned DataFrame as input (from Part 2)
2. Creates a new column `total_price` = `quantity` * `price`
3. Calculates total revenue by region (sum of `total_price` grouped by `region`)
4. Finds the top 3 products by total quantity sold
5. Returns a dictionary with:
   ```python
   {
       'revenue_by_region': Series,  # region as index, total revenue as values
       'top_3_products': Series       # product names as index, quantities as values
   }
   ```

**Expected output format:**
```python
results = analyze_orders(df_clean)
# results['revenue_by_region']:
#   North    15234.50
#   South    12456.75
#   dtype: float64

# results['top_3_products']:
#   Widget A    45
#   Widget C    38
#   Widget B    32
#   dtype: int64
```

**Testing:** `pytest test_assignment.py::test_analyze_orders -v`

## Starter Code

```python
# main.py
import pandas as pd

def load_and_explore():
    """
    Load orders.csv and explore the data.

    Returns:
        pd.DataFrame: The loaded data
    """
    # TODO: Implement this function
    pass

def clean_and_filter(df):
    """
    Clean and filter the orders data.

    Args:
        df (pd.DataFrame): Raw orders data

    Returns:
        pd.DataFrame: Cleaned and filtered data
    """
    # TODO: Implement this function
    pass

def analyze_orders(df_clean):
    """
    Analyze cleaned orders data.

    Args:
        df_clean (pd.DataFrame): Cleaned orders data

    Returns:
        dict: Dictionary with 'revenue_by_region' and 'top_3_products' Series
    """
    # TODO: Implement this function
    pass

if __name__ == "__main__":
    # Test your functions
    df = load_and_explore()
    df_clean = clean_and_filter(df)
    results = analyze_orders(df_clean)

    print("\n=== Revenue by Region ===")
    print(results['revenue_by_region'])

    print("\n=== Top 3 Products ===")
    print(results['top_3_products'])
```

## Testing Your Code

Run all tests:
```bash
pytest test_assignment.py -v
```

Run individual tests:
```bash
pytest test_assignment.py::test_load_and_explore -v
pytest test_assignment.py::test_clean_and_filter -v
pytest test_assignment.py::test_analyze_orders -v
```

## Grading

- **Part 1:** 33% - Data loading and exploration
- **Part 2:** 33% - Data cleaning and filtering
- **Part 3:** 34% - Analysis and insights

**Pass threshold:** All 3 parts must pass their tests

## Tips

1. **Part 1:**
   - Use `pd.read_csv()` to load the file
   - Use `.shape` for dimensions
   - Use `.head()` to preview data

2. **Part 2:**
   - Chain multiple filtering operations
   - Remember: `df[df['column'] != 'value']` removes rows
   - Use `.fillna()` for missing values
   - `.isin()` is useful for multiple values

3. **Part 3:**
   - Create column before grouping: `df['total'] = df['a'] * df['b']`
   - Use `.groupby()` with `.sum()` for aggregation
   - Use `.nlargest(3)` or `.sort_values()` + `.head(3)` for top 3

## Common Errors

- **FileNotFoundError:** Make sure `orders.csv` is in the same directory as `main.py`
- **KeyError:** Check column names are exactly as specified (case-sensitive!)
- **AssertionError in tests:** Your return values don't match expected format - check data types
- **AttributeError:** Make sure you're returning DataFrames/Series, not lists or dicts

## Submission

Your repository should contain:
- `main.py` - Your completed solution
- `orders.csv` - Provided data file (don't modify)
- `test_assignment.py` - Test file (don't modify)

Push to GitHub and verify the auto-grading passes (green checkmark).
