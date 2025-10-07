---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Assignment 4: E-commerce Data Analysis

Welcome to your first Jupyter-based assignment! In this assignment, you'll use pandas to analyze e-commerce order data.

## Setup

First, import the necessary libraries:

```python
import pandas as pd
import numpy as np
```

## Part 1: Load and Explore Data (Foundation)

Load the `orders.csv` file and explore its structure.

### Task 1.1: Load the data

```python
def load_and_explore():
    """
    Load the orders.csv file and return basic information about the dataset.

    Returns:
        dict: A dictionary containing:
            - 'df': The loaded DataFrame
            - 'num_rows': Number of rows
            - 'num_cols': Number of columns
            - 'columns': List of column names
    """
    # TODO: Implement this function
    # Hint: Use pd.read_csv() to load the data
    # Hint: Use .shape to get dimensions
    # Hint: Use .columns to get column names

    pass
```

```python
# Test your function
result = load_and_explore()
print(f"Dataset has {result['num_rows']} rows and {result['num_cols']} columns")
print(f"Columns: {result['columns']}")
print("\nFirst few rows:")
print(result['df'].head())
```

### Task 1.2: Check for missing values

```python
# TODO: Check which columns have missing values
# Hint: Use .isnull().sum()
```

### Task 1.3: Explore data types

```python
# TODO: Display the data types of each column
# Hint: Use .dtypes or .info()
```

## Part 2: Clean and Filter (Builds on Part 1)

Clean the data by handling missing values and filtering out invalid records.

### Task 2.1: Implement the cleaning function

```python
def clean_and_filter(df):
    """
    Clean the DataFrame by:
    1. Filling missing 'quantity' values with 1
    2. Filling missing 'price' values with the median price
    3. Removing rows where status is 'cancelled'
    4. Removing rows with missing 'product' values

    Args:
        df (pd.DataFrame): Raw order data

    Returns:
        pd.DataFrame: Cleaned data
    """
    # TODO: Implement this function
    # Hint: Use .fillna() for missing values
    # Hint: Use df[df['column'] != 'value'] to filter rows
    # Hint: Use .dropna(subset=['column']) to drop rows with missing values

    pass
```

```python
# Test your function
df_raw = load_and_explore()['df']
df_clean = clean_and_filter(df_raw)
print(f"Original rows: {len(df_raw)}, Cleaned rows: {len(df_clean)}")
print("\nCleaned data:")
print(df_clean.head())
```

### Task 2.2: Verify cleaning

```python
# TODO: Verify there are no missing values in critical columns
# TODO: Verify there are no cancelled orders
# Hint: Use assertions or print statements to check
```

## Part 3: Analysis and Insights (Combines Parts 1 & 2)

Analyze the cleaned data to extract business insights.

### Task 3.1: Implement the analysis function

```python
def analyze_orders(df_clean):
    """
    Analyze the cleaned order data.

    Args:
        df_clean (pd.DataFrame): Cleaned order data

    Returns:
        dict: Analysis results containing:
            - 'total_revenue': Sum of all order totals (price * quantity)
            - 'avg_order_value': Average order total
            - 'top_product': Most frequently ordered product
            - 'orders_by_region': Series with count of orders per region
    """
    # TODO: Implement this function
    # Hint: Create a 'total' column by multiplying price and quantity
    # Hint: Use .sum(), .mean() for aggregations
    # Hint: Use .value_counts() to count by category
    # Hint: Use .mode() or .value_counts().index[0] for most common

    pass
```

```python
# Test your function
analysis = analyze_orders(df_clean)
print("=== ORDER ANALYSIS ===")
print(f"Total Revenue: ${analysis['total_revenue']:,.2f}")
print(f"Average Order Value: ${analysis['avg_order_value']:.2f}")
print(f"Top Product: {analysis['top_product']}")
print(f"\nOrders by Region:")
print(analysis['orders_by_region'])
```

### Task 3.2: Additional exploration (optional)

```python
# TODO: Try additional analysis:
# - Which region generates the most revenue?
# - What's the most expensive product?
# - What's the average quantity per order?
```

<!-- #region -->
## Submission

Make sure all three functions (`load_and_explore`, `clean_and_filter`, `analyze_orders`) are implemented and working correctly. Copy your final implementations to `main.py` for autograding.

## Tips

- Test each function independently before moving to the next part
- Use `df.head()` frequently to inspect your data
- Check the shape and info of DataFrames after each transformation
- Remember: each part builds on the previous one!
<!-- #endregion -->
