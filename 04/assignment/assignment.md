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

## Instructions

Complete the three questions below. Each question builds on the previous one.

**Important:**
- Run cells in order from top to bottom
- Each question must save a CSV file to the `output/` directory
- Test your work locally before submitting: `pytest .github/test/test_assignment.py -v`
- Before final submission: Restart kernel and "Run All" to verify everything works

---

## Setup

Run this cell first to import libraries and create the output directory:

```python
import pandas as pd
import numpy as np
import os

# Create output directory
os.makedirs('output', exist_ok=True)

print("✓ Setup complete")
```

---

## Question 1: Data Loading & Exploration

**Objective:** Load the dataset, select specific columns, perform basic inspection, and generate summary statistics.

### Part A: Load and Inspect the Data

Load `data/customer_purchases.csv` and display basic information about the dataset.

```python
# TODO: Load the CSV file into a DataFrame called 'df'
df = None  # Replace with pd.read_csv(...)

# TODO: Display the shape of the DataFrame
# Hint: Use .shape

# TODO: Display the first 5 rows
# Hint: Use .head()

# TODO: Display column names and data types
# Hint: Use .info()
```

### Part B: Check for Missing Values

Identify which columns have missing values and how many.

```python
# TODO: Display the count of missing values for each column
# Hint: Use .isnull().sum()
```

### Part C: Select Numeric Columns

Create a new DataFrame containing only the numeric columns from the original data.

```python
# TODO: Select only numeric columns into a new DataFrame called 'df_numeric'
# Hint: Use .select_dtypes(include=['number'])
df_numeric = None  # Replace with your code
```

### Part D: Generate Summary Statistics

Calculate summary statistics for the numeric columns.

```python
# TODO: Generate summary statistics for df_numeric
# Hint: Use .describe()

# TODO: Save the summary statistics to 'output/exploration_summary.csv'
# Hint: Use .to_csv()
# Example: summary_stats.to_csv('output/exploration_summary.csv')
```

**Expected output file:** `output/exploration_summary.csv` containing summary statistics (count, mean, std, min, 25%, 50%, 75%, max) for numeric columns.

---

## Question 2: Data Cleaning & Transformation

**Objective:** Handle missing values, convert data types, filter the data, and save the cleaned dataset.

### Part A: Handle Missing Values

Fill missing values in the `quantity` column with 1, and drop rows where `shipping_method` is missing.

```python
# TODO: Fill missing 'quantity' values with 1
# Hint: Use .fillna()

# TODO: Drop rows where 'shipping_method' is missing
# Hint: Use .dropna(subset=['column_name'])

# TODO: Verify no missing values remain in 'quantity' and 'shipping_method'
# Hint: Use .isnull().sum()
```

### Part B: Convert Data Types

Convert the `purchase_date` column from string to datetime, and `quantity` to integer.

```python
# TODO: Convert 'purchase_date' to datetime
# Hint: Use pd.to_datetime()

# TODO: Convert 'quantity' to integer type
# Hint: Use .astype('int64')

# TODO: Verify the data types changed
# Hint: Use .dtypes
```

### Part C: Filter the Data

Keep only purchases from California (CA) and New York (NY) with quantity greater than or equal to 2.

```python
# TODO: Filter the DataFrame to keep only:
#       - customer_state is 'CA' or 'NY'
#       - quantity >= 2
# Hint: Use boolean indexing with &
# Hint: Use .isin(['CA', 'NY']) for multiple values
df_filtered = None  # Replace with your filtering code

# TODO: Display how many rows remain after filtering
```

### Part D: Save Cleaned Data

Save the cleaned and filtered DataFrame.

```python
# TODO: Save df_filtered to 'output/cleaned_data.csv' (without the index)
# Hint: Use .to_csv('filename.csv', index=False)
```

**Expected output file:** `output/cleaned_data.csv` containing the cleaned and filtered data (no missing values in quantity/shipping_method, only CA/NY states, quantity >= 2, datetime and integer types).

---

## Question 3: Analysis & Aggregation

**Objective:** Create calculated columns, perform groupby aggregations, find top products, and save results.

**Note:** Use the cleaned DataFrame (`df_filtered`) from Question 2 for this question.

### Part A: Create a Calculated Column

Create a new column `total_price` by multiplying `quantity` and `price_per_item`.

```python
# TODO: Create 'total_price' column
# Hint: df['new_col'] = df['col1'] * df['col2']

# TODO: Display the first few rows to verify
```

### Part B: Calculate Total Revenue by Product Category

Group the data by `product_category` and calculate the sum of `total_price` for each category.

```python
# TODO: Group by 'product_category' and sum 'total_price'
# Hint: Use .groupby('column')['target'].sum()
revenue_by_category = None  # Replace with your code

# TODO: Sort by revenue (descending)
# Hint: Use .sort_values(ascending=False)

# TODO: Display the results
```

### Part C: Find Top 5 Products by Quantity Sold

Find the 5 products with the highest total quantity sold.

```python
# TODO: Group by 'product_name' and sum 'quantity'
# TODO: Get the top 5 using .nlargest(5) or .sort_values().head(5)
top_5_products = None  # Replace with your code

# TODO: Display the results
```

### Part D: Save Analysis Results

Combine the revenue by category and top 5 products into a summary and save it.

```python
# TODO: Create a DataFrame with your analysis results
# Hint: You can create a DataFrame with:
#   pd.DataFrame({
#       'category_revenue': revenue_by_category,
#       'top_products': top_5_products  # You may need to reindex or align these
#   })
#
# OR save them separately and combine in a way that makes sense
# The tests will check that you saved the correct aggregated data

# TODO: Save to 'output/analysis_results.csv'
# Make sure the CSV has at least these columns or data:
#   - Product categories with their total revenue
#   - Top products by quantity
# Format the output so the auto-grader can find the aggregated values

# Example structure (adjust as needed):
analysis_summary = pd.DataFrame({
    'product_category': revenue_by_category.index,
    'total_revenue': revenue_by_category.values
})

# TODO: Save analysis_summary or your own structured result
# analysis_summary.to_csv('output/analysis_results.csv', index=False)
```

**Expected output file:** `output/analysis_results.csv` containing aggregated analysis results (revenue by category, top products).
