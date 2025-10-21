---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Assignment 6: Data Wrangling with Merge, Concat, and Reshape

**Deliverable:** Completed notebook with output files in `output/`

---

## Setup

First, make sure you've generated the data by running `data_generator.ipynb`.

```python
import pandas as pd
import numpy as np
import os

# Verify data files exist
required_files = ['data/customers.csv', 'data/products.csv', 'data/purchases.csv']
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found. Run data_generator.ipynb first!")

print("✓ All data files found")
```

---

## Dataset Column Reference

Use this reference when writing merge operations and selecting columns. Each dataset's columns are listed below with their data types and descriptions.

**`customers.csv` columns:**
- `customer_id` - Unique ID (C001, C002, ...)
- `name` - Customer full name
- `city` - Customer city
- `signup_date` - Registration date

**`products.csv` columns:**
- `product_id` - Unique ID (P001, P002, ...)
- `product_name` - Product name
- `category` - Product category (Electronics, Clothing, Home & Garden, Books, Sports)
- `price` - Product price in dollars

**`purchases.csv` columns:**
- `purchase_id` - Unique ID (T0001, T0002, ...)
- `customer_id` - Links to customers
- `product_id` - Links to products
- `quantity` - Number of items purchased
- `purchase_date` - Purchase date
- `store` - Store location (Store A, B, or C)

---

## Question 1: Merging Datasets

### Part A: Basic Merge Operations

Load the datasets and perform merge operations.

```python
# TODO: Load the three datasets
customers = None  # Load data/customers.csv
products = None   # Load data/products.csv
purchases = None  # Load data/purchases.csv

# Display first few rows of each
print("Customers:")
display(customers.head())
print("\nProducts:")
display(products.head())
print("\nPurchases:")
display(purchases.head())
```

```python
# TODO: Merge purchases with customers (left join)
# Keep all purchases, add customer information
purchase_customer = None

display(purchase_customer.head(10))
```

```python
# TODO: Merge the result with products to add product information
# Use left join to keep all purchases
full_data = None

display(full_data.head(10))
```

```python
# TODO: Calculate total_price for each purchase
# Multiply quantity by price to get the total cost
# Round to 2 decimal places
# Hint: full_data['total_price'] = (full_data['quantity'] * full_data['price']).round(2)

display(full_data.head(10))
```

### Part B: Join Type Analysis

Compare different join types to understand data relationships.

```python
# TODO: Inner join - only customers who made purchases
inner_result = None

print(f"Inner join result: {len(inner_result)} rows")
display(inner_result.head())
```

```python
# TODO: Left join - all customers (including those with no purchases)
left_result = None

print(f"Left join result: {len(left_result)} rows")
display(left_result.head())
```

```python
# TODO: Find customers who haven't made any purchases
# Hint: Use left join result and check where purchase_id is NaN
# Use .isna() to find NaN values: left_result[left_result['purchase_id'].isna()]
no_purchases = None

print(f"Customers with no purchases: {len(no_purchases)}")
display(no_purchases.head())
```

### Part C: Multi-Column Merge

Merge on multiple columns when single columns aren't unique enough.

```python
# Create store-specific product pricing
# (Different stores may have different prices for same product)
store_pricing = pd.DataFrame({
    'product_id': ['P001', 'P001', 'P002', 'P002', 'P003', 'P003'],
    'store': ['Store A', 'Store B', 'Store A', 'Store B', 'Store A', 'Store B'],
    'discount_pct': [5, 10, 8, 5, 0, 15]
})

# TODO: Merge purchases with store_pricing on BOTH product_id AND store
# Hint: Use on=['product_id', 'store']
purchases_with_discount = None

display(purchases_with_discount.head(10))
```

### Part D: Save Results

```python
# Create output directory
os.makedirs('output', exist_ok=True)

# TODO: Save full_data to output/q1_merged_data.csv
# Hint: Use .to_csv() with index=False

print("✓ Saved output/q1_merged_data.csv")
```

```python
# Create a validation report
validation_report = f"""
Question 1 Validation Report
============================

Dataset Sizes:
  - Customers: {len(customers)} rows
  - Products: {len(products)} rows
  - Purchases: {len(purchases)} rows

Merge Results:
  - Full merged data: {len(full_data)} rows
  - Inner join: {len(inner_result)} rows
  - Left join: {len(left_result)} rows
  - Customers with no purchases: {len(no_purchases)}

Data Quality:
  - Missing customer names: {full_data['name'].isna().sum()}
  - Missing product names: {full_data['product_name'].isna().sum()}
"""

# TODO: Save validation_report to output/q1_validation.txt
# Hint: Use open() with 'w' mode

print("✓ Saved output/q1_validation.txt")
```

---

## Question 2: Concatenating DataFrames

### Part A: Vertical Concatenation

Combine multiple DataFrames by stacking rows.

```python
# Split purchases into quarterly datasets
q1_purchases = purchases[purchases['purchase_date'] < '2023-04-01']
q2_purchases = purchases[(purchases['purchase_date'] >= '2023-04-01') &
                          (purchases['purchase_date'] < '2023-07-01')]
q3_purchases = purchases[(purchases['purchase_date'] >= '2023-07-01') &
                          (purchases['purchase_date'] < '2023-10-01')]
q4_purchases = purchases[purchases['purchase_date'] >= '2023-10-01']

print(f"Q1: {len(q1_purchases)} purchases")
print(f"Q2: {len(q2_purchases)} purchases")
print(f"Q3: {len(q3_purchases)} purchases")
print(f"Q4: {len(q4_purchases)} purchases")
```

```python
# TODO: Concatenate all quarters back together
# Use ignore_index=True for clean sequential indexing
# Hint: pd.concat([df1, df2, df3, df4], ignore_index=True)
all_purchases = None

print(f"Total after concat: {len(all_purchases)} purchases")
display(all_purchases.head())
print(f"\nVerify total rows: {len(q1_purchases)} + {len(q2_purchases)} + {len(q3_purchases)} + {len(q4_purchases)} = {len(all_purchases)}")
```

### Part B: Horizontal Concatenation

Add related information as new columns.

```python
# Create customer satisfaction scores (subset of customers)
satisfaction = pd.DataFrame({
    'customer_id': customers['customer_id'].sample(50, random_state=42),
    'satisfaction_score': np.random.randint(1, 11, size=50),
    'survey_date': pd.date_range('2023-12-01', periods=50, freq='D')
})

# Create customer loyalty tier (different subset)
loyalty = pd.DataFrame({
    'customer_id': customers['customer_id'].sample(60, random_state=123),
    'tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=60),
    'points': np.random.randint(100, 10000, size=60)
})

# Set customer_id as index for both
satisfaction = satisfaction.set_index('customer_id')
loyalty = loyalty.set_index('customer_id')

print("Satisfaction scores:")
display(satisfaction.head())
print("\nLoyalty tiers:")
display(loyalty.head())
```

```python
# TODO: Horizontal concat to combine satisfaction and loyalty
# Use outer join to keep all customers from both datasets
# Hint: pd.concat([df1, df2], axis=1, join='outer')
customer_metrics = None

display(customer_metrics.head(10))
```

```python
# Handle misaligned indexes - how many NaN values?
print(f"Missing satisfaction scores: {customer_metrics['satisfaction_score'].isna().sum()}")
print(f"Missing loyalty tiers: {customer_metrics['tier'].isna().sum()}")
```

```python
# TODO: Save customer_metrics to output/q2_combined_data.csv
# Hint: Use .to_csv() - index will be saved automatically

print("✓ Saved output/q2_combined_data.csv")
```

---

## Question 3: Reshaping and Analysis

### Part A: Pivot Table Analysis

Transform data to analyze patterns.

```python
# TODO: Load the merged data from Question 1
# This already has purchases merged with customers and products (and total_price calculated)
# Hint: pd.read_csv('output/q1_merged_data.csv')
full_data = None

# Add month column for grouping (YYYY-MM format like "2023-01")
full_data['month'] = pd.to_datetime(full_data['purchase_date']).dt.strftime('%Y-%m')

display(full_data.head())
```

```python
# TODO: Create pivot table - sales by category and month
# Use pivot_table to handle duplicate entries (aggregate with sum)
# Hint: pd.pivot_table(df, values='total_price', index='month', columns='category', aggfunc='sum')
sales_pivot = None

display(sales_pivot)
```

```python
# TODO: Save sales_pivot to output/q3_category_sales_wide.csv
# Hint: Use .to_csv()

print("✓ Saved output/q3_category_sales_wide.csv")
```

### Part B: Melt and Long Format

Convert wide format back to long for different analysis.

```python
# Reset index to make month a column
sales_wide = sales_pivot.reset_index()

# TODO: Melt to convert category columns back to rows
# Hint: pd.melt(df, id_vars=['month'], var_name='category', value_name='sales')
sales_long = None

display(sales_long.head(15))
```

```python
# TODO: Calculate summary statistics using the long format
# Group by category and calculate total sales, average monthly sales
# Hint: Use .groupby('category')['sales'].agg(['sum', 'mean']) and sort by sum descending
category_summary = None

display(category_summary)
```

```python
# Create final analysis report
analysis_report = f"""
Question 3 Analysis Report
==========================

Sales by Category (Total):
{category_summary.to_string()}

Time Period:
  - Start: {full_data['purchase_date'].min()}
  - End: {full_data['purchase_date'].max()}
  - Months: {full_data['month'].nunique()}

Top Category: {category_summary.index[0]}
Bottom Category: {category_summary.index[-1]}
"""

# TODO: Save analysis_report to output/q3_analysis_report.txt
# Hint: Use open() with 'w' mode

print("✓ Saved output/q3_analysis_report.txt")
```

---

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_merged_data.csv` - Merged customer/product/purchase data
- [ ] `output/q1_validation.txt` - Merge validation report
- [ ] `output/q2_combined_data.csv` - Concatenated data with metrics
- [ ] `output/q3_category_sales_wide.csv` - Pivoted category sales
- [ ] `output/q3_analysis_report.txt` - Sales analysis report

```python
# Run this cell to check all outputs exist
required_outputs = [
    'output/q1_merged_data.csv',
    'output/q1_validation.txt',
    'output/q2_combined_data.csv',
    'output/q3_category_sales_wide.csv',
    'output/q3_analysis_report.txt'
]

print("Checking required outputs:")
for file in required_outputs:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")

all_exist = all(os.path.exists(f) for f in required_outputs)
if all_exist:
    print("\n✓ All required files created! Ready to submit.")
else:
    print("\n✗ Some files are missing. Review the questions above.")
```
