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

# Complete Data Cleaning Workflow

End-to-end cleaning pipeline: detect → handle → validate → transform → save

```python
import pandas as pd
import numpy as np

print("Complete workflow tools loaded!")
```

## Load Dirty E-commerce Data

```python
# Realistic e-commerce data with multiple issues
ecommerce_data = """order_id,customer,product_name,price,quantity,order_date,status
O001,John Doe,  Widget A  ,29.99,2,2024-01-15,complete
O002,JANE SMITH,Widget B,-1,1,2024-01-16,COMPLETE
O003,john doe,widget a,29.99,,2024-XX-17,pending
O004,Bob Jones,Widget C,19.99,5,2024-01-18,Complete
O005,Jane Smith,Widget B,49.99,3,2024-01-19,cancelled
O006,BOB JONES,  ,35.50,2,2024-01-20,complete"""

with open('orders_dirty.csv', 'w') as f:
    f.write(ecommerce_data)

df = pd.read_csv('orders_dirty.csv')
print("Dirty e-commerce data:")
print(df)
```

## Detect Issues (Audit Data Quality)

```python
# Missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Duplicates (same customer + product + date)
print(f"\n=== DUPLICATES ===")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Data issues
print("\n=== DATA ISSUES ===")
print(f"Negative prices: {(df['price'] < 0).sum()}")
print(f"Missing quantities: {df['quantity'].isnull().sum()}")
print(f"Invalid dates: {df['order_date'].str.contains('XX', na=False).sum()}")
```

## Handle Issues Systematically

```python
df_clean = df.copy()

# 1. Fix customer names (standardize)
df_clean['customer'] = df_clean['customer'].str.strip().str.title()

# 2. Fix product names (strip whitespace, title case)
df_clean['product_name'] = df_clean['product_name'].str.strip().str.title()

# 3. Replace negative prices with NaN, then fill with median
df_clean.loc[df_clean['price'] < 0, 'price'] = np.nan
df_clean['price'] = df_clean['price'].fillna(df_clean['price'].median())

# 4. Fill missing quantities with 1
df_clean['quantity'] = df_clean['quantity'].fillna(1)

# 5. Fix dates - replace invalid with NaT
df_clean['order_date'] = pd.to_datetime(df_clean['order_date'], errors='coerce')

# 6. Standardize status
df_clean['status'] = df_clean['status'].str.lower().str.strip()

print("\n=== CLEANED DATA ===")
print(df_clean)
```

## Validate Cleaning

```python
# Validation checks
print("\n=== VALIDATION ===")
print(f"Missing values remaining: {df_clean.isnull().sum().sum()}")
print(f"Negative prices: {(df_clean['price'] < 0).sum()}")
print(f"Missing quantities: {df_clean['quantity'].isnull().sum()}")

# Verify data quality improved
print(f"\nData types:\n{df_clean.dtypes}")
print(f"\nUnique statuses: {df_clean['status'].unique()}")
```

## Transform for Analysis

```python
# Add calculated fields
df_clean['total_price'] = df_clean['quantity'] * df_clean['price']
df_clean['order_month'] = df_clean['order_date'].dt.to_period('M')

# Create customer spending summary
customer_summary = df_clean.groupby('customer').agg({
    'order_id': 'count',
    'total_price': 'sum'
}).rename(columns={'order_id': 'num_orders', 'total_price': 'total_spent'})

print("\n=== CUSTOMER SUMMARY ===")
print(customer_summary)
```

## Detect Outliers

```python
# IQR method for outlier detection
Q1 = df_clean['total_price'].quantile(0.25)
Q3 = df_clean['total_price'].quantile(0.75)
IQR = Q3 - Q1

outliers = df_clean[(df_clean['total_price'] < Q1 - 1.5 * IQR) |
                     (df_clean['total_price'] > Q3 + 1.5 * IQR)]

print(f"\nOutlier orders: {len(outliers)}")
if len(outliers) > 0:
    print(outliers[['order_id', 'customer', 'total_price']])
```

## Save Results

```python
# Save cleaned data
df_clean.to_csv('orders_clean.csv', index=False)
print("\n✓ Saved orders_clean.csv")

# Save summary
customer_summary.to_csv('customer_summary.csv')
print("✓ Saved customer_summary.csv")

# Create data quality report
report = f"""DATA CLEANING REPORT
====================

Original rows: {len(df)}
Cleaned rows: {len(df_clean)}
Rows removed: {len(df) - len(df_clean)}

Issues fixed:
- Standardized {df['customer'].nunique()} customer names
- Fixed {(df['price'] < 0).sum()} negative prices
- Filled {df['quantity'].isnull().sum()} missing quantities
- Corrected {df['order_date'].str.contains('XX', na=False).sum()} invalid dates

Final data quality:
- Missing values: {df_clean.isnull().sum().sum()}
- Duplicate rows: {df_clean.duplicated().sum()}
- Outliers detected: {len(outliers)}
"""

with open('cleaning_report.txt', 'w') as f:
    f.write(report)
print("✓ Saved cleaning_report.txt")

print("\n=== WORKFLOW COMPLETE ===")
print("All files saved. Data is clean and ready for analysis!")
```
