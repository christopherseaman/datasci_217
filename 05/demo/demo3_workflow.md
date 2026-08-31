---
jupyter:
  jupytext:
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    name: python
    version: 3.12.13
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

from pathlib import Path

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
orders_path = output_dir / 'orders_dirty.csv'
with orders_path.open('w') as f:
    f.write(ecommerce_data)

df = pd.read_csv(orders_path)
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

# 2. Standardize product names; retain blank names as explicitly missing.
df_clean['product_name'] = df_clean['product_name'].str.strip().str.title()
df_clean['product_name'] = df_clean['product_name'].replace('', pd.NA)

# 3. Replace negative prices with NaN, then fill with median
df_clean.loc[df_clean['price'] < 0, 'price'] = np.nan
df_clean['price'] = df_clean['price'].fillna(df_clean['price'].median())

# 4. Fill missing quantities with 1
df_clean['quantity'] = df_clean['quantity'].fillna(1)

# 5. Parse dates; retain invalid values as NaT for source review.
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
# Add row-level derived fields.  Keep this cleaning workflow independent of
# grouped summaries; aggregation belongs to the later GroupBy lecture.
df_clean['total_price'] = df_clean['quantity'] * df_clean['price']
print("\n=== ROW-LEVEL DERIVED FIELDS ===")
print(df_clean[['order_id', 'total_price']])
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
df_clean.to_csv(output_dir / 'orders_clean.csv', index=False)
print("\n✓ Saved output/orders_clean.csv")

# Create data quality report
report = f"""DATA CLEANING REPORT
====================

Original rows: {len(df)}
Cleaned rows: {len(df_clean)}
Rows removed: {len(df) - len(df_clean)}

Issues handled:
- Standardized {df['customer'].nunique()} customer names
- Fixed {(df['price'] < 0).sum()} negative prices
- Filled {df['quantity'].isnull().sum()} missing quantities
- Retained {df_clean['order_date'].isnull().sum()} unparseable dates as missing for source review
- Retained {df_clean['product_name'].isnull().sum()} blank product names as missing for source review

Final data quality:
- Missing values: {df_clean.isnull().sum().sum()}
- Duplicate rows: {df_clean.duplicated().sum()}
- Outliers detected: {len(outliers)}

Decision: resolved numeric and formatting defects passed validation. Missing
dates and product names remain documented review items and must not be silently
invented before analysis.
"""

with (output_dir / 'cleaning_report.txt').open('w') as f:
    f.write(report)
print("✓ Saved output/cleaning_report.txt")

print("\n=== WORKFLOW COMPLETE ===")
print("All files saved with unresolved fields documented for review.")
```
