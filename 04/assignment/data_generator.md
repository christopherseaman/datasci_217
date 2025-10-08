---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Data Generator - Customer Purchases Dataset

This notebook generates the sample dataset for Assignment 4.

**Run this once to create `data/customer_purchases.csv`**

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(217)  # For reproducibility
```

```python
# Generate sample customer purchase data
n_records = 15000  # 100x larger dataset

data = {
    'purchase_id': [f'P{i:05d}' for i in range(1, n_records + 1)],
    'customer_id': np.random.choice([f'C{i:04d}' for i in range(1, 501)], n_records),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports'], n_records),
    'product_name': np.random.choice([
        'Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Monitor',
        'T-Shirt', 'Jeans', 'Jacket', 'Shoes', 'Hat',
        'Plant Pot', 'Garden Tools', 'Lamp', 'Rug', 'Cushion',
        'Novel', 'Cookbook', 'Magazine', 'Textbook',
        'Yoga Mat', 'Dumbbells', 'Tennis Racket', 'Basketball', 'Running Shoes'
    ], n_records),
    'quantity': np.random.choice([1, 2, 3, 4, 5], n_records),
    'price_per_item': np.round(np.random.uniform(9.99, 499.99, n_records), 2),
    'purchase_date': [(datetime(2024, 1, 1) + timedelta(days=int(x))).strftime('%Y-%m-%d') 
                      for x in np.random.uniform(0, 365, n_records)],
    'customer_state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'WA', 'IL'], n_records),
    'shipping_method': np.random.choice(['Standard', 'Express', 'Overnight'], n_records)
}

df = pd.DataFrame(data)

# Introduce some missing values (realistic scenario)
# About 4% missing values in quantity and shipping_method
n_missing = int(n_records * 0.04)
missing_indices = np.random.choice(df.index, size=n_missing * 2, replace=False)
df.loc[missing_indices[:n_missing], 'quantity'] = np.nan
df.loc[missing_indices[n_missing:], 'shipping_method'] = np.nan

# Show sample
print(f"Generated {len(df):,} records")
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nFirst few rows:")
df.head(10)
```

```python
# Save to CSV
df.to_csv('data/customer_purchases.csv', index=False)
print(f"✓ Saved {len(df):,} records to data/customer_purchases.csv")
print(f"  File size: {len(df.to_csv(index=False)) / 1024:.1f} KB")
```

## Dataset Preview & Summary Statistics

This section shows some analysis techniques you'll learn later in the course!

```python
# Summary statistics
print("=== NUMERIC COLUMN SUMMARY ===")
df.describe()
```

```python
# Category distributions
print("=== PRODUCT CATEGORIES ===")
df['product_category'].value_counts()
```

```python
# Preview: Simple visualization (you'll learn this in Lecture 9!)
import matplotlib.pyplot as plt

df['product_category'].value_counts().plot(kind='bar', title='Purchases by Category')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

```python
# Preview: Time series (Lecture 11!)
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
daily_purchases = df.groupby('purchase_date').size()

plt.figure(figsize=(12, 4))
daily_purchases.plot(title='Purchases Over Time')
plt.ylabel('Number of Purchases')
plt.xlabel('Date')
plt.tight_layout()
plt.show()
```
