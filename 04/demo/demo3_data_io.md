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

# Data I/O and Real-World Workflow

Loading real data, handling missing values, and performing analysis

```python
import pandas as pd
import os

# Create sample sales data
sales_data = """date,product,quantity,price,region
2024-01-15,Widget A,5,29.99,North
2024-01-16,Widget B,3,49.99,South
2024-01-16,Widget A,2,29.99,East
2024-01-17,Widget C,7,19.99,North
2024-01-17,Widget B,,49.99,South
2024-01-18,Widget A,4,29.99,West
2024-01-18,Widget C,6,,North
2024-01-19,Widget B,8,49.99,East"""

# Write to file
with open('sales_data.csv', 'w') as f:
    f.write(sales_data)

print("Created sales_data.csv")
```

## Load and Inspect Data

```python
# Read CSV
df_sales = pd.read_csv('sales_data.csv')

print("Data loaded:")
print(df_sales)

print("\nData types:")
print(df_sales.dtypes)

print("\nMissing values per column:")
print(df_sales.isnull().sum())
```

## Handle Missing Data

```python
# See rows with missing values
print("Rows with missing values:")
print(df_sales[df_sales.isnull().any(axis=1)])

# Fill missing quantity with 0
df_sales['quantity'] = df_sales['quantity'].fillna(0)

# Fill missing price with median price for that product
df_sales['price'] = df_sales.groupby('product')['price'].transform(lambda x: x.fillna(x.median()))

print("\nAfter handling missing values:")
print(df_sales)
print(f"\nRemaining missing values: {df_sales.isnull().sum().sum()}")
```

## Basic Analysis

```python
# Add calculated column
df_sales['total_sale'] = df_sales['quantity'] * df_sales['price']

print("Sales with totals:")
print(df_sales[['date', 'product', 'quantity', 'price', 'total_sale']])

# Summary by product
print("\nSales by product:")
product_summary = df_sales.groupby('product').agg({
    'quantity': 'sum',
    'total_sale': 'sum'
}).round(2)
print(product_summary)

# Summary by region
print("\nSales by region:")
region_summary = df_sales.groupby('region')['total_sale'].sum().sort_values(ascending=False)
print(region_summary)
```

## Value Counts and Quick Insights

```python
# Most common products
print("Product frequencies:")
print(df_sales['product'].value_counts())

# Regions ranked by number of sales
print("\nSales transactions by region:")
print(df_sales['region'].value_counts())

# Quick statistics
print(f"\nTotal revenue: ${df_sales['total_sale'].sum():.2f}")
print(f"Average transaction: ${df_sales['total_sale'].mean():.2f}")
print(f"Largest sale: ${df_sales['total_sale'].max():.2f}")
```

## Save Results

```python
# Save cleaned data
df_sales.to_csv('sales_data_clean.csv', index=False)
print("Saved cleaned data to sales_data_clean.csv")

# Save summary
product_summary.to_csv('product_summary.csv')
print("Saved product summary to product_summary.csv")

print("\nFiles created:")
for file in ['sales_data.csv', 'sales_data_clean.csv', 'product_summary.csv']:
    if os.path.exists(file):
        print(f"  ✓ {file}")
```
