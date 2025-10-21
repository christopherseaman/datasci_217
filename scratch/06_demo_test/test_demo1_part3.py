#!/usr/bin/env python3
"""Test Demo 1 Part 3: Overlapping Columns and Real-World Application"""

import pandas as pd
import numpy as np
from datetime import datetime

# Track execution time
start_time = datetime.now()

# Set random seed for reproducibility
np.random.seed(42)

print('='*60)
print('CELL 13-15: Overlapping Column Names')
print('='*60)

# Create two datasets with overlapping 'total' column
monthly_sales = pd.DataFrame({
    'product_id': ['P001', 'P002', 'P003', 'P004'],
    'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'total': [150000, 45000, 32000, 78000],  # Sales total
    'units_sold': [150, 1800, 400, 260]
})

monthly_inventory = pd.DataFrame({
    'product_id': ['P001', 'P002', 'P003', 'P004'],
    'total': [45, 520, 125, 85],  # Inventory total
    'warehouse': ['Seattle', 'Seattle', 'Portland', 'Portland']
})

print('\nMonthly Sales:')
print(monthly_sales)
print('\nMonthly Inventory:')
print(monthly_inventory)

# Merge with default suffixes (_x and _y)
merged_default = pd.merge(monthly_sales, monthly_inventory, on='product_id')
print('\nDefault suffixes (_x and _y):')
print(merged_default)

# Merge with descriptive suffixes
merged_clear = pd.merge(monthly_sales, monthly_inventory,
                        on='product_id',
                        suffixes=('_sales', '_inventory'))

print('\nClear suffixes (_sales and _inventory):')
print(merged_clear)

# Calculate inventory turnover rate
merged_clear['turnover_rate'] = (merged_clear['units_sold'] /
                                 merged_clear['total_inventory']).round(1)

print('\nInventory Analysis:')
print(merged_clear[['product_name', 'units_sold', 'total_inventory', 'turnover_rate']])

print('='*60)
print('CELL 13-15: Overlapping Columns - PASSED')
print('='*60)

# Real-world application
print('\n' + '='*60)
print('CELL 16: Real-World Application')
print('='*60)

# Recreate base data
customers = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
    'name': ['Alice Chen', 'Bob Martinez', 'Charlie Kim', 'Diana Patel', 'Eric Thompson'],
    'city': ['Seattle', 'Portland', 'Seattle', 'Eugene', 'Tacoma'],
    'signup_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10',
                                    '2023-04-05', '2023-05-12'])
})

purchases = pd.DataFrame({
    'customer_id': ['C001', 'C001', 'C002', 'C003', 'C006', 'C001', 'C002'],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Tablet', 'USB Cable', 'Webcam'],
    'amount': [999.99, 25.99, 79.99, 299.99, 449.99, 12.99, 89.99],
    'purchase_date': pd.to_datetime(['2023-06-01', '2023-06-15', '2023-06-20',
                                     '2023-07-01', '2023-07-05', '2023-07-10',
                                     '2023-07-15'])
})

# Step 1: Left join to keep all customers
customer_purchases = pd.merge(customers, purchases, on='customer_id', how='left')

# Step 2: Fill missing amounts with 0 for customers without purchases
customer_purchases['amount'] = customer_purchases['amount'].fillna(0)

# Step 3: Group by city and calculate total spending
city_spending = customer_purchases.groupby('city').agg({
    'amount': 'sum',
    'customer_id': 'nunique',
    'product': 'count'
}).round(2)

city_spending.columns = ['total_revenue', 'unique_customers', 'total_transactions']
city_spending['avg_transaction'] = (city_spending['total_revenue'] /
                                    city_spending['total_transactions']).round(2)

print('\nCity Spending Analysis:')
print(city_spending.sort_values('total_revenue', ascending=False))

print('='*60)
print('CELL 16: Real-World Application - PASSED')
print('='*60)

# Execution summary
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print(f'\n\nEXECUTION SUMMARY - PART 3 (Cells 13-16):')
print(f'Duration: {duration:.2f} seconds')
print('All cells executed successfully!')
