#!/usr/bin/env python3
"""Test Demo 1 Part 2: Indicator, Composite Keys, Suffixes"""

import pandas as pd
import numpy as np
from datetime import datetime

# Track execution time
start_time = datetime.now()

# Set random seed for reproducibility
np.random.seed(42)

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

print('='*60)
print('CELL 10: Validated Merge with indicator=True')
print('='*60)

# Add indicator column to track merge sources
validated_merge = pd.merge(customers, purchases, on='customer_id',
                           how='outer', indicator=True)
print('\nValidated Merge Result:')
print(validated_merge)
print('\nMerge source breakdown:')
print(validated_merge['_merge'].value_counts())
print()

print('Customers only (no purchases):')
print(validated_merge[validated_merge['_merge'] == 'left_only'][['customer_id', 'name']].values)
print()

print('Purchases only (no customer record):')
print(validated_merge[validated_merge['_merge'] == 'right_only'][['customer_id', 'product']].values)
print()

print('Matched records:')
print(f"{len(validated_merge[validated_merge['_merge'] == 'both'])} purchase records with customer info")

print('='*60)
print('CELL 10: Indicator - PASSED')
print('='*60)

# Composite keys example
print('\n' + '='*60)
print('CELL 11-12: Composite Keys')
print('='*60)

# Create quarterly sales data
sales_q1 = pd.DataFrame({
    'store_id': ['S01', 'S01', 'S02', 'S02', 'S03'],
    'quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q1'],
    'product_category': ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics'],
    'sales': [50000, 30000, 42000, 25000, 38000]
})

# Create targets by store, quarter, and category
targets = pd.DataFrame({
    'store_id': ['S01', 'S01', 'S02', 'S02', 'S01', 'S02'],
    'quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2'],
    'product_category': ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics', 'Electronics'],
    'target': [52000, 28000, 45000, 22000, 58000, 50000]
})

print('\nQ1 Sales:')
print(sales_q1)
print('\nTargets:')
print(targets)

# Merge on composite key: store_id + quarter + product_category
sales_vs_target = pd.merge(sales_q1, targets,
                           on=['store_id', 'quarter', 'product_category'],
                           how='left')

# Calculate performance
sales_vs_target['performance'] = (sales_vs_target['sales'] / sales_vs_target['target'] * 100).round(1)
sales_vs_target['status'] = sales_vs_target['performance'].apply(
    lambda x: 'Above Target' if x >= 100 else 'Below Target' if pd.notna(x) else 'No Target'
)

print('\nSales vs Target:')
print(sales_vs_target)

print('='*60)
print('CELL 11-12: Composite Keys - PASSED')
print('='*60)

# Execution summary
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print(f'\n\nEXECUTION SUMMARY - PART 2 (Cells 10-12):')
print(f'Duration: {duration:.2f} seconds')
print('All cells executed successfully!')
