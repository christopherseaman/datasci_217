#!/usr/bin/env python3
"""Test Demo 1 Part 1: Basic Merge Operations"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime

# Track execution time
start_time = datetime.now()

# Set random seed for reproducibility
np.random.seed(42)

print('='*60)
print('CELL 1: Setup - PASSED')
print('='*60)

# Customer master data
customers = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
    'name': ['Alice Chen', 'Bob Martinez', 'Charlie Kim', 'Diana Patel', 'Eric Thompson'],
    'city': ['Seattle', 'Portland', 'Seattle', 'Eugene', 'Tacoma'],
    'signup_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10',
                                    '2023-04-05', '2023-05-12'])
})

print('\nCustomers Table:')
print(customers)
print('='*60)
print('CELL 2: Create Customers - PASSED')
print('='*60)

# Purchase transaction data
purchases = pd.DataFrame({
    'customer_id': ['C001', 'C001', 'C002', 'C003', 'C006', 'C001', 'C002'],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Tablet', 'USB Cable', 'Webcam'],
    'amount': [999.99, 25.99, 79.99, 299.99, 449.99, 12.99, 89.99],
    'purchase_date': pd.to_datetime(['2023-06-01', '2023-06-15', '2023-06-20',
                                     '2023-07-01', '2023-07-05', '2023-07-10',
                                     '2023-07-15'])
})

print('\nPurchases Table:')
print(purchases)
print('='*60)
print('CELL 3: Create Purchases - PASSED')
print('='*60)

# Inner join
inner_merge = pd.merge(customers, purchases, on='customer_id', how='inner')
print('\nInner Merge Result:')
print(inner_merge)
print(f'\nRow count: {len(inner_merge)}')
print('='*60)
print('CELL 4: Inner Join - PASSED')
print('='*60)

# Left join
left_merge = pd.merge(customers, purchases, on='customer_id', how='left')
print('\nLeft Merge Result:')
print(left_merge)
print(f'\nRow count: {len(left_merge)}')
print('='*60)
print('CELL 5: Left Join - PASSED')
print('='*60)

# Find customers with no purchases
no_purchases = left_merge[left_merge['product'].isna()][['customer_id', 'name', 'city']]
print('\nCustomers with no purchases (marketing opportunity):')
print(no_purchases)
print('='*60)
print('CELL 6: Find No Purchases - PASSED')
print('='*60)

# Right join
right_merge = pd.merge(customers, purchases, on='customer_id', how='right')
print('\nRight Merge Result:')
print(right_merge)
print(f'\nRow count: {len(right_merge)}')
print('='*60)
print('CELL 7: Right Join - PASSED')
print('='*60)

# Find orphaned purchases
orphaned = right_merge[right_merge['name'].isna()][['customer_id', 'product', 'amount']]
print('\nOrphaned purchases (data quality issue):')
print(orphaned)
print('='*60)
print('CELL 8: Find Orphaned Purchases - PASSED')
print('='*60)

# Outer join
outer_merge = pd.merge(customers, purchases, on='customer_id', how='outer')
print('\nOuter Merge Result:')
print(outer_merge)
print(f'\nRow count: {len(outer_merge)}')
print('='*60)
print('CELL 9: Outer Join - PASSED')
print('='*60)

# Execution summary
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print(f'\n\nEXECUTION SUMMARY - PART 1 (Cells 1-9):')
print(f'Duration: {duration:.2f} seconds')
print('All cells executed successfully!')
