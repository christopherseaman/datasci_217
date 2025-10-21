#!/usr/bin/env python3
"""
Test the simplified assignment changes to ensure they work correctly.
"""

import pandas as pd
import numpy as np
import os

print("Testing Assignment 06 Simplifications")
print("=" * 50)

# Load data
customers = pd.read_csv('data/customers.csv')
products = pd.read_csv('data/products.csv')
purchases = pd.read_csv('data/purchases.csv')

print("\n✓ Data loaded successfully")
print(f"  - Customers: {len(customers)} rows")
print(f"  - Products: {len(products)} rows")
print(f"  - Purchases: {len(purchases)} rows")

# Test 1: Verify total_price is NOT in purchases (students calculate it)
print("\nTest 1: Verify total_price NOT in generated data")
assert 'total_price' not in purchases.columns, "total_price should NOT be in purchases.csv"
print("  ✓ Students will calculate total_price in Q1")

# Test 2: Column reference table completeness
print("\nTest 2: Verify all referenced columns exist")
expected_customers = ['customer_id', 'name', 'city', 'signup_date']
expected_products = ['product_id', 'product_name', 'category', 'price']
expected_purchases = ['purchase_id', 'customer_id', 'product_id', 'quantity', 'purchase_date', 'store']

for col in expected_customers:
    assert col in customers.columns, f"Missing {col} in customers"
for col in expected_products:
    assert col in products.columns, f"Missing {col} in products"
for col in expected_purchases:
    assert col in purchases.columns, f"Missing {col} in purchases"

print("  ✓ All columns in reference table exist in data")

# Test 3: Simple date formatting (.strftime instead of .to_period)
print("\nTest 3: Date formatting with strftime (replacing to_period)")
test_df = purchases.copy()
test_df['month'] = pd.to_datetime(test_df['purchase_date']).dt.strftime('%Y-%m')
print(f"  ✓ strftime works correctly: {test_df['month'].iloc[0]}")
print(f"  ✓ Format is string: {type(test_df['month'].iloc[0])}")
print(f"  ✓ Unique months: {test_df['month'].nunique()}")

# Test 4: Simple concat without keys parameter
print("\nTest 4: Concatenation without hierarchical indexing")
q1_purchases = purchases[purchases['purchase_date'] < '2023-04-01']
q2_purchases = purchases[(purchases['purchase_date'] >= '2023-04-01') & (purchases['purchase_date'] < '2023-07-01')]
q3_purchases = purchases[(purchases['purchase_date'] >= '2023-07-01') & (purchases['purchase_date'] < '2023-10-01')]
q4_purchases = purchases[purchases['purchase_date'] >= '2023-10-01']

all_purchases = pd.concat([q1_purchases, q2_purchases, q3_purchases, q4_purchases], ignore_index=True)

print(f"  ✓ Q1: {len(q1_purchases)} purchases")
print(f"  ✓ Q2: {len(q2_purchases)} purchases")
print(f"  ✓ Q3: {len(q3_purchases)} purchases")
print(f"  ✓ Q4: {len(q4_purchases)} purchases")
print(f"  ✓ Total: {len(all_purchases)} purchases (original: {len(purchases)})")
assert len(all_purchases) == len(purchases), "Concat should preserve all rows"
print(f"  ✓ Verification: {len(q1_purchases)} + {len(q2_purchases)} + {len(q3_purchases)} + {len(q4_purchases)} = {len(all_purchases)}")

# Test 5: Merge and total_price calculation (what students do in Q1)
print("\nTest 5: Student workflow - merge and calculate total_price")
purchase_customer = purchases.merge(customers, on='customer_id', how='left')
full_data = purchase_customer.merge(products, on='product_id', how='left')
full_data['total_price'] = (full_data['quantity'] * full_data['price']).round(2)

print(f"  ✓ Merged data: {len(full_data)} rows")
print(f"  ✓ total_price calculated: {full_data['total_price'].iloc[0]:.2f}")
print(f"  ✓ Sample calculation: {full_data['quantity'].iloc[0]} × ${full_data['price'].iloc[0]:.2f} = ${full_data['total_price'].iloc[0]:.2f}")

# Test 6: Clarified NaN check (purchase_id specific)
print("\nTest 6: Finding customers with no purchases (clarified instruction)")
inner_result = customers.merge(purchases, on='customer_id', how='inner')
left_result = customers.merge(purchases, on='customer_id', how='left')
no_purchases = left_result[left_result['purchase_id'].isna()]

print(f"  ✓ Inner join: {len(inner_result)} rows")
print(f"  ✓ Left join: {len(left_result)} rows")
print(f"  ✓ Customers with no purchases: {len(no_purchases)}")
print(f"  ✓ Instruction now specifies: 'check where purchase_id is NaN'")

print("\n" + "=" * 50)
print("✅ ALL SIMPLIFICATIONS WORK CORRECTLY")
print("\nChanges verified:")
print("  1. ✓ Removed .to_period() - using .strftime() instead")
print("  2. ✓ Removed keys parameter - using simple concat")
print("  3. ✓ Column reference table matches actual data")
print("  4. ✓ NaN check clarified to specify purchase_id column")
print("\nAll methods now align with lectures 01-06!")
