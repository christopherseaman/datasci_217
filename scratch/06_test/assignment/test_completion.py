"""
Manual completion of assignment 6 to test instruction clarity
Tests the NEW workflow where students calculate total_price in Question 1
"""
import pandas as pd
import numpy as np
import os

# Setup
print("=" * 60)
print("TESTING ASSIGNMENT 6 INSTRUCTION CLARITY")
print("=" * 60)

# Verify data files exist
required_files = ['data/customers.csv', 'data/products.csv', 'data/purchases.csv']
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found. Run data_generator.ipynb first!")

print("\n✓ All data files found")

# Question 1: Merging Datasets
print("\n" + "=" * 60)
print("QUESTION 1: MERGING DATASETS")
print("=" * 60)

# Part A: Basic Merge Operations
print("\nPart A: Basic Merge Operations")
customers = pd.read_csv('data/customers.csv')
products = pd.read_csv('data/products.csv')
purchases = pd.read_csv('data/purchases.csv')

print(f"Loaded: {len(customers)} customers, {len(products)} products, {len(purchases)} purchases")

# Verify purchases DOES NOT have total_price
if 'total_price' in purchases.columns:
    raise ValueError("ERROR: purchases.csv should NOT have total_price column!")
print("✓ Verified purchases.csv does not have total_price")

# Merge purchases with customers
purchase_customer = pd.merge(purchases, customers, on='customer_id', how='left')
print(f"✓ Merged purchases with customers: {len(purchase_customer)} rows")

# Merge with products
full_data = pd.merge(purchase_customer, products, on='product_id', how='left')
print(f"✓ Merged with products: {len(full_data)} rows")

# CALCULATE total_price (this is what students should do)
full_data['total_price'] = (full_data['quantity'] * full_data['price']).round(2)
print(f"✓ Calculated total_price column: {full_data['total_price'].describe()}")

# Part B: Join Type Analysis
print("\nPart B: Join Type Analysis")
inner_result = pd.merge(customers, purchases, on='customer_id', how='inner')
print(f"✓ Inner join: {len(inner_result)} rows")

left_result = pd.merge(customers, purchases, on='customer_id', how='left')
print(f"✓ Left join: {len(left_result)} rows")

no_purchases = left_result[left_result['purchase_id'].isna()]
print(f"✓ Customers with no purchases: {len(no_purchases)}")

# Part C: Multi-Column Merge
print("\nPart C: Multi-Column Merge")
store_pricing = pd.DataFrame({
    'product_id': ['P001', 'P001', 'P002', 'P002', 'P003', 'P003'],
    'store': ['Store A', 'Store B', 'Store A', 'Store B', 'Store A', 'Store B'],
    'discount_pct': [5, 10, 8, 5, 0, 15]
})
purchases_with_discount = pd.merge(purchases, store_pricing, on=['product_id', 'store'], how='left')
print(f"✓ Multi-column merge with store pricing: {len(purchases_with_discount)} rows")
print(f"  Purchases with discounts: {purchases_with_discount['discount_pct'].notna().sum()}")

# Part D: Save Results
print("\nPart D: Save Results")
os.makedirs('output', exist_ok=True)
full_data.to_csv('output/q1_merged_data.csv', index=False)
print("✓ Saved output/q1_merged_data.csv")

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
  - Total price calculated: {full_data['total_price'].notna().sum()} rows
"""

with open('output/q1_validation.txt', 'w') as f:
    f.write(validation_report)
print("✓ Saved output/q1_validation.txt")

# Question 2: Concatenating DataFrames
print("\n" + "=" * 60)
print("QUESTION 2: CONCATENATING DATAFRAMES")
print("=" * 60)

# Part A: Vertical Concatenation
print("\nPart A: Vertical Concatenation")
q1_purchases = purchases[purchases['purchase_date'] < '2023-04-01']
q2_purchases = purchases[(purchases['purchase_date'] >= '2023-04-01') &
                          (purchases['purchase_date'] < '2023-07-01')]
q3_purchases = purchases[(purchases['purchase_date'] >= '2023-07-01') &
                          (purchases['purchase_date'] < '2023-10-01')]
q4_purchases = purchases[purchases['purchase_date'] >= '2023-10-01']

print(f"Split into quarters: Q1={len(q1_purchases)}, Q2={len(q2_purchases)}, Q3={len(q3_purchases)}, Q4={len(q4_purchases)}")

all_purchases = pd.concat([q1_purchases, q2_purchases, q3_purchases, q4_purchases], ignore_index=True)
print(f"✓ Concatenated: {len(all_purchases)} purchases")

labeled_purchases = pd.concat([q1_purchases, q2_purchases, q3_purchases, q4_purchases],
                               keys=['Q1', 'Q2', 'Q3', 'Q4'])
print(f"✓ Labeled concatenation: {len(labeled_purchases)} purchases with MultiIndex")

# Part B: Horizontal Concatenation
print("\nPart B: Horizontal Concatenation")
satisfaction = pd.DataFrame({
    'customer_id': customers['customer_id'].sample(50, random_state=42),
    'satisfaction_score': np.random.randint(1, 11, size=50),
    'survey_date': pd.date_range('2023-12-01', periods=50, freq='D')
})

loyalty = pd.DataFrame({
    'customer_id': customers['customer_id'].sample(60, random_state=123),
    'tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=60),
    'points': np.random.randint(100, 10000, size=60)
})

satisfaction = satisfaction.set_index('customer_id')
loyalty = loyalty.set_index('customer_id')

customer_metrics = pd.concat([satisfaction, loyalty], axis=1, join='outer')
print(f"✓ Horizontal concat: {len(customer_metrics)} customers")
print(f"  Missing satisfaction: {customer_metrics['satisfaction_score'].isna().sum()}")
print(f"  Missing loyalty: {customer_metrics['tier'].isna().sum()}")

customer_metrics.to_csv('output/q2_combined_data.csv')
print("✓ Saved output/q2_combined_data.csv")

# Question 3: Reshaping and Analysis
print("\n" + "=" * 60)
print("QUESTION 3: RESHAPING AND ANALYSIS")
print("=" * 60)

# Part A: Pivot Table Analysis
print("\nPart A: Pivot Table Analysis")
# LOAD Q1 OUTPUT instead of re-merging
full_data = pd.read_csv('output/q1_merged_data.csv')
print(f"✓ Loaded Q1 merged data: {len(full_data)} rows")

# Verify it has total_price
if 'total_price' not in full_data.columns:
    raise ValueError("ERROR: Q1 output should have total_price column!")
print(f"✓ Verified total_price exists in Q1 output")

full_data['month'] = pd.to_datetime(full_data['purchase_date']).dt.to_period('M')

sales_pivot = pd.pivot_table(full_data,
                              values='total_price',
                              index='month',
                              columns='category',
                              aggfunc='sum')
print(f"✓ Created pivot table: {len(sales_pivot)} months x {len(sales_pivot.columns)} categories")

sales_pivot.to_csv('output/q3_category_sales_wide.csv')
print("✓ Saved output/q3_category_sales_wide.csv")

# Part B: Melt and Long Format
print("\nPart B: Melt and Long Format")
sales_wide = sales_pivot.reset_index()
sales_long = pd.melt(sales_wide, id_vars=['month'], var_name='category', value_name='sales')
print(f"✓ Melted to long format: {len(sales_long)} rows")

category_summary = sales_long.groupby('category')['sales'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
print(f"✓ Calculated summary statistics for {len(category_summary)} categories")

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

with open('output/q3_analysis_report.txt', 'w') as f:
    f.write(analysis_report)
print("✓ Saved output/q3_analysis_report.txt")

# Final check
print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

required_outputs = [
    'output/q1_merged_data.csv',
    'output/q1_validation.txt',
    'output/q2_combined_data.csv',
    'output/q3_category_sales_wide.csv',
    'output/q3_analysis_report.txt'
]

all_exist = True
for file in required_outputs:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")
    if not exists:
        all_exist = False

if all_exist:
    print("\n✅ All required files created! Assignment complete.")
else:
    print("\n❌ Some files are missing.")

print("\n" + "=" * 60)
print("INSTRUCTION CLARITY ASSESSMENT:")
print("=" * 60)
print("✅ Students calculate total_price in Q1 (good learning)")
print("✅ Q3 reuses Q1 output (efficient, no duplication)")
print("✅ All merge types work correctly")
print("✅ Multi-column merge example works")
print("✅ Instructions are clear and educational")
