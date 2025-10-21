"""
Assignment 6 Solution - Working as a student with lectures 01-06 knowledge
"""
import pandas as pd
import numpy as np
import os

print("="*60)
print("ASSIGNMENT 6: DATA WRANGLING")
print("="*60)

# ============================================================================
# SETUP: Verify data files
# ============================================================================
print("\n" + "="*60)
print("SETUP: Verifying data files")
print("="*60)

required_files = ['data/customers.csv', 'data/products.csv', 'data/purchases.csv']
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found. Run data_generator.ipynb first!")

print("✓ All data files found")

# ============================================================================
# QUESTION 1: MERGING DATASETS
# ============================================================================
print("\n" + "="*60)
print("QUESTION 1: MERGING DATASETS")
print("="*60)

# --- Part A: Basic Merge Operations ---
print("\n--- Part A: Basic Merge Operations ---")

# Load the three datasets
print("\nLoading datasets...")
customers = pd.read_csv('data/customers.csv')
products = pd.read_csv('data/products.csv')
purchases = pd.read_csv('data/purchases.csv')

print(f"Loaded {len(customers)} customers")
print(f"Loaded {len(products)} products")
print(f"Loaded {len(purchases)} purchases")

print("\nCustomers sample:")
print(customers.head())

print("\nProducts sample:")
print(products.head())

print("\nPurchases sample:")
print(purchases.head())

# Merge purchases with customers (left join)
print("\nMerging purchases with customers...")
purchase_customer = pd.merge(purchases, customers, on='customer_id', how='left')
print(f"Result: {len(purchase_customer)} rows")
print(purchase_customer.head(10))

# Merge the result with products
print("\nMerging with products...")
full_data = pd.merge(purchase_customer, products, on='product_id', how='left')
print(f"Result: {len(full_data)} rows")
print(full_data.head(10))

# Calculate total_price
print("\nCalculating total_price...")
full_data['total_price'] = (full_data['quantity'] * full_data['price']).round(2)
print(full_data.head(10))

# --- Part B: Join Type Analysis ---
print("\n--- Part B: Join Type Analysis ---")

# Inner join - only customers who made purchases
print("\nInner join (customers with purchases)...")
inner_result = pd.merge(customers, purchases, on='customer_id', how='inner')
print(f"Inner join result: {len(inner_result)} rows")
print(inner_result.head())

# Left join - all customers (including those with no purchases)
print("\nLeft join (all customers)...")
left_result = pd.merge(customers, purchases, on='customer_id', how='left')
print(f"Left join result: {len(left_result)} rows")
print(left_result.head())

# Find customers who haven't made any purchases
print("\nFinding customers with no purchases...")
no_purchases = left_result[left_result['purchase_id'].isna()]
print(f"Customers with no purchases: {len(no_purchases)}")
print(no_purchases.head())

# --- Part C: Multi-Column Merge ---
print("\n--- Part C: Multi-Column Merge ---")

# Create store-specific product pricing
store_pricing = pd.DataFrame({
    'product_id': ['P001', 'P001', 'P002', 'P002', 'P003', 'P003'],
    'store': ['Store A', 'Store B', 'Store A', 'Store B', 'Store A', 'Store B'],
    'discount_pct': [5, 10, 8, 5, 0, 15]
})

print("\nStore pricing:")
print(store_pricing)

# Merge on both product_id AND store
print("\nMerging on multiple columns...")
purchases_with_discount = pd.merge(
    purchases,
    store_pricing,
    on=['product_id', 'store'],
    how='left'
)
print(f"Result: {len(purchases_with_discount)} rows")
print(purchases_with_discount.head(10))

# --- Part D: Save Results ---
print("\n--- Part D: Save Results ---")

# Create output directory
os.makedirs('output', exist_ok=True)

# Save full_data
full_data.to_csv('output/q1_merged_data.csv', index=False)
print("✓ Saved output/q1_merged_data.csv")

# Create validation report
validation_report = f"""Question 1 Validation Report
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

with open('output/q1_validation.txt', 'w') as f:
    f.write(validation_report)
print("✓ Saved output/q1_validation.txt")

# ============================================================================
# QUESTION 2: CONCATENATING DATAFRAMES
# ============================================================================
print("\n" + "="*60)
print("QUESTION 2: CONCATENATING DATAFRAMES")
print("="*60)

# --- Part A: Vertical Concatenation ---
print("\n--- Part A: Vertical Concatenation ---")

# Split purchases into quarterly datasets
print("\nSplitting purchases by quarter...")
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

# Concatenate all quarters back together
print("\nConcatenating quarters...")
all_purchases = pd.concat([q1_purchases, q2_purchases, q3_purchases, q4_purchases],
                          ignore_index=True)
print(f"Total after concat: {len(all_purchases)} purchases")
print(all_purchases.head())

# Concatenate with source tracking using keys
print("\nConcatenating with keys...")
labeled_purchases = pd.concat([q1_purchases, q2_purchases, q3_purchases, q4_purchases],
                              keys=['Q1', 'Q2', 'Q3', 'Q4'])
print(labeled_purchases.head(10))

# --- Part B: Horizontal Concatenation ---
print("\n--- Part B: Horizontal Concatenation ---")

# Create customer satisfaction scores (subset of customers)
print("\nCreating customer metrics...")
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
print(satisfaction.head())
print("\nLoyalty tiers:")
print(loyalty.head())

# Horizontal concat to combine satisfaction and loyalty
print("\nHorizontal concatenation...")
customer_metrics = pd.concat([satisfaction, loyalty], axis=1, join='outer')
print(customer_metrics.head(10))

# Check for NaN values
print(f"\nMissing satisfaction scores: {customer_metrics['satisfaction_score'].isna().sum()}")
print(f"Missing loyalty tiers: {customer_metrics['tier'].isna().sum()}")

# Save customer_metrics
customer_metrics.to_csv('output/q2_combined_data.csv')
print("✓ Saved output/q2_combined_data.csv")

# ============================================================================
# QUESTION 3: RESHAPING AND ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("QUESTION 3: RESHAPING AND ANALYSIS")
print("="*60)

# --- Part A: Pivot Table Analysis ---
print("\n--- Part A: Pivot Table Analysis ---")

# Load the merged data from Question 1
print("\nLoading merged data...")
full_data = pd.read_csv('output/q1_merged_data.csv')

# Add month column for grouping
full_data['month'] = pd.to_datetime(full_data['purchase_date']).dt.to_period('M')

print(full_data.head())

# Create pivot table - sales by category and month
print("\nCreating pivot table...")
sales_pivot = pd.pivot_table(
    full_data,
    values='total_price',
    index='month',
    columns='category',
    aggfunc='sum'
)

print(sales_pivot)

# Save sales_pivot
sales_pivot.to_csv('output/q3_category_sales_wide.csv')
print("✓ Saved output/q3_category_sales_wide.csv")

# --- Part B: Melt and Long Format ---
print("\n--- Part B: Melt and Long Format ---")

# Reset index to make month a column
sales_wide = sales_pivot.reset_index()

# Melt to convert category columns back to rows
print("\nMelting to long format...")
sales_long = pd.melt(
    sales_wide,
    id_vars=['month'],
    var_name='category',
    value_name='sales'
)

print(sales_long.head(15))

# Calculate summary statistics
print("\nCalculating summary statistics...")
category_summary = sales_long.groupby('category')['sales'].agg(['sum', 'mean']).sort_values('sum', ascending=False)

print(category_summary)

# Create final analysis report
analysis_report = f"""Question 3 Analysis Report
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

# ============================================================================
# FINAL VERIFICATION
# ============================================================================
print("\n" + "="*60)
print("FINAL VERIFICATION")
print("="*60)

required_outputs = [
    'output/q1_merged_data.csv',
    'output/q1_validation.txt',
    'output/q2_combined_data.csv',
    'output/q3_category_sales_wide.csv',
    'output/q3_analysis_report.txt'
]

print("\nChecking required outputs:")
all_exist = True
for file in required_outputs:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")
    if not exists:
        all_exist = False

if all_exist:
    print("\n✓ All required files created! Assignment complete.")
else:
    print("\n✗ Some files are missing. Review the questions above.")

print("\n" + "="*60)
print("ASSIGNMENT COMPLETE!")
print("="*60)
