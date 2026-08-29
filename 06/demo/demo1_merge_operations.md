---
jupyter:
  jupytext:
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

# Demo 1: Customer Purchase Analysis with pd.merge()

## Learning Objectives

- Master database-style joins using `pd.merge()`
- Understand the four join types: inner, left, right, outer
- Handle duplicate keys in merge operations
- Merge on multiple columns with composite keys
- Handle overlapping column names with suffixes
- Validate merge results with `indicator=True`


## Setup

```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
```

## Create Sample Data

We'll create two realistic datasets:
- **Customers table**: Customer information (master list)
- **Purchases table**: Purchase transactions

```python
# Customer master data
customers = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
    'name': ['Alice Chen', 'Bob Martinez', 'Charlie Kim', 'Diana Patel', 'Eric Thompson'],
    'city': ['Seattle', 'Portland', 'Seattle', 'Eugene', 'Tacoma'],
    'signup_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10',
                                    '2023-04-05', '2023-05-12'])
})

print("Customers Table:")
customers.head()
```

```python
# Purchase transaction data
purchases = pd.DataFrame({
    'customer_id': ['C001', 'C001', 'C002', 'C003', 'C006', 'C001', 'C002'],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Tablet', 'USB Cable', 'Webcam'],
    'amount': [999.99, 25.99, 79.99, 299.99, 449.99, 12.99, 89.99],
    'purchase_date': pd.to_datetime(['2023-06-01', '2023-06-15', '2023-06-20',
                                     '2023-07-01', '2023-07-05', '2023-07-10',
                                     '2023-07-15'])
})

print("Purchases Table:")
purchases.head(10)
```

**Key Observations:**
- C001 (Alice) has **3 purchases** - many-to-one relationship
- C004 (Diana) and C005 (Eric) have **no purchases** - will be relevant for join types
- C006 has a purchase but **no customer record** - orphaned transaction


## Inner Join (Default)

Inner join returns **only rows with matching keys in BOTH tables**.

**Use case:** "Show me all purchases with complete customer information"

```python
# Inner join - only matching customer_ids
inner_merge = pd.merge(customers, purchases, on='customer_id', how='inner')
inner_merge
```

**Interpretation:**
- Result has **6 rows** (only customers C001, C002, C003 who made purchases)
- Missing customers: C004, C005 (no purchases)
- Missing purchase: C006's tablet (no customer record)
- Alice (C001) appears **3 times** because she has 3 purchases

**Common pitfall:** Always check row counts! If you expected all customers, inner join silently dropped C004 and C005.


## Left Join

Left join returns **ALL rows from left table** + matching rows from right.

**Use case:** "Show me all customers, including those who haven't purchased anything"

```python
# Left join - keep ALL customers
left_merge = pd.merge(customers, purchases, on='customer_id', how='left')
left_merge
```

**Interpretation:**
- Result has **8 rows** (all 5 unique customers, but C001 appears 3x)
- C004 and C005 appear with **NaN values** for purchase columns
- C006's orphaned purchase is **still excluded** (not in customers table)

**Real-world application:** Identify customers who need marketing engagement (those with NaN purchases).

```python
# Find customers who haven't made purchases
no_purchases = left_merge[left_merge['product'].isna()][['customer_id', 'name', 'city']]
print("Customers with no purchases (marketing opportunity):")
no_purchases
```

## Right Join

Right join returns **ALL rows from right table** + matching rows from left.

**Use case:** "Show me all purchases, including those with missing customer info"

```python
# Right join - keep ALL purchases
right_merge = pd.merge(customers, purchases, on='customer_id', how='right')
right_merge
```

**Interpretation:**
- Result has **7 rows** (all purchases preserved)
- C006's tablet purchase appears with **NaN customer info**
- C004 and C005 are **excluded** (no purchases)

**Real-world application:** Identify data quality issues (orphaned transactions).

```python
# Find orphaned purchases (data quality issue)
orphaned = right_merge[right_merge['name'].isna()][['customer_id', 'product', 'amount']]
print("Orphaned purchases (data quality issue):")
orphaned
```

## Outer Join

Outer join returns **ALL rows from BOTH tables**.

**Use case:** "Show me everything - all customers AND all purchases, regardless of matches"

```python
# Outer join - keep EVERYTHING
outer_merge = pd.merge(customers, purchases, on='customer_id', how='outer')
outer_merge
```

**Interpretation:**
- Result has **9 rows** (complete picture of all data)
- Includes customers with no purchases (C004, C005)
- Includes orphaned purchase (C006)
- NaN values appear where no match exists

**Real-world application:** Comprehensive data audit to identify all gaps.


## Validating Merge Results with indicator=True

The `indicator=True` parameter adds a `_merge` column showing the source of each row.

```python
# Add indicator column to track merge sources
validated_merge = pd.merge(customers, purchases, on='customer_id',
                           how='outer', indicator=True)
validated_merge
```

```python
# Check merge statistics
print("Merge source breakdown:")
print(validated_merge['_merge'].value_counts())
print()

# Find each category
print("Customers only (no purchases):")
print(validated_merge[validated_merge['_merge'] == 'left_only'][['customer_id', 'name']].values)
print()

print("Purchases only (no customer record):")
print(validated_merge[validated_merge['_merge'] == 'right_only'][['customer_id', 'product']].values)
print()

print("Matched records:")
print(f"{len(validated_merge[validated_merge['_merge'] == 'both'])} purchase records with customer info")
```

**Pro tip:** Always use `indicator=True` when debugging merges to understand what matched and what didn't!


## Merging on Multiple Columns (Composite Keys)

Sometimes a single column isn't unique enough - you need multiple columns to identify matches.

```python
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

print("Q1 Sales:")
display(sales_q1)
print("\nTargets:")
display(targets)
```

```python
# Merge on composite key: store_id + quarter + product_category
sales_vs_target = pd.merge(sales_q1, targets,
                           on=['store_id', 'quarter', 'product_category'],
                           how='left')

# Calculate performance
sales_vs_target['performance'] = (sales_vs_target['sales'] / sales_vs_target['target'] * 100).round(1)
sales_vs_target['status'] = sales_vs_target['performance'].apply(
    lambda x: 'Above Target' if x >= 100 else 'Below Target' if pd.notna(x) else 'No Target'
)

sales_vs_target
```

**Interpretation:**
- Each row requires **all three keys** to match (store + quarter + category)
- S03 has no target set (NaN values)
- Most stores are performing close to targets

**Why composite keys matter:** If we merged only on `store_id`, we'd match Q1 sales with Q2 targets - wrong!


## Handling Overlapping Column Names

When both DataFrames have columns with the same name (besides merge keys), pandas adds suffixes.

```python
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

print("Monthly Sales:")
display(monthly_sales)
print("\nMonthly Inventory:")
display(monthly_inventory)
```

```python
# Merge with default suffixes (_x and _y)
merged_default = pd.merge(monthly_sales, monthly_inventory, on='product_id')
print("Default suffixes (_x and _y):")
merged_default
```

**Problem:** `total_x` and `total_y` are confusing! Which is which?

```python
# Merge with descriptive suffixes
merged_clear = pd.merge(monthly_sales, monthly_inventory,
                        on='product_id',
                        suffixes=('_sales', '_inventory'))

print("Clear suffixes (_sales and _inventory):")
merged_clear
```

**Much better!** Now it's immediately clear:
- `total_sales` = revenue from sales
- `total_inventory` = units in stock

**Best practice:** Always use descriptive suffixes that explain what each column represents.

```python
# Calculate inventory turnover rate
merged_clear['turnover_rate'] = (merged_clear['units_sold'] /
                                 merged_clear['total_inventory']).round(1)

print("Inventory Analysis:")
merged_clear[['product_name', 'units_sold', 'total_inventory', 'turnover_rate']]
```

**Interpretation:**
- Mouse has highest turnover (3.5x) - selling fast!
- Laptop has lowest turnover (3.3x) - slower movement
- This analysis was only possible by properly merging overlapping column names


## Real-World Application: Complete Customer Analysis

Combining multiple merge operations to answer: "What's the total spending by city?"

```python
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

city_spending.sort_values('total_revenue', ascending=False)
```

**Business insights from our merge:**
- Seattle generates highest revenue ($1,338.96) from 2 customers
- Portland has best average transaction value ($84.99)
- Eugene and Tacoma have customers but no purchases (engagement opportunity)

**Key workflow:** merge → fill missing → group → analyze


## Key Takeaways

1. **Join types matter:** Choose based on which data you want to preserve
   - **Inner:** Only matches (most restrictive)
   - **Left:** All left table + matches (common for "master" tables)
   - **Right:** All right table + matches (less common)
   - **Outer:** Everything (comprehensive audit)

2. **Always validate merges:**
   - Check row counts before and after
   - Use `indicator=True` to track merge sources
   - Look for unexpected NaN values

3. **Composite keys prevent wrong matches:**
   - Use `on=['col1', 'col2']` when single column isn't unique
   - Common for hierarchical data (store + date, region + quarter)

4. **Handle overlapping columns properly:**
   - Use descriptive suffixes like `('_sales', '_inventory')`
   - Avoid default `_x` and `_y` for readability

5. **Left joins are most common in practice:**
   - Preserve your "master" table (customers, products, etc.)
   - Add related information from other tables
   - Handle missing data appropriately

**Next steps:** Practice with your own datasets. Start simple, validate results, then build complexity!
