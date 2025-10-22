---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Demo 3: Time Series Concatenation and Index Management

## Learning Objectives

- Concatenate DataFrames vertically (stacking rows) and horizontally (adding columns)
- Understand when to use `ignore_index=True` vs preserving indexes
- Master `set_index()` and `reset_index()` for index manipulation
- Handle misaligned indexes during concatenation
- Combine `concat()` and `merge()` in practical workflows
- Work with time-based indexes for temporal data

+++

## Setup

```{code-cell}
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
```

## Create Sample Data: Quarterly Sales Reports

We'll simulate monthly sales data that arrives in separate quarterly files.

```{code-cell}
# Q1 Sales (Jan-Mar 2023)
q1_sales = pd.DataFrame({
    'month': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
    'revenue': [125000, 132000, 145000],
    'units_sold': [1250, 1320, 1450],
    'returns': [50, 45, 60]
})

# Q2 Sales (Apr-Jun 2023)
q2_sales = pd.DataFrame({
    'month': pd.to_datetime(['2023-04-01', '2023-05-01', '2023-06-01']),
    'revenue': [158000, 165000, 178000],
    'units_sold': [1580, 1650, 1780],
    'returns': [55, 70, 65]
})

# Q3 Sales (Jul-Sep 2023)
q3_sales = pd.DataFrame({
    'month': pd.to_datetime(['2023-07-01', '2023-08-01', '2023-09-01']),
    'revenue': [185000, 192000, 175000],
    'units_sold': [1850, 1920, 1750],
    'returns': [80, 75, 68]
})

print("Q1 Sales:")
display(q1_sales)
print("\nQ2 Sales:")
display(q2_sales)
print("\nQ3 Sales:")
display(q3_sales)
```

**Scenario:** You receive quarterly sales files and need to combine them into a single dataset for annual analysis.

+++

## Vertical Concatenation: Stacking Rows

Use `pd.concat()` to stack DataFrames vertically (add more rows).

```{code-cell}
# Basic vertical concatenation
year_sales = pd.concat([q1_sales, q2_sales, q3_sales])

print("Combined Sales (Default):")
year_sales
```

**Problem:** Notice the index! It repeats (0, 1, 2, 0, 1, 2, 0, 1, 2)

**Why:** Each DataFrame has its own 0-2 index, and concat preserved them.

**Two solutions:**
1. Use `ignore_index=True` to create new sequential index
2. Use `set_index()` to make month the index

```{code-cell}
# Solution 1: ignore_index=True for clean sequential index
year_sales_clean = pd.concat([q1_sales, q2_sales, q3_sales], ignore_index=True)

print("Combined Sales (Clean Index):")
year_sales_clean
```

**Much better!** Now we have a clean 0-8 index.

**When to use `ignore_index=True`:**
- When original indexes don't matter (default numeric indexes)
- When you want clean sequential numbering
- When combining similar datasets from different sources

+++

## Using set_index() for Meaningful Row Labels

For time series data, the date should be the index!

```{code-cell}
# Solution 2: Use month as index (better for time series!)
year_sales_indexed = pd.concat([q1_sales, q2_sales, q3_sales], ignore_index=True)
year_sales_indexed = year_sales_indexed.set_index('month')

print("Combined Sales (Month as Index):")
year_sales_indexed
```

**Advantages of datetime index:**
- Can select by date: `year_sales_indexed.loc['2023-06']`
- Easy time-based filtering and resampling
- More meaningful than numeric index

```{code-cell}
# Example: Select Q2 data using datetime index
q2_data = year_sales_indexed.loc['2023-04':'2023-06']
print("Q2 Data (using datetime index):")
q2_data
```

```{code-cell}
# Example: Calculate quarterly totals
quarterly_totals = year_sales_indexed.resample('QE').sum()
print("\nQuarterly Totals (resample magic!):")
quarterly_totals
```

**This is why datetime indexes are powerful for time series!**

+++

## Horizontal Concatenation: Adding Columns

Use `axis=1` to concatenate side-by-side (adding more columns).

```{code-cell}
# Create additional metrics in separate DataFrames
# Marketing spend data
marketing = pd.DataFrame({
    'month': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01',
                             '2023-04-01', '2023-05-01', '2023-06-01']),
    'ad_spend': [12000, 15000, 18000, 20000, 22000, 25000],
    'impressions': [500000, 600000, 700000, 800000, 850000, 900000]
}).set_index('month')

# Customer satisfaction scores
satisfaction = pd.DataFrame({
    'month': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01',
                             '2023-04-01', '2023-05-01', '2023-06-01']),
    'nps_score': [45, 48, 52, 55, 58, 60],
    'survey_responses': [120, 135, 150, 165, 180, 195]
}).set_index('month')

print("Marketing Data:")
display(marketing.head())
print("\nSatisfaction Data:")
display(satisfaction.head())
```

```{code-cell}
# Get first 6 months of sales for this example
sales_h1 = year_sales_indexed.loc['2023-01':'2023-06']

# Horizontal concatenation (add columns)
combined_metrics = pd.concat([sales_h1, marketing, satisfaction], axis=1)

print("Combined Metrics (Horizontal Concat):")
combined_metrics
```

**What happened:**
- All DataFrames aligned by their **month index**
- Columns from each DataFrame added side-by-side
- Index values matched up automatically

**Key insight:** Horizontal concat uses index for alignment!

+++

## Handling Misaligned Indexes

What happens when indexes don't match perfectly?

```{code-cell}
# Create data with missing/extra months
partial_data = pd.DataFrame({
    'month': pd.to_datetime(['2023-02-01', '2023-03-01', '2023-04-01', 
                             '2023-07-01']),  # Missing Jan, May, Jun
    'social_engagement': [5000, 5500, 6000, 7000]
}).set_index('month')

print("Partial Data (Missing Some Months):")
display(partial_data)

# Concatenate with misaligned indexes
combined_misaligned = pd.concat([sales_h1, partial_data], axis=1)
print("\nCombined with Misaligned Indexes:")
combined_misaligned
```

**Result:** NaN values appear where indexes don't match!

**Default behavior:** `join='outer'` keeps all index values from both DataFrames.

**Alternative:** Use `join='inner'` to keep only matching indexes.

```{code-cell}
# Inner join - only keep matching months
combined_inner = pd.concat([sales_h1, partial_data], axis=1, join='inner')

print("Combined with Inner Join (Only Matching Months):")
combined_inner
```

**Now only months present in BOTH DataFrames appear!**

**Common pitfall:** Using horizontal concat when you should use merge. If indexes don't align well, consider `pd.merge()` instead!

+++

## Alternative: combine_first() for Filling Missing Values

When you have two DataFrames with overlapping indexes and want to fill missing values, `combine_first()` is simpler than concat.

```{code-cell}
# Create primary sales data with some missing values (NaN)
primary_sales = pd.DataFrame({
    'month': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01',
                             '2023-04-01', '2023-05-01']),
    'actual_revenue': [125000, np.nan, 145000, np.nan, 165000],
    'units_sold': [1250, np.nan, 1450, np.nan, 1650]
}).set_index('month')

# Create backup/estimated data
estimated_sales = pd.DataFrame({
    'month': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01',
                             '2023-04-01', '2023-05-01', '2023-06-01']),
    'actual_revenue': [120000, 130000, 140000, 155000, 160000, 175000],
    'units_sold': [1200, 1300, 1400, 1550, 1600, 1750]
}).set_index('month')

print("Primary Sales (with missing data):")
display(primary_sales)
print("\nEstimated Sales (backup data):")
display(estimated_sales)
```

**Scenario:** You have actual sales data but some months are missing. You have estimated/forecast data as backup.

```{code-cell}
# Use combine_first() to fill missing values
filled_sales = primary_sales.combine_first(estimated_sales)

print("Filled Sales (actual where available, estimated where missing):")
filled_sales
```

**What happened:**
- February and April: Used **estimated** values (130000, 155000)
- Jan, Mar, May: Kept **actual** values (125000, 145000, 165000)
- June: Added from estimated data (175000) - not in primary

**How combine_first() works:**
1. Starts with primary_sales (the caller DataFrame)
2. For each NaN value, looks up value from estimated_sales
3. Fills NaN with estimated value if available
4. Adds any extra indexes from estimated_sales

**vs concat():** Much cleaner syntax for this specific use case!

```{code-cell}
# Compare with concat approach (more complex)
concat_result = pd.concat([primary_sales, estimated_sales], axis=1, join='outer')
print("Concat result (creates duplicate columns):")
display(concat_result)

# Would need additional steps to merge columns
# This is why combine_first() is better for this use case!
```

**See the difference?** concat creates duplicate columns, combine_first() merges them intelligently.

```{code-cell}
# Real-world application: Add data quality flags
filled_sales['data_source'] = 'actual'
# Mark rows where primary had NaN as 'estimated'
mask = primary_sales['actual_revenue'].isna()
filled_sales.loc[mask[mask].index, 'data_source'] = 'estimated'

print("Final Dataset with Source Tracking:")
filled_sales
```

**Best practice:** Always track which values came from estimates vs actuals for transparency!

**When to use combine_first():**
- Filling missing values from a backup DataFrame
- Preferring one data source over another
- Combining forecasts with actuals
- Merging duplicate datasets with different coverage

**When NOT to use combine_first():**
- Stacking rows from different time periods (use concat)
- Joining by keys other than index (use merge)
- Need complex aggregation logic (use fillna with custom functions)

+++

## reset_index(): Moving Index Back to Columns

Sometimes you need to convert the index back to a regular column.

```{code-cell}
# Current state: month is the index
print("Before reset_index():")
display(combined_metrics.head())
print(f"Index name: {combined_metrics.index.name}")
print(f"Columns: {list(combined_metrics.columns)}")
```

```{code-cell}
# Reset index to make month a regular column
combined_reset = combined_metrics.reset_index()

print("\nAfter reset_index():")
display(combined_reset.head())
print(f"Index: {list(combined_reset.index)}")
print(f"Columns: {list(combined_reset.columns)}")
```

**What happened:**
- `month` moved from index to a regular column
- New default numeric index (0, 1, 2, ...) created

**When to use reset_index():**
- After groupby operations (groups become index)
- Before saving to CSV (indexes aren't always preserved)
- When you need the index as a column for analysis

```{code-cell}
# Alternative: drop the index instead of converting to column
combined_dropped = combined_metrics.reset_index(drop=True)

print("reset_index(drop=True) - Index Discarded:")
combined_dropped.head()
```

**Use `drop=True` when:** The index contains no useful information.

+++

## Combining concat() and merge() in Workflows

Real-world scenarios often require both operations.

```{code-cell}
# Step 1: Concatenate quarterly sales files
all_sales = pd.concat([q1_sales, q2_sales, q3_sales], ignore_index=True)
print("Step 1: Concatenated Sales Data")
display(all_sales.head())
```

```{code-cell}
# Step 2: Create product category data
# (This would come from a separate database table in reality)
products = pd.DataFrame({
    'month': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01',
                             '2023-04-01', '2023-05-01', '2023-06-01',
                             '2023-07-01', '2023-08-01', '2023-09-01']),
    'top_category': ['Electronics', 'Electronics', 'Clothing',
                     'Clothing', 'Electronics', 'Home Goods',
                     'Home Goods', 'Electronics', 'Clothing'],
    'new_customers': [120, 135, 150, 165, 180, 195, 210, 225, 240]
})

print("\nStep 2: Product Category Data")
display(products.head())
```

```{code-cell}
# Step 3: Merge sales with product data
sales_enriched = pd.merge(all_sales, products, on='month', how='left')

print("\nStep 3: Merged Sales + Product Data")
display(sales_enriched.head())
```

```{code-cell}
# Step 4: Calculate metrics and analyze
sales_enriched['return_rate'] = (sales_enriched['returns'] / 
                                 sales_enriched['units_sold'] * 100).round(2)
sales_enriched['revenue_per_unit'] = (sales_enriched['revenue'] / 
                                      sales_enriched['units_sold']).round(2)

print("\nStep 4: Final Analysis Dataset")
display(sales_enriched)
```

```{code-cell}
# Step 5: Analyze by product category
category_summary = sales_enriched.groupby('top_category').agg({
    'revenue': 'sum',
    'units_sold': 'sum',
    'new_customers': 'sum',
    'return_rate': 'mean'
}).round(2)

category_summary['avg_revenue_per_unit'] = (
    category_summary['revenue'] / category_summary['units_sold']
).round(2)

print("\nStep 5: Category Summary")
category_summary.sort_values('revenue', ascending=False)
```

**Complete workflow:**
1. **concat()** - Combine quarterly files (same structure)
2. **merge()** - Add related data from other sources (different structure)
3. **groupby()** - Analyze the enriched dataset

**Key insight:** concat for stacking, merge for joining!

+++

## Tracking Data Sources with keys Parameter

Use `keys` to label where data came from during concatenation.

```{code-cell}
# Concatenate with source labels
labeled_sales = pd.concat(
    [q1_sales, q2_sales, q3_sales],
    keys=['Q1', 'Q2', 'Q3'],
    names=['quarter', 'month_index']
)

print("Sales with Quarter Labels (MultiIndex):")
labeled_sales
```

**Created a MultiIndex!**
- Outer level: quarter (Q1, Q2, Q3)
- Inner level: month_index (0, 1, 2)

**Use case:** Track data provenance when combining multiple sources.

```{code-cell}
# Select all Q2 data using the outer index level
q2_only = labeled_sales.loc['Q2']
print("Q2 Data Only:")
q2_only
```

```{code-cell}
# Flatten the MultiIndex with reset_index
labeled_flat = labeled_sales.reset_index()
print("\nFlattened with Quarter Column:")
labeled_flat
```

**Perfect!** Now we have a `quarter` column showing data source.

+++

## Real-World Application: Year-Over-Year Analysis

Combining techniques to compare 2023 vs 2024 performance.

```{code-cell}
# Create 2024 Q1 data for comparison
q1_2024 = pd.DataFrame({
    'month': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
    'revenue': [145000, 152000, 168000],
    'units_sold': [1450, 1520, 1680],
    'returns': [48, 52, 58]
})

# Prepare both years with year label
q1_2023_labeled = q1_sales.copy()
q1_2023_labeled['year'] = 2023

q1_2024_labeled = q1_2024.copy()
q1_2024_labeled['year'] = 2024

# Concatenate both years
yoy_data = pd.concat([q1_2023_labeled, q1_2024_labeled], ignore_index=True)

# Add month name for grouping
yoy_data['month_name'] = yoy_data['month'].dt.strftime('%B')

print("Year-Over-Year Q1 Data:")
yoy_data
```

```{code-cell}
# Pivot to compare 2023 vs 2024 side-by-side
yoy_comparison = yoy_data.pivot_table(
    index='month_name',
    columns='year',
    values=['revenue', 'units_sold']
)

print("\nYear-Over-Year Comparison (Pivoted):")
yoy_comparison
```

```{code-cell}
# Calculate growth rates
# Flatten column names for easier access
yoy_flat = yoy_comparison.copy()
yoy_flat.columns = ['_'.join(map(str, col)) for col in yoy_flat.columns]

yoy_flat['revenue_growth_%'] = (
    (yoy_flat['revenue_2024'] - yoy_flat['revenue_2023']) / 
    yoy_flat['revenue_2023'] * 100
).round(1)

yoy_flat['units_growth_%'] = (
    (yoy_flat['units_sold_2024'] - yoy_flat['units_sold_2023']) / 
    yoy_flat['units_sold_2023'] * 100
).round(1)

print("\nYear-Over-Year Growth Analysis:")
yoy_flat[['revenue_2023', 'revenue_2024', 'revenue_growth_%',
          'units_sold_2023', 'units_sold_2024', 'units_growth_%']]
```

**Business insights:**
- February 2024 revenue up **15.2%** vs 2023
- March 2024 shows strongest growth: **15.9%** revenue, **15.9%** units
- Consistent growth across all months

**Workflow used:**
1. **concat()** - Stack 2023 and 2024 data
2. **pivot_table()** - Create side-by-side comparison
3. Calculate derived metrics (growth rates)

+++

## Key Takeaways

1. **concat() for stacking similar DataFrames:**
   - Vertical (`axis=0`): Add more rows (default)
   - Horizontal (`axis=1`): Add more columns
   - Use `ignore_index=True` for clean sequential indexing

2. **set_index() makes columns into indexes:**
   - Essential for time series (use dates as index)
   - Enables powerful time-based operations
   - Makes selection more intuitive

3. **reset_index() moves indexes back to columns:**
   - After groupby operations
   - When saving to files
   - Use `drop=True` to discard index

4. **Index alignment in horizontal concat:**
   - Default: `join='outer'` (keep all indexes)
   - Alternative: `join='inner'` (only matching)
   - Creates NaN where indexes don't match

5. **Common workflow patterns:**
   - **concat → set_index:** Stack files then create meaningful index
   - **concat → merge:** Stack similar data, then join with related data
   - **concat with keys:** Track data sources with MultiIndex

6. **When to use concat vs merge:**
   - **concat:** Same structure, different time periods/sources
   - **merge:** Different structures, need to join by keys

7. **Index management best practices:**
   - Use datetime indexes for time series
   - Use meaningful indexes (not just 0, 1, 2)
   - Reset index before groupby results
   - Set index for better selection

**Practice tip:** Think of concat as "stacking LEGO bricks" - vertically or horizontally. Merge is like "connecting LEGO pieces by their studs" (keys)!
