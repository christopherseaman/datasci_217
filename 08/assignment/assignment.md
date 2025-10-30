---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Assignment 8: Data Aggregation and Group Operations

## Overview
This assignment covers data aggregation and group operations using a health data lens (think EHR-like tables for departments, staff, regions, and activity). We’ll use the existing `department`, `employee`, and `sales` columns as proxies for clinical departments, clinicians, and encounters to keep the provided dataset and tests unchanged.

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('output', exist_ok=True)
```

## Question 1: Basic GroupBy Operations

### Part 1.1: Load and Explore Data

```python
# Load the datasets
employee_df = pd.read_csv('data/employee_data.csv')
department_df = pd.read_csv('data/department_data.csv')
sales_df = pd.read_csv('data/sales_data.csv')

print("Employee data shape:", employee_df.shape)
print("Department data shape:", department_df.shape)
print("Sales data shape:", sales_df.shape)

# Merge data for analysis
merged_df = sales_df.merge(employee_df, on='employee_id').merge(department_df, on='department_id')

print("\nMerged data shape:", merged_df.shape)
print("\nColumns:", merged_df.columns.tolist())
print("\nFirst few rows:")
print(merged_df.head())
```

### Part 1.2: Basic Aggregation (health context)

**TODO: Perform basic groupby operations**

```python
# Group by department (clinical department proxy) and calculate basic stats
dept_stats = merged_df.groupby('department_name').agg({
    'salary': ['mean', 'sum', 'count'],
    'performance_score': 'mean'
})

# Flatten multi-level columns for safe joins
dept_stats.columns = [
    '_'.join([str(c) for c in col if c is not None]).replace('salary_', 'salary_')
    if isinstance(col, tuple) else str(col)
    for col in dept_stats.columns.values
]

# Total encounters by department
if 'total_amount' in merged_df.columns:
    merged_df['_encounters'] = merged_df['total_amount']
elif {'quantity', 'unit_price'}.issubset(merged_df.columns):
    merged_df['_encounters'] = merged_df['quantity'] * merged_df['unit_price']
else:
    merged_df['_encounters'] = 1

dept_encounters = merged_df.groupby('department_name')['_encounters'].sum().to_frame('total_encounters')

# Combine and write
q1_df = dept_stats.join(dept_encounters)
q1_df.to_csv('output/q1_groupby_analysis.csv', index=True)

# Top-performing department by encounter volume
top_dept = dept_encounters['total_encounters'].idxmax()
with open('output/q1_aggregation_report.txt', 'w') as f:
    f.write('Q1 Aggregation Report\n')
    f.write(f"Top department by encounters: {top_dept}\n")
    f.write(f"Departments: {len(dept_encounters)}\n")
    f.write(q1_df.head().to_string())
```

### Part 1.3: Transform Operations (within-department norms)

**TODO: Use transform operations to add group statistics**

```python
# Add department (clinical unit) mean/std salary and normalized salary (z-score within department)
merged_df['dept_salary_mean'] = merged_df.groupby('department_name')['salary'].transform('mean')
merged_df['dept_salary_std'] = merged_df.groupby('department_name')['salary'].transform('std')
merged_df['salary_z'] = (merged_df['salary'] - merged_df['dept_salary_mean']) / merged_df['dept_salary_std']

# Add department total encounters as new column
dept_total_map = dept_encounters['total_encounters']
merged_df['dept_total_encounters'] = merged_df['department_name'].map(dept_total_map)

# Preview
merged_df[['department_name','employee_id','salary','dept_salary_mean','dept_salary_std','salary_z','dept_total_encounters']].head()
```

## Question 2: Advanced GroupBy Operations

### Part 2.1: Filter Operations (quality/scale gates)

**TODO: Use filter operations to remove groups**

```python
dept_counts = merged_df.groupby('department_name')['employee_id'].nunique()
dept_salary_mean = merged_df.groupby('department_name')['salary'].mean()
threshold_encounters = max(merged_df.get('_encounters', pd.Series([0])).sum() * 0.05, 10000)

keepers = (
    (dept_counts > 5) &
    (dept_salary_mean > 60000) &
    (dept_encounters['total_encounters'] > threshold_encounters)
)

filtered_depts = dept_encounters.loc[keepers.index[keepers]].copy()
filtered_depts['num_employees'] = dept_counts.loc[filtered_depts.index]
filtered_depts['avg_salary'] = dept_salary_mean.loc[filtered_depts.index]

# Save summary
filtered_depts.to_csv('output/q2_hierarchical_analysis.csv')

# Write a simple selection report
with open('output/q2_performance_report.txt','w') as f:
    f.write('Q2 Filter Summary\n')
    f.write(f"Kept departments: {len(filtered_depts)} / {dept_counts.size}\n")
    f.write(filtered_depts.head().to_string())
```

### Part 2.2: Apply Operations

**TODO: Use apply operations with custom functions**

```python
def salary_stats(group):
    return pd.Series({
        'count': len(group),
        'mean': group['salary'].mean(),
        'std': group['salary'].std(),
        'min': group['salary'].min(),
        'max': group['salary'].max(),
        'range': group['salary'].max() - group['salary'].min()
    })

dept_salary_summary = merged_df.groupby('department_name').apply(salary_stats)

top2_earners = merged_df.sort_values('salary', ascending=False).groupby('department_name').head(2)
top2_earners[['department_name','employee_id','salary']].head()
```

### Part 2.3: Hierarchical Grouping (dept × region)

**TODO: Perform multi-level grouping**

```python
hier_stats = merged_df.groupby(['department_name','region']).agg({
    '_encounters':'sum',
    'salary':'mean'
})
hier_stats_wide = hier_stats.unstack(fill_value=0)
hier_stats_wide.to_csv('output/q2_hierarchical_analysis.csv')
hier_stats.head()
```

## Question 3: Pivot Tables and Cross-Tabulations

### Part 3.1: Basic Pivot Tables (health service mix)

**TODO: Create pivot tables for multi-dimensional analysis**

```python
product_col = 'product_id' if 'product_id' in sales_df.columns else ('Product' if 'Product' in sales_df.columns else None)
region_col = 'region' if 'region' in sales_df.columns else ('Region' if 'Region' in sales_df.columns else None)

if 'total_amount' in sales_df.columns:
    measure = sales_df['total_amount']
elif {'quantity','unit_price'}.issubset(sales_df.columns):
    measure = sales_df['quantity'] * sales_df['unit_price']
else:
    measure = pd.Series(1, index=sales_df.index)

df_pvt = sales_df.copy()
df_pvt['_measure'] = measure

pivot = pd.pivot_table(df_pvt,
                       values='_measure',
                       index=product_col,
                       columns=region_col,
                       aggfunc='sum',
                       fill_value=0,
                       margins=True)
pivot.to_csv('output/q3_pivot_analysis.csv')
pivot.head()
```

### Part 3.2: Cross-Tabulations (caseload distribution)

**TODO: Create cross-tabulations for categorical analysis**

```python
dept_region_xtab = pd.crosstab(merged_df['department_name'], merged_df['region'], margins=True)
dept_region_xtab.to_csv('output/q3_crosstab_analysis.csv')
dept_region_xtab.head()
```

### Part 3.3: Pivot Table Visualization

**TODO: Create visualizations from pivot tables**

```python
import matplotlib.pyplot as plt
import seaborn as sns

pivot_viz = pd.read_csv('output/q3_pivot_analysis.csv', index_col=0)
if 'All' in pivot_viz.columns:
    pivot_viz = pivot_viz.drop(columns=['All'])
if 'All' in pivot_viz.index:
    pivot_viz = pivot_viz.drop(index=['All'])

plt.figure(figsize=(8, 5))
sns.heatmap(pivot_viz, annot=False, cmap='Blues')
plt.title('Encounters by Product and Region')
plt.tight_layout()
plt.savefig('output/q3_pivot_visualization.png', dpi=150)
plt.close()
```

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_groupby_analysis.csv` - Basic groupby analysis
- [ ] `output/q1_aggregation_report.txt` - Aggregation report
- [ ] `output/q2_hierarchical_analysis.csv` - Hierarchical analysis
- [ ] `output/q2_performance_report.txt` - Performance report
- [ ] `output/q3_pivot_analysis.csv` - Pivot table analysis
- [ ] `output/q3_crosstab_analysis.csv` - Cross-tabulation analysis
- [ ] `output/q3_pivot_visualization.png` - Pivot visualization

## Key Learning Objectives

- Master the split-apply-combine paradigm
- Apply aggregation functions and transformations
- Create pivot tables for multi-dimensional analysis
- Apply advanced groupby techniques
