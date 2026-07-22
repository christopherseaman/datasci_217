# Demo 1: GroupBy Operations

## Learning Objectives
- Master the split-apply-combine paradigm
- Apply aggregation functions to grouped data
- Use transform, filter, and apply operations
- Handle hierarchical grouping

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set inline plotting for Jupyter
%matplotlib inline

# Set random seed for reproducibility
np.random.seed(42)
```

## Part 1: Basic GroupBy Operations

### Create Sample Data

```python
# Create sample sales data
data = {
    'Department': ['Sales', 'Sales', 'Engineering', 'Engineering', 'Marketing', 'Marketing'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Salary': [50000, 55000, 80000, 85000, 60000, 65000],
    'Experience': [2, 3, 5, 7, 4, 6],
    'Region': ['North', 'South', 'North', 'South', 'North', 'South']
}

df = pd.DataFrame(data)
print("Sample Data:")
print(df)
```

### Basic Aggregation

```python
# Group by department and calculate statistics
print("=== Basic Aggregation ===")
print("Mean salary by department:")
print(df.groupby('Department')['Salary'].mean())

print("\nMultiple aggregations:")
print(df.groupby('Department').agg({
    'Salary': ['mean', 'sum', 'count'],
    'Experience': 'mean'
}))
```

### GroupBy with Multiple Columns

```python
# Group by multiple columns
print("=== Multi-column Grouping ===")
result = df.groupby(['Department', 'Region']).agg({
    'Salary': 'mean',
    'Experience': 'mean'
})
print(result)
```

## Part 2: Advanced GroupBy Operations

### Transform Operations

```python
# Transform: Add group statistics as new columns
print("=== Transform Operations ===")
df['Salary_Mean'] = df.groupby('Department')['Salary'].transform('mean')
df['Salary_Std'] = df.groupby('Department')['Salary'].transform('std')
df['Salary_Normalized'] = df.groupby('Department')['Salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

print("Data with group statistics:")
print(df[['Department', 'Employee', 'Salary', 'Salary_Mean', 'Salary_Std', 'Salary_Normalized']])
```

### Filter Operations

```python
# Filter: Keep only departments with more than 1 employee
print("=== Filter Operations ===")
filtered = df.groupby('Department').filter(lambda x: len(x) > 1)
print("Departments with multiple employees:")
print(filtered)

# Filter: Keep only departments with average salary > 60000
high_salary_depts = df.groupby('Department').filter(lambda x: x['Salary'].mean() > 60000)
print("\nHigh-salary departments:")
print(high_salary_depts)
```

### Apply Operations

```python
# Apply: Custom function for salary statistics
def salary_stats(group):
    return pd.Series({
        'count': len(group),
        'mean': group['Salary'].mean(),
        'std': group['Salary'].std(),
        'range': group['Salary'].max() - group['Salary'].min()
    })

print("=== Apply Operations ===")
print("Custom statistics by department:")
print(df.groupby('Department').apply(salary_stats))

# Apply: Get top earners in each department
top_earners = df.groupby('Department').apply(lambda x: x.nlargest(1, 'Salary'))
print("\nTop earners per department:")
print(top_earners)
```

## Part 3: Hierarchical Grouping

### Multi-level Grouping

```python
# Create hierarchical data
hierarchical_data = {
    'Region': ['North', 'North', 'South', 'South', 'North', 'South'],
    'Department': ['Sales', 'Engineering', 'Sales', 'Engineering', 'Marketing', 'Marketing'],
    'Revenue': [100000, 150000, 120000, 180000, 80000, 90000],
    'Employees': [5, 8, 6, 10, 4, 5]
}

hierarchical_df = pd.DataFrame(hierarchical_data)
print("=== Hierarchical Grouping ===")
print("Original data:")
print(hierarchical_df)

# Hierarchical grouping
hierarchical_grouped = hierarchical_df.groupby(['Region', 'Department']).sum()
print("\nHierarchical grouping:")
print(hierarchical_grouped)

# Unstack to wide format
wide_format = hierarchical_grouped.unstack()
print("\nWide format:")
print(wide_format)
```

### MultiIndex Operations

```python
# Work with MultiIndex
print("=== MultiIndex Operations ===")
print("Index levels:", hierarchical_grouped.index.names)
print("Index values:", hierarchical_grouped.index.values)

# Access specific groups
print("\nNorth region data:")
print(hierarchical_grouped.loc['North'])

# Reset index to flatten
flattened = hierarchical_grouped.reset_index()
print("\nFlattened data:")
print(flattened)
```

## Part 4: Real-world Example

### Sales Analysis

```python
# Create more realistic sales data
np.random.seed(42)
n_sales = 1000

sales_data = {
    'Date': pd.date_range('2023-01-01', periods=n_sales, freq='D'),
    'Product': np.random.choice(['A', 'B', 'C', 'D'], n_sales),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], n_sales),
    'Salesperson': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana'], n_sales),
    'Quantity': np.random.randint(1, 10, n_sales),
    'Unit_Price': np.random.uniform(10, 100, n_sales)
}

sales_df = pd.DataFrame(sales_data)
sales_df['Total_Sales'] = sales_df['Quantity'] * sales_df['Unit_Price']

print("=== Sales Analysis ===")
print("Sample sales data:")
print(sales_df.head())

# Monthly sales by region
monthly_sales = sales_df.groupby([sales_df['Date'].dt.to_period('M'), 'Region'])['Total_Sales'].sum()
print("\nMonthly sales by region:")
print(monthly_sales.head(10))

# Top salesperson by region
top_salesperson = sales_df.groupby('Region').apply(
    lambda x: x.groupby('Salesperson')['Total_Sales'].sum().idxmax()
)
print("\nTop salesperson by region:")
print(top_salesperson)
```

### Performance Analysis

```python
# Performance comparison
print("=== Performance Analysis ===")

# Method 1: Multiple groupby operations
start_time = pd.Timestamp.now()
result1 = sales_df.groupby('Region')['Total_Sales'].sum()
result2 = sales_df.groupby('Region')['Quantity'].sum()
end_time = pd.Timestamp.now()
method1_time = (end_time - start_time).total_seconds()

# Method 2: Single groupby with multiple aggregations
start_time = pd.Timestamp.now()
result3 = sales_df.groupby('Region').agg({
    'Total_Sales': 'sum',
    'Quantity': 'sum'
})
end_time = pd.Timestamp.now()
method2_time = (end_time - start_time).total_seconds()

print(f"Method 1 (multiple groupby): {method1_time:.6f} seconds")
print(f"Method 2 (single groupby): {method2_time:.6f} seconds")
print(f"Performance improvement: {method1_time/method2_time:.2f}x")
```

## Key Takeaways

1. **Split-Apply-Combine**: The fundamental pattern of data aggregation
2. **Aggregation Functions**: Use mean, sum, count, and custom functions
3. **Transform Operations**: Add group statistics to original data
4. **Filter Operations**: Remove groups based on conditions
5. **Apply Operations**: Use custom functions on groups
6. **Hierarchical Grouping**: Work with multi-level group structures
7. **Performance**: Single groupby with multiple aggregations is more efficient

## Next Steps

- Practice with your own datasets
- Experiment with different aggregation functions
- Learn about pivot tables for multi-dimensional analysis
- Explore remote computing for large datasets
