# Demo 2: Pivot Tables and Cross-Tabulations

## Learning Objectives
- Create pivot tables for multi-dimensional analysis
- Use cross-tabulations for frequency analysis
- Apply advanced pivot operations
- Handle missing values and totals

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
```

## Part 1: Basic Pivot Tables

### Create Sample Data

```python
# Create sample sales data
data = {
    'Product': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],
    'Region': ['North', 'South', 'North', 'South', 'North', 'South', 'North', 'South', 'North', 'South'],
    'Sales': [1000, 1500, 2000, 1200, 800, 900, 1100, 1300, 700, 1600],
    'Quantity': [10, 15, 20, 12, 8, 9, 11, 13, 7, 16]
}

df = pd.DataFrame(data)
print("Sample Data:")
print(df)
```

### Basic Pivot Table

```python
# Create basic pivot table
print("=== Basic Pivot Table ===")
pivot = pd.pivot_table(df, 
                      values='Sales', 
                      index='Product', 
                      columns='Region', 
                      aggfunc='sum')
print("Sales by Product and Region:")
print(pivot)
```

### Pivot Table with Multiple Aggregations

```python
# Pivot table with multiple aggregations
print("=== Multiple Aggregations ===")
pivot_multi = pd.pivot_table(df,
                            values='Sales',
                            index='Product',
                            columns='Region',
                            aggfunc=['sum', 'mean', 'count'])
print("Multiple aggregations:")
print(pivot_multi)
```

## Part 2: Advanced Pivot Operations

### Pivot Table with Totals

```python
# Pivot table with totals
print("=== Pivot Table with Totals ===")
pivot_totals = pd.pivot_table(df,
                             values='Sales',
                             index='Product',
                             columns='Region',
                             aggfunc='sum',
                             margins=True,
                             margins_name='Total')
print("Sales with totals:")
print(pivot_totals)
```

### Handling Missing Values

```python
# Create data with missing combinations
data_missing = {
    'Product': ['A', 'A', 'B', 'B', 'C'],
    'Region': ['North', 'South', 'North', 'South', 'North'],
    'Sales': [1000, 1500, 2000, 1200, 800]
}

df_missing = pd.DataFrame(data_missing)
print("=== Handling Missing Values ===")
print("Data with missing combinations:")
print(df_missing)

# Pivot with fill_value
pivot_filled = pd.pivot_table(df_missing,
                             values='Sales',
                             index='Product',
                             columns='Region',
                             aggfunc='sum',
                             fill_value=0)
print("\nPivot with filled missing values:")
print(pivot_filled)
```

### Cross-Tabulation

```python
# Create categorical data for cross-tabulation
np.random.seed(42)
n_observations = 500

categorical_data = {
    'Gender': np.random.choice(['M', 'F'], n_observations),
    'Age_Group': np.random.choice(['18-25', '26-35', '36-45', '46-55'], n_observations),
    'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_observations),
    'Income_Level': np.random.choice(['Low', 'Medium', 'High'], n_observations)
}

cat_df = pd.DataFrame(categorical_data)
print("=== Cross-Tabulation ===")
print("Sample categorical data:")
print(cat_df.head())

# Cross-tabulation
crosstab = pd.crosstab(cat_df['Gender'], cat_df['Age_Group'], margins=True)
print("\nGender vs Age Group:")
print(crosstab)

# Multi-dimensional cross-tabulation
crosstab_multi = pd.crosstab([cat_df['Gender'], cat_df['Age_Group']], 
                            cat_df['Education'], 
                            margins=True)
print("\nMulti-dimensional cross-tabulation:")
print(crosstab_multi.head(10))
```

## Part 3: Real-world Analysis

### Sales Performance Analysis

```python
# Create comprehensive sales data
np.random.seed(42)
n_sales = 1000

sales_data = {
    'Date': pd.date_range('2023-01-01', periods=n_sales, freq='D'),
    'Product': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_sales),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], n_sales),
    'Salesperson': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana'], n_sales),
    'Quantity': np.random.randint(1, 10, n_sales),
    'Unit_Price': np.random.uniform(10, 100, n_sales)
}

sales_df = pd.DataFrame(sales_data)
sales_df['Total_Sales'] = sales_df['Quantity'] * sales_df['Unit_Price']
sales_df['Month'] = sales_df['Date'].dt.to_period('M')

print("=== Sales Performance Analysis ===")
print("Sample sales data:")
print(sales_df.head())
```

### Monthly Sales by Product and Region

```python
# Monthly sales pivot table
monthly_pivot = pd.pivot_table(sales_df,
                               values='Total_Sales',
                               index='Product',
                               columns='Month',
                               aggfunc='sum',
                               margins=True)
print("Monthly sales by product:")
print(monthly_pivot)
```

### Salesperson Performance Analysis

```python
# Salesperson performance
salesperson_pivot = pd.pivot_table(sales_df,
                                  values=['Total_Sales', 'Quantity'],
                                  index='Salesperson',
                                  columns='Region',
                                  aggfunc={'Total_Sales': 'sum', 'Quantity': 'sum'},
                                  margins=True)
print("Salesperson performance by region:")
print(salesperson_pivot)
```

### Product Performance by Region

```python
# Product performance by region
product_pivot = pd.pivot_table(sales_df,
                              values='Total_Sales',
                              index='Product',
                              columns='Region',
                              aggfunc=['sum', 'mean', 'count'],
                              margins=True)
print("Product performance by region:")
print(product_pivot)
```

## Part 4: Visualization with Pivot Tables

### Heatmap Visualization

```python
# Create heatmap from pivot table
pivot_heatmap = pd.pivot_table(sales_df,
                               values='Total_Sales',
                               index='Product',
                               columns='Region',
                               aggfunc='sum')

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pivot_heatmap.values, cmap='YlOrRd', aspect='auto')

# Set ticks and labels
ax.set_xticks(range(len(pivot_heatmap.columns)))
ax.set_yticks(range(len(pivot_heatmap.index)))
ax.set_xticklabels(pivot_heatmap.columns)
ax.set_yticklabels(pivot_heatmap.index)

# Add colorbar
plt.colorbar(im, ax=ax, label='Total Sales')

# Add title and labels
plt.title('Sales Heatmap: Product vs Region')
plt.xlabel('Region')
plt.ylabel('Product')

# Add text annotations
for i in range(len(pivot_heatmap.index)):
    for j in range(len(pivot_heatmap.columns)):
        text = ax.text(j, i, f'{pivot_heatmap.iloc[i, j]:.0f}',
                      ha="center", va="center", color="black")

plt.tight_layout()
plt.show()
```

### Bar Chart from Pivot Table

```python
# Create bar chart from pivot table
pivot_bar = pd.pivot_table(sales_df,
                           values='Total_Sales',
                           index='Product',
                           columns='Region',
                           aggfunc='sum')

# Create bar chart
pivot_bar.plot(kind='bar', figsize=(12, 6))
plt.title('Sales by Product and Region')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.legend(title='Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Part 5: Advanced Pivot Operations

### Custom Aggregation Functions

```python
# Custom aggregation function
def sales_range(series):
    return series.max() - series.min()

# Pivot table with custom function
custom_pivot = pd.pivot_table(sales_df,
                              values='Total_Sales',
                              index='Product',
                              columns='Region',
                              aggfunc=sales_range)
print("=== Custom Aggregation ===")
print("Sales range by product and region:")
print(custom_pivot)
```

### Pivot Table with Multiple Values

```python
# Pivot table with multiple value columns
multi_value_pivot = pd.pivot_table(sales_df,
                                  values=['Total_Sales', 'Quantity'],
                                  index='Product',
                                  columns='Region',
                                  aggfunc={'Total_Sales': 'sum', 'Quantity': 'mean'})
print("=== Multiple Value Columns ===")
print("Sales and quantity by product and region:")
print(multi_value_pivot)
```

### Reshaping Pivot Results

```python
# Reshape pivot table results
pivot_reshaped = pd.pivot_table(sales_df,
                                values='Total_Sales',
                                index='Product',
                                columns='Region',
                                aggfunc='sum')

# Stack and unstack operations
stacked = pivot_reshaped.stack()
print("=== Reshaping Operations ===")
print("Stacked pivot table:")
print(stacked.head(10))

# Unstack back to wide format
unstacked = stacked.unstack()
print("\nUnstacked back to wide format:")
print(unstacked)
```

## Key Takeaways

1. **Pivot Tables**: Transform long-format data to wide-format summaries
2. **Multiple Aggregations**: Use different functions for different insights
3. **Totals and Margins**: Add row and column totals for comprehensive analysis
4. **Missing Values**: Handle missing combinations with fill_value
5. **Cross-Tabulation**: Analyze categorical data relationships
6. **Visualization**: Create heatmaps and charts from pivot tables
7. **Custom Functions**: Apply business-specific aggregation logic

## Next Steps

- Practice with your own datasets
- Experiment with different aggregation functions
- Learn about remote computing for large datasets
- Explore performance optimization techniques
