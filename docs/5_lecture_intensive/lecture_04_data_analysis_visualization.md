# Lecture 04: Data Analysis & Visualization

*Advanced Data Manipulation and Effective Visual Communication*

## Learning Objectives

By the end of this lecture, you will be able to:
- Perform advanced data manipulation using pandas operations
- Create comprehensive data analysis workflows
- Design effective visualizations using matplotlib, seaborn, and pandas
- Apply visualization design principles for clear communication
- Build complete data analysis pipelines from raw data to insights

## Introduction: From Data to Insights

Data analysis is the art of transforming raw data into actionable insights. Today we'll combine advanced pandas operations with powerful visualization techniques to tell compelling stories with data. We'll learn not just how to create charts, but how to design visualizations that communicate effectively and drive decision-making.

The key to successful data analysis lies in understanding both the technical capabilities of your tools and the principles of clear communication. We'll explore advanced pandas operations that enable complex analyses, then master visualization techniques that make your findings accessible and impactful.

## Part 1: Advanced Data Manipulation with Pandas

### Grouping and Aggregation

**Basic Grouping Operations:**
```python
import pandas as pd
import numpy as np

# Sample sales data
sales_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100, freq='D'),
    'product': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'sales': np.random.randint(100, 1000, 100),
    'units': np.random.randint(1, 20, 100),
    'cost': np.random.randint(50, 500, 100)
})

# Basic grouping
by_product = sales_data.groupby('product')['sales'].sum()
print(by_product)

# Multiple grouping columns
by_product_region = sales_data.groupby(['product', 'region'])['sales'].sum()
print(by_product_region)

# Multiple aggregation functions
summary_stats = sales_data.groupby('product').agg({
    'sales': ['sum', 'mean', 'count'],
    'units': 'sum',
    'cost': ['min', 'max']
})
print(summary_stats)
```

**Advanced Grouping Patterns:**
```python
# Custom aggregation functions
def sales_range(series):
    return series.max() - series.min()

def top_quartile_mean(series):
    return series.quantile(0.75)

custom_aggs = sales_data.groupby('product').agg({
    'sales': [sales_range, top_quartile_mean, 'std']
})

# Named aggregations (pandas 0.25+)
named_agg = sales_data.groupby('region').agg(
    total_sales=('sales', 'sum'),
    avg_sales=('sales', 'mean'),
    total_units=('units', 'sum'),
    sales_std=('sales', 'std')
)

# Apply custom functions
def analyze_region(group):
    """Custom function for detailed regional analysis"""
    return pd.Series({
        'total_sales': group['sales'].sum(),
        'avg_daily_sales': group['sales'].mean(),
        'best_product': group.loc[group['sales'].idxmax(), 'product'],
        'sales_consistency': group['sales'].std() / group['sales'].mean()
    })

region_analysis = sales_data.groupby('region').apply(analyze_region)
```

**Transform and Filter Operations:**
```python
# Transform: add group statistics to original data
sales_data['region_avg'] = sales_data.groupby('region')['sales'].transform('mean')
sales_data['sales_vs_region_avg'] = sales_data['sales'] - sales_data['region_avg']

# Filter: keep only groups meeting criteria
high_volume_regions = sales_data.groupby('region').filter(lambda x: x['sales'].sum() > 10000)

# Rolling operations within groups
sales_data = sales_data.sort_values(['region', 'date'])
sales_data['7day_avg'] = sales_data.groupby('region')['sales'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
```

### Pivoting and Reshaping Data

**Pivot Tables:**
```python
# Create pivot table
pivot_table = pd.pivot_table(
    sales_data,
    values='sales',
    index='product',
    columns='region',
    aggfunc='sum',
    fill_value=0,
    margins=True,  # Add totals
    margins_name='Total'
)

# Multiple value columns
multi_pivot = pd.pivot_table(
    sales_data,
    values=['sales', 'units'],
    index='product',
    columns='region',
    aggfunc={'sales': 'sum', 'units': 'sum'}
)

# Cross-tabulation
cross_tab = pd.crosstab(
    sales_data['product'],
    sales_data['region'],
    values=sales_data['sales'],
    aggfunc='sum',
    normalize='index'  # Show percentages by row
)
```

**Melt and Stack Operations:**
```python
# Wide to long format
wide_data = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'Q1_sales': [100, 150, 120],
    'Q2_sales': [110, 140, 130],
    'Q3_sales': [105, 160, 125],
    'Q4_sales': [120, 155, 135]
})

long_data = pd.melt(
    wide_data,
    id_vars=['product'],
    value_vars=['Q1_sales', 'Q2_sales', 'Q3_sales', 'Q4_sales'],
    var_name='quarter',
    value_name='sales'
)

# Clean up the quarter column
long_data['quarter'] = long_data['quarter'].str.replace('_sales', '')

# Stack and unstack operations
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['X', 'Y', 'Z'])

stacked = df.stack()              # Convert columns to rows
unstacked = stacked.unstack()     # Convert back to columns
```

### Time Series Analysis

**Date/Time Operations:**
```python
# Create time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
time_series_data = pd.DataFrame({
    'date': dates,
    'sales': np.random.normal(1000, 200, 365) + 100 * np.sin(np.arange(365) * 2 * np.pi / 365)
})
time_series_data.set_index('date', inplace=True)

# Resampling operations
daily_to_weekly = time_series_data.resample('W').sum()      # Weekly totals
daily_to_monthly = time_series_data.resample('M').mean()   # Monthly averages
quarterly = time_series_data.resample('Q').agg(['sum', 'mean', 'std'])

# Rolling operations
time_series_data['7day_ma'] = time_series_data['sales'].rolling(7).mean()
time_series_data['30day_ma'] = time_series_data['sales'].rolling(30).mean()

# Seasonal decomposition patterns
time_series_data['day_of_week'] = time_series_data.index.dayofweek
time_series_data['month'] = time_series_data.index.month
time_series_data['quarter'] = time_series_data.index.quarter

# Seasonal analysis
seasonal_pattern = time_series_data.groupby('day_of_week')['sales'].mean()
monthly_pattern = time_series_data.groupby('month')['sales'].mean()
```

**Advanced Time Series Operations:**
```python
# Lag and lead operations
time_series_data['sales_lag1'] = time_series_data['sales'].shift(1)
time_series_data['sales_lead1'] = time_series_data['sales'].shift(-1)

# Calculate changes and growth rates
time_series_data['daily_change'] = time_series_data['sales'].diff()
time_series_data['pct_change'] = time_series_data['sales'].pct_change()

# Window functions for complex calculations
def rolling_correlation(series1, series2, window):
    """Calculate rolling correlation between two series"""
    return series1.rolling(window).corr(series2)

# Example: correlation with lagged version
time_series_data['autocorr'] = rolling_correlation(
    time_series_data['sales'], 
    time_series_data['sales_lag1'], 
    30
)
```

### Merging and Joining Data

**DataFrame Joining:**
```python
# Sample datasets
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'region': ['North', 'South', 'East', 'West', 'North']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105, 106],
    'customer_id': [1, 2, 2, 3, 4, 6],  # Note: customer 6 doesn't exist in customers
    'amount': [500, 300, 200, 800, 150, 400],
    'date': pd.date_range('2023-01-01', periods=6, freq='D')
})

# Different join types
inner_join = pd.merge(customers, orders, on='customer_id', how='inner')
left_join = pd.merge(customers, orders, on='customer_id', how='left')
right_join = pd.merge(customers, orders, on='customer_id', how='right')
outer_join = pd.merge(customers, orders, on='customer_id', how='outer')

# Join on multiple columns
sales_detail = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'product_id': ['A', 'B', 'C'],
    'region': ['North', 'South', 'East'],
    'discount': [0.1, 0.05, 0.15]
})

complex_join = pd.merge(
    customers, 
    sales_detail, 
    on=['customer_id', 'region'], 
    how='left'
)
```

**Advanced Joining Patterns:**
```python
# Joining on index
customers_indexed = customers.set_index('customer_id')
orders_indexed = orders.set_index('customer_id')
index_join = customers_indexed.join(orders_indexed, how='left')

# Suffix handling for overlapping columns
result = pd.merge(customers, orders, on='customer_id', suffixes=('_customer', '_order'))

# Concatenation
q1_sales = pd.DataFrame({'month': [1, 2, 3], 'sales': [100, 120, 110]})
q2_sales = pd.DataFrame({'month': [4, 5, 6], 'sales': [130, 125, 140]})

# Vertical concatenation
yearly_sales = pd.concat([q1_sales, q2_sales], ignore_index=True)

# Horizontal concatenation
customer_info = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
customer_scores = pd.DataFrame({'score': [85, 92], 'grade': ['B', 'A']})
customer_complete = pd.concat([customer_info, customer_scores], axis=1)
```

### Data Quality and Profiling

**Comprehensive Data Profiling:**
```python
def profile_dataframe(df):
    """
    Generate comprehensive data profile for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to profile
        
    Returns:
        dict: Comprehensive data profile
    """
    profile = {
        'basic_info': {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': df.duplicated().sum()
        },
        'columns': {}
    }
    
    for column in df.columns:
        col_profile = {
            'dtype': str(df[column].dtype),
            'non_null_count': df[column].notna().sum(),
            'null_count': df[column].isna().sum(),
            'null_percentage': (df[column].isna().sum() / len(df)) * 100,
            'unique_count': df[column].nunique(),
            'unique_percentage': (df[column].nunique() / len(df)) * 100
        }
        
        # Numeric column statistics
        if df[column].dtype in ['int64', 'float64']:
            col_profile.update({
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'q25': df[column].quantile(0.25),
                'q75': df[column].quantile(0.75),
                'outliers_iqr': detect_outliers_iqr(df[column]).sum()
            })
        
        # Categorical column statistics
        elif df[column].dtype == 'object':
            value_counts = df[column].value_counts()
            col_profile.update({
                'most_frequent': value_counts.index[0] if not value_counts.empty else None,
                'most_frequent_count': value_counts.iloc[0] if not value_counts.empty else 0,
                'avg_length': df[column].astype(str).str.len().mean()
            })
        
        profile['columns'][column] = col_profile
    
    return profile

def detect_outliers_iqr(series):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# Example usage
sample_data = pd.DataFrame({
    'id': range(1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'value': np.random.normal(100, 20, 1000),
    'text': ['sample_text_' + str(i) for i in range(1000)]
})

# Add some missing values and outliers
sample_data.loc[sample_data.index[:50], 'value'] = np.nan
sample_data.loc[sample_data.index[900:], 'value'] = np.random.normal(500, 50, 100)  # Outliers

profile = profile_dataframe(sample_data)

# Display profile results
for column, stats in profile['columns'].items():
    print(f"\nColumn: {column}")
    for stat, value in stats.items():
        if isinstance(value, float):
            print(f"  {stat}: {value:.2f}")
        else:
            print(f"  {stat}: {value}")
```

## Part 2: Data Visualization Fundamentals

### Matplotlib: The Foundation

**Basic Plot Types:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Line plot
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots()
ax.plot(x, y1, label='sin(x)', linewidth=2)
ax.plot(x, y2, label='cos(x)', linewidth=2, linestyle='--')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('Trigonometric Functions')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Scatter plot with customization
np.random.seed(42)
x = np.random.randn(200)
y = 2 * x + np.random.randn(200)
colors = np.random.rand(200)
sizes = 1000 * np.random.rand(200)

fig, ax = plt.subplots()
scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('Scatter Plot with Color and Size Mapping')
plt.colorbar(scatter, ax=ax, label='Color Scale')
plt.tight_layout()
plt.show()
```

**Subplots and Complex Layouts:**
```python
# Create sample data
time = np.arange(0, 24, 0.1)
temperature = 20 + 10 * np.sin((time - 6) * np.pi / 12) + np.random.normal(0, 1, len(time))
humidity = 50 + 20 * np.sin((time - 8) * np.pi / 12) + np.random.normal(0, 3, len(time))
pressure = 1013 + 5 * np.sin(time * np.pi / 12) + np.random.normal(0, 2, len(time))

# Multiple subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Temperature plot
axes[0].plot(time, temperature, 'r-', linewidth=2, alpha=0.8)
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Weather Data Over 24 Hours')
axes[0].grid(True, alpha=0.3)

# Humidity plot
axes[1].plot(time, humidity, 'b-', linewidth=2, alpha=0.8)
axes[1].set_ylabel('Humidity (%)')
axes[1].grid(True, alpha=0.3)

# Pressure plot
axes[2].plot(time, pressure, 'g-', linewidth=2, alpha=0.8)
axes[2].set_ylabel('Pressure (hPa)')
axes[2].set_xlabel('Hour of Day')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Complex subplot layout
fig = plt.figure(figsize=(15, 10))

# Main plot
ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
ax_main.scatter(x, y, alpha=0.6)
ax_main.set_title('Main Scatter Plot')

# Top histogram
ax_top = plt.subplot2grid((3, 3), (0, 2))
ax_top.hist(x, bins=20, alpha=0.7, orientation='vertical')
ax_top.set_title('X Distribution')

# Side histogram
ax_side = plt.subplot2grid((3, 3), (1, 2))
ax_side.hist(y, bins=20, alpha=0.7, orientation='horizontal')
ax_side.set_title('Y Distribution')

# Bottom summary stats
ax_stats = plt.subplot2grid((3, 3), (2, 0), colspan=3)
ax_stats.text(0.1, 0.8, f'X: mean={np.mean(x):.2f}, std={np.std(x):.2f}', transform=ax_stats.transAxes)
ax_stats.text(0.1, 0.4, f'Y: mean={np.mean(y):.2f}, std={np.std(y):.2f}', transform=ax_stats.transAxes)
ax_stats.text(0.1, 0.0, f'Correlation: {np.corrcoef(x, y)[0,1]:.2f}', transform=ax_stats.transAxes)
ax_stats.set_title('Summary Statistics')
ax_stats.axis('off')

plt.tight_layout()
plt.show()
```

### Seaborn: Statistical Visualization

**Distribution Plots:**
```python
# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'group': np.repeat(['A', 'B', 'C'], 100),
    'value': np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(2, 1.5, 100),
        np.random.normal(-1, 0.8, 100)
    ])
})

# Histogram with KDE
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Histogram
sns.histplot(data=data, x='value', hue='group', kde=True, ax=axes[0,0])
axes[0,0].set_title('Histogram with KDE')

# Box plot
sns.boxplot(data=data, x='group', y='value', ax=axes[0,1])
axes[0,1].set_title('Box Plot by Group')

# Violin plot
sns.violinplot(data=data, x='group', y='value', ax=axes[1,0])
axes[1,0].set_title('Violin Plot by Group')

# Strip plot with jitter
sns.stripplot(data=data, x='group', y='value', size=4, alpha=0.7, ax=axes[1,1])
axes[1,1].set_title('Strip Plot with Jitter')

plt.tight_layout()
plt.show()
```

**Relationship Plots:**
```python
# Create sample dataset
np.random.seed(42)
n = 200
dataset = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n),
    'category': np.random.choice(['Type A', 'Type B', 'Type C'], n),
    'size_var': np.random.randint(20, 200, n)
})
dataset['z'] = dataset['x'] * 0.5 + dataset['y'] * 0.3 + np.random.randn(n) * 0.2

# Scatter plot with regression line
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.scatterplot(data=dataset, x='x', y='z', hue='category', size='size_var', 
                alpha=0.7, ax=axes[0,0])
axes[0,0].set_title('Scatter Plot with Hue and Size')

# Regression plot
sns.regplot(data=dataset, x='x', y='z', ax=axes[0,1], scatter_kws={'alpha':0.6})
axes[0,1].set_title('Regression Plot')

# Joint plot (using separate figure)
g = sns.JointGrid(data=dataset, x='x', y='z', space=0)
g.plot_joint(sns.scatterplot, alpha=0.6)
g.plot_marginals(sns.histplot, kde=True)
plt.suptitle('Joint Distribution Plot')
plt.show()

# Pair plot for multiple variables
numeric_data = dataset[['x', 'y', 'z', 'category']]
g = sns.pairplot(numeric_data, hue='category', diag_kind='kde')
g.fig.suptitle('Pair Plot Matrix', y=1.02)
plt.show()
```

**Categorical Plots:**
```python
# Create sample sales data
sales_data = pd.DataFrame({
    'month': np.tile(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 20),
    'region': np.repeat(['North', 'South', 'East', 'West'], 30),
    'sales': np.random.normal(100, 20, 120) + np.random.normal(0, 10, 120),
    'product': np.random.choice(['Product A', 'Product B', 'Product C'], 120)
})

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Bar plot
sns.barplot(data=sales_data, x='month', y='sales', hue='region', ax=axes[0,0])
axes[0,0].set_title('Average Sales by Month and Region')
axes[0,0].tick_params(axis='x', rotation=45)

# Count plot
sns.countplot(data=sales_data, x='product', hue='region', ax=axes[0,1])
axes[0,1].set_title('Count by Product and Region')

# Heatmap of pivot table
pivot_data = sales_data.pivot_table(values='sales', index='region', columns='month', aggfunc='mean')
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1,0])
axes[1,0].set_title('Sales Heatmap: Region vs Month')

# FacetGrid
g = sns.FacetGrid(sales_data, col='region', row='product', margin_titles=True, height=3)
g.map(plt.hist, 'sales', alpha=0.7, bins=15)
g.add_legend()
plt.suptitle('Distribution of Sales by Region and Product', y=1.02)
plt.show()
```

### Pandas Built-in Visualization

**Quick Plotting with Pandas:**
```python
# Sample time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'sales': 1000 + 200 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 50, 365),
    'marketing_spend': 500 + 100 * np.sin(np.arange(365) * 2 * np.pi / 365 + np.pi/4) + np.random.normal(0, 25, 365),
    'temperature': 15 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 3, 365)
})
ts_data.set_index('date', inplace=True)

# Line plot
ax = ts_data[['sales', 'marketing_spend']].plot(kind='line', figsize=(12, 6), 
                                                title='Sales and Marketing Spend Over Time')
ax.set_ylabel('Amount ($)')
plt.show()

# Multiple plot types
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Line plot
ts_data['sales'].plot(ax=axes[0,0], title='Sales Trend')

# Histogram
ts_data['sales'].plot(kind='hist', bins=30, ax=axes[0,1], title='Sales Distribution')

# Box plot by quarter
ts_data['quarter'] = ts_data.index.quarter
ts_data.boxplot(column='sales', by='quarter', ax=axes[1,0])
axes[1,0].set_title('Sales by Quarter')

# Scatter plot
ts_data.plot(kind='scatter', x='marketing_spend', y='sales', ax=axes[1,1], 
             title='Sales vs Marketing Spend', alpha=0.6)

plt.tight_layout()
plt.show()
```

## Part 3: Design Principles for Effective Visualization

### Understanding Your Audience and Purpose

**The Visualization Purpose Framework:**
```python
def choose_visualization(data_type, purpose, audience):
    """
    Guide for choosing appropriate visualization types.
    
    Args:
        data_type: 'categorical', 'numerical', 'time_series', 'relationship'
        purpose: 'compare', 'trend', 'distribution', 'composition', 'relationship'
        audience: 'technical', 'business', 'general_public'
        
    Returns:
        list: Recommended visualization types
    """
    recommendations = {
        ('categorical', 'compare', 'business'): ['bar_chart', 'horizontal_bar'],
        ('categorical', 'composition', 'business'): ['pie_chart', 'stacked_bar'],
        ('numerical', 'distribution', 'technical'): ['histogram', 'box_plot', 'violin_plot'],
        ('numerical', 'distribution', 'business'): ['histogram', 'summary_table'],
        ('time_series', 'trend', 'all'): ['line_chart', 'area_chart'],
        ('relationship', 'correlation', 'technical'): ['scatter_plot', 'correlation_matrix'],
        ('relationship', 'correlation', 'business'): ['scatter_plot', 'trend_line']
    }
    
    key = (data_type, purpose, audience)
    return recommendations.get(key, ['explore_multiple_options'])

# Example usage
print("For comparing categories to business audience:", 
      choose_visualization('categorical', 'compare', 'business'))
```

### Color and Design Best Practices

**Color Palettes and Accessibility:**
```python
# Color-blind friendly palettes
def create_accessible_plots():
    # Sample data
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    
    # Color-blind friendly palette
    colorblind_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Good: Color-blind friendly
    axes[0].bar(categories, values, color=colorblind_palette)
    axes[0].set_title('Good: Colorblind Friendly Palette')
    axes[0].set_ylabel('Values')
    
    # Bad: Red-green combination
    bad_colors = ['red', 'green', 'red', 'green', 'red']
    axes[1].bar(categories, values, color=bad_colors)
    axes[1].set_title('Bad: Red-Green Combination')
    axes[1].set_ylabel('Values')
    
    # Better: Using patterns for accessibility
    patterns = ['', '///', '...', '+++', 'xxx']
    bars = axes[2].bar(categories, values, color=colorblind_palette)
    for bar, pattern in zip(bars, patterns):
        bar.set_hatch(pattern)
    axes[2].set_title('Best: Colors + Patterns')
    axes[2].set_ylabel('Values')
    
    plt.tight_layout()
    plt.show()

create_accessible_plots()
```

**Minimizing Chart Junk:**
```python
def demonstrate_chart_junk():
    # Sample data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    sales = [100, 120, 140, 110, 160, 180]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bad: Chart junk
    axes[0].bar(months, sales, color=['red', 'blue', 'green', 'yellow', 'purple', 'orange'])
    axes[0].set_title('BAD: Too Much Visual Noise', fontsize=16, color='red', 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
    axes[0].set_ylabel('Sales ($000)', fontsize=14, color='blue')
    axes[0].grid(True, linestyle='--', linewidth=2, color='red', alpha=0.8)
    axes[0].spines['top'].set_linewidth(3)
    axes[0].spines['right'].set_linewidth(3)
    axes[0].spines['bottom'].set_linewidth(3)
    axes[0].spines['left'].set_linewidth(3)
    
    # Good: Clean design
    axes[1].bar(months, sales, color='steelblue', alpha=0.8)
    axes[1].set_title('GOOD: Clean and Clear', fontsize=14)
    axes[1].set_ylabel('Sales ($000)')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

demonstrate_chart_junk()
```

### Effective Data Storytelling

**Creating Narrative with Data:**
```python
def create_data_story():
    """Example of effective data storytelling"""
    
    # Sample sales data with a story
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sales_2022 = [100, 105, 110, 108, 95, 85, 90, 95, 120, 135, 150, 180]
    sales_2023 = [120, 125, 130, 140, 145, 150, 160, 165, 170, 175, 180, 200]
    
    # Calculate key metrics
    total_2022 = sum(sales_2022)
    total_2023 = sum(sales_2023)
    growth = ((total_2023 - total_2022) / total_2022) * 100
    
    # Create the story
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Main trend
    axes[0,0].plot(months, sales_2022, 'o-', label='2022', linewidth=3, markersize=8)
    axes[0,0].plot(months, sales_2023, 'o-', label='2023', linewidth=3, markersize=8)
    axes[0,0].set_title('Sales Recovery Story: From Summer Slump to Holiday Success', 
                        fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Sales ($000)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Annotate key events
    axes[0,0].annotate('Summer Slump 2022', xy=(5, 85), xytext=(3, 70),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red')
    axes[0,0].annotate('Recovery Begins', xy=(8, 95), xytext=(6, 60),
                       arrowprops=dict(arrowstyle='->', color='green'),
                       fontsize=10, color='green')
    
    # Growth comparison
    categories = ['Total 2022', 'Total 2023']
    totals = [total_2022, total_2023]
    colors = ['lightcoral', 'lightgreen']
    
    bars = axes[0,1].bar(categories, totals, color=colors)
    axes[0,1].set_title(f'Annual Performance: +{growth:.1f}% Growth')
    axes[0,1].set_ylabel('Total Sales ($000)')
    
    # Add value labels on bars
    for bar, value in zip(bars, totals):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 10,
                       f'${value:,.0f}K', ha='center', va='bottom', fontweight='bold')
    
    # Month-over-month growth
    growth_2023 = [(sales_2023[i] - sales_2022[i]) / sales_2022[i] * 100 
                   for i in range(len(months))]
    
    colors = ['green' if g > 0 else 'red' for g in growth_2023]
    axes[1,0].bar(months, growth_2023, color=colors, alpha=0.7)
    axes[1,0].set_title('Month-over-Month Growth (2023 vs 2022)')
    axes[1,0].set_ylabel('Growth (%)')
    axes[1,0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Key insights
    axes[1,1].text(0.05, 0.9, 'KEY INSIGHTS', fontsize=16, fontweight='bold',
                   transform=axes[1,1].transAxes)
    
    insights = [
        f'• Overall growth: +{growth:.1f}%',
        '• Consistent month-over-month improvement',
        '• No summer slump in 2023',
        '• Holiday season exceeded expectations',
        '• Strategy: Maintain momentum in Q1 2024'
    ]
    
    for i, insight in enumerate(insights):
        axes[1,1].text(0.05, 0.7 - i*0.12, insight, fontsize=12,
                       transform=axes[1,1].transAxes)
    
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()

create_data_story()
```

## Part 4: Complete Analysis Workflow

### End-to-End Analysis Example

```python
def complete_analysis_workflow():
    """
    Demonstrate complete analysis workflow from data loading to insights.
    """
    
    # Step 1: Generate realistic sample data
    np.random.seed(42)
    
    # Simulate e-commerce data
    n_customers = 1000
    n_orders = 5000
    
    # Customer data
    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'signup_date': pd.date_range('2022-01-01', periods=n_customers, freq='D')[:n_customers],
        'age': np.random.normal(35, 12, n_customers).clip(18, 80),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_customers, p=[0.5, 0.3, 0.2]),
        'acquisition_channel': np.random.choice(['Organic', 'Paid', 'Social', 'Email'], n_customers, p=[0.4, 0.3, 0.2, 0.1])
    })
    
    # Order data
    orders = pd.DataFrame({
        'order_id': range(1, n_orders + 1),
        'customer_id': np.random.choice(customers['customer_id'], n_orders),
        'order_date': pd.date_range('2022-01-01', '2023-12-31', periods=n_orders),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_orders),
        'order_value': np.random.lognormal(4, 0.8, n_orders),
        'discount_applied': np.random.choice([0, 5, 10, 15, 20], n_orders, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    })
    
    # Step 2: Data cleaning and preparation
    print("=== DATA QUALITY CHECK ===")
    
    # Check for missing values
    print("Missing values in customers:", customers.isnull().sum().sum())
    print("Missing values in orders:", orders.isnull().sum().sum())
    
    # Data type conversions
    customers['age'] = customers['age'].round().astype(int)
    orders['order_value'] = orders['order_value'].round(2)
    
    # Create age groups
    customers['age_group'] = pd.cut(customers['age'], 
                                   bins=[0, 25, 35, 50, 100], 
                                   labels=['18-25', '26-35', '36-50', '50+'])
    
    # Add time-based features
    orders['order_month'] = orders['order_date'].dt.month
    orders['order_year'] = orders['order_date'].dt.year
    orders['order_quarter'] = orders['order_date'].dt.quarter
    
    # Step 3: Join datasets
    customer_orders = pd.merge(orders, customers, on='customer_id', how='left')
    
    # Step 4: Exploratory Data Analysis
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    
    # Customer analysis
    customer_summary = customer_orders.groupby('customer_id').agg({
        'order_value': ['count', 'sum', 'mean'],
        'order_date': ['min', 'max']
    }).round(2)
    
    customer_summary.columns = ['order_count', 'total_spent', 'avg_order_value', 'first_order', 'last_order']
    customer_summary = customer_summary.merge(customers[['customer_id', 'age_group', 'location', 'acquisition_channel']], 
                                            on='customer_id')
    
    print("Top customers by total spent:")
    print(customer_summary.nlargest(5, 'total_spent')[['total_spent', 'order_count', 'avg_order_value']])
    
    # Step 5: Create comprehensive visualization dashboard
    fig = plt.figure(figsize=(20, 16))
    
    # Sales trend over time
    ax1 = plt.subplot(3, 3, 1)
    monthly_sales = customer_orders.groupby([customer_orders['order_date'].dt.to_period('M')])['order_value'].sum()
    monthly_sales.plot(kind='line', ax=ax1, color='steelblue', linewidth=3)
    ax1.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sales ($)')
    ax1.grid(True, alpha=0.3)
    
    # Category performance
    ax2 = plt.subplot(3, 3, 2)
    category_sales = customer_orders.groupby('product_category')['order_value'].sum().sort_values(ascending=True)
    category_sales.plot(kind='barh', ax=ax2, color='lightcoral')
    ax2.set_title('Sales by Product Category', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Total Sales ($)')
    
    # Customer age distribution
    ax3 = plt.subplot(3, 3, 3)
    customer_orders['age'].hist(bins=20, ax=ax3, alpha=0.7, color='lightgreen')
    ax3.set_title('Customer Age Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Frequency')
    
    # Average order value by age group
    ax4 = plt.subplot(3, 3, 4)
    avg_by_age = customer_orders.groupby('age_group')['order_value'].mean()
    avg_by_age.plot(kind='bar', ax=ax4, color='orange', alpha=0.8)
    ax4.set_title('Avg Order Value by Age Group', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Average Order Value ($)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Location analysis
    ax5 = plt.subplot(3, 3, 5)
    location_orders = customer_orders.groupby('location')['order_value'].agg(['count', 'sum'])
    location_orders['avg'] = location_orders['sum'] / location_orders['count']
    location_orders['avg'].plot(kind='bar', ax=ax5, color='purple', alpha=0.8)
    ax5.set_title('Avg Order Value by Location', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Average Order Value ($)')
    ax5.tick_params(axis='x', rotation=45)
    
    # Acquisition channel effectiveness
    ax6 = plt.subplot(3, 3, 6)
    channel_summary = customer_summary.groupby('acquisition_channel').agg({
        'total_spent': 'mean',
        'order_count': 'mean'
    })
    
    x = range(len(channel_summary.index))
    width = 0.35
    ax6.bar([i - width/2 for i in x], channel_summary['total_spent'], width, 
            label='Avg Total Spent', alpha=0.8)
    ax6_twin = ax6.twinx()
    ax6_twin.bar([i + width/2 for i in x], channel_summary['order_count'], width, 
                 label='Avg Order Count', alpha=0.8, color='orange')
    
    ax6.set_title('Customer Value by Acquisition Channel', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(channel_summary.index, rotation=45)
    ax6.set_ylabel('Average Total Spent ($)')
    ax6_twin.set_ylabel('Average Order Count')
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    
    # Seasonal patterns
    ax7 = plt.subplot(3, 3, 7)
    seasonal_pattern = customer_orders.groupby('order_month')['order_value'].mean()
    seasonal_pattern.plot(kind='line', ax=ax7, marker='o', linewidth=3, markersize=8)
    ax7.set_title('Seasonal Order Value Pattern', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Month')
    ax7.set_ylabel('Average Order Value ($)')
    ax7.set_xticks(range(1, 13))
    ax7.grid(True, alpha=0.3)
    
    # Customer lifetime value distribution
    ax8 = plt.subplot(3, 3, 8)
    customer_summary['total_spent'].hist(bins=30, ax=ax8, alpha=0.7, color='teal')
    ax8.set_title('Customer Lifetime Value Distribution', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Total Spent ($)')
    ax8.set_ylabel('Number of Customers')
    
    # Key metrics summary
    ax9 = plt.subplot(3, 3, 9)
    
    # Calculate key metrics
    total_revenue = customer_orders['order_value'].sum()
    avg_order_value = customer_orders['order_value'].mean()
    total_customers = customer_orders['customer_id'].nunique()
    avg_customer_value = customer_summary['total_spent'].mean()
    repeat_customer_rate = (customer_summary['order_count'] > 1).mean() * 100
    
    metrics_text = f"""
    KEY METRICS
    
    Total Revenue: ${total_revenue:,.0f}
    
    Average Order Value: ${avg_order_value:.2f}
    
    Total Customers: {total_customers:,}
    
    Avg Customer LTV: ${avg_customer_value:.2f}
    
    Repeat Purchase Rate: {repeat_customer_rate:.1f}%
    
    Top Category: {category_sales.index[-1]}
    
    Best Channel: {channel_summary['total_spent'].idxmax()}
    """
    
    ax9.text(0.05, 0.95, metrics_text, transform=ax9.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    plt.suptitle('E-Commerce Business Dashboard - Comprehensive Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    # Step 6: Generate insights and recommendations
    print("\n=== BUSINESS INSIGHTS & RECOMMENDATIONS ===")
    
    insights = []
    
    # Revenue trend analysis
    if monthly_sales.iloc[-1] > monthly_sales.iloc[0]:
        insights.append("✓ Positive revenue growth trend observed")
    else:
        insights.append("⚠ Revenue decline needs attention")
    
    # Customer segmentation insights
    high_value_customers = (customer_summary['total_spent'] > customer_summary['total_spent'].quantile(0.8)).sum()
    insights.append(f"• {high_value_customers} high-value customers (top 20%) drive significant revenue")
    
    # Channel performance
    best_channel = channel_summary['total_spent'].idxmax()
    insights.append(f"• {best_channel} acquisition channel shows highest customer value")
    
    # Seasonal patterns
    peak_month = seasonal_pattern.idxmax()
    insights.append(f"• Month {peak_month} shows peak order values - plan inventory accordingly")
    
    for insight in insights:
        print(insight)
    
    return customer_orders, customer_summary

# Run the complete analysis
customer_orders, customer_summary = complete_analysis_workflow()
```

## Summary and Next Steps

### What We've Accomplished

Today we've mastered the complete data analysis workflow:

1. **Advanced pandas Operations**: Grouping, pivoting, merging, and time series analysis
2. **Data Quality and Profiling**: Comprehensive data validation and quality assessment
3. **Visualization Mastery**: Created effective visualizations using matplotlib, seaborn, and pandas
4. **Design Principles**: Applied visual design principles for clear communication
5. **Complete Workflow**: Built end-to-end analysis pipelines from raw data to insights

### Key Takeaways

- **Data First**: Always understand your data quality before analysis
- **Purpose-Driven Visualization**: Choose chart types based on your message and audience
- **Progressive Disclosure**: Start with overview, then provide details
- **Design Matters**: Clean, accessible visualizations communicate better than complex ones
- **Tell a Story**: Use data to build narrative and drive action

### Advanced Techniques Covered

- Complex groupby operations with custom functions
- Multi-level indexing and pivot table creation
- Time series analysis and seasonal decomposition
- Advanced visualization layouts and subplots
- Color theory and accessibility in data visualization
- Professional data profiling and validation

### Practice Exercises

1. **Complete Analysis Project**: Find a dataset of personal interest and create a comprehensive analysis dashboard

2. **Visualization Redesign**: Take poorly designed charts from the web and redesign them using the principles learned

3. **Storytelling Challenge**: Create a data story that guides viewers through insights using annotations and progressive disclosure

4. **Performance Analysis**: Build a data profiling system that can handle large datasets efficiently

### Preparation for Next Lecture

In our final lecture, we'll integrate everything with applied projects and professional best practices. To prepare:

1. Practice the advanced pandas operations and visualization techniques
2. Think about a dataset or problem you'd like to work on for the capstone project  
3. Install additional libraries we'll use: `pip install scikit-learn statsmodels plotly`
4. Review statistical concepts like hypothesis testing and regression

You now have the tools to perform professional-grade data analysis and create compelling visualizations. These skills form the foundation for machine learning, statistical analysis, and data-driven decision making. The combination of technical proficiency and design thinking will set you apart as a data scientist who can not only find insights but communicate them effectively.