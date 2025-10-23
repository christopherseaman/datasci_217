# Demo 3: pandas Plotting and Modern Tools

## Learning Objectives
- Use pandas plotting for quick data exploration
- Apply visualization best practices
- Explore modern visualization tools
- Choose appropriate chart types

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_style('whitegrid')
```

## Part 1: pandas Plotting for Exploration

### Quick Data Exploration

```python
# Create sample dataset
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = {
    'sales': np.cumsum(np.random.randn(100)) + 100,
    'customers': np.random.randint(50, 200, 100),
    'revenue': np.random.uniform(1000, 5000, 100)
}
df = pd.DataFrame(data, index=dates)

# Quick exploration with pandas
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Line plot
df['sales'].plot(ax=axes[0, 0], title='Sales Over Time')
axes[0, 0].grid(True, alpha=0.3)

# Histogram
df['revenue'].plot(kind='hist', ax=axes[0, 1], title='Revenue Distribution', bins=20)
axes[0, 1].grid(True, alpha=0.3)

# Scatter plot
df.plot(kind='scatter', x='customers', y='revenue', ax=axes[1, 0], title='Customers vs Revenue')
axes[1, 0].grid(True, alpha=0.3)

# Box plot
df[['sales', 'revenue']].plot(kind='box', ax=axes[1, 1], title='Sales and Revenue Distribution')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Multiple Series Plotting

```python
# Create multiple time series
df['sales_ma'] = df['sales'].rolling(window=7).mean()
df['sales_ema'] = df['sales'].ewm(span=7).mean()

# Plot multiple series
fig, ax = plt.subplots(figsize=(12, 6))
df[['sales', 'sales_ma', 'sales_ema']].plot(ax=ax, title='Sales with Moving Averages')
ax.grid(True, alpha=0.3)
ax.legend(['Sales', '7-day MA', '7-day EMA'])
plt.show()
```

## Part 2: Visualization Best Practices

### Chart Selection Guide

```python
# Create sample data for different chart types
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 12, 78]
time_series = np.cumsum(np.random.randn(50)) + 100
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100)

# Create subplots showing different chart types
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Line chart - time series
axes[0, 0].plot(time_series)
axes[0, 0].set_title('Line Chart: Time Series')
axes[0, 0].grid(True, alpha=0.3)

# Bar chart - categories
axes[0, 1].bar(categories, values)
axes[0, 1].set_title('Bar Chart: Categories')
axes[0, 1].grid(True, alpha=0.3)

# Scatter plot - relationships
axes[0, 2].scatter(x_scatter, y_scatter, alpha=0.6)
axes[0, 2].set_title('Scatter Plot: Relationships')
axes[0, 2].grid(True, alpha=0.3)

# Histogram - distribution
data = np.random.normal(0, 1, 1000)
axes[1, 0].hist(data, bins=30, alpha=0.7)
axes[1, 0].set_title('Histogram: Distribution')
axes[1, 0].grid(True, alpha=0.3)

# Box plot - distribution with outliers
data_by_group = [np.random.normal(i, 1, 50) for i in range(3)]
axes[1, 1].boxplot(data_by_group)
axes[1, 1].set_title('Box Plot: Distribution + Outliers')
axes[1, 1].grid(True, alpha=0.3)

# Heatmap - 2D patterns
heatmap_data = np.random.randn(10, 10)
im = axes[1, 2].imshow(heatmap_data, cmap='coolwarm')
axes[1, 2].set_title('Heatmap: 2D Patterns')
plt.colorbar(im, ax=axes[1, 2])

plt.tight_layout()
plt.show()
```

### Good vs Bad Visualization

```python
# Create sample data
data = [10, 20, 30, 40]
categories = ['A', 'B', 'C', 'D']

# Create comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# BAD: Misleading scale, poor colors, no labels
ax1.bar(categories, data, color=['red', 'blue', 'green', 'yellow'])
ax1.set_title('Sales by Region')
ax1.set_ylim(0, 50)  # Misleading scale

# GOOD: Clear scale, good colors, proper labels
ax2.bar(categories, data, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax2.set_title('Sales by Region (in thousands)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Sales (thousands)', fontsize=12)
ax2.set_xlabel('Region', fontsize=12)
ax2.set_ylim(0, 45)  # Appropriate scale
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(data):
    ax2.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

## Part 3: Modern Visualization Tools

### vega-altair (if available)

```python
# Try altair if available
try:
    import altair as alt
    
    # Create sample data
    tips = sns.load_dataset('tips')
    
    # Create altair chart
    chart = alt.Chart(tips).mark_circle().encode(
        x='total_bill:Q',
        y='tip:Q',
        color='time:N',
        size='size:O'
    ).interactive()
    
    # Display (requires Jupyter or altair viewer)
    print("Altair chart created successfully!")
    print("Note: This requires altair to be installed and a compatible environment")
    
except ImportError:
    print("altair not installed. Install with: pip install altair")
```

### plotnine (if available)

```python
# Try plotnine if available
try:
    from plotnine import ggplot, aes, geom_point, geom_smooth, theme_minimal
    
    # Create sample data
    tips = sns.load_dataset('tips')
    
    # Create plotnine chart
    p = (ggplot(tips, aes(x='total_bill', y='tip', color='time')) 
         + geom_point(alpha=0.6)
         + geom_smooth(method='lm')
         + theme_minimal())
    
    print("plotnine chart created successfully!")
    print("Note: This requires plotnine to be installed")
    
except ImportError:
    print("plotnine not installed. Install with: pip install plotnine")
```

## Part 4: Interactive Visualization

### Basic Interactivity with matplotlib

```python
# Create interactive plot
fig, ax = plt.subplots(figsize=(10, 6))

# Generate data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Plot with interactivity
line1, = ax.plot(x, y1, label='sin(x)', linewidth=2)
line2, = ax.plot(x, y2, label='cos(x)', linewidth=2)

# Customize
ax.set_title('Interactive Trigonometric Functions')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.legend()
ax.grid(True, alpha=0.3)

# Add annotations
ax.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red')

plt.tight_layout()
plt.show()
```

## Part 5: Visualization Workflow

### Complete Analysis Workflow

```python
# Create comprehensive analysis
def analyze_data(df):
    """Complete data analysis workflow"""
    
    # 1. Quick overview
    print("Data Overview:")
    print(df.describe())
    print("\nData Types:")
    print(df.dtypes)
    
    # 2. Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Time series plot
    if 'sales' in df.columns:
        df['sales'].plot(ax=axes[0, 0], title='Sales Over Time')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Distribution
    if 'revenue' in df.columns:
        df['revenue'].hist(ax=axes[0, 1], bins=20, alpha=0.7)
        axes[0, 1].set_title('Revenue Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Correlation
    if len(df.columns) > 1:
        corr_matrix = df.corr()
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title('Correlation Matrix')
        plt.colorbar(im, ax=axes[1, 0])
    
    # Box plot
    if len(df.columns) > 1:
        df.boxplot(ax=axes[1, 1])
        axes[1, 1].set_title('Distribution by Variable')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run analysis
analyze_data(df)
```

## Key Takeaways

1. **pandas Plotting**: Use for quick data exploration and time series analysis
2. **Chart Selection**: Choose appropriate chart types for your data and message
3. **Best Practices**: Use clear labels, appropriate scales, and good color choices
4. **Modern Tools**: Explore altair and plotnine for grammar-of-graphics approaches
5. **Workflow**: Combine multiple visualization techniques for comprehensive analysis

## Next Steps

- Practice with your own datasets
- Experiment with different visualization libraries
- Learn about interactive visualization tools
- Apply visualization best practices to your projects
