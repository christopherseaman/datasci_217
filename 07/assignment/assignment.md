# Assignment 7: Data Visualization

## Overview
This assignment covers the essential tools for data visualization: matplotlib fundamentals, seaborn statistical plots, pandas plotting, and visualization best practices.

## Setup

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set style
plt.style.use('default')
sns.set_style('whitegrid')

# Create output directory
os.makedirs('output', exist_ok=True)
```

## Question 1: matplotlib Fundamentals

### Part 1.1: Basic Figures and Subplots

**TODO: Create a figure with 2x2 subplots showing different plot types**

```python
# Create sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(-x/5)
y4 = np.log(x + 1)

# TODO: Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# TODO: Plot on each subplot
# Top-left: Sine wave
# Top-right: Cosine wave  
# Bottom-left: Exponential decay
# Bottom-right: Logarithmic function

# TODO: Add titles and labels to each subplot
# TODO: Add grid to each subplot
# TODO: Use tight_layout() to prevent overlapping

plt.show()
```

### Part 1.2: Plot Customization

**TODO: Create a customized line plot with multiple series**

```python
# Create sample data
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

# TODO: Create a figure with custom styling
fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot three lines with different styles
# Line 1: Solid line, blue color, label 'sin(x)'
# Line 2: Dashed line, red color, label 'cos(x)'  
# Line 3: Dotted line, green color, label 'sin(x)*cos(x)'

# TODO: Add title, xlabel, ylabel
# TODO: Add legend
# TODO: Add grid with transparency
# TODO: Set appropriate axis limits

plt.show()
```

### Part 1.3: Different Plot Types

**TODO: Create a comprehensive visualization with multiple plot types**

```python
# Create sample data
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 12, 78]
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100)
data_hist = np.random.normal(0, 1, 1000)

# TODO: Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# TODO: Bar chart (top-left)
# TODO: Scatter plot (top-right)
# TODO: Histogram (bottom-left)
# TODO: Box plot (bottom-right)

# TODO: Add titles and labels to each subplot
# TODO: Use tight_layout()

plt.show()
```

### Part 1.4: Save Plots

**TODO: Create and save a publication-quality plot**

```python
# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/5)

# TODO: Create a high-quality figure
fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot the data with custom styling
# TODO: Add title, labels, grid
# TODO: Save as PNG with high DPI
# TODO: Save as PDF
# TODO: Save as SVG

plt.show()
print("Plots saved successfully!")
```

## Question 2: seaborn Statistical Visualization

### Part 2.1: Load and Explore Data

**TODO: Load the sales data and create basic seaborn plots**

```python
# Load the sales data
sales_data = pd.read_csv('data/sales_data.csv')
customer_data = pd.read_csv('data/customer_data.csv')
product_data = pd.read_csv('data/product_data.csv')

# TODO: Merge the datasets
# Merge sales with customer data
# Merge result with product data
# TODO: Add calculated columns (age_group, price_category)

print("Data shape:", merged_data.shape)
print("Columns:", merged_data.columns.tolist())
```

### Part 2.2: Statistical Plots

**TODO: Create statistical visualizations with seaborn**

```python
# TODO: Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# TODO: Box plot of total_amount by store_location
# TODO: Violin plot of total_amount by age_group
# TODO: Histogram of total_amount with KDE
# TODO: Scatter plot of quantity vs total_amount

# TODO: Add titles and labels
plt.tight_layout()
plt.show()
```

### Part 2.3: Relationship Analysis

**TODO: Create relationship visualizations**

```python
# TODO: Create correlation heatmap
# TODO: Create pair plot for numerical columns
# TODO: Create joint plot for quantity vs total_amount

# TODO: Save the correlation heatmap
plt.savefig('output/q2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Part 2.4: Advanced seaborn Features

**TODO: Create advanced statistical plots**

```python
# TODO: Create faceted plots
# TODO: Create categorical plots
# TODO: Create distribution plots
# TODO: Apply seaborn styling

# TODO: Save the plots
plt.savefig('output/q2_seaborn_plots.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Question 3: pandas Plotting and Data Exploration

### Part 3.1: Quick Data Exploration

**TODO: Use pandas plotting for quick exploration**

```python
# TODO: Create time series plot of sales over time
# TODO: Create histogram of total_amount
# TODO: Create box plot by category
# TODO: Create scatter plot of quantity vs total_amount

# TODO: Save the plots
plt.savefig('output/q3_pandas_plots.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Part 3.2: Time Series Analysis

**TODO: Create time series visualizations**

```python
# TODO: Convert transaction_date to datetime
# TODO: Set transaction_date as index
# TODO: Create daily sales aggregation
# TODO: Plot daily sales with moving average
# TODO: Create seasonal analysis

# TODO: Save the time series plot
plt.savefig('output/q3_data_overview.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Part 3.3: Data Overview

**TODO: Create comprehensive data overview**

```python
# TODO: Create summary statistics
# TODO: Create distribution plots
# TODO: Create correlation analysis
# TODO: Create categorical analysis

# TODO: Save the comprehensive overview
plt.savefig('output/q3_data_overview.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Question 4: Modern Visualization and Best Practices

### Part 4.1: Chart Selection

**TODO: Create appropriate charts for different data types**

```python
# TODO: Create line chart for time series data
# TODO: Create bar chart for categorical data
# TODO: Create scatter plot for relationship data
# TODO: Create histogram for distribution data

# TODO: Apply best practices for each chart type
# TODO: Use appropriate colors and styling
# TODO: Add clear titles and labels

plt.tight_layout()
plt.show()
```

### Part 4.2: Good vs Bad Visualization

**TODO: Create examples of good and bad visualization**

```python
# Create sample data
data = [10, 20, 30, 40]
categories = ['A', 'B', 'C', 'D']

# TODO: Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# TODO: BAD visualization (misleading scale, poor colors)
# TODO: GOOD visualization (clear scale, good colors, labels)

# TODO: Add value labels on bars
# TODO: Save the comparison
plt.savefig('output/q4_best_practices.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Part 4.3: Publication Quality

**TODO: Create publication-quality visualizations**

```python
# TODO: Create a comprehensive analysis plot
# TODO: Use consistent styling
# TODO: Apply design principles
# TODO: Create clear legends and annotations

# TODO: Save the final report
plt.savefig('output/q4_final_report.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Part 4.4: Final Report

**TODO: Create a comprehensive visualization report**

```python
# TODO: Create a multi-panel analysis
# TODO: Include summary statistics
# TODO: Add insights and conclusions
# TODO: Use professional styling

# TODO: Save the final report
plt.savefig('output/q4_final_report.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_matplotlib_plots.png` - matplotlib fundamentals
- [ ] `output/q1_multi_panel.png` - multi-panel visualization
- [ ] `output/q2_seaborn_plots.png` - seaborn statistical plots
- [ ] `output/q2_correlation_heatmap.png` - correlation analysis
- [ ] `output/q3_pandas_plots.png` - pandas plotting
- [ ] `output/q3_data_overview.png` - data exploration
- [ ] `output/q4_best_practices.png` - visualization best practices
- [ ] `output/q4_final_report.png` - comprehensive report

## Key Learning Objectives

1. **matplotlib Fundamentals**: Create figures, subplots, and customize plots
2. **seaborn Statistical Plots**: Visualize relationships and distributions
3. **pandas Plotting**: Quick data exploration and time series analysis
4. **Best Practices**: Choose appropriate charts and apply design principles

## Tips for Success

- Read each TODO item carefully
- Use the hints provided in comments
- Test your code incrementally
- Save your plots with descriptive filenames
- Apply consistent styling throughout
- Choose appropriate chart types for your data
