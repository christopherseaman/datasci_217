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
hist_data = np.random.normal(0, 1, 1000)

# TODO: Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# TODO: Top-left: Bar chart
# TODO: Top-right: Scatter plot
# TODO: Bottom-left: Histogram
# TODO: Bottom-right: Pie chart

# TODO: Add appropriate titles and labels
# TODO: Customize colors and styles
# TODO: Save the plot as 'output/q1_matplotlib_plots.png'

plt.tight_layout()
plt.savefig('output/q1_matplotlib_plots.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Question 2: seaborn Statistical Visualization

### Part 2.1: Load and Explore Data

```python
# Load the sales data
sales_df = pd.read_csv('data/sales_data.csv')
customer_df = pd.read_csv('data/customer_data.csv')
product_df = pd.read_csv('data/product_data.csv')

# Merge data for analysis
merged_df = sales_df.merge(customer_df, on='customer_id').merge(product_df, on='product_id')

print("Data shape:", merged_df.shape)
print("\nColumns:", merged_df.columns.tolist())
print("\nFirst few rows:")
print(merged_df.head())
```

### Part 2.2: Statistical Plots

**TODO: Create statistical visualizations with seaborn**

```python
# TODO: Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# TODO: Top-left: Box plot of total_amount by store_location
# TODO: Top-right: Violin plot of total_amount by category
# TODO: Bottom-left: Scatter plot of quantity vs total_amount with hue by gender
# TODO: Bottom-right: Histogram of total_amount with kde overlay

# TODO: Add appropriate titles and labels
# TODO: Customize colors and styling
# TODO: Save the plot as 'output/q2_seaborn_plots.png'

plt.tight_layout()
plt.savefig('output/q2_seaborn_plots.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Part 2.3: Correlation Analysis

**TODO: Create a correlation heatmap**

```python
# TODO: Select numeric columns for correlation analysis
# First, explore the merged data to see what columns are available
print("Available columns:", merged_df.columns.tolist())
print("Data types:")
print(merged_df.dtypes)

# TODO: Select only the numeric columns that exist for correlation analysis
# Hint: After merging, pandas adds _x and _y suffixes to duplicate column names
numeric_cols = []  # Fill this with the appropriate column names

# TODO: Calculate correlation matrix
# correlation_matrix = ...

# TODO: Create heatmap with seaborn
# TODO: Add title and customize appearance
# TODO: Save the plot as 'output/q2_correlation_heatmap.png'

plt.tight_layout()
plt.savefig('output/q2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Question 3: pandas Plotting and Data Exploration

### Part 3.1: Time Series Visualization

**TODO: Create time series plots with pandas**

```python
# TODO: Convert transaction_date to datetime
# merged_df['transaction_date'] = pd.to_datetime(merged_df['transaction_date'])

# TODO: Create daily sales aggregation
# daily_sales = merged_df.groupby('transaction_date')['total_amount'].sum()

# TODO: Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# TODO: Top-left: Line plot of daily sales
# TODO: Top-right: Rolling 7-day average
# TODO: Bottom-left: Monthly sales bar chart (use resample('ME') for month-end frequency)
# TODO: Bottom-right: Sales distribution histogram

# TODO: Add titles and labels
# TODO: Save the plot as 'output/q3_pandas_plots.png'

plt.tight_layout()
plt.savefig('output/q3_pandas_plots.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Part 3.2: Data Overview Dashboard

**TODO: Create a comprehensive data overview**

```python
# TODO: Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# TODO: Plot 1: Sales by category (bar chart)
# TODO: Plot 2: Sales by store location (pie chart)
# TODO: Plot 3: Age distribution (histogram)
# TODO: Plot 4: Quantity vs total_amount scatter
# TODO: Plot 5: Sales over time (line plot)
# TODO: Plot 6: Top 10 products by sales (horizontal bar)

# TODO: Add titles and customize appearance
# TODO: Save the plot as 'output/q3_data_overview.png'

plt.tight_layout()
plt.savefig('output/q3_data_overview.png', dpi=300, bbox_inches='tight')
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

## Key Learning Objectives

- Master matplotlib fundamentals for custom plots
- Create statistical visualizations with seaborn
- Use pandas plotting for quick exploration
- Apply visualization best practices
- Choose appropriate chart types for different data