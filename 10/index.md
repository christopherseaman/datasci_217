Advanced Visualization and Reporting

Welcome to week 10! You've mastered data analysis and automation, and now it's time to communicate your findings effectively through professional visualizations and reports. You'll learn to create publication-ready plots, design compelling visual narratives, and integrate analysis with visualization in comprehensive reports.

By the end of today, you'll create visualizations that clearly communicate insights to stakeholders and build complete analytical reports that combine code, analysis, and professional presentation.

![xkcd 552: Correlation](media/xkcd_552.png)

Today we'll make sure your visualizations tell the right story - clearly and convincingly!

Advanced matplotlib Fundamentals

Understanding the matplotlib Architecture

**Visualization Hierarchy:**
- **Figure** - The overall container (like a canvas)
- **Axes** - Individual plot areas within the figure
- **Artists** - Everything you see (lines, text, legends, etc.)
- **Backend** - How the figure gets displayed or saved

Professional Figure Creation

**Reference:**
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Professional figure setup
def create_professional_figure(figsize=(12, 8), style='seaborn-v0_8'):
 """
 Create a professional matplotlib figure with proper styling
 """
 # Set style (use default if seaborn not available)
 try:
 plt.style.use(style)
 except:
 plt.style.use('default')
 
 # Create figure with specific size and DPI for quality
 fig = plt.figure(figsize=figsize, dpi=100, facecolor='white')
 
 # Set overall properties
 plt.rcParams.update({
 'font.size': 12,
 'axes.titlesize': 16,
 'axes.labelsize': 14,
 'xtick.labelsize': 11,
 'ytick.labelsize': 11,
 'legend.fontsize': 12
 })
 
 return fig

Example usage with sample data
np.random.seed(42)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
product_a_sales = np.random.normal(1000, 150, 12)
product_b_sales = np.random.normal(800, 120, 12)
product_c_sales = np.random.normal(1200, 200, 12)

Professional single plot
fig = create_professional_figure()
ax = fig.add_subplot(111)

Plot data with professional styling
ax.plot(months, product_a_sales, marker='o', linewidth=2.5, label='Product A', color='#2E86AB')
ax.plot(months, product_b_sales, marker='s', linewidth=2.5, label='Product B', color='#A23B72')
ax.plot(months, product_c_sales, marker='^', linewidth=2.5, label='Product C', color='#F18F01')

Professional formatting
ax.set_title('Monthly Sales Performance by Product', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=14, fontweight='bold')
ax.set_ylabel('Sales ($)', fontsize=14, fontweight='bold')

Format y-axis as currency
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

Add grid for readability
ax.grid(True, alpha=0.3)

Position legend professionally
ax.legend(loc='upper left', frameon=True, shadow=True, fancybox=True)

Adjust layout to prevent clipping
plt.tight_layout()

Show the plot
plt.show()
```

Subplot Creation and Management

**Reference:**
```python
def create_dashboard_layout():
 """
 Create a professional dashboard layout with multiple subplots
 """
 # Create figure with subplots
 fig, axes = plt.subplots(2, 2, figsize=(15, 10))
 fig.suptitle('Sales Dashboard - Q4 2024', fontsize=20, fontweight='bold', y=0.98)
 
 # Sample data for different visualizations
 categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
 q4_sales = [45000, 32000, 18000, 25000, 22000]
 monthly_trend = [38000, 42000, 45000]
 customer_ages = np.random.normal(35, 12, 500)
 customer_purchases = np.random.normal(150, 50, 500)
 
 # Subplot 1: Category Sales (Bar Chart)
 ax1 = axes[0, 0]
 bars = ax1.bar(categories, q4_sales, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
 ax1.set_title('Sales by Category', fontweight='bold', fontsize=14)
 ax1.set_ylabel('Sales ($)')
 ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
 
 # Add value labels on bars
 for bar in bars:
 height = bar.get_height()
 ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
 f'${height/1000:.0f}K', ha='center', va='bottom', fontweight='bold')
 
 plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
 
 # Subplot 2: Monthly Trend (Line Chart)
 ax2 = axes[0, 1]
 months_q4 = ['Oct', 'Nov', 'Dec']
 ax2.plot(months_q4, monthly_trend, marker='o', linewidth=3, markersize=8, color='#2E86AB')
 ax2.fill_between(months_q4, monthly_trend, alpha=0.3, color='#2E86AB')
 ax2.set_title('Monthly Sales Trend', fontweight='bold', fontsize=14)
 ax2.set_ylabel('Sales ($)')
 ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
 ax2.grid(True, alpha=0.3)
 
 # Subplot 3: Customer Age Distribution (Histogram)
 ax3 = axes[1, 0]
 ax3.hist(customer_ages, bins=20, color='#96CEB4', alpha=0.7, edgecolor='black')
 ax3.set_title('Customer Age Distribution', fontweight='bold', fontsize=14)
 ax3.set_xlabel('Age')
 ax3.set_ylabel('Number of Customers')
 ax3.axvline(customer_ages.mean(), color='red', linestyle='--', linewidth=2, 
 label=f'Mean: {customer_ages.mean():.1f}')
 ax3.legend()
 
 # Subplot 4: Purchase Amount vs Age (Scatter Plot)
 ax4 = axes[1, 1]
 scatter = ax4.scatter(customer_ages, customer_purchases, alpha=0.6, 
 c=customer_purchases, cmap='viridis', s=30)
 ax4.set_title('Purchase Amount vs Customer Age', fontweight='bold', fontsize=14)
 ax4.set_xlabel('Customer Age')
 ax4.set_ylabel('Purchase Amount ($)')
 
 # Add colorbar
 cbar = plt.colorbar(scatter, ax=ax4)
 cbar.set_label('Purchase Amount ($)', fontweight='bold')
 
 # Add trend line
 z = np.polyfit(customer_ages, customer_purchases, 1)
 p = np.poly1d(z)
 ax4.plot(customer_ages, p(customer_ages), "r--", linewidth=2, alpha=0.8)
 
 # Adjust layout to prevent overlap
 plt.tight_layout()
 plt.subplots_adjust(top=0.93) # Make room for main title
 
 return fig, axes

Create and display the dashboard
fig, axes = create_dashboard_layout()
plt.show()
```

Color Schemes and Visual Aesthetics

Professional Color Selection

**Reference:**
```python
Professional color palettes
PROFESSIONAL_COLORS = {
 'corporate': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600'],
 'accessible': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22'],
 'scientific': ['#0173b2', '#de8f05', '#029e73', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133'],
 'presentation': ['#E31A1C', '#1F78B4', '#33A02C', '#FF7F00', '#6A3D9A', '#B15928', '#A6CEE3', '#FDBF6F']
}

def demonstrate_color_schemes():
 """
 Demonstrate different color schemes and their applications
 """
 fig, axes = plt.subplots(2, 2, figsize=(16, 12))
 fig.suptitle('Professional Color Scheme Applications', fontsize=18, fontweight='bold')
 
 # Sample data
 categories = ['Q1', 'Q2', 'Q3', 'Q4']
 values_2023 = [45, 52, 48, 61]
 values_2024 = [48, 58, 55, 67]
 
 # Corporate colors - grouped bar chart
 ax1 = axes[0, 0]
 x = np.arange(len(categories))
 width = 0.35
 
 ax1.bar(x - width/2, values_2023, width, label='2023', color=PROFESSIONAL_COLORS['corporate'][0])
 ax1.bar(x + width/2, values_2024, width, label='2024', color=PROFESSIONAL_COLORS['corporate'][2])
 
 ax1.set_title('Corporate Color Scheme', fontweight='bold')
 ax1.set_xlabel('Quarter')
 ax1.set_ylabel('Sales (K$)')
 ax1.set_xticks(x)
 ax1.set_xticklabels(categories)
 ax1.legend()
 ax1.grid(True, alpha=0.3)
 
 # Accessible colors - pie chart
 ax2 = axes[0, 1]
 pie_data = [30, 25, 20, 15, 10]
 pie_labels = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
 
 wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
 colors=PROFESSIONAL_COLORS['accessible'][:5],
 startangle=90, textprops={'fontweight': 'bold'})
 ax2.set_title('Accessible Color Scheme', fontweight='bold')
 
 # Scientific colors - line plot with multiple series
 ax3 = axes[1, 0]
 months = np.arange(1, 13)
 
 for i, metric in enumerate(['Revenue', 'Profit', 'Costs', 'Growth']):
 data = np.random.normal(100, 20, 12) + i*10
 ax3.plot(months, data, marker='o', linewidth=2.5, 
 label=metric, color=PROFESSIONAL_COLORS['scientific'][i])
 
 ax3.set_title('Scientific Color Scheme', fontweight='bold')
 ax3.set_xlabel('Month')
 ax3.set_ylabel('Value')
 ax3.legend()
 ax3.grid(True, alpha=0.3)
 
 # Presentation colors - stacked area chart
 ax4 = axes[1, 1]
 quarters = ['Q1', 'Q2', 'Q3', 'Q4']
 product_a = [20, 25, 30, 35]
 product_b = [15, 20, 18, 25]
 product_c = [10, 15, 20, 25]
 
 ax4.stackplot(quarters, product_a, product_b, product_c,
 labels=['Product A', 'Product B', 'Product C'],
 colors=PROFESSIONAL_COLORS['presentation'][:3],
 alpha=0.8)
 
 ax4.set_title('Presentation Color Scheme', fontweight='bold')
 ax4.set_ylabel('Sales (K$)')
 ax4.legend(loc='upper left')
 
 plt.tight_layout()
 return fig

Demonstrate color schemes
color_demo_fig = demonstrate_color_schemes()
plt.show()
```

Visual Design Best Practices

**Reference:**
```python
def create_before_after_comparison():
 """
 Show the difference between poor and professional visualization design
 """
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
 
 # Sample data
 categories = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
 sales = [45000, 32000, 58000, 23000, 41000]
 
 # BEFORE: Poor design practices
 ax1.bar(categories, sales, color=['red', 'blue', 'green', 'purple', 'orange'])
 ax1.set_title('sales by product') # Poor capitalization
 ax1.set_ylabel('sales') # No units
 # No formatting on axis labels
 # Harsh colors
 # No grid or visual aids
 # Cramped labels
 
 # AFTER: Professional design
 # Use professional colors
 colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
 bars = ax2.bar(categories, sales, color=colors, edgecolor='white', linewidth=1.5)
 
 # Professional title and labels
 ax2.set_title('Sales Performance by Product Line', fontsize=16, fontweight='bold', pad=20)
 ax2.set_ylabel('Sales Revenue ($)', fontsize=12, fontweight='bold')
 ax2.set_xlabel('Product Category', fontsize=12, fontweight='bold')
 
 # Format currency on y-axis
 ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
 
 # Add value labels on bars
 for bar in bars:
 height = bar.get_height()
 ax2.text(bar.get_x() + bar.get_width()/2., height + 1500,
 f'${height/1000:.0f}K', ha='center', va='bottom', fontweight='bold')
 
 # Add subtle grid
 ax2.grid(True, alpha=0.3, axis='y')
 ax2.set_axisbelow(True)
 
 # Rotate x-labels if needed
 plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
 
 # Remove unnecessary borders
 ax2.spines['top'].set_visible(False)
 ax2.spines['right'].set_visible(False)
 
 # Add comparison labels
 ax1.text(0.5, 0.95, 'BEFORE: Poor Design', transform=ax1.transAxes, 
 ha='center', va='top', fontsize=14, fontweight='bold', 
 bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7, edgecolor="white"))
 
 ax2.text(0.5, 0.95, 'AFTER: Professional Design', transform=ax2.transAxes, 
 ha='center', va='top', fontsize=14, fontweight='bold',
 bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7, edgecolor="white"))
 
 plt.tight_layout()
 return fig

Create the comparison
comparison_fig = create_before_after_comparison()
plt.show()
```

Statistical Plots for Communication

Distribution Visualization

**Reference:**
```python
def create_statistical_plots():
 """
 Create statistical plots that effectively communicate data insights
 """
 # Generate sample data
 np.random.seed(42)
 group_a = np.random.normal(100, 15, 200)
 group_b = np.random.normal(110, 12, 180)
 group_c = np.random.normal(95, 18, 220)
 
 fig, axes = plt.subplots(2, 2, figsize=(15, 12))
 fig.suptitle('Statistical Communication Plots', fontsize=18, fontweight='bold')
 
 # 1. Box plot for distribution comparison
 ax1 = axes[0, 0]
 box_data = [group_a, group_b, group_c]
 box_labels = ['Treatment A', 'Treatment B', 'Control']
 colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
 
 bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
 for patch, color in zip(bp['boxes'], colors):
 patch.set_facecolor(color)
 patch.set_alpha(0.7)
 
 ax1.set_title('Distribution Comparison', fontweight='bold', fontsize=14)
 ax1.set_ylabel('Score')
 ax1.grid(True, alpha=0.3)
 
 # Add statistical annotations
 ax1.text(0.02, 0.98, f'Treatment A: μ={group_a.mean():.1f}±{group_a.std():.1f}', 
 transform=ax1.transAxes, va='top', fontsize=10,
 bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[0], alpha=0.3))
 ax1.text(0.02, 0.88, f'Treatment B: μ={group_b.mean():.1f}±{group_b.std():.1f}', 
 transform=ax1.transAxes, va='top', fontsize=10,
 bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[1], alpha=0.3))
 ax1.text(0.02, 0.78, f'Control: μ={group_c.mean():.1f}±{group_c.std():.1f}', 
 transform=ax1.transAxes, va='top', fontsize=10,
 bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[2], alpha=0.3))
 
 # 2. Histogram with density overlay
 ax2 = axes[0, 1]
 n, bins, patches = ax2.hist(group_a, bins=25, density=True, alpha=0.7, 
 color='#FF6B6B', edgecolor='black', linewidth=0.5)
 
 # Add normal distribution overlay
 mu, sigma = group_a.mean(), group_a.std()
 x = np.linspace(group_a.min(), group_a.max(), 100)
 normal_curve = ((1/(sigma * np.sqrt(2 * np.pi))) * 
 np.exp(-0.5 * ((x - mu) / sigma) ** 2))
 ax2.plot(x, normal_curve, 'k-', linewidth=2, label=f'Normal (μ={mu:.1f}, σ={sigma:.1f})')
 
 ax2.set_title('Distribution with Normal Overlay', fontweight='bold', fontsize=14)
 ax2.set_xlabel('Score')
 ax2.set_ylabel('Density')
 ax2.legend()
 ax2.grid(True, alpha=0.3)
 
 # 3. Correlation plot with trend line
 ax3 = axes[1, 0]
 x_data = np.random.normal(50, 10, 100)
 y_data = 2 * x_data + np.random.normal(0, 8, 100)
 
 # Scatter plot
 scatter = ax3.scatter(x_data, y_data, alpha=0.6, c=y_data, cmap='viridis', s=50)
 
 # Add trend line
 z = np.polyfit(x_data, y_data, 1)
 p = np.poly1d(z)
 ax3.plot(x_data, p(x_data), "r-", linewidth=2, alpha=0.8)
 
 # Calculate and display correlation
 correlation = np.corrcoef(x_data, y_data)[0, 1]
 ax3.set_title(f'Correlation Analysis (r = {correlation:.3f})', fontweight='bold', fontsize=14)
 ax3.set_xlabel('X Variable')
 ax3.set_ylabel('Y Variable')
 
 # Add R-squared
 r_squared = correlation ** 2
 ax3.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax3.transAxes, 
 va='top', fontsize=12, fontweight='bold',
 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
 
 # Add colorbar
 cbar = plt.colorbar(scatter, ax=ax3)
 cbar.set_label('Y Variable Value')
 
 # 4. Error bars and confidence intervals
 ax4 = axes[1, 1]
 categories = ['A', 'B', 'C', 'D']
 means = [20, 35, 30, 25]
 std_errors = [2, 3, 2.5, 1.8]
 
 bars = ax4.bar(categories, means, yerr=std_errors, capsize=5,
 color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
 alpha=0.8, edgecolor='black', linewidth=1)
 
 ax4.set_title('Means with Error Bars', fontweight='bold', fontsize=14)
 ax4.set_ylabel('Value ± Standard Error')
 ax4.grid(True, alpha=0.3, axis='y')
 
 # Add value labels
 for bar, mean, se in zip(bars, means, std_errors):
 height = bar.get_height()
 ax4.text(bar.get_x() + bar.get_width()/2., height + se + 0.5,
 f'{mean:.1f}±{se:.1f}', ha='center', va='bottom', fontweight='bold')
 
 plt.tight_layout()
 return fig

Create statistical plots
stats_fig = create_statistical_plots()
plt.show()
```

**Brief Example:**
```python
Quick professional visualization template
def quick_professional_plot(x_data, y_data, title, xlabel, ylabel, plot_type='line'):
 """
 Create a professional plot quickly
 """
 fig, ax = plt.subplots(figsize=(10, 6))
 
 if plot_type == 'line':
 ax.plot(x_data, y_data, marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
 elif plot_type == 'bar':
 ax.bar(x_data, y_data, color='#2E86AB', alpha=0.8, edgecolor='white', linewidth=1.5)
 elif plot_type == 'scatter':
 ax.scatter(x_data, y_data, alpha=0.7, s=50, c='#2E86AB')
 
 # Professional formatting
 ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
 ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
 ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
 ax.grid(True, alpha=0.3)
 
 # Remove top and right spines
 ax.spines['top'].set_visible(False)
 ax.spines['right'].set_visible(False)
 
 plt.tight_layout()
 return fig, ax

Example usage
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
revenue = [45000, 52000, 48000, 61000, 58000, 67000]

fig, ax = quick_professional_plot(months, revenue, 
 'Monthly Revenue Growth', 
 'Month', 'Revenue ($)', 
 'line')
plt.show()
```

LIVE DEMO!
*Creating a complete data visualization dashboard: from raw data to publication-ready plots*

Jupyter for Professional Reporting

Combining Analysis and Visualization

Structured Report Template

**Reference:**
```python
Jupyter notebook cell structure for professional reports

Cell 1: Setup and imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

Configure matplotlib for high-quality output
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

Display all outputs in cells
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

```

```markdown
Sales Performance Analysis Report

**Author:** Data Science Team 
**Date:** October 2024 
**Objective:** Analyze Q3 sales performance and identify growth opportunities

Executive Summary

This report analyzes Q3 2024 sales performance across all product categories and regions. Key findings include:
- Overall revenue increased by 15% compared to Q2
- Electronics category showed strongest growth (22%)
- Regional performance varied significantly, with West region leading

---
```

```python
Cell 3: Data loading and initial exploration
def load_and_explore_data():
 """
 Load sales data and perform initial exploration
 """
 
 # In real scenario, load from file:
 # df = pd.read_csv('q3_sales_data.csv')
 
 # Sample data for demonstration
 np.random.seed(42)
 dates = pd.date_range('2024-07-01', '2024-09-30', freq='D')
 products = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
 regions = ['North', 'South', 'East', 'West']
 
 # Create sample dataset
 data = []
 for date in dates:
 for _ in range(np.random.poisson(10)):
 data.append({
 'date': date,
 'product': np.random.choice(products),
 'region': np.random.choice(regions),
 'revenue': np.random.exponential(100) + 20,
 'units_sold': np.random.poisson(3) + 1
 })
 
 df = pd.DataFrame(data)
 df['month'] = df['date'].dt.month_name()
 df['week'] = df['date'].dt.isocalendar().week
 
 
 return df

Load data
sales_df = load_and_explore_data()

Quick data quality check

Display first few records
sales_df.head()
```

```python
Cell 4: Summary statistics with professional formatting
def generate_summary_statistics(df):
 """
 Generate comprehensive summary statistics
 """
 
 # Overall metrics
 total_revenue = df['revenue'].sum()
 total_transactions = len(df)
 avg_transaction = df['revenue'].mean()
 
 
 # Monthly breakdown
 monthly_summary = df.groupby('month').agg({
 'revenue': ['sum', 'mean', 'count'],
 'units_sold': 'sum'
 }).round(2)
 
 display(monthly_summary) # Use display() for better formatting in Jupyter
 
 # Product performance
 product_summary = df.groupby('product').agg({
 'revenue': ['sum', 'mean', 'count'],
 'units_sold': 'sum'
 }).round(2).sort_values(('revenue', 'sum'), ascending=False)
 
 display(product_summary)
 
 return {
 'total_revenue': total_revenue,
 'total_transactions': total_transactions,
 'avg_transaction': avg_transaction,
 'monthly': monthly_summary,
 'product': product_summary
 }

Generate statistics
summary_stats = generate_summary_statistics(sales_df)
```

```python
Cell 5: Professional visualization dashboard
def create_report_visualizations(df):
 """
 Create professional visualizations for the report
 """
 # Set up the figure layout
 fig = plt.figure(figsize=(20, 15))
 gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
 
 # Color palette
 colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
 
 # 1. Monthly revenue trend (spans 2 columns)
 ax1 = fig.add_subplot(gs[0, :2])
 monthly_revenue = df.groupby('month')['revenue'].sum()
 month_order = ['July', 'August', 'September']
 monthly_revenue = monthly_revenue.reindex(month_order)
 
 bars = ax1.bar(monthly_revenue.index, monthly_revenue.values, color=colors[0], alpha=0.8)
 ax1.set_title('Monthly Revenue Trend', fontsize=16, fontweight='bold')
 ax1.set_ylabel('Revenue ($)', fontweight='bold')
 ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
 
 # Add value labels
 for bar in bars:
 height = bar.get_height()
 ax1.text(bar.get_x() + bar.get_width()/2., height,
 f'${height/1000:.0f}K', ha='center', va='bottom', fontweight='bold')
 ax1.grid(axis='y', alpha=0.3)
 
 # 2. Product category pie chart
 ax2 = fig.add_subplot(gs[0, 2])
 product_revenue = df.groupby('product')['revenue'].sum()
 wedges, texts, autotexts = ax2.pie(product_revenue.values, labels=product_revenue.index,
 autopct='%1.1f%%', colors=colors,
 startangle=90)
 ax2.set_title('Revenue by Product Category', fontsize=16, fontweight='bold')
 
 # 3. Regional performance comparison
 ax3 = fig.add_subplot(gs[1, :2])
 regional_data = df.groupby(['region', 'product'])['revenue'].sum().unstack(fill_value=0)
 
 regional_data.plot(kind='bar', ax=ax3, color=colors, width=0.8)
 ax3.set_title('Regional Performance by Product', fontsize=16, fontweight='bold')
 ax3.set_ylabel('Revenue ($)', fontweight='bold')
 ax3.set_xlabel('Region', fontweight='bold')
 ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
 ax3.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')
 plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)
 ax3.grid(axis='y', alpha=0.3)
 
 # 4. Transaction value distribution
 ax4 = fig.add_subplot(gs[1, 2])
 ax4.hist(df['revenue'], bins=30, color=colors[1], alpha=0.7, edgecolor='black')
 ax4.axvline(df['revenue'].mean(), color='red', linestyle='--', linewidth=2,
 label=f'Mean: ${df["revenue"].mean():.2f}')
 ax4.set_title('Transaction Value Distribution', fontsize=16, fontweight='bold')
 ax4.set_xlabel('Transaction Value ($)', fontweight='bold')
 ax4.set_ylabel('Frequency', fontweight='bold')
 ax4.legend()
 ax4.grid(alpha=0.3)
 
 # 5. Weekly trend analysis (spans all columns)
 ax5 = fig.add_subplot(gs[2, :])
 weekly_data = df.groupby(['week', 'product'])['revenue'].sum().unstack(fill_value=0)
 
 for i, product in enumerate(weekly_data.columns):
 ax5.plot(weekly_data.index, weekly_data[product], marker='o', 
 linewidth=2.5, label=product, color=colors[i])
 
 ax5.set_title('Weekly Revenue Trends by Product', fontsize=16, fontweight='bold')
 ax5.set_xlabel('Week Number', fontweight='bold')
 ax5.set_ylabel('Revenue ($)', fontweight='bold')
 ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
 ax5.legend(title='Product Category')
 ax5.grid(True, alpha=0.3)
 
 # Main title
 fig.suptitle('Q3 2024 Sales Performance Dashboard', fontsize=24, fontweight='bold', y=0.98)
 
 plt.show()
 return fig

Create visualizations
dashboard_fig = create_report_visualizations(sales_df)
```

```markdown
Key Findings

Revenue Performance
- **Total Q3 Revenue:** $X,XXX,XXX (15% increase from Q2)
- **Highest performing month:** September with $XXX,XXX
- **Average transaction value:** $XX.XX

Product Analysis
- **Electronics** leads with XX% of total revenue
- **Sports equipment** shows strongest growth trajectory
- **Books** category underperformed expectations

Regional Insights
- **West region** contributed XX% of total revenue
- **North region** shows potential for expansion
- **Product preferences vary significantly by region**

Recommendations

1. **Increase marketing investment** in Electronics category during peak months
2. **Expand Sports equipment inventory** based on growing demand
3. **Regional strategy**: Focus expansion efforts on North region
4. **Product mix optimization**: Consider regional preferences in inventory planning

---
```

Export Formats and Sharing

Professional Export Options

**Reference:**
```python
def export_report_assets(fig, df, output_dir='report_outputs'):
 """
 Export report components in multiple formats
 """
 import os
 from pathlib import Path
 
 # Create output directory
 Path(output_dir).mkdir(exist_ok=True)
 
 
 # 1. High-resolution figure export
 fig.savefig(f'{output_dir}/sales_dashboard.png', 
 dpi=300, bbox_inches='tight', facecolor='white')
 fig.savefig(f'{output_dir}/sales_dashboard.pdf', 
 bbox_inches='tight', facecolor='white')
 
 # 2. Data export for stakeholders
 # Summary data
 summary_data = df.groupby('product').agg({
 'revenue': ['sum', 'mean', 'count'],
 'units_sold': 'sum'
 }).round(2)
 summary_data.to_excel(f'{output_dir}/summary_statistics.xlsx')
 
 # 3. Raw data export (cleaned)
 df.to_csv(f'{output_dir}/q3_sales_data_cleaned.csv', index=False)
 
 # 4. Create PowerPoint-ready slides export
 individual_plots = ['monthly_trend', 'product_breakdown', 'regional_analysis']
 
 for plot_name in individual_plots:
 # Create individual plots for presentations
 fig_small, ax = plt.subplots(figsize=(10, 6))
 
 if plot_name == 'monthly_trend':
 monthly_revenue = df.groupby('month')['revenue'].sum()
 month_order = ['July', 'August', 'September']
 monthly_revenue = monthly_revenue.reindex(month_order)
 ax.bar(monthly_revenue.index, monthly_revenue.values, color='#2E86AB')
 ax.set_title('Monthly Revenue Trend', fontsize=16, fontweight='bold')
 
 elif plot_name == 'product_breakdown':
 product_revenue = df.groupby('product')['revenue'].sum()
 ax.pie(product_revenue.values, labels=product_revenue.index, autopct='%1.1f%%')
 ax.set_title('Revenue by Product Category', fontsize=16, fontweight='bold')
 
 elif plot_name == 'regional_analysis':
 regional_revenue = df.groupby('region')['revenue'].sum()
 ax.bar(regional_revenue.index, regional_revenue.values, color='#A23B72')
 ax.set_title('Revenue by Region', fontsize=16, fontweight='bold')
 
 fig_small.savefig(f'{output_dir}/{plot_name}_slide.png', 
 dpi=300, bbox_inches='tight', facecolor='white')
 plt.close(fig_small)
 
 
 # 5. Create report metadata
 metadata = {
 'report_date': datetime.now().isoformat(),
 'data_period': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
 'total_records': len(df),
 'total_revenue': df['revenue'].sum(),
 'files_created': [
 'sales_dashboard.png', 'sales_dashboard.pdf',
 'summary_statistics.xlsx', 'q3_sales_data_cleaned.csv'
 ] + [f'{plot}_slide.png' for plot in individual_plots]
 }
 
 import json
 with open(f'{output_dir}/report_metadata.json', 'w') as f:
 json.dump(metadata, f, indent=2)
 
 
 return metadata

Export report assets
metadata = export_report_assets(dashboard_fig, sales_df)
```

Jupyter to HTML/PDF Export

**Reference:**
```python
Command line exports (run in terminal)
"""
Convert Jupyter notebook to HTML report
jupyter nbconvert --to html --template classic sales_analysis.ipynb

Convert to PDF (requires latex)
jupyter nbconvert --to pdf sales_analysis.ipynb

Convert to slides for presentation
jupyter nbconvert --to slides sales_analysis.ipynb --post serve

Custom HTML template with company branding
jupyter nbconvert --to html --template custom_template.tpl sales_analysis.ipynb
"""

Programmatic export from within notebook
def export_notebook_programmatically():
 """
 Export current notebook to various formats programmatically
 """
 import subprocess
 import os
 
 # Get current notebook name (if running in Jupyter)
 try:
 notebook_name = 'sales_analysis.ipynb' # Replace with actual notebook name
 
 # Export to HTML
 subprocess.run(['jupyter', 'nbconvert', '--to', 'html', notebook_name])
 
 # Export to PDF (if latex available)
 try:
 subprocess.run(['jupyter', 'nbconvert', '--to', 'pdf', notebook_name])
 except:
 
 except Exception as e:

Uncomment to run export
export_notebook_programmatically()
```

Visualization Design Principles

Effective Communication Guidelines

**The CLEAR Framework:**
- **C**ontext - Provide sufficient background and explanation
- **L**abels - Clear titles, axis labels, and legends
- **E**mphasis - Highlight the most important information
- **A**ccessibility - Use colorblind-friendly palettes and sufficient contrast
- **R**elevance - Every element should serve the analysis purpose

Common Pitfalls to Avoid

**Reference:**
```python
DON'T DO THESE - Common visualization mistakes

1. DON'T: Use 3D effects unnecessarily
ax.bar3d(x, y, z, dx, dy, dz) # Avoid unless truly needed

2. DON'T: Use pie charts with too many categories
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] # Too many!

3. DON'T: Start bar charts at non-zero baselines (misleading)
ax.set_ylim(50, 60) # Makes small differences look huge

4. DON'T: Use rainbow color schemes for continuous data
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple'] # Hard to interpret order

5. DON'T: Forget to label axes and provide context
ax.plot(x, y) # No titles, labels, or context

INSTEAD, DO THIS:
def create_effective_visualization():
 """
 Demonstrate effective visualization principles
 """
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
 
 # Example data
 categories = ['Product A', 'Product B', 'Product C', 'Product D']
 values = [85, 92, 78, 88]
 
 # DO: Clear, accessible design
 colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Accessible colors
 bars = ax1.bar(categories, values, color=colors, alpha=0.8, 
 edgecolor='black', linewidth=1)
 
 # DO: Comprehensive labeling
 ax1.set_title('Customer Satisfaction Scores by Product\n(Scale: 0-100)', 
 fontsize=14, fontweight='bold', pad=20)
 ax1.set_ylabel('Satisfaction Score', fontweight='bold')
 ax1.set_xlabel('Product Category', fontweight='bold')
 
 # DO: Start at zero for honest comparison
 ax1.set_ylim(0, 100)
 
 # DO: Add value labels for precision
 for bar in bars:
 height = bar.get_height()
 ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
 f'{height}', ha='center', va='bottom', fontweight='bold')
 
 # DO: Add reference lines for context
 ax1.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Target (80)')
 ax1.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Minimum (60)')
 ax1.legend()
 
 # DO: Use subtle grid for readability
 ax1.grid(axis='y', alpha=0.3)
 
 # Example of time series best practices
 dates = pd.date_range('2024-01-01', periods=12, freq='M')
 values = [100, 105, 102, 108, 110, 107, 112, 115, 118, 114, 120, 125]
 
 ax2.plot(dates, values, marker='o', linewidth=2.5, markersize=6, color='#2E86AB')
 ax2.fill_between(dates, values, alpha=0.3, color='#2E86AB')
 
 # DO: Format dates properly
 ax2.set_title('Revenue Growth Over Time', fontsize=14, fontweight='bold', pad=20)
 ax2.set_ylabel('Revenue (Thousands $)', fontweight='bold')
 
 # DO: Format axis labels for readability
 ax2.tick_params(axis='x', rotation=45)
 ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x}K'))
 
 # DO: Add trend information
 from scipy import stats
 slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(values)), values)
 ax2.text(0.05, 0.95, f'Monthly Growth Rate: {slope:.1f}K', 
 transform=ax2.transAxes, va='top', fontweight='bold',
 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
 
 plt.tight_layout()
 return fig

effective_viz = create_effective_visualization()
plt.show()
```

Key Takeaways

1. **matplotlib mastery** enables professional publication-quality visualizations
2. **Subplot layouts** create comprehensive dashboards and reports
3. **Color schemes** must be accessible and purposeful
4. **Statistical plots** effectively communicate data insights
5. **Jupyter integration** enables seamless analysis-to-report workflows
6. **Multiple export formats** serve different stakeholder needs
7. **Design principles** ensure clear, honest communication
8. **Professional presentation** requires attention to every visual detail

You now have the skills to create compelling visualizations that effectively communicate your data analysis findings to any audience, from technical teams to executive stakeholders.

Next week: We'll integrate everything into comprehensive projects and prepare for real-world applications!

Practice Challenge

Before next class:
1. **Visualization Mastery:**
 - Create a complete dashboard with multiple subplot types
 - Practice using professional color schemes and formatting
 - Export visualizations in multiple formats for different uses
 
2. **Report Integration:**
 - Build a Jupyter notebook report combining analysis and visualization
 - Practice the CLEAR framework for effective communication
 - Export your report as HTML and PDF formats
 
3. **Design Principles:**
 - Review and improve an existing visualization using best practices
 - Create before/after examples showing design improvements
 - Focus on accessibility and clear communication

Remember: Great visualizations don't just show data - they tell compelling, accurate stories!