# Demo 3: Time Series Visualization and Integration

**Placement**: After "Time Zone Handling" section (end of lecture)  
**Duration**: 25 minutes  
**Focus**: Time series visualization using matplotlib, seaborn, and altair; combining concepts from earlier lectures

## Learning Objectives
- Create effective time series visualizations with multiple libraries
- Integrate pandas, matplotlib, and seaborn for time series plots
- Use altair for interactive time series visualizations
- Identify seasonal patterns in health data
- Create publication-quality plots for medical research

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Enable altair for Jupyter
alt.renderers.enable('default')
```

## Part 1: Multi-Year Disease Surveillance Data

### Load and Prepare Data

```python
# Simulate multi-year disease surveillance data (3 years, monthly)
print("=== Disease Surveillance Data ===\n")

# Create monthly dates for 3 years
monthly_dates = pd.date_range('2020-01-01', periods=36, freq='ME')

# Generate realistic disease surveillance data with seasonality
n = len(monthly_dates)

# Base trend (slight increase over time)
trend = np.linspace(100, 120, n)

# Seasonal component (flu season peaks in winter)
seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 12) + 10

# Noise component
noise = np.random.normal(0, 5, n)

# Combine components
case_counts = trend + seasonal + noise
case_counts = np.maximum(case_counts, 0)  # No negative cases

# Create DataFrame with multiple sites
surveillance_data = pd.DataFrame({
    'cases': case_counts,
    'temperature': 60 + 30 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 2, n),
    'humidity': 50 + 20 * np.sin(2 * np.pi * np.arange(n) / 12 + np.pi) + np.random.normal(0, 3, n),
    'site': 'Site_A'
}, index=monthly_dates)

# Add second site
site_b_cases = case_counts * 0.8 + np.random.normal(0, 3, n)
site_b_cases = np.maximum(site_b_cases, 0)

site_b_data = pd.DataFrame({
    'cases': site_b_cases,
    'temperature': surveillance_data['temperature'] + np.random.normal(0, 1, n),
    'humidity': surveillance_data['humidity'] + np.random.normal(0, 2, n),
    'site': 'Site_B'
}, index=monthly_dates)

# Combine sites
surveillance_data = pd.concat([surveillance_data, site_b_data]).sort_index()

print(f"Surveillance data shape: {surveillance_data.shape}")
print(f"Date range: {surveillance_data.index.min()} to {surveillance_data.index.max()}")
print(f"\nSample data:")
print(surveillance_data.head(10))
```

## Part 2: Time Series Visualization with matplotlib

### Basic Time Series Plots (from Lecture 07)

```python
# Create comprehensive time series visualization with matplotlib
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Disease Surveillance Data - Time Series Analysis', fontsize=16, fontweight='bold')

# Line plot (basic time series)
site_a = surveillance_data[surveillance_data['site'] == 'Site_A']
axes[0, 0].plot(site_a.index, site_a['cases'], 
                marker='o', markersize=4, linewidth=2, label='Site A', color='blue')
axes[0, 0].set_title('Case Counts Over Time', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Cases')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Time series with rolling mean
rolling_mean = site_a['cases'].rolling(window=6).mean()
axes[0, 1].plot(site_a.index, site_a['cases'], 
                alpha=0.5, linewidth=1, label='Monthly Cases', color='gray')
axes[0, 1].plot(rolling_mean.index, rolling_mean.values, 
                linewidth=2, label='6-Month Rolling Mean', color='red')
axes[0, 1].set_title('Cases with Rolling Mean', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Cases')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# Distribution
axes[1, 0].hist(site_a['cases'], bins=15, alpha=0.7, edgecolor='black', color='blue')
axes[1, 0].set_title('Case Count Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Cases')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Box plot by month (seasonal pattern)
monthly_data = [site_a[site_a.index.month == i]['cases'].values for i in range(1, 13)]
axes[1, 1].boxplot(monthly_data, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
axes[1, 1].set_title('Monthly Distribution (Seasonal Pattern)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Cases')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Part 3: Statistical Visualization with seaborn

### Multi-Variable Time Series with seaborn (from Lecture 07)

```python
# Use seaborn for statistical visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Disease Surveillance - Statistical Analysis', fontsize=16, fontweight='bold')

# Prepare data for seaborn
site_a_long = site_a.reset_index()
site_a_long['month'] = site_a_long['date'].dt.month
site_a_long['year'] = site_a_long['date'].dt.year

# Scatter plot: Cases vs Temperature
sns.scatterplot(data=site_a_long, x='temperature', y='cases', 
                hue='month', palette='viridis', ax=axes[0, 0], s=100)
axes[0, 0].set_title('Cases vs Temperature (colored by month)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Temperature (°F)')
axes[0, 0].set_ylabel('Cases')
axes[0, 0].grid(True, alpha=0.3)

# Line plot with multiple variables
ax_twin = axes[0, 1].twinx()
line1 = axes[0, 1].plot(site_a.index, site_a['cases'], 
                        color='blue', linewidth=2, label='Cases', marker='o', markersize=4)
line2 = ax_twin.plot(site_a.index, site_a['temperature'], 
                     color='red', linewidth=2, label='Temperature', linestyle='--')
axes[0, 1].set_title('Cases and Temperature Over Time', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Cases', color='blue')
ax_twin.set_ylabel('Temperature (°F)', color='red')
axes[0, 1].tick_params(axis='y', labelcolor='blue')
ax_twin.tick_params(axis='y', labelcolor='red')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Violin plot by month (distribution shape)
sns.violinplot(data=site_a_long, x='month', y='cases', ax=axes[1, 0])
axes[1, 0].set_title('Monthly Case Distribution (Violin Plot)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Cases')
axes[1, 0].grid(True, alpha=0.3)

# Heatmap: Cases by month and year
heatmap_data = site_a_long.pivot(index='year', columns='month', values='cases')
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1, 1])
axes[1, 1].set_title('Cases Heatmap: Year vs Month', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Year')

plt.tight_layout()
plt.show()
```

## Part 4: Interactive Visualization with altair

### Interactive Time Series with altair (from Lecture 07)

```python
# Create interactive time series visualization with altair
print("=== Interactive Time Series with altair ===\n")

# Prepare data for altair
chart_data = site_a.reset_index()
chart_data = chart_data.rename(columns={'date': 'Date', 'cases': 'Cases', 
                                       'temperature': 'Temperature', 'humidity': 'Humidity'})

# Basic interactive line chart
chart = alt.Chart(chart_data).mark_line(point=True).encode(
    x=alt.X('Date:T', title='Date'),
    y=alt.Y('Cases:Q', title='Cases'),
    tooltip=['Date', 'Cases', 'Temperature']
).properties(
    width=700,
    height=400,
    title='Disease Surveillance - Interactive Time Series'
).interactive()

chart.save('media/interactive_time_series.html')
print("✓ Saved interactive chart: media/interactive_time_series.html")

# Multi-line chart: Cases and Temperature
base = alt.Chart(chart_data).encode(x='Date:T')

line1 = base.mark_line(color='blue', strokeWidth=2).encode(
    y=alt.Y('Cases:Q', title='Cases', axis=alt.Axis(titleColor='blue'))
)

line2 = base.mark_line(color='red', strokeWidth=2, strokeDash=[5, 5]).encode(
    y=alt.Y('Temperature:Q', title='Temperature (°F)', axis=alt.Axis(titleColor='red'))
)

multi_line = alt.layer(line1, line2).resolve_scale(
    y='independent'
).properties(
    width=700,
    height=400,
    title='Cases and Temperature Over Time'
).interactive()

multi_line.save('media/multi_line_interactive.html')
print("✓ Saved multi-line chart: media/multi_line_interactive.html")
```

### Faceted Time Series (Small Multiples)

```python
# Create faceted charts for multiple sites (small multiples)
print("=== Faceted Charts for Multiple Sites ===\n")

# Prepare data for faceting
all_sites = surveillance_data.reset_index()
all_sites = all_sites.rename(columns={'date': 'Date', 'cases': 'Cases', 'site': 'Site'})

# Faceted line chart
faceted = alt.Chart(all_sites).mark_line(point=True).encode(
    x='Date:T',
    y='Cases:Q',
    color='Site:N',
    tooltip=['Date', 'Cases', 'Site']
).facet(
    column='Site:N'
).properties(
    width=300,
    height=300,
    title='Disease Surveillance by Site'
)

faceted.save('media/faceted_sites.html')
print("✓ Saved faceted chart: media/faceted_sites.html")
```

## Part 5: Seasonal Pattern Analysis

### Identifying Seasonal Patterns

```python
# Analyze seasonal patterns
print("=== Seasonal Pattern Analysis ===\n")

# Group by month to identify seasonal pattern
monthly_avg = site_a.groupby(site_a.index.month)['cases'].mean()
monthly_std = site_a.groupby(site_a.index.month)['cases'].std()

print("Average cases by month:")
for month, avg in monthly_avg.items():
    print(f"Month {month:2d}: {avg:6.1f} cases (std: {monthly_std[month]:.1f})")

# Visualize seasonal pattern
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Monthly average bar chart
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
axes[0].bar(month_names, monthly_avg.values, yerr=monthly_std.values, 
           alpha=0.7, color='steelblue', edgecolor='black')
axes[0].set_title('Average Cases by Month (Seasonal Pattern)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Average Cases')
axes[0].grid(True, alpha=0.3, axis='y')

# Seasonal decomposition visualization
trend = site_a['cases'].rolling(window=12, center=True).mean()
seasonal = site_a['cases'] - trend
residual = seasonal - site_a['cases'].groupby(site_a.index.month).transform('mean')

axes[1].plot(site_a.index, site_a['cases'], alpha=0.5, label='Original', color='gray')
axes[1].plot(site_a.index, trend, linewidth=2, label='Trend (12-month)', color='blue')
axes[1].set_title('Trend Component Extraction', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Cases')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## Part 6: Integration with Earlier Concepts

### Combining Resampling, Rolling Windows, and Visualization

```python
# Integrate all concepts: resampling, rolling windows, and visualization
print("=== Integration: Resampling + Rolling + Visualization ===\n")

# Create daily data and resample to different frequencies
daily_dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
daily_cases = 100 + 20 * np.sin(2 * np.pi * np.arange(len(daily_dates)) / 365.25) + np.random.normal(0, 5, len(daily_dates))
daily_cases = np.maximum(daily_cases, 0)
daily_ts = pd.Series(daily_cases, index=daily_dates)

# Resample to different frequencies
weekly_ts = daily_ts.resample('W').mean()
monthly_ts = daily_ts.resample('ME').mean()

# Calculate rolling statistics
daily_ts['rolling_7d'] = daily_ts.rolling(window=7).mean()
daily_ts['rolling_30d'] = daily_ts.rolling(window=30).mean()
daily_ts['rolling_90d'] = daily_ts.rolling(window=90).mean()

# Create comprehensive visualization
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# Original daily with rolling windows
sample_data = daily_ts['2021-01-01':'2021-12-31']
axes[0].plot(sample_data.index, sample_data.values, 
             alpha=0.3, linewidth=1, label='Daily', color='gray')
axes[0].plot(sample_data.index, sample_data['rolling_7d'], 
             linewidth=2, label='7-Day Rolling', color='blue')
axes[0].plot(sample_data.index, sample_data['rolling_30d'], 
             linewidth=2, label='30-Day Rolling', color='green')
axes[0].plot(sample_data.index, sample_data['rolling_90d'], 
             linewidth=2, label='90-Day Rolling', color='red')
axes[0].set_title('Daily Data with Rolling Windows', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Cases')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Weekly resampled
weekly_sample = weekly_ts['2021-01-01':'2021-12-31']
axes[1].plot(weekly_sample.index, weekly_sample.values, 
             marker='o', markersize=6, linewidth=2, label='Weekly Mean', color='blue')
axes[1].set_title('Weekly Resampled Data', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Cases')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Monthly resampled
monthly_sample = monthly_ts['2021-01-01':'2021-12-31']
axes[2].plot(monthly_sample.index, monthly_sample.values, 
             marker='s', markersize=8, linewidth=2, label='Monthly Mean', color='red')
axes[2].set_title('Monthly Resampled Data', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Cases')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Key Takeaways

1. **matplotlib**: Basic time series plots with customization
2. **seaborn**: Statistical visualizations for time series (violin plots, heatmaps)
3. **altair**: Interactive time series visualizations for web
4. **Seasonal Patterns**: Identify and visualize seasonal trends
5. **Integration**: Combine resampling, rolling windows, and visualization
6. **Publication Quality**: Create professional plots for medical research

## Next Steps

- Practice with your own health/medical time series data
- Explore advanced visualization techniques
- Set up automated visualization pipelines
- Integrate with dashboard tools for monitoring
