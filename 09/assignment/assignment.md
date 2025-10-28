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

# Assignment 9: Time Series Analysis

## Overview
This assignment covers time series analysis using real-world healthcare data including patient monitoring, clinical trials, and medical device sensor data.

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('output', exist_ok=True)
```

## Question 1: datetime Fundamentals

### Part 1.1: Load and Explore Data

```python
# Load the datasets
stock_df = pd.read_csv('data/stock_prices.csv')
weather_df = pd.read_csv('data/weather_data.csv')
sales_df = pd.read_csv('data/sales_data.csv')

print("Stock data shape:", stock_df.shape)
print("Weather data shape:", weather_df.shape)
print("Sales data shape:", sales_df.shape)

print("\nStock data columns:", stock_df.columns.tolist())
print("Weather data columns:", weather_df.columns.tolist())
print("Sales data columns:", sales_df.columns.tolist())
```

### Part 1.2: datetime Operations

**TODO: Perform datetime operations**

```python
# TODO: Convert date columns to datetime
# stock_df['date'] = pd.to_datetime(stock_df['date'])
# weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
# sales_df['date'] = pd.to_datetime(sales_df['date'])

# TODO: Set datetime columns as index
# TODO: Extract year, month, day, hour components
# TODO: Calculate time differences
# TODO: Create business day ranges

# TODO: Save results as 'output/q1_datetime_analysis.csv'
```

### Part 1.3: Time Zone Handling

**TODO: Handle time zones**

```python
# TODO: Create timezone-aware datetime
# TODO: Convert between different timezones
# TODO: Handle daylight saving time
# TODO: Create timezone-aware DataFrames

# TODO: Document timezone operations
# TODO: Save results as 'output/q1_timezone_report.txt'
```

## Question 2: Time Series Indexing and Resampling

### Part 2.1: Time Series Selection

**TODO: Perform time series indexing and selection**

```python
# TODO: Select data by specific dates
# TODO: Select data by date ranges
# TODO: Select data by time periods (business hours, specific months)
# TODO: Use first() and last() methods
# TODO: Use truncate() method

# TODO: Demonstrate different selection methods
# TODO: Save results as 'output/q2_resampling_analysis.csv'
```

### Part 2.2: Resampling Operations

**TODO: Perform resampling and frequency conversion**

```python
# TODO: Resample daily data to weekly
# TODO: Resample daily data to monthly
# TODO: Resample hourly data to daily
# TODO: Use different aggregation functions (mean, sum, max, min)
# TODO: Handle missing values during resampling

# TODO: Compare different resampling frequencies
# TODO: Save results as 'output/q2_resampling_analysis.csv'
```

### Part 2.3: Missing Data Handling

**TODO: Handle missing data in time series**

```python
# TODO: Identify missing values in time series
# TODO: Use forward fill and backward fill
# TODO: Use interpolation methods
# TODO: Use rolling mean for imputation
# TODO: Create missing data report

# TODO: Document missing data patterns
# TODO: Save results as 'output/q2_missing_data_report.txt'
```

## Question 3: Rolling Window Operations

### Part 3.1: Basic Rolling Operations

**TODO: Apply rolling window operations**

```python
# TODO: Calculate 7-day rolling mean
# TODO: Calculate 30-day rolling standard deviation
# TODO: Calculate rolling min and max
# TODO: Calculate rolling sum
# TODO: Use different window sizes

# TODO: Create rolling statistics dataframe
# TODO: Save results as 'output/q3_rolling_analysis.csv'
```

### Part 3.2: Advanced Rolling Operations

**TODO: Apply advanced rolling operations**

```python
# TODO: Use centered rolling windows
# TODO: Use expanding windows
# TODO: Calculate exponentially weighted moving averages
# TODO: Create custom rolling functions
# TODO: Handle minimum periods requirement

# TODO: Compare different rolling methods
# TODO: Save results as 'output/q3_rolling_analysis.csv'
```

### Part 3.3: Trend Analysis

**TODO: Create trend analysis visualization**

```python
# TODO: Create time series plot with original data
# TODO: Add rolling mean overlay
# TODO: Add rolling standard deviation bands
# TODO: Add exponentially weighted moving average
# TODO: Customize colors and styling

# TODO: Save the plot as 'output/q3_trend_analysis.png'
```

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_datetime_analysis.csv` - datetime analysis
- [ ] `output/q1_timezone_report.txt` - timezone report
- [ ] `output/q2_resampling_analysis.csv` - resampling analysis
- [ ] `output/q2_missing_data_report.txt` - missing data report
- [ ] `output/q3_rolling_analysis.csv` - rolling analysis
- [ ] `output/q3_trend_analysis.png` - trend analysis plot

## Key Learning Objectives

- Master datetime handling and parsing
- Perform time series indexing and selection
- Use resampling and frequency conversion
- Apply rolling window operations
- Handle time zones and temporal data
