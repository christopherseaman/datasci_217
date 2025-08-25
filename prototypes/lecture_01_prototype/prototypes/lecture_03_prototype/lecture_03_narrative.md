# Lecture 3: NumPy and Pandas Foundations

## Overview

The transition from basic Python programming to data science begins with mastering the tools that make large-scale data analysis both possible and practical. This lecture introduces NumPy and Pandas - the foundational libraries that transform Python from a general-purpose programming language into a powerful data analysis environment.

NumPy provides the mathematical foundation for scientific computing, enabling efficient operations on large arrays of numerical data. Its power lies not just in speed, but in the way it allows you to think about data operations vectorially - applying functions to entire datasets rather than processing individual elements. This shift in thinking is fundamental to data science.

Pandas builds on NumPy's foundation to provide data structures and operations specifically designed for data analysis. The DataFrame, pandas' central data structure, mirrors the familiar spreadsheet metaphor while providing the power and flexibility needed for sophisticated analysis. Together, these libraries form the core of the Python data science ecosystem.

Understanding these tools is essential because real data science involves working with datasets too large for manual processing. The techniques you'll learn today - vectorized operations, data alignment, grouping, and transformation - are the building blocks for machine learning, statistical analysis, and data visualization. By the end of this lecture, you'll understand not just how to use these libraries, but when and why they're essential for professional data analysis.

## Learning Objectives

By the end of this lecture, students will be able to:

- Create and manipulate NumPy arrays for efficient numerical computing and statistical analysis
- Apply vectorized operations to perform calculations across entire datasets without explicit loops
- Load, explore, and clean datasets using Pandas DataFrames and Series
- Filter, sort, and transform data using Pandas operations for analytical workflows
- Perform grouping and aggregation operations to summarize data by categories
- Handle missing data and data quality issues using pandas built-in methods
- Integrate NumPy and Pandas workflows with the data structures and version control skills from previous lectures

## Prerequisites

This lecture builds on the foundation from Lectures 1 and 2, assuming proficiency with:

- Python data structures (lists, dictionaries, sets, tuples) and their appropriate usage
- Control structures, functions, and basic Python programming patterns
- Git workflows for tracking changes and collaborative development
- Command line operations and file management for data science projects

Students should be comfortable with mathematical concepts including basic statistics (mean, median, standard deviation) and understand how these concepts apply to analyzing datasets. Familiarity with spreadsheet operations (filtering, sorting, basic formulas) will help contextualize pandas operations.

## Core Concepts

### NumPy: The Mathematical Foundation

NumPy (Numerical Python) transforms Python into a powerful platform for numerical computing by providing efficient array operations and mathematical functions. The key insight behind NumPy is that most data analysis involves applying the same operation to many data points - exactly what vectorized operations excel at.

#### Understanding NumPy Arrays

NumPy arrays differ fundamentally from Python lists in both structure and performance. Where Python lists store references to objects scattered throughout memory, NumPy arrays store homogeneous data in contiguous memory blocks, enabling much faster operations.

```python
import numpy as np

# Creating arrays - the foundation of numerical analysis
temperature_readings = np.array([23.1, 25.4, 22.8, 24.7, 26.1, 23.9, 22.3])
patient_ages = np.array([34, 67, 45, 78, 23, 56, 41, 62])

print(f"Temperature readings: {temperature_readings}")
print(f"Array shape: {temperature_readings.shape}")
print(f"Data type: {temperature_readings.dtype}")
print(f"Memory usage: {temperature_readings.nbytes} bytes")

# Compare with equivalent Python list operations
import sys
python_list = [23.1, 25.4, 22.8, 24.7, 26.1, 23.9, 22.3]
print(f"Python list memory: {sys.getsizeof(python_list)} bytes")
```

The memory efficiency becomes crucial when working with large datasets. A NumPy array of one million numbers uses approximately 8MB of memory, while an equivalent Python list might use 80MB or more.

#### Vectorized Operations: Thinking in Arrays

Vectorized operations apply functions to entire arrays at once, eliminating the need for explicit loops and dramatically improving both performance and code readability.

```python
# Traditional Python approach with loops
def celsius_to_fahrenheit_loop(celsius_list):
    fahrenheit_list = []
    for temp in celsius_list:
        fahrenheit_list.append((temp * 9/5) + 32)
    return fahrenheit_list

# NumPy vectorized approach
def celsius_to_fahrenheit_vectorized(celsius_array):
    return (celsius_array * 9/5) + 32

# Performance demonstration
celsius_temps = np.array([20.5, 22.1, 24.7, 19.8, 26.3, 21.9, 23.4])
fahrenheit_temps = celsius_to_fahrenheit_vectorized(celsius_temps)

print(f"Celsius: {celsius_temps}")
print(f"Fahrenheit: {fahrenheit_temps}")

# Statistical operations are similarly vectorized
print(f"Mean temperature: {np.mean(celsius_temps):.1f}°C")
print(f"Standard deviation: {np.std(celsius_temps):.1f}°C")
print(f"Temperature range: {np.min(celsius_temps):.1f}°C to {np.max(celsius_temps):.1f}°C")
```

Vectorized operations extend beyond simple arithmetic to complex mathematical functions:

```python
# Advanced statistical operations
readings = np.array([22.1, 24.5, 23.8, 25.1, 22.9, 24.2, 23.6, 25.4, 22.7, 24.8])

# Calculate z-scores (standardization) - fundamental for many analyses
mean_temp = np.mean(readings)
std_temp = np.std(readings)
z_scores = (readings - mean_temp) / std_temp

print(f"Z-scores: {z_scores}")
print(f"Outliers (|z| > 2): {readings[np.abs(z_scores) > 2]}")

# Percentile calculations for distribution analysis
percentiles = np.percentile(readings, [25, 50, 75])
print(f"Quartiles: Q1={percentiles[0]:.1f}, Q2={percentiles[1]:.1f}, Q3={percentiles[2]:.1f}")
```

#### Multi-dimensional Arrays: Matrices and Beyond

Real datasets often have multiple dimensions - observations by variables, time series by locations, or images as pixel grids. NumPy handles multi-dimensional arrays naturally:

```python
# Temperature readings from multiple sensors over time
# Rows: time points, Columns: sensor locations
temperature_matrix = np.array([
    [22.1, 24.5, 23.2],  # Hour 1: Sensor A, B, C
    [22.8, 24.1, 23.8],  # Hour 2
    [23.4, 25.2, 24.1],  # Hour 3
    [23.1, 24.8, 23.7],  # Hour 4
    [22.6, 24.3, 23.4]   # Hour 5
])

print(f"Matrix shape: {temperature_matrix.shape}")  # (5 hours, 3 sensors)
print(f"Sensor A readings: {temperature_matrix[:, 0]}")  # All hours, sensor A
print(f"Hour 3 readings: {temperature_matrix[2, :]}")    # Hour 3, all sensors

# Statistical operations across dimensions
hourly_averages = np.mean(temperature_matrix, axis=1)  # Average across sensors
sensor_averages = np.mean(temperature_matrix, axis=0)  # Average across hours

print(f"Hourly averages: {hourly_averages}")
print(f"Sensor averages: {sensor_averages}")
```

Understanding array dimensions and axis operations is crucial for data analysis, as many real datasets have this structure.

#### Array Broadcasting: Flexible Operations

Broadcasting allows NumPy to perform operations on arrays of different shapes, following intuitive rules that mirror how we think about mathematical operations:

```python
# Temperature readings from three sensors
sensor_readings = np.array([22.1, 24.5, 23.2])

# Calibration adjustments for each sensor (broadcasting)
calibration_factors = np.array([1.02, 0.98, 1.01])
adjusted_readings = sensor_readings * calibration_factors

print(f"Original readings: {sensor_readings}")
print(f"Calibration factors: {calibration_factors}")
print(f"Adjusted readings: {adjusted_readings}")

# Broadcasting with temperature matrix
# Add a constant offset to all readings
temperature_offset = 0.5
adjusted_matrix = temperature_matrix + temperature_offset

# Apply different offsets to each sensor
sensor_offsets = np.array([0.2, -0.1, 0.3])
calibrated_matrix = temperature_matrix + sensor_offsets  # Broadcasts across rows

print(f"Original matrix shape: {temperature_matrix.shape}")
print(f"Sensor offsets shape: {sensor_offsets.shape}")
print(f"Result shape: {calibrated_matrix.shape}")
```

Broadcasting eliminates the need for explicit loops and makes array operations more intuitive and efficient.

### Pandas: Data Analysis Made Practical

Pandas transforms NumPy's numerical capabilities into a comprehensive data analysis toolkit. Built around two primary data structures - Series and DataFrame - pandas provides the tools needed for real-world data analysis workflows.

#### Series: Enhanced One-Dimensional Data

A pandas Series is like a NumPy array with labels, providing the foundation for more sophisticated data operations:

```python
import pandas as pd

# Create a Series with meaningful labels
temperature_series = pd.Series(
    data=[22.1, 24.5, 23.2, 25.1, 22.8],
    index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    name='Temperature_C'
)

print(temperature_series)
print(f"\nWednesday temperature: {temperature_series['Wednesday']}°C")
print(f"Above 23°C days: {temperature_series[temperature_series > 23].index.tolist()}")

# Statistical operations maintain labels
print(f"\nDescriptive statistics:")
print(temperature_series.describe())
```

Series provide labeled access to data, making operations more readable and less error-prone than working with raw arrays.

#### DataFrames: The Heart of Data Analysis

DataFrames represent tabular data - the format most familiar from spreadsheets and databases. They're essentially collections of Series that share the same index:

```python
# Patient monitoring data - typical DataFrame structure
patient_data = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
    'age': [34, 67, 45, 78, 23],
    'temperature': [36.8, 38.2, 37.1, 39.1, 36.9],
    'blood_pressure_systolic': [120, 140, 135, 160, 110],
    'blood_pressure_diastolic': [80, 90, 85, 95, 70],
    'condition': ['healthy', 'fever', 'mild_fever', 'high_fever', 'healthy']
})

print("Patient Monitoring Data:")
print(patient_data)

# Basic DataFrame exploration
print(f"\nDataFrame shape: {patient_data.shape}")
print(f"Column names: {patient_data.columns.tolist()}")
print(f"Data types:\n{patient_data.dtypes}")
```

DataFrames provide a familiar interface while enabling sophisticated operations that would be complex with traditional spreadsheet tools.

#### Data Loading and Initial Exploration

Real data analysis begins with loading data from external sources. Pandas provides robust tools for reading various file formats:

```python
# Simulating data loading (normally from CSV files)
# This would typically be: df = pd.read_csv('patient_data.csv')

# Create sample dataset for demonstration
climate_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=30, freq='D'),
    'location': ['NYC'] * 10 + ['LA'] * 10 + ['Chicago'] * 10,
    'temperature': np.random.normal(loc=20, scale=5, size=30),
    'humidity': np.random.normal(loc=65, scale=10, size=30),
    'pressure': np.random.normal(loc=1013, scale=15, size=30)
})

# Initial data exploration - crucial first steps
print("Climate Dataset Overview:")
print(f"Shape: {climate_data.shape}")
print(f"Columns: {climate_data.columns.tolist()}")

print("\nFirst few rows:")
print(climate_data.head())

print("\nData types and missing values:")
print(climate_data.info())

print("\nStatistical summary:")
print(climate_data.describe())
```

These exploration steps provide essential insights before beginning detailed analysis, helping identify data quality issues and analysis opportunities.

#### Data Selection and Filtering

Pandas provides multiple ways to select and filter data, each optimized for different scenarios:

```python
# Column selection
temperatures = climate_data['temperature']
location_temp = climate_data[['location', 'temperature']]

print("Temperature column:")
print(temperatures.head())

# Boolean indexing - fundamental for data analysis
high_temp_days = climate_data[climate_data['temperature'] > 25]
print(f"\nHigh temperature days: {len(high_temp_days)}")

# Complex filtering with multiple conditions
nyc_cold_days = climate_data[
    (climate_data['location'] == 'NYC') & 
    (climate_data['temperature'] < 15)
]

print(f"NYC cold days: {len(nyc_cold_days)}")

# .loc and .iloc for precise selection
# .loc uses labels, .iloc uses integer positions
recent_nyc_data = climate_data.loc[
    climate_data['location'] == 'NYC', 
    ['date', 'temperature', 'humidity']
]

print("\nRecent NYC data:")
print(recent_nyc_data)
```

Effective filtering is essential for focusing analysis on relevant subsets of data.

#### Data Transformation and Feature Engineering

Creating new variables and transforming existing ones is central to data analysis. Pandas provides efficient tools for these operations:

```python
# Creating new columns
climate_data['temperature_f'] = (climate_data['temperature'] * 9/5) + 32
climate_data['temp_category'] = pd.cut(
    climate_data['temperature'], 
    bins=[-np.inf, 10, 20, 30, np.inf],
    labels=['cold', 'cool', 'warm', 'hot']
)

# Date-based features
climate_data['month'] = climate_data['date'].dt.month
climate_data['day_of_week'] = climate_data['date'].dt.day_name()

print("Enhanced dataset with new features:")
print(climate_data[['date', 'temperature', 'temperature_f', 'temp_category', 'month']].head())

# Applying custom functions
def comfort_index(temp, humidity):
    """Calculate a simple comfort index."""
    return (temp * 1.8) + 32 - ((humidity / 100) * (temp * 0.55))

climate_data['comfort_index'] = climate_data.apply(
    lambda row: comfort_index(row['temperature'], row['humidity']), 
    axis=1
)

print(f"\nComfort index range: {climate_data['comfort_index'].min():.1f} to {climate_data['comfort_index'].max():.1f}")
```

Feature engineering transforms raw data into variables more suitable for analysis and modeling.

#### Grouping and Aggregation: Summarizing Data

One of pandas' most powerful features is the ability to group data by categories and perform aggregated calculations:

```python
# Group by location for analysis
location_summary = climate_data.groupby('location').agg({
    'temperature': ['mean', 'std', 'min', 'max'],
    'humidity': ['mean', 'median'],
    'pressure': ['mean'],
    'comfort_index': ['mean']
}).round(2)

print("Location-based climate summary:")
print(location_summary)

# Multiple grouping variables
monthly_location_summary = climate_data.groupby(['location', 'month']).agg({
    'temperature': 'mean',
    'humidity': 'mean'
}).round(1)

print("\nMonthly summaries by location:")
print(monthly_location_summary)

# Custom aggregation functions
def temperature_range(series):
    return series.max() - series.min()

custom_summary = climate_data.groupby('location').agg({
    'temperature': [temperature_range, 'count']
})

print("\nCustom aggregation - temperature ranges:")
print(custom_summary)
```

Grouping operations enable the kind of summary statistics essential for understanding patterns in data.

#### Handling Missing Data

Real datasets inevitably contain missing values. Pandas provides sophisticated tools for detecting, understanding, and handling missing data:

```python
# Introduce some missing values for demonstration
climate_with_missing = climate_data.copy()
climate_with_missing.loc[5:7, 'humidity'] = np.nan
climate_with_missing.loc[15:16, 'pressure'] = np.nan

# Detect missing values
print("Missing data summary:")
print(climate_with_missing.isnull().sum())

# Strategies for handling missing data
print("\nMissing data handling strategies:")

# 1. Drop rows with any missing values
cleaned_dropna = climate_with_missing.dropna()
print(f"Original rows: {len(climate_with_missing)}, After dropna: {len(cleaned_dropna)}")

# 2. Fill with specific values
filled_median = climate_with_missing.fillna({
    'humidity': climate_with_missing['humidity'].median(),
    'pressure': climate_with_missing['pressure'].median()
})

# 3. Forward fill or backward fill (useful for time series)
filled_forward = climate_with_missing.fillna(method='ffill')

# 4. Interpolation (sophisticated for time series)
filled_interpolated = climate_with_missing.interpolate()

print("Missing data successfully handled using multiple strategies")
```

Proper missing data handling prevents biased analysis results and maintains dataset integrity.

### Integration: NumPy and Pandas Working Together

NumPy and Pandas are designed to work seamlessly together, with pandas built on top of NumPy arrays. Understanding this relationship helps you choose the right tool for each task:

```python
# Pandas DataFrames use NumPy arrays internally
df_values = climate_data.select_dtypes(include=[np.number]).values
print(f"Underlying NumPy array shape: {df_values.shape}")
print(f"Data type: {df_values.dtype}")

# NumPy operations on pandas data
temperature_array = climate_data['temperature'].values
normalized_temps = (temperature_array - np.mean(temperature_array)) / np.std(temperature_array)

# Add normalized values back to DataFrame
climate_data['temperature_normalized'] = normalized_temps

print("Integration example - normalized temperatures:")
print(climate_data[['location', 'temperature', 'temperature_normalized']].head())

# Complex operations combining both libraries
# Calculate moving average using numpy on pandas data
window_size = 3
moving_avg = np.convolve(
    climate_data['temperature'].values, 
    np.ones(window_size)/window_size, 
    mode='valid'
)

# Add to DataFrame (handling size difference from convolution)
climate_data.loc[window_size-1:, 'temp_moving_avg'] = moving_avg

print(f"\nMoving average calculation complete")
print(climate_data[['date', 'temperature', 'temp_moving_avg']].tail())
```

This integration enables sophisticated analysis workflows that leverage the strengths of both libraries.

## Hands-On Practice

### Exercise 1: NumPy Array Operations for Statistical Analysis

Build proficiency with NumPy arrays through realistic statistical analysis scenarios.

```python
# Create file: exercises/numpy_statistics_practice.py
"""
NumPy Statistics Practice Exercise

This exercise builds familiarity with NumPy arrays and vectorized operations
through realistic statistical analysis scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt

def exercise_1_basic_array_operations():
    """
    Exercise 1: Basic array operations with medical data.
    
    Practice creating arrays, basic operations, and statistical functions
    using patient vital signs data.
    """
    print("Exercise 1: Medical Data Analysis with NumPy")
    print("-" * 45)
    
    # Patient vital signs data
    patient_temperatures = np.array([36.2, 37.8, 38.5, 36.9, 39.1, 37.2, 36.8, 38.9])
    patient_heart_rates = np.array([72, 88, 95, 76, 102, 81, 74, 92])
    patient_ages = np.array([34, 67, 45, 78, 23, 56, 41, 62])
    
    # TODO: Complete these statistical analyses
    
    # 1. Basic descriptive statistics
    temp_mean = np.mean(patient_temperatures)
    temp_std = np.std(patient_temperatures)
    
    print(f"Temperature Statistics:")
    print(f"  Mean: {temp_mean:.1f}°C")
    print(f"  Standard deviation: {temp_std:.1f}°C")
    
    # TODO: Calculate median, min, max for temperatures
    
    # 2. Identify outliers using z-scores
    temp_z_scores = (patient_temperatures - temp_mean) / temp_std
    outlier_threshold = 2.0
    
    # TODO: Find which patients have temperature outliers
    # outlier_indices = np.where(np.abs(temp_z_scores) > outlier_threshold)[0]
    
    # 3. Correlation analysis
    # TODO: Calculate correlation between age and heart rate
    # age_hr_correlation = np.corrcoef(patient_ages, patient_heart_rates)[0, 1]
    
    # 4. Vectorized operations
    # Convert temperatures to Fahrenheit
    temps_fahrenheit = (patient_temperatures * 9/5) + 32
    
    # Create age groups using vectorized operations
    # TODO: Create boolean arrays for different age groups
    # young_patients = patient_ages < 40
    # middle_aged_patients = (patient_ages >= 40) & (patient_ages < 65)
    # senior_patients = patient_ages >= 65
    
    print(f"Age group analysis:")
    # TODO: Print statistics for each age group
    
    print("✅ Basic array operations completed")

def exercise_2_multidimensional_analysis():
    """
    Exercise 2: Multi-dimensional array operations with time series data.
    
    Practice working with 2D arrays representing multiple sensors over time.
    """
    print("\nExercise 2: Multi-dimensional Time Series Analysis")
    print("-" * 52)
    
    # Temperature readings from 4 sensors over 7 days
    # Rows: days, Columns: sensors
    sensor_data = np.array([
        [22.1, 24.5, 23.2, 21.8],  # Day 1
        [22.8, 24.1, 23.8, 22.4],  # Day 2
        [23.4, 25.2, 24.1, 22.9],  # Day 3
        [23.1, 24.8, 23.7, 22.6],  # Day 4
        [22.6, 24.3, 23.4, 22.1],  # Day 5
        [23.9, 25.7, 24.8, 23.3],  # Day 6
        [24.2, 26.1, 25.1, 23.7]   # Day 7
    ])
    
    print(f"Sensor data shape: {sensor_data.shape}")
    
    # TODO: Complete these multi-dimensional analyses
    
    # 1. Calculate daily averages (across all sensors)
    daily_averages = np.mean(sensor_data, axis=1)
    
    # TODO: Calculate sensor averages (across all days)
    # sensor_averages = np.mean(sensor_data, axis=0)
    
    # 2. Find the day with highest overall temperature
    # TODO: Find which day had the maximum average temperature
    # max_temp_day = np.argmax(daily_averages) + 1  # +1 for 1-indexed day
    
    # 3. Sensor reliability analysis
    # Calculate standard deviation for each sensor across days
    # TODO: sensor_reliability = np.std(sensor_data, axis=0)
    
    # 4. Temperature trends
    # Calculate day-to-day changes for each sensor
    # TODO: daily_changes = np.diff(sensor_data, axis=0)
    
    # 5. Outlier detection across the entire dataset
    overall_mean = np.mean(sensor_data)
    overall_std = np.std(sensor_data)
    z_scores = (sensor_data - overall_mean) / overall_std
    
    # TODO: Find positions of outliers (|z-score| > 2)
    # outlier_positions = np.where(np.abs(z_scores) > 2)
    
    print(f"Overall temperature statistics:")
    print(f"  Mean: {overall_mean:.1f}°C")
    print(f"  Standard deviation: {overall_std:.1f}°C")
    
    print("✅ Multi-dimensional analysis completed")

def exercise_3_advanced_numpy_operations():
    """
    Exercise 3: Advanced NumPy operations for data science.
    
    Practice advanced array operations, broadcasting, and mathematical functions.
    """
    print("\nExercise 3: Advanced NumPy Operations")
    print("-" * 38)
    
    # Simulated experimental data
    np.random.seed(42)  # For reproducible results
    
    # Three experimental conditions, 20 observations each
    condition_a = np.random.normal(loc=100, scale=10, size=20)
    condition_b = np.random.normal(loc=105, scale=12, size=20)
    condition_c = np.random.normal(loc=95, scale=8, size=20)
    
    # Combine into 2D array
    experimental_data = np.column_stack([condition_a, condition_b, condition_c])
    
    print(f"Experimental data shape: {experimental_data.shape}")
    
    # TODO: Complete these advanced analyses
    
    # 1. Standardization (z-score normalization)
    # Calculate mean and std for each condition
    condition_means = np.mean(experimental_data, axis=0)
    condition_stds = np.std(experimental_data, axis=0)
    
    # Standardize using broadcasting
    standardized_data = (experimental_data - condition_means) / condition_stds
    
    # TODO: Verify standardization worked (means ≈ 0, stds ≈ 1)
    
    # 2. Percentile analysis
    # Calculate 25th, 50th, and 75th percentiles for each condition
    percentiles = [25, 50, 75]
    # TODO: condition_percentiles = np.percentile(experimental_data, percentiles, axis=0)
    
    # 3. Effect size calculation (Cohen's d between conditions)
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    # TODO: Calculate effect sizes between all condition pairs
    # effect_a_b = cohens_d(condition_a, condition_b)
    # effect_a_c = cohens_d(condition_a, condition_c)  
    # effect_b_c = cohens_d(condition_b, condition_c)
    
    # 4. Bootstrap sampling for confidence intervals
    n_bootstrap = 1000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # TODO: Create bootstrap sample and calculate mean
        # bootstrap_sample = np.random.choice(condition_a, size=len(condition_a), replace=True)
        # bootstrap_means.append(np.mean(bootstrap_sample))
        pass
    
    # TODO: Calculate 95% confidence interval
    # bootstrap_means = np.array(bootstrap_means)
    # ci_lower = np.percentile(bootstrap_means, 2.5)
    # ci_upper = np.percentile(bootstrap_means, 97.5)
    
    print("✅ Advanced NumPy operations completed")

if __name__ == "__main__":
    exercise_1_basic_array_operations()
    exercise_2_multidimensional_analysis()
    exercise_3_advanced_numpy_operations()
    
    print("\n" + "=" * 60)
    print("All NumPy exercises completed!")
    print("Next: Practice pandas operations with real datasets.")
```

### Exercise 2: Pandas Data Analysis Workflow

Apply pandas operations to realistic datasets while building professional analysis workflows.

```python
# Create file: exercises/pandas_analysis_practice.py
"""
Pandas Data Analysis Practice Exercise

This exercise builds proficiency with pandas DataFrames and analysis workflows
using realistic healthcare and climate datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_datasets():
    """Create sample datasets for practice exercises."""
    
    # Healthcare dataset
    np.random.seed(42)
    n_patients = 100
    
    healthcare_data = pd.DataFrame({
        'patient_id': [f'P{i:03d}' for i in range(1, n_patients+1)],
        'age': np.random.randint(18, 85, n_patients),
        'gender': np.random.choice(['M', 'F'], n_patients),
        'admission_date': pd.date_range('2024-01-01', periods=n_patients, freq='D'),
        'temperature': np.random.normal(37.0, 1.5, n_patients),
        'blood_pressure_sys': np.random.randint(90, 180, n_patients),
        'blood_pressure_dia': np.random.randint(60, 120, n_patients),
        'heart_rate': np.random.randint(60, 120, n_patients),
        'diagnosis': np.random.choice(['flu', 'pneumonia', 'diabetes', 'hypertension', 'healthy'], n_patients),
        'length_of_stay': np.random.randint(1, 10, n_patients)
    })
    
    # Introduce some missing values
    healthcare_data.loc[np.random.choice(n_patients, 5), 'blood_pressure_sys'] = np.nan
    healthcare_data.loc[np.random.choice(n_patients, 3), 'heart_rate'] = np.nan
    
    # Climate dataset
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    
    climate_data = []
    for location in locations:
        for date in dates[:73]:  # ~2.5 months per location
            climate_data.append({
                'date': date,
                'location': location,
                'temperature': np.random.normal(20, 10),  # Celsius
                'humidity': np.random.normal(60, 15),     # Percent
                'precipitation': np.random.exponential(2), # mm
                'wind_speed': np.random.exponential(5)    # km/h
            })
    
    climate_df = pd.DataFrame(climate_data)
    
    return healthcare_data, climate_df

def exercise_1_data_exploration():
    """
    Exercise 1: Data exploration and initial analysis.
    
    Practice loading data, understanding structure, and performing
    initial exploratory analysis.
    """
    print("Exercise 1: Healthcare Data Exploration")
    print("-" * 40)
    
    healthcare_data, _ = create_sample_datasets()
    
    # TODO: Complete these exploration tasks
    
    # 1. Basic dataset information
    print(f"Dataset shape: {healthcare_data.shape}")
    print(f"Column names: {healthcare_data.columns.tolist()}")
    
    # TODO: Display data types
    # print(f"Data types:\n{healthcare_data.dtypes}")
    
    # TODO: Check for missing values
    # print(f"Missing values:\n{healthcare_data.isnull().sum()}")
    
    # 2. Statistical summary
    print("\nNumerical columns summary:")
    numerical_summary = healthcare_data.describe()
    print(numerical_summary)
    
    # TODO: Categorical columns summary
    # print("\nCategorical columns summary:")
    # categorical_cols = healthcare_data.select_dtypes(include=['object']).columns
    # for col in categorical_cols:
    #     print(f"{col}: {healthcare_data[col].value_counts().head()}")
    
    # 3. Data quality checks
    # TODO: Check for impossible values
    # high_temp_patients = healthcare_data[healthcare_data['temperature'] > 42]
    # low_temp_patients = healthcare_data[healthcare_data['temperature'] < 35]
    
    # TODO: Check for logical inconsistencies
    # invalid_bp = healthcare_data[healthcare_data['blood_pressure_sys'] <= healthcare_data['blood_pressure_dia']]
    
    print("✅ Data exploration completed")

def exercise_2_data_filtering_transformation():
    """
    Exercise 2: Data filtering and transformation operations.
    
    Practice selecting subsets of data and creating new variables
    for analysis.
    """
    print("\nExercise 2: Data Filtering and Transformation")
    print("-" * 48)
    
    healthcare_data, climate_data = create_sample_datasets()
    
    # TODO: Complete these filtering and transformation tasks
    
    # 1. Basic filtering
    # Find patients with fever (temperature > 38°C)
    fever_patients = healthcare_data[healthcare_data['temperature'] > 38.0]
    print(f"Patients with fever: {len(fever_patients)}")
    
    # TODO: Find elderly patients (age >= 65) with hypertension
    # elderly_hypertension = healthcare_data[
    #     (healthcare_data['age'] >= 65) & 
    #     (healthcare_data['diagnosis'] == 'hypertension')
    # ]
    
    # 2. Creating new columns
    # BMI categories (simulated)
    healthcare_data['age_group'] = pd.cut(
        healthcare_data['age'], 
        bins=[0, 30, 50, 65, 100],
        labels=['young', 'middle_aged', 'senior', 'elderly']
    )
    
    # TODO: Create fever severity categories
    # healthcare_data['fever_severity'] = pd.cut(
    #     healthcare_data['temperature'],
    #     bins=[0, 37.5, 38.5, 40.0, 50],
    #     labels=['normal', 'low_fever', 'moderate_fever', 'high_fever']
    # )
    
    # 3. Date-based operations
    healthcare_data['admission_month'] = healthcare_data['admission_date'].dt.month
    healthcare_data['admission_weekday'] = healthcare_data['admission_date'].dt.day_name()
    
    # TODO: Calculate length of stay categories
    # healthcare_data['stay_category'] = pd.cut(
    #     healthcare_data['length_of_stay'],
    #     bins=[0, 2, 5, 10],
    #     labels=['short', 'medium', 'long']
    # )
    
    # 4. Complex transformations using .apply()
    def calculate_blood_pressure_risk(row):
        """Calculate blood pressure risk level."""
        sys_bp = row['blood_pressure_sys']
        dia_bp = row['blood_pressure_dia']
        
        if pd.isna(sys_bp) or pd.isna(dia_bp):
            return 'unknown'
        elif sys_bp >= 140 or dia_bp >= 90:
            return 'high'
        elif sys_bp >= 130 or dia_bp >= 80:
            return 'elevated'
        else:
            return 'normal'
    
    healthcare_data['bp_risk'] = healthcare_data.apply(calculate_blood_pressure_risk, axis=1)
    
    # TODO: Create a composite health score
    # def calculate_health_score(row):
    #     """Calculate composite health score (0-100)."""
    #     # Implementation here
    #     pass
    # 
    # healthcare_data['health_score'] = healthcare_data.apply(calculate_health_score, axis=1)
    
    print(f"New columns created: {healthcare_data.columns[-4:].tolist()}")
    print("✅ Data filtering and transformation completed")

def exercise_3_grouping_aggregation():
    """
    Exercise 3: Grouping and aggregation operations.
    
    Practice summarizing data by groups and calculating
    meaningful aggregate statistics.
    """
    print("\nExercise 3: Grouping and Aggregation Analysis")
    print("-" * 47)
    
    healthcare_data, climate_data = create_sample_datasets()
    
    # Add calculated columns from previous exercise
    healthcare_data['age_group'] = pd.cut(healthcare_data['age'], bins=[0, 30, 50, 65, 100], 
                                        labels=['young', 'middle_aged', 'senior', 'elderly'])
    
    # TODO: Complete these grouping and aggregation tasks
    
    # 1. Basic grouping by diagnosis
    diagnosis_summary = healthcare_data.groupby('diagnosis').agg({
        'age': ['mean', 'std'],
        'temperature': ['mean', 'max'],
        'length_of_stay': ['mean', 'median'],
        'patient_id': 'count'
    }).round(2)
    
    print("Summary by diagnosis:")
    print(diagnosis_summary)
    
    # TODO: Grouping by age group
    # age_group_summary = healthcare_data.groupby('age_group').agg({
    #     'temperature': ['mean', 'std'],
    #     'blood_pressure_sys': 'mean',
    #     'heart_rate': 'mean',
    #     'length_of_stay': 'mean'
    # }).round(1)
    
    # 2. Multiple grouping variables
    # TODO: Group by both diagnosis and age group
    # diagnosis_age_summary = healthcare_data.groupby(['diagnosis', 'age_group']).size().unstack(fill_value=0)
    
    # 3. Custom aggregation functions
    def coefficient_of_variation(series):
        """Calculate coefficient of variation."""
        return series.std() / series.mean() if series.mean() != 0 else 0
    
    custom_agg = healthcare_data.groupby('diagnosis').agg({
        'temperature': [coefficient_of_variation, 'count'],
        'age': ['min', 'max']
    })
    
    print("\nCustom aggregation results:")
    print(custom_agg)
    
    # 4. Time-based grouping
    # TODO: Group by admission month
    # monthly_admissions = healthcare_data.groupby('admission_month').agg({
    #     'patient_id': 'count',
    #     'length_of_stay': 'mean',
    #     'temperature': 'mean'
    # }).round(1)
    
    # 5. Climate data grouping
    # TODO: Analyze climate patterns by location
    # climate_summary = climate_data.groupby('location').agg({
    #     'temperature': ['mean', 'std', 'min', 'max'],
    #     'humidity': ['mean', 'median'],
    #     'precipitation': ['sum', 'mean'],
    #     'wind_speed': 'mean'
    # }).round(2)
    
    print("✅ Grouping and aggregation completed")

def exercise_4_missing_data_handling():
    """
    Exercise 4: Missing data detection and handling.
    
    Practice identifying missing data patterns and applying
    appropriate strategies for handling missing values.
    """
    print("\nExercise 4: Missing Data Handling")
    print("-" * 35)
    
    healthcare_data, _ = create_sample_datasets()
    
    # TODO: Complete these missing data tasks
    
    # 1. Missing data analysis
    print("Missing data summary:")
    missing_summary = healthcare_data.isnull().sum()
    print(missing_summary[missing_summary > 0])
    
    # TODO: Calculate missing data percentages
    # missing_percentages = (healthcare_data.isnull().sum() / len(healthcare_data)) * 100
    # print(f"Missing data percentages:\n{missing_percentages[missing_percentages > 0]}")
    
    # 2. Missing data patterns
    # TODO: Check if missing data is related to other variables
    # missing_bp_by_diagnosis = healthcare_data.groupby('diagnosis')['blood_pressure_sys'].apply(
    #     lambda x: x.isnull().sum()
    # )
    
    # 3. Different strategies for handling missing data
    
    # Strategy 1: Drop rows with missing values
    cleaned_dropna = healthcare_data.dropna()
    print(f"Original rows: {len(healthcare_data)}, After dropna: {len(cleaned_dropna)}")
    
    # Strategy 2: Fill with statistical measures
    filled_median = healthcare_data.copy()
    filled_median['blood_pressure_sys'] = filled_median['blood_pressure_sys'].fillna(
        filled_median['blood_pressure_sys'].median()
    )
    filled_median['heart_rate'] = filled_median['heart_rate'].fillna(
        filled_median['heart_rate'].mean()
    )
    
    # TODO: Strategy 3: Fill based on group characteristics
    # filled_group = healthcare_data.copy()
    # filled_group['blood_pressure_sys'] = filled_group.groupby('diagnosis')['blood_pressure_sys'].transform(
    #     lambda x: x.fillna(x.median())
    # )
    
    # 4. Advanced imputation
    # TODO: Use interpolation for time-series like data
    # healthcare_data_sorted = healthcare_data.sort_values('admission_date')
    # healthcare_data_sorted['temperature_interpolated'] = healthcare_data_sorted['temperature'].interpolate()
    
    print("✅ Missing data handling completed")

if __name__ == "__main__":
    exercise_1_data_exploration()
    exercise_2_data_filtering_transformation()
    exercise_3_grouping_aggregation()
    exercise_4_missing_data_handling()
    
    print("\n" + "=" * 60)
    print("All pandas exercises completed!")
    print("Practice integrating these skills with NumPy for advanced analysis.")
```

### Exercise 3: Integration Challenge - Complete Data Analysis Pipeline

Combine NumPy, Pandas, and previous skills in a comprehensive data analysis project.

```python
# Create file: exercises/complete_analysis_pipeline.py
"""
Complete Data Analysis Pipeline Integration Challenge

This exercise combines NumPy, Pandas, data structures, and version control
in a realistic data science project workflow.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class HealthcareAnalyticsPipeline:
    """
    Complete healthcare analytics pipeline demonstrating integration
    of NumPy, Pandas, and professional development practices.
    """
    
    def __init__(self, config_path: str = 'config/analysis_config.json'):
        """Initialize pipeline with configuration."""
        
        # Load configuration (using data structures from Lecture 2)
        self.config = {
            'data_quality': {
                'temperature_range': [35.0, 42.0],
                'heart_rate_range': [40, 200],
                'bp_sys_range': [70, 250],
                'bp_dia_range': [40, 150]
            },
            'analysis_parameters': {
                'fever_threshold': 38.0,
                'hypertension_sys_threshold': 140,
                'hypertension_dia_threshold': 90,
                'elderly_age_threshold': 65
            },
            'output_settings': {
                'decimal_places': 2,
                'save_intermediates': True,
                'export_format': 'json'
            }
        }
        
        # Data structures for tracking analysis (from Lecture 2)
        self.processed_files: set = set()  # Track processed files
        self.analysis_history: List[Dict] = []  # Ordered processing steps
        self.results_cache: Dict[str, any] = {}  # Cache results
        
        # Analysis results storage
        self.data_quality_report: Optional[Dict] = None
        self.descriptive_statistics: Optional[Dict] = None
        self.patient_risk_scores: Optional[pd.DataFrame] = None
        
    def load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """
        Load healthcare data and perform initial validation.
        
        This function demonstrates pandas data loading and initial
        quality checks using NumPy operations.
        """
        # TODO: Implement data loading and validation
        
        # Record processing step
        self.analysis_history.append({
            'timestamp': datetime.now().isoformat(),
            'step': 'data_loading',
            'file': data_path,
            'status': 'started'
        })
        
        try:
            # Simulate data loading (normally pd.read_csv(data_path))
            # Create realistic healthcare dataset
            np.random.seed(42)
            n_patients = 500
            
            data = pd.DataFrame({
                'patient_id': [f'P{i:04d}' for i in range(1, n_patients+1)],
                'age': np.random.randint(18, 90, n_patients),
                'gender': np.random.choice(['M', 'F'], n_patients, p=[0.48, 0.52]),
                'admission_date': pd.date_range('2024-01-01', periods=n_patients, freq='H')[:n_patients],
                'temperature': np.random.normal(37.2, 1.2, n_patients),
                'heart_rate': np.random.randint(55, 130, n_patients),
                'blood_pressure_systolic': np.random.randint(95, 180, n_patients),
                'blood_pressure_diastolic': np.random.randint(60, 110, n_patients),
                'diagnosis': np.random.choice(
                    ['healthy', 'flu', 'pneumonia', 'hypertension', 'diabetes'], 
                    n_patients, p=[0.3, 0.2, 0.15, 0.2, 0.15]
                ),
                'length_of_stay': np.random.poisson(3, n_patients) + 1
            })
            
            # Introduce realistic missing values
            missing_indices = np.random.choice(n_patients, size=int(n_patients*0.02), replace=False)
            data.loc[missing_indices, 'blood_pressure_systolic'] = np.nan
            
            # Add the file to processed files set
            self.processed_files.add(data_path)
            
            # Update processing history
            self.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'step': 'data_loading',
                'file': data_path,
                'status': 'completed',
                'records_loaded': len(data),
                'columns': list(data.columns)
            })
            
            return data
            
        except Exception as e:
            self.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'step': 'data_loading',
                'file': data_path,
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    def perform_data_quality_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality analysis using NumPy and Pandas.
        
        Demonstrates integration of both libraries for quality assessment.
        """
        # TODO: Implement comprehensive data quality analysis
        
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'missing_data': {},
            'outliers': {},
            'data_consistency': {},
            'summary_statistics': {}
        }
        
        # 1. Missing data analysis (Pandas)
        missing_counts = data.isnull().sum()
        quality_report['missing_data'] = {
            col: int(count) for col, count in missing_counts.items() if count > 0
        }
        
        # 2. Outlier detection using NumPy statistical functions
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in data.columns:
                values = data[col].dropna().values  # Convert to NumPy array
                
                # Use NumPy for statistical calculations
                mean_val = np.mean(values)
                std_val = np.std(values)
                z_scores = np.abs((values - mean_val) / std_val)
                
                outlier_count = np.sum(z_scores > 3.0)  # Count outliers
                
                quality_report['outliers'][col] = {
                    'count': int(outlier_count),
                    'percentage': float((outlier_count / len(values)) * 100),
                    'mean': float(mean_val),
                    'std': float(std_val)
                }
        
        # 3. Data consistency checks
        # TODO: Check for logical inconsistencies
        # - Blood pressure: systolic should be > diastolic
        # - Temperature: should be within biological range
        # - Age: should be positive and reasonable
        
        # 4. Summary statistics using both libraries
        numerical_summary = data.describe()
        quality_report['summary_statistics'] = {
            col: {
                'count': int(numerical_summary.loc['count', col]),
                'mean': float(numerical_summary.loc['mean', col]),
                'std': float(numerical_summary.loc['std', col]),
                'min': float(numerical_summary.loc['min', col]),
                'max': float(numerical_summary.loc['max', col])
            } for col in numerical_summary.columns
        }
        
        # Cache results
        self.results_cache['data_quality_report'] = quality_report
        self.data_quality_report = quality_report
        
        return quality_report
    
    def calculate_patient_risk_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive patient risk scores.
        
        Demonstrates advanced pandas operations with NumPy calculations.
        """
        # TODO: Implement patient risk scoring system
        
        risk_data = data.copy()
        
        # 1. Age-based risk (using NumPy for calculations)
        ages = risk_data['age'].values
        age_risk = np.where(ages < 30, 0.1,
                   np.where(ages < 50, 0.3,
                   np.where(ages < 70, 0.5, 0.8)))
        
        risk_data['age_risk'] = age_risk
        
        # 2. Vital signs risk (combining pandas and NumPy)
        # Temperature risk
        temp_values = risk_data['temperature'].values
        fever_threshold = self.config['analysis_parameters']['fever_threshold']
        temp_risk = np.where(temp_values < fever_threshold, 0.0,
                    np.where(temp_values < fever_threshold + 1, 0.3,
                    np.where(temp_values < fever_threshold + 2, 0.7, 1.0)))
        
        risk_data['temperature_risk'] = temp_risk
        
        # Heart rate risk
        # TODO: Calculate heart rate risk based on age-adjusted norms
        
        # Blood pressure risk  
        # TODO: Calculate BP risk using both systolic and diastolic values
        
        # 3. Diagnosis-based risk (using pandas categorical operations)
        diagnosis_risk_map = {
            'healthy': 0.0,
            'flu': 0.2,
            'pneumonia': 0.7,
            'hypertension': 0.5,
            'diabetes': 0.6
        }
        
        risk_data['diagnosis_risk'] = risk_data['diagnosis'].map(diagnosis_risk_map)
        
        # 4. Composite risk score (NumPy weighted average)
        risk_columns = ['age_risk', 'temperature_risk', 'diagnosis_risk']
        weights = np.array([0.3, 0.4, 0.3])  # Adjust weights as needed
        
        # Calculate weighted average using NumPy
        risk_matrix = risk_data[risk_columns].values
        composite_risk = np.average(risk_matrix, axis=1, weights=weights)
        
        risk_data['composite_risk_score'] = composite_risk
        
        # 5. Risk categories (using pandas categorical)
        risk_data['risk_category'] = pd.cut(
            risk_data['composite_risk_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
        
        # Store results
        self.patient_risk_scores = risk_data
        self.results_cache['patient_risk_scores'] = risk_data.to_dict('records')
        
        return risk_data
    
    def perform_cohort_analysis(self, risk_data: pd.DataFrame) -> Dict:
        """
        Advanced cohort analysis using pandas grouping operations.
        
        Demonstrates sophisticated grouping and aggregation with NumPy calculations.
        """
        # TODO: Implement comprehensive cohort analysis
        
        cohort_analysis = {}
        
        # 1. Risk category analysis
        risk_summary = risk_data.groupby('risk_category').agg({
            'patient_id': 'count',
            'age': ['mean', 'std'],
            'temperature': ['mean', 'std'],
            'composite_risk_score': ['mean', 'std'],
            'length_of_stay': ['mean', 'median']
        }).round(3)
        
        # Convert to nested dictionary for JSON serialization
        cohort_analysis['risk_category_summary'] = {}
        for risk_cat in risk_summary.index:
            cohort_analysis['risk_category_summary'][risk_cat] = {
                'patient_count': int(risk_summary.loc[risk_cat, ('patient_id', 'count')]),
                'avg_age': float(risk_summary.loc[risk_cat, ('age', 'mean')]),
                'avg_temperature': float(risk_summary.loc[risk_cat, ('temperature', 'mean')]),
                'avg_risk_score': float(risk_summary.loc[risk_cat, ('composite_risk_score', 'mean')]),
                'avg_length_of_stay': float(risk_summary.loc[risk_cat, ('length_of_stay', 'mean')])
            }
        
        # 2. Diagnosis-based analysis
        # TODO: Analyze outcomes by diagnosis
        
        # 3. Age group analysis
        # TODO: Create age groups and analyze risk patterns
        
        # 4. Temporal analysis (if admission dates are meaningful)
        # TODO: Analyze trends over time
        
        # Store results
        self.results_cache['cohort_analysis'] = cohort_analysis
        
        return cohort_analysis
    
    def generate_analysis_report(self, output_dir: str = 'results') -> str:
        """
        Generate comprehensive analysis report.
        
        Demonstrates file I/O and report generation using all previous results.
        """
        # TODO: Create comprehensive report
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Compile comprehensive report
        report = {
            'analysis_metadata': {
                'generated_at': datetime.now().isoformat(),
                'pipeline_version': '1.0',
                'processed_files': list(self.processed_files),
                'analysis_steps': len(self.analysis_history)
            },
            'data_quality_report': self.data_quality_report,
            'analysis_history': self.analysis_history,
            'patient_count': len(self.patient_risk_scores) if self.patient_risk_scores is not None else 0,
            'results_summary': {
                'cached_results': len(self.results_cache),
                'analysis_completed': True
            }
        }
        
        # Save detailed report
        report_path = os.path.join(output_dir, 'analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save patient risk scores if available
        if self.patient_risk_scores is not None:
            csv_path = os.path.join(output_dir, 'patient_risk_scores.csv')
            self.patient_risk_scores.to_csv(csv_path, index=False)
        
        return report_path


def main():
    """
    Main analysis workflow demonstrating integration of all concepts.
    
    This workflow should be managed with Git commits:
    1. Initial data loading and quality assessment
    2. Risk score calculation and validation
    3. Cohort analysis and insights generation  
    4. Report generation and results export
    """
    print("Healthcare Analytics Pipeline - Complete Integration")
    print("=" * 55)
    
    # Initialize pipeline
    pipeline = HealthcareAnalyticsPipeline()
    
    try:
        # Step 1: Data Loading and Quality Assessment
        print("Step 1: Loading and validating data...")
        data = pipeline.load_and_validate_data('data/healthcare_data.csv')
        
        print(f"Loaded {len(data)} patient records with {len(data.columns)} variables")
        # Git: git add . && git commit -m "Load healthcare data and complete initial validation"
        
        # Step 2: Data Quality Analysis
        print("\nStep 2: Performing data quality analysis...")
        quality_report = pipeline.perform_data_quality_analysis(data)
        
        missing_data_cols = len(quality_report['missing_data'])
        print(f"Quality analysis complete: {missing_data_cols} columns with missing data")
        # Git: git add . && git commit -m "Complete comprehensive data quality analysis"
        
        # Step 3: Risk Score Calculation
        print("\nStep 3: Calculating patient risk scores...")
        risk_data = pipeline.calculate_patient_risk_scores(data)
        
        high_risk_count = len(risk_data[risk_data['risk_category'] == 'high'])
        print(f"Risk scoring complete: {high_risk_count} high-risk patients identified")
        # Git: git add . && git commit -m "Implement patient risk scoring system"
        
        # Step 4: Cohort Analysis
        print("\nStep 4: Performing cohort analysis...")
        cohort_results = pipeline.perform_cohort_analysis(risk_data)
        
        risk_categories = len(cohort_results.get('risk_category_summary', {}))
        print(f"Cohort analysis complete: {risk_categories} risk categories analyzed")
        # Git: git add . && git commit -m "Complete cohort analysis and risk stratification"
        
        # Step 5: Report Generation
        print("\nStep 5: Generating analysis reports...")
        report_path = pipeline.generate_analysis_report()
        
        print(f"Analysis complete! Report saved to: {report_path}")
        # Git: git add . && git commit -m "Generate comprehensive analysis report and export results"
        
        # Final summary
        print(f"\n" + "=" * 55)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 55)
        print(f"Files processed: {len(pipeline.processed_files)}")
        print(f"Analysis steps: {len(pipeline.analysis_history)}")
        print(f"Results cached: {len(pipeline.results_cache)}")
        print(f"Patients analyzed: {len(risk_data)}")
        
        print(f"\nKey Insights:")
        if pipeline.data_quality_report:
            missing_cols = len(pipeline.data_quality_report.get('missing_data', {}))
            print(f"• Data quality: {missing_cols} columns require attention")
        
        if 'risk_category_summary' in cohort_results:
            risk_summary = cohort_results['risk_category_summary']
            for category, stats in risk_summary.items():
                print(f"• {category.title()} risk patients: {stats['patient_count']}")
        
        print(f"\nNext steps:")
        print("1. Review data quality issues and implement corrections")
        print("2. Validate risk scoring algorithm with clinical experts")
        print("3. Develop intervention strategies for high-risk patients")
        print("4. Set up monitoring dashboard for ongoing analysis")
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        # Git: git add . && git commit -m "Handle pipeline failure - investigate and fix"


if __name__ == "__main__":
    main()
```

## Real-World Applications

NumPy and Pandas form the foundation for virtually all Python-based data analysis across industries and research domains. Their integration enables sophisticated workflows that scale from exploratory analysis to production systems.

**Healthcare Analytics**: Medical institutions use pandas to manage patient records, clinical trial data, and electronic health records while NumPy powers statistical analyses and risk modeling. The combination enables real-time patient monitoring systems that process thousands of vital signs readings per minute.

**Financial Services**: Investment firms rely on pandas for portfolio management, market data analysis, and risk assessment while NumPy handles the mathematical computations underlying trading algorithms and risk models. High-frequency trading systems process millions of market data points using these tools.

**Climate Research**: Environmental scientists use pandas to organize multi-dimensional climate datasets (temperature, pressure, humidity across locations and time) while NumPy performs the numerical modeling and statistical analysis. Climate models processing decades of observational data depend on these libraries for efficiency.

**Business Intelligence**: Companies across industries use pandas to transform raw business data into actionable insights - sales trends, customer behavior, operational efficiency. NumPy enables the statistical modeling that powers machine learning recommendations and forecasting systems.

**Academic Research**: Researchers across disciplines - from psychology to physics - use these tools to analyze experimental data, conduct statistical tests, and validate hypotheses. The combination of structured data manipulation (pandas) with mathematical operations (NumPy) makes complex analyses accessible.

## Assessment Integration

### Formative Assessment

Understanding check questions throughout the lecture:

1. **NumPy Efficiency**: "Explain why NumPy arrays are more memory-efficient than Python lists for numerical data, and when this efficiency matters most in data science."

2. **Vectorized Operations**: "Given a list of patient temperatures, describe how you would identify outliers using both traditional Python loops and NumPy vectorized operations. What are the advantages of each approach?"

3. **Pandas Data Structures**: "When would you choose a pandas Series over a DataFrame, and vice versa? Provide a concrete example from healthcare or climate data analysis."

4. **Integration Thinking**: "Describe a scenario where you would use NumPy arrays within a pandas DataFrame operation. Why would this combination be beneficial?"

### Summative Assessment Preview

Your assignment combines these tools in a comprehensive data analysis project:

- **Data Pipeline Development**: Create a complete analysis workflow using both NumPy and Pandas for a real dataset
- **Statistical Analysis**: Implement statistical functions using NumPy and organize results using Pandas
- **Data Quality Assessment**: Use pandas for data exploration and NumPy for outlier detection and validation
- **Professional Documentation**: Create analysis reports that demonstrate understanding of when and why to use each tool

This mirrors professional data science projects where tool selection, efficiency, and reproducibility are essential for success.

## Further Reading and Resources

### Essential Resources
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) - Comprehensive documentation with tutorials and examples
- [Pandas Documentation](https://pandas.pydata.org/docs/) - Complete guide to pandas data structures and operations
- [Python for Data Analysis](https://wesmckinney.com/pages/book.html) - Authoritative guide by pandas creator Wes McKinney

### Advanced Topics
- [NumPy Performance Optimization](https://numpy.org/doc/stable/user/theory.broadcasting.html) - Understanding broadcasting and vectorization
- [Pandas Performance Tips](https://pandas.pydata.org/docs/user_guide/enhancingperf.html) - Optimizing pandas operations for large datasets
- [Scientific Python Ecosystem](https://scipy.org/) - Integration with SciPy, Matplotlib, and other libraries

### Practice Environments
- [NumPy Tutorials](https://numpy.org/numpy-tutorials/) - Interactive tutorials for hands-on learning
- [Pandas Exercises](https://github.com/guipsamora/pandas_exercises) - Comprehensive practice problems
- [Kaggle Learn](https://www.kaggle.com/learn/pandas) - Real-world datasets for pandas practice

## Next Steps

In our next lecture, **Data Visualization and Statistical Analysis**, you'll see how today's numerical computing foundation enables sophisticated visual and statistical analysis. The data manipulation skills you've learned directly support advanced topics:

- Matplotlib and Seaborn build on NumPy arrays for creating publication-quality visualizations
- Statistical analysis libraries use pandas DataFrames as their primary input format  
- The data cleaning and transformation techniques from today enable meaningful visualization and modeling
- Version control becomes essential for managing analysis scripts that combine multiple libraries

The computational foundation you've built today - efficient numerical operations and structured data manipulation - makes advanced data science techniques both accessible and practical.

Start applying these tools to your own data immediately. The muscle memory developed through regular practice with NumPy and Pandas distinguishes competent data scientists from those who struggle with real-world analytical challenges.

---

*Lecture Format: Notion-Compatible Narrative with Embedded Interactive Code*
*Progressive Complexity: Fundamentals → Integration → Real-World Applications*  
*Version: 2.0 - Generated by automated conversion tool*