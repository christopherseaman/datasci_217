# Notebook 2: Wrangling & Feature Engineering

**Phases 4-5:** Data Wrangling & Transformation, Feature Engineering & Aggregation

**Dataset:** NYC Taxi Trip Dataset (continuing from Notebook 1)

**Focus:** Transforming and enriching data - merging datasets, working with datetime data, reshaping, and creating features for modeling.

---

## Phase 4: Data Wrangling & Transformation

### Learning Objectives
- Merge and join multiple datasets
- Handle datetime columns and set datetime index
- Extract time-based features
- Reshape data for analysis
- Work with indexes

### Step 1: Load Cleaned Data from Previous Notebook

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline

# Load cleaned data from Notebook 1
df = pd.read_csv('../output/01_cleaned_taxi_data.csv')
print(f"Loaded {len(df):,} cleaned taxi trips")
print(f"Date range: {df['pickup_datetime'].min()} to {df['pickup_datetime'].max()}")
```

### Step 2: Convert to Datetime and Set Datetime Index

```python
# Convert datetime columns
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

# Recalculate trip_duration if needed
df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60

# Set pickup_datetime as index for datetime-based operations
df_ts = df.set_index('pickup_datetime').sort_index()

print(f"Datetime index set. Shape: {df_ts.shape}")
print(f"Index range: {df_ts.index.min()} to {df_ts.index.max()}")
display(df_ts.head())
```

### Step 3: Extract Time-Based Features

```python
# Extract various time-based features from the datetime index
df_ts['hour'] = df_ts.index.hour
df_ts['day_of_week'] = df_ts.index.dayofweek  # 0=Monday, 6=Sunday
df_ts['day_name'] = df_ts.index.day_name()
df_ts['month'] = df_ts.index.month
df_ts['month_name'] = df_ts.index.month_name()
df_ts['year'] = df_ts.index.year
df_ts['is_weekend'] = df_ts['day_of_week'].isin([5, 6]).astype(int)

# Create time-of-day categories
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df_ts['time_of_day'] = df_ts['hour'].apply(get_time_of_day)

print("Time-based features extracted:")
print(df_ts[['hour', 'day_of_week', 'day_name', 'month', 'is_weekend', 'time_of_day']].head(10))
```

### Step 4: Merge with Additional Data (Zone Lookup Table)

```python
# In practice, you'd load a real zone lookup table
# For this demo, we'll create a simplified zone lookup based on coordinates

# Create zone lookup table (simplified - in practice this would be a real NYC zone file)
zone_lookup = pd.DataFrame({
    'zone_id': range(1, 21),
    'zone_name': [f'Zone_{i}' for i in range(1, 21)],
    'borough': np.random.choice(['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'], 20)
})

# Create pickup and dropoff zones based on coordinates (simplified)
# In practice, you'd use actual zone boundaries
np.random.seed(42)
df_ts['pickup_zone_id'] = np.random.choice(zone_lookup['zone_id'], len(df_ts))
df_ts['dropoff_zone_id'] = np.random.choice(zone_lookup['zone_id'], len(df_ts))

# Merge pickup zone information using LEFT JOIN
# LEFT JOIN keeps all rows from left DataFrame (df_ts), adds matching data from right (zone_lookup)
# This is the most common join type - we want all trips, even if zone info is missing
# IMPORTANT: Reset index before merge, then set it back to preserve DatetimeIndex
df_ts_reset = df_ts.reset_index()
df_ts_reset = df_ts_reset.merge(
    zone_lookup.rename(columns={'zone_id': 'pickup_zone_id', 'zone_name': 'pickup_zone_name', 'borough': 'pickup_borough'}),
    on='pickup_zone_id',
    how='left'  # LEFT JOIN: keep all trips, add zone info where available
)

# Merge dropoff zone information
df_ts_reset = df_ts_reset.merge(
    zone_lookup.rename(columns={'zone_id': 'dropoff_zone_id', 'zone_name': 'dropoff_zone_name', 'borough': 'dropoff_borough'}),
    on='dropoff_zone_id',
    how='left'  # LEFT JOIN: keep all trips
)

# Set datetime index back
df_ts = df_ts_reset.set_index('pickup_datetime').sort_index()

print("Zone information merged:")
print(f"Columns: {df_ts.shape[1]}")
display(df_ts[['pickup_zone_name', 'pickup_borough', 'dropoff_zone_name', 'dropoff_borough']].head())

# Demonstrate other join types (for educational purposes)
print("\n" + "=" * 60)
print("JOIN TYPE EXAMPLES (Educational)")
print("=" * 60)

# Create example DataFrames to demonstrate join types
left_df = pd.DataFrame({'key': [1, 2, 3, 4], 'left_value': ['A', 'B', 'C', 'D']})
right_df = pd.DataFrame({'key': [2, 3, 4, 5], 'right_value': ['X', 'Y', 'Z', 'W']})

print("Left DataFrame:")
display(left_df)
print("\nRight DataFrame:")
display(right_df)

# INNER JOIN: Only rows with matching keys in both DataFrames
inner_result = pd.merge(left_df, right_df, on='key', how='inner')
print("\nINNER JOIN (only matching keys):")
display(inner_result)

# LEFT JOIN: All rows from left, matching from right
left_result = pd.merge(left_df, right_df, on='key', how='left')
print("\nLEFT JOIN (all from left, matching from right):")
display(left_result)

# RIGHT JOIN: All rows from right, matching from left
right_result = pd.merge(left_df, right_df, on='key', how='right')
print("\nRIGHT JOIN (all from right, matching from left):")
display(right_result)

# OUTER JOIN: All rows from both DataFrames
outer_result = pd.merge(left_df, right_df, on='key', how='outer')
print("\nOUTER JOIN (all rows from both):")
display(outer_result)
```

### Step 5: Reshape Data - Pivot Table Example

```python
# Create a pivot table: Average fare by day of week and time of day
pivot_fare = df_ts.pivot_table(
    values='fare_amount',
    index='day_name',
    columns='time_of_day',
    aggfunc='mean'
)

print("Average Fare by Day of Week and Time of Day:")
display(pivot_fare.round(2))

# Visualize the pivot table
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_fare, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Average Fare ($)'})
plt.title('Average Fare Amount by Day and Time of Day', fontsize=14, fontweight='bold')
plt.xlabel('Time of Day')
plt.ylabel('Day of Week')
plt.tight_layout()
plt.show()
```

### Step 6: Reshape Data - Melt Example

```python
# Example: Convert wide format to long format
# Let's create a summary by hour with multiple metrics

hourly_summary = df_ts.groupby('hour').agg({
    'fare_amount': 'mean',
    'trip_distance': 'mean',
    'trip_duration': 'mean',
    'passenger_count': 'mean'
}).reset_index()

print("Hourly Summary (Wide Format):")
display(hourly_summary.head())

# Melt to long format
hourly_long = hourly_summary.melt(
    id_vars='hour',
    value_vars=['fare_amount', 'trip_distance', 'trip_duration', 'passenger_count'],
    var_name='metric',
    value_name='value'
)

print("\nHourly Summary (Long Format):")
display(hourly_long.head(10))
```

---

## Phase 5: Feature Engineering & Aggregation

### Learning Objectives
- Create derived features
- Perform groupby aggregations
- Calculate rolling window statistics
- Create time-based features
- Aggregate by multiple dimensions

### Step 1: Create Derived Features

```python
# Speed (miles per hour) - derived from distance and duration
df_ts['speed_mph'] = df_ts['trip_distance'] / (df_ts['trip_duration'] / 60)  # Convert minutes to hours
df_ts['speed_mph'] = df_ts['speed_mph'].replace([np.inf, -np.inf], np.nan)  # Handle division by zero
df_ts['speed_mph'] = df_ts['speed_mph'].clip(upper=60)  # Cap at 60 mph (reasonable for NYC)

# Fare per mile
df_ts['fare_per_mile'] = df_ts['fare_amount'] / df_ts['trip_distance']
df_ts['fare_per_mile'] = df_ts['fare_per_mile'].replace([np.inf, -np.inf], np.nan)

# Tip percentage
df_ts['tip_percentage'] = (df_ts['tip_amount'] / df_ts['fare_amount']) * 100
df_ts['tip_percentage'] = df_ts['tip_percentage'].fillna(0)  # No tip = 0%

# Distance category
def categorize_distance(dist):
    if dist < 1:
        return 'Short'
    elif dist < 3:
        return 'Medium'
    elif dist < 10:
        return 'Long'
    else:
        return 'Very Long'

df_ts['distance_category'] = df_ts['trip_distance'].apply(categorize_distance)

print("Derived features created:")
print(df_ts[['speed_mph', 'fare_per_mile', 'tip_percentage', 'distance_category']].head(10))
```

### Step 2: GroupBy Aggregations

```python
# Aggregate by day of week
daily_stats = df_ts.groupby('day_name').agg({
    'fare_amount': ['mean', 'median', 'std', 'count'],
    'trip_distance': ['mean', 'median'],
    'trip_duration': ['mean', 'median'],
    'passenger_count': 'mean'
}).round(2)

print("Statistics by Day of Week:")
display(daily_stats)

# Aggregate by multiple dimensions: day of week and time of day
multi_agg = df_ts.groupby(['day_name', 'time_of_day']).agg({
    'fare_amount': 'mean',
    'trip_distance': 'count'  # Count of trips
}).rename(columns={'fare_amount': 'avg_fare', 'trip_distance': 'trip_count'})

print("\nAverage Fare by Day and Time:")
display(multi_agg.head(15))
```

### Step 3: Rolling Window Calculations

```python
# Resample to hourly for rolling calculations
hourly_trips = df_ts.resample('h').agg({
    'fare_amount': ['mean', 'count'],
    'trip_distance': 'mean',
    'total_amount': 'sum'
})
hourly_trips.columns = ['fare_amount', 'trip_count', 'trip_distance', 'total_amount']
hourly_trips = hourly_trips[['fare_amount', 'trip_distance', 'total_amount', 'trip_count']]

# Calculate rolling averages (7-day and 30-day windows)
hourly_trips['fare_7d_avg'] = hourly_trips['fare_amount'].rolling(window=7*24, min_periods=1).mean()  # 7 days * 24 hours
hourly_trips['fare_30d_avg'] = hourly_trips['fare_amount'].rolling(window=30*24, min_periods=1).mean()  # 30 days * 24 hours

# Calculate exponentially weighted moving average
hourly_trips['fare_ewm'] = hourly_trips['fare_amount'].ewm(span=7*24, adjust=False).mean()

print("Rolling window calculations:")
display(hourly_trips[['fare_amount', 'fare_7d_avg', 'fare_30d_avg', 'fare_ewm']].head(20))

# Visualize rolling averages
plt.figure(figsize=(14, 6))
plt.plot(hourly_trips.index, hourly_trips['fare_amount'], alpha=0.3, label='Hourly Average', linewidth=1)
plt.plot(hourly_trips.index, hourly_trips['fare_7d_avg'], label='7-Day Rolling Average', linewidth=2)
plt.plot(hourly_trips.index, hourly_trips['fare_30d_avg'], label='30-Day Rolling Average', linewidth=2)
plt.plot(hourly_trips.index, hourly_trips['fare_ewm'], label='Exponentially Weighted', linewidth=2, linestyle='--')
plt.title('Fare Amount Trends with Rolling Averages', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Average Fare Amount ($)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Step 4: Time-Based Aggregations

```python
# Aggregate by hour of day (across all days)
hourly_pattern = df_ts.groupby('hour').agg({
    'fare_amount': ['mean', 'count'],
    'trip_distance': 'mean',
    'total_amount': 'sum'
})
hourly_pattern.columns = ['fare_amount', 'trip_count', 'trip_distance', 'total_amount']
hourly_pattern = hourly_pattern[['fare_amount', 'trip_count', 'trip_distance', 'total_amount']]

print("Hourly Patterns (aggregated across all days):")
display(hourly_pattern.head(10))

# Visualize hourly patterns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Hourly Patterns in Taxi Trips', fontsize=16, fontweight='bold')

# Average fare by hour
axes[0, 0].plot(hourly_pattern.index, hourly_pattern['fare_amount'], marker='o', linewidth=2)
axes[0, 0].set_title('Average Fare by Hour of Day')
axes[0, 0].set_xlabel('Hour')
axes[0, 0].set_ylabel('Average Fare ($)')
axes[0, 0].grid(True, alpha=0.3)

# Trip count by hour
axes[0, 1].bar(hourly_pattern.index, hourly_pattern['trip_count'], alpha=0.7)
axes[0, 1].set_title('Number of Trips by Hour of Day')
axes[0, 1].set_xlabel('Hour')
axes[0, 1].set_ylabel('Number of Trips')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Average distance by hour
axes[1, 0].plot(hourly_pattern.index, hourly_pattern['trip_distance'], marker='s', color='green', linewidth=2)
axes[1, 0].set_title('Average Distance by Hour of Day')
axes[1, 0].set_xlabel('Hour')
axes[1, 0].set_ylabel('Average Distance (miles)')
axes[1, 0].grid(True, alpha=0.3)

# Total revenue by hour
axes[1, 1].bar(hourly_pattern.index, hourly_pattern['total_amount'], alpha=0.7, color='orange')
axes[1, 1].set_title('Total Revenue by Hour of Day')
axes[1, 1].set_xlabel('Hour')
axes[1, 1].set_ylabel('Total Revenue ($)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### Step 5: Cross-Tabulation

```python
# Cross-tabulation: Day of week vs Time of day
crosstab = pd.crosstab(
    df_ts['day_name'],
    df_ts['time_of_day'],
    margins=True
)

print("Trip Count: Day of Week × Time of Day")
display(crosstab)

# Cross-tabulation with aggregation
crosstab_fare = pd.crosstab(
    df_ts['day_name'],
    df_ts['time_of_day'],
    values=df_ts['fare_amount'],
    aggfunc='mean',
    margins=True
).round(2)

print("\nAverage Fare: Day of Week × Time of Day")
display(crosstab_fare)
```

### Step 6: Save Processed Data

```python
# Reset index to make pickup_datetime a regular column again
df_processed = df_ts.reset_index()

# Save processed dataset for next notebook
df_processed.to_csv('../output/02_processed_taxi_data.csv', index=False)
print(f"\nProcessed data saved: {len(df_processed):,} trips")
print(f"Columns: {df_processed.shape[1]}")
print("\nReady for next phase: Pattern Analysis & Modeling Prep!")
```

---

## Summary

**What we accomplished:**

1. ✅ **Set datetime index** for time-based operations
2. ✅ **Extracted time-based features** (hour, day, month, etc.)
3. ✅ **Merged zone lookup data** using pandas merge
4. ✅ **Reshaped data** using pivot and melt
5. ✅ **Created derived features** (speed, fare per mile, etc.)
6. ✅ **Performed aggregations** by multiple dimensions
7. ✅ **Calculated rolling windows** for trend analysis
8. ✅ **Created time-based patterns** and visualizations

**Key Takeaways:**
- Datetime indexing enables time-based operations and aggregations
- Merging enriches data with additional context
- Feature engineering creates predictive signals
- Rolling windows reveal trends and patterns
- GroupBy aggregations summarize data at different levels

**Next:** Notebook 3 will focus on pattern analysis, advanced visualizations, and preparing data for modeling.

