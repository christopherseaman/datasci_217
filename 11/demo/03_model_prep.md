# Notebook 3: Pattern Analysis & Modeling Prep

**Phases 6-7:** Pattern Analysis & Advanced Visualization, Modeling Preparation

**Dataset:** NYC Taxi Trip Dataset (continuing from Notebook 2)

**Focus:** Deep analysis of patterns, advanced visualizations, and preparing data for predictive modeling.

---

## Phase 6: Pattern Analysis & Advanced Visualization

### Learning Objectives
- Create advanced multi-panel visualizations
- Identify trends and seasonal patterns
- Perform statistical analysis
- Visualize relationships across multiple dimensions

### Step 1: Load Processed Data

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

# Load processed data from Notebook 2
df = pd.read_csv('../output/02_processed_taxi_data.csv')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

print(f"Loaded {len(df):,} processed trips")
print(f"Date range: {df['pickup_datetime'].min()} to {df['pickup_datetime'].max()}")
```

### Step 2: Trends Analysis Over Time

```python
# Set datetime index for time-based operations
df_ts = df.set_index('pickup_datetime').sort_index()

# Resample to daily for trend analysis
daily = df_ts.resample('d').agg({
    'fare_amount': 'mean',
    'total_amount': 'sum',
    'trip_distance': 'count'  # Count trips
}).rename(columns={'trip_distance': 'trip_count'})

# Calculate moving averages for trend detection
daily['fare_7d_ma'] = daily['fare_amount'].rolling(window=7, min_periods=1).mean()
daily['fare_30d_ma'] = daily['fare_amount'].rolling(window=30, min_periods=1).mean()

# Visualize trends
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Trends Analysis Over Time', fontsize=16, fontweight='bold')

# Average fare over time
axes[0].plot(daily.index, daily['fare_amount'], alpha=0.5, label='Daily Average', linewidth=1)
axes[0].plot(daily.index, daily['fare_7d_ma'], label='7-Day Moving Average', linewidth=2)
axes[0].plot(daily.index, daily['fare_30d_ma'], label='30-Day Moving Average', linewidth=2)
axes[0].set_title('Average Fare Amount Over Time')
axes[0].set_ylabel('Fare Amount ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Total revenue over time
axes[1].plot(daily.index, daily['total_amount'], linewidth=2, color='green')
axes[1].set_title('Total Daily Revenue')
axes[1].set_ylabel('Revenue ($)')
axes[1].grid(True, alpha=0.3)

# Trip count over time
axes[2].bar(daily.index, daily['trip_count'], alpha=0.7, width=1)
axes[2].set_title('Number of Trips per Day')
axes[2].set_ylabel('Trip Count')
axes[2].set_xlabel('Date')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### Step 3: Seasonal Pattern Analysis

```python
# Analyze patterns by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_ts['day_name'] = df_ts.index.day_name()

daily_by_dow = df_ts.groupby('day_name')['fare_amount'].agg(['mean', 'std', 'count']).reindex(day_order)

# Analyze patterns by month
monthly = df_ts.groupby('month')['fare_amount'].agg(['mean', 'std', 'count'])

# Visualize seasonal patterns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Seasonal Pattern Analysis', fontsize=16, fontweight='bold')

# Average fare by day of week
axes[0, 0].bar(range(len(daily_by_dow)), daily_by_dow['mean'], alpha=0.7)
axes[0, 0].set_xticks(range(len(daily_by_dow)))
axes[0, 0].set_xticklabels(daily_by_dow.index, rotation=45, ha='right')
axes[0, 0].set_title('Average Fare by Day of Week')
axes[0, 0].set_ylabel('Average Fare ($)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Average fare by month
axes[0, 1].plot(monthly.index, monthly['mean'], marker='o', linewidth=2, markersize=8)
axes[0, 1].set_title('Average Fare by Month')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Average Fare ($)')
axes[0, 1].set_xticks(monthly.index)
axes[0, 1].grid(True, alpha=0.3)

# Hourly pattern (heatmap by day of week)
hourly_dow = df_ts.groupby(['day_name', 'hour'])['fare_amount'].mean().unstack(level=0).reindex(columns=day_order)
sns.heatmap(hourly_dow, annot=False, cmap='YlOrRd', ax=axes[1, 0], cbar_kws={'label': 'Avg Fare ($)'})
axes[1, 0].set_title('Average Fare: Hour × Day of Week')
axes[1, 0].set_xlabel('Day of Week')
axes[1, 0].set_ylabel('Hour of Day')

# Weekend vs weekday comparison
df_ts['is_weekend'] = df_ts['day_of_week'].isin([5, 6])
weekend_comparison = df_ts.groupby(['is_weekend', 'hour'])['fare_amount'].mean().unstack(level=0)
axes[1, 1].plot(weekend_comparison.index, weekend_comparison[False], label='Weekday', marker='o', linewidth=2)
axes[1, 1].plot(weekend_comparison.index, weekend_comparison[True], label='Weekend', marker='s', linewidth=2)
axes[1, 1].set_title('Average Fare: Weekday vs Weekend by Hour')
axes[1, 1].set_xlabel('Hour of Day')
axes[1, 1].set_ylabel('Average Fare ($)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Step 4: Correlation Analysis

```python
# Select numeric features for correlation
numeric_features = ['fare_amount', 'trip_distance', 'trip_duration', 'passenger_count', 
                    'tip_amount', 'total_amount', 'speed_mph', 'fare_per_mile', 'tip_percentage']

corr_matrix = df_ts[numeric_features].corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Key Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Identify strongest correlations
print("Strongest Correlations (absolute value > 0.5):")
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

for feat1, feat2, corr in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True):
    print(f"  {feat1} ↔ {feat2}: {corr:.3f}")
```

### Step 5: Multi-Dimensional Analysis

```python
# Analyze fare by multiple dimensions: distance category, time of day, day type
multi_dim = df_ts.groupby(['distance_category', 'time_of_day', 'is_weekend'])['fare_amount'].mean().unstack(level=2)

# Note: unstack converts boolean to int (False=0, True=1)
# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Multi-Dimensional Fare Analysis', fontsize=16, fontweight='bold')

# Distance category vs time of day (weekday, is_weekend=0)
# Need to unstack the Series to create a 2D DataFrame for heatmap
weekday_col = 0 if 0 in multi_dim.columns else multi_dim.columns[0]
weekday_data = multi_dim[weekday_col].unstack(level=1)  # Unstack time_of_day to columns
sns.heatmap(weekday_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 0], cbar_kws={'label': 'Avg Fare ($)'})
axes[0, 0].set_title('Weekday: Distance Category × Time of Day')
axes[0, 0].set_xlabel('Time of Day')
axes[0, 0].set_ylabel('Distance Category')

# Distance category vs time of day (weekend, is_weekend=1)
weekend_col = 1 if 1 in multi_dim.columns else multi_dim.columns[-1]
weekend_data = multi_dim[weekend_col].unstack(level=1)  # Unstack time_of_day to columns
sns.heatmap(weekend_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 1], cbar_kws={'label': 'Avg Fare ($)'})
axes[0, 1].set_title('Weekend: Distance Category × Time of Day')
axes[0, 1].set_xlabel('Time of Day')
axes[0, 1].set_ylabel('Distance Category')

# Box plot: Fare by distance category
sns.boxplot(data=df_ts, x='distance_category', y='fare_amount', ax=axes[1, 0])
axes[1, 0].set_title('Fare Distribution by Distance Category')
axes[1, 0].set_xlabel('Distance Category')
axes[1, 0].set_ylabel('Fare Amount ($)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Violin plot: Fare by time of day
sns.violinplot(data=df_ts, x='time_of_day', y='fare_amount', ax=axes[1, 1])
axes[1, 1].set_title('Fare Distribution by Time of Day')
axes[1, 1].set_xlabel('Time of Day')
axes[1, 1].set_ylabel('Fare Amount ($)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

---

## Phase 7: Modeling Preparation

### Learning Objectives
- Split data temporally for time series
- Select and prepare features
- Handle categorical variables
- Create final modeling dataset

### Step 1: Temporal Train/Test Split

```python
# For data with temporal structure, we must split by time (not randomly)
# Train on earlier data, test on later data

# Sort by datetime to ensure temporal order
df_model = df_ts.reset_index().sort_values('pickup_datetime').copy()

# Define split point (e.g., 80% train, 20% test)
split_date = df_model['pickup_datetime'].quantile(0.8)
print(f"Split date: {split_date}")
print(f"Train: {df_model[df_model['pickup_datetime'] < split_date].shape[0]:,} trips")
print(f"Test: {df_model[df_model['pickup_datetime'] >= split_date].shape[0]:,} trips")

# Create train/test split
train = df_model[df_model['pickup_datetime'] < split_date].copy()
test = df_model[df_model['pickup_datetime'] >= split_date].copy()

print(f"\nTrain date range: {train['pickup_datetime'].min()} to {train['pickup_datetime'].max()}")
print(f"Test date range: {test['pickup_datetime'].min()} to {test['pickup_datetime'].max()}")

# Visualize the split
plt.figure(figsize=(14, 4))
plt.plot(train['pickup_datetime'], train['fare_amount'], alpha=0.3, label='Train', linewidth=0.5)
plt.plot(test['pickup_datetime'], test['fare_amount'], alpha=0.3, label='Test', linewidth=0.5, color='red')
plt.axvline(split_date, color='black', linestyle='--', linewidth=2, label='Split Point')
plt.title('Temporal Train/Test Split', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Fare Amount ($)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Step 2: Feature Selection

```python
# Define target variable
target = 'fare_amount'

# Select features for modeling
# Include temporal, geographic, and trip characteristics
feature_cols = [
    # Temporal features
    'hour', 'day_of_week', 'month', 'is_weekend',
    # Trip characteristics
    'trip_distance', 'passenger_count', 'trip_duration',
    # Derived features
    'speed_mph', 'fare_per_mile',
    # Categorical (will need encoding)
    'time_of_day', 'distance_category', 'pickup_borough'
]

# Check feature availability
available_features = [f for f in feature_cols if f in df_model.columns]
missing_features = [f for f in feature_cols if f not in df_model.columns]

print("Available features:", available_features)
if missing_features:
    print("Missing features (will create or skip):", missing_features)

# Select available features
X_train = train[available_features].copy()
X_test = test[available_features].copy()
y_train = train[target].copy()
y_test = test[target].copy()

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

### Step 3: Handle Categorical Variables

```python
# Identify categorical variables
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical features:", categorical_cols)
print("Numeric features:", numeric_cols)

# For simplicity, we'll use pandas get_dummies for one-hot encoding
# In practice, you might use sklearn's OneHotEncoder

X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, prefix=categorical_cols, drop_first=True)

# Ensure test set has same columns as training set
# Add missing columns (with 0s) and remove extra columns
for col in X_train_encoded.columns:
    if col not in X_test_encoded.columns:
        X_test_encoded[col] = 0

X_test_encoded = X_test_encoded[X_train_encoded.columns]

print(f"\nAfter encoding:")
print(f"Training features: {X_train_encoded.shape}")
print(f"Test features: {X_test_encoded.shape}")
print(f"Feature names: {list(X_train_encoded.columns)}")
```

### Step 4: Handle Missing Values in Features

```python
# Check for missing values
print("Missing values in training set:")
print(X_train_encoded.isnull().sum()[X_train_encoded.isnull().sum() > 0])

# Fill missing values (using training set statistics)
# For numeric columns, use median
for col in numeric_cols:
    if col in X_train_encoded.columns:
        median_val = X_train_encoded[col].median()
        X_train_encoded[col] = X_train_encoded[col].fillna(median_val)
        X_test_encoded[col] = X_test_encoded[col].fillna(median_val)

print("\nMissing values after imputation:")
print(f"Train: {X_train_encoded.isnull().sum().sum()}")
print(f"Test: {X_test_encoded.isnull().sum().sum()}")
```

### Step 5: Save Prepared Data

```python
# Save prepared datasets for modeling
X_train_encoded.to_csv('../output/03_X_train.csv', index=False)
X_test_encoded.to_csv('../output/03_X_test.csv', index=False)
y_train.to_csv('../output/03_y_train.csv', index=False)
y_test.to_csv('../output/03_y_test.csv', index=False)

print("Prepared datasets saved:")
print(f"  X_train: {X_train_encoded.shape}")
print(f"  X_test: {X_test_encoded.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test: {y_test.shape}")
print("\nReady for next phase: Modeling & Results!")
```

---

## Summary

**What we accomplished:**

1. ✅ **Analyzed trends over time** with moving averages
2. ✅ **Identified seasonal patterns** (day of week, month, hour)
3. ✅ **Performed correlation analysis** to understand relationships
4. ✅ **Created advanced visualizations** (heatmaps, multi-panel plots)
5. ✅ **Split data temporally** to prevent data leakage
6. ✅ **Selected and prepared features** for modeling
7. ✅ **Handled categorical variables** with encoding
8. ✅ **Prepared final datasets** for modeling

**Key Takeaways:**
- Temporal splits prevent data leakage when data has time structure
- Feature engineering creates predictive signals
- Categorical encoding is essential for ML models
- Advanced visualizations reveal hidden patterns
- Proper preparation ensures model quality

**Next:** Notebook 4 will build and evaluate predictive models.

