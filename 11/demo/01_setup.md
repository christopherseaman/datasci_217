# Notebook 1: Setup, Exploration & Cleaning

**Phases 1-3:** Project Setup, Data Exploration, and Data Cleaning

**Dataset:** NYC Taxi Trip Dataset

**Focus:** Getting data ready for analysis - loading, understanding, and cleaning messy real-world data.

---

## Phase 1: Project Setup & Data Acquisition

### Learning Objectives
- Set up the analysis environment
- Load data from files
- Perform initial data inspection
- Understand data structure and schema

### Step 1: Import Libraries

```python
# Core data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Jupyter display
from IPython.display import display

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline

# Display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("Libraries imported successfully!")
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
```

### Step 2: Load the Data

**NYC Taxi Trip Dataset**

**Source:** [NYC Taxi & Limousine Commission (TLC)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

We'll use actual NYC Taxi Trip data downloaded from the NYC TLC website. The data is available in Parquet or CSV format.

```python
# Load actual NYC Taxi Trip data
# Download from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
# NYC TLC provides trip record data in Parquet or CSV format
# Place the downloaded file in the data/ directory

import os

# Check for data file (downloaded by download_data.sh)
data_file = 'data/yellow_tripdata_2023-01.parquet'

if not os.path.exists(data_file):
    print("⚠️  NYC Taxi data file not found!")
    print("Please run download_data.sh to download the data:")
    print("  chmod +x download_data.sh")
    print("  ./download_data.sh")
    raise FileNotFoundError(f"NYC Taxi data file not found: {data_file}. Run download_data.sh first.")

print(f"Loading NYC Taxi Trip data from: {data_file}")

# Load data - Parquet format (downloaded by download_data.sh)
# Parquet requires pyarrow: pip install pyarrow
try:
    df = pd.read_parquet(data_file)
    print(f"✅ Loaded Parquet file: {len(df):,} rows")
except ImportError as e:
    if 'pyarrow' in str(e).lower() or 'fastparquet' in str(e).lower():
        print("❌ Parquet file requires pyarrow library")
        print("Please install pyarrow:")
        print("  pip install pyarrow")
        print("Or use conda:")
        print("  conda install pyarrow")
        raise ImportError("pyarrow is required to read Parquet files. Install with: pip install pyarrow")
    else:
        raise
except Exception as e:
    print(f"❌ Error loading Parquet file: {e}")
    raise

# Standardize column names for consistency across all notebooks
# NYC TLC data uses 'tpep_' prefix for Yellow taxis, 'lpep_' for Green taxis
# We'll standardize to simpler names: pickup_datetime, dropoff_datetime, etc.

column_mapping = {
    # Datetime columns (most important - used throughout)
    'tpep_pickup_datetime': 'pickup_datetime',
    'tpep_dropoff_datetime': 'dropoff_datetime',
    'lpep_pickup_datetime': 'pickup_datetime',  # Green taxi
    'lpep_dropoff_datetime': 'dropoff_datetime',
}

# Apply mapping
for old_name, new_name in column_mapping.items():
    if old_name in df.columns:
        df = df.rename(columns={old_name: new_name})

# Parse datetime columns immediately (needed for trip_duration calculation)
if 'pickup_datetime' in df.columns:
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
if 'dropoff_datetime' in df.columns:
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

# Calculate total_amount if not present (NYC TLC has component columns)
if 'total_amount' not in df.columns:
    components = ['fare_amount']
    if 'tip_amount' in df.columns:
        components.append('tip_amount')
    if 'tolls_amount' in df.columns:
        components.append('tolls_amount')
    if 'extra' in df.columns:
        components.append('extra')
    if 'mta_tax' in df.columns:
        components.append('mta_tax')
    if 'improvement_surcharge' in df.columns:
        components.append('improvement_surcharge')
    
    if len(components) > 1:
        df['total_amount'] = df[components].sum(axis=1)
        print(f"✓ Calculated total_amount from: {', '.join(components)}")

# Verify essential columns exist
essential_cols = ['pickup_datetime', 'dropoff_datetime', 'fare_amount', 'trip_distance', 'passenger_count']
missing_essential = [col for col in essential_cols if col not in df.columns]

if missing_essential:
    print(f"⚠️  Warning: Missing essential columns: {missing_essential}")
    print(f"Available columns: {list(df.columns)}")
    print("\nNote: NYC TLC data structure may vary. Adjust column references as needed.")
else:
    print("✅ All essential columns found!")

print(f"\n✅ Loaded {len(df):,} taxi trips")
print(f"Columns: {list(df.columns)[:10]}... ({len(df.columns)} total)")
if 'pickup_datetime' in df.columns:
    print(f"Date range: {df['pickup_datetime'].min()} to {df['pickup_datetime'].max()}")

```

### Step 3: Initial Data Inspection

Now that we've loaded the data, we need to understand its structure and quality. This initial inspection helps us identify potential issues before diving deeper into analysis.

**What to look for:**
- Dataset size (rows and columns)
- Column names and data types
- Memory usage (important for large datasets)
- Sample of actual data values
- Summary statistics
- Missing data patterns

```python
# Basic information about the dataset
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn names:")
print(df.columns.tolist())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMemory usage:")
print(df.memory_usage(deep=True).sum() / 1024**2, "MB")
```

**Interpreting the output:** The shape tells us how much data we're working with. Column names help us understand what information is available. Data types are crucial - we need to ensure datetime columns are properly parsed, and numeric columns are numeric (not strings). Memory usage helps us plan for processing - large datasets may require chunking or sampling.

Now let's look at actual data values to see what the records look like:

```python
# First few rows
print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
display(df.head())
```

**What to observe:** Look at the actual values - do they make sense? Are there any obvious data quality issues? Are the datetime columns properly formatted? Do numeric values seem reasonable?

Next, we'll compute summary statistics to understand the distributions of our numeric variables:

```python
# Summary statistics
print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
display(df.describe())
```

**Key insights from summary statistics:**
- **Mean vs Median:** Large differences suggest skewed distributions (common with trip distances, fares)
- **Min/Max values:** Extreme values may indicate outliers or data errors
- **Standard deviation:** High std dev relative to mean suggests high variability
- **25th/75th percentiles:** Help identify the range where most data falls

Finally, we need to check for missing data, which is critical for data quality assessment:

```python
# Check for missing values
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
display(missing_df)
```

**Missing data considerations:**
- **High percentage (>10-20%):** May need to drop the column or use sophisticated imputation
- **Low percentage (<5%):** Can often be handled with simple imputation or removal
- **Patterns matter:** Is missingness random, or systematic (e.g., all missing on weekends)?
- **Domain knowledge:** Some missing values may be meaningful (e.g., missing tip = no tip)

---

## Phase 2: Data Exploration & Understanding

### Learning Objectives
- Understand data distributions
- Identify relationships between variables
- Create initial visualizations
- Spot potential data quality issues

### Step 1: Basic Statistics and Distributions

Visualizing distributions helps us understand the shape of our data, identify potential outliers, and see if variables are normally distributed or skewed. This is crucial before any modeling or analysis.

**Why distributions matter:**
- **Skewed distributions** may need transformation (log, square root)
- **Outliers** can heavily influence models
- **Bimodal distributions** suggest subgroups in the data
- **Normal distributions** are ideal for many statistical methods

Let's start by creating histograms for our key numeric variables:

```python
# Distribution of key numeric variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Key Variables', fontsize=16, fontweight='bold')

# Trip distance
axes[0, 0].hist(df['trip_distance'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Trip Distance (miles)')
axes[0, 0].set_xlabel('Distance')
axes[0, 0].set_ylabel('Frequency')

# Fare amount
axes[0, 1].hist(df['fare_amount'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Fare Amount ($)')
axes[0, 1].set_xlabel('Fare')
axes[0, 1].set_ylabel('Frequency')

# Tip amount (excluding zeros)
tips_nonzero = df['tip_amount'].dropna()
axes[0, 2].hist(tips_nonzero, bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].set_title('Tip Amount ($) - Non-zero only')
axes[0, 2].set_xlabel('Tip')
axes[0, 2].set_ylabel('Frequency')

# Passenger count
passenger_counts = df['passenger_count'].value_counts().sort_index()
axes[1, 0].bar(passenger_counts.index, passenger_counts.values, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Passenger Count Distribution')
axes[1, 0].set_xlabel('Passengers')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_xticks(passenger_counts.index)

# Total amount
axes[1, 1].hist(df['total_amount'], bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Total Amount ($)')
axes[1, 1].set_xlabel('Total')
axes[1, 1].set_ylabel('Frequency')

# Trip duration (calculate from datetime columns)
df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60  # minutes
axes[1, 2].hist(df['trip_duration'], bins=50, edgecolor='black', alpha=0.7)
axes[1, 2].set_title('Trip Duration (minutes)')
axes[1, 2].set_xlabel('Duration')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

**What to look for in these distributions:**
- **Trip distance:** Likely right-skewed (many short trips, few very long ones)
- **Fare amount:** Should correlate with distance; watch for negative or zero fares
- **Tip amount:** Many zeros (no tip), then a distribution of tip amounts
- **Passenger count:** Discrete values (1-6 typically); check for unrealistic values
- **Total amount:** Should be sum of fare + tip + taxes; verify consistency
- **Trip duration:** Right-skewed; very long durations may be errors

**Key observations:**
- Most distributions will be right-skewed (common in real-world data)
- Extreme values in the tails may be outliers or errors
- Bimodal patterns might indicate different trip types (e.g., airport vs local)

### Step 2: Trends Over Time

Time series analysis is a required component of this project. Let's examine how trip volume changes over time to identify patterns, seasonality, and anomalies.

**Why temporal patterns matter:**
- **Daily patterns:** Rush hours, lunch breaks, late night
- **Weekly patterns:** Weekday vs weekend differences
- **Monthly patterns:** Seasonal effects, holidays
- **Anomalies:** Unusual days (events, weather, holidays)

Let's aggregate trips by date and visualize the trend:

```python
# Plot trips over time to see temporal patterns
df['pickup_date'] = pd.to_datetime(df['pickup_datetime']).dt.date
trips_by_date = df.groupby('pickup_date').size()

plt.figure(figsize=(14, 6))
plt.plot(trips_by_date.index, trips_by_date.values, linewidth=2)
plt.title('Number of Taxi Trips Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Average trips per day: {trips_by_date.mean():.0f}")
print(f"Peak day: {trips_by_date.idxmax()} with {trips_by_date.max()} trips")
```

**Interpreting the time series plot:**
- **Trend:** Is trip volume increasing, decreasing, or stable?
- **Weekly pattern:** Look for regular dips (weekends typically have fewer trips)
- **Outliers:** Days with unusually high or low trip counts (holidays, events, data issues)
- **Variability:** How much does daily volume fluctuate?

**Common patterns in taxi data:**
- Lower volume on weekends
- Higher volume during rush hours
- Holiday effects (New Year's Eve, Thanksgiving)
- Weather impacts (snow, rain increase demand)

### Step 3: Relationships Between Variables

Understanding relationships between variables helps us identify which features might be useful for modeling and which might be redundant (highly correlated).

**Why correlation matters:**
- **Feature selection:** Highly correlated features may be redundant
- **Model assumptions:** Some models assume independence
- **Business insights:** Understanding what drives fares, tips, etc.

Let's start with a scatter plot to visualize the relationship between distance and fare:

```python
# Scatter plot: Distance vs Fare
plt.figure(figsize=(10, 6))
plt.scatter(df['trip_distance'], df['fare_amount'], alpha=0.3, s=10)
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Fare Amount ($)')
plt.title('Trip Distance vs Fare Amount', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**What to observe in the scatter plot:**
- **Positive relationship:** Longer trips should cost more (expected)
- **Linear vs non-linear:** Is the relationship linear or curved?
- **Outliers:** Points far from the main cluster (e.g., very short trips with high fares, or very long trips with low fares)
- **Clusters:** Multiple groups might indicate different fare structures (e.g., airport flat rates)

**Expected patterns:**
- Strong positive correlation (longer trips = higher fares)
- Some variation due to traffic, time of day, tolls
- Outliers might be data errors or special fare types

Now let's compute a correlation matrix to quantify relationships between all numeric variables:

```python
# Correlation matrix
numeric_cols = ['trip_distance', 'fare_amount', 'tip_amount', 'total_amount', 'passenger_count', 'trip_duration']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Interpreting correlation values:**
- **|r| > 0.7:** Strong correlation (may indicate redundancy)
- **0.3 < |r| < 0.7:** Moderate correlation (useful relationships)
- **|r| < 0.3:** Weak correlation (little relationship)
- **Positive values:** Variables increase together
- **Negative values:** Variables move in opposite directions

**Expected correlations:**
- `trip_distance` ↔ `fare_amount`: Strong positive (longer trips cost more)
- `fare_amount` ↔ `total_amount`: Very strong positive (fare is major component)
- `trip_distance` ↔ `trip_duration`: Moderate positive (longer trips take more time, but traffic matters)
- `tip_amount` ↔ `fare_amount`: Moderate positive (higher fares often get higher tips)

---

## Phase 3: Data Cleaning & Preprocessing

### Learning Objectives
- Identify and handle missing data
- Detect and handle outliers
- Validate data ranges
- Clean data systematically

### Step 1: Missing Data Analysis

Before handling missing data, we need to understand the extent and pattern of missingness. This informs our strategy for dealing with it.

**Why missing data analysis matters:**
- **Extent:** How much data is missing? (affects our sample size)
- **Pattern:** Is missingness random or systematic? (affects imputation strategy)
- **Impact:** Which variables are affected? (affects which features we can use)

Let's create a comprehensive missing data report:

```python
print("=" * 60)
print("MISSING DATA ANALYSIS")
print("=" * 60)

# Detailed missing data analysis
missing_analysis = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': df.isnull().sum().values,
    'Missing Percentage': (df.isnull().sum() / len(df) * 100).values,
    'Data Type': df.dtypes.values
})
missing_analysis = missing_analysis[missing_analysis['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

display(missing_analysis)
```

**Interpreting missing data:**
- **High percentage (>20%):** Consider dropping the column or using advanced imputation
- **Medium percentage (5-20%):** Can use imputation, but be cautious
- **Low percentage (<5%):** Usually safe to impute or drop rows
- **Systematic missingness:** All missing on weekends? All missing for certain trip types?

Now let's visualize the missing data pattern:

```python
# Visualize missing data pattern
if len(missing_analysis) > 0:
    plt.figure(figsize=(10, 6))
    plt.barh(missing_analysis['Column'], missing_analysis['Missing Percentage'])
    plt.xlabel('Missing Percentage (%)')
    plt.title('Missing Data by Column', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
```

**Visual insights:**
- Which columns have the most missing data?
- Are there patterns (e.g., all optional fields missing together)?
- Does missingness correlate with other variables?

### Step 2: Handle Missing Data

Now that we understand the missing data, we need to decide on a strategy. The approach depends on:
- **Type of variable:** Categorical vs numeric
- **Amount missing:** High vs low percentage
- **Domain knowledge:** Is missing meaningful? (e.g., missing tip = no tip)

**Common strategies:**
- **Drop rows:** If missing is rare and random
- **Drop columns:** If too much is missing
- **Impute with 0:** For counts/amounts where 0 is meaningful
- **Impute with median/mean:** For numeric variables
- **Impute with mode:** For categorical variables
- **Advanced imputation:** KNN, regression-based, etc.

For this dataset, let's handle tip_amount specifically:

```python
# Strategy for handling missing data
# For tip_amount: Missing likely means no tip (0), but we'll be conservative
# and use median imputation for now

print("Handling missing data...")
print(f"Missing tip_amount before: {df['tip_amount'].isnull().sum()}")
```

**Decision point:** For `tip_amount`, we have two reasonable options:
1. **Fill with 0:** Assumes missing = no tip (common in taxi data)
2. **Fill with median:** More conservative, preserves distribution

We'll use median imputation to be conservative, but in practice, you might choose 0 based on domain knowledge:

```python
# Option 1: Fill with 0 (assuming missing = no tip)
# df['tip_amount'] = df['tip_amount'].fillna(0)

# Option 2: Fill with median (more conservative)
df['tip_amount'] = df['tip_amount'].fillna(df['tip_amount'].median())

print(f"Missing tip_amount after: {df['tip_amount'].isnull().sum()}")
```

**Note:** After imputing tip_amount, we should recalculate total_amount to ensure consistency:

```python
# Recalculate total_amount to ensure consistency
df['total_amount'] = df['fare_amount'] + df['tip_amount'] + 0.5

print("Missing data handling complete!")
```

**Why recalculate?** If we imputed tip_amount, the total_amount might have been calculated before imputation, so we need to update it.

### Step 3: Outlier Detection

Outliers can significantly impact models and analysis. We need to identify them using statistical methods, then decide whether they're errors (remove) or valid extreme values (keep or transform).

**Why detect outliers:**
- **Model impact:** Outliers can heavily influence regression models
- **Data quality:** Extreme values may be data entry errors
- **Domain knowledge:** Some outliers are valid (e.g., very long airport trips)

**Common methods:**
- **IQR method:** Uses quartiles (robust to outliers)
- **Z-score method:** Uses standard deviations (assumes normal distribution)
- **Domain-based:** Use business rules (e.g., trips > 50 miles are errors)

Let's use the IQR (Interquartile Range) method, which is robust and doesn't assume normality:

```python
print("=" * 60)
print("OUTLIER DETECTION")
print("=" * 60)

# Identify outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound
```

**How IQR works:**
- **Q1 (25th percentile):** 25% of data below this value
- **Q3 (75th percentile):** 75% of data below this value
- **IQR = Q3 - Q1:** The "middle 50%" range
- **Bounds:** Q1 - 1.5×IQR (lower) and Q3 + 1.5×IQR (upper)
- **Outliers:** Values outside these bounds

Now let's check for outliers in trip distance:

```python
# Check trip_distance outliers
distance_outliers, dist_lower, dist_upper = detect_outliers_iqr(df, 'trip_distance')
print(f"\nTrip Distance Outliers:")
print(f"  Lower bound: {dist_lower:.2f} miles")
print(f"  Upper bound: {dist_upper:.2f} miles")
print(f"  Number of outliers: {len(distance_outliers):,} ({len(distance_outliers)/len(df)*100:.2f}%)")
```

**Interpreting trip distance outliers:**
- **Negative values:** Data errors (must remove)
- **Very high values (>50 miles):** Unusual for NYC, might be airport trips or errors
- **Zero values:** Data errors (trip must have distance)

Next, let's check fare amount:

```python
# Check fare_amount outliers
fare_outliers, fare_lower, fare_upper = detect_outliers_iqr(df, 'fare_amount')
print(f"\nFare Amount Outliers:")
print(f"  Lower bound: ${fare_lower:.2f}")
print(f"  Upper bound: ${fare_upper:.2f}")
print(f"  Number of outliers: {len(fare_outliers):,} ({len(fare_outliers)/len(df)*100:.2f}%)")
```

**Interpreting fare outliers:**
- **Negative fares:** Data errors (must remove)
- **Zero fares:** Could be errors or promotional rides
- **Very high fares:** Might be valid (long trips, tolls, surcharges) or errors

Finally, let's check trip duration:

```python
# Check trip_duration outliers (unrealistic trips)
duration_outliers, dur_lower, dur_upper = detect_outliers_iqr(df, 'trip_duration')
print(f"\nTrip Duration Outliers:")
print(f"  Lower bound: {dur_lower:.2f} minutes")
print(f"  Upper bound: {dur_upper:.2f} minutes")
print(f"  Number of outliers: {len(duration_outliers):,} ({len(duration_outliers)/len(df)*100:.2f}%)")
```

**Interpreting duration outliers:**
- **Negative duration:** Data errors (dropoff before pickup - must remove)
- **Very short (<1 minute):** Might be errors or very short trips
- **Very long (>2 hours):** Unusual for NYC taxis, might be errors or special cases

### Step 4: Check for Duplicates

Duplicate records can occur due to data collection errors, system glitches, or legitimate re-submissions. We need to identify and handle them.

**Types of duplicates:**
- **Exact duplicates:** Identical rows (likely data errors)
- **Near-duplicates:** Same trip recorded multiple times (same location/time)

**Why remove duplicates:**
- **Model bias:** Duplicates give extra weight to certain observations
- **Data quality:** Indicates potential data collection issues
- **Memory/performance:** Reduces dataset size

Let's first check for completely identical rows:

```python
# Check for duplicate rows
print("=" * 60)
print("DUPLICATE DETECTION")
print("=" * 60)

# Check for completely duplicate rows
n_duplicates = df.duplicated().sum()
print(f"Completely duplicate rows: {n_duplicates:,}")
```

**Interpreting exact duplicates:**
- **Zero duplicates:** Good data quality
- **Many duplicates:** May indicate data collection or processing issues
- **Few duplicates:** Might be legitimate (e.g., system retry)

Now let's check for near-duplicates - trips that have the same pickup/dropoff location and time:

```python
# Check for duplicates based on key columns (same trip recorded twice)
# NYC TLC data uses location IDs (PULocationID, DOLocationID) instead of lat/long
# Note: If latitude/longitude coordinates were available, duplicate detection could be
# more robust using fuzzy matching (e.g., trips within ~100m of each other at similar times).
# For this dataset, exact matching on location IDs and timestamps is a reasonable approach.
key_cols = ['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID']
# Only check if these columns exist
available_key_cols = [col for col in key_cols if col in df.columns]
if len(available_key_cols) >= 2:
    n_key_duplicates = df.duplicated(subset=available_key_cols).sum()
    print(f"Duplicate trips (same pickup/dropoff location and time): {n_key_duplicates:,}")
else:
    print("Location columns not available for duplicate detection")
```

**Why check near-duplicates?**
- Same trip might be recorded multiple times due to system issues
- Location IDs are less precise than coordinates, so exact matches are reasonable
- If we had lat/long, we could use fuzzy matching (e.g., trips within 100m at similar times)

If we find duplicates, let's examine them:

```python
# Show examples if any duplicates exist
if n_duplicates > 0:
    print("\nExample duplicate rows:")
    display(df[df.duplicated(keep=False)].head(10))
```

**What to look for:**
- Are duplicates truly identical, or do they differ in some columns?
- If they differ, which columns vary? (helps understand the issue)
- Should we keep the first occurrence, last, or merge them?

### Step 5: Handle Outliers

Now that we've identified outliers, we need to decide how to handle them. The strategy depends on:
- **Are they errors?** (remove them)
- **Are they valid but extreme?** (keep, cap, or transform)
- **Domain knowledge:** What makes sense for taxi trips in NYC?

**Common strategies:**
- **Remove:** If clearly errors (negative values, impossible combinations)
- **Cap:** Set extreme values to reasonable maximums
- **Transform:** Use log transformation for highly skewed data
- **Keep:** If valid extreme cases (e.g., airport trips)

Let's apply domain-specific cleaning rules:

```python
# Handle outliers based on domain knowledge
print("\nHandling outliers...")

# Remove unrealistic trip distances (> 50 miles in NYC is very unusual)
# Or cap them at a reasonable maximum
df_clean = df.copy()
print(f"Original shape: {df_clean.shape}")
```

**Our cleaning strategy:**
1. Remove duplicates
2. Cap extreme trip distances (NYC trips rarely exceed 50 miles)
3. Remove negative or zero distances
4. Remove unrealistic trip durations (>2 hours)
5. Remove negative fares
6. Remove unrealistic passenger counts

Let's apply these rules step by step:

```python
# Remove duplicate rows (if any)
df_clean = df_clean.drop_duplicates()
print(f"After removing duplicates: {df_clean.shape}")
```

**Why remove duplicates first?** Duplicates can inflate our counts and affect outlier detection.

```python
# Cap trip_distance at 50 miles (very generous for NYC)
df_clean['trip_distance'] = df_clean['trip_distance'].clip(upper=50)

# Remove trips with negative or zero distance
df_clean = df_clean[df_clean['trip_distance'] > 0]
```

**Distance cleaning rationale:**
- **Cap at 50 miles:** NYC is ~13 miles across, so 50 miles is very generous (includes airport trips)
- **Remove zero/negative:** These are data errors

```python
# Remove trips with unrealistic duration (> 2 hours is very unusual)
df_clean = df_clean[df_clean['trip_duration'] <= 120]  # 2 hours max

# Remove trips with negative fare
df_clean = df_clean[df_clean['fare_amount'] > 0]

# Remove trips with unrealistic passenger counts
df_clean = df_clean[df_clean['passenger_count'].between(1, 6)]
```

**Additional cleaning rules:**
- **Duration ≤ 120 minutes:** Very long trips are unusual in NYC (traffic is bad, but not that bad)
- **Fare > 0:** Negative fares are errors
- **Passenger count 1-6:** Taxis typically hold 1-6 passengers

Let's see the impact of our cleaning:

```python
print(f"Cleaned shape: {df_clean.shape}")
print(f"Removed {len(df) - len(df_clean):,} rows ({(len(df) - len(df_clean))/len(df)*100:.2f}%)")
```

**Interpreting the results:**
- **Removal percentage:** How much data did we lose?
- **If >10%:** Might be too aggressive, reconsider thresholds
- **If <1%:** Very clean data, or thresholds too lenient
- **Balance:** Remove errors while preserving valid extreme cases

### Step 6: Data Type Validation and Conversion

After cleaning, we need to ensure all data types are correct. This is crucial for:
- **Performance:** Correct types use less memory and process faster
- **Functionality:** Some operations require specific types (e.g., datetime operations)
- **Modeling:** Machine learning models expect numeric types

**Common type issues:**
- Datetime columns stored as strings
- Numeric columns stored as strings (e.g., "12.5" instead of 12.5)
- Categorical columns stored as numeric codes

Let's validate and convert data types:

```python
# Ensure datetime columns are properly typed
print("\nValidating and converting data types...")

df_clean['pickup_datetime'] = pd.to_datetime(df_clean['pickup_datetime'])
df_clean['dropoff_datetime'] = pd.to_datetime(df_clean['dropoff_datetime'])
```

**Why datetime conversion matters:**
- Enables time-based operations (resampling, rolling windows)
- Allows extraction of temporal features (hour, day of week, month)
- Required for time series analysis

Now let's ensure numeric columns are properly typed:

```python
# Ensure numeric columns are proper types
numeric_cols = ['trip_distance', 'fare_amount', 'tip_amount', 'total_amount', 
                'passenger_count', 'trip_duration']
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
```

**Why numeric conversion matters:**
- Mathematical operations require numeric types
- Models expect numeric inputs
- `errors='coerce'` converts invalid values to NaN (which we can then handle)

Let's verify everything is correct:

```python
# Check for any remaining issues
print("\nFinal data quality check:")
print(f"Missing values: {df_clean.isnull().sum().sum()}")
print(f"Data types:\n{df_clean.dtypes}")
print(f"\nFinal dataset shape: {df_clean.shape}")
```

**Final validation checklist:**
- ✅ All datetime columns are datetime type
- ✅ All numeric columns are numeric type
- ✅ Missing values are handled or acceptable
- ✅ Dataset shape is reasonable (not too much data removed)

### Step 7: Save Cleaned Data

Now that we've cleaned and validated our data, we should save it for use in the next notebook. This ensures we don't have to repeat the cleaning process.

**Why save intermediate results:**
- **Efficiency:** Avoid re-running time-consuming cleaning steps
- **Reproducibility:** Others can use the cleaned data
- **Version control:** Track data transformations
- **Backup:** In case we need to revert changes

Let's save the cleaned dataset:

```python
# Save cleaned dataset for next notebook
output_dir = '../output'
import os
os.makedirs(output_dir, exist_ok=True)

df_clean.to_csv(f'{output_dir}/01_cleaned_taxi_data.csv', index=False)
print(f"\nCleaned data saved to: {output_dir}/01_cleaned_taxi_data.csv")
print("Ready for next phase: Data Wrangling & Feature Engineering!")
```

**File format considerations:**
- **CSV:** Human-readable, universal compatibility, but larger file size
- **Parquet:** Compressed, faster to read/write, preserves data types, but requires special libraries
- **Pickle:** Python-specific, preserves everything, but not portable across languages

For this project, CSV is a good choice for compatibility and ease of use.

---

## Summary

**What we accomplished:**

1. ✅ **Loaded data** and performed initial inspection
2. ✅ **Explored distributions** and relationships
3. ✅ **Identified missing data** and handled it appropriately
4. ✅ **Detected outliers** using statistical methods
5. ✅ **Cleaned data** based on domain knowledge
6. ✅ **Validated data types** and ranges
7. ✅ **Saved cleaned dataset** for next phase

**Key Takeaways:**
- Always inspect data before cleaning
- Use domain knowledge to guide cleaning decisions
- Document your cleaning steps
- Save intermediate results

**Next:** Notebook 2 will focus on data wrangling, merging, and feature engineering.

