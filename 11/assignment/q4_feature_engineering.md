# Q4: Feature Engineering

**Phase 5:** Feature Engineering & Aggregation  
**Points: 9 points**

**Focus:** Create derived features, perform time-based aggregations, calculate rolling windows.

**Lecture Reference:** See **Lecture 11, Notebook 2** (`11/demo/02_wrangling_feature_engineering.ipynb`), Phase 5 for examples of feature engineering, time-based aggregations, and rolling window calculations. Also see **Lecture 09** for time series rolling window operations.

---

## Setup

```python
# Import libraries
import pandas as pd
import numpy as np
import os

# Load wrangled data from Q3
df = pd.read_csv('output/q3_wrangled_data.csv', parse_dates=['Measurement Timestamp'], index_col='Measurement Timestamp')
# Or if you saved without index:
# df = pd.read_csv('output/q3_wrangled_data.csv')
# df['Measurement Timestamp'] = pd.to_datetime(df['Measurement Timestamp'])
# df = df.set_index('Measurement Timestamp')
print(f"Loaded {len(df):,} records with datetime index")
```

---

## Objective

Create derived features, perform time-based aggregations, and calculate rolling windows for time series analysis.

**Time Series Note:** Rolling windows are essential for time series data. They capture temporal dependencies (e.g., 7-hour rolling mean captures short-term patterns). See **Lecture 09** for time series rolling window operations. For hourly data, common window sizes are 7-24 hours (capturing daily patterns). Use pandas `rolling()` method with `window` parameter to specify the number of periods.

---

## Required Artifacts

You must create exactly these 3 files in the `output/` directory:

### 1. `output/q4_features.csv`
**Format:** CSV file
**Content:** Dataset with all derived features added
**Requirements:**
- All original columns from Q3
- All new derived features added as columns
- **No index column** (save with `index=False`)

### 2. `output/q4_rolling_features.csv`
**Format:** CSV file
**Content:** Dataset with rolling window features
**Required Columns:**
- Original datetime column
- At least one rolling window calculation column (e.g., `water_temp_rolling_7h`, `air_temp_rolling_24h`)

**Requirements:**
- Must include at least one rolling window calculation
- Rolling window names should be descriptive (e.g., `temp_rolling_7h` for 7-hour rolling mean)
- **No index column** (save with `index=False`)

**Example columns:**
```csv
Measurement Timestamp,wind_speed_rolling_7h,humidity_rolling_24h,pressure_rolling_7h
2022-01-01 00:00:00,6.8,65.2,1013.5
2022-01-01 01:00:00,6.9,65.3,1013.6
...
```

**Note:** The example shows rolling windows of predictor variables (wind speed, humidity, pressure), not the target variable. If you're predicting Air Temperature, do NOT create rolling windows of Air Temperature - this causes data leakage.

### 3. `output/q4_feature_list.txt`
**Format:** Plain text file
**Content:** List of new features created (one per line)
**Requirements:**
- One feature name per line
- No extra text, just feature names
- Include all derived features, rolling features, and categorical features created

**Example format:**
```
temp_difference
temp_ratio
wind_speed_squared
comfort_index
water_temp_rolling_7h
air_temp_rolling_24h
wind_speed_rolling_7h
temp_category
wind_category
```

---

## Requirements Checklist

- [ ] Derived features created (differences, ratios, interactions, etc.)
- [ ] Time-based aggregations performed (by hour, day, month, etc.) - optional but recommended
- [ ] At least one rolling window calculation (rolling mean, rolling median, etc.)
- [ ] Categorical features created (if applicable)
- [ ] Feature list documented
- [ ] All 3 required artifacts saved with exact filenames

---

## Your Approach

1. **Create derived features:**
   ```python
   # Examples:
   df['temp_difference'] = df['Air Temperature'] - df['Water Temperature']
   df['temp_ratio'] = df['Air Temperature'] / (df['Water Temperature'] + 0.1)  # Avoid division by zero
   df['wind_speed_squared'] = df['Wind Speed'] ** 2
   df['comfort_index'] = (df['Air Temperature'] * 0.4 + 
                         (100 - df['Humidity']) * 0.3 + 
                         (20 - df['Wind Speed']) * 0.3)
   ```

2. **Calculate rolling windows:**
   ```python
   # Ensure datetime index is set and sorted
   df = df.sort_index()
   
   # Rolling windows (examples - use predictor variables only, not your target variable)
   df['wind_speed_rolling_7h'] = df['Wind Speed'].rolling(window=7, min_periods=1).mean()
   df['wind_speed_rolling_24h'] = df['Wind Speed'].rolling(window=24, min_periods=1).mean()
   df['humidity_rolling_7h'] = df['Humidity'].rolling(window=7, min_periods=1).mean()
   df['pressure_rolling_7h'] = df['Barometric Pressure'].rolling(window=7, min_periods=1).mean()
   ```
   
   ⚠️ **Important:** Only create rolling windows of **predictor variables**, not your target variable. Creating rolling windows of the target variable causes data leakage (you'd be predicting the target from a smoothed version of itself). See Q6 and Q7 for more details on avoiding data leakage.

3. **Create categorical features (optional):**
   ```python
   df['temp_category'] = pd.cut(df['Air Temperature'], 
                                bins=[-np.inf, 10, 20, 30, np.inf],
                                labels=['Cold', 'Cool', 'Warm', 'Hot'])
   ```

4. **Check for infinity values:**
   - After creating derived features (especially ratios), check for infinity values:
   ```python
   # Replace infinity with NaN, then handle appropriately
   df = df.replace([np.inf, -np.inf], np.nan)
   # Fill or drop as needed based on your analysis
   ```

5. **Document and save:**
   - Save all features: `df.reset_index().to_csv('output/q4_features.csv', index=False)`
     - **Important:** When saving CSVs with datetime index, use `reset_index()` to convert index to column, otherwise it will not be included in the output.
   - Save rolling features: `df[['rolling_col1', 'rolling_col2', ...]].reset_index().to_csv('output/q4_rolling_features.csv', index=False)`
     - **Important:** Remember to use `reset_index()` before saving to include the datetime as a column.
   - Write feature list: `with open('output/q4_feature_list.txt', 'w') as f: f.write('\n'.join(feature_names))`

---

## Decision Points

- **Derived features:** What relationships might be useful? Temperature differences? Ratios? Interactions between variables?
- **Rolling windows:** What window size makes sense? 7 hours? 24 hours? Consider the temporal scale of your data. For hourly data, 7-24 hours captures daily patterns.
- **Time-based aggregations:** Aggregate by hour? Day? Week? What temporal granularity is useful for your analysis?

---

## Checkpoint

After Q4, you should have:
- [ ] Derived features created
- [ ] At least one rolling window calculation
- [ ] Feature list documented
- [ ] All 3 artifacts saved: `q4_features.csv`, `q4_rolling_features.csv`, `q4_feature_list.txt`

---

**Next:** Continue to `q5_pattern_analysis.md` for Pattern Analysis.

