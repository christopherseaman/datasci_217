# Q4: Feature Engineering

**Phase 5:** Feature Engineering & Aggregation  
**Points: 9 points**

**Focus:** Create derived features, perform time-based aggregations, calculate rolling windows.

**Lecture Reference:** Lecture 11, Notebook 2 ([`11/demo/02_wrangling_feature_engineering.ipynb`](https://github.com/christopherseaman/datasci_217/blob/main/11/demo/02_wrangling_feature_engineering.ipynb)), Phase 5. Also see Lecture 09 (rolling windows).

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

1. **Create derived features** - Differences, ratios, interactions between variables (watch for division by zero)
2. **Calculate rolling windows** - Use `.rolling()` on predictor variables to capture temporal patterns

   ⚠️ **Data Leakage Warning:** Do not create ANY features that use your target variable - this includes rolling windows, differences, ratios, or interactions involving the target. For example, if predicting Air Temperature, do not create `air_temp * humidity` or `air_temp - wet_bulb`. Only derive features from other predictor variables.

3. **Create categorical features** - Bin continuous variables if useful (optional)
4. **Check for infinity values** - Ratios can produce infinity; replace with NaN and handle appropriately
5. **Document and save** - Remember to `reset_index()` before saving CSVs

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

