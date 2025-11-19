# Q6: Modeling Preparation

**Phase 7:** Modeling Preparation  
**Points: 3 points**

**Focus:** Perform temporal train/test split, select features, handle categorical variables.

**Lecture Reference:** See **Lecture 11, Notebook 3** (`11/demo/03_pattern_analysis_modeling_prep.ipynb`), Phase 7 for examples of temporal train/test splitting and feature preparation. **CRITICAL:** The lecture emphasizes why temporal splitting is required (not random split) for time series data.

---

## Setup

```python
# Import libraries
import pandas as pd
import numpy as np
import os

# Load feature-engineered data from Q4
df = pd.read_csv('output/q4_features.csv', parse_dates=['Measurement Timestamp'], index_col='Measurement Timestamp')
# Or if you saved without index:
# df = pd.read_csv('output/q4_features.csv')
# df['Measurement Timestamp'] = pd.to_datetime(df['Measurement Timestamp'])
# df = df.set_index('Measurement Timestamp')
print(f"Loaded {len(df):,} records with features")
```

---

## Objective

Prepare data for modeling by performing temporal train/test split, selecting features, and handling categorical variables.

**CRITICAL - Temporal Split:** For time series data, you **MUST** use temporal splitting (earlier data for training, later data for testing). **DO NOT** use random split. Why? Time series data has temporal dependencies - using future data to predict the past would be data leakage.

---

## Required Artifacts

You must create exactly these 5 files in the `output/` directory:

### 1. `output/q6_X_train.csv`
**Format:** CSV file
**Content:** Training features (X)
**Requirements:**
- All feature columns (no target variable)
- Only training data (earlier time periods)
- **No index column** (save with `index=False`)
- **No datetime column** (unless it's a feature, not the index)

### 2. `output/q6_X_test.csv`
**Format:** CSV file
**Content:** Test features (X)
**Requirements:**
- All feature columns (same as X_train)
- Only test data (later time periods)
- **No index column** (save with `index=False`)
- **No datetime column** (unless it's a feature, not the index)

### 3. `output/q6_y_train.csv`
**Format:** CSV file
**Content:** Training target variable (y)
**Requirements:**
- Single column with target variable name as header
- Only training data (corresponding to X_train)
- **No index column** (save with `index=False`)

**Example:**
```csv
Water Temperature
15.2
15.3
15.1
...
```

### 4. `output/q6_y_test.csv`
**Format:** CSV file
**Content:** Test target variable (y)
**Requirements:**
- Single column with target variable name as header
- Only test data (corresponding to X_test)
- **No index column** (save with `index=False`)

### 5. `output/q6_train_test_info.txt`
**Format:** Plain text file
**Content:** Train/test split information
**Required information:**
- Split method: Temporal (80/20 or similar)
- Training set size: [number] samples
- Test set size: [number] samples
- Training date range: [start] to [end]
- Test date range: [start] to [end]
- Number of features: [number]
- Target variable: [name]

**Example format:**
```
TRAIN/TEST SPLIT INFORMATION
==========================

Split Method: Temporal (80/20 split by time)

Training Set Size: 40000 samples
Test Set Size: 10000 samples

Training Date Range: 2022-01-01 00:00:00 to 2026-09-15 07:00:00
Test Date Range: 2026-09-15 08:00:00 to 2027-09-15 07:00:00

Number of Features: 22
Target Variable: Water Temperature
```

---

## Requirements Checklist

- [ ] Target variable selected
- [ ] Temporal train/test split performed (train on earlier data, test on later data - **NOT random split**)
- [ ] Features selected and prepared
- [ ] Categorical variables handled (encoding if needed)
- [ ] No data leakage (future data not in training set)
- [ ] All 5 required artifacts saved with exact filenames

---

## Your Approach

1. **Select target variable:**
   ```python
   target = 'Water Temperature'  # Choose a meaningful target
   ```

2. **Select features:**
   ```python
   # Exclude target and non-numeric columns
   feature_cols = [col for col in df.columns 
                   if col != target 
                   and pd.api.types.is_numeric_dtype(df[col])]
   
   # CRITICAL: Exclude features derived from the target variable
   # If predicting Air Temperature, exclude features that use Air Temperature
   features_to_exclude = [
       'temp_difference',  # Uses Air Temperature
       'temp_ratio',  # Uses Air Temperature
       'temp_category',  # Derived from Air Temperature
       'comfort_index',  # Uses Air Temperature
   ]
   feature_cols = [col for col in feature_cols if col not in features_to_exclude]
   ```

3. **Handle categorical variables:**
   ```python
   # One-hot encode if needed
   # Example: If you have 'Station Name' or 'temp_category' columns
   categorical_cols = ['Station Name', 'temp_category', 'wind_category']  # Adjust based on your data
   categorical_cols = [col for col in categorical_cols if col in df.columns]  # Only encode columns that exist
   
   if categorical_cols:
       df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
       # Update feature_cols to include encoded columns
       encoded_cols = [col for col in df_encoded.columns if any(cat in col for cat in categorical_cols)]
       feature_cols = [col for col in feature_cols if col not in categorical_cols] + encoded_cols
       df = df_encoded
   ```

4. **Perform temporal train/test split:**
   ```python
   # CRITICAL: Sort by datetime first
   df = df.sort_index()
   
   # Split by time (e.g., 80/20)
   split_idx = int(len(df) * 0.8)
   
   X_train = df[feature_cols].iloc[:split_idx]
   X_test = df[feature_cols].iloc[split_idx:]
   y_train = df[target].iloc[:split_idx]
   y_test = df[target].iloc[split_idx:]
   
   # Verify no data leakage
   assert df.index[split_idx-1] < df.index[split_idx], "Data leakage detected!"
   ```

5. **Save artifacts:**
   ```python
   X_train.to_csv('output/q6_X_train.csv', index=False)
   X_test.to_csv('output/q6_X_test.csv', index=False)
   y_train.to_csv('output/q6_y_train.csv', index=False, header=[target])
   y_test.to_csv('output/q6_y_test.csv', index=False, header=[target])
   ```

6. **Document split:**
   - Write split info to `output/q6_train_test_info.txt`

---

## Feature Selection Guidelines

When selecting features for modeling, think critically about each feature:

**Red Flags to Watch For:**
- **Circular logic**: Does this feature use the target variable to predict the target?
  - Example: Rolling mean of target, lag of target (if not handled carefully)
  - Example: If predicting `Air Temperature`, using `air_temp_rolling_7h` is circular - you're predicting temperature from smoothed temperature
- **Data leakage**: Does this feature contain information that wouldn't be available at prediction time?
  - Example: Future values, aggregated statistics that include the current value
- **Near-duplicates**: Is this feature nearly identical to the target?
  - Check correlations - if correlation > 0.95, investigate whether it's legitimate
  - Example: A feature with 99%+ correlation with the target is likely problematic

**Good Practices:**
- Use external predictors (other weather variables, temporal features)
- Create rolling windows of **predictors**, not the target
  - Good: `wind_speed_rolling_7h`, `humidity_rolling_24h`
  - Bad: `air_temp_rolling_7h` when predicting Air Temperature
- Use derived features that combine multiple predictors
- Think: "Would I have this information when making a real prediction?"

**Remember:** The goal is to predict the target from **other** information, not from the target itself.

---

## Decision Points

- **Target variable:** What do you want to predict? Temperature? Water conditions? Choose something meaningful and measurable.
- **Temporal split:** **CRITICAL** - Use temporal split (earlier data for training, later data for testing), NOT random split. Why? Time series data has temporal dependencies. Typical split: 80/20 or 70/30.
- **Feature selection:** Which features are most relevant? Consider correlations, domain knowledge, and feature importance from previous analysis.
- **Categorical encoding:** If you have categorical variables, encode them (one-hot encoding, label encoding, etc.) before modeling.

---

## Checkpoint

After Q6, you should have:
- [ ] Temporal train/test split completed (earlier → train, later → test)
- [ ] Features prepared (no target, no datetime index)
- [ ] Categorical variables encoded
- [ ] No data leakage verified
- [ ] All 5 artifacts saved: `q6_X_train.csv`, `q6_X_test.csv`, `q6_y_train.csv`, `q6_y_test.csv`, `q6_train_test_info.txt`

---

**Next:** Continue to `q7_modeling.md` for Modeling.

