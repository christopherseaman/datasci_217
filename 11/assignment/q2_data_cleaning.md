# Q2: Data Cleaning

**Phase 3:** Data Cleaning & Preprocessing  
**Points: 9 points**

**Focus:** Handle missing data, outliers, validate data types, remove duplicates.

**Lecture Reference:** Lecture 11, Notebook 1 ([`11/demo/01_setup_exploration_cleaning.ipynb`](https://github.com/christopherseaman/datasci_217/blob/main/11/demo/01_setup_exploration_cleaning.ipynb)), Phase 3. Also see Lecture 05 (data cleaning).

---

## Setup

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data from Q1 (or directly from source)
df = pd.read_csv('data/beach_sensors.csv')
# If you saved cleaned data from Q1, you can load it:
# df = pd.read_csv('output/q1_exploration.csv')  # This won't work - load original
```

---

## Objective

Clean the dataset by handling missing data, outliers, validating data types, and removing duplicates.

**Time Series Note:** For time series data, forward-fill (`ffill()`) is often appropriate for missing values since sensor readings are continuous. However, you may choose other strategies based on your analysis.

---

## Required Artifacts

You must create exactly these 3 files in the `output/` directory:

### 1. `output/q2_cleaned_data.csv`
**Format:** CSV file
**Content:** Cleaned dataset with same structure as original (same columns)
**Requirements:**
- Same columns as original dataset
- Missing values handled (filled, dropped, or imputed)
- Outliers handled (removed, capped, or transformed)
- Data types validated and converted
- Duplicates removed
- **Sanity check:** Dataset should retain most rows after cleaning (at least 1,000 rows). If you're removing more than 50% of data, reconsider your strategy—imputation is usually preferable to dropping rows for this dataset.
- **No index column** (save with `index=False`)

### 2. `output/q2_cleaning_report.txt`
**Format:** Plain text file
**Content:** Detailed report of cleaning operations
**Required information:**
- Rows before cleaning: [number]
- Missing data handling method: [description]
  - Which columns had missing data
  - Method used (drop, forward-fill, impute, etc.)
  - Number of values handled
- Outlier handling: [description]
  - Detection method (IQR, z-scores, domain knowledge)
  - Which columns had outliers
  - Method used (remove, cap, transform)
  - Number of outliers handled
- Duplicates removed: [number]
- Data type conversions: [list any conversions]
- Rows after cleaning: [number]

**Example format:**
```
DATA CLEANING REPORT
====================

Rows before cleaning: 50000

Missing Data Handling:
- Water Temperature: 2500 missing values (5.0%)
  Method: Forward-fill (time series appropriate)
  Result: All missing values filled
  
- Air Temperature: 1500 missing values (3.0%)
  Method: Forward-fill, then median imputation for remaining
  Result: All missing values filled

Outlier Handling:
- Water Temperature: Detected 500 outliers using IQR method (3×IQR)
  Method: Capped at bounds [Q1 - 3×IQR, Q3 + 3×IQR]
  Bounds: [-5.2, 35.8]
  Result: 500 values capped

Duplicates Removed: 0

Data Type Conversions:
- Measurement Timestamp: Converted to datetime64[ns]

Rows after cleaning: 50000
```

### 3. `output/q2_rows_cleaned.txt`
**Format:** Plain text file
**Content:** Single integer number (total rows after cleaning)
**Requirements:**
- Only the number, no text, no labels
- No whitespace before or after
- Example: `50000`

---

## Requirements Checklist

- [ ] Missing data handling strategy chosen and implemented
- [ ] Outliers detected and handled (IQR method, z-scores, or domain knowledge)
- [ ] Data types validated and converted
- [ ] Duplicates identified and removed
- [ ] Cleaning decisions documented in report
- [ ] All 3 required artifacts saved with exact filenames

---

## Your Approach

1. **Handle missing data** - Choose appropriate strategy (drop, forward-fill, impute) based on data characteristics
2. **Detect and handle outliers** - Use IQR method or z-scores; decide whether to remove, cap, or transform
3. **Validate data types** - Ensure numeric and datetime columns are properly typed
4. **Remove duplicates**
5. **Document and save** - Write detailed cleaning report explaining your decisions

---

## Decision Points

- **Missing data:** Should you drop rows, impute values, or forward-fill? Consider: How much data is missing? Is it random or systematic? For time series, forward-fill is often appropriate.
- **Outliers:** Are they errors or valid extreme values? Use IQR method or z-scores to detect, then decide: remove, cap, or transform. Document your reasoning.
- **Data types:** Are numeric columns actually numeric? Are datetime columns properly formatted? Convert as needed.

---

## Checkpoint

After Q2, you should have:
- [ ] Missing data handled
- [ ] Outliers addressed
- [ ] Data types validated
- [ ] Duplicates removed
- [ ] All 3 artifacts saved: `q2_cleaned_data.csv`, `q2_cleaning_report.txt`, `q2_rows_cleaned.txt`

---

**Next:** Continue to `q3_data_wrangling.md` for Data Wrangling.

