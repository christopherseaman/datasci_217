# Q3: Data Wrangling

**Phase 4:** Data Wrangling & Transformation  
**Points: 9 points**

**Focus:** Parse datetime columns, set datetime index, extract time-based features.

**Lecture Reference:** Lecture 11, Notebook 2 ([`11/demo/02_wrangling_feature_engineering.ipynb`](https://github.com/christopherseaman/datasci_217/blob/main/11/demo/02_wrangling_feature_engineering.ipynb)), Phase 4. Also see Lecture 09 (time series).

---

## Setup

```python
# Import libraries
import pandas as pd
import numpy as np
import os

# Load cleaned data from Q2
df = pd.read_csv('output/q2_cleaned_data.csv')
print(f"Loaded {len(df):,} cleaned records")
```

---

## Objective

Parse datetime columns, set datetime index, and extract temporal features for time series analysis.

**Time Series Note:** This dataset is time-series data (sensor readings over time), unlike the lecture's event-based taxi data. You'll work with a datetime index and extract temporal features (hour, day_of_week, month) that are essential for time series analysis. See **Lecture 09** for time series operations. Use pandas datetime index properties (`.hour`, `.dayofweek`, `.month`, etc.) to extract temporal features from your datetime index.

---

## Required Artifacts

You must create exactly these 3 files in the `output/` directory:

### 1. `output/q3_wrangled_data.csv`
**Format:** CSV file
**Content:** Dataset with datetime index set
**Requirements:**
- Datetime column parsed using `pd.to_datetime()`
- Datetime column set as index using `df.set_index()`
- Index sorted chronologically using `df.sort_index()`
- **When saving:** Reset index to save datetime as column: `df.reset_index().to_csv(..., index=False)`
- All original columns preserved
- **No extra index column** (save with `index=False`)

### 2. `output/q3_temporal_features.csv`
**Format:** CSV file
**Required Columns (exact names):** Must include at minimum:
- Original datetime column (e.g., `Measurement Timestamp` or `datetime`)
- `hour` (integer, 0-23)
- `day_of_week` (integer, 0=Monday, 6=Sunday)
- `month` (integer, 1-12)

**Optional but recommended:**
- `year` (integer)
- `day_name` (string, e.g., "Monday")
- `is_weekend` (integer, 0 or 1)

**Content:** DataFrame with datetime column and extracted temporal features
**Requirements:**
- At minimum: datetime column, `hour`, `day_of_week`, `month`
- All values must be valid (no NaN in required columns)
- **No index column** (save with `index=False`)

**Example columns:**
```csv
Measurement Timestamp,hour,day_of_week,month,year,day_name,is_weekend
2022-01-01 00:00:00,0,5,1,2022,Saturday,1
2022-01-01 01:00:00,1,5,1,2022,Saturday,1
...
```

### 3. `output/q3_datetime_info.txt`
**Format:** Plain text file
**Content:** Date range information after datetime parsing
**Required information:**
- Start date (earliest datetime)
- End date (latest datetime)
- Total duration (optional but recommended)

**Example format:**
```
Date Range After Datetime Parsing:
Start: 2022-01-01 00:00:00
End: 2027-09-15 07:00:00
Total Duration: 5 years, 8 months, 14 days, 7 hours
```

---

## Requirements Checklist

- [ ] Datetime columns parsed correctly using `pd.to_datetime()`
- [ ] Datetime index set using `df.set_index()`
- [ ] Index sorted chronologically using `df.sort_index()`
- [ ] Temporal features extracted: `hour`, `day_of_week`, `month` (minimum)
- [ ] All 3 required artifacts saved with exact filenames

---

## Your Approach

1. **Parse datetime** - Convert datetime column using `pd.to_datetime()`
2. **Set datetime index** - Set as index and sort chronologically
3. **Extract temporal features** - Use datetime index properties (`.hour`, `.dayofweek`, `.month`, etc.)
4. **Save artifacts** - Remember to `reset_index()` before saving CSVs so the datetime becomes a column

---

## Decision Points

- **Datetime parsing:** What format is your datetime column? Use `pd.to_datetime()` with appropriate format string if needed: `pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')`
- **Temporal features:** Extract at minimum: hour, day_of_week, month. Consider also: year, day_name, is_weekend, time_of_day categories. What makes sense for your analysis?

---

## Checkpoint

After Q3, you should have:
- [ ] Datetime columns parsed
- [ ] Datetime index set and sorted
- [ ] Temporal features extracted (at minimum: hour, day_of_week, month)
- [ ] All 3 artifacts saved: `q3_wrangled_data.csv`, `q3_temporal_features.csv`, `q3_datetime_info.txt`

---

**Next:** Continue to `q4_feature_engineering.md` for Feature Engineering.

