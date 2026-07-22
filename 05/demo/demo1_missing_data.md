---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Missing Data Detective Work

Mastering missing data detection, analysis, and strategic handling

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Missing data detective tools loaded!")
```

## Create Messy Dataset

```python
# Realistic patient data with missing values
patient_data = {
    'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
    'age': [45, np.nan, 62, 34, np.nan, 58, 41, np.nan],
    'blood_pressure': [120, 135, np.nan, 118, 142, np.nan, 125, 130],
    'cholesterol': [200, np.nan, 185, np.nan, 220, 195, np.nan, 210],
    'test_date': ['2024-01-15', '2024-01-16', np.nan, '2024-01-18', '2024-01-19', np.nan, '2024-01-21', '2024-01-22']
}

df = pd.DataFrame(patient_data)
print("Raw patient data:")
print(df)
```

## Detect Missing Patterns

```python
# Count missing values per column
print("\nMissing values per column:")
print(df.isnull().sum())

# Calculate percentage missing
print("\nPercentage missing:")
print((df.isnull().sum() / len(df) * 100).round(1))

# Find rows with ANY missing values
print(f"\nRows with missing data: {df.isnull().any(axis=1).sum()} out of {len(df)}")
```

## Visualize Missing Data Pattern

```python
# Heatmap of missing values
plt.figure(figsize=(10, 4))
plt.imshow(df.isnull(), cmap='RdYlGn_r', aspect='auto')
plt.colorbar(label='Missing (1) vs Present (0)')
plt.yticks(range(len(df)), df.index)
plt.xticks(range(len(df.columns)), df.columns, rotation=45)
plt.title('Missing Data Pattern')
plt.tight_layout()
plt.show()
```

## Strategic Missing Data Handling

```python
# Strategy 1: Fill age with median (robust to outliers)
df['age_filled'] = df['age'].fillna(df['age'].median())
print("\nAge - filled with median:")
print(df[['patient_id', 'age', 'age_filled']])

# Strategy 2: Forward fill test dates (temporal data)
df['test_date_filled'] = pd.to_datetime(df['test_date']).fillna(method='ffill')
print("\nTest dates - forward filled:")
print(df[['patient_id', 'test_date', 'test_date_filled']])

# Strategy 3: Drop rows with critical missing data
# If BOTH blood_pressure AND cholesterol missing, row is useless
df_complete = df.dropna(subset=['blood_pressure', 'cholesterol'], how='all')
print(f"\nAfter dropping rows missing both BP and cholesterol: {len(df_complete)} rows remain")
```

## Compare Strategies

```python
print("\n=== SUMMARY OF STRATEGIES ===")
print(f"Original rows: {len(df)}")
print(f"Age: filled {df['age'].isnull().sum()} missing values with median")
print(f"Test dates: forward filled {pd.to_datetime(df['test_date']).isnull().sum()} missing dates")
print(f"Dropped {len(df) - len(df_complete)} rows with both BP and cholesterol missing")
```
