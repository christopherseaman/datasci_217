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

# Data Transformation and Cleaning Pipeline

Building a complete data cleaning pipeline with transformations, renaming, and type conversions

```python
import pandas as pd
import numpy as np

print("Data transformation tools ready!")
```

## Load Messy Survey Data

```python
# Realistic survey data with common issues
survey_data = """respondent_id,AGE,income,education,satisfaction,region
R001,  25  ,45000,bachelors,very satisfied,NORTH
R002,thirty-two,65000,masters,satisfied,south
R003,45,-999,bachelors,neutral,North
R004,28,55000,phd,very satisfied,SOUTH
R005,52,85000,masters,dissatisfied,north
R006,19,  ,high school,satisfied,South
R007,invalid,72000,bachelors,very satisfied,NORTH"""

# Write and load
with open('survey.csv', 'w') as f:
    f.write(survey_data)

df = pd.read_csv('survey.csv')
print("Raw survey data:")
print(df)
print(f"\nData types:\n{df.dtypes}")
```

## Clean Column Names

```python
# Standardize column names
df_clean = df.rename(columns=str.lower)  # Lowercase all
df_clean = df_clean.rename(columns=str.strip)  # Remove whitespace
df_clean.columns = df_clean.columns.str.replace(' ', '_')  # Replace spaces

print("Cleaned column names:")
print(df_clean.columns.tolist())
```

## Handle Sentinel Values and Bad Data

```python
# Replace sentinel values (-999, "N/A", etc.)
df_clean['income'] = df_clean['income'].replace(-999, np.nan)

# Fix data type issues - invalid values become NaN
df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')

# Fill missing income with median
median_income = df_clean['income'].median()
df_clean['income'] = df_clean['income'].fillna(median_income)

# Fill missing age with mean
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].mean())

print("\nAfter handling sentinel values and bad data:")
print(df_clean[['respondent_id', 'age', 'income']])
```

## Standardize Text Data

```python
# Standardize region column
df_clean['region'] = df_clean['region'].str.strip().str.title()
print("\nStandardized regions:")
print(df_clean['region'].value_counts())

# Standardize education levels
education_map = {
    'high school': 'High School',
    'bachelors': 'Bachelors',
    'masters': 'Masters',
    'phd': 'PhD'
}
df_clean['education'] = df_clean['education'].replace(education_map)
print("\nStandardized education:")
print(df_clean['education'].value_counts())
```

## Create Categories and Dummy Variables

```python
# Create age groups with equal-width bins
df_clean['age_group'] = pd.cut(
    df_clean['age'],
    bins=[0, 30, 50, 100],
    labels=['Young', 'Middle', 'Senior']
)

# Create income categories with equal-frequency bins (quantiles)
df_clean['income_level'] = pd.qcut(
    df_clean['income'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

print("\nAge groups and income levels:")
print(df_clean[['age', 'age_group', 'income', 'income_level']])
```

## Create Dummy Variables for Modeling

```python
# Create dummy variables for region
region_dummies = pd.get_dummies(df_clean['region'], prefix='region')
df_final = pd.concat([df_clean, region_dummies], axis=1)

print("\nWith region dummies:")
print(df_final[['region', 'region_North', 'region_South']])

print("\nFinal cleaned dataset:")
print(df_final)
```
