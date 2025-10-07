Demo Guide - Lecture 5: Data Cleaning and Preparation

# Demo 1: Missing Data Detective Work

**Objective**: Master missing data detection, analysis, and handling strategies.

**Key Concepts**: Missing data patterns, fillna strategies, dropna decisions

## Step 1: Create Messy Dataset

```python
import pandas as pd
import numpy as np

# Create realistic messy data
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

**What this demonstrates**: Real-world data comes with missing values across different columns

## Step 2: Detect and Visualize Missing Patterns

```python
# Count missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Percentage missing
print("\nPercentage missing:")
print((df.isnull().sum() / len(df) * 100).round(1))

# Which rows have ANY missing values?
print(f"\nRows with missing data: {df.isnull().any(axis=1).sum()} out of {len(df)}")

# Visualize missing pattern
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.imshow(df.isnull(), cmap='RdYlGn_r', aspect='auto')
plt.colorbar(label='Missing (1) vs Present (0)')
plt.yticks(range(len(df)), df.index)
plt.xticks(range(len(df.columns)), df.columns, rotation=45)
plt.title('Missing Data Pattern')
plt.tight_layout()
plt.show()
```

**What this demonstrates**:
- Multiple ways to quantify missingness
- Visualization reveals patterns (is it random or systematic?)
- Different columns have different amounts of missing data

## Step 3: Strategic Missing Data Handling

```python
# Strategy 1: Fill age with median (numerical, no clear pattern)
df['age_filled'] = df['age'].fillna(df['age'].median())
print("\nAge - filled with median:")
print(df[['patient_id', 'age', 'age_filled']])

# Strategy 2: Forward fill test dates (temporal data)
df['test_date_filled'] = pd.to_datetime(df['test_date']).fillna(method='ffill')
print("\nTest dates - forward filled:")
print(df[['patient_id', 'test_date', 'test_date_filled']])

# Strategy 3: Drop rows with critical missing data
# If blood_pressure AND cholesterol both missing, row is useless
df_complete = df.dropna(subset=['blood_pressure', 'cholesterol'], how='all')
print(f"\nAfter dropping rows missing both BP and cholesterol: {len(df_complete)} rows remain")
```

**What this demonstrates**:
- Different strategies for different data types
- Median for numerical (robust to outliers)
- Forward fill for temporal/sequential data
- Drop when data is too incomplete to be useful

---

# Demo 2: Data Transformation and Cleaning Pipeline

**Objective**: Build a complete data cleaning pipeline with transformations, renaming, and type conversions.

**Key Concepts**: Replace, rename, astype, cut/qcut, categorical data

## Step 1: Messy Survey Data

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

**What this demonstrates**: Real surveys have inconsistent formats, sentinel values, encoding issues

## Step 2: Clean Column Names

```python
# Standardize column names
df_clean = df.rename(columns=str.lower)  # Lowercase all
df_clean = df_clean.rename(columns=str.strip)  # Remove whitespace
df_clean.columns = df_clean.columns.str.replace(' ', '_')  # Replace spaces

print("Cleaned column names:")
print(df_clean.columns.tolist())
```

**What this demonstrates**: Column names need standardization - lowercase, no spaces, consistent format

## Step 3: Handle Sentinel Values and Bad Data

```python
# Replace sentinel values
df_clean['income'] = df_clean['income'].replace(-999, np.nan)

# Fix data type issues
df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')  # Invalid → NaN

# Fill missing income with median
median_income = df_clean['income'].median()
df_clean['income'] = df_clean['income'].fillna(median_income)

# Fill missing age with mean
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].mean())

print("\nAfter handling sentinel values and bad data:")
print(df_clean[['respondent_id', 'age', 'income']])
```

**What this demonstrates**:
- Sentinel values (-999, "N/A", etc.) need replacement
- `errors='coerce'` converts invalid data to NaN
- Choose fill strategy based on distribution (median vs mean)

## Step 4: Standardize Text Data

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

**What this demonstrates**: Text data needs consistent capitalization and spelling

## Step 5: Create Categories and Dummy Variables

```python
# Create age groups
df_clean['age_group'] = pd.cut(
    df_clean['age'],
    bins=[0, 30, 50, 100],
    labels=['Young', 'Middle', 'Senior']
)

# Create income categories
df_clean['income_level'] = pd.qcut(
    df_clean['income'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

print("\nAge groups and income levels:")
print(df_clean[['age', 'age_group', 'income', 'income_level']])

# Create dummy variables for region
region_dummies = pd.get_dummies(df_clean['region'], prefix='region')
df_final = pd.concat([df_clean, region_dummies], axis=1)

print("\nWith region dummies:")
print(df_final[['region', 'region_North', 'region_South']])
```

**What this demonstrates**:
- `cut()` creates equal-width bins
- `qcut()` creates equal-frequency bins (quantiles)
- Dummy variables prepare categorical data for modeling

---

# Demo 3: Complete Data Cleaning Workflow

**Objective**: Put it all together - a realistic, end-to-end cleaning pipeline.

**Key Concepts**: Detect → Handle → Validate → Transform → Save

## Step 1: Load Dirty E-commerce Data

```python
# Realistic e-commerce data with multiple issues
ecommerce_data = """order_id,customer,product_name,price,quantity,order_date,status
O001,John Doe,  Widget A  ,29.99,2,2024-01-15,complete
O002,JANE SMITH,Widget B,-1,1,2024-01-16,COMPLETE
O003,john doe,widget a,29.99,,2024-XX-17,pending
O004,Bob Jones,Widget C,19.99,5,2024-01-18,Complete
O005,Jane Smith,Widget B,49.99,3,2024-01-19,cancelled
O006,BOB JONES,  ,35.50,2,2024-01-20,complete"""

with open('orders_dirty.csv', 'w') as f:
    f.write(ecommerce_data)

df = pd.read_csv('orders_dirty.csv')
print("Dirty e-commerce data:")
print(df)
```

## Step 2: Detect Issues

```python
# Missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Duplicates (same customer + product + date)
print(f"\n=== DUPLICATES ===")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Data issues
print("\n=== DATA ISSUES ===")
print(f"Negative prices: {(df['price'] < 0).sum()}")
print(f"Missing quantities: {df['quantity'].isnull().sum()}")
print(f"Invalid dates: {df['order_date'].str.contains('XX', na=False).sum()}")
```

**What this demonstrates**: Always audit data quality before cleaning

## Step 3: Handle Issues Systematically

```python
df_clean = df.copy()

# 1. Fix customer names (standardize)
df_clean['customer'] = df_clean['customer'].str.strip().str.title()

# 2. Fix product names (strip whitespace, lowercase)
df_clean['product_name'] = df_clean['product_name'].str.strip().str.title()

# 3. Replace negative prices with NaN, then fill with median
df_clean.loc[df_clean['price'] < 0, 'price'] = np.nan
df_clean['price'] = df_clean['price'].fillna(df_clean['price'].median())

# 4. Fill missing quantities with 1
df_clean['quantity'] = df_clean['quantity'].fillna(1)

# 5. Fix dates - replace invalid with NaT
df_clean['order_date'] = pd.to_datetime(df_clean['order_date'], errors='coerce')

# 6. Standardize status
df_clean['status'] = df_clean['status'].str.lower().str.strip()

print("\n=== CLEANED DATA ===")
print(df_clean)
```

## Step 4: Validate Cleaning

```python
# Validation checks
print("\n=== VALIDATION ===")
print(f"Missing values remaining: {df_clean.isnull().sum().sum()}")
print(f"Negative prices: {(df_clean['price'] < 0).sum()}")
print(f"Missing quantities: {df_clean['quantity'].isnull().sum()}")

# Verify data quality improved
print(f"\nData types:\n{df_clean.dtypes}")
print(f"\nUnique statuses: {df_clean['status'].unique()}")
```

**What this demonstrates**: Always validate that cleaning achieved its goals

## Step 5: Transform for Analysis

```python
# Add calculated fields
df_clean['total_price'] = df_clean['quantity'] * df_clean['price']
df_clean['order_month'] = df_clean['order_date'].dt.to_period('M')

# Create customer spending summary
customer_summary = df_clean.groupby('customer').agg({
    'order_id': 'count',
    'total_price': 'sum'
}).rename(columns={'order_id': 'num_orders', 'total_price': 'total_spent'})

print("\n=== CUSTOMER SUMMARY ===")
print(customer_summary)

# Detect outliers in spending
Q1 = df_clean['total_price'].quantile(0.25)
Q3 = df_clean['total_price'].quantile(0.75)
IQR = Q3 - Q1
outliers = df_clean[(df_clean['total_price'] < Q1 - 1.5 * IQR) |
                     (df_clean['total_price'] > Q3 + 1.5 * IQR)]
print(f"\nOutlier orders: {len(outliers)}")
print(outliers[['order_id', 'customer', 'total_price']])
```

## Step 6: Save Results

```python
# Save cleaned data
df_clean.to_csv('orders_clean.csv', index=False)
print("\n✓ Saved orders_clean.csv")

# Save summary
customer_summary.to_csv('customer_summary.csv')
print("✓ Saved customer_summary.csv")

# Create data quality report
report = f"""DATA CLEANING REPORT
====================

Original rows: {len(df)}
Cleaned rows: {len(df_clean)}
Rows removed: {len(df) - len(df_clean)}

Issues fixed:
- Standardized {df['customer'].nunique()} customer names
- Fixed {(df['price'] < 0).sum()} negative prices
- Filled {df['quantity'].isnull().sum()} missing quantities
- Corrected {df['order_date'].str.contains('XX', na=False).sum()} invalid dates

Final data quality:
- Missing values: {df_clean.isnull().sum().sum()}
- Duplicate rows: {df_clean.duplicated().sum()}
- Outliers detected: {len(outliers)}
"""

with open('cleaning_report.txt', 'w') as f:
    f.write(report)
print("✓ Saved cleaning_report.txt")
```

**What this demonstrates**:
- Complete pipeline: detect → handle → validate → transform → save
- Always save both cleaned data and metadata/reports
- Document what was done for reproducibility

---

# Key Takeaways

**Demo 1 - Missing Data**:
- Quantify and visualize missing patterns
- Different fill strategies: median (numerical), forward fill (temporal), drop (critical missing)
- Always understand WHY data is missing before deciding how to handle it

**Demo 2 - Transformations**:
- Clean column names first (lowercase, no spaces)
- Handle sentinel values (-999, "N/A") before analysis
- Standardize text data (capitalization, spelling)
- Create categories with cut/qcut for analysis
- Dummy variables prepare data for modeling

**Demo 3 - Complete Workflow**:
- Systematic process: detect → handle → validate → transform → save
- Always validate cleaning worked
- Calculate derived fields after cleaning
- Detect outliers with IQR method
- Save cleaned data, summaries, and reports

**Best Practices**:
1. Never modify original data - always use `.copy()`
2. Document every cleaning decision
3. Validate at each step
4. Save intermediate results
5. Create audit trails (reports, logs)

**Next Steps for Students**:
- Practice with their own messy datasets
- Build reusable cleaning functions
- Create data quality checklists
- Develop validation test suites
