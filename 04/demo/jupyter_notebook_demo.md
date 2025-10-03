# Jupyter Notebook Demo Guide

## Demo 1: Jupyter Notebook Basics

### Setup
1. Open Jupyter Notebook: `jupyter notebook` or `jupyter lab`
2. Create a new notebook
3. Follow along with the demo

### Cell Types and Execution

```python
# Code Cell - Execute with Shift+Enter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Display options
pd.set_option('display.max_columns', None)
%matplotlib inline
```

```markdown
# Markdown Cell - Documentation
This is a **markdown cell** for documentation.

## Features:
- **Bold text**
- *Italic text*
- `Code snippets`
- Lists and tables
```

### Magic Commands

```python
# Line magics (single %)
%matplotlib inline          # Display plots in notebook
%timeit sum(range(1000))    # Time execution
%load_ext autoreload        # Auto-reload modules

# Cell magics (double %%)
%%time
# This entire cell will be timed
data = [i**2 for i in range(10000)]
```

### Data Loading and Exploration

```python
# Load sample data
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')

# Quick exploration
df.head()
df.info()
df.describe()
```

```python
# Data visualization
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()
```

## Demo 2: Interactive Data Analysis

### Creating Sample Data

```python
# Generate sample data
np.random.seed(42)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'] * 20,
    'Age': np.random.randint(18, 65, 100),
    'Salary': np.random.normal(50000, 15000, 100),
    'Department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR'], 100),
    'City': np.random.choice(['New York', 'London', 'Tokyo', 'Paris'], 100)
}
df = pd.DataFrame(data)
```

### Data Exploration

```python
# Basic info
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

```python
# Missing data check
print("Missing values:")
print(df.isna().sum())
```

```python
# Data types
print("Data types:")
print(df.dtypes)
```

### Data Analysis

```python
# Summary statistics
df.describe()
```

```python
# Group by department
dept_summary = df.groupby('Department').agg({
    'Age': ['mean', 'std'],
    'Salary': ['mean', 'std', 'count']
}).round(2)
dept_summary
```

```python
# Visualization
import seaborn as sns

plt.figure(figsize=(12, 8))

# Salary distribution by department
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='Department', y='Salary')
plt.title('Salary by Department')

# Age distribution
plt.subplot(2, 2, 2)
df['Age'].hist(bins=20, alpha=0.7)
plt.title('Age Distribution')

# City counts
plt.subplot(2, 2, 3)
df['City'].value_counts().plot(kind='bar')
plt.title('Employees by City')
plt.xticks(rotation=45)

# Salary vs Age scatter
plt.subplot(2, 2, 4)
plt.scatter(df['Age'], df['Salary'], alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Salary vs Age')

plt.tight_layout()
plt.show()
```

## Demo 3: File I/O Operations

### Reading Different File Formats

```python
# CSV file
df_csv = pd.read_csv('data.csv')

# Excel file
df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON file
df_json = pd.read_json('data.json')

# HTML table
tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)')
df_html = tables[0]
```

### Writing Files

```python
# Save to different formats
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)
df.to_json('output.json', orient='records')
```

### Handling Large Files

```python
# Read large file in chunks
chunk_size = 1000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    processed_chunk = chunk[chunk['value'] > 100]
    chunks.append(processed_chunk)

# Combine chunks
df_combined = pd.concat(chunks, ignore_index=True)
```

## Demo 4: Data Cleaning Workflow

### Load Messy Data

```python
# Create messy data
messy_data = {
    'Name': ['  Alice  ', 'Bob', '  Charlie  ', 'Diana', ''],
    'Age': [25, 30, None, 28, 35],
    'Email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'diana@email.com', ''],
    'Salary': [50000, 60000, 70000, 55000, 65000],
    'Department': ['Sales', 'Marketing', 'IT', 'HR', 'Sales']
}
messy_df = pd.DataFrame(messy_data)
```

### Data Cleaning Steps

```python
# Step 1: Remove whitespace
messy_df['Name'] = messy_df['Name'].str.strip()

# Step 2: Handle missing values
print("Missing values before cleaning:")
print(messy_df.isna().sum())
```

```python
# Step 3: Fill missing values
messy_df['Age'] = messy_df['Age'].fillna(messy_df['Age'].mean())
messy_df['Email'] = messy_df['Email'].fillna('No email provided')

# Step 4: Remove empty rows
messy_df = messy_df[messy_df['Name'] != '']
```

```python
# Step 5: Data type conversion
messy_df['Age'] = messy_df['Age'].astype(int)
messy_df['Salary'] = messy_df['Salary'].astype(float)
```

```python
# Step 6: Final validation
print("Cleaned data:")
print(messy_df)
print(f"Shape: {messy_df.shape}")
print(f"Data types:\n{messy_df.dtypes}")
```

## Demo 5: Advanced Pandas Operations

### String Operations

```python
# String methods
df['Name_Upper'] = df['Name'].str.upper()
df['Name_Length'] = df['Name'].str.len()
df['Name_First_Letter'] = df['Name'].str[0]
```

### Conditional Operations

```python
# Create new column based on conditions
df['Salary_Category'] = df['Salary'].apply(
    lambda x: 'High' if x > 60000 else 'Medium' if x > 40000 else 'Low'
)
```

### Pivot Tables

```python
# Create pivot table
pivot_table = df.pivot_table(
    values='Salary',
    index='Department',
    columns='City',
    aggfunc='mean',
    fill_value=0
)
pivot_table
```

### Data Aggregation

```python
# Complex aggregation
summary = df.groupby('Department').agg({
    'Age': ['mean', 'std', 'count'],
    'Salary': ['mean', 'min', 'max'],
    'Name': 'count'
}).round(2)
summary
```

## Best Practices Demo

### Notebook Organization

```python
# 1. Imports and setup (first cell)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
%matplotlib inline
```

```python
# 2. Data loading
df = pd.read_csv('data.csv')
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
```

```python
# 3. Data exploration
def explore_data(df):
    print("=== DATA OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    missing = df.isna().sum()
    print(missing[missing > 0])
    
    print("\n=== BASIC STATISTICS ===")
    print(df.describe())

explore_data(df)
```

```python
# 4. Data analysis
# ... your analysis code ...
```

```python
# 5. Results and visualization
# ... plots and conclusions ...
```

### Performance Tips

```python
# Use vectorized operations
# Slow: df.apply(lambda x: x**2)
# Fast: df**2

# Use appropriate data types
df['Category'] = df['Category'].astype('category')

# Use query() for complex filtering
df.query('Age > 25 and City == "New York"')
```

## Demo Conclusion

This demo covered:
- ✅ Jupyter notebook basics
- ✅ Data loading and exploration
- ✅ File I/O operations
- ✅ Data cleaning workflows
- ✅ Advanced pandas operations
- ✅ Best practices

**Next steps:**
- Practice with your own datasets
- Explore more advanced pandas features
- Learn about data visualization
- Move on to data cleaning and preparation techniques
