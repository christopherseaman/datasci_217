# Additional Data Wrangling Techniques

1. Quick Data Visualization with pandas
2. Data Quality Assessment Techniques
3. Custom Operations with apply() and applymap()

---

## 1. Quick Data Visualization with pandas

Visualizing data can help identify patterns, outliers, and issues during the cleaning process. Pandas provides built-in plotting capabilities that integrate with matplotlib.

---

### Basic Plotting in pandas

```python
# Line plot
df['column'].plot(kind='line')

# Histogram
df['column'].hist()

# Box plot
df.boxplot(column=['col1', 'col2', 'col3'])
```

These simple plots can quickly reveal distributions and trends in your data.

---

### Advanced Plotting with Seaborn

Seaborn is a statistical data visualization library built on top of matplotlib.

```python
import seaborn as sns

# Scatter plot
df.plot.scatter(x='col1', y='col2')

# Correlation heatmap
sns.heatmap(df.corr(), annot=True)

# Pair plot
sns.pairplot(df)
```

These plots help visualize relationships between multiple variables simultaneously.

---

## 2. Data Quality Assessment Techniques

Before diving into analysis, it's crucial to assess the quality of your data. This involves checking for issues like duplicates, outliers, and inconsistent data types.

---

### Checking for Duplicates and Missing Values

Duplicate rows can skew your analysis, while missing values need to be addressed.

```python
# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)
```

---

### Identifying Outliers

Outliers can significantly impact statistical analyses and machine learning models.

```python
# Identify outliers using Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df['column']))
outliers = df[z_scores > 3]
```

This method flags values that are more than 3 standard deviations from the mean.

---

### Validating Data Types and Unique Values

Ensuring correct data types and examining unique values can reveal inconsistencies.

```python
# Validate data types
print(df.dtypes)

# Unique value counts
unique_counts = df.nunique()
print(unique_counts)

# Check categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts())
```

---

## 3. Custom Operations with apply() and applymap()

For more complex data transformations, pandas provides `apply()` and `applymap()` functions. These allow you to apply custom functions to your data.

---

### Using apply() on Columns or Rows

`apply()` lets you use custom functions on a whole Series or DataFrame.

```python
# apply() on a single column
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

df['temp_fahrenheit'] = df['temp_celsius'].apply(celsius_to_fahrenheit)

# apply() on multiple columns
def calculate_bmi(row):
    return row['weight'] / (row['height'] / 100) ** 2

df['bmi'] = df.apply(calculate_bmi, axis=1)
```

---

### Using applymap() on Entire DataFrames

`applymap()` applies a function to every element in the DataFrame.

```python
def format_currency(value):
    return f"${value:.2f}" if isinstance(value, (int, float)) else value

df_formatted = df.applymap(format_currency)
```

This is useful for operations that need to be applied uniformly across all elements.

---

# Conclusion

These additional techniques enhance your data wrangling toolkit:
- Quick visualizations help you understand your data at a glance
- Data quality assessments ensure you're working with reliable information
- Custom operations with apply() and applymap() allow for complex transformations

Practice these methods to become more efficient in your data cleaning and preparation process.