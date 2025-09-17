# Data Cleaning and Basic Visualization

Welcome to week 7! Now that you're comfortable with pandas DataFrames, it's time to master data cleaning - the most important (and time-consuming) part of any data science project. You'll also create your first visualizations to understand and communicate your data insights.

By the end of today, you'll know how to handle messy real-world data and create clear, informative plots that tell compelling stories.

*[xkcd 2054: "Data Quality" - Shows a person looking at messy data saying "I spent 80% of my time cleaning the data and 20% complaining about how messy it was."]*

That's actually pretty accurate - but we'll make the cleaning part less painful!

# Why Data Cleaning Matters

## The Reality of Real Data

In textbooks, data looks like this:
```
name,age,grade,major
Alice Smith,20,85,Biology
Bob Jones,21,92,Chemistry
```

In the real world, data looks like this:
```
NAME,age,grade,Major
  Alice Smith  ,20,85,biology
BOB JONES,21,,Chemistry
charlie brown,19,78.5,BIOLOGY
Diana Wilson,, 96,chemistry
Eve Chen,22,92,Phys ics
```

**Common data quality issues:**
- Inconsistent capitalization and formatting
- Extra whitespace and special characters
- Missing values in critical columns
- Inconsistent categorical values
- Mixed data types in single columns
- Duplicate records with slight variations

## The Cost of Dirty Data

Dirty data leads to:
- **Incorrect analysis results** - Garbage in, garbage out
- **Failed joins and merges** - Inconsistent keys prevent combining datasets  
- **Visualization problems** - Charts become unreadable with inconsistent categories
- **Model failures** - Machine learning algorithms require clean, consistent data
- **Lost credibility** - Stakeholders lose trust when results are obviously wrong

**Brief Example:**
If "Biology", "biology", "BIOLOGY", and "Bio" are all the same major, but your analysis treats them as different categories, you'll get completely wrong results about major distributions.

# Advanced Missing Data Handling

## Understanding Missing Data Patterns

**Reference:**
```python
import pandas as pd
import numpy as np

# Check missing data patterns
df.isna().sum()                    # Count missing per column
df.isna().sum() / len(df) * 100    # Percentage missing per column

# Visualize missing data patterns
missing_matrix = df.isna()
print("Missing data by row:")
print(missing_matrix.sum(axis=1).value_counts().sort_index())

# Find rows with most missing data
rows_missing = df.isna().sum(axis=1)
print("Rows with missing data:")
print(df[rows_missing > 0][['name'] + list(df.columns[df.isna().any()])])
```

## Strategic Missing Data Decisions

**Reference:**
```python
# Different strategies for different situations

# Strategy 1: Drop if too much missing data
threshold = 0.5  # Drop columns missing >50% of data
df_cleaned = df.dropna(thresh=len(df) * threshold, axis=1)

# Strategy 2: Drop specific problematic rows
df_cleaned = df.dropna(subset=['critical_column'])

# Strategy 3: Fill with appropriate values
# For categorical data
df['major'] = df['major'].fillna('Unknown')
df['status'] = df['status'].fillna(df['status'].mode()[0])  # Most common value

# For numerical data  
df['age'] = df['age'].fillna(df['age'].median())            # Median for skewed data
df['grade'] = df['grade'].fillna(df['grade'].mean())        # Mean for normal data

# Strategy 4: Fill based on other columns
df['grade'] = df.groupby('major')['grade'].transform(
    lambda x: x.fillna(x.mean())
)  # Fill with mean grade for that major
```

## Advanced Filling Techniques

**Reference:**
```python
# Forward and backward filling (for time series data)
df['temperature'] = df['temperature'].fillna(method='forward')
df['sales'] = df['sales'].fillna(method='backward')

# Interpolation for numeric data
df['score'] = df['score'].interpolate(method='linear')

# Fill with calculated values based on conditions
mask_seniors = df['year'] == 2023
mask_juniors = df['year'] == 2024

df.loc[mask_seniors, 'expected_grade'] = df.loc[mask_seniors, 'expected_grade'].fillna(85)
df.loc[mask_juniors, 'expected_grade'] = df.loc[mask_juniors, 'expected_grade'].fillna(82)
```

**Brief Example:**
```python
# Real-world missing data handling
print("Missing data analysis:")
print("-" * 25)

# Check patterns
missing_counts = df.isna().sum()
missing_pct = (missing_counts / len(df) * 100).round(1)

missing_summary = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Missing_Percent': missing_pct
})
print(missing_summary[missing_summary['Missing_Count'] > 0])

# Apply appropriate strategies
if missing_summary.loc['grade', 'Missing_Percent'] < 10:
    # Few missing grades - fill with median by major
    df['grade'] = df.groupby('major')['grade'].transform(
        lambda x: x.fillna(x.median())
    )
    print("Filled missing grades with major-specific median")
else:
    # Too many missing - might need to drop or investigate further
    print("Too many missing grades - investigate data collection issues")
```

# String Data Cleaning

## Standardizing Text Data

**Reference:**
```python
# Common text cleaning operations
df['name'] = df['name'].str.strip()              # Remove leading/trailing whitespace
df['name'] = df['name'].str.title()              # Proper case: "Alice Smith"
df['major'] = df['major'].str.upper()            # All uppercase: "BIOLOGY" 
df['email'] = df['email'].str.lower()            # All lowercase for emails

# Replace multiple whitespace with single space
df['name'] = df['name'].str.replace(r'\s+', ' ', regex=True)

# Remove special characters
df['phone'] = df['phone'].str.replace(r'[^\d]', '', regex=True)  # Keep only digits

# Standardize categorical values
major_mapping = {
    'bio': 'Biology',
    'biology': 'Biology', 
    'BIOLOGY': 'Biology',
    'chem': 'Chemistry',
    'chemistry': 'Chemistry',
    'CHEMISTRY': 'Chemistry',
    'phys': 'Physics',
    'physics': 'Physics',
    'PHYSICS': 'Physics'
}
df['major'] = df['major'].str.lower().replace(major_mapping)
```

## Advanced String Operations

**Reference:**
```python
# Extract parts of strings
df['first_name'] = df['name'].str.split(' ').str[0]
df['last_name'] = df['name'].str.split(' ').str[-1]
df['domain'] = df['email'].str.split('@').str[1]

# Check string patterns
df['valid_email'] = df['email'].str.contains(r'^[^@]+@[^@]+\.[^@]+$', regex=True)
df['has_number'] = df['student_id'].str.contains(r'\d', regex=True)

# String length checks
df['name_length'] = df['name'].str.len()
short_names = df[df['name_length'] < 3]  # Possibly problematic entries

# Handle encoding issues
df['comments'] = df['comments'].str.encode('ascii', errors='ignore').str.decode('ascii')
```

**Brief Example:**
```python
# Clean student names and majors
print("String cleaning example:")
print("Before cleaning:")
print(df[['name', 'major']].head())

# Standardize names
df['name'] = df['name'].str.strip().str.title()

# Standardize majors using mapping
major_cleanup = {
    'bio': 'Biology',
    'biology': 'Biology',
    'chem': 'Chemistry', 
    'chemistry': 'Chemistry',
    'phys': 'Physics',
    'physics': 'Physics'
}

df['major'] = df['major'].str.lower().str.strip()
df['major'] = df['major'].replace(major_cleanup)

print("\nAfter cleaning:")
print(df[['name', 'major']].head())

print(f"\nUnique majors: {df['major'].unique()}")
```

# LIVE DEMO!
*Loading messy real-world dataset, identifying quality issues, applying systematic cleaning approaches*

# Duplicate Detection and Removal

## Finding Duplicates

**Reference:**
```python
# Check for exact duplicates
duplicates = df.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")

# Check specific columns for duplicates
name_duplicates = df.duplicated(subset=['name'])
id_duplicates = df.duplicated(subset=['student_id'])

# Find duplicate rows (not just count them)
duplicate_rows = df[df.duplicated()]
print("Duplicate rows:")
print(duplicate_rows)

# Find all copies of duplicated rows (including originals)
all_duplicates = df[df.duplicated(keep=False)]
```

## Strategic Duplicate Removal

**Reference:**
```python
# Remove exact duplicates (keep first occurrence)
df_no_dupes = df.drop_duplicates()

# Remove duplicates based on specific columns
df_unique_students = df.drop_duplicates(subset=['student_id'])
df_unique_names = df.drop_duplicates(subset=['name'])

# Keep last occurrence instead of first
df_latest = df.drop_duplicates(subset=['student_id'], keep='last')

# Handle "fuzzy" duplicates (similar but not identical)
# Group by similar names and choose best record
def choose_best_record(group):
    # Logic: keep record with most complete data
    completeness = group.isna().sum(axis=1)  # Count missing values per row
    return group.loc[completeness.idxmin()]   # Keep row with fewest missing values

df_deduped = df.groupby('student_id').apply(choose_best_record).reset_index(drop=True)
```

**Brief Example:**
```python
# Handle duplicates systematically
print("Duplicate analysis:")
print(f"Total rows: {len(df)}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Check for student ID duplicates (more concerning)
id_dupes = df.duplicated(subset=['student_id'], keep=False)
if id_dupes.sum() > 0:
    print(f"Students with duplicate IDs: {id_dupes.sum()}")
    print("Duplicate ID records:")
    print(df[id_dupes].sort_values('student_id'))
    
    # Keep record with most recent data or most complete information
    df_cleaned = df.drop_duplicates(subset=['student_id'], keep='last')
    print(f"After removing duplicates: {len(df_cleaned)} rows")
```

# Introduction to Data Visualization

## Why Visualization Matters

Visualizations help you:
- **Spot patterns** that are invisible in raw numbers
- **Identify outliers** and data quality issues quickly
- **Communicate findings** to both technical and non-technical audiences
- **Validate assumptions** about your data
- **Guide further analysis** by revealing interesting relationships

### The Right Chart for the Job

**Distribution of single variable:** Histogram, box plot
**Comparison across categories:** Bar chart, violin plot  
**Relationship between variables:** Scatter plot, line chart
**Part-of-whole relationships:** Pie chart (use sparingly!)
**Time series:** Line chart with time on x-axis

## pandas Plotting Basics

pandas has built-in plotting powered by matplotlib - perfect for quick exploration.

### Simple Plotting

**Reference:**
```python
import matplotlib.pyplot as plt

# Set up for inline plots (Jupyter)
%matplotlib inline

# Basic plots using pandas
df['grade'].plot(kind='hist')                    # Histogram
plt.title('Grade Distribution')
plt.xlabel('Grade')
plt.show()

df['major'].value_counts().plot(kind='bar')      # Bar chart  
plt.title('Students by Major')
plt.ylabel('Number of Students')
plt.show()

# Box plot for comparing groups
df.boxplot(column='grade', by='major')
plt.title('Grade Distribution by Major')
plt.show()
```

### Scatter Plots and Correlations

**Reference:**
```python
# Scatter plot to show relationships
df.plot.scatter(x='study_hours', y='grade')
plt.title('Study Hours vs Grade')
plt.show()

# Multiple scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df.plot.scatter(x='study_hours', y='grade', ax=axes[0])
axes[0].set_title('Study Hours vs Grade')

df.plot.scatter(x='attendance', y='grade', ax=axes[1])  
axes[1].set_title('Attendance vs Grade')
plt.tight_layout()
plt.show()
```

## Introduction to matplotlib

For more control over your plots, use matplotlib directly.

**Reference:**
```python
import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data
ax.hist(df['grade'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')

# Customize the plot
ax.set_title('Student Grade Distribution', fontsize=16, fontweight='bold')
ax.set_xlabel('Grade', fontsize=12)
ax.set_ylabel('Number of Students', fontsize=12)
ax.grid(True, alpha=0.3)

# Add statistical information
mean_grade = df['grade'].mean()
ax.axvline(mean_grade, color='red', linestyle='--', 
           label=f'Mean: {mean_grade:.1f}')
ax.legend()

plt.tight_layout()
plt.show()
```

### Subplots for Multiple Visualizations

**Reference:**
```python
# Create multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Grade histogram
axes[0,0].hist(df['grade'], bins=10, alpha=0.7)
axes[0,0].set_title('Grade Distribution')
axes[0,0].set_xlabel('Grade')

# Plot 2: Major counts
major_counts = df['major'].value_counts()
axes[0,1].bar(major_counts.index, major_counts.values)
axes[0,1].set_title('Students by Major')
axes[0,1].tick_params(axis='x', rotation=45)

# Plot 3: Box plot by major
df.boxplot(column='grade', by='major', ax=axes[1,0])
axes[1,0].set_title('Grades by Major')

# Plot 4: Scatter plot (if you have another numeric column)
if 'study_hours' in df.columns:
    axes[1,1].scatter(df['study_hours'], df['grade'], alpha=0.6)
    axes[1,1].set_xlabel('Study Hours')
    axes[1,1].set_ylabel('Grade')
    axes[1,1].set_title('Study Hours vs Grade')

plt.tight_layout()
plt.show()
```

**Brief Example:**
```python
# Create a comprehensive data overview visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Grade distribution
axes[0,0].hist(df['grade'], bins=15, alpha=0.7, color='lightblue', edgecolor='black')
axes[0,0].set_title('Grade Distribution')
axes[0,0].set_xlabel('Grade')
axes[0,0].axvline(df['grade'].mean(), color='red', linestyle='--', alpha=0.8)

# Major distribution
major_counts = df['major'].value_counts()
axes[0,1].bar(major_counts.index, major_counts.values, color='lightgreen')
axes[0,1].set_title('Students by Major')
axes[0,1].tick_params(axis='x', rotation=45)

# Grade by major (box plot)
majors = df['major'].unique()
grade_by_major = [df[df['major'] == major]['grade'] for major in majors]
axes[1,0].boxplot(grade_by_major, labels=majors)
axes[1,0].set_title('Grade Distribution by Major')
axes[1,0].tick_params(axis='x', rotation=45)

# Summary statistics table (as text)
stats_text = f"""Summary Statistics:
Total Students: {len(df)}
Average Grade: {df['grade'].mean():.1f}
Median Grade: {df['grade'].median():.1f}
Grade Std Dev: {df['grade'].std():.1f}

Missing Data:
{df.isna().sum().to_string()}
"""
axes[1,1].text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10)
axes[1,1].axis('off')
axes[1,1].set_title('Dataset Summary')

plt.tight_layout()
plt.show()
```

# LIVE DEMO!
*Creating a complete data cleaning and visualization workflow: load messy data, clean systematically, create informative plots*

# Debugging Data Issues

## Systematic Data Investigation

**Reference:**
```python
def investigate_data_quality(df, column=None):
    """Comprehensive data quality investigation"""
    print("DATA QUALITY REPORT")
    print("=" * 40)
    
    # Overall info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Missing data
    print("\nMISSING DATA:")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_report = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percent': missing_pct
    })
    print(missing_report[missing_report['Missing_Count'] > 0])
    
    # Data types
    print("\nDATA TYPES:")
    print(df.dtypes)
    
    # Duplicates
    print(f"\nDUPLICATES:")
    print(f"Exact duplicates: {df.duplicated().sum()}")
    
    # For specific column investigation
    if column and column in df.columns:
        print(f"\nCOLUMN ANALYSIS: {column}")
        print(f"Unique values: {df[column].nunique()}")
        print(f"Value counts:")
        print(df[column].value_counts().head(10))
        
        if df[column].dtype in ['object', 'string']:
            # String analysis
            print(f"Average length: {df[column].str.len().mean():.1f}")
            print(f"Contains whitespace: {df[column].str.contains(r'^\s|\s$').sum()}")
```

## Common Debugging Patterns

**Reference:**
```python
# Find outliers
def find_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Outliers in {column}: {len(outliers)}")
    return outliers

# Check for impossible values
def validate_ranges(df):
    """Check for values outside reasonable ranges"""
    issues = []
    
    if 'age' in df.columns:
        invalid_ages = df[(df['age'] < 0) | (df['age'] > 120)]
        if len(invalid_ages) > 0:
            issues.append(f"Invalid ages: {len(invalid_ages)} records")
    
    if 'grade' in df.columns:
        invalid_grades = df[(df['grade'] < 0) | (df['grade'] > 100)]
        if len(invalid_grades) > 0:
            issues.append(f"Invalid grades: {len(invalid_grades)} records")
    
    return issues

# Check categorical consistency  
def check_categories(df, column):
    """Find similar categorical values that might be the same"""
    values = df[column].dropna().astype(str).str.lower().unique()
    
    print(f"Unique values in {column}:")
    for value in sorted(values):
        count = df[df[column].str.lower() == value].shape[0]
        print(f"  {value}: {count}")
    
    # Look for similar values
    from difflib import get_close_matches
    potential_dupes = []
    for value in values:
        matches = get_close_matches(value, values, n=3, cutoff=0.8)
        if len(matches) > 1:
            potential_dupes.append(matches)
    
    if potential_dupes:
        print(f"\nPotential duplicate categories:")
        for group in potential_dupes:
            print(f"  {group}")
```

**Brief Example:**
```python
# Complete data debugging workflow
print("Starting data quality investigation...")

# Run comprehensive check
investigate_data_quality(df)

# Check specific columns
if 'grade' in df.columns:
    grade_outliers = find_outliers(df, 'grade')
    if len(grade_outliers) > 0:
        print("Grade outliers found:")
        print(grade_outliers[['name', 'grade']])

# Check for validation issues
validation_issues = validate_ranges(df)
if validation_issues:
    print("Data validation issues:")
    for issue in validation_issues:
        print(f"  - {issue}")

# Check categorical consistency
if 'major' in df.columns:
    check_categories(df, 'major')

print("\nData quality investigation complete.")
```

# Key Takeaways

1. **Data cleaning is iterative** - expect to cycle through multiple cleaning steps
2. **Understand your missing data** before deciding how to handle it
3. **String standardization** is crucial for categorical data consistency
4. **Duplicate detection** requires strategic thinking about what constitutes a "duplicate"
5. **Visualization reveals patterns** that are invisible in raw data
6. **pandas plotting** is perfect for quick exploration; matplotlib for publication-quality plots
7. **Systematic debugging** prevents overlooking critical data quality issues

You now have the skills to handle messy real-world data and create meaningful visualizations. These are fundamental skills you'll use in every data science project.

Next week: We'll dive deeper into data analysis patterns and debugging techniques!

# Practice Challenge

Before next class:
1. **Find messy data:**
   - Download a real-world dataset (Kaggle, government data, etc.)
   - Or create artificially messy data from clean data
   
2. **Complete cleaning workflow:**
   - Investigate data quality issues systematically
   - Apply appropriate cleaning techniques for each issue
   - Document your decisions and rationale
   
3. **Create informative visualizations:**
   - Use both pandas plotting and matplotlib
   - Create at least 3 different plot types
   - Include proper titles, labels, and legends

Remember: Real data is always messy - learning to clean it efficiently is what separates beginners from practitioners!