# Pandas Fundamentals and Jupyter Introduction

Welcome to week 6! Today marks a major transition - we're introducing Jupyter notebooks and pandas, the powerhouse library that makes data analysis in Python incredibly productive. You'll learn when to use notebooks versus scripts and master pandas' core data structures.

By the end of today, you'll understand when and how to use Jupyter for exploratory analysis and perform efficient data manipulation with pandas DataFrames.

*[xkcd 2083: "Data Pipeline" - Shows a complex data processing pipeline with the caption "Look, I'm not saying the data is definitely wrong, but have you considered that maybe the entire field of data science is just an elaborate excuse to avoid having to make decisions?"]*

Don't worry - pandas makes data science decisions much clearer!

# When to Use Jupyter vs Scripts

## Understanding the Tools

**Jupyter Notebooks** - Interactive environment perfect for:
- Exploratory data analysis (EDA)
- Data visualization and prototyping
- Sharing analysis with explanations
- Teaching and learning data science
- Experimenting with new datasets

**Python Scripts** - Traditional files perfect for:
- Production data pipelines  
- Automated analysis workflows
- Reusable functions and modules
- Version control and collaboration
- Large-scale data processing

### The Right Tool for the Job

**Use Jupyter when:**
- Exploring new datasets for the first time
- Creating visualizations to understand patterns
- Documenting your analysis process
- Sharing results with non-technical stakeholders
- Learning new techniques interactively

**Use Scripts when:**
- Building production data pipelines
- Creating reusable analysis functions
- Processing data on a schedule
- Working on large codebases with teams
- Need robust error handling and testing

**Brief Example Workflow:**
1. **Jupyter**: Explore new customer data, create visualizations, test hypotheses
2. **Script**: Convert successful analysis into automated monthly report
3. **Jupyter**: Present findings to stakeholders with embedded charts and explanations

# Jupyter Notebook Basics

## Setting Up Jupyter

**Reference:**
```bash
# Make sure your environment is activated
conda activate datasci217

# Install Jupyter (if not already installed)
conda install jupyter

# Start Jupyter server
jupyter notebook

# Alternative: JupyterLab (more features)
jupyter lab
```

## Essential Jupyter Operations

### Cell Types and Navigation

**Reference:**
- **Code cells**: Execute Python code, show results below
- **Markdown cells**: Formatted text, equations, documentation
- **Raw cells**: Plain text (rarely used)

**Keyboard shortcuts:**
```
Esc - Enter command mode (blue border)
Enter - Enter edit mode (green border)

In command mode:
A - Insert cell above
B - Insert cell below  
DD - Delete cell
M - Convert to markdown
Y - Convert to code
Shift+Enter - Run cell and move to next
Ctrl+Enter - Run cell and stay
```

### Markdown for Documentation

**Reference:**
```markdown
# Main Heading
## Sub Heading

**Bold text** and *italic text*

- Bullet point 1
- Bullet point 2

1. Numbered list
2. Second item

```python
# Code block
import pandas as pd
```

[Link text](https://example.com)

Mathematical equations: $x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$
```

**Brief Example:**
```markdown
# Customer Analysis - Week 6

## Data Overview
We're analyzing customer purchase data from **Q3 2024**.

### Key Questions:
1. What are the most popular products?
2. Which customers have the highest lifetime value?
3. Are there seasonal patterns in purchases?

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load customer data
customers = pd.read_csv('customer_data.csv')
```
```

## Jupyter Best Practices

### Organization and Structure

**Reference:**
```python
# Cell 1: Imports and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cell 2: Load data
data = pd.read_csv('analysis_data.csv')

# Cell 3: Quick data exploration
print(f"Dataset shape: {data.shape}")
data.head()

# Cell 4: Data cleaning
# (cleaning code here)

# Cell 5: Analysis
# (analysis code here)

# Cell 6: Visualization
# (plotting code here)
```

### Documentation Strategy

**Reference:**
- **Start with markdown cell**: Explain the analysis goals
- **Document each major section**: Use markdown headers
- **Explain complex code**: Add markdown cells before tricky analysis
- **Summarize findings**: End with conclusions in markdown
- **Include data sources**: Document where data came from

**Brief Example Structure:**
```
1. [Markdown] # Sales Analysis - October 2024
2. [Markdown] ## Objective: Identify top-performing products
3. [Code] Import libraries and load data
4. [Code] data.head() and basic exploration
5. [Markdown] ## Data Cleaning Notes
6. [Code] Handle missing values and outliers
7. [Markdown] ## Analysis Results
8. [Code] Calculate product performance metrics
9. [Code] Create visualizations
10. [Markdown] ## Key Findings and Recommendations
```

# LIVE DEMO!
*Setting up Jupyter, creating a notebook, mixing code and markdown, demonstrating interactive data exploration*

# Introduction to pandas

## Why pandas is Essential

pandas is built on NumPy but adds powerful data structures specifically designed for data analysis:
- **DataFrames** - Like Excel spreadsheets but much more powerful
- **Series** - Like NumPy arrays but with labeled indices  
- **Built-in data operations** - Grouping, joining, filtering made easy
- **File I/O** - Read/write CSV, Excel, JSON, SQL databases
- **Missing data handling** - Sophisticated tools for incomplete data

Think of pandas as "Excel on steroids" - all the familiar concepts but with programming power.

## pandas Data Structures

### Series - 1D Labeled Arrays

**Reference:**
```python
import pandas as pd
import numpy as np

# Create Series from lists
grades = pd.Series([85, 92, 78, 96, 88])
print(grades)
# Output:
# 0    85
# 1    92  
# 2    78
# 3    96
# 4    88

# Create Series with custom index
student_grades = pd.Series([85, 92, 78, 96, 88], 
                          index=['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'])
print(student_grades)
# Output:
# Alice      85
# Bob        92
# Charlie    78
# Diana      96  
# Eve        88

# Access by label
print(student_grades['Alice'])      # 85
print(student_grades[['Alice', 'Diana']])  # Multiple students
```

### DataFrames - 2D Labeled Arrays

**Reference:**
```python
# Create DataFrame from dictionary
student_data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'major': ['Biology', 'Chemistry', 'Biology', 'Physics', 'Chemistry'],
    'grade': [85, 92, 78, 96, 88],
    'year': [2024, 2025, 2024, 2023, 2025]
}

df = pd.DataFrame(student_data)
print(df)
# Output:
#       name      major  grade  year
# 0    Alice    Biology     85  2024
# 1      Bob  Chemistry     92  2025
# 2  Charlie    Biology     78  2024
# 3    Diana    Physics     96  2023
# 4      Eve  Chemistry     88  2025

# DataFrame attributes
print(f"Shape: {df.shape}")           # (5, 4) - 5 rows, 4 columns  
print(f"Columns: {df.columns}")       # ['name', 'major', 'grade', 'year']
print(f"Index: {df.index}")           # [0, 1, 2, 3, 4]
print(f"Data types: {df.dtypes}")     # Shows data type of each column
```

**Brief Example:**
```python
# Create a small dataset for practice
sales_data = {
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Tablet'],
    'price': [999.99, 25.99, 79.99, 299.99, 599.99],
    'quantity': [50, 200, 150, 75, 30],
    'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics']
}

sales_df = pd.DataFrame(sales_data)
print("Sales Data Overview:")
print(sales_df)
```

# Reading Data from Files

## CSV Files - The Most Common Format

**Reference:**
```python
# Basic CSV reading
df = pd.read_csv('student_data.csv')

# Common parameters
df = pd.read_csv('data.csv',
                sep=',',              # Delimiter (default is comma)
                header=0,             # Row to use as column names (default is 0)
                index_col=None,       # Column to use as row labels
                names=None,           # Custom column names
                skiprows=None,        # Rows to skip at beginning
                na_values=['NA', ''],  # Additional strings to recognize as NaN
                encoding='utf-8')     # File encoding

# Handle different separators
df_semicolon = pd.read_csv('data.csv', sep=';')
df_tab = pd.read_csv('data.tsv', sep='\t')

# Skip header rows
df = pd.read_csv('data_with_notes.csv', skiprows=3)

# Specify column names
df = pd.read_csv('data_no_header.csv', 
                names=['name', 'age', 'grade', 'major'])
```

## Other File Formats

**Reference:**
```python
# Excel files
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df_multi = pd.read_excel('data.xlsx', sheet_name=['Sheet1', 'Sheet2'])

# JSON files  
df = pd.read_json('data.json')

# From URLs
df = pd.read_csv('https://example.com/data.csv')

# From clipboard (useful for quick testing)
df = pd.read_clipboard()
```

**Brief Example:**
```python
# Load sample data
try:
    student_df = pd.read_csv('student_grades.csv')
    print(f"Successfully loaded {len(student_df)} student records")
    print(f"Columns: {list(student_df.columns)}")
except FileNotFoundError:
    print("File not found - creating sample data instead")
    # Create sample data for demonstration
    student_df = pd.DataFrame({
        'student_id': range(1, 21),
        'name': [f'Student_{i}' for i in range(1, 21)],
        'grade': np.random.randint(70, 100, 20),
        'major': np.random.choice(['Biology', 'Chemistry', 'Physics'], 20)
    })
```

# Basic Data Exploration

## Essential DataFrame Methods

**Reference:**
```python
# Load data
df = pd.read_csv('student_data.csv')

# Quick overview
df.head()          # First 5 rows (default)
df.head(10)        # First 10 rows
df.tail()          # Last 5 rows
df.tail(3)         # Last 3 rows

# Data information
df.info()          # Column types, non-null counts, memory usage
df.describe()      # Statistical summary of numeric columns
df.shape           # (rows, columns)
len(df)           # Number of rows
df.columns        # Column names
df.dtypes         # Data types of each column

# Memory usage
df.memory_usage()  # Memory usage by column
df.memory_usage(deep=True)  # More accurate memory usage
```

### Understanding describe() Output

**Reference:**
```python
df.describe()
# Output for numeric columns:
#        grade        year
# count   5.000000   5.000000  # Non-null values
# mean   87.800000  2024.200000  # Average
# std     7.259477    0.836660  # Standard deviation
# min    78.000000  2023.000000  # Minimum value
# 25%    85.000000  2024.000000  # 25th percentile  
# 50%    88.000000  2024.000000  # Median (50th percentile)
# 75%    92.000000  2025.000000  # 75th percentile
# max    96.000000  2025.000000  # Maximum value

# For non-numeric columns
df.describe(include='object')
#          name     major
# count       5        5  # Non-null values
# unique      5        3  # Number of unique values
# top     Alice  Biology  # Most frequent value
# freq        1        2  # Frequency of most frequent value
```

## Basic Selection and Filtering

### Column Selection

**Reference:**
```python
# Single column (returns Series)
names = df['name']
grades = df['grade']

# Multiple columns (returns DataFrame)  
subset = df[['name', 'grade']]
student_info = df[['name', 'major', 'year']]

# All columns except specific ones
df_no_year = df.drop('year', axis=1)
df_minimal = df.drop(['year', 'major'], axis=1)
```

### Row Selection with .loc and .iloc

**Reference:**
```python
# .loc - label-based selection
first_student = df.loc[0]                    # First row
first_three = df.loc[0:2]                   # Rows 0, 1, 2
specific_rows = df.loc[[0, 2, 4]]           # Rows 0, 2, 4

# .iloc - position-based selection  
first_student = df.iloc[0]                  # First row
first_three = df.iloc[0:3]                 # Rows 0, 1, 2 (excludes 3)
last_two = df.iloc[-2:]                    # Last 2 rows

# Select both rows and columns
subset = df.loc[0:2, ['name', 'grade']]    # First 3 rows, specific columns
subset = df.iloc[0:3, [0, 2]]             # First 3 rows, first and third columns
```

### Boolean Filtering

**Reference:**
```python
# Simple conditions
high_grades = df[df['grade'] > 90]
biology_students = df[df['major'] == 'Biology']
seniors = df[df['year'] == 2023]

# Multiple conditions (use & for AND, | for OR)
high_bio = df[(df['grade'] > 85) & (df['major'] == 'Biology')]
good_students = df[(df['grade'] > 85) | (df['major'] == 'Physics')]

# String operations
long_names = df[df['name'].str.len() > 5]
chemistry = df[df['major'].str.contains('Chemistry')]

# NOT conditions
not_biology = df[df['major'] != 'Biology']
not_seniors = df[~(df['year'] == 2023)]    # ~ means NOT
```

**Brief Example:**
```python
# Explore the dataset
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\nFirst few records:")
print(df.head(3))

print("\nSummary statistics:")
print(df.describe())

print("\nHigh-performing students (grade > 90):")
top_students = df[df['grade'] > 90]
print(top_students[['name', 'grade', 'major']])

print(f"\nNumber of students by major:")
print(df['major'].value_counts())
```

# LIVE DEMO!
*Loading real dataset into pandas, exploring with head/tail/info/describe, practicing selection and filtering*

# Data Cleaning Basics

## Handling Missing Data

**Reference:**
```python
# Check for missing values
df.isna()           # Boolean DataFrame showing missing values
df.isna().sum()     # Count of missing values per column  
df.isna().any()     # True if column has any missing values
df.info()           # Shows non-null counts

# Remove missing values
df.dropna()                    # Drop rows with any missing values
df.dropna(subset=['grade'])    # Drop rows missing grade only
df.dropna(axis=1)             # Drop columns with any missing values

# Fill missing values
df.fillna(0)                              # Fill with 0
df.fillna({'grade': 0, 'name': 'Unknown'})  # Different values per column
df.fillna(method='forward')               # Forward fill (use previous value)
df.fillna(method='backward')              # Backward fill (use next value)

# Fill with calculated values
df['grade'].fillna(df['grade'].mean())    # Fill with column mean
df['grade'].fillna(df['grade'].median())  # Fill with column median
```

## Data Type Conversion

**Reference:**
```python
# Check current data types
print(df.dtypes)

# Convert data types
df['grade'] = df['grade'].astype(int)           # To integer
df['year'] = df['year'].astype(str)             # To string
df['gpa'] = df['gpa'].astype(float)             # To float

# Convert to datetime
df['enrollment_date'] = pd.to_datetime(df['enrollment_date'])

# Convert to categorical (saves memory for repeated values)
df['major'] = df['major'].astype('category')
```

**Brief Example:**
```python
# Check data quality
print("Data Quality Check:")
print("-" * 20)
print("Missing values per column:")
print(df.isna().sum())

print("\nData types:")
print(df.dtypes)

# Clean example issues
print("\nCleaning data...")
# Fill missing grades with median
if df['grade'].isna().any():
    median_grade = df['grade'].median()
    df['grade'] = df['grade'].fillna(median_grade)
    print(f"Filled {df['grade'].isna().sum()} missing grades with median ({median_grade})")

# Ensure grades are integers
df['grade'] = df['grade'].astype(int)

print("Data cleaning complete!")
```

# Key Takeaways

1. **Jupyter notebooks** are perfect for exploration and documentation; **scripts** for production
2. **pandas DataFrames** are like spreadsheets with programming power
3. **pd.read_csv()** handles most data loading needs with flexible parameters
4. **head(), info(), describe()** provide quick dataset understanding
5. **Boolean filtering** enables powerful data selection using conditions
6. **Missing data handling** is crucial for real-world datasets
7. **Proper documentation** in notebooks makes analysis reproducible and shareable

You now have the foundation for data exploration and manipulation in Python. pandas makes data analysis much more efficient than pure NumPy or Python lists, especially for structured data with mixed types.

Next week: We'll dive deeper into data cleaning techniques and introduce basic visualization!

# Practice Challenge

Before next class:
1. **Jupyter Setup:**
   - Install Jupyter in your environment
   - Create a new notebook and practice mixing code and markdown cells
   - Document a simple analysis workflow
   
2. **pandas Practice:**
   - Find or create a CSV dataset with at least 50 rows
   - Load it into pandas and explore with head(), info(), describe()
   - Practice different selection methods (.loc, .iloc, boolean filtering)
   
3. **Data Quality:**
   - Check for missing values and data type issues
   - Practice cleaning operations (fillna, dropna, astype)
   - Create summary statistics by different groups

Remember: Start thinking in terms of DataFrames - they're your primary tool for data analysis!